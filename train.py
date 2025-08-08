import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import os
from torch.utils.data import Dataset, DataLoader
from data import MyAudioDataset, GenerateData
from model import AudioClassifier
from sklearn.metrics import recall_score, f1_score

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def save_model(save_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, save_name))

# 训练超参数
epochs = 60
batch_size = 16
lr = 1e-4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random_seed = 20240121
save_path = 'D:/Desktop/competition/code/audio_model'
setup_seed(random_seed)

# 定义模型
model = AudioClassifier()
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=lr)
model = model.to(device)     
criterion = criterion.to(device)
print('model has been defined')

# 构建数据集
train_dataset = GenerateData(mode='train')
dev_dataset = GenerateData(mode='val')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
print('data has prepared')

# 训练
best_dev_acc = 0
for epoch_num in range(epochs):
    total_acc_train = 0
    total_loss_train = 0
    
    model.train()
    for spectrograms, labels in tqdm(train_loader):
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)
        
        # 对输入进行归一化
        inputs_m, inputs_s = spectrograms.mean(), spectrograms.std()
        spectrograms = (spectrograms - inputs_m) / inputs_s
        
        output = model(spectrograms)
        
        batch_loss = criterion(output, labels)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        acc = (output.argmax(dim=1) == labels).sum().item()
        total_acc_train += acc
        total_loss_train += batch_loss.item()

    # ----------- 验证模型 -----------
    model.eval()
    total_acc_val = 0
    total_loss_val = 0
    
    with torch.no_grad():
        all_preds = []
        all_labels = []
        for spectrograms, labels in dev_loader:
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)
            
            # 对输入进行归一化
            inputs_m, inputs_s = spectrograms.mean(), spectrograms.std()
            spectrograms = (spectrograms - inputs_m) / inputs_s
            
            output = model(spectrograms)

            batch_loss = criterion(output, labels)
            acc = (output.argmax(dim=1) == labels).sum().item()
            total_acc_val += acc
            total_loss_val += batch_loss.item()
        
            preds = output.argmax(dim=1).cpu().numpy()
            true = labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(true)
            
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        print(f'''Epochs: {epoch_num + 1} 
          | Train Loss: {total_loss_train / len(train_dataset): .3f} 
          | Train Accuracy: {total_acc_train / len(train_dataset): .3f} 
          | Val Loss: {total_loss_val / len(dev_dataset): .3f} 
          | Val Accuracy: {total_acc_val / len(dev_dataset): .3f}
          | Val Recall: {recall:.3f}
          | Val F1: {f1:.3f}''')
        
        # 保存最优的模型
        if total_acc_val / len(dev_dataset) > best_dev_acc:
            best_dev_acc = total_acc_val / len(dev_dataset)
            save_model('best.pt')
        
    model.train()

# 保存最后的模型
save_model('last.pt')
print(f'训练完成！最佳验证准确率: {best_dev_acc:.3f}')