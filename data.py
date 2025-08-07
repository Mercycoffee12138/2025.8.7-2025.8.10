# filepath: d:\Desktop\校赛\code\data.py
import pandas as pd
import torch
import torchaudio
from torchaudio import transforms
import librosa
import os
import random
from pathlib import Path
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader, random_split

class AudioUtil():
    @staticmethod
    def open(audio_file):
        """读取音频文件（用librosa）"""
        try:
            y, sr = librosa.load(audio_file, sr=None, mono=False)  # 支持多声道
            if y.ndim == 1:
                y = y[np.newaxis, :]  # (1, N)
            else:
                y = y  # (C, N)
            sig = torch.from_numpy(y)
            return (sig, sr)
        except Exception as e:
            print(f"Error loading {audio_file}: {e}")
            return None
    
    @staticmethod
    def normalize_amplitude(aud):
        """将音频幅度归一化到[-1, 1]范围"""
        sig, sr = aud
        max_val = sig.abs().max()
        if max_val > 0:
            sig = sig / max_val
        return (sig, sr)
    
    @staticmethod
    def remove_silence(aud, threshold=0.01):
        """简单的静音段去除"""
        sig, sr = aud
        energy = sig.pow(2).mean(dim=0)
        non_silent = energy > threshold
        if non_silent.any():
            start_idx = non_silent.nonzero()[0].item()
            end_idx = non_silent.nonzero()[-1].item() + 1
            sig = sig[:, start_idx:end_idx]
        return (sig, sr)
    
    @staticmethod
    def rechannel(aud, new_channel):
        """转换声道数"""
        sig, sr = aud
        if (sig.shape[0] == new_channel):
            return aud
        if (new_channel == 1):
            resig = sig.mean(dim=0, keepdim=True)
        else:
            resig = torch.cat([sig, sig])
        return ((resig, sr))

    @staticmethod
    def resample(aud, newsr):
        """重采样到目标采样率"""
        sig, sr = aud
        if (sr == newsr):
            return aud
        num_channels = sig.shape[0]
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1, :])
        if (num_channels > 1):
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:, :])
            resig = torch.cat([resig, retwo])
        return ((resig, newsr))

    @staticmethod
    def pad_trunc(aud, max_ms):
        """调整音频长度"""
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr // 1000 * max_ms

        if (sig_len > max_len):
            sig = sig[:, :max_len]
        elif (sig_len < max_len):
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))
            sig = torch.cat((pad_begin, sig, pad_end), 1)
        return (sig, sr)

    @staticmethod
    def time_shift(aud, shift_limit):
        """时间移位数据增强"""
        sig, sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)

    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        """生成梅尔频谱图"""
        sig, sr = aud
        top_db = 80
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)

    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        """频谱图数据增强"""
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec

class MyAudioDataset(Dataset):
    def __init__(self, processed_spectrograms, processed_labels):
        """
        使用预处理好的频谱图和标签创建数据集
        processed_spectrograms: 预处理好的频谱图张量
        processed_labels: 对应的标签张量
        """
        self.spectrograms = processed_spectrograms
        self.labels = processed_labels

    def __getitem__(self, idx):
        return self.spectrograms[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

def create_dataset_csv(ai_folder, nonai_folder, output_csv="ai_music_dataset.csv"):
    """创建数据集CSV文件"""
    data = []
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg','.aac']
    
    # 处理AI音乐文件夹 (标签为1)
    ai_path = Path(ai_folder)
    if ai_path.exists():
        for file in ai_path.rglob('*'):
            if file.suffix.lower() in audio_extensions:
                relative_path = str(file.relative_to(ai_path.parent))
                data.append({
                    'relative_path': relative_path,
                    'classID': 1
                })
    
    # 处理非AI音乐文件夹 (标签为0)
    nonai_path = Path(nonai_folder)
    if nonai_path.exists():
        for file in nonai_path.rglob('*'):
            if file.suffix.lower() in audio_extensions:
                abs_path = str(file.resolve())
                abs_path = abs_path.replace("\\", "/")  # 统一分隔符
                data.append({
                    'relative_path': abs_path,
                    'classID': 0
                })
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    return df

def preprocess_all_data(ai_folder="aidata_wav", nonai_folder="noneaidata_wav", save_dir="processed_data"):
    """预处理所有数据并保存"""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # 如果已经存在预处理数据，直接返回路径
    if (save_path / "spectrograms.pt").exists() and (save_path / "labels.pt").exists():
        print(f"发现已存在的预处理数据: {save_path}")
        return save_path
    
    print("开始预处理音频数据...")
    
    # 创建数据集CSV
    df = create_dataset_csv(ai_folder, nonai_folder, save_path / "ai_music_dataset.csv")
    
    # 预处理每个音频文件
    processed_data = []
    labels = []
    
    for idx, row in df.iterrows():
        if idx % 50 == 0:
            print(f"处理进度: {idx}/{len(df)}")
        
        audio_file = Path(".") / row['relative_path']
        class_id = row['classID']
        
        try:
            # 音频预处理流水线
            aud = AudioUtil.open(str(audio_file))
            if aud is None:
                continue
            
            aud = AudioUtil.normalize_amplitude(aud)
            aud = AudioUtil.remove_silence(aud, threshold=0.01)
            aud = AudioUtil.resample(aud, 16000)
            aud = AudioUtil.rechannel(aud, 1)
            aud = AudioUtil.pad_trunc(aud, 4000)
            aud = AudioUtil.time_shift(aud, 0.4)
            sgram = AudioUtil.spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None)
            aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
            
            processed_data.append(aug_sgram)
            labels.append(class_id)
            
        except Exception as e:
            print(f"处理文件 {audio_file} 时出错: {e}")
    
    # 保存预处理数据
    all_spectrograms = torch.stack(processed_data)
    all_labels = torch.tensor(labels)
    
    torch.save(all_spectrograms, save_path / "spectrograms.pt")
    torch.save(all_labels, save_path / "labels.pt")
    
    # 保存数据信息
    data_info = {
        'num_samples': len(processed_data),
        'spectrogram_shape': all_spectrograms.shape,
        'num_ai_samples': (all_labels == 1).sum().item(),
        'num_human_samples': (all_labels == 0).sum().item(),
    }
    
    with open(save_path / "data_info.pkl", 'wb') as f:
        pickle.dump(data_info, f)
    
    print(f"预处理完成！数据已保存到 {save_path}")
    return save_path

def GenerateData(mode, data_dir="processed_data"):
    """
    生成数据集，类似于BERT的GenerateData函数
    mode: 'train', 'val', 'test'
    data_dir: 预处理数据存储目录
    """
    data_path = Path(data_dir)
    
    # 检查预处理数据是否存在，不存在则先预处理
    if not (data_path / "spectrograms.pt").exists():
        print("未找到预处理数据，开始预处理...")
        preprocess_all_data(save_dir=data_dir)
    
    # 加载预处理数据
    spectrograms = torch.load(data_path / "spectrograms.pt")
    labels = torch.load(data_path / "labels.pt")
    
    print(f"加载数据: {spectrograms.shape}, 标签: {len(labels)}")
    
    # 创建完整数据集
    full_dataset = MyAudioDataset(spectrograms, labels)
    
    # 按模式返回数据集
    if mode == 'train':
        # 训练集占80%
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, _ = random_split(full_dataset, [train_size, val_size])
        return train_dataset
    
    elif mode == 'val':
        # 验证集占20%
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        _, val_dataset = random_split(full_dataset, [train_size, val_size])
        return val_dataset
    
    elif mode == 'test':
        # 测试集使用全部数据（实际应用中可能需要单独的测试集）
        return full_dataset
    
    else:
        raise ValueError("mode must be 'train', 'val', or 'test'")
    
if __name__ == "__main__":
    # 生成csv文件，保存所有音频文件的路径和标签
    create_dataset_csv("aidata_wav", "noneaidata_wav", "ai_music_dataset.csv")
    print("CSV文件已生成：ai_music_dataset.csv")