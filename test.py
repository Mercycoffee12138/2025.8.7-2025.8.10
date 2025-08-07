import torch
import os
import pandas as pd
from model import AudioClassifier
from data import AudioUtil
import glob

# 参数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = "D://Desktop/competition/code/audio_model/best.pt"
test_folder = "test_wav"
output_csv = "test_result.csv"

# 加载模型
model = AudioClassifier()
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# 收集所有音频文件
audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']
test_files = []
for ext in audio_extensions:
    test_files.extend(glob.glob(os.path.join(test_folder, f"*{ext}")))

results = []

for audio_file in test_files:
    aud = AudioUtil.open(audio_file)
    if aud is None:
        pred = -1  # 加载失败
    else:
        aud = AudioUtil.normalize_amplitude(aud)
        aud = AudioUtil.remove_silence(aud, threshold=0.01)
        aud = AudioUtil.resample(aud, 16000)
        aud = AudioUtil.rechannel(aud, 1)
        aud = AudioUtil.pad_trunc(aud, 4000)
        sgram = AudioUtil.spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None)
        # 归一化
        sgram = (sgram - sgram.mean()) / (sgram.std() + 1e-6)
        # 模型输入格式 [B, C, H, W]
        sgram = sgram.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(sgram)
            pred = output.argmax(dim=1).item()
    results.append({
        "filename": os.path.basename(audio_file),
        "predict": pred  # 1=AI音乐，0=人类音乐，-1=加载失败
    })

# 保存结果
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"预测完成，结果已保存到 {output_csv}")