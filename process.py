import ffmpeg
import os

def convert_and_rename(root_dir, output_root, prefix):
    os.makedirs(output_root, exist_ok=True)
    idx = 1
    for folder, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.aac') or file.lower().endswith('.mp3'):
                audio_path = os.path.join(folder, file)
                wav_name = f"{prefix}_{idx:05d}.wav"
                wav_path = os.path.join(output_root, wav_name)
                print(f"正在转码: {audio_path} -> {wav_path}")
                try:
                    (
                        ffmpeg
                        .input(audio_path)
                        .output(
                            wav_path,
                            format='wav',
                            acodec='pcm_s16le', # 16bit PCM
                            ac=1,               # 单声道
                            ar='16000'          # 16kHz
                        )
                        .overwrite_output()
                        .run()
                    )
                    idx += 1
                except Exception as e:
                    print(f"转码失败: {audio_path}, 错误: {e}")

convert_and_rename('test', 'test_wav', 'test')
