import pyxdf
import pylsl
import numpy as np
import time


# 1. 音声ファイルを読み込み
audio_data, _ = pyxdf.load_xdf('sub-P001_ses-S001_task-Default_run-001_eeg.xdf')
audio_samples = audio_data[0]['time_series'].flatten()

# 2. 音声からEEGデータを生成
eeg_info = pylsl.StreamInfo('AudioBasedEEG', 'EEG', 8, 250, 'float32')
eeg_outlet = pylsl.StreamOutlet(eeg_info)

# 3. 音声→EEG変換
for i in range(0, len(audio_samples), 176):  # 44100Hz→250Hzにダウンサンプリング
    audio_chunk = audio_samples[i:i+176]
    amplitude = np.mean(np.abs(audio_chunk))
    
    # 8チャンネルのEEGデータ生成
    eeg_sample = []
    for ch in range(8):
        # 音声振幅をEEG様データに変換
        eeg_value = amplitude * (0.1 + ch * 0.05) + np.random.normal(0, 2.0)
        eeg_sample.append(float(eeg_value))
    
    eeg_outlet.push_sample(eeg_sample)
    time.sleep(1/250)