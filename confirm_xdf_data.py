import pyxdf
data, header = pyxdf.load_xdf('sub-P001_ses-S001_task-Default_run-001_eeg.xdf')

print(data[0]['info']['name'])         # → "MyAudioStream"
print(data[0]['info']['type'])         # → "Audio" (EEGではない！)
print(data[0]['info']['channel_count']) # → "1" (EEGなら8+チャンネル)
print(data[0]['info']['nominal_srate']) # → "44100" (EEGなら250-1000Hz)

# データの中身
print(data[0]['time_series'][0:10])    # → 音声の波形データ