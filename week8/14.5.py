import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

sample_rate, audio_data = wav.read('week8\起风了.wav')

# 将音频数据转换为单声道（如果是多声道的话）
if audio_data.ndim > 1:
    audio_data = audio_data[:, 0]

fft_data = np.fft.fft(audio_data)

freq_axis = np.fft.fftfreq(len(audio_data), 1 / sample_rate)

plt.figure(figsize=(12, 6))
plt.plot(freq_axis, np.abs(fft_data))
plt.title('FFT Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()