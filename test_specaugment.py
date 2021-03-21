import pickle
from specAugment import spec_augment_pytorch
import torch
import scipy.io
import scipy.io.wavfile
import librosa
import matplotlib.pyplot as plt
import torchaudio
import numpy as np

_, sample_rate = librosa.load(r'C:\Data\NLP\LibriSpeech\train-clean-100\LibriSpeech\train-clean-100\19\198\19-198-0004.flac')
frame_stride = 0.01
hop_length = np.round(frame_stride * sample_rate).astype(np.int32)

train_dataset = pickle.load(open(r'C:\Data\NLP\LibriSpeech\train-clean-100_mfcc.pickle', 'rb'))

time_mask = torchaudio.transforms.TimeMasking(100)
freq_mask = torchaudio.transforms.FrequencyMasking(100)
time_stretch = torchaudio.transforms.TimeStretch(hop_length, 13, 1.5)

for i in range(10):
    spectrogram = train_dataset['audio_spectrograms'][i]
    #-spectrogram_audio = librosa.feature.inverse.mfcc_to_audio(spectrogram, 13)
    #scipy.io.wavfile.write(r'augmented_audio_samples\train-clean-100_mfcc\{0}.wav'.format(i), sample_rate, spectrogram_audio)
    spectrogram_tensor = torch.tensor(spectrogram)
    spec_augment_pytorch.visualization_spectrogram(spectrogram_tensor, 'Before augment')
    spectrogram_tensor = torch.unsqueeze(spectrogram_tensor, 0)
    spectrogram_augmented = time_stretch(spectrogram_tensor)
    spectrogram_augmented = torch.squeeze(spectrogram_augmented).numpy()
    spec_augment_pytorch.visualization_spectrogram(spectrogram_augmented, 'After augment')
    #spectrogram_augmented_audio = librosa.feature.inverse.mfcc_to_audio(spectrogram_augmented, 13)
    #scipy.io.wavfile.write(r'augmented_audio_samples\train-clean-100_mfcc\{0}_augmented.wav'.format(i), sample_rate, spectrogram_augmented_audio)
    
    #fig, axes = plt.subplots(2)
    #axes[0].plot(spectrogram_audio)
    #axes[0].set_title('Before augmentation')
    #axes[1].plot(spectrogram_augmented_audio)
    #axes[1].set_title('After augmentation')
    #plt.show()