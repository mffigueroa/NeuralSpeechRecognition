import numpy as np
import librosa

import code

def audio_file_to_mfcc(file_path, frame_size=None, frame_stride=None, normalized=None, resampling_rate=None):
    normalized = normalized or True
    frame_size = frame_size or 0.025 # 25 ms windows, aka window length
    frame_stride = frame_stride or 0.01 # 10 ms window stride, aka hop_length
    
    signal_y, sample_rate = librosa.load(file_path)
    signal_length = signal_y.shape[0] / sample_rate
    if resampling_rate is not None:
        signal_y = librosa.resample(signal_y, sample_rate, resampling_rate)
        sample_rate = signal_y.shape[0] / signal_length
    
    win_length = np.round(frame_size * sample_rate).astype(np.int32)
    hop_length = np.round(frame_stride * sample_rate).astype(np.int32)
    
    signal_y_preem = librosa.effects.preemphasis(signal_y)
    
    mfcc = librosa.feature.mfcc(y=signal_y_preem, sr=sample_rate, win_length=win_length, hop_length=hop_length, n_mfcc=13, window='hamming')
    if not normalized:
        return mfcc
    else:
        mfcc_mean = np.mean(mfcc, axis=0, keepdims=True)
        mfcc_std = np.std(mfcc, axis=0, keepdims=True)
        mfcc_normed = (mfcc - mfcc_mean) / (mfcc_std + 1e-8)
        return mfcc_normed

def audio_file_to_spectrogram(file_path, frame_size=None, frame_stride=None, normalized=None, num_filters=None, resampling_rate=None):
    normalized = normalized or True
    frame_size = frame_size or 0.025 # 25 ms windows, aka window length
    frame_stride = frame_stride or (frame_size / 2.5) # 10 ms window stride, aka hop_length
    num_filters = num_filters or 40
    
    signal_y, sample_rate = librosa.load(file_path)
    signal_length = signal_y.shape[0] / sample_rate
    if resampling_rate is not None:
        signal_y = librosa.resample(signal_y, sample_rate, resampling_rate)
        sample_rate = signal_y.shape[0] / signal_length
    
    win_length = np.round(frame_size * sample_rate).astype(np.int32)
    hop_length = np.round(frame_stride * sample_rate).astype(np.int32)
    
    signal_y_preem = librosa.effects.preemphasis(signal_y)
    
    spectrogram = librosa.feature.melspectrogram(y=signal_y_preem, sr=sample_rate, win_length=win_length, hop_length=hop_length, n_mels=num_filters, window='hamming')
    total_frames = spectrogram.shape[1]
    effective_frames_per_second = total_frames / signal_length
    if not normalized:
        return spectrogram
    else:
        spectrogram_mean = np.mean(spectrogram, axis=0, keepdims=True)
        spectrogram_std = np.std(spectrogram, axis=0, keepdims=True)
        spectrogram_normed = (spectrogram - spectrogram_mean) / (spectrogram_std + 1e-8)
        return spectrogram_normed

if __name__ == '__main__':
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    example_file = r'C:\Data\NLP\LibriSpeech\dev-clean\LibriSpeech\dev-clean\84\121123\84-121123-0010.flac'
    mfcc = audio_file_to_mfcc(example_file)
    spectrogram = audio_file_to_spectrogram(example_file)
    
    sns.heatmap(mfcc)
    plt.title('MFCC')
    plt.show()
    
    sns.heatmap(spectrogram)
    plt.title('Mel-Spectrogram')
    plt.show()