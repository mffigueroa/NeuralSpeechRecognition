import argparse
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser(description='')
parser.add_argument('librispeech_file', help='Path to input corpus sequences data file')
args = parser.parse_args()

data = joblib.load(open(args.librispeech_file, 'rb'))

assert len(data['audio_spectrograms']) == len(data['transcription_tokens'])

total_length = len(data['audio_spectrograms'])
print(f'Dataset length: {total_length}')

spectrogram_seq_lengths = []
transcription_seq_lengths = []
for idx in tqdm(range(len(data['audio_spectrograms']))):
    spectrogram = data['audio_spectrograms'][idx]
    transcription = data['transcription_tokens'][idx]
    spectrogram_length = spectrogram.shape[1]
    transcription_length = len(transcription)
    spectrogram_seq_lengths.append(spectrogram_length)
    transcription_seq_lengths.append(transcription_length)

fig, axes = plt.subplots(2)
axes[0].set_title('Spectrogram Feature Lengths')
axes[0].hist(spectrogram_seq_lengths)
axes[1].set_title('Transcription Lengths')
axes[1].hist(transcription_seq_lengths)
plt.show()

