import argparse
import re
import os
import numpy as np
import pickle

parser = argparse.ArgumentParser(description='')
parser.add_argument('librispeech_file', help='Path to input corpus sequences data file')
parser.add_argument('output_path', help='Path to output corpus sequences data file')
parser.add_argument('--min_spectrogram_sequence_length', type=int)
parser.add_argument('--max_spectrogram_sequence_length', type=int)
parser.add_argument('--min_transcription_length', type=int)
parser.add_argument('--max_transcription_length', type=int)
args = parser.parse_args()

data = pickle.load(open(args.librispeech_file, 'rb'))

assert len(data['audio_spectrograms']) == len(data['transcription_tokens'])

num_spectrograms_under_min = 0
num_spectrograms_over_max = 0
num_transcriptions_under_min = 0
num_transcriptions_over_max = 0

filtered_audio_spectrograms = []
filtered_transcription_tokens = []
for idx in range(len(data['audio_spectrograms'])):
    spectrogram = data['audio_spectrograms'][idx]
    transcription = data['transcription_tokens'][idx]
    spectrogram_length = spectrogram.shape[1]
    transcription_length = len(transcription)
    if args.min_spectrogram_sequence_length is not None and spectrogram_length < args.min_spectrogram_sequence_length:
        num_spectrograms_under_min += 1
        continue
    if args.max_spectrogram_sequence_length is not None and spectrogram_length > args.max_spectrogram_sequence_length:
        num_spectrograms_over_max += 1
        continue
    if args.min_transcription_length is not None and transcription_length < args.min_transcription_length:
        num_transcriptions_under_min += 1
        continue
    if args.max_transcription_length is not None and transcription_length > args.max_transcription_length:
        num_transcriptions_over_max += 1
        continue
    filtered_audio_spectrograms.append(spectrogram)
    filtered_transcription_tokens.append(transcription)

sequence_data = { 'audio_spectrograms' : filtered_audio_spectrograms, 'transcription_tokens' : filtered_transcription_tokens }

original_length = len(data['audio_spectrograms'])
filtered_length = len(sequence_data['audio_spectrograms'])
print(f'Original Dataset Length: {original_length}')
print(f'Filtered Dataset Length: {filtered_length}')
print(f'# Dataset Items Filtered Out: {original_length-filtered_length}\n\n')

print(f'# Dataset Spectrograms Under Minimum: {num_spectrograms_under_min}')
print(f'# Dataset Spectrograms Over Maximum: {num_spectrograms_over_max}')
print(f'# Dataset Transcriptions Under Minimum: {num_transcriptions_under_min}')
print(f'# Dataset Transcriptions Over Maximum: {num_transcriptions_over_max}\n\n')

print(f'# Dataset Spectrograms Under Minimum: {num_spectrograms_under_min}')

with open(args.output_path, 'wb') as output:
    pickle.dump(sequence_data, output)

