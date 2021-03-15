import argparse
import re
import os
import pickle
from text_tokenizer import tokenize_line
from audio_to_mfcc import audio_file_to_spectrogram, audio_file_to_mfcc
from tqdm import tqdm

def get_librispeech_files(folder):
    transcription_for_folder = {}
    audio_files_in_folder = {}
    transcription_suffix = '.trans.txt'
    transcription_suffix_len = len(transcription_suffix)
    for root, dirs, files in os.walk(folder):
        for file in files:
            if len(file) >= transcription_suffix_len and file[-transcription_suffix_len:].lower() == transcription_suffix:
                transcription_for_folder[root] = os.path.join(root, file)
            elif os.path.splitext(file)[1].lower() == '.flac':
                if not root in audio_files_in_folder:
                    audio_files_in_folder[root] = []
                audio_files_in_folder[root].append(os.path.join(root, file))
    return transcription_for_folder, audio_files_in_folder

parser = argparse.ArgumentParser(description='Process text and speech files from LibriSpeech Dataset to produce a set of text/speech sequence pairs.')
parser.add_argument('librispeech_dir', help='Path to LibriSpeech dataset folder with subfolders containing speech and transcription files')
parser.add_argument('output_path', help='Path to output corpus sequences data file')
parser.add_argument('--frames_per_second', help='Used to calculate the proper sampling rate and windowing parameters', type=float)
parser.add_argument('--show_length_histogram', help='Show histogram of transcription and spectrogram sequence lengths', action='store_true')
args = parser.parse_args()

transcription_for_folder, audio_files_in_folder = get_librispeech_files(args.librispeech_dir)

transcription_for_audio_file = {}
transcription_file_offset_for_audio_file = {}
for folder, transcription_file in tqdm(transcription_for_folder.items()):
    for audio_file in audio_files_in_folder[folder]:
        transcription_for_audio_file[audio_file] = transcription_file
    with open(transcription_file, 'r') as transcription_file_obj:
        while True:
            file_offset = transcription_file_obj.tell()
            line = transcription_file_obj.readline()
            if len(line) < 1 or line is None:
                break
            line_split = line.split(' ')
            if len(line_split) < 1:
                continue
            audio_file_id = line_split[0]
            audio_filename = f'{audio_file_id}.flac'
            audio_filepath = os.path.join(folder, audio_filename)
            transcription_file_offset = file_offset + len(audio_file_id) + 1
            transcription_file_offset_for_audio_file[audio_filepath] = transcription_file_offset

if args.frames_per_second is not None:
    samples_per_frame = 2048
    frame_window_to_stride_ratio =  1 / 2.5
    
    sampling_rate = (args.frames_per_second - 2.5) * samples_per_frame * frame_window_to_stride_ratio + samples_per_frame
    frame_size = samples_per_frame / sampling_rate
    frame_stride = frame_size * frame_window_to_stride_ratio
else:
    sampling_rate = None
    frame_size = None
    frame_stride = None

audio_spectrograms = []
transcription_tokens = []
for audio_file, transcription in tqdm(transcription_for_audio_file.items()):
    with open(transcription, 'r') as transcription_file_obj:
        transcription_file_offset = transcription_file_offset_for_audio_file[audio_file]
        transcription_file_obj.seek(transcription_file_offset)
        transcription_line = transcription_file_obj.readline()
        tokens = tokenize_line(transcription_line)
        spectrogram = audio_file_to_mfcc(audio_file, frame_size=frame_size, frame_stride=frame_stride, resampling_rate=sampling_rate)
        transcription_tokens.append(tokens)
        audio_spectrograms.append(spectrogram)

sequence_data = { 'audio_spectrograms' : audio_spectrograms, 'transcription_tokens' : transcription_tokens }

if args.show_length_histogram:
    import matplotlib.pyplot as plt
    audio_spectrogram_lengths = [ spectrogram.shape[1] for spectrogram in audio_spectrograms ]
    transcription_token_lengths = [ len(transcription) for transcription in transcription_tokens ]
    fig, axes = plt.subplots(2)
    axes[0].set_title('Spectrogram Lengths')
    axes[0].set_ylabel('Count')
    axes[0].hist(audio_spectrogram_lengths)
    axes[1].set_title('Transcription Lengths')
    axes[1].set_ylabel('Count')
    axes[1].hist(transcription_token_lengths)
    plt.show()

with open(args.output_path, 'wb') as output_file:
    pickle.dump(sequence_data, output_file)

