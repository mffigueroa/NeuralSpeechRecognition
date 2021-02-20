import argparse
import re
import os
import pickle
from text_tokenizer import tokenize_line
from audio_to_mfcc import audio_file_to_spectrogram

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
args = parser.parse_args()

transcription_for_folder, audio_files_in_folder = get_librispeech_files(args.librispeech_dir)

transcription_for_audio_file = {}
transcription_file_offset_for_audio_file = {}
for folder, transcription_file in transcription_for_folder.items():
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

audio_spectrograms = []
transcription_tokens = []
file_num = 0
for audio_file, transcription in transcription_for_audio_file.items():
    with open(transcription, 'r') as transcription_file_obj:
        transcription_file_offset = transcription_file_offset_for_audio_file[audio_file]
        transcription_file_obj.seek(transcription_file_offset)
        transcription_line = transcription_file_obj.readline()
        tokens = tokenize_line(transcription_line)
        spectrogram = audio_file_to_spectrogram(audio_file, frame_size=0.09)
        transcription_tokens.append(tokens)
        audio_spectrograms.append(spectrogram)
        print(spectrogram.shape)
        file_num += 1
        if file_num > 10:
            break

sequence_data = { 'audio_spectrograms' : audio_spectrograms, 'transcription_tokens' : transcription_tokens }
with open(args.output_path, 'wb') as output_file:
    pickle.dump(sequence_data, output_file)
