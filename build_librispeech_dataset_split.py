import argparse
import os
import sys
import numpy as np
import shutil

from LibrispeechDatasetToSequenceData import get_librispeech_files

np.random.seed(1234)

parser = argparse.ArgumentParser(description='Split the LibriSpeech Dataset into train, validation, and test sets.')
parser.add_argument('librispeech_dir', help='Path to LibriSpeech dataset folder with subfolders containing speech and transcription files')
parser.add_argument('output_path', help='Path to create dataset directories')
parser.add_argument('--train_split', help='Percentage of dataset to split into the train subset', type=float, default=0.95)
parser.add_argument('--test_split', help='Percentage of dataset to split into the train subset', type=float, default=0.04)
args = parser.parse_args()

train_dir = os.path.join(args.output_path, 'train')
valid_dir = os.path.join(args.output_path, 'valid')
test_dir = os.path.join(args.output_path, 'test')

try:
    os.makedirs(train_dir)
    os.makedirs(valid_dir)
    os.makedirs(test_dir)
except OSError as e:
    print('Unable to create dataset split directories or they already exist.')
    sys.exit(1)

transcription_for_folder, audio_files_in_folder = get_librispeech_files(args.librispeech_dir)
folder_transcription_file_size = { folder : os.stat(transcription_file).st_size for folder,transcription_file in transcription_for_folder.items() }
total_size = sum(folder_transcription_file_size.values())

train_size = int(total_size * args.train_split)
test_size = int(total_size * args.test_split)
valid_size = total_size - train_size - test_size

folders_scrambled = list(transcription_for_folder.keys())
folders_indices = np.arange(len(folders_scrambled))
np.random.shuffle(folders_indices)

folder_dataset_assignment = {}
dataset_order = ['train','test','valid']
dataset_sizes = {dataset : 0 for dataset in dataset_order}
dataset_optimal_sizes = [train_size,test_size,valid_size]
current_dataset_index = 0

for folder in folders_scrambled:
    dataset = dataset_order[current_dataset_index]
    dataset_optimal_size = dataset_optimal_sizes[current_dataset_index]
    folder_dataset_assignment[folder] = dataset
    folder_size = folder_transcription_file_size[folder]
    dataset_sizes[dataset] += folder_size
    if dataset_sizes[dataset] >= dataset_optimal_size:
        current_dataset_index += 1

dataset_dirs = {'train':train_dir, 'test':test_dir, 'valid':valid_dir}
for folder, dataset_assigned in folder_dataset_assignment.items():
    dataset_dir = dataset_dirs[dataset_assigned]
    src_path = folder
    dest_path = os.path.join(dataset_dir, folder)
    shutil.copytree(src_path, dest_path)