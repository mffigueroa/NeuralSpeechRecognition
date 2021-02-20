import argparse
import os
import sys
import numpy as np
import shutil

np.random.seed(1234)

parser = argparse.ArgumentParser(description='Split the Gutenberg Dataset into train, validation, and test sets.')
parser.add_argument('gutenberg_dir', help='Path to Gutenberg dataset folder containing .txt files')
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

txt_files = os.listdir(args.gutenberg_dir)
txt_files = filter(lambda file: os.path.splitext(file)[1].lower() == '.txt', txt_files)
txt_files = { file : os.path.join(args.gutenberg_dir, file) for file in txt_files }
txt_file_size = { file : os.stat(path).st_size for file,path in txt_files.items() }
total_size = sum(txt_file_size.values())

train_size = int(total_size * args.train_split)
test_size = int(total_size * args.test_split)
valid_size = total_size - train_size - test_size

files_scrambled = list(txt_files.keys())
files_indices = np.arange(len(files_scrambled))
np.random.shuffle(files_indices)

file_dataset_assignment = {}
dataset_order = ['train','test','valid']
dataset_sizes = {dataset : 0 for dataset in dataset_order}
dataset_optimal_sizes = [train_size,test_size,valid_size]
current_dataset_index = 0

for file in files_scrambled:
    dataset = dataset_order[current_dataset_index]
    dataset_optimal_size = dataset_optimal_sizes[current_dataset_index]
    file_dataset_assignment[file] = dataset
    file_size = txt_file_size[file]
    dataset_sizes[dataset] += file_size
    if dataset_sizes[dataset] >= dataset_optimal_size:
        current_dataset_index += 1

dataset_dirs = {'train':train_dir, 'test':test_dir, 'valid':valid_dir}
for file, dataset_assigned in file_dataset_assignment.items():
    dataset_dir = dataset_dirs[dataset_assigned]
    src_path = txt_files[file]
    dest_path = os.path.join(dataset_dir, file)
    shutil.copyfile(src_path, dest_path)