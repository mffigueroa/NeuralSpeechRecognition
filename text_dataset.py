import os
import nltk
import nltk.tokenize
import pickle
import re
from vocabulary import Vocabulary
from torch.utils.data import Dataset
import numpy as np

class TextDataset(Dataset):
    def __init__(self, sequence_list_file, expected_sequence_length, transform=None):
        self.sequence_list_file = sequence_list_file
        sequence_list_file_dir = os.path.dirname(self.sequence_list_file)
        sequence_list_file_prefix = os.path.splitext(self.sequence_list_file)[0]
        pickle_file_name = f'{sequence_list_file_prefix}_TextDataset.pickle'
        pickle_file_path = os.path.join(sequence_list_file_dir, pickle_file_name)
        self.sequence_list_fileobj = open(self.sequence_list_file, 'r', encoding='utf-8')
        if not os.path.isfile(pickle_file_path):
            dataset_info = self.process_dataset(self.sequence_list_fileobj)
            pickle.dump(dataset_info, open(pickle_file_path, 'wb'))
        else:
            dataset_info = pickle.load(open(pickle_file_path, 'rb'))
        self.vocabulary = dataset_info['vocabulary']
        self.sequence_delimiters = dataset_info['sequence_delimiters']
        self.transform = transform
        self.expected_sequence_length = expected_sequence_length
        
    def set_transform(self, transform):
        self.transform = transform
    
    def __len__(self):
        return len(self.sequence_delimiters) - 1
    
    def __getitem__(self, idx):
        sequence_begin = self.sequence_delimiters[idx]
        sequence_end = self.sequence_delimiters[idx+1]
        sequence_length = sequence_end - sequence_begin - 1
        self.sequence_list_fileobj.seek(sequence_begin)
        sequence_comma_separated = self.sequence_list_fileobj.read(sequence_length)
        sequence_words = sequence_comma_separated.strip().split(',')
        
        assert len(sequence_words) == self.expected_sequence_length
        
        sequence_words = self.vocabulary.sentence_to_tokens(sequence_words)
        sample = {'dataset_idx': np.array([idx]), 'words' : sequence_words}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
    
    def use_vocabulary_from_dataset(self, dataset):
        self.vocabulary = dataset.vocabulary
    
    def process_dataset(self, fileobj):
        vocabulary = Vocabulary()
        sequence_delimiters = [0]
        while True:
            line = fileobj.readline()
            if line is None or len(line) < 1:
                break
            sequence_delimiters.append(fileobj.tell())
            words = line.strip().split(',')
            for word in words:
                vocabulary.add_word(word)
        dataset_info = { 'sequence_delimiters' : sequence_delimiters, 'vocabulary' : vocabulary }
        return dataset_info
    
    def unk_vocabulary_with_true_oov_rate(self, other_dataset):
        other_vocabulary = other_dataset.vocabulary
        oov_rate = self.vocabulary.get_oov_rate(other_vocabulary)
        return self.unk_vocabulary_with_oov_rate(oov_rate)
    
    def unk_vocabulary_with_oov_rate(self, oov_rate):
        self.vocabulary.unk_words(oov_rate)