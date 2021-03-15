import os
import nltk
import nltk.tokenize
import pickle
import re
from vocabulary import Vocabulary
from torch.utils.data import Dataset
import numpy as np

class SpeechDataset(Dataset):
    def __init__(self, data_file, character_level=None, vocabulary=None, transform=None):
        self.data_file = data_file
        self.data = pickle.load(open(self.data_file, 'rb'))
        self.character_level = character_level
        
        if self.character_level:
            characters = [ chr(c) for c in range(ord('a'),ord('z')+1) ]
            characters += [ ' ' ]
            character_vocab = Vocabulary()
            for character in characters:
                character_vocab.add_word(character)
            self.vocabulary = character_vocab
        elif vocabulary is None:
            data_file_dir = os.path.dirname(self.data_file)
            data_file_prefix = os.path.splitext(self.data_file)[0]
            pickle_file_name = f'{data_file_prefix}_SpeechDataset.pickle'
            pickle_file_path = os.path.join(data_file_dir, pickle_file_name)
            if not os.path.isfile(pickle_file_path):
                dataset_info = self.build_vocabulary_from_dataset(self.data)
                pickle.dump(dataset_info, open(pickle_file_path, 'wb'))
            else:
                dataset_info = pickle.load(open(pickle_file_path, 'rb'))
            self.vocabulary = dataset_info['vocabulary']
        else:
            self.vocabulary = vocabulary
        self.transform = transform
        self.max_transcription_length = max([len(transcription) for transcription in self.data['transcription_tokens']])
        self.max_input_length = max([spectrogram.shape[1] for spectrogram in self.data['audio_spectrograms']])
    
    def set_transform(self, transform):
        self.transform = transform
    
    def get_max_transcription_length(self):
        return self.max_transcription_length
    
    def get_max_input_length(self):
        return self.max_input_length
    
    def __len__(self):
        return len(self.data['transcription_tokens']) - 1
    
    def __getitem__(self, idx):
        words = self.data['transcription_tokens'][idx]
        spectral_features = self.data['audio_spectrograms'][idx]
        if self.character_level:
            words = list(' '.join(words).lower())
        sequence_words = self.vocabulary.sentence_to_tokens(words)
        sample = {'dataset_idx': np.array([idx]), 'target' : sequence_words, 'input' : spectral_features}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
    
    def use_vocabulary_from_dataset(self, dataset):
        self.vocabulary = dataset.vocabulary
    
    def build_vocabulary_from_dataset(self, data):
        vocabulary = Vocabulary()
        for transcription in data['transcription_tokens']:
            for word in transcription:
                vocabulary.add_word(word)
        dataset_info = { 'vocabulary' : vocabulary }
        return dataset_info
    
    def unk_vocabulary_with_true_oov_rate(self, other_dataset):
        other_vocabulary = other_dataset.vocabulary
        oov_rate = self.vocabulary.get_oov_rate(other_vocabulary)
        return self.unk_vocabulary_with_oov_rate(oov_rate)
    
    def unk_vocabulary_with_oov_rate(self, oov_rate):
        self.vocabulary.unk_words(oov_rate)