import os
import nltk
import nltk.tokenize
import pickle
import joblib
import re
from vocabulary import Vocabulary
from torch.utils.data import Dataset
import numpy as np
import cmudict
import functools

class SpeechDataset(Dataset):
    def __init__(self, data_file, character_level=None, phoneme_level=None, vocabulary=None, transform=None):
        self.data_file = data_file
        self.data = joblib.load(open(self.data_file, 'rb'))
        self.character_level = character_level
        self.phoneme_level = phoneme_level
        self.transcription_processor = lambda words: words
        
        if self.character_level:
            characters = [ chr(c) for c in range(ord('a'),ord('z')+1) ]
            characters += [ ' ' ]
            character_vocab = Vocabulary()
            for character in characters:
                character_vocab.add_word(character)
            self.vocabulary = character_vocab
            self.transcription_processor = self._character_level_transcription_processor
        elif self.phoneme_level:
            cmu_phones = list(map(lambda x: x[0], cmudict.phones()))
            cmu_phones += [ ' ' ]
            phones_vocab = Vocabulary(custom_unk_word=' ')
            for phone in cmu_phones:
                phones_vocab.add_word(phone)
            self.vocabulary = phones_vocab
            self.phones_dict = cmudict.dict()
            self.transcription_processor = self._phone_level_transcription_processor
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
    
    def _character_level_transcription_processor(self, words):
        return list(' '.join(words).lower())
    
    def _phone_level_transcription_processor(self, words):
        phones = map(self.phones_dict.get, words)
        phones = filter(lambda phone_list: phone_list is not None, phones) # drop unknown words
        phones = map(lambda phone_list: phone_list[0], phones)
        phones = functools.reduce(lambda phone_list,phone: phone_list+phone, phones, [])
        
        remove_numerals = lambda phone: ''.join(filter(lambda character: character.isalpha(), phone))
        phones = map(remove_numerals, phones)
        return list(phones)
    
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
        words = self.transcription_processor(words)
        sequence_words = self.vocabulary.sentence_to_tokens(words)
        sample = {'dataset_idx': np.array([idx]), 'target' : sequence_words, 'input' : spectral_features}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
    
    def use_vocabulary_from_dataset(self, dataset):
        self.vocabulary = dataset.vocabulary
    
    def build_vocabulary_from_dataset(self, data):
        vocabulary = Vocabulary(custom_unk_word=' ')
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