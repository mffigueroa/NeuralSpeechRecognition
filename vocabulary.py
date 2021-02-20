from collections import Counter
from enum import Enum
import numpy as np

class VocabularySpecialWords(Enum):
    START = 1
    STOP = 2
    UNK = 3

class Vocabulary(object):
    def __init__(self):
        self.word_frequency = Counter()
        self.unked_words = set()
        self.word_to_id = {}
        self.id_to_word = {}
        self.word_id_after_unking = {}
        special_symbol_ids = [e.value for e in VocabularySpecialWords]
        self.min_word_id = min(special_symbol_ids)
        self.min_known_word_id = max(special_symbol_ids) + 1
        self.next_id_to_assign = self.min_known_word_id
        self.known_words = set()
    
    def add_word(self, word):
        self.word_frequency[word] += 1
        if not word in self.word_to_id:
            self.word_to_id[word] = self.next_id_to_assign
            self.id_to_word[self.next_id_to_assign] = word
            self.next_id_to_assign += 1
    
    def get_oov_rate(self, other_vocabulary):
        words_in_other_vocab = set(other_vocabulary.word_frequency.keys())
        words_in_this_vocab = set(self.word_frequency.keys())
        assert len(words_in_this_vocab) > 0
        num_oov_words = len(words_in_other_vocab.difference(words_in_this_vocab))
        return num_oov_words / len(words_in_this_vocab)
    
    def unk_words(self, unk_rate):
        vocab_size = len(self.word_frequency.keys())
        words_by_freq_desc = sorted(self.word_frequency.keys(), key=lambda k: self.word_frequency[k], reverse=True)
        known_word_rate = 1 - unk_rate
        truncated_vocab_size = int(known_word_rate * vocab_size)
        self.unked_words = set(words_by_freq_desc[truncated_vocab_size:])
        self.remap_word_ids_after_unking()
    
    # After UNKing, ensure word IDs are in range [0, self.get_max_word_id() - 1]
    def remap_word_ids_after_unking(self):
        known_words = list(self.get_known_words())
        known_word_ids = map(self.word_to_id.get, known_words)
        remapped_word_ids = range(self.min_known_word_id, self.min_known_word_id+len(known_words))
        self.word_id_after_unking = dict(zip(known_word_ids, remapped_word_ids))
        assert min(self.word_id_after_unking.values()) == self.min_known_word_id and max(self.word_id_after_unking.values()) == self.get_max_word_id() - 1
        
    def get_word_id(self, word):
        # if we never UNKed, just use the original word IDs
        if len(self.word_id_after_unking) < 1:
            self.remap_word_ids_after_unking()
        
        if word in self.unked_words or not word in self.word_to_id:
            remapped_word_id = VocabularySpecialWords.UNK.value
        else:
            remapped_word_id = self.word_id_after_unking[self.word_to_id[word]]
        assert remapped_word_id > 0 and remapped_word_id < self.get_max_word_id()
        return remapped_word_id
    
    def get_word_from_id(self, id):
        if id in self.id_to_word:
            return self.id_to_word[id]
        elif id == VocabularySpecialWords.START.value:
            return '[START]'
        elif id == VocabularySpecialWords.STOP.value:
            return '[STOP]'
        elif id == VocabularySpecialWords.UNK.value:
            return '[UNK]'
        else:
            import pdb; pdb.set_trace()
            return None
    
    def sentence_to_tokens(self, sentence):
        tokens = [VocabularySpecialWords.START.value]
        tokens += list(map(self.get_word_id, sentence))
        tokens += [VocabularySpecialWords.STOP.value]
        return np.array(tokens, dtype=np.int64)
    
    def tokens_to_sentence(self, word_ids):
        words = map(self.get_word_from_id, word_ids)
        return ' '.join(words)
        
    def get_known_words(self):
        if len(self.known_words) < 1:
            self.known_words = set(self.word_to_id.keys()).difference(self.unked_words)
        return self.known_words
    
    def get_vocab_size(self):
        number_of_word_id_outputs = len(self.get_known_words()) + 3 # include the special symbols as part of the vocabulary
        return number_of_word_id_outputs
    
    def get_max_word_id(self):
        return self.get_vocab_size() + self.get_min_word_id()
    
    def get_min_word_id(self):
        return self.min_word_id
    
    def get_min_valid_lm_output_word_id(self):
        return VocabularySpecialWords.STOP.value

