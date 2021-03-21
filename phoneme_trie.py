import cmudict

class PhonemeTrie(object):
    WORDS_AT_PHONE = 'WORDS_AT_PHONE'
    TERMINATING_WORDS = 'TERMINATING_WORDS'
    
    def __init__(self):
        self.phones_for_word = cmudict.dict()
        self.id_to_word = list(self.phones_for_word.keys())
        self.word_to_id = { word : i for i,word in enumerate(self.id_to_word) }
        self.id_to_phoneme = list(map(lambda x: x[0], cmudict.phones()))
        self.phoneme_to_id = { phoneme : i for i,phoneme in enumerate(self.id_to_phoneme) }
        self.root_node = {}
        for word, phoneme_sequences in self.phones_for_word.items():
            word_id = self.word_to_id[word]
            for phoneme_sequence in phoneme_sequences:
                phoneme_sequence = map(self.remove_phoneme_numerals, phoneme_sequence)
                phoneme_sequence = list(phoneme_sequence)
                
                current_node = self.root_node
                
                for phoneme in phoneme_sequence[:-1]:
                    phoneme_id = self.phoneme_to_id[phoneme]
                    if not phoneme_id in current_node:
                        current_node[phoneme_id] = {PhonemeTrie.WORDS_AT_PHONE : [word_id]}
                    else:
                        current_node[phoneme_id][PhonemeTrie.WORDS_AT_PHONE].append(word_id)
                    current_node = current_node[phoneme_id]
                
                terminating_phoneme = phoneme_sequence[-1]
                terminating_phoneme_id = self.phoneme_to_id[terminating_phoneme]
                
                if not terminating_phoneme_id in current_node:
                    current_node[terminating_phoneme_id] = {PhonemeTrie.WORDS_AT_PHONE : [word_id]}
                else:
                    current_node[terminating_phoneme_id][PhonemeTrie.WORDS_AT_PHONE].append(word_id)
                
                if not PhonemeTrie.TERMINATING_WORDS in current_node[terminating_phoneme_id]:
                    current_node[terminating_phoneme_id][PhonemeTrie.TERMINATING_WORDS] = []
                
                current_node[terminating_phoneme_id][PhonemeTrie.TERMINATING_WORDS].append(word_id)
                
    def remove_phoneme_numerals(self, phoneme):
        phone_chars = filter(lambda character: character.isalpha(), phoneme)
        return ''.join(phone_chars)
    
    def decode_phoneme_sequence(self, phoneme_sequence):
        return self._decode_phoneme_sequence(phoneme_sequence, 0)
    
    def _decode_phoneme_sequence(self, phoneme_sequence, idx):
        if idx < 0:
            return None
        if idx >= len(phoneme_sequence):
            return [[]]
        
        decoded_words_sequences = []
        current_node = self.root_node
        for phoneme_idx in range(idx, len(phoneme_sequence)):
            phoneme = phoneme_sequence[phoneme_idx]
            phoneme_id = self.phoneme_to_id[phoneme]
            if not phoneme_id in current_node:
                break
            
            current_node = current_node[phoneme_id]
            
            if PhonemeTrie.TERMINATING_WORDS in current_node:
                words_terminating_here = current_node[PhonemeTrie.TERMINATING_WORDS]
                words_terminating_here = set(map(self.id_to_word.__getitem__, words_terminating_here))
                words_terminating_here = list(words_terminating_here)
                next_words_if_word_terminates_here = self._decode_phoneme_sequence(phoneme_sequence, phoneme_idx+1)
                for decoded_sequence in next_words_if_word_terminates_here:
                    decoded_words_sequences.append(words_terminating_here + decoded_sequence)
        
        return decoded_words_sequences