import torch
import torch.nn.functional as F

class ToTensor(object):
    def __call__(self, sample):
        output = {}
        for key, input in sample.items():
            output[key] = torch.from_numpy(input)
        return output
        
class Seq2Seq(object):
    def __init__(self, min_output_word_id=None):
        if min_output_word_id is None:
            min_output_word_id = 0
        self.min_output_word_id = min_output_word_id
    def __call__(self, sample):
        words = sample['words']
        input = words[:-1]
        target = words[1:] - self.min_output_word_id
        return {**sample, 'input':input, 'target':target}
        
class OneHotSeq2SeqTarget(object):
    def __init__(self, max_vocab_word_id):
        self.max_vocab_word_id = max_vocab_word_id
    
    def __call__(self, sample):
        target_one_hot = F.one_hot(sample['target'], self.max_vocab_word_id)
        return { **sample, 'target' : target_one_hot }