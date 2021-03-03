import torch
import torch.nn.functional as F
import torch.nn.utils as utils

class ToTensor(object):
    def __call__(self, sample):
        output = {}
        for key, input in sample.items():
            output[key] = torch.from_numpy(input)
        return output

class RemapUsingMinWordID(object):
    def __init__(self, sample_key, min_output_word_id):
        self.sample_key = sample_key
        self.min_output_word_id = min_output_word_id
    
    def __call__(self, sample):
        output = {**sample}
        output[self.sample_key] -= self.min_output_word_id
        return output
        
class Seq2Seq(object):
    def __call__(self, sample):
        words = sample['words']
        input = words[:-1]
        target = words[1:]
        return {**sample, 'input':input, 'target':target}

class DeleteSequencePrefix(object):
    def __init__(self, sample_key, units_to_delete):
        self.sample_key = sample_key
        self.units_to_delete = units_to_delete
    
    def __call__(self, sample):
        output = {**sample}
        output[self.sample_key] = output[self.sample_key][self.units_to_delete:]
        return output
        
class DeleteSequenceSuffix(object):
    def __init__(self, sample_key, units_to_delete):
        self.sample_key = sample_key
        self.units_to_delete = units_to_delete
    
    def __call__(self, sample):
        output = {**sample}
        output[self.sample_key] = output[self.sample_key][:-self.units_to_delete]
        return output
        
class PadSequencesCollater(object):
    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x['input'].shape[1], reverse=True)
        input_sequences = [torch.transpose(x['input'],0,1) for x in sorted_batch]
        target_sequences = [x['target'] for x in sorted_batch]
        input_sequences_padded = torch.nn.utils.rnn.pad_sequence(input_sequences, batch_first=True)
        input_sequences_padded = torch.transpose(input_sequences_padded, 1, 2)
        target_sequences_padded = torch.nn.utils.rnn.pad_sequence(target_sequences, batch_first=True)
        input_lengths = torch.LongTensor([x.size()[0] for x in input_sequences])
        target_lengths = torch.LongTensor([x.size()[0] for x in target_sequences])
        dataset_idx = torch.cat([x['dataset_idx'] for x in batch])
        return {
            'input' : input_sequences_padded,
            'target' : target_sequences_padded,
            'dataset_idx' : dataset_idx,
            'input_lengths' : input_lengths,
            'target_lengths' : target_lengths
        }