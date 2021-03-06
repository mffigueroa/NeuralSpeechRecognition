import argparse
import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms, utils
from text_dataset import TextDataset
from torch_transforms import Seq2Seq, ToTensor, RemapUsingMinWordID

from lstm_lm import LSTM_LanguageModel
from tqdm import tqdm

from vocabulary import VocabularySpecialWords

def get_model_attributes_from_weights_filename(weights_file):
    attributes = {}
    weights_file = os.path.split(weights_file)[-1]
    model_name = os.path.splitext(weights_file)[0]
    model_name = model_name.replace('model_', '')
    model_attribute_keys = ['hidden','embedding','layers']
    for attribute_key in model_attribute_keys:
        attribute_key_w_suffix = attribute_key + '_'
        try:
            attribute_key_index = model_name.index(attribute_key_w_suffix)
        except IndexError:
            continue
        attribute_index = attribute_key_index + len(attribute_key_w_suffix)
        attribute_value = model_name[attribute_index:].split('_')[0]
        attribute_value = int(attribute_value)
        attributes[attribute_key] = attribute_value
    return attributes

if __name__ == '__main__':
    torch.manual_seed(1234)
    np.random.seed(1234)
    
    parser = argparse.ArgumentParser(description='Train a simple LSTM language model.')
    parser.add_argument('weights_file', help='Path to model weights file')
    parser.add_argument('train_dataset', help='Path to processed train dataset file')
    parser.add_argument('valid_dataset', help='Path to processed train dataset file')
    parser.add_argument('test_dataset', help='Path to processed test dataset file')
    parser.add_argument('--vocab_unk_rate', help='UNKing rate to use for vocabulary, by default will use true UNK rate based on validation set OOV rate', default=-1.0)
    args = parser.parse_args()
    
    train_dataset = TextDataset(args.train_dataset, 50)
    valid_dataset = TextDataset(args.valid_dataset, 50)
    test_dataset = TextDataset(args.test_dataset, 50)
    
    if args.vocab_unk_rate == -1.0:
        train_dataset.unk_vocabulary_with_true_oov_rate(valid_dataset)
    elif args.vocab_unk_rate > 0:
        train_dataset.unk_vocabulary_with_oov_rate(args.vocab_unk_rate)
    test_dataset.use_vocabulary_from_dataset(train_dataset)
    
    max_word_id = train_dataset.vocabulary.get_max_word_id()
    lm_min_word_id = train_dataset.vocabulary.get_min_valid_lm_output_word_id()
    
    dataset_transformer = transforms.Compose([Seq2Seq(), RemapUsingMinWordID('target',lm_min_word_id), ToTensor()])
    test_dataset.set_transform(dataset_transformer)
    
    num_test_samples = 500
    batch_size = 4
    test_sampler = RandomSampler(test_dataset, replacement=True, num_samples=num_test_samples)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, sampler=test_sampler)
    
    model_attributes = get_model_attributes_from_weights_filename(args.weights_file)
    model = LSTM_LanguageModel(model_attributes['embedding'], model_attributes['hidden'], lm_min_word_id, max_word_id, num_lstm_layers=model_attributes['layers'])
    model.load_state_dict(torch.load(args.weights_file))
    model.eval()
    model.train(False)
    model.cuda()
    
    number_of_words = 0
    total_test_loss = 0
    loss_function = nn.NLLLoss()
    batches_loop = tqdm(test_dataloader)
    with torch.no_grad():
        for batch_sample in batches_loop:
            model_inputs = batch_sample['input'].cuda()
            model_targets = batch_sample['target'].cuda()
            word_scores = model(model_inputs)
            
            loss = loss_function(torch.transpose(word_scores, 1, 2), model_targets)
            
            number_of_words += np.prod(batch_sample['words'].size())
            total_test_loss += loss.cpu().numpy()
            perplexity = 2.0 ** (total_test_loss / number_of_words)
            
            batches_loop.set_postfix(loss=loss.item(), perplexity=perplexity.item())
    
    avg_test_loss = total_test_loss / number_of_words
    test_perplexity = 2.0 ** (total_test_loss / number_of_words)
    
    print(f'Average Test Loss: {avg_test_loss}')
    print(f'Test Perplexity: {test_perplexity}')

