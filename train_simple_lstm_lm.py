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
from torch_transforms import Seq2Seq, ToTensor, OneHotSeq2SeqTarget

from lstm_lm import LSTM_LanguageModel
from tqdm import tqdm

torch.manual_seed(1234)
np.random.seed(1234)

parser = argparse.ArgumentParser(description='Train a simple LSTM language model.')
parser.add_argument('log_file', help='Path to output log file')
parser.add_argument('train_dataset', help='Path to processed train dataset file')
parser.add_argument('valid_dataset', help='Path to processed train dataset file')
parser.add_argument('--vocab_unk_rate', help='UNKing rate to use for vocabulary, by default will use true UNK rate based on validation set OOV rate', default=-1.0)
args = parser.parse_args()

n_epochs = 1000
train_samples_per_epoch = 100#10000
valid_samples_per_epoch = 1000
batch_size = 4
max_sequence_length = 50

train_dataset = TextDataset(args.train_dataset, max_sequence_length)
valid_dataset = TextDataset(args.valid_dataset, max_sequence_length)

if args.vocab_unk_rate == -1.0:
    train_dataset.unk_vocabulary_with_true_oov_rate(valid_dataset)
elif args.vocab_unk_rate > 0:
    train_dataset.unk_vocabulary_with_oov_rate(args.vocab_unk_rate)

max_word_id = train_dataset.vocabulary.get_max_word_id()
lm_min_word_id = train_dataset.vocabulary.get_min_valid_lm_output_word_id()
vocabulary_size = train_dataset.vocabulary.get_vocab_size()
valid_dataset.use_vocabulary_from_dataset(train_dataset)
print(f'Vocabulary Size: {vocabulary_size}')

dataset_transformer = transforms.Compose([Seq2Seq(lm_min_word_id), ToTensor()])
train_dataset.set_transform(dataset_transformer)
valid_dataset.set_transform(dataset_transformer)

train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=train_samples_per_epoch)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0)
valid_sampler = RandomSampler(valid_dataset, replacement=True, num_samples=valid_samples_per_epoch)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=0)

num_layers = np.arange(1,2)
embedding_sizes = np.arange(16, 64)
hidden_sizes = np.arange(16, 64)
learning_rates = 10.**np.arange(-5,-1)
hyperparameters_tried = set()

'''
num_layers = np.arange(5,10)
embedding_sizes = np.arange(64, 128)
hidden_sizes = np.arange(64, 128)
learning_rates = 10.**np.arange(-5,-1)
hyperparameters_tried = set()
'''

logfile_prefix = os.path.splitext(args.log_file)[0]
logfile_dir = os.path.dirname(args.log_file)

while True:
    random_hidden_size = np.random.choice(hidden_sizes)
    random_embedding_size = np.random.choice(embedding_sizes)
    random_learning_rate = np.random.choice(learning_rates)
    random_num_lstm_layers = np.random.choice(num_layers)
    hyperparameter_combo = (random_hidden_size, random_embedding_size, random_learning_rate, random_num_lstm_layers)
    if hyperparameter_combo in hyperparameters_tried:
        continue
    hyperparameters_tried.add(hyperparameter_combo)
    
    model_suffix = f'hidden_{random_hidden_size}_embedding_{random_embedding_size}_lr_{random_learning_rate}_layers_{random_num_lstm_layers}'
    model_save_name = f'model_{model_suffix}.pth'
    model_save_path = os.path.join(logfile_dir, model_save_name)
    if os.path.isfile(model_save_path):
        continue
    
    model = LSTM_LanguageModel(random_embedding_size, random_hidden_size, lm_min_word_id, max_word_id, num_lstm_layers=random_num_lstm_layers)
    model.cuda()    
    
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=random_learning_rate)
    
    logfile_path = f'{logfile_prefix}_{model_suffix}.txt'
    log_file = open(logfile_path, 'w')
    model_samples_filepath = f'{logfile_prefix}_output_samples_{model_suffix}.txt'
    model_samples_file = open(model_samples_filepath, 'w')
    
    model_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_file.write(f'Model Total Parameters: {model_total_params}\n')
    log_file.write(f'Epoch #,Train Average Loss,Train Perplexity,Validation Average Loss,Validation Perplexity\n')
    log_file.flush()
    model.train()
    for epoch in range(n_epochs):
        train_number_of_words = 0
        batches_loop = tqdm(train_dataloader)
        total_train_loss = 0
        
        for batch_sample in batches_loop:
            model.zero_grad()
            model_inputs = batch_sample['input'].cuda()
            model_targets = batch_sample['target'].cuda()
            
            word_scores = model(model_inputs)
            
            loss = loss_function(torch.transpose(word_scores, 1, 2), model_targets)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                train_number_of_words += np.prod(batch_sample['words'].size())
                total_train_loss += loss
                perplexity = 2.0 ** (total_train_loss / train_number_of_words)
            
            batches_loop.set_description('Train Epoch {}/{}'.format(epoch + 1, n_epochs))
            batches_loop.set_postfix(loss=loss.item(), perplexity=perplexity.item())
        
        valid_number_of_words = 0
        batches_loop = tqdm(valid_dataloader)
        total_valid_loss = 0
        with torch.no_grad():
            avg_train_loss = total_train_loss / train_number_of_words
            train_perplexity = 2.0 ** (total_train_loss / train_number_of_words)
            for batch_sample in batches_loop:
                model_inputs = batch_sample['input'].cuda()
                model_targets = batch_sample['target'].cuda()
                
                word_scores = model(model_inputs)
                
                loss = loss_function(torch.transpose(word_scores, 1, 2), model_targets)
                valid_number_of_words += np.prod(batch_sample['words'].size())
                total_valid_loss += loss
                perplexity = 2.0 ** (total_valid_loss / valid_number_of_words)
                
                batches_loop.set_description('Validation Epoch {}/{}'.format(epoch + 1, n_epochs))
                batches_loop.set_postfix(loss=loss.item(), perplexity=perplexity.item())
            
            avg_valid_loss = total_valid_loss / valid_number_of_words
            valid_perplexity = 2.0 ** (total_valid_loss / valid_number_of_words)
            log_file.write(f'{epoch+1},{avg_train_loss.cpu().numpy()},{train_perplexity.cpu().numpy()},{avg_valid_loss.cpu().numpy()},{valid_perplexity.cpu().numpy()}\n')
            log_file.flush()
            
        sampled_sentence_tokens = model.sample_from_model(max_sequence_length)
        sampled_sentence_tokens_str = ','.join(map(str, sampled_sentence_tokens))
        sampled_sentence = train_dataset.vocabulary.tokens_to_sentence(sampled_sentence_tokens)
        model_samples_file.write(f'Epoch{epoch}\nTokens\n{sampled_sentence_tokens_str}\nWords\n{sampled_sentence}\n\n')
        model_samples_file.flush()
    
    torch.save(model.state_dict(), model_save_path)
