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

torch.manual_seed(1234)
np.random.seed(1234)

parser = argparse.ArgumentParser(description='Train a simple LSTM language model.')
parser.add_argument('weights_file', help='Path to model weights file')
parser.add_argument('train_dataset', help='Path to processed train dataset file')
parser.add_argument('valid_dataset', help='Path to processed train dataset file')
args = parser.parse_args()

train_dataset = TextDataset(args.train_dataset, 50)
valid_dataset = TextDataset(args.valid_dataset, 50)

dataset_transformer = transforms.Compose([Seq2Seq(), ToTensor()])
train_dataset.set_transform(dataset_transformer)
valid_dataset.set_transform(dataset_transformer)

vocabulary_size = train_dataset.vocabulary.get_max_word_id()

model_attributes = get_model_attributes_from_weights_filename(args.weights_file)
model = LSTM_LanguageModel(model_attributes['embedding'], model_attributes['hidden'], vocabulary_size, num_lstm_layers=model_attributes['layers'])
model.load_state_dict(torch.load(args.weights_file))
model.eval()
model.train(False)
model.cuda()

for _ in range(10):
    sampled_tokens = model.sample_from_model(train_dataset.vocabulary, 50)
    sampled_sentence = train_dataset.vocabulary.tokens_to_sentence(sampled_tokens)
    print(sampled_sentence)