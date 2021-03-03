import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

class ConvLSTM_AcousticModel(nn.Module):
    def __init__(self, lstm_hidden_dim, spectral_filter_dims, min_output_word_id, max_word_id, num_lstm_layers=None, num_conv_blocks=None, num_conv_layers_per_block=None):
        super(ConvLSTM_AcousticModel, self).__init__()
        if num_lstm_layers is None:
            num_lstm_layers = 1
        if num_conv_blocks is None:
            num_conv_blocks = 1
        if num_conv_layers_per_block is None:
            num_conv_layers_per_block = 1
        self.num_conv_blocks = num_conv_blocks
        self.num_conv_layers_per_block = num_conv_layers_per_block
        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden_dim = lstm_hidden_dim
        self.max_word_id = max_word_id
        self.min_output_word_id = min_output_word_id
        self.spectral_filter_dims = spectral_filter_dims
        
        cnn_layer_list = []
        input_channels = self.spectral_filter_dims
        for conv_block_num in range(self.num_conv_blocks):
            for conv_layer_num in range(self.num_conv_layers_per_block):
                conv_layer_name = f'conv{conv_block_num}_layer{conv_layer_num}'
                bn_layer_name = f'bn{conv_block_num}_layer{conv_layer_num}'
                gelu_layer_name = f'gelu{conv_block_num}_layer{conv_layer_num}'
                output_channels = int(input_channels * 2)
                cnn_layer_list.append((conv_layer_name, nn.Conv1d(input_channels, output_channels, 3)))
                cnn_layer_list.append((bn_layer_name, nn.BatchNorm1d(output_channels)))
                cnn_layer_list.append((gelu_layer_name, nn.GELU()))
                input_channels = output_channels
            maxpool_layer_name = f'maxpool_{conv_block_num}'
            cnn_layer_list.append((maxpool_layer_name, nn.MaxPool1d(2)))
        
        # The CNN takes in a sequence of spectral inputs, and outputs
        # an embedding of size (BatchSize, spectral_filter_dims * 2**[num_conv_blocks*num_conv_layers_per_block], input_size / (2**num_conv_blocks))
        self.cnn = nn.Sequential(OrderedDict(cnn_layer_list))
        
        # The LSTM takes CNN embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(output_channels, lstm_hidden_dim, num_layers=self.num_lstm_layers)
        
        # The linear layer that maps from hidden state space to word space
        self.hidden2word = nn.Linear(lstm_hidden_dim, self.max_word_id - self.min_output_word_id + 1) # outputs range from [self.min_output_word_id, self.max_word_id]
    
    def forward(self, spectral_input):
        cnn_embeddings = self.cnn(spectral_input)
        cnn_embeddings = torch.transpose(cnn_embeddings, 0, 2)
        cnn_embeddings = torch.transpose(cnn_embeddings, 1, 2)
        lstm_out, _ = self.lstm(cnn_embeddings)
        word_space = self.hidden2word(lstm_out.view(-1, self.lstm_hidden_dim))
        batch_size = lstm_out.size()[0]
        sequence_length = lstm_out.size()[1]
        word_scores = F.log_softmax(word_space.view(batch_size, sequence_length, -1), dim=2)
        return word_scores