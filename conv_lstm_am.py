import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

class ConvLSTM_AcousticModel(nn.Module):
    def __init__(self, lstm_hidden_dim, spectral_filter_dims, min_output_word_id, max_word_id, num_lstm_layers=None, num_conv_blocks=None, num_conv_layers_per_block=None, bottleneck_size=None):
        super(ConvLSTM_AcousticModel, self).__init__()
        if num_lstm_layers is None:
            num_lstm_layers = 1
        if num_conv_blocks is None:
            num_conv_blocks = 1
        if num_conv_layers_per_block is None:
            num_conv_layers_per_block = 1
        if bottleneck_size is None:
            bottleneck_size = 4
        self.num_conv_blocks = num_conv_blocks
        self.num_conv_layers_per_block = num_conv_layers_per_block
        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden_dim = lstm_hidden_dim
        self.bottleneck_size = bottleneck_size
        self.max_word_id = max_word_id
        self.min_output_word_id = min_output_word_id
        self.spectral_filter_dims = spectral_filter_dims
        
        input_channels = self.spectral_filter_dims
        if self.num_conv_blocks > 0:
            cnn_layer_list = []
            for conv_block_num in range(self.num_conv_blocks):
                for conv_layer_num in range(self.num_conv_layers_per_block):
                    conv_layer_name = f'conv{conv_block_num}_layer{conv_layer_num}'
                    bn_layer_name = f'bn{conv_block_num}_layer{conv_layer_num}'
                    gelu_layer_name = f'gelu{conv_block_num}_layer{conv_layer_num}'
                    output_channels = int(input_channels * 2)
                    output_channels = min(output_channels, 4096)
                    cnn_layer_list.append((conv_layer_name, nn.Conv1d(input_channels, output_channels, 3)))
                    cnn_layer_list.append((bn_layer_name, nn.BatchNorm1d(output_channels)))
                    cnn_layer_list.append((gelu_layer_name, nn.GELU()))
                    input_channels = output_channels
                maxpool_layer_name = f'maxpool_{conv_block_num}'
                cnn_layer_list.append((maxpool_layer_name, nn.MaxPool1d(2)))
            
            # The CNN takes in a sequence of spectral inputs, and outputs
            # an embedding of size (BatchSize, spectral_filter_dims * 2**[num_conv_blocks*num_conv_layers_per_block], input_size / (2**num_conv_blocks))
            self.cnn = nn.Sequential(OrderedDict(cnn_layer_list))
        else:
            self.cnn = None
            output_channels = input_channels
        
        # The LSTM takes CNN embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(output_channels, self.lstm_hidden_dim, num_layers=self.num_lstm_layers)
        
        if self.bottleneck_size > 0:
            self.lstm2bottleneck = nn.Linear(self.lstm_hidden_dim, self.bottleneck_size)
        else:
            self.lstm2bottleneck = None
            self.bottleneck_size = self.lstm_hidden_dim
        
        # The linear layer that maps from hidden state space to word space
        self.bottleneck2word = nn.Linear(self.bottleneck_size, self.max_word_id - self.min_output_word_id + 1) # outputs range from [self.min_output_word_id, self.max_word_id]
    
    def forward(self, spectral_input):
        if self.num_conv_blocks > 0:
            cnn_embeddings = self.cnn(spectral_input)
            lstm_input = cnn_embeddings
        else:
            lstm_input = spectral_input
        lstm_input = torch.transpose(lstm_input, 0, 2)
        lstm_input = torch.transpose(lstm_input, 1, 2)
        lstm_out, _ = self.lstm(lstm_input)
        lstm_out = lstm_out.view(-1, self.lstm_hidden_dim)
        
        if self.bottleneck_size > 0:
            bottleneck = self.lstm2bottleneck(lstm_out)
        else:
            bottleneck = lstm_out
        
        word_space = self.bottleneck2word(bottleneck)
        batch_size = lstm_out.size()[0]
        sequence_length = lstm_out.size()[1]
        word_scores = F.log_softmax(word_space.view(batch_size, sequence_length, -1), dim=2)
        return word_scores