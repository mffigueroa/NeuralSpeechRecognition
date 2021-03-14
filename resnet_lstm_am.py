import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from resnet import resnet_1d

class GroupNormWrapper(nn.Module):
    def __init__(self, num_channels):
        super(GroupNormWrapper, self).__init__()
        self.norm = nn.GroupNorm(num_groups=2, num_channels=num_channels,
                                 eps=1e-5, affine=True)
    
    def forward(self, x):
        x = self.norm(x)
        return x
    
    @property
    def weight(self):
        return self.norm.weight
        
    @property
    def bias(self):
        return self.norm.bias

class ResNetLSTM_AcousticModel(nn.Module):
    def __init__(self, lstm_hidden_dim, spectral_filter_dims, min_output_word_id, max_word_id, num_lstm_layers=None, resnet_backbone_name=None, bottleneck_size=None):
        super(ResNetLSTM_AcousticModel, self).__init__()
        if num_lstm_layers is None:
            num_lstm_layers = 1
        if resnet_backbone_name is None:
            resnet_backbone_name = 'resnet18'
        if bottleneck_size is None:
            bottleneck_size = 4
        
        resnet_backbone_name_to_constructor = {
            'resnet18' : resnet_1d.resnet18
        }
        resnet_backbone = resnet_backbone_name_to_constructor[resnet_backbone_name]
        
        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden_dim = lstm_hidden_dim
        self.bottleneck_size = bottleneck_size
        self.max_word_id = max_word_id
        self.min_output_word_id = min_output_word_id
        self.spectral_filter_dims = spectral_filter_dims
        
        input_channels = self.spectral_filter_dims
        self.backbone = resnet_backbone(include_top=False, norm_layer=GroupNormWrapper, input_channels=input_channels, zero_init_residual=True)
        
        # The LSTM takes CNN embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(self.backbone.get_output_channels(), self.lstm_hidden_dim, num_layers=self.num_lstm_layers)
        if self.bottleneck_size > 0:
            self.lstm2bottleneck = nn.Linear(self.lstm_hidden_dim, self.bottleneck_size)
            bottleneck2word_input_size = self.bottleneck_size
        else:
            self.lstm2bottleneck = None
            bottleneck2word_input_size = self.lstm_hidden_dim
        
        # The linear layer that maps from hidden state space to word space
        self.bottleneck2word = nn.Linear(bottleneck2word_input_size, self.max_word_id - self.min_output_word_id + 1) # outputs range from [self.min_output_word_id, self.max_word_id]
    
    def forward(self, spectral_input):
        resnet_embeddings = self.backbone(spectral_input)
        
        resnet_output_seq_length = resnet_embeddings.size()[-1]
        spectral_input_seq_length = spectral_input.size()[-1]
        #print(f'ResNet input length: {spectral_input_seq_length} -> {resnet_output_seq_length}')
        
        lstm_input = torch.transpose(resnet_embeddings, 0, 2)
        lstm_input = torch.transpose(lstm_input, 1, 2)
        lstm_out, _ = self.lstm(lstm_input)
        batch_size = lstm_out.size()[0]
        sequence_length = lstm_out.size()[1]
        
        lstm_out = lstm_out.view(-1, self.lstm_hidden_dim)
        
        if self.bottleneck_size > 0:
            bottleneck = self.lstm2bottleneck(lstm_out)
        else:
            bottleneck = lstm_out
        
        word_space = self.bottleneck2word(bottleneck)
        word_scores = F.log_softmax(word_space.view(batch_size, sequence_length, -1), dim=2)
        return word_scores