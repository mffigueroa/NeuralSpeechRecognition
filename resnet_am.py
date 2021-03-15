import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from resnet import resnet_1d

class GroupNormWrapper(nn.Module):
    def __init__(self, num_channels):
        super(GroupNormWrapper, self).__init__()
        self.norm = nn.GroupNorm(num_groups=8, num_channels=num_channels,
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

class ResNet_AcousticModel(nn.Module):
    def __init__(self, spectral_filter_dims, min_output_word_id, max_word_id, resnet_backbone_name=None, bottleneck_size=None):
        super(ResNet_AcousticModel, self).__init__()
        if resnet_backbone_name is None:
            resnet_backbone_name = 'resnet50'
        if bottleneck_size is None:
            bottleneck_size = 32
        
        resnet_backbone_name_to_constructor = {
            'resnet18' : resnet_1d.resnet18,
            'resnet50' : resnet_1d.resnet50,
            'resnext101' : resnet_1d.resnext101_32x8d
        }
        resnet_backbone = resnet_backbone_name_to_constructor[resnet_backbone_name]
        
        self.bottleneck_size = bottleneck_size
        self.max_word_id = max_word_id
        self.min_output_word_id = min_output_word_id
        self.spectral_filter_dims = spectral_filter_dims
        
        input_channels = self.spectral_filter_dims
        self.backbone = resnet_backbone(include_top=False, norm_layer=GroupNormWrapper, input_channels=input_channels, zero_init_residual=True, replace_stride_with_dilation=[True,True,True,True])
        self.backbone_output_channels = self.backbone.get_output_channels()
        
        if self.bottleneck_size > 0:
            self.backbone2bottleneck = nn.Linear(self.backbone_output_channels, self.bottleneck_size)
            bottleneck2word_input_size = self.bottleneck_size
        else:
            self.backbone2bottleneck = None
            bottleneck2word_input_size = self.backbone.get_output_channels()
        
        # The linear layer that maps from hidden state space to word space
        self.bottleneck2word = nn.Linear(bottleneck2word_input_size, self.max_word_id - self.min_output_word_id + 1) # outputs range from [self.min_output_word_id, self.max_word_id]
    
    def forward(self, spectral_input):
        resnet_output = self.backbone(spectral_input)
        #print(f'spectral_input: {spectral_input.size()} -> resnet_output: {resnet_output.size()}')
        
        bottleneck_input = torch.transpose(resnet_output, 0, 2)
        bottleneck_input = torch.transpose(bottleneck_input, 1, 2)
        
        sequence_length = bottleneck_input.size()[0]
        batch_size = bottleneck_input.size()[1]
        
        bottleneck_input = bottleneck_input.reshape(-1, self.backbone_output_channels)
        
        if self.bottleneck_size > 0:
            bottleneck = self.backbone2bottleneck(bottleneck_input)
        else:
            bottleneck = bottleneck_input
        
        word_space = self.bottleneck2word(bottleneck)
        word_scores = F.log_softmax(word_space.view(sequence_length, batch_size, -1), dim=2)
        return word_scores