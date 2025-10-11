#! /usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init
try:
    from .quantized_layers import quan_Conv1d, CustomBlock, SEBlock, DownsampleA
except ImportError:
    from quantized_layers import quan_Conv1d, CustomBlock, SEBlock, DownsampleA


class ResNetSEBlockIoT(nn.Module):
    """
    ResNet-style architecture with Squeeze-and-Excitation blocks for IoT intrusion detection.
    
    This model uses quantized Conv1d layers and SEBlocks to create an efficient
    network for processing IoT network traffic features. The architecture follows
    a ResNet design pattern with residual connections and attention mechanisms.
    
    Args:
        input_size (int): Number of input features (default: 69 for IoTID20 dataset)
        hidden_sizes (list): List of hidden layer sizes for each stage (default: [32, 64, 128, 256, 512])
        output_size (int): Number of output classes (default: 5)
    """
    
    def __init__(self, input_size=69, hidden_sizes=[32, 64, 128, 256, 512], output_size=5):
        super(ResNetSEBlockIoT, self).__init__()
        
        # Initial convolution layer
        self.fc1 = quan_Conv1d(input_size, hidden_sizes[0], kernel_size=3, stride=1, padding=1)
        self.bn_1 = nn.BatchNorm1d(hidden_sizes[0])

        # ResNet-style stages with SEBlocks
        self.inplanes = 32
        self.stage_1 = self._make_layer(SEBlock, 32, 16, 1)
        self.stage_2 = self._make_layer(SEBlock, 64, 16, 2)
        self.stage_3 = self._make_layer(SEBlock, 128, 16, 2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Final classifier
        self.classifier = CustomBlock(128 * SEBlock.expansion, output_size)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()
        
    def _make_layer(self, block, planes, blocks, stride=1):
        """Create a ResNet layer with the specified block type"""
        downsample = None
        if stride == 2 or self.inplanes != planes * SEBlock.expansion:
            downsample = DownsampleA(self.inplanes, planes * SEBlock.expansion, stride) if stride == 2 else None

        layers = []
        layers.append(block(self.inplanes, planes, stride=1, downsample=downsample))
        self.inplanes = planes * SEBlock.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the network"""
        # Handle input shape: expect [B, features] -> [B, features, 1] for Conv1d
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [B, features] -> [B, features, 1]
        
        # Initial convolution and batch normalization
        x = self.fc1(x)
        x = F.relu(self.bn_1(x), inplace=True)
        
        # ResNet stages
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)

        # Global pooling and classification
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# Alias for backward compatibility
CustomModel = ResNetSEBlockIoT
