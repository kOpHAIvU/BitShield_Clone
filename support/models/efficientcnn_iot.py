#! /usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientCNN(nn.Module):
    """
    Lightweight CNN with depthwise separable convolutions for IoT intrusion detection.
    
    This model uses depthwise separable convolutions to create an efficient
    network architecture that reduces computational complexity while maintaining
    good performance. The design is particularly suitable for resource-constrained
    IoT environments.
    
    Args:
        input_size (int): Number of input features (default: 69 for IoTID20 dataset)
        output_size (int): Number of output classes (default: 5)
    """
    
    def __init__(self, input_size=69, output_size=5):
        super().__init__()

        # Depthwise separable convolution blocks
        self.conv1 = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1, groups=1)
        self.pw_conv1 = nn.Conv1d(1, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, groups=64)
        self.pw_conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1, groups=128)
        self.pw_conv3 = nn.Conv1d(128, 256, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Global pooling and classification
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(256, output_size)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize with Xavier uniform"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """Forward pass through the network"""
        # Handle input shape: expect [B, features] or [B, 1, features]
        if x.dim() == 2:
            # Input is [B, features], add channel dimension
            x = x.unsqueeze(1)  # [B, features] -> [B, 1, features]
        elif x.dim() == 3 and x.size(1) == 1:
            # Input is already [B, 1, features], keep as is
            pass
        else:
            # Unexpected shape, try to handle
            if x.dim() == 3 and x.size(-1) == 1:
                # Input is [B, features, 1], transpose to [B, 1, features]
                x = x.transpose(1, 2)  # [B, features, 1] -> [B, 1, features]
            else:
                raise ValueError(f"Unexpected input shape: {x.shape}")

        # Depthwise separable blocks
        x = F.relu(self.bn1(self.pw_conv1(self.conv1(x))))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.pw_conv2(self.conv2(x))))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.pw_conv3(self.conv3(x))))
        x = self.pool3(x)

        # Global pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        x = self.dropout(x)
        return self.classifier(x)
