#! /usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .quantized_layers import CustomBlock
except ImportError:
    from quantized_layers import CustomBlock


class SimpleCNNIoT(nn.Module):
    """
    Simple CNN architecture for IoT intrusion detection.
    
    This model uses a straightforward CNN design with multiple convolutional
    stages followed by adaptive pooling and a custom quantized classifier.
    The architecture is designed to be lightweight while maintaining good
    performance on IoT network traffic classification tasks.
    
    Args:
        input_size (int): Number of input features (default: 69 for IoTID20 dataset)
        hidden_sizes (list): List of hidden layer sizes for each stage (default: [32, 64, 128, 100])
        output_size (int): Number of output classes (default: 5)
    """
    
    def __init__(self, input_size=69, hidden_sizes=[64, 128, 256, 128], output_size=5):
        super(SimpleCNNIoT, self).__init__()
        self.hidden_sizes = hidden_sizes

        # Define layers
        # Stage 1
        self.conv1 = nn.Conv1d(input_size, hidden_sizes[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        # Note: Removed MaxPool because input is treated as [B, C, 1] (Length=1), 
        # so spatial pooling is impossible/redundant.
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        # Stage 2
        self.stage_1 = nn.Conv1d(hidden_sizes[0], hidden_sizes[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        
        # Stage 3
        self.stage_2 = nn.Conv1d(hidden_sizes[1], hidden_sizes[2], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        
        # Stage 4
        self.stage_3 = nn.Conv1d(hidden_sizes[2], hidden_sizes[3], kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(hidden_sizes[3])

        # Global Pooling (Identity if length is 1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classifier
        self.classifier = CustomBlock(hidden_sizes[-1], output_size, apply_softmax=True)
        
    def forward(self, x):
        """Forward pass through the network"""
        # Handle input shape: expect [B, features] -> [B, features, 1] for Conv1d
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [B, features] -> [B, features, 1]
        
        # Pass through layers
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        # Block 2
        x = self.stage_1(x)
        x = self.bn2(x)
        x = self.activation(x)

        # Block 3
        x = self.stage_2(x)
        x = self.bn3(x)
        x = self.activation(x)

        # Block 4
        x = self.stage_3(x)
        x = self.bn4(x)
        x = self.activation(x)

        # Global Pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.classifier(x)


# Alias for backward compatibility
CustomModel2 = SimpleCNNIoT
