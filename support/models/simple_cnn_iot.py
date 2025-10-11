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
    
    def __init__(self, input_size=69, hidden_sizes=[32, 64, 128, 100], output_size=5):
        super(SimpleCNNIoT, self).__init__()
        self.hidden_sizes = hidden_sizes

        # Define layers
        self.fc1 = nn.Conv1d(input_size, hidden_sizes[0], kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        self.stage_1 = nn.Conv1d(hidden_sizes[0], hidden_sizes[1], kernel_size=3, stride=2, padding=1)
        self.stage_2 = nn.Conv1d(hidden_sizes[1], hidden_sizes[2], kernel_size=3, stride=2, padding=1)
        self.stage_3 = nn.Conv1d(hidden_sizes[2], hidden_sizes[3], kernel_size=3, stride=2, padding=1)

        # Global Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classifier
        self.classifier = CustomBlock(hidden_sizes[-1], output_size, apply_softmax=True)
        nn.Dropout(0.15)
        
    def forward(self, x):
        """Forward pass through the network"""
        # Handle input shape: expect [B, features] -> [B, features, 1] for Conv1d
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [B, features] -> [B, features, 1]
        
        # Pass through layers
        x = self.fc1(x)
        x = self.activation(self.pool(x))

        x = self.stage_1(x)
        x = self.activation(self.pool(x))

        x = self.stage_2(x)
        x = self.activation(self.pool(x))

        x = self.stage_3(x)
        x = self.activation(self.pool(x))

        # Global Pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.classifier(x)


# Alias for backward compatibility
CustomModel2 = SimpleCNNIoT
