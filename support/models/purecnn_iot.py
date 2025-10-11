#! /usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F


class PureCNN(nn.Module):
    """
    Pure CNN model without quantization for IoT intrusion detection baseline comparison.
    
    This model provides a baseline CNN architecture without any quantization
    techniques, making it suitable for comparison with quantized models.
    The architecture uses standard Conv1d layers with batch normalization,
    max pooling, and dropout for regularization.
    
    Args:
        input_size (int): Number of input features (default: 69 for IoTID20 dataset)
        output_size (int): Number of output classes (default: 5)
    """
    
    def __init__(self, input_size=69, output_size=5):
        super().__init__()

        # Convolutional layers with increasing channels
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Adaptive pooling to handle variable input lengths
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(0.2)
        self.classifier = nn.Linear(128, output_size)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """Forward pass through the network"""
        # Handle input shape: expect [B, 69] or [B, 69, 1]
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [B, 69] -> [B, 69, 1]

        # If input is [B, 69, 1], transpose to [B, 1, 69] for Conv1d
        if x.size(-1) == 1:
            x = x.transpose(1, 2)  # [B, 69, 1] -> [B, 1, 69]

        # Convolutional blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)

        # Global average pooling
        x = self.adaptive_pool(x)  # [B, 512, 1]
        x = x.view(x.size(0), -1)  # [B, 512]

        # Fully connected layers
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))

        x = self.dropout2(x)
        x = F.relu(self.fc2(x))

        x = self.dropout3(x)
        x = self.classifier(x)
        return x
