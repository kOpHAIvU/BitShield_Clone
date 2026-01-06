#! /usr/bin/env python3
"""
TabularDNN - A true Deep Neural Network (MLP) for IoT Intrusion Detection

This model is a Multi-Layer Perceptron (fully-connected network) designed
for tabular network traffic data. Unlike CNN which uses convolution operations,
DNN/MLP uses only Linear (fully-connected) layers.

Architecture comparison:
- DNN (this model): Linear layers only, no spatial/temporal assumptions
- CNN: Conv1d layers, assumes local patterns in features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from .quantized_layers import quan_Linear, quantize
except ImportError:
    from quantized_layers import quan_Linear, quantize


class quan_Linear_v2(nn.Linear):
    """Quantized Linear layer with 8-bit quantization for DNN"""
    def __init__(self, in_features, out_features, bias=True):
        super(quan_Linear_v2, self).__init__(in_features, out_features, bias=bias)

        self.N_bits = 8
        self.full_lvls = 2**self.N_bits
        self.half_lvls = (self.full_lvls - 2) / 2
        self.step_size = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.__reset_stepsize__()
        self.inf_with_weight = False

        self.b_w = nn.Parameter(2**torch.arange(start=self.N_bits - 1,
                                                end=-1,
                                                step=-1).unsqueeze(-1).float(),
                                requires_grad=False)
        self.b_w[0] = -self.b_w[0]

    def forward(self, input):
        if self.inf_with_weight:
            return F.linear(input, self.weight * self.step_size, self.bias)
        else:
            self.__reset_stepsize__()
            weight_quan = self._quantize(self.weight, self.step_size, self.half_lvls) * self.step_size
            return F.linear(input, weight_quan, self.bias)

    def _quantize(self, input, step_size, half_lvls):
        output = F.hardtanh(input,
                           min_val=-half_lvls * step_size.item(),
                           max_val=half_lvls * step_size.item())
        output = torch.round(output / step_size)
        return output

    def __reset_stepsize__(self):
        with torch.no_grad():
            self.step_size.data = self.weight.abs().max() / self.half_lvls

    def __reset_weight__(self):
        with torch.no_grad():
            self.weight.data = self._quantize(self.weight, self.step_size, self.half_lvls)
        self.inf_with_weight = True


class ResidualBlock(nn.Module):
    """Residual block for DNN with skip connection"""
    def __init__(self, in_features, out_features, dropout=0.3):
        super(ResidualBlock, self).__init__()
        self.fc1 = quan_Linear_v2(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.fc2 = quan_Linear_v2(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection: if dimensions differ, use projection
        self.skip = None
        if in_features != out_features:
            self.skip = nn.Sequential(
                quan_Linear_v2(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        
        if self.skip is not None:
            residual = self.skip(x)
        
        out = F.relu(out + residual, inplace=True)
        return out


class TabularDNN(nn.Module):
    """
    Deep Neural Network (MLP) for IoT Intrusion Detection.
    
    This is a TRUE DNN/MLP architecture using only Linear (fully-connected) layers.
    It includes:
    - Residual connections (similar to ResNet but for MLP)
    - Batch Normalization
    - Dropout for regularization
    - 8-bit Quantization support
    
    Args:
        input_size (int): Number of input features (default: 69 for IoTID20)
        hidden_sizes (list): Hidden layer sizes (default: [256, 512, 256, 128])
        output_size (int): Number of output classes (default: 5)
        dropout (float): Dropout rate (default: 0.3)
        use_residual (bool): Use residual connections (default: True)
    """
    
    def __init__(self, input_size=69, hidden_sizes=[256, 512, 256, 128], 
                 output_size=5, dropout=0.3, use_residual=True):
        super(TabularDNN, self).__init__()
        
        self.use_residual = use_residual
        
        # Input layer
        self.input_layer = nn.Sequential(
            quan_Linear_v2(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Hidden layers with optional residual connections
        self.hidden_layers = nn.ModuleList()
        
        if use_residual:
            # Use Residual Blocks
            for i in range(len(hidden_sizes) - 1):
                self.hidden_layers.append(
                    ResidualBlock(hidden_sizes[i], hidden_sizes[i+1], dropout)
                )
        else:
            # Simple fully-connected layers
            for i in range(len(hidden_sizes) - 1):
                self.hidden_layers.append(nn.Sequential(
                    quan_Linear_v2(hidden_sizes[i], hidden_sizes[i+1]),
                    nn.BatchNorm1d(hidden_sizes[i+1]),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout)
                ))
        
        # Output classifier
        self.classifier = quan_Linear_v2(hidden_sizes[-1], output_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Forward pass through the network"""
        # Ensure input is 2D: [B, features]
        if x.dim() == 3:
            x = x.squeeze(-1)  # [B, features, 1] -> [B, features]
        
        # Input layer
        x = self.input_layer(x)
        
        # Hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
        
        # Classifier
        x = self.classifier(x)
        return x
    
    def get_layer_names(self):
        """Get names of all layers for CIG tracking"""
        names = ['input_layer']
        for i in range(len(self.hidden_layers)):
            names.append(f'hidden_layer_{i}')
        names.append('classifier')
        return names


# Alias for compatibility
DNNModel = TabularDNN


if __name__ == "__main__":
    # Test the model
    model = TabularDNN(input_size=69, output_size=5)
    print("TabularDNN Architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    x = torch.randn(32, 69)
    y = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
