#! /usr/bin/env python3
"""
ResNetSEBlockIoT - Deep Neural Network with Residual Connections for IoT Intrusion Detection

UPDATED: Changed from CNN (Conv1d) to DNN (Linear layers) architecture.
This is now a TRUE DNN/MLP with SE-style attention and residual connections.

Key differences from CNN version:
- Uses Linear layers instead of Conv1d
- No spatial assumptions - treats input as flat feature vector
- SE attention now works on feature dimensions, not channels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init

try:
    from .quantized_layers import CustomBlock
except ImportError:
    from quantized_layers import CustomBlock


class quan_Linear(nn.Linear):
    """Quantized Linear layer with 8-bit quantization"""
    def __init__(self, in_features, out_features, bias=True):
        super(quan_Linear, self).__init__(in_features, out_features, bias=bias)

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


class SEBlockLinear(nn.Module):
    """
    Squeeze-and-Excitation Block using Linear layers (DNN version).
    
    This is the DNN equivalent of SEBlock, using fully-connected layers
    instead of convolutions. Includes residual connection.
    """
    expansion = 1

    def __init__(self, in_features, out_features, dropout=0.3, downsample=None):
        super(SEBlockLinear, self).__init__()
        
        # Main path: two Linear layers with BN and dropout
        self.fc_a = quan_Linear(in_features, out_features)
        self.bn_a = nn.BatchNorm1d(out_features)
        self.dropout_a = nn.Dropout(p=dropout)

        self.fc_b = quan_Linear(out_features, out_features)
        self.bn_b = nn.BatchNorm1d(out_features)
        self.dropout_b = nn.Dropout(p=dropout)

        # SE attention mechanism (squeeze-excitation for features)
        self.se_fc1 = nn.Linear(out_features, out_features // 4)
        self.se_fc2 = nn.Linear(out_features // 4, out_features)

        # Skip connection
        self.downsample = downsample
        if in_features != out_features and downsample is None:
            self.downsample = nn.Sequential(
                quan_Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )

    def forward(self, x):
        residual = x
        
        # Main path
        out = self.fc_a(x)
        out = self.bn_a(out)
        out = F.relu(out, inplace=True)
        out = self.dropout_a(out)

        out = self.fc_b(out)
        out = self.bn_b(out)
        out = self.dropout_b(out)

        # SE attention: squeeze (global) -> excite (per-feature weights)
        se = out.mean(dim=0, keepdim=True) if out.dim() > 1 else out  # Global squeeze
        se = F.relu(self.se_fc1(out))
        se = torch.sigmoid(self.se_fc2(se))
        out = out * se  # Scale features by attention weights

        # Residual connection
        if self.downsample is not None:
            residual = self.downsample(x)

        return F.relu(residual + out, inplace=True)


class ResNetSEBlockIoT(nn.Module):
    """
    ResNet-style DNN with Squeeze-and-Excitation for IoT Intrusion Detection.
    
    ARCHITECTURE TYPE: DNN (Deep Neural Network / Multi-Layer Perceptron)
    
    This model uses quantized Linear layers and SE-style attention blocks.
    Unlike CNN, it treats input features as a flat vector without assuming
    any spatial or temporal structure.
    
    Key features:
    - Fully-connected (Linear) layers only - TRUE DNN architecture
    - Residual connections for training deep networks
    - SE attention for feature re-weighting
    - 8-bit quantization support
    
    Args:
        input_size (int): Number of input features (default: 69 for IoTID20)
        hidden_sizes (list): Hidden layer sizes for each stage
        output_size (int): Number of output classes (default: 5)
    """
    
    def __init__(self, input_size=69, hidden_sizes=[128, 256, 128], output_size=5):
        super(ResNetSEBlockIoT, self).__init__()
        
        # Initial linear layer (replaces initial Conv1d)
        self.fc1 = quan_Linear(input_size, hidden_sizes[0])
        self.bn_1 = nn.BatchNorm1d(hidden_sizes[0])
        self.dropout_1 = nn.Dropout(0.2)

        # DNN stages with SEBlockLinear (replaces Conv-based stages)
        self.stage_1 = self._make_layer(hidden_sizes[0], hidden_sizes[0], num_blocks=4)
        self.stage_2 = self._make_layer(hidden_sizes[0], hidden_sizes[1], num_blocks=4)
        self.stage_3 = self._make_layer(hidden_sizes[1], hidden_sizes[2], num_blocks=4)

        # Final classifier using CustomBlock (16-bit quantization, bit-flip support)
        self.classifier = CustomBlock(hidden_sizes[2], output_size, apply_softmax=False)

        # Initialize weights
        self._init_weights()
        
    def _make_layer(self, in_features, out_features, num_blocks):
        """Create a stage with multiple SEBlockLinear layers"""
        layers = []
        
        # First block may need dimension change
        layers.append(SEBlockLinear(in_features, out_features))
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(SEBlockLinear(out_features, out_features))

        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization"""
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
        # Handle input shape: ensure [B, features] (2D)
        if x.dim() == 3:
            x = x.squeeze(-1)  # [B, features, 1] -> [B, features]
        
        # Initial layer
        x = self.fc1(x)
        x = self.bn_1(x)
        x = F.relu(x, inplace=True)
        x = self.dropout_1(x)
        
        # DNN stages with residual connections
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)

        # Classification
        return self.classifier(x)


# Alias for backward compatibility
CustomModel = ResNetSEBlockIoT


if __name__ == "__main__":
    # Test the model
    print("="*60)
    print("ResNetSEBlockIoT - DNN Architecture")
    print("="*60)
    
    model = ResNetSEBlockIoT(input_size=69, output_size=5)
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    import torch
    x = torch.randn(32, 69)
    y = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    print("\n" + "="*60)
    print("✅ This is a TRUE DNN (uses Linear layers, not Conv1d)")
    print("="*60)
