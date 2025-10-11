import torch
import torch.nn as nn
import torch.nn.functional as F


class _quantize_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, step_size, half_lvls):
        ctx.step_size = step_size
        ctx.half_lvls = half_lvls
        output = F.hardtanh(input,
                            min_val=-ctx.half_lvls * ctx.step_size.item(),
                            max_val=ctx.half_lvls * ctx.step_size.item())
        output = torch.round(output / ctx.step_size)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone() / ctx.step_size
        return grad_input, None, None


quantize = _quantize_func.apply


class CustomBlock(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.N_bits = 8
        self.full_lvls = 2 ** self.N_bits
        self.half_lvls = (self.full_lvls - 2) / 2
        self.step_size = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.__reset_stepsize__()
        self.inf_with_weight = False
        self.b_w = nn.Parameter(2 ** torch.arange(start=self.N_bits - 1,
                                                  end=-1,
                                                  step=-1).unsqueeze(-1).float(),
                                requires_grad=False)
        self.b_w[0] = -self.b_w[0]
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def __reset_stepsize__(self):
        with torch.no_grad():
            self.step_size.data = self.weight.abs().max() / self.half_lvls if self.weight.numel() else torch.tensor(1.)

    def __reset_weight__(self):
        with torch.no_grad():
            self.weight.data = quantize(self.weight, self.step_size, self.half_lvls)
        self.inf_with_weight = True

    def forward(self, input):
        if self.inf_with_weight:
            weight_applied = self.weight * self.step_size
        else:
            self.__reset_stepsize__()
            weight_quan = quantize(self.weight, self.step_size, self.half_lvls) * self.step_size
            weight_applied = weight_quan
        return F.softmax(input @ weight_applied.T, dim=-1)


class quan_Conv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.N_bits = 8
        self.full_lvls = 2 ** self.N_bits
        self.half_lvls = (self.full_lvls - 2) / 2
        self.step_size = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.__reset_stepsize__()
        self.inf_with_weight = False
        self.b_w = nn.Parameter(2 ** torch.arange(start=self.N_bits - 1, end=-1, step=-1).unsqueeze(-1).float(), requires_grad=False)
        self.b_w[0] = -self.b_w[0]

    def __reset_stepsize__(self):
        self.step_size.data.fill_(1.0)

    def forward(self, x):
        if self.inf_with_weight:
            quantized_weight = self.quantize_weight(self.weight)
            return F.conv1d(x, quantized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            return F.conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def quantize_weight(self, weight):
        quantized_weight = torch.round(weight / self.step_size) * self.step_size
        quantized_weight = torch.clamp(quantized_weight, -self.half_lvls * self.step_size, (self.half_lvls - 1) * self.step_size)
        return quantized_weight


class CustomModel1(nn.Module):
    def __init__(self, input_size=69, hidden_size1=32, hidden_size2=64, hidden_size3=128, hidden_size4=100, output_size=5):
        super().__init__()
        self.fc1 = quan_Conv1d(input_size, hidden_size1, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.stage_1 = quan_Conv1d(hidden_size1, hidden_size2, kernel_size=3, stride=2, padding=1)
        self.stage_1_1 = quan_Conv1d(hidden_size2, hidden_size2, kernel_size=3, stride=2, padding=1)
        self.stage_2 = quan_Conv1d(hidden_size2, hidden_size3, kernel_size=3, stride=2, padding=1)
        self.stage_2_1 = quan_Conv1d(hidden_size3, hidden_size3, kernel_size=3, stride=2, padding=1)
        self.stage_3 = quan_Conv1d(hidden_size3, hidden_size4, kernel_size=3, stride=2, padding=1)
        self.stage_3_1 = quan_Conv1d(hidden_size4, hidden_size4, kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = CustomBlock(hidden_size4, output_size)

    def forward(self, x):
        # Expect x shape [B, 69] or [B, 69, 1]
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = self.fc1(x)
        x = F.relu(self.pool(x), inplace=True)
        x = self.stage_1(x)
        x = F.relu(self.pool(x), inplace=True)
        x = self.stage_1_1(x)
        x = F.relu(self.pool(x), inplace=True)
        x = self.stage_2(x)
        x = F.relu(self.pool(x), inplace=True)
        x = self.stage_2_1(x)
        x = F.relu(self.pool(x), inplace=True)
        x = self.stage_3(x)
        x = F.relu(self.pool(x), inplace=True)
        x = self.stage_3_1(x)
        x = F.relu(self.pool(x), inplace=True)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.classifier(x)


class CustomModel2(CustomModel1):
    def __init__(self, input_size=69, output_size=5):
        # a slightly larger variant
        super().__init__(input_size=input_size, hidden_size1=64, hidden_size2=96, hidden_size3=160, hidden_size4=128, output_size=output_size)

class PureCNN(nn.Module):
    """Pure CNN model without quantization for comparison"""
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


class EfficientCNN(nn.Module):
    """Lightweight CNN with depthwise separable convolutions"""
    def __init__(self, input_size=69, output_size=5):
        super().__init__()

        # Depthwise separable convolution blocks
        self.conv1 = nn.Conv1d(input_size, input_size, kernel_size=3, stride=1, padding=1, groups=input_size)
        self.pw_conv1 = nn.Conv1d(input_size, 64, kernel_size=1)
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
        # Handle input shape
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        if x.size(-1) == 1:
            x = x.transpose(1, 2)

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


