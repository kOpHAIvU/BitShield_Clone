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

