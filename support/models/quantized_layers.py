#! /usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init


class quan_Linear(nn.Linear):
    """Quantized Linear layer with 8-bit quantization"""
    def __init__(self, in_features, out_features, bias=True):
        super(quan_Linear, self).__init__(in_features, out_features, bias=bias)

        self.N_bits = 8
        self.full_lvls = 2**self.N_bits
        self.half_lvls = (self.full_lvls - 2) / 2
        # Initialize the step size
        self.step_size = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.__reset_stepsize__()
        # flag to enable the inference with quantized weight or self.weight
        self.inf_with_weight = False  # disabled by default

        # create a vector to identify the weight to each bit
        self.b_w = nn.Parameter(2**torch.arange(start=self.N_bits - 1,
                                                end=-1,
                                                step=-1).unsqueeze(-1).float(),
                                requires_grad=False)

        self.b_w[0] = -self.b_w[0]  #in-place reverse

    def forward(self, input):
        if self.inf_with_weight:
            return F.linear(input, self.weight * self.step_size, self.bias)
        else:
            self.__reset_stepsize__()
            weight_quan = quantize(self.weight, self.step_size,
                                   self.half_lvls) * self.step_size
            return F.linear(input, weight_quan, self.bias)

    def __reset_stepsize__(self):
        with torch.no_grad():
            self.step_size.data = self.weight.abs().max() / self.half_lvls

    def __reset_weight__(self):
        '''
        This function will reconstruct the weight stored in self.weight.
        Replacing the orginal floating-point with the quantized fix-point
        weight representation.
        '''
        # replace the weight with the quantized version
        with torch.no_grad():
            self.weight.data = quantize(self.weight, self.step_size,
                                        self.half_lvls)
        # enable the flag, thus now computation does not invovle weight quantization
        self.inf_with_weight = True


class quan_Conv1d(nn.Conv1d):
    """Quantized 1D Convolution layer with 8-bit quantization"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(quan_Conv1d, self).__init__(in_channels,
                                          out_channels,
                                          kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=bias)

        # Số lượng bit để lượng tử hóa trọng số
        self.N_bits = 8
        self.full_lvls = 2 ** self.N_bits
        self.half_lvls = (self.full_lvls - 2) / 2

        # Bước lượng tử hóa (step size), là một tham số có thể học được
        self.step_size = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.__reset_stepsize__()

        # Cờ để bật hoặc tắt sử dụng trọng số lượng tử hóa
        self.inf_with_weight = False  # Tắt theo mặc định

        # Tạo một vector để biểu diễn trọng số cho từng bit
        self.b_w = nn.Parameter(2 ** torch.arange(start=self.N_bits - 1,
                                                  end=-1,
                                                  step=-1).unsqueeze(-1).float(),
                                requires_grad=False)
        self.b_w[0] = -self.b_w[0]  # Biến đổi MSB thành giá trị âm để hỗ trợ bù hai

    def __reset_stepsize__(self):
        """Hàm này dùng để đặt lại giá trị `step_size`."""
        # Giá trị này có thể được tùy chỉnh tùy thuộc vào yêu cầu của mô hình
        self.step_size.data.fill_(1.0)

    def forward(self, x):
        # Kiểm tra cờ `inf_with_weight` để quyết định sử dụng trọng số đã lượng tử hóa hay không
        if self.inf_with_weight:
            quantized_weight = self.quantize_weight(self.weight)
            return nn.functional.conv1d(x, quantized_weight, self.bias, self.stride,
                                        self.padding, self.dilation, self.groups)
        else:
            return nn.functional.conv1d(x, self.weight, self.bias, self.stride,
                                        self.padding, self.dilation, self.groups)

    def quantize_weight(self, weight):
        """Lượng tử hóa trọng số theo số bit đã định."""
        # Tạo trọng số lượng tử hóa bằng cách sử dụng step_size
        quantized_weight = torch.round(weight / self.step_size) * self.step_size
        quantized_weight = torch.clamp(quantized_weight, -self.half_lvls * self.step_size,
                                       (self.half_lvls - 1) * self.step_size)
        return quantized_weight


def change_quan_bitwidth(model, n_bit):
    '''This script change the quantization bit-width of entire model to n_bit'''
    for m in model.modules():
        if isinstance(m, quan_Conv1d) or isinstance(m, quan_Linear):
            m.N_bits = n_bit
            # print("Change weight bit-width as {}.".format(m.N_bits))
            m.b_w.data = m.b_w.data[-m.N_bits:]
            m.b_w[0] = -m.b_w[0]
            print(m.b_w)
    return 


class _quantize_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, step_size, half_lvls):
        # ctx is a context object that can be used to stash information
        # for backward computation
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


class _bin_func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, mu):
        ctx.mu = mu
        output = input.clone().zero_()
        output[input.ge(0)] = 1
        output[input.lt(0)] = -1

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone() / ctx.mu
        return grad_input, None

w_bin = _bin_func.apply


class CustomBlock(nn.Module):
    """Custom quantized block for IoT models"""
    def __init__(self, in_features, out_features, bias=True, apply_softmax=False):
        super(CustomBlock, self).__init__()
        self.N_bits = 16
        self.full_lvls = 2 ** self.N_bits
        self.half_lvls = (self.full_lvls - 2) / 2
        self.apply_softmax = apply_softmax

        # Initialize the step size
        self.step_size = nn.Parameter(torch.Tensor([1]), requires_grad=True)

        # Initialize weights and bias
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None

        # Reset parameters
        self.__reset_stepsize__()
        self.reset_parameters()

        # Flag for inference with quantized weights
        self.inf_with_weight = False

        self.b_w = nn.Parameter(2**torch.arange(start=self.N_bits - 1,
                                             end=-1,
                                             step=-1).unsqueeze(-1).float(),
                           requires_grad=False)
        self.b_w[0] = -self.b_w[0]  #in-place reverse
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        if self.inf_with_weight:
            weight_applied = self.weight * self.step_size
        else:
            self.__reset_stepsize__()
            weight_quan = quantize(self.weight, self.step_size, self.half_lvls) * self.step_size
            weight_applied = weight_quan

        # Linear transformation
        input = input.view(input.size(0), -1)  # Flatten input to 2D for matmul
        output = input @ weight_applied.T
        if self.bias is not None:
            output += self.bias

        if self.apply_softmax:
            output = F.softmax(output, dim=-1)
        return output

    def __reset_stepsize__(self):
        with torch.no_grad():
            self.step_size.data = self.weight.abs().max() / self.half_lvls

    def __reset_weight__(self):
        with torch.no_grad():
            self.weight.data = quantize(self.weight, self.step_size, self.half_lvls)
        self.inf_with_weight = True


class DownsampleA(nn.Module):
    """Downsampling layer for ResNet-style architectures"""
    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool1d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)
        

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for ResNet architectures"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SEBlock, self).__init__()
        self.conv_a = quan_Conv1d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_a = nn.BatchNorm1d(planes)
        self.dropout_a = nn.Dropout(p=0.3)  # Dropout sau BatchNorm

        self.conv_b = quan_Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm1d(planes)
        self.dropout_b = nn.Dropout(p=0.3)  # Dropout sau BatchNorm

        self.downsample = downsample

    def forward(self, x):
        residual = x
        
        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)
        basicblock = self.dropout_a(basicblock)  # Áp dụng dropout

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)
        basicblock = self.dropout_b(basicblock)  # Áp dụng dropout

        if self.downsample:
            residual = self.downsample(x)

        return F.relu(residual + basicblock, inplace=True)
