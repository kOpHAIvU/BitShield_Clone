# This file implements DIG on PyTorch

import torch
import torch.nn as nn
from torch.autograd import grad

import utils

# Since calculating this on PyTorch/CPU is too slow
cached_sus_score_ranges = {
    ('resnet50', 'CIFAR10'): [1931.778222373047, 2777.2466794042966],
    ('resnet50', 'MNISTC'): [1e-6, 730168.8208257324],
    ('resnet50', 'FashionC'): [1e-6, 367547.7471652832],
    ('resnet50', 'ImageNet'): [5829.662276359733, 45814.322139640986],  # mean=32422.926201196722, max=42724.0, min=11966.5693359375
    ('googlenet', 'CIFAR10'): [511.8928237158203, 1077.7412306982421],
    ('googlenet', 'MNISTC'): [4580.245536230469, 10190.49709873047],
    ('googlenet', 'FashionC'): [3540.325566748047, 10827.591728857422],
    ('googlenet', 'ImageNet'): [2097.448810887777, 26011.71306870028],  # mean=17905.086645999076, max=24140.953125, min=5745.365234375
    ('densenet121', 'CIFAR10'): [3912.626290566406, 11357.675509316407],
    ('densenet121', 'MNISTC'): [4416.494133886719, 11545.765862402344],
    ('densenet121', 'FashionC'): [4385.031740966797, 11876.160500732421],
    ('densenet121', 'ImageNet'): [3223.8052272047917, 31852.901613923543],  # mean=22771.050609838196, max=29757.08984375, min=7734.7080078125
}

class DIGProtectedModule(nn.Module):
    # NOTE: This also assumes the model's last layer is FC

    def __init__(self, model, model_fc=None):
        super().__init__()
        self.model = model
        self.model_fc = model_fc or getattr(model, 'fc', getattr(model, 'classifier', None))
        assert self.model_fc is not None, 'No fc/classifier layer found'

    def forward(self, x):
        # Pass through
        return self.model(x)

    def calc_sus_score(self, x):
        logits = self.model(x)
        nclasses = logits.shape[1]
        self.model.zero_grad()
        return utils.thread_first(
            grad(
                -1/nclasses * utils.thread_first(
                    logits,
                    (torch.log_softmax, {'dim': 1}),
                    (torch.sum, {'dim': 1}),
                    torch.sum,  # PyTorch requires a scalar for backward
                ),
                self.model_fc.weight,
                create_graph=True,
            )[0],
            torch.abs,
            torch.sum,
        )
