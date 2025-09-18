import random
import torch
import torch.nn as nn
from custom_models import CustomBlock, quan_Conv1d

class RandomBitFlip:
    def __init__(self, model: nn.Module, n_bits=8):
        self.model = model
        self.n_bits = n_bits
        self.modules = [m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv1d, CustomBlock, quan_Conv1d))]

    @staticmethod
    def _float_to_int8(x: torch.Tensor, scale: float):
        q = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int16)
        return q

    @staticmethod
    def _int8_to_float(q: torch.Tensor, scale: float):
        return (q.to(torch.float32)) * scale

    def flip_one_bit(self):
        if not self.modules:
            return None
        layer = random.choice(self.modules)
        with torch.no_grad():
            w = layer.weight.view(-1)
            idx = random.randrange(w.numel())
            scale = (w.abs().max() / 127.0).item() if w.numel() else 1e-3
            if scale == 0:
                scale = 1e-3
            q = self._float_to_int8(w, scale)
            bit = 1 << random.randrange(0, 7)  # 7 bits + sign approximation
            q[idx] = torch.bitwise_xor(q[idx], torch.tensor(bit, dtype=q.dtype))
            w[idx] = self._int8_to_float(q[idx], scale)
        return {'layer': layer.__class__.__name__, 'index': idx, 'bit': bit}

