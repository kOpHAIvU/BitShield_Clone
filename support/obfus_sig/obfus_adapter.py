import torch
import torch.nn as nn
from typing import Optional


def _index_select_along_dim(x: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    if index.device != x.device:
        index = index.to(x.device)
    return torch.index_select(x, dim, index)


class ObfusAdapter(nn.Module):
    """
    Stateless permutation adapter with reseed() to redraw a permutation.
    It permutes along a given dimension.
    """
    def __init__(self, size: int, dim: int = 1, seed: Optional[int] = None) -> None:
        super().__init__()
        self.dim = dim
        self.size = int(size)
        perm = self._make_perm(seed)
        self.register_buffer("perm", perm, persistent=False)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(self.size, device=perm.device)
        self.register_buffer("inv_perm", inv, persistent=False)

    def _make_perm(self, seed: Optional[int]) -> torch.Tensor:
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(int(seed))
        return torch.randperm(self.size, generator=gen)

    @torch.no_grad()
    def reseed(self, seed: Optional[int] = None) -> None:
        perm = self._make_perm(seed).to(self.perm.device)
        self.perm.copy_(perm)
        inv = torch.empty_like(self.perm)
        inv[self.perm] = torch.arange(self.size, device=self.perm.device)
        self.inv_perm.copy_(inv)

    def forward(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        index = self.inv_perm if inverse else self.perm
        return _index_select_along_dim(x, self.dim, index)


class ObfusPair(nn.Module):
    """
    Wrap a child module with a permutation before and its inverse after the child.
    This preserves function (ideally) but obfuscates intermediate representation.
    """
    def __init__(self, child: nn.Module, size: int, dim: int = 1, seed: Optional[int] = None) -> None:
        super().__init__()
        self.child = child
        self.adapter = ObfusAdapter(size=size, dim=dim, seed=seed)
        self.dim = dim

    @torch.no_grad()
    def reseed(self, seed: Optional[int] = None) -> None:
        self.adapter.reseed(seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adapter(x, inverse=False)
        y = self.child(x)
        y = self.adapter(y, inverse=True)
        return y


def wrap_last_linear_with_obfus(model: nn.Module, seed: Optional[int] = None) -> nn.Module:
    """
    Find the last nn.Linear in model and wrap it with ObfusPair.
    The permutation size equals in_features (feature dimension) and dim depends on input coming to Linear:
    - For standard tabular models: inputs to Linear are [batch, features] => dim=1
    """
    last_name = None
    last_linear = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_name = name
            last_linear = module
    if last_linear is None or last_name is None:
        return model

    parent = model
    path = last_name.split(".")
    for p in path[:-1]:
        parent = getattr(parent, p)
    leaf_name = path[-1]
    wrapped = ObfusPair(child=last_linear, size=last_linear.in_features, dim=1, seed=seed)
    setattr(parent, leaf_name, wrapped)
    return model


