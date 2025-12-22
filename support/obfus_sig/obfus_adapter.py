import torch
import torch.nn as nn
from typing import List, Optional, Sequence, Tuple


def _index_select_along_dim(x: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    if index.device != x.device:
        index = index.to(x.device)
    return torch.index_select(x, dim, index)


class ObfusAdapter(nn.Module):
    """
    Lightweight permutation holder with reseed() support.
    """
    def __init__(self, size: int, seed: Optional[int] = None) -> None:
        super().__init__()
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


class ObfusPair(nn.Module):
    """
    Wrap a child module by permuting its weight columns (or input channels) in memory.
    During forward pass, inputs are reordered by the inverse permutation to preserve
    functional behaviour while obfuscating the physical bit layout.
    """
    def __init__(
        self,
        child: nn.Module,
        size: int,
        dim: int = 1,
        seed: Optional[int] = None,
        allow_fallback: bool = True,
    ) -> None:
        super().__init__()
        self.child = child
        self.dim = dim
        self.size = int(size)
        self.adapter = ObfusAdapter(size=self.size, seed=seed)
        self.register_buffer("_active_perm_buf", self.adapter.perm.clone(), persistent=False)
        self.register_buffer("_active_inv_buf", self.adapter.inv_perm.clone(), persistent=False)
        self._allow_fallback = allow_fallback
        self._supports_weight_shuffle = self._check_weight_dim()
        # Apply initial permutation so that physical layout is obfuscated from the start.
        if self._supports_weight_shuffle:
            identity = torch.arange(self.size, device=self._active_perm_buf.device)
            self._sync_child_weights(self.adapter.perm, identity)
        elif not self._allow_fallback:
            raise ValueError(f"Module {child.__class__.__name__} does not support weight obfuscation.")

    def _check_weight_dim(self) -> bool:
        weight = getattr(self.child, "weight", None)
        if weight is None:
            return False
        if weight.dim() <= self.dim:
            return False
        if weight.size(self.dim) != self.size:
            return False
        return True

    @torch.no_grad()
    def _sync_child_weights(self, new_perm: torch.Tensor, prev_inv: torch.Tensor) -> None:
        if not self._supports_weight_shuffle:
            return
        weight = getattr(self.child, "weight", None)
        if weight is None:
            return
        device = weight.data.device
        new_perm = new_perm.to(device)
        prev_inv = prev_inv.to(device)
        logical = torch.index_select(weight.data, self.dim, prev_inv)
        shuffled = torch.index_select(logical, self.dim, new_perm)
        weight.data.copy_(shuffled)

    @torch.no_grad()
    def reseed(self, seed: Optional[int] = None) -> None:
        prev_inv = self._active_inv_buf.clone()
        self.adapter.reseed(seed)
        new_perm = self.adapter.perm.clone()
        new_inv = self.adapter.inv_perm.clone()
        if self._supports_weight_shuffle:
            self._sync_child_weights(new_perm, prev_inv)
        elif self._allow_fallback:
            # Fall back to activation permutation when weights cannot be shuffled.
            pass
        self._active_perm_buf.copy_(new_perm)
        self._active_inv_buf.copy_(new_inv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._supports_weight_shuffle:
            x = _index_select_along_dim(x, self.dim, self._active_perm_buf)
            return self.child(x)
        # Fallback: behave like legacy adapter (permute activations pre/post).
        # This path is only used when shuffling weights is not supported.
        x = _index_select_along_dim(x, self.dim, self.adapter.perm)
        y = self.child(x)
        y = _index_select_along_dim(y, self.dim, self.adapter.inv_perm)
        return y


def _resolve_parent(model: nn.Module, path: str) -> Tuple[nn.Module, str]:
    if not path:
        raise ValueError("Cannot wrap the root module with ObfusPair.")
    parent = model
    fragments = path.split(".")
    for frag in fragments[:-1]:
        parent = getattr(parent, frag)
    return parent, fragments[-1]


def wrap_model_with_obfus(
    model: nn.Module,
    targets: Sequence[str] = ("linear",),
    max_wrapped: Optional[int] = None,
    seed: Optional[int] = None,
    allow_fallback: bool = True,
) -> Tuple[nn.Module, List[ObfusPair]]:
    """
    Wrap selected layers with ObfusPair to obfuscate their weight layout.

    Args:
        model: The module to wrap in-place.
        targets: Iterable of layer type aliases ('linear', 'conv1d', 'conv2d').
        max_wrapped: Limit on number of layers to wrap (closest to output). None or <=0 keeps all.
        seed: Optional base seed for deterministic permutations.
        allow_fallback: If False, raise error when a target cannot shuffle its weights.
    Returns:
        (model, adapters) with adapters preserving order of wrapping.
    """
    type_map = {
        "linear": (nn.Linear,),
        "conv1d": (nn.Conv1d,),
        "conv2d": (nn.Conv2d,),
    }
    resolved_types: Tuple[type, ...] = tuple(
        {tp for alias in targets for tp in type_map.get(alias.lower(), tuple())}
    )
    if not resolved_types:
        return model, []

    candidates: List[Tuple[str, nn.Module, int]] = []
    seen_ids = set()
    for name, module in model.named_modules():
        if not name:
            continue
        if isinstance(module, ObfusPair):
            continue
        if id(module) in seen_ids:
            continue
        if not any(isinstance(module, t) for t in resolved_types):
            continue
        weight = getattr(module, "weight", None)
        if weight is None or weight.dim() <= 1:
            if allow_fallback:
                size = getattr(module, "in_features", None) or getattr(module, "in_channels", None)
                if size is None:
                    continue
            else:
                continue
        else:
            size = int(weight.size(1 if weight.dim() > 1 else 0))
        if size is None or size < 2:
            continue
        seen_ids.add(id(module))
        candidates.append((name, module, size))

    if not candidates:
        return model, []

    if max_wrapped is not None and max_wrapped > 0:
        candidates = candidates[-max_wrapped:]

    adapters: List[ObfusPair] = []
    for idx, (name, module, size) in enumerate(candidates):
        parent, leaf = _resolve_parent(model, name)
        wrapper_seed = None if seed is None else int(seed) + idx
        wrapper = ObfusPair(
            child=module,
            size=size,
            dim=1,
            seed=wrapper_seed,
            allow_fallback=allow_fallback,
        )
        setattr(parent, leaf, wrapper)
        adapters.append(wrapper)
    return model, adapters


def wrap_last_linear_with_obfus(model: nn.Module, seed: Optional[int] = None) -> nn.Module:
    wrap_model_with_obfus(model, targets=("linear",), max_wrapped=1, seed=seed)
    return model
