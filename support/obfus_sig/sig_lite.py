import math
from typing import Dict, Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def _median_and_mad(values: torch.Tensor) -> Tuple[float, float]:
    med = values.median().item()
    mad = (values - values.median()).abs().median().item()
    return med, mad


def _find_last_linear(model: nn.Module) -> Optional[nn.Module]:
    """
    Find the "head" layer we will monitor.
    Prefer the last nn.Linear; if none exists, fall back to the last Conv1d/Conv2d.
    """
    last: Optional[nn.Module] = None
    # First, try to find the last Linear layer
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is not None:
        return last
    # Fallback: use the last Conv layer as head (for fully-conv models)
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            last = m
    return last


class SigLiteMonitor:
    """
    Lightweight semantic integrity guard:
    - KL divergence D_KL(u || p) with u = uniform prior over classes
    - L1 norm of gradient of KL wrt last layer weights
    Thresholds via robust stats (median ± k * MAD).
    """
    def __init__(
        self,
        model: nn.Module,
        probe_loader: torch.utils.data.DataLoader,
        period: int = 500,
        k: float = 3.0,
        device: Optional[torch.device] = None,
        grad_norm_type: Literal["l1","l2"] = "l1",
        normalize_by_params: bool = True,
    ) -> None:
        self.model = model
        self.probe_loader = probe_loader
        self.period = int(period)
        self.k = float(k)
        self.device = device or next(model.parameters()).device
        self.last_layer = _find_last_linear(model)
        if self.last_layer is None:
            raise ValueError("SigLiteMonitor requires a model with a final nn.Linear/Conv layer.")
        self._probe_iter = None
        # Baseline thresholds
        self.kl_med = None
        self.kl_mad = None
        self.gn_med = None
        self.gn_mad = None
        self.grad_norm_type = grad_norm_type
        self.normalize_by_params = normalize_by_params

    def _next_probe_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._probe_iter is None:
            self._probe_iter = iter(self.probe_loader)
        try:
            batch = next(self._probe_iter)
        except StopIteration:
            self._probe_iter = iter(self.probe_loader)
            batch = next(self._probe_iter)
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            x, y = batch, None
        x = x.to(self.device)
        if y is not None:
            y = y.to(self.device)
        return x, y

    @torch.no_grad()
    def _compute_probs(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits, dim=-1)

    def _compute_kl_uniform(self, probs: torch.Tensor) -> torch.Tensor:
        # D_KL(u || p) = sum_i u_i * log(u_i / p_i), with u_i = 1/C
        c = probs.size(-1)
        u = 1.0 / float(c)
        # Avoid log(0)
        eps = 1e-8
        log_term = math.log(u + eps) - torch.log(probs + eps)
        kl = (u * log_term).sum(dim=-1).mean()
        return kl

    def _compute_grad_norm(self, kl: torch.Tensor) -> torch.Tensor:
        # Compute gradient of KL wrt last layer weights
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad = None
        grad_w = torch.autograd.grad(
            kl, self.last_layer.weight, retain_graph=False, allow_unused=False, create_graph=False
        )[0]
        if self.grad_norm_type == "l2":
            norm = torch.linalg.vector_norm(grad_w, ord=2)
        else:
            norm = grad_w.abs().sum()
        if self.normalize_by_params:
            denom = float(grad_w.numel())
            if denom > 0:
                norm = norm / denom
        return norm

    def fit_baseline(self, steps: int = 50) -> Dict[str, float]:
        """
        Run a few probe steps on a clean model to calibrate thresholds.
        """
        self.model.eval()
        kl_vals = []
        gn_vals = []
        with torch.enable_grad():
            for _ in range(steps):
                x, _ = self._next_probe_batch()
                logits = self.model(x)
                probs = self._compute_probs(logits)
                kl = self._compute_kl_uniform(probs)
                gn = self._compute_grad_norm(kl)
                kl_vals.append(kl.detach())
                gn_vals.append(gn.detach())
        kl_tensor = torch.stack(kl_vals)
        gn_tensor = torch.stack(gn_vals)
        self.kl_med, self.kl_mad = _median_and_mad(kl_tensor)
        self.gn_med, self.gn_mad = _median_and_mad(gn_tensor)
        return {
            "kl_median": self.kl_med,
            "kl_mad": self.kl_mad,
            "grad_norm_median": self.gn_med,
            "grad_norm_mad": self.gn_mad,
        }

    def _thresholds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        if any(v is None for v in [self.kl_med, self.kl_mad, self.gn_med, self.gn_mad]):
            raise RuntimeError("Call fit_baseline() before using SigLiteMonitor.")
        kl_L = self.kl_med - self.k * self.kl_mad
        kl_U = self.kl_med + self.k * self.kl_mad
        gn_L = self.gn_med - self.k * self.gn_mad
        gn_U = self.gn_med + self.k * self.gn_mad
        return (kl_L, kl_U), (gn_L, gn_U)

    def step(self, batch_idx: int) -> Dict[str, float]:
        """
        Run probe every 'period' steps. Returns metrics and 'alert' flag.
        """
        run_now = (batch_idx % self.period) == 0
        if not run_now:
            return {"ran": 0, "alert": 0}
        self.model.eval()
        with torch.enable_grad():
            x, _ = self._next_probe_batch()
            logits = self.model(x)
            probs = self._compute_probs(logits)
            kl = self._compute_kl_uniform(probs)
            gn = self._compute_grad_norm(kl)
        (kl_L, kl_U), (gn_L, gn_U) = self._thresholds()
        kl_val = float(kl.detach().cpu().item())
        gn_val = float(gn.detach().cpu().item())
        kl_alert = int(kl_val < kl_L or kl_val > kl_U)
        gn_alert = int(gn_val < gn_L or gn_val > gn_U)
        alert = int(kl_alert or gn_alert)
        return {
            "ran": 1,
            "alert": alert,
            "kl": kl_val,
            "grad_norm_l1": gn_val,
            "kl_L": kl_L,
            "kl_U": kl_U,
            "gn_L": gn_L,
            "gn_U": gn_U,
        }


