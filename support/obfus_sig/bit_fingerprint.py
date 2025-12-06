from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def _quantize_int8(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().float().cpu().numpy()
    scale = np.max(np.abs(arr))
    if scale <= 1e-12:
        scale = 1.0
    q = np.clip(np.round(arr / (scale / 127.0)), -128, 127).astype(np.int8)
    return q


def _hist256(q: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(q.astype(np.int16), bins=256, range=(-128, 128))
    return counts.astype(np.float64)


def _psi(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    ratio = (p + eps) / (q + eps)
    return float(np.sum((p - q) * np.log(ratio)))


class BitFingerprint:
    """
    Per-parameter histogram on int8-quantized weights. PSI drift against baseline.
    Additionally track per bit-plane entropy and alert on entropy drift.
    Alert if any layer's PSI exceeds threshold_psi or entropy drift exceeds threshold_entropy.
    """
    def __init__(self, model: nn.Module, threshold_psi: float = 0.1, threshold_entropy: float = 0.15) -> None:
        self.model = model
        self.threshold_psi = float(threshold_psi)
        self.threshold_entropy = float(threshold_entropy)
        self.baseline_hists: Dict[str, np.ndarray] = {}
        self.baseline_entropy: Dict[str, np.ndarray] = {}

    @staticmethod
    def _bit_plane_entropy(q: np.ndarray) -> np.ndarray:
        # Compute entropy for each of 8 bit planes (int8 two's complement)
        q_u = q.view(np.uint8)
        ent = []
        for b in range(8):
            bits = (q_u >> b) & 1
            p1 = bits.mean()
            p0 = 1.0 - p1
            # binary entropy
            eps = 1e-12
            h = 0.0
            if 0.0 < p1 < 1.0:
                h = -(p1 * np.log2(p1 + eps) + p0 * np.log2(p0 + eps))
            ent.append(h)
        return np.array(ent, dtype=np.float64)

    def build_baseline(self) -> Dict[str, float]:
        self.baseline_hists.clear()
        self.baseline_entropy.clear()
        num_params = 0
        for name, p in self.model.named_parameters():
            if p.data is None:
                continue
            q = _quantize_int8(p.data)
            h = _hist256(q)
            self.baseline_hists[name] = h
            self.baseline_entropy[name] = self._bit_plane_entropy(q)
            num_params += 1
        return {"num_params": float(num_params)}

    def update(self) -> Dict[str, object]:
        if not self.baseline_hists:
            raise RuntimeError("Call build_baseline() before update().")
        psi_per_layer: Dict[str, float] = {}
        alert_layers: List[str] = []
        entropy_drift_per_layer: Dict[str, float] = {}
        entropy_alert_layers: List[str] = []
        for name, p in self.model.named_parameters():
            if name not in self.baseline_hists:
                continue
            q = _quantize_int8(p.data)
            h = _hist256(q)
            psi_val = _psi(self.baseline_hists[name], h)
            psi_per_layer[name] = psi_val
            if psi_val > self.threshold_psi:
                alert_layers.append(name)
            # entropy drift (L_inf across planes)
            cur_ent = self._bit_plane_entropy(q)
            base_ent = self.baseline_entropy.get(name, cur_ent)
            drift = float(np.max(np.abs(cur_ent - base_ent)))
            entropy_drift_per_layer[name] = drift
            if drift > self.threshold_entropy:
                entropy_alert_layers.append(name)
        max_psi = max(psi_per_layer.values()) if psi_per_layer else 0.0
        max_entropy_drift = max(entropy_drift_per_layer.values()) if entropy_drift_per_layer else 0.0
        any_alert = int(len(alert_layers) > 0 or len(entropy_alert_layers) > 0)
        return {
            "alert": any_alert,
            "max_psi": float(max_psi),
            "max_entropy_drift": float(max_entropy_drift),
            "psi_per_layer": psi_per_layer,
            "alert_layers": alert_layers,
            "entropy_drift_per_layer": entropy_drift_per_layer,
            "entropy_alert_layers": entropy_alert_layers,
        }

    def localize(self, topk: int = 5) -> List[Tuple[str, float]]:
        """Return top-k layers with largest PSI."""
        upd = self.update()
        psi_dict = upd["psi_per_layer"]  # type: ignore[assignment]
        items = sorted(psi_dict.items(), key=lambda x: x[1], reverse=True)
        return items[:topk]


