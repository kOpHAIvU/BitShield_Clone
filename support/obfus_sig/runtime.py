from typing import Dict, Optional

import copy
import torch
import torch.nn as nn

from .obfus_adapter import ObfusPair, wrap_last_linear_with_obfus
from .sig_lite import SigLiteMonitor
from .bit_fingerprint import BitFingerprint
from .controller import ControllerPolicy


class ObfusSigRuntime:
    """
    High-level runtime to combine:
      - Obfuscation (wrap last Linear with ObfusPair)
      - SigLiteMonitor (KL + grad-norm probes)
      - BitFingerprint (per-parameter PSI)
      - ControllerPolicy (fuse alerts, reseed)

    Usage:
      runtime = ObfusSigRuntime(model, probe_loader)
      runtime.calibrate()  # on clean model
      for step, (x, y) in enumerate(loader):
          out = runtime.model(x)  # normal inference/training
          runtime.periodic_check(step)
    """
    def __init__(
        self,
        model: nn.Module,
        probe_loader: torch.utils.data.DataLoader,
        alert_mode: str = "or",
        sig_period: int = 500,
        sig_k: float = 3.0,
        fp_threshold: float = 0.1,
        fp_entropy_threshold: float = 0.15,
        grad_norm_type: str = "l1",
        normalize_grad: bool = True,
        make_shadow: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or next(model.parameters()).device
        self.model = model
        # Wrap last linear with obfuscation pair
        self.model = wrap_last_linear_with_obfus(self.model)
        # Collect adapters
        self.adapters = []
        for m in self.model.modules():
            if isinstance(m, ObfusPair):
                self.adapters.append(m)
        # Monitors
        self.sig = SigLiteMonitor(
            self.model,
            probe_loader,
            period=sig_period,
            k=sig_k,
            device=self.device,
            grad_norm_type=grad_norm_type,  # type: ignore[arg-type]
            normalize_by_params=normalize_grad,
        )
        self.fp = BitFingerprint(self.model, threshold_psi=fp_threshold, threshold_entropy=fp_entropy_threshold)
        # Controller
        self.ctrl = ControllerPolicy(alert_mode=alert_mode, cooldown_steps=max(2, sig_period))
        self.ctrl.register_adapters(self.adapters)
        # Optional shadow model
        self.shadow_model = copy.deepcopy(self.model).to(self.device) if make_shadow else None
        if self.shadow_model is not None:
            self.ctrl.register_shadow_model(self.shadow_model)

    def calibrate(self, sig_steps: int = 50) -> Dict[str, float]:
        fp_stats = self.fp.build_baseline()
        sig_stats = self.sig.fit_baseline(steps=sig_steps)
        ret = {**fp_stats, **sig_stats}
        return ret

    def periodic_check(self, batch_idx: int) -> Dict[str, object]:
        # Run SigLite if due
        sig_ret = self.sig.step(batch_idx)
        sig_alert = int(sig_ret.get("alert", 0))  # type: ignore[arg-type]
        # Run fingerprint update on same cadence to amortize cost
        fp_ret = self.fp.update()
        fp_alert = int(fp_ret.get("alert", 0))  # type: ignore[arg-type]
        # Fuse in controller
        ctrl_ret = self.ctrl.step(
            {
                "sig_alert": sig_alert,
                "fp_alert": fp_alert,
                "sig": sig_ret,
                "fp": {
                    "max_psi": fp_ret.get("max_psi", 0.0),
                    "max_entropy_drift": fp_ret.get("max_entropy_drift", 0.0),
                },
            }
        )
        return {"sig": sig_ret, "fp": fp_ret, "ctrl": ctrl_ret}


