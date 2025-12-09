import time
from typing import Dict, List, Optional

import torch.nn as nn

from .obfus_adapter import ObfusPair


class ControllerPolicy:
    """
    Fuse alerts from SigLite and BitFingerprint and trigger actions:
    - reseed all registered ObfusPairs
    - optionally switch to a shadow model
    """
    def __init__(
        self,
        alert_mode: str = "or",  # "or" or "and"
        cooldown_steps: int = 1000,
        proactive_period: int = 0,
    ) -> None:
        assert alert_mode in ("or", "and")
        self.alert_mode = alert_mode
        self.cooldown_steps = int(cooldown_steps)
        self.proactive_period = max(0, int(proactive_period))
        self._last_reseed_step = -10**9
        self._step = 0
        self._adapters: List[ObfusPair] = []
        self._shadow_model: Optional[nn.Module] = None
        self._logs: List[Dict[str, object]] = []

    def register_adapters(self, adapters: List[ObfusPair]) -> None:
        self._adapters = adapters

    def register_shadow_model(self, shadow: nn.Module) -> None:
        self._shadow_model = shadow

    def _should_act(self, alerts: Dict[str, int]) -> bool:
        values = list(alerts.values())
        if not values:
            return False
        fire = any(v > 0 for v in values) if self.alert_mode == "or" else all(v > 0 for v in values)
        return fire and (self._step - self._last_reseed_step >= self.cooldown_steps)

    def _perform_reseed(self) -> None:
        for adapter in self._adapters:
            adapter.reseed()
        self._last_reseed_step = self._step

    def step(self, metrics: Dict[str, object]) -> Dict[str, object]:
        """
        metrics: should include {'sig_alert': 0/1, 'fp_alert': 0/1, ... plus aux}
        """
        alerts = {
            "sig": int(metrics.get("sig_alert", 0)),  # type: ignore[arg-type]
            "fp": int(metrics.get("fp_alert", 0)),  # type: ignore[arg-type]
        }
        action_taken = "none"
        if self._should_act(alerts):
            self._perform_reseed()
            action_taken = "reseed_adapters_alert"
        elif self.proactive_period > 0 and (self._step - self._last_reseed_step) >= self.proactive_period:
            self._perform_reseed()
            action_taken = "reseed_adapters_proactive"
        # Optionally escalate to shadow switch if repeated alerts (not implemented escalation logic here)
        log_entry = {
            "t": time.time(),
            "step": self._step,
            "alerts": alerts,
            "action": action_taken,
            "extras": {k: v for k, v in metrics.items() if k not in ("sig_alert", "fp_alert")},
        }
        self._logs.append(log_entry)
        self._step += 1
        return {"action": action_taken, "step": self._step - 1}

    def get_logs(self) -> List[Dict[str, object]]:
        return self._logs


