"""Adaptive RL and rule-based policy decision engine."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from gully_system.config import PolicyConfig
from gully_system.sensors import SensorSnapshot

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PolicyDecision:
    mode: str
    inference_interval_s: float
    roi_profile: str = "normal"
    reason: str = ""


class Policy(Protocol):
    def decide(self, snapshot: SensorSnapshot) -> PolicyDecision:
        ...


class RuleBasedPolicy:
    def __init__(self, config: PolicyConfig) -> None:
        self.config = config

    def _decision(self, mode: str, reason: str, roi_profile: str = "normal") -> PolicyDecision:
        interval = self.config.mode_intervals.get(mode, 1.0)
        return PolicyDecision(mode=mode, inference_interval_s=interval, roi_profile=roi_profile, reason=reason)

    def decide(self, snapshot: SensorSnapshot) -> PolicyDecision:
        if snapshot.battery_pct is not None and snapshot.battery_pct <= self.config.critical_battery:
            return self._decision("low", "critical battery cutoff", "expanded")
        if snapshot.rain_level >= self.config.emergency_rain_level:
            return self._decision("high", "rain emergency", "expanded")
        if snapshot.water_level is not None and snapshot.water_level >= self.config.emergency_water_level:
            return self._decision("high", "high water level emergency", "expanded")
        if snapshot.battery_pct is not None and snapshot.battery_pct < self.config.min_safe_battery:
            return self._decision("low", "battery saving mode", "narrow")
        return self._decision("medium", "normal conditions", "normal")


class TablePolicy:
    def __init__(self, table_path: str | Path, config: PolicyConfig) -> None:
        self.config = config
        with Path(table_path).open("r", encoding="utf-8") as f:
            self.table = json.load(f)

    def decide(self, snapshot: SensorSnapshot) -> PolicyDecision:
        key = f"{int(snapshot.rain_level)}_{int(round(snapshot.battery_pct or 100))}"
        mode = self.table.get(key, "medium")
        interval = self.config.mode_intervals.get(mode, 1.0)
        return PolicyDecision(mode=mode, inference_interval_s=interval, reason=f"table lookup ({key})")


class DQNPolicy:
    def __init__(self, model_path: str, config: PolicyConfig) -> None:
        self.config = config
        self.model_path = model_path
        self._model = None
        self._action_to_mode = {0: "low", 1: "medium", 2: "high"}

        path = Path(model_path) if model_path else None
        if path and path.exists():
            try:
                from stable_baselines3 import DQN
                self._model = DQN.load(str(path))
                LOGGER.info("DQNPolicy successfully loaded model from %s", path)
            except Exception as exc:
                LOGGER.warning("Could not load DQN model from %s (%s). Using fallback mode.", path, exc)

    def decide(self, snapshot: SensorSnapshot) -> PolicyDecision:
        if self._model is not None:
            try:
                import numpy as np
                battery = snapshot.battery_pct if snapshot.battery_pct is not None else 100.0
                obs = np.array([float(snapshot.rain_level), float(battery)], dtype=np.float32)
                action, _ = self._model.predict(obs, deterministic=True)
                mode = self._action_to_mode.get(int(action), "medium")
                interval = self.config.mode_intervals.get(mode, 1.0)
                return PolicyDecision(mode=mode, inference_interval_s=interval, reason=f"DQN RL inference (action={action})")
            except Exception as exc:
                LOGGER.warning("DQN inference failed: %s. Using default.", exc)

        interval = self.config.mode_intervals.get("medium", 1.0)
        return PolicyDecision(mode="medium", inference_interval_s=interval, reason="DQN fallback (default medium)")



class SafePolicy:
    def __init__(self, policy: Policy, fallback: RuleBasedPolicy, config: PolicyConfig) -> None:
        self.policy = policy
        self.fallback = fallback
        self.config = config

    def _wrap_override(self, base_decision: PolicyDecision) -> PolicyDecision:
        return PolicyDecision(
            mode=base_decision.mode,
            inference_interval_s=base_decision.inference_interval_s,
            roi_profile=base_decision.roi_profile,
            reason="safety override",
        )

    def decide(self, snapshot: SensorSnapshot) -> PolicyDecision:
        if snapshot.battery_pct is not None and snapshot.battery_pct <= self.config.critical_battery:
            return self._wrap_override(self.fallback.decide(snapshot))
        if snapshot.rain_level >= self.config.emergency_rain_level:
            return self._wrap_override(self.fallback.decide(snapshot))
        if snapshot.water_level is not None and snapshot.water_level >= self.config.emergency_water_level:
            return self._wrap_override(self.fallback.decide(snapshot))
        try:
            return self.policy.decide(snapshot)
        except Exception:
            return self._wrap_override(self.fallback.decide(snapshot))
