from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class GullyEnv(gym.Env):
    """Small offline environment for choosing an inference interval.

    Observation: [battery percentage, rain level].
    Action: 0=low, 1=medium, 2=high inference mode.
    This environment is deliberately conservative: missing a hazard costs more
    than spending additional energy.
    """

    metadata = {"render_modes": []}

    def __init__(self, episode_steps: int = 1440, seed: int | None = None) -> None:
        super().__init__()
        self.episode_steps = episode_steps
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.asarray([0.0, 0.0], dtype=np.float32),
            high=np.asarray([100.0, 2.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._rng = np.random.default_rng(seed)
        self.battery = 100.0
        self.rain_level = 0
        self.steps = 0

    def _observation(self) -> np.ndarray:
        return np.asarray([self.battery, self.rain_level], dtype=np.float32)

    def _next_rain_level(self) -> int:
        # Rain tends to persist, but weather can change during an episode.
        if self.rain_level == 0:
            return int(self._rng.choice([0, 0, 0, 1, 2]))
        if self.rain_level == 1:
            return int(self._rng.choice([0, 1, 1, 2]))
        return int(self._rng.choice([1, 2, 2, 2]))

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.battery = 100.0
        self.rain_level = int(self._rng.integers(0, 3))
        self.steps = 0
        return self._observation(), {}

    def step(self, action: int):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        energy_cost = (0.03, 0.10, 0.30)[action]
        detection_probability = (0.45, 0.75, 0.95)[action]
        hazard_probability = (0.02, 0.12, 0.30)[self.rain_level]
        hazard_seen = bool(self._rng.random() < hazard_probability)
        detected = bool(hazard_seen and self._rng.random() < detection_probability)

        self.battery = max(0.0, self.battery - energy_cost)
        reward = -energy_cost
        if detected:
            reward += 4.0
        elif hazard_seen:
            reward -= 12.0
        if self.battery < 20.0:
            reward -= 1.0

        self.rain_level = self._next_rain_level()
        self.steps += 1
        terminated = self.battery <= 0.0
        truncated = self.steps >= self.episode_steps
        return self._observation(), float(reward), terminated, truncated, {
            "hazard_seen": hazard_seen,
            "detected": detected,
        }
