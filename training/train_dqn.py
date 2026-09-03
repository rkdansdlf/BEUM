from __future__ import annotations

import argparse
from pathlib import Path

from .gully_env import GullyEnv


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the offline gully sampling policy")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--output", default="models/gully_dqn")
    args = parser.parse_args()

    try:
        from stable_baselines3 import DQN
    except ImportError as exc:
        raise SystemExit("Install gymnasium and stable-baselines3 on the training machine") from exc

    environment = GullyEnv()
    model = DQN(
        "MlpPolicy",
        environment,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        exploration_fraction=0.25,
        verbose=1,
        device="auto",
    )
    model.learn(total_timesteps=args.steps)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output))
    print(f"Saved offline policy to {output}.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
