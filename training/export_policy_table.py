from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a DQN to a small Pi runtime table")
    parser.add_argument("model", help="Path to a stable-baselines3 DQN zip")
    parser.add_argument("--output", default="models/gully_policy.json")
    parser.add_argument("--battery-step", type=int, default=5)
    args = parser.parse_args()
    if args.battery_step <= 0 or 100 % args.battery_step:
        raise SystemExit("battery-step must be a positive divisor of 100")

    try:
        from stable_baselines3 import DQN
    except ImportError as exc:
        raise SystemExit("stable-baselines3 is required on the training machine") from exc

    model = DQN.load(args.model, device="cpu")
    actions: dict[str, int] = {}
    for battery in range(0, 101, args.battery_step):
        for rain_level in range(3):
            observation = np.asarray([battery, rain_level], dtype=np.float32)
            action, _ = model.predict(observation, deterministic=True)
            actions[f"{battery}:{rain_level}"] = int(action)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"battery_step": args.battery_step, "actions": actions},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"Saved runtime policy table to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
