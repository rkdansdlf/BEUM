"""CLI entry point for the gully monitoring runtime."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from gully_system.config import SystemConfig
from gully_system.runtime import GullyRuntime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Raspberry Pi gully monitoring runtime")
    parser.add_argument("--config", default="config.example.json", help="Path to a JSON configuration")
    parser.add_argument("--model", help="Override detector model path")
    parser.add_argument("--source", help="Override camera index or video path")
    parser.add_argument("--max-frames", type=int, help="Override frame limit; 0 means unlimited")
    parser.add_argument("--realtime", action="store_true", help="Replay a video at source FPS with a latest-frame buffer")
    parser.add_argument("--display", action="store_true", help="Show a local preview window")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"Configuration file not found: {config_path}")

    config = SystemConfig.from_json(config_path)
    if args.model:
        config = replace(config, detector=replace(config.detector, model_path=args.model))
    if args.source:
        config = replace(config, source=args.source)
    if args.max_frames is not None:
        config = replace(config, max_frames=args.max_frames)
    if args.realtime:
        config = replace(config, realtime_source=True)

    runtime = GullyRuntime(config)
    result = runtime.run(display=args.display)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
