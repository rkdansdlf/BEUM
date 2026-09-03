"""Compatibility entry point for the polygon-aware COCO converter."""

try:
    from .convert_coco_to_yolo import main
except ImportError:  # Supports direct execution as well as python -m.
    from convert_coco_to_yolo import main


if __name__ == "__main__":
    raise SystemExit(main())
