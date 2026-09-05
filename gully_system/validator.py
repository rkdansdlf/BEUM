"""Model-Config mapping validator and resolver."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

LOGGER = logging.getLogger(__name__)

# Known domain aliases
GULLY_ALIASES: frozenset[str] = frozenset({
    "drain_area",
    "drain",
    "gully",
    "catch_basin",
    "grating",
    "gutter",
    "drainage",
})

OBSTACLE_ALIASES: frozenset[str] = frozenset({
    "drain_full",
    "debris",
    "sediment",
    "trash",
    "leaf",
    "leaves",
    "blockage",
    "obstacle",
    "dirt",
    "litter",
    "mud",
})


class MappingMode(str, Enum):
    STRICT = "strict"
    AUTO = "auto"
    WARN = "warn"


class ModelMappingError(ValueError):
    """Raised when model and config class mappings are incompatible."""


@dataclass(frozen=True)
class MappingResult:
    is_exact_match: bool
    model_classes: tuple[str, ...]
    config_classes: tuple[str, ...]
    missing_in_config: tuple[str, ...]
    missing_in_model: tuple[str, ...]
    resolved_class_names: tuple[str, ...]
    resolved_gully_classes: tuple[str, ...]
    resolved_obstacle_classes: tuple[str, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


def extract_model_classes(model: Any) -> tuple[str, ...]:
    """Extract class names list in order from a YOLO model object."""
    names_attr = getattr(model, "names", None)
    if not names_attr:
        return ()

    if isinstance(names_attr, dict):
        # Sort by integer class index if possible
        try:
            sorted_items = sorted(names_attr.items(), key=lambda item: int(item[0]))
            return tuple(str(v) for _, v in sorted_items)
        except (TypeError, ValueError):
            return tuple(str(v) for v in names_attr.values())
    elif isinstance(names_attr, (list, tuple)):
        return tuple(str(v) for v in names_attr)
    return ()


def validate_and_resolve_mapping(
    model_classes: tuple[str, ...] | list[str],
    config_class_names: tuple[str, ...] | list[str] = (),
    gully_class_names: tuple[str, ...] | list[str] = ("gully",),
    obstacle_class_names: tuple[str, ...] | list[str] = ("debris", "sediment", "trash", "leaf"),
    mapping_mode: str | MappingMode = MappingMode.AUTO,
) -> MappingResult:
    """
    Validates model class names against config and resolves active classes based on mode.

    Modes:
      - 'strict': Mismatch raises ModelMappingError immediately (Fail-Fast).
      - 'auto': Automatically map classes using domain aliases and populate missing values.
                If no gully class can be resolved, raises ModelMappingError.
      - 'warn': Logs warnings on mismatch and preserves config classes.
    """
    mode_str = mapping_mode.value if isinstance(mapping_mode, MappingMode) else str(mapping_mode).lower()
    m_classes = tuple(model_classes)
    c_classes = tuple(config_class_names)
    g_classes = tuple(gully_class_names)
    o_classes = tuple(obstacle_class_names)

    warnings: list[str] = []
    errors: list[str] = []

    if not m_classes:
        msg = "Model has no classes defined in 'model.names'."
        if mode_str == MappingMode.STRICT.value:
            raise ModelMappingError(msg)
        warnings.append(msg)
        LOGGER.warning(msg)
        return MappingResult(
            is_exact_match=False,
            model_classes=(),
            config_classes=c_classes,
            missing_in_config=(),
            missing_in_model=c_classes,
            resolved_class_names=c_classes,
            resolved_gully_classes=g_classes,
            resolved_obstacle_classes=o_classes,
            warnings=tuple(warnings),
            errors=(msg,) if mode_str == MappingMode.STRICT.value else (),
        )

    # Sets for comparison (case-insensitive for alias logic, exact for match logic)
    m_set_lower = {cls_name.lower(): cls_name for cls_name in m_classes}
    c_set_lower = {cls_name.lower(): cls_name for cls_name in c_classes}

    missing_in_config = tuple(cls_name for cls_name in m_classes if cls_name.lower() not in c_set_lower)
    missing_in_model = tuple(cls_name for cls_name in c_classes if cls_name.lower() not in m_set_lower)

    is_exact_match = (
        len(missing_in_config) == 0
        and len(missing_in_model) == 0
        and len(m_classes) == len(c_classes)
        and list(m_classes) == list(c_classes)
    )

    if not is_exact_match:
        mismatch_details = []
        if missing_in_config:
            mismatch_details.append(f"Model classes missing in config: {list(missing_in_config)}")
        if missing_in_model:
            mismatch_details.append(f"Config classes missing in model: {list(missing_in_model)}")
        if not mismatch_details and list(m_classes) != list(c_classes):
            mismatch_details.append(
                f"Class ordering or casing differs (model: {list(m_classes)}, config: {list(c_classes)})"
            )

        warn_msg = f"Model-Config class mapping mismatch detected. {' | '.join(mismatch_details)}"
        warnings.append(warn_msg)
        LOGGER.warning(warn_msg)

    # Strict mode handling
    if mode_str == MappingMode.STRICT.value:
        if not is_exact_match:
            err_msg = f"Strict validation failed: {warnings[-1]}"
            errors.append(err_msg)
            raise ModelMappingError(err_msg)

        return MappingResult(
            is_exact_match=True,
            model_classes=m_classes,
            config_classes=c_classes,
            missing_in_config=(),
            missing_in_model=(),
            resolved_class_names=c_classes,
            resolved_gully_classes=g_classes,
            resolved_obstacle_classes=o_classes,
            warnings=(),
            errors=(),
        )

    # Auto mode handling
    if mode_str == MappingMode.AUTO.value:
        auto_gullies: list[str] = []
        auto_obstacles: list[str] = []
        unrecognized: list[str] = []

        for m_name in m_classes:
            low = m_name.lower()
            if low in GULLY_ALIASES or any(low in g.lower() for g in g_classes):
                auto_gullies.append(m_name)
            elif low in OBSTACLE_ALIASES or any(low in o.lower() for o in o_classes):
                auto_obstacles.append(m_name)
            else:
                unrecognized.append(m_name)

        if not auto_gullies:
            err_msg = (
                f"Auto-mapping failed: No gully/drain class found in model classes {list(m_classes)}. "
                f"Recognized gully aliases: {sorted(GULLY_ALIASES)}"
            )
            errors.append(err_msg)
            LOGGER.error(err_msg)
            raise ModelMappingError(err_msg)

        if unrecognized:
            msg = f"Unrecognized model classes assigned to obstacles by fallback: {unrecognized}"
            warnings.append(msg)
            LOGGER.info(msg)
            auto_obstacles.extend(unrecognized)

        resolved_gullies = tuple(auto_gullies)
        resolved_obstacles = tuple(auto_obstacles)
        resolved_all = tuple(m_classes)

        LOGGER.info(
            "Auto-mapping applied: gully=%s, obstacles=%s (all=%s)",
            resolved_gullies,
            resolved_obstacles,
            resolved_all,
        )

        return MappingResult(
            is_exact_match=is_exact_match,
            model_classes=m_classes,
            config_classes=c_classes,
            missing_in_config=missing_in_config,
            missing_in_model=missing_in_model,
            resolved_class_names=resolved_all,
            resolved_gully_classes=resolved_gullies,
            resolved_obstacle_classes=resolved_obstacles,
            warnings=tuple(warnings),
            errors=tuple(errors),
        )

    # Warn mode handling
    resolved_all = c_classes if c_classes else m_classes
    return MappingResult(
        is_exact_match=is_exact_match,
        model_classes=m_classes,
        config_classes=c_classes,
        missing_in_config=missing_in_config,
        missing_in_model=missing_in_model,
        resolved_class_names=resolved_all,
        resolved_gully_classes=g_classes,
        resolved_obstacle_classes=o_classes,
        warnings=tuple(warnings),
        errors=tuple(errors),
    )
