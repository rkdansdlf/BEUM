"""Tests for model-config class mapping validation, auto-mapping, and strict blocking."""

import pytest
from gully_system.config import DetectorConfig, SystemConfig
from gully_system.validator import (
    MappingMode,
    MappingResult,
    ModelMappingError,
    extract_model_classes,
    validate_and_resolve_mapping,
)


class DummyYOLOModel:
    def __init__(self, names):
        self.names = names


def test_extract_model_classes():
    # Dict format with int keys
    m1 = DummyYOLOModel({0: "drain_area", 1: "drain_full"})
    assert extract_model_classes(m1) == ("drain_area", "drain_full")

    # Dict format with string keys
    m2 = DummyYOLOModel({"0": "gully", "1": "debris", "2": "sediment"})
    assert extract_model_classes(m2) == ("gully", "debris", "sediment")

    # List format
    m3 = DummyYOLOModel(["gully", "debris"])
    assert extract_model_classes(m3) == ("gully", "debris")

    # None / Empty
    m4 = DummyYOLOModel(None)
    assert extract_model_classes(m4) == ()


def test_exact_match_validation():
    model_classes = ("drain_area", "drain_full")
    config_classes = ("drain_area", "drain_full")

    res = validate_and_resolve_mapping(
        model_classes=model_classes,
        config_class_names=config_classes,
        gully_class_names=("drain_area",),
        obstacle_class_names=("drain_full",),
        mapping_mode="strict",
    )

    assert res.is_exact_match is True
    assert res.is_valid is True
    assert res.missing_in_config == ()
    assert res.missing_in_model == ()
    assert res.resolved_class_names == ("drain_area", "drain_full")
    assert res.resolved_gully_classes == ("drain_area",)
    assert res.resolved_obstacle_classes == ("drain_full",)


def test_strict_mode_blocks_on_mismatch():
    # 2-class model with 5-class config
    model_classes = ("drain_area", "drain_full")
    config_classes = ("gully", "debris", "sediment", "trash", "leaf")

    with pytest.raises(ModelMappingError) as exc_info:
        validate_and_resolve_mapping(
            model_classes=model_classes,
            config_class_names=config_classes,
            mapping_mode=MappingMode.STRICT,
        )

    assert "Strict validation failed" in str(exc_info.value)


def test_auto_mapping_resolves_synonyms():
    # 2-class model with 5-class config in auto mode
    model_classes = ("drain_area", "drain_full")
    config_classes = ("gully", "debris", "sediment", "trash", "leaf")

    res = validate_and_resolve_mapping(
        model_classes=model_classes,
        config_class_names=config_classes,
        gully_class_names=("gully",),
        obstacle_class_names=("debris", "sediment", "trash", "leaf"),
        mapping_mode="auto",
    )

    assert res.is_exact_match is False
    assert res.is_valid is True
    assert len(res.warnings) > 0
    # Auto-mapping should identify drain_area as gully and drain_full as obstacle
    assert res.resolved_gully_classes == ("drain_area",)
    assert res.resolved_obstacle_classes == ("drain_full",)
    assert res.resolved_class_names == ("drain_area", "drain_full")


def test_auto_mapping_case_insensitivity():
    model_classes = ("GULLY", "Debris")
    config_classes = ("gully", "debris")

    res = validate_and_resolve_mapping(
        model_classes=model_classes,
        config_class_names=config_classes,
        gully_class_names=("gully",),
        obstacle_class_names=("debris",),
        mapping_mode="auto",
    )

    assert res.resolved_gully_classes == ("GULLY",)
    assert res.resolved_obstacle_classes == ("Debris",)


def test_auto_mapping_fails_without_gully_class():
    # Model has only non-drain classes (e.g., COCO object classes)
    model_classes = ("car", "person", "dog")
    config_classes = ("gully", "debris")

    with pytest.raises(ModelMappingError) as exc_info:
        validate_and_resolve_mapping(
            model_classes=model_classes,
            config_class_names=config_classes,
            mapping_mode="auto",
        )

    assert "No gully/drain class found" in str(exc_info.value)


def test_warn_mode_preserves_config():
    model_classes = ("drain_area", "drain_full")
    config_classes = ("gully", "debris")

    res = validate_and_resolve_mapping(
        model_classes=model_classes,
        config_class_names=config_classes,
        gully_class_names=("gully",),
        obstacle_class_names=("debris",),
        mapping_mode="warn",
    )

    assert res.is_exact_match is False
    assert res.resolved_class_names == ("gully", "debris")
    assert res.resolved_gully_classes == ("gully",)
    assert res.resolved_obstacle_classes == ("debris",)
    assert len(res.warnings) > 0
