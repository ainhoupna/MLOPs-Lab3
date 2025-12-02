"""
Tests for model artifacts.

This module tests that the required model files exist before deployment.
"""

import pytest
from pathlib import Path


def test_model_file_exists():
    """Test that the ONNX model file exists."""
    model_path = Path("model.onnx")
    assert model_path.exists(), (
        f"Model file not found: {model_path}. "
        "Please run scripts/select_best_model.py to generate the model."
    )


def test_class_labels_file_exists():
    """Test that the class labels JSON file exists."""
    labels_path = Path("class_labels.json")
    assert labels_path.exists(), (
        f"Class labels file not found: {labels_path}. "
        "Please run scripts/select_best_model.py to generate the class labels."
    )


def test_model_file_not_empty():
    """Test that the ONNX model file is not empty."""
    model_path = Path("model.onnx")
    if model_path.exists():
        assert model_path.stat().st_size > 0, "Model file is empty"


def test_class_labels_file_not_empty():
    """Test that the class labels file is not empty."""
    labels_path = Path("class_labels.json")
    if labels_path.exists():
        assert labels_path.stat().st_size > 0, "Class labels file is empty"


def test_class_labels_valid_json():
    """Test that the class labels file contains valid JSON."""
    import json

    labels_path = Path("class_labels.json")
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            class_labels = json.load(f)
        
        assert isinstance(class_labels, dict), "Class labels should be a dictionary"
        assert len(class_labels) > 0, "Class labels should not be empty"
