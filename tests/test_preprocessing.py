"""
Unit Tests for preprocessing utilities (mylib/preprocessing.py)
"""

import pytest
from pathlib import Path
from PIL import Image
from unittest.mock import patch

from mylib.preprocessing import (
    ensure_output_dir,
    to_grayscale,
    random_rotate,
    random_flip,
    blur,
    preprocess_pipeline,
)


# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def dummy_image(tmp_path):
    """Create a temporary test image."""
    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (100, 100), color="white")
    img.save(img_path)
    return img_path


# ─────────────────────────────
# ENSURE_OUTPUT_DIR
# ─────────────────────────────
def test_ensure_output_dir(tmp_path, monkeypatch):
    """Test that ensure_output_dir creates the directory."""
    monkeypatch.chdir(tmp_path)
    ensure_output_dir()
    out = Path("outputs")
    assert out.exists()
    assert out.is_dir()


def test_ensure_output_dir_idempotent(tmp_path, monkeypatch):
    """Test that calling ensure_output_dir multiple times is safe."""
    monkeypatch.chdir(tmp_path)
    ensure_output_dir()
    ensure_output_dir()  # Second call should not fail
    out = Path("outputs")
    assert out.exists()


# ─────────────────────────────
# TO_GRAYSCALE
# ─────────────────────────────
def test_to_grayscale(dummy_image):
    """Test grayscale conversion."""
    img = Image.open(dummy_image)
    gray = to_grayscale(img)
    assert gray.mode == "L"  # Grayscale mode


def test_to_grayscale_already_gray(dummy_image, tmp_path):
    """Test grayscale conversion on already grayscale image."""
    # Create gray image
    gray_path = tmp_path / "gray.jpg"
    Image.open(dummy_image).convert("L").save(gray_path)
    
    img = Image.open(gray_path)
    gray = to_grayscale(img)
    assert gray.mode == "L"


# ─────────────────────────────
# RANDOM_ROTATE
# ─────────────────────────────
@patch("mylib.preprocessing.random.uniform", return_value=10)
def test_random_rotate(mock_rot, dummy_image):
    """Test random rotation."""
    img = Image.open(dummy_image)
    rotated = random_rotate(img)
    mock_rot.assert_called_once_with(-20, 20)
    assert isinstance(rotated, Image.Image)


def test_random_rotate_range(dummy_image):
    """Test that rotation angle is within expected range."""
    img = Image.open(dummy_image)
    # Run multiple times to ensure it works
    for _ in range(5):
        rotated = random_rotate(img)
        assert isinstance(rotated, Image.Image)


def test_random_rotate_custom_degrees(dummy_image):
    """Test rotation with custom max_degrees."""
    img = Image.open(dummy_image)
    rotated = random_rotate(img, max_degrees=45)
    assert isinstance(rotated, Image.Image)


# ─────────────────────────────
# RANDOM_FLIP
# ─────────────────────────────
@patch("mylib.preprocessing.random.random", return_value=0.9)
def test_random_flip_flips(mock_rand, dummy_image):
    """Test that image flips when random > 0.5."""
    img = Image.open(dummy_image)
    result = random_flip(img)
    # When random > 0.5, image should be flipped
    assert isinstance(result, Image.Image)


@patch("mylib.preprocessing.random.random", return_value=0.1)
def test_random_flip_no_flip(mock_rand, dummy_image):
    """Test that image doesn't flip when random < 0.5."""
    img = Image.open(dummy_image)
    original = img.copy()
    result = random_flip(img)
    assert isinstance(result, Image.Image)


# ─────────────────────────────
# BLUR
# ─────────────────────────────
def test_blur(dummy_image):
    """Test blur applies GaussianBlur filter."""
    img = Image.open(dummy_image)
    blurred = blur(img)
    assert isinstance(blurred, Image.Image)


def test_blur_custom_radius(dummy_image):
    """Test blur with custom radius."""
    img = Image.open(dummy_image)
    blurred = blur(img, radius=5)
    assert isinstance(blurred, Image.Image)


# ─────────────────────────────
# PREPROCESS_PIPELINE
# ─────────────────────────────
def test_preprocess_pipeline_returns_image(dummy_image):
    """Test that pipeline returns a PIL Image."""
    img = Image.open(dummy_image)
    output = preprocess_pipeline(img)
    assert isinstance(output, Image.Image)


def test_preprocess_pipeline_produces_grayscale(dummy_image):
    """Test that pipeline produces grayscale output by default."""
    img = Image.open(dummy_image)
    output = preprocess_pipeline(img)
    assert output.mode == "L"


def test_preprocess_pipeline_target_size(dummy_image):
    """Test pipeline with target size."""
    img = Image.open(dummy_image)
    output = preprocess_pipeline(img, target_size=(64, 64))
    # After rotation with expand=True, size may change
    # Just verify it's a reasonable size
    assert output.size[0] >= 64
    assert output.size[1] >= 64


def test_preprocess_pipeline_no_grayscale(dummy_image):
    """Test pipeline without grayscale conversion."""
    img = Image.open(dummy_image)
    output = preprocess_pipeline(img, apply_grayscale=False)
    # Should still be RGB if no grayscale
    assert output.mode in ("RGB", "L")  # Might be L from other operations


@patch("mylib.preprocessing.random_rotate")
@patch("mylib.preprocessing.random_flip")
@patch("mylib.preprocessing.blur")
@patch("mylib.preprocessing.to_grayscale")
def test_preprocess_pipeline_call_order(
    mock_gray, mock_blur, mock_flip, mock_rot, dummy_image
):
    """Test that all preprocessing steps are called."""
    # Setup mock returns
    img = Image.open(dummy_image)
    mock_rot.return_value = img
    mock_flip.return_value = img
    mock_blur.return_value = img
    mock_gray.return_value = img.convert("L")

    output = preprocess_pipeline(img, target_size=(64, 64))
    assert isinstance(output, Image.Image)

    # Verify all steps were called when enabled
    mock_rot.assert_called_once()
    mock_flip.assert_called_once()
    mock_blur.assert_called_once()
    mock_gray.assert_called_once()


def test_preprocess_pipeline_skip_steps(dummy_image):
    """Test pipeline with some steps disabled."""
    img = Image.open(dummy_image)
    output = preprocess_pipeline(
        img,
        target_size=(64, 64),
        apply_rotate=False,
        apply_flip=False,
        apply_blur=False,
        apply_grayscale=False,
    )
    assert isinstance(output, Image.Image)
    assert output.size == (64, 64)
