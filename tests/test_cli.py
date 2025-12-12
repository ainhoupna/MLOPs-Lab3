"""
Unit Testing of the CLI with command groups
"""

import pytest
import json
from click.testing import CliRunner
from pathlib import Path
from PIL import Image
from unittest.mock import patch

from cli.cli import cli


# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def runner():
    """Create a CliRunner instance for testing Click commands."""
    return CliRunner()


@pytest.fixture
def test_image(tmp_path):
    """Create a temporary test image."""
    img_path = tmp_path / "test_image.jpg"
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img.save(img_path)
    return img_path


@pytest.fixture
def mock_outputs_dir(tmp_path, monkeypatch):
    """Mock the outputs directory to use a temporary path."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    
    # Mock ensure_output_dir
    def mock_ensure():
        outputs.mkdir(exist_ok=True)
        return outputs
    
    monkeypatch.setattr("cli.cli.ensure_output_dir", mock_ensure)
    
    # Mock Path to redirect outputs/
    original_path = Path
    
    def custom_path(x):
        if isinstance(x, str) and x.startswith("outputs/"):
            return outputs / x.replace("outputs/", "")
        return original_path(x)
    
    monkeypatch.setattr("cli.cli.Path", custom_path)
    
    return outputs


@pytest.fixture(scope="session")
def expected_classes():
    """Load class_labels.json or return fallback class names."""
    class_labels_path = Path("class_labels.json")

    if class_labels_path.exists():
        with open(class_labels_path, encoding='utf-8') as f:
            labels = json.load(f)
            # Handle both dict and list formats
            if isinstance(labels, dict):
                return list(labels.values())
            return labels

    # Fallback for CI
    return [
        "Abyssinian", "American_Bulldog", "American_Pit_Bull_Terrier",
        "Basset_Hound", "Beagle", "Bengal", "Birman", "Bombay", "Boxer",
        "British_Shorthair", "Chihuahua", "Egyptian_Mau",
        "English_Cocker_Spaniel", "English_Setter", "German_Shorthaired",
        "Great_Pyrenees", "Havanese", "Japanese_Chin", "Keeshond",
        "Leonberger", "Maine_Coon", "Miniature_Pinscher", "Newfoundland",
        "Persian", "Pomeranian", "Pug", "Ragdoll", "Russian_Blue",
        "Saint_Bernard", "Samoyed", "Scottish_Terrier", "Shiba_Inu",
        "Siamese", "Sphynx", "Staffordshire_Bull_Terrier",
        "Wheaten_Terrier", "Yorkshire_Terrier",
    ]


# ═════════════════════════════════════════════════════════════
# CLASSIFY GROUP TESTS
# ═════════════════════════════════════════════════════════════

def test_classify_predict_success(runner, test_image, expected_classes):
    """Test successful prediction with valid image."""
    result = runner.invoke(cli, ["classify", "predict", str(test_image)])
    
    assert result.exit_code == 0
    assert "Predicted class:" in result.output
    assert str(test_image) in result.output or "test_image" in result.output


def test_classify_predict_nonexistent_image(runner):
    """Test prediction with non-existent image file."""
    result = runner.invoke(cli, ["classify", "predict", "nonexistent.jpg"])
    
    # Click returns exit code 2 for missing file
    assert result.exit_code == 2


# ═════════════════════════════════════════════════════════════
# PREPROCESS GROUP TESTS
# ═════════════════════════════════════════════════════════════

def test_preprocess_resize_with_dimensions(runner, test_image, tmp_path):
    """Test resize with explicit width and height."""
    output_path = tmp_path / "resized.jpg"
    
    result = runner.invoke(cli, [
        "preprocess", "resize", str(test_image),
        "--width", "50",
        "--height", "60",
        "--output", str(output_path)
    ])
    
    assert result.exit_code == 0
    assert output_path.exists()
    
    # Verify dimensions
    img = Image.open(output_path)
    assert img.size == (50, 60)


def test_preprocess_resize_random_size(runner, test_image, tmp_path):
    """Test resize with random dimensions."""
    output_path = tmp_path / "resized_random.jpg"
    
    result = runner.invoke(cli, [
        "preprocess", "resize", str(test_image),
        "--output", str(output_path)
    ])
    
    assert result.exit_code == 0
    assert output_path.exists()
    
    # Verify dimensions are within random range
    img = Image.open(output_path)
    assert 28 <= img.size[0] <= 225
    assert 28 <= img.size[1] <= 225


def test_preprocess_grayscale(runner, test_image, tmp_path):
    """Test grayscale conversion."""
    output_path = tmp_path / "grayscale.jpg"
    
    result = runner.invoke(cli, [
        "preprocess", "grayscale", str(test_image),
        "--output", str(output_path)
    ])
    
    assert result.exit_code == 0
    assert output_path.exists()
    
    # Verify it's grayscale
    img = Image.open(output_path)
    assert img.mode == "L"


def test_preprocess_rotate(runner, test_image, tmp_path):
    """Test random rotation."""
    output_path = tmp_path / "rotated.jpg"
    
    result = runner.invoke(cli, [
        "preprocess", "rotate", str(test_image),
        "--output", str(output_path)
    ])
    
    assert result.exit_code == 0
    assert output_path.exists()


def test_preprocess_flip(runner, test_image, tmp_path):
    """Test random horizontal flip."""
    output_path = tmp_path / "flipped.jpg"
    
    result = runner.invoke(cli, [
        "preprocess", "flip", str(test_image),
        "--output", str(output_path)
    ])
    
    assert result.exit_code == 0
    assert output_path.exists()


def test_preprocess_blur(runner, test_image, tmp_path):
    """Test Gaussian blur."""
    output_path = tmp_path / "blurred.jpg"
    
    result = runner.invoke(cli, [
        "preprocess", "blur", str(test_image),
        "--output", str(output_path)
    ])
    
    assert result.exit_code == 0
    assert output_path.exists()


def test_preprocess_pipeline(runner, test_image, tmp_path):
    """Test full preprocessing pipeline."""
    output_path = tmp_path / "processed.jpg"
    
    result = runner.invoke(cli, [
        "preprocess", "pipeline", str(test_image),
        "--output", str(output_path)
    ])
    
    assert result.exit_code == 0
    assert output_path.exists()
    
    # Pipeline includes grayscale
    img = Image.open(output_path)
    assert img.mode == "L"


# ═════════════════════════════════════════════════════════════
# CLI GROUP STRUCTURE TESTS
# ═════════════════════════════════════════════════════════════

def test_cli_help(runner):
    """Test main CLI help message."""
    result = runner.invoke(cli, ["--help"])
    
    assert result.exit_code == 0
    assert "classify" in result.output
    assert "preprocess" in result.output


def test_classify_group_help(runner):
    """Test classify group help message."""
    result = runner.invoke(cli, ["classify", "--help"])
    
    assert result.exit_code == 0
    assert "predict" in result.output


def test_preprocess_group_help(runner):
    """Test preprocess group help message."""
    result = runner.invoke(cli, ["preprocess", "--help"])
    
    assert result.exit_code == 0
    assert "resize" in result.output
    assert "grayscale" in result.output
    assert "rotate" in result.output
    assert "flip" in result.output
    assert "blur" in result.output
    assert "pipeline" in result.output