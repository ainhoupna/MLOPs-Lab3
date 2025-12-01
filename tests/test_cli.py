"""
Integration testing with the CLI

"""

import os
from pathlib import Path
import pytest
from click.testing import CliRunner
from cli.cli import cli

# --- Fixtures for CLI Tests ---

@pytest.fixture(scope="module")
def cli_runner():
    """Fixture for the CliRunner."""
    return CliRunner()

@pytest.fixture(scope="module")
def sample_image_path(tmp_path_factory):
    """Creates a temporary JPEG image file for use in tests."""
    from PIL import Image
    img_dir = tmp_path_factory.mktemp("data_cli")
    img_path = img_dir / "test_image.jpg"
    
    img = Image.new('RGB', (10, 10), color = 'red')
    img.save(img_path, "jpeg")
    
    return img_path

# --- Tests ---

def test_cli_predict_success(cli_runner, sample_image_path):
    """Tests the 'predict' command with a valid image path."""
    result = cli_runner.invoke(cli, ["predict", str(sample_image_path)])
    
    assert result.exit_code == 0
    assert "predicted" in result.output

def test_cli_resize_success(cli_runner, sample_image_path, tmp_path):
    """Tests the 'resize' command with width and height options."""
    output_path = tmp_path / "resized_output.jpg"
    
    args = [
        "resize", 
        str(sample_image_path), 
        str(output_path), 
        "--width", "50", 
        "--height", "50"
    ]
    result = cli_runner.invoke(cli, args)
    
    assert result.exit_code == 0
    assert output_path.exists()
    assert "Image resized" in result.output
    
def test_cli_predict_file_not_found(cli_runner):
    """Tests the 'predict' command with a non-existent file."""
    result = cli_runner.invoke(cli, ["predict", "non_existent_file.jpg"])
    
    assert result.exit_code != 0
    assert "Path 'non_existent_file.jpg' does not exist." in result.output