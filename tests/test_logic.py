"""
Unit Tests for the image processing logic (mylib.image_classificator).
"""

import io
import pytest
from PIL import Image

from mylib.image_classificator import predict_image_class, resize_image, rotate_image, convert_to_grayscale


# --- Fixtures for Logic Tests ---

@pytest.fixture(scope="module")
def sample_image_path(tmp_path_factory):
    """Creates a temporary JPEG image file for use in tests."""
    img_dir = tmp_path_factory.mktemp("data")
    img_path = img_dir / "test_image.jpg"
    
    # Create a simple 10x10 pixel image
    img = Image.new('RGB', (10, 10), color = 'red')
    img.save(img_path, "jpeg")
    
    return img_path

@pytest.fixture
def image_buffer(sample_image_path):
    """Reads the test image into an io.BytesIO buffer."""
    with open(sample_image_path, "rb") as f:
        img_bytes = io.BytesIO(f.read())
    
    img_bytes.seek(0)
    yield img_bytes

# --- Tests ---

def test_logic_predict_returns_valid_string(image_buffer):
    """Test that predict_image_class returns a non-empty string."""
    prediction = predict_image_class(image_buffer)
    assert isinstance(prediction, str)
    assert len(prediction) > 0

def test_logic_resize_returns_bytesio_and_correct_size(image_buffer):
    """Test that resize_image returns a BytesIO and the image has the correct new size."""
    new_width = 75
    new_height = 75
    
    resized_buffer = resize_image(image_buffer, new_width, new_height)
    
    assert isinstance(resized_buffer, io.BytesIO)
    
    resized_img = Image.open(resized_buffer)
    assert resized_img.size == (new_width, new_height)

def test_logic_resize_raises_error_on_invalid_data():
    """Test that resize_image raises an error when given non-image data."""
    invalid_buffer = io.BytesIO(b"This is not an image")
    
    with pytest.raises(ValueError):
        resize_image(invalid_buffer, 50, 50)

def test_logic_convert_to_grayscale_returns_grayscale_image(image_buffer):
    """Test that convert_to_grayscale returns a grayscale image."""
    grayscale_buffer = convert_to_grayscale(image_buffer)
    
    assert isinstance(grayscale_buffer, io.BytesIO)
    
    grayscale_img = Image.open(grayscale_buffer)
    assert grayscale_img.mode == "L"

def test_logic_rotate_image_returns_rotated_image(image_buffer):
    """Test that rotate_image returns a rotated image."""
    degrees = 90
    
    rotated_buffer = rotate_image(image_buffer, degrees)
    
    assert isinstance(rotated_buffer, io.BytesIO)
    
    rotated_img = Image.open(rotated_buffer)
    assert rotated_img is not None

def test_logic_rotate_image_negative_degrees(image_buffer):
    """Test rotation with negative degrees."""
    degrees = -45
    
    rotated_buffer = rotate_image(image_buffer, degrees)
    
    assert isinstance(rotated_buffer, io.BytesIO)
    rotated_img = Image.open(rotated_buffer)
    assert rotated_img is not None

def test_logic_resize_preserves_image_format(image_buffer):
    """Test that resize preserves JPEG format."""
    resized_buffer = resize_image(image_buffer, 50, 50)
    
    resized_img = Image.open(resized_buffer)
    assert resized_img.format == "JPEG"

def test_logic_grayscale_preserves_dimensions(image_buffer):
    """Test that grayscale conversion preserves image dimensions."""
    original_img = Image.open(image_buffer)
    original_size = original_img.size
    
    image_buffer.seek(0)  # Reset
    grayscale_buffer = convert_to_grayscale(image_buffer)
    
    grayscale_img = Image.open(grayscale_buffer)
    assert grayscale_img.size == original_size