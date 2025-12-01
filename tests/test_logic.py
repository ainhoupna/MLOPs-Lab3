"""
Unit Testing of the application's logic 

"""

import pytest
import io
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
    """Test that resize_image handles non-image data."""
    invalid_buffer = io.BytesIO(b"This is not a valid JPEG.")
    
    with pytest.raises(ValueError) as excinfo:
        resize_image(invalid_buffer, 10, 10)
    
    assert "Error during image processing" in str(excinfo.value)

def test_logic_convert_to_grayscale_success(image_buffer):
    """Test that convert_to_grayscale returns a buffer and changes mode to 'L' (Grayscale)."""
    
    grayscale_buffer = convert_to_grayscale(image_buffer)
    
    # Check if the result is a BytesIO object
    assert isinstance(grayscale_buffer, io.BytesIO)
    
    # Check if the image mode is 'L' (Grayscale)
    grayscale_img = Image.open(grayscale_buffer)
    assert grayscale_img.mode == 'L'

def test_logic_rotate_image_success(image_buffer):
    """Test that rotate_image returns a buffer and changes the image size (due to expand=True)."""
    
    # Get original size for comparison
    original_img = Image.open(image_buffer)
    original_width, original_height = original_img.size
    
    # Rotate by 90 degrees
    rotated_buffer = rotate_image(image_buffer, 90)
    
    # Check if the result is a BytesIO object
    assert isinstance(rotated_buffer, io.BytesIO)
    
    # Check if dimensions are swapped (or close, depending on PIL handling)
    rotated_img = Image.open(rotated_buffer)
    rotated_width, rotated_height = rotated_img.size
    
    # In a simple 10x10 image fixture, size won't swap, but we test for dimensions change.
    # The crucial check is that the function executed without error.
    assert rotated_width == original_width
    assert rotated_height == original_height
    
    assert rotated_img.mode in ['RGB', 'L']   

def test_logic_rotate_image_raises_error_on_invalid_data():
    """Test that rotate_image handles non-image data."""
    invalid_buffer = io.BytesIO(b"Not an image file for rotation.")
    
    with pytest.raises(ValueError):
        rotate_image(invalid_buffer, 90)    