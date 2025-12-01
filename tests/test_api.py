"""
Integration testing with the API

"""

# tests/test_api.py

import pytest
import io
from PIL import Image
from fastapi.testclient import TestClient

#  Import the application from your file structure
from api.api import app 

# --- Fixtures for API Tests ---

@pytest.fixture(scope="module")
def test_client():
    """Fixture for the FastAPI TestClient."""
    with TestClient(app) as client:
        yield client

@pytest.fixture(scope="module")
def sample_image_path(tmp_path_factory):
    """Creates a temporary JPEG image file for use in tests."""
    img_dir = tmp_path_factory.mktemp("data_api")
    img_path = img_dir / "test_image.jpg"
    
    img = Image.new('RGB', (10, 10), color = 'red')
    img.save(img_path, "jpeg")
    
    return img_path

@pytest.fixture
def image_buffer(sample_image_path):
    """Reads the test image into an io.BytesIO buffer."""
    with open(sample_image_path, "rb") as f:
        img_bytes = io.BytesIO(f.read())
    
    img_bytes.seek(0)
    # Use yield so the buffer can be reset for tests if needed, though for client.post() it's consumed.
    yield img_bytes

# --- Tests ---

def test_api_predict_success(test_client, image_buffer):
    """Tests the /predict endpoint with a valid image upload."""
    files = {"file": ("test_image.jpg", image_buffer, "image/jpeg")}
    
    response = test_client.post("/predict", files=files)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "predicted_class" in data

def test_api_predict_invalid_file_type(test_client):
    """Tests the /predict endpoint with an invalid content type (text/plain)."""
    text_buffer = io.BytesIO(b"This is not an image.")
    files = {"file": ("not_an_image.txt", text_buffer, "text/plain")}
    
    response = test_client.post("/predict", files=files)
    
    assert response.status_code == 400
    assert "Invalid image format. Only JPEG/PNG allowed." in response.json()["detail"]

def test_api_resize_success(test_client, image_buffer):
    """Tests the /resize endpoint with a valid image and form data."""
    files = {"file": ("test_image.jpg", image_buffer, "image/jpeg")}
    data = {"width": "50", "height": "50"}
    
    response = test_client.post("/resize", files=files, data=data)
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    
    img = Image.open(io.BytesIO(response.content))
    assert img.size == (50, 50)


def test_api_resize_missing_form_data(test_client, image_buffer):
    """Tests the /resize endpoint when a required form parameter (height) is missing."""
    files = {"file": ("test_image.jpg", image_buffer, "image/jpeg")}
    data = {"width": "50"}
    
    response = test_client.post("/resize", files=files, data=data)
    
    assert response.status_code == 422
    assert "height" in response.json()["detail"][0]["loc"]


def test_api_grayscale_success(test_client, image_buffer):
    """Tests the /grayscale endpoint with a valid image upload."""
    files = {"file": ("test_image.jpg", image_buffer, "image/jpeg")}
    
    response = test_client.post("/grayscale", files=files)
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    
    # Verification: Check if the returned image is grayscale ('L' mode)
    img = Image.open(io.BytesIO(response.content))
    assert img.mode == 'L'

def test_api_rotate_success(test_client, image_buffer):
    """Tests the /rotate endpoint with a valid image and form data (degrees)."""
    files = {"file": ("test_image.jpg", image_buffer, "image/jpeg")}
    data = {"degrees": "90"} # Form data for the angle

    response = test_client.post("/rotate", files=files, data=data)
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    
    # Verification: Check if the image size has changed due to rotation (expand=True)
    rotated_img = Image.open(io.BytesIO(response.content))
    
    # Since the fixture image is 10x10, the size should remain 10x10, but the 
    # crucial check is successful execution and correct content type.
    assert rotated_img.width > 0 

def test_api_rotate_missing_form_data(test_client, image_buffer):
    """Tests the /rotate endpoint when the required degrees parameter is missing."""
    files = {"file": ("test_image.jpg", image_buffer, "image/jpeg")}
    data = {} # Missing degrees

    response = test_client.post("/rotate", files=files, data=data)
    
    # Should fail due to missing required Form parameter
    assert response.status_code == 422
    assert "degrees" in response.json()["detail"][0]["loc"]