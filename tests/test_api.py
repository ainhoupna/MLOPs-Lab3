"""
Integration testing with the API

Unit Testing of the API endpoints
"""

import io
import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from PIL import Image
from unittest.mock import patch

from api.api import app

client = TestClient(app)


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
    yield img_bytes


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


# --- Tests ---

def test_api_home_page(test_client):
    """Tests the home page endpoint."""
    response = test_client.get("/")
    
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_api_predict_success(test_client, image_buffer, expected_classes):
    """Tests the /predict endpoint with a valid image upload."""
    files = {"file": ("test_image.jpg", image_buffer, "image/jpeg")}
    
    response = test_client.post("/predict", files=files)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "predicted_class" in data
    assert "confidence" in data
    assert "filename" in data
    assert data["filename"] == "test_image.jpg"
    assert isinstance(data["confidence"], float)


def test_api_predict_invalid_image_type(test_client):
    """Tests the /predict endpoint with an invalid file type."""
    invalid_file = io.BytesIO(b"This is not an image")
    files = {"file": ("test.txt", invalid_file, "text/plain")}
    
    response = test_client.post("/predict", files=files)
    
    # Should reject non-image MIME type
    assert response.status_code in (400, 500)


def test_api_resize_success(test_client, image_buffer):
    """Tests the /resize endpoint with valid width and height."""
    image_buffer.seek(0)
    files = {"file": ("test_image.jpg", image_buffer, "image/jpeg")}
    data = {"width": 50, "height": 50}
    
    response = test_client.post("/resize", files=files, data=data)
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    
    # Verify the resized image
    resized_img = Image.open(io.BytesIO(response.content))
    assert resized_img.size == (50, 50)


def test_api_resize_invalid_dimensions(test_client, image_buffer):
    """Tests the /resize endpoint with invalid dimensions (e.g., negative)."""
    image_buffer.seek(0)
    files = {"file": ("test_image.jpg", image_buffer, "image/jpeg")}
    data = {"width": -10, "height": 50}
    
    response = test_client.post("/resize", files=files, data=data)
    
    # Should handle gracefully (400 error or error message)
    assert response.status_code in (200, 400)


def test_api_grayscale_success(test_client, image_buffer):
    """Tests the /grayscale endpoint."""
    image_buffer.seek(0)
    files = {"file": ("test_image.jpg", image_buffer, "image/jpeg")}
    
    response = test_client.post("/grayscale", files=files)
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    
    # Verify it's grayscale
    gray_img = Image.open(io.BytesIO(response.content))
    assert gray_img.mode == "L"


def test_api_rotate_success(test_client, image_buffer):
    """Tests the /rotate endpoint."""
    image_buffer.seek(0)
    files = {"file": ("test_image.jpg", image_buffer, "image/jpeg")}
    data = {"degrees": 90}
    
    response = test_client.post("/rotate", files=files, data=data)
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    
    rotated_img = Image.open(io.BytesIO(response.content))
    assert rotated_img is not None


def test_api_rotate_negative_degrees(test_client, image_buffer):
    """Tests the /rotate endpoint with negative degrees."""
    image_buffer.seek(0)
    files = {"file": ("test_image.jpg", image_buffer, "image/jpeg")}
    data = {"degrees": -45}
    
    response = test_client.post("/rotate", files=files, data=data)
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"