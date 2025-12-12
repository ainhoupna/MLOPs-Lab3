"""
Image preprocessing utilities for CLI commands.
"""

import random
from pathlib import Path
from PIL import Image, ImageFilter


def ensure_output_dir(output_dir: str = "outputs") -> None:
    """
    Ensure the output directory exists, creating it if necessary.
    
    Args:
        output_dir: Path to output directory (default: "outputs")
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def to_grayscale(img: Image.Image) -> Image.Image:
    """
    Convert image to grayscale.
    
    Args:
        img: PIL Image object
        
    Returns:
        Grayscale PIL Image
    """
    return img.convert("L")


def random_rotate(img: Image.Image, max_degrees: int = 20) -> Image.Image:
    """
    Randomly rotate image by up to ±max_degrees.
    
    Args:
        img: PIL Image object
        max_degrees: Maximum rotation in degrees (default: 20)
        
    Returns:
        Rotated PIL Image
    """
    angle = random.uniform(-max_degrees, max_degrees)
    return img.rotate(angle, expand=True, fillcolor=(255, 255, 255))


def random_flip(img: Image.Image, probability: float = 0.5) -> Image.Image:
    """
    Randomly flip image horizontally with given probability.
    
    Args:
        img: PIL Image object
        probability: Probability of flipping (default: 0.5)
        
    Returns:
        PIL Image (flipped or original)
    """
    if random.random() < probability:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def blur(img: Image.Image, radius: int = 2) -> Image.Image:
    """
    Apply Gaussian blur to image.
    
    Args:
        img: PIL Image object
        radius: Blur radius (default: 2)
        
    Returns:
        Blurred PIL Image
    """
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def preprocess_pipeline(
    img: Image.Image,
    target_size: tuple = None,
    apply_grayscale: bool = True,
    apply_rotate: bool = True,
    apply_flip: bool = True,
    apply_blur: bool = True,
) -> Image.Image:
    """
    Apply full preprocessing pipeline to image.
    
    Args:
        img: PIL Image object
        target_size: Target size as (width, height), None for random
        apply_grayscale: Whether to convert to grayscale
        apply_rotate: Whether to apply random rotation
        apply_flip: Whether to apply random flip
        apply_blur: Whether to apply blur
        
    Returns:
        Preprocessed PIL Image
    """
    # Resize
    if target_size:
        img = img.resize(target_size)
    else:
        # Random size between 28 and 225
        size = random.randint(28, 225)
        img = img.resize((size, size))
    
    # Apply transformations in sequence
    if apply_rotate:
        img = random_rotate(img)
    
    if apply_flip:
        img = random_flip(img)
    
    if apply_blur:
        img = blur(img)
    
    if apply_grayscale:
        img = to_grayscale(img)
    
    return img
