import io
import random
from PIL import Image

IMAGE_CLASSES = ["person", "airplane", "ball", "house", "truck"]


def resize_image(image_file: io.BytesIO, width: int, height: int) -> io.BytesIO:
    """
    Resizes an image uploaded as a BytesIO buffer to a specified width and height.

    Parameters
    ----------
    image_file : io.BytesIO
        Buffer containing the binary image data.
    width : int
        The new desired width in pixels.
    height : int
    # ... (docstring continues)

    Returns
    -------
    io.BytesIO
        Buffer containing the resized image data in JPEG format.

    Raises
    ------
    ValueError
        If the input file cannot be opened or processed by PIL.
    """
    try:
        img = Image.open(image_file)

        resized_img = img.resize((width, height))

        img_bytes = io.BytesIO()
        resized_img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        return img_bytes

    except Exception as e:
        # FIX: Added 'from e' to fix W0707
        raise ValueError(f"Error during image processing: {e}") from e


def convert_to_grayscale(image_file: io.BytesIO) -> io.BytesIO:
    """
    Converts the input image to grayscale.

    Parameters
    ----------
    image_file : io.BytesIO
        Buffer containing the binary image data.
    # ... (docstring continues)

    Returns
    -------
    io.BytesIO
        Buffer containing the grayscale image data in JPEG format.

    Raises
    ------
    ValueError
        If the input file cannot be opened or processed by PIL.
    """
    try:
        img = Image.open(image_file)
        grayscale_img = img.convert("L")

        img_bytes = io.BytesIO()
        grayscale_img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        return img_bytes

    except Exception as e:
        raise ValueError(f"Error during grayscale conversion: {e}") from e


def rotate_image(image_file: io.BytesIO, degrees: int) -> io.BytesIO:
    """
    Rotates the input image counter-clockwise by a specified number of degrees.

    Parameters
    ----------
    image_file : io.BytesIO
    # ... (docstring continues)

    Returns
    -------
    io.BytesIO
        Buffer containing the rotated image data in JPEG format.

    Raises
    ------
    ValueError
        If the input file cannot be opened or processed by PIL.
    """
    try:
        img = Image.open(image_file)
        rotated_img = img.rotate(degrees, expand=True)

        img_bytes = io.BytesIO()
        rotated_img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        return img_bytes

    except Exception as e:
        raise ValueError(f"Error during image rotation: {e}") from e


def predict_image_class(image_file: io.BytesIO) -> str:
    """
    Predicts the pet breed class using the ONNX model.

    Parameters
    ----------
    image_file : io.BytesIO
        Buffer containing the binary image data.

    Returns
    -------
    str
        The predicted class name from the model.

    Raises
    ------
    ValueError
        If the image cannot be processed or the model fails.
    """
    try:
        from mylib.onnx_classifier import get_classifier

        # Load image
        img = Image.open(image_file)

        # Get classifier and predict
        classifier = get_classifier()
        predicted_class = classifier.predict(img)

        return predicted_class

    except Exception as e:
        raise ValueError(f"Error during image classification: {e}") from e
