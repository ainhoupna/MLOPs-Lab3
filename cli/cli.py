"""
Main CLI or app entry point with command groups.
"""

import click
import io
from pathlib import Path
from PIL import Image
from mylib.image_classificator import predict_image_class, resize_image
from mylib.preprocessing import (
    to_grayscale,
    random_rotate,
    random_flip,
    blur,
    preprocess_pipeline,
    ensure_output_dir,
)


# ─────────────────────────────
# MAIN GROUP
# ─────────────────────────────
@click.group(help="MLOps Image Classification Command Line Interface.")
def cli():
    """
    Main entry point of the CLI.

    Commands:
        - classify
        - preprocess
    """


def _validate_image_path(image_path: Path) -> None:
    """Helper function to validate the file extension."""
    if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        raise ValueError(
            "Invalid file extension. Only .jpg, .jpeg, or .png are allowed."
        )


# ─────────────────────────────
# CLASSIFICATION GROUP
# ─────────────────────────────
@cli.group(help="Commands related to classification.")
def classify():
    """Group for prediction-related commands."""


@classify.command(
    name="predict",
    help="Predict the class of an image using the ONNX model. "
    "Example: python -m cli.cli classify predict image.jpg",
)
@click.argument("image_path", type=click.Path(exists=True))
def classify_predict(image_path: str):
    """
    Predicts the class of the image using the ONNX model.

    Parameters
    ----------
    image_path : str
        The file path to the image to be classified.
        
    Example:
        python -m cli.cli classify predict image.jpg
    """
    try:
        path = Path(image_path)
        _validate_image_path(path)

        with open(path, "rb") as f:
            image_bytes = io.BytesIO(f.read())

        predicted_class = predict_image_class(image_bytes)
        click.echo(f"Image: {image_path}")
        click.echo(f"Predicted class: {predicted_class}")

    except (ValueError, FileNotFoundError) as e:
        click.echo(f"ERROR: {e}", err=True)


# ─────────────────────────────
# PREPROCESSING GROUP
# ─────────────────────────────
@cli.group(help="Commands related to preprocessing images.")
def preprocess():
    """Group for preprocessing commands."""


@preprocess.command(
    name="resize",
    help="Resize an image to a specified size (or random 28-225 if not specified). "
    "Example: python -m cli.cli preprocess resize input.jpg",
)
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--width", type=int, default=None, help="Target width (random if not specified)"
)
@click.option(
    "--height", type=int, default=None, help="Target height (random if not specified)"
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output path (default: outputs/resized_<filename>)",
)
def preprocess_resize(input_path: str, width: int, height: int, output: str):
    """
    Resize an image and save the result.

    Args:
        input_path: Input image path
        width: Target width
        height: Target height
        output: Path to save resized image
    """
    try:
        ensure_output_dir()

        input_path_obj = Path(input_path)
        _validate_image_path(input_path_obj)

        # Generate output path if not provided
        if output is None:
            output = f"outputs/resized_{input_path_obj.name}"

        # Use existing resize_image function or manual resize
        if width and height:
            with open(input_path_obj, "rb") as f:
                image_bytes = io.BytesIO(f.read())
            resized_bytes = resize_image(image_bytes, width, height)
            with open(output, "wb") as f:
                f.write(resized_bytes.read())
        else:
            # Random size
            img = Image.open(input_path)
            import random
            size = random.randint(28, 225)
            img = img.resize((size, size))
            img.save(output)

        click.echo(f"Saved resized image to: {output}")

    except (ValueError, FileNotFoundError) as e:
        click.echo(f"ERROR: {e}", err=True)


@preprocess.command(name="grayscale", help="Convert image to grayscale.")
@click.argument("image_path", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output path (default: outputs/grayscale_<filename>)",
)
def preprocess_grayscale(image_path: str, output: str):
    """Convert image to grayscale."""
    try:
        ensure_output_dir()

        input_file = Path(image_path)
        _validate_image_path(input_file)

        if output is None:
            output = f"outputs/grayscale_{input_file.name}"

        img = Image.open(image_path)
        img = to_grayscale(img)
        img.save(output)
        click.echo(f"Saved grayscale image to: {output}")

    except (ValueError, FileNotFoundError) as e:
        click.echo(f"ERROR: {e}", err=True)


@preprocess.command(
    name="rotate", help="Randomly rotate the image by up to ±20 degrees."
)
@click.argument("image_path", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output path (default: outputs/rotated_<filename>)",
)
def preprocess_rotate(image_path: str, output: str):
    """Randomly rotate image."""
    try:
        ensure_output_dir()

        input_file = Path(image_path)
        _validate_image_path(input_file)

        if output is None:
            output = f"outputs/rotated_{input_file.name}"

        img = Image.open(image_path)
        img = random_rotate(img)
        img.save(output)
        click.echo(f"Saved rotated image to: {output}")

    except (ValueError, FileNotFoundError) as e:
        click.echo(f"ERROR: {e}", err=True)


@preprocess.command(
    name="flip", help="Randomly flip the image horizontally (50% probability)."
)
@click.argument("image_path", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output path (default: outputs/flipped_<filename>)",
)
def preprocess_flip(image_path: str, output: str):
    """Randomly flip image horizontally."""
    try:
        ensure_output_dir()

        input_file = Path(image_path)
        _validate_image_path(input_file)

        if output is None:
            output = f"outputs/flipped_{input_file.name}"

        img = Image.open(image_path)
        img = random_flip(img)
        img.save(output)
        click.echo(f"Saved flipped image to: {output}")

    except (ValueError, FileNotFoundError) as e:
        click.echo(f"ERROR: {e}", err=True)


@preprocess.command(name="blur", help="Apply Gaussian blur to the image.")
@click.argument("image_path", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output path (default: outputs/blurred_<filename>)",
)
def preprocess_blur(image_path: str, output: str):
    """Apply Gaussian blur."""
    try:
        ensure_output_dir()

        input_file = Path(image_path)
        _validate_image_path(input_file)

        if output is None:
            output = f"outputs/blurred_{input_file.name}"

        img = Image.open(image_path)
        img = blur(img)
        img.save(output)
        click.echo(f"Saved blurred image to: {output}")

    except (ValueError, FileNotFoundError) as e:
        click.echo(f"ERROR: {e}", err=True)


@preprocess.command(
    name="pipeline",
    help="Apply full preprocessing pipeline (resize, rotate, flip, blur, grayscale).",
)
@click.argument("image_path", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output path (default: outputs/processed_<filename>)",
)
def preprocess_full_pipeline(image_path: str, output: str):
    """Apply full preprocessing pipeline."""
    try:
        ensure_output_dir()

        input_file = Path(image_path)
        _validate_image_path(input_file)

        if output is None:
            output = f"outputs/processed_{input_file.name}"

        img = Image.open(image_path)
        img = preprocess_pipeline(img)
        img.save(output)
        click.echo(f"Saved fully preprocessed image to: {output}")

    except (ValueError, FileNotFoundError) as e:
        click.echo(f"ERROR: {e}", err=True)


# ─────────────────────────────
# ENTRY POINT
# ─────────────────────────────
if __name__ == "__main__":  # pragma: no cover
    cli()
