"""
Main CLI or app entry point
"""

import click
import io
from pathlib import Path
from mylib.image_classificator import predict_image_class, resize_image


@click.group()
def cli():
    """MLOps Image Classification Command Line Interface."""


def _validate_image_path(image_path: Path) -> None:
    """Helper function to validate the file extension."""
    if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        raise ValueError(
            "Invalid file extension. Only .jpg, .jpeg, or .png are allowed."
        )


@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
def predict(image_path: str):
    """
    Predicts the class of the image at IMAGE_PATH (randomly chosen for Lab 1).

    Parameters
    ----------
    image_path : str
        The file path to the image to be classified.
    """
    try:
        path = Path(image_path)
        _validate_image_path(path)

        with open(path, "rb") as f:
            image_bytes = io.BytesIO(f.read())

        predicted_class = predict_image_class(image_bytes)
        click.echo(f"The class predicted for {path.name} is: {predicted_class}")

    except (ValueError, FileNotFoundError) as e:
        click.echo(f"ERROR: {e}", err=True)


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--width", type=int, default=100, help="New width of the image.")
@click.option("--height", type=int, default=100, help="New height of the image.")
def resize(input_path: str, output_path: str, width: int, height: int):
    """
    Resizes the image at INPUT_PATH to WIDTHxHEIGHT and saves it to OUTPUT_PATH.

    Parameters
    ----------
    input_path : str
        The file path to the image to be resized.
    output_path : str
        The file path where the resized image will be saved.
    width : int
        The target width for the resized image.
    height : int
        The target height for the resized image.
    """
    try:
        input_path_obj = Path(input_path)
        output_path_obj = Path(output_path)

        _validate_image_path(input_path_obj)

        with open(input_path_obj, "rb") as f:
            image_bytes = io.BytesIO(f.read())

        resized_bytes = resize_image(image_bytes, width, height)

        with open(output_path_obj, "wb") as f:
            f.write(resized_bytes.read())

        click.echo(
            f"Image resized from {input_path_obj.name} to {width}x{height} "
            f"and saved to {output_path_obj.name}"
        )

    except (ValueError, FileNotFoundError) as e:
        click.echo(f"ERROR: {e}", err=True)


if __name__ == "__main__":
    cli()
