import io
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from mylib.image_classificator import (
    predict_image_class,
    resize_image,
    convert_to_grayscale,  # Nuevo
    rotate_image,  # Nuevo
)

# Configure FastAPI
app = FastAPI(
    title="MLOps Image Classification API",
    description="API to perform image classification (simulated) and resizing, with extra transformations.",
)

# Configure templates (directory="templates")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Renders the home page of the API, which includes a demo form.

    Parameters
    ----------
    request : Request
        The incoming HTTP request object.

    Returns
    -------
    TemplateResponse
        The rendered home.html template.
    """
    return templates.TemplateResponse(
        "home.html", {"request": request, "api_title": app.title}
    )


def _validate_image_type(content_type: str):
    """Helper function to validate the MIME type."""
    if content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400, detail="Invalid image format. Only JPEG/PNG allowed."
        )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image file and returns a mock prediction (randomly chosen class).

    Parameters
    ----------
    file : UploadFile
        The image file uploaded via multipart/form-data.

    Returns
    -------
    dict
        A dictionary containing the filename and the predicted class.

    Raises
    ------
    HTTPException
        If the file format is invalid (400) or an internal error occurs (500).
    """
    _validate_image_type(file.content_type)

    try:
        image_bytes = io.BytesIO(await file.read())
        predicted_class = predict_image_class(image_bytes)

        return {"filename": file.filename, "predicted_class": predicted_class}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/resize")
async def resize(
    file: UploadFile = File(...), width: int = Form(...), height: int = Form(...)
):
    """
    Accepts an image file and resizes it to the specified dimensions.

    Parameters
    ----------
    file : UploadFile
        The image file uploaded via multipart/form-data.
    width : int
        The target width for the resized image.
    height : int
        The target height for the resized image.

    Returns
    -------
    StreamingResponse
        The resized image data returned as a JPEG stream.

    Raises
    ------
    HTTPException
        If the file format is invalid (400), resizing fails (400), or for internal errors (500).
    """
    _validate_image_type(file.content_type)

    try:
        image_bytes = io.BytesIO(await file.read())
        resized_bytes = resize_image(image_bytes, width, height)

        return StreamingResponse(resized_bytes, media_type="image/jpeg")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# --- NUEVOS ENDPOINTS PARA LAS TRANSFORMACIONES EXTRA ---


@app.post("/grayscale")
async def grayscale(file: UploadFile = File(...)):
    """
    Accepts an image file and converts it to grayscale.

    Parameters
    ----------
    file : UploadFile
        The image file uploaded via multipart/form-data.

    Returns
    -------
    StreamingResponse
        The grayscale image data returned as a JPEG stream.

    Raises
    ------
    HTTPException
        If the file format is invalid (400) or processing fails (400/500).
    """
    _validate_image_type(file.content_type)

    try:
        image_bytes = io.BytesIO(await file.read())
        grayscale_bytes = convert_to_grayscale(image_bytes)

        return StreamingResponse(grayscale_bytes, media_type="image/jpeg")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/rotate")
async def rotate(file: UploadFile = File(...), degrees: int = Form(...)):
    """
    Accepts an image file and rotates it by the specified degrees (counter-clockwise).

    Parameters
    ----------
    file : UploadFile
        The image file uploaded via multipart/form-data.
    degrees : int
        The angle of rotation in degrees (e.g., 90, 180, 270).

    Returns
    -------
    StreamingResponse
        The rotated image data returned as a JPEG stream.

    Raises
    ------
    HTTPException
        If the file format is invalid (400) or processing fails (400/500).
    """
    _validate_image_type(file.content_type)

    try:
        image_bytes = io.BytesIO(await file.read())
        rotated_bytes = rotate_image(image_bytes, degrees)

        return StreamingResponse(rotated_bytes, media_type="image/jpeg")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
