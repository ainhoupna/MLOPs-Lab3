"""
Gradio Application for MLOps Lab3 Image Classification
This app provides a user-friendly interface to interact with the FastAPI backend hosted on Render.
"""

import gradio as gr
import requests
from PIL import Image
import io

API_URL = "https://mlops-lab3-latest-kx2m.onrender.com"

def predict_image(image):
    """
    Send image to the API for prediction.
    
    Args:
        image: PIL Image object from Gradio
        
    Returns:
        str: Predicted class label
    """
    if image is None:
        return "Please upload an image first."
    
    try:
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        # Send request to API
        files = {'file': ('image.jpg', img_byte_arr, 'image/jpeg')}
        response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return f"Predicted Class: **{result['predicted_class']}**"
        else:
            return f"Error: {response.status_code} - {response.text}"
            
    except requests.exceptions.Timeout:
        return "Request timed out. The API might be waking up (cold start). Please try again in a few seconds."
    except requests.exceptions.RequestException as e:
        return f"Connection Error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


def resize_image(image, width, height):
    """
    Resize image using the API.
    
    Args:
        image: PIL Image object from Gradio
        width: Target width
        height: Target height
        
    Returns:
        PIL Image: Resized image
    """
    if image is None:
        return None
    
    try:
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        # Send request to API
        files = {'file': ('image.jpg', img_byte_arr, 'image/jpeg')}
        data = {'width': width, 'height': height}
        response = requests.post(f"{API_URL}/resize", files=files, data=data, timeout=30)
        
        if response.status_code == 200:
            # Convert response bytes to PIL Image
            resized_image = Image.open(io.BytesIO(response.content))
            return resized_image
        else:
            return None
            
    except Exception as e:
        print(f"Error resizing image: {e}")
        return None


def grayscale_image(image):
    """
    Convert image to grayscale using the API.
    
    Args:
        image: PIL Image object from Gradio
        
    Returns:
        PIL Image: Grayscale image
    """
    if image is None:
        return None
    
    try:
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        # Send request to API
        files = {'file': ('image.jpg', img_byte_arr, 'image/jpeg')}
        response = requests.post(f"{API_URL}/grayscale", files=files, timeout=30)
        
        if response.status_code == 200:
            # Convert response bytes to PIL Image
            gray_image = Image.open(io.BytesIO(response.content))
            return gray_image
        else:
            return None
            
    except Exception as e:
        print(f"Error converting to grayscale: {e}")
        return None


def rotate_image(image, degrees):
    """
    Rotate image using the API.
    
    Args:
        image: PIL Image object from Gradio
        degrees: Rotation angle in degrees
        
    Returns:
        PIL Image: Rotated image
    """
    if image is None:
        return None
    
    try:
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        # Send request to API
        files = {'file': ('image.jpg', img_byte_arr, 'image/jpeg')}
        data = {'degrees': degrees}
        response = requests.post(f"{API_URL}/rotate", files=files, data=data, timeout=30)
        
        if response.status_code == 200:
            # Convert response bytes to PIL Image
            rotated_image = Image.open(io.BytesIO(response.content))
            return rotated_image
        else:
            return None
            
    except Exception as e:
        print(f"Error rotating image: {e}")
        return None


# Create Gradio interface with tabs
with gr.Blocks(title="MLOps Lab3 - Image Classification") as demo:
    gr.Markdown(
        """
        # MLOps Lab3 - Image Classification and Processing
        
        This application demonstrates a complete MLOps pipeline with:
        - **FastAPI** backend hosted on **Render**
        - **Docker** containerization
        - **CI/CD** with GitHub Actions
        - **Gradio** frontend on **HuggingFace Spaces**
        
        Upload an image and try the different features!
        """
    )
    
    with gr.Tabs():
        # Tab 1: Prediction
        with gr.Tab("Predict"):
            with gr.Row():
                with gr.Column():
                    predict_input = gr.Image(type="pil", label="Upload Image")
                    predict_btn = gr.Button("Predict Class", variant="primary")
                with gr.Column():
                    predict_output = gr.Markdown(label="Prediction Result")
            
            predict_btn.click(
                fn=predict_image,
                inputs=predict_input,
                outputs=predict_output
            )
        
        # Tab 2: Resize
        with gr.Tab("Resize"):
            with gr.Row():
                with gr.Column():
                    resize_input = gr.Image(type="pil", label="Upload Image")
                    resize_width = gr.Slider(minimum=50, maximum=1000, value=300, step=10, label="Width")
                    resize_height = gr.Slider(minimum=50, maximum=1000, value=300, step=10, label="Height")
                    resize_btn = gr.Button("Resize Image", variant="primary")
                with gr.Column():
                    resize_output = gr.Image(type="pil", label="Resized Image")
            
            resize_btn.click(
                fn=resize_image,
                inputs=[resize_input, resize_width, resize_height],
                outputs=resize_output
            )
        
        # Tab 3: Grayscale
        with gr.Tab("Grayscale"):
            with gr.Row():
                with gr.Column():
                    gray_input = gr.Image(type="pil", label="Upload Image")
                    gray_btn = gr.Button("Convert to Grayscale", variant="primary")
                with gr.Column():
                    gray_output = gr.Image(type="pil", label="Grayscale Image")
            
            gray_btn.click(
                fn=grayscale_image,
                inputs=gray_input,
                outputs=gray_output
            )
        
        # Tab 4: Rotate
        with gr.Tab("Rotate"):
            with gr.Row():
                with gr.Column():
                    rotate_input = gr.Image(type="pil", label="Upload Image")
                    rotate_degrees = gr.Slider(minimum=0, maximum=360, value=90, step=15, label="Rotation Degrees")
                    rotate_btn = gr.Button("Rotate Image", variant="primary")
                with gr.Column():
                    rotate_output = gr.Image(type="pil", label="Rotated Image")
            
            rotate_btn.click(
                fn=rotate_image,
                inputs=[rotate_input, rotate_degrees],
                outputs=rotate_output
            )
    
    gr.Markdown(
        """
        ---
        ### Note
        - The first request might take 30-60 seconds due to cold start (free tier limitation)
        - Subsequent requests will be faster
        - API is hosted on Render (free tier)
        """
    )

if __name__ == "__main__":
    demo.launch()