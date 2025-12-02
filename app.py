"""
Gradio Application for MLOps Lab3 Image Classification
Compatible with Gradio 4.x
"""

import gradio as gr
import requests
from PIL import Image
import io

# URL of the API created with FastAPI
API_URL = "https://mlops-lab3-latest-kx2m.onrender.com"


def predict_image(image):
    """
    Send an image to the API for prediction
    """
    try:
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Send to API
        files = {"file": ("image.png", img_byte_arr, "image/png")}
        response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        predicted_class = data.get("predicted_class", "Unknown")
        confidence = data.get("confidence", 0.0)
        
        return f"**Prediction:** {predicted_class}\n\n**Confidence:** {confidence:.2%}"
    
    except requests.exceptions.Timeout:
        return "Error: Request timed out. The API might be starting up (cold start)."
    except requests.exceptions.HTTPError as e:
        error_detail = "Unknown error"
        try:
            error_detail = response.json().get('detail', str(e))
        except:
            error_detail = str(e)
        return f"Error: {error_detail}"
    except Exception as e:
        return f"Error: {str(e)}"


# Create Gradio interface
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload Pet Image"),
    outputs=gr.Textbox(label="Prediction Result"),
    title="MLOps Lab3 - Pet Breed Classifier",
    description="Upload an image of a dog or cat to classify its breed. Powered by ResNet50 + ONNX Runtime.",
    allow_flagging="never"
)

# Launch the GUI
if __name__ == "__main__":
    iface.launch()