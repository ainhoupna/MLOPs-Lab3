import gradio as gr
import requests
import io
from PIL import Image

RENDER_API_URL = "https://mlops-lab3-n3sg.onrender.com"
PREDICT_ENDPOINT = f"{RENDER_API_URL}/predict"


def predict_pet_breed(image: Image.Image):
    """
    Handles the Gradio image input, sends it to the Render FastAPI /predict endpoint,
    and returns the predicted class label.
    """
    if image is None:
        return "Please upload an image to predict its class."

    # 1. Convert the PIL Image object (received from Gradio) to a byte buffer
    img_byte_arr = io.BytesIO()
    # Save the image as PNG into the buffer
    image.save(img_byte_arr, format='PNG') 
    img_byte_arr.seek(0)  # Rewind the buffer pointer to the start

    # 2. Prepare file payload for multipart/form-data upload
    # 'file' must match the parameter name in your FastAPI endpoint: file: UploadFile = File(...)
    files = {'file': ('image.png', img_byte_arr, 'image/png')}

    try:
        # 3. Send POST request to the remote FastAPI API on Render
        response = requests.post(PREDICT_ENDPOINT, files=files, timeout=30)
        response.raise_for_status()  # Raise exception for bad status codes (4xx or 5xx)
        
        data = response.json()
        
        # 4. Extract and display the prediction
        if 'predicted_class' in data:
            predicted_class = data['predicted_class']
            confidence = data.get('confidence', 0.0)
            return f"**Predicted Breed:** {predicted_class}\n\n**Confidence:** {confidence:.2%}"
        
        return f"API Error: {data.get('detail', 'API response missing prediction.')}"

    except requests.exceptions.Timeout:
        return "API Timeout: The prediction took too long. The service might be starting up (cold start). Please try again."
    except requests.exceptions.RequestException as e:
        # Handle connection errors, DNS errors, timeout, etc.
        return f"API Connection Error: Could not reach API or invalid response. Check Render URL and API status. ({e})"


# --- Gradio Interface ---

# Define the interface components
image_input = gr.Image(
    type="pil", 
    label="Upload Pet Image", 
    width=400
)
prediction_output = gr.Textbox(
    label="Prediction Result", 
    lines=3
)

# Build the Gradio interface
iface = gr.Interface(
    fn=predict_pet_breed,
    inputs=image_input,
    outputs=prediction_output,
    title="MLOps Lab 3: Pet Breed Classifier",
    description=f"Upload an image of a dog or cat to classify its breed. Powered by MobileNetV2 + ONNX Runtime (API: {RENDER_API_URL})"
)

# Launch the GUI (necessary for local testing, ignored by HuggingFace Spaces)
if __name__ == "__main__":
    iface.launch()
