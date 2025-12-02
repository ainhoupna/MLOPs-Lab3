"""
ONNX Runtime Inference for Pet Classification.

This module provides a classifier wrapper for loading and using
the ONNX-serialized model for pet breed classification.
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


class ONNXPetClassifier:
    """
    ONNX Runtime classifier for pet breed classification.

    Args:
        model_path: Path to the ONNX model file
        labels_path: Path to the class labels JSON file
    """

    def __init__(self, model_path="model.onnx", labels_path="class_labels.json"):
        """Initialize the ONNX classifier."""
        # Start ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4

        self.session = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )

        # Get session input name
        self.input_name = self.session.get_inputs()[0].name

        # Load class labels
        with open(labels_path, "r", encoding="utf-8") as f:
            class_labels_dict = json.load(f)

        # Convert to list (sorted by index)
        self.class_labels = [
            class_labels_dict[str(i)] for i in range(len(class_labels_dict))
        ]

    def preprocess(self, image):
        """
        Preprocess the input image for the model.

        Args:
            image: PIL Image object

        Returns:
            Preprocessed image as numpy array
        """
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize to 224x224
        image = image.resize((224, 224))

        # Convert to numpy array and normalize
        img_array = np.array(image).astype(np.float32) / 255.0

        # Normalize using ImageNet mean and std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std

        # Transpose from HWC to CHW format
        img_array = np.transpose(img_array, (2, 0, 1))

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def predict(self, image):
        """
        Predict the class label for the input image.

        Args:
            image: PIL Image object

        Returns:
            Predicted class label as string
        """
        # Preprocess the image
        input_data = self.preprocess(image)

        # Create inputs dictionary
        inputs = {self.input_name: input_data}

        # Run inference
        outputs = self.session.run(None, inputs)

        # Get logits (first output)
        logits = outputs[0]

        # Get predicted class index
        predicted_idx = np.argmax(logits, axis=1)[0]

        # Return class label
        return self.class_labels[predicted_idx]


# Create a global classifier instance
# This will be initialized when the module is imported
_classifier = None


def get_classifier():
    """Get or create the global classifier instance."""
    global _classifier
    if _classifier is None:
        model_path = Path(__file__).parent.parent / "model.onnx"
        labels_path = Path(__file__).parent.parent / "class_labels.json"

        # Check if files exist
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                "Please run scripts/select_best_model.py first."
            )
        if not labels_path.exists():
            raise FileNotFoundError(
                f"Class labels file not found: {labels_path}. "
                "Please run scripts/select_best_model.py first."
            )

        _classifier = ONNXPetClassifier(str(model_path), str(labels_path))

    return _classifier
