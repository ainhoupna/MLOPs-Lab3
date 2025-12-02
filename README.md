---
title: MLOps Lab3 - Transfer Learning with MLflow
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# MLOps Lab3 - Transfer Learning with MLflow Experiment Tracking

This project demonstrates a complete MLOps pipeline with transfer learning, experiment tracking, and model deployment.

## Features

- **Transfer Learning**: Train pet breed classifiers using pre-trained models (ResNet, VGG, EfficientNet)
- **MLflow Experiment Tracking**: Log parameters, metrics, artifacts, and models
- **Model Selection**: Automatically select best model based on validation accuracy
- **ONNX Deployment**: Serialize models to ONNX format for production
- **Image Processing**: Resize, grayscale conversion, and rotation capabilities

## Architecture

This project implements a full MLOps stack:

1. **Training**: PyTorch transfer learning with Oxford-IIIT Pet dataset (37 classes)
2. **Experiment Tracking**: MLflow for logging experiments and model registry
3. **Model Serialization**: ONNX format for efficient inference
4. **Backend API**: FastAPI application with ONNX Runtime inference
5. **Container Registry**: Docker Hub for image storage
6. **Deployment**: Render for hosting the API
7. **Frontend**: Gradio interface on HuggingFace Spaces
8. **CI/CD**: GitHub Actions for automated deployment

## Model Performance

- **Best Model**: ResNet50
- **Validation Accuracy**: 90.90%
- **Dataset**: Oxford-IIIT Pet (3,680 training samples, 37 classes)
- **Training**: Transfer learning with frozen feature extractor

## Technical Stack

- **ML Framework**: PyTorch, torchvision
- **Experiment Tracking**: MLflow
- **Inference**: ONNX Runtime
- **Backend**: FastAPI, Python 3.13
- **Frontend**: Gradio
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Hosting**: Render (API), HuggingFace Spaces (Frontend)

## Note

The first request might take 30-60 seconds due to cold start on the free tier. Subsequent requests will be faster.