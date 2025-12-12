# MLOps Lab3 - Transfer Learning with MLflow Experiment Tracking

[![CI/CD](https://github.com/ainhoupna/MLOPs-Lab3/actions/workflows/cicd.yml/badge.svg)](https://github.com/ainhoupna/MLOPs-Lab3/actions/workflows/cicd.yml)

This project demonstrates a complete MLOps pipeline with transfer learning, experiment tracking, and model deployment.

## Features

- **Transfer Learning**: Train pet breed classifiers using pre-trained models (MobileNetV2, ResNet, EfficientNet)
- **MLflow Experiment Tracking**: Log parameters, metrics, artifacts, and models
- **Model Selection**: Automatically select best model based on validation accuracy
- **ONNX Deployment**: Serialize models to ONNX format for production
- **Comprehensive CLI**: Image classification and preprocessing utilities
- **Automated CI/CD**: GitHub Actions pipeline for testing and deployment

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

- **Deployed Model**: MobileNetV2 (optimized for 512MB memory limit)
- **Validation Accuracy**: ~90%
- **Dataset**: Oxford-IIIT Pet (37 classes)
- **Training**: Transfer learning with frozen feature extractor
- **Model Size**: ~9MB (ONNX format)

## Live Deployments

- **Frontend (Gradio)**: [HuggingFace Space](https://huggingface.co/spaces/ainhoupna/mlops-lab-3)
- **Backend (FastAPI)**: [Render API](https://mlops-lab3-n3sg.onrender.com)
- **Docker Image**: [Docker Hub](https://hub.docker.com/r/ainhoupna/mlops-lab3)

## Technical Stack

- **ML Framework**: PyTorch, torchvision
- **Experiment Tracking**: MLflow
- **Inference**: ONNX Runtime
- **Backend**: FastAPI, Python 3.11
- **Frontend**: Gradio
- **Containerization**: Docker (multi-stage build)
- **CI/CD**: GitHub Actions
- **Testing**: pytest (50 tests, 80% coverage)
- **Code Quality**: pylint (10/10 score)
- **Hosting**: Render (API), HuggingFace Spaces (Frontend)

## Quick Start

### Run Locally

```bash
# Install dependencies
make install

# Run tests
make test

# Start API server
uvicorn api.api:app --reload

# Use CLI
python -m cli.cli classify predict image.jpg
python -m cli.cli preprocess pipeline image.jpg --output processed.jpg
```

### Docker

```bash
docker build -t mlops-lab3 .
docker run -p 8000:8000 mlops-lab3
```

## CI/CD Pipeline

The project uses GitHub Actions for automated testing and deployment:

1. **Build**: Run tests, linting, and formatting checks
2. **Deploy**: Build and push Docker image to Docker Hub, trigger Render deployment
3. **Deploy-HF**: Deploy Gradio frontend to HuggingFace Spaces

## Note

The first API request might take 30-60 seconds due to cold start on Render's free tier. Subsequent requests will be faster.