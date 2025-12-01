---
title: MLOps Lab2 - Image Classification
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# MLOps Lab2 - Image Classification and Processing

This project demonstrates a complete MLOps pipeline for image classification and processing.

## Features

- **Image Classification**: Predict image classes using a machine learning model
- **Image Resize**: Resize images to custom dimensions
- **Grayscale Conversion**: Convert images to grayscale
- **Image Rotation**: Rotate images by specified degrees

## Architecture

This project implements a full MLOps stack:

1. **Backend API**: FastAPI application containerized with Docker
2. **Container Registry**: Docker Hub for image storage
3. **Deployment**: Render for hosting the API
4. **Frontend**: Gradio interface on HuggingFace Spaces
5. **CI/CD**: GitHub Actions for automated deployment

## Links

- **Source Code**: [GitHub Repository](https://github.com/ainhoupna/MLOPs-Lab2)
- **Docker Image**: [Docker Hub](https://hub.docker.com/r/ainhoupna/mlops-lab2)

## Technical Stack

- **Backend**: FastAPI, Python 3.13, Pillow
- **Frontend**: Gradio
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Hosting**: Render (API), HuggingFace Spaces (Frontend)

## Note

The first request might take 30-60 seconds due to cold start on the free tier. Subsequent requests will be faster.