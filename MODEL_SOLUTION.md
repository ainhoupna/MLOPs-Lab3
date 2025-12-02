# Solution: Model Too Large for GitHub

## The Problem
The ONNX model file is ~90MB, which exceeds GitHub's 100MB limit (or is close to it).

## Solution: Store Model Externally

We'll use **GitHub Releases** or **HuggingFace Hub** to store the model, then download it during Docker build.

### Option 1: Use GitHub Releases (Recommended)

1. **Create a GitHub Release with the model file**
2. **Modify Dockerfile to download model during build**

Let me implement this for you:

```dockerfile
# In Dockerfile, add before COPY commands:
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*
RUN wget https://github.com/ainhoupna/MLOPs-Lab3/releases/download/v1.0/model.onnx -O /app/model.onnx
RUN wget https://github.com/ainhoupna/MLOPs-Lab3/releases/download/v1.0/class_labels.json -O /app/class_labels.json
```

### Option 2: Use a Smaller Model

Train a smaller model (like MobileNet) that will be under 50MB.

### Option 3: Use Model from HuggingFace Hub

Upload model to HuggingFace and download during Docker build.

## What I'll Do Now

I'll modify the Dockerfile to download the model from a URL you'll provide after creating a GitHub Release.

For now, let's:
1. Push the code WITHOUT the model files
2. You manually create a GitHub Release with the model
3. Update Dockerfile to download from that release
4. Redeploy to Render
