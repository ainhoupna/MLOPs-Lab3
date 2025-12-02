# Use the official Python 3.11 image as the base image
FROM python:3.11-slim AS builder

# Recommended environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1

WORKDIR /app

# Intall the requiered dependencies of the system 
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv and the dependencies of the project
FROM base AS builder
# Install uv
RUN pip install --no-cache-dir uv
# Copy the dependencies file
COPY pyproject.toml .
# Copy the lock file if exists
COPY uv.lock* .
# Install the dependencies of the project in the system's environment
RUN uv pip install --system --no-cache .

# Copy the source code and prepare the execution environment
# Copy the source code and prepare the execution environment
FROM python:3.11-slim AS runtime
# Copy the installed dependencies
COPY --from=builder /usr/local /usr/local

# Install wget for downloading model files
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# Copy the source code of the API, logic and home.html
COPY api ./api
COPY mylib ./mylib
COPY templates ./templates

# Download model artifacts from GitHub Releases
# Using v1.0 tag. If download fails, build will fail.
RUN wget -q https://github.com/ainhoupna/MLOPs-Lab3/releases/download/v1.0/model.onnx -O ./model.onnx && \
    wget -q https://github.com/ainhoupna/MLOPs-Lab3/releases/download/v1.0/class_labels.json -O ./class_labels.json && \
    ls -lh ./model.onnx ./class_labels.json

# Expose the port associated with the API created with FastAPI
EXPOSE 8000
# Default command: it starts the API with uvicorn
CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000"]