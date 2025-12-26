FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Flask and API dependencies
RUN pip install --no-cache-dir \
    flask==3.0.0 \
    flask-cors==4.0.0 \
    werkzeug==3.0.1 \
    gunicorn==21.2.0

# Install ML dependencies
RUN pip install --no-cache-dir \
    opencv-contrib-python \
    timm \
    transformers \
    scipy \
    accelerate \
    Pillow \
    matplotlib

# Copy application code
COPY app/ ./app/
COPY config.py ./

# Copy model weights
COPY weights/ ./weights/

# Create outputs directory
RUN mkdir -p outputs/api

# Expose Flask port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD curl -f http://localhost:5000/health || exit 1

# Run server with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "300", "app.server:app"]

