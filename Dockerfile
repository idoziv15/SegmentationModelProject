FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Create directories for models and images if they don't exist
RUN mkdir -p models images onnx_models

# Expose the port that FastAPI will run on
EXPOSE 8000

# Set environment variables
ENV MODEL_NAME="deeplabv3_mobilenet_v3_large"
ENV IMAGE_PATH="images/cat2.jpg"
ENV USE_ONNX="False"
ENV PYTHONPATH=/app

# Command to run the FastAPI server
CMD ["uvicorn", "phase_3.api:app", "--host", "0.0.0.0", "--port", "8000"]


