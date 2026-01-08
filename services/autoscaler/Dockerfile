# Use x86_64 base image to avoid PyTorch ARM64/MKL-DNN compatibility issues on Apple Silicon
FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model code and weights
COPY autoscaler_model.py .
COPY autoscaler_model.pth .

# Copy application code
COPY app.py .

# Run the application
CMD ["python", "app.py"]
