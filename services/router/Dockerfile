# Use x86_64 base image to avoid PyTorch ARM64/MKL-DNN compatibility issues on Apple Silicon
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy routing model code and weights
COPY routing_model.py ./routing_model.py
COPY routing_model.pth ./routing_model.pth

# Copy application code
COPY app.py .

# Expose port
EXPOSE 8080

# Run the application
CMD ["python", "app.py"]
