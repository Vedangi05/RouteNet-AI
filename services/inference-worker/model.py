"""
Dummy PyTorch model for simulating inference workload.
This is a simple linear model to ensure CPU-only operation.
"""

import torch
import torch.nn as nn


class DummyInferenceModel(nn.Module):
    """
    Simple linear model for demonstration purposes.
    Input: tensor of shape (batch_size, 784) - simulating flattened 28x28 images
    Output: tensor of shape (batch_size, 10) - simulating 10-class classification
    """
    
    def __init__(self):
        super(DummyInferenceModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def create_model():
    """Create and return the inference model."""
    model = DummyInferenceModel()
    model.eval()  # Set to evaluation mode
    return model


def run_inference(model, input_data=None):
    """
    Run inference on the model.
    
    Args:
        model: PyTorch model
        input_data: Optional input tensor. If None, generates random input.
    
    Returns:
        Model output as a list
    """
    if input_data is None:
        # Generate random input (batch_size=1, features=784)
        input_data = torch.randn(1, 784)
    
    with torch.no_grad():
        output = model(input_data)
    
    return output.tolist()
