"""
Autoscaler Model Architecture
Simple feedforward network for predicting future load based on recent traffic patterns.

Input features:
  - Recent request rates (time series window)
  - Recent average latencies (time series window)
  - Current pod count

Output:
  - Desired pod count (1-10 range)
"""

import torch
import torch.nn as nn


class AutoscalerModel(nn.Module):
    """
    Time-series prediction model for autoscaling decisions.
    
    Architecture:
      - Input: Flattened time series (window_size * 2 features + 1)
      - Hidden layers: Dense network with dropout
      - Output: Single value (desired replica count)
    """
    
    def __init__(self, window_size=5, hidden_size=32):
        super(AutoscalerModel, self).__init__()
        
        self.window_size = window_size
        # Input: (request_rate * window_size) + (latency * window_size) + current_pods
        input_size = window_size * 2 + 1
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size // 2, 1),
            nn.ReLU()  # Ensure positive output
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Tensor of shape (batch_size, input_size)
        
        Returns:
            Tensor of shape (batch_size, 1) - predicted desired pods
        """
        return self.network(x)


def normalize_features(request_rates, latencies, current_pods, max_rate=100, max_latency=5000):
    """
    Normalize input features for the autoscaler.
    
    Args:
        request_rates: List of recent request rates
        latencies: List of recent average latencies (ms)
        current_pods: Current number of pods
        max_rate: Maximum expected request rate
        max_latency: Maximum expected latency
    
    Returns:
        Normalized tensor
    """
    # Normalize request rates
    norm_rates = [min(r / max_rate, 1.0) for r in request_rates]
    
    # Normalize latencies
    norm_latencies = [min(l / max_latency, 1.0) for l in latencies]
    
    # Normalize current pods (assume max 10 pods)
    norm_pods = current_pods / 10.0
    
    # Concatenate all features
    features = norm_rates + norm_latencies + [norm_pods]
    
    return torch.tensor(features, dtype=torch.float32)


def predict_desired_pods(model, request_rates, latencies, current_pods):
    """
    Predict desired number of pods.
    
    Args:
        model: Trained AutoscalerModel
        request_rates: List of recent request rates
        latencies: List of recent latencies
        current_pods: Current pod count
    
    Returns:
        int: Desired pod count (clamped to 1-10 range)
    """
    model.eval()
    with torch.no_grad():
        features = normalize_features(request_rates, latencies, current_pods)
        features = features.unsqueeze(0)  # Add batch dimension
        prediction = model(features).item()
        
        # Clamp to valid range
        desired_pods = max(1, min(10, int(round(prediction))))
        
        return desired_pods


def create_autoscaler_model(window_size=5):
    """Create and return a new autoscaler model."""
    model = AutoscalerModel(window_size=window_size)
    model.eval()
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing Autoscaler Model...")
    
    model = create_autoscaler_model(window_size=5)
    print(f"Model architecture:\n{model}\n")
    
    # Test with sample data
    request_rates = [10, 15, 20, 25, 30]
    latencies = [100, 150, 200, 250, 300]
    current_pods = 3
    
    desired = predict_desired_pods(model, request_rates, latencies, current_pods)
    print(f"Sample prediction:")
    print(f"  Request rates: {request_rates}")
    print(f"  Latencies: {latencies}")
    print(f"  Current pods: {current_pods}")
    print(f"  Predicted desired pods: {desired}")