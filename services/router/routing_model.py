"""
Routing Model Architecture
PyTorch MLP for intelligent pod selection based on metrics.

Input features (4):
  - pod_latency_ms: Average latency of the pod
  - queue_depth: Number of requests in pod's queue
  - cpu_utilization: Current CPU usage percentage
  - request_rate: Requests per second the pod is processing

Output:
  - routing_score: Confidence score for routing to this pod (higher is better)
"""

import torch
import torch.nn as nn


class RoutingModel(nn.Module):
    """
    Multi-Layer Perceptron for AI-driven routing decisions.
    
    Architecture:
      - Input layer: 4 features
      - Hidden layer 1: 16 neurons with ReLU
      - Hidden layer 2: 8 neurons with ReLU
      - Output layer: 1 neuron (routing score)
    """
    
    def __init__(self, input_size=4, hidden_size1=16, hidden_size2=8):
        super(RoutingModel, self).__init__()
        
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Layer 2
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Output layer (routing score)
            nn.Linear(hidden_size2, 1),
            nn.Sigmoid()  # Output between 0 and 1 (confidence score)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Tensor of shape (batch_size, 4) containing pod metrics
        
        Returns:
            Tensor of shape (batch_size, 1) containing routing scores
        """
        return self.network(x)


def create_routing_model():
    """Create and return a new routing model instance."""
    model = RoutingModel()
    model.eval()  # Set to evaluation mode
    return model


def normalize_features(features):
    """
    Normalize input features to [0, 1] range.
    
    Expected ranges:
      - latency_ms: 0-1000ms
      - queue_depth: 0-100
      - cpu_utilization: 0-100%
      - request_rate: 0-100 req/s
    
    Args:
        features: dict with keys [latency_ms, queue_depth, cpu_utilization, request_rate]
    
    Returns:
        Normalized tensor of shape (4,)
    """
    latency_normalized = min(features['latency_ms'] / 1000.0, 1.0)
    queue_normalized = min(features['queue_depth'] / 100.0, 1.0)
    cpu_normalized = features['cpu_utilization'] / 100.0
    rate_normalized = min(features['request_rate'] / 100.0, 1.0)
    
    return torch.tensor([
        latency_normalized,
        queue_normalized,
        cpu_normalized,
        rate_normalized
    ], dtype=torch.float32)


def predict_routing_score(model, pod_metrics):
    """
    Predict routing score for a single pod.
    
    Args:
        model: Trained RoutingModel instance
        pod_metrics: dict with pod metrics
    
    Returns:
        float: Routing score (0-1, higher is better)
    """
    model.eval()
    with torch.no_grad():
        features = normalize_features(pod_metrics)
        features = features.unsqueeze(0)  # Add batch dimension
        score = model(features)
        return score.item()


def select_best_pod(model, all_pod_metrics):
    """
    Select the best pod from a list of pod metrics.
    
    Args:
        model: Trained RoutingModel instance
        all_pod_metrics: list of dicts, each containing pod metrics
    
    Returns:
        int: Index of the best pod
    """
    scores = []
    for metrics in all_pod_metrics:
        score = predict_routing_score(model, metrics)
        scores.append(score)
    
    # Return index of pod with highest score
    return scores.index(max(scores))


if __name__ == "__main__":
    # Test the model
    print("Testing Routing Model...")
    
    model = create_routing_model()
    print(f"Model architecture:\n{model}\n")
    
    # Test with sample metrics
    sample_metrics = {
        'latency_ms': 50.0,
        'queue_depth': 5,
        'cpu_utilization': 30.0,
        'request_rate': 10.0
    }
    
    score = predict_routing_score(model, sample_metrics)
    print(f"Sample prediction for metrics {sample_metrics}")
    print(f"Routing score: {score:.4f}\n")
    
    # Test with multiple pods
    pod_metrics_list = [
        {'latency_ms': 50.0, 'queue_depth': 5, 'cpu_utilization': 30.0, 'request_rate': 10.0},
        {'latency_ms': 100.0, 'queue_depth': 15, 'cpu_utilization': 60.0, 'request_rate': 20.0},
        {'latency_ms': 25.0, 'queue_depth': 2, 'cpu_utilization': 15.0, 'request_rate': 5.0},
    ]
    
    best_idx = select_best_pod(model, pod_metrics_list)
    print(f"Best pod index: {best_idx}")
    print(f"Best pod metrics: {pod_metrics_list[best_idx]}")
