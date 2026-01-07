"""
Training Script for Routing Model

Trains a PyTorch MLP to predict the best pod for routing based on metrics.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from models.routing_model import RoutingModel, normalize_features


class RoutingDataset(Dataset):
    """Custom dataset for routing model training."""
    
    def __init__(self, dataframe):
        self.data = dataframe
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Features
        features = {
            'latency_ms': row['latency_ms'],
            'queue_depth': row['queue_depth'],
            'cpu_utilization': row['cpu_utilization'],
            'request_rate': row['request_rate']
        }
        
        # Normalize features
        x = normalize_features(features)
        
        # Label (0 or 1)
        y = torch.tensor([row['is_optimal']], dtype=torch.float32)
        
        return x, y


def train_model(data_file='data/routing_training_data.csv', 
                model_output='models/routing_model.pth',
                epochs=50,
                batch_size=32,
                learning_rate=0.001):
    """
    Train the routing model.
    
    Args:
        data_file: Path to training data CSV
        model_output: Path to save trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
    """
    print("=" * 60)
    print("Training Routing Model")
    print("=" * 60)
    print()
    
    # Load data
    print(f"📊 Loading training data from {data_file}...")
    df = pd.read_csv(data_file)
    print(f"   Total samples: {len(df)}")
    print(f"   Positive samples (optimal pods): {df['is_optimal'].sum()}")
    print(f"   Negative samples: {len(df) - df['is_optimal'].sum()}")
    print()
    
    # Split into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"📦 Split data:")
    print(f"   Training samples: {len(train_df)}")
    print(f"   Validation samples: {len(val_df)}")
    print()
    
    # Create datasets and dataloaders
    train_dataset = RoutingDataset(train_df)
    val_dataset = RoutingDataset(val_df)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = RoutingModel()
    print(f"🧠 Model architecture:")
    print(model)
    print()
    
    # Loss and optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy for binary classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print(f"🚀 Starting training for {epochs} epochs...")
    print()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                # Calculate accuracy
                predicted = (outputs > 0.5).float()
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}")
            print(f"  Val Accuracy: {val_accuracy:.2f}%")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_output)
    
    print()
    print("✅ Training complete!")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Final validation accuracy: {val_accuracies[-1]:.2f}%")
    print(f"   Model saved to: {model_output}")
    print()
    
    # Plot training history
    plot_training_history(train_losses, val_losses, val_accuracies)
    
    return model


def plot_training_history(train_losses, val_losses, val_accuracies):
    """Plot training and validation metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    output_path = 'results/routing_model_training.png'
    Path('results').mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"📊 Training plots saved to: {output_path}")
    plt.close()


def test_model(model_path='models/routing_model.pth'):
    """Test the trained model with sample inputs."""
    print("=" * 60)
    print("Testing Trained Model")
    print("=" * 60)
    print()
    
    # Load model
    model = RoutingModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Test scenarios
    test_scenarios = [
        {
            'description': 'Pod 1: Low load (should score high)',
            'metrics': {'latency_ms': 20, 'queue_depth': 1, 'cpu_utilization': 15, 'request_rate': 5}
        },
        {
            'description': 'Pod 2: Medium load (should score medium)',
            'metrics': {'latency_ms': 60, 'queue_depth': 10, 'cpu_utilization': 45, 'request_rate': 20}
        },
        {
            'description': 'Pod 3: High load (should score low)',
            'metrics': {'latency_ms': 150, 'queue_depth': 35, 'cpu_utilization': 80, 'request_rate': 40}
        }
    ]
    
    print("Testing with sample pod metrics:\n")
    
    for i, scenario in enumerate(test_scenarios, 1):
        metrics = scenario['metrics']
        features = normalize_features(metrics)
        features = features.unsqueeze(0)
        
        with torch.no_grad():
            score = model(features).item()
        
        print(f"Scenario {i}: {scenario['description']}")
        print(f"  Metrics: {metrics}")
        print(f"  Routing Score: {score:.4f}")
        print()
    
    print("=" * 60)


def main():
    """Main training pipeline."""
    # Train model
    model = train_model(
        data_file='data/routing_training_data.csv',
        model_output='models/routing_model.pth',
        epochs=50,
        batch_size=64,
        learning_rate=0.001
    )
    
    # Test the trained model
    test_model('models/routing_model.pth')


if __name__ == "__main__":
    main()
