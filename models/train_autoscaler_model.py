"""
Training Script for Autoscaler Model
Trains a PyTorch model to predict optimal pod count based on traffic patterns.
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

from models.autoscaler_model import AutoscalerModel, normalize_features


class AutoscalerDataset(Dataset):
    """Custom dataset for autoscaler training."""
    
    def __init__(self, dataframe, window_size=5):
        self.data = dataframe
        self.window_size = window_size
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Extract features
        request_rates = [row[f'rate_t_minus_{i}'] for i in range(self.window_size, 0, -1)]
        latencies = [row[f'latency_t_minus_{i}'] for i in range(self.window_size, 0, -1)]
        current_pods = row['current_pods']
        
        # Normalize
        x = normalize_features(request_rates, latencies, current_pods)
        
        # Target (optimal pods)
        y = torch.tensor([row['optimal_pods']], dtype=torch.float32)
        
        return x, y


def train_model(data_file='data/autoscaler_training_data.csv',
                model_output='models/autoscaler_model.pth',
                epochs=50,
                batch_size=64,
                learning_rate=0.001,
                window_size=5):
    """Train the autoscaler model."""
    
    print("=" * 70)
    print("Training Autoscaler Model")
    print("=" * 70)
    print()
    
    # Load data
    print(f"📊 Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    print(f"   Total samples: {len(df)}")
    print()
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"📦 Split:")
    print(f"   Training: {len(train_df)}")
    print(f"   Validation: {len(val_df)}")
    print()
    
    # Create datasets
    train_dataset = AutoscalerDataset(train_df, window_size)
    val_dataset = AutoscalerDataset(val_df, window_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = AutoscalerModel(window_size=window_size)
    print(f"🧠 Model:")
    print(model)
    print()
    
    # Loss and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    train_losses = []
    val_losses = []
    
    print(f"🚀 Training for {epochs} epochs...")
    print()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_output)
    
    print()
    print("✅ Training complete!")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Model saved to: {model_output}")
    print()
    
    # Plot training history
    plot_training_history(train_losses, val_losses)
    
    return model


def plot_training_history(train_losses, val_losses):
    """Plot training curves."""
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Autoscaler Model Training')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    output_path = 'results/autoscaler_training.png'
    plt.savefig(output_path, dpi=150)
    print(f"📊 Training plot saved: {output_path}")
    plt.close()


def main():
    """Main training pipeline."""
    model = train_model(
        data_file='data/autoscaler_training_data.csv',
        model_output='models/autoscaler_model.pth',
        epochs=50,
        batch_size=64,
        learning_rate=0.001,
        window_size=5
    )
    
    print("=" * 70)


if __name__ == "__main__":
    main()
