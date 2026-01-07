"""
Traffic Pattern Generator for Autoscaler Training
Generates synthetic traffic patterns with various characteristics for training the predictive autoscaler.
"""

import random
import csv
import numpy as np
from pathlib import Path


def generate_linear_growth(duration=60, start_rate=5, growth_rate=0.5):
    """Generate linearly increasing traffic."""
    timestamps = list(range(duration))
    rates = [start_rate + (growth_rate * t) for t in timestamps]
    return timestamps, rates


def generate_sudden_spike(duration=60, base_rate=10, spike_start=30, spike_magnitude=50, spike_duration=10):
    """Generate sudden traffic spike."""
    timestamps = list(range(duration))
    rates = []
    
    for t in timestamps:
        if spike_start <= t < spike_start + spike_duration:
            rates.append(base_rate + spike_magnitude)
        else:
            rates.append(base_rate + random.gauss(0, 2))
    
    return timestamps, rates


def generate_periodic_pattern(duration=120, base_rate=10, amplitude=20, period=30):
    """Generate periodic traffic (sine wave)."""
    timestamps = list(range(duration))
    rates = [base_rate + amplitude * np.sin(2 * np.pi * t / period) for t in timestamps]
    return timestamps, rates


def generate_stepwise_increase(duration=90, levels=[10, 25, 50, 75], step_duration=20):
    """Generate st epwise traffic increases."""
    timestamps = list(range(duration))
    rates = []
    
    for t in timestamps:
        level_index = min(t // step_duration, len(levels) - 1)
        rates.append(levels[level_index] + random.gauss(0, 2))
    
    return timestamps, rates


def calculate_optimal_pods(request_rate, target_rate_per_pod=15, min_pods=1, max_pods=10):
    """
    Calculate optimal pod count for given request rate.
    
    Args:
        request_rate: Current request rate
        target_rate_per_pod: Target requests per pod
        min_pods: Minimum pods
        max_pods: Maximum pods
    
    Returns:
        Optimal pod count
    """
    # Calculate needed pods with some headroom (1.2x buffer)
    needed = (request_rate / target_rate_per_pod) * 1.2
    optimal = max(min_pods, min(max_pods, int(np.ceil(needed))))
    
    return optimal


def simulate_latency(request_rate, pod_count, base_latency=50):
    """
    Simulate latency based on load.
    
    Args:
        request_rate: Current request rate
        pod_count: Number of pods
        base_latency: Base latency in ms
    
    Returns:
        Simulated latency in ms
    """
    # Calculate load per pod
    load_per_pod = request_rate / pod_count if pod_count > 0 else request_rate
    
    # Latency increases with load (quadratic relationship for queue effects)
    latency = base_latency + (load_per_pod ** 1.5) * 5
    
    # Add some random noise
    latency += random.gauss(0, latency * 0.1)
    
    return max(base_latency, latency)


def generate_training_scenario(pattern_type='linear', window_size=5):
    """
    Generate a complete training scenario.
    
    Args:
        pattern_type: Type of traffic pattern
        window_size: Window size for time series
    
    Returns:
        List of training samples
    """
    # Generate traffic pattern
    if pattern_type == 'linear':
        timestamps, rates = generate_linear_growth(duration=60)
    elif pattern_type == 'spike':
        timestamps, rates = generate_sudden_spike(duration=60)
    elif pattern_type == 'periodic':
        timestamps, rates = generate_periodic_pattern(duration=120)
    elif pattern_type == 'stepwise':
        timestamps, rates = generate_stepwise_increase(duration=90)
    else:
        timestamps, rates = generate_linear_growth(duration=60)
    
    # Simulate system behavior
    samples = []
    current_pods = 2  # Start with 2 pods
    
    for i in range(window_size, len(timestamps)):
        # Get window of historical data
        rate_window = rates[i-window_size:i]
        
        # Calculate latencies for window
        latency_window = []
        for j in range(i-window_size, i):
            lat = simulate_latency(rates[j], current_pods)
            latency_window.append(lat)
        
        # Calculate optimal pods for current rate
        optimal_pods = calculate_optimal_pods(rates[i])
        
        # Create sample
        sample = {
            'timestamp': timestamps[i],
            'pattern_type': pattern_type,
            'current_pods': current_pods,
            'optimal_pods': optimal_pods
        }
        
        # Add windowed features
        for k in range(window_size):
            sample[f'rate_t_minus_{window_size-k}'] = rate_window[k]
            sample[f'latency_t_minus_{window_size-k}'] = latency_window[k]
        
        samples.append(sample)
        
        # Update current pods (gradually move toward optimal)
        if current_pods < optimal_pods:
            current_pods = min(current_pods + 1, optimal_pods)
        elif current_pods > optimal_pods:
            current_pods = max(current_pods - 1, optimal_pods)
    
    return samples


def generate_dataset(num_samples_per_pattern=500, window_size=5, output_file='data/autoscaler_training_data.csv'):
    """
    Generate complete dataset for autoscaler training.
    
    Args:
        num_samples_per_pattern: Number of scenarios per pattern type
        window_size: Time series window size
        output_file: Output CSV path
    
    Returns:
        Path to generated file
    """
    print(f"Generating autoscaler training data...")
    print(f"  Window size: {window_size}")
    print(f"  Samples per pattern: {num_samples_per_pattern}")
    print()
    
    patterns = ['linear', 'spike', 'periodic', 'stepwise']
    all_samples = []
    
    for pattern in patterns:
        print(f"  Generating '{pattern}' patterns...")
        
        for _ in range(num_samples_per_pattern):
            scenario_samples = generate_training_scenario(pattern, window_size)
            all_samples.extend(scenario_samples)
    
    print(f"\n✅ Generated {len(all_samples)} training samples")
    
    # Write to CSV
    if all_samples:
        fieldnames = list(all_samples[0].keys())
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_samples)
        
        print(f"📁 Saved to: {output_file}")
    
    return output_file


def main():
    """Generate the autoscaler training dataset."""
    print("=" * 70)
    print("Autoscaler Training Data Generator")
    print("=" * 70)
    print()
    
    output_file = generate_dataset(
        num_samples_per_pattern=100,
        window_size=5
    )
    
    print()
    print("✅ Data generation complete!")
    print(f"   Use this file to train: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
