"""
Synthetic Data Generator for Routing Model Training

Generates realistic scenarios with varying pod loads to train the routing model.
The goal is to learn which pod is optimal given different load conditions.
"""

import random
import csv
import numpy as np
from pathlib import Path


def generate_pod_metrics(base_load=0.0, variance=0.2):
    """
    Generate realistic pod metrics.
    
    Args:
        base_load: Base load level (0.0 = idle, 1.0 = saturated)
        variance: Random variance to add
    
    Returns:
        dict with pod metrics
    """
    # Add random variance
    load = max(0.0, min(1.0, base_load + random.gauss(0, variance)))
    
    # Metrics scale with load
    latency_ms = 10 + (load * 200) + random.gauss(0, 10)  # 10-210ms
    queue_depth = int(load * 50) + random.randint(0, 10)  # 0-60
    cpu_utilization = (load * 70) + random.gauss(0, 10)  # 0-80%
    request_rate = (load * 50) + random.gauss(0, 5)  # 0-55 req/s
    
    return {
        'latency_ms': max(5, latency_ms),
        'queue_depth': max(0, queue_depth),
        'cpu_utilization': max(0, min(100, cpu_utilization)),
        'request_rate': max(0, request_rate)
    }


def calculate_optimal_pod(pod_metrics_list):
    """
    Calculate which pod is optimal based on a scoring function.
    
    Lower is better for: latency, queue_depth, cpu_utilization
    We compute a composite score where lower total score = better pod
    
    Args:
        pod_metrics_list: list of dicts with pod metrics
    
    Returns:
        int: Index of optimal pod (0-based)
    """
    scores = []
    for metrics in pod_metrics_list:
        # Weighted scoring: prioritize latency and queue depth
        score = (
            metrics['latency_ms'] * 0.4 +          # 40% weight
            metrics['queue_depth'] * 5.0 * 0.3 +   # 30% weight (scaled up)
            metrics['cpu_utilization'] * 0.2 +     # 20% weight
            (100 - metrics['request_rate']) * 0.1  # 10% weight (inverse)
        )
        scores.append(score)
    
    # Return index of pod with lowest score (best pod)
    return scores.index(min(scores))


def generate_training_scenario(num_pods=3):
    """
    Generate a single training scenario.
    
    Simulates various load distributions across pods:
    - Balanced load
    - Unbalanced load (one pod overloaded)
    - Mixed conditions
    
    Returns:
        tuple: (pod_metrics_list, optimal_pod_index)
    """
    scenario_type = random.choice(['balanced', 'unbalanced', 'mixed'])
    
    pod_metrics_list = []
    
    if scenario_type == 'balanced':
        # All pods have similar load
        base_load = random.uniform(0.2, 0.7)
        for _ in range(num_pods):
            pod_metrics_list.append(generate_pod_metrics(base_load, variance=0.1))
    
    elif scenario_type == 'unbalanced':
        # One pod is heavily loaded, others are light
        for i in range(num_pods):
            if i == 0:
                # First pod is overloaded
                pod_metrics_list.append(generate_pod_metrics(0.8, variance=0.1))
            else:
                # Others are lightly loaded
                pod_metrics_list.append(generate_pod_metrics(0.2, variance=0.1))
    
    else:  # mixed
        # Random load distribution
        for _ in range(num_pods):
            base_load = random.uniform(0.1, 0.9)
            pod_metrics_list.append(generate_pod_metrics(base_load, variance=0.2))
    
    optimal_pod = calculate_optimal_pod(pod_metrics_list)
    
    return pod_metrics_list, optimal_pod


def generate_dataset(num_samples=5000, num_pods=3, output_file='data/routing_training_data.csv'):
    """
    Generate a complete dataset for training the routing model.
    
    Args:
        num_samples: Number of training samples to generate
        num_pods: Number of pods per scenario
        output_file: Path to save CSV file
    
    Returns:
        str: Path to generated CSV file
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_samples} training samples with {num_pods} pods each...")
    
    # Generate data
    samples = []
    for i in range(num_samples):
        pod_metrics_list, optimal_pod = generate_training_scenario(num_pods)
        
        # Create a sample for each pod
        for pod_idx, metrics in enumerate(pod_metrics_list):
            # Label: 1 if this is the optimal pod, 0 otherwise
            label = 1 if pod_idx == optimal_pod else 0
            
            sample = {
                'scenario_id': i,
                'pod_id': pod_idx,
                'latency_ms': metrics['latency_ms'],
                'queue_depth': metrics['queue_depth'],
                'cpu_utilization': metrics['cpu_utilization'],
                'request_rate': metrics['request_rate'],
                'is_optimal': label
            }
            samples.append(sample)
        
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{num_samples} scenarios...")
    
    # Write to CSV
    fieldnames = ['scenario_id', 'pod_id', 'latency_ms', 'queue_depth', 
                  'cpu_utilization', 'request_rate', 'is_optimal']
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)
    
    print(f"✅ Dataset generated: {output_file}")
    print(f"   Total samples: {len(samples)} ({num_samples} scenarios x {num_pods} pods)")
    
    # Print statistics
    optimal_count = sum(1 for s in samples if s['is_optimal'] == 1)
    print(f"   Optimal pods: {optimal_count} ({(optimal_count/len(samples)*100):.1f}%)")
    
    return output_file


def main():
    """Generate the training dataset."""
    print("=" * 60)
    print("Routing Model Training Data Generator")
    print("=" * 60)
    print()
    
    # Generate training data
    output_file = generate_dataset(num_samples=5000, num_pods=3)
    
    print()
    print("✅ Data generation complete!")
    print(f"   Use this file to train the routing model: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
