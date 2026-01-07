"""
Visualization Script for Phase 2 - Stress Testing Results
Generates plots to demonstrate network congestion effects.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime


def load_stress_test_results(results_dir='results'):
    """Load all stress test result files."""
    results_path = Path(results_dir)
    json_files = sorted(results_path.glob('stress_test_*.json'))
    
    if not json_files:
        print("No stress test results found!")
        return []
    
    results = []
    for file in json_files:
        with open(file, 'r') as f:
            data = json.load(f)
            results.append(data)
    
    return results


def plot_latency_over_time(requests, output_file='results/latency_over_time.png'):
    """Plot latency over time."""
    successful_requests = [r for r in requests if r['success']]
    
    if not successful_requests:
        print("No successful requests to plot!")
        return
    
    # Extract timestamps and latencies
    timestamps = [datetime.fromisoformat(r['timestamp']) for r in successful_requests]
    latencies = [r['latency_ms'] for r in successful_requests]
    
    # Convert to relative time (seconds from start)
    start_time = min(timestamps)
    relative_times = [(t - start_time).total_seconds() for t in timestamps]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.scatter(relative_times, latencies, alpha=0.6, s=20)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Latency (ms)')
    plt.title('Request Latency Over Time - Network Congestion Effects')
    plt.grid(True, alpha=0.3)
    
    # Add moving average
    if len(latencies) > 10:
        window_size = min(10, len(latencies) // 5)
        moving_avg = np.convolve(latencies, np.ones(window_size)/window_size, mode='valid')
        avg_times = relative_times[window_size-1:]
        plt.plot(avg_times, moving_avg, 'r-', linewidth=2, label=f'{window_size}-request moving average')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"✅ Saved latency plot: {output_file}")
    plt.close()


def plot_latency_distribution(requests, output_file='results/latency_distribution.png'):
    """Plot latency distribution including percentiles."""
    successful_requests = [r for r in requests if r['success']]
    
    if not successful_requests:
        return
    
    latencies = [r['latency_ms'] for r in successful_requests]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(latencies, bins=30, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Latency (ms)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Latency Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Add percentile lines
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    
    ax1.axvline(p50, color='g', linestyle='--', linewidth=2, label=f'p50: {p50:.0f}ms')
    ax1.axvline(p95, color='orange', linestyle='--', linewidth=2, label=f'p95: {p95:.0f}ms')
    ax1.axvline(p99, color='r', linestyle='--', linewidth=2, label=f'p99: {p99:.0f}ms')
    ax1.legend()
    
    # Box plot
    ax2.boxplot(latencies, vert=True)
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('Latency Box Plot')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add percentile annotations
    ax2.text(1.15, p50, f'Median: {p50:.0f}ms', va='center')
    ax2.text(1.15, p95, f'p95: {p95:.0f}ms', va='center', color='orange')
    ax2.text(1.15, p99, f'p99: {p99:.0f}ms', va='center', color='red')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"✅ Saved distribution plot: {output_file}")
    plt.close()


def plot_pod_distribution(requests, output_file='results/pod_distribution.png'):
    """Plot request distribution across pods."""
    successful_requests = [r for r in requests if r['success'] and 'routed_to_pod' in r]
    
    if not successful_requests:
        return
    
    # Count requests per pod
    pod_counts = {}
    for r in successful_requests:
        pod = r['routed_to_pod']
        pod_counts[pod] = pod_counts.get(pod, 0) + 1
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    pods = list(pod_counts.keys())
    counts = list(pod_counts.values())
    
    # Color code by pod type (fast/medium/slow)
    colors = []
    for pod in pods:
        if 'fast' in pod:
            colors.append('green')
        elif 'medium' in pod:
            colors.append('orange')
        elif 'slow' in pod:
            colors.append('red')
        else:
            colors.append('blue')
    
    bars = plt.bar(range(len(pods)), counts, color=colors, alpha=0.7, edgecolor='black')
    plt.xlabel('Pod')
    plt.ylabel('Number of Requests')
    plt.title('Request Distribution Across Pods')
    plt.xticks(range(len(pods)), [p.split('-')[-1] for p in pods], rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    total = sum(counts)
    for i, (bar, count) in enumerate(zip(bars, counts)):
        percentage = (count / total) * 100
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{percentage:.1f}%', ha='center', va='bottom')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Fast Pod (20ms)'),
        Patch(facecolor='orange', label='Medium Pod (100ms)'),
        Patch(facecolor='red', label='Slow Pod (250ms)')
    ]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"✅ Saved pod distribution plot: {output_file}")
    plt.close()


def plot_burst_comparison(results_list, output_file='results/burst_comparison.png'):
    """Compare latency across different burst patterns."""
    if not results_list:
        return
    
    plt.figure(figsize=(12, 6))
    
    for idx, result_data in enumerate(results_list):
        requests = result_data.get('requests', [])
        successful = [r for r in requests if r['success']]
        
        if not successful:
            continue
        
        latencies = [r['latency_ms'] for r in successful]
        label = f"Test {idx+1}"
        
        # Plot as time series
        plt.plot(latencies, alpha=0.7, label=label)
    
    plt.xlabel('Request Number')
    plt.ylabel('Latency (ms)')
    plt.title('Latency Variation Across Multiple Stress Tests')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"✅ Saved burst comparison: {output_file}")
    plt.close()


def generate_all_visualizations():
    """Generate all visualizations from stress test results."""
    print("=" * 70)
    print("📊 Generating Phase 2 Visualizations")
    print("=" * 70)
    print()
    
    # Load results
    results_list = load_stress_test_results()
    
    if not results_list:
        print("❌ No results found to visualize!")
        return
    
    print(f"Found {len(results_list)} stress test result file(s)")
    print()
    
    # Use the most recent result
    latest_result = results_list[-1]
    requests = latest_result.get('requests', [])
    
    print(f"Processing {len(requests)} requests from latest test...")
    print()
    
    # Generate plots
    plot_latency_over_time(requests)
    plot_latency_distribution(requests)
    plot_pod_distribution(requests)
    
    if len(results_list) > 1:
        plot_burst_comparison(results_list)
    
    print()
    print("=" * 70)
    print("✅ All visualizations generated in results/ directory")
    print("=" * 70)


if __name__ == "__main__":
    generate_all_visualizations()
