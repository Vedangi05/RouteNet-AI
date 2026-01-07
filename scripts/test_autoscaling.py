"""
Comparison Test: Predictive Autoscaler vs Reactive HPA
Generates traffic patterns and monitors scaling response.
"""

import time
import threading
import json
import subprocess
import matplotlib.pyplot as plt
from datetime import datetime
import requests
import queue
from statistics import mean

# Import stress test functions
import sys
import os
sys.path.append(os.path.dirname(__file__))
# Adapt imports if needed or copy functions
# Simple request generator reuse
ROUTER_URL = "http://localhost:30080"

class Monitor(threading.Thread):
    def __init__(self, interval=1.0):
        super().__init__()
        self.interval = interval
        self.running = True
        self.data = []
        self.start_time = None

    def get_replica_count(self):
        try:
            cmd = "kubectl get deployment inference-worker -o jsonpath='{.spec.replicas}'"
            result = subprocess.check_output(cmd, shell=True).decode().strip()
            return int(result)
        except:
            return 0

    def run(self):
        self.start_time = time.time()
        while self.running:
            replicas = self.get_replica_count()
            elapsed = time.time() - self.start_time
            self.data.append({
                'time': elapsed,
                'replicas': replicas,
                'timestamp': datetime.now().isoformat()
            })
            time.sleep(self.interval)

    def stop(self):
        self.running = False


def send_traffic(duration, rate, request_log):
    """Send traffic at fixed rate."""
    start_time = time.time()
    end_time = start_time + duration
    interval = 1.0 / rate
    
    while time.time() < end_time:
        req_start = time.time()
        try:
            requests.post(f"{ROUTER_URL}/route", json={"request_id": "auto-test"}, timeout=2)
            request_log.append((time.time(), True))
        except:
            request_log.append((time.time(), False))
        
        # Sleep to maintain rate
        taken = time.time() - req_start
        sleep_time = max(0, interval - taken)
        time.sleep(sleep_time)


def run_scaling_test(test_name, traffic_pattern):
    """
    Run a scaling test.
    traffic_pattern: list of (duration, rate) tuples
    """
    print(f"🚀 Starting {test_name}...")
    
    # Start monitor
    monitor = Monitor(interval=0.5)
    monitor.start()
    
    request_log = []
    
    # Execute traffic pattern
    total_start = time.time()
    for duration, rate in traffic_pattern:
        print(f"  Generating traffic: {rate} req/s for {duration}s")
        send_traffic(duration, rate, request_log)
    
    # Cool down
    print("  Cooling down (10s)...")
    time.sleep(10)
    
    monitor.stop()
    monitor.join()
    
    print("✅ Test complete. Analyzing...")
    return monitor.data, request_log, total_start


def plot_results(test_name, monitor_data, request_log, start_time):
    """Plot scaling behavior."""
    times = [d['time'] for d in monitor_data]
    replicas = [d['replicas'] for d in monitor_data]
    
    # Process request log to get rate over time
    # Bin requests by second
    max_time = int(times[-1]) + 1 if times else 0
    request_rates = [0] * max_time
    
    for req_time, success in request_log:
        rel_time = int(req_time - start_time)
        if 0 <= rel_time < max_time:
            request_rates[rel_time] += 1
    
    # Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Requests / sec', color=color)
    ax1.plot(range(max_time), request_rates, color=color, alpha=0.6, label='Traffic Load')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Replica Count', color=color)
    ax2.plot(times, replicas, color=color, linewidth=2, label='Pod Count')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 12)
    
    plt.title(f'Autoscaling Response: {test_name}')
    fig.tight_layout()
    
    output_file = f"results/autoscaling_{test_name.lower().replace(' ', '_')}.png"
    plt.savefig(output_file, dpi=150)
    print(f"📊 Saved plot: {output_file}")


def main():
    print("="*60)
    print("Autoscaler Performance Test")
    print("="*60)
    
    # Define traffic pattern: 
    # 20s low load -> 30s spike -> 20s low load
    pattern = [
        (20, 5),   # 5 req/s (should need ~1-2 pods)
        (30, 30),  # 30 req/s (should spike to ~6-7 pods)
        (20, 5)    # Back to 5 req/s
    ]
    
    test_name = sys.argv[1] if len(sys.argv) > 1 else "Predictive Scaling"
    monitor_data, request_log, start_time = run_scaling_test(test_name, pattern)
    plot_results(test_name, monitor_data, request_log, start_time)


if __name__ == "__main__":
    main()
