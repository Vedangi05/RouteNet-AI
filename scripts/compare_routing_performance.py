"""
Comparative Performance Test: AI vs Round-Robin
Runs identical stress tests against different router configurations to prove AI superiority.
"""

import time
import subprocess
import json
import matplotlib.pyplot as plt
import requests
import sys
import os

ROUTER_URL = "http://localhost:30080"

def set_routing_mode(mode):
    """Update router configuration via kubectl."""
    print(f"\n🔄 Switching router to {mode} mode...")
    cmd = f"kubectl set env deployment/router ROUTING_MODE={mode}"
    subprocess.run(cmd, shell=True, check=True)
    
    print("⏳ Waiting for rollout...")
    subprocess.run("kubectl rollout status deployment/router", shell=True, check=True)
    
    # Wait for pods to be fully ready and serving
    time.sleep(10)
    print("✅ Router updated.")

def run_stress_test(label):
    """Run stress test and return latency metrics."""
    print(f"\n🚀 Running stress test for: {label}")
    
    # We'll use the existing stress_test.py but capture its results
    # Or implement a simple version here to be self-contained and consistent
    
    latencies = []
    
    # Burst parameters
    concurrent_requests = 5  # Reduced for emulation stability
    num_bursts = 3
    burst_interval = 5
    
    import concurrent.futures
    
    def send_request():
        start = time.time()
        try:
            requests.post(f"{ROUTER_URL}/route", json={"request_id": f"comp-{time.time()}"}, timeout=30)
            return (time.time() - start) * 1000
        except Exception as e:
            return None

    for i in range(num_bursts):
        print(f"  🌊 Burst {i+1}/{num_bursts}...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [executor.submit(send_request) for _ in range(concurrent_requests)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
            
        valid_results = [r for r in results if r is not None]
        latencies.extend(valid_results)
        print(f"     Avg latency: {sum(valid_results)/len(valid_results):.1f}ms")
        time.sleep(burst_interval)
        
    return latencies

def plot_comparison(rr_latencies, ai_latencies):
    """Generate comparison plot."""
    plt.figure(figsize=(10, 6))
    
    plt.boxplot([rr_latencies, ai_latencies], labels=['Round-Robin', 'AI-Driven'])
    plt.ylabel('Latency (ms)')
    plt.title('Performance Comparison: Round-Robin vs AI Routing')
    plt.grid(True, alpha=0.3)
    
    # Annotate p99
    import numpy as np
    p99_rr = np.percentile(rr_latencies, 99)
    p99_ai = np.percentile(ai_latencies, 99)
    
    plt.text(1.1, p99_rr, f'p99: {p99_rr:.0f}ms', color='red')
    plt.text(2.1, p99_ai, f'p99: {p99_ai:.0f}ms', color='green')
    
    output_file = "results/comparison_rr_vs_ai.png"
    plt.savefig(output_file, dpi=150)
    print(f"\n📊 Comparison plot saved: {output_file}")

def main():
    print("="*60)
    print("🔍 Final Verification: AI vs Round-Robin")
    print("="*60)
    
    # 1. Test Round-Robin
    set_routing_mode("round-robin")
    rr_latencies = run_stress_test("Round-Robin")
    
    # 2. Test AI-Driven
    set_routing_mode("ai-driven")
    ai_latencies = run_stress_test("AI-Driven")
    
    # 3. Analyze
    plot_comparison(rr_latencies, ai_latencies)
    
    # 4. Report
    import numpy as np
    avg_rr = np.mean(rr_latencies)
    avg_ai = np.mean(ai_latencies)
    p99_rr = np.percentile(rr_latencies, 99)
    p99_ai = np.percentile(ai_latencies, 99)
    
    print("\n🏆 Final Results")
    print(f"Round-Robin: Avg={avg_rr:.0f}ms, p99={p99_rr:.0f}ms")
    print(f"AI-Driven:   Avg={avg_ai:.0f}ms, p99={p99_ai:.0f}ms")
    
    if p99_ai < p99_rr:
        improvement = (p99_rr - p99_ai) / p99_rr * 100
        print(f"\n✅ AI Routing Improved Tail Latency by {improvement:.1f}%!")
    else:
        print("\n⚠️ AI Routing did not improve latency (check load conditions).")

if __name__ == "__main__":
    main()
