"""
Stress Testing Script for Phase 2
Generates burst traffic to test network congestion and queue buildup.
"""

import time
import requests
import threading
import statistics
from typing import List, Dict
from datetime import datetime
import json


ROUTER_URL = "http://localhost:30080"


class StressTestResults:
    """Container for stress test results."""
    
    def __init__(self):
        self.requests = []
        self.start_time = None
        self.end_time = None
    
    def add_result(self, result: Dict):
        """Add a request result."""
        self.requests.append(result)
    
    def get_stats(self) -> Dict:
        """Calculate statistics from results."""
        if not self.requests:
            return {}
        
        successful = [r for r in self.requests if r['success']]
        failed = [r for r in self.requests if not r['success']]
        
        if successful:
            latencies = [r['latency_ms'] for r in successful]
            latencies_sorted = sorted(latencies)
            
            # Calculate percentiles
            p50_idx = int(len(latencies_sorted) * 0.50)
            p95_idx = int(len(latencies_sorted) * 0.95)
            p99_idx = int(len(latencies_sorted) * 0.99)
            
            stats = {
                'total_requests': len(self.requests),
                'successful': len(successful),
                'failed': len(failed),
                'success_rate': len(successful) / len(self.requests) * 100,
                'duration_seconds': (self.end_time - self.start_time) if self.start_time and self.end_time else 0,
                'latency': {
                    'min': min(latencies),
                    'max': max(latencies),
                    'mean': statistics.mean(latencies),
                    'median': statistics.median(latencies),
                    'p50': latencies_sorted[p50_idx],
                    'p95': latencies_sorted[p95_idx],
                    'p99': latencies_sorted[p99_idx],
                    'stdev': statistics.stdev(latencies) if len(latencies) > 1 else 0
                }
            }
        else:
            stats = {
                'total_requests': len(self.requests),
                'successful': 0,
                'failed': len(failed),
                'success_rate': 0,
                'duration_seconds': 0,
                'latency': {}
            }
        
        return stats


def send_request(request_id: int, results: StressTestResults):
    """Send a single request and record results."""
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{ROUTER_URL}/route",
            json={"request_id": f"stress-{request_id}"},
            timeout=30
        )
        response.raise_for_status()
        elapsed = (time.time() - start_time) * 1000  # ms
        
        data = response.json()
        
        results.add_result({
            'success': True,
            'request_id': request_id,
            'latency_ms': elapsed,
            'routed_to_pod': data.get('routed_to_pod'),
            'timestamp': datetime.utcnow().isoformat()
        })
    
    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        results.add_result({
            'success': False,
            'request_id': request_id,
            'latency_ms': elapsed,
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        })


def burst_traffic_test(burst_size: int = 50, num_bursts: int = 3, burst_interval: float = 5.0):
    """
    Generate burst traffic pattern.
    
    Args:
        burst_size: Number of concurrent requests per burst
        num_bursts: Number of bursts to generate
        burst_interval: Seconds between bursts
    """
    print("=" * 70)
    print("🌊 BURST TRAFFIC TEST")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Burst size: {burst_size} concurrent requests")
    print(f"  Number of bursts: {num_bursts}")
    print(f"  Interval between bursts: {burst_interval}s")
    print()
    
    all_results = StressTestResults()
    all_results.start_time = time.time()
    
    for burst_num in range(num_bursts):
        print(f"🚀 Burst {burst_num + 1}/{num_bursts}")
        
        burst_results = StressTestResults()
        burst_results.start_time = time.time()
        
        # Launch concurrent requests
        threads = []
        for i in range(burst_size):
            request_id = burst_num * burst_size + i
            thread = threading.Thread(target=send_request, args=(request_id, burst_results))
            threads.append(thread)
            thread.start()
        
        # Wait for all requests to complete
        for thread in threads:
            thread.join()
        
        burst_results.end_time = time.time()
        
        # Add to overall results
        all_results.requests.extend(burst_results.requests)
        
        # Print burst stats
        stats = burst_results.get_stats()
        print(f"  ✅ Completed: {stats['successful']}/{stats['total_requests']} requests")
        if stats['latency']:
            print(f"  ⏱️  Latency: mean={stats['latency']['mean']:.1f}ms, "
                  f"p95={stats['latency']['p95']:.1f}ms, p99={stats['latency']['p99']:.1f}ms")
        print()
        
        # Wait before next burst (except for last burst)
        if burst_num < num_bursts - 1:
            print(f"⏸️  Waiting {burst_interval}s before next burst...")
            time.sleep(burst_interval)
            print()
    
    all_results.end_time = time.time()
    
    # Print overall stats
    print("=" * 70)
    print("📊 OVERALL RESULTS")
    print("=" * 70)
    
    stats = all_results.get_stats()
    print(f"Total requests: {stats['total_requests']}")
    print(f"Successful: {stats['successful']} ({stats['success_rate']:.1f}%)")
    print(f"Failed: {stats['failed']}")
    print(f"Duration: {stats['duration_seconds']:.2f}s")
    print()
    
    if stats['latency']:
        print("Latency Statistics:")
        print(f"  Min: {stats['latency']['min']:.2f}ms")
        print(f"  Max: {stats['latency']['max']:.2f}ms")
        print(f"  Mean: {stats['latency']['mean']:.2f}ms")
        print(f"  Median: {stats['latency']['median']:.2f}ms")
        print(f"  p95: {stats['latency']['p95']:.2f}ms")
        print(f"  p99: {stats['latency']['p99']:.2f}ms")
        print(f"  Std Dev: {stats['latency']['stdev']:.2f}ms")
    
    print()
    print("=" * 70)
    
    # Save results
    output_file = f"results/stress_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'config': {
                'burst_size': burst_size,
                'num_bursts': num_bursts,
                'burst_interval': burst_interval
            },
            'stats': stats,
            'requests': all_results.requests
        }, f, indent=2)
    
    print(f"📁 Results saved to: {output_file}")
    
    return all_results


def sustained_load_test(duration_seconds: int = 60, rate_per_second: int = 10):
    """
    Generate sustained load.
    
    Args:
        duration_seconds: How long to run the test
        rate_per_second: Target requests per second
    """
    print("=" * 70)
    print("📈 SUSTAINED LOAD TEST")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Duration: {duration_seconds}s")
    print(f"  Target rate: {rate_per_second} req/s")
    print()
    
    results = StressTestResults()
    results.start_time = time.time()
    
    request_interval = 1.0 / rate_per_second
    request_id = 0
    
    end_time = time.time() + duration_seconds
    
    while time.time() < end_time:
        send_request(request_id, results)
        request_id += 1
        
        # Progress indicator
        if request_id % 50 == 0:
            elapsed = time.time() - results.start_time
            print(f"  Progress: {request_id} requests sent ({elapsed:.1f}s elapsed)")
        
        time.sleep(request_interval)
    
    results.end_time = time.time()
    
    # Print stats
    print()
    print("=" * 70)
    print("📊 RESULTS")
    print("=" * 70)
    
    stats = results.get_stats()
    print(f"Total requests: {stats['total_requests']}")
    print(f"Successful: {stats['successful']} ({stats['success_rate']:.1f}%)")
    print(f"Actual rate: {stats['total_requests'] / stats['duration_seconds']:.2f} req/s")
    print()
    
    if stats['latency']:
        print("Latency Statistics:")
        print(f"  Mean: {stats['latency']['mean']:.2f}ms")
        print(f"  p95: {stats['latency']['p95']:.2f}ms")
        print(f"  p99: {stats['latency']['p99']:.2f}ms")
    
    print()
    print("=" * 70)
    
    return results


def main():
    """Run stress tests."""
    print()
    print("🧪 RouteNet-AI Stress Testing")
    print()
    
    # Test 1: Burst traffic
    print("\nTest 1: Burst Traffic Pattern")
    print("-" * 70)
    burst_traffic_test(burst_size=30, num_bursts=3, burst_interval=5.0)
    
    print("\n\n")
    
    # Test 2: Sustained load
    print("Test 2: Sustained Load")
    print("-" * 70)
    sustained_load_test(duration_seconds=30, rate_per_second=5)
    
    print("\n✅ All stress tests complete!")


if __name__ == "__main__":
    main()
