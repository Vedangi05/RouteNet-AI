"""
Test script to verify routing functionality.
Sends requests to the router and measures latency.
"""

import time
import requests
import statistics
from typing import List, Dict


ROUTER_URL = "http://localhost:30080"


def test_router_health():
    """Test if router is accessible."""
    print("🔍 Testing router health...")
    try:
        response = requests.get(f"{ROUTER_URL}/health", timeout=5)
        response.raise_for_status()
        print(f"✅ Router is healthy: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Router health check failed: {e}")
        return False


def send_inference_request(request_id: int) -> Dict:
    """Send a single inference request through the router."""
    try:
        start_time = time.time()
        response = requests.post(
            f"{ROUTER_URL}/route",
            json={"request_id": f"test-{request_id}"},
            timeout=30
        )
        response.raise_for_status()
        elapsed_time = (time.time() - start_time) * 1000  # ms
        
        data = response.json()
        return {
            "success": True,
            "request_id": request_id,
            "routed_to_pod": data.get("routed_to_pod"),
            "routing_time_ms": data.get("routing_time_ms"),
            "total_time_ms": data.get("total_time_ms"),
            "client_measured_time_ms": elapsed_time
        }
    except Exception as e:
        return {
            "success": False,
            "request_id": request_id,
            "error": str(e)
        }


def run_load_test(num_requests: int = 50):
    """Run a simple load test."""
    print(f"\n🚀 Running load test with {num_requests} requests...")
    
    results = []
    pod_distribution = {}
    
    for i in range(num_requests):
        result = send_inference_request(i)
        results.append(result)
        
        if result["success"]:
            pod = result["routed_to_pod"]
            pod_distribution[pod] = pod_distribution.get(pod, 0) + 1
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{num_requests} requests completed")
    
    # Calculate statistics
    successful_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]
    
    if not successful_results:
        print("❌ All requests failed!")
        return
    
    latencies = [r["client_measured_time_ms"] for r in successful_results]
    
    print(f"\n📊 Results:")
    print(f"  Total requests: {num_requests}")
    print(f"  Successful: {len(successful_results)}")
    print(f"  Failed: {len(failed_results)}")
    print(f"\n⏱️  Latency Statistics:")
    print(f"  Average: {statistics.mean(latencies):.2f} ms")
    print(f"  Median: {statistics.median(latencies):.2f} ms")
    print(f"  Min: {min(latencies):.2f} ms")
    print(f"  Max: {max(latencies):.2f} ms")
    
    if len(latencies) > 1:
        print(f"  Std Dev: {statistics.stdev(latencies):.2f} ms")
    
    print(f"\n🎯 Pod Distribution:")
    for pod, count in sorted(pod_distribution.items()):
        percentage = (count / len(successful_results)) * 100
        print(f"  {pod}: {count} requests ({percentage:.1f}%)")
    
    if failed_results:
        print(f"\n❌ Failed Requests:")
        for result in failed_results[:5]:  # Show first 5 failures
            print(f"  Request {result['request_id']}: {result['error']}")


def main():
    print("=" * 60)
    print("RouteNet-AI Router Test")
    print("=" * 60)
    
    # Test router health
    if not test_router_health():
        print("\n⚠️  Router is not accessible. Make sure the cluster is running.")
        print("   Run: bash scripts/build-and-deploy.sh")
        return
    
    # Run load test
    run_load_test(num_requests=50)
    
    print("\n✅ Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
