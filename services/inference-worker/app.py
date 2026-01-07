"""
Inference Worker Service
FastAPI-based service that handles inference requests and exposes metrics.
"""

import os
import time
import random
from collections import deque
from datetime import datetime
from typing import Optional, List

import psutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from model import create_model, run_inference


app = FastAPI(title="Inference Worker")

# Configuration from environment variables
POD_NAME = os.getenv("POD_NAME", "unknown-pod")
ARTIFICIAL_DELAY = float(os.getenv("ARTIFICIAL_DELAY", "0.0"))  # seconds
MAX_QUEUE_SIZE = 100

# Model and metrics
model = create_model()
request_queue = deque(maxlen=MAX_QUEUE_SIZE)
latency_history = deque(maxlen=100)  # Keep last 100 latencies
request_count = 0
start_time = time.time()


class InferenceRequest(BaseModel):
    data: Optional[List] = None
    request_id: Optional[str] = None


class InferenceResponse(BaseModel):
    result: list
    pod_name: str
    processing_time: float
    request_id: Optional[str] = None


@app.get("/")
async def root():
    return {
        "service": "inference-worker",
        "pod_name": POD_NAME,
        "status": "running"
    }


@app.get("/health")
async def health():
    """Health check endpoint for Kubernetes probes."""
    return {"status": "healthy", "pod_name": POD_NAME}


@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    """
    Main inference endpoint.
    Simulates processing with artificial delay and returns model output.
    """
    global request_count
    
    start_processing = time.time()
    request_count += 1
    
    # Add to queue (for metrics)
    request_queue.append({
        "request_id": request.request_id,
        "timestamp": start_processing
    })
    
    try:
        # Artificial delay to simulate heterogeneous pod behavior
        if ARTIFICIAL_DELAY > 0:
            time.sleep(ARTIFICIAL_DELAY)
        
        # Add some random variance (0-50ms)
        variance = random.uniform(0, 0.05)
        time.sleep(variance)
        
        # Run actual inference
        result = run_inference(model, None)
        
        # Calculate processing time
        processing_time = time.time() - start_processing
        latency_history.append(processing_time)
        
        # Remove from queue
        if request_queue:
            request_queue.popleft()
        
        return InferenceResponse(
            result=result,
            pod_name=POD_NAME,
            processing_time=processing_time,
            request_id=request.request_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """
    Expose metrics for the router to query.
    
    Returns:
        - pod_name: Identifier for this pod
        - latency: Average latency over recent requests (ms)
        - queue_depth: Current number of requests in queue
        - cpu_utilization: Current CPU usage percentage
        - request_rate: Requests per second
        - request_count: Total requests processed
        - uptime: Seconds since pod started
    """
    # Calculate average latency
    avg_latency = sum(latency_history) / len(latency_history) if latency_history else 0.0
    
    # Calculate request rate (requests per second)
    uptime = time.time() - start_time
    request_rate = request_count / uptime if uptime > 0 else 0.0
    
    # Get CPU utilization
    cpu_percent = psutil.cpu_percent(interval=0.1)
    
    # Clean old queue entries (older than 10 seconds)
    current_time = time.time()
    while request_queue and (current_time - request_queue[0]["timestamp"]) > 10:
        request_queue.popleft()
    
    return {
        "pod_name": POD_NAME,
        "latency_ms": avg_latency * 1000,  # Convert to milliseconds
        "queue_depth": len(request_queue),
        "cpu_utilization": cpu_percent,
        "request_rate": request_rate,
        "request_count": request_count,
        "uptime_seconds": uptime,
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    print(f"🚀 Starting Inference Worker: {POD_NAME}")
    print(f"   Artificial delay: {ARTIFICIAL_DELAY}s")
    uvicorn.run(app, host="0.0.0.0", port=8000)
