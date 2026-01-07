"""
Router Service
Routes inference requests to the best available inference worker pod.

Phase 0: Hardcoded round-robin routing
Phase 1: AI-driven routing using PyTorch model
"""

import os
import random
import time
import asyncio
from typing import List, Dict, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


app = FastAPI(title="AI Router")

# Configuration
INFERENCE_SERVICE_URL = os.getenv(
    "INFERENCE_SERVICE_URL", 
    "http://inference-service:8000"
)
ROUTING_MODE = os.getenv("ROUTING_MODE", "round-robin")  # or "ai-driven"

# State
current_pod_index = 0
pod_endpoints = []  # Will be populated dynamically


class RouteRequest(BaseModel):
    data: Optional[List] = None
    request_id: Optional[str] = None


class RouteResponse(BaseModel):
    result: list
    routed_to_pod: str
    routing_time_ms: float
    total_time_ms: float
    request_id: Optional[str] = None


async def discover_pods() -> List[str]:
    """
    Discover available inference worker pods.
    
    In Kubernetes, we'll use the Service DNS to reach pods.
    For now, we'll try to get individual pod IPs by querying the service.
    """
    # For Phase 0, we'll use the service endpoint directly
    # The Kubernetes service will handle load balancing
    return [INFERENCE_SERVICE_URL]


async def get_pod_metrics(pod_url: str) -> Dict:
    """
    Query metrics from an inference worker pod.
    
    Args:
        pod_url: URL of the pod's metrics endpoint
    
    Returns:
        Dict containing pod metrics
    """
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{pod_url}/metrics")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"Error fetching metrics from {pod_url}: {e}")
        return {
            "pod_name": "unknown",
            "latency_ms": 9999.0,
            "queue_depth": 100,
            "cpu_utilization": 100.0,
            "request_rate": 0.0
        }


def select_pod_round_robin(pods: List[str]) -> str:
    """Simple round-robin pod selection."""
    global current_pod_index
    
    if not pods:
        raise HTTPException(status_code=503, detail="No inference pods available")
    
    selected = pods[current_pod_index]
    current_pod_index = (current_pod_index + 1) % len(pods)
    return selected


def select_pod_random(pods: List[str]) -> str:
    """Random pod selection."""
    if not pods:
        raise HTTPException(status_code=503, detail="No inference pods available")
    
    return random.choice(pods)


async def select_pod_ai_driven(pods: List[str]) -> str:
    """
    AI-driven pod selection based on metrics using trained PyTorch model.
    
    Phase 1: Uses trained routing model to score pods
    """
    try:
        # Import here to avoid failure if torch not available
        import torch
        from routing_model import RoutingModel, normalize_features
        
        # Load model if not already loaded
        if not hasattr(select_pod_ai_driven, 'model'):
            model_path = '/app/routing_model.pth'
            if os.path.exists(model_path):
                select_pod_ai_driven.model = RoutingModel()
                select_pod_ai_driven.model.load_state_dict(torch.load(model_path))
                select_pod_ai_driven.model.eval()
                print(f"✅ Loaded AI routing model from {model_path}")
            else:
                print(f"⚠️ Model file not found at {model_path}, falling back to round-robin")
                return select_pod_round_robin(pods)
        
        #  Get metrics from all pods
        all_metrics = []
        for pod_url in pods:
            metrics = await get_pod_metrics(pod_url)
            all_metrics.append(metrics)
        
        # Score each pod using the model
        scores = []
        for metrics in all_metrics:
            features = normalize_features({
                'latency_ms': metrics.get('latency_ms', 100),
                'queue_depth': metrics.get('queue_depth', 10),
                'cpu_utilization': metrics.get('cpu_utilization', 50),
                'request_rate': metrics.get('request_rate', 10)
            })
           
            with torch.no_grad():
                features = features.unsqueeze(0)
                score = select_pod_ai_driven.model(features).item()
                scores.append(score)
        
        # Select pod with highest score
        best_idx = scores.index(max(scores))
        selected_pod = pods[best_idx]
        
        print(f"🤖 AI routing scores: {[f'{s:.3f}' for s in scores]} → selected pod {best_idx}")
        return selected_pod
    
    except Exception as e:
        print(f"❌ Error in AI routing: {e}, falling back to round-robin")
        return select_pod_round_robin(pods)



@app.get("/")
async def root():
    return {
        "service": "router",
        "status": "running",
        "routing_mode": ROUTING_MODE
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/route", response_model=RouteResponse)
async def route(request: RouteRequest):
    """
    Route an inference request to the best available pod.
    
    Process:
    1. Discover available pods
    2. Select best pod (round-robin or AI-driven)
    3. Forward request to selected pod
    4. Return response with routing metadata
    """
    start_time = time.time()
    
    try:
        # Discover pods
        pods = await discover_pods()
        
        if not pods:
            raise HTTPException(status_code=503, detail="No inference pods available")
        
        # Select pod based on routing mode
        routing_start = time.time()
        
        if ROUTING_MODE == "ai-driven":
            selected_pod = await select_pod_ai_driven(pods)
        elif ROUTING_MODE == "random":
            selected_pod = select_pod_random(pods)
        else:  # default to round-robin
            selected_pod = select_pod_round_robin(pods)
        
        routing_time = (time.time() - routing_start) * 1000  # ms
        
        # Forward request to selected pod
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{selected_pod}/infer",
                json=request.dict()
            )
            response.raise_for_status()
            result = response.json()
        
        total_time = (time.time() - start_time) * 1000  # ms
        
        return RouteResponse(
            result=result["result"],
            routed_to_pod=result["pod_name"],
            routing_time_ms=routing_time,
            total_time_ms=total_time,
            request_id=request.request_id
        )
    
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Error contacting inference pod: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pods/metrics")
async def get_all_metrics():
    """
    Get metrics from all available pods.
    Useful for debugging and monitoring.
    """
    pods = await discover_pods()
    
    metrics = []
    for pod_url in pods:
        pod_metrics = await get_pod_metrics(pod_url)
        metrics.append(pod_metrics)
    
    return {"pods": metrics, "count": len(metrics)}


if __name__ == "__main__":
    print(f"🚀 Starting Router Service")
    print(f"   Routing mode: {ROUTING_MODE}")
    print(f"   Inference service: {INFERENCE_SERVICE_URL}")
    uvicorn.run(app, host="0.0.0.0", port=8080)
