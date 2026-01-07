"""
Predictive Autoscaler Service
Monitors traffic and scales inference pods using a trained PyTorch model.
"""

import os
import time
import requests
import logging
import torch
from collections import deque
from kubernetes import client, config

from autoscaler_model import AutoscalerModel, normalize_features, predict_desired_pods

# Configuration
ROUTER_URL = os.getenv("ROUTER_URL", "http://router-service:8080")
NAMESPACE = os.getenv("NAMESPACE", "default")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "inference-worker")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5"))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "5"))
MIN_PODS = int(os.getenv("MIN_PODS", "1"))
MAX_PODS = int(os.getenv("MAX_PODS", "10"))

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# K8s setup
try:
    config.load_incluster_config()
except:
    try:
        config.load_kube_config()
    except:
        logger.warning("Could not load k8s config. Scaling disabled.")

k8s_apps = client.AppsV1Api()


class Autoscaler:
    def __init__(self):
        self.request_rate_history = deque(maxlen=WINDOW_SIZE)
        self.latency_history = deque(maxlen=WINDOW_SIZE)
        self.model = self.load_model()
        self.current_pods = 0
        
    def load_model(self):
        try:
            model = AutoscalerModel(window_size=WINDOW_SIZE)
            model.load_state_dict(torch.load("autoscaler_model.pth"))
            model.eval()
            logger.info("✅ Loaded autoscaler model")
            return model
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return None

    def get_metrics(self):
        """Fetch aggregated metrics from router."""
        try:
            response = requests.get(f"{ROUTER_URL}/pods/metrics", timeout=2)
            if response.status_code == 200:
                data = response.json()
                pods = data.get("pods", [])
                
                # Calculate average metrics across all pods
                total_latency = 0
                total_rate = 0
                valid_pods = 0
                
                for pod in pods:
                    # Filter out error/dummy metrics
                    if pod.get("latency_ms", 9999) < 5000:
                        total_latency += pod.get("latency_ms", 0)
                        total_rate += pod.get("request_rate", 0)
                        valid_pods += 1
                
                avg_latency = total_latency / valid_pods if valid_pods > 0 else 0
                # Use max rate as representative of load, or sum? 
                # Model trained on per-pod metrics usually implies per-pod load or total load?
                # The generator used `generate_training_scenario` where `rates` seem to be system-wide load?
                # Actually, `generate_training_scenario` uses `calculate_optimal_pods` based on `request_rate`.
                # And `simulate_latency` uses `load_per_pod = request_rate / pod_count`.
                # So the input `request_rate` to the model should be system-wide request rate.
                
                total_system_rate = total_rate  # Sum of rates from all pods?
                # Wait, inference-worker reports its OWN request rate (req/sec processed).
                # So sum of all pods' processing rates ~ system throughput.
                # If there is a queue, system throughput might be capped.
                # Ideally we want incoming request rate. But router doesn't expose that directly yet.
                # Router exposes pod metrics.
                # We can sum up the processing rates as a proxy for load, or check queue depths.
                
                # Let's sum up the rates for total system throughput
                system_rate = total_rate
                
                return system_rate, avg_latency, len(pods)
            else:
                logger.warning(f"Failed to fetch metrics: {response.text}")
                return 0, 0, 0
        except Exception as e:
            logger.error(f"Error fetching metrics: {e}")
            return 0, 0, 0

    def scale_deployment(self, replicas):
        """Scale the Kubernetes deployment."""
        if replicas == self.current_pods:
            return
            
        try:
            # Clamp replicas
            replicas = max(MIN_PODS, min(MAX_PODS, replicas))
            
            logger.info(f"⚖️ Scaling {DEPLOYMENT_NAME} from {self.current_pods} to {replicas}")
            
            body = {"spec": {"replicas": replicas}}
            k8s_apps.patch_namespaced_deployment_scale(
                name=DEPLOYMENT_NAME,
                namespace=NAMESPACE,
                body=body
            )
            self.current_pods = replicas
        except Exception as e:
            logger.error(f"Failed to scale deployment: {e}")

    def run(self):
        logger.info("🚀 Starting predictive autoscaler...")
        
        while True:
            # 1. Get metrics
            request_rate, latency, pod_count = self.get_metrics()
            self.current_pods = pod_count
            
            # 2. Update history
            self.request_rate_history.append(request_rate)
            self.latency_history.append(latency)
            
            # 3. Predict and scale if we have enough history
            if self.model and len(self.request_rate_history) == WINDOW_SIZE:
                # Convert deques to lists
                rates = list(self.request_rate_history)
                latencies = list(self.latency_history)
                
                desired_pods = predict_desired_pods(
                    self.model, 
                    rates, 
                    latencies, 
                    pod_count
                )
                
                logger.info(f"📊 Stats: Rate={request_rate:.1f}/s, Latency={latency:.1f}ms, Pods={pod_count} -> Predicted={desired_pods}")
                
                if desired_pods != pod_count:
                    self.scale_deployment(desired_pods)
            else:
                logger.info(f"⏳ Gathering data... ({len(self.request_rate_history)}/{WINDOW_SIZE})")
            
            time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    autoscaler = Autoscaler()
    autoscaler.run()
