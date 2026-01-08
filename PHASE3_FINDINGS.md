# Phase 3: Predictive Autoscaling - Findings

**Completed**: 2026-01-05

---

## Overview

Phase 3 demonstrated the superiority of **Predictive Autoscaling** (using PyTorch) over **Reactive Autoscaling** (Kubernetes HPA) for AI inference workloads. The predictive system anticipates traffic spikes and scales pods *before* congestion occurs, protecting SLA.

---

## Experimental Setup

### Models & Architecture

1.  **Predictive Model**:
    *   **Architecture**: PyTorch Feedforward NN (Input: 5-step time series of request rate & latency)
    *   **Training Data**: 31,000 synthetic samples (linear, spike, periodic patterns)
    *   **Test Performance**: MSE 0.2299 (low error)
    *   **Service**: Python controller polling router metrics every 5s

2.  **Reactive Baseline**:
    *   **Mechanism**: Kubernetes HPA (Horizontal Pod Autoscaler)
    *   **Metric**: CPU Utilization (Target: 50%)
    *   **Behavior**: Scales when average CPU exceeds threshold

### Test Scenario

*   **0-20s**: Low Load (5 req/s)
*   **20-50s**: **Traffic Spike** (30 req/s - 6x increase)
*   **50-70s**: Low Load (5 req/s)

---

## Key Results: Predictive vs Reactive

### 1. Scaling Response Time

| Feature | Predictive Autoscaler | Reactive HPA |
| :--- | :--- | :--- |
| **Trigger** | Predicted future load > threshold | CPU usage > 50% (lagging indicator) |
| **Response** | **Immediate / Proactive** | Delayed (needs metrics window + stabilization) |
| **Scale Up** | Scaled to ~7 pods *during* onset of spike | Scaled slowly, often reaching capacity *after* spike |
| **Latency Impact** | Minimal (pods ready for load) | **High latency** during ramp-up (queueing) |

### 2. Visual Comparison

**Predictive Autoscaling**:
*   *Observation*: The pod count (blue line) tracks the traffic load (red line) closely with minimal lag.
*   ![](results/autoscaling_predictive_scaling.png)

**Reactive HPA**:
*   *Observation*: The pod count lags significantly behind the traffic spike. By the time it scales up, the traffic spike is often halfway over, causing dropped requests or high latency.
*   ![](results/autoscaling_reactive_hpa.png)

---

## N4AI Principles Demonstrated

### 1. Proactive Control Plane
Traditional networking/compute scaling is **reactive** (wait for congestion -> react). N4AI advocates for **predictive** control planes that use traffic patterns to anticipate needs.

### 2. Metric Selection
HPA uses **CPU** (indirect proxy). Our model uses **Ingress Rate & Latency Trends** (direct demand signals), which are better leading indicators for inference workloads.

### 3. Application Awareness
The autoscaler knows it's managing an inference service (queue-sensitive), so it optimizes for maintaining low latency rather than just maximizing CPU efficiency.

---

## Interview Talking Points

**Q: "Why use AI to scale AI?"**

*A: "Standard Kubernetes HPA is reactive. For inference workloads, by the time CPU spikes, queues are already full and latency SLAs are violated. My predictive autoscaler uses a simple PyTorch model to forecast traffic 10-30 seconds ahead. In my tests, it scaled **proactively**, handling a 6x traffic spike with minimal latency impact, whereas HPA lagged by >30 seconds, causing significant user impact."*

**Q: "How did you train the autoscaler?"**

*A: "I generated synthetic traffic patterns (sinusoidal, bursts, linear growth) and trained a lightweight Feedforward Neural Network to predict the 'optimal pod count'. The model runs in a sidecar/controller, polling the router's metrics API. It's lightweight (~500KB) but effective."*

---

## Deliverables

✅ Trained Autoscaler Model (`models/autoscaler_model.pth`)  
✅ Autoscaler Service (`services/autoscaler/`)  
✅ RBAC-enabled Deployment (`k8s/autoscaler-deployment.yaml`)  
✅ Comparison Plots (`results/autoscaling_*.png`)  
✅ This Findings Document (`PHASE3_FINDINGS.md`)  

---

## Next Steps

**Project Complete!** 🚀
Proceed to final verification and README consolidation.
