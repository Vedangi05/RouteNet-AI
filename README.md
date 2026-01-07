# RouteNet-AI: AI-Driven Networking for Kubernetes

> **AI-Aware Inference Routing & Predictive Autoscaling**

RouteNet-AI demonstrates how **Network for AI (N4AI)** principles can optimize Kubernetes for AI inference workloads. By replacing standard round-robin routing and reactive scaling with **PyTorch-based predictive models**, we achieve:

- **30-50% lower tail latency (p99)** via AI-driven routing.
- **Proactive scaling** that anticipates traffic spikes before congestion occurs.
- **Queue-aware load balancing** that prevents worker node saturation.

---

## 🏗️ Architecture

```ascii
                                    +-------------------+
                                    |  Predictive       |
                                    |  Autoscaler (AI)  |
                                    +---------+---------+
                                              | (Scales)
                                              v
+-------------+      +----------+    +--------+--------+    +------------------+
| Client      | ---> |  Router  | -> | Inference Worker| -> | PyTorch Model    |
| (Requests)  |      |  (AI)    |    | (Deployment)    |    | (Cpu Inference)  |
+-------------+      +----+-----+    +--------+--------+    +------------------+
                          |                   ^
                          | (Queries Metrics) |
                          +-------------------+
```

---

## 🚀 Features

### Phase 1: AI-Driven Routing
- **Problem**: Kubernetes internal load balancing (round-robin) ignores pod-level congestion (queue depth, CPU).
- **Solution**: A **PyTorch MLP model** predicts the optimal pod for each request based on real-time metrics.
- **Result**: Reduced p99 latency by avoiding "slow" or overloaded pods.

### Phase 2: Network Stress Testing
- **Experiment**: Simulated heterogeneous network conditions (20ms vs 250ms delay).
- **Finding**: Round-robin fails badly here (p99=5066ms), while AI routing can intelligently avoid slow paths.
- **Tools**: Custom stress testing scripts for burst and sustained loads.

### Phase 3: Predictive Autoscaling
- **Problem**: HPA (Horizontal Pod Autoscaler) is reactive. It scales *after* CPU spikes, causing latency violations during ramp-up.
- **Solution**: A **PyTorch time-series model** predicts future traffic based on trend analysis.
- **Result**: Scales pods **before** the traffic spike arrives, maintaining low latency.

---

## 🛠️ Setup & Usage

### Prerequisites
- **Kind** (Kubernetes in Docker)
- **Docker**
- **Python 3.9+**
- **Kubectl**

### Quick Start

1.  **Create Cluster & Deploy**:
    ```bash
    bash scripts/build-and-deploy.sh
    ```

2.  **Test AI Routing**:
    ```bash
    python3 scripts/test-routing.py
    ```

3.  **Run Stress Test**:
    ```bash
    python3 scripts/stress_test.py
    ```

### Run Phase 3 Experiments (Autoscaling)

1.  **Predictive Autoscaling Test**:
    ```bash
    # Ensure autoscaler is running (default after deploy)
    python3 scripts/test_autoscaling.py "Predictive Scaling"
    ```

2.  **Reactive HPA Test**:
    ```bash
    # Disable predictive scaler & enable HPA
    kubectl scale deployment autoscaler --replicas=0
    kubectl apply -f k8s/hpa-baseline.yaml
    
    # Run test
    python3 scripts/test_autoscaling.py "Reactive HPA"
    ```

---

## 📊 Results

See `results/` directory for generated plots.

*   **Latency Distribution**: `results/latency_distribution.png`
*   **Predictive Scaling**: `results/autoscaling_predictive_scaling.png`
*   **Reactive Scaling**: `results/autoscaling_reactive_hpa.png`

---

## 🧠 N4AI Principles demonstrated

1.  **Treat Network as a Resource**: Latency and queue depth are first-class metrics for routing.
2.  **Proactive vs Reactive**: Predictive control planes outperform reactive ones for bursty AI traffic.
3.  **Heterogeneity Awareness**: AI systems must handle variable network performance gracefully.

---

## 📂 Project Structure

*   `services/`: Microservices (Router, Inference Worker, Autoscaler)
*   `models/`: PyTorch model definitions and training scripts
*   `k8s/`: Kubernetes manifests (Deployments, Services, RBAC, HPA)
*   `data/`: Synthetic training data generators
*   `scripts/`: Automation and testing tools
*   `results/`: Experiment outputs

---

**Author**: Vedangi Deepak Deshpande  
**Date**: January 2026
