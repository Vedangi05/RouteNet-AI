Below is a **single, all-in-one, production-grade technical prompt** you can **directly paste into ChatGPT inside your IDE** (Cursor / VS Code / Copilot Chat / Continue.dev, etc.).

This is written **as a senior prompt engineer with ~5 years of infra + AI experience**, optimized for:

* novice Kubernetes users
* strong AI background
* N4AI / Ultra Ethernet–relevant outcomes
* realistic scope
* clean abstractions
* step-by-step build without overwhelm

---

# 🧠📡 ALL-IN-ONE TECHNICAL PROMPT

### *AI-Aware Inference Routing + Predictive Autoscaling on Kubernetes (N4AI-Focused)*

---

## ROLE & EXPECTATIONS

You are a **senior AI infrastructure engineer** specializing in:

* Kubernetes
* networking for AI (N4AI)
* inference systems
* control-plane intelligence
* PyTorch-based decision models

You will **design and implement** a **local, reproducible Kubernetes project** that demonstrates how **network-aware AI routing and predictive autoscaling outperform Kubernetes defaults** for AI inference workloads.

The solution must be:

* runnable on **MacOS and Windows**
* GPU-optional (CPU-only by default)
* understandable by a **Kubernetes novice**
* interview-ready for **N4AI / Ultra Ethernet Consortium–style audiences**

Do **NOT** assume deep Kubernetes expertise.
Do **NOT** modify Kubernetes internals.
Treat Kubernetes as a **platform**, not the subject.

---

## PROJECT GOAL (NON-NEGOTIABLE)

Build a **Kubernetes-based AI inference system** where:

1. **Request routing** is decided by a **PyTorch model** instead of round-robin
2. **Autoscaling** is **predictive**, not reactive
3. Decisions are driven by **network-relevant signals**:

   * latency
   * queue depth
   * request rate
   * tail latency
4. The system clearly demonstrates **Networking for AI (N4AI)** principles

---

## HIGH-LEVEL ARCHITECTURE

```
Client / Load Generator
        |
        v
+----------------------+
|  AI Inference Router |  ← FastAPI + PyTorch
|  (Control Plane)     |
+----------------------+
        |
        v
+----------------------+
|  Kubernetes Service  |
+----------------------+
        |
        v
+--------------------------------+
| Inference Pods (Multiple)      |
| PyTorch model + queue + delay  |
+--------------------------------+
```

A **separate AI Autoscaler** observes metrics and scales inference pods **before congestion**.

---

## REQUIRED COMPONENTS (MUST IMPLEMENT)

### 1️⃣ Kubernetes Environment

* Use **Kind** (Kubernetes-in-Docker)
* Single-node cluster
* No cloud dependencies
* Minimal YAML, heavily commented

Must include:

* Deployment
* Service
* ConfigMap
* (Optional) HorizontalPodAutoscaler for baseline comparison

---

### 2️⃣ Inference Worker Service

* Python + FastAPI
* Simple PyTorch model (dummy CNN or linear model)
* Each pod exposes:

  * `/infer`
  * `/metrics`

Simulate **heterogeneous behavior**:

* configurable artificial latency
* request queue
* variable processing time

No real GPU required.

---

### 3️⃣ AI Inference Router (CORE)

This is the **heart of the project**.

Responsibilities:

* Receive all client requests
* Query metrics from inference pods
* Use a **PyTorch model** to decide **which pod to route to**

#### Routing Model (PyTorch)

Input features (minimum):

```
[pod_latency, queue_depth, cpu_utilization, request_rate]
```

Output:

```
routing_score or pod_id
```

Model:

* small MLP
* trained on **synthetic data**
* inference only during routing

The router must:

* outperform round-robin under congestion
* reduce tail latency

---

### 4️⃣ Metrics & Networking Signals

Collect and expose:

* per-pod latency
* queue depth
* request rate
* response time

Use:

* Prometheus-style `/metrics` endpoint
* or lightweight in-memory metrics (acceptable)

Explicitly model:

* queueing delay
* backpressure
* request bursts

---

### 5️⃣ Predictive AI Autoscaler (EXTENSION)

A second PyTorch model that:

* predicts **future load**
* triggers scaling **before latency spikes**

Inputs:

```
[past_request_rate, latency_trend]
```

Outputs:

```
desired_replica_count
```

This component:

* runs as a separate service
* compares itself against Kubernetes HPA behavior
* demonstrates **control-plane intelligence**

---

## PHASED IMPLEMENTATION (STRICT ORDER)

### PHASE 0 — Skeleton (Confidence Builder)

* Kind cluster
* One inference pod
* One router pod
* Hardcoded routing

Goal: *“Kubernetes fear should be gone.”*

---

### PHASE 1 — AI Routing (MOST IMPORTANT)

* Add routing model
* Compare:

  * round-robin vs AI routing
* Measure:

  * average latency
  * tail latency (p95 / p99)

---

### PHASE 2 — Networking Stress

* Inject artificial delays
* Create request bursts
* Observe queue buildup
* Explain **why latency spikes even when compute is idle**

---

### PHASE 3 — Predictive Autoscaling

* Add autoscaler service
* Predict traffic spikes
* Scale inference pods early
* Compare with reactive HPA

---

## VISUALIZATIONS (MANDATORY)

Generate plots for:

1. latency vs time
2. queue depth vs time
3. request distribution per pod
4. scale events vs traffic

Use:

* matplotlib or simple CSV + plots

---

## CONSTRAINTS (IMPORTANT)

* Python only
* No CUDA dependency
* No service mesh
* No CRDs
* No scheduler plugins
* No eBPF
* No cloud APIs

This is **conceptual clarity > infra complexity**.

---

## CODE QUALITY EXPECTATIONS

* Clean folder structure
* Each service independently runnable
* README with:

  * architecture diagram (ASCII acceptable)
  * setup instructions
  * demo steps
  * experiment results
  * N4AI insights

---

## INTERVIEW STORY (YOU MUST ENABLE THIS)

The final system must allow the user to confidently explain:

> “Why Kubernetes’ default routing and scaling are insufficient for AI inference workloads, and how AI-driven control planes can improve tail latency under network congestion.”

---

## OUTPUT FORMAT

Produce:

1. Folder structure
2. Kubernetes YAMLs
3. FastAPI services
4. PyTorch models
5. Synthetic data generator
6. Load test scripts
7. Step-by-step local run instructions

Explain **why** each component exists.

---

## FINAL CHECK

If any part:

* feels too complex for a Kubernetes beginner
* requires deep K8s internals
* hides networking effects

👉 simplify it.

**This project is about understanding, not showing off.**

---

### BEGIN IMPLEMENTATION NOW.
