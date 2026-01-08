# Known Issues

## Phase 1: PyTorch ARM64 Compatibility Issue

**Status**: 🟡 Deferred for later resolution

**Description**:
The trained routing model loads successfully into the router service, but PyTorch encounters a `matmul primitive` error when running inference on ARM64 (Apple Silicon) architecture.

**Symptoms**:
```
❌ Error in AI routing: could not create a primitive descriptor for a matmul primitive, falling back to round-robin
```

**Current Behavior**:
- Model file loads: `✅ Loaded AI routing model from /app/routing_model.pth`
- Router falls back to round-robin routing
- All requests succeed (100% uptime maintained)
- No impact on system stability

**Root Cause**:
PyTorch CPU backend optimization incompatibility with ARM64 architecture in Docker container.

**Proposed Solutions** (in order of preference):

### Solution 1: Environment Variable Fix (Quick)
Add environment variable to router deployment:

```yaml
# k8s/router-deployment.yaml
env:
  - name: ATEN_CPU_CAPABILITY
    value: "default"
```

Then redeploy:
```bash
kubectl apply -f k8s/router-deployment.yaml
kubectl rollout restart deployment/router
```

### Solution 2: Use CPU-Only PyTorch Build
Update router requirements.txt:
```
torch==2.1.0+cpu
```

### Solution 3: Use x86_64 Container
Modify Dockerfile to use x86_64 base image (slower but more compatible):
```dockerfile
FROM --platform=linux/amd64 python:3.11-slim
```

**Impact**:
- Low - AI routing features unavailable until fixed
- Router operates in fallback mode (round-robin)
- All other functionality working perfectly

**Priority**: Medium (P2)

**Owner**: To be addressed in Phase 1 cleanup or Phase 4

**Logged**: 2026-01-05T22:10:00-05:00
