#!/bin/bash
# Build and deploy RouteNet services to Kind cluster

set -e

CLUSTER_NAME="routenet"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🏗️  Building and deploying RouteNet to Kind cluster..."
echo ""

# Check if cluster exists
if ! kind get clusters | grep -q "^${CLUSTER_NAME}$"; then
    echo "❌ Cluster '${CLUSTER_NAME}' not found. Creating cluster..."
    kind create cluster --config "${PROJECT_ROOT}/cluster/kind-config.yaml" --name "${CLUSTER_NAME}"
else
    echo "✅ Cluster '${CLUSTER_NAME}' already exists"
fi

echo ""
echo "🐳 Building Docker images..."

# Build inference worker
echo "  📦 Building inference-worker..."
docker build -t routenet/inference-worker:latest \
    "${PROJECT_ROOT}/services/inference-worker"

# Build router
# Build router
echo "  📦 Building router..."
docker build -t routenet/router:latest \
    "${PROJECT_ROOT}/services/router"

# Build autoscaler
echo "  📦 Building autoscaler..."
# Copy model files to build context temporarily
cp "${PROJECT_ROOT}/models/autoscaler_model.py" "${PROJECT_ROOT}/services/autoscaler/"
cp "${PROJECT_ROOT}/models/autoscaler_model.pth" "${PROJECT_ROOT}/services/autoscaler/"

docker build -t routenet/autoscaler:latest \
    "${PROJECT_ROOT}/services/autoscaler"

# Cleanup temp files
rm "${PROJECT_ROOT}/services/autoscaler/autoscaler_model.py"
rm "${PROJECT_ROOT}/services/autoscaler/autoscaler_model.pth"

echo ""
echo "📤 Loading images into Kind cluster..."

kind load docker-image routenet/inference-worker:latest --name "${CLUSTER_NAME}"
kind load docker-image routenet/router:latest --name "${CLUSTER_NAME}"
kind load docker-image routenet/autoscaler:latest --name "${CLUSTER_NAME}"

echo ""
echo "☸️  Applying Kubernetes manifests..."

kubectl apply -f "${PROJECT_ROOT}/k8s/autoscaler-deployment.yaml"
kubectl apply -f "${PROJECT_ROOT}/k8s/inference-deployment.yaml"
kubectl apply -f "${PROJECT_ROOT}/k8s/inference-service.yaml"
kubectl apply -f "${PROJECT_ROOT}/k8s/router-deployment.yaml"
kubectl apply -f "${PROJECT_ROOT}/k8s/router-service.yaml"

echo ""
echo "⏳ Waiting for pods to be ready..."

kubectl wait --for=condition=ready pod -l app=inference-worker --timeout=120s || true
kubectl wait --for=condition=ready pod -l app=router --timeout=120s || true

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📊 Pod status:"
kubectl get pods -o wide

echo ""
echo "🌐 Services:"
kubectl get services

echo ""
echo "🎯 Router is accessible at: http://localhost:30080"
echo "   Test with: curl http://localhost:30080/"
echo ""
