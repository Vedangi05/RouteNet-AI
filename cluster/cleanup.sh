#!/bin/bash
# Cleanup script to delete the RouteNet Kind cluster

set -e

echo "🧹 Cleaning up RouteNet Kind cluster..."

# Delete the cluster
kind delete cluster --name routenet

echo "✅ Cluster deleted successfully!"
