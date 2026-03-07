# AI DJ Project - Deployment Guide

## Prerequisites

- Docker & Docker Compose (for local/containerized deployment)
- Kubernetes cluster (for K8s deployment)
- kubectl CLI
- kustomize (optional, for K8s)

## Quick Start

### Docker Compose (Development)

```bash
cd deploy
docker-compose up -d
```

The server will be available at http://localhost:5000

### Docker (Standalone)

```bash
# Build
docker build -t ai-dj-server:latest -f deploy/Dockerfile .

# Run
docker run -d -p 5000:5000 --name ai-dj-server \
  -v ai-dj-data:/app/data \
  -v ai-dj-exports:/app/exports \
  ai-dj-server:latest
```

### Kubernetes

Using kubectl:
```bash
# Apply all manifests
kubectl apply -f deploy/k8s/

# Or using kustomize
kubectl apply -k deploy/k8s/
```

Using Docker Desktop K8s:
```bash
kubectl apply -f deploy/k8s/00-namespace.yaml
kubectl apply -f deploy/k8s/01-configmap.yaml
kubectl apply -f deploy/k8s/02-secret.yaml
kubectl apply -f deploy/k8s/03-pvc.yaml
kubectl apply -f deploy/k8s/04-deployment.yaml
kubectl apply -f deploy/k8s/05-service.yaml
kubectl apply -f deploy/k8s/06-ingress.yaml
kubectl apply -f deploy/k8s/07-hpa.yaml
```

## Configuration

1. Copy `.env.example` to `.env` and configure environment variables
2. Update `k8s/02-secret.yaml` with actual secrets
3. Update `k8s/06-ingress.yaml` with your domain

## Resource Limits

The Kubernetes deployment includes:
- 2-10 replicas (auto-scaling)
- 2-4GB RAM per pod
- 1-2 CPU cores per pod

Adjust these in `k8s/04-deployment.yaml` based on your needs.

## Health Checks

- Liveness probe: `/health` endpoint (60s initial delay)
- Readiness probe: `/health` endpoint (30s initial delay)

## Persistence

- Data: 10GB PVC (`ai-dj-data-pvc`)
- Exports: 20GB PVC (`ai-dj-exports-pvc`)

Adjust storage size in `k8s/03-pvc.yaml`.
