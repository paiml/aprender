# Docker Deployment

Realizar provides production-ready Docker images for both CPU and GPU deployments.

## Quick Start

### CPU-Only Deployment

```bash
# Build image
docker build -t realizar:latest .

# Run container
docker run -p 3000:3000 realizar:latest

# Test
curl http://localhost:3000/health
```

### GPU-Enabled Deployment

Requires [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) and NVIDIA GPU with CUDA 12.3+.

```bash
# Build GPU image
docker build -f Dockerfile.gpu -t realizar:gpu .

# Run with GPU support
docker run --gpus all -p 3000:3000 realizar:gpu

# Test
curl http://localhost:3000/health
```

## Docker Compose

Use Docker Compose for multi-container deployments:

```bash
# CPU-only deployment
docker-compose up realizar-cpu

# GPU deployment
docker-compose up realizar-gpu

# With monitoring stack (Prometheus + Grafana)
docker-compose --profile monitoring up
```

## Image Details

### CPU Image (`Dockerfile`)

- **Base**: Rust 1.75 (builder), Debian Bookworm Slim (runtime)
- **Size**: ~200MB (optimized multi-stage build)
- **Features**: Server API, batch processing, streaming
- **Architecture**: x86_64, ARM64 (multi-arch support)

### GPU Image (`Dockerfile.gpu`)

- **Base**: NVIDIA CUDA 12.3 (builder + runtime)
- **Size**: ~1.5GB (includes CUDA libraries)
- **Features**: All CPU features + GPU acceleration
- **Requirements**: NVIDIA GPU with Compute Capability 7.0+

## Configuration

### Environment Variables

```bash
# Logging
RUST_LOG=info              # Options: error, warn, info, debug, trace

# GPU selection (GPU image only)
CUDA_VISIBLE_DEVICES=0     # GPU device ID

# Server settings
HOST=0.0.0.0              # Bind address
PORT=3000                  # API port
```

### Volume Mounts

Mount custom models directory:

```bash
docker run -v /path/to/models:/app/models:ro \
  -p 3000:3000 realizar:latest
```

### Custom Command

Override default command:

```bash
# Serve with custom model
docker run -p 3000:3000 realizar:latest \
  realizar serve --model /app/models/llama-7b.gguf --host 0.0.0.0

# Run CLI commands
docker run realizar:latest realizar --help
```

## Production Deployment

### Kubernetes

Deploy to Kubernetes with GPU support:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: realizar
spec:
  replicas: 3
  selector:
    matchLabels:
      app: realizar
  template:
    metadata:
      labels:
        app: realizar
    spec:
      containers:
      - name: realizar
        image: realizar:gpu
        ports:
        - containerPort: 3000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "4"
          requests:
            nvidia.com/gpu: 1
            memory: "4Gi"
            cpu: "2"
        env:
        - name: RUST_LOG
          value: "info"
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        livenessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: realizar
spec:
  selector:
    app: realizar
  ports:
  - port: 80
    targetPort: 3000
  type: LoadBalancer
```

### Helm Chart

Create `values.yaml`:

```yaml
image:
  repository: realizar
  tag: gpu
  pullPolicy: IfNotPresent

replicaCount: 3

resources:
  limits:
    nvidia.com/gpu: 1
    memory: 8Gi
    cpu: 4
  requests:
    nvidia.com/gpu: 1
    memory: 4Gi
    cpu: 2

env:
  - name: RUST_LOG
    value: "info"

service:
  type: LoadBalancer
  port: 80

ingress:
  enabled: true
  annotations:
    kubernetes.io/ingress.class: nginx
  hosts:
    - host: realizar.example.com
      paths:
        - path: /
          pathType: Prefix

monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
    interval: 30s
```

### AWS ECS

Task definition for ECS with GPU:

```json
{
  "family": "realizar",
  "requiresCompatibilities": ["EC2"],
  "networkMode": "bridge",
  "containerDefinitions": [
    {
      "name": "realizar",
      "image": "realizar:gpu",
      "memory": 8192,
      "cpu": 2048,
      "essential": true,
      "portMappings": [
        {
          "containerPort": 3000,
          "hostPort": 3000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "RUST_LOG",
          "value": "info"
        }
      ],
      "resourceRequirements": [
        {
          "type": "GPU",
          "value": "1"
        }
      ],
      "healthCheck": {
        "command": [
          "CMD-SHELL",
          "wget --no-verbose --tries=1 --spider http://localhost:3000/health || exit 1"
        ],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 10
      }
    }
  ]
}
```

## Monitoring

### Prometheus Metrics

Metrics are exposed at `/metrics` endpoint:

```bash
# Scrape metrics
curl http://localhost:3000/metrics

# Output:
# realizar_total_requests 1234
# realizar_successful_requests 1200
# realizar_total_tokens 56789
# ...
```

### Grafana Dashboard

1. Access Grafana: `http://localhost:3030` (default password: `admin`)
2. Add Prometheus data source: `http://prometheus:9090`
3. Import Realizar dashboard (JSON provided in `/grafana` directory)

### Monitoring Stack

Full monitoring setup with Docker Compose:

```bash
# Start all services
docker-compose --profile monitoring up

# Services:
# - Realizar API: http://localhost:3000
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3030
```

## Security

### Non-Root User

Both images run as non-root user `realizar` (UID 1000):

```dockerfile
USER realizar
WORKDIR /app
```

### Read-Only Root Filesystem

Run with read-only root filesystem:

```bash
docker run --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=100m \
  -p 3000:3000 realizar:latest
```

### Network Security

Limit network access:

```bash
docker run --cap-drop=ALL \
  --security-opt=no-new-privileges \
  -p 3000:3000 realizar:latest
```

### Secrets Management

Use Docker secrets for sensitive data:

```bash
# Create secret
echo "secret-token" | docker secret create api_key -

# Use in swarm
docker service create \
  --name realizar \
  --secret api_key \
  --publish 3000:3000 \
  realizar:latest
```

## Performance Tuning

### CPU Optimization

Set CPU affinity for better performance:

```bash
docker run --cpuset-cpus="0-3" \
  -p 3000:3000 realizar:latest
```

### Memory Limits

Prevent OOM with memory limits:

```bash
docker run --memory="4g" \
  --memory-swap="6g" \
  -p 3000:3000 realizar:latest
```

### GPU Memory

Limit GPU memory fraction:

```bash
docker run --gpus '"device=0"' \
  -e CUDA_VISIBLE_DEVICES=0 \
  -p 3000:3000 realizar:gpu
```

## Troubleshooting

### Check Container Logs

```bash
docker logs <container-id>

# Follow logs
docker logs -f <container-id>

# Last 100 lines
docker logs --tail 100 <container-id>
```

### Debug Build Issues

```bash
# Build with no cache
docker build --no-cache -t realizar:latest .

# Build with build args
docker build --build-arg RUST_LOG=debug -t realizar:debug .
```

### GPU Verification

```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi

# Verify GPU access in container
docker run --rm --gpus all realizar:gpu nvidia-smi
```

### Health Check Failures

```bash
# Manual health check
docker exec <container-id> wget -O- http://localhost:3000/health

# Check port binding
docker port <container-id>
```

## Build Optimization

### Multi-Architecture Build

Build for multiple platforms:

```bash
# Setup buildx
docker buildx create --use

# Build for multiple architectures
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t realizar:latest \
  --push .
```

### Layer Caching

Optimize build time with BuildKit:

```bash
export DOCKER_BUILDKIT=1

docker build \
  --cache-from realizar:latest \
  -t realizar:latest .
```

### Dependency Caching

The Dockerfile uses a dummy source pattern to cache dependencies:

```dockerfile
# Cache dependencies (rebuilt only when Cargo.toml changes)
RUN cargo build --release && rm -rf src

# Build actual code (fast incremental builds)
COPY src ./src
RUN cargo build --release
```

## Registry and Distribution

### Push to Registry

```bash
# Tag image
docker tag realizar:latest myregistry.com/realizar:latest

# Push
docker push myregistry.com/realizar:latest

# Pull on deployment
docker pull myregistry.com/realizar:latest
```

### Private Registry

```bash
# Login to private registry
docker login myregistry.com

# Build and push
docker build -t myregistry.com/realizar:1.0.0 .
docker push myregistry.com/realizar:1.0.0
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Docker Build

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build Docker image
        run: docker build -t realizar:${{ github.sha }} .

      - name: Run tests
        run: docker run realizar:${{ github.sha }} cargo test

      - name: Push to registry
        run: |
          docker tag realizar:${{ github.sha }} myregistry.com/realizar:latest
          docker push myregistry.com/realizar:latest
```

## Best Practices

1. **Use multi-stage builds** - Minimize image size by separating build and runtime
2. **Pin base images** - Use specific versions (e.g., `rust:1.75`) not `latest`
3. **Non-root user** - Run as non-privileged user for security
4. **Health checks** - Always define health check endpoints
5. **Resource limits** - Set memory/CPU limits to prevent resource exhaustion
6. **Logging** - Use structured logging and ship to centralized logging
7. **Secrets** - Never bake secrets into images, use runtime injection
8. **Scanning** - Scan images for vulnerabilities before deployment

## References

- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [Kubernetes GPU Support](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
- [Prometheus Monitoring](https://prometheus.io/docs/introduction/overview/)
