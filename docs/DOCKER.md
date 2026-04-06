# Docker Guide for Batuta

Comprehensive guide for using Batuta with Docker and Docker Compose.

## Quick Start

```bash
# Build production image
./scripts/docker-build.sh prod

# Run Batuta on current directory
docker run -v $(pwd):/workspace batuta:latest analyze /workspace

# Or use docker-compose for development
docker-compose up dev
```

## Table of Contents

- [Images](#images)
- [Building](#building)
- [Running](#running)
- [Docker Compose Services](#docker-compose-services)
- [Development Workflow](#development-workflow)
- [CI/CD Integration](#cicd-integration)
- [Troubleshooting](#troubleshooting)

## Images

Batuta provides multiple Docker images for different use cases:

### Production Image (`batuta:latest`)

**Dockerfile:** `Dockerfile`
**Size:** ~150-200 MB (multi-stage build)
**Use case:** Running Batuta CLI in production

**Features:**
- Minimal Debian base image
- Only runtime dependencies
- Non-root user for security
- Health check included
- Optimized for size and security

### Development Image (`batuta:dev`)

**Dockerfile:** `Dockerfile.dev`
**Size:** ~2-3 GB (includes dev tools)
**Use case:** Active development with hot reload

**Features:**
- Full Rust toolchain
- cargo-watch for automatic rebuilds
- Development tools (vim, curl, etc.)
- Python and C/C++ for testing transpilation
- Persistent volumes for fast rebuilds

## Building

### Using Build Script (Recommended)

```bash
# Build production image
./scripts/docker-build.sh prod

# Build development image
./scripts/docker-build.sh dev

# Build all images
./scripts/docker-build.sh all
```

### Manual Build

```bash
# Production
docker build -t batuta:latest .

# Development
docker build -f Dockerfile.dev -t batuta:dev .

# With version tag
VERSION=$(grep '^version' Cargo.toml | head -1 | cut -d'"' -f2)
docker build -t batuta:$VERSION .
```

### Build Options

```bash
# No cache (clean build)
docker build --no-cache -t batuta:latest .

# Build for specific platform
docker build --platform linux/amd64 -t batuta:latest .

# Build with BuildKit
DOCKER_BUILDKIT=1 docker build -t batuta:latest .
```

## Running

### Basic Usage

```bash
# Analyze current directory
docker run -v $(pwd):/workspace batuta:latest analyze /workspace

# Check version
docker run --rm batuta:latest batuta --version

# Show help
docker run --rm batuta:latest batuta --help
```

### Common Commands

```bash
# Analyze with language detection
docker run -v $(pwd):/workspace batuta:latest \
  analyze --languages /workspace

# Transpile Python project
docker run -v $(pwd):/workspace batuta:latest \
  transpile --input /workspace/python_project --output /workspace/rust_project

# Generate report
docker run -v $(pwd):/workspace batuta:latest \
  report --format html --output /workspace/report.html

# Run PARF analysis
docker run -v $(pwd):/workspace batuta:latest \
  parf --patterns --dead-code /workspace/src
```

### Interactive Shell

```bash
# Start interactive bash session
docker run -it -v $(pwd):/workspace batuta:latest /bin/bash

# As root (for debugging)
docker run -it --user root -v $(pwd):/workspace batuta:latest /bin/bash
```

### Environment Variables

```bash
# Set log level
docker run -e BATUTA_LOG_LEVEL=debug -v $(pwd):/workspace batuta:latest analyze /workspace

# Enable Rust backtrace
docker run -e RUST_BACKTRACE=full -v $(pwd):/workspace batuta:latest analyze /workspace
```

## Docker Compose Services

### Services Overview

```yaml
services:
  batuta    # Production CLI
  dev       # Development with hot reload
  ci        # CI/CD testing
  wasm      # WASM build
  docs      # Documentation server
```

### Using Services

#### Development Service

```bash
# Start development environment
docker-compose up dev

# Run in background
docker-compose up -d dev

# View logs
docker-compose logs -f dev

# Stop service
docker-compose down dev
```

#### CI Service

```bash
# Run all tests and checks
docker-compose up ci

# Run with output
docker-compose up --abort-on-container-exit ci
```

#### WASM Build Service

```bash
# Build WASM artifacts
docker-compose up wasm

# Output will be in wasm-dist/
```

#### Documentation Server

```bash
# Start docs server on http://localhost:8000
docker-compose up docs

# Access at http://localhost:8000
# WASM demo at http://localhost:8000/wasm
```

### Service Configuration

#### Custom Commands

```bash
# Override command
docker-compose run batuta batuta analyze /workspace

# Run custom cargo command
docker-compose run dev cargo test backend

# Run specific example
docker-compose run batuta cargo run --example pipeline_demo
```

#### Volume Mounts

```bash
# Mount different directory
docker-compose run -v /path/to/project:/workspace batuta analyze /workspace

# Read-only mount
docker-compose run -v $(pwd):/workspace:ro batuta analyze /workspace
```

## Development Workflow

### Local Development Setup

```bash
# 1. Start development container
docker-compose up -d dev

# 2. Attach to container
docker exec -it batuta-dev /bin/bash

# 3. Make changes in your editor (VSCode, vim, etc.)
# cargo-watch will automatically rebuild

# 4. View logs
docker-compose logs -f dev

# 5. Run tests manually
docker-compose exec dev cargo test

# 6. Stop when done
docker-compose down
```

### Hot Reload Development

The development service uses `cargo-watch` for automatic rebuilds:

```bash
# Start with default watch command
docker-compose up dev

# Custom watch command
docker-compose run dev cargo watch -x 'test --lib'

# Watch specific file patterns
docker-compose run dev cargo watch -w src -x check
```

### Persistent Caches

Docker Compose uses named volumes for faster rebuilds:

- `cargo-cache`: Cargo registry cache
- `cargo-git`: Git dependencies
- `target-cache`: Compiled artifacts

```bash
# List volumes
docker volume ls | grep batuta

# Remove caches (clean rebuild)
docker-compose down -v

# Inspect volume
docker volume inspect batuta_cargo-cache
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Docker Build

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run tests in Docker
        run: docker-compose up --abort-on-container-exit ci

      - name: Build production image
        run: ./scripts/docker-build.sh prod
```

### GitLab CI Example

```yaml
docker-build:
  image: docker:latest
  services:
    - docker:dind
  script:
    - ./scripts/docker-build.sh all
    - docker-compose up --abort-on-container-exit ci
```

### Local CI Testing

```bash
# Run the same checks as CI
docker-compose up ci

# Or manually
docker run -v $(pwd):/workspace batuta:latest bash -c "
  cargo test --all --features native &&
  cargo clippy --all-targets --all-features -- -D warnings &&
  cargo fmt -- --check
"
```

## Troubleshooting

### Build Issues

**Problem:** Build fails with "no space left on device"

```bash
# Clean up Docker
docker system prune -a

# Remove build cache
docker builder prune
```

**Problem:** Slow builds

```bash
# Use BuildKit for faster builds
export DOCKER_BUILDKIT=1
docker build -t batuta:latest .

# Or enable BuildKit in docker-compose
export COMPOSE_DOCKER_CLI_BUILD=1
docker-compose build
```

### Runtime Issues

**Problem:** Permission denied errors

```bash
# Check volume permissions
ls -la $(pwd)

# Run as current user
docker run --user $(id -u):$(id -g) -v $(pwd):/workspace batuta:latest analyze /workspace

# Or fix permissions in container
docker run --user root -v $(pwd):/workspace batuta:latest chown -R batuta:batuta /workspace
```

**Problem:** Container exits immediately

```bash
# Check logs
docker logs batuta-prod

# Run with interactive shell
docker run -it batuta:latest /bin/bash

# Check health
docker inspect batuta-prod | grep Health
```

### Network Issues

**Problem:** Cannot access documentation server

```bash
# Check port binding
docker ps | grep batuta-docs

# Check if port is in use
lsof -i :8000

# Use different port
docker run -p 9000:80 -v ./book/book:/usr/share/nginx/html:ro nginx:alpine
```

### Volume Issues

**Problem:** Changes not reflected in container

```bash
# Ensure volume is mounted correctly
docker inspect batuta-dev | grep Mounts -A 10

# Restart container
docker-compose restart dev

# Force recreation
docker-compose up --force-recreate dev
```

## Advanced Usage

### Multi-Platform Builds

```bash
# Setup buildx
docker buildx create --use

# Build for multiple platforms
docker buildx build --platform linux/amd64,linux/arm64 -t batuta:latest .
```

### Custom Base Image

```dockerfile
# Use Alpine instead of Debian
FROM rust:1.75-alpine as builder
# ... rest of Dockerfile
```

### Registry Push

```bash
# Tag for registry
docker tag batuta:latest myregistry.com/batuta:latest

# Push to registry
docker push myregistry.com/batuta:latest

# Pull from registry
docker pull myregistry.com/batuta:latest
```

### Resource Limits

```bash
# Limit memory
docker run -m 2g -v $(pwd):/workspace batuta:latest analyze /workspace

# Limit CPU
docker run --cpus="1.5" -v $(pwd):/workspace batuta:latest analyze /workspace

# In docker-compose.yml
services:
  batuta:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

## Best Practices

1. **Use multi-stage builds** for smaller production images
2. **Leverage build cache** with proper layer ordering
3. **Use named volumes** for persistent data
4. **Run as non-root user** for security
5. **Include health checks** for monitoring
6. **Document environment variables** in docker-compose.yml
7. **Use .dockerignore** to exclude unnecessary files
8. **Version your images** with semantic tags
9. **Test locally** before pushing to registry
10. **Monitor resource usage** with `docker stats`

## Security Considerations

- Images run as non-root user (`batuta`)
- Minimal attack surface (slim base images)
- No unnecessary packages
- Health checks for monitoring
- Read-only file systems where possible
- Secrets via environment variables (not baked in)

## Performance Tips

- Use BuildKit for parallel builds
- Cache cargo dependencies separately
- Use named volumes for target directory
- Limit unnecessary file copies
- Consider distroless images for even smaller size
- Profile with `docker stats` and adjust resources

## Resources

- [Dockerfile Reference](https://docs.docker.com/engine/reference/builder/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
- [Rust Docker Best Practices](https://docs.docker.com/language/rust/)
- [Batuta Repository](https://github.com/paiml/Batuta)

## Support

For issues and questions:
- GitHub Issues: https://github.com/paiml/Batuta/issues
- Documentation: https://github.com/paiml/Batuta#readme
