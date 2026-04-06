# Docker Containerization

> **"Package Batuta and all transpilation tools in reproducible containers for consistent development, CI/CD, and deployment."**

## Overview

Batuta provides comprehensive **Docker support** for containerized development, testing, and deployment. Docker ensures:

- **Reproducible environments** across development, CI/CD, and production
- **Isolated toolchains** with all transpilers (Decy, Depyler, Bashrs) pre-installed
- **Zero setup time** for new team members
- **Consistent CI/CD** builds without "works on my machine" issues
- **Multi-stage builds** for minimal production image sizes

## Quick Start

### Running Batuta in Docker

```bash
# Pull the production image (when published)
docker pull paiml/batuta:latest

# Run Batuta CLI
docker run --rm -v $(pwd):/workspace paiml/batuta:latest \
    batuta analyze /workspace/my_project
```

### Building Locally

```bash
# Build production image
make docker

# Build development image (with hot reload)
make docker-dev

# Run tests in container
make docker-test
```

## Docker Images

Batuta provides **three Docker images** for different use cases:

### **1. Production Image** (`batuta:latest`)

Minimal image for running Batuta CLI in production:

- **Base**: `debian:bookworm-slim` (minimal Debian)
- **Size**: ~150-200 MB (multi-stage build)
- **Contents**: Batuta binary only, minimal runtime dependencies
- **User**: Non-root user (`batuta:1000`)
- **Use case**: Production deployments, CI/CD pipelines

```bash
docker build -t batuta:latest .
```

### **2. Development Image** (`batuta:dev`)

Full development environment with hot reload:

- **Base**: `rust:1.75-slim`
- **Size**: ~2-3 GB (includes Rust toolchain, build cache)
- **Contents**: Full Rust toolchain, source code, cargo watch
- **Volumes**: Cargo cache, target directory, source code
- **Use case**: Local development, interactive debugging

```bash
docker build -f Dockerfile.dev -t batuta:dev .
```

### **3. CI Image** (`batuta:ci`)

Optimized for CI/CD pipelines:

- **Base**: Same as production
- **Size**: ~150-200 MB
- **Contents**: Batuta + test dependencies
- **Use case**: Automated testing, quality gates, PR checks

```bash
docker-compose up --abort-on-container-exit ci
```

## Multi-Stage Build

The production Dockerfile uses **multi-stage builds** to minimize image size:

```dockerfile
# ============================================
# Stage 1: Builder
# ============================================
FROM rust:1.75-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy dependency files first (layer caching)
COPY Cargo.toml Cargo.lock ./

# Build dependencies only (cached layer)
RUN mkdir src && \
    echo "fn main() {}" > src/main.rs && \
    cargo build --release --features native --locked && \
    rm -rf src

# Copy source code
COPY src ./src
COPY examples ./examples

# Build Batuta (only rebuilds if source changed)
RUN cargo build --release --features native --locked

# ============================================
# Stage 2: Runtime
# ============================================
FROM debian:bookworm-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash batuta

# Copy binary from builder
COPY --from=builder /build/target/release/batuta /usr/local/bin/batuta

# Set working directory
WORKDIR /workspace

# Switch to non-root user
USER batuta

# Default command
CMD ["batuta", "--help"]
```

**Key optimizations:**

1. **Dependency caching**: Build dependencies in separate layer (rarely changes)
2. **Minimal runtime**: Only copy final binary to runtime stage
3. **Clean APT cache**: Remove package lists after installation
4. **Non-root user**: Security best practice
5. **Locked dependencies**: Use `Cargo.lock` for reproducibility

**Size reduction:**
- Before multi-stage: ~1.5 GB (includes Rust toolchain)
- After multi-stage: ~150 MB (only runtime dependencies)
- **Savings**: ~1.35 GB (90% reduction)

## Docker Compose

Batuta includes `docker-compose.yml` for orchestrating **5 services**:

```yaml
version: '3.8'

services:
  # ==========================================
  # Production CLI
  # ==========================================
  batuta:
    build:
      context: .
      dockerfile: Dockerfile
    image: batuta:latest
    volumes:
      - .:/workspace:rw
      - cargo-cache:/usr/local/cargo/registry
    working_dir: /workspace
    command: batuta --help

  # ==========================================
  # Development (hot reload)
  # ==========================================
  dev:
    build:
      context: .
      dockerfile: Dockerfile.dev
    image: batuta:dev
    volumes:
      - .:/workspace:rw
      - cargo-cache:/usr/local/cargo/registry
      - cargo-git:/usr/local/cargo/git
      - target-cache:/workspace/target
    working_dir: /workspace
    command: cargo watch -x check -x test -x run
    environment:
      - RUST_LOG=batuta=debug

  # ==========================================
  # CI/CD Testing
  # ==========================================
  ci:
    image: batuta:latest
    volumes:
      - .:/workspace:ro  # Read-only for CI
    working_dir: /workspace
    command: >
      bash -c "cargo test --all --features native &&
               cargo clippy --all-targets --all-features -- -D warnings"

  # ==========================================
  # WASM Build
  # ==========================================
  wasm:
    image: batuta:dev
    volumes:
      - .:/workspace:rw
      - cargo-cache:/usr/local/cargo/registry
      - target-cache:/workspace/target
    working_dir: /workspace
    command: cargo build --target wasm32-unknown-unknown --no-default-features --features wasm

  # ==========================================
  # Documentation Server
  # ==========================================
  docs:
    image: nginx:alpine
    volumes:
      - ./target/doc:/usr/share/nginx/html:ro
    ports:
      - "8000:80"
    depends_on:
      - batuta

# ==========================================
# Named Volumes (persistent cache)
# ==========================================
volumes:
  cargo-cache:
    driver: local
  cargo-git:
    driver: local
  target-cache:
    driver: local
```

### Service Descriptions

| Service | Purpose | Command | Ports |
|---------|---------|---------|-------|
| `batuta` | Production CLI | `batuta --help` | None |
| `dev` | Hot reload development | `cargo watch -x check -x test -x run` | None |
| `ci` | CI/CD testing | Run tests + clippy | None |
| `wasm` | WASM build | Build for `wasm32-unknown-unknown` | None |
| `docs` | Documentation server | Serve rustdoc HTML | 8000 |

### Volume Mounts

**Named volumes** for caching (persist across container restarts):

- `cargo-cache`: Cargo registry cache (~500 MB, rarely changes)
- `cargo-git`: Git dependencies cache
- `target-cache`: Build artifacts cache (~1-2 GB, speeds up rebuilds)

**Bind mounts** for live editing:

- `.:/workspace:rw`: Source code (read-write)
- `.:/workspace:ro`: Source code (read-only for CI)

## Usage Patterns

### **1. Local Development**

Start development container with hot reload:

```bash
# Start dev container
docker-compose up dev

# In another terminal, edit source code
vim src/main.rs

# Container automatically recompiles and runs tests
# Output shows in first terminal
```

**Features:**
- Automatic recompilation on file save
- Runs tests on every change
- Persistent cargo cache across restarts
- Full Rust toolchain available

### **2. Running CLI Commands**

Execute Batuta commands in isolated container:

```bash
# Analyze a Python project
docker-compose run --rm batuta \
    batuta analyze /workspace/my_python_project

# Transpile with Depyler
docker-compose run --rm batuta \
    batuta transpile --input /workspace/src --output /workspace/target/rust

# Generate migration report
docker-compose run --rm batuta \
    batuta report --format html --output /workspace/report.html
```

**Note:** Use `/workspace/` prefix for paths (container working directory).

### **3. CI/CD Integration**

Run tests in clean container (CI/CD pipeline):

```bash
# Run full test suite + linting
docker-compose up --abort-on-container-exit ci

# Exit code indicates pass/fail
echo $?  # 0 = success, non-zero = failure
```

**GitHub Actions example:**

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run tests in Docker
        run: docker-compose up --abort-on-container-exit ci

      - name: Check exit code
        run: |
          if [ $? -ne 0 ]; then
            echo "Tests failed!"
            exit 1
          fi
```

**GitLab CI example:**

```yaml
# .gitlab-ci.yml
test:
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker-compose up --abort-on-container-exit ci
```

### **4. Building WASM**

Build WASM in container:

```bash
# Build WASM target
docker-compose run --rm wasm

# Generated files in target/wasm32-unknown-unknown/
ls -lh target/wasm32-unknown-unknown/release/batuta.wasm
```

### **5. Serving Documentation**

Build and serve rustdoc:

```bash
# Build documentation
docker-compose run --rm batuta cargo doc --no-deps

# Start documentation server
docker-compose up docs

# Open browser
open http://localhost:8000/batuta/
```

### **6. One-Off Commands**

Run arbitrary commands in container:

```bash
# Run specific example
docker-compose run --rm batuta \
    cargo run --example full_transpilation

# Check clippy lints
docker-compose run --rm batuta \
    cargo clippy -- -D warnings

# Format code
docker-compose run --rm batuta \
    cargo fmt --all

# Run benchmarks
docker-compose run --rm batuta \
    cargo bench
```

## Build Script

The `scripts/docker-build.sh` script automates Docker builds:

```bash
#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-prod}"

case "$MODE" in
    prod)
        echo "🐳 Building production Docker image..."
        docker build -t batuta:latest \
            --target runtime \
            --build-arg FEATURES=native \
            .
        echo "✅ Built: batuta:latest"
        ;;

    dev)
        echo "🐳 Building development Docker image..."
        docker build -f Dockerfile.dev -t batuta:dev .
        echo "✅ Built: batuta:dev"
        ;;

    ci)
        echo "🐳 Building CI Docker image..."
        docker build -t batuta:ci \
            --target runtime \
            --build-arg FEATURES=native \
            .
        echo "✅ Built: batuta:ci"
        ;;

    wasm)
        echo "🐳 Building WASM Docker image..."
        docker build -t batuta:wasm \
            --target builder \
            --build-arg FEATURES=wasm \
            --build-arg TARGET=wasm32-unknown-unknown \
            .
        echo "✅ Built: batuta:wasm"
        ;;

    *)
        echo "Usage: $0 {prod|dev|ci|wasm}"
        exit 1
        ;;
esac
```

**Usage:**

```bash
# Build production image
./scripts/docker-build.sh prod

# Build development image
./scripts/docker-build.sh dev

# Build CI image
./scripts/docker-build.sh ci

# Build WASM-capable image
./scripts/docker-build.sh wasm
```

## Dockerfile.dev

The development Dockerfile includes additional tools:

```dockerfile
FROM rust:1.75-slim

# Install development dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install cargo-watch for hot reload
RUN cargo install cargo-watch

# Install wasm toolchain
RUN rustup target add wasm32-unknown-unknown

# Install external transpilation tools
RUN cargo install depyler bashrs pmat

WORKDIR /workspace

# Default: watch mode
CMD ["cargo", "watch", "-x", "check", "-x", "test"]
```

**Additional tools:**
- `cargo-watch`: Automatic recompilation on file changes
- `wasm32-unknown-unknown`: WASM build target
- `depyler`, `bashrs`, `pmat`: External transpilers

## .dockerignore

Exclude unnecessary files from Docker build context:

```
# Build artifacts
target/
wasm-dist/
dist/

# Dependency cache
Cargo.lock  # Keep if you want reproducible builds

# Git
.git/
.gitignore

# IDE
.vscode/
.idea/
*.swp
*.swo

# Documentation build
book/book/

# CI/CD
.github/
.gitlab-ci.yml

# Local config
.env
.batuta-state.json

# macOS
.DS_Store

# Logs
*.log
```

**Benefits:**
- Faster Docker builds (smaller context)
- No accidental secrets in images
- Cleaner build logs

## Environment Variables

Configure Batuta via environment variables:

```bash
# Enable debug logging
docker-compose run -e RUST_LOG=batuta=debug batuta \
    batuta analyze /workspace/project

# Set custom config path
docker-compose run -e BATUTA_CONFIG=/workspace/custom.toml batuta \
    batuta transpile --input /workspace/src

# Disable GPU backend
docker-compose run -e BATUTA_DISABLE_GPU=1 batuta \
    batuta optimize --input /workspace/project
```

**Supported variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `RUST_LOG` | Logging level | `info` |
| `BATUTA_CONFIG` | Config file path | `batuta.toml` |
| `BATUTA_DISABLE_GPU` | Disable GPU backend | `0` (enabled) |
| `BATUTA_CACHE_DIR` | Cache directory | `/tmp/batuta-cache` |

## Security Best Practices

### **1. Non-Root User**

All images run as non-root user `batuta:1000`:

```dockerfile
# Create user
RUN useradd -m -u 1000 -s /bin/bash batuta

# Switch user
USER batuta
```

**Benefits:**
- Limits container breakout impact
- Matches host user permissions (if UID=1000)
- Industry security standard

### **2. Read-Only Volumes**

CI containers use read-only mounts:

```yaml
volumes:
  - .:/workspace:ro  # Read-only
```

Prevents CI from modifying source code.

### **3. Minimal Attack Surface**

Production image:
- No Rust toolchain (can't compile malicious code)
- No package managers (can't install backdoors)
- Only essential runtime dependencies

### **4. Trusted Base Images**

Use official images:
- `rust:1.75-slim` (official Rust image)
- `debian:bookworm-slim` (official Debian)
- `nginx:alpine` (official nginx)

Avoid unknown/untrusted bases.

### **5. Dependency Scanning**

Scan for vulnerabilities:

```bash
# Using Trivy
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
    aquasec/trivy image batuta:latest

# Using Snyk
snyk container test batuta:latest
```

## Cleanup

Remove Docker artifacts:

```bash
# Clean all Batuta containers and images
make docker-clean

# Manually remove containers
docker-compose down

# Remove volumes (deletes cache!)
docker-compose down -v

# Remove all images
docker rmi batuta:latest batuta:dev batuta:ci

# Prune unused Docker resources
docker system prune -a --volumes
```

## Performance Tips

### **1. Use BuildKit**

Enable Docker BuildKit for faster builds:

```bash
# Enable BuildKit
export DOCKER_BUILDKIT=1

# Build with BuildKit
docker build -t batuta:latest .
```

**Benefits:**
- Parallel layer building
- Better caching
- Smaller images

### **2. Layer Caching**

Order Dockerfile commands by change frequency:

```dockerfile
# 1. Base image (rarely changes)
FROM rust:1.75-slim

# 2. System dependencies (rarely changes)
RUN apt-get update && apt-get install -y ...

# 3. Cargo dependencies (changes occasionally)
COPY Cargo.toml Cargo.lock ./
RUN cargo build --release

# 4. Source code (changes frequently)
COPY src ./src
RUN cargo build --release
```

### **3. Cargo Cache Volumes**

Use named volumes for cargo cache:

```yaml
volumes:
  - cargo-cache:/usr/local/cargo/registry  # Persistent cache
```

**Speedup:** 5-10x faster dependency builds after first run.

### **4. Parallel Builds**

Build multiple images in parallel:

```bash
# Build prod and dev simultaneously
docker-compose build batuta dev &
wait
```

## Integration with Makefile

The Makefile includes Docker targets:

```makefile
# Build production Docker image
docker:
\t@echo "🐳 Building production Docker image..."
\t./scripts/docker-build.sh prod

# Build development Docker image
docker-dev:
\t@echo "🐳 Building development Docker image..."
\t./scripts/docker-build.sh dev

# Run tests in Docker
docker-test:
\t@echo "🧪 Running tests in Docker..."
\tdocker-compose up --abort-on-container-exit ci

# Clean Docker artifacts
docker-clean:
\t@echo "🧹 Cleaning Docker images and volumes..."
\tdocker-compose down -v
\tdocker rmi batuta:latest batuta:dev batuta:ci 2>/dev/null || true
\t@echo "✅ Docker cleanup complete"
```

**Usage:**

```bash
make docker       # Build production image
make docker-dev   # Build development image
make docker-test  # Run tests in container
make docker-clean # Remove all artifacts
```

## Troubleshooting

### **Issue: Slow builds**

**Cause:** Docker not using layer cache.

**Solution:**
```bash
# Use BuildKit
export DOCKER_BUILDKIT=1
docker build --cache-from batuta:latest -t batuta:latest .
```

### **Issue: Permission denied**

**Cause:** Container user UID doesn't match host user.

**Solution:**
```bash
# Build with custom UID
docker build --build-arg UID=$(id -u) -t batuta:latest .
```

Or:
```bash
# Run as current user
docker-compose run --user $(id -u):$(id -g) batuta batuta --help
```

### **Issue: Out of disk space**

**Cause:** Docker images and volumes consuming disk.

**Solution:**
```bash
# Check disk usage
docker system df

# Clean unused resources
docker system prune -a --volumes

# Remove specific volumes
docker volume rm batuta_cargo-cache batuta_target-cache
```

### **Issue: Cannot connect to Docker daemon**

**Cause:** Docker service not running or permissions issue.

**Solution:**
```bash
# Start Docker service
sudo systemctl start docker

# Add user to docker group (Linux)
sudo usermod -aG docker $USER
newgrp docker
```

## Next Steps

- **[Distribution](./distribution.md)**: Publishing Batuta packages
- **[Release Builds](./release-builds.md)**: Production optimization
- **[Phase 4: Validation](./phase4-validation.md)**: Testing transpiled code

---

**Navigate:** [Table of Contents](../SUMMARY.md)
