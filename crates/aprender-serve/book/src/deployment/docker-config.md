# Docker Configuration

Generate optimized Dockerfiles for realizar deployments.

## DockerConfig

```rust
pub struct DockerConfig {
    /// Base image (default: debian:bookworm-slim)
    pub base_image: String,
    /// Target architecture
    pub target_arch: String,
    /// Enable multi-stage build
    pub multi_stage: bool,
    /// Additional runtime dependencies
    pub runtime_deps: Vec<String>,
    /// Environment variables
    pub env_vars: Vec<(String, String)>,
    /// Exposed ports
    pub expose_ports: Vec<u16>,
    /// Health check configuration
    pub health_check: Option<HealthCheckConfig>,
}
```

## Default Configuration

```rust
let config = DockerConfig::default();
// base_image: "debian:bookworm-slim"
// target_arch: "x86_64-unknown-linux-gnu"
// multi_stage: true
// expose_ports: [3000]
```

## Generated Dockerfile

```rust
let dockerfile = config.generate_dockerfile();
```

Output:

```dockerfile
# Build stage
FROM rust:1.75-bookworm AS builder
WORKDIR /app
COPY . .
RUN cargo build --release --features server

# Runtime stage
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/realizar /usr/local/bin/

ENV RUST_LOG=info
EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

ENTRYPOINT ["/usr/local/bin/realizar"]
CMD ["serve", "--bind", "0.0.0.0:3000"]
```

## ARM64 Docker

For AWS Graviton or Apple Silicon:

```rust
let config = DockerConfig {
    target_arch: "aarch64-unknown-linux-gnu".to_string(),
    ..Default::default()
};

let dockerfile = config.generate_dockerfile();
```

## Multi-Architecture Build

```bash
# Build for both AMD64 and ARM64
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    -t realizar:latest \
    --push .
```

## Resource Limits

Recommended container limits:

| Resource | Minimum | Recommended | Maximum |
|----------|---------|-------------|---------|
| CPU | 0.5 | 2.0 | 4.0 |
| Memory | 256MB | 1GB | 4GB |
| Storage | 100MB | 500MB | 2GB |

## Docker Compose

```yaml
version: '3.8'
services:
  realizar:
    image: realizar:latest
    ports:
      - "3000:3000"
    environment:
      - RUST_LOG=info
      - MODEL_PATH=/models/model.apr
    volumes:
      - ./models:/models:ro
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 1G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 3s
      retries: 3
```
