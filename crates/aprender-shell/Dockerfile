# aprender-shell Docker Test Environment
# Reproducible testing for shell completion model
#
# Usage:
#   docker build -t aprender-shell-test .
#   docker run --rm aprender-shell-test make docker-test-all
#
# Toyota Way: Genchi Genbutsu - Go and see for yourself

FROM rust:1.83-slim AS builder

WORKDIR /app

# Install build dependencies and nightly toolchain (required for trueno AVX-512)
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && rustup toolchain install nightly \
    && rustup default nightly

# Copy workspace files (full structure needed for Cargo.toml references)
COPY . .

# Build release binary
RUN cargo build --release --package aprender-shell

# Runtime stage
FROM debian:bookworm-slim AS runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy binary from builder
COPY --from=builder /app/target/release/aprender-shell /usr/local/bin/

# Create test user (non-root)
RUN useradd -m testuser
USER testuser
WORKDIR /home/testuser

# Create sample shell history
RUN mkdir -p /home/testuser && \
    printf '%s\n' \
    "git status" \
    "git add ." \
    "git commit -m 'feat: add feature'" \
    "git push origin main" \
    "cargo build --release" \
    "cargo test --lib" \
    "cargo clippy -- -D warnings" \
    "docker ps -a" \
    "docker run -it ubuntu bash" \
    "kubectl get pods" \
    "kubectl describe pod test" \
    "npm install" \
    "npm run build" \
    "aws s3 ls" \
    "python3 -m pytest" \
    > /home/testuser/.bash_history

# Default command: run tests
CMD ["aprender-shell", "--help"]

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
    CMD aprender-shell --version || exit 1
