# Batuta Dockerfile
# Multi-stage build for minimal production image
#
# Build: docker build -t batuta:latest .
# Run:   docker run -v $(pwd):/workspace batuta analyze /workspace

# Stage 1: Builder
FROM rust:1.75-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy manifests
COPY Cargo.toml Cargo.lock ./
COPY .cargo .cargo

# Copy source code
COPY src ./src
COPY examples ./examples

# Build release binary with native features
RUN cargo build --release --features native --locked

# Verify binary exists
RUN ls -lh target/release/batuta

# Stage 2: Runtime
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    git \
    python3 \
    python3-pip \
    python3-venv \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Create batuta user for security
RUN useradd -m -u 1000 -s /bin/bash batuta

# Set working directory
WORKDIR /workspace

# Copy binary from builder
COPY --from=builder /build/target/release/batuta /usr/local/bin/batuta

# Verify binary works
RUN batuta --version

# Set ownership
RUN chown -R batuta:batuta /workspace

# Switch to non-root user
USER batuta

# Set environment variables
ENV RUST_BACKTRACE=1
ENV BATUTA_LOG_LEVEL=info

# Default command
CMD ["batuta", "--help"]

# Labels
LABEL maintainer="Pragmatic AI Labs"
LABEL description="Batuta - Sovereign AI orchestration: agents, ML serving, code analysis, and transpilation"
LABEL version="0.7.2"
LABEL org.opencontainers.image.source="https://github.com/paiml/Batuta"
LABEL org.opencontainers.image.documentation="https://github.com/paiml/Batuta#readme"
LABEL org.opencontainers.image.licenses="MIT"

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD batuta --version || exit 1
