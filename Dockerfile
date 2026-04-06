# Realizar - Multi-stage Docker build for production deployment
#
# Build: docker build -t realizar:latest .
# Run: docker run -p 3000:3000 realizar:latest

# Stage 1: Build
FROM rust:1.75-bookworm as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy manifests
COPY Cargo.toml Cargo.lock ./

# Create dummy source to cache dependencies
RUN mkdir -p src && \
    echo "fn main() {}" > src/main.rs && \
    echo "pub fn lib() {}" > src/lib.rs

# Build dependencies (cached layer)
RUN cargo build --release --features server && \
    rm -rf src target/release/deps/realizar*

# Copy actual source code
COPY src ./src
COPY tests ./tests
COPY examples ./examples
COPY benches ./benches

# Build application
RUN cargo build --release --features server

# Stage 2: Runtime
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 realizar && \
    mkdir -p /app/models && \
    chown -R realizar:realizar /app

USER realizar
WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/realizar /usr/local/bin/realizar

# Expose API port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000/health || exit 1

# Default command: serve in demo mode
CMD ["realizar", "serve", "--demo", "--host", "0.0.0.0", "--port", "3000"]
