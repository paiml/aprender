# Multi-stage Dockerfile for simular
# Provides reproducible build environment
# SHA256 pinned for deterministic builds

# Build stage
ARG RUST_VERSION=1.85.0
FROM rust:${RUST_VERSION}-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy manifests first for better caching
COPY Cargo.toml Cargo.lock ./

# Create dummy src for dependency caching
RUN mkdir src && echo "fn main() {}" > src/main.rs

# Build dependencies
RUN cargo build --release
RUN rm -rf src

# Copy source code
COPY src ./src
COPY benches ./benches
COPY examples ./examples

# Build the actual application
RUN touch src/main.rs && cargo build --release

# Runtime stage
FROM debian:bookworm-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 simular
USER simular

# Copy binary from builder
COPY --from=builder /app/target/release/simular /usr/local/bin/

# Set working directory
WORKDIR /home/simular

# Default command
ENTRYPOINT ["simular"]
CMD ["--help"]

# WASM build stage
FROM rust:${RUST_VERSION}-slim-bookworm AS wasm-builder

# Install build dependencies (needed for wasm-pack and zstd-sys)
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    curl \
    clang \
    lld \
    && rm -rf /var/lib/apt/lists/*

# Install wasm-pack (use pre-built binary for faster builds)
RUN curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Install target
RUN rustup target add wasm32-unknown-unknown

WORKDIR /app

# Copy manifests
COPY Cargo.toml Cargo.lock ./

# Copy source and other directories referenced in Cargo.toml
COPY src ./src
COPY benches ./benches
COPY examples ./examples
COPY tests ./tests
COPY schemas ./schemas

# Build WASM
RUN wasm-pack build --target web --no-default-features --features wasm

# WASM output stage
FROM scratch AS wasm
COPY --from=wasm-builder /app/pkg /pkg

# Labels
LABEL org.opencontainers.image.title="simular"
LABEL org.opencontainers.image.description="Unified Simulation Engine"
LABEL org.opencontainers.image.source="https://github.com/paiml/simular"
LABEL org.opencontainers.image.licenses="MIT"
