# entrenar - Sovereign AI Training Library
# Multi-stage build for reproducible training environments

# Stage 1: Build
FROM rust:1.87-bookworm AS builder

RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY crates/ crates/
COPY examples/ examples/
COPY benches/ benches/
COPY tests/ tests/

# Build release binary (CPU-only by default)
RUN cargo build --release --lib

# Stage 2: Test
FROM builder AS tester
RUN cargo test --lib --quiet

# Stage 3: Runtime
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /build/target/release/libentrenar.rlib /app/

# For CUDA builds, use nvidia/cuda base image instead:
# FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder-cuda

ENV RUST_LOG=info
