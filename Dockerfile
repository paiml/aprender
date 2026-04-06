# Multi-stage Dockerfile for Hermetic Builds
# Certeza Scientific Benchmarking - Phase 3.5
#
# This Dockerfile enables reproducible builds and benchmarking by:
# 1. Pinning exact Rust toolchain version
# 2. Isolating from host environment
# 3. Capturing complete build environment
# 4. Supporting byte-identical binary verification

# =============================================================================
# Stage 1: Builder - Hermetic Rust build environment
# =============================================================================
FROM rust:1.82.0-slim-bookworm AS builder

# Set working directory
WORKDIR /build

# Install system dependencies needed for build and benchmarking
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Deno for reporting scripts
RUN curl -fsSL https://deno.land/install.sh | sh
ENV PATH="/root/.deno/bin:$PATH"

# Install hyperfine for benchmarking
RUN cargo install hyperfine --version 1.19.0

# Copy rust-toolchain.toml to ensure exact toolchain version
COPY rust-toolchain.toml ./

# Copy Cargo files for dependency caching
COPY Cargo.toml Cargo.lock ./

# Create dummy source to cache dependencies
RUN mkdir -p src benches && \
    echo "fn main() {}" > src/lib.rs && \
    echo "fn main() {}" > benches/vec_benchmarks.rs && \
    cargo build --release && \
    cargo build --release --benches && \
    rm -rf src benches

# Copy actual source code
COPY src/ ./src/
COPY benches/ ./benches/
COPY scripts/ ./scripts/
COPY benchmarks/ ./benchmarks/

# Build release binaries
RUN cargo build --release --all-targets

# Generate reproducibility manifest
COPY scripts/generate_reproducibility_manifest.sh ./scripts/
RUN chmod +x scripts/generate_reproducibility_manifest.sh && \
    ./scripts/generate_reproducibility_manifest.sh

# =============================================================================
# Stage 2: Runner - Minimal runtime for benchmarking
# =============================================================================
FROM debian:bookworm-slim AS runner

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Deno for reporting
RUN curl -fsSL https://deno.land/install.sh | sh
ENV PATH="/root/.deno/bin:$PATH"

# Create application directory
WORKDIR /app

# Copy built artifacts from builder
COPY --from=builder /build/target/release/certeza /app/
COPY --from=builder /build/target/release/deps/*benchmarks* /app/benches/
COPY --from=builder /root/.cargo/bin/hyperfine /usr/local/bin/

# Copy scripts and configuration
COPY --from=builder /build/scripts/ /app/scripts/
COPY --from=builder /build/benchmarks/ /app/benchmarks/
COPY --from=builder /build/Cargo.toml /build/Cargo.lock /app/
COPY --from=builder /build/rust-toolchain.toml /app/

# Copy reproducibility manifest
COPY --from=builder /build/benchmarks/metadata/toolchain_manifest*.txt /app/benchmarks/metadata/

# Set permissions
RUN chmod +x /app/scripts/*.sh /app/scripts/*.ts

# Create output directories
RUN mkdir -p /app/benchmarks/results /app/benchmarks/baselines /app/benchmarks/history

# Set environment variables for reproducibility
ENV RUST_BACKTRACE=1
ENV CARGO_TERM_COLOR=always

# Default command: run comprehensive benchmark suite
CMD ["/bin/bash", "/app/scripts/run_benchmarks.sh", "--benchmarks", "all", "--warmup", "5", "--iterations", "20"]

# =============================================================================
# Stage 3: Verifier - Validate reproducibility
# =============================================================================
FROM builder AS verifier

WORKDIR /verify

# Copy source again for rebuild verification
COPY . .

# Rebuild and compare binaries
RUN cargo build --release && \
    sha256sum target/release/certeza > /tmp/rebuild_hash.txt && \
    echo "Binary hashes:" && \
    cat /tmp/rebuild_hash.txt

# =============================================================================
# Metadata Labels
# =============================================================================
LABEL org.opencontainers.image.title="certeza"
LABEL org.opencontainers.image.description="Scientific benchmarking framework for Rust"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.authors="Pragmatic AI Labs"
LABEL org.opencontainers.image.source="https://github.com/paiml/certeza"
LABEL org.opencontainers.image.licenses="MIT OR Apache-2.0"

# Build metadata
LABEL certeza.rust-version="1.82.0"
LABEL certeza.stage="Phase 3.5: Reproducibility and Archival"
LABEL certeza.reproducibility="hermetic-build"
