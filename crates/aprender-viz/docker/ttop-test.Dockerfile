# Dockerfile for ttop Ubuntu testing
# Build: docker build -t ttop-test -f docker/ttop-test.Dockerfile .
# Run:   docker run --rm -it ttop-test

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Copy entire workspace (simpler and ensures all deps available)
COPY . .

# Build ttop from its crate directory (excluded from main workspace)
WORKDIR /app/crates/ttop
RUN cargo build --release

# Verify binary
RUN ./target/release/ttop --version
RUN ./target/release/ttop --help

# Default: run debug mode to verify collectors initialize
CMD ["sh", "-c", "timeout 5 ./target/release/ttop --debug 2>&1; echo 'Docker test complete'"]
