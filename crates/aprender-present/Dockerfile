# Reproducible Build Environment for Presentar
# Ensures identical builds across all systems
#
# Build: docker build -t presentar .
# Run:   docker run -it presentar cargo test
# Dev:   docker run -it -v $(pwd):/app presentar bash

FROM rust:1.83.0-bookworm AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set reproducibility environment
ENV CARGO_INCREMENTAL=0
ENV RUSTFLAGS="-D warnings"
ENV RUST_BACKTRACE=1

# Create app directory
WORKDIR /app

# Copy dependency files first for caching
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates/*/Cargo.toml ./crates/

# Create dummy source files for dependency caching
RUN mkdir -p crates/presentar/src && echo "fn main() {}" > crates/presentar/src/lib.rs
RUN mkdir -p crates/presentar-terminal/src && echo "fn main() {}" > crates/presentar-terminal/src/lib.rs
RUN mkdir -p crates/presentar-core/src && echo "fn main() {}" > crates/presentar-core/src/lib.rs
RUN mkdir -p crates/presentar-test/src && echo "fn main() {}" > crates/presentar-test/src/lib.rs
RUN mkdir -p crates/presentar-test-macros/src && echo "fn main() {}" > crates/presentar-test-macros/src/lib.rs
RUN mkdir -p crates/presentar-yaml/src && echo "fn main() {}" > crates/presentar-yaml/src/lib.rs

# Build dependencies only
RUN cargo build --release 2>/dev/null || true

# Copy actual source
COPY . .

# Build the project
RUN cargo build --release

# Runtime image
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/ptop /usr/local/bin/

ENTRYPOINT ["ptop"]

# Development image
FROM builder AS development

# Install additional dev tools
RUN rustup component add rustfmt clippy
RUN cargo install cargo-llvm-cov cargo-nextest

CMD ["bash"]
