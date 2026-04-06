# Justfile for Presentar
# B1: Reproducibility via standard build commands

# Default recipe
default: test

# Set random seeds for reproducibility
export RANDOM_SEED := "42"
export PRESENTAR_TEST_SEED := "42"
export PRESENTAR_BENCH_SEED := "12345"
export PROPTEST_SEED := "0xdeadbeef"

# Build all targets
build:
    cargo build --workspace --all-features

# Build release
build-release:
    cargo build --workspace --all-features --release

# Build ptop binary
build-ptop:
    cargo build -p presentar-terminal --bin ptop --features ptop --release

# Run all tests with fixed seed
test:
    cargo test --workspace --all-features -- --test-threads=1

# Run tests with coverage
coverage:
    cargo llvm-cov --workspace --all-features --lcov --output-path lcov.info

# Run benchmarks
bench:
    cargo criterion --all-features

# Run specific benchmark
bench-render:
    cargo criterion --bench terminal -- full_render

# Format code
fmt:
    cargo fmt --all

# Check formatting
fmt-check:
    cargo fmt --all -- --check

# Run clippy
clippy:
    cargo clippy --workspace --all-features -- -D warnings

# Run all quality checks
check: fmt-check clippy test

# Clean build artifacts
clean:
    cargo clean

# Generate documentation
docs:
    cargo doc --workspace --all-features --no-deps

# Install development tools
setup:
    rustup component add llvm-tools-preview
    cargo install cargo-llvm-cov cargo-criterion

# Run ptop
run-ptop: build-ptop
    ./target/release/ptop

# Pre-commit checks
pre-commit: fmt clippy test

# CI pipeline
ci: check coverage bench
