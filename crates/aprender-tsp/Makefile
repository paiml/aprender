# aprender-tsp Makefile
#
# Scientific TSP solver benchmarking and reproducibility toolkit
#
# Usage:
#   make help        - Show this help
#   make test        - Run all tests
#   make bench       - Run criterion benchmarks
#   make coverage    - Generate coverage report
#   make examples    - Run all examples
#   make paper-bench - Run benchmarks for academic papers

.PHONY: all build test bench coverage lint fmt examples paper-bench help clean

# Default target
all: fmt lint test

# Build release binary
build:
	cargo build --release

# Run all tests
test:
	cargo test --all-features

# Run tests with verbose output
test-verbose:
	cargo test --all-features -- --nocapture

# Run clippy lints
lint:
	cargo clippy --all-features -- -D warnings

# Format code
fmt:
	cargo fmt --all

# Check formatting
fmt-check:
	cargo fmt --all -- --check

# Generate coverage report (bashrs-style, requires cargo-llvm-cov)
# This follows the established pattern from bashrs for reliable coverage generation
coverage:
	@echo "📊 Running coverage analysis..."
	@which cargo-llvm-cov > /dev/null 2>&1 || (echo "📦 Installing cargo-llvm-cov..." && cargo install cargo-llvm-cov --locked)
	@echo "🧹 Cleaning old coverage data..."
	@cargo llvm-cov clean --workspace
	@mkdir -p target/coverage
	@echo "⚙️  Temporarily disabling global cargo config (mold breaks coverage)..."
	@test -f ~/.cargo/config.toml && mv ~/.cargo/config.toml ~/.cargo/config.toml.cov-backup || true
	@echo "🧪 Phase 1: Running tests with instrumentation..."
	@cargo llvm-cov --no-report --all-features
	@echo "📊 Phase 2: Generating coverage reports..."
	@cargo llvm-cov report --html --output-dir target/coverage/html
	@cargo llvm-cov report --lcov --output-path target/coverage/lcov.info
	@echo "⚙️  Restoring global cargo config..."
	@test -f ~/.cargo/config.toml.cov-backup && mv ~/.cargo/config.toml.cov-backup ~/.cargo/config.toml || true
	@echo ""
	@echo "📊 Coverage Summary:"
	@cargo llvm-cov report --summary-only
	@echo ""
	@echo "💡 HTML report: target/coverage/html/index.html"

coverage-summary:
	@cargo llvm-cov report --summary-only 2>/dev/null || echo "Run 'make coverage' first"

# Run criterion benchmarks
bench:
	cargo bench --bench tsp_benchmarks

# Run all examples
examples:
	@echo "=== Running TSP Benchmark Example ==="
	cargo run --example tsp_benchmark --release
	@echo ""
	@echo "=== Running Model Persistence Example ==="
	cargo run --example tsp_model_persistence --release
	@echo ""
	@echo "=== Running Algorithm Comparison Example ==="
	cargo run --example tsp_algorithm_comparison --release

# Run benchmarks suitable for academic papers
# Uses fixed seeds and multiple iterations for statistical significance
paper-bench:
	@echo "=== Academic Paper Benchmarks ==="
	@echo "Date: $$(date -Iseconds)"
	@echo "Git SHA: $$(git rev-parse --short HEAD)"
	@echo ""
	cargo run --example tsp_benchmark --release 2>/dev/null
	@echo ""
	cargo run --example tsp_algorithm_comparison --release 2>/dev/null

# Train model on TSPLIB instances
train-berlin52:
	cargo run --release -- train data/berlin52.tsp \
		--algorithm aco \
		--iterations 1000 \
		--seed 42 \
		--output models/berlin52_aco.apr

# Benchmark trained model
benchmark-model:
	cargo run --release -- benchmark models/berlin52_aco.apr \
		--instances data/berlin52.tsp data/eil51.tsp

# Clean build artifacts
clean:
	cargo clean
	rm -f models/*.apr
	rm -f results/*.json

# Create directories
setup:
	mkdir -p models results

# Help target
help:
	@echo "aprender-tsp Makefile"
	@echo ""
	@echo "Quality Gates:"
	@echo "  make test        - Run all tests"
	@echo "  make lint        - Run clippy lints"
	@echo "  make fmt         - Format code"
	@echo "  make coverage    - Generate coverage report"
	@echo ""
	@echo "Benchmarking:"
	@echo "  make bench       - Run criterion benchmarks"
	@echo "  make paper-bench - Run benchmarks for academic papers"
	@echo "  make examples    - Run all examples"
	@echo ""
	@echo "Training:"
	@echo "  make train-berlin52  - Train ACO on berlin52.tsp"
	@echo "  make benchmark-model - Benchmark trained model"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean       - Clean build artifacts"
	@echo "  make setup       - Create required directories"

# Tier 1: Fast feedback (<1s)
tier1: fmt-check lint

# Tier 2: Pre-commit (<5s)
tier2: tier1 test

# Tier 3: Pre-push (1-5 min)
tier3: tier2 coverage

# Full CI pipeline
ci: tier3 bench paper-bench
