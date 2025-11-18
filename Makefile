# Aprender Makefile
# Certeza Methodology - Tiered Quality Gates

.PHONY: all build test lint fmt clean doc tier1 tier2 tier3 tier4 coverage profile

# Default target
all: tier2

# Build
build:
	cargo build --release

# Run all tests
test:
	cargo test --all

# Fast tests (unit tests only)
test-fast:
	cargo test --lib

# Linting
lint:
	cargo clippy -- -D warnings

# Format check
fmt:
	cargo fmt

fmt-check:
	cargo fmt --check

# Clean build artifacts
clean:
	cargo clean

# Generate documentation
doc:
	cargo doc --no-deps --open

# Tier 1: On-save (<1 second, non-blocking)
tier1:
	@echo "Running Tier 1: Fast feedback..."
	@cargo fmt --check
	@cargo clippy -- -W clippy::all
	@cargo check
	@echo "Tier 1: PASSED"

# Tier 2: Pre-commit (<5 seconds, changed files only)
tier2:
	@echo "Running Tier 2: Pre-commit checks..."
	@cargo test --lib
	@cargo clippy -- -D warnings
	@echo "Tier 2: PASSED"

# Tier 3: Pre-push (1-5 minutes, full validation)
tier3:
	@echo "Running Tier 3: Full validation..."
	@cargo test --all
	@cargo clippy -- -D warnings
	@echo "Tier 3: PASSED"

# Tier 4: CI/CD (5-60 minutes, heavyweight)
tier4: tier3
	@echo "Running Tier 4: CI/CD validation..."
	@cargo test --release
	@echo "Running pmat analysis..."
	-pmat tdg . --include-components
	@echo "Tier 4: PASSED"

# Coverage report (requires cargo-llvm-cov)
coverage: ## Generate coverage report (>85% required, <10 min target)
	@echo "📊 Generating coverage report (target: >85%, <10 min)..."
	@# Temporarily disable mold linker (breaks LLVM coverage)
	@test -f ~/.cargo/config.toml && mv ~/.cargo/config.toml ~/.cargo/config.toml.cov-backup || true
	@cargo llvm-cov --all-features --workspace --lcov --output-path lcov.info
	@cargo llvm-cov report --html --output-dir target/coverage/html
	@# Restore mold linker
	@test -f ~/.cargo/config.toml.cov-backup && mv ~/.cargo/config.toml.cov-backup ~/.cargo/config.toml || true
	@echo "✅ Coverage report: target/coverage/html/index.html"
	@cargo llvm-cov report | grep TOTAL

# Profiling (requires renacer)
profile:
	renacer --function-time --source -- cargo bench

# Benchmarks
bench:
	cargo bench

# Development workflow
dev: tier1

# Pre-push checks
pre-push: tier3

# CI/CD checks
ci: tier4

# Quick check (compile only)
check:
	cargo check --all
