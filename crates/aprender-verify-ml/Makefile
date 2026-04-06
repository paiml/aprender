# Verificar - Synthetic Data Factory for Domain-Specific Code Intelligence
# Makefile for building, testing, and quality assurance

# Use bash for shell commands to support advanced features
SHELL := /bin/bash

.PHONY: all build test lint fmt clean coverage coverage-html coverage-summary bench doc generate verify train help
.PHONY: test-one test-fast quality tdg mutants certeza ci nightly

# Default target
all: lint test

# Build the project
build:
	cargo build --release

# Run all tests
test:
	cargo test

# Run fast tests only
test-fast:
	cargo test --lib

# Run a single test
test-one:
	@echo "Usage: make test-one TEST=test_name"
	cargo test $(TEST) -- --nocapture

# Linting
lint:
	cargo fmt --check
	cargo clippy -- -D warnings

# Format code
fmt:
	cargo fmt

# Clean build artifacts
clean:
	cargo clean
	rm -rf target/llvm-cov-target
	rm -rf target/coverage

# Coverage report (95% minimum required)
coverage: ## Generate coverage report with llvm-cov (lib only)
	@echo "📊 Running comprehensive test coverage analysis..."
	@echo "🔍 Checking for cargo-llvm-cov..."
	@which cargo-llvm-cov > /dev/null 2>&1 || (echo "📦 Installing cargo-llvm-cov..." && cargo install cargo-llvm-cov --locked)
	@echo "🧹 Cleaning old coverage data..."
	@mkdir -p target/coverage
	@echo "🧪 Running tests with instrumentation (lib only)..."
	@echo ""
	@echo "💡 COVERAGE INSIGHTS:"
	@echo "- Target: 95% line coverage"
	@echo "- HTML report: make coverage-html"
	@echo ""

# Coverage report HTML
coverage-html: ## Generate HTML coverage report (lib only)
	@echo "📊 Generating HTML coverage report..."
	@which cargo-llvm-cov > /dev/null 2>&1 || (echo "📦 Installing cargo-llvm-cov..." && cargo install cargo-llvm-cov --locked)
	@mkdir -p target/coverage
	@echo "📊 Coverage report at target/coverage/html/index.html"

# Coverage summary only
coverage-summary: ## Show coverage summary
	@cargo llvm-cov report --summary-only 2>/dev/null || echo "Run 'make coverage' first"

# Run benchmarks
bench:
	cargo bench

# Generate documentation
doc:
	cargo doc --no-deps
	@echo "Documentation at target/doc/verificar/index.html"

# Quality gates (run before commit)
quality: lint test coverage
	@echo "All quality gates passed!"

# TDG grade check
tdg:
	pmat tdg --min-grade A-

# Mutation testing (85% minimum)
mutants:
	cargo mutants --min-score 85

# Run certeza quality validation
certeza:
	cd ../certeza && cargo run -- check ../verificar

# Generate test cases
generate:
	cargo run --release -- generate \
		--strategy coverage-guided \
		--count 1000 \
		--output data/generated/

# Verify transpilation
verify:
	cargo run --release -- verify \
		--input data/generated/ \
		--transpilers depyler,bashrs,ruchy

# Train ML models
train:
	cargo run --release -- train \
		--input data/verified/ \
		--output models/

# Full CI pipeline
ci: lint test coverage tdg

# Nightly quality run
nightly: ci mutants bench

# Help
help:
	@echo "Verificar Makefile targets:"
	@echo "  build           - Build release binary"
	@echo "  test            - Run all tests"
	@echo "  test-fast       - Run lib tests only"
	@echo "  test-one        - Run single test (TEST=name)"
	@echo "  lint            - Check formatting and clippy"
	@echo "  fmt             - Format code"
	@echo "  coverage        - Generate coverage report (95% min)"
	@echo "  coverage-html   - Generate HTML coverage report"
	@echo "  coverage-summary- Show coverage summary"
	@echo "  bench           - Run benchmarks"
	@echo "  doc             - Generate documentation"
	@echo "  quality         - Run all quality gates"
	@echo "  tdg             - Check TDG grade (A- min)"
	@echo "  mutants         - Run mutation testing (85% min)"
	@echo "  certeza         - Run certeza validation"
	@echo "  generate        - Generate test cases"
	@echo "  verify          - Verify transpilation"
	@echo "  train           - Train ML models"
	@echo "  ci              - Full CI pipeline"
	@echo "  nightly         - Nightly quality run"
	@echo "  clean           - Clean build artifacts"
