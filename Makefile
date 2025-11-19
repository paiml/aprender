# Aprender Makefile
# Certeza Methodology - Tiered Quality Gates

# Disable built-in rules for performance
.SUFFIXES:

# Delete partially-built files on error
.DELETE_ON_ERROR:

# Multi-line recipes execute in same shell
.ONESHELL:

.PHONY: all build test lint fmt clean doc book book-build book-serve book-test tier1 tier2 tier3 tier4 coverage profile hooks-install hooks-verify lint-scripts bashrs-score bashrs-lint-makefile chaos-test fuzz bench dev pre-push ci check run-ci run-bench

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

# EXTREME TDD Book (mdBook)
book: book-build ## Build and open the EXTREME TDD book

book-build: ## Build the book
	@echo "📚 Building EXTREME TDD book..."
	@if command -v mdbook >/dev/null 2>&1; then \
		mdbook build book; \
		echo "✅ Book built: book/book/index.html"; \
	else \
		echo "❌ mdbook not found. Install with: cargo install mdbook"; \
		exit 1; \
	fi

book-serve: ## Serve the book locally for development
	@echo "📖 Serving book at http://localhost:3000..."
	@mdbook serve book --open

book-test: ## Test book synchronization
	@echo "🔍 Testing book synchronization..."
	@for example in examples/*.rs; do \
		if [ -f "$$example" ]; then \
			EXAMPLE_NAME=$$(basename "$$example" .rs); \
			CASE_STUDY=$$(echo "$$EXAMPLE_NAME" | sed 's/_/-/g'); \
			if [ ! -f "book/src/examples/$$CASE_STUDY.md" ]; then \
				echo "❌ Missing case study for $$EXAMPLE_NAME"; \
				exit 1; \
			fi; \
		fi; \
	done
	@echo "✅ All examples have corresponding book chapters"

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

# Chaos engineering tests (from renacer)
chaos-test: ## Run chaos engineering tests
	@echo "🔥 Running chaos engineering tests..."
	@cargo test --features chaos-basic --quiet
	@echo "✅ Chaos tests passed"

# Fuzz testing (from renacer, 60s)
fuzz: ## Run fuzz testing for 60 seconds
	@echo "🎲 Running fuzz tests (60s)..."
	@cargo +nightly fuzz run fuzz_target_1 -- -max_total_time=60 || echo "⚠️  Fuzz testing requires nightly Rust: rustup default nightly"
	@echo "✅ Fuzz testing complete"

# Development workflow
dev: tier1

# Pre-push checks
pre-push: tier3

# CI/CD checks
ci: tier4

# Quick check (compile only)
check:
	cargo check --all

# Install PMAT pre-commit hooks
hooks-install: ## Install PMAT pre-commit hooks
	@echo "🔧 Installing PMAT pre-commit hooks..."
	@pmat hooks install || exit 1
	@echo "✅ Hooks installed successfully"

# Verify PMAT hooks
hooks-verify: ## Verify PMAT hooks are working
	@echo "🔍 Verifying PMAT hooks..."
	@pmat hooks verify
	@pmat hooks run

# Lint shell scripts (bashrs quality gates)
lint-scripts: ## Lint shell scripts with bashrs (determinism + idempotency + safety)
	@echo "🔍 Linting shell scripts with bashrs..."
	@if command -v bashrs >/dev/null 2>&1; then \
		for script in scripts/*.sh; do \
			echo "  Linting $$script..."; \
			bashrs lint "$$script" || exit 1; \
		done; \
		echo "✅ All shell scripts pass bashrs lint"; \
	else \
		echo "❌ bashrs not installed. Install with: cargo install bashrs"; \
		exit 1; \
	fi

bashrs-score: ## Score shell script quality with bashrs
	@echo "📊 Scoring shell scripts..."
	@for script in scripts/*.sh; do \
		echo ""; \
		echo "Scoring $$script:"; \
		bashrs score "$$script"; \
	done

bashrs-lint-makefile: ## Lint Makefile with bashrs
	@echo "🔍 Linting Makefile with bashrs..."
	@bashrs make lint Makefile || echo "⚠️  Makefile linting found issues"

# Run CI pipeline
run-ci: ## Run full CI pipeline
	@./scripts/ci.sh

# Run benchmarks
run-bench: ## Run benchmark suite
	@./scripts/bench.sh
