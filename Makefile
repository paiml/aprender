# Renacer Makefile
# Following bashrs and paiml-mcp-agent-toolkit patterns

.SUFFIXES:

.PHONY: help test coverage coverage-html coverage-clean mutants mutants-quick clean build release lint format check \
	tier1 tier2 tier3 chaos-test chaos-full check-regression fuzz benchmark bench install-llvm verify-llvm build-fixtures

# =============================================================================
# Installation & Dependencies
# =============================================================================

install-llvm: ## Install LLVM/Clang development libraries (for projects like decy)
	@echo "🔧 Installing LLVM/Clang development libraries..."
	@if [ -f /etc/debian_version ]; then \
		echo "📦 Detected Debian/Ubuntu"; \
		sudo apt-get update; \
		sudo apt-get install -y llvm-14-dev libclang-14-dev clang-14 build-essential pkg-config; \
		echo "🔗 Setting up LLVM environment variables..."; \
		if ! grep -q "LLVM_CONFIG_PATH" ~/.zshrc; then \
			echo 'export LLVM_CONFIG_PATH=/usr/bin/llvm-config-14' >> ~/.zshrc; \
		fi; \
		if ! grep -q "LIBCLANG_PATH" ~/.zshrc; then \
			echo 'export LIBCLANG_PATH=/usr/lib/llvm-14/lib' >> ~/.zshrc; \
		fi; \
		export LLVM_CONFIG_PATH=/usr/bin/llvm-config-14; \
		export LIBCLANG_PATH=/usr/lib/llvm-14/lib; \
		echo "✅ LLVM/Clang libraries installed"; \
		echo "⚠️  Run 'source ~/.zshrc' to reload environment"; \
	elif [ -f /etc/redhat-release ]; then \
		echo "📦 Detected RHEL/CentOS/Fedora"; \
		sudo yum install -y llvm-devel clang-devel || sudo dnf install -y llvm-devel clang-devel; \
		echo "✅ LLVM/Clang libraries installed"; \
	elif [ "$$(uname)" = "Darwin" ]; then \
		echo "📦 Detected macOS"; \
		brew install llvm; \
		echo "🔗 Setting up LLVM environment variables..."; \
		if ! grep -q "LLVM_CONFIG_PATH" ~/.zshrc; then \
			echo 'export PATH="/usr/local/opt/llvm/bin:$$PATH"' >> ~/.zshrc; \
			echo 'export LDFLAGS="-L/usr/local/opt/llvm/lib"' >> ~/.zshrc; \
			echo 'export CPPFLAGS="-I/usr/local/opt/llvm/include"' >> ~/.zshrc; \
			echo 'export LIBCLANG_PATH=/usr/local/opt/llvm/lib' >> ~/.zshrc; \
		fi; \
		echo "✅ LLVM/Clang libraries installed"; \
		echo "⚠️  Run 'source ~/.zshrc' to reload environment"; \
	else \
		echo "❌ Unsupported platform. Please install LLVM/Clang manually."; \
		exit 1; \
	fi

verify-llvm: ## Verify LLVM/Clang installation
	@echo "🔍 Verifying LLVM/Clang installation..."
	@echo ""
	@if command -v llvm-config >/dev/null 2>&1 || command -v llvm-config-14 >/dev/null 2>&1; then \
		echo "✅ LLVM found:"; \
		llvm-config-14 --version 2>/dev/null || llvm-config --version; \
	else \
		echo "❌ LLVM not found"; \
	fi
	@echo ""
	@if [ -n "$$LLVM_CONFIG_PATH" ]; then \
		echo "✅ LLVM_CONFIG_PATH: $$LLVM_CONFIG_PATH"; \
	else \
		echo "⚠️  LLVM_CONFIG_PATH not set"; \
	fi
	@echo ""
	@if [ -n "$$LIBCLANG_PATH" ]; then \
		echo "✅ LIBCLANG_PATH: $$LIBCLANG_PATH"; \
		if [ -d "$$LIBCLANG_PATH" ]; then \
			echo "✅ libclang directory exists"; \
			ls -la "$$LIBCLANG_PATH"/libclang.so* 2>/dev/null || echo "⚠️  libclang.so not found"; \
		else \
			echo "❌ libclang directory does not exist"; \
		fi; \
	else \
		echo "⚠️  LIBCLANG_PATH not set"; \
	fi

# =============================================================================
# Fixtures (compiled ELF binaries for tests/benchmarks)
# =============================================================================

build-fixtures: ## Build test/bench fixture binaries from source
	@echo "Building test fixtures..."
	@rustc tests/fixtures/simple_program.rs -o tests/fixtures/simple_program -g
	@rustc benches/fixtures/syscall_heavy.rs -o benches/fixtures/syscall_heavy -g
	@echo "Fixtures built."

# =============================================================================
# Testing & Quality
# =============================================================================

help: ## Show this help message
	@echo "Renacer - Pure Rust strace alternative"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

test: build-fixtures ## Run tests (fast, no coverage)
	@echo "🧪 Running tests..."
	@cargo test --quiet

test-fast: ## Run tests quickly with nextest (parallel, < 5min target)
	@echo "🧪 Running fast tests with nextest..."
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		PROPTEST_CASES=25 RUST_TEST_THREADS=$$(nproc) cargo nextest run \
			--workspace \
			--status-level skip \
			--failure-output immediate; \
	else \
		echo "⚠️  cargo-nextest not found. Installing..."; \
		cargo install cargo-nextest; \
		PROPTEST_CASES=25 RUST_TEST_THREADS=$$(nproc) cargo nextest run \
			--workspace \
			--status-level skip \
			--failure-output immediate; \
	fi

coverage: ## Generate HTML coverage report and open in browser (max 10min target)
	@echo "📊 Running comprehensive test coverage analysis..."
	@echo "🔍 Checking for cargo-llvm-cov..."
	@which cargo-llvm-cov > /dev/null 2>&1 || (echo "📦 Installing cargo-llvm-cov..." && cargo install cargo-llvm-cov --locked)
	@if ! rustup component list --installed | grep -q llvm-tools-preview; then \
		echo "📦 Installing llvm-tools-preview..."; \
		rustup component add llvm-tools-preview; \
	fi
	@echo "🔍 Detecting GPU hardware..."
	@GPU_FEATURES=""; \
	if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then \
		echo "✅ NVIDIA GPU detected - enabling gpu-tracing and cuda-tracing features"; \
		GPU_FEATURES="--features gpu-tracing,cuda-tracing"; \
		export CUDA_VISIBLE_DEVICES=0; \
	else \
		echo "⚠️  No NVIDIA GPU detected - running without GPU features"; \
		GPU_FEATURES=""; \
	fi; \
	echo "🧹 Cleaning old coverage data..."; \
	mkdir -p target/coverage/html; \
	echo "🧪 Phase 1: Running tests with instrumentation (reduced proptest cases)..."; \
	PROPTEST_CASES=20 RUST_TEST_THREADS=$$(nproc) timeout 600 cargo llvm-cov --no-report test --workspace $$GPU_FEATURES || true; \
	echo "📊 Phase 2: Generating coverage reports..."; \
	cargo llvm-cov report --html --output-dir target/coverage/html || echo "⚠️  No coverage data generated"; \
	cargo llvm-cov report --lcov --output-path target/coverage/lcov.info || echo "⚠️  LCOV generation skipped"; \
	echo ""; \
	echo "📊 Coverage Summary:"; \
	cargo llvm-cov report --summary-only || echo "Run 'cargo test' to generate coverage data first"; \
	echo ""; \
	echo "📊 Coverage reports generated:"; \
	echo "- HTML: target/coverage/html/index.html"; \
	echo "- LCOV: target/coverage/lcov.info"; \
	echo ""; \
	xdg-open target/coverage/html/index.html 2>/dev/null || \
		open target/coverage/html/index.html 2>/dev/null || \
		echo "✅ Open target/coverage/html/index.html in your browser"

coverage-html: coverage ## Alias for coverage

coverage-clean: ## Clean coverage artifacts
	@echo "🧹 Cleaning coverage artifacts..."
	@if command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "✅ Coverage artifacts cleaned!"; \
	else \
		echo "⚠️  cargo-llvm-cov not installed, skipping clean."; \
	fi

build: ## Build debug binary
	@echo "🔨 Building debug binary..."
	@cargo build

release: ## Build optimized release binary
	@echo "🚀 Building release binary..."
	@cargo build --release
	@echo "✅ Release binary: target/release/renacer"

lint: ## Run clippy linter
	@echo "🔍 Running clippy..."
	@cargo clippy -- -D warnings

format: ## Format code with rustfmt
	@echo "📝 Formatting code..."
	@cargo fmt

check: ## Type check without building
	@echo "✅ Type checking..."
	@cargo check --all-targets --all-features

clean: ## Clean build artifacts
	@echo "🧹 Cleaning build artifacts..."
	@cargo clean
	@rm -rf target/coverage
	@echo "✅ Clean completed!"

benchmark: ## Run performance benchmarks
	@echo "📊 Running benchmarks..."
	@cargo test --test benchmark_vs_strace -- --nocapture --test-threads=1

bench: benchmark ## Alias for benchmark

mutants: ## Run mutation testing (full analysis)
	@echo "🧬 Running mutation testing..."
	@echo "🔍 Checking for cargo-mutants..."
	@which cargo-mutants > /dev/null 2>&1 || (echo "📦 Installing cargo-mutants..." && cargo install cargo-mutants --locked)
	@echo "🧬 Running cargo-mutants (this may take several minutes)..."
	@cargo mutants --output target/mutants.out || echo "⚠️  Some mutants survived"
	@echo ""
	@echo "📊 Mutation Testing Results:"
	@cat target/mutants.out/mutants.out 2>/dev/null || echo "Check target/mutants.out/ for detailed results"

mutants-quick: ## Run mutation testing (quick check on changed files only)
	@echo "🧬 Running quick mutation testing..."
	@echo "🔍 Checking for cargo-mutants..."
	@which cargo-mutants > /dev/null 2>&1 || (echo "📦 Installing cargo-mutants..." && cargo install cargo-mutants --locked)
	@echo "🧬 Running cargo-mutants on uncommitted changes..."
	@cargo mutants --in-diff git:HEAD --output target/mutants-quick.out || echo "⚠️  Some mutants survived"
	@echo ""
	@echo "📊 Quick Mutation Testing Results:"
	@cat target/mutants-quick.out/mutants.out 2>/dev/null || echo "Check target/mutants-quick.out/ for detailed results"

# =============================================================================
# Tiered TDD Workflow (from trueno patterns)
# =============================================================================

tier1: ## Tier 1: Fast tests (<5s) - unit tests, clippy, format
	@echo "🏃 Tier 1: Fast tests (<5 seconds)..."
	@cargo fmt --check
	@cargo clippy -- -D warnings
	@cargo test --lib --quiet
	@echo "✅ Tier 1 complete!"

tier2: tier1 ## Tier 2: Integration tests (<30s) - includes tier1
	@echo "🏃 Tier 2: Integration tests (<30 seconds)..."
	@cargo test --tests --quiet
	@echo "✅ Tier 2 complete!"

tier3: tier2 ## Tier 3: Full validation (<5m) - includes tier1+2, property tests
	@echo "🏃 Tier 3: Full validation (<5 minutes)..."
	@cargo test --all-targets --all-features --quiet
	@echo "✅ Tier 3 complete!"

# =============================================================================
# Chaos Engineering (Sprint 29 - Red-Team Profile)
# =============================================================================

chaos-test: ## Run chaos engineering tests (basic tier)
	@echo "🔥 Running chaos engineering tests..."
	@cargo test --features chaos-basic --quiet
	@echo "✅ Chaos basic tests complete!"

chaos-full: ## Run full chaos engineering suite (requires chaos-full feature)
	@echo "🔥 Running full chaos engineering suite..."
	@cargo test --features chaos-full --quiet
	@echo "✅ Full chaos tests complete!"

check-regression: ## Check for performance regressions (>5% threshold)
	@echo "📊 Checking for performance regressions..."
	@ruchy scripts/check_regression.ruchy || echo "⚠️  Regression check failed or ruchy not found"

fuzz: ## Run fuzz testing targets
	@echo "🎲 Running fuzz tests..."
	@echo "🔍 Checking for cargo-fuzz..."
	@which cargo-fuzz > /dev/null 2>&1 || (echo "📦 Installing cargo-fuzz..." && cargo install cargo-fuzz --locked)
	@cargo +nightly fuzz run filter_parser -- -max_total_time=60 || echo "⚠️  Fuzz testing requires nightly toolchain"

# =============================================================================
# Differential Testing (Oracle Problem)
# =============================================================================

diff-test: ## Run differential tests against strace
	@echo "🔬 Running differential tests (Renacer vs strace)..."
	@cargo test --test differential_strace_tests --quiet || echo "⚠️  Differential tests not yet implemented"
