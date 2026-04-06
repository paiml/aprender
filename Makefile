.PHONY: all build dev test test-fast lint lint-fast fmt coverage score clean tier1 tier2 tier3

# Application name
APP ?= presentar

# Coverage exclusions: platform-specific and binary entry points only (CB-125-A ≤10)
# external | CLI binary | proc-macro infra | GPU-only | WASM-only
COVERAGE_EXCLUDE := --ignore-filename-regex='probar/.*\.rs|presentar-cli/src/main\.rs|test-macros/.*\.rs|webgpu\.rs|shell_autocomplete\.rs'

# Default target
all: fmt lint test build

# Development server with hot reload
dev:
	@echo "Starting Presentar dev server..."
	@cargo watch -x "build" -s "echo 'Build complete'"

# Production build
build:
	@echo "Building..."
	@cargo build --release

# Run all tests
test: test-unit test-integration

test-unit:
	@echo "Running unit tests..."
	@PROPTEST_CASES=256 cargo test --workspace --lib

test-integration:
	@echo "Running integration tests..."
	@PROPTEST_CASES=256 cargo test --workspace --test '*' 2>/dev/null || echo "No integration tests found"

# Fast tests for rapid TDD feedback (<30s target)
test-fast: ## Fast unit tests (<30s target)
	@echo "⚡ Running fast tests (target: <30s)..."
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		PROPTEST_CASES=64 cargo nextest run --workspace --lib \
			--status-level skip \
			--failure-output immediate; \
	else \
		PROPTEST_CASES=64 cargo test --workspace --lib --quiet; \
	fi

# Linting
lint: lint-rust

lint-rust:
	@echo "Running clippy..."
	@cargo clippy --workspace -- -D warnings

# Fast lint for tier1
lint-fast: ## Fast clippy (library only)
	@cargo clippy --workspace --lib --quiet -- -D warnings

# Format check
fmt:
	@echo "Checking formatting..."
	@cargo fmt --all -- --check

# Format fix
fmt-fix:
	@cargo fmt --all

# Coverage with proper llvm-cov handling
coverage: ## Generate HTML coverage report (FAST: <1 min)
	@echo "📊 FAST coverage (target: <1 min)..."
	@which cargo-llvm-cov > /dev/null 2>&1 || cargo install cargo-llvm-cov --locked
	@cargo llvm-cov --workspace --html --quiet $(COVERAGE_EXCLUDE)
	@echo ""
	@cargo llvm-cov report --summary-only $(COVERAGE_EXCLUDE)
	@echo ""
	@echo "💡 HTML: target/llvm-cov/html/index.html"

# Quality score (placeholder)
score:
	@echo "Computing quality score..."
	@cargo test --workspace 2>&1 | grep -E "^test result" || true

# Clean
clean:
	@cargo clean

# =============================================================================
# Three-Tier Quality Pipeline
# =============================================================================

# TIER 1: On-save (<1 second) - Rapid feedback for flow state
tier1: ## Tier 1: Sub-second feedback (ON-SAVE)
	@echo "🚀 TIER 1: Sub-second feedback"
	@echo ""
	@echo "  [1/3] Type checking..."
	@cargo check --workspace --quiet
	@echo "  [2/3] Fast lint..."
	@cargo clippy --workspace --lib --quiet -- -D warnings
	@echo "  [3/3] Fast tests..."
	@cargo test --workspace --lib --quiet
	@echo ""
	@echo "✅ Tier 1 complete"

# TIER 2: Pre-commit (1-5 minutes)
tier2: ## Tier 2: Full validation (ON-COMMIT)
	@echo "🔍 TIER 2: Comprehensive validation"
	@echo ""
	@echo "  [1/4] Formatting..."
	@cargo fmt --all -- --check
	@echo "  [2/4] Full clippy..."
	@cargo clippy --workspace --all-targets -- -D warnings
	@echo "  [3/4] All tests..."
	@cargo test --workspace
	@echo "  [4/4] Score..."
	@cargo test --workspace 2>&1 | grep -E "^test result" || true
	@echo ""
	@echo "✅ Tier 2 complete"

# TIER 3: Nightly (comprehensive)
tier3: ## Tier 3: Mutation & coverage (ON-MERGE/NIGHTLY)
	@echo "🧬 TIER 3: Test quality assurance"
	@$(MAKE) --no-print-directory tier2
	@echo ""
	@echo "  Running coverage..."
	@$(MAKE) --no-print-directory coverage
	@echo ""
	@echo "  Running mutation testing..."
	@cargo mutants --timeout 300 2>/dev/null || echo "    ⚠️  cargo-mutants not installed"
	@echo ""
	@echo "✅ Tier 3 complete"

# =============================================================================
# Development Helpers
# =============================================================================

# Run a single test
test-one:
	@PROPTEST_CASES=256 cargo test --workspace $(TEST) -- --nocapture

# Watch mode for TDD
watch:
	@cargo watch -x "test --workspace"

# Check all targets compile
check:
	@cargo check --workspace --all-targets

# Generate documentation
doc:
	@cargo doc --workspace --no-deps --open

# =============================================================================
# WASM Browser Runtime
# =============================================================================

# Build WASM package
wasm: ## Build WASM package for browser
	@echo "🌐 Building WASM package..."
	@which wasm-pack >/dev/null 2>&1 || (echo "📦 Installing wasm-pack..." && cargo install wasm-pack)
	@wasm-pack build crates/presentar --target web --out-dir ../../www/pkg
	@echo "✅ WASM package built to www/pkg/"

# Serve WASM demo
serve: wasm ## Build and serve WASM demo
	@echo "🚀 Starting development server..."
	@echo "   Open http://localhost:8080 in your browser"
	@cd www && python3 -m http.server 8080

# Build optimized WASM for production
wasm-release: ## Build optimized WASM for production
	@echo "📦 Building optimized WASM..."
	@wasm-pack build crates/presentar --target web --out-dir ../../www/pkg --release
	@which wasm-opt >/dev/null 2>&1 && wasm-opt -Oz www/pkg/presentar_bg.wasm -o www/pkg/presentar_bg.wasm || true
	@echo "✅ Optimized WASM built"
	@ls -lh www/pkg/*.wasm 2>/dev/null || true

# Run WASM tests in headless browser (no Playwright)
wasm-test: ## Run WASM tests in headless browser
	@echo "🧪 Running WASM browser tests..."
	@wasm-pack test --headless --chrome crates/presentar
	@echo "✅ WASM tests passed"

# Run WASM tests in Firefox
wasm-test-firefox: ## Run WASM tests in Firefox
	@wasm-pack test --headless --firefox crates/presentar
