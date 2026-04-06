# Trueno-DB Makefile
# Toyota Way: Extreme TDD with quality gates

# Quality directives (bashrs enforcement)
.SUFFIXES:
.DELETE_ON_ERROR:
.ONESHELL:

.PHONY: help build test test-fast bench lint check coverage mutants tdg clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

## Development Commands

build: ## Build the project
	cargo build

build-release: ## Build release version
	cargo build --release

test: ## Run all tests
	cargo test --all-features

test-fast: ## Run fast unit tests only (tier 1)
	cargo test --lib

test-verbose: ## Run tests with verbose output
	cargo test --all-features -- --nocapture

bench: ## Run benchmarks
	cargo bench

## Quality Gates (EXTREME TDD)

lint: ## Run clippy with zero tolerance (ALL targets: lib, tests, examples, benches)
	cargo clippy --all-targets --all-features -- -D warnings

lint-pedantic: ## Run clippy with pedantic lints (for continuous improvement)
	cargo clippy --all-targets --all-features -- -D warnings -D clippy::pedantic

check: lint test ## Run basic quality checks

coverage: ## Generate coverage report (≥90% required, GPU excluded due to LLVM instrumentation limits)
	@echo "📊 Generating coverage report (target: ≥90%, GPU excluded)..."
	@echo "    Note: GPU backend excluded (LLVM coverage cannot instrument GPU shaders)"
	@# Temporarily disable mold linker (breaks LLVM coverage)
	@cargo llvm-cov --all-features --workspace --lcov --output-path lcov.info
	@cargo llvm-cov report --html --output-dir target/coverage/html
	@# Restore mold linker
	@echo "✅ Coverage report: target/coverage/html/index.html"
	@echo ""
	@echo "📊 Coverage by Component:"
	@cargo llvm-cov report | python3 -c "import sys; lines = [l.strip() for l in sys.stdin if l.strip()]; total_line = [l for l in lines if l.startswith('TOTAL')]; parts = total_line[0].split() if total_line else []; cov_str = parts[-4].rstrip('%') if len(parts) >= 10 else '0'; cov = float(cov_str); total_lines = int(parts[7]) if len(parts) >= 10 else 0; missed_lines = int(parts[8]) if len(parts) >= 10 else 0; covered_lines = total_lines - missed_lines; print(f'   Trueno-DB:      {cov:.2f}% ({covered_lines:,}/{total_lines:,} lines)'); print(''); print('   ✅ PASS: Coverage ≥90%' if cov >= 90 else f'   ❌ FAIL: Coverage ({cov:.2f}%) below 90%')"

coverage-check: ## Enforce 90% coverage threshold (BLOCKS on failure, GPU excluded)
	@echo "🔒 Enforcing 90% coverage threshold (GPU excluded)..."
	@# Temporarily disable mold linker (breaks LLVM coverage)
	@cargo llvm-cov --all-features --workspace --lcov --output-path lcov.info > /dev/null 2>&1
	@# Restore mold linker
	@cargo llvm-cov report | python3 -c "import sys; lines = [l.strip() for l in sys.stdin if l.strip()]; total_line = [l for l in lines if l.startswith('TOTAL')]; parts = total_line[0].split() if total_line else []; cov_str = parts[-4].rstrip('%') if len(parts) >= 4 else '0'; cov = float(cov_str); print(f'Overall coverage: {cov:.2f}%'); exit_code = 1 if cov < 90 else 0; print(f'✅ Coverage threshold met (≥90%)' if exit_code == 0 else f'❌ FAIL: Coverage {cov:.2f}% below 90% threshold'); sys.exit(exit_code)"

mutants: ## Run mutation testing (target: ≥85% kill rate, certeza formula)
	@echo "🧬 Running mutation testing (this will take a while)..."
	@echo "Target: >85% mutation score"
	@if command -v cargo-mutants >/dev/null 2>&1; then \
		cargo mutants --no-times --output mutants.out || true; \
		echo "✅ Mutation testing complete. Results in mutants.out/"; \
	else \
		echo "📥 Installing cargo-mutants..."; \
		cargo install cargo-mutants && cargo mutants --no-times --output mutants.out || true; \
	fi

mutation-report: ## Analyze mutation test results
	@echo "📊 Analyzing mutation test results..."
	@if [ -d "mutants.out" ]; then \
		cat mutants.out/mutants.out 2>/dev/null || echo "No mutation results yet"; \
	else \
		echo "No mutation results found. Run 'make mutants' first."; \
	fi

mutation-clean: ## Clean mutation testing artifacts
	@rm -rf mutants.out mutants.out.old
	@echo "✓ Mutation testing artifacts cleaned"

tdg: ## Run TDG analysis (target: ≥B+ / 85)
	pmat analyze tdg

quality-gate: lint test coverage-check ## Run full quality gate (BLOCKS if coverage < 90%)
	@echo "✅ All quality gates passed"

## Backend Equivalence Tests (Critical)

test-equivalence: ## Test GPU == SIMD == Scalar
	cargo test --test backend_equivalence --all-features

## WASM Build

wasm: ## Build for WASM target
	cargo build --target wasm32-unknown-unknown --release --features wasm

wasm-test: ## Test WASM build
	wasm-pack test --headless --firefox

## Benchmarking

bench-gpu: ## Benchmark GPU backend
	cargo bench --bench aggregations -- --gpu

bench-simd: ## Benchmark SIMD backend
	cargo bench --bench aggregations -- --simd

bench-comparison: ## Compare all backends
	cargo bench --bench backend_comparison

bench-competitive: ## Benchmark vs DuckDB/SQLite (temporarily disables mold linker)
	@echo "🏁 Running competitive benchmarks (Trueno vs DuckDB vs SQLite)..."
	@echo "    Note: Mold linker temporarily disabled (DuckDB build compatibility)"
	@# Temporarily disable mold linker (breaks DuckDB build)
	@test -f ~/.cargo/config.toml && mv ~/.cargo/config.toml ~/.cargo/config.toml.bench-backup || true
	@cargo bench --bench competitive_benchmarks
	@# Restore mold linker
	@test -f ~/.cargo/config.toml.bench-backup && mv ~/.cargo/config.toml.bench-backup ~/.cargo/config.toml || true
	@echo "✅ Competitive benchmarks complete"

## Documentation

book: ## Build the mdBook
	cd book && mdbook build

book-serve: ## Serve the book locally
	cd book && mdbook serve --open

book-watch: ## Watch and rebuild book on changes
	cd book && mdbook watch

## Maintenance

clean: ## Clean build artifacts
	cargo clean
	rm -rf target/ || exit 1
	rm -rf coverage/ || exit 1
	rm -rf book/book/ || exit 1

update-trueno: ## Update trueno to latest version
	@echo "Checking latest trueno version..."
	cargo search trueno | head -1
	@echo "Current version:"
	cargo tree | grep trueno
	@echo "To update: cargo update trueno"

check-deps: ## Check dependency versions
	cargo tree | grep -E "(trueno|wgpu|arrow|parquet)"

## Docker (Future)

docker-build: ## Build Docker image
	docker build -t trueno-db:latest .

docker-test: ## Run tests in Docker
	docker run --rm trueno-db:latest cargo test

## CI/CD

ci: lint test coverage ## Run CI pipeline
	@echo "✅ CI pipeline passed"

pre-commit: lint test ## Pre-commit hook
	@echo "✅ Pre-commit checks passed"

## WASM Build (Phase 4)

WASM_PKG_DIR := wasm-pkg
WASM_OUT_DIR := $(WASM_PKG_DIR)/pkg

wasm-build: ## Build WASM package with wasm-pack
	@echo "Building WASM package..."
	@command -v wasm-pack >/dev/null 2>&1 || { echo "Installing wasm-pack..."; cargo install wasm-pack; }
	cd $(WASM_PKG_DIR) && wasm-pack build --target web --release
	@echo "WASM built: $(WASM_OUT_DIR)/"

wasm-build-simd: ## Build WASM with SIMD128 + WebGPU
	@echo "Building WASM with SIMD128 + WebGPU..."
	cd $(WASM_PKG_DIR) && RUSTFLAGS="-C target-feature=+simd128 --cfg=web_sys_unstable_apis" wasm-pack build --target web --release --features webgpu
	@echo "SIMD128 + WebGPU WASM built"

WASM_PORT ?= 8080
wasm-serve: wasm-build-simd ## Build and serve WASM demo (ruchy or python)
	@echo "Starting demo at http://localhost:$(WASM_PORT)/"
	@echo "Press Ctrl+C to stop"
	@if command -v ruchy >/dev/null 2>&1; then \
		echo "Using ruchy (fast)"; \
		cd $(WASM_PKG_DIR) && ruchy serve . --port $(WASM_PORT); \
	else \
		echo "Using Python (install ruchy for faster: cargo install ruchy)"; \
		cd $(WASM_PKG_DIR) && python3 -m http.server $(WASM_PORT); \
	fi

wasm-clean: ## Clean WASM build artifacts
	rm -rf $(WASM_OUT_DIR) $(WASM_PKG_DIR)/target

wasm-check: ## Check WASM compiles
	cd $(WASM_PKG_DIR) && cargo check --target wasm32-unknown-unknown

wasm-e2e: wasm-build-simd ## Run E2E tests with Playwright
	@echo "Running E2E tests..."
	@cd $(WASM_PKG_DIR) && npm install --silent 2>/dev/null || npm install
	@cd $(WASM_PKG_DIR) && npx playwright install chromium --with-deps 2>/dev/null || true
	cd $(WASM_PKG_DIR) && npx playwright test

wasm-e2e-headed: wasm-build-simd ## Run E2E tests with visible browser
	@cd $(WASM_PKG_DIR) && npm install --silent 2>/dev/null || npm install
	cd $(WASM_PKG_DIR) && npx playwright test --headed

wasm-e2e-debug: wasm-build-simd ## Run E2E tests in debug mode
	@cd $(WASM_PKG_DIR) && npm install --silent 2>/dev/null || npm install
	cd $(WASM_PKG_DIR) && npx playwright test --debug

wasm-e2e-update: wasm-build-simd ## Update E2E test screenshots
	@cd $(WASM_PKG_DIR) && npm install --silent 2>/dev/null || npm install
	cd $(WASM_PKG_DIR) && npx playwright test --update-snapshots

# ============================================================================
# RELEASE (crates.io publishing)
# ============================================================================

TRUENO_VERSION := 0.7.4

release-prep: ## Prepare Cargo.toml for release (swap path → version)
	@echo "🔄 Preparing Cargo.toml for release..."
	@sed -i 's|trueno = { path = "../trueno" }.*|trueno = "$(TRUENO_VERSION)"  # SIMD fallback + hash module|' Cargo.toml
	@echo "✅ Updated trueno dependency to version $(TRUENO_VERSION)"
	@grep "^trueno = " Cargo.toml

release-dev: ## Restore Cargo.toml for local development (swap version → path)
	@echo "🔄 Restoring Cargo.toml for local development..."
	@sed -i 's|trueno = "$(TRUENO_VERSION)".*|trueno = { path = "../trueno" }  # For local dev; change to "$(TRUENO_VERSION)" for release|' Cargo.toml
	@echo "✅ Restored trueno to path dependency"
	@grep "^trueno = " Cargo.toml

release-check: release-prep ## Verify package can be published (dry-run)
	@echo "🔍 Checking release readiness..."
	cargo publish --dry-run --allow-dirty || ($(MAKE) release-dev && exit 1)
	@$(MAKE) release-dev
	@echo "✅ Package ready for release"

release: release-prep ## Publish to crates.io (requires cargo login, trueno must be published first)
	@echo "🚀 Publishing trueno-db to crates.io..."
	@echo "⚠️  Ensure trueno $(TRUENO_VERSION) is already published!"
	cargo publish --allow-dirty || ($(MAKE) release-dev && exit 1)
	@$(MAKE) release-dev
	@echo "✅ Published successfully"
	@echo "📦 Create GitHub release: gh release create v$$(cargo pkgid | cut -d# -f2)"

release-tag: ## Create git tag for current version
	@VERSION=$$(cargo pkgid | cut -d# -f2) && \
	echo "🏷️  Creating tag v$$VERSION..." && \
	git tag -a "v$$VERSION" -m "Release v$$VERSION" && \
	git push origin "v$$VERSION" && \
	echo "✅ Tag v$$VERSION pushed"
