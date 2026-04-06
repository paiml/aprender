# Trueno-Viz Makefile
# Lean-Scientific Development Workflow with PMAT Integration
#
# Quality Philosophy:
#   - Toyota Way: Jidoka (stop the line), Kaizen (continuous improvement)
#   - Scientific Rigor: Evidence-based, reproducible, peer-reviewed
#   - Zero Tolerance: 95% coverage, 80% mutation score, A- TDG grade

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

# ============================================================================
# Configuration
# ============================================================================

CARGO := cargo
PMAT := pmat
RUSTFLAGS := -D warnings

# Quality thresholds (enforced)
MIN_COVERAGE := 95
MIN_MUTATION_SCORE := 80
MIN_TDG_GRADE := A-
MAX_COMPLEXITY := 15
MAX_DEAD_CODE_PCT := 5.0
MAX_UNWRAP_CALLS := 50

# Paths
SRC_DIR := src
TEST_DIR := tests
DOCS_DIR := docs
PMAT_DIR := .pmat
BASELINE_FILE := $(PMAT_DIR)/tdg-baseline.json

# ============================================================================
# Help
# ============================================================================

.PHONY: help
help: ## Show this help
	@echo "Trueno-Viz Development Commands"
	@echo "================================"
	@echo ""
	@echo "Build & Test:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(build|test|check)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
	@echo ""
	@echo "Quality Gates (PMAT):"
	@grep -E '^pmat-[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
	@echo ""
	@echo "Development:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -vE '(build|test|check|pmat-)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

# ============================================================================
# Build Commands
# ============================================================================

.PHONY: build
build: ## Build the project (debug)
	$(CARGO) build

.PHONY: build-release
build-release: ## Build the project (release, optimized)
	RUSTFLAGS="$(RUSTFLAGS)" $(CARGO) build --release

.PHONY: build-wasm
build-wasm: ## Build for WebAssembly target (wasm-pack)
	@command -v wasm-pack >/dev/null 2>&1 || { echo "Installing wasm-pack..."; cargo install wasm-pack; }
	wasm-pack build --target web --features wasm --out-dir pkg
	@cp pkg/README.md pkg/README.md.bak 2>/dev/null || true
	@if [ -f pkg/package.json.template ]; then \
		jq -s '.[0] * .[1]' pkg/package.json pkg/package.json.template > pkg/package.json.tmp && \
		mv pkg/package.json.tmp pkg/package.json; \
	fi
	@mv pkg/README.md.bak pkg/README.md 2>/dev/null || true
	@echo "WASM package built in pkg/"

.PHONY: build-wasm-nodejs
build-wasm-nodejs: ## Build WASM for Node.js
	wasm-pack build --target nodejs --features wasm --out-dir pkg-node

.PHONY: build-wasm-bundler
build-wasm-bundler: ## Build WASM for bundlers (webpack, etc)
	wasm-pack build --target bundler --features wasm --out-dir pkg-bundler

# ============================================================================
# Test Commands
# ============================================================================

# Native features (excludes wasm which requires wasm-bindgen)
NATIVE_FEATURES := monitor,terminal,svg,parallel

.PHONY: test
test: ## Run all tests (native features)
	PROPTEST_CASES=100 $(CARGO) test --features $(NATIVE_FEATURES)

.PHONY: test-fast
test-fast: ## Run fast tests (<5s)
	@echo "⚡ Running fast tests..."
	@$(CARGO) test --lib --features $(NATIVE_FEATURES)

.PHONY: test-doc
test-doc: ## Run documentation tests
	$(CARGO) test --doc

.PHONY: test-coverage
test-coverage: ## Run tests with tarpaulin coverage
	$(CARGO) tarpaulin --out Html --output-dir target/coverage \
		--fail-under $(MIN_COVERAGE) \
		--skip-clean \
		--timeout 300

# Coverage exclusions: platform-specific code that can't be tested on current OS
# - app.rs: UI event loop (requires terminal)
# - gpu_amd.rs: AMD GPU hardware required
# - gpu_apple.rs: Apple Silicon required
# - battery.rs: Hardware-specific
# - battery_sensors_simd.rs: Hardware-specific SIMD sensors
# - kernels.rs: SIMD intrinsics require specific CPU features (AVX2/NEON)
# - /home/*/src/*: Exclude local dependencies (probar, etc.) - only cover workspace code
COVERAGE_EXCLUDE := --ignore-filename-regex='monitor/app\.rs|monitor/collectors/gpu_amd\.rs|monitor/collectors/gpu_apple\.rs|monitor/collectors/battery\.rs|monitor/collectors/battery_sensors_simd\.rs|monitor/simd/kernels\.rs|wasm\.rs|^/home/.*/src/(?!trueno-viz)'

.PHONY: coverage
coverage: ## Generate HTML coverage report with llvm-cov (fast: ~30s warm)
	@echo "📊 Running FAST coverage analysis (lib tests only)..."
	@which cargo-llvm-cov > /dev/null 2>&1 || (echo "📦 Installing cargo-llvm-cov..." && cargo install cargo-llvm-cov --locked)
	@mkdir -p target/coverage
	@echo "🧪 Running lib tests with instrumentation..."
	@cargo llvm-cov --no-report test --lib --features $(NATIVE_FEATURES)
	@echo "📊 Generating coverage reports..."
	@cargo llvm-cov report $(COVERAGE_EXCLUDE) --html --output-dir target/coverage/html
	@cargo llvm-cov report $(COVERAGE_EXCLUDE) --lcov --output-path target/coverage/lcov.info
	@echo ""
	@echo "📊 Coverage Summary (excluding platform-specific code):"
	@echo "======================================================="
	@cargo llvm-cov report $(COVERAGE_EXCLUDE) --summary-only
	@echo ""
	@echo "💡 COVERAGE INSIGHTS:"
	@echo "- HTML report: target/coverage/html/index.html"
	@echo "- LCOV file: target/coverage/lcov.info"
	@echo "- Open HTML: xdg-open target/coverage/html/index.html"
	@echo ""

.PHONY: coverage-full
coverage-full: ## Generate FULL coverage report (all tests, ~3min)
	@echo "📊 Running FULL coverage analysis..."
	@which cargo-llvm-cov > /dev/null 2>&1 || (echo "📦 Installing cargo-llvm-cov..." && cargo install cargo-llvm-cov --locked)
	@mkdir -p target/coverage
	@echo "🧪 Running ALL tests with instrumentation..."
	@PROPTEST_CASES=10 cargo llvm-cov test --no-report --lib --features $(NATIVE_FEATURES) || true
	@echo "📊 Generating coverage reports..."
	@cargo llvm-cov report $(COVERAGE_EXCLUDE) --html --output-dir target/coverage/html
	@cargo llvm-cov report $(COVERAGE_EXCLUDE) --summary-only
	@echo ""

.PHONY: coverage-summary
coverage-summary: ## Show coverage summary (run after 'make coverage')
	@cargo llvm-cov report $(COVERAGE_EXCLUDE) --summary-only 2>/dev/null || echo "Run 'make coverage' first"

.PHONY: coverage-open
coverage-open: ## Open coverage HTML report in browser
	@xdg-open target/coverage/html/index.html 2>/dev/null || open target/coverage/html/index.html 2>/dev/null || echo "Open target/coverage/html/index.html manually"

.PHONY: coverage-check
coverage-check: ## Enforce 95% coverage threshold (BLOCKS on failure, excludes wasm.rs)
	@echo "🔒 Enforcing 95% coverage threshold (wasm.rs excluded)..."
	@which cargo-llvm-cov > /dev/null 2>&1 || (echo "📦 Installing cargo-llvm-cov..." && cargo install cargo-llvm-cov --locked)
	@PROPTEST_CASES=10 cargo llvm-cov test --no-report --lib --features $(NATIVE_FEATURES) > /dev/null 2>&1
	@./scripts/check-coverage.sh 95

# ============================================================================
# Probar Testing (GUI/UX Coverage)
# ============================================================================

.PHONY: probar
probar: ## Run all probar tests (GUI/pixel/UX coverage)
	@echo "🎯 Running Probar GUI/UX coverage tests..."
	cd crates/ttop && CARGO_TARGET_DIR=./target cargo test --release --test probar_full_test -- --nocapture

.PHONY: probar-full
probar-full: ## Run full probar test suite with all flavors
	@echo "🎯 Running full Probar test suite..."
	cd crates/ttop && CARGO_TARGET_DIR=./target cargo test --release --test '*' -- --nocapture

.PHONY: test-ttop
test-ttop: ## Run ttop-specific tests
	@echo "🖥️  Running ttop tests..."
	cd crates/ttop && CARGO_TARGET_DIR=./target cargo test --release -- --nocapture

# ============================================================================
# Lint & Format
# ============================================================================

.PHONY: fmt
fmt: ## Format code
	$(CARGO) fmt

.PHONY: fmt-check
fmt-check: ## Check code formatting
	$(CARGO) fmt -- --check

.PHONY: clippy
clippy: ## Run Clippy lints
	$(CARGO) clippy --all-targets --all-features -- -D warnings

.PHONY: lint
lint: fmt-check clippy ## Run all lints

# ============================================================================
# PMAT Quality Gates
# ============================================================================

.PHONY: pmat-init
pmat-init: ## Initialize PMAT configuration
	@mkdir -p $(PMAT_DIR)
	@echo "Initializing PMAT quality gates..."
	$(PMAT) diagnose --full || true

.PHONY: pmat-baseline
pmat-baseline: pmat-init ## Create TDG baseline (run once, then on major releases)
	$(PMAT) tdg baseline create --output $(BASELINE_FILE) --path .
	@echo "Baseline created at $(BASELINE_FILE)"

.PHONY: pmat-quality
pmat-quality: ## Run O(1) quality gates (<30ms)
	$(PMAT) quality-gate --fail-on-violation

.PHONY: pmat-score
pmat-score: ## Full Rust project quality assessment (211 pts scale)
	$(PMAT) rust-project-score --full

.PHONY: pmat-score-json
pmat-score-json: ## Rust project score as JSON (for CI)
	$(PMAT) rust-project-score --full --format json --output $(PMAT_DIR)/score.json
	@cat $(PMAT_DIR)/score.json | jq '.total_earned, .total_possible'

.PHONY: pmat-tdg
pmat-tdg: ## Analyze technical debt (A+ to F grade)
	$(PMAT) analyze satd --path . --format table

.PHONY: pmat-tdg-check
pmat-tdg-check: ## Check for TDG regression against baseline
	@if [ -f $(BASELINE_FILE) ]; then \
		$(PMAT) tdg check-regression \
			--baseline $(BASELINE_FILE) \
			--max-score-drop 5.0 \
			--fail-on-regression; \
	else \
		echo "No baseline found. Run 'make pmat-baseline' first."; \
		exit 1; \
	fi

.PHONY: pmat-complexity
pmat-complexity: ## Analyze cyclomatic/cognitive complexity hotspots
	$(PMAT) analyze complexity --top-files 10

.PHONY: pmat-dead-code
pmat-dead-code: ## Detect dead/unreachable code
	$(PMAT) analyze dead-code --max-percentage $(MAX_DEAD_CODE_PCT)

.PHONY: pmat-defects
pmat-defects: ## Scan for known defect patterns (unwrap, expect, etc.)
	$(PMAT) analyze defects --path .

.PHONY: pmat-mutate
pmat-mutate: ## Run mutation testing (test quality validation)
	$(PMAT) mutate --target $(SRC_DIR)/ --threshold $(MIN_MUTATION_SCORE)

.PHONY: pmat-mutate-failures
pmat-mutate-failures: ## Show only surviving mutants (test gaps)
	$(PMAT) mutate --target $(SRC_DIR)/ --failures-only

.PHONY: pmat-context
pmat-context: ## Generate AI-ready deep context for Claude
	$(PMAT) context --output deep_context.md --format llm-optimized

.PHONY: pmat-dag
pmat-dag: ## Generate dependency graph (Mermaid)
	$(PMAT) analyze dag --format mermaid --output $(DOCS_DIR)/dependency-graph.md

.PHONY: pmat-churn
pmat-churn: ## Analyze code churn (change frequency)
	$(PMAT) analyze churn --days 30

.PHONY: pmat-validate-docs
pmat-validate-docs: ## Validate README accuracy (hallucination detection)
	$(PMAT) validate-readme --targets README.md CONTRIBUTING.md

.PHONY: pmat-repo-health
pmat-repo-health: ## Quick repository health score (0-110)
	$(PMAT) repo-score

.PHONY: pmat-metrics
pmat-metrics: ## Show quality metrics trend
	$(PMAT) show-metrics --trend

# ============================================================================
# Composite Quality Commands
# ============================================================================

.PHONY: check
check: lint test ## Basic CI check (lint + test)

.PHONY: quality
quality: lint pmat-quality pmat-tdg-check test ## Full quality gate (pre-commit)
	@echo "All quality gates passed!"

.PHONY: quality-full
quality-full: quality pmat-mutate pmat-dead-code pmat-complexity ## Comprehensive quality (pre-merge)
	@echo "Comprehensive quality validation complete!"

.PHONY: self-check
self-check: ## Author's pre-submission checklist
	@echo "=== Self-Check: Pre-Submission Validation ==="
	@echo ""
	@echo "1. Formatting..."
	@$(CARGO) fmt -- --check || (echo "FAIL: Run 'make fmt'" && exit 1)
	@echo "   PASS"
	@echo ""
	@echo "2. Clippy lints..."
	@$(CARGO) clippy --all-targets --all-features -- -D warnings || (echo "FAIL: Fix Clippy warnings" && exit 1)
	@echo "   PASS"
	@echo ""
	@echo "3. Tests..."
	@$(CARGO) test --quiet || (echo "FAIL: Tests failed" && exit 1)
	@echo "   PASS"
	@echo ""
	@echo "4. Quality gates..."
	@$(PMAT) quality-gate --fail-on-violation || (echo "FAIL: Quality gate violation" && exit 1)
	@echo "   PASS"
	@echo ""
	@echo "5. Technical debt..."
	@$(PMAT) analyze satd --path . --format table
	@echo ""
	@echo "=== Self-Check Complete: Ready for Review ==="

.PHONY: verify-reproducibility
verify-reproducibility: ## Reviewer's reproducibility audit
	@echo "=== Reproducibility Audit ==="
	@echo ""
	@echo "1. Clean build..."
	@$(CARGO) clean
	@$(CARGO) build || (echo "FAIL: Build failed" && exit 1)
	@echo "   PASS"
	@echo ""
	@echo "2. Test run 1/3..."
	@$(CARGO) test --quiet || (echo "FAIL: Tests failed (run 1)" && exit 1)
	@echo "   PASS"
	@echo ""
	@echo "3. Test run 2/3..."
	@$(CARGO) test --quiet || (echo "FAIL: Tests failed (run 2)" && exit 1)
	@echo "   PASS"
	@echo ""
	@echo "4. Test run 3/3..."
	@$(CARGO) test --quiet || (echo "FAIL: Tests failed (run 3)" && exit 1)
	@echo "   PASS"
	@echo ""
	@echo "=== Reproducibility Audit: PASS ==="
	@echo "Record: 'Reproduced locally on $$(date +%Y-%m-%d)'"

# ============================================================================
# CI/CD Pipeline Targets
# ============================================================================

.PHONY: ci
ci: lint test-coverage pmat-quality pmat-score-json ## Full CI pipeline
	@echo "CI pipeline complete"

.PHONY: ci-quick
ci-quick: lint test-fast pmat-quality ## Quick CI (PRs)
	@echo "Quick CI complete"

.PHONY: ci-nightly
ci-nightly: quality-full pmat-mutate test-coverage ## Nightly comprehensive CI
	@echo "Nightly CI complete"

# ============================================================================
# Development Utilities
# ============================================================================

.PHONY: dev
dev: ## Start development (watch mode)
	$(CARGO) watch -x check -x test

.PHONY: doc
doc: ## Generate documentation
	$(CARGO) doc --no-deps --open

.PHONY: readme
readme: ## Regenerate README.md from template (reproducible docs)
	@echo "Regenerating README.md..."
	@$(CARGO) build --example readme_demo --quiet
	@./scripts/generate-readme.sh > README.md
	@echo "README.md updated"

.PHONY: bench
bench: ## Run benchmarks
	$(CARGO) bench

.PHONY: clean
clean: ## Clean build artifacts
	$(CARGO) clean
	rm -rf target/coverage
	rm -f deep_context.md

.PHONY: clean-all
clean-all: clean ## Clean everything including PMAT cache
	rm -rf $(PMAT_DIR)
	$(PMAT) cache clear || true

# ============================================================================
# Release
# ============================================================================

.PHONY: release-check
release-check: quality-full pmat-score ## Pre-release validation
	@echo ""
	@echo "=== Release Readiness Check ==="
	@$(PMAT) rust-project-score --full
	@echo ""
	@echo "Review the score above. Target: 180+/211 (85%+)"
	@echo "If ready, run: make release-tag VERSION=x.y.z"

.PHONY: release-tag
release-tag: ## Tag a release (requires VERSION=x.y.z)
ifndef VERSION
	$(error VERSION is required. Usage: make release-tag VERSION=0.1.0)
endif
	@echo "Tagging release v$(VERSION)..."
	git tag -a "v$(VERSION)" -m "Release v$(VERSION)"
	@echo "Run 'git push origin v$(VERSION)' to publish"

# ============================================================================
# MCP Server (for Claude Code integration)
# ============================================================================

.PHONY: mcp-server
mcp-server: ## Start PMAT MCP server for Claude Code
	$(PMAT) mcp

.PHONY: mcp-demo
mcp-demo: ## Interactive PMAT demo (web-based)
	$(PMAT) demo --url .

# ============================================================================
# Hooks Management
# ============================================================================

.PHONY: hooks-install
hooks-install: ## Install Git pre-commit hooks (95% coverage enforced)
	@echo "📦 Installing pre-commit hooks..."
	@cp .githooks/pre-commit .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "✅ Pre-commit hook installed"
	@echo "   Enforces: fmt, clippy, tests, 95% coverage"
	@echo "   Bypass (emergency): git commit --no-verify"

.PHONY: hooks-uninstall
hooks-uninstall: ## Uninstall Git pre-commit hooks
	@rm -f .git/hooks/pre-commit
	@echo "✅ Pre-commit hook removed"

.PHONY: hooks-status
hooks-status: ## Check hook installation status
	@if [ -f .git/hooks/pre-commit ]; then \
		echo "✅ Pre-commit hook installed"; \
		echo "   Location: .git/hooks/pre-commit"; \
	else \
		echo "❌ No pre-commit hook installed"; \
		echo "   Run: make hooks-install"; \
	fi

# ============================================================================
# Roadmap Integration
# ============================================================================

.PHONY: roadmap
roadmap: ## Show project roadmap with quality gates
	$(PMAT) roadmap status || echo "No roadmap configured. Use 'pmat roadmap create'"

.PHONY: roadmap-create
roadmap-create: ## Create initial roadmap
	$(PMAT) roadmap create --output $(PMAT_DIR)/roadmap.toml

# ============================================================================
# WASM Build (GPU Demo)
# ============================================================================

WASM_PKG_DIR := wasm-pkg
WASM_OUT_DIR := $(WASM_PKG_DIR)/pkg

.PHONY: wasm
wasm: wasm-build ## Build WASM package (alias)

.PHONY: wasm-build
wasm-build: ## Build WASM package with wasm-pack
	@echo "Building WASM package..."
	@command -v wasm-pack >/dev/null 2>&1 || { \
		echo "Installing wasm-pack..."; \
		cargo install wasm-pack; \
	}
	cd $(WASM_PKG_DIR) && wasm-pack build --target web --release
	@echo ""
	@echo "WASM Build Complete"
	@echo "   Package: $(WASM_OUT_DIR)/"
	@ls -lh $(WASM_OUT_DIR)/*.wasm 2>/dev/null || true

.PHONY: wasm-build-simd
wasm-build-simd: ## Build WASM with SIMD128 + WebGPU enabled
	@echo "Building WASM with SIMD128 + WebGPU..."
	cd $(WASM_PKG_DIR) && RUSTFLAGS="-C target-feature=+simd128 --cfg=web_sys_unstable_apis" wasm-pack build --target web --release --features webgpu
	@echo "SIMD128 + WebGPU WASM built"

# Serve WASM demo (override port: make wasm-serve WASM_PORT=8000)
WASM_PORT ?= 9876
.PHONY: wasm-serve
wasm-serve: wasm-build-simd ## Build and serve WASM demo with SIMD (port 9876)
	@echo "Starting demo server at http://localhost:$(WASM_PORT)/"
	@echo "Press Ctrl+C to stop"
	@if command -v ruchy >/dev/null 2>&1; then \
		echo "Using ruchy (fast)"; \
		cd $(WASM_PKG_DIR) && ruchy serve . --port $(WASM_PORT); \
	else \
		echo "Using Python (install ruchy for faster: cargo install ruchy)"; \
		cd $(WASM_PKG_DIR) && python3 -m http.server $(WASM_PORT); \
	fi

.PHONY: wasm-clean
wasm-clean: ## Clean WASM build artifacts
	rm -rf $(WASM_OUT_DIR)
	rm -rf $(WASM_PKG_DIR)/target

.PHONY: wasm-check
wasm-check: ## Check WASM package compiles
	cd $(WASM_PKG_DIR) && cargo check --target wasm32-unknown-unknown

# ============================================================================
# Phony declarations for safety
# ============================================================================

.PHONY: all
all: build test quality
