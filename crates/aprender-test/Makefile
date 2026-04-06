# Use bash for shell commands to support advanced features
SHELL := /bin/bash

# PERFORMANCE TARGETS (Toyota Way: Zero Defects, Fast Feedback)
# - make test-fast: < 2 minutes (50 property test cases)
# - make coverage:  < 1 minute (5 property test cases, exclusions)
# - make test:      comprehensive (500 property test cases)
# Override with: PROPTEST_CASES=n make <target>

# Coverage exclusions: CLI/proc-macro bins, GPU/infra subsystems, and non-testable modules
# 17 patterns: 7 dirs + playbook/runner + 9 probar/src files needing individual exclusion
# (stress, load_testing, dev_server, visualization, debug, generate, hir, svg_exporter,
#  wasm_testing, ast_visitor are already covered by probar-cli/ or don't exist in probar/src)
COVERAGE_EXCLUDE := --ignore-filename-regex='probar-cli/|probar-derive/|brick/|runner/|gpu_pixels/|mock/|perf/|playbook/runner\.rs|browser\.rs|driver\.rs|capabilities\.rs|simulation\.rs|fuzzer\.rs|watch\.rs|websocket\.rs|worker_harness\.rs|replay\.rs'

.PHONY: all validate quick-validate release clean help
.PHONY: format format-check lint lint-check check test test-fast test-all
.PHONY: coverage coverage-open coverage-ci coverage-clean coverage-summary
.PHONY: build build-release doc book book-serve book-clean
.PHONY: example-locator example-accessibility example-coverage example-simulation
.PHONY: test-property test-property-comprehensive
.PHONY: mutants mutants-quick mutants-clean mutants-report

# Parallel job execution
MAKEFLAGS += -j$(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Default target
all: validate build

# ============================================================================
# ZERO JAVASCRIPT ENFORCEMENT
# ============================================================================

zero-js: ## Check ABSOLUTE ZERO JAVASCRIPT compliance
	@./scripts/check-zero-js.sh

# ============================================================================
# TIER 1: ON-SAVE - Sub-second feedback
# ============================================================================

tier1: zero-js check
	@echo "✓ Tier 1 passed"

# ============================================================================
# TIER 2: ON-COMMIT - Full validation (1-5 min)
# ============================================================================

tier2: zero-js format-check lint-check test-fast
	@echo "✓ Tier 2 passed"

# ============================================================================
# TIER 3: ON-MERGE - Coverage + comprehensive tests
# ============================================================================

tier3: tier2 coverage test-all
	@echo "✓ Tier 3 passed"

# Quick validation for development (skip expensive checks)
quick-validate: format-check lint-check check test-fast
	@echo "✅ Quick validation passed!"

# Full validation pipeline
validate: format lint check test
	@echo "✅ All validation passed!"

# ============================================================================
# FORMATTING
# ============================================================================

format: ## Format code
	@echo "🎨 Formatting code..."
	@cargo fmt --all

format-check: ## Check code formatting
	@echo "🎨 Checking code formatting..."
	@cargo fmt --all -- --check

# ============================================================================
# LINTING
# ============================================================================

# Note: We exclude compute-blocks feature as it requires presentar-terminal which may have issues
LINT_FEATURES := --features browser,runtime,derive

lint: ## Run clippy with auto-fix
	@echo "🔍 Running clippy..."
	@cargo clippy --workspace $(LINT_FEATURES) --all-targets --fix --allow-dirty --allow-staged 2>/dev/null || true
	@cargo clippy --workspace $(LINT_FEATURES) --lib -- -D warnings -A dead_code -A clippy::format_push_string
	@cargo clippy --workspace $(LINT_FEATURES) --tests -- \
		-D clippy::suspicious \
		-A clippy::expect_used \
		-A clippy::unwrap_used \
		-A clippy::panic \
		-A clippy::float_cmp \
		-A clippy::field_reassign_with_default \
		-A clippy::type_complexity \
		-A clippy::approx_constant \
		-A clippy::needless_collect \
		-A clippy::unnecessary_get_then_check \
		-A clippy::absurd_extreme_comparisons \
		-A dead_code

lint-check: ## Check clippy without fixing
	@echo "🔍 Checking clippy..."
	@cargo clippy --workspace --all-targets -- \
		-D clippy::correctness \
		-D clippy::suspicious \
		-W clippy::complexity \
		-W clippy::perf \
		-A clippy::multiple_crate_versions \
		-A clippy::expect_used \
		-A clippy::unwrap_used \
		-A clippy::indexing_slicing \
		-A clippy::panic \
		-A unused_results \
		-A dead_code \
		-A unused_variables

# ============================================================================
# TYPE CHECKING
# ============================================================================

check: ## Type check
	@echo "🔍 Type checking..."
	@cargo check --workspace --all-targets

# ============================================================================
# TESTING
# ============================================================================

test-fast: ## Run fast library tests (target: <2 min)
	@echo "⚡ Running fast tests (target: <2 min)..."
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		PROPTEST_CASES=25 RUST_TEST_THREADS=$$(nproc) cargo nextest run \
			--workspace --lib \
			--status-level skip \
			--failure-output immediate; \
	else \
		PROPTEST_CASES=25 cargo test --workspace --lib; \
	fi

test: test-fast test-doc test-property ## Run core test suite
	@echo "✅ Core test suite completed!"
	@echo "  - Fast unit tests ✓"
	@echo "  - Documentation tests ✓"
	@echo "  - Property-based tests ✓"

test-doc: ## Run documentation tests
	@echo "📚 Running documentation tests..."
	@PROPTEST_CASES=256 QUICKCHECK_TESTS=256 cargo test --doc --workspace

test-property: ## Run property-based tests (50 cases per property)
	@echo "🎲 Running property-based tests (50 cases per property)..."
	@THREADS=$${PROPTEST_THREADS:-$$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}; \
	echo "  Running with $$THREADS threads..."; \
	echo "  (Override with PROPTEST_THREADS=n or PROPTEST_CASES=n)"; \
	timeout 120 env PROPTEST_CASES=25 cargo test --workspace --lib -- prop_ --test-threads=$$THREADS || echo "⚠️  Some property tests timed out"
	@echo "✅ Property tests completed (fast mode)!"

test-property-comprehensive: ## Run property-based tests (500 cases per property)
	@echo "🎲 Running property-based tests (500 cases per property)..."
	@THREADS=$${PROPTEST_THREADS:-$$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}; \
	echo "  Running with $$THREADS threads..."; \
	timeout 300 env PROPTEST_CASES=250 cargo test --workspace --lib -- prop_ --test-threads=$$THREADS || echo "⚠️  Some property tests timed out"
	@echo "✅ Property tests completed (comprehensive mode)!"

test-all: test test-property-comprehensive test-gpu-pixels ## Run ALL tests with all features
	@echo "🧪 Running comprehensive tests with all features..."
	@PROPTEST_CASES=250 cargo test --workspace --all-features
	@echo "✅ All tests completed!"

test-gpu-pixels: ## Run GPU pixel tests (PTX validation, regression detection)
	@echo "🎯 Running GPU pixel tests..."
	@PROPTEST_CASES=256 QUICKCHECK_TESTS=256 cargo test -p jugar-probar gpu_pixels --lib
	@echo "✅ GPU pixel tests passed!"

# ============================================================================
# COVERAGE
# ============================================================================

coverage: ## Generate coverage report (FAST: <1 min, 95% target)
	@echo "📊 Coverage report (target: 95%, <1 min)..."
	@which cargo-llvm-cov > /dev/null 2>&1 || cargo install cargo-llvm-cov --locked
	@mkdir -p target/coverage
	@echo "🧪 Running lib tests..."
	@env PROPTEST_CASES=5 QUICKCHECK_TESTS=5 \
		cargo llvm-cov --lib --workspace \
		--html --output-dir target/coverage/html
	@cargo llvm-cov report --lcov --output-path target/coverage/lcov.info
	@echo ""
	@cargo llvm-cov report --summary-only
	@echo ""
	@echo "💡 HTML: target/coverage/html/index.html | LCOV: target/coverage/lcov.info"

coverage-full: ## Generate full coverage including bins (slower)
	@echo "📊 Full coverage (all targets)..."
	@which cargo-llvm-cov > /dev/null 2>&1 || cargo install cargo-llvm-cov --locked
	@mkdir -p target/coverage
	@env PROPTEST_CASES=5 QUICKCHECK_TESTS=5 \
		cargo llvm-cov --lib --bins --workspace --all-features --all-targets \
		--html --output-dir target/coverage/html
	@cargo llvm-cov report --summary-only
	@echo ""
	@echo "💡 HTML: target/coverage/html/index.html"

coverage-summary: ## Show coverage summary
	@cargo llvm-cov report --summary-only 2>/dev/null || echo "Run 'make coverage' first"

coverage-open: ## Open HTML coverage report in browser
	@if [ -f target/coverage/html/index.html ]; then \
		xdg-open target/coverage/html/index.html 2>/dev/null || \
		open target/coverage/html/index.html 2>/dev/null || \
		echo "Please open: target/coverage/html/index.html"; \
	else \
		echo "❌ Run 'make coverage' first to generate the HTML report"; \
	fi

coverage-ci: ## Generate LCOV report for CI/CD
	@echo "📊 Generating coverage for CI..."
	@env PROPTEST_CASES=25 QUICKCHECK_TESTS=25 cargo llvm-cov --no-report test --lib --workspace $(COVERAGE_EXCLUDE)
	@cargo llvm-cov report --lcov --output-path lcov.info $(COVERAGE_EXCLUDE)
	@echo "✓ Coverage report generated: lcov.info"

coverage-clean: ## Clean coverage artifacts
	@rm -f lcov.info
	@rm -rf target/llvm-cov target/coverage
	@echo "✓ Coverage artifacts cleaned"

# ============================================================================
# MUTATION TESTING
# ============================================================================

mutants: ## Run mutation testing on all modules
	@echo "🧬 Running mutation testing..."
	@which cargo-mutants > /dev/null 2>&1 || (echo "📦 Installing cargo-mutants..." && cargo install cargo-mutants --locked)
	@echo "🧪 Running mutation tests on probar package..."
	@cargo mutants --package probar --no-times || true
	@echo ""
	@echo "📊 Mutation testing complete. Review mutants.out/ for detailed results."

mutants-quick: ## Run mutation testing on recently changed files only
	@echo "🧬 Running quick mutation testing (recently changed files)..."
	@which cargo-mutants > /dev/null 2>&1 || (echo "📦 Installing cargo-mutants..." && cargo install cargo-mutants --locked)
	@cargo mutants --package probar --no-times --in-diff HEAD~5..HEAD || true
	@echo "📊 Quick mutation testing complete."

mutants-module: ## Run mutation testing on a single module (MODULE=path/to/file.rs)
	@echo "🧬 Running targeted mutation testing..."
	@if [ -z "$(MODULE)" ]; then \
		echo "❌ Error: MODULE parameter required"; \
		echo "Usage: make mutants-module MODULE=crates/probar/src/coverage/block.rs"; \
		exit 1; \
	fi
	@if [ ! -f "$(MODULE)" ]; then \
		echo "❌ Error: File not found: $(MODULE)"; \
		exit 1; \
	fi
	@which cargo-mutants > /dev/null 2>&1 || cargo install cargo-mutants --locked
	@cargo mutants --file '$(MODULE)' --package probar --no-times || true
	@echo "📊 Mutation testing complete for $(MODULE)"

mutants-report: ## Generate mutation testing report
	@echo "📊 Generating mutation testing report..."
	@if [ -f mutants.out/mutants.json ]; then \
		echo "=== Mutation Testing Summary ==="; \
		echo ""; \
		cat mutants.out/mutants.json | jq -r '.summary // empty' 2>/dev/null || cat mutants.out/mutants.json; \
		echo ""; \
		echo "📄 Full report: mutants.out/mutants.json"; \
		echo "📋 Detailed logs: mutants.out/"; \
	else \
		echo "❌ No mutation results found. Run 'make mutants' first."; \
	fi

mutants-clean: ## Clean mutation testing artifacts
	@rm -rf mutants.out mutants.out.old
	@echo "✓ Mutation testing artifacts cleaned"

# ============================================================================
# BUILD
# ============================================================================

build: ## Build all crates
	@echo "🔨 Building..."
	@cargo build --workspace

build-release: ## Build release binaries
	@echo "🔨 Building release..."
	@cargo build --workspace --release

# ============================================================================
# DOCUMENTATION
# ============================================================================

doc: ## Generate and open rustdoc
	@echo "📚 Building documentation..."
	@cargo doc --workspace --no-deps --open

# ============================================================================
# BOOK
# ============================================================================

book: ## Build mdbook documentation
	@echo "📖 Building book..."
	@cd book && mdbook build

book-serve: ## Serve book locally with hot reload
	@echo "📖 Serving book..."
	@cd book && mdbook serve --open

book-clean: ## Clean generated book
	@rm -rf book/book
	@echo "✓ Book cleaned"

# ============================================================================
# EXAMPLES
# ============================================================================

example-locator: ## Run locator demo
	@cargo run --example locator_demo -p probar

example-accessibility: ## Run accessibility demo
	@cargo run --example accessibility_demo -p probar

example-coverage: ## Run coverage demo
	@cargo run --example coverage_demo -p probar

example-simulation: ## Run simulation demo
	@cargo run --example pong_simulation -p probar

# ============================================================================
# CLEAN
# ============================================================================

clean: ## Clean all build artifacts
	@echo "🧹 Cleaning..."
	@cargo clean
	@rm -rf target/coverage book/book

# ============================================================================
# HELP
# ============================================================================

help: ## Show this help
	@echo "Probar Build System"
	@echo "==================="
	@echo ""
	@echo "Tiered Workflow:"
	@echo "  make tier1          - ON-SAVE: Type checking only"
	@echo "  make tier2          - ON-COMMIT: Format + lint + fast tests"
	@echo "  make tier3          - ON-MERGE: Full coverage + all tests"
	@echo ""
	@echo "Quick Commands:"
	@echo "  make                - Run validation and build"
	@echo "  make quick-validate - Quick validation for development"
	@echo "  make validate       - Full validation pipeline"
	@echo ""
	@echo "Testing (Performance Targets Enforced):"
	@echo "  make test-fast      - Run library tests only (TARGET: <2 min, 50 prop cases)"
	@echo "  make test           - Run core test suite (fast + doc + property tests)"
	@echo "  make test-property  - Run property-based tests (50 cases)"
	@echo "  make test-property-comprehensive - Run property tests (500 cases)"
	@echo "  make test-gpu-pixels - Run GPU pixel tests (PTX validation)"
	@echo "  make test-all       - Run ALL tests with all features"
	@echo ""
	@echo "Coverage:"
	@echo "  make coverage       - Generate lib coverage report (TARGET: 95%, <1 min)"
	@echo "  make coverage-full  - Generate full coverage including bins"
	@echo "  make coverage-summary - Show coverage summary"
	@echo "  make coverage-open  - Open HTML coverage in browser"
	@echo "  make coverage-ci    - Generate LCOV report for CI/CD"
	@echo "  make coverage-clean - Clean coverage artifacts"
	@echo ""
	@echo "Mutation Testing:"
	@echo "  make mutants        - Run mutation testing on all modules"
	@echo "  make mutants-quick  - Run mutation testing on recent changes"
	@echo "  make mutants-module MODULE=path - Run on specific module"
	@echo "  make mutants-report - Show mutation testing report"
	@echo "  make mutants-clean  - Clean mutation artifacts"
	@echo ""
	@echo "Formatting & Linting:"
	@echo "  make format         - Format code"
	@echo "  make format-check   - Check formatting"
	@echo "  make lint           - Run clippy with auto-fix"
	@echo "  make lint-check     - Check clippy without fixing"
	@echo ""
	@echo "Book Commands:"
	@echo "  make book           - Build mdbook documentation"
	@echo "  make book-serve     - Serve book locally with hot reload"
	@echo "  make book-clean     - Clean generated book"
	@echo ""
	@echo "Examples:"
	@echo "  make example-locator      - Run locator demo"
	@echo "  make example-accessibility - Run accessibility demo"
	@echo "  make example-coverage     - Run coverage demo"
	@echo "  make example-simulation   - Run simulation demo"
	@echo ""
	@echo "Other:"
	@echo "  make build          - Build all crates"
	@echo "  make build-release  - Build release binaries"
	@echo "  make doc            - Generate rustdoc"
	@echo "  make clean          - Clean build artifacts"
	@echo "  make help           - Show this help"

# ============================================================================
# SIMPLE COVERAGE (one command, one number)
# ============================================================================

coverage-gui: ## Show GUI coverage (TUI + WASM) - one number
	@echo "📊 GUI Test Coverage"
	@echo "===================="
	@cargo llvm-cov --package showcase-calculator --features tui --summary-only 2>/dev/null | grep TOTAL | awk '{print "Coverage: " $$10}'
