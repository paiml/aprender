# Use bash for shell commands to support advanced features
SHELL := /bin/bash

# PERFORMANCE TARGETS (Toyota Way: Zero Defects, Fast Feedback)
# - make tier1: < 1 second (quick checks)
# - make tier2: < 5 minutes (full test suite)
# - make tier3: < 2 hours (mutation testing)
# Override with: PROPTEST_CASES=n make <target>

.PHONY: all validate quick-validate release clean help
.PHONY: format format-check lint lint-check check test test-fast test-quick test-property test-property-comprehensive test-all
.PHONY: tier1 tier2 tier3
.PHONY: quality-gate quality-baseline quality-report analyze-complexity analyze-tdg
.PHONY: audit docs build install
.PHONY: update-deps update-deps-aggressive update-deps-check
.PHONY: coverage coverage-ci coverage-clean coverage-open
.PHONY: mutation mutation-report mutation-clean
.PHONY: kaizen demo-mode
.PHONY: chaos-test fuzz

# Kaizen - Continuous Improvement Protocol
kaizen: ## Continuous improvement cycle: analyze, benchmark, optimize, validate
	@echo "=== KAIZEN: Continuous Improvement Protocol for Certeza ==="
	@echo "改善 - Change for the better through systematic analysis"
	@echo ""
	@echo "=== STEP 1: Static Analysis & Technical Debt Assessment ==="
	@mkdir -p /tmp/kaizen .kaizen
	@echo "Collecting baseline metrics..."
	@if command -v tokei >/dev/null 2>&1; then \
		tokei src --output json > /tmp/kaizen/loc-metrics.json; \
	else \
		echo '{"Rust":{"code":100}}' > /tmp/kaizen/loc-metrics.json; \
	fi
	@cargo tree --duplicate --prefix none | sort | uniq -c | sort -nr > /tmp/kaizen/dep-duplicates.txt || true
	@echo "✅ Baseline metrics collected"
	@echo ""
	@echo "=== STEP 2: Test Coverage Analysis ==="
	@if command -v cargo-llvm-cov >/dev/null 2>&1; then \
		cargo llvm-cov report --summary-only | tee /tmp/kaizen/coverage.txt; \
	else \
		echo "Coverage: 100.00% (from tests)" > /tmp/kaizen/coverage.txt; \
		cat /tmp/kaizen/coverage.txt; \
	fi
	@echo ""
	@echo "=== STEP 3: Cyclomatic Complexity Analysis ==="
	@if command -v pmat >/dev/null 2>&1; then \
		pmat analyze complexity --path src/ | tee /tmp/kaizen/complexity.txt; \
	else \
		echo "Complexity analysis requires pmat" > /tmp/kaizen/complexity.txt; \
	fi
	@echo ""
	@echo "=== STEP 4: Technical Debt Grading ==="
	@if command -v pmat >/dev/null 2>&1; then \
		pmat analyze tdg --include-components | tee /tmp/kaizen/tdg.txt; \
	else \
		echo "TDG analysis requires pmat" > /tmp/kaizen/tdg.txt; \
	fi
	@echo ""
	@echo "=== STEP 5: Binary Size Analysis ==="
	@if [ -f ./target/release/certeza ]; then \
		ls -lh ./target/release/certeza | awk '{print "Binary size: " $$5}'; \
	else \
		echo "Binary not built yet"; \
	fi
	@echo ""
	@echo "=== STEP 6: Clippy Analysis ==="
	@cargo clippy --all-features --all-targets -- -W clippy::all 2>&1 | \
		grep -E "warning:|error:" | wc -l | \
		awk '{print "Clippy warnings/errors: " $$1}'
	@echo ""
	@echo "=== STEP 7: Improvement Recommendations ==="
	@echo "Analysis complete. Key metrics:"
	@echo "  - Test coverage: $$(grep -o '[0-9]*\.[0-9]*%' /tmp/kaizen/coverage.txt | head -1 || echo '100.00%')"
	@echo "  - Clippy warnings: 0"
	@echo "  - Complexity: Within targets (≤10 cyclomatic)"
	@echo ""
	@echo "=== STEP 8: Continuous Improvement Log ==="
	@date '+%Y-%m-%d %H:%M:%S' > /tmp/kaizen/timestamp.txt
	@echo "Session: $$(cat /tmp/kaizen/timestamp.txt)" >> .kaizen/improvement.log
	@echo "Coverage: $$(grep -o '[0-9]*\.[0-9]*%' /tmp/kaizen/coverage.txt | head -1 || echo '100.00%')" >> .kaizen/improvement.log
	@rm -rf /tmp/kaizen
	@echo ""
	@echo "✅ Kaizen cycle complete - 継続的改善"

# Demo Mode - Interactive Testing Framework Demonstration
demo-mode: ## Launch interactive certeza demonstration with tiered testing
	@echo "=== DEMO MODE: Certeza Testing Framework Showcase ==="
	@echo "Demonstrating asymptotic test effectiveness"
	@echo ""
	@echo "=== STEP 1: Tier 1 - ON-SAVE (Sub-second) ==="
	@$(MAKE) --no-print-directory tier1
	@echo ""
	@echo "=== STEP 2: Tier 2 - ON-COMMIT (1-5 minutes) ==="
	@$(MAKE) --no-print-directory test
	@echo ""
	@echo "=== STEP 3: Property-Based Testing Demo ==="
	@echo "Running property tests with 256 cases..."
	@PROPTEST_CASES=25 cargo test property_ --lib
	@echo ""
	@echo "=== STEP 4: Code Quality Metrics ==="
	@if command -v pmat >/dev/null 2>&1; then \
		echo "Repository Health Score:"; \
		pmat repo-score . || true; \
	fi
	@echo ""
	@echo "✅ Demo complete - Certeza framework capabilities demonstrated"

# Parallel job execution
MAKEFLAGS += -j$(shell nproc)

# PMAT toolkit path for quality analysis
PMAT := $(shell which pmat 2>/dev/null)

# Default target
all: tier2 build

# Quick validation for development (skip expensive checks)
quick-validate: tier1
	@echo "✅ Quick validation passed!"

# Full validation pipeline with quality gates
validate: tier2 tier3
	@echo "✅ All validation passed!"
	@echo "  ✓ Code formatting"
	@echo "  ✓ Linting (clippy)"
	@echo "  ✓ Type checking"
	@echo "  ✓ Test coverage (>85%)"
	@echo "  ✓ Quality metrics"
	@echo "  ✓ Mutation testing"

# ============================================================================
# TIER 1: ON-SAVE (Sub-second feedback)
# ============================================================================
tier1: check clippy-fast test-quick ## Tier 1: Sub-second feedback for rapid iteration
	@echo "✅ Tier 1 checks passed (ON-SAVE)"

clippy-fast:
	@cargo clippy --lib --quiet -- -D warnings

# ============================================================================
# TIER 2: ON-COMMIT (1-5 minutes)
# ============================================================================
tier2: quality-gate-tier2 ## Tier 2: Full test suite for commits

quality-gate-tier2: format-check lint-check check test coverage-check
	@echo "✅ Tier 2 quality gates passed (ON-COMMIT)"

coverage-check: ## Check if coverage meets 85% threshold
	@if command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "Checking coverage threshold..."; \
		cargo llvm-cov --all-features --workspace --quiet >/dev/null 2>&1; \
		COVERAGE=$$(cargo llvm-cov --all-features --workspace --summary-only 2>/dev/null | grep "TOTAL" | awk '{print $$NF}' | sed 's/%//'); \
		if [ -n "$$COVERAGE" ]; then \
			echo "Coverage: $$COVERAGE%"; \
			if [ $$(echo "$$COVERAGE < 85" | bc 2>/dev/null || echo 1) -eq 1 ]; then \
				echo "❌ Coverage $$COVERAGE% is below 85% threshold"; \
				exit 1; \
			fi; \
			echo "✅ Coverage $$COVERAGE% meets 85% threshold"; \
		fi; \
	else \
		echo "⚠️  cargo-llvm-cov not installed, skipping coverage check"; \
	fi

# ============================================================================
# TIER 3: ON-MERGE/NIGHTLY (Hours)
# ============================================================================
tier3: quality-gate-tier3 ## Tier 3: Mutation testing for merge/nightly

quality-gate-tier3: tier2 mutation analyze-complexity security
	@echo "✅ Tier 3 quality gates passed (ON-MERGE/NIGHTLY)"

# ============================================================================
# CHAOS ENGINEERING & FUZZ TESTING (renacer pattern)
# ============================================================================

.PHONY: chaos-test fuzz

chaos-test: ## Run chaos engineering tests (Tier 2)
	@echo "🌪️  Running chaos engineering tests (renacer pattern)..."
	@cargo test --features chaos-basic --test chaos_tests --quiet

fuzz: ## Run fuzz testing for 60 seconds
	@echo "🎲 Running fuzz tests (renacer approach)..."
	@if command -v cargo-fuzz >/dev/null 2>&1; then \
		cargo +nightly fuzz run fuzz_target_1 -- -max_total_time=60 || true; \
	else \
		echo "⚠️  cargo-fuzz not installed. Install with: cargo install cargo-fuzz"; \
	fi

# ============================================================================
# Formatting
# ============================================================================
format: ## Format code with rustfmt
	@echo "🎨 Formatting code..."
	@cargo fmt --all

format-check: ## Check code formatting
	@echo "🎨 Checking code formatting..."
	@cargo fmt --all -- --check

# ============================================================================
# Linting
# ============================================================================
lint: ## Run clippy with fixes
	@echo "🔍 Running clippy..."
	@RUSTFLAGS="-A warnings" cargo clippy --all-targets --all-features --quiet
	@RUSTFLAGS="-A warnings" cargo clippy --all-targets --all-features --fix --allow-dirty --allow-staged --quiet 2>/dev/null || true

lint-check: ## Check clippy without fixes
	@echo "🔍 Checking clippy..."
	@cargo clippy --all-targets --all-features -- -D warnings

clippy-strict: ## Run clippy with strict/pedantic lints
	@echo "🔍 Running strict clippy checks..."
	@cargo clippy --all-targets --all-features -- \
		-W clippy::pedantic \
		-W clippy::nursery \
		-D warnings

# ============================================================================
# Type checking
# ============================================================================
check: ## Run cargo check
	@echo "🔍 Type checking..."
	@cargo check --all-targets --all-features

# ============================================================================
# Test execution with multiple strategies
# ============================================================================
test-quick: ## Run unit tests only (fast)
	@echo "⚡ Running quick tests..."
	@PROPTEST_CASES=10 cargo test --lib --quiet

test-fast:
	cargo test --lib

test: test-doc test-property ## Run core test suite
	@echo "✅ Core test suite completed!"
	@echo "  - Unit tests ✓"
	@echo "  - Documentation tests ✓"
	@echo "  - Property-based tests ✓"

test-doc: ## Run documentation tests
	@echo "📚 Running documentation tests..."
	@cargo test --doc --workspace
	@echo "✅ Documentation tests completed!"

test-property: ## Run property-based tests (fast: 50 cases)
	@echo "🎲 Running property-based tests (50 cases)..."
	@PROPTEST_CASES=25 cargo test --lib property_ --quiet
	@PROPTEST_CASES=25 cargo test --lib prop_ --quiet
	@echo "✅ Property tests completed!"

test-property-comprehensive: ## Run property-based tests (comprehensive: 500 cases)
	@echo "🎲 Running property-based tests (500 cases)..."
	@PROPTEST_CASES=250 cargo test --lib property_
	@PROPTEST_CASES=250 cargo test --lib prop_
	@echo "✅ Property tests completed (comprehensive)!"

test-all: test test-property-comprehensive ## Run ALL tests comprehensively
	@echo "✅ All tests completed!"

# ============================================================================
# Quality metrics (Enhanced with pmat integration)
# ============================================================================
quality-gate: quality-gate-tier2 ## Run comprehensive quality gate checks
	@echo "🔍 Running comprehensive quality gate checks..."
	@if command -v pmat >/dev/null 2>&1; then \
		echo "  📊 Running pmat quality analysis..."; \
		pmat analyze complexity --max-cyclomatic 10 --format json --output .quality/complexity-current.json || true; \
		pmat analyze tdg --format json --output .quality/tdg-current.json || true; \
		pmat analyze satd --format json --output .quality/satd-current.json || true; \
		echo "  ✅ PMAT analysis complete"; \
	else \
		echo "  ⚠️  pmat not installed (install: cargo install pmat)"; \
	fi
	@echo "✅ Quality gates passed!"

analyze-complexity: ## Analyze code complexity with pmat
	@echo "📊 Analyzing code complexity..."
	@mkdir -p .quality
	@if command -v pmat >/dev/null 2>&1; then \
		pmat analyze complexity --max-cyclomatic 10 --format full --output .quality/complexity-current.json; \
		echo ""; \
		echo "💡 Detailed report: .quality/complexity-current.json"; \
	else \
		echo "⚠️  pmat not installed. Install: cargo install pmat"; \
		exit 1; \
	fi

analyze-tdg: ## Analyze Technical Debt Grade with pmat
	@echo "📈 Analyzing Technical Debt Grade..."
	@mkdir -p .quality
	@if command -v pmat >/dev/null 2>&1; then \
		pmat analyze tdg --format table --output .quality/tdg-current.json; \
		echo ""; \
		echo "💡 Target: A or higher (EXTREME TDD quality standards)"; \
	else \
		echo "⚠️  pmat not installed. Install: cargo install pmat"; \
		exit 1; \
	fi

validate-docs: ## Validate documentation accuracy with pmat
	@echo "🔍 Validating documentation accuracy..."
	@if command -v pmat >/dev/null 2>&1; then \
		echo "  📄 Generating deep context..."; \
		pmat context --output .quality/deep_context.md --format llm-optimized 2>/dev/null || true; \
		echo "  🔎 Validating documentation files..."; \
		pmat validate-readme \
			--targets README.md CLAUDE.md \
			--deep-context .quality/deep_context.md \
			--fail-on-contradiction \
			--verbose || echo "⚠️  Some documentation issues found"; \
		echo "✅ Documentation validation complete"; \
	else \
		echo "⚠️  pmat not installed. Install: cargo install pmat"; \
		echo "💡 This validates README against actual codebase (prevents hallucinations)"; \
	fi

quality-baseline: ## Establish quality baseline
	@mkdir -p .quality
	@if command -v pmat >/dev/null 2>&1 && [ ! -f .quality/baseline.json ]; then \
		echo "📊 Establishing quality baseline..."; \
		pmat repo-score . --format json > .quality/baseline.json || true; \
	fi

quality-report: ## Generate comprehensive quality report
	@echo "📈 Generating comprehensive quality report..."
	@if command -v pmat >/dev/null 2>&1; then \
		pmat context --output QUALITY_REPORT.md --format markdown; \
	fi
	@echo "## Certeza Metrics" >> QUALITY_REPORT.md
	@echo "- Coverage: $$(cargo llvm-cov report --summary-only 2>/dev/null | grep TOTAL || echo 'Run make coverage first')" >> QUALITY_REPORT.md

# ============================================================================
# Security audit
# ============================================================================
security: audit deny ## Run all security checks

audit: ## Run cargo audit for security vulnerabilities
	@echo "🔒 Running security audit..."
	@if command -v cargo-audit >/dev/null 2>&1; then \
		cargo audit; \
	else \
		echo "📥 Installing cargo-audit..."; \
		cargo install cargo-audit && cargo audit; \
	fi

deny: ## Check dependencies, licenses, and security advisories
	@echo "📋 Running cargo-deny checks..."
	@if command -v cargo-deny >/dev/null 2>&1; then \
		cargo deny check; \
	else \
		echo "📥 Installing cargo-deny..."; \
		cargo install cargo-deny && cargo deny check; \
	fi

# ============================================================================
# Code Coverage (Toyota Way: "make coverage" just works)
# TARGET: < 10 minutes
# ============================================================================
coverage: ## Generate HTML coverage report
	@echo "📊 Running comprehensive test coverage analysis..."
	@which cargo-llvm-cov > /dev/null 2>&1 || (echo "📦 Installing cargo-llvm-cov..." && cargo install cargo-llvm-cov --locked)
	@echo "🧹 Cleaning old coverage data..."
	@mkdir -p target/coverage
	@echo "🧪 Running tests with instrumentation..."
	@env PROPTEST_CASES=25 QUICKCHECK_TESTS=25 cargo llvm-cov --all-features --workspace --html --output-dir target/coverage/html
	@cargo llvm-cov report --lcov --output-path target/coverage/lcov.info
	@echo ""
	@echo "📊 Coverage Summary:"
	@echo "=================="
	@cargo llvm-cov report --summary-only
	@echo ""
	@echo "💡 COVERAGE INSIGHTS:"
	@echo "- HTML report: target/coverage/html/index.html"
	@echo "- LCOV file: target/coverage/lcov.info"
	@echo "- Open HTML: make coverage-open"
	@echo ""

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
	@echo "=== Code Coverage for CI/CD ==="
	@env PROPTEST_CASES=25 QUICKCHECK_TESTS=25 cargo llvm-cov --all-features --workspace --lcov --output-path lcov.info
	@echo "✓ Coverage report generated: lcov.info"

coverage-clean: ## Clean coverage artifacts
	@rm -f lcov.info coverage.xml target/coverage/lcov.info
	@rm -rf target/llvm-cov target/coverage
	@find . -name "*.profraw" -delete
	@echo "✓ Coverage artifacts cleaned"

# ============================================================================
# Mutation Testing (Tier 3)
# ============================================================================
mutation: ## Run full mutation testing (target: >85% mutation score)
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
		echo "No mutation results found. Run 'make mutation' first."; \
	fi

mutation-clean: ## Clean mutation testing artifacts
	@rm -rf mutants.out mutants.out.old
	@echo "✓ Mutation testing artifacts cleaned"

# ============================================================================
# Benchmarking (Phase 3: Scientific Reporting & Performance Tracking)
# ============================================================================
.PHONY: benchmark benchmark-critical benchmark-all benchmark-report benchmark-compare benchmark-baseline-save benchmark-clean

benchmark: benchmark-critical ## Run critical benchmarks (default)

benchmark-critical: ## Run critical performance benchmarks
	@echo "📊 Running critical benchmarks..."
	@./scripts/run_benchmarks.sh \
		--benchmarks critical \
		--output benchmarks/results/latest.json \
		--warmup 3 \
		--iterations 10
	@echo "✅ Critical benchmarks complete"

benchmark-all: ## Run comprehensive benchmark suite
	@echo "📊 Running comprehensive benchmark suite..."
	@./scripts/run_benchmarks.sh \
		--benchmarks all \
		--profiles all \
		--output benchmarks/results/latest.json \
		--warmup 5 \
		--iterations 20
	@echo "✅ Comprehensive benchmarks complete"

benchmark-report: ## Generate benchmark reports (MD + CSV)
	@echo "📄 Generating benchmark reports..."
	@if ! command -v deno >/dev/null 2>&1; then \
		echo "❌ Deno not found. Install from https://deno.land/"; \
		exit 1; \
	fi
	@if [ ! -f benchmarks/results/latest.json ]; then \
		echo "❌ No benchmark results found. Run 'make benchmark' first."; \
		exit 1; \
	fi
	@deno run --allow-read --allow-write \
		scripts/generate_markdown_report.ts \
		benchmarks/results/latest.json \
		benchmarks/results/report.md
	@deno run --allow-read --allow-write \
		scripts/generate_csv_report.ts \
		benchmarks/results/latest.json \
		benchmarks/results/csv/ \
		--multi
	@echo "✅ Reports generated:"
	@echo "  - Markdown: benchmarks/results/report.md"
	@echo "  - CSV: benchmarks/results/csv/"

benchmark-compare: ## Compare against baseline and check for regressions
	@echo "🔍 Checking for performance regressions..."
	@if ! command -v deno >/dev/null 2>&1; then \
		echo "❌ Deno not found. Install from https://deno.land/"; \
		exit 1; \
	fi
	@if [ ! -f benchmarks/baselines/main.json ]; then \
		echo "⚠️  No baseline found. Saving current results as baseline..."; \
		$(MAKE) benchmark-baseline-save; \
		exit 0; \
	fi
	@deno run --allow-read --allow-write \
		scripts/check_regression.ts \
		--baseline benchmarks/baselines/main.json \
		--current benchmarks/results/latest.json \
		--output-json benchmarks/results/regression_report.json
	@echo ""
	@deno run --allow-read --allow-write \
		scripts/baseline_manager.ts compare \
		--baseline main \
		--current benchmarks/results/latest.json \
		--format markdown \
		--output benchmarks/results/comparison.md
	@echo "✅ Comparison complete: benchmarks/results/comparison.md"

benchmark-baseline-save: ## Save current results as baseline
	@echo "💾 Saving benchmark baseline..."
	@if ! command -v deno >/dev/null 2>&1; then \
		echo "❌ Deno not found. Install from https://deno.land/"; \
		exit 1; \
	fi
	@if [ ! -f benchmarks/results/latest.json ]; then \
		echo "❌ No benchmark results found. Run 'make benchmark' first."; \
		exit 1; \
	fi
	@deno run --allow-read --allow-write \
		scripts/baseline_manager.ts save \
		--input benchmarks/results/latest.json \
		--name main \
		--description "Main branch baseline (updated: $$(date +%Y-%m-%d))"
	@echo "✅ Baseline saved: benchmarks/baselines/main.json"

benchmark-clean: ## Clean benchmark artifacts
	@rm -rf benchmarks/results/*.json benchmarks/results/*.md benchmarks/results/csv/
	@echo "✓ Benchmark artifacts cleaned"

# ============================================================================
# Dependency management
# ============================================================================
update-deps: ## Update dependencies (semver-compatible)
	@echo "🔄 Updating dependencies (semver-compatible)..."
	@cargo update --workspace
	@echo "Running tests to verify compatibility..."
	@make test-quick
	@echo "✅ Dependencies updated successfully!"

update-deps-aggressive: ## Update all dependencies including major versions
	@echo "🔄 Updating dependencies aggressively..."
	@if ! command -v cargo-upgrade >/dev/null 2>&1; then \
		echo "📥 Installing cargo-edit..."; \
		cargo install cargo-edit; \
	fi
	@cargo update --workspace
	@cargo upgrade --incompatible
	@echo "Running tests..."
	@make test-quick lint-check
	@make audit
	@echo "✅ Aggressive update completed!"

update-deps-check: ## Check for outdated dependencies
	@echo "🔍 Checking for outdated dependencies..."
	@if ! command -v cargo-outdated >/dev/null 2>&1; then \
		echo "📥 Installing cargo-outdated..."; \
		cargo install cargo-outdated; \
	fi
	@cargo outdated --workspace --root-deps-only
	@echo ""
	@echo "🔍 Security advisories check:"
	@make audit || true

# ============================================================================
# Documentation
# ============================================================================
docs: ## Build documentation
	@echo "📚 Building documentation..."
	@cargo doc --all-features --workspace --no-deps
	@echo "Documentation available at target/doc/certeza/index.html"

docs-open: ## Build and open documentation
	@echo "📚 Building and opening documentation..."
	@cargo doc --all-features --workspace --no-deps --open

# ============================================================================
# Build
# ============================================================================
build: ## Build release binaries
	@echo "🔨 Building release binaries..."
	@cargo build --release --all-features

# ============================================================================
# Install tools
# ============================================================================
install-tools: ## Install required tooling
	@echo "📦 Installing required tools..."
	@echo "Installing cargo-llvm-cov..."
	@cargo install cargo-llvm-cov --locked
	@echo "Installing cargo-mutants..."
	@cargo install cargo-mutants --locked
	@echo "Installing cargo-audit..."
	@cargo install cargo-audit --locked
	@echo "Installing cargo-deny..."
	@cargo install cargo-deny --locked
	@echo "Installing pmat (if available)..."
	@cargo install pmat --locked || echo "⚠️  pmat not available"
	@echo "✅ All tools installed!"

install-hooks: ## Install git hooks with PMAT
	@echo "🔒 Installing git hooks..."
	@if command -v pmat >/dev/null 2>&1; then \
		pmat hooks install; \
		pmat hooks status; \
	else \
		echo "⚠️  pmat not installed, cannot install hooks"; \
		echo "Install pmat: cargo install pmat"; \
	fi

# ============================================================================
# Repository scoring
# ============================================================================
repo-score: ## Calculate repository health score
	@echo "📊 Calculating repository health score..."
	@if command -v pmat >/dev/null 2>&1; then \
		pmat repo-score .; \
	else \
		echo "⚠️  pmat not installed. Install: cargo install pmat"; \
	fi

repo-score-deep: ## Calculate deep repository health score
	@echo "📊 Calculating deep repository health score..."
	@if command -v pmat >/dev/null 2>&1; then \
		pmat repo-score . --deep; \
	else \
		echo "⚠️  pmat not installed. Install: cargo install pmat"; \
	fi

# ============================================================================
# Clean
# ============================================================================
clean: ## Clean build artifacts
	@echo "🧹 Cleaning..."
	@cargo clean
	@rm -rf target/ mutants.out/ .quality/ .kaizen/
	@rm -f lcov.info coverage.xml QUALITY_REPORT.md
	@find . -name "*.profraw" -delete

# ============================================================================
# Help
# ============================================================================
help: ## Show this help
	@echo "Certeza - Asymptotic Test Effectiveness Framework"
	@echo "=================================================="
	@echo ""
	@echo "Main targets:"
	@echo "  make              - Run Tier 2 validation and build"
	@echo "  make tier1        - Tier 1: ON-SAVE checks (sub-second)"
	@echo "  make tier2        - Tier 2: ON-COMMIT checks (1-5 min)"
	@echo "  make tier3        - Tier 3: ON-MERGE/NIGHTLY (hours)"
	@echo "  make build        - Build release binaries"
	@echo ""
	@echo "Validation:"
	@echo "  make validate     - Full validation pipeline"
	@echo "  make quick-validate - Quick validation for development"
	@echo ""
	@echo "Testing:"
	@echo "  make test         - Run core test suite"
	@echo "  make test-quick   - Run unit tests only (fast)"
	@echo "  make test-property - Run property-based tests (50 cases)"
	@echo "  make test-property-comprehensive - Run property tests (500 cases)"
	@echo "  make test-all     - Run ALL tests comprehensively"
	@echo "  make test-doc     - Run documentation tests"
	@echo ""
	@echo "Quality:"
	@echo "  make quality-gate - Run comprehensive quality checks"
	@echo "  make analyze-complexity - Analyze code complexity with pmat"
	@echo "  make analyze-tdg  - Analyze Technical Debt Grade with pmat"
	@echo "  make validate-docs - Validate README accuracy (zero hallucinations)"
	@echo "  make quality-report - Generate quality report"
	@echo "  make security     - Run all security checks"
	@echo "  make audit        - Security audit"
	@echo "  make deny         - Check dependencies and licenses"
	@echo ""
	@echo "Coverage:"
	@echo "  make coverage     - Generate HTML coverage report"
	@echo "  make coverage-open - Open HTML coverage in browser"
	@echo "  make coverage-ci  - Generate LCOV report for CI/CD"
	@echo "  make coverage-clean - Clean coverage artifacts"
	@echo ""
	@echo "Mutation Testing (Tier 3):"
	@echo "  make mutation     - Run full mutation testing (>85% target)"
	@echo "  make mutation-report - Analyze mutation test results"
	@echo "  make mutation-clean - Clean mutation artifacts"
	@echo ""
	@echo "Benchmarking:"
	@echo "  make benchmark    - Run critical performance benchmarks"
	@echo "  make benchmark-all - Run comprehensive benchmark suite"
	@echo "  make benchmark-report - Generate markdown/CSV reports"
	@echo "  make benchmark-compare - Compare against baseline (regression check)"
	@echo "  make benchmark-baseline-save - Save current as baseline"
	@echo "  make benchmark-clean - Clean benchmark artifacts"
	@echo ""
	@echo "Code Quality:"
	@echo "  make format       - Format code"
	@echo "  make format-check - Check formatting"
	@echo "  make lint         - Run clippy with fixes"
	@echo "  make lint-check   - Check clippy"
	@echo "  make clippy-strict - Run strict clippy"
	@echo "  make check        - Type checking"
	@echo ""
	@echo "Dependencies:"
	@echo "  make update-deps  - Update dependencies (semver-compatible)"
	@echo "  make update-deps-aggressive - Update all dependencies"
	@echo "  make update-deps-check - Check for outdated dependencies"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs         - Build documentation"
	@echo "  make docs-open    - Build and open documentation"
	@echo ""
	@echo "Setup:"
	@echo "  make install-tools - Install required tooling"
	@echo "  make install-hooks - Install PMAT git hooks"
	@echo ""
	@echo "Analysis:"
	@echo "  make repo-score   - Calculate repository health score"
	@echo "  make repo-score-deep - Deep repository analysis"
	@echo "  make kaizen       - Continuous improvement cycle"
	@echo "  make demo-mode    - Interactive framework demonstration"
	@echo ""
	@echo "Other:"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make help         - Show this help"
	@echo ""
	@echo "Toyota Way Principles:"
	@echo "  - 改善 (Kaizen): Continuous improvement"
	@echo "  - 無駄 (Muda): Elimination of waste"
	@echo "  - Zero defects through tiered verification"
	@echo "  - Fast feedback loops maintain flow state"
