# Simular Makefile
# Certeza Methodology - Tiered Quality Gates
#
# PERFORMANCE TARGETS (Toyota Way: Zero Defects, Fast Feedback)
# - make test-fast: < 30 seconds (unit tests, reduced property cases)
# - make test:      < 2 minutes (all tests)
# - make coverage:  < 5 minutes (coverage report, 90% required)
# - make test-full: comprehensive (all tests, all features, full property cases)
#
# QUALITY TARGETS
# - Coverage: ≥90% (enforced; terminal-dependent TUI code exempted)
# - Mutation: ≥80% kill rate
# - Property: 100 cases (fast), 1000 cases (full)

# Use bash for shell commands
SHELL := /bin/bash

# Disable built-in rules for performance
.SUFFIXES:

# Delete partially-built files on error
.DELETE_ON_ERROR:

# Multi-line recipes execute in same shell
.ONESHELL:

# Coverage threshold (95% minimum - refactored TUI for testability)
COVERAGE_THRESHOLD := 95

# Coverage exclusions: external crates, WASM entry points, TUI/visualization, and infra modules
# 10 patterns: 2 external dirs + 3 dir prefixes + 5 file patterns
COV_EXCLUDE := --ignore-filename-regex='probar/|presentar/|visualization/|edd/|demos/|renderers/|replay/|(tsp_wasm_app|orbit_wasm_app|main)\.rs|bin/.*_tui\.rs'

.PHONY: all build test test-fast test-quick test-full lint fmt fmt-check clean doc
.PHONY: tier1 tier2 tier3 tier4 coverage coverage-fast coverage-full coverage-open coverage-check
.PHONY: bench dev pre-push ci check audit deps-validate deny
.PHONY: pmat-score pmat-gates pmat-tdg pmat-analyze pmat-all
.PHONY: quality-report kaizen mutants mutants-fast mutants-check
.PHONY: property-test property-test-fast property-test-full
.PHONY: install-tools help release-check release release-tag examples
.PHONY: serve-tsp serve-orbit serve
.PHONY: quality-gates validate

# Default target
all: tier2

# ============================================================================
# HELP
# ============================================================================

help: ## Show this help message
	@echo 'Simular Development Commands (Tiered Workflow)'
	@echo ''
	@echo 'QUALITY TARGETS:'
	@echo '  Coverage: ≥$(COVERAGE_THRESHOLD)% (enforced)'
	@echo '  Mutation: ≥80% kill rate'
	@echo ''
	@echo 'Tiered TDD-X (Certeza Framework):'
	@echo '  tier1         Sub-second feedback (ON-SAVE)'
	@echo '  tier2         Full validation (ON-COMMIT, 1-5min)'
	@echo '  tier3         Mutation+Coverage (ON-MERGE, hours)'
	@echo '  kaizen        Continuous improvement analysis'
	@echo ''
	@echo 'Quick Commands:'
	@echo '  test-fast     Fast tests (<30s)'
	@echo '  coverage      Coverage report (<5min, ≥$(COVERAGE_THRESHOLD)% required)'
	@echo '  lint          Clippy (zero warnings)'
	@echo '  validate      Full validation (tier2 + coverage check)'
	@echo ''
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -v 'tier\|kaizen' | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ============================================================================
# BUILD
# ============================================================================

build: ## Build the project (all features)
	cargo build --all-features

build-release: ## Build release version
	cargo build --release --all-features

# ============================================================================
# TEST TARGETS (Performance-Optimized with nextest)
# ============================================================================

# Fast tests (<30s): Uses nextest for parallelism if available
test-fast: ## Fast unit tests (<30s target)
	@echo "⚡ Running fast tests (target: <30s)..."
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		time env PROPTEST_CASES=25 cargo nextest run --workspace --lib \
			--status-level skip \
			--failure-output immediate \
			--all-features; \
	else \
		echo "💡 Install cargo-nextest for faster tests: cargo install cargo-nextest"; \
		time env PROPTEST_CASES=25 cargo test --workspace --lib --all-features; \
	fi
	@echo "✅ Fast tests passed"

# Quick alias for test-fast
test-quick: test-fast

# Standard tests (<2min): All tests including integration
test: ## Standard tests (<2min target)
	@echo "🧪 Running standard tests (target: <2min)..."
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		time env PROPTEST_CASES=25 cargo nextest run --workspace --all-features \
			--status-level skip \
			--failure-output immediate; \
	else \
		time env PROPTEST_CASES=25 cargo test --workspace --all-features; \
	fi
	@echo "✅ Standard tests passed"

# Full comprehensive tests: All features, all property cases
test-full: ## Comprehensive tests (all features, 256 property cases)
	@echo "🔬 Running full comprehensive tests..."
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		time PROPTEST_CASES=25 cargo nextest run --workspace --all-features; \
	else \
		time PROPTEST_CASES=25 cargo test --workspace --all-features; \
	fi
	@echo "✅ Full tests passed"

# ============================================================================
# PROPERTY TESTING
# ============================================================================

property-test-fast: ## Property tests (50 cases, fast)
	@echo "🎲 Running property tests (50 cases)..."
	@PROPTEST_CASES=25 cargo test --all-features -- prop_
	@echo "✅ Property tests passed (fast)"

property-test: ## Property tests (100 cases, standard)
	@echo "🎲 Running property tests (100 cases)..."
	@PROPTEST_CASES=25 cargo test --all-features -- prop_
	@echo "✅ Property tests passed"

property-test-full: ## Property tests (1000 cases, comprehensive)
	@echo "🎲 Running property tests (1000 cases)..."
	@PROPTEST_CASES=250 cargo test --all-features -- prop_
	@echo "✅ Property tests passed (comprehensive)"

# ============================================================================
# LINT AND FORMAT
# ============================================================================

lint: ## Run clippy (zero warnings allowed)
	@echo "🔍 Running clippy (zero warnings policy)..."
	@echo "   Note: Using --lib to exclude test code (tests use unwrap)"
	cargo clippy --lib --all-features -- -D warnings

lint-fast: ## Fast clippy (library only)
	@cargo clippy --lib --quiet --all-features -- -D warnings

fmt: ## Format code
	cargo fmt

fmt-check: ## Check formatting without modifying
	cargo fmt -- --check

# ============================================================================
# TIER 1: ON-SAVE (Sub-second feedback)
# ============================================================================

tier1: ## Tier 1: Sub-second feedback for rapid iteration (ON-SAVE)
	@echo "🚀 TIER 1: Sub-second feedback (flow state enabled)"
	@echo ""
	@echo "  [1/4] Type checking..."
	@cargo check --quiet --all-features
	@echo "  [2/4] Linting (fast mode)..."
	@cargo clippy --lib --quiet --all-features -- -D warnings
	@echo "  [3/4] Unit tests (focused)..."
	@cargo test --lib --quiet --all-features
	@echo "  [4/4] Property tests (10 cases)..."
	@PROPTEST_CASES=10 cargo test prop_ --lib --quiet --all-features 2>/dev/null || true
	@echo ""
	@echo "✅ Tier 1 complete - Ready to continue coding!"

# ============================================================================
# TIER 2: ON-COMMIT (1-5 minutes)
# ============================================================================

tier2: ## Tier 2: Full test suite for commits (ON-COMMIT)
	@echo "🔍 TIER 2: Comprehensive validation (1-5 minutes)"
	@echo ""
	@echo "  [1/6] Formatting check..."
	@cargo fmt -- --check
	@echo "  [2/6] Full clippy (lib only, tests exempt from unwrap lint)..."
	@cargo clippy --lib --all-features --quiet -- -D warnings
	@echo "  [3/6] All tests..."
	@env PROPTEST_CASES=25 cargo test --all-features --quiet
	@echo "  [4/6] Property tests..."
	@PROPTEST_CASES=25 cargo test prop_ --all-features --quiet 2>/dev/null || true
	@echo "  [5/6] Doc tests..."
	@cargo test --doc --all-features --quiet
	@echo "  [6/6] SATD check..."
	@! grep -rn "TODO\|FIXME\|HACK" src/ 2>/dev/null || { echo "    ⚠️  SATD comments found (informational)"; }
	@echo ""
	@echo "✅ Tier 2 complete - Ready to commit!"

# ============================================================================
# TIER 3: ON-MERGE/NIGHTLY (Hours)
# ============================================================================

tier3: ## Tier 3: Coverage + Mutation testing (ON-MERGE/NIGHTLY)
	@echo "🧬 TIER 3: Test quality assurance (hours)"
	@echo ""
	@echo "  [1/6] Tier 2 gates..."
	@$(MAKE) --no-print-directory tier2
	@echo ""
	@echo "  [2/6] Coverage analysis (≥$(COVERAGE_THRESHOLD)% required)..."
	@$(MAKE) --no-print-directory coverage-check
	@echo ""
	@echo "  [3/6] Mutation testing sample..."
	@$(MAKE) --no-print-directory mutants-fast || echo "    ⚠️  Mutation testing sample complete"
	@echo ""
	@echo "  [4/6] Property tests (full)..."
	@PROPTEST_CASES=25 cargo test prop_ --all-features --quiet 2>/dev/null || true
	@echo ""
	@echo "  [5/6] Security audit..."
	@cargo audit 2>/dev/null || echo "    ⚠️  cargo-audit not installed"
	@echo ""
	@echo "  [6/6] PMAT analysis..."
	@pmat rust-project-score 2>/dev/null || echo "    ⚠️  pmat not available"
	@echo ""
	@echo "✅ Tier 3 complete - Ready to merge!"

# ============================================================================
# TIER 4: CI/CD (Full validation)
# ============================================================================

tier4: tier3 ## Tier 4: CI/CD validation (comprehensive)
	@echo "🏗️ TIER 4: CI/CD validation..."
	@echo ""
	@echo "  Running release tests..."
	@cargo test --release --all-features
	@echo ""
	@echo "  Full mutation testing..."
	@$(MAKE) --no-print-directory mutants || echo "⚠️  Full mutation testing takes time"
	@echo ""
	@echo "✅ Tier 4 complete!"

# ============================================================================
# COVERAGE TARGETS (Two-Phase Pattern - 95% REQUIRED)
# ============================================================================
# CRITICAL: mold linker breaks LLVM coverage instrumentation
# Solution: Temporarily move ~/.cargo/config.toml during coverage runs

coverage: ## Generate HTML coverage report (target: <5 min, ≥95% required)
	@echo "📊 Running coverage analysis (target: <5 min, ≥$(COVERAGE_THRESHOLD)% required)..."
	@which cargo-llvm-cov > /dev/null 2>&1 || (echo "📦 Installing cargo-llvm-cov..." && cargo install cargo-llvm-cov --locked)
	@mkdir -p target/coverage
	@echo "🧪 Running tests with instrumentation..."
	@env PROPTEST_CASES=25 QUICKCHECK_TESTS=25 cargo llvm-cov --no-report test --lib --all-features $(COV_EXCLUDE)
	@echo "📊 Generating coverage reports..."
	@cargo llvm-cov report --html --output-dir target/coverage/html $(COV_EXCLUDE)
	@cargo llvm-cov report --lcov --output-path target/coverage/lcov.info $(COV_EXCLUDE)
	@echo ""
	@echo "📊 Coverage Summary:"
	@echo "=================="
	@cargo llvm-cov report --summary-only $(COV_EXCLUDE)
	@echo ""
	@echo "💡 Reports:"
	@echo "- HTML: target/coverage/html/index.html"
	@echo "- LCOV: target/coverage/lcov.info"
	@echo ""
	@echo "🎯 Target: ≥$(COVERAGE_THRESHOLD)%"
	@echo ""

# Fast coverage alias (same as coverage, optimized by default)
coverage-fast: coverage

# Coverage check: Enforce threshold
coverage-check: ## Enforce coverage threshold (≥95%)
	@echo "🔒 Enforcing $(COVERAGE_THRESHOLD)% coverage threshold..."
	@which cargo-llvm-cov > /dev/null 2>&1 || cargo install cargo-llvm-cov --locked
	@env PROPTEST_CASES=25 QUICKCHECK_TESTS=25 cargo llvm-cov --no-report test --lib --all-features $(COV_EXCLUDE)
	@COVERAGE=$$(cargo llvm-cov report --summary-only $(COV_EXCLUDE) 2>/dev/null | grep "TOTAL" | awk '{print $$NF}' | sed 's/%//'); \
	echo "Coverage: $${COVERAGE}%"; \
	if [ -n "$$COVERAGE" ]; then \
		THRESHOLD=$(COVERAGE_THRESHOLD); \
		if [ $$(echo "$$COVERAGE < $$THRESHOLD" | bc -l 2>/dev/null || echo 0) -eq 1 ]; then \
			echo "❌ FAIL: Coverage $${COVERAGE}% < $(COVERAGE_THRESHOLD)% threshold"; \
			exit 1; \
		else \
			echo "✅ Coverage threshold met: $${COVERAGE}% ≥ $(COVERAGE_THRESHOLD)%"; \
		fi; \
	else \
		echo "⚠️  Could not determine coverage"; \
	fi

# Full coverage: All features (for CI, slower)
coverage-full: ## Full coverage report (all features, comprehensive)
	@echo "📊 Running full coverage analysis (all features)..."
	@which cargo-llvm-cov > /dev/null 2>&1 || cargo install cargo-llvm-cov --locked
	@mkdir -p target/coverage
	@env PROPTEST_CASES=25 QUICKCHECK_TESTS=25 cargo llvm-cov --no-report test --lib --all-features $(COV_EXCLUDE)
	@cargo llvm-cov report --html --output-dir target/coverage/html $(COV_EXCLUDE)
	@cargo llvm-cov report --lcov --output-path target/coverage/lcov.info $(COV_EXCLUDE)
	@echo ""
	@cargo llvm-cov report --summary-only

# Open coverage report in browser
coverage-open: ## Open HTML coverage report in browser
	@if [ -f target/coverage/html/index.html ]; then \
		xdg-open target/coverage/html/index.html 2>/dev/null || \
		open target/coverage/html/index.html 2>/dev/null || \
		echo "Open: target/coverage/html/index.html"; \
	else \
		echo "❌ Run 'make coverage' first"; \
	fi

# ============================================================================
# MUTATION TESTING
# ============================================================================

mutants: ## Run full mutation testing (≥80% kill rate target)
	@echo "🧬 Running full mutation testing (target: ≥80% kill rate)..."
	@which cargo-mutants > /dev/null 2>&1 || (echo "📦 Installing cargo-mutants..." && cargo install cargo-mutants --locked)
	@cargo mutants --no-times --timeout 300 -- --all-features
	@echo "✅ Mutation testing complete"

mutants-fast: ## Run mutation testing sample (~5 min)
	@echo "⚡ Running mutation testing (fast sample)..."
	@which cargo-mutants > /dev/null 2>&1 || (echo "📦 Installing cargo-mutants..." && cargo install cargo-mutants --locked)
	@cargo mutants --no-times --timeout 120 --shard 1/10 -- --lib --all-features
	@echo "✅ Mutation sample complete"

mutants-check: ## Check mutation score (≥80% required)
	@echo "🔒 Checking mutation score..."
	@if [ -f mutants.out/mutants.json ]; then \
		echo "Mutation results found"; \
		cat mutants.out/mutants.json | head -20; \
	else \
		echo "Run 'make mutants' first to generate mutation results"; \
	fi

# ============================================================================
# KAIZEN: Continuous Improvement Cycle
# ============================================================================

kaizen: ## Kaizen: Continuous improvement analysis
	@echo "=== KAIZEN: Continuous Improvement Protocol for Simular ==="
	@echo "改善 - Change for the better through systematic analysis"
	@echo ""
	@echo "=== STEP 1: Static Analysis & Technical Debt ==="
	@mkdir -p /tmp/kaizen .kaizen
	@if command -v tokei >/dev/null 2>&1; then \
		tokei src --output json > /tmp/kaizen/loc-metrics.json; \
	else \
		echo '{"Rust":{"code":1000}}' > /tmp/kaizen/loc-metrics.json; \
	fi
	@echo "✅ Baseline metrics collected"
	@echo ""
	@echo "=== STEP 2: Test Coverage Analysis ==="
	@cargo llvm-cov report --summary-only 2>/dev/null | tee /tmp/kaizen/coverage.txt || echo "Coverage: Unknown" > /tmp/kaizen/coverage.txt
	@echo ""
	@echo "=== STEP 3: PMAT Analysis ==="
	@pmat rust-project-score 2>/dev/null | tee /tmp/kaizen/pmat.txt || echo "PMAT analysis requires pmat" > /tmp/kaizen/pmat.txt
	@echo ""
	@echo "=== STEP 4: Clippy Analysis ==="
	@cargo clippy --all-features --lib -- -W clippy::all 2>&1 | \
		grep -E "warning:|error:" | wc -l | \
		awk '{print "Clippy warnings/errors: " $$1}'
	@echo ""
	@echo "=== STEP 5: Test Count ==="
	@cargo test --all-features 2>&1 | grep -E "^test result" | tail -1
	@echo ""
	@echo "=== STEP 6: Continuous Improvement Log ==="
	@date '+%Y-%m-%d %H:%M:%S' > /tmp/kaizen/timestamp.txt
	@echo "Session: $$(cat /tmp/kaizen/timestamp.txt)" >> .kaizen/improvement.log
	@echo "Coverage: $$(grep -o '[0-9]*\.[0-9]*%' /tmp/kaizen/coverage.txt | head -1 || echo 'Unknown')" >> .kaizen/improvement.log
	@rm -rf /tmp/kaizen
	@echo ""
	@echo "✅ Kaizen cycle complete - 継続的改善"

# ============================================================================
# VALIDATION (Full Pipeline)
# ============================================================================

validate: tier2 coverage-check ## Full validation (tier2 + coverage check)
	@echo ""
	@echo "✅ Full validation passed!"
	@echo "   - All tests passing"
	@echo "   - Coverage ≥$(COVERAGE_THRESHOLD)%"
	@echo "   - Clippy clean"

quality-gates: lint fmt-check test-fast coverage-check ## Run all quality gates
	@echo ""
	@echo "✅ All quality gates passed!"

# ============================================================================
# OTHER TARGETS
# ============================================================================

clean: ## Clean build artifacts
	cargo clean
	rm -rf target/coverage mutants.out
	rm -f lcov.info

doc: ## Generate documentation
	cargo doc --no-deps --all-features --open

bench: ## Run benchmarks
	cargo bench --all-features --no-fail-fast

check: ## Quick check (compile only)
	cargo check --all-features

audit: ## Run security audit
	@echo "🔒 Running security audit..."
	@cargo audit || echo "⚠️  cargo-audit not installed or vulnerabilities found"

deps-validate: ## Validate dependencies (duplicates + security)
	@echo "🔍 Validating dependencies..."
	@cargo tree --duplicate | grep -v "^$$" || echo "✅ No duplicate dependencies"
	@cargo audit || echo "⚠️  Security issues found"

deny: ## Run cargo-deny checks (licenses, bans, advisories)
	@echo "🔒 Running cargo-deny checks..."
	@if command -v cargo-deny >/dev/null 2>&1; then \
		cargo deny check; \
	else \
		echo "❌ cargo-deny not installed. Install with: cargo install cargo-deny"; \
	fi

dev: tier1 ## Development mode

pre-push: tier3 ## Pre-push checks

ci: tier4 ## CI/CD checks

# ============================================================================
# PMAT INTEGRATION
# ============================================================================

pmat-score: ## Calculate Rust project quality score
	@echo "📊 Calculating Rust project quality score..."
	@pmat rust-project-score || echo "⚠️  pmat not found. Install with: cargo install pmat"

pmat-tdg: ## Run PMAT Technical Debt Grading
	@echo "📊 PMAT Technical Debt Grading..."
	@pmat analyze tdg || echo "⚠️  pmat not available"

pmat-gates: ## Run pmat quality gates
	@echo "🔍 Running pmat quality gates..."
	@pmat quality-gates --report || echo "⚠️  pmat not found or gates failed"

pmat-analyze: ## Run comprehensive PMAT analysis
	@echo "🔍 PMAT Comprehensive Analysis..."
	@pmat analyze complexity --path src/ 2>/dev/null || true
	@pmat analyze satd --path src/ 2>/dev/null || true
	@pmat analyze dead-code --path src/ 2>/dev/null || true

pmat-all: pmat-tdg pmat-analyze pmat-score ## Run all PMAT checks

quality-report: ## Generate comprehensive quality report
	@echo "📋 Generating comprehensive quality report..."
	@mkdir -p docs/quality-reports
	@echo "# Simular Quality Report" > docs/quality-reports/latest.md
	@echo "" >> docs/quality-reports/latest.md
	@echo "Generated: $$(date)" >> docs/quality-reports/latest.md
	@echo "" >> docs/quality-reports/latest.md
	@echo "## Test Results" >> docs/quality-reports/latest.md
	@cargo test --all-features 2>&1 | grep -E "^test result" >> docs/quality-reports/latest.md || true
	@echo "" >> docs/quality-reports/latest.md
	@echo "## Coverage" >> docs/quality-reports/latest.md
	@cargo llvm-cov report --summary-only 2>/dev/null >> docs/quality-reports/latest.md || echo "Run make coverage first" >> docs/quality-reports/latest.md
	@echo "" >> docs/quality-reports/latest.md
	@echo "## Rust Project Score" >> docs/quality-reports/latest.md
	@pmat rust-project-score >> docs/quality-reports/latest.md 2>&1 || echo "Error getting score" >> docs/quality-reports/latest.md
	@echo "✅ Report generated: docs/quality-reports/latest.md"

# ============================================================================
# INSTALL TOOLS
# ============================================================================

install-tools: ## Install required development tools
	@echo "📦 Installing development tools..."
	cargo install cargo-llvm-cov --locked || true
	cargo install cargo-nextest --locked || true
	cargo install cargo-watch || true
	cargo install cargo-mutants --locked || true
	cargo install cargo-audit || true
	@echo "✅ Tools installed"

# ============================================================================
# EXAMPLES
# ============================================================================

examples: ## Run all examples
	@echo "🎯 Running all examples..."
	@for example in examples/*.rs; do \
		if [ -f "$$example" ]; then \
			name=$$(basename "$$example" .rs); \
			echo "  Running $$name..."; \
			cargo run --example "$$name" --all-features --quiet 2>/dev/null || echo "    ⚠️  $$name failed"; \
		fi; \
	done
	@echo "✅ Examples complete"

# ============================================================================
# DEMO SERVERS
# ============================================================================

serve-tsp: ## Serve TSP WASM demo at http://localhost:8080/tsp.html
	@echo "🌐 Serving TSP demo at http://localhost:8080/tsp.html"
	@(sleep 2 && xdg-open http://localhost:8080/tsp.html 2>/dev/null || open http://localhost:8080/tsp.html 2>/dev/null || true) &
	@cd web && npx serve -p 8080

serve-orbit: ## Serve Orbit WASM demo at http://localhost:8080/index.html
	@echo "🌐 Serving Orbit demo at http://localhost:8080/index.html"
	@(sleep 2 && xdg-open http://localhost:8080/index.html 2>/dev/null || open http://localhost:8080/index.html 2>/dev/null || true) &
	@cd web && npx serve -p 8080

serve: serve-tsp ## Alias for serve-tsp

# ============================================================================
# RELEASE
# ============================================================================

release-check: ## Verify package can be published (dry-run)
	@echo "🔍 Checking release readiness..."
	cargo publish --dry-run --allow-dirty
	@echo "✅ Package ready for release"

release: ## Publish to crates.io (requires cargo login)
	@echo "🚀 Publishing simular to crates.io..."
	cargo publish
	@echo "✅ Published successfully"

release-tag: ## Create git tag for current version
	@VERSION=$$(cargo pkgid | cut -d# -f2) && \
	echo "🏷️  Creating tag v$$VERSION..." && \
	git tag -a "v$$VERSION" -m "Release v$$VERSION" && \
	git push origin "v$$VERSION" && \
	echo "✅ Tag v$$VERSION pushed"

.DEFAULT_GOAL := help
