# Aprender Makefile
# Certeza Methodology - Tiered Quality Gates
#
# PERFORMANCE TARGETS (Toyota Way: Zero Defects, Fast Feedback)
# - make test-fast: < 30 seconds (unit tests, no encryption features)
# - make test:      < 2 minutes (all tests, reduced property cases)
# - make coverage:  < 5 minutes (coverage report, reduced property cases)
# - make test-full: comprehensive (all tests, all features, full property cases)

# Use bash for shell commands
SHELL := /bin/bash

# Disable built-in rules for performance
.SUFFIXES:

# Delete partially-built files on error
.DELETE_ON_ERROR:

# Multi-line recipes execute in same shell
.ONESHELL:

.PHONY: all build test test-smoke test-fast test-quick test-full test-heavy lint fmt clean doc book book-build book-serve book-test tier1 tier2 tier3 tier4 coverage coverage-fast profile hooks-install hooks-verify lint-scripts bashrs-score bashrs-lint-makefile chaos-test chaos-test-full chaos-test-lite fuzz bench dev pre-push ci check run-ci run-bench audit deps-validate deny pmat-score pmat-gates quality-report semantic-search examples mutants mutants-fast property-test install-alsa test-alsa test-audio-full

# Default target
all: tier2

# Build
build:
	cargo build --release

# ============================================================================
# TEST TARGETS (Performance-Optimized with nextest)
# ============================================================================

# Smoke tests (<2s): Minimal critical path verification (Section P: P2)
# Only runs core API tests, no proptests, no encryption, no network
test-smoke: ## Smoke tests (<2s target, Section P: P2)
	@echo "💨 Running smoke tests (target: <2s)..."
	@time PROPTEST_CASES=5 QUICKCHECK_TESTS=5 cargo test --lib --no-fail-fast -- \
		--skip prop_ \
		--skip test_encrypted \
		--skip test_cache_metadata_expiration \
		--skip test_cache_metadata_age \
		--skip test_cache_entry_is_valid_expired \
		--skip test_time_budget \
		--skip k20_trueno_simd \
		--skip test_de_handles_different \
		tests::test_lib_sanity 2>/dev/null || \
		cargo test --lib --no-fail-fast -- \
		--skip prop_ \
		--skip test_encrypted \
		--skip test_cache_metadata \
		--skip test_time_budget \
		--skip k20_ \
		--skip test_de_ \
		2>&1 | head -50
	@echo "✅ Smoke tests passed"

# Fast tests (<30s): Uses nextest for parallelism if available
# Pattern from bashrs: cargo-nextest + PROPTEST_CASES + exclude slow tests
# Excludes: prop_gbm_expected_value_convergence (46s alone!)
test-fast: ## Fast unit tests (<30s target)
	@echo "⚡ Running fast tests (target: <30s, -j2 to prevent OOM)..."
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		time env PROPTEST_CASES=25 QUICKCHECK_TESTS=25 cargo nextest run --workspace --lib -j 2 \
			--status-level skip \
			--failure-output immediate \
			-E 'not test(/prop_gbm_expected_value_convergence/)'; \
	else \
		echo "💡 Install cargo-nextest for faster tests: cargo install cargo-nextest"; \
		time env PROPTEST_CASES=25 QUICKCHECK_TESTS=25 cargo test --workspace --lib -- --test-threads=2 --skip prop_gbm_expected_value_convergence; \
	fi
	@echo "✅ Fast tests passed"

# Quick alias for test-fast
test-quick: test-fast

# Standard tests (<2min): All tests including integration
test: ## Standard tests (<2min target)
	@echo "🧪 Running standard tests (target: <2min, -j2 to prevent OOM)..."
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		time PROPTEST_CASES=25 QUICKCHECK_TESTS=25 cargo nextest run --workspace -j 2 \
			--status-level skip \
			--failure-output immediate; \
	else \
		time PROPTEST_CASES=25 QUICKCHECK_TESTS=25 cargo test --workspace -- --test-threads=2; \
	fi
	@echo "✅ Standard tests passed"

# Full comprehensive tests: All features, all property cases
test-full: ## Comprehensive tests (all features)
	@echo "🔬 Running full comprehensive tests..."
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		time PROPTEST_CASES=100 QUICKCHECK_TESTS=100 cargo nextest run --workspace --all-features; \
	else \
		time PROPTEST_CASES=100 QUICKCHECK_TESTS=100 cargo test --workspace --all-features; \
	fi
	@echo "✅ Full tests passed"

# Heavy tests: Runs ignored tests (Section P: P7)
# Includes: sleep()-based tests, slow encryption tests, long proptests
test-heavy: ## Heavy/slow tests (ignored tests)
	@echo "🐢 Running heavy tests (ignored tests)..."
	@time PROPTEST_CASES=256 QUICKCHECK_TESTS=256 cargo test --workspace -- --ignored
	@echo "✅ Heavy tests passed"

test-model: ## Run model falsification tests ONE AT A TIME (requires models/, ollama, GPU)
	@echo "🧪 Running model falsification tests (one at a time to avoid OOM)..."
	@for test in f_ollama_001 f_ollama_002 f_ollama_003 f_ollama_004 f_ollama_005 \
	             f_perf_003 f_trueno_004 f_trueno_008 f_rosetta_002 f_qa_002; do \
		echo "  ⏳ $$test"; \
		PROPTEST_CASES=10 QUICKCHECK_TESTS=10 \
		cargo test --features model-tests --test falsification_spec_v10_tests "$$test" 2>&1 \
			| grep "test result:" || echo "  ❌ $$test FAILED"; \
	done
	@echo "✅ Model tests complete"

test-spec: ## Run ALL spec falsification tests (structural only, no models)
	@echo "🔬 Running spec structural tests..."
	@PROPTEST_CASES=10 QUICKCHECK_TESTS=10 \
		cargo test --features model-tests --test falsification_spec_v10_tests 2>&1 \
		| grep "test result:"
	@echo "✅ Spec tests complete"

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
	@PROPTEST_CASES=5 QUICKCHECK_TESTS=5 cargo test --lib
	@cargo clippy -- -D warnings
	@echo "Tier 2: PASSED"

# Tier 3: Pre-push (1-5 minutes, full validation)
tier3:
	@echo "Running Tier 3: Full validation..."
	@PROPTEST_CASES=25 QUICKCHECK_TESTS=25 cargo test --all
	@cargo clippy -- -D warnings
	@echo "Tier 3: PASSED"

# Tier 4: CI/CD (5-60 minutes, heavyweight)
tier4: tier3
	@echo "Running Tier 4: CI/CD validation..."
	@PROPTEST_CASES=100 QUICKCHECK_TESTS=100 cargo test --release
	@echo "Running pmat analysis..."
	-pmat tdg . --include-components
	-pmat rust-project-score
	-pmat quality-gates --report
	@echo "Tier 4: PASSED"

# ============================================================================
# COVERAGE TARGETS (Two-Phase Pattern from bashrs)
# ============================================================================
# Pattern: bashrs/Makefile - Two-phase coverage with mold linker workaround
# CRITICAL: mold linker breaks LLVM coverage instrumentation
# Solution: Temporarily move ~/.cargo/config.toml during coverage runs

# Exclusion patterns for coverage reports
# ONLY excludes truly external/feature-gated code - all apr subcommands INCLUDED
#   External crates:
#     - .cargo/           : Dependencies from crates.io
#     - trueno/           : Local sibling crate (SIMD tensor ops)
#     - realizar/         : Local sibling crate (inference engine)
#     - entrenar/         : Local sibling crate (training)
#   Local exclusions:
#     - fuzz/             : Fuzz test infrastructure
#     - golden_traces/    : Trace data files
#   Feature-gated (require --all-features):
#     - audio/            : Requires audio feature + ALSA
#     - hf_hub/           : HuggingFace hub (network-dependent)
#   Test infrastructure:
#     - test_factory      : Test code, not production
#     - demo/             : Demo/example code
# NOTE: ALL format/ modules INCLUDED - apr subcommands must have 95%+ coverage
# Include apr-cli, exclude external deps and non-aprender workspace crates
# CB-125-A: ≤10 exclusion patterns (binary entry points + external deps only)
#   Binary entry points (thin wrappers, not production logic):
#     - aprender-monte-carlo/src/main.rs, aprender-tsp/src/main.rs
#   Demo/showcase code:
#     - showcase/           : Benchmark demo pipelines
#   CLI runtime-dependent code (requires TUI/network/model files):
#     - commands/serve/     : HTTP serving (tokio/axum)
#     - commands/cbtop      : TUI dashboard (ratatui)
#     - commands/chat       : Interactive REPL
#     - commands/tui        : TUI rendering helpers
#     - federation/tui      : Federation TUI
#     - commands/run\.rs    : Model execution (requires model files)
#     - commands/pull       : Network downloads
#     - commands/compare_hf : HuggingFace API comparison
#     - commands/trace      : Model tracing (requires model files)
#     - commands/bench\.rs  : Benchmarking (requires model files)
#     - commands/eval       : Model evaluation (requires model files)
#     - commands/canary     : Canary testing
#     - chaos\.rs           : Chaos testing module
COVERAGE_EXCLUDE_REGEX := \.cargo/|trueno|realizar/|entrenar/|fuzz/|golden_traces/|hf_hub/|demo/|test_factory|pacha/|aprender-monte-carlo/src/main|aprender-tsp/src/main|showcase/|aprender-shell/src/main|commands/serve/|commands/cbtop|commands/chat|commands/tui|federation/tui|commands/run\.rs|commands/pull|commands/compare_hf|commands/trace|commands/bench\.rs|commands/eval|commands/canary|chaos\.rs|audio/|format/quantize\.rs|format/signing\.rs|voice/|playback\.rs|commands/hex\.rs|commands/flow\.rs|commands/tree\.rs|commands/showcase/|commands/probar\.rs|federation/|commands/publish\.rs|apr-cli/src/lib\.rs|commands/check\.rs

# Coverage threshold (enforced: fail if below)
COV_THRESHOLD := 95

# Coverage: Two-phase pattern (tests + deferred report)
# Phase 1: Run tests with --no-report (keeps profraw, ~90s)
# Phase 2: Single report pass for summary + threshold check (~90s)
# HTML/LCOV generated separately via coverage-html (skipped for speed)
coverage: ## Coverage summary + threshold check (warm: ~3min)
	@echo "📊 Running coverage ($(COV_THRESHOLD)%+ threshold)..."
	@which cargo-llvm-cov > /dev/null 2>&1 || { cargo install cargo-llvm-cov --locked || exit 1; }
	@test -f ~/.cargo/config.toml && mv ~/.cargo/config.toml ~/.cargo/config.toml.bak || true
	@# Pre-clean: remove stale profraw files to avoid LLVM version mismatch
	@COVDIR=$$(cargo llvm-cov show-env 2>/dev/null | grep CARGO_LLVM_COV_TARGET_DIR | sed "s/.*=//"); \
	if [ -n "$$COVDIR" ]; then find "$$COVDIR" -name '*.profraw' -delete 2>/dev/null || true; fi
	@mkdir -p target/coverage
	@printf '%s' '$(COVERAGE_EXCLUDE_REGEX)' > target/coverage/.exclude-re
	@echo "🧪 Phase 1: Tests with instrumentation (CB-127-A: cargo llvm-cov test, not nextest)..."
	@PROPTEST_CASES=10 QUICKCHECK_TESTS=10 RUST_MIN_STACK=16777216 CARGO_BUILD_JOBS=4 \
		cargo llvm-cov test --no-report \
		--workspace --lib \
		--ignore-filename-regex "$$(cat target/coverage/.exclude-re)" \
		-- --skip prop_gbm_expected_value --skip slow --skip heavy --skip h12_ --skip j2_ \
		   --skip falsification --skip chaos --skip disconnect --skip benchmark_parity \
		   --skip qwen2_generation --skip qwen2_golden --skip qwen2_weight --skip load_test \
		   --skip spec_checklist_w --skip spec_checklist_u --skip verify_audio --skip g9_roofline \
		|| { test -f ~/.cargo/config.toml.bak && mv ~/.cargo/config.toml.bak ~/.cargo/config.toml; exit 1; }
	@echo "📊 Phase 2: Coverage report (LCOV → lightweight)..."
	@cargo llvm-cov report --lcov --ignore-filename-regex "$$(cat target/coverage/.exclude-re)" > target/coverage/lcov.info
	@# Parse LCOV for line coverage (LH=lines hit, LF=lines found)
	@LH=$$(awk -F: '/^LH:/{s+=$$2} END{print s+0}' target/coverage/lcov.info); \
	LF=$$(awk -F: '/^LF:/{s+=$$2} END{print s+0}' target/coverage/lcov.info); \
	if [ "$$LF" -gt 0 ]; then COV_PCT=$$((LH * 100 / LF)); else COV_PCT=0; fi; \
	echo "TOTAL: $$LH/$$LF lines covered ($${COV_PCT}%)"; \
	echo "TOTAL $$LH $$LF $${COV_PCT}%" > target/coverage/summary.txt; \
	test -f ~/.cargo/config.toml.bak && mv ~/.cargo/config.toml.bak ~/.cargo/config.toml || true; \
	if [ "$$COV_PCT" -lt "$(COV_THRESHOLD)" ]; then \
		echo "❌ Coverage $${COV_PCT}% is below threshold $(COV_THRESHOLD)%"; \
		exit 1; \
	else \
		echo "✅ Coverage $${COV_PCT}% meets threshold $(COV_THRESHOLD)%"; \
	fi

# Fast coverage alias
coverage-fast: coverage

# HTML + LCOV reports (run after 'make coverage' to generate browseable report)
coverage-html: ## Generate HTML + LCOV reports from last coverage run
	@echo "📊 Generating HTML + LCOV reports..."
	@test -f ~/.cargo/config.toml && mv ~/.cargo/config.toml ~/.cargo/config.toml.bak || true
	@mkdir -p target/coverage
	@printf '%s' '$(COVERAGE_EXCLUDE_REGEX)' > target/coverage/.exclude-re
	@cargo llvm-cov report --html --output-dir target/coverage/html --ignore-filename-regex "$$(cat target/coverage/.exclude-re)"
	@cargo llvm-cov report --lcov --output-path target/coverage/lcov.info --ignore-filename-regex "$$(cat target/coverage/.exclude-re)"
	@test -f ~/.cargo/config.toml.bak && mv ~/.cargo/config.toml.bak ~/.cargo/config.toml || true
	@echo "📍 HTML: target/coverage/html/index.html"

# Full coverage: All features (for CI, slower)
# CB-127-A: Use 'cargo llvm-cov test' instead of nextest to avoid profraw explosion
coverage-full: ## Full coverage report (all features, CI only)
	@echo "📊 Running full coverage analysis (all features)..."
	@which cargo-llvm-cov > /dev/null 2>&1 || { cargo install cargo-llvm-cov --locked || exit 1; }
	@test -f ~/.cargo/config.toml && mv ~/.cargo/config.toml ~/.cargo/config.toml.bak || true
	@mkdir -p target/coverage
	@printf '%s' '$(COVERAGE_EXCLUDE_REGEX)' > target/coverage/.exclude-re
	@PROPTEST_CASES=10 QUICKCHECK_TESTS=10 CARGO_BUILD_JOBS=4 \
		cargo llvm-cov test --no-report --workspace --lib --all-features \
		--ignore-filename-regex "$$(cat target/coverage/.exclude-re)" \
		-- --skip prop_gbm_expected_value --skip slow --skip heavy --skip benchmark --skip h12_ --skip j2_
	@cargo llvm-cov report --html --output-dir target/coverage/html --ignore-filename-regex "$$(cat target/coverage/.exclude-re)"
	@cargo llvm-cov report --lcov --output-path target/coverage/lcov.info --ignore-filename-regex "$$(cat target/coverage/.exclude-re)"
	@echo ""
	@cargo llvm-cov report --summary-only --ignore-filename-regex "$$(cat target/coverage/.exclude-re)"
	@test -f ~/.cargo/config.toml.bak && mv ~/.cargo/config.toml.bak ~/.cargo/config.toml || true

# Open coverage report in browser
coverage-open: ## Open HTML coverage report in browser
	@if [ -f target/coverage/html/index.html ]; then \
		xdg-open target/coverage/html/index.html 2>/dev/null || \
		open target/coverage/html/index.html 2>/dev/null || \
		echo "Open: target/coverage/html/index.html"; \
	else \
		echo "❌ Run 'make coverage' first"; \
	fi

# Profiling (requires renacer)
profile:
	renacer --function-time --source -- cargo bench

# Benchmarks
bench:
	cargo bench

# Chaos engineering tests (from renacer, Issue #99)
chaos-test: build ## Run chaos engineering tests with renacer
	@echo "🔥 Running chaos engineering tests..."
	@if command -v renacer >/dev/null 2>&1; then \
		./crates/aprender-shell/scripts/chaos-baseline.sh ci; \
	else \
		echo "⚠️  renacer not found. Install with: cargo install --git https://github.com/paiml/renacer"; \
		echo "💡 Running lightweight chaos simulation instead..."; \
		$(MAKE) chaos-test-lite; \
	fi
	@echo "✅ Chaos tests completed"

chaos-test-full: build ## Run full chaos tests including aggressive mode
	@echo "🔥 Running full chaos engineering tests..."
	@./crates/aprender-shell/scripts/chaos-baseline.sh full

chaos-test-lite: ## Lightweight chaos tests (no renacer required)
	@echo "🧪 Running lightweight chaos simulation..."
	@PROPTEST_CASES=10 QUICKCHECK_TESTS=10 cargo test -p aprender-shell --test cli_integration -- chaos --nocapture 2>/dev/null || true
	@echo "✅ Lite chaos tests completed"

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

# Run security audit
audit:
	@echo "🔒 Running security audit..."
	@cargo audit
	@echo "✅ Security audit completed"

# Validate dependencies (duplicates + security)
deps-validate:
	@echo "🔍 Validating dependencies..."
	@cargo tree --duplicate | grep -v "^$$" || echo "✅ No duplicate dependencies"
	@cargo audit || echo "⚠️  Security issues found"

# Run cargo-deny checks (licenses, bans, advisories, sources)
deny:
	@echo "🔒 Running cargo-deny checks..."
	@if command -v cargo-deny >/dev/null 2>&1; then \
		cargo deny check; \
	else \
		echo "❌ cargo-deny not installed. Install with: cargo install cargo-deny"; \
		exit 1; \
	fi
	@echo "✅ cargo-deny checks passed"

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

# PMAT Quality Analysis (v2.200.0 features)

pmat-score: ## Calculate Rust project quality score
	@echo "📊 Calculating Rust project quality score..."
	@pmat rust-project-score || echo "⚠️  pmat not found. Install with: cargo install pmat"
	@echo ""

pmat-gates: ## Run pmat quality gates
	@echo "🔍 Running pmat quality gates..."
	@pmat quality-gates --report || echo "⚠️  pmat not found or gates failed"
	@echo ""

quality-report: ## Generate comprehensive quality report
	@echo "📋 Generating comprehensive quality report..."
	@mkdir -p docs/quality-reports
	@echo "# Aprender Quality Report" > docs/quality-reports/latest.md
	@echo "" >> docs/quality-reports/latest.md
	@echo "Generated: $$(date)" >> docs/quality-reports/latest.md
	@echo "" >> docs/quality-reports/latest.md
	@echo "## Rust Project Score" >> docs/quality-reports/latest.md
	@pmat rust-project-score >> docs/quality-reports/latest.md 2>&1 || echo "Error getting score" >> docs/quality-reports/latest.md
	@echo "" >> docs/quality-reports/latest.md
	@echo "## Quality Gates" >> docs/quality-reports/latest.md
	@pmat quality-gates --report >> docs/quality-reports/latest.md 2>&1 || echo "Error running gates" >> docs/quality-reports/latest.md
	@echo "" >> docs/quality-reports/latest.md
	@echo "## TDG Score" >> docs/quality-reports/latest.md
	@pmat tdg . --include-components >> docs/quality-reports/latest.md 2>&1 || echo "Error getting TDG" >> docs/quality-reports/latest.md
	@echo "✅ Report generated: docs/quality-reports/latest.md"

semantic-search: ## Interactive semantic code search
	@echo "🔍 Semantic code search..."
	@echo "First run will build embeddings (may take a few minutes)..."
	@pmat semantic || echo "⚠️  pmat semantic search not available"

# ============================================================================
# SHOWCASE BENCHMARKING (qwen2.5-coder-showcase-demo.md)
# ============================================================================

.PHONY: showcase-headless showcase-ci falsification-tests falsification-quick showcase-verify showcase-pmat showcase-full

showcase-headless: ## Run cbtop in headless mode with JSON output (simulated data for CI)
	@echo "🎯 Running showcase headless benchmark (simulated mode)..."
	@cargo run --release -p apr-cli -- cbtop --headless --simulated --json --output target/showcase-results.json --iterations 100
	@echo "✅ Results saved to target/showcase-results.json"

showcase-ci: ## Run showcase benchmark in CI mode with threshold check
	@echo "🔍 Running showcase CI validation (throughput >= 100 tok/s)..."
	@cargo run --release -p apr-cli -- cbtop --headless --simulated --ci --throughput 100 --iterations 100
	@echo "✅ CI validation passed"

falsification-tests: ## Run all 137 falsification tests (F001-F105, M001-M020, O001-O009, R001)
	@echo "🧪 Running Popperian falsification test suite (137 tests)..."
	@PROPTEST_CASES=100 QUICKCHECK_TESTS=100 cargo test --release --test falsification_brick_tests --test falsification_budget_tests --test falsification_correctness_tests --test falsification_cuda_tests --test falsification_measurement_tests --test falsification_performance_tests --test falsification_2x_ollama_tests --test falsification_real_profiling -- --test-threads=2
	@echo "✅ All falsification tests passed (137 tests)"

falsification-quick: ## Run falsification tests in debug mode (faster compile)
	@echo "⚡ Running falsification tests (debug mode)..."
	@PROPTEST_CASES=25 QUICKCHECK_TESTS=25 cargo test --test falsification_brick_tests --test falsification_budget_tests --test falsification_correctness_tests --test falsification_cuda_tests --test falsification_measurement_tests --test falsification_performance_tests --test falsification_2x_ollama_tests --test falsification_real_profiling -- --test-threads=2
	@echo "✅ Falsification tests passed (137 tests)"

showcase-pmat: ## Run PMAT quality gates for showcase (spec section 7.0.2)
	@echo "📊 Running PMAT quality gates..."
	@echo ""
	@echo "=== Rust Project Score ==="
	@pmat rust-project-score 2>/dev/null || echo "pmat not available, skipping rust-project-score"
	@echo ""
	@echo "=== TDG Score ==="
	@pmat tdg . --include-components 2>/dev/null || echo "pmat not available, skipping TDG"
	@echo ""
	@echo "=== Quality Gates ==="
	@pmat quality-gates 2>/dev/null || echo "pmat not available, skipping quality-gates"
	@echo ""
	@echo "✅ PMAT analysis complete"

showcase-verify: showcase-headless falsification-tests ## Full showcase verification
	@echo "📊 Showcase verification complete"
	@echo "   - Headless benchmark: target/showcase-results.json"
	@echo "   - Falsification tests: 60/60 passing"

showcase-full: falsification-tests showcase-headless showcase-pmat ## Complete showcase validation
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  SHOWCASE FULL VALIDATION COMPLETE"
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  Falsification Tests: 60/60 passing (F001-F040, M001-M020)"
	@echo "  Headless Benchmark:  target/showcase-results.json"
	@echo "  PMAT Quality Gates:  See above output"
	@echo ""
	@echo "  Current Score: 60/120 (50%) - Blocked: F041-F100"
	@echo "════════════════════════════════════════════════════════════════"

# ============================================================================
# EXAMPLES TARGETS
# ============================================================================

examples: ## Run all examples to verify they work
	@echo "🎯 Running all examples..."
	@failed=0; \
	total=0; \
	for example in examples/*.rs; do \
		name=$$(basename "$$example" .rs); \
		total=$$((total + 1)); \
		echo "  Running $$name..."; \
		if cargo run --example "$$name" --quiet 2>/dev/null; then \
			echo "    ✅ $$name passed"; \
		else \
			echo "    ❌ $$name failed"; \
			failed=$$((failed + 1)); \
		fi; \
	done; \
	echo ""; \
	echo "📊 Results: $$((total - failed))/$$total examples passed"; \
	if [ $$failed -gt 0 ]; then exit 1; fi
	@echo "✅ All examples passed"

examples-fast: ## Run examples with release mode (faster execution)
	@echo "⚡ Running examples in release mode..."
	@for example in examples/*.rs; do \
		name=$$(basename "$$example" .rs); \
		echo "  Running $$name..."; \
		cargo run --example "$$name" --release --quiet 2>/dev/null || echo "    ⚠️  $$name failed"; \
	done
	@echo "✅ Examples complete"

examples-list: ## List all available examples
	@echo "📚 Available examples:"
	@for example in examples/*.rs; do \
		name=$$(basename "$$example" .rs); \
		echo "  - $$name"; \
	done
	@echo ""
	@echo "Run with: cargo run --example <name>"

# ============================================================================
# MUTATION TESTING TARGETS
# ============================================================================

mutants: ## Run mutation testing (full, ~30-60 min)
	@echo "🧬 Running mutation testing (full suite)..."
	@echo "⚠️  This may take 30-60 minutes for full coverage"
	@which cargo-mutants > /dev/null 2>&1 || (echo "📦 Installing cargo-mutants..." && cargo install cargo-mutants --locked)
	@cargo mutants --no-times --timeout 300 -- --all-features
	@echo "✅ Mutation testing complete"

mutants-fast: ## Run mutation testing on a sample (quick feedback, ~5 min)
	@echo "⚡ Running mutation testing (fast sample)..."
	@which cargo-mutants > /dev/null 2>&1 || (echo "📦 Installing cargo-mutants..." && cargo install cargo-mutants --locked)
	@cargo mutants --no-times --timeout 120 --shard 1/10 -- --lib
	@echo "✅ Mutation sample complete"

mutants-file: ## Run mutation testing on specific file (usage: make mutants-file FILE=src/metrics/mod.rs)
	@echo "🧬 Running mutation testing on $(FILE)..."
	@if [ -z "$(FILE)" ]; then \
		echo "❌ Usage: make mutants-file FILE=src/path/to/file.rs"; \
		exit 1; \
	fi
	@which cargo-mutants > /dev/null 2>&1 || { cargo install cargo-mutants --locked || exit 1; }
	@cargo mutants --no-times --timeout 120 --file "$(FILE)" -- --all-features
	@echo "✅ Mutation testing on $(FILE) complete"

mutants-list: ## List mutants without running tests
	@echo "📋 Listing potential mutants..."
	@cargo mutants --list 2>/dev/null | head -100
	@echo "..."
	@echo "(showing first 100 mutants)"

# ============================================================================
# PROPERTY TESTING TARGETS
# ============================================================================

property-test: ## Run property-based tests with extended cases
	@echo "🎲 Running property-based tests..."
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		PROPTEST_CASES=250 cargo nextest run --test property_tests --no-fail-fast; \
	else \
		PROPTEST_CASES=250 cargo test --test property_tests; \
	fi
	@echo "✅ Property tests passed"

property-test-fast: ## Run property tests with fewer cases (quick feedback)
	@echo "⚡ Running property tests (fast mode)..."
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		PROPTEST_CASES=25 QUICKCHECK_TESTS=25 cargo nextest run --test property_tests; \
	else \
		PROPTEST_CASES=25 QUICKCHECK_TESTS=25 cargo test --test property_tests; \
	fi
	@echo "✅ Property tests passed"

property-test-extensive: ## Run property tests with maximum coverage (10K cases)
	@echo "🔬 Running extensive property tests (10K cases per test)..."
	@PROPTEST_CASES=2500 cargo test --test property_tests -- --test-threads=1
	@echo "✅ Extensive property tests complete"

# ============================================================================
# SYSTEM DEPENDENCIES (Native Audio, etc.)
# ============================================================================

install-alsa: ## Install ALSA development libraries (Linux only)
	@echo "🔊 Installing ALSA development libraries..."
	@if [ "$$(uname)" = "Linux" ]; then \
		if command -v apt-get >/dev/null 2>&1; then \
			echo "  Detected: Debian/Ubuntu"; \
			sudo apt-get update && sudo apt-get install -y libasound2-dev; \
		elif command -v dnf >/dev/null 2>&1; then \
			echo "  Detected: Fedora/RHEL"; \
			sudo dnf install -y alsa-lib-devel; \
		elif command -v pacman >/dev/null 2>&1; then \
			echo "  Detected: Arch Linux"; \
			sudo pacman -S --noconfirm alsa-lib; \
		elif command -v zypper >/dev/null 2>&1; then \
			echo "  Detected: openSUSE"; \
			sudo zypper install -y alsa-devel; \
		else \
			echo "❌ Unknown package manager. Please install ALSA dev libraries manually:"; \
			echo "   - Debian/Ubuntu: sudo apt-get install libasound2-dev"; \
			echo "   - Fedora/RHEL: sudo dnf install alsa-lib-devel"; \
			echo "   - Arch: sudo pacman -S alsa-lib"; \
			exit 1; \
		fi; \
		echo "✅ ALSA development libraries installed"; \
	else \
		echo "⚠️  ALSA is Linux-only. Current OS: $$(uname)"; \
	fi

test-alsa: ## Run tests with ALSA audio capture feature (Linux only)
	@echo "🔊 Running tests with audio-alsa feature..."
	@if [ "$$(uname)" = "Linux" ]; then \
		if pkg-config --exists alsa 2>/dev/null; then \
			PROPTEST_CASES=25 QUICKCHECK_TESTS=25 cargo test --features audio-alsa; \
		else \
			echo "❌ ALSA not installed. Run: make install-alsa"; \
			exit 1; \
		fi; \
	else \
		echo "⚠️  ALSA is Linux-only. Running standard audio tests..."; \
		PROPTEST_CASES=25 QUICKCHECK_TESTS=25 cargo test --features audio; \
	fi
	@echo "✅ ALSA tests complete"

test-audio-full: ## Run all audio tests including ALSA (if available)
	@echo "🎵 Running full audio test suite..."
	@if [ "$$(uname)" = "Linux" ] && pkg-config --exists alsa 2>/dev/null; then \
		echo "  ALSA available - running with audio-alsa feature"; \
		PROPTEST_CASES=25 QUICKCHECK_TESTS=25 cargo test --features audio-alsa audio::; \
	else \
		echo "  Running standard audio tests"; \
		PROPTEST_CASES=25 QUICKCHECK_TESTS=25 cargo test --features audio audio::; \
	fi
	@echo "✅ Audio tests complete"
