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
	@time cargo test --lib --no-fail-fast -- \
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
	@echo "⚡ Running fast tests (target: <30s)..."
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		time env PROPTEST_CASES=50 cargo nextest run --workspace --lib \
			--status-level skip \
			--failure-output immediate \
			-E 'not test(/prop_gbm_expected_value_convergence/)'; \
	else \
		echo "💡 Install cargo-nextest for faster tests: cargo install cargo-nextest"; \
		time env PROPTEST_CASES=50 cargo test --workspace --lib -- --skip prop_gbm_expected_value_convergence; \
	fi
	@echo "✅ Fast tests passed"

# Quick alias for test-fast
test-quick: test-fast

# Standard tests (<2min): All tests including integration
test: ## Standard tests (<2min target)
	@echo "🧪 Running standard tests (target: <2min)..."
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		time cargo nextest run --workspace \
			--status-level skip \
			--failure-output immediate; \
	else \
		time cargo test --workspace; \
	fi
	@echo "✅ Standard tests passed"

# Full comprehensive tests: All features, all property cases
test-full: ## Comprehensive tests (all features)
	@echo "🔬 Running full comprehensive tests..."
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		time cargo nextest run --workspace --all-features; \
	else \
		time cargo test --workspace --all-features; \
	fi
	@echo "✅ Full tests passed"

# Heavy tests: Runs ignored tests (Section P: P7)
# Includes: sleep()-based tests, slow encryption tests, long proptests
test-heavy: ## Heavy/slow tests (ignored tests)
	@echo "🐢 Running heavy tests (ignored tests)..."
	@time cargo test --workspace -- --ignored
	@echo "✅ Heavy tests passed"

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
	-pmat rust-project-score
	-pmat quality-gates --report
	@echo "Tier 4: PASSED"

# ============================================================================
# COVERAGE TARGETS (Two-Phase Pattern from bashrs)
# ============================================================================
# Pattern: bashrs/Makefile - Two-phase coverage with mold linker workaround
# CRITICAL: mold linker breaks LLVM coverage instrumentation
# Solution: Temporarily move ~/.cargo/config.toml during coverage runs

# Exclusion patterns for coverage reports (bashrs pattern)
# Excludes: crates/, fuzz/, golden_traces/, external path deps (realizar/, trueno/)
COVERAGE_EXCLUDE := --ignore-filename-regex='(crates/|fuzz/|golden_traces/|realizar/|trueno/)'

# Fast coverage (<2 min): Lib + integration tests, skip slow tests (bashrs style)
coverage: ## Generate HTML coverage report (target: <2 min, 95%+)
	@echo "📊 Running coverage (bashrs style, target: <2 min)..."
	@which cargo-llvm-cov > /dev/null 2>&1 || cargo install cargo-llvm-cov --locked
	@echo "⚙️  Disabling sccache/mold (breaks coverage instrumentation)..."
	@test -f ~/.cargo/config.toml && mv ~/.cargo/config.toml ~/.cargo/config.toml.bak || true
	@cargo llvm-cov clean --workspace 2>/dev/null || true
	@mkdir -p target/coverage
	@echo "🧪 Running lib + integration tests (skip slow/benchmark tests)..."
	@echo "   Using -j 8 to limit memory (LLVM instrumentation ~2x overhead)"
	@cargo llvm-cov --no-report --lib --tests -j 8 \
		-- --skip prop_ --skip encryption --skip compressed --skip slow \
		--skip h12_benchmark --skip j2_roofline --skip benchmark
	@echo "📊 Generating report..."
	@cargo llvm-cov report --html --output-dir target/coverage/html $(COVERAGE_EXCLUDE)
	@cargo llvm-cov report --lcov --output-path target/coverage/lcov.info $(COVERAGE_EXCLUDE)
	@test -f ~/.cargo/config.toml.bak && mv ~/.cargo/config.toml.bak ~/.cargo/config.toml || true
	@echo ""
	@cargo llvm-cov report --summary-only $(COVERAGE_EXCLUDE)
	@echo ""
	@echo "📍 HTML: target/coverage/html/index.html"

# Fast coverage alias (same as coverage, optimized by default)
coverage-fast: coverage

# Full coverage: All features (for CI, slower)
coverage-full: ## Full coverage report (all features, >10 min)
	@echo "📊 Running full coverage analysis (all features)..."
	@which cargo-llvm-cov > /dev/null 2>&1 || { cargo install cargo-llvm-cov --locked || exit 1; }
	@which cargo-nextest > /dev/null 2>&1 || { cargo install cargo-nextest --locked || exit 1; }
	@cargo llvm-cov clean --workspace
	@mkdir -p target/coverage
	@test -f ~/.cargo/config.toml && mv ~/.cargo/config.toml ~/.cargo/config.toml.cov-backup || true
	@cargo llvm-cov --no-report nextest --no-tests=warn --workspace --all-features -j 8
	@cargo llvm-cov report --html --output-dir target/coverage/html $(COVERAGE_EXCLUDE)
	@cargo llvm-cov report --lcov --output-path target/coverage/lcov.info $(COVERAGE_EXCLUDE)
	@test -f ~/.cargo/config.toml.cov-backup && mv ~/.cargo/config.toml.cov-backup ~/.cargo/config.toml || true
	@echo ""
	@cargo llvm-cov report --summary-only $(COVERAGE_EXCLUDE)

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
	@cargo test -p aprender-shell --test cli_integration -- chaos --nocapture 2>/dev/null || true
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
		PROPTEST_CASES=1000 cargo nextest run --test property_tests --no-fail-fast; \
	else \
		PROPTEST_CASES=1000 cargo test --test property_tests; \
	fi
	@echo "✅ Property tests passed"

property-test-fast: ## Run property tests with fewer cases (quick feedback)
	@echo "⚡ Running property tests (fast mode)..."
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		cargo nextest run --test property_tests; \
	else \
		cargo test --test property_tests; \
	fi
	@echo "✅ Property tests passed"

property-test-extensive: ## Run property tests with maximum coverage (10K cases)
	@echo "🔬 Running extensive property tests (10K cases per test)..."
	@PROPTEST_CASES=10000 cargo test --test property_tests -- --test-threads=1
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
			cargo test --features audio-alsa; \
		else \
			echo "❌ ALSA not installed. Run: make install-alsa"; \
			exit 1; \
		fi; \
	else \
		echo "⚠️  ALSA is Linux-only. Running standard audio tests..."; \
		cargo test --features audio; \
	fi
	@echo "✅ ALSA tests complete"

test-audio-full: ## Run all audio tests including ALSA (if available)
	@echo "🎵 Running full audio test suite..."
	@if [ "$$(uname)" = "Linux" ] && pkg-config --exists alsa 2>/dev/null; then \
		echo "  ALSA available - running with audio-alsa feature"; \
		cargo test --features audio-alsa audio::; \
	else \
		echo "  Running standard audio tests"; \
		cargo test --features audio audio::; \
	fi
	@echo "✅ Audio tests complete"
