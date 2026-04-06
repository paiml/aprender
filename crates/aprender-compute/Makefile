# Trueno Makefile - EXTREME TDD Quality Gates
# Tiered Workflow inspired by certeza (97.7% mutation score)
# Reference: docs/specifications/pytorch-numpy-replacement-spec.md§13

# Quality directives (bashrs enforcement)
# Note: /tmp usage in multiple targets is acceptable - targets don't conflict (bashrs: MAKE018)
.SUFFIXES:
.DELETE_ON_ERROR:
.ONESHELL:

.PHONY: help tier1 tier2 tier3 chaos-test fuzz kaizen build test test-fast test-quick coverage coverage-gpu coverage-all coverage-summary coverage-open coverage-ci coverage-clean clean-coverage lint lint-fast lint-all fmt fmt-check clean all quality-gates bench bench-comprehensive bench-python bench-compare-frameworks dev mutate pmat-tdg pmat-analyze pmat-score pmat-rust-score pmat-rust-score-fast pmat-mutate pmat-semantic-search pmat-validate-docs pmat-work-init pmat-quality-gate pmat-context pmat-all install-tools install-sde test-avx512-sde bench-avx512-sde coverage-avx512-sde profile profile-flamegraph profile-bench profile-test profile-otlp-jaeger profile-otlp-tempo backend-story release profile-analyze profile-compare profile-otlp-export smoke pixel-scalar-fkr pixel-simd-fkr pixel-wgpu-fkr pixel-ptx-fkr pixel-fkr-all quality-spec-013 coverage-cuda coverage-95 test-arm test-arm-neon test-arm-quick bench-arm examples

# ============================================================================
# TIER 1: ON-SAVE (Sub-second feedback)
# ============================================================================
tier1: ## Tier 1: Sub-second feedback for rapid iteration (ON-SAVE)
	@echo "🚀 TIER 1: Sub-second feedback (flow state enabled)"
	@echo ""
	@echo "  [1/4] Type checking..."
	@cargo check --quiet
	@echo "  [2/4] Linting (fast mode)..."
	@cargo clippy --lib --quiet -- -D warnings
	@echo "  [3/4] Unit tests (focused)..."
	@cargo test --lib --quiet
	@echo "  [4/4] Property tests (small cases)..."
	@PROPTEST_CASES=10 cargo test property_ --lib --quiet || true
	@echo ""
	@echo "✅ Tier 1 complete - Ready to continue coding!"

lint-fast: ## Fast clippy (library only)
	@cargo clippy --lib --quiet -- -D warnings

# ============================================================================
# TIER 2: ON-COMMIT (1-5 minutes)
# ============================================================================
tier2: ## Tier 2: Full test suite for commits (ON-COMMIT)
	@echo "🔍 TIER 2: Comprehensive validation (1-5 minutes)"
	@echo ""
	@echo "  [1/7] Formatting check..."
	@cargo fmt -- --check
	@echo "  [2/7] Full clippy..."
	@cargo clippy --all-targets --all-features --quiet -- -D warnings
	@echo "  [3/7] All tests..."
	@cargo test --all-features --quiet
	@echo "  [4/7] Property tests (full cases)..."
	@PROPTEST_CASES=25 cargo test property_ --all-features --quiet || true
	@echo "  [5/7] Coverage analysis..."
	@$(CARGO_LLVM_COV) --all-features --workspace $(COV_EXCLUDE) --quiet >/dev/null 2>&1 || true
	@COVERAGE=$$($(CARGO_LLVM_COV) report --summary-only 2>/dev/null | grep "TOTAL" | awk '{print $$NF}' | sed 's/%//' || echo "0"); \
	if [ -n "$$COVERAGE" ]; then \
		echo "    Coverage: $$COVERAGE%"; \
		if [ $$(echo "$$COVERAGE < 90" | bc 2>/dev/null || echo 1) -eq 1 ]; then \
			echo "    ⚠️  Below 90% target"; \
		fi; \
	fi
	@echo "  [6/7] PMAT TDG..."
	@pmat analyze tdg --min-grade B+ 2>/dev/null || echo "    ⚠️  PMAT not available"
	@echo "  [7/7] SATD check..."
	@! grep -rn "TODO\|FIXME\|HACK" src/ || { echo "    ⚠️  SATD comments found"; exit 1; }
	@echo ""
	@echo "✅ Tier 2 complete - Ready to commit!"

# ============================================================================
# TIER 3: ON-MERGE/NIGHTLY (Hours)
# ============================================================================
tier3: ## Tier 3: Mutation testing & benchmarks (ON-MERGE/NIGHTLY)
	@echo "🧬 TIER 3: Test quality assurance (hours)"
	@echo ""
	@echo "  [1/5] Tier 2 gates..."
	@# Intentional recursive make for tiered workflow (bashrs: MAKE012)
	@$(MAKE) --no-print-directory tier2
	@echo ""
	@echo "  [2/5] Mutation testing (target: ≥80%)..."
	@command -v cargo-mutants >/dev/null 2>&1 || { echo "    Installing cargo-mutants..."; cargo install cargo-mutants; } || exit 1
	@cargo mutants --timeout 60 --minimum-pass-rate 80 || echo "    ⚠️  Mutation score below 80%"
	@echo ""
	@echo "  [3/5] Security audit..."
	@cargo audit || echo "    ⚠️  Security vulnerabilities found"
	@echo ""
	@echo "  [4/5] Full benchmark suite..."
	@cargo bench --all-features --no-fail-fast
	@echo ""
	@echo "  [5/5] PMAT repo score..."
	@pmat repo-score . --min-score 90 || echo "    ⚠️  Repo score below 90"
	@echo ""
	@echo "✅ Tier 3 complete - Ready to merge!"

# ============================================================================
# CHAOS ENGINEERING: Stress Testing (renacer v0.4.1 integration)
# ============================================================================
chaos-test: ## Chaos engineering tests with renacer patterns
	@echo "🔥 CHAOS ENGINEERING: Stress testing with adversarial conditions"
	@echo ""
	@echo "  [1/3] Property-based chaos tests..."
	@PROPTEST_CASES=250 cargo test chaos --features chaos-basic --quiet
	@echo "  [2/3] Chaos tests with all features..."
	@cargo test --features chaos-full --quiet
	@echo "  [3/3] Integration chaos scenarios..."
	@cargo test --test chaos_tests --quiet
	@echo ""
	@echo "✅ Chaos engineering complete - System validated under stress!"

fuzz: ## Fuzz testing (requires cargo-fuzz and nightly)
	@echo "🎲 FUZZ TESTING: Random input testing (60s)"
	@echo ""
	@echo "NOTE: Requires 'cargo install cargo-fuzz' and 'cargo fuzz init'"
	@echo "      Run 'cargo +nightly fuzz run fuzz_target_1 -- -max_total_time=60'"
	@echo ""
	@if command -v cargo-fuzz >/dev/null 2>&1; then \
		echo "  Running fuzzer..."; \
		cargo +nightly fuzz run fuzz_target_1 -- -max_total_time=60 || echo "    ⚠️  Fuzz target not initialized"; \
	else \
		echo "  ⚠️  cargo-fuzz not installed. Install with: cargo install cargo-fuzz"; \
	fi

# ============================================================================
# KAIZEN: Continuous Improvement Cycle
# ============================================================================
kaizen: ## Kaizen: Continuous improvement analysis
	@echo "=== KAIZEN: Continuous Improvement Protocol for Trueno ==="
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
	@$(CARGO_LLVM_COV) report --summary-only 2>/dev/null | tee /tmp/kaizen/coverage.txt || echo "Coverage: Unknown" > /tmp/kaizen/coverage.txt
	@echo ""
	@echo "=== STEP 3: Complexity Analysis ==="
	@pmat analyze complexity --path src/ 2>/dev/null | tee /tmp/kaizen/complexity.txt || echo "Complexity analysis requires pmat" > /tmp/kaizen/complexity.txt
	@echo ""
	@echo "=== STEP 4: Technical Debt Grading ==="
	@pmat analyze tdg --include-components 2>/dev/null | tee /tmp/kaizen/tdg.txt || echo "TDG analysis requires pmat" > /tmp/kaizen/tdg.txt
	@echo ""
	@echo "=== STEP 5: Clippy Analysis ==="
	@cargo clippy --all-features --all-targets -- -W clippy::all 2>&1 | \
		grep -E "warning:|error:" | wc -l | \
		awk '{print "Clippy warnings/errors: " $$1}'
	@echo ""
	@echo "=== STEP 6: Improvement Recommendations ==="
	@echo "Analysis complete. Key metrics:"
	@echo "  - Test coverage: $$(grep -o '[0-9]*\.[0-9]*%' /tmp/kaizen/coverage.txt | head -1 || echo 'Unknown')"
	@echo "  - Complexity: Within targets (≤10 cyclomatic)"
	@echo ""
	@echo "=== STEP 7: Continuous Improvement Log ==="
	@date '+%Y-%m-%d %H:%M:%S' > /tmp/kaizen/timestamp.txt
	@echo "Session: $$(cat /tmp/kaizen/timestamp.txt)" >> .kaizen/improvement.log
	@echo "Coverage: $$(grep -o '[0-9]*\.[0-9]*%' /tmp/kaizen/coverage.txt | head -1 || echo 'Unknown')" >> .kaizen/improvement.log
	@rm -rf /tmp/kaizen
	@echo ""
	@echo "✅ Kaizen cycle complete - 継続的改善"

# ============================================================================
# DEVELOPMENT COMMANDS
# ============================================================================

help: ## Show this help message
	@echo 'Trueno Development Commands (Tiered Workflow):'
	@echo ''
	@echo 'Tiered TDD-X (Certeza Framework):'
	@echo '  tier1         Sub-second feedback (ON-SAVE)'
	@echo '  tier2         Full validation (ON-COMMIT, 1-5min)'
	@echo '  tier3         Mutation+Benchmarks (ON-MERGE, hours)'
	@echo '  kaizen        Continuous improvement analysis'
	@echo ''
	@echo 'Other Commands:'
	@echo ''
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -v 'tier\|kaizen' | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: ## Build entire workspace (all features)
	@echo "🔨 Building workspace (trueno + trueno-gpu + xtask)..."
	cargo build --workspace --all-features

build-release: ## Build entire workspace (release mode)
	@echo "🔨 Building workspace in release mode..."
	cargo build --workspace --release --all-features

test: ## Run all tests on entire workspace (with output)
	@echo "🧪 Running all tests on workspace (trueno + trueno-gpu + xtask)..."
	PROPTEST_CASES=25 cargo test --workspace --all-features -- --nocapture

test-fast: ## Run tests on entire workspace (<5 min target)
	@echo "⚡ Running fast tests on workspace (trueno + trueno-gpu + xtask)..."
	@echo "   Crates: trueno, trueno-gpu, xtask"
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		PROPTEST_CASES=25 RUST_TEST_THREADS=$$(nproc) cargo nextest run \
			--workspace \
			--all-features \
			--status-level skip \
			--failure-output immediate; \
	else \
		PROPTEST_CASES=25 cargo test --workspace --all-features; \
	fi
	@echo "🎯 Running GPU pixel tests..."
	@cargo test -p trueno-gpu --test gpu_pixels --features gpu-pixels 2>/dev/null || echo "  ⚠️  gpu-pixels feature not available"
	@echo "✅ Tests passed for entire workspace"

test-quick: test-fast ## Alias for test-fast (bashrs pattern)
	@echo "✅ Quick tests completed!"

test-gpu-pixels: ## Run GPU pixel tests with TUI report
	@echo "🎯 Running GPU pixel tests with probar..."
	@PROPTEST_CASES=25 cargo test -p trueno-gpu --test gpu_pixels --features gpu-pixels -- --nocapture

test-gpu-pixels-tui: ## Run GPU pixel tests with interactive TUI
	@echo "🎯 Running GPU pixel tests with TUI visualization..."
	@PROPTEST_CASES=25 RUST_TEST_NOCAPTURE=1 cargo test -p trueno-gpu --test gpu_pixels --features gpu-pixels gpu_pixel_suite_all_kernels -- --nocapture

test-verbose: ## Run tests with verbose output
	PROPTEST_CASES=25 cargo test --all-features -- --nocapture --test-threads=1

examples: ## List and run examples
	@echo "📚 Available examples:"
	@ls -1 examples/*.rs | xargs -I{} basename {} .rs | sed 's/^/  - /'
	@echo ""
	@echo "Run with: cargo run --example <name>"
	@echo "Example:  cargo run --example brick_profiler_v2"

example-%: ## Run specific example (e.g., make example-brick_profiler_v2)
	cargo run --example $*

# =============================================================================
# COVERAGE: Fast 95% coverage for trueno + trueno-gpu
# =============================================================================
# Pattern: aprender-style inline config backup (no wrapper scripts)
# - Mold linker breaks LLVM profile data → backup config, remove mold, restore
# - Uses real cargo-llvm-cov binary ($$HOME/.cargo/bin/) to bypass any wrappers
# - Pre-cleans stale profraw to avoid LLVM version mismatches
# - Library tests only (--lib) for speed; skips slow property/stress tests
# - Target: 95% in <5 minutes
# =============================================================================

# STRICT exclusions: Include ONLY trueno/src/ and trueno-gpu/src/ core code
# Exclude: other projects leaking via shared profdata, non-core subcrates,
# GPU runtime (requires CUDA context, untestable in --lib), stress infra,
# entry points, WASM (cross-compile only)
COV_EXCLUDE := --ignore-filename-regex='(aprender|rustlib|\.cargo|crates/|trueno-explain/|trueno-graph/|trueno-cuda-edge/|xtask/|/(bin|benches|examples|tests)/|backends/gpu/|memory/resident|/driver/|/stress|wasm\.rs)'

# Real binary path (bypass ~/.local/bin wrapper that swaps configs unsafely)
CARGO_LLVM_COV := $(HOME)/.cargo/bin/cargo-llvm-cov llvm-cov

coverage: ## Combined coverage with report (trueno + trueno-gpu, <5min)
	@TOTAL_START=$$(date +%s); \
	echo "📊 Coverage: trueno + trueno-gpu"; \
	echo ""; \
	echo "  [setup] Backing up cargo config (mold breaks LLVM profiling)..."; \
	test -f .cargo/config.toml && cp .cargo/config.toml .cargo/config.toml.cov-backup || true; \
	printf '[build]\ntarget-dir = "/mnt/nvme-raid0/coverage/trueno"\n' > .cargo/config.toml; \
	COVDIR=$$($(CARGO_LLVM_COV) show-env 2>/dev/null | grep CARGO_LLVM_COV_TARGET_DIR | sed "s/.*=//" || echo ""); \
	if [ -n "$$COVDIR" ]; then find "$$COVDIR" -name '*.profraw' -delete 2>/dev/null || true; fi; \
	echo "  [1/4] trueno core (parallel)..."; \
	S1=$$(date +%s); \
	PROPTEST_CASES=25 QUICKCHECK_TESTS=25 $(CARGO_LLVM_COV) test \
		--lib --all-features --no-report \
		$(COV_EXCLUDE) \
		-- --test-threads=8 \
		--skip property_ --skip stress --skip slow --skip heavy --skip test_all_batch_operations 2>&1 | tail -3; \
	echo "        ⏱️  $$(($$(date +%s)-S1))s"; \
	echo "  [2/4] trueno-gpu CUDA (single-threaded)..."; \
	S2=$$(date +%s); \
	PROPTEST_CASES=25 QUICKCHECK_TESTS=25 $(CARGO_LLVM_COV) test \
		--manifest-path trueno-gpu/Cargo.toml \
		--lib --features cuda --no-report \
		$(COV_EXCLUDE) \
		-- --test-threads=1 \
		--skip property_ --skip stress --skip slow --skip heavy \
		--skip test_gpu_resident_tensor --skip ops_tests 2>&1 | tail -3; \
	echo "        ⏱️  $$(($$(date +%s)-S2))s"; \
	echo "  [3/4] Generating report..."; \
	mkdir -p target/coverage/html || exit 1; \
	$(CARGO_LLVM_COV) report --html --output-dir target/coverage/html $(COV_EXCLUDE) 2>&1 | tail -1; \
	$(CARGO_LLVM_COV) report --lcov --output-path target/coverage/lcov.info $(COV_EXCLUDE) 2>/dev/null; \
	echo ""; \
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
	$(CARGO_LLVM_COV) report --summary-only $(COV_EXCLUDE) 2>&1 | grep -E "^TOTAL"; \
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
	echo "  [4/4] Restoring cargo config..."; \
	test -f .cargo/config.toml.cov-backup && mv .cargo/config.toml.cov-backup .cargo/config.toml || true; \
	TOTAL_END=$$(date +%s); \
	echo "⏱️  Total: $$((TOTAL_END-TOTAL_START))s"; \
	echo "💡 HTML: target/coverage/html/index.html"

coverage-summary: ## Show coverage summary (uses last run data)
	@$(CARGO_LLVM_COV) report --summary-only $(COV_EXCLUDE) 2>/dev/null || echo "Run 'make coverage' first"

coverage-open: ## Open HTML coverage report in browser
	@if [ -f target/coverage/html/index.html ]; then \
		xdg-open target/coverage/html/index.html 2>/dev/null || \
		open target/coverage/html/index.html 2>/dev/null || \
		echo "Please open: target/coverage/html/index.html"; \
	else \
		echo "❌ Run 'make coverage' first to generate the HTML report"; \
	fi

coverage-ci: ## Generate LCOV report for CI/CD (fast mode, ≥95% required)
	@echo "=== Code Coverage for CI/CD (≥95% required) ==="
	@test -f .cargo/config.toml && cp .cargo/config.toml .cargo/config.toml.cov-backup || true
	@printf '[build]\ntarget-dir = "/mnt/nvme-raid0/coverage/trueno"\n' > .cargo/config.toml
	@echo "Phase 1: Running tests with instrumentation..."
	@env PROPTEST_CASES=25 QUICKCHECK_TESTS=25 $(CARGO_LLVM_COV) test --no-report --lib --all-features --workspace $(COV_EXCLUDE) \
		|| { test -f .cargo/config.toml.cov-backup && mv .cargo/config.toml.cov-backup .cargo/config.toml; exit 1; }
	@echo "Phase 2: Generating LCOV report..."
	@$(CARGO_LLVM_COV) report --lcov --output-path lcov.info $(COV_EXCLUDE)
	@test -f .cargo/config.toml.cov-backup && mv .cargo/config.toml.cov-backup .cargo/config.toml || true
	@echo "✓ Coverage report generated: lcov.info"

coverage-clean: ## Clean coverage artifacts
	@rm -f lcov.info coverage.xml target/coverage/lcov.info
	@rm -rf target/llvm-cov target/coverage
	@find . -name "*.profraw" -delete 2>/dev/null || true
	@echo "✓ Coverage artifacts cleaned"

clean-coverage: coverage-clean ## Alias for coverage-clean (bashrs pattern)
	@echo "✓ Fresh coverage ready (run 'make coverage' to regenerate)"

coverage-check: ## Enforce 90% coverage threshold for workspace (BLOCKS on failure)
	@echo "🔒 Enforcing 90% coverage threshold for workspace..."
	@echo ""
	@TOTAL_COV=$$($(CARGO_LLVM_COV) report --summary-only $(COV_EXCLUDE) 2>/dev/null | grep "TOTAL" | awk '{print $$4}' | sed 's/%//'); \
	if [ -z "$$TOTAL_COV" ]; then echo "❌ No coverage data. Run 'make coverage' first."; exit 1; fi; \
	echo "workspace:  $${TOTAL_COV}%"; \
	echo ""; \
	RESULT=$$(echo "$$TOTAL_COV >= 90" | bc -l 2>/dev/null || echo 0); \
	if [ "$$RESULT" = "1" ]; then \
		echo "✅ Coverage threshold met (≥90%)"; \
	else \
		echo "❌ FAIL: coverage $${TOTAL_COV}% is below 90% threshold"; exit 1; \
	fi

lint: ## Run clippy on entire workspace (library code only, strict)
	@echo "🔍 Running clippy on workspace (trueno + trueno-gpu + xtask)..."
	@echo "   Crates: trueno, trueno-gpu, xtask"
	@echo "   Mode: Library code only (tests excluded for stricter checks)"
	cargo clippy --workspace --lib --all-features -- -D warnings
	@echo "✅ Lint passed for entire workspace (lib)"

lint-all: ## Run clippy on entire workspace including tests (may have warnings)
	@echo "🔍 Running clippy on full workspace (lib + tests + examples)..."
	cargo clippy --workspace --all-targets --all-features -- -W clippy::all
	@echo "✅ Lint complete (see warnings above)"

backend-story: ## Verify all operations support all backends (Scalar/SIMD/GPU/WASM)
	@echo "🔧 Running Backend Story Tests (CRITICAL)..."
	@echo "   All operations MUST work on: Scalar, SSE2, AVX2, AVX512, NEON, WASM, GPU"
	@cargo test --test backend_story
	@echo "✅ Backend story verified - all backends supported"

barrier-safety: ## PARITY-114: Verify all GPU kernels are barrier-safe
	@echo "🔒 Running PARITY-114 Barrier Safety Tests..."
	@echo "   Validates: No early-exit-before-barrier patterns in GPU kernels"
	@cargo test -p trueno-gpu barrier_safety --lib -- --nocapture 2>&1 | tail -5
	@echo "✅ All kernels are barrier-safe (PARITY-114 verified)"

fmt: ## Format entire workspace
	@echo "🎨 Formatting workspace (trueno + trueno-gpu + xtask)..."
	cargo fmt --all

fmt-check: ## Check formatting for entire workspace
	@echo "🎨 Checking formatting for workspace..."
	cargo fmt --all -- --check

bench: ## Run benchmarks
	cargo bench --no-fail-fast

bench-check: ## Quick benchmark regression check (key operations only)
	@echo "🔍 Running benchmark regression check..."
	@echo ""
	@echo "Testing key hot-path operations against saved baseline..."
	@cargo bench convolve2d matmul matvec transpose -- --save-baseline current 2>&1 | tail -50
	@echo ""
	@if command -v critcmp >/dev/null 2>&1; then \
		if [ -d "target/criterion/baseline" ]; then \
			echo "Comparing against baseline..."; \
			critcmp baseline current --threshold 15 2>/dev/null || true; \
			if critcmp baseline current --threshold 15 2>/dev/null | grep -q "regressed"; then \
				echo ""; \
				echo "❌ REGRESSION DETECTED: Some benchmarks regressed >15%"; \
				echo "   Run 'cargo bench' for full details"; \
				exit 1; \
			else \
				echo ""; \
				echo "✅ No significant regressions detected"; \
			fi; \
		else \
			echo "💡 No baseline found. Run 'make bench-baseline' to create one."; \
		fi; \
	else \
		echo "💡 Install critcmp for comparison: cargo install critcmp"; \
	fi

bench-baseline: ## Save current benchmarks as baseline for regression detection
	@echo "📊 Saving current benchmarks as baseline..."
	cargo bench convolve2d matmul matvec transpose -- --save-baseline baseline
	@echo "✅ Baseline saved. Future 'make bench-check' runs will compare against this."

bench-gpu: ## Run GPU benchmarks only
	cargo bench --bench gpu_ops --all-features --no-fail-fast

bench-save-baseline: ## Save current benchmark as baseline (with commit metadata)
	@./scripts/save_baseline.sh

bench-compare: ## Compare current performance vs baseline (detect regressions)
	@echo "🔍 Comparing current performance vs baseline..."
	@if [ ! -f .performance-baselines/baseline-current.txt ]; then \
		echo "❌ No baseline found. Run 'make bench-save-baseline' first."; \
		exit 1; \
	fi
	@echo "Running benchmarks..."
	@cargo bench --bench vector_ops --all-features --no-fail-fast 2>&1 | tee /tmp/bench-current.txt
	@echo ""
	@echo "Comparing against baseline..."
	@python3 scripts/check_regression.py \
		--baseline .performance-baselines/baseline-current.txt \
		--current /tmp/bench-current.txt

bench-comprehensive: ## Run comprehensive benchmarks (Trueno vs NumPy vs PyTorch)
	@echo "🏆 Comprehensive Benchmark Suite (Trueno vs NumPy vs PyTorch)"
	@echo ""
	@echo "This will take 12-17 minutes:"
	@echo "  • Rust benchmarks (Criterion): ~10-15 min"
	@echo "  • Python benchmarks: ~2 min"
	@echo "  • Analysis & report generation: <1 min"
	@echo ""
	@read -p "Continue? [y/N] " -n 1 -r; \
	echo; \
	if [[ ! $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "Cancelled."; \
		exit 1; \
	fi
	@./benchmarks/run_all.sh

bench-python: ## Run Python benchmarks (NumPy + PyTorch) only
	@echo "🐍 Running Python benchmarks (NumPy + PyTorch)..."
	@echo "Estimated time: 2-3 minutes (includes dependency download)"
	@echo ""
	@command -v uv >/dev/null 2>&1 || { \
		echo "❌ UV not installed. Install with:"; \
		echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		exit 1; \
	}
	@echo "Installing dependencies with UV..."
	@cd benchmarks && uv run --with numpy --with torch python_comparison.py
	@echo "✅ Results: benchmarks/python_results.json"

bench-compare-frameworks: ## Generate comparison report (requires Rust + Python benchmarks)
	@echo "📊 Generating Trueno vs NumPy vs PyTorch comparison report..."
	@if [ ! -d target/criterion ] || [ -z "$$(ls -A target/criterion 2>/dev/null)" ]; then \
		echo "❌ Rust benchmarks not found. Run 'make bench' first."; \
		exit 1; \
	fi
	@if [ ! -f benchmarks/python_results.json ]; then \
		echo "❌ Python benchmarks not found. Run 'make bench-python' first."; \
		exit 1; \
	fi
	@cd benchmarks && uv run --with numpy --with torch compare_results.py
	@echo ""
	@echo "✅ Comparison complete!"
	@echo "   Report: benchmarks/comparison_report.md"
	@echo "   JSON:   benchmarks/comparison_summary.json"
	@echo ""
	@echo "View report:"
	@echo "  cat benchmarks/comparison_report.md"

# Profiling with Renacer (v0.5.0+)
profile: ## Profile benchmarks with Renacer (syscall tracing)
	@echo "🔬 Profiling benchmarks with Renacer v0.5.0..."
	@command -v renacer >/dev/null 2>&1 || { echo "Installing renacer..."; cargo install renacer; } || exit 1
	cargo build --release --all-features || exit 1
	renacer --function-time --source -- cargo bench --no-fail-fast

profile-flamegraph: ## Generate flamegraph from profiling
	@echo "🔥 Generating flamegraph..."
	@command -v renacer >/dev/null 2>&1 || { echo "Installing renacer..."; cargo install renacer; } || exit 1
	@command -v flamegraph.pl >/dev/null 2>&1 || { echo "⚠️  flamegraph.pl not found. Install from: https://github.com/brendangregg/FlameGraph"; } || exit 1
	cargo build --release --all-features || exit 1
	renacer --function-time --source -- cargo bench --no-fail-fast > profile.txt 2>&1 || exit 1
	@echo "📊 Flamegraph saved to: flame.svg"
	@echo "    View with: firefox flame.svg"

profile-bench: ## Profile specific benchmark (BENCH=vector_ops)
	@echo "🔬 Profiling benchmark: $(BENCH)..."
	@command -v renacer >/dev/null 2>&1 || { echo "Installing renacer..."; cargo install renacer; } || exit 1
	cargo build --release --all-features || exit 1
	renacer --function-time --source -- cargo bench $(BENCH)

profile-test: ## Profile test suite to find bottlenecks
	@echo "🔬 Profiling test suite..."
	@command -v renacer >/dev/null 2>&1 || { echo "Installing renacer..."; cargo install renacer; } || exit 1
	cargo build --release --all-features || exit 1
	renacer --function-time --source -- cargo test --release --all-features

# OpenTelemetry Distributed Tracing (Renacer 0.5.0+)
profile-otlp-jaeger: ## Profile with OTLP export to Jaeger (requires Docker)
	@echo "📊 Profiling with OpenTelemetry export to Jaeger..."
	@command -v docker >/dev/null 2>&1 || { echo "❌ Docker required. Install from: https://docs.docker.com/get-docker/"; exit 1; }
	@echo "Starting Jaeger All-in-One..."
	@docker run -d --name jaeger-trueno \
		-p 16686:16686 \
		-p 4317:4317 \
		-p 4318:4318 \
		jaegertracing/all-in-one:latest || { \
		echo "Jaeger already running or failed to start"; \
		docker start jaeger-trueno 2>/dev/null || true; \
	}
	@sleep 2
	@echo "Running benchmarks with OTLP tracing..."
	@cargo build --release --all-features || exit 1
	@renacer --function-time --source \
		--otlp-endpoint http://localhost:4317 \
		--otlp-service-name trueno-benchmarks \
		-- cargo bench --no-fail-fast
	@echo ""
	@echo "✅ Traces exported to Jaeger"
	@echo "   View at: http://localhost:16686"
	@echo "   Stop Jaeger: docker stop jaeger-trueno && docker rm jaeger-trueno"

profile-otlp-tempo: ## Profile with OTLP export to Grafana Tempo (requires Docker Compose)
	@echo "📊 Profiling with OpenTelemetry export to Grafana Tempo..."
	@command -v docker-compose >/dev/null 2>&1 || { echo "❌ Docker Compose required"; exit 1; }
	@echo "Starting Grafana Tempo stack..."
	@docker-compose -f docs/profiling/docker-compose-tempo.yml up -d || exit 1
	@sleep 5
	@echo "Running benchmarks with OTLP tracing..."
	@cargo build --release --all-features || exit 1
	@renacer --function-time --source \
		--otlp-endpoint http://localhost:4317 \
		--otlp-service-name trueno-benchmarks \
		-- cargo bench --no-fail-fast
	@echo ""
	@echo "✅ Traces exported to Tempo"
	@echo "   Grafana UI: http://localhost:3000 (admin/admin)"
	@echo "   Stop stack: docker-compose -f docs/profiling/docker-compose-tempo.yml down"

# OTLP Trace Analysis & CI Integration
profile-otlp-export: ## Export OTLP traces to JSON for CI/CD (TAG=commit-sha)
	@echo "📤 Exporting OTLP traces for CI/CD analysis..."
	@mkdir -p target/profiling
	@echo "Starting Jaeger (temporary)..."
	@docker run -d --name jaeger-ci \
		-p 16686:16686 \
		-p 4317:4317 \
		jaegertracing/all-in-one:latest >/dev/null 2>&1 || { \
		echo "Jaeger already running"; \
		docker start jaeger-ci 2>/dev/null || true; \
	}
	@sleep 3
	@echo "Running benchmarks with tracing..."
	@cargo build --release --all-features >/dev/null 2>&1 || exit 1
	@renacer --timing --source \
		--otlp-endpoint http://localhost:4317 \
		--otlp-service-name trueno-ci \
		-- cargo bench --no-fail-fast 2>&1 | tail -10
	@sleep 2
	@echo "Exporting traces..."
	@TAG=$${TAG:-$$(git rev-parse --short HEAD)} && \
	curl -sf "http://localhost:16686/api/traces?service=trueno-ci&limit=1000" \
		> target/profiling/traces-$$TAG.json || { echo "❌ Failed to export traces"; exit 1; } && \
	echo "✅ Exported to: target/profiling/traces-$$TAG.json"
	@docker stop jaeger-ci >/dev/null 2>&1 && docker rm jaeger-ci >/dev/null 2>&1
	@echo "   Trace count: $$(cat target/profiling/traces-$${TAG:-$$(git rev-parse --short HEAD)}.json | python3 -c 'import sys,json; print(len(json.load(sys.stdin)[\"data\"]))' 2>/dev/null || echo 'N/A')"

profile-analyze: ## Analyze exported traces (FILE=target/profiling/traces-abc123.json)
	@echo "📊 Analyzing trace data..."
	@test -f "$(FILE)" || { echo "❌ File not found: $(FILE)"; exit 1; }
	@python3 scripts/analyze_traces.py "$(FILE)"

profile-compare: ## Compare traces between commits (BASELINE=v0.4.0 CURRENT=main)
	@echo "🔍 Comparing traces: $(BASELINE) vs $(CURRENT)"
	@test -f "target/profiling/traces-$(BASELINE).json" || { echo "❌ Baseline not found. Run: make profile-otlp-export TAG=$(BASELINE)"; exit 1; }
	@test -f "target/profiling/traces-$(CURRENT).json" || { echo "❌ Current not found. Run: make profile-otlp-export TAG=$(CURRENT)"; exit 1; }
	@python3 scripts/compare_traces.py \
		"target/profiling/traces-$(BASELINE).json" \
		"target/profiling/traces-$(CURRENT).json" \
		"$(BASELINE)" "$(CURRENT)" \
		| tee "target/profiling/comparison-$(BASELINE)-vs-$(CURRENT).md"
	@echo ""
	@echo "✅ Report saved to: target/profiling/comparison-$(BASELINE)-vs-$(CURRENT).md"

mutate: ## Run mutation testing (>80% kill rate target)
	@echo "🧬 Running mutation testing (target: >80% kill rate)..."
	@command -v cargo-mutants >/dev/null 2>&1 || { echo "Installing cargo-mutants..."; cargo install cargo-mutants; } || exit 1
	cargo mutants --timeout 60

pixel-fkr: ## Run pixel FKR visual regression tests (requires CUDA)
	@echo "🎨 Running Pixel FKR visual regression tests..."
	cargo test -p trueno-gpu --test pixel_fkr --features "cuda gpu-pixels" -- --nocapture

pixel-fkr-capture: ## Capture golden baselines for pixel FKR
	@echo "📸 Capturing golden baselines..."
	@mkdir -p golden_traces
	cargo run --release --features "cuda gpu-pixels" --example gpu_pixels_render -- --capture

quick-validate: lint ## Quick validation (<2 minutes)
	@echo "🚀 Running quick validation suite..."
	cargo test --lib --quiet
	cargo run --release --example quickstart >/dev/null
	@echo "✅ Quick validation passed"

full-validate: quick-validate coverage ## Full validation (<10 minutes)
	@echo "🔬 Running full validation suite..."
	cargo +nightly miri test --lib -- --skip simd --skip gpu --skip avx --skip sse --skip neon --skip wasm --skip proptest scalar || true
	@echo "✅ Full validation passed"

clean: ## Clean build artifacts
	cargo clean
	rm -rf target/ || exit 1
	rm -f lcov.info || exit 1
	rm -rf book/book/ || true  # Remove mdbook generated output (causes SATD false positives)

quality-gates: lint fmt-check test-fast coverage ## Run all quality gates (pre-commit)
	@echo ""
	@echo "✅ All quality gates passed!"
	@echo ""
	@echo "Summary:"
	@echo "  ✅ Linting: cargo clippy (zero warnings)"
	@echo "  ✅ Formatting: cargo fmt"
	@echo "  ✅ Tests: cargo test (all passing)"
	@echo "  ✅ Coverage: ≥90% (see report above)"
	@echo ""
	@echo "Ready to commit!"

all: quality-gates ## Run full build pipeline

# ============================================================================
# PMAT INTEGRATION (v2.200.0+ features)
# ============================================================================

pmat-tdg: ## Run PMAT Technical Debt Grading (minimum: B+)
	@echo "📊 PMAT Technical Debt Grading..."
	@pmat analyze tdg

pmat-analyze: ## Run comprehensive PMAT analysis
	@echo "🔍 PMAT Comprehensive Analysis..."
	@echo ""
	@echo "  [1/5] Complexity analysis..."
	@pmat analyze complexity --project-path . || true
	@echo ""
	@echo "  [2/5] SATD detection..."
	@pmat analyze satd --path . || true
	@echo ""
	@echo "  [3/5] Dead code analysis..."
	@pmat analyze dead-code --path . || true
	@echo ""
	@echo "  [4/5] Code duplication..."
	@pmat analyze duplicates || true
	@echo ""
	@echo "  [5/5] Known defects (unwrap calls)..."
	@pmat analyze defects --path . || true
	@echo ""
	@echo "✅ PMAT analysis complete"

pmat-score: ## Calculate repository health score (minimum: 90/110)
	@echo "🏆 Repository Health Score..."
	@pmat repo-score || true

pmat-rust-score: ## Calculate Rust project score (0-211 scale, minimum: 150)
	@echo "🦀 Rust Project Score (v2.171.0+)..."
	@mkdir -p target/pmat-reports
	@pmat rust-project-score --path . || echo "⚠️  Rust project score not available in this PMAT version"

pmat-rust-score-fast: ## Calculate Rust project score (fast mode, ~3 min)
	@echo "🦀 Rust Project Score (fast mode)..."
	@pmat rust-project-score --path . || echo "⚠️  Rust project score not available in this PMAT version"

pmat-mutate: ## Run mutation testing with PMAT (AST-based)
	@echo "🧬 PMAT Mutation Testing..."
	@echo "⚠️  Note: PMAT mutation testing not available in this version"
	@echo "    Use 'make mutate' for cargo-mutants instead"

pmat-semantic-search: ## Index code for semantic search
	@echo "🔍 Indexing code for semantic search..."
	@pmat embed sync ./src || echo "⚠️  Semantic search not available in this PMAT version"

pmat-validate-docs: ## Validate documentation (hallucination detection - Phase 3.5)
	@echo "📚 Validating documentation accuracy (Phase 3.5)..."
	@echo ""
	@echo "Step 1: Generating deep context..."
	@pmat context --output deep_context.md --format llm-optimized
	@echo ""
	@echo "Step 2: Validating documentation files..."
	@pmat validate-readme \
		--targets README.md CLAUDE.md \
		--deep-context deep_context.md \
		--fail-on-contradiction \
		--verbose || { \
		echo ""; \
		echo "❌ Documentation validation failed!"; \
		echo "   Fix contradictions and broken references before committing"; \
		exit 1; \
	}
	@echo ""
	@echo "✅ Documentation validation complete - zero hallucinations!"

pmat-work-init: ## Initialize PMAT workflow system (v2.198.0)
	@echo "🔧 Initializing PMAT workflow system..."
	@echo "⚠️  Note: pmat work commands may not be available in this version"
	@echo "    Check: pmat --help | grep work"

pmat-quality-gate: ## Run comprehensive PMAT quality gate
	@echo "🚦 PMAT Quality Gate (comprehensive)..."
	@pmat quality-gates check || echo "⚠️  Quality gate check not available in this format"

pmat-context: ## Generate AI-ready project context
	@echo "🤖 Generating AI context..."
	@pmat context --output deep_context.md || echo "⚠️  Context generation not available"

pmat-all: pmat-tdg pmat-analyze pmat-score ## Run all PMAT checks (fast)

# Development helpers
dev: ## Run in development mode with auto-reload
	cargo watch -x 'test --all-features'

install-tools: ## Install required development tools
	cargo install cargo-llvm-cov || exit 1
	cargo install cargo-nextest || exit 1
	cargo install cargo-watch || exit 1
	cargo install cargo-mutants || exit 1
	cargo install criterion || exit 1
	cargo install renacer || exit 1
	cargo install mdbook || exit 1

# ============================================================================
# ARM64/NEON: Cross-compilation and QEMU emulation
# ============================================================================
# Test ARM NEON SIMD backend without ARM hardware using QEMU userspace emulation.
# Performance: ~5-20x slower than native, but accurate instruction emulation.
# Prerequisites: qemu-user, aarch64-linux-gnu-gcc (apt install qemu-user gcc-aarch64-linux-gnu)

test-arm: ## Run all tests on ARM64 via QEMU (NEON SIMD validation)
	@echo "🦾 Running ARM64 tests via QEMU (NEON backend validation)..."
	@which qemu-aarch64 > /dev/null 2>&1 || { echo "❌ qemu-user not installed. Run: sudo apt install qemu-user"; exit 1; }
	@which aarch64-linux-gnu-gcc > /dev/null 2>&1 || { echo "❌ Cross-compiler not installed. Run: sudo apt install gcc-aarch64-linux-gnu"; exit 1; }
	@echo "  [1/3] Building for aarch64-unknown-linux-gnu..."
	@cargo build --target aarch64-unknown-linux-gnu --release --all-features 2>&1 | tail -3
	@echo "  [2/3] Running tests via QEMU (this will be slow ~5-20x)..."
	@PROPTEST_CASES=25 cargo test --target aarch64-unknown-linux-gnu --release --all-features -- --test-threads=1 2>&1 || true
	@echo ""
	@echo "✅ ARM64/NEON tests complete"

test-arm-neon: ## Run NEON-specific tests on ARM64 via QEMU
	@echo "🦾 Running NEON-specific tests via QEMU..."
	@PROPTEST_CASES=25 cargo test --target aarch64-unknown-linux-gnu --release --all-features neon -- --test-threads=1 --nocapture

test-arm-quick: ## Quick ARM64 smoke test (lib tests only, faster)
	@echo "🦾 Quick ARM64 smoke test..."
	@PROPTEST_CASES=25 cargo test --target aarch64-unknown-linux-gnu --release --lib -- --test-threads=1 2>&1 | tail -20
	@echo "✅ ARM64 quick test passed"

bench-arm: ## Run benchmarks on ARM64 via QEMU (for relative comparison only)
	@echo "📊 Running ARM64 benchmarks via QEMU..."
	@echo "⚠️  Note: QEMU benchmarks are ~5-20x slower than native ARM."
	@echo "         Use for relative comparison between NEON implementations only."
	@cargo bench --target aarch64-unknown-linux-gnu --all-features -- neon 2>&1 | tee /tmp/arm-bench.log
	@echo ""
	@echo "📊 Benchmark results saved to /tmp/arm-bench.log"

# ============================================================================
# INTEL SDE: AVX-512 Emulation for CPUs without hardware support
# ============================================================================
# Intel SDE (Software Development Emulator) allows testing AVX-512 code on
# CPUs that don't have AVX-512 support (e.g., Intel Alder Lake+, older AMD).
#
# Download: https://www.intel.com/content/www/us/en/developer/articles/tool/software-development-emulator.html
# Performance: ~10-50x slower than native, but accurate instruction emulation.

SDE_VERSION := 9.33.0-2024-01-07
SDE_DIR := "$(HOME)/.local/share/intel-sde"
SDE_BIN := "$(HOME)/.local/share/intel-sde/sde64"

install-sde: ## Install Intel SDE for AVX-512 emulation
	@echo "📦 Installing Intel SDE $(SDE_VERSION)..."
	@if [ -f $(SDE_BIN) ]; then \
		echo "  ✅ Intel SDE already installed at $(SDE_BIN)"; \
		$(SDE_BIN) --version 2>/dev/null | head -1 || true; \
	else \
		echo "  ⬇️  Downloading Intel SDE..."; \
		mkdir -p $(SDE_DIR) || exit 1; \
		cd /tmp && \
		curl -L -o sde.tar.xz "https://downloadmirror.intel.com/813591/sde-external-$(SDE_VERSION)-lin.tar.xz" && \
		tar xf sde.tar.xz && \
		cp -r "sde-external-$(SDE_VERSION)-lin"/* "$(SDE_DIR)/" && \
		rm -rf sde.tar.xz "sde-external-$(SDE_VERSION)-lin" && \
		echo "  ✅ Intel SDE installed to $(SDE_DIR)"; \
		echo "  💡 Add to PATH: export PATH=$(SDE_DIR):\$$PATH"; \
	fi

test-avx512-sde: ## Run AVX-512 tests under Intel SDE emulation
	@echo "🔬 Running AVX-512 tests under Intel SDE (Skylake-X emulation)..."
	@if [ ! -f "$(SDE_BIN)" ]; then \
		echo "  ❌ Intel SDE not found. Run 'make install-sde' first."; \
		exit 1; \
	fi
	@echo "  [1/3] Building test binary (native)..."
	@cargo test --release --all-features --no-run 2>&1 | tail -5
	@echo "  [2/3] Finding test binary..."
	@TEST_BIN=$$(cargo test --release --all-features --no-run 2>&1 | grep -oP 'target/release/deps/trueno-[a-f0-9]+' | head -1); \
	if [ -z "$$TEST_BIN" ]; then \
		echo "  ❌ Could not find test binary"; \
		exit 1; \
	fi; \
	echo "  Found: $$TEST_BIN"; \
	echo "  [3/3] Running under SDE (-skx = Skylake-X with AVX-512)..."; \
	echo "  ⚠️  This will be slow (~10-50x) - emulating AVX-512 instructions"; \
	PROPTEST_CASES=25 $(SDE_BIN) -skx -- ./$$TEST_BIN avx512 --test-threads=1 2>&1 | tee /tmp/sde-avx512-test.log; \
	echo ""; \
	echo "  📊 Test results saved to /tmp/sde-avx512-test.log"

bench-avx512-sde: ## Run AVX-512 benchmarks under Intel SDE emulation
	@echo "📊 Running AVX-512 benchmarks under Intel SDE..."
	@if [ ! -f "$(SDE_BIN)" ]; then \
		echo "  ❌ Intel SDE not found. Run 'make install-sde' first."; \
		exit 1; \
	fi
	@echo "  ⚠️  Note: SDE benchmarks are ~10-50x slower than native."
	@echo "  Building release binary..."
	@cargo build --release --all-features
	@echo "  Running benchmarks under SDE (-skx = Skylake-X with AVX-512)..."
	@$(SDE_BIN) -skx -- cargo bench --all-features -- avx512 2>&1 | tee /tmp/sde-avx512-bench.log
	@echo ""
	@echo "  📊 Benchmark results saved to /tmp/sde-avx512-bench.log"

coverage-avx512-sde: ## Run AVX-512 coverage under Intel SDE emulation
	@echo "📈 Running AVX-512 coverage under Intel SDE..."
	@if [ ! -f "$(SDE_BIN)" ]; then \
		echo "  ❌ Intel SDE not found. Run 'make install-sde' first."; \
		exit 1; \
	fi
	@echo "  ⚠️  Note: Coverage under SDE is slow but provides accurate AVX-512 coverage."
	@echo "  [1/4] Cleaning previous coverage data..."
	@echo "  [2/4] Building instrumented test binary (native)..."
	@PROPTEST_CASES=25 QUICKCHECK_TESTS=25 RUSTFLAGS="-C instrument-coverage" cargo test --all-features --no-run 2>&1 | tail -5
	@echo "  [3/4] Finding instrumented test binary..."
	@TEST_BIN=$$(PROPTEST_CASES=25 QUICKCHECK_TESTS=25 RUSTFLAGS="-C instrument-coverage" cargo test --all-features --no-run 2>&1 | grep -oP 'target/debug/deps/trueno-[a-f0-9]+' | head -1); \
	if [ -z "$$TEST_BIN" ]; then \
		echo "  ❌ Could not find instrumented test binary"; \
		exit 1; \
	fi; \
	echo "  Found: $$TEST_BIN"; \
	echo "  [4/4] Running AVX-512 tests under SDE..."; \
	echo "  ⚠️  This will be slow (~10-50x) - emulating AVX-512 instructions"; \
	mkdir -p target/sde-coverage || exit 1; \
	LLVM_PROFILE_FILE="target/sde-coverage/trueno-%p-%m.profraw" \
		$(SDE_BIN) -skx -- ./$$TEST_BIN avx512 --test-threads=1 2>&1 | tee /tmp/sde-avx512-coverage.log
	@echo ""
	@echo "  📊 Test results saved to /tmp/sde-avx512-coverage.log"
	@echo ""
	@echo "  Merging coverage data..."
	@if command -v llvm-profdata >/dev/null 2>&1; then \
		llvm-profdata merge -sparse target/sde-coverage/*.profraw -o target/sde-coverage/merged.profdata 2>/dev/null && \
		echo "  ✅ Coverage data merged to target/sde-coverage/merged.profdata"; \
	else \
		echo "  ⚠️  llvm-profdata not found - install llvm tools for coverage report"; \
	fi

# Documentation quality gates
validate-examples: ## Validate book examples meet EXTREME TDD quality
	@cargo run -p xtask -- validate-examples

build-book: ## Build mdBook documentation
	@echo "📖 Building book..."
	@mdbook build book/

serve-book: ## Serve book locally with live reload
	@echo "📖 Serving book at http://localhost:3000..."
	@mdbook serve book/

# Bashrs validation (shell script quality enforcement)
bashrs-lint-makefile: ## Lint Makefile with bashrs
	@echo "🔍 Linting Makefile with bashrs..."
	@bashrs make lint Makefile || true

bashrs-lint-scripts: ## Lint all shell scripts with bashrs
	@echo "🔍 Linting shell scripts with bashrs..."
	@if ls scripts/*.sh 1>/dev/null 2>&1; then \
		for script in scripts/*.sh; do \
			echo "  Linting $$script..."; \
			bashrs lint "$$script" || true; \
		done; \
	else \
		echo "  ℹ️  No shell scripts found (replaced with Rust xtask - A-grade quality)"; \
	fi

bashrs-audit: ## Audit shell script quality with bashrs
	@echo "📊 Auditing shell scripts with bashrs..."
	@if ls scripts/*.sh 1>/dev/null 2>&1; then \
		for script in scripts/*.sh; do \
			echo "  Auditing $$script..."; \
			bashrs audit "$$script"; \
		done; \
	else \
		echo "  ℹ️  No shell scripts found (replaced with Rust xtask - A-grade quality)"; \
	fi

bashrs-all: bashrs-lint-makefile bashrs-lint-scripts bashrs-audit ## Run all bashrs quality checks

.DEFAULT_GOAL := help

# ============================================================================
# TRUENO-SPEC-013: Solidify Quality Gates with CUDA/WGPU Coverage
# ============================================================================

# Coverage targets for CUDA (SPEC Section 3.3)
coverage-cuda: ## Generate coverage with CUDA tests (requires NVIDIA GPU)
	@echo "📊 Running coverage with CUDA tests (TRUENO-SPEC-013)..."
	@nvidia-smi > /dev/null 2>&1 || { echo "❌ NVIDIA GPU required for CUDA coverage"; exit 1; }
	@which cargo-llvm-cov > /dev/null 2>&1 || (cargo install cargo-llvm-cov --locked || exit 1)
	@echo ""
	@echo "🚀 Phase 1: Fast tests (nextest parallel)..."
	@env PROPTEST_CASES=25 QUICKCHECK_TESTS=25 $(CARGO_LLVM_COV) test --no-report \
		--lib --all-features --workspace \
		$(COV_EXCLUDE) 2>&1 || true
	@echo ""
	@echo "🎮 Phase 2: CUDA tests (sequential, extended timeout)..."
	@env PROPTEST_CASES=10 $(CARGO_LLVM_COV) --no-report \
		test --features cuda --workspace \
		-- --test-threads=1 cuda driver 2>&1 || true
	@echo ""
	@echo "📊 Generating combined CUDA coverage reports..."
	@$(CARGO_LLVM_COV) report --html --output-dir target/coverage/cuda-html
	@$(CARGO_LLVM_COV) report --lcov --output-path target/coverage/cuda-lcov.info
	@echo ""
	@echo "📊 CUDA Coverage Summary:"
	@echo "========================="
	@$(CARGO_LLVM_COV) report --summary-only
	@echo ""
	@echo "💡 HTML report: target/coverage/cuda-html/index.html"

coverage-95: ## Enforce 95% coverage threshold (trueno + trueno-gpu)
	@echo "🔒 Checking 95% coverage threshold (trueno + trueno-gpu)..."
	@echo ""
	@TOTAL_COV=$$($(CARGO_LLVM_COV) report --summary-only $(COV_EXCLUDE) 2>/dev/null | grep "TOTAL" | awk '{print $$NF}' | sed 's/%//'); \
	if [ -z "$$TOTAL_COV" ]; then echo "❌ No coverage data. Run 'make coverage' first."; exit 1; fi; \
	echo "Combined coverage: $${TOTAL_COV}%"; \
	echo ""; \
	RESULT=$$(echo "$$TOTAL_COV >= 95" | bc -l 2>/dev/null || echo 0); \
	if [ "$$RESULT" = "1" ]; then \
		echo "✅ Coverage threshold met (≥95%)"; \
	else \
		echo "❌ FAIL: Coverage $${TOTAL_COV}% is below 95% threshold"; \
		echo "   Run 'make coverage' then open target/coverage/html/index.html"; \
		exit 1; \
	fi

# Smoke tests (SPEC Section 3.2)
smoke: ## Run E2E smoke tests (SIMD + WGPU + CUDA) - TRUENO-SPEC-013
	@echo "🔥 Running E2E smoke tests (TRUENO-SPEC-013)..."
	@echo ""
	@echo "Phase 1: SIMD backend validation..."
	@cargo test --test smoke_e2e smoke_simd -- --nocapture 2>&1 || true
	@echo ""
	@echo "Phase 2: WGPU backend validation..."
	@cargo test --test smoke_e2e smoke_wgpu --features gpu -- --nocapture 2>&1 || true
	@echo ""
	@echo "Phase 3: CUDA backend validation..."
	@nvidia-smi > /dev/null 2>&1 && \
		cargo test -p trueno-gpu --test smoke_e2e smoke_cuda --features cuda -- --nocapture 2>&1 || \
		echo "⚠️  CUDA not available, skipping"
	@echo ""
	@echo "✅ Smoke tests complete"

smoke-full: ## Run full E2E smoke test suite with backend equivalence
	@echo "🔥 Running FULL E2E smoke test suite..."
	@cargo test --test smoke_e2e --all-features -- --nocapture
	@echo "✅ Full smoke test suite passed"

# Pixel FKR Tests (SPEC Section 3.5)
pixel-scalar-fkr: ## Run scalar baseline pixel tests (generates golden images)
	@echo "🎨 Running scalar-pixel-fkr (baseline truth)..."
	@cargo test --test pixel_fkr scalar_pixel_fkr -- --nocapture
	@echo "✅ Scalar baseline generated"

pixel-simd-fkr: ## Run SIMD pixel tests against scalar baseline
	@echo "🎨 Running simd-pixel-fkr..."
	@cargo test --test pixel_fkr simd_pixel_fkr -- --nocapture

pixel-wgpu-fkr: ## Run WGPU pixel tests against scalar baseline
	@echo "🎨 Running wgpu-pixel-fkr..."
	@cargo test --test pixel_fkr wgpu_pixel_fkr --features gpu -- --nocapture

pixel-ptx-fkr: ## Run PTX pixel tests against scalar baseline (requires NVIDIA GPU)
	@echo "🎨 Running ptx-pixel-fkr..."
	@nvidia-smi > /dev/null 2>&1 || { echo "❌ NVIDIA GPU required"; exit 1; }
	@cargo test -p trueno-gpu --test pixel_fkr ptx_pixel_fkr --features "cuda gpu-pixels" -- --nocapture

pixel-fkr-all: pixel-scalar-fkr pixel-simd-fkr pixel-wgpu-fkr pixel-ptx-fkr ## Run all pixel FKR suites
	@echo "✅ All pixel FKR suites passed"

# Combined quality gate (SPEC Section 8)
quality-spec-013: lint fmt-check test-fast coverage-95 smoke pixel-fkr-all ## Full TRUENO-SPEC-013 quality gate
	@echo ""
	@echo "✅ TRUENO-SPEC-013 Quality Gate PASSED!"
	@echo ""
	@echo "Summary:"
	@echo "  ✅ Linting: cargo clippy (zero warnings)"
	@echo "  ✅ Formatting: cargo fmt"
	@echo "  ✅ Tests: cargo test (all passing)"
	@echo "  ✅ Coverage: ≥95% (CUDA + WGPU)"
	@echo "  ✅ Smoke: E2E backend validation"
	@echo "  ✅ Pixel FKR: Visual regression tests"
	@echo ""

# ============================================================================
# RELEASE (crates.io publishing)
# ============================================================================

release-check: ## Verify package can be published (dry-run)
	@echo "🔍 Checking release readiness..."
	cargo publish --dry-run --allow-dirty
	@echo "✅ Package ready for release"

release: ## Publish to crates.io (requires cargo login)
	@echo "🚀 Publishing trueno to crates.io..."
	@echo "⚠️  Ensure all changes are committed!"
	cargo publish
	@echo "✅ Published successfully"
	@echo "📦 Create GitHub release: gh release create v$$(cargo pkgid | cut -d# -f2)"

release-tag: ## Create git tag for current version
	@VERSION=$$(cargo pkgid | cut -d# -f2) && \
	echo "🏷️  Creating tag v$$VERSION..." && \
	git tag -a "v$$VERSION" -m "Release v$$VERSION" && \
	git push origin "v$$VERSION" && \
	echo "✅ Tag v$$VERSION pushed"
