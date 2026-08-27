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
# Recipes ran as `bash -c` with NO pipefail, so any `cmd | tail`/`| grep` reported
# the LAST command's status and a failing producer was laundered to success. That
# is the defect class this repo keeps rediscovering (see the Verification
# Discipline section of CLAUDE.md: "Never read $? through a pipe").
#
# Measured on this Makefile before the change: 577 recipe lines, 14 with a pipe.
# The worst was the release gate itself, `contracts:` -> `pv lint contracts/ 2>&1
# | tail -5`, which could never fail the build no matter what pv reported.
#
# DELIBERATELY `-o pipefail` ONLY, not `-eu -o pipefail`. Measured exposure of the
# other two flags on this file: 248 recipe lines use `;` chains (-e would abort
# them mid-recipe) and 74 reference `$$VAR` (-u would error on any unset one).
# Changing three variables at once across 577 lines is how a "small" fix becomes
# an outage. pipefail is provably orthogonal to both -- verified with fixtures:
# a `;` chain and an unset var both still exit 0 under pipefail alone -- so it
# closes the laundering class and touches nothing else. Add -e/-u later, one at
# a time, each with its own blast-radius measurement.
.SHELLFLAGS := -o pipefail -c

# Disable built-in rules for performance
.SUFFIXES:

# Delete partially-built files on error
.DELETE_ON_ERROR:

# Multi-line recipes execute in same shell
.ONESHELL:

.PHONY: all build test test-smoke test-fast test-quick test-full test-heavy lint lint-current fmt clean doc book book-build book-serve book-test tier1 tier2 tier3 tier4 coverage coverage-fast profile hooks-install hooks-verify lint-scripts bashrs-score bashrs-lint-makefile chaos-test chaos-test-full chaos-test-lite fuzz bench dev pre-push ci check run-ci run-bench audit deps-validate deny pmat-score pmat-gates quality-report semantic-search examples mutants mutants-fast property-test install-alsa test-alsa test-audio-full contract-validate contract-test contract-audit contract-regen contract-check dev-setup check-siblings check-wasm32 contrastive-data-boundary

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

# aprender#2522: both targets below piped cargo into `grep`, so make read
# GREP's exit status and never cargo's. `test-spec` therefore printed
# "✅ Spec tests complete" for months while the suite was 38-red — grep found
# the "test result:" line, which is exactly what it does when tests FAIL. These
# were the suite's only callers anywhere, so nothing could observe the failures.
# CLAUDE.md "Verification Discipline" rule 1: never read `$?` through a pipe.
test-model: ## Run model falsification tests ONE AT A TIME (requires models/, ollama, GPU)
	@echo "🧪 Running model falsification tests (one at a time to avoid OOM)..."
	@rc=0; for test in f_ollama_001 f_ollama_002 f_ollama_003 f_ollama_004 f_ollama_005 \
	             f_perf_003 f_trueno_004 f_trueno_008 f_rosetta_002 f_qa_002; do \
		echo "  ⏳ $$test"; \
		PROPTEST_CASES=10 QUICKCHECK_TESTS=10 \
		cargo test --features model-tests --test falsification_spec_v10_tests "$$test" \
			> /tmp/apr-test-model-$$test.log 2>&1 \
			|| { rc=1; echo "  ❌ $$test FAILED"; }; \
		grep "test result:" /tmp/apr-test-model-$$test.log || true; \
	done; \
	[ "$$rc" -eq 0 ] || { echo "❌ Model tests FAILED"; exit 1; }
	@echo "✅ Model tests complete"

test-spec: ## Run ALL spec falsification tests (structural only, no models)
	@echo "🔬 Running spec structural tests..."
	@PROPTEST_CASES=10 QUICKCHECK_TESTS=10 \
		cargo test --features model-tests \
			--test falsification_spec_v10_tests \
			--test falsification_stress_tests \
			--test falsification_gpu_state_tests \
		> /tmp/apr-test-spec.log 2>&1; \
	rc=$$?; \
	grep "test result:" /tmp/apr-test-spec.log || true; \
	[ "$$rc" -eq 0 ] || { sed -n '/^failures:/,$$p' /tmp/apr-test-spec.log; \
		echo "❌ Spec tests FAILED"; exit 1; }
	@echo "✅ Spec tests complete"

# Linting
lint:
	cargo clippy -- -D warnings

# Toolchain CEILING gate (aprender#2370). `lint` above runs through the
# rust-toolchain.toml pin, so clippy findings from NEWER releases accumulate
# unseen until someone's toolchain outruns the pin. This lints on current
# stable instead, and refuses to pass vacuously. Mirror of `check_msrv.sh`,
# which guards the floor.
lint-current:
	@bash scripts/check_clippy_current_stable.sh

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
# PMAT-484: probar golden regression if tests/golden/ exists
tier2:
	@echo "Running Tier 2: Pre-commit checks..."
	@PROPTEST_CASES=5 QUICKCHECK_TESTS=5 cargo test --lib
	@cargo clippy -- -D warnings
	@if [ -d tests/golden ]; then \
		if . scripts/apr_bin.sh 2>/dev/null; then \
			echo "Running probar golden regression... ($$APR)"; \
			"$$APR" probar tests/golden/model.apr --golden tests/golden/ --assert --tolerance 0.98 2>/dev/null || true; \
		else \
			echo "Skipping probar golden regression: no apr built from HEAD (scripts/apr_bin.sh)"; \
		fi; \
	fi
	@echo "Tier 2: PASSED"

# Tier 3: Pre-push (1-5 minutes, full validation)
# PMAT-484: probar golden regression + profile if tests/golden/ exists
tier3:
	@echo "Running Tier 3: Full validation..."
	@PROPTEST_CASES=25 QUICKCHECK_TESTS=25 cargo test --all
	@cargo clippy -- -D warnings
	@echo "Checking include!() files tracked by git..."
	@bash scripts/check_include_files.sh
	@echo "Checking publish safety (symlinks, companion lookups)..."
	@bash scripts/check_publish_safety.sh
	@echo "Checking cargo-deny policy (licences, bans, sources, advisories)..."
	@$(MAKE) --no-print-directory deny
	@echo "Checking exclude patterns are root-anchored (CB-510 class)..."
	@bash scripts/check_exclude_anchored.sh
	@echo "Checking build.rs crate-root escapes (v0.31.1 yank class)..."
	@bash scripts/check_build_rs_paths.sh
	@echo "Checking self-hosted CI jobs pin a discriminating runner label..."
	@bash scripts/check_runner_labels.sh
	@echo "Checking the toolchain-ceiling guard's comparator (aprender#2370)..."
	@bash scripts/check_clippy_current_stable.sh --self-test
	@echo "Checking no contract cites a test that does not exist (aprender#2465)..."
	@bash scripts/check_contract_test_binding.sh --self-test
	@bash scripts/check_contract_test_binding.sh
	@echo "Checking no contract names an enforcement command that cannot run (aprender#2504)..."
	@bash scripts/check_contract_enforcement.sh --self-test
	@bash scripts/check_contract_enforcement.sh
	@echo "Checking no test asserts about the fd 0 it inherited (aprender#2307)..."
	@bash scripts/check_hermetic_stdin_tests.sh --self-test
	@bash scripts/check_hermetic_stdin_tests.sh
	@if [ -d tests/golden ]; then \
		if . scripts/apr_bin.sh 2>/dev/null; then \
			echo "Running probar golden regression with profiling... ($$APR)"; \
			"$$APR" probar tests/golden/model.apr --golden tests/golden/ --assert --tolerance 0.98 2>/dev/null || true; \
		else \
			echo "Skipping probar golden regression: no apr built from HEAD (scripts/apr_bin.sh)"; \
		fi; \
	fi
# D-04, wired here because a target outside the tiers is a target that stops
# being run. `make contrastive-data-boundary` was run STANDALONE first with its
# status captured directly (`> /tmp/cdb.log 2>&1; rc=$$?`, never through a pipe):
# rc=0 in 1 s wall. Its four failure modes were each induced, observed and
# reverted rather than assumed — see the target's own comment block.
	@$(MAKE) contrastive-data-boundary
	@echo "Tier 3: PASSED"

# D-04: the aprender-contrastive-data bytes boundary. Wired into tier3 above.
#
# BOTH HALVES ARE POSITIVE CHECKS, and that is the whole design. The first draft
# of this gate was a dependency DENY-list plus a grep that skipped #[cfg(test)],
# and both can report PASS while the property is false: a deny-list only ever
# catches the hazards someone already enumerated, so the first dependency nobody
# thought to name passes silently; and a cfg-blind grep cannot actually tell test
# code from library code, so it either exempts too much or claims a precision it
# does not have. Replaced by (a) a POSITIVE allowlist compared against the
# resolved closure, so a new transitive dependency fails by DEFAULT, and (b) a
# src/-wide symbol ban with NO cfg(test) exemption, which turns "the public API
# contains no path types" into a mechanical consequence.
#
# EVERY FAILURE MODE WAS OBSERVED, not assumed (2026-08-08, each mutation applied,
# run, and reverted):
#   add `tempfile` to [dependencies] ........ FAIL, prints tempfile as an offender
#   `use std::path::PathBuf;` in src/schema.rs FAIL, names schema.rs and the line
#   the same line inside #[cfg(test)] mod tests FAIL (no exemption, by design)
#   rename allowed-deps.txt away ............ FAIL with a missing-allowlist message,
#                                             NOT a vacuous pass on an empty list
# Standalone timing before wiring: 1 s wall, rc=0.
contrastive-data-boundary: ## D-04: bytes boundary for aprender-contrastive-data (positive allowlist + src symbol ban)
	@echo "Bytes boundary: aprender-contrastive-data (D-04)"
	@mkdir -p target
# (a) DEPENDENCY ALLOWLIST. cargo tree's OWN status is checked FIRST. Piping it
# into the comparison would read the comparison's status (CLAUDE.md rule 1), and
# a cargo tree that failed outright would feed an EMPTY closure into a subset
# test — which passes vacuously and silently disarms the supply-chain half.
	@cargo tree -p aprender-contrastive-data -e normal --prefix none --no-dedupe \
		> target/contrastive-data-tree.txt 2>&1 || \
		{ echo "FAIL: cargo tree failed; the D-04 dependency check would pass vacuously"; \
		  cat target/contrastive-data-tree.txt; exit 1; }
	@if [ ! -s target/contrastive-data-tree.txt ]; then \
		echo "FAIL: cargo tree produced no output; the D-04 dependency check would pass vacuously"; \
		exit 1; \
	fi
	@awk 'NF { print $$1 }' target/contrastive-data-tree.txt | sort -u \
		> target/contrastive-data-deps.txt
	@if [ ! -f crates/aprender-contrastive-data/allowed-deps.txt ]; then \
		echo "FAIL: crates/aprender-contrastive-data/allowed-deps.txt is MISSING."; \
		echo "      Without it every dependency would be admitted and this gate would"; \
		echo "      report PASS while checking nothing."; \
		exit 1; \
	fi
	@grep -v '^[[:space:]]*#' crates/aprender-contrastive-data/allowed-deps.txt \
		| grep -v '^[[:space:]]*$$' | sort -u > target/contrastive-data-allowed.txt
	@if [ ! -s target/contrastive-data-allowed.txt ]; then \
		echo "FAIL: allowed-deps.txt has no entries. An empty allowlist cannot admit even"; \
		echo "      the crate itself, so this is a broken gate rather than a strict one."; \
		exit 1; \
	fi
	@comm -23 target/contrastive-data-deps.txt target/contrastive-data-allowed.txt \
		> target/contrastive-data-offenders.txt
	@if [ -s target/contrastive-data-offenders.txt ]; then \
		echo "FAIL: packages in the resolved normal-dependency closure but ABSENT from"; \
		echo "      crates/aprender-contrastive-data/allowed-deps.txt (D-04):"; \
		sed 's/^/        /' target/contrastive-data-offenders.txt; \
		echo "      Do NOT widen the allowlist just to turn this green: the allowlist"; \
		echo "      entry IS the review. Read what the package pulls in first."; \
		exit 1; \
	fi
	@echo "  deps:   resolved closure is a subset of allowed-deps.txt"
# (b) SOURCE SURFACE BAN. Matches are taken with true line numbers first, then
# comment lines are dropped from the RESULTS, so a doc comment can neither trip
# the gate nor satisfy it and the reported line number still points at the real
# file. There is deliberately NO #[cfg(test)] exemption — tests that genuinely
# need a filesystem belong in tests/ (outside the library boundary) or in apr-cli.
	@find crates/aprender-contrastive-data/src -type f -name '*.rs' \
		> target/contrastive-data-srcfiles.txt 2>&1 || \
		{ echo "FAIL: could not enumerate src/; the D-04 source check would pass vacuously"; \
		  exit 1; }
	@if [ ! -s target/contrastive-data-srcfiles.txt ]; then \
		echo "FAIL: no .rs files found under crates/aprender-contrastive-data/src;"; \
		echo "      the D-04 source check would pass vacuously"; \
		exit 1; \
	fi
	@: > target/contrastive-data-symbols.txt
	@while IFS= read -r srcfile; do \
		{ grep -nE 'std::fs|std::net|std::path' "$$srcfile" || true; \
		  grep -nwE 'Path|PathBuf' "$$srcfile" || true; } \
		| grep -vE '^[0-9]+:[[:space:]]*//' \
		| sed "s|^|$$srcfile:|" >> target/contrastive-data-symbols.txt || true; \
	done < target/contrastive-data-srcfiles.txt
	@if [ -s target/contrastive-data-symbols.txt ]; then \
		echo "FAIL: forbidden filesystem/network/path symbols under src/ (D-04)."; \
		echo "      The crate is bytes-in/bytes-out; apr-cli owns every fs adapter."; \
		sort -u target/contrastive-data-symbols.txt | sed 's/^/        /'; \
		exit 1; \
	fi
	@echo "  source: no fs/net/path symbols under src/ (no cfg(test) exemption)"
	@echo "contrastive-data-boundary: PASSED"

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
# NOTE: Coverage tracks the main aprender library only.
# Subcrate tests still RUN (--workspace), exercising main lib code paths,
# but subcrate source files are excluded from the coverage REPORT.
# External deps (trueno, realizar, .cargo) also excluded.
# Subcrate code, external deps, and modules requiring external model files for coverage.
# models/ = dead code per UCBD §9.1 (scheduled for deletion).
# serialization/ = SafeTensors IO (needs actual .safetensors files).
# speech/ = like audio/ (already excluded), speech recognition IO.
# format/onnx = ONNX format support (needs .onnx files).
# format/converter = format conversion (needs model files, covered by integration tests).
# format/rosetta = cross-format parity (needs model files).
# transfer/ = transfer learning (needs pretrained models).
# bench/ = benchmark visualization (non-core).
COVERAGE_EXCLUDE_REGEX := \.cargo/|trueno|realizar/|entrenar/|fuzz/|golden_traces/|hf_hub/|demo/|test_factory|pacha/|showcase/|apr-cli/|aprender-shell/|aprender-tsp/|aprender-monte-carlo/|chaos\.rs|audio/|format/quantize\.rs|format/signing\.rs|voice/|playback\.rs|rustlib/src/rust|models/|serialization/|speech/|format/onnx|format/converter|format/rosetta|transfer/|bench_viz/

# Coverage threshold (enforced: fail if below)
COV_THRESHOLD := 95

# Enforced RATCHET floor, distinct from the aspirational target above.
#
# Measured 2026-07-29 by the nightly on 95145584f (the commit that fixed the
# measurement itself): TOTAL: 786448/885829 lines covered = 88.78%. The 95%
# target is real but is NOT where the tree is, so gating on 95 today would paint
# the nightly permanently red and train everyone to ignore it - the exact
# "gate that cannot turn red usefully" failure this repo keeps finding.
#
# So the enforced condition is "do not regress below what we actually have".
# Raise this number whenever a run comes in higher; never lower it to make red
# go away. Integer truncation gives ~0.78pt of headroom before 88 becomes 87.
COV_FLOOR := 88

# NVMe target dir (mirrors cargo() shell function that sets CARGO_TARGET_DIR)
# Without this, Make's subshell bypasses the function and uses ./target/ instead
# of /mnt/nvme-raid0/targets/aprender, causing profraw/binary mismatch.
NVME_TARGET_DIR := $(wildcard /mnt/nvme-raid0)
ifdef NVME_TARGET_DIR
  COV_TARGET_DIR := /mnt/nvme-raid0/targets/aprender
else
  COV_TARGET_DIR :=
endif
COV_CARGO_ENV := $(if $(COV_TARGET_DIR),CARGO_TARGET_DIR=$(COV_TARGET_DIR))

# Coverage: SINGLE-phase (tests instrument AND write the report in one invocation).
#
# This was a two-phase pattern (`test --no-report`, then a separate `report`) and it
# silently measured NOTHING: every run reported "TOTAL: 0/0 lines covered (0%)".
#
# Why: `cargo llvm-cov report` takes its package scope from the CURRENT package, and it
# does NOT accept --workspace/--exclude ("--workspace is specific to [test,nextest,...]
# and not supported for subcommand 'report'"). Phase 1 instrumented
# `--workspace --exclude aprender-gpu`, phase 2 then reported on the ROOT package - which
# is a facade with no code - so the LCOV came out empty and COV_PCT computed to 0. Same
# facade trap .github/workflows/ci.yml:60-64 already documents for sovereign-ci.
#
# Verified on a multi-package run (aprender-common + aprender-bench-compute), in the
# SHARED target dir this Makefile uses:
#   two-phase, unscoped report  -> LH=0   LF=0    (empty)
#   report --summary-only -p A -p B -> LH=686 LF=737  (93.08%)
#   single-phase --lcov --output-path -> LH=686 LF=737  (93.08%)
# Single-phase is chosen over an explicit -p list because the invocation that selects the
# scope is the one that writes the report, so the two cannot drift apart again. profraw
# survive it (31 present afterwards), so coverage-html still has data to work from.
.PHONY: coverage-check contracts

# Alias the dogfood pre-release protocol looks for. It expects `coverage-check`;
# without it the gate reports WARN ("verify >=95% manually"), i.e. a release gate
# that asks a human to do the measurement is not a gate. `coverage` already
# enforces COV_FLOOR, so this is a name, not a new policy.
coverage-check: coverage

# Ditto for `contracts`. The provable-contract tier is a HARD release gate per
# CLAUDE.md, and the dogfood protocol looked for a target that did not exist, so
# it WARNed instead of checking. `pv lint` runs validate + audit + score across
# contracts/ and is the documented entry point (never hand-rolled bash).
contracts:
	@echo "== provable contracts: pv lint contracts/ =="
	@. scripts/pv_bin.sh && "$$PV" lint contracts/ 2>&1 | tail -5
	@echo "== contract engine tests =="
	@cargo test -p aprender-contracts --lib 2>&1 | grep -E "test result" | tail -1

coverage: ## Coverage summary + threshold check (warm: ~3min)
	@echo "📊 Running coverage ($(COV_THRESHOLD)%+ threshold)..."
	@which cargo-llvm-cov > /dev/null 2>&1 || { cargo install cargo-llvm-cov --locked || exit 1; }
	@test -f ~/.cargo/config.toml && mv ~/.cargo/config.toml ~/.cargo/config.toml.bak || true
	@# Pre-clean: remove stale profraw files to avoid LLVM version mismatch
	@COVDIR=$$($(COV_CARGO_ENV) cargo llvm-cov show-env 2>/dev/null | grep CARGO_LLVM_COV_TARGET_DIR | sed "s/.*=//"); \
	if [ -n "$$COVDIR" ]; then find "$$COVDIR" -name '*.profraw' -delete 2>/dev/null || true; fi
	@mkdir -p target/coverage
	@printf '%s' '$(COVERAGE_EXCLUDE_REGEX)' > target/coverage/.exclude-re
	@echo "🧪 Tests with instrumentation + report in ONE invocation (CB-127-A: cargo llvm-cov test, not nextest)..."
	@PROPTEST_CASES=10 QUICKCHECK_TESTS=10 RUST_MIN_STACK=16777216 CARGO_BUILD_JOBS=4 \
		$(COV_CARGO_ENV) cargo llvm-cov test \
		--workspace --exclude aprender-gpu --lib \
		--lcov --output-path target/coverage/lcov.info \
		--ignore-filename-regex "$$(cat target/coverage/.exclude-re)" \
		-- --skip prop_gbm_expected_value --skip slow --skip heavy --skip h12_ --skip j2_ \
		   --skip falsification --skip chaos --skip disconnect --skip benchmark_parity \
		   --skip qwen2_generation --skip qwen2_golden --skip qwen2_weight --skip load_test \
		   --skip spec_checklist_w --skip spec_checklist_u --skip verify_audio --skip g9_roofline \
		   --skip cuda --skip gpu_ \
		|| { test -f ~/.cargo/config.toml.bak && mv ~/.cargo/config.toml.bak ~/.cargo/config.toml; exit 1; }
	@echo "📊 Parsing LCOV for the threshold check..."
	@# Parse LCOV for line coverage (LH=lines hit, LF=lines found)
	@LH=$$(awk -F: '/^LH:/{s+=$$2} END{print s+0}' target/coverage/lcov.info); \
	LF=$$(awk -F: '/^LF:/{s+=$$2} END{print s+0}' target/coverage/lcov.info); \
	if [ "$$LF" -gt 0 ]; then COV_PCT=$$((LH * 100 / LF)); else COV_PCT=0; fi; \
	echo "TOTAL: $$LH/$$LF lines covered ($${COV_PCT}%)"; \
	echo "TOTAL $$LH $$LF $${COV_PCT}%" > target/coverage/summary.txt; \
	mkdir -p .pmat-metrics || exit 1; \
	printf '{"coverage_pct":%s}' "$$COV_PCT" > .pmat-metrics/coverage.result; \
	echo "   wrote .pmat-metrics/coverage.result ($${COV_PCT}%) for pmat score"; \
	test -f ~/.cargo/config.toml.bak && mv ~/.cargo/config.toml.bak ~/.cargo/config.toml || true; \
	if [ "$$COV_PCT" -lt "$(COV_FLOOR)" ]; then \
		echo "❌ REGRESSION: coverage $${COV_PCT}% fell below the enforced floor $(COV_FLOOR)%"; \
		echo "   The floor is the last measured value, so this means coverage went DOWN."; \
		echo "   Add tests for what you changed, or justify and lower COV_FLOOR deliberately."; \
		exit 1; \
	elif [ "$$COV_PCT" -lt "$(COV_THRESHOLD)" ]; then \
		echo "✅ Coverage $${COV_PCT}% holds the floor $(COV_FLOOR)% (target is $(COV_THRESHOLD)%, not yet reached)"; \
		if [ "$$COV_PCT" -gt "$(COV_FLOOR)" ]; then \
			echo "   ⬆  Above the floor - raise COV_FLOOR to $${COV_PCT} to lock the gain in."; \
		fi; \
	else \
		echo "✅ Coverage $${COV_PCT}% meets threshold $(COV_THRESHOLD)%"; \
	fi

# Fast coverage alias
coverage-fast: coverage

# HTML + LCOV reports (run after 'make coverage' to generate browseable report)
# KNOWN DEFECT (same root cause as `coverage` above, NOT yet fixed): both `report` calls
# below are unscoped, so they report the root facade and produce an EMPTY html/lcov. They
# need either an explicit `-p <pkg>` list or to be folded into the instrumenting run, the
# way `coverage` now is. Left as-is here because it is report-only cosmetics and does not
# gate anything - unlike `coverage`, whose 0% fed the >=95% threshold check.
coverage-html: ## Generate HTML + LCOV reports from last coverage run
	@echo "📊 Generating HTML + LCOV reports..."
	@test -f ~/.cargo/config.toml && mv ~/.cargo/config.toml ~/.cargo/config.toml.bak || true
	@mkdir -p target/coverage
	@printf '%s' '$(COVERAGE_EXCLUDE_REGEX)' > target/coverage/.exclude-re
	@$(COV_CARGO_ENV) cargo llvm-cov report --html --output-dir target/coverage/html --ignore-filename-regex "$$(cat target/coverage/.exclude-re)"
	@$(COV_CARGO_ENV) cargo llvm-cov report --lcov --output-path target/coverage/lcov.info --ignore-filename-regex "$$(cat target/coverage/.exclude-re)"
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
		$(COV_CARGO_ENV) cargo llvm-cov test --no-report --workspace --lib --all-features \
		--ignore-filename-regex "$$(cat target/coverage/.exclude-re)" \
		-- --skip prop_gbm_expected_value --skip slow --skip heavy --skip benchmark --skip h12_ --skip j2_
	@$(COV_CARGO_ENV) cargo llvm-cov report --html --output-dir target/coverage/html --ignore-filename-regex "$$(cat target/coverage/.exclude-re)"
	@$(COV_CARGO_ENV) cargo llvm-cov report --lcov --output-path target/coverage/lcov.info --ignore-filename-regex "$$(cat target/coverage/.exclude-re)"
	@echo ""
	@$(COV_CARGO_ENV) cargo llvm-cov report --summary-only --ignore-filename-regex "$$(cat target/coverage/.exclude-re)"
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
	@# `cmd | grep ... || echo` reads GREP's status, not cargo's, so this target
	@# exited 0 while printing 1,828 lines of duplicates. Nothing invoked it either.
	@# Same class as #2336/#2360. Redirect, then read the real status.
	@cargo tree --duplicates > /tmp/apr-dup.txt 2>&1; \
	if [ -s /tmp/apr-dup.txt ]; then \
		echo "FAIL: duplicate dependencies present:"; cat /tmp/apr-dup.txt; exit 1; \
	fi; \
	echo "OK: no duplicate dependencies"
	@cargo audit > /tmp/apr-audit.txt 2>&1; rc=$$?; \
	if [ $$rc -ne 0 ]; then \
		echo "FAIL: cargo audit reported issues:"; cat /tmp/apr-audit.txt; exit 1; \
	fi; \
	echo "OK: cargo audit clean"

# Run cargo-deny checks (licenses, bans, advisories, sources)
deny:
	@echo "🔒 Running cargo-deny checks..."
	@bash scripts/check_deny_exemptions_live.sh
	@bash scripts/check_no_ghsa_banned_crates.sh --self-test
	@bash scripts/check_no_ghsa_banned_crates.sh
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
	@pmat rust-project-score || echo "⚠️  pmat not found — run: cargo install pmat"
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

# NOTE (#2397): `cbtop --ci` now honours the report's own FAIL/red verdict, not
# just the explicit --throughput number. The --simulated pipeline jitters each
# brick +/-20% around its budget, so roughly half land over budget and this
# target exits non-zero. That is the true state of the simulated data; it used
# to print "CI validation passed" over a report that read "Status: FAIL | CI:
# red" only because the exit path never consulted the verdict.
showcase-ci: ## Run showcase benchmark in CI mode with threshold check (RED on simulated data — see #2397)
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
			sudo dnf install -y alsa-lib-devel || exit 1; \
		elif command -v pacman >/dev/null 2>&1; then \
			echo "  Detected: Arch Linux"; \
			sudo pacman -S --noconfirm alsa-lib; \
		elif command -v zypper >/dev/null 2>&1; then \
			echo "  Detected: openSUSE"; \
			sudo zypper install -y alsa-devel || exit 1; \
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

# ============================================================================
# CONTRACT ENFORCEMENT (provable-contracts integration)
# ============================================================================
# Kernel contracts live in-tree at contracts/ (APR-MONO Phase 2b
# consolidation, 2026-04-18). Binding registry:
# contracts/aprender/binding.yaml. Generated tests: tests/contracts/.
# Pre-consolidation `../provable-contracts/` references retired.

PV_BIN := cargo run --release -p aprender-contracts-cli --bin pv --
BINDING := contracts/aprender/binding.yaml
CONTRACTS := contracts/softmax-kernel-v1.yaml \
             contracts/rmsnorm-kernel-v1.yaml \
             contracts/rope-kernel-v1.yaml \
             contracts/attention-kernel-v1.yaml \
             contracts/activation-kernel-v1.yaml \
             contracts/matmul-kernel-v1.yaml \
             contracts/flash-attention-v1.yaml \
             contracts/swiglu-kernel-v1.yaml \
             contracts/gqa-kernel-v1.yaml \
             contracts/layernorm-kernel-v1.yaml \
             contracts/silu-kernel-v1.yaml \
             contracts/cross-entropy-kernel-v1.yaml \
             contracts/adamw-kernel-v1.yaml \
             contracts/ssm-kernel-v1.yaml \
             contracts/conv1d-kernel-v1.yaml \
             contracts/batchnorm-kernel-v1.yaml \
             contracts/kmeans-kernel-v1.yaml \
             contracts/pagerank-kernel-v1.yaml \
             contracts/lbfgs-kernel-v1.yaml \
             contracts/cma-es-kernel-v1.yaml \
             contracts/model-config-algebra-v1.yaml \
             contracts/qk-norm-v1.yaml \
             contracts/tensor-shape-flow-v1.yaml \
             contracts/roofline-model-v1.yaml \
             contracts/gated-delta-net-v1.yaml \
             contracts/format-parity-v1.yaml \
             contracts/shannon-entropy-v1.yaml \
             contracts/f16-conversion-v1.yaml \
             contracts/kernel-launch-budget-v1.yaml \
             contracts/tensor-inventory-v1.yaml \
             contracts/performance-grading-v1.yaml \
             contracts/lora-algebra-v1.yaml \
             contracts/quantization-ordering-v1.yaml \
             contracts/q4k-q6k-superblock-v1.yaml \
             contracts/sampling-algorithms-v1.yaml \
             contracts/validated-tensor-v1.yaml \
             contracts/hybrid-layer-dispatch-v1.yaml \
             contracts/qwen35-shapes-v1.yaml \
             contracts/kv-cache-sizing-v1.yaml \
             contracts/backend-dispatch-v1.yaml \
             contracts/kv-cache-equivalence-v1.yaml

contract-validate: ## Validate all kernel contracts (schema + staleness)
	@echo "Validating kernel contracts..."
	@for contract in $(CONTRACTS); do \
		echo "  $$contract"; \
		$(PV_BIN) validate "$$contract" || exit 1; \
	done
	@echo "Contract validation passed"

contract-test: ## Run contract-driven property tests
	@echo "Running contract property tests..."
	@PROPTEST_CASES=100 cargo test --test contract_tests
	@echo "Contract tests passed"

contract-audit: ## Audit binding coverage (equations -> implementations)
	@echo "Running binding audit..."
	@for contract in $(CONTRACTS); do \
		echo ""; \
		$(PV_BIN) audit "$$contract" --binding $(BINDING); \
	done
	@echo ""
	@echo "Binding audit complete"

contract-regen: ## Regenerate wired test files from contracts
	@echo "Regenerating contract test files..."
	@for contract in $(CONTRACTS); do \
		name=$$(basename "$$contract" .yaml | sed 's/-kernel-v[0-9]*//;s/-v[0-9]*//'); \
		echo "  $$name <- $$contract"; \
		$(PV_BIN) probar "$$contract" --binding $(BINDING) > tests/contracts/$${name}_contract.rs.new 2>/dev/null || true; \
	done
	@echo "Regeneration complete (review .rs.new files)"

contract-check: contract-validate contract-test contract-audit ## Full contract compliance check
	@echo ""
	@echo "Contract compliance check: PASSED"

# ============================================================================
# DEVELOPMENT ENVIRONMENT SETUP (GH-344, GH-345)
# ============================================================================

# Sibling repos required for full-stack development
SIBLINGS := ../realizar ../entrenar ../trueno ../renacer ../provable-contracts ../pacha

dev-setup: ## Set up the dev environment with sibling repo overrides
	@echo "Setting up full-stack development environment..."
	@if [ ! -f .cargo/config.toml ]; then \
		cp .cargo/config.toml.dev-overrides .cargo/config.toml || exit 1; \
		echo "Created .cargo/config.toml with sibling overrides"; \
	elif ! grep -q '\[patch.crates-io\]' .cargo/config.toml; then \
		echo "" >> .cargo/config.toml; \
		cat .cargo/config.toml.dev-overrides >> .cargo/config.toml; \
		echo "Appended sibling overrides to .cargo/config.toml"; \
	else \
		echo ".cargo/config.toml already has [patch.crates-io] section"; \
	fi
	@echo ""
	@$(MAKE) --no-print-directory check-siblings

publish: ## Publish crate(s) to crates.io — strips [patch], publishes, then verifies cargo install
	@echo "Publishing to crates.io (removing [patch.crates-io] temporarily)..."
	@if [ -f .cargo/config.toml ]; then \
		cp .cargo/config.toml .cargo/config.toml.publish-backup || exit 1; \
		echo "# Clean config for publishing" > .cargo/config.toml; \
	fi
	@CRATE=$(CRATE); \
	if [ -z "$$CRATE" ]; then \
		echo "Usage: make publish CRATE=aprender   (or apr-cli, provable-contracts, ...)"; \
		echo "       any crate listed by: python3 scripts/lib/cascade_universe.py ."; \
		echo "       -- INCLUDING the crates/facades/ workspace, which is excluded"; \
		echo "          from the root and which this target could not reach at all"; \
		echo "          before aprender#2559."; \
		echo "Restoring config..."; \
		if [ -f .cargo/config.toml.publish-backup ]; then \
			cp .cargo/config.toml.publish-backup .cargo/config.toml && \
			rm -f .cargo/config.toml.publish-backup; \
		fi; \
		exit 1; \
	fi; \
	echo "Publishing $$CRATE..."; \
	SEL="-p $$CRATE"; \
	MANIFEST=$$(python3 scripts/lib/cascade_universe.py . | awk -F'\t' -v c="$$CRATE" '$$1==c{print $$3}'); \
	WSROOT=$$(python3 scripts/lib/cascade_universe.py . | awk -F'\t' -v c="$$CRATE" '$$1==c{print $$4}'); \
	if [ -z "$$MANIFEST" ]; then \
		echo "FAIL: $$CRATE is not a publishable crate in ANY workspace here."; \
		echo "      (scripts/lib/cascade_universe.py enumerates all of them)"; \
		if [ -f .cargo/config.toml.publish-backup ]; then \
			cp .cargo/config.toml.publish-backup .cargo/config.toml && \
			rm -f .cargo/config.toml.publish-backup; \
		fi; \
		exit 1; \
	fi; \
	if [ "$$WSROOT" != "$$(pwd)" ]; then \
		echo "  ($$CRATE lives in the excluded workspace $$WSROOT; selecting by --manifest-path,"; \
		echo "   because \`cargo publish -p $$CRATE\` from here is rc=101 'did not match any packages')"; \
		SEL="--manifest-path $$MANIFEST"; \
	fi; \
	DRY=""; \
	if [ -n "$$PUBLISH_DRY_RUN" ]; then \
		echo "  (PUBLISH_DRY_RUN set: packaging and resolving, but NOT uploading)"; \
		DRY="--dry-run --no-verify"; \
	fi; \
	cargo publish $$SEL $$DRY --allow-dirty --locked; \
	STATUS=$$?; \
	echo "Restoring .cargo/config.toml..."; \
	if [ -f .cargo/config.toml.publish-backup ]; then \
		cp .cargo/config.toml.publish-backup .cargo/config.toml && \
		rm -f .cargo/config.toml.publish-backup; \
	fi; \
	if [ $$STATUS -ne 0 ]; then \
		echo "FAIL: cargo publish failed"; \
		exit $$STATUS; \
	fi; \
	if [ -n "$$PUBLISH_DRY_RUN" ]; then \
		echo "DRY RUN OK: $$CRATE resolved and packaged; nothing was uploaded."; \
		exit 0; \
	fi; \
	echo ""; \
	echo "=== POST-PUBLISH VERIFICATION (PMAT-517) ==="; \
	echo "Waiting for crates.io index to update..."; \
	sleep 15; \
	if [ "$$CRATE" = "apr-cli" ]; then \
		echo "Verifying: cargo install apr-cli --force ..."; \
		cargo install apr-cli --force 2>&1 | tee /tmp/publish-verify-$$CRATE.log; \
		INSTALL_STATUS=$${PIPESTATUS[0]}; \
		if [ $$INSTALL_STATUS -ne 0 ]; then \
			echo ""; \
			echo "FATAL: cargo install apr-cli FAILED after publish!"; \
			echo "The published crate is BROKEN. You must fix and republish."; \
			echo "Build log: /tmp/publish-verify-$$CRATE.log"; \
			exit 1; \
		fi; \
		echo "Verifying apr --version..."; \
		WANT=$$(grep -m1 '^version' Cargo.toml | sed 's/.*"\(.*\)".*/\1/'); \
		APR_BIN_PATH="$${CARGO_HOME:-$$HOME/.cargo}/bin/apr"; \
		GOT=$$("$$APR_BIN_PATH" --version 2>&1); \
		echo "  expected $$WANT, $$APR_BIN_PATH reports: $$GOT"; \
		case "$$GOT" in \
			*"$$WANT"*) echo "POST-PUBLISH VERIFICATION: PASSED" ;; \
			*) echo "FATAL: published apr reports '$$GOT' but this tree is $$WANT."; \
			   echo "The publish did not produce the binary we think it did."; \
			   exit 1 ;; \
		esac; \
	else \
		echo "Verifying: cargo install apr-cli --force (depends on $$CRATE)..."; \
		cargo install apr-cli --force 2>&1 | tee /tmp/publish-verify-$$CRATE.log; \
		INSTALL_STATUS=$${PIPESTATUS[0]}; \
		if [ $$INSTALL_STATUS -ne 0 ]; then \
			echo ""; \
			echo "FATAL: cargo install apr-cli FAILED after publishing $$CRATE!"; \
			echo "The published $$CRATE broke the apr-cli build."; \
			echo "Build log: /tmp/publish-verify-$$CRATE.log"; \
			exit 1; \
		fi; \
		echo "POST-PUBLISH VERIFICATION: PASSED"; \
	fi

check-wasm32: ## Verify aprender-core still compiles for wasm32-unknown-unknown (aprender#2310)
	@bash scripts/check_wasm32_core_builds.sh

check-siblings: ## Verify sibling repos exist and versions are compatible
	@echo "Checking sibling repositories..."
	@all_ok=true; \
	for repo in $(SIBLINGS); do \
		name=$$(basename "$$repo"); \
		if [ -d "$$repo" ]; then \
			version=$$(grep '^version' "$$repo/Cargo.toml" 2>/dev/null | head -1 | sed 's/.*"\(.*\)"/\1/'); \
			echo "  ✓ $$name ($$version)"; \
		else \
			echo "  ✗ $$name — not found at $$repo"; \
			all_ok=false; \
		fi; \
	done; \
	echo ""; \
	if [ "$$all_ok" = true ]; then \
		echo "All sibling repos present"; \
	else \
		echo "Missing sibling repos. Clone them alongside aprender:"; \
		echo "  cd .. && git clone <repo-url>"; \
		echo ""; \
		echo "Or build standalone (uses crates.io versions):"; \
		echo "  Remove [patch.crates-io] from .cargo/config.toml"; \
	fi
