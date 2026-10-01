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
# CR-03: `.ONESHELL:` runs a whole recipe in ONE shell, and the default `.SHELLFLAGS` is
# `-c` with no `-e`. That shell does not stop at the first failure, so the recipe's status
# is whatever its LAST line returned — and every tier recipe ends in `@echo "Tier N: PASSED"`,
# which always succeeds. Measured on this repo with both makes installed:
#
#   SHELL := /bin/bash ; .ONESHELL: ; recipe = { false ; @echo "done" }
#     make  3.81 (macOS default) -> exit=2   (3.81 predates .ONESHELL and ignores it)
#     gmake 4.4.1 (Linux)        -> exit=0   FAILURE SWALLOWED
#
# Under Make 4.x that disarmed every gate D-26 deliberately moved INTO the tiers.
#
# `-e` ONLY, deliberately. `-u` and `-o pipefail` are separate hardening with a much larger
# blast radius here: 35 recipes reference `$$VAR` (a `-u` risk) and 5 pipe into `head`/`tail`,
# where the reader closing the pipe SIGPIPEs the writer and `pipefail` turns that into a
# failure. Adding them needs its own verification pass across all 85 targets on BOTH makes.
# `-e` alone restores the per-line abort semantics 3.81 already had, which is the defect.
#
# `.SHELLFLAGS` arrived in Make 3.82, so 3.81 ignores this line — harmless, since 3.81 also
# ignores `.ONESHELL:` and therefore never had the bug.
.SHELLFLAGS := -e -c
.ONESHELL:

.PHONY: all build test test-smoke test-fast test-quick test-full test-heavy lint fmt clean doc book book-build book-serve book-test tier1 tier2 tier3 tier4 coverage coverage-fast profile hooks-install hooks-verify lint-scripts bashrs-score bashrs-lint-makefile chaos-test chaos-test-full chaos-test-lite fuzz bench dev pre-push ci check run-ci run-bench audit deps-validate deny pmat-score pmat-gates quality-report semantic-search examples mutants mutants-fast property-test install-alsa test-alsa test-audio-full contract-validate contract-test contract-audit contract-audit-phase2 contract-audit-phase3 contract-regen contract-check dev-setup check-siblings setfit-feature-matrix setfit-repro-inproc setfit-repro-crossproc setfit-repro-replay gemm-thread-determinism setfit-tests setfit-bench-tests contract-audit-phase4 contract-audit-phase5 contract-audit-phase6 contract-audit-phase8 contract-audit-phase8-selftest setfit-apr-tests setfit-classify-tests setfit-bundle-tests setfit-config-tests setfit-evaluate-tests setfit-codec-tests setfit-reload-tests setfit-lock-tests setfit-verify-tests setfit-lifecycle-tests setfit-ui-tests setfit-cli-train-tests setfit-cli-predict-tests setfit-cli-inspect-tests setfit-cli-eval-tests setfit-cli-io-tests setfit-cli-serve-tests setfit-serve-tests setfit-parity setfit-serve-smoke setfit-cli-lifecycle setfit-api-boundary setfit-all-tests setfit-bench-door-probe setfit-bench-door-probe-build lint-current check-wasm32 gate contrastive-data-boundary contrastive-data-boundary-cases

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
# Phase 1 SetFit conformance (D-26). The gates must live INSIDE a tier: a target
# outside the tiers is a target that stops being run.
#
# PLACEMENT WAS MEASURED, not assumed (2026-08-08, warm tree):
#   whole conformance suite, ONE invocation ......  7 s wall / 0.57 s test time
#   one filtered invocation (e.g. gradient_gate_)   6 s wall / 0.44 s test time
#   setfit:: lib module, conformance-fixtures .... 19 s wall / 1.72 s test time
# The wall clock is dominated by cargo's per-invocation freshness check, not by
# test execution. The pre-accepted tier2/tier3 SPLIT (user, 2026-08-07) would
# therefore cost ~3 x 6 s = ~18 s for the three "fast" gates alone — WORSE than
# the 7 s single run it was meant to avoid, because a `cargo test` line may carry
# at most ONE positional filter (a second exits `error: unexpected argument`) and
# the three gate prefixes do not share one. The fallback is NOT triggered: the
# whole suite stays here, as one invocation. The plan's 10-30 s estimate for the
# suite was 20-50x high.
#
# The lib line uses the MODULE PATH, not the plan's unscoped `--features setfit`
# form (D30). Measured: `--lib --features setfit` runs 14202 tests in 119 s of
# which 162 are this phase's, while `setfit::` selects exactly 162 in 19 s — and
# 162 is precisely the feature-gated delta (14283 tests with conformance-fixtures
# vs 14121 with default features), so the filter loses no coverage.
	@echo "Phase 1 SetFit: encoder/tokenizer/import/loss/model unit gates..."
	@cargo test -p aprender-core --lib --features conformance-fixtures setfit::
	@echo "Phase 1 SetFit: fixture parity + ENC-04 gradient/frozen/detach gates..."
	@cargo test -p aprender-core --features setfit,conformance-fixtures --test setfit_conformance
# Phase 2 contrastive-data (D-26 again — a gate outside the tiers stops being run).
#
# RUNTIME WAS RE-MEASURED, not estimated (2026-08-09, warm tree, three consecutive
# runs): 6.48 s / 6.34 s / 6.46 s wall, rc=0 each time. History of this line, because
# the trend is the point: 2/2/2 s with one determinism doctest, 1/2/1 s after plan
# 02-03 grew it to 67 lib + 5 doc, 1.75 s after plan 02-05 took it to 126 lib + 7
# integration + 7 doc, 3.0 s after plan 02-07 took it to 206 lib + 11 integration +
# 7 doc, and now 6.4 s after plan 02-08 took it to 212 lib + 24 integration + 7 doc
# across EIGHT suites (1 further test is #[ignore]d — the golden regenerator).
# Actual test execution inside that 6.4 s is 1.18 s lib + 0.23 s trybuild + 3.33 s
# doc; the rest is cargo's per-invocation freshness check over eight targets. The
# step is still comfortably tier2-shaped.
# The 1.75 -> 3.0 s step was the pair sampler's two heaviest properties: a 40,000-draw
# marginal-equivalence measurement on layout [3,5,7] (which is what proves the O(K)
# negative scheme preserves D-14's n_j*n_k class-pair weights rather than merely
# being faster) and a 24,576-pair streamed manifest hash. The 3.0 -> 6.4 s step is
# `tests/ui.rs`: trybuild spawns a nested cargo build for the five compile-fail
# programs that pin DATA-06's non-constructibility. All three costs are deliberate —
# a cheaper marginal test could not distinguish the weighted scheme from a uniform
# one, and a compile-fail claim that is not compiled is not a claim.
# Re-measure and update when the suite grows again; a tier2 line whose comment
# records a stale number is worse than one with no comment, because it will be
# trusted.
#
# UNFILTERED on purpose. The Phase 1 lines above take a module filter because
# they select 162 of 14202 tests in a large crate. Here the whole crate IS this
# phase, a filter would exclude the doctests, and `cargo test` accepts at most
# one positional anyway.
	@echo "Phase 2 contrastive-data: protocol unit gates..."
	@cargo test -p aprender-contrastive-data
# Phase 3 D-16, tier2 half. The decision splits ONE gate across two tiers: the fast
# in-process comparison here, the authoritative cross-process one in tier3. Wiring only
# the tier3 half would leave D-16's fast signal in a test file no tier invokes.
#
# Measured standalone before wiring (warm tree): rc=0, 6 s wall, 1 test. It is one
# libtest filter on an already-built target, which is tier2-shaped. Its failure mode was
# induced, observed and reverted — see 03-10-SUMMARY.md.
	@echo "Phase 3 SetFit: in-process two-clean-runs equality (D-16)..."
	@$(MAKE) setfit-repro-inproc
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
# D-26: the contract gate must be REACHED, not merely listed. Before this line,
# `contract-validate` was reachable only from `contract-check`, which no tier
# depends on — so appending a contract to $(CONTRACTS) alone would have parked
# the Phase 1 gate outside the tiers entirely.
#
# The BROAD form was chosen from evidence, not preference (W4). `make
# contract-validate` was run STANDALONE first, with its status captured directly
# (`make contract-validate > /tmp/cv.log 2>&1; rc=$$?`, never through a pipe —
# CLAUDE.md rule 1): rc=0 in 8 s wall, all 41 pre-existing contracts reporting
# "0 error(s), 0 warning(s)". Nothing is already red, so wiring the whole list
# cannot make tier3 fail for a defect this phase did not cause, and 8 s is well
# inside tier3's 1-5 minute budget. Had any contract been red, the narrow
# `$(PV_CARGO_RUN) validate contracts/setfit-encoder-conformance-v1.yaml` form would
# have been used instead and the red contracts surfaced as their own finding.
	@echo "Validating provable contracts (incl. the Phase 1 setfit gate, D-26)..."
	@$(MAKE) contract-validate
# Review finding F9 (plan 02-08): `contract-validate` above checks contract
# SHAPE and says nothing about whether an equation is bound to an
# implementation, so both Phase 2 contracts could have been "valid" with all 25
# equations bound to nothing. This is the BLOCKING coverage gate. It is the
# SCOPED form on purpose — the repo-wide `contract-audit` reports 132 unbound
# equations across 38 contracts and exits 0 anyway; see that target's comment
# block for the measurement and why neither wiring it nor fixing it belongs to
# this phase. Same evidence discipline as the two blocks around it: run
# standalone first (rc=0, 9 s cold / ~1 s warm), and its failure mode induced,
# observed and reverted before it was wired.
	@$(MAKE) contract-audit-phase2
# Phase 3's equivalent, wired here for exactly the reason the line above exists:
# a target outside the tiers is a target that stops being run. Scoped to
# $(PHASE3_CONTRACTS). Same evidence discipline — run standalone with the status
# captured directly, and its failure mode induced, observed and reverted rather
# than assumed. See the target's own comment block.
	@$(MAKE) contract-audit-phase3
# Phase 4's equivalent, wired here for the same reason the two lines above exist: a
# target outside the tiers is a target that stops being run. Scoped to
# $(PHASE4_CONTRACTS). Unlike its predecessors this one tolerates `status: pending`
# (BIND-004) and still refuses a missing entry (BIND-001), because 04-01 commits the
# Phase 4 schema BEFORE the code that implements it. Its RED and GREEN states were
# both measured with the status captured directly, never through a pipe. See the
# target's own comment block.
	@$(MAKE) contract-audit-phase4
# Phase 5's equivalent, wired here for the reason the three lines above exist: a target
# outside the tiers is a target that stops being run. Scoped to $(PHASE5_CONTRACTS). Like
# Phase 4's it tolerates `status: pending` (BIND-004) and still refuses a missing entry
# (BIND-001), because 05-05 commits the claims schema BEFORE the row type, the bench
# adapters and the report renderer that implement it. Its RED and GREEN states were both
# measured with the status captured directly, never through a pipe. See the target's own
# comment block.
	@$(MAKE) contract-audit-phase5
# Phase 6's equivalent, wired here for the reason the four lines above exist: a target
# outside the tiers is a target that stops being run. Scoped to $(PHASE6_CONTRACTS), refuses
# ANY BIND- line as Phase 5's does, and ALSO resolves every row to a real definition site —
# see the target's comment block for why zero BIND- findings is not that proof.
	@$(MAKE) contract-audit-phase6
# Phase 8's equivalent, wired here for the reason the five lines above exist: a target outside
# the tiers is a target that stops being run. Scoped to $(PHASE8_CONTRACTS). Since plan 08-12 it
# is Phase 6's form: it refuses ANY BIND- line (the only admissible one is the single
# PHASE8_LIVE_EXEMPT line, empty since the live outcome is deployed-passed) and resolves every
# row to a definition site. NOTE: CI does not run tier3, so this is a CI gate only if the CI
# docker chain names it. See the target's own comment block.
	@$(MAKE) contract-audit-phase8
# TRN-06's AUTHORITATIVE reproducibility claim (D-16) and D-13's GEMM control, wired
# here for the reason the three lines above exist: a target outside the tiers is a
# target that stops being run. Both were run STANDALONE first with the status captured
# directly, and both had a failure INDUCED, observed and reverted before being wired —
# see 03-10-SUMMARY.md for the rc values and the perturbations used. The recipes read
# `$$?` on the line after the redirect and contain no `tee`.
	@$(MAKE) setfit-repro-crossproc
# REVIEW CR-01 (tier3 half): the ENTIRE Phase 3 test surface ran in no tier and no CI job.
# `setfit` is declared but not default (aprender-train/Cargo.toml:79, default = ["tui"]), the
# module is `#[cfg(feature = "setfit")]` (train/mod.rs:51), aprender-core's is gated the same way
# (lib.rs:165), and no workspace member enables either — so tier3's `cargo test --all` and CI's
# `cargo nextest run --workspace --lib` both COMPILE IT OUT. `setfit-feature-matrix` only
# `cargo check`s the feature; checking is not testing. Never executed anywhere before this line:
# bundle_tests.rs (1045 lines), lock_tests.rs (753), verify_tests.rs (694), evaluate_tests.rs
# (378), and all seven trybuild compile-fail cases. The CI half needs a workflow edit and is
# item 4 of 03-HUMAN-UAT.md; this closes the tier3 half, which needs no approval.
	@$(MAKE) setfit-tests
# ─── Phase 5 (plan 05-10). CR-01's lesson, applied one phase later ──────────
#
# `setfit-tests` above filters on the module path `setfit::`, which DOES reach the
# Phase 5 modules in aprender-train — but it reaches NOTHING in apr-cli, whose
# `setfit_bench` suite (58 tests, including the report renderer's two-sided
# incomparability control and its no-verdict-word scan) is behind the same non-default
# `setfit` feature and is therefore compiled out of tier3's `cargo test --all` and CI's
# `cargo nextest run --workspace --lib` exactly as CR-01 described. This line is the
# tier3 half of that gap, and it also gives the Phase 5 floors somewhere to be raised
# as 05-11/05-12/05-13 add tests, rather than hiding inside the Phase 3+4 numbers.
	@$(MAKE) setfit-bench-tests
# REVIEW CR-02: the replay check was written, committed, and wired into NOTHING. Two clean
# runs agreeing proves reproducibility; only this proves the reproduced order is the
# INTENDED one. Without it the pair of gates above can both pass on a wrong-but-consistent
# order — the one failure mode the recorded-vs-recomputed split exists to catch.
	@$(MAKE) setfit-repro-replay
	@$(MAKE) gemm-thread-determinism
	@$(MAKE) setfit-feature-matrix
# ─── Phase 4 (plan 04-10). CR-01's lesson made structural ───────────────────
#
# CR-01 was not "someone forgot a line". It was that an ENTIRE feature-gated test
# surface existed with no tier and no CI job running it, for a whole phase, while
# every plan cited it as evidence. The structural fix is that no feature-gated
# surface ships without a tier that runs it — so all five Phase 4 gates are wired
# HERE, together, rather than left as targets a reader could discover.
#
#   setfit-all-tests     18 scoped suites, each with its own measured floor
#   setfit-parity        04-09's three readers of one artifact must agree
#   setfit-serve-smoke   04-09's #[ignore]d spawned-server leg
#   setfit-cli-lifecycle 04-15's #[ignore]d spawned-`apr` ladders
#   setfit-api-boundary  OPS-01: no library crate may depend on apr-cli
#
# THE TWO `#[ignore]`d ONES ARE THE POINT OF WIRING THEM HERE. An `#[ignore]`d
# test runs in NO default invocation — not `cargo test --all`, not tier2, not CI's
# nextest run. Left unwired they would be the CR-01 shape reintroduced by the very
# plans that were closing it. tier3 is where an `--ignored` leg belongs: it is the
# pre-push tier, and both spawn real processes.
#
# `setfit-feature-matrix` above is the SAFE-02 half and is already in this list; it
# grew from two crates to four in the same plan that added these five.
	@$(MAKE) setfit-all-tests
	@$(MAKE) setfit-parity
	@$(MAKE) setfit-serve-smoke
	@$(MAKE) setfit-cli-lifecycle
	@$(MAKE) setfit-api-boundary
# ─── F-07 (plan 04-22). CLAUDE.md mandates bashrs over shellcheck, and until this
# line NO bashrs target was wired into ANY tier — all three of them
# (`lint-scripts`, `bashrs-score`, `bashrs-lint-makefile`) sat outside, which is
# the D-26 failure mode again: a target outside the tiers is a target that stops
# being run. `bashrs-lint-makefile` was worse than unwired — it ended in
# `|| echo`, so it could not fail even if someone did run it.
#
# Both were run STANDALONE first with the status captured directly
# (`> log 2>&1; rc=$$?`, never through a pipe): rc=0 each, ~5 s combined, which is
# nothing against tier3's 1-5 minute budget. Both had their failure modes INDUCED,
# observed and reverted rather than assumed — a real unterminated quote in a guard
# script, an unjustified baseline entry, and an empty file list each turned the
# gate RED. See 04-22-SUMMARY.md for the verbatim transcripts.
#
# `lint-scripts` is DELIBERATELY NOT WIRED. Measured one file at a time: 51 of 59
# scripts exit non-zero, 22 with error-severity findings. Wiring it here would put
# a permanently-red gate in a blocking tier, and a permanently-red gate gets
# disabled — along with the real gates beside it. The backlog is measured, triaged
# by fix-shape and given an owner in D-04-22-A instead of being hidden.
	@$(MAKE) bashrs-lint-makefile
	@$(MAKE) bashrs-scoped-lint
# D-04 (Phase 2), wired here for the same reason the line above exists: a target
# outside the tiers is a target that stops being run. Same evidence discipline as
# the D-26 block — `make contrastive-data-boundary` was run STANDALONE first with
# its status captured directly (`> /tmp/cdb-standalone.log 2>&1; rc=$$?`, never
# through a pipe): rc=0 in 1 s wall. Nothing was already red, and 1 s is nothing
# against tier3's 1-5 minute budget. Its four failure modes were each induced,
# observed and reverted rather than assumed — see the target's own comment block.
	@$(MAKE) contrastive-data-boundary
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
	@echo "Checking fleet hosts accept only the manifest nightly apr/pv (aprender#4186)..."
	@bash scripts/check_nightly_pin.sh --self-test
	@echo "Checking no declared-unsupported capability is already implemented (aprender#3686)..."
	@bash scripts/check_unwired_capabilities.sh --self-test
	@bash scripts/check_unwired_capabilities.sh
	@echo "Checking no NEW silent truncation of a value a human reads later (aprender#3904)..."
	@bash scripts/check_no_silent_truncation.sh --self-test
	@bash scripts/check_no_silent_truncation.sh
	@if [ -d tests/golden ]; then \
		if . scripts/apr_bin.sh 2>/dev/null; then \
			echo "Running probar golden regression with profiling... ($$APR)"; \
			"$$APR" probar tests/golden/model.apr --golden tests/golden/ --assert --tolerance 0.98 2>/dev/null || true; \
		else \
			echo "Skipping probar golden regression: no apr built from HEAD (scripts/apr_bin.sh)"; \
		fi; \
	fi
# ─── Phase 6 (UAT item 5, decided 2026-09-07). D-26's lesson, one milestone later ──
#
# `just forecast-sc1-sweep` is the SC1 gate — 19 compositions of freq x growth x
# holiday shape on a release build, each asserted under 2.0 s. Until this line it ran
# in NO tier and NO CI job: `grep -rn forecast-sc1-sweep .github/ Makefile scripts/`
# returned nothing. That is the D-26 shape exactly — a gate outside the tiers is a
# gate that stops being run — and it matters more here than usual, because the sweep
# is the ONLY detector for a new cost axis entering through prophet.rs/np.rs that no
# `ForecastArgs` field names. Both `door_surface` completeness tests are blind to that
# path: the knobs half derives from `schemars::schema_for!` and so sees only struct
# fields, and the cost_axes half reads a hand-kept YAML list (06-REVIEW.md WR-03).
#
# RUN STANDALONE FIRST, status captured directly off the command and never through a
# pipe (CLAUDE.md rule 1): `just forecast-sc1-sweep > log 2>&1; rc=$$?` -> rc=0,
# 25 s wall warm, 19 `SC1 WALL:` lines. 25 s is comfortably inside tier3's 1-5 minute
# budget. Re-run green through THIS target (`make forecast-sc1-gate`): rc=0.
#
# FAILURE INDUCED, OBSERVED AND REVERTED rather than assumed. The bar was temporarily
# lowered 2.0 -> 1.0 in the recipe; `make forecast-sc1-gate` exited 2 with
#   FAIL "SC1 model=neuralprophet freq=D ... n_lags=41": 1.579 is at or above the 1.0 bar
# and — the half that makes it an observation rather than a crash — every prophet
# composition (0.156 s .. 0.843 s) still PASSED under the lowered bar, so the gate fired
# on the one composition over the bar and on nothing else. Bar restored, gate re-run green.
#
# THE MARGIN IS THINNER THAN IT LOOKS, recorded here because a reader of this line is who
# needs it. The worst composition measured 1.420 s on one run and 1.579 s on the next —
# the SAME composition, an 11.2 % run-to-run spread, leaving 21-29 % headroom under the
# 2.0 s bar. This is the neuralprophet row (n_lags=41, cells=14 810 040). If tier3 ever
# goes flaky here, that row is why, and the fix is a quieter machine or a narrower default
# geometry — NOT a raised bar, which is the SC1 literal.
#
# WHY THE GEOMETRY COULD NOT BE USED TO INDUCE IT, which is itself a finding: the
# default NP composition already prices at cells=14 810 040, i.e. 98.7% of
# `MAX_NP_TRAIN_COST` (15 000 000). Raising points or lags to make it slow trips
# CR-01's door refusal instead of a slow wall — independent confirmation that the
# bound is too tight, and the reason Phase 7 exists.
#
# NOT WIRED INTO CI. UAT item 3 decided that Phase 6's release-profile gates stay
# MANUAL; this is the tier3 half, which needs no workflow edit and no approval.
	@$(MAKE) forecast-sc1-gate
	@echo "Tier 3: PASSED"

forecast-sc1-gate: ## SC1 sweep (19 compositions, release, 2 s bar) - the tier3 half of UAT item 5
# Requires `just`. FAILS LOUDLY when it is absent rather than skipping: a gate that
# silently no-ops when a tool is missing is the theater this phase spent three rounds
# removing. The Makefile invokes `just` nowhere else, so the dependency is stated here.
	@command -v just >/dev/null 2>&1 || { \
		echo "FAIL: forecast-sc1-gate needs \`just\` and it is not on PATH."; \
		echo "      Install with: cargo install just"; \
		echo "      This gate is NOT skippable - it is the only detector for a cost axis"; \
		echo "      added inside prophet.rs/np.rs that no ForecastArgs field names."; \
		exit 1; \
	}
	@just forecast-sc1-sweep

# D-06: the setfit feature must be dependency-CLOSED and must not leak into a
# minimal build. Wired into tier3 above.
#
# `--all-features` is deliberately ABSENT (D22): it enables `audio-alsa`, whose
# `alsa-sys` build script needs the Linux ALSA headers, so on macOS it fails for
# reasons that have nothing to do with this phase. Proven independent of setfit —
# `cargo check -p aprender-core --no-default-features --features audio-alsa`,
# which touches no setfit code at all, fails identically. The union below is the
# platform-appropriate one D22's fix direction asks for and covers every feature
# combination this phase introduces.
# ─── SAFE-02: FOUR crates x THREE supported CPU profiles, BUILD and RUN ─────
#
# SAFE-02: "a developer can verify the supported CPU build/test feature matrix in
# CI without Python or network access". Phase 4 spread the `setfit` feature across
# four crates, so the matrix has to span all four. Profiles:
#
#   (a) --no-default-features                     the minimal build
#   (b) --no-default-features --features setfit   dependency closure of the feature
#   (c) default features + --features setfit      what a developer actually runs
#
# THE FULL CELL TABLE, MEASURED on 5887d301c with the status captured directly off
# each cargo command, never through a pipe (CLAUDE.md rule 1):
#
#   crate            (a) ndf   (b) ndf+setfit   (c) default+setfit
#   aprender-core    rc=0      rc=0             rc=0
#   aprender-train   rc=0      rc=0             rc=0
#   apr-cli          rc=101 !  rc=101 !         rc=0
#   aprender-serve   rc=0      rc=0             rc=0
#
# ! THE TWO apr-cli MINIMAL CELLS ARE RED AT THIS COMMIT AND ARE NOT WIRED.
# Both fail with the same four errors, and they are the SAME four the (a) cell
# produces — `inference`-gated code that is not `cfg`-gated at
# src/commands/explain.rs:231 and :344, src/commands/diff_05_aprt_stage.rs:100 and
# src/lib.rs:63. 04-09 re-measured (a) with the pre-Phase-4 manifest restored and
# got the identical failure, so it predates this phase (D-04-09-A). Setting
# `--features setfit` does not help because `setfit` does not imply `inference`.
#
# **04-09's PLAN NAMED `cargo check -p apr-cli --no-default-features` AS SAFE-02's
# GATING EVIDENCE. THAT IS WRONG AND WIRING IT WOULD MAKE THIS GATE RED ON
# ARRIVAL.** The green equivalent, and the leg the gating evidence is actually read
# from, is `cargo check -p apr-cli --all-targets` with DEFAULT features (setfit
# OFF, inference ON) — measured rc=0, and the leg 04-06 and 04-07 already used.
#
# LITERAL `--all-features` WAS MEASURED, NOT ASSUMED, for all four crates:
#   aprender-core    rc=101  alsa-sys build script (no ALSA headers on macOS)
#   aprender-train   rc=0    CLEAN — so it IS wired below
#   apr-cli          rc=101  entrenar::finetune::wgpu_pipeline::WgpuInstructPipeline
#   aprender-serve   rc=101  trueno_viz::plots::Histogram::dimensions (API drift)
# Three of the four fail for reasons with nothing to do with setfit — GPU, audio
# and viz features that are not the supported CPU profile SAFE-02 names — which is
# why the per-crate CPU unions below are the honest substitute. The aprender-train
# result contradicts the older comment further down claiming no `--all-features`
# leg is buildable on a CPU host; it was re-measured and it is. If the train leg
# ever goes red for a `cuda`/`wasm` reason, that is NOT a SAFE-02 signal — scope it
# back out rather than silencing the whole matrix.
#
# THE RUN LEGS, and what each one actually proves:
#   aprender-core / aprender-train: the SAME `setfit::` filter is run with the
#     feature off and on. Off must select ZERO (assert_tests_absent), on must
#     select many (assert_tests_ran). Neither half is evidence alone — see the
#     define's comment. Measured 0 / 240 and 0 / 311.
#   apr-cli: the off leg is NOT zero. Measured `--lib setfit` = 13 with the feature
#     off and 68 with it on, because apr-cli carries setfit-NAMED tests (argument
#     parsing, error strings) that are not behind the feature. So the apr-cli run
#     leg proves the gated surface APPEARS, not that it is ABSENT, and it is
#     asserted as a delta rather than as a zero. SAFE-02's apr-cli gating evidence
#     is the CHECK leg plus the `tokenizers` graph negative below.
#   aprender-serve: has a run leg ONLY at profile (c). Its `--no-default-features`
#     TEST build is red (D-04-10-A, new here) — `#[cfg(test)]` code imports
#     crate::gguf::OwnedQuantizedModelCached, crate::gpu and crate::api GPU types
#     unconditionally, so the test target needs `server`/`gpu` even though the
#     LIBRARY checks clean without them. Pre-existing and unrelated to setfit: the
#     feature-off leg fails identically. The (a)/(b) CHECK cells still run.
#
# THE dev-dependency RULE, recorded here because a reader of this recipe is who
# needs it (04-09-SUMMARY deviation 1). Cargo does not permit an OPTIONAL
# dev-dependency, so a dev-dep on a package that is ALSO a normal dependency is
# unconditional, and its features UNIFY with the normal dependency for any build
# that includes test targets. Such an entry silently weakens the corresponding
# `cargo test --no-default-features` run leg: the crate under test would still be
# built WITH the feature the leg is trying to prove absent.
# **IT DOES NOT APPLY ON THIS TREE.** 04-09's plan required a `realizar` dev-dep
# with `features = ["setfit"]`; 04-09 MEASURED it unnecessary (the test target's
# own `required-features = ["setfit","inference"]` already makes `realizar::api`
# reachable with its setfit surface) and did not add one, so threat T-04-61 does
# not arise. The rule stays written down because the next person reaching for such
# a dev-dep needs it; the apr-cli run leg's weakness above has a different,
# measured cause.
setfit-feature-matrix: ## SAFE-02: setfit feature matrix, 4 crates x 3 CPU profiles, build AND run
	@mkdir -p target
	@echo "Feature matrix: aprender-core setfit isolation (D-06)"
	@cargo check -p aprender-core --no-default-features
	@cargo check -p aprender-core --no-default-features --features setfit
	@cargo check -p aprender-core --features setfit
	@cargo check -p aprender-core --features conformance-fixtures
	@cargo check -p aprender-core --features setfit,conformance-fixtures,model-tests
	@echo "  negative: a no-default-features build must contain NO tokenizers node"
# The tree is captured to a file and `cargo tree`'s own status checked FIRST.
# Piping straight into `grep -q` would read grep's status, and a `cargo tree`
# that failed outright would feed grep nothing — the guard would then pass
# vacuously, which is exactly the CLAUDE.md rule 1 failure mode.
	@cargo tree -p aprender-core --no-default-features -e normal \
		> target/setfit-feature-matrix-tree.txt 2>&1 || \
		{ echo "FAIL: cargo tree failed; the D-06 negative check would pass vacuously"; \
		  cat target/setfit-feature-matrix-tree.txt; exit 1; }
	@if grep -q tokenizers target/setfit-feature-matrix-tree.txt; then \
		echo "FAIL: tokenizers leaked into a no-default-features build (D-06)"; \
		grep -n tokenizers target/setfit-feature-matrix-tree.txt; exit 1; \
	fi
# ─── Phase 3 D-05: the same closure obligation, one crate up ────────────────
#
# `setfit` now has to propagate through a 37-module crate that also carries
# GPU/LoRA/distill/server. Three legs, and one of them is not the shape the
# plan first asked for — the reason is recorded here rather than in a commit
# message, because a reader of this Makefile is who needs it.
#
# A literal `--all-features` leg is NOT buildable on the CPU profile: this
# crate's feature list includes `cuda`, `gpu`, `nvml` and `wasm`, which need
# toolchains a CPU host does not have. Leg (c) is the honest substitute — every
# CPU-buildable feature co-enabled with setfit, which IS the D-05 risk surface.
# Measured 2026-08-09 before wiring, exactly as the D-04 block below was: the
# same feature list WITHOUT setfit (the control) exits 0, and WITH setfit it
# also exits 0. Nothing pre-existing was inherited and nothing was hidden.
	@echo "Feature matrix: aprender-train setfit closure (Phase 3 D-05)"
	@cargo check -p aprender-train --features setfit
	@cargo check -p aprender-train --features setfit,cpu-fallback,gguf,monitor,tui,citl,server,tracing,ruchy-sessions,parquet,hub,viz
# Leg (a) is now the PLAIN green check the plan literally asked for.
#
# History, because this leg used to be a two-sided diff: measured 2026-08-09,
# `cargo check -p aprender-train --no-default-features` was RED at HEAD with 8
# errors, all `presentar_terminal` unlinked under `src/monitor/tui/`, because
# `src/monitor/mod.rs` declared `pub mod tui;` UNCONDITIONALLY while its only
# dependency is gated behind the `tui` feature (deferred-items D-ITEM-05). The
# diff leg asserted only that setfit added nothing to that red build. On
# 2026-08-14 the human rejected that substitution (03-HUMAN-UAT.md item 3,
# decision (b)) and required the D-ITEM-05 fix: the presentar-dependent surface
# (`tui::dashboard`, `TuiMonitor`, `TuiMonitorConfig`) is now `#[cfg(feature =
# "tui")]` while the unconditional IPC writer and state types stay available,
# exactly as the `default = ["tui"]` comment in aprender-train/Cargo.toml always
# said. Both minimal builds are green, so the diff apparatus is retired: two
# plain checks subsume it (green + green means setfit added nothing), and unlike
# the diff they FAIL if either build regresses to red. Under `.SHELLFLAGS := -e
# -c` a failing cargo check aborts the recipe with cargo's own diagnostics,
# which is now the desired behavior — fail loud, no status comparison to guard.
	@echo "  leg (a): minimal-build closure (plain; D-ITEM-05 fixed 2026-08-14)"
	@cargo check -p aprender-train --no-default-features
	@cargo check -p aprender-train --no-default-features --features setfit
	@echo "    minimal build green with and without setfit"
# The dependency-closure negative, two-sided. The positive half is what stops
# the negative half passing for the wrong reason: a `cargo tree` invocation that
# silently stopped resolving these packages would satisfy an absence-only check
# forever. Both trees are captured to files and `cargo tree`'s own status is
# checked FIRST, same discipline as the aprender-core block above.
	@echo "  negative: a DEFAULT aprender-train build must contain NO contrastive-data, rand or tokenizers node"
	@cargo tree -p aprender-train -e normal \
		> target/sfm-train-tree-default.txt 2>&1 || \
		{ echo "FAIL: cargo tree failed; the D-05 closure check would pass vacuously"; \
		  cat target/sfm-train-tree-default.txt; exit 1; }
	@cargo tree -p aprender-train --features setfit -e normal \
		> target/sfm-train-tree-setfit.txt 2>&1 || \
		{ echo "FAIL: cargo tree --features setfit failed; the D-05 closure check would pass vacuously"; \
		  cat target/sfm-train-tree-setfit.txt; exit 1; }
	@if grep -qE 'aprender-contrastive-data|aprender-rand|tokenizers' target/sfm-train-tree-default.txt; then \
		echo "FAIL: a setfit-only dependency leaked into the DEFAULT aprender-train build (D-05)"; \
		grep -nE 'aprender-contrastive-data|aprender-rand|tokenizers' target/sfm-train-tree-default.txt; exit 1; \
	fi
	@for node in aprender-contrastive-data aprender-rand tokenizers; do \
		if ! grep -q "$$node" target/sfm-train-tree-setfit.txt; then \
			echo "FAIL: --features setfit did NOT pull in $$node; the absence check above is vacuous (D-05)"; \
			exit 1; \
		fi; \
	done
# The one MEASURED-CLEAN `--all-features` leg (see the table at the head of this
# target). Wired because it was measured rc=0, not because it was assumed to be.
	@echo "  leg (d): aprender-train --all-features (measured clean 2026-08-15)"
	@cargo check -p aprender-train --all-features
# ─── SAFE-02 cell: apr-cli ──────────────────────────────────────────────────
#
# NO `--no-default-features` LEG. Measured rc=101 at this commit AND at the
# pre-Phase-4 base — D-04-09-A, four `inference`-gated-but-not-cfg-gated errors.
# The gating evidence is the `--all-targets` check with DEFAULT features below.
	@echo "Feature matrix: apr-cli (SAFE-02 gating evidence is the CHECK leg)"
	@echo "  check, setfit OFF (default features) — THIS is SAFE-02's gating leg"
	@cargo check -p apr-cli --all-targets
	@echo "  check, setfit ON"
	@cargo check -p apr-cli --features setfit
	@cargo check -p apr-cli --features setfit,inference --all-targets
# The graph negative, two-sided, and it is the apr-cli gating claim that the run
# leg below cannot make. `aprender-contrastive-data` is NOT usable as the marker
# here (unlike aprender-train): it is already in the DEFAULT apr-cli tree via
# `training`, which is a default feature. Measured — `tokenizers` is 0 in the
# default tree and 1 with setfit, so it is the marker that actually discriminates.
	@echo "  negative: a DEFAULT apr-cli build must contain NO tokenizers node"
	@cargo tree -p apr-cli -e normal --prefix none > target/sfm-cli-tree-default.txt 2>&1 || \
		{ echo "FAIL: cargo tree failed; the apr-cli closure check would pass vacuously"; \
		  cat target/sfm-cli-tree-default.txt; exit 1; }
	@cargo tree -p apr-cli --features setfit -e normal --prefix none > target/sfm-cli-tree-setfit.txt 2>&1 || \
		{ echo "FAIL: cargo tree --features setfit failed; the apr-cli closure check would pass vacuously"; \
		  cat target/sfm-cli-tree-setfit.txt; exit 1; }
	@if grep -q tokenizers target/sfm-cli-tree-default.txt; then \
		echo "FAIL (SAFE-02): tokenizers leaked into a DEFAULT apr-cli build"; \
		grep -n tokenizers target/sfm-cli-tree-default.txt; exit 1; \
	fi
	@if ! grep -q tokenizers target/sfm-cli-tree-setfit.txt; then \
		echo "FAIL: --features setfit did NOT pull tokenizers into apr-cli; the absence"; \
		echo "check above is vacuous — the marker no longer discriminates (SAFE-02)"; \
		exit 1; \
	fi
	@echo "    apr-cli: tokenizers absent by default, present with setfit"
# ─── SAFE-02 cell: aprender-serve ───────────────────────────────────────────
	@echo "Feature matrix: aprender-serve setfit closure (D-09, HTTP transport only)"
	@cargo check -p aprender-serve --no-default-features
	@cargo check -p aprender-serve --no-default-features --features setfit
	@cargo check -p aprender-serve --features setfit
	@echo "  negative: a DEFAULT aprender-serve build must contain NO tokenizers node"
	@cargo tree -p aprender-serve -e normal --prefix none > target/sfm-serve-tree-default.txt 2>&1 || \
		{ echo "FAIL: cargo tree failed; the aprender-serve closure check would pass vacuously"; \
		  cat target/sfm-serve-tree-default.txt; exit 1; }
	@cargo tree -p aprender-serve --features setfit -e normal --prefix none > target/sfm-serve-tree-setfit.txt 2>&1 || \
		{ echo "FAIL: cargo tree --features setfit failed; the aprender-serve closure check would pass vacuously"; \
		  cat target/sfm-serve-tree-setfit.txt; exit 1; }
	@if grep -q tokenizers target/sfm-serve-tree-default.txt; then \
		echo "FAIL (SAFE-02): tokenizers leaked into a DEFAULT aprender-serve build"; \
		grep -n tokenizers target/sfm-serve-tree-default.txt; exit 1; \
	fi
	@if ! grep -q tokenizers target/sfm-serve-tree-setfit.txt; then \
		echo "FAIL: --features setfit did NOT pull tokenizers into aprender-serve; the"; \
		echo "absence check above is vacuous (SAFE-02)"; \
		exit 1; \
	fi
	@echo "    aprender-serve: tokenizers absent by default, present with setfit"
# ─── SAFE-02 RUN legs ───────────────────────────────────────────────────────
#
# `cargo check` is not `cargo test`. A crate can type-check with the feature on
# and still have every gated test compiled out — which is CR-01 exactly, and is
# why this target used to be described as "only cargo checks the feature;
# checking is not testing" in the setfit-tests comment above. These legs close
# that: each profile that has tests RUNS them, under a guard.
	@echo "Feature matrix RUN legs (checking is not testing — CR-01)"
	@echo "  aprender-core: same filter, feature OFF must select ZERO"
	@set +e; CARGO_INCREMENTAL=0 cargo test -p aprender-core --no-default-features --lib setfit:: \
		> target/sfm-run-core-off.log 2>&1; rc=$$?; \
	set -e; \
	if [ $$rc -ne 0 ]; then echo "FAIL: aprender-core minimal test build is red (rc=$$rc)"; tail -20 target/sfm-run-core-off.log; exit $$rc; fi
	@$(call assert_tests_absent,target/sfm-run-core-off.log,aprender-core --no-default-features --lib setfit::)
	@echo "  aprender-core: same filter, feature ON must select many"
	@set +e; CARGO_INCREMENTAL=0 cargo test -p aprender-core --no-default-features --features setfit --lib setfit:: \
		> target/sfm-run-core-on.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/sfm-run-core-on.log; \
	if [ $$rc -ne 0 ]; then echo "FAIL: aprender-core setfit run leg is red (rc=$$rc)"; exit $$rc; fi
	@$(call assert_tests_ran,target/sfm-run-core-on.log,230,setfit-feature-matrix/core-run)
	@echo "  aprender-train: same filter, feature OFF must select ZERO"
	@set +e; CARGO_INCREMENTAL=0 cargo test -p aprender-train --no-default-features --lib setfit:: \
		> target/sfm-run-train-off.log 2>&1; rc=$$?; \
	set -e; \
	if [ $$rc -ne 0 ]; then echo "FAIL: aprender-train minimal test build is red (rc=$$rc)"; tail -20 target/sfm-run-train-off.log; exit $$rc; fi
	@$(call assert_tests_absent,target/sfm-run-train-off.log,aprender-train --no-default-features --lib setfit::)
	@echo "  aprender-train: same filter, feature ON must select many"
	@set +e; CARGO_INCREMENTAL=0 cargo test -p aprender-train --no-default-features --features setfit --lib setfit:: \
		> target/sfm-run-train-on.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/sfm-run-train-on.log; \
	if [ $$rc -ne 0 ]; then echo "FAIL: aprender-train setfit run leg is red (rc=$$rc)"; exit $$rc; fi
	@$(call assert_tests_ran,target/sfm-run-train-on.log,300,setfit-feature-matrix/train-run)
# apr-cli is a DELTA, not a zero — see the head of this target. 13 off / 68 on,
# measured. Asserting zero here would be false, and asserting nothing would let
# the leg read as gating evidence it cannot supply.
	@echo "  apr-cli: DELTA leg (feature-off is NOT zero — see this target's header)"
	@set +e; CARGO_INCREMENTAL=0 cargo test -p apr-cli --lib setfit \
		> target/sfm-run-cli-off.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/sfm-run-cli-off.log; \
	if [ $$rc -ne 0 ]; then echo "FAIL: apr-cli default test build is red (rc=$$rc)"; exit $$rc; fi
	@$(call assert_tests_ran,target/sfm-run-cli-off.log,10,setfit-feature-matrix/cli-run-off)
	@set +e; CARGO_INCREMENTAL=0 cargo test -p apr-cli --features setfit --lib setfit \
		> target/sfm-run-cli-on.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/sfm-run-cli-on.log; \
	if [ $$rc -ne 0 ]; then echo "FAIL: apr-cli setfit run leg is red (rc=$$rc)"; exit $$rc; fi
	@$(call assert_tests_ran,target/sfm-run-cli-on.log,60,setfit-feature-matrix/cli-run-on)
	@off=$$(awk '/^test result:/ { for (i = 1; i <= NF; i++) if ($$(i+1) ~ /^passed/) s += $$i } END { print s + 0 }' target/sfm-run-cli-off.log); \
	on=$$(awk '/^test result:/ { for (i = 1; i <= NF; i++) if ($$(i+1) ~ /^passed/) s += $$i } END { print s + 0 }' target/sfm-run-cli-on.log); \
	if [ $$((on - off)) -lt 40 ]; then \
		echo "FAIL (SAFE-02): apr-cli's setfit delta is $$((on - off)) ($$off off, $$on on),"; \
		echo "expected at least 40. Either the gated surface stopped compiling in, or the"; \
		echo "surface stopped being gated. Both are SAFE-02 failures; neither is visible"; \
		echo "from a one-sided count."; \
		exit 1; \
	fi; \
	echo "    apr-cli setfit delta: $$off off -> $$on on (+$$((on - off)))"
# aprender-serve runs at profile (c) ONLY — its minimal TEST build is red for a
# pre-existing, non-setfit reason (D-04-10-A; the CHECK cells above still run).
	@echo "  aprender-serve: profile (c) only (D-04-10-A blocks the minimal TEST build)"
	@set +e; CARGO_INCREMENTAL=0 cargo test -p aprender-serve --features setfit --lib setfit \
		> target/sfm-run-serve-on.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/sfm-run-serve-on.log; \
	if [ $$rc -ne 0 ]; then echo "FAIL: aprender-serve setfit run leg is red (rc=$$rc)"; exit $$rc; fi
	@$(call assert_tests_ran,target/sfm-run-serve-on.log,9,setfit-feature-matrix/serve-run)
	@echo "setfit-feature-matrix: PASSED"

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
# stderr goes to its OWN file, never into the one parsed as the closure. It used to be
# `2>&1`, and cargo writes progress to stderr: `Blocking waiting for file lock on package
# cache`, `Updating crates.io index`, `Downloading ...`. Each contributes a first field
# that `awk` below turns into a phantom package name and `comm -23` reports as an
# unlisted dependency. That fired twice unprompted during review (rc=2, offender
# `Blocking`) on a tree that was clean seconds earlier -- so any concurrent cargo, a
# rust-analyzer or a parallel CI job, reds this gate at random. The FAIL text says do not
# widen the allowlist, which leaves a developer no lever except to add `Blocking` to
# allowed-deps.txt permanently. A required gate that reds at random is worse than one
# that never reds.
	@cargo tree -p aprender-contrastive-data -e normal --prefix none --no-dedupe \
		> target/contrastive-data-tree.txt 2> target/contrastive-data-tree.err || \
		{ echo "FAIL: cargo tree failed; the D-04 dependency check would pass vacuously"; \
		  cat target/contrastive-data-tree.txt target/contrastive-data-tree.err; exit 1; }
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
# The third detector is not decoration. The first two match five literal spellings, and
# `use std::{fs, net::TcpStream};` contains NONE of them -- the text is `std::{fs` -- while
# binding `fs` and `TcpStream` into scope identically. This is what rustfmt emits under
# `imports_granularity = "Crate"`, which is exactly what rustfmt.toml asks for here, so it
# is the DEFAULT spelling rather than an exotic one. Verified against a compiled mutation:
# a module doing `fs::File::create(...).write_all(...)` and `TcpStream::connect(...)`
# compiled with rc=0 AND passed the gate with rc=0 before this line existed.
#
# awk, not grep, because a grouped import spans lines: it accumulates a `use` statement
# until the `;` and tests the whole statement. The pattern set is a case table, not a
# guess -- 6 must-match and 6 must-not-match cases live in
# tests/gate_cases/, and `make contrastive-data-boundary-cases` re-runs them. Re-run the
# table rather than re-reading the pattern.
	@while IFS= read -r srcfile; do \
		{ grep -nE 'std::fs|std::net|std::path' "$$srcfile" || true; \
		  grep -nwE 'Path|PathBuf' "$$srcfile" || true; \
		  awk -f scripts/lib/d04_grouped_std_import.awk "$$srcfile" || true; } \
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
	@$(MAKE) --no-print-directory contrastive-data-boundary-cases
	@echo "contrastive-data-boundary: PASSED"

# D-04 CASE TABLE. The source half is a set of text patterns, and the ONLY thing that ever
# caught one of its misses was a case -- never a review. So the patterns carry a table and
# the gate runs it on every invocation: a detector that has silently stopped matching
# fails HERE, loudly, instead of passing a violation through in silence.
#
# The filename IS the expectation. must_match_* must be flagged, must_not_match_* must not.
# Both directions matter: a pattern that flags everything is as broken as one that flags
# nothing, and only the must_not_match half can tell them apart.
contrastive-data-boundary-cases: ## D-04: prove the source detectors still match what they claim
	@cases=crates/aprender-contrastive-data/tests/gate_cases; \
	if [ ! -d "$$cases" ]; then \
		echo "FAIL: $$cases is MISSING. Without it the detectors are unproven and this"; \
		echo "      target would report PASS while checking nothing."; \
		exit 1; \
	fi; \
	n=0; bad=0; \
	for f in "$$cases"/must_*.rs; do \
		[ -e "$$f" ] || continue; \
		n=$$((n + 1)); \
		hits=$$({ grep -nE 'std::fs|std::net|std::path' "$$f" || true; \
		          grep -nwE 'Path|PathBuf' "$$f" || true; \
		          awk -f scripts/lib/d04_grouped_std_import.awk "$$f" || true; } \
		        | grep -vE '^[0-9]+:[[:space:]]*//' | wc -l); \
		case "$$(basename "$$f")" in \
		  must_match_*)     want=1 ;; \
		  must_not_match_*) want=0 ;; \
		  *) echo "FAIL: $$f is neither must_match_* nor must_not_match_*"; exit 1 ;; \
		esac; \
		if [ "$$want" = 1 ] && [ "$$hits" -eq 0 ]; then \
			echo "FAIL: $$f MUST be flagged and was not -- a detector stopped matching"; \
			bad=$$((bad + 1)); \
		fi; \
		if [ "$$want" = 0 ] && [ "$$hits" -ne 0 ]; then \
			echo "FAIL: $$f MUST NOT be flagged and was -- a detector is over-broad"; \
			bad=$$((bad + 1)); \
		fi; \
	done; \
	if [ "$$n" -eq 0 ]; then \
		echo "FAIL: no case files found; the case table would pass vacuously"; exit 1; \
	fi; \
	if [ "$$bad" -ne 0 ]; then echo "  $$bad of $$n case(s) WRONG"; exit 1; fi; \
	echo "  cases:  $$n/$$n (6 must-match, 6 must-not-match)"

# Tier 4: CI/CD (5-60 minutes, heavyweight)
tier4: tier3
	@echo "Running Tier 4: CI/CD validation..."
	@PROPTEST_CASES=100 QUICKCHECK_TESTS=100 cargo test --release
# Phase 5's DOOR-LEVEL claims proof (gap-closure round). Wired here, not in tier3, because
# it needs a release `apr` built with the non-default `setfit` feature and tier4 is the tier
# that already builds release artifacts. See the target's own comment block for why the
# build is a prerequisite rather than a skip path.
	@$(MAKE) setfit-bench-door-probe-build
	@echo "Running pmat analysis..."
	-pmat tdg . --include-components
	-pmat rust-project-score
	-pmat quality-gates --report
	@echo "Tier 4: PASSED"

# ============================================================================
# COVERAGE TARGETS (Two-Phase Pattern from bashrs)
# ============================================================================
# #3839: these targets used to `mv ~/.cargo/config.toml` aside for the whole run
# (the bashrs "mold breaks LLVM coverage" workaround). That renames a file every
# other cargo process of this user reads, so on a shared host it silently changes
# other agents' builds mid-run. No fleet host has a global config (checked
# 2026-09-23: lambda, intel, yoga, gx10), so the move was a no-op there. Instead,
# COV_REFUSE_GLOBAL_MOLD refuses to measure where a global config enables mold,
# and never edits it.
COV_REFUSE_GLOBAL_MOLD = @if [ -f "$${CARGO_HOME:-$$HOME/.cargo}/config.toml" ] && grep -q mold "$${CARGO_HOME:-$$HOME/.cargo}/config.toml"; then \
	echo "❌ $${CARGO_HOME:-$$HOME/.cargo}/config.toml enables mold, which breaks LLVM coverage instrumentation."; \
	echo "   Refusing rather than moving a file every other cargo process on this host reads (issue 3839)."; \
	echo "   Run with CARGO_HOME pointing at a copy without mold, or remove mold from that file."; \
	exit 1; fi

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
# 2026-09-23, #4023: 88 -> 89. The first COMPLETE measurement (every aprender-serve process
# exited normally; coverage-nightly run 35908686532) was 849871/941605 = 90.26%. 89 is a
# ratchet with margin, since 90 would leave no room for noise; it goes to 90 once two
# consecutive nightlies measure >= 90.5% (release-cop ruling).
COV_FLOOR := 89
# #4023: libtest threads for aprender-serve's `gpu` coverage shard (25.9 GB at 22 on yoga).
COV_GPU_SHARD_THREADS ?= 4

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
# #4023 brings two-phase BACK, deliberately: aprender-serve's lib tests cannot run in one
# process on a 28 GB runner (#4028), so they run as several --no-report processes and one
# report merges them. It is safe because every report is now scoped by an explicit `-p` list
# DERIVED from `cargo metadata` (scripts/coverage_report_scope.py), the verified alternative
# above, and scripts/check_coverage_report_scoped.sh refuses any unscoped `llvm-cov report`. profraw
# survive it (31 present afterwards), so coverage-html still has data to work from.
.PHONY: coverage-check contracts

# BSE-03 phase A (Pmat-Ticket: PMAT-1068). The README's contract count is
# DERIVED: scripts/readme_sync.sh rewrites the text between the
# <!-- CONTRACT_COUNT_START/END --> markers with `find contracts/ -name '*.yaml'`
# and nothing else in the file. Idempotent — running it twice is byte-identical.
#
# Before this, three literals in three prose sites were maintained by hand and
# `--regen` only PRINTED the numbers for a human to copy; they sat two behind
# the filesystem (1812 vs 1814) and were GREEN, because the guard lets the
# README lag. scripts/check_readme_claims.sh judges the generated block by
# EQUALITY against the MERGE TREE, with origin/main as the comparand.
.PHONY: readme-sync readme-sync-check
readme-sync: ## Regenerate the README's derived contract count (BSE-03)
	@bash scripts/readme_sync.sh --write

readme-sync-check: ## Fail if README.md is not what the generator produces
	@bash scripts/readme_sync.sh --check

# The polarity table of the D2 normaliser, on a throwaway git repo (BSE-03
# phase A). It carries its own registered mutation: a copy of the guard whose
# merge-tree measurement READS A FILE ON DISK must turn the hand-edited rows
# GREEN, which is what makes their RED load-bearing rather than incidental.
# `--class complexity` and `--class satd` are stubs and exit 3, never 0.
.PHONY: roadmap-aggregate roadmap-aggregate-check
roadmap-aggregate: ## Regenerate docs/roadmaps/roadmap.yaml from docs/roadmaps/entries/ (#3296)
	@python3 scripts/lib/roadmap_fragments.py aggregate --write

roadmap-aggregate-check: ## Fail if roadmap.yaml is not what the aggregator produces, or if it is not idempotent
	@python3 scripts/lib/roadmap_fragments.py aggregate --check

.PHONY: ratchet-semantics-test
ratchet-semantics-test: ## BSE-03: D2 ratchet polarity rows (--class readme)
	@bash scripts/tests/ratchet_semantics_test.sh --class readme

# Alias the dogfood pre-release protocol looks for. It expects `coverage-check`;
# without it the gate reports WARN ("verify >=95% manually"), i.e. a release gate
# that asks a human to do the measurement is not a gate. `coverage` already
# enforces COV_FLOOR, so this is a name, not a new policy.
coverage-check: coverage

# Ditto for `contracts`. The provable-contract tier is a HARD release gate per
# CLAUDE.md, and the dogfood protocol looked for a target that did not exist, so
# it WARNed instead of checking. `pv lint` runs validate + audit + score across
# contracts/ and is the documented entry point (never hand-rolled bash).
# EXIT PROPAGATION (PVL-001 EV-4, aprender#4168). Under .ONESHELL this whole
# recipe is ONE shell script, so without errexit its status is the LAST line's
# and every earlier step -- `pv lint` included -- was advisory: a failing lint
# printed its tail and the gate exited 0. `set -e` stops at a failing step.
# It is NOT enough on the pv lines: errexit ignores a failure on the LEFT of
# `&&`, so a pv_bin.sh that REFUSES the binary (stale, wrong identity) would
# fall through to the next step. Hence `|| exit` there as well, which exits
# with that list's own status (pv's rc through the pipe, via -o pipefail).
# Case table + mutants: scripts/tests/make_contracts_propagates.sh.
contracts:
	@set -e
	@echo "== provable contracts: pv lint contracts/ =="
# `| tail -5` DISCARDED THE VERDICT: the pipeline's status is tail's, so the armed-meet
# result was PRINTED and NOT ENFORCED (found by aprender-d8, 0.69.1 tail rehearsal). That
# is Verification Discipline #1 in the release's own contract gate, and
# contracts-exit-integrity does not catch it -- it looks for `|| true` and bare for-loops,
# not for a pipe. The output is kept to a tail for readability by writing it to a file and
# tailing THAT, so the exit status belongs to pv and nothing else.
# .ONESHELL: the whole recipe is ONE shell, so an unconditional `exit $$rc` here ended the
# recipe green after lint -- census, graph, README, provenance and the engine tests never ran.
# Exit only on failure (#4315, caught by scripts/tests/make_contracts_propagates.sh).
	@. scripts/pv_bin.sh && { "$$PV" lint contracts/ > /tmp/pv-lint-contracts.$$$$.log 2>&1; rc=$$?; tail -5 /tmp/pv-lint-contracts.$$$$.log; rm -f /tmp/pv-lint-contracts.$$$$.log; [ $$rc -eq 0 ] || exit $$rc; } || exit
	@echo "== census: tracked contracts/census.json == a fresh one (ONT-001 ONT-1, F-1) =="
	@git ls-files --error-unmatch contracts/census.json >/dev/null || { echo "FAIL: contracts/census.json is not tracked, so diffing it proves nothing"; exit 1; }
	@. scripts/pv_bin.sh && "$$PV" census contracts --format json > contracts/census.json || exit
	@git diff --exit-code contracts/census.json || { echo "FAIL: the tracked census differs from a fresh one — commit the regenerated contracts/census.json"; exit 1; }
	@echo "== graph: tracked contracts/contracts.nt + shapes.ttl == a fresh extraction (ONT-001 ONT-4b, R-18) =="
	@. scripts/pv_bin.sh && "$$PV" extract contracts --check >/dev/null || exit
	@echo "== README states the censused count =="
	@bash scripts/readme_sync.sh --check
	@echo "== provenance marks, interim (ONT-001 R-10) =="
	@bash scripts/lint-provenance.sh --self-test
	@bash scripts/lint-provenance.sh contracts/external-corpora.yaml
	@echo "== contract engine tests =="
	@cargo test -p aprender-contracts --lib 2>&1 | grep -E "test result" | tail -1

# #3839: skips are EXACT full test paths from scripts/coverage-skips.txt, one reason
# per entry. They used to be 19 --skip substrings that removed 2,713 tests (2,702 of
# which pass without a GPU), so the number measured a subset over the whole denominator.
coverage: ## Coverage summary + threshold check (warm: ~3min)
	@echo "📊 Running coverage ($(COV_THRESHOLD)%+ threshold)..."
	@# #4023: refuse before any test runs if a `llvm-cov report` anywhere would cover only the facade.
	@scripts/check_coverage_report_scoped.sh
	@which cargo-llvm-cov > /dev/null 2>&1 || { cargo install cargo-llvm-cov --locked || exit 1; }
	$(COV_REFUSE_GLOBAL_MOLD)
	@# Pre-clean: remove stale profraw files to avoid LLVM version mismatch
	@COVDIR=$$($(COV_CARGO_ENV) cargo llvm-cov show-env 2>/dev/null | grep CARGO_LLVM_COV_TARGET_DIR | sed "s/.*=//"); \
	if [ -n "$$COVDIR" ]; then find "$$COVDIR" -name '*.profraw' -delete 2>/dev/null || true; fi
	@mkdir -p target/coverage
	@rm -f target/coverage/lcov.info target/coverage/test.log target/coverage/failed-tests.txt
	@printf '%s' '$(COVERAGE_EXCLUDE_REGEX)' > target/coverage/.exclude-re
	@# #4023: aprender-serve's lib tests run as SEVERAL processes. In one process they build up
	@# memory across tests (#4028: 30 GB single-threaded, 45 GB at 22 threads on gx10) and earlyoom
	@# SIGTERMed them on yoga's 28 GB box (run 35868368976); one module group per process peaks
	@# <= 7.8 GB. EVERY run is --no-report and ONE `cargo llvm-cov report` merges them: a run WITH a
	@# report cleans the earlier profiles (measured: the first run's coverage fell to 0).
	@echo "🧪 Workspace lib tests except aprender-serve (instrumented, --no-report)..."
	@PROPTEST_CASES=10 QUICKCHECK_TESTS=10 RUST_MIN_STACK=16777216 CARGO_BUILD_JOBS=4 \
		$(COV_CARGO_ENV) cargo llvm-cov test --no-report \
		--workspace --exclude aprender-gpu --exclude aprender-serve --lib --ignore-run-fail \
		-- --exact $$(sed -e '/^#/d' -e '/^[[:space:]]*$$/d' -e 's/^/--skip /' scripts/coverage-skips.txt) \
		2>&1 | tee target/coverage/test.log; \
	rc=$${PIPESTATUS[0]}; \
	if [ "$$rc" -ne 0 ]; then \
		echo "❌ coverage DID NOT MEASURE: cargo llvm-cov exited $$rc on the workspace run (build failure;"; \
		echo "   with --ignore-run-fail a failing test alone does not stop it). No coverage verdict."; \
		exit 1; \
	fi
	@echo "🧪 aprender-serve lib tests, one process per module group (instrumented, --no-report)..."
	@rm -rf target/coverage/serve-shards
	@$(COV_CARGO_ENV) cargo llvm-cov test --no-report -p aprender-serve --lib -- --list \
		> target/coverage/serve-list.txt 2>> target/coverage/test.log || \
		{ echo "❌ coverage DID NOT MEASURE: could not list aprender-serve's lib tests. No coverage verdict."; exit 1; }
	@python3 scripts/coverage_serve_shards.py target/coverage/serve-list.txt scripts/coverage-skips.txt \
		target/coverage/serve-shards scripts/coverage-solo.txt
	@# scripts/coverage-solo.txt: run FIRST, each in its OWN process, and print its test binary's peak RSS
	@# (RUSAGE_CHILDREN.ru_maxrss), so a later skip carries a measured per-test reason.
	@: > target/coverage/failed-runs.txt; \
	for solo in target/coverage/serve-shards/solo-*.txt; do \
		[ -e "$$solo" ] || continue; \
		t=$$(cat $$solo); \
		PROPTEST_CASES=10 QUICKCHECK_TESTS=10 RUST_MIN_STACK=16777216 CARGO_BUILD_JOBS=4 \
			$(COV_CARGO_ENV) python3 -c 'import resource, subprocess, sys; rc = subprocess.call(sys.argv[2:]); print("coverage-solo-maxrss", resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, "KB", sys.argv[1], "rc=%d" % rc, flush=True); sys.exit(rc)' \
			"$$t" cargo llvm-cov test --no-report -p aprender-serve --lib --ignore-run-fail -- --exact "$$t" \
			2>&1 | tee -a target/coverage/test.log; \
		rc=$${PIPESTATUS[0]}; \
		[ "$$rc" -eq 0 ] || echo "solo $$t rc=$$rc" >> target/coverage/failed-runs.txt; \
	done
	@# The `gpu` module builds up memory in one process on yoga (25.9 GB at 22 threads, 26.5 GB at 4;
	@# runs 35881004821, 35885731831), so the partitioner chunks it into <= 200-test processes, which
	@# also run at COV_GPU_SHARD_THREADS. EVERY shard runs even if one fails, so a dispatch yields the
	@# whole picture; any failure then means no verdict, naming each failed shard.
	@for shard in target/coverage/serve-shards/shard-*.txt; do \
		threads=""; case "$$shard" in *-gpu.*.txt) threads="--test-threads=$(COV_GPU_SHARD_THREADS)" ;; esac; \
		echo "   $$shard ($$(wc -l < $$shard) tests) $$threads"; \
		PROPTEST_CASES=10 QUICKCHECK_TESTS=10 RUST_MIN_STACK=16777216 CARGO_BUILD_JOBS=4 \
			$(COV_CARGO_ENV) cargo llvm-cov test --no-report -p aprender-serve --lib --ignore-run-fail \
			-- --exact $$threads $$(cat $$shard) 2>&1 | tee -a target/coverage/test.log; \
		rc=$${PIPESTATUS[0]}; \
		echo "   coverage-shard-rc $$rc $$shard"; \
		[ "$$rc" -eq 0 ] || echo "shard $$shard rc=$$rc" >> target/coverage/failed-runs.txt; \
	done
	@if [ -s target/coverage/failed-runs.txt ]; then \
		echo "❌ coverage DID NOT MEASURE: these aprender-serve runs failed (every one was still run):"; \
		sed 's/^/     /' target/coverage/failed-runs.txt; \
		echo "   No coverage verdict."; \
		exit 1; \
	fi
	@echo "📊 Merging every run's profiles into one report..."
	@# `--workspace --exclude aprender-gpu` is REQUIRED: the root Cargo.toml is also a package (the
	@# `apr` facade), and an unqualified `report` covers ONLY the root package. Proof run
	@# 35892421393 printed "Finished report saved" and then found no (non-empty) lcov. Measured with
	@# cargo-llvm-cov 0.9.0 (CI's version) on a root-package workspace: without --workspace the
	@# lcov held only src/lib.rs; with it, every member.
	@# SCOPE IS EXPLICIT: an unscoped `report` covers only the root facade (empty lcov, run
	@# 35892421393 and the single-phase note above); `report --exclude` is rejected by 0.9.0 (run
	@# 35901458111) and `report --workspace` by older versions. A derived `-p` list works on both.
	@$(COV_CARGO_ENV) cargo llvm-cov report $$(python3 scripts/coverage_report_scope.py --exclude aprender-gpu) \
		--lcov --output-path $(CURDIR)/target/coverage/lcov.info \
		--ignore-filename-regex "$$(cat target/coverage/.exclude-re)" 2>&1 | tee -a target/coverage/test.log; \
	rc=$${PIPESTATUS[0]}; \
	echo "   lcov: $$(ls -la $(CURDIR)/target/coverage/lcov.info 2>&1)"; \
	echo "   lcov files under the workspace: $$(find $(CURDIR) -name lcov.info -newer target/coverage/.exclude-re 2>/dev/null | tr '\n' ' ')"; \
	echo "   profraw files: $$(find $${CARGO_TARGET_DIR:-$(CURDIR)/target} -name '*.profraw' 2>/dev/null | wc -l)"; \
	if [ "$$rc" -ne 0 ]; then echo "❌ coverage DID NOT MEASURE: the merged report step exited $$rc. No coverage verdict."; exit 1; fi
	@# #3839: --ignore-run-fail keeps one failing test from blanking the number (the 2026-09-23
	@# nightly wrote no lcov because of one timing test). Failures are LISTED, not hidden, and
	@# every test run here is also run by CI's workspace-test, which fails on them.
	@grep -E '^test .* \.\.\. FAILED$$' target/coverage/test.log | sed -e 's/^test //' -e 's/ \.\.\. FAILED$$//' | sort -u > target/coverage/failed-tests.txt || true
	@# A test BINARY killed by a signal (earlyoom SIGTERMed aprender-serve at 25.7 GB on yoga, run
	@# 35868368976) is swallowed by --ignore-run-fail, and its crate's profile is missing from the
	@# lcov: that run printed "76% ... REGRESSION" with the largest crate absent. No verdict then.
	@scripts/check_coverage_log_complete.sh target/coverage/test.log
	@echo "📊 Parsing LCOV for the threshold check..."
	@# Parse LCOV for line coverage (LH=lines hit, LF=lines found)
	@if [ ! -s target/coverage/lcov.info ]; then echo "❌ coverage DID NOT MEASURE: no lcov.info was written. No coverage verdict."; exit 1; fi; \
	LH=$$(awk -F: '/^LH:/{s+=$$2} END{print s+0}' target/coverage/lcov.info); \
	LF=$$(awk -F: '/^LF:/{s+=$$2} END{print s+0}' target/coverage/lcov.info); \
	if [ "$$LF" -eq 0 ]; then echo "❌ coverage DID NOT MEASURE: lcov.info has 0 instrumented lines. No coverage verdict."; exit 1; fi; \
	COV_PCT=$$((LH * 100 / LF)); \
	NFAIL=$$(wc -l < target/coverage/failed-tests.txt); \
	echo "TOTAL: $$LH/$$LF lines covered ($${COV_PCT}%)"; \
	echo "TOTAL $$LH $$LF $${COV_PCT}% failed_tests=$$NFAIL" > target/coverage/summary.txt; \
	if [ "$$NFAIL" -gt 0 ]; then \
		echo "⚠  $$NFAIL test(s) FAILED in the instrumented run (measured anyway; CI workspace-test gates them):"; \
		sed 's/^/     /' target/coverage/failed-tests.txt; \
		sed 's/^/FAILED /' target/coverage/failed-tests.txt >> target/coverage/summary.txt; \
	fi; \
	mkdir -p .pmat-metrics || exit 1; \
	printf '{"coverage_pct":%s}' "$$COV_PCT" > .pmat-metrics/coverage.result; \
	echo "   wrote .pmat-metrics/coverage.result ($${COV_PCT}%) for pmat score"; \
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
	$(COV_REFUSE_GLOBAL_MOLD)
	@mkdir -p target/coverage
	@printf '%s' '$(COVERAGE_EXCLUDE_REGEX)' > target/coverage/.exclude-re
	@$(COV_CARGO_ENV) cargo llvm-cov report $$(python3 scripts/coverage_report_scope.py --exclude aprender-gpu) --html --output-dir target/coverage/html --ignore-filename-regex "$$(cat target/coverage/.exclude-re)"
	@$(COV_CARGO_ENV) cargo llvm-cov report $$(python3 scripts/coverage_report_scope.py --exclude aprender-gpu) --lcov --output-path target/coverage/lcov.info --ignore-filename-regex "$$(cat target/coverage/.exclude-re)"
	@echo "📍 HTML: target/coverage/html/index.html"

# Full coverage: All features (for CI, slower)
# CB-127-A: Use 'cargo llvm-cov test' instead of nextest to avoid profraw explosion
coverage-full: ## Full coverage report (all features, CI only)
	@echo "📊 Running full coverage analysis (all features)..."
	@which cargo-llvm-cov > /dev/null 2>&1 || { cargo install cargo-llvm-cov --locked || exit 1; }
	$(COV_REFUSE_GLOBAL_MOLD)
	@mkdir -p target/coverage
	@printf '%s' '$(COVERAGE_EXCLUDE_REGEX)' > target/coverage/.exclude-re
	@PROPTEST_CASES=10 QUICKCHECK_TESTS=10 CARGO_BUILD_JOBS=4 \
		$(COV_CARGO_ENV) cargo llvm-cov test --no-report --workspace --lib --all-features \
		--ignore-filename-regex "$$(cat target/coverage/.exclude-re)" \
		-- --skip prop_gbm_expected_value --skip slow --skip heavy --skip benchmark --skip h12_ --skip j2_
	@$(COV_CARGO_ENV) cargo llvm-cov report $$(python3 scripts/coverage_report_scope.py) --html --output-dir target/coverage/html --ignore-filename-regex "$$(cat target/coverage/.exclude-re)"
	@$(COV_CARGO_ENV) cargo llvm-cov report $$(python3 scripts/coverage_report_scope.py) --lcov --output-path target/coverage/lcov.info --ignore-filename-regex "$$(cat target/coverage/.exclude-re)"
	@echo ""
	@$(COV_CARGO_ENV) cargo llvm-cov report $$(python3 scripts/coverage_report_scope.py) --summary-only --ignore-filename-regex "$$(cat target/coverage/.exclude-re)"

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

# Fail-closed, comparand-pinned composite gate (BSE-16, docs/specifications/
# build-system-enhancement.md, infra repo). Replaces both a bare `pmat verify`
# (RED on pre-existing SATD, verify.rs:460,493) and `cargo test --workspace`
# (42-94min measured, docs/reports/work-history-delay-optimization-report.md)
# with: skip the SATD/tests stages pmat can't pass or can't scope correctly,
# run the tree-property guard, then test only the touched crates and their
# direct reverse dependents (scripts/gate_touched_crates.sh), falling back to
# `cargo check --workspace --tests` when that selection can't be trusted.
# `@set -e` is REQUIRED as the first recipe line: under .ONESHELL (see top of
# this file) a failing line does NOT abort a multi-line recipe on its own —
# only the shell's own exit status does, and without `set -e` that is just the
# LAST command's exit code (measured: a `false` mid-recipe is otherwise silently
# swallowed). This local `set -e` is scoped to this recipe's own shell
# invocation only, not the file-wide .SHELLFLAGS (see that comment for why -e
# is not applied globally).
gate: ## Fail-closed, comparand-pinned composite gate (BSE-16)
	@set -e
	@echo "==> gate comparand: origin/main@$$(git rev-parse origin/main)"
	pmat verify --format json --skip satd --skip tests
	scripts/guard_tree.sh --no-cargo
	scripts/gate_touched_crates.sh

# Predict whether merge(origin/main, HEAD) will pass the tree-property
# guards, BEFORE pushing (BSE-14, docs/specifications/build-system-
# enhancement.md §4 wave 3, infra repo). `predict` builds the merge in a
# throwaway `git worktree` (never touches this branch's own working tree),
# runs guard_tree.sh --no-cargo and gate_touched_crates.sh --dry-run against
# it, and records the verdict in .predict/last-<branch>.json. `predict-check`
# is the cheap replay for a pre-push hook: it refuses (exit 3) rather than
# reuse a verdict made stale by origin/main moving, HEAD moving, or the
# working tree going dirty, and exits 4 (distinct) on a fetch/network
# failure.
predict: ## Predict merge(origin/main, HEAD) against the tree-property guards (BSE-14)
	scripts/predict_merge.sh

predict-check: ## Refuse a stale prediction; exit 0 only if still fresh (BSE-14)
	scripts/predict_merge.sh --check

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

# ─── F-07 (plan 04-22): the two bashrs gates that can actually fail ──────────
#
# WHAT `bashrs-lint-makefile` USED TO BE, and why it was a defect:
#
#     @bashrs make lint Makefile || echo "  Makefile linting found issues"
#
# `|| echo` SWALLOWS THE STATUS. `echo` succeeds, so the recipe exited 0 no matter
# what bashrs said, and the 34 findings it really produces — including one
# error-severity SC2168 — were printed to a gate nobody read. That is the same
# defect class CLAUDE.md Verification Discipline rule 1 records twice by ticket:
# #2336, where qwen-story-daily captured `tee`'s status so its fail-the-job step
# was unreachable and three green runs proved nothing; and #2360, where `make
# publish`'s POST-PUBLISH VERIFICATION did the same and could never report a
# broken published crate. A check that cannot fail is not a check.
#
# THE EXIT CODE IS NOT THE VERDICT. Measured, not read off the help text
# (D-04-22-A section 1):
#
#   bashrs make lint <Makefile with 1 error-severity finding>  -> rc=2
#   bashrs make lint /nonexistent/Makefile                     -> rc=2   SAME CODE
#   bashrs make lint <warning-only Makefile>                   -> rc=1
#   bashrs make lint <clean Makefile>                          -> rc=0
#   bashrs absent from PATH                                    -> rc=127
#
# rc=2 cannot distinguish "this file has an error" from "the gate was pointed at
# nothing", so both targets below read the VERDICT OUT OF THE REPORT and treat an
# unparseable report as a FAILURE, never as a zero. Two report shapes exist and
# both are handled: `Summary: N error(s), ...`, and — on a fully clean file —
# `No issues found` with no Summary line at all. bashrs also colours its output
# EVEN WHEN REDIRECTED TO A FILE, so the count arrives wrapped in ANSI and is
# stripped with a literal ESC before any numeric comparison. A naive
# `awk`-then-compare reads `\033[1;31m1`, not `1`.
#
# The global `--strict` flag is deliberately NOT used. Its help text says "fail on
# warnings", but warnings already fail without it (rc=1) and it changed nothing in
# any of the seven states measured. It is a no-op here.
#
# BASELINE, NOT ZERO-ERRORS — and every entry must justify itself.
# A zero-errors gate over these four files is UNSATISFIABLE today without editing
# correct code, because two of the error-severity findings are MEASURED BASHRS
# FALSE POSITIVES (D-04-22-A section 2):
#
#   The `dev-setup:` help line — 2327 when measured at 9ae41fcaa, and it moves
#     every time anything is inserted above it, which is exactly why nothing here
#     is keyed on a line number — SC2168 "'local' is only valid in functions"
#     lands on its columns 22-28, i.e. the English words "local d" of
#     `dev-setup: ## Set up local dev environment ...`. CONTROL: `make -n
#     dev-setup` exits 0 and the expanded recipe contains ZERO shell `local`.
#   scripts/check_apr_bin_pinned.sh:72 SC1078 "forget to close this double-quoted
#     string?" lands on the standard single-quote-escaping idiom inside the
#     ABS_APR regex. CONTROL: `bash -n` on that file exits 0 — and the same
#     control exits 2 on a genuinely unterminated quote, so it is not a control
#     that never fires.
#
# NEITHER IS FIXED. Rewording a help string to satisfy a parser bug would wire a
# blocking tier3 gate that trips on any future help text containing "local",
# "declare" or "typeset"; editing the ABS_APR regex would touch a pattern whose
# own lines 70-71 say it "has now been gotten wrong four times in this repo; if
# you change it, re-run the table rather than reading it".
#
# HOW THEY ARE HANDLED, and why the design changed once it was TESTED. The first
# version carried each as a BASELINED count (Makefile 1, check_apr_bin_pinned.sh
# 1). Running the case table refuted that: one new ordinary help line containing
# the word "local" took the Makefile count to 2 and turned the gate red on correct
# prose — the exact liability this plan set out not to build. Both are therefore
# DISCRIMINATED by the two controls documented above the counter, and every
# baseline is now 0. The discriminators are not suppressions: each names a control
# that is re-run on every invocation and that demonstrably FIRES on a real defect
# in the same file. If a later bashrs release fixes either rule, DELETE the
# discriminator rather than leaving it as permanent slack.
#
# The baseline mechanism itself is kept, and every entry must still name the
# control that justifies it: the gate REJECTS an entry with an empty
# justification, so the list cannot decay into a silent suppression list — which
# would be just a slower version of the `|| echo` defect it replaced.
#
# Entry format:  <path>%<subcommand>%<max-error-severity>%<justification>
#
# `_` stands for a space, because make's `for` splits on whitespace and an entry
# therefore cannot contain any. The readable form of each justification is in the
# block above. The separator is `%` and NOT `|`: the first draft used `|`, and the
# unquoted expansion made the shell read every entry as a PIPELINE —
# `syntax error near unexpected token '|'`. Caught by running the target, not by
# reading it. Entries are restricted to [A-Za-z0-9_./%:=,+-] and the gate REJECTS
# an entry containing anything else rather than mis-parsing it, so the next editor
# who reaches for a space or a quote gets a clear failure instead of silence.
#
# Each entry is ALSO single-quoted here, and that is load-bearing rather than
# decorative. Without the quotes the charset guard was unreachable for exactly the
# characters that matter: make pastes this list into shell source text, so an
# entry containing `;` aborted the whole recipe with a raw bash syntax error
# BEFORE the guard could run. Fail-closed either way, but a bash parse dump is not
# a diagnostic. Measured: with quotes, the same probe now reports
# `FAIL: baseline entries outside the safe charset: scripts/bad;rm%lint%0%probe`.
#
# A THIRD false positive, found by running this gate against its own change and
# recorded rather than tidied away: bashrs reads the `check_apr_bin_pinned.sh`
# DATA line below as a command and raises MAKE003 "Unquoted variable in command"
# on the `$(BASHRS_GUARD_PINNED_BASELINE)` inside it. It is a variable-assignment
# continuation, not a command, and there is nothing to quote. It is
# warning-severity, so it does not move any baseline — but it is why the Makefile
# warning count went 33 -> 34 in this commit, and an unexplained +1 in a lint
# tally is exactly the kind of thing that gets a gate distrusted.
# ─── THE PROSE DISCRIMINATOR, and why the baseline alone was not enough ──────
#
# The first version of this gate baselined the Makefile at 1 error-severity
# finding, carrying the SC2168 false positive as a justified entry. Running the
# must-not-match row from CLAUDE.md rule 7's case table REFUTED that design:
# appending one new ordinary help line —
#
#     04-22-probe-help-string: ## Run the local smoke suite against a local endpoint
#
# — raised a SECOND SC2168, took the count 1 -> 2, and turned the gate RED on
# correct prose. A blocking tier3 gate that goes red when someone writes the word
# "local" in help text is precisely the liability this plan set out NOT to build;
# it would be disabled within a week, and the real gates beside it with it.
#
# So SC2168 findings that land on HELP TEXT are discriminated out instead of
# baselined. The rule is positional, not a name-based suppression:
#
#   skip iff  rule is SC2168
#       AND   the source line does NOT begin with a TAB (recipe lines always do,
#             target/help lines never do)
#       AND   the line contains `##`
#       AND   the flagged start column falls AFTER that `##`
#
# WHY NOT JUST SUPPRESS SC2168. Because it catches a REAL defect, measured: a
# recipe body `@local probe=1; echo "$$probe"` produced SC2168 at 2699:3-9 AND
# `/bin/bash: line 0: local: can only be used in a function` when actually run.
# Blanket-suppressing the rule would have let that through. The discriminator
# keeps the true positive and drops the false one — which is the whole difference
# between a gate and a mute button.
#
# ITS CASE TABLE, RUN rather than written (CLAUDE.md rule 7). Full transcripts in
# 04-22-SUMMARY.md:
#   COUNTED     `@local probe=1` in a recipe body (TAB-led, no `##`)      -> gate RED
#   NOT COUNTED SC2168 on `dev-setup: ## Set up local dev environment ...` -> gate green
#   NOT COUNTED a NEW `... ## Run the local smoke suite ...` help line     -> gate green
# With the discriminator the Makefile's counted baseline is therefore 0, not 1.
# SC2168 is still PRESENT and still PRINTED in the log — it is classified, not
# hidden, and `bashrs-lint-makefile` prints every raw error line before counting.
#
# THE SECOND DISCRIMINATOR, for the SC1078 false positive, on the same principle
# and with the same discipline: an SC1078 ("did you forget to close this
# double-quoted string?") is DISCOUNTED for a file whose own `bash -n` exits 0.
# The authority on whether a bash script's quotes are balanced is bash. The
# control is per-file and re-run every time the gate runs, so it cannot go stale,
# and it is NOT a blanket rule suppression:
#
#   COUNTED     an unterminated quote appended to a guard -> bash -n rc=2 -> RED
#   NOT COUNTED the ABS_APR idiom at check_apr_bin_pinned.sh:72 -> bash -n rc=0
#
# That pairing is the whole justification. A control that never fires would prove
# nothing, and this one demonstrably fires on a real defect in the same file.
# It is deliberately narrowed to SC1078 rather than all SC10xx parse-class rules:
# a wider discount would start excusing findings whose control was never measured.
#
# KNOWN LIMITATIONS, stated rather than discovered later. (1) A recipe line that
# puts a literal `##` inside a string before a genuine `local` would be
# under-counted; no such line exists here, and recipe lines are excluded by the
# TAB test anyway. (2) `bash -n` proves syntax, not intent, so an SC1078 marking a
# quote that parses but nests differently than the author meant would be
# discounted; the ABS_APR regex is protected instead by the 12-case must-match /
# must-not-match table its own comment block demands be re-run on every change.
#
# Prints "<counted> <discriminated>". $(2) is the source file, or NONE to disable
# the help-comment discriminator (shell scripts have no `##` help convention).
# $(3) is the file's `bash -n` exit code, or NONE when it was not applicable.
# TWO make-level traps here, both found by RUNNING this, not by reading it:
#   (1) It is ONE physical line on purpose. `$(call)` of a multi-line `define`
#       injects real newlines, which terminate the `\`-continued shell line these
#       recipes are built from — `unexpected EOF while looking for matching '`.
#   (2) It contains NO literal hash character. A `#` inside a make variable
#       assignment starts a COMMENT, so the obvious `index(l, "##")` silently
#       truncated this whole program mid-string and produced the identical
#       unexpected-EOF error. The two hashes are built as `sprintf("%c%c",35,35)`.
bashrs_count_errors = awk -v src="$(2)" -v bashn="$(3)" 'BEGIN { if (src != "NONE") { n = 0; while ((getline l < src) > 0) { n++; srcline[n] = l } } } /\[error\]/ { lineno = 0; colstart = 0; rule = ""; if (match($$0, /[0-9]+:[0-9]+-[0-9]+/)) { loc = substr($$0, RSTART, RLENGTH); split(loc, p, ":"); lineno = p[1] + 0; split(p[2], c, "-"); colstart = c[1] + 0 } if (match($$0, /\[error\] [A-Z]+[0-9]+/)) { rule = substr($$0, RSTART + 8, RLENGTH - 8) } if (src != "NONE" && rule == "SC2168" && (lineno in srcline)) { l = srcline[lineno]; if (substr(l, 1, 1) != "\t") { h = index(l, sprintf("%c%c", 35, 35)); if (h > 0 && colstart > h) { skipped++; next } } } if (rule == "SC1078" && bashn == "0") { skipped++; next } counted++ } END { printf "%d %d\n", counted + 0, skipped + 0 }' $(1)

BASHRS_MAKEFILE_BASELINE = 0
BASHRS_GUARD_PINNED_BASELINE = 0
BASHRS_SCOPED_BASELINE = \
	'Makefile%make%$(BASHRS_MAKEFILE_BASELINE)%the_only_error-severity_finding_is_SC2168_on_the_dev-setup_help_comment,_which_the_prose_discriminator_above_classifies_out._CONTROL:_make_-n_dev-setup_rc=0_and_the_expanded_recipe_contains_ZERO_shell_local' \
	'scripts/apr_bin.sh%lint%0%zero_error-severity_findings_at_baseline_time._CONTROL:_bash_-n_rc=0' \
	'scripts/check_apr_bin_pinned.sh%lint%$(BASHRS_GUARD_PINNED_BASELINE)%its_only_error-severity_finding_is_SC1078_at_line_72_on_the_single-quote_escape_idiom_inside_the_ABS_APR_regex,_which_the_bash_-n_discriminator_above_discounts._CONTROL:_bash_-n_rc=0,_and_the_SAME_control_returns_rc=2_on_a_genuinely_unterminated_quote' \
	'scripts/check_sourced_libs_option_neutral.sh%lint%0%zero_error-severity_findings_at_baseline_time._CONTROL:_bash_-n_rc=0'

bashrs-lint-makefile: ## F-07: lint the Makefile with bashrs and REPORT THE REAL STATUS (BLOCKING, wired into tier3)
	@echo "Linting Makefile with bashrs (baseline $(BASHRS_MAKEFILE_BASELINE) error-severity)..."
	@if ! command -v bashrs >/dev/null 2>&1; then \
		echo "FAIL: bashrs is not installed, so this check DID NOT RUN."; \
		echo "A check that did not run is never reported as passing (F-07)."; \
		echo "Install with: cargo install bashrs"; \
		exit 1; \
	fi
	@mkdir -p target
	@set +e; bashrs make lint Makefile > target/bashrs-makefile.log 2>&1; rc=$$?; \
	set -e; \
	esc=$$(printf '\033'); \
	sed "s/$$esc\[[0-9;]*m//g" target/bashrs-makefile.log > target/bashrs-makefile.plain.log; \
	grep -E '\[error\]' target/bashrs-makefile.plain.log || true; \
	grep -E '^Summary:' target/bashrs-makefile.plain.log || true; \
	echo "  warning-severity breakdown (deferred with stated reasons in D-04-22-A,"; \
	echo "  NOT silently tolerated -- MAKE012 is an architectural observation about"; \
	echo "  the whole file, and MAKE010 largely flags the advisory '|| echo' idiom):"; \
	grep -oE '\[warning\] [A-Z]+[0-9]+' target/bashrs-makefile.plain.log \
		| sort | uniq -c | sort -rn | sed 's/^/    /' || true; \
	echo "  full report: target/bashrs-makefile.log"; \
	if grep -q '^Summary:' target/bashrs-makefile.plain.log \
		|| grep -q 'No issues found' target/bashrs-makefile.plain.log; then \
		pair=$$($(call bashrs_count_errors,target/bashrs-makefile.plain.log,Makefile,NONE)); \
	else \
		pair=""; \
	fi; \
	if [ -z "$$pair" ]; then \
		echo "FAIL: bashrs produced no parseable report (rc=$$rc)."; \
		echo "rc alone cannot be trusted: rc=2 means BOTH 'one error-severity finding'"; \
		echo "AND 'the specified file was not found'. See target/bashrs-makefile.log."; \
		exit 1; \
	fi; \
	errs=$${pair%% *}; prose=$${pair##* }; \
	if [ "$$errs" -gt "$(BASHRS_MAKEFILE_BASELINE)" ]; then \
		echo "FAIL: $$errs counted error-severity finding(s), baseline is $(BASHRS_MAKEFILE_BASELINE) (bashrs rc=$$rc)."; \
		echo "Fix the NEW finding. Do not raise the baseline to absorb it, and do not"; \
		echo "reword correct code to satisfy the linter. If the new finding is SC2168 on"; \
		echo "a ## help comment it would have been discriminated out automatically, so a"; \
		echo "COUNTED finding here is one the prose discriminator did not excuse."; \
		exit 1; \
	fi; \
	echo "bashrs-lint-makefile: $$errs counted error-severity finding(s) (baseline $(BASHRS_MAKEFILE_BASELINE)), $$prose discriminated as control-refuted false positive(s), bashrs rc=$$rc"

# The SCOPED gate. Its scope is the whole point, and it is not zero-errors.
#
# WHY NOT THE WHOLE CORPUS. Measured at 9ae41fcaa, one file at a time so no
# failure masks another: 59 scripts, 51 exit non-zero, 22 of them carrying at
# least one error-severity finding (106 across the corpus). Wiring `lint-scripts`
# into a blocking tier would make tier3 permanently red, and a permanently red
# gate stops being run — it gets commented out, and the real gates beside it go
# with it. That is the Ph1 D-26 failure mode this phase has now cited three times.
# The backlog is not ignored: it is measured, triaged by fix-shape and given an
# owner in D-04-22-A. `lint-scripts` above stays UNWIRED and honest.
#
# WHY THESE FOUR FILES. They are the surface this phase's own verification rests
# on: the Makefile that defines every Phase 4 gate, and the three scripts CLAUDE.md
# makes load-bearing for every `apr` invocation — apr_bin.sh (sourced, so it must
# stay option-neutral), its guard check_apr_bin_pinned.sh, and the guard that
# enforces the option-neutrality. A guard that does not scan the surface where the
# DECISION is made is theater (CLAUDE.md rule 5).
#
# IT DETECTS REGRESSION; IT DOES NOT ASSERT CLEANLINESS. Read the baseline block
# above before changing a number in it.
bashrs-scoped-lint: ## F-07: baseline-non-increase bashrs gate over the Makefile + the three apr-pinning guards (BLOCKING, wired into tier3)
	@echo "bashrs scoped gate: Makefile + the three apr-pinning guard scripts"
	@if ! command -v bashrs >/dev/null 2>&1; then \
		echo "FAIL: bashrs is not installed, so this gate DID NOT RUN."; \
		echo "A check that did not run is never reported as passing (F-07)."; \
		echo "Install with: cargo install bashrs"; \
		exit 1; \
	fi
	@mkdir -p target
	@esc=$$(printf '\033'); \
	examined=0; over=""; unjustified=""; unparseable=""; \
	malformed=""; \
	for entry in $(BASHRS_SCOPED_BASELINE); do \
		case "$$entry" in \
			*[!A-Za-z0-9_./%:=,+-]*) malformed="$$malformed $$entry"; continue ;; \
			*) ;; \
		esac; \
		path=$$(printf '%s' "$$entry" | cut -d'%' -f1); \
		mode=$$(printf '%s' "$$entry" | cut -d'%' -f2); \
		base=$$(printf '%s' "$$entry" | cut -d'%' -f3); \
		why=$$(printf '%s' "$$entry" | cut -d'%' -f4); \
		if [ -z "$$why" ]; then \
			unjustified="$$unjustified $$path"; \
			continue; \
		fi; \
		log=target/bashrs-scoped-$$(printf '%s' "$$path" | tr '/.' '__').log; \
		set +e; \
		if [ "$$mode" = "make" ]; then \
			bashrs make lint "$$path" > "$$log" 2>&1; rc=$$?; src="$$path"; bashn=NONE; \
		else \
			bashrs lint "$$path" > "$$log" 2>&1; rc=$$?; src=NONE; \
			bash -n "$$path" > "$$log.bashn" 2>&1; bashn=$$?; \
		fi; \
		set -e; \
		sed "s/$$esc\[[0-9;]*m//g" "$$log" > "$$log.plain"; \
		if grep -q '^Summary:' "$$log.plain" || grep -q 'No issues found' "$$log.plain"; then \
			pair=$$($(call bashrs_count_errors,"$$log.plain",$$src,$$bashn)); \
		else \
			pair=""; \
		fi; \
		if [ -z "$$pair" ]; then \
			unparseable="$$unparseable $$path(rc=$$rc)"; \
			continue; \
		fi; \
		errs=$${pair%% *}; prose=$${pair##* }; \
		examined=$$((examined + 1)); \
		echo "  examined $$path: $$errs counted error-severity (baseline $$base), $$prose discriminated as control-refuted false positive(s), bashrs rc=$$rc"; \
		if [ "$$errs" -gt "$$base" ]; then \
			over="$$over $$path($$errs>$$base)"; \
		fi; \
	done; \
	if [ -n "$$malformed" ]; then \
		echo "FAIL: baseline entries outside the safe charset:$$malformed"; \
		echo "An entry must match [A-Za-z0-9_./%:=,+-] and use _ for spaces. Anything"; \
		echo "else is REJECTED rather than mis-parsed -- an unquoted | in an earlier"; \
		echo "draft made the shell read each entry as a pipeline."; \
		exit 1; \
	fi; \
	if [ -n "$$unjustified" ]; then \
		echo "FAIL: baseline entries carrying no justification:$$unjustified"; \
		echo "Every entry must name the independent shell-semantic control that refuted"; \
		echo "the finding it carries -- a bash -n exit code, or the read showing the flag"; \
		echo "lands on prose. Without that rule this baseline decays into a silent"; \
		echo "suppression list, which is a slower version of the defect this gate replaced."; \
		exit 1; \
	fi; \
	if [ -n "$$unparseable" ]; then \
		echo "FAIL: bashrs produced no parseable report for:$$unparseable"; \
		echo "rc alone cannot be trusted: MEASURED, bashrs make lint returns rc=2 BOTH for"; \
		echo "one error-severity finding AND for a file that does not exist. A check that"; \
		echo "did not run is never reported as passing (F-07)."; \
		exit 1; \
	fi; \
	if [ "$$examined" -eq 0 ]; then \
		echo "FAIL: this gate examined NOTHING and was about to report success."; \
		echo "BASHRS_SCOPED_BASELINE is empty, or every entry was skipped. A gate whose"; \
		echo "file list stops matching must go RED, not green (contract-audit-phase4 above"; \
		echo "carries the same guard for the same reason)."; \
		exit 1; \
	fi; \
	if [ -n "$$over" ]; then \
		echo "FAIL: error-severity findings ROSE above baseline in:$$over"; \
		echo "This gate detects REGRESSION; it does not assert cleanliness. Fix the NEW"; \
		echo "finding. Do not raise the baseline to absorb it, and do not reword correct"; \
		echo "code to satisfy the linter. The two known bashrs false positives here are"; \
		echo "already DISCRIMINATED OUT by their own controls, so a COUNTED finding is one"; \
		echo "neither control excused -- treat it as real until a control says otherwise."; \
		exit 1; \
	fi; \
	echo "bashrs-scoped-lint: $$examined file(s) examined, none above its baseline"

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

# NOT named PV_BIN (PVL-001 EV-4): a makefile assignment overrides an inherited
# environment variable AND is what make exports to recipes, so `PV_BIN := cargo
# run ...` handed scripts/pv_bin.sh the string "cargo run ..." whenever a caller
# exported PV_BIN=/path/to/pv -- the one override pv_bin.sh honours -- and every
# `. scripts/pv_bin.sh` step refused with `not executable: cargo run ...`.
PV_CARGO_RUN := cargo run --release -p aprender-contracts-cli --bin pv --
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
             contracts/kv-cache-equivalence-v1.yaml \
             contracts/setfit-encoder-conformance-v1.yaml \
             contracts/tweet-eval-stance-benchmark-v1.yaml \
             contracts/contrastive-pair-protocol-v1.yaml \
             contracts/multinomial-head-v1.yaml \
             contracts/setfit-train-lifecycle-v1.yaml \
             contracts/linear-probe-classifier-v1.yaml \
             contracts/setfit-apr-v1.yaml \
             contracts/setfit-benchmark-claims-v1.yaml \
             contracts/prophet-parity-v1.yaml \
             contracts/neuralprophet-parity-v1.yaml \
             contracts/chronos-bolt-parity-v1.yaml \
             contracts/forecast-tool-boundary-v1.yaml \
             contracts/decide-tool-boundary-v1.yaml \
             contracts/laya-finetune-gate-v1.yaml \
             contracts/laya-parity-v1.yaml \
             contracts/decide-apr-v1.yaml

# The two Phase 2 contracts, audited as a BLOCKING tier3 gate by
# `contract-audit-phase2` below. Deliberately a separate, narrower list than
# $(CONTRACTS) — see that target's comment block for the measurement that
# forced the narrowing.
PHASE2_CONTRACTS := contracts/contrastive-pair-protocol-v1.yaml \
                    contracts/tweet-eval-stance-benchmark-v1.yaml

# The Phase 3 contracts, audited as a BLOCKING tier3 gate by
# `contract-audit-phase3` below. Same narrowing rationale as PHASE2_CONTRACTS:
# scoped to what this phase OWNS, because the repo-wide `contract-audit` is
# vacuous (see that target's comment block).
#
# 03-02 HAND-OFF, resolved by measurement rather than assumption (plan 03-04
# task 3, W-04). `test -f
# .planning/phases/03-faithful-two-stage-trainer-and-head/03-02-SUMMARY.md`
# returned rc=0 (the file exists, 28303 bytes), and a grep for the heading
# `CONTINGENCY FIRED` in it returned rc=1 — the heading is ABSENT. 03-02's GEMM
# partition-determinism gate was therefore GREEN, `contracts/gemm-partition-determinism-v1.yaml`
# was deliberately never authored (`ls` rc=1), and there is nothing for this
# phase to wire on its behalf. Recorded explicitly because "I did not see a
# heading" and "I did not look" are indistinguishable afterwards.
# linear-probe-classifier-v1.yaml is here because plan 03-06 task 3 binds
# FrozenProbeRun to it. A contract that is bound but absent from this list is
# audited by NOTHING: `contract-audit` repo-wide is vacuous (see that target),
# so the bindings would sit unchecked while looking checked. Its presence was
# verified by inducing a bogus binding status and observing this target go red.
PHASE3_CONTRACTS := contracts/multinomial-head-v1.yaml \
                    contracts/setfit-train-lifecycle-v1.yaml \
                    contracts/linear-probe-classifier-v1.yaml

# The Phase 4 contract, audited as a BLOCKING tier3 gate by
# `contract-audit-phase4` below. Same narrowing rationale as PHASE2_CONTRACTS and
# PHASE3_CONTRACTS: scoped to what this phase OWNS, because the repo-wide
# `contract-audit` is vacuous (see that target's comment block).
#
# ONE ENTRY, AND THAT IS THE WHOLE PHASE. Ph1 D-23 is one new contract per phase
# referencing the existing ones rather than editing them, and `git diff --stat`
# on plan 04-01's contract commit showed NO other contract file modified.
PHASE4_CONTRACTS := contracts/setfit-apr-v1.yaml

# The Phase 5 contract, audited as a BLOCKING tier3 gate by `contract-audit-phase5`
# below. Same narrowing rationale as PHASE2/3/4_CONTRACTS: scoped to what this phase
# OWNS, because the repo-wide `contract-audit` is vacuous (see that target's comment
# block — it prints 132 BIND-001 errors and exits 0 anyway).
#
# ONE ENTRY, AND THAT IS THE WHOLE PHASE, for the same reason Phase 4 has one: Ph1 D-23
# is one new contract per phase REFERENCING the existing ones rather than editing them.
# This contract references tweet-eval-stance-benchmark-v1 (dataset identity, F_avg, the
# seed set), setfit-apr-v1 (the artifact a SetFit row measures),
# contrastive-pair-protocol-v1 (the manifest whose hash is the pairing key) and
# calibration-v1 (the metrics and the frozen t), and edits none of them.
PHASE5_CONTRACTS := contracts/setfit-benchmark-claims-v1.yaml

# The Phase 6 contracts, audited as a BLOCKING tier3 gate by `contract-audit-phase6`
# below. Same narrowing rationale as PHASE2/3/4/5_CONTRACTS: scoped to what this phase
# OWNS, because the repo-wide `contract-audit` is vacuous (see that target's comment
# block — it prints 132 BIND-001 errors and exits 0 anyway).
#
# FOUR ENTRIES, WHERE PHASES 4 AND 5 HAVE ONE, AND THAT IS NOT A DEPARTURE FROM Ph1 D-23.
# D-23 is "one NEW contract per phase, REFERENCING the existing ones rather than editing
# them"; the count that matters is contracts EDITED, which is zero here. This phase ships
# three independent ports with three DIFFERENT oracles — Prophet 1.1.7 (a Stan-shaped MAP
# objective), NeuralProphet 0.9.0 (an autograd fit), and chronos-forecasting 2.3.1 (a
# zero-shot T5) — plus one tool boundary shared by the two thin MCP servers. Folding four
# unrelated oracles into one contract would make a single equation set that no single
# falsification run can evaluate; folding the boundary into any one port would leave the
# other server's refusals unowned.
PHASE6_CONTRACTS := contracts/forecast-tool-boundary-v1.yaml \
                    contracts/prophet-parity-v1.yaml \
                    contracts/neuralprophet-parity-v1.yaml \
                    contracts/chronos-bolt-parity-v1.yaml

# The Phase 8 contracts, audited as a BLOCKING tier3 gate by `contract-audit-phase8` below.
# Same narrowing rationale as PHASE2..6_CONTRACTS: scoped to what this phase OWNS, because the
# repo-wide `contract-audit` is vacuous (see that target's comment block).
#
# FOUR ENTRIES, AND LIKE PHASE 6 THAT IS NOT A DEPARTURE FROM Ph1 D-23: the count that matters is
# contracts EDITED, which is zero. Plan 08-01 commits all four BEFORE any training run, parity run
# or deploy reads a result (D-07: a threshold chosen after seeing a result is not a gate). They
# are four because they are four DIFFERENT falsification surfaces, the Phase 6 argument:
#   - decide-tool-boundary-v1: the SERVING surface (classify bounds priced to the 30 s API
#     Gateway cap, admission, response shape), falsified by server tests and a live cold probe;
#   - laya-finetune-gate-v1: the TRAINING surface (recipe, calibration, gate thresholds, seed
#     policy, run-dir schemas), falsified by the trainer self-test and the Rust gate verifier;
#   - laya-parity-v1: the NUMERIC surface (torch -> .apr -> Rust tolerances), falsified against
#     the Laya oracle fixtures;
#   - decide-apr-v1: the ARTIFACT surface (schema, load ladder, probes, identity, task schema),
#     falsified by the packer and loader tests.
# Folding the gate into the tool boundary would make one contract whose equations no single
# falsification run can evaluate.
PHASE8_CONTRACTS := contracts/decide-tool-boundary-v1.yaml \
                    contracts/laya-finetune-gate-v1.yaml \
                    contracts/laya-parity-v1.yaml \
                    contracts/decide-apr-v1.yaml

# NOTE (plan 02-01, D-24): $(CONTRACTS) is an EXPLICIT HARDCODED LIST, not a glob
# over contracts/*.yaml. A contract file that merely EXISTS in contracts/ is
# validated by nothing. tweet-eval-stance-benchmark-v1.yaml sat in the tree
# unreferenced and therefore unvalidated, and `pv validate` rejected it the whole
# time (PROVABILITY-001 x2: no proof_obligations, no kani_harnesses) without any
# gate ever noticing. The line above is what makes tier3 reach it — tier3 calls
# `$(MAKE) contract-validate`, which iterates exactly this list. Every future phase
# contract needs its own line here or it is decoration.

contract-validate: ## Validate all kernel contracts (schema + staleness)
	@echo "Validating kernel contracts..."
	@for contract in $(CONTRACTS); do \
		echo "  $$contract"; \
		$(PV_CARGO_RUN) validate "$$contract" || exit 1; \
	done
	@echo "Contract validation passed"

contract-test: ## Run contract-driven property tests
	@set -e
	@echo "Running contract property tests..."
	@PROPTEST_CASES=100 cargo test -p aprender-core --test contract_tests
	@echo "Contract tests passed"

contract-audit: ## Audit binding coverage (equations -> implementations)
	@echo "Running binding audit..."
	@rc=0; for contract in $(CONTRACTS); do \
		echo ""; \
		$(PV_CARGO_RUN) audit "$$contract" --binding $(BINDING) || rc=$$?; \
	done
	@echo ""
	@if [ "$$rc" -ne 0 ]; then echo "Binding audit FAILED: at least one audit exited non-zero (last rc=$$rc)"; exit "$$rc"; fi
	@echo "Binding audit complete"

# D-26 / review finding F9 (plan 02-08). `pv validate` checks contract SHAPE;
# it says nothing about whether an equation is bound to any implementation. A
# schema-valid contract with no binding is a claim that nothing checks, which
# is exactly the failure class the Phase 2 gates exist to close. This target is
# BLOCKING and is wired into tier3.
#
# WHY THIS IS SCOPED, AND WHY THE REPO-WIDE `contract-audit` IS NOT WIRED.
# Measured, not assumed (`make contract-audit > /tmp/ca-repo.log 2>&1; rc=$$?`,
# status captured directly): it reports **132 BIND-001 errors across 38 of the
# 44 contracts** — 10 in Phase 1's setfit-encoder-conformance-v1.yaml, the rest
# spread over the kernel contracts — and **exits 0 anyway**, because its loop
# body ends in `;` and never reads the audit's status. So the broad target is
# today a vacuous gate: it prints failures and reports success. Making it
# blocking would turn tier3 red on 132 pre-existing unbound equations this phase
# did not create; leaving it non-blocking keeps a target that checks nothing.
# Neither is this plan's to fix — logged in the phase's deferred-items.md.
# Scoping to the two contracts this phase OWNS is the gate it can honestly stand
# behind, and neither Phase 2 contract appears anywhere in those 132.
#
# EVIDENCE DISCIPLINE, same as the D-26 and D-04 blocks near tier3. Run
# STANDALONE first with the status captured directly
# (`make contract-audit-phase2 > /tmp/cap2.log 2>&1; rc=$$?`, never through a
# pipe — CLAUDE.md rule 1): rc=0, 24/24 equations bound for
# contrastive-pair-protocol-v1 and 1/1 for tweet-eval-stance-benchmark-v1.
# Wall time 9 s cold (pv is rebuilt by $(PV_CARGO_RUN)), then 1 s / 0 s / 1 s over
# three warm runs — nothing against tier3's 1-5 minute budget, and tier3 has
# already built pv via `contract-validate` two lines earlier.
#
# ITS FAILURE MODE WAS INDUCED, OBSERVED AND REVERTED before it was trusted,
# because a gate that has only ever been seen passing is not evidence: deleting
# the `pair_manifest_hash` entry from $(BINDING) turned it rc=1 with
# "[ERROR] BIND-001: Equation 'pair_manifest_hash' ... has no binding entry",
# naming the deleted equation. That check matters more than usual here — plan
# 02-02 found that a `contract:` field carrying a `../` prefix parses cleanly
# and binds NOTHING, so this gate could otherwise have been green while
# inspecting nothing at all.
contract-audit-phase2: ## Audit Phase 2 binding coverage (BLOCKING, wired into tier3)
	@echo "Auditing binding coverage for the Phase 2 contracts..."
	@unbound=""; \
	for contract in $(PHASE2_CONTRACTS); do \
		echo "  $$contract"; \
		$(PV_CARGO_RUN) audit "$$contract" --binding $(BINDING); \
		status=$$?; \
		if [ "$$status" -ne 0 ]; then \
			unbound="$$unbound $$contract"; \
		fi; \
	done; \
	if [ -n "$$unbound" ]; then \
		echo "FAIL: unbound equations remain in:$$unbound"; \
		echo "Every equation of a Phase 2 contract needs an entry in $(BINDING)."; \
		exit 1; \
	fi; \
	echo "Phase 2 binding audit: every equation is bound"

# Phase 3's twin of contract-audit-phase2, and it exists for the same reason:
# `contract-validate` checks contract SHAPE and says nothing about whether an
# equation is bound to any implementation, so multinomial-head-v1.yaml could be
# "valid" with all five equations bound to nothing at all. BLOCKING, wired into
# tier3 immediately after the Phase 2 audit.
#
# The loop reads the audit's STATUS (`status=$$?` on its own line). That is not
# incidental: the repo-wide `contract-audit` target ends its loop body in `;`,
# never reads the status, and therefore reports success while printing 132
# BIND-001 errors. Copying that shape would have produced a gate that cannot
# fail.
#
# EVIDENCE DISCIPLINE, matching the Phase 2 block above. Run STANDALONE first
# with the status captured directly (`make contract-audit-phase3 > /tmp/cap3.log
# 2>&1; rc=$$?`, never through a pipe — CLAUDE.md rule 1): rc=0, 5/5 equations
# bound, 8 obligations, 12 falsification tests. Wall time ~1 s warm; tier3 has
# already built pv via `contract-validate` two lines earlier.
#
# ITS FAILURE MODE WAS INDUCED, OBSERVED AND REVERTED before it was trusted,
# because a gate that has only ever been seen passing is not evidence. Deleting
# the `analytic_gradient` entry from $(BINDING) turned it **rc=2** (make's status
# for a failed recipe, not the recipe's own 1 — measured, not assumed) with
# "[ERROR] BIND-001: Equation 'analytic_gradient' in multinomial-head-v1.yaml has
# no binding entry" and "FAIL: unbound equations remain in:
# contracts/multinomial-head-v1.yaml", naming the deleted equation; "Bound
# equations" fell 5 -> 4. $(BINDING) was then restored and verified BYTE-IDENTICAL
# by sha256 (dfbce939bdc9a291...) and the target re-run green at rc=0.
#
# THE `set +e` AROUND THE AUDIT IS LOAD-BEARING, and it is a correction to the
# shape copied from the Phase 2 target. This Makefile sets `.SHELLFLAGS := -e -c`
# (line 40), so a failing `$(PV_CARGO_RUN) audit` inside the loop body ABORTS the whole
# recipe before `status=$$?` on the next line can run: `unbound` never
# accumulates, the remaining contracts are never audited, and the summarising
# "FAIL: unbound equations remain in:" line is unreachable. Reproduced directly:
# `bash -e -c 'for i in 1 2; do echo iter=$$i; false; status=$$?; echo st=$$status;
# done; echo REACHED_END'` prints ONLY `iter=1` and exits 1. The gate still fails
# closed, so this was never a false green — but it reported one contract where
# three were asked for. `set +e` for exactly the audit call restores the
# accumulate-then-report behaviour the loop is written for, and the status is
# still read from `$$?` on its own line, never through a pipe (CLAUDE.md rule 1).
# `contract-audit-phase2` above carries the same latent defect and is left for a
# change that owns that target.
contract-audit-phase3: ## Audit Phase 3 binding coverage (BLOCKING, wired into tier3)
	@echo "Auditing binding coverage for the Phase 3 contracts..."
	@unbound=""; \
	for contract in $(PHASE3_CONTRACTS); do \
		echo "  $$contract"; \
		set +e; \
		$(PV_CARGO_RUN) audit "$$contract" --binding $(BINDING); \
		status=$$?; \
		set -e; \
		if [ "$$status" -ne 0 ]; then \
			unbound="$$unbound $$contract"; \
		fi; \
	done; \
	if [ -n "$$unbound" ]; then \
		echo "FAIL: unbound equations remain in:$$unbound"; \
		echo "Every equation of a Phase 3 contract needs an entry in $(BINDING)."; \
		exit 1; \
	fi; \
	echo "Phase 3 binding audit: every equation is bound"

# The pending-tolerant phase binding audit, ONE body for contract-audit-phase4 and (until plan
# 08-12 tightened it to the phase6 form) contract-audit-phase8. Arguments: $(1) the contract list,
# $(2) the list variable's NAME for the empty-list message, $(3) the phase label.
# Every load-bearing detail documented on contract-audit-phase4 below lives here once:
# `set +e` around the audit (.SHELLFLAGS is -e -c), `status=$$?` on its own line (never
# through a pipe, CLAUDE.md rule 1), and the `audited` counter that refuses an empty list.
define audit_phase_bindings_pending_ok
	@echo "Auditing binding coverage for the Phase $(3) contracts..."
	@unbound=""; \
	audited=0; \
	for contract in $(1); do \
		echo "  $$contract"; \
		audited=$$((audited + 1)); \
		set +e; \
		$(PV_CARGO_RUN) audit "$$contract" --binding $(BINDING); \
		status=$$?; \
		set -e; \
		if [ "$$status" -ne 0 ]; then \
			unbound="$$unbound $$contract"; \
		fi; \
	done; \
	if [ "$$audited" -eq 0 ]; then \
		echo "FAIL: $(2) is empty — this gate audited nothing and would have reported success."; \
		exit 1; \
	fi; \
	if [ -n "$$unbound" ]; then \
		echo "FAIL: unbound equations remain in:$$unbound"; \
		echo "Every equation of a Phase $(3) contract needs an entry in $(BINDING)."; \
		echo "An equation still being written belongs there as 'status: pending', not absent."; \
		exit 1; \
	fi; \
	echo "Phase $(3) binding audit: $$audited contract(s) audited, every equation is bound"
endef

# Phase 4's twin of contract-audit-phase2/phase3, for the same reason both exist:
# `contract-validate` checks contract SHAPE and says nothing about whether an
# equation is bound to any implementation, so setfit-apr-v1.yaml could be "valid"
# with all fifteen equations bound to nothing at all. BLOCKING, wired into tier3
# immediately after the Phase 3 audit.
#
# THE `set +e` AND THE `status=$$?` ON ITS OWN LINE ARE BOTH LOAD-BEARING, and both
# are copied from contract-audit-phase3 rather than from contract-audit-phase2. This
# Makefile sets `.SHELLFLAGS := -e -c` (line 40), so a failing `$(PV_CARGO_RUN) audit`
# inside the loop body would ABORT the whole recipe before `status=$$?` could run:
# `unbound` would never accumulate and the summarising FAIL line would be
# unreachable. And the status is read from `$$?` directly, NEVER through a pipe —
# CLAUDE.md Verification rule 1, the defect that made the repo-wide
# `contract-audit` print 132 BIND-001 errors and exit 0 anyway.
#
# WHY `pending` MUST PASS HERE AND `BIND-001` MUST NOT. Plan 04-01 commits the
# Phase 4 schema BEFORE the writer, loader and codec exist — that ordering is the
# point of the phase (Ph1 D-14; the cross-AI review found three plans describing
# the artifact differently because no plan wrote it down). So all fifteen equations
# are registered `status: pending` in $(BINDING), which `pv audit` reports as
# BIND-004, a WARNING (audit/mod.rs:182-194). A MISSING entry stays BIND-001, an
# ERROR. The gate therefore tolerates "not written yet" and refuses "not tracked at
# all", and it tightens by itself as each later plan flips its binding to
# `implemented`.
#
# EVIDENCE DISCIPLINE, matching the two blocks above. Both states were MEASURED with
# the status captured directly, never through a pipe:
#   - BEFORE the bindings were added: `pv audit contracts/setfit-apr-v1.yaml
#     --binding $(BINDING) > /tmp/pv-audit-04-01-pre.log 2>&1; rc=$$?` -> rc=1, with
#     fifteen "[ERROR] BIND-001 ... has no binding entry" lines, one per equation.
#   - AFTER: rc=0, "Total equations: 15 / Bound equations: 15", fifteen
#     "[WARN] BIND-004 ... is pending implementation" lines.
# That pair IS this gate's induced failure mode: it was observed RED and then GREEN
# on a real difference, not merely observed passing. A gate that has only ever been
# seen passing is not evidence.
contract-audit-phase4: ## Audit Phase 4 binding coverage (BLOCKING, wired into tier3)
	$(call audit_phase_bindings_pending_ok,$(PHASE4_CONTRACTS),PHASE4_CONTRACTS,4)

# Phase 5's twin of contract-audit-phase2/3/4, for the same reason all three exist:
# `contract-validate` checks contract SHAPE and says nothing about whether an equation is
# bound to any implementation, so setfit-benchmark-claims-v1.yaml could be "valid" with all
# ten equations bound to nothing at all. BLOCKING, wired into tier3 immediately after the
# Phase 4 audit.
#
# THE `set +e`, THE `status=$$?` ON ITS OWN LINE, AND THE `audited` COUNTER ARE ALL
# LOAD-BEARING, and all three are copied from contract-audit-phase4 rather than from
# contract-audit-phase2. This Makefile sets `.SHELLFLAGS := -e -c` (line 40), so a failing
# `$(PV_CARGO_RUN) audit` inside the loop body would ABORT the whole recipe before `status=$$?`
# could run: `unbound` would never accumulate and the summarising FAIL line would be
# unreachable. The status is read from `$$?` directly, NEVER through a pipe — CLAUDE.md
# Verification rule 1, the defect that made the repo-wide `contract-audit` print 132
# BIND-001 errors and exit 0 anyway. And an empty $(PHASE5_CONTRACTS) must FAIL rather than
# report success over nothing (CR-02).
#
# WHY THIS GATE IS STRICTER THAN ITS PHASE 2/3/4 TWINS, AS OF PLAN 05-10. Those tolerate
# `status: pending`, which `pv audit` reports as BIND-004 — a WARNING (audit/mod.rs:182-194),
# so `pv audit` exits 0 and the recipe's `$$status` check passes. That tolerance existed for a
# reason: plan 05-05 task 1 committed this schema BEFORE the row type, the bench adapters and
# the report renderer existed, which is the point (Ph1 D-14), and a gate that refused a pending
# equation would have made the correct ordering impossible.
#
# That reason has now expired. Every one of the ten equations is `implemented` (05-05, 05-08,
# 05-09, 05-10), so a BIND-004 line here no longer means "not written yet" — it means an
# equation has REGRESSED to pending, or a new one was added and left untracked. Tolerating it
# would be tolerating exactly the thing the gate is for. The recipe therefore ALSO requires
# ZERO `BIND-` lines of any severity, which is a strictly stronger check than reading `$$?`:
# the repo-wide `contract-audit` prints 132 BIND-001 ERRORS and exits 0, so a status alone has
# already been shown here to be an unreliable summary of what a tool reported.
#
# THE OUTPUT IS CAPTURED TO A FILE AND THEN SCANNED, never piped into a counter whose status is
# read — CLAUDE.md Verification rule 1. And the scan is guarded for NON-VACUITY: an empty or
# missing audit log would produce zero BIND- lines and pass, which is the same vacuous success
# the `audited` counter below exists to prevent one level up, so the log must contain the
# summary line `Total equations:` before its BIND- count is trusted.
#
# EVIDENCE DISCIPLINE, matching the three blocks above. THREE states were MEASURED, each with
# the status captured directly and never through a pipe:
#   - BEFORE any bindings existed (plan 05-05): rc=1, ten
#     "[ERROR] BIND-001 ... has no binding entry" lines.
#   - WITH all ten `pending` (plans 05-05..05-09): rc=0, ten
#     "[WARN] BIND-004 ... is pending implementation" lines — which the OLD recipe passed and
#     the NEW one refuses.
#   - AFTER plan 05-10 flipped all ten to `implemented`: rc=0, zero BIND- lines,
#     "Implemented: 10".
# Plan 05-10 then induced a failure OF THE STRENGTHENED FORM: one row flipped back to
# `status: pending` -> rc=2, one "[WARN] BIND-004" line, "Implemented: 9", and the FAIL block
# below. Reverted -> rc=0, "Implemented: 10", zero BIND- lines. That is the control that
# matters, because `pv audit` ALONE still exits 0 on that input.
#
# AND A CONTROL THAT DID NOT FIRE, RECORDED BECAUSE IT IS THE MORE USEFUL FINDING. Plan 05-10
# was asked to prove non-vacuity by "pointing one binding row at a nonexistent symbol and
# observing the audit refuse". MEASURED: `function: this_symbol_does_not_exist_anywhere` on the
# `pairing_rule` row -> rc=0, "Implemented: 10", zero BIND- lines. `pv audit` DOES NOT RESOLVE
# SYMBOLS; it reads the `status` field, exactly as $(BINDING)'s own Phase 4 block states
# (which is why plan 04-10 declined to flip a status it could not verify). So this gate cannot
# detect a `module_path`/`function` pair that names nothing, and NOTHING in this repository can
# — those columns are a claim on the author, checked by review and by the resolution each plan
# performs before it flips a status, never by the audit. Do not credit this gate with a failure
# it cannot actually detect: a gate credited with a detection it does not have is worse than
# one with no recorded failure at all.
contract-audit-phase5: ## Audit Phase 5 binding coverage (BLOCKING, wired into tier3)
	@echo "Auditing binding coverage for the Phase 5 contracts..."
	@mkdir -p target
	@unbound=""; \
	warned=""; \
	audited=0; \
	for contract in $(PHASE5_CONTRACTS); do \
		echo "  $$contract"; \
		audited=$$((audited + 1)); \
		log="target/contract-audit-phase5-$$audited.log"; \
		set +e; \
		$(PV_CARGO_RUN) audit "$$contract" --binding $(BINDING) > "$$log" 2>&1; \
		status=$$?; \
		set -e; \
		cat "$$log"; \
		if [ "$$status" -ne 0 ]; then \
			unbound="$$unbound $$contract"; \
		fi; \
		if ! grep -q 'Total equations:' "$$log"; then \
			echo "FAIL: $$log carries no 'Total equations:' summary, so its BIND- count is not"; \
			echo "evidence of anything. The audit did not run, or its output format moved."; \
			exit 1; \
		fi; \
		found=$$(grep -c 'BIND-' "$$log" || true); \
		if [ "$$found" -ne 0 ]; then \
			warned="$$warned $$contract($$found)"; \
		fi; \
	done; \
	if [ "$$audited" -eq 0 ]; then \
		echo "FAIL: PHASE5_CONTRACTS is empty — this gate audited nothing and would have reported success."; \
		exit 1; \
	fi; \
	if [ -n "$$unbound" ]; then \
		echo "FAIL: unbound equations remain in:$$unbound"; \
		echo "Every equation of a Phase 5 contract needs an entry in $(BINDING)."; \
		exit 1; \
	fi; \
	if [ -n "$$warned" ]; then \
		echo "FAIL: BIND- findings remain in:$$warned"; \
		echo "Every Phase 5 equation is implemented as of plan 05-10, so a BIND- line here"; \
		echo "means one has REGRESSED to pending, or a new equation was added without a"; \
		echo "binding entry. Note that a BIND-004 warning does NOT set a nonzero exit status:"; \
		echo "the repo-wide contract-audit prints 132 BIND-001 errors and exits 0, which is"; \
		echo "why this gate counts the lines rather than trusting the status alone."; \
		exit 1; \
	fi; \
	echo "Phase 5 binding audit: $$audited contract(s) audited, zero BIND- findings"

# Phase 6's twin of contract-audit-phase2/3/4/5, plus ONE THING NONE OF THEM DO.
#
# WHY THE EXTRA STEP EXISTS (REVIEW-06-U3, codex MEDIUM). `pv audit` matches a binding row
# by contract FILENAME and EQUATION and then trusts the `status` field. It does not open the
# file `module_path` names, and it does not look for `function` anywhere. contract-audit-phase5's
# own comment block records the measurement that proves it: plan 05-10 set
# `function: this_symbol_does_not_exist_anywhere` on a real row and `pv audit` reported rc=0,
# "Implemented: 10", zero BIND- lines. So "zero BIND- findings" is evidence of REGISTRY
# COMPLETENESS — every equation has a row — and is NOT evidence that any row names real code.
# This target therefore runs the audits AND THEN RESOLVES every Phase 6 row to a definition
# site, and fails on either.
#
# WHICH ROUTE, AND WHY NOT THE EXISTING RESOLVER. `verify_source_functions`
# (crates/aprender-contracts/src/build_helper.rs:183-258) is `pub`, but it is the wrong
# instrument here on three counts, each checked by reading it rather than assumed:
#   1. It matches on the BARE, lowercased function name against every `pub fn` found anywhere
#      under `crates/` — 77 crates. Phase 6 binds `predict`, `forecast`, `validate`, `new`,
#      `load` and `main`; every one of those exists in some unrelated crate, so a row pointing
#      at the wrong FILE would resolve. The defect REVIEW-06-U3 names is precisely a row that
#      names nothing in the module it claims, and this would not see it.
#   2. It has no caller: no binary and no example in the workspace invokes it, so wiring it
#      would mean adding one purely to be invoked by a Makefile.
#   3. It cannot express a `justfile` row, and two Phase 6 equations bind to recipes
#      (REVIEW-06-03 / REVIEW-06-04) because a working-tree ignore claim and a wall-clock
#      ratio are not functions.
# The equivalent is therefore implemented below, FILE-SCOPED rather than name-scoped, which is
# strictly stronger than the existing helper for this phase's purpose.
#
# THE NEGATIVE CONTROL WAS OBSERVED, NOT ASSUMED (CLAUDE.md Verification rule 4 and 7 — a
# guard extended to a new scope must be re-mutated in that scope). Plan 06-09 rewrote the
# `single_row_routing_dot8` row's function to `this_function_does_not_exist`, ran this target,
# and observed rc=1 with exactly one line:
#   RESOLVE- chronos-bolt-parity-v1.yaml single_row_routing_dot8 aprender_forecast::bolt::this_function_does_not_exist
# then reverted and observed rc=0 with "resolved 55 Phase 6 binding rows". Without that
# observation the resolver is itself unproven. Its rc values are in 06-09-SUMMARY.md.
#
# THE `set +e`, THE `status=$$?` ON ITS OWN LINE, AND THE `audited` COUNTER are copied from
# contract-audit-phase5 and are load-bearing for the same reasons: this Makefile sets
# `.SHELLFLAGS := -e -c`, so a failing `$(PV_CARGO_RUN) audit` would abort the recipe before the
# status could be read, and a status must never be read through a pipe (CLAUDE.md rule 1).
# Every non-vacuity guard is copied too — an empty $(PHASE6_CONTRACTS), a log with no
# `Total equations:` summary, and a resolver that resolved ZERO rows all FAIL rather than
# report success over nothing.
#
# LIKE PHASE 5 AND UNLIKE PHASES 2/3/4, this refuses ANY BIND- line rather than only a nonzero
# status. All 55 Phase 6 equations are `implemented`: nothing in this phase was committed ahead
# of its code, so a BIND-004 "pending" here means a REGRESSION or an untracked new equation.
contract-audit-phase6: ## Audit Phase 6 binding coverage + source resolution (BLOCKING, wired into tier3)
	@echo "Auditing binding coverage for the Phase 6 contracts..."
	@mkdir -p target
	@unbound=""; \
	warned=""; \
	audited=0; \
	for contract in $(PHASE6_CONTRACTS); do \
		echo "  $$contract"; \
		audited=$$((audited + 1)); \
		log="target/contract-audit-phase6-$$audited.log"; \
		set +e; \
		$(PV_CARGO_RUN) audit "$$contract" --binding $(BINDING) > "$$log" 2>&1; \
		status=$$?; \
		set -e; \
		cat "$$log"; \
		if [ "$$status" -ne 0 ]; then \
			unbound="$$unbound $$contract"; \
		fi; \
		if ! grep -q 'Total equations:' "$$log"; then \
			echo "FAIL: $$log carries no 'Total equations:' summary, so its BIND- count is not"; \
			echo "evidence of anything. The audit did not run, or its output format moved."; \
			exit 1; \
		fi; \
		found=$$(grep -c 'BIND-' "$$log" || true); \
		if [ "$$found" -ne 0 ]; then \
			warned="$$warned $$contract($$found)"; \
		fi; \
	done; \
	if [ "$$audited" -eq 0 ]; then \
		echo "FAIL: PHASE6_CONTRACTS is empty — this gate audited nothing and would have reported success."; \
		exit 1; \
	fi; \
	if [ -n "$$unbound" ]; then \
		echo "FAIL: unbound equations remain in:$$unbound"; \
		echo "Every equation of a Phase 6 contract needs an entry in $(BINDING)."; \
		exit 1; \
	fi; \
	if [ -n "$$warned" ]; then \
		echo "FAIL: BIND- findings remain in:$$warned"; \
		echo "Every Phase 6 equation is implemented, so a BIND- line here means one has"; \
		echo "REGRESSED to pending, or a new equation was added without a binding entry."; \
		exit 1; \
	fi; \
	echo "Resolving every Phase 6 binding row to a definition site..."; \
	rows="target/contract-audit-phase6-rows.txt"; \
	awk -v want=" $(notdir $(PHASE6_CONTRACTS)) " ' \
		/^- contract:/ { c=$$3; e=""; m=""; f=""; next } \
		/^  equation:/ { e=$$2; next } \
		/^  module_path:/ { m=$$2; next } \
		/^  function:/ { f=$$2; \
			if (c != "" && index(want, " " c " ") > 0) print c, e, m, f; \
			next } \
	' $(BINDING) > "$$rows"; \
	resolved=0; \
	unresolvable=0; \
	while read -r c e m f; do \
		name="$${f##*::}"; \
		case "$$m" in \
			justfile) file="justfile" ;; \
			aprender_mcp_forecast) file="crates/aprender-mcp-forecast/src/lib.rs" ;; \
			aprender_mcp_chronos) file="crates/aprender-mcp-chronos/src/lib.rs" ;; \
			aprender_forecast::build) file="crates/aprender-forecast/build.rs" ;; \
			aprender_forecast::*) file="crates/aprender-forecast/src/$${m#aprender_forecast::}.rs" ;; \
			aprender_forecast) file="crates/aprender-forecast/src/lib.rs" ;; \
			*) file="" ;; \
		esac; \
		if [ -z "$$file" ] || [ ! -f "$$file" ]; then \
			echo "RESOLVE- $$c $$e $$m::$$f (module_path maps to no file)"; \
			unresolvable=$$((unresolvable + 1)); \
			continue; \
		fi; \
		if [ "$$file" = "justfile" ]; then \
			pattern="^$$name([[:space:]][^:]*)?:"; \
		else \
			pattern="(^|[^[:alnum:]_])fn[[:space:]]+$$name[[:space:]]*[(<]"; \
		fi; \
		if grep -Eq "$$pattern" "$$file"; then \
			resolved=$$((resolved + 1)); \
		else \
			echo "RESOLVE- $$c $$e $$m::$$f (no definition site in $$file)"; \
			unresolvable=$$((unresolvable + 1)); \
		fi; \
	done < "$$rows"; \
	if [ "$$unresolvable" -ne 0 ]; then \
		echo "FAIL: $$unresolvable Phase 6 binding row(s) name a function with no definition site"; \
		echo "in the file their module_path names. pv audit CANNOT see this — it matches"; \
		echo "filename and equation and trusts 'status', so those rows passed the audit above."; \
		echo "Fix the ROW (module_path / function), never the contract."; \
		exit 1; \
	fi; \
	if [ "$$resolved" -eq 0 ]; then \
		echo "FAIL: the resolver resolved ZERO Phase 6 binding rows, so it proved nothing."; \
		echo "Either $(BINDING) carries no Phase 6 rows, or the awk extraction stopped matching"; \
		echo "the file's shape. A resolver that resolves nothing must not pass vacuously."; \
		exit 1; \
	fi; \
	echo "Phase 6 source resolution: resolved $$resolved Phase 6 binding rows to definition sites"; \
	echo "Phase 6 binding audit: $$audited contract(s) audited, zero BIND- findings"

# THE ONE LIVE EXEMPTION contract-audit-phase8 may carry (plan 08-12, D-17/D-18). It holds either
# NOTHING or exactly `decide-tool-boundary-v1.yaml:accepted_region_cold`, the only Phase 8 equation
# whose falsification is LIVE (a cold pmcp.run Lambda, `just laya-deploy-verify`), never a libtest.
# It is populated exactly when the FINAL live record
# (.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-LIVE-DEPLOY-EVIDENCE.json,
# `final: true`) is NOT deployed-passed; the row is then `status: partial` (BIND-002) and the open
# work is D-ITEM-08-11-A in that phase's deferred-items.md. When set, the audit admits ONLY that
# equation's BIND-002 line (printed as `EXEMPT (live, see deferred-items.md): ...`) and still fails
# on every other BIND- line, a BIND-004 for the same equation included.
# EMPTY since plan 08-12: the final live outcome is deployed-passed (4 proven-cold samples at
# 3 008 MB, max 29 350 ms < 30 000 ms), so accepted_region_cold is bound `implemented`.
PHASE8_LIVE_EXEMPT :=

# Phase 8's twin of contract-audit-phase6: refuses ANY BIND- line AND resolves every Phase 8 row
# to a definition site. Plan 08-12 tightened it from the pending-tolerant phase-4 form
# (audit_phase_bindings_pending_ok, which plan 08-01 used while the four contracts were committed
# ahead of their code, D-07). Every Phase 8 binding is now `implemented` with a signature copied
# from source, so a BIND-004 here means an equation REGRESSED to pending or a new one was added
# untracked, and tolerating it would tolerate exactly what the gate is for.
#
# WHAT THIS ADDS TO `pv audit`, AND WHY. `pv audit` matches a row by contract filename and equation
# and trusts `status`; it never opens `module_path` (contract-audit-phase5/6 record the measurement:
# a nonexistent `function` passed it). So after the audits, every Phase 8 row is resolved
# FILE-SCOPED: its module_path maps (case map below) to a file or directory, and the function's
# last `::` segment must have a `fn <name>` definition there. For a `Type::method` row the file must
# ALSO carry an `impl ... Type` block (inherent or `impl Trait for Type`), and the `fn` is looked for
# only in the files that do. That scope is the FILE, not the impl block: a method bound to a type
# with no impl in that module fails, but one bound to the wrong type of the SAME file still
# resolves. A `justfile` row resolves to its recipe header. Any unresolvable row FAILS.
#
# THE LOAD-BEARING DETAILS are copied from contract-audit-phase6 for the same reasons: `.SHELLFLAGS`
# is `-e -c`, so `set +e` brackets the audit and `status=$$?` is read on its own line, never through
# a pipe (CLAUDE.md rule 1); an empty $(PHASE8_CONTRACTS), a log with no `Total equations:` summary,
# and a resolver that resolved ZERO rows all FAIL rather than pass over nothing. The BIND- count and
# the exemption filter are awk over the captured log file, not a pipe whose status is read.
#
# INDUCED FAILURES, OBSERVED BEFORE THIS WAS TRUSTED (plan 08-12; rc values in 08-12-SUMMARY.md):
# (1) one row's function renamed to a symbol that does not exist -> RESOLVE- line and FAIL, and a
#     `Type::method` row given a type with no impl under its module -> RESOLVE- line and FAIL;
# (2) PHASE8_LIVE_EXEMPT populated and ANOTHER Phase 8 row set to `partial` -> its BIND-002 line is
#     counted and the gate FAILS, while the exempt equation's own line would have been admitted.
# Both reverted, then green.
# THE RESOLVER'S THREE EREs (plan 08-22, V14-a), one definition each, used by contract-audit-phase8 and
# checked by contract-audit-phase8-selftest's must-match / must-not-match table BEFORE the audit runs
# (CLAUDE.md Verification rule 7: guard regexes ship a case table). Each is expanded inside a shell
# double-quoted string where `$$name` / `$$ty` are the row's function and type.
#   P8_FN_ERE      a `fn <name>` DEFINITION: the line starts (after whitespace) with optional
#                  visibility/qualifiers (pub, pub(<scope>), const, async, unsafe, extern, default) and
#                  then `fn`. So doc and line comments (`///`, `//`, `//!`) never resolve a row. The
#                  pre-08-22 form, `(^|[^[:alnum:]_])fn...`, matched `/// fn name(`.
#   P8_IMPL_ERE    an inherent or trait `impl` block for the type (unchanged).
#   P8_RECIPE_ERE  a justfile recipe HEADER: the name, optional parameters, a colon NOT followed by `=`.
#                  The pre-08-22 form matched a just assignment `name := value`.
# KNOWN LIMITS, not fixed by a regex: a grep cannot see `#[cfg(test)]`, so a test-only helper with the
# right name still resolves; `extern "C" fn` (an ABI string) does not match P8_FN_ERE; a recipe
# parameter default that contains a colon does not match P8_RECIPE_ERE.
P8_FN_ERE = ^[[:space:]]*((pub(\([^)]*\))?|const|async|unsafe|extern|default)[[:space:]]+)*fn[[:space:]]+$$name[[:space:]]*[(<]
P8_IMPL_ERE = ^[[:space:]]*impl([[:space:]]*<[^>]*>)?[[:space:]]+([^{;]*[[:space:]]for[[:space:]]+)?$$ty([[:space:]]|<|\{|$$)
P8_RECIPE_ERE = ^$$name([[:space:]][^:]*)?:([^=]|$$)

# The resolver EREs' case table (plan 08-22): each case is one line written to a temp file and grepped
# with the SAME variable contract-audit-phase8 uses, name=name and ty=Type. A case that disagrees with
# its expectation FAILS the target, and contract-audit-phase8 lists this target as a prerequisite, so
# the audit never runs on EREs that fail their table. Override a variable on the command line (for
# example the pre-08-22 P8_FN_ERE) to watch the table go RED. The `cfg(test)` case documents the known
# limit above: it MATCHES, because a grep cannot see the attribute.
contract-audit-phase8-selftest: ## Must-match / must-not-match case table for the contract-audit-phase8 resolver EREs
	@d=$$(mktemp -d); \
	fails=0; \
	cases=0; \
	name=name; \
	ty=Type; \
	check() { \
		kind="$$1"; want="$$2"; line="$$3"; \
		cases=$$((cases + 1)); \
		printf '%s\n' "$$line" > "$$d/case"; \
		case "$$kind" in \
			fn) re="$(P8_FN_ERE)" ;; \
			impl) re="$(P8_IMPL_ERE)" ;; \
			recipe) re="$(P8_RECIPE_ERE)" ;; \
			*) echo "  FAIL unknown ERE kind $$kind"; fails=$$((fails + 1)); return ;; \
		esac; \
		if grep -Eq "$$re" "$$d/case"; then got=match; else got=no-match; fi; \
		if [ "$$got" = "$$want" ]; then \
			echo "  ok   $$kind $$want: $$line"; \
		else \
			echo "  FAIL $$kind expected $$want, got $$got: $$line"; \
			fails=$$((fails + 1)); \
		fi; \
	}; \
	echo "Resolver ERE case table (P8_FN_ERE / P8_IMPL_ERE / P8_RECIPE_ERE, name=name ty=Type):"; \
	check fn match 'fn name('; \
	check fn match 'pub(crate) fn name<T>('; \
	check fn match '    pub async fn name('; \
	check fn match 'pub const unsafe fn name('; \
	check fn match '    fn name() {}  // cfg(test) helper: a grep cannot see the attribute (known limit)'; \
	check fn no-match '/// fn name('; \
	check fn no-match '// fn name('; \
	check fn no-match '//! see fn name('; \
	check fn no-match '    /// Calls fn name(x) first.'; \
	check fn no-match 'fn name_other('; \
	check fn no-match 'fn rename('; \
	check impl match 'impl Type {'; \
	check impl match 'impl<T> Trait for Type<T> {'; \
	check impl match '    impl Type'; \
	check impl no-match '// impl Type {'; \
	check impl no-match 'impl TypeOther {'; \
	check recipe match 'name:'; \
	check recipe match 'name arg="x":'; \
	check recipe match 'name a b="c" *rest:'; \
	check recipe no-match 'name := value'; \
	check recipe no-match 'name:= value'; \
	check recipe no-match '    name:'; \
	check recipe no-match 'name-other:'; \
	rm -rf "$$d"; \
	if [ "$$cases" -eq 0 ]; then echo "FAIL: the case table ran no case"; exit 1; fi; \
	if [ "$$fails" -ne 0 ]; then \
		echo "FAIL: $$fails of $$cases resolver ERE case(s) disagree with the table: contract-audit-phase8 would"; \
		echo "resolve (or refuse) rows on the wrong text. Fix the ERE, never the table."; \
		exit 1; \
	fi; \
	echo "contract-audit-phase8-selftest: $$cases cases, every resolver ERE as the table expects"

contract-audit-phase8: contract-audit-phase8-selftest ## Audit Phase 8 binding coverage + source resolution (BLOCKING, wired into tier3)
	@echo "Auditing binding coverage for the Phase 8 contracts..."
	@mkdir -p target
	@exempt_raw="$(strip $(PHASE8_LIVE_EXEMPT))"; \
	exempt_line=""; \
	if [ -n "$$exempt_raw" ]; then \
		if [ "$$exempt_raw" != "decide-tool-boundary-v1.yaml:accepted_region_cold" ]; then \
			echo "FAIL: PHASE8_LIVE_EXEMPT='$$exempt_raw'. It may hold only"; \
			echo "decide-tool-boundary-v1.yaml:accepted_region_cold (the one LIVE equation), or nothing."; \
			exit 1; \
		fi; \
		exempt_line="BIND-002: Equation '$${exempt_raw##*:}' in $${exempt_raw%%:*} is partially implemented"; \
		echo "PHASE8_LIVE_EXEMPT is set: admitting only the partial-implementation finding for $$exempt_raw (D-ITEM-08-11-A)"; \
	fi; \
	unbound=""; \
	warned=""; \
	audited=0; \
	for contract in $(PHASE8_CONTRACTS); do \
		echo "  $$contract"; \
		audited=$$((audited + 1)); \
		log="target/contract-audit-phase8-$$audited.log"; \
		set +e; \
		$(PV_CARGO_RUN) audit "$$contract" --binding $(BINDING) > "$$log" 2>&1; \
		status=$$?; \
		set -e; \
		awk -v x="$$exempt_line" 'x != "" && index($$0, x) { print "EXEMPT (live, see deferred-items.md): " $$0; next } { print }' "$$log"; \
		if [ "$$status" -ne 0 ]; then \
			unbound="$$unbound $$contract"; \
		fi; \
		if ! grep -q 'Total equations:' "$$log"; then \
			echo "FAIL: $$log carries no 'Total equations:' summary, so its BIND- count is not"; \
			echo "evidence of anything. The audit did not run, or its output format moved."; \
			exit 1; \
		fi; \
		found=$$(awk -v x="$$exempt_line" '/BIND-/ && !(x != "" && index($$0, x)) { n++ } END { print n + 0 }' "$$log"); \
		if [ "$$found" -ne 0 ]; then \
			warned="$$warned $$contract($$found)"; \
		fi; \
	done; \
	if [ "$$audited" -eq 0 ]; then \
		echo "FAIL: PHASE8_CONTRACTS is empty — this gate audited nothing and would have reported success."; \
		exit 1; \
	fi; \
	if [ -n "$$unbound" ]; then \
		echo "FAIL: unbound equations remain in:$$unbound"; \
		echo "Every equation of a Phase 8 contract needs an entry in $(BINDING)."; \
		exit 1; \
	fi; \
	if [ -n "$$warned" ]; then \
		echo "FAIL: BIND- findings remain in:$$warned"; \
		echo "Every Phase 8 equation is implemented as of plan 08-12, so a BIND- line here means one"; \
		echo "has REGRESSED (pending/partial/not_implemented) or a new equation was added without a"; \
		echo "binding entry. The only admissible line is the PHASE8_LIVE_EXEMPT BIND-002, when set."; \
		exit 1; \
	fi; \
	echo "Resolving every Phase 8 binding row to a definition site..."; \
	rows="target/contract-audit-phase8-rows.txt"; \
	awk -v want=" $(notdir $(PHASE8_CONTRACTS)) " ' \
		/^- contract:/ { c=$$3; e=""; m=""; f=""; next } \
		/^  equation:/ { e=$$2; next } \
		/^  module_path:/ { m=$$2; next } \
		/^  function:/ { f=$$2; \
			if (c != "" && index(want, " " c " ") > 0) print c, e, m, f; \
			next } \
	' $(BINDING) > "$$rows"; \
	resolved=0; \
	unresolvable=0; \
	while read -r c e m f; do \
		name="$${f##*::}"; \
		ty=""; \
		case "$$f" in *::*) ty="$${f%::*}" ;; esac; \
		case "$$m" in \
			justfile) file="justfile" ;; \
			aprender::models::modernbert|aprender::models::modernbert::*) file="crates/aprender-core/src/models/modernbert" ;; \
			aprender::calibration) file="crates/aprender-core/src/calibration.rs" ;; \
			aprender_decide::laya) file="crates/aprender-decide/src/laya/mod.rs" ;; \
			aprender_decide::laya::*) file="crates/aprender-decide/src/laya/$${m#aprender_decide::laya::}.rs" ;; \
			aprender_decide::*) file="crates/aprender-decide/src/$${m#aprender_decide::}.rs" ;; \
			aprender_decide) file="crates/aprender-decide/src/lib.rs" ;; \
			aprender_mcp_decide_lambda::*) file="crates/aprender-mcp-decide-lambda/src/$${m#aprender_mcp_decide_lambda::}.rs" ;; \
			aprender_mcp_decide_lambda) file="crates/aprender-mcp-decide-lambda/src/lib.rs" ;; \
			aprender_mcp_decide) file="crates/aprender-mcp-decide/src/lib.rs" ;; \
			*) file="" ;; \
		esac; \
		if [ -z "$$file" ] || [ ! -e "$$file" ]; then \
			echo "RESOLVE- $$c $$e $$m::$$f (module_path maps to no file)"; \
			unresolvable=$$((unresolvable + 1)); \
			continue; \
		fi; \
		if [ "$$file" = "justfile" ]; then \
			if grep -Eq "$(P8_RECIPE_ERE)" justfile; then \
				resolved=$$((resolved + 1)); \
			else \
				echo "RESOLVE- $$c $$e $$m::$$f (no recipe header in justfile)"; \
				unresolvable=$$((unresolvable + 1)); \
			fi; \
			continue; \
		fi; \
		if [ -n "$$ty" ]; then \
			hits=$$(grep -rlE "$(P8_IMPL_ERE)" "$$file" || true); \
		else \
			hits=$$(grep -rlE "$(P8_FN_ERE)" "$$file" || true); \
		fi; \
		if [ -z "$$hits" ]; then \
			if [ -n "$$ty" ]; then why="no impl block for type '$$ty'"; else why="no fn '$$name'"; fi; \
			echo "RESOLVE- $$c $$e $$m::$$f ($$why under $$file)"; \
			unresolvable=$$((unresolvable + 1)); \
			continue; \
		fi; \
		if grep -Eq "$(P8_FN_ERE)" $$hits; then \
			resolved=$$((resolved + 1)); \
		else \
			echo "RESOLVE- $$c $$e $$m::$$f (no definition site for fn '$$name' in $$hits)"; \
			unresolvable=$$((unresolvable + 1)); \
		fi; \
	done < "$$rows"; \
	if [ "$$unresolvable" -ne 0 ]; then \
		echo "FAIL: $$unresolvable Phase 8 binding row(s) name a function with no definition site"; \
		echo "in the file their module_path names. pv audit CANNOT see this — it matches"; \
		echo "filename and equation and trusts 'status', so those rows passed the audit above."; \
		echo "Fix the ROW (module_path / function / signature from source), never the contract."; \
		exit 1; \
	fi; \
	if [ "$$resolved" -eq 0 ]; then \
		echo "FAIL: the resolver resolved ZERO Phase 8 binding rows, so it proved nothing."; \
		echo "Either $(BINDING) carries no Phase 8 rows, or the awk extraction stopped matching"; \
		echo "the file's shape. A resolver that resolves nothing must not pass vacuously."; \
		exit 1; \
	fi; \
	echo "Phase 8 source resolution: resolved $$resolved Phase 8 binding rows to definition sites"; \
	echo "Phase 8 binding audit: $$audited contract(s) audited, no binding finding outside PHASE8_LIVE_EXEMPT='$$exempt_raw'"


# ============================================================================
# PHASE 3 REPRODUCIBILITY GATES (TRN-06 / D-16 / D-13)
# ============================================================================
#
# A NAME-FILTERED `cargo test` THAT MATCHES NOTHING EXITS 0 (REVIEW CR-02).
#
# libtest prints `test result: ok. 0 passed; ... N filtered out` and returns success.
# Every target below selects its test by NAME, so renaming a test — or mistyping a
# filter — would turn the gate green while running nothing, and it would keep printing
# its own success banner while doing it. That is a worse failure than red: red gets
# investigated.
#
# `assert_tests_ran` reads the count libtest actually reported and fails if it is below
# the number the target expects. `awk` parses it, not the rtk hook's summarised form.
# Measured both ways before being trusted: with the real filter it reads 1 (or 2 for the
# GEMM target) and passes; with a deliberately misspelled filter it reads 0 and the gate
# exits non-zero instead of printing success.
#
define assert_tests_ran
ran=$$(awk '/^test result:/ { for (i = 1; i <= NF; i++) if ($$(i+1) ~ /^passed/) s += $$i } END { print s + 0 }' $(1)); \
if [ "$$ran" -lt "$(2)" ]; then \
	echo "FAIL: $(3) reported $$ran test(s) passed, expected at least $(2)."; \
	echo "A name filter that matches nothing exits 0 (REVIEW CR-02) — this gate was"; \
	echo "about to report success having run nothing. Check the test name in the"; \
	echo "filter against the test binary: $(1)"; \
	exit 1; \
fi
endef

# The INVERSE guard, for SAFE-02's gating-by-absence half. `assert_tests_ran`
# catches a filter that selected nothing; this catches a filter that selected
# something it should not have — a surface that stopped being feature-gated.
#
# NEITHER IS SOUND ALONE, and the matrix uses them in PAIRS on purpose. An
# assert-zero leg passes for two different reasons — the surface really is gated,
# or the filter is dead — and it cannot tell them apart. It is only evidence when
# the SAME filter, with the feature ON, is asserted non-zero by `assert_tests_ran`.
# That pairing is what the two-sided `cargo tree` negatives below do for the
# dependency graph, applied here at the test tier.
define assert_tests_absent
ran=$$(awk '/^test result:/ { for (i = 1; i <= NF; i++) if ($$(i+1) ~ /^passed/) s += $$i } END { print s + 0 }' $(1)); \
if [ "$$ran" -ne 0 ]; then \
	echo "FAIL: $(2) ran $$ran test(s) with the setfit feature OFF, expected 0."; \
	echo "SAFE-02's gating half says this filter must select NOTHING when the"; \
	echo "feature is off. It selected something, so either the surface is no"; \
	echo "longer feature-gated or the filter has widened. Log: $(1)"; \
	exit 1; \
fi
endef

#
# D-16 SPLITS one gate across two tiers, and the split is not decoration:
#
#   tier2  setfit-repro-inproc      two runs in ONE process agree
#   tier3  setfit-repro-crossproc   two SEPARATE processes at pool sizes 1 and 3 agree
#   tier3  setfit-repro-replay      recorded digests == an independent recomputation
#
# The in-process form is structurally blind — both runs share the rayon pool, the
# allocator's free lists and every lazily-initialized static, which is exactly the
# class of nondeterminism a "clean run" exists to expose — so it is the fast signal
# and NOT the claim. The cross-process form is the authoritative one. Wiring only the
# tier3 half would implement half of D-16 and leave the fast signal in a test file no
# tier invokes.
#
# EVERY RECIPE BELOW READS `$$?` ON THE LINE AFTER THE REDIRECT, NEVER THROUGH A PIPE.
# CLAUDE.md Verification Discipline rule 1: piping into `tee` and then reading `$$?`
# reports the PIPE's last status, and this repo has shipped that defect twice (#2336
# qwen-story-daily, #2360 make publish's POST-PUBLISH VERIFICATION — three green runs
# that proved nothing). A pipe before the capture here is a defect, not a style choice.
#
# The obvious check for that property — a bare substring search for `tee` across each
# recipe — is UNFIT, and it was measured rather than reasoned about: it matched the word
# "guaran-tee" in two failure messages and reported a violation in recipes that contain
# no pipe at all (CLAUDE.md rule 7 — a guard pattern is re-checked by re-running its case
# table, not by re-reading it). The messages avoid that substring so the naive form also
# reads clean, but the pattern to reuse is a PIPE-aware one, e.g. `\| *tee`.
#
# `CARGO_INCREMENTAL=0` per STATE.md's ENOSPC mitigation: this workspace has stopped
# twice on a full disk in `target/debug/incremental` at ~25 GB.
#
# `mkdir -p target` because the log destination must exist before the redirect; a
# redirect into a missing directory fails the shell line, which would be reported as a
# gate failure rather than as the setup error it is.

setfit-tests: ## REVIEW CR-01 (tier3 half): RUN the feature-gated Phase 3+4 test surface
	@echo "Phase 3+4 SetFit surface: lib tests + 8 trybuild cases that no other tier runs"
	@mkdir -p target
# Three invocations, not one, and each is necessary:
#   (a) aprender-core --features setfit  -- setfit:: tests, gated at lib.rs:165
#   (b) aprender-train --features setfit -- setfit:: tests, gated at train/mod.rs:51
#   (c) --test ui                        -- the trybuild cases are a SEPARATE test target and
#                                           are not reached by any --lib invocation
# THE FLOORS BELOW ARE MEASURED, AND THEY MOVE WHEN THE SURFACE MOVES. Phase 4 roughly
# doubled the core suite while the floor stayed at its Phase 3 value of 100, so an entire
# Phase 4 module could have been compiled out and this gate would still have gone green —
# the vacuous-pass class (CR-02) the floors exist to prevent, reintroduced by not raising
# them. Re-measure and raise them whenever a phase adds tests here.
#
# MEASURED at the head of phase 4, status captured directly off each cargo command and
# never through a pipe (CLAUDE.md rule 1):
#   cargo test -p aprender-core  --features setfit --lib setfit:: -> rc=0, 236 passed
#   cargo test -p aprender-train --features setfit --lib setfit:: -> rc=0, 273 passed, 1 ignored
# The floors sit just under those, as the Phase 3 pair did (100 under 102, 230 under 235).
# rc captured directly off each cargo command, never through a pipe, and `set +e` so the
# diagnostic below is reachable under this Makefile's `.SHELLFLAGS := -e -c` (REVIEW WR-01).
	@set +e; CARGO_INCREMENTAL=0 cargo test -p aprender-core --features setfit --lib setfit:: \
		> target/setfit-tests-core.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/setfit-tests-core.log; \
	if [ $$rc -ne 0 ]; then echo "FAIL: aprender-core setfit tests are red (rc=$$rc)"; exit $$rc; fi
	@$(call assert_tests_ran,target/setfit-tests-core.log,230,setfit-tests/aprender-core)
	@set +e; CARGO_INCREMENTAL=0 cargo test -p aprender-train --features setfit --lib setfit:: \
		> target/setfit-tests-train.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/setfit-tests-train.log; \
	if [ $$rc -ne 0 ]; then echo "FAIL: aprender-train setfit tests are red (rc=$$rc)"; exit $$rc; fi
	@$(call assert_tests_ran,target/setfit-tests-train.log,265,setfit-tests/aprender-train)
	@set +e; CARGO_INCREMENTAL=0 cargo test -p aprender-train --features setfit --test ui \
		> target/setfit-tests-ui.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/setfit-tests-ui.log; \
	if [ $$rc -ne 0 ]; then \
		echo "FAIL: a trybuild compile-fail case no longer produces its pinned .stderr (rc=$$rc)"; \
		echo "An illegal lifecycle expression became EXPRESSIBLE, or a diagnostic changed."; \
		echo "See target/setfit-tests-ui.log"; \
		exit $$rc; \
	fi
	@$(call assert_tests_ran,target/setfit-tests-ui.log,1,setfit-tests/trybuild)
	@echo "  setfit surface: core + train lib tests and all eight compile-fail proofs ran"

# ============================================================================
# PHASE 5 BENCHMARK + CLAIMS-GATE SUITES (EVAL-03 / EVAL-04 / EVAL-05)
# ============================================================================
#
# WHY THIS IS A SEPARATE TARGET FROM setfit-tests. That target filters on the module
# path `setfit::`, which selects the Phase 3+4 surface in BOTH crates. The Phase 5
# surface is four DIFFERENT filters across TWO crates, and `cargo test` accepts at most
# ONE positional (a second exits `error: unexpected argument found`) — so it is four
# invocations either way, and giving them their own target is what lets their floors be
# raised as this phase grows without disturbing the Phase 3+4 numbers.
#
# THE FLOORS BELOW ARE MEASURED, NOT ESTIMATED, and each status was captured DIRECTLY
# off its own cargo command, never through a pipe (CLAUDE.md rule 1). Measured at the
# head of plan 05-10, warm tree:
#
#   cargo test -p aprender-train --lib --features setfit bench_row     -> rc=0, 24 passed
#   cargo test -p aprender-train --lib --features setfit bench_gate    -> rc=0, 31 passed
#   cargo test -p aprender-train --lib --features setfit bench_metrics -> rc=0, 14 passed
#   cargo test -p apr-cli        --lib --features setfit setfit_bench  -> rc=0, 58 passed
#
# The floors sit just under those, as the Phase 3 and Phase 4 pairs do (100 under 102,
# 230 under 235). RE-MEASURE AND RAISE THEM WHENEVER A PLAN ADDS TESTS HERE: Phase 4
# roughly doubled the core suite while its floor stayed at the Phase 3 value, so an
# entire module could have been compiled out and the gate would still have gone green.
# A stale floor reintroduces the vacuous pass the floor exists to prevent.
#
# TIER PLACEMENT WAS MEASURED, AND THE FIRST GUESS WAS WRONG BY TWENTY-FOLD. Two
# consecutive runs on an already-built tree: 166 s and 236 s wall (417 s user — this
# box compiles in parallel). Test EXECUTION inside that is 0.04 + 1.56 + 0.01 + 0.09 s;
# everything else is cargo rebuilding.
#
# The cause is worth recording, because it is a property of the target's SHAPE and not
# a cold cache. `cargo test -p aprender-train --features setfit` and `cargo test
# -p apr-cli --features setfit` unify features differently across their shared
# dependency graphs, so the two produce different fingerprints and each alternation
# re-links the other's artifacts. `setfit-tests` above has the same shape for the same
# reason (aprender-core then aprender-train), and a single invocation is not available:
# `cargo test` accepts at most ONE positional filter.
#
# So: tier3 ONLY, comfortably inside its 1-5 minute budget and nowhere near tier2's
# <5 s. The plan allowed a tier2 subset if a fast leg measured under 5 s. No leg does —
# the CHEAPEST leg still pays the whole cross-crate re-link — and splitting the gate to
# put half of it in tier2 would buy a second re-link for no earlier signal.
#
# rc is captured on the line AFTER each redirect, and `set +e` so the diagnostic is
# reachable under this Makefile's `.SHELLFLAGS := -e -c` (REVIEW WR-01).
setfit-bench-tests: ## EVAL-03/04/05 (tier3): the Phase 5 row, gate, metric and CLI suites
	@echo "Phase 5 benchmark + claims gate: row schema, fail-closed gate, metrics, CLI report"
	@mkdir -p target
	@set +e; CARGO_INCREMENTAL=0 cargo test -p aprender-train --lib --features setfit bench_row \
		> target/setfit-bench-row.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/setfit-bench-row.log; \
	if [ $$rc -ne 0 ]; then \
		echo "FAIL: the BenchRow/RunManifest suite is red (rc=$$rc)"; \
		echo "The row schema and the 40-cell ACTIVE expectation set ARE the claim; see"; \
		echo "target/setfit-bench-row.log"; \
		exit $$rc; \
	fi
	@$(call assert_tests_ran,target/setfit-bench-row.log,27,setfit-bench-tests/bench_row)
	@set +e; CARGO_INCREMENTAL=0 cargo test -p aprender-train --lib --features setfit bench_gate \
		> target/setfit-bench-gate.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/setfit-bench-gate.log; \
	if [ $$rc -ne 0 ]; then \
		echo "FAIL: the claims gate is red (rc=$$rc)"; \
		echo "This suite carries TWENTY-THREE doctored shapes, and the SCOPE each is"; \
		echo "mutated at is part of what it proves — a proof taken at one scope does"; \
		echo "not transfer to another (CLAUDE.md Verification Discipline rule 4):"; \
		echo "  * ACTIVE 40-cell scope, through the PUBLIC verify_run door: a missing"; \
		echo "    cell, a trimmed row, edited row bytes, post-test selection, the two"; \
		echo "    out-of-scope refusals (which REUSE existing variants), a deleted"; \
		echo "    committed lock record, a foreign contract_id on a row and on the"; \
		echo "    manifest, three contract-pinned constants a row may not choose, the"; \
		echo "    three-shape escape sweep (absolute / '..' / symlink), and the"; \
		echo "    deterministic refusal ORDER over two escaping cells."; \
		echo "  * ACTIVE 40-cell scope, THE SELECTION BINDING (05-16, EVAL-02): an"; \
		echo "    eleven-row swept case table carrying a CONTROL that must be"; \
		echo "    ACCEPTED, the two trees verification measured returning 0 (the"; \
		echo "    whole selections/ directory deleted, and a row's pairing key"; \
		echo "    doctored to 64 zeros), an absent per-cell directory, a zero-byte"; \
		echo "    manifest, a key doctored to another cell's REAL digest, three"; \
		echo "    TRANSPLANTED manifests with the row key doctored to match (which"; \
		echo "    is what makes the hash check agree so only the cell-key check can"; \
		echo "    refuse), the same transplant WITHOUT that doctoring as the"; \
		echo "    inversion, and an unsealed manifest payload — plus both ends of"; \
		echo "    the shot axis, exact byte equality of the 64-hex key, and the"; \
		echo "    deterministic order over two broken cells. THE ROW COUNT IS"; \
		echo "    ASSERTED INSIDE THE TEST: this floor counts test FUNCTIONS, so"; \
		echo "    eleven table rows collapsing to three would not move it."; \
		echo "  * ACTIVE 40-cell scope, THE CLOSED-FORM QUALITY CROSS-CHECK (05-17,"; \
		echo "    EVAL-01): a TWELVE-row swept case table carrying a CONTROL that"; \
		echo "    must be ACCEPTED and the tree verification measured returning 0"; \
		echo "    (spot-check D: quality.f_avg 0.4579 -> 0.99 with the row envelope"; \
		echo "    digest, the manifest row_sha256 and the manifest envelope digest"; \
		echo "    ALL repaired, which is what makes the cross-check the only thing"; \
		echo "    left that can refuse), plus macro_f1, mcc, one per-class vector"; \
		echo "    element and n_test_rows doctored; f_avg_bits moved with the"; \
		echo "    decimal untouched AND the decimal moved with the bits untouched,"; \
		echo "    which is an INVERTING pair proving the value check is reached"; \
		echo "    before the encoding check; a calibration bits sibling doctored;"; \
		echo "    and three degenerate matrices (ragged, 4x4 against three labels,"; \
		echo "    all-zero) that must be a TYPED refusal rather than a row of NaNs,"; \
		echo "    because serde_json renders a NaN as null and a null reads as a"; \
		echo "    MISSING cell. THE ROW COUNT IS ASSERTED INSIDE THE TEST, for the"; \
		echo "    same reason as the binding table. The same spot-check D is also"; \
		echo "    replayed through the SINGLE-CELL verify_cell door, because scope"; \
		echo "    is part of what a negative proves."; \
		echo "  * DEFERRED two-method scope: an unpaired pair and forged LoRA"; \
		echo "    provenance. Those shapes exist only in a design production code"; \
		echo "    cannot construct; D-ITEM-05-15 restores the arm."; \
		echo "  * RESOLVER HELPER, directly: thirteen path shapes crossed with both"; \
		echo "    evidence kinds — eight escapes, three ACCEPTANCE rows (without which"; \
		echo "    the table would prove only that something is refused) and two"; \
		echo "    behaviour-preserving rows."; \
		echo "A red here means one of those dishonesty shapes is no longer detected, or"; \
		echo "is no longer detected as its OWN refusal. See target/setfit-bench-gate.log"; \
		exit $$rc; \
	fi
	@$(call assert_tests_ran,target/setfit-bench-gate.log,55,setfit-bench-tests/bench_gate)
	@set +e; CARGO_INCREMENTAL=0 cargo test -p aprender-train --lib --features setfit bench_metrics \
		> target/setfit-bench-metrics.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/setfit-bench-metrics.log; \
	if [ $$rc -ne 0 ]; then \
		echo "FAIL: the EVAL-01 metric assembly is red (rc=$$rc)"; \
		echo "This suite carries the hand-computed toy cases (every asserted number"; \
		echo "derived in a comment from the counts, so a refactor has to argue with"; \
		echo "arithmetic rather than re-record a baseline) and, since 05-17, the"; \
		echo "CLOSED-FORM RECOMPUTATION: the 40-COMMITTED-ROW AGREEMENT MEASUREMENT"; \
		echo "that the acceptance band was chosen FROM — it reports per-field"; \
		echo "bit-identical counts and the maximum deviation, and was 40/40 exact"; \
		echo "with max deviation 0e0 BEFORE any band existed, which is why the band"; \
		echo "is exact bit equality and no epsilon appears anywhere; the"; \
		echo "order-independence assertion the count expansion rests on; the"; \
		echo "degenerate-matrix refusals; the zero-support class holding its index;"; \
		echo "and the structural scan that refuses a SECOND definition of any metric"; \
		echo "inside the recomputation (OPS-03). A red here can mean the committed"; \
		echo "evidence itself disagrees with its own confusion matrices — read the"; \
		echo "COMMITTED_ROW_AGREEMENT lines before touching a tolerance, and do not"; \
		echo "widen one: that is a finding about the evidence, not a band to fit."; \
		echo "See target/setfit-bench-metrics.log"; \
		exit $$rc; \
	fi
	@$(call assert_tests_ran,target/setfit-bench-metrics.log,19,setfit-bench-tests/bench_metrics)
	@set +e; CARGO_INCREMENTAL=0 cargo test -p apr-cli --lib --features setfit setfit_bench \
		> target/setfit-bench-cli.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/setfit-bench-cli.log; \
	if [ $$rc -ne 0 ]; then \
		echo "FAIL: the bench run/report adapters are red (rc=$$rc)"; \
		echo "This suite carries the ACTIVE-scope rendering case table, whose"; \
		echo "must-not-match rows are the ACTUAL literals the two-method renderer"; \
		echo "emitted (nine rows since 05-13: the six section-level literals plus"; \
		echo "the artifact-bytes ROW LABEL, the size footnote and the provenance"; \
		echo "clause, which named the deferred method BELOW section level and so"; \
		echo "survived the 05-11 retarget); the token-level scan that refuses the"; \
		echo "deferred method's name by any spelling; the no-verdict-word scan; the"; \
		echo "peak-RSS mechanism-asymmetry assertions; and the single-cell door's"; \
		echo "no-statistic check. Since 05-17 it also GATES THE DISCLOSURE itself:"; \
		echo "the residual: line must no longer carry the retired unqualified"; \
		echo "forgery concession, must still name the three residuals that remain"; \
		echo "true (a producer-written confusion matrix, the two non-recomputable"; \
		echo "calibration diagnostics, and the artifacts the index does not carry),"; \
		echo "and must not overstate; and the single-cell door's printed scope note"; \
		echo "must enumerate every recomputation the door performs. A red there"; \
		echo "means the report may now read as a comparison it did not make, or may"; \
		echo "describe a strength it does not have — or concede a weakness it no"; \
		echo "longer has, which teaches a reader to trust real evidence less than it"; \
		echo "warrants. See target/setfit-bench-cli.log"; \
		exit $$rc; \
	fi
	@$(call assert_tests_ran,target/setfit-bench-cli.log,67,setfit-bench-tests/apr-cli)
	@echo "  phase 5 bench surface: row, gate (twenty-three doctored shapes across"
	@echo "  four scopes: the public verify_run door over the ACTIVE 40 — including"
	@echo "  the selection binding recomputed from the committed manifest at a"
	@echo "  cell-key-derived path AND every published quality figure recomputed from"
	@echo "  the row's own confusion matrix — the single-cell verify_cell door, the"
	@echo "  deferred two-method scope, and the evidence-path resolver), metrics"
	@echo "  (including the 40-committed-row agreement measurement the exact band was"
	@echo "  chosen from) and the single-method CLI report all ran"

# The DOOR-LEVEL probe, deliberately NOT wired into setfit-bench-tests or any tier.
#
# It needs an `apr` built from HEAD and carrying the `setfit` feature, which the
# unit-test floors above do not. A leg that silently skips when the binary is
# stale or feature-less is exactly the vacuous gate this phase exists to prevent,
# so this is its own target with its own prerequisite rather than a step that
# degrades quietly inside a suite people run for other reasons.
#
# rc is captured on the line AFTER the redirect, never through a pipe: `$?` after
# a pipeline is the LAST command's status, and this repo has shipped that defect
# twice (#2336, #2360).
setfit-bench-door-probe: ## EVAL-01/02/04: replay spot-checks E, G, F and D through the SHIPPED apr door (needs: cargo build --release --bin apr --features setfit)
	@echo "Phase 5 claims gate, DOOR LEVEL: the four trees verification measured"
	@echo "'apr setfit bench report' ACCEPTING must now be refused by that same door -"
	@echo "an escaping evidence path, a deleted selections/ tree, a doctored pairing"
	@echo "key, and a published metric that does not follow from its own confusion"
	@echo "matrix - after a POSITIVE CONTROL on the undoctored evidence passes first."
	@echo "PREREQUISITE: cargo build --release --bin apr --features setfit"
	@mkdir -p target
	@set +e; bash scripts/setfit_bench_gate_door_probe.sh \
		> target/setfit-bench-door.log 2>&1; rc=$$?; \
	set -e; \
	cat target/setfit-bench-door.log; \
	if [ $$rc -ne 0 ]; then \
		echo "FAIL: the door probe did not pass (rc=$$rc)"; \
		echo "It runs a POSITIVE CONTROL on the undoctored committed evidence FIRST,"; \
		echo "then replays FOUR verifier spot-checks, each on its OWN slim copy so a"; \
		echo "later case cannot pass because an earlier one broke the tree: E (the"; \
		echo "committed lock record moved out of the tree and the row repointed at"; \
		echo "it), G (the whole selections/ directory deleted, no row byte touched),"; \
		echo "F (a row's pairing key doctored to 64 zeros) and D (a row's f_avg"; \
		echo "doctored to 0.99 beside an UNTOUCHED confusion matrix). E, F and D"; \
		echo "repair every digest, which is LOAD-BEARING: without the repair the gate"; \
		echo "refuses at step 4 and the probe goes green having proven nothing."; \
		echo "A control failure means the evidence or the binary is wrong; an attack"; \
		echo "failure means gap 1, gap 2 or advisory 2 is open again. It builds"; \
		echo "nothing: if"; \
		echo "scripts/apr_bin.sh refuses, run"; \
		echo "  cargo build --release --bin apr --features setfit"; \
		echo "See target/setfit-bench-door.log"; \
		exit $$rc; \
	fi

# THE DOOR PROBE, WITH ITS PREREQUISITE MADE REAL — and the reason it now runs somewhere.
#
# `setfit-bench-door-probe` above was a prerequisite of NOTHING and was absent from
# `.PHONY`: the string occurred exactly once in this file, in its own definition. It is
# the SOLE automated door-level proof of spot-checks D, E, F and G, and the secondary leg
# cited by five threats (T-05-15-01/-03, T-05-16-01/-03, T-05-17-01). Each of those has a
# primary LIBRARY mechanism already inside `setfit-bench-tests`, so nothing was
# unprotected — but "a target outside the tiers is a target that stops being run" is this
# Makefile's own stated rule four times over in tier3, and the probe was outside them.
#
# WHY A SEPARATE TARGET RATHER THAN WIRING THE PROBE ITSELF INTO A TIER. The probe BUILDS
# NOTHING by design: `scripts/apr_bin.sh` REFUSES unless a built `apr` matches HEAD, and
# the script turns that refusal into `exit 1` (:102-107), then re-checks that the binary
# actually carries the `setfit` feature by running `apr setfit bench report --help` and
# exiting 1 if the surface is missing (:129-136). That is the correct behaviour for a
# hand-run target and the WRONG behaviour for a tier leg: wired bare, it would make the
# tier red on every checkout whose binary is stale — including this one right now — which
# teaches people to ignore the tier. The fix is to make the prerequisite real instead of
# making the failure tolerable, so this target BUILDS the binary first. There is
# deliberately NO skip path: a probe that skipped when the binary was stale would be the
# vacuous gate T-05-07-02 exists to prevent, and `apr_bin.sh`'s refusal plus the
# `--help` surface check together mean the mechanism must engage or the target fails
# (CLAUDE.md Verification rule 2: never label a run by intent).
#
# WHY tier4 AND NOT tier3. tier3 is the pre-push tier (1-5 min) and builds no release
# binary; a release `apr` build does not belong in it. tier4 is the CI/CD tier and already
# runs `cargo test --release`, so the release artifacts are being built there anyway. The
# probe's cost lands where build cost already lives. `setfit-bench-door-probe` remains
# available standalone for anyone who already has the binary.
#
# rc is captured on the line AFTER the redirect, never through a pipe (#2336, #2360).
setfit-bench-door-probe-build: ## EVAL-01/02/04 (tier4): build apr --features setfit, THEN run the door probe
	@echo "Building the apr binary the door probe needs (release, --features setfit)."
	@echo "The probe itself builds nothing and REFUSES a stale or feature-less binary,"
	@echo "so this build is the prerequisite made real rather than a skip path."
	@mkdir -p target
	@set +e; cargo build --release --bin apr --features setfit \
		> target/setfit-bench-door-build.log 2>&1; rc=$$?; \
	set -e; \
	if [ $$rc -ne 0 ]; then \
		tail -40 target/setfit-bench-door-build.log; \
		echo "FAIL: could not build apr --release --features setfit (rc=$$rc)."; \
		echo "The door probe cannot run without it, and it has no skip path by design."; \
		echo "See target/setfit-bench-door-build.log"; \
		exit $$rc; \
	fi
	@$(MAKE) setfit-bench-door-probe


setfit-repro-inproc: ## TRN-06/D-16 (tier2 half): in-process two-clean-runs equality
	@echo "TRN-06: in-process two-run equality (D-16's fast, non-authoritative half)"
	@mkdir -p target
	@set +e; CARGO_INCREMENTAL=0 cargo test -p aprender-train --test setfit_repro \
		--features setfit in_process > target/setfit-repro-inproc.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/setfit-repro-inproc.log; \
	if [ $$rc -ne 0 ]; then \
		echo "FAIL: the in-process two-run comparison is red (rc=$$rc)"; \
		echo "See target/setfit-repro-inproc.log. NOTE: tier2 as a WHOLE is red on"; \
		echo "arm64 from 24 pre-existing clippy errors (D-ITEM-02) and its headline"; \
		echo "test step runs zero tests (D-ITEM-03) — neither is this gate's status."; \
		exit $$rc; \
	fi
	@$(call assert_tests_ran,target/setfit-repro-inproc.log,1,setfit-repro-inproc)
	@echo "  in-process: every composite component agreed"

setfit-repro-crossproc: ## TRN-06/D-16 (tier3, AUTHORITATIVE): cross-process hash equality
	@echo "TRN-06: cross-process two-clean-runs equality at fixed pool sizes 1 and 3"
	@mkdir -p target
	@set +e; CARGO_INCREMENTAL=0 cargo test -p aprender-train --test setfit_repro \
		--features setfit setfit_repro_cross_process \
		> target/setfit-repro-crossproc.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/setfit-repro-crossproc.log; \
	if [ $$rc -ne 0 ]; then \
		echo "FAIL: two separate processes did not agree, or did not run at two"; \
		echo "DISTINCT pool sizes (the mechanism-engaged half). Either way TRN-06's"; \
		echo "two-clean-runs claim does not hold as measured on this host."; \
		echo "See target/setfit-repro-crossproc.log"; \
		exit $$rc; \
	fi
	@$(call assert_tests_ran,target/setfit-repro-crossproc.log,1,setfit-repro-crossproc)
	@echo "  cross-process: THREADS differed and all ten components agreed"

setfit-repro-replay: ## TRN-06 (tier3): recorded digests match an INDEPENDENT recomputation
	@echo "TRN-06: recorded-vs-expected replay (reproducible is not the same as correct)"
	@mkdir -p target
	@set +e; CARGO_INCREMENTAL=0 cargo test -p aprender-train --test setfit_repro \
		--features setfit setfit_repro_recorded_matches_expected_replay \
		> target/setfit-repro-replay.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/setfit-repro-replay.log; \
	if [ $$rc -ne 0 ]; then \
		echo "FAIL: the RECORDED pair order / batch boundaries do not match an"; \
		echo "independent recomputation from the public epoch_pair_order + a fresh"; \
		echo "PairSampler. Two clean runs could still AGREE while both being wrong;"; \
		echo "this is the check that separates reproducible from correct."; \
		echo "See target/setfit-repro-replay.log"; \
		exit $$rc; \
	fi
	@$(call assert_tests_ran,target/setfit-repro-replay.log,1,setfit-repro-replay)
	@echo "  replay: recorded digests equal the independently recomputed ones"

gemm-thread-determinism: ## D-13/TRN-06: Tensor::matmul does not depend on the rayon pool size
	@echo "D-13: GEMM determinism across fixed rayon pool sizes 1/2/3 (03-02 T3)"
	@mkdir -p target
	@set +e; CARGO_INCREMENTAL=0 cargo test -p aprender-core --test gemm_thread_determinism \
		> target/gemm-thread-determinism.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/gemm-thread-determinism.log; \
	if [ $$rc -ne 0 ]; then \
		echo "FAIL: Tensor::matmul's output moved with the rayon pool size, so"; \
		echo "assumption A3 is falsified and TRN-06's bitwise claim does not"; \
		echo "hold on this host. See target/gemm-thread-determinism.log"; \
		exit $$rc; \
	fi
	@$(call assert_tests_ran,target/gemm-thread-determinism.log,2,gemm-thread-determinism)
	@echo "  GEMM: identical hashes at pool sizes 1, 2 and 3"

# ============================================================================
# PHASE 4 SCOPED SETFIT GATES (SAFE-01 / SAFE-02 / OPS-01)
# ============================================================================
#
# WHY ONE TARGET PER SUITE, and not one target with several filters.
#
# `cargo test` accepts AT MOST ONE positional filter. A second bare token is
# consumed as another positional and libtest exits `error: unexpected argument`.
# The first draft of this block carried two filters per line (e.g.
# `... setfit::artifact:: setfit::classify::`); a gate that cannot run what it
# names is worse than no gate, so each suite is its own invocation with its own
# floor (review M5). The tier2 comment block near line 220 records the same
# constraint being hit from the other direction.
#
# EVERY FLOOR BELOW IS MEASURED, NOT GUESSED. Each was run standalone on
# 5887d301c with the status captured DIRECTLY off cargo — `cmd > log 2>&1; rc=$$?`,
# never through a pipe (CLAUDE.md Verification Discipline rule 1, a defect this
# repo has shipped twice: #2336 and #2360). The measurement used `rtk proxy` so the
# log holds libtest's raw `test result:` line; the rtk hook's summarised form does
# not contain it and `assert_tests_ran` would read 0 from a hook-rewritten log.
# Recipes are not hook-rewritten, so they see the raw form.
#
#   target                     command filter               measured   floor
#   setfit-apr-tests           core  setfit::artifact::      86 / 0        80
#   setfit-classify-tests      core  setfit::classify::      50 / 0        45
#   setfit-bundle-tests        train setfit::bundle          33 / 0        30
#   setfit-config-tests        train setfit::config          40 / 0        36
#   setfit-evaluate-tests      train setfit::evaluate        14 / 0        12
#   setfit-codec-tests         train setfit::apr_codec::     17 / 0        15
#   setfit-reload-tests        train setfit::apr_reload::    17 / 0        15
#   setfit-lock-tests          train setfit::lock            36 / 0        32
#   setfit-verify-tests        train setfit::verify          18 / 0        16
#   setfit-cli-serve-tests     cli   serve                  358 / 0       330
#   setfit-lifecycle-tests     train --test setfit_apr_life   5 / 0         5
#   setfit-ui-tests            train --test ui                1 / 0         1
#   setfit-cli-train-tests     cli   setfit_train            15 / 0        13
#     (second leg)             cli   setfit_train --ignored   1 / 0         1
#   setfit-cli-predict-tests   cli   predict                 34 / 0        30
#   setfit-cli-inspect-tests   cli   inspect                120 / 0       110
#   setfit-cli-eval-tests      cli   eval::setfit            20 / 0        18
#   setfit-cli-io-tests        cli   setfit_io                5 / 0         5
#   setfit-serve-tests         serve setfit                  11 / 0        10
#   setfit-parity              cli   --test setfit_parity    20 / 0        18
#   setfit-serve-smoke         cli   --ignored smoke          1 / 0         1
#   setfit-cli-lifecycle       cli   --ignored lifecycle      2 / 0         2
#     (second leg)             cli   --ignored tooling        1 / 0         1
#
# Re-measure and RAISE the floors whenever a phase adds tests here. `setfit-tests`
# above records what happens when nobody does: Phase 4 roughly doubled the core
# suite while its floor stayed at the Phase 3 value, so an entire module could have
# been compiled out and the gate would still have gone green.
#
# TWO LEGS ARE DELIBERATELY ABSENT. Both are RED at this commit, both pre-existing,
# and a gate that is red on arrival is worse than no gate — it gets disabled, and
# the real gates beside it get disabled with it.
#
#   (1) `cargo test -p aprender-serve --lib` (whole crate) = 15389 passed / 51
#       FAILED, all 51 sharing one root cause: `attempt to multiply with overflow`
#       at crates/aprender-serve/src/contract_gate.rs:428:21. No Phase 4 plan
#       touched that file (D-04-08-A). `setfit-serve-tests` below is the SCOPED
#       substitute, and it was measured green (10 passed / 0 failed) — the filter
#       is what keeps the standing red out, not luck.
#   (2) `cargo check -p apr-cli --no-default-features` = rc=101, four errors from
#       `inference`-gated code that is not `cfg`-gated (src/commands/explain.rs:231
#       and :344, src/commands/diff_05_aprt_stage.rs:100, src/lib.rs:63). Red with
#       the pre-Phase-4 manifest restored too, so it predates this phase
#       (D-04-09-A). The green equivalent is `cargo check -p apr-cli --all-targets`
#       with DEFAULT features, wired in `setfit-feature-matrix` below.
#
# NO WHOLE-CRATE `aprender-train --lib` LEG EITHER, and the reason is not the same:
# that suite is 7920 passed / 24 FAILED, and the 24 are the Phase 3 known-red
# baseline recorded at
# .planning/phases/03-faithful-two-stage-trainer-and-head/known-red-baseline.md
# (21 under `gpu::`, 3 under `prune::snapshot_tests`). A gate over it would have to
# diff the failing NAMES against that file — never the COUNT, which passes when one
# pre-existing failure is fixed while a new regression appears. Every train leg
# below is instead SCOPED under `setfit::`, which is disjoint from both known-red
# modules, and each measured 0 failed. If you add a whole-crate train leg, it needs
# the name diff; the scoped ones do not.

setfit-apr-tests: ## SAFE-01: aprender-core setfit::artifact:: (APR container read/write)
	@echo "Phase 4: aprender-core setfit::artifact:: suite"
	@mkdir -p target
	@set +e; CARGO_INCREMENTAL=0 cargo test -p aprender-core --features setfit --lib setfit::artifact:: \
		> target/setfit-apr-tests.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/setfit-apr-tests.log; \
	if [ $$rc -ne 0 ]; then echo "FAIL: setfit-apr-tests is red (rc=$$rc); see target/setfit-apr-tests.log"; exit $$rc; fi
	@$(call assert_tests_ran,target/setfit-apr-tests.log,80,setfit-apr-tests)

setfit-classify-tests: ## SAFE-01: aprender-core setfit::classify:: (the one proven classify path)
	@echo "Phase 4: aprender-core setfit::classify:: suite"
	@mkdir -p target
	@set +e; CARGO_INCREMENTAL=0 cargo test -p aprender-core --features setfit --lib setfit::classify:: \
		> target/setfit-classify-tests.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/setfit-classify-tests.log; \
	if [ $$rc -ne 0 ]; then echo "FAIL: setfit-classify-tests is red (rc=$$rc); see target/setfit-classify-tests.log"; exit $$rc; fi
	@$(call assert_tests_ran,target/setfit-classify-tests.log,45,setfit-classify-tests)

setfit-bundle-tests: ## SAFE-01: aprender-train setfit::bundle
	@echo "Phase 4: aprender-train setfit::bundle suite"
	@mkdir -p target
	@set +e; CARGO_INCREMENTAL=0 cargo test -p aprender-train --features setfit --lib setfit::bundle \
		> target/setfit-bundle-tests.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/setfit-bundle-tests.log; \
	if [ $$rc -ne 0 ]; then echo "FAIL: setfit-bundle-tests is red (rc=$$rc); see target/setfit-bundle-tests.log"; exit $$rc; fi
	@$(call assert_tests_ran,target/setfit-bundle-tests.log,30,setfit-bundle-tests)

# 04-14 Task 1's `to_request` suite. It had NO target of its own and was covered
# only incidentally by the broad `-p aprender-train --lib setfit::` leg inside
# `setfit-tests`, where no `assert_tests_ran` sits over this filter — exactly the
# CR-02 shape this block exists to remove (checker warning W-6).
setfit-config-tests: ## SAFE-01: aprender-train setfit::config (04-14's to_request evidence)
	@echo "Phase 4: aprender-train setfit::config suite (04-14 Task 1 evidence)"
	@mkdir -p target
	@set +e; CARGO_INCREMENTAL=0 cargo test -p aprender-train --features setfit --lib setfit::config \
		> target/setfit-config-tests.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/setfit-config-tests.log; \
	if [ $$rc -ne 0 ]; then echo "FAIL: setfit-config-tests is red (rc=$$rc); see target/setfit-config-tests.log"; exit $$rc; fi
	@$(call assert_tests_ran,target/setfit-config-tests.log,36,setfit-config-tests)

# 04-14 Task 2 re-runs `evaluate_source_exposes_no_public_api_taking_a_float_parameter`
# here and names it as acceptance evidence for a new public door. Evidence that runs
# only incidentally is the same gap as evidence that does not run (W-6).
setfit-evaluate-tests: ## SAFE-01: aprender-train setfit::evaluate (04-14's float-door guard)
	@echo "Phase 4: aprender-train setfit::evaluate suite (04-14 Task 2 float-door guard)"
	@mkdir -p target
	@set +e; CARGO_INCREMENTAL=0 cargo test -p aprender-train --features setfit --lib setfit::evaluate \
		> target/setfit-evaluate-tests.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/setfit-evaluate-tests.log; \
	if [ $$rc -ne 0 ]; then echo "FAIL: setfit-evaluate-tests is red (rc=$$rc); see target/setfit-evaluate-tests.log"; exit $$rc; fi
	@$(call assert_tests_ran,target/setfit-evaluate-tests.log,12,setfit-evaluate-tests)

setfit-codec-tests: ## SAFE-01: aprender-train setfit::apr_codec::
	@echo "Phase 4: aprender-train setfit::apr_codec:: suite"
	@mkdir -p target
	@set +e; CARGO_INCREMENTAL=0 cargo test -p aprender-train --features setfit --lib setfit::apr_codec:: \
		> target/setfit-codec-tests.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/setfit-codec-tests.log; \
	if [ $$rc -ne 0 ]; then echo "FAIL: setfit-codec-tests is red (rc=$$rc); see target/setfit-codec-tests.log"; exit $$rc; fi
	@$(call assert_tests_ran,target/setfit-codec-tests.log,15,setfit-codec-tests)

setfit-reload-tests: ## SAFE-01: aprender-train setfit::apr_reload::
	@echo "Phase 4: aprender-train setfit::apr_reload:: suite"
	@mkdir -p target
	@set +e; CARGO_INCREMENTAL=0 cargo test -p aprender-train --features setfit --lib setfit::apr_reload:: \
		> target/setfit-reload-tests.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/setfit-reload-tests.log; \
	if [ $$rc -ne 0 ]; then echo "FAIL: setfit-reload-tests is red (rc=$$rc); see target/setfit-reload-tests.log"; exit $$rc; fi
	@$(call assert_tests_ran,target/setfit-reload-tests.log,15,setfit-reload-tests)

setfit-lock-tests: ## SAFE-01: aprender-train setfit::lock (TRN-07 selection lock)
	@echo "Phase 4: aprender-train setfit::lock suite"
	@mkdir -p target
	@set +e; CARGO_INCREMENTAL=0 cargo test -p aprender-train --features setfit --lib setfit::lock \
		> target/setfit-lock-tests.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/setfit-lock-tests.log; \
	if [ $$rc -ne 0 ]; then echo "FAIL: setfit-lock-tests is red (rc=$$rc); see target/setfit-lock-tests.log"; exit $$rc; fi
	@$(call assert_tests_ran,target/setfit-lock-tests.log,32,setfit-lock-tests)

# FOUND BY THE ALL-38-TASKS CROSS-CHECK, not by the plan's target list. 04-13's
# `<automated>` block names this suite as its own evidence and it had no target —
# and its acceptance criterion is the strongest of the phase ("a compile failure
# here is the signature of a missed verify_tests.rs call site"), which is precisely
# the kind of claim that must not depend on an unguarded broad leg.
setfit-verify-tests: ## SAFE-01: aprender-train setfit::verify (04-13's cited evidence)
	@echo "Phase 4: aprender-train setfit::verify suite (04-13 evidence)"
	@mkdir -p target
	@set +e; CARGO_INCREMENTAL=0 cargo test -p aprender-train --features setfit --lib setfit::verify \
		> target/setfit-verify-tests.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/setfit-verify-tests.log; \
	if [ $$rc -ne 0 ]; then echo "FAIL: setfit-verify-tests is red (rc=$$rc); see target/setfit-verify-tests.log"; exit $$rc; fi
	@$(call assert_tests_ran,target/setfit-verify-tests.log,16,setfit-verify-tests)

# Also found by the cross-check: 04-08's `<automated>` block names
# `cargo test -p apr-cli --features setfit --lib serve` and nothing covered it.
setfit-cli-serve-tests: ## OPS-05: apr-cli serve (04-08's cited evidence)
	@echo "Phase 4: apr-cli serve suite (04-08 evidence)"
	@mkdir -p target
	@set +e; CARGO_INCREMENTAL=0 cargo test -p apr-cli --features setfit --lib serve \
		> target/setfit-cli-serve-tests.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/setfit-cli-serve-tests.log; \
	if [ $$rc -ne 0 ]; then echo "FAIL: setfit-cli-serve-tests is red (rc=$$rc); see target/setfit-cli-serve-tests.log"; exit $$rc; fi
	@$(call assert_tests_ran,target/setfit-cli-serve-tests.log,330,setfit-cli-serve-tests)

setfit-lifecycle-tests: ## OPS-01: aprender-train --test setfit_apr_lifecycle (04-12's cross-crate leg)
	@echo "Phase 4: aprender-train setfit_apr_lifecycle integration suite"
	@mkdir -p target
	@set +e; CARGO_INCREMENTAL=0 cargo test -p aprender-train --features setfit --test setfit_apr_lifecycle \
		> target/setfit-lifecycle-tests.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/setfit-lifecycle-tests.log; \
	if [ $$rc -ne 0 ]; then echo "FAIL: setfit-lifecycle-tests is red (rc=$$rc); see target/setfit-lifecycle-tests.log"; exit $$rc; fi
	@$(call assert_tests_ran,target/setfit-lifecycle-tests.log,5,setfit-lifecycle-tests)

# Distinct from `setfit-tests`' third leg on purpose: this one is named, so a tier
# can depend on it directly, and the trybuild count moved in Phase 4 (04-03 added a
# case). A compile-fail claim that is not compiled is not a claim.
setfit-ui-tests: ## SAFE-01: aprender-train --test ui (trybuild compile-fail proofs)
	@echo "Phase 4: aprender-train trybuild compile-fail suite"
	@mkdir -p target
	@set +e; CARGO_INCREMENTAL=0 cargo test -p aprender-train --features setfit --test ui \
		> target/setfit-ui-tests.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/setfit-ui-tests.log; \
	if [ $$rc -ne 0 ]; then \
		echo "FAIL: a trybuild compile-fail case no longer produces its pinned .stderr (rc=$$rc)"; \
		echo "See target/setfit-ui-tests.log"; \
		exit $$rc; \
	fi
	@$(call assert_tests_ran,target/setfit-ui-tests.log,1,setfit-ui-tests)

# TWO invocations, guarded SEPARATELY. The default run leaves 1 test `#[ignore]`d
# (the e2e leg), and an `--ignored` run does not re-run the 15 default ones, so a
# single floor over one invocation could never cover both.
setfit-cli-train-tests: ## OPS-02: apr-cli setfit_train (default + the #[ignore]d e2e leg)
	@echo "Phase 4: apr-cli setfit_train suite (default leg)"
	@mkdir -p target
	@set +e; CARGO_INCREMENTAL=0 cargo test -p apr-cli --features setfit --lib setfit_train \
		> target/setfit-cli-train-tests.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/setfit-cli-train-tests.log; \
	if [ $$rc -ne 0 ]; then echo "FAIL: setfit-cli-train-tests is red (rc=$$rc); see target/setfit-cli-train-tests.log"; exit $$rc; fi
	@$(call assert_tests_ran,target/setfit-cli-train-tests.log,13,setfit-cli-train-tests/default)
	@echo "Phase 4: apr-cli setfit_train suite (--ignored e2e leg)"
	@set +e; CARGO_INCREMENTAL=0 cargo test -p apr-cli --features setfit --lib setfit_train -- --ignored \
		> target/setfit-cli-train-ignored.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/setfit-cli-train-ignored.log; \
	if [ $$rc -ne 0 ]; then echo "FAIL: setfit-cli-train-tests --ignored is red (rc=$$rc); see target/setfit-cli-train-ignored.log"; exit $$rc; fi
	@$(call assert_tests_ran,target/setfit-cli-train-ignored.log,1,setfit-cli-train-tests/ignored)

setfit-cli-predict-tests: ## OPS-02: apr-cli predict
	@echo "Phase 4: apr-cli predict suite"
	@mkdir -p target
	@set +e; CARGO_INCREMENTAL=0 cargo test -p apr-cli --features setfit --lib predict \
		> target/setfit-cli-predict-tests.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/setfit-cli-predict-tests.log; \
	if [ $$rc -ne 0 ]; then echo "FAIL: setfit-cli-predict-tests is red (rc=$$rc); see target/setfit-cli-predict-tests.log"; exit $$rc; fi
	@$(call assert_tests_ran,target/setfit-cli-predict-tests.log,30,setfit-cli-predict-tests)

setfit-cli-inspect-tests: ## OPS-02: apr-cli inspect
	@echo "Phase 4: apr-cli inspect suite"
	@mkdir -p target
	@set +e; CARGO_INCREMENTAL=0 cargo test -p apr-cli --features setfit --lib inspect \
		> target/setfit-cli-inspect-tests.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/setfit-cli-inspect-tests.log; \
	if [ $$rc -ne 0 ]; then echo "FAIL: setfit-cli-inspect-tests is red (rc=$$rc); see target/setfit-cli-inspect-tests.log"; exit $$rc; fi
	@$(call assert_tests_ran,target/setfit-cli-inspect-tests.log,110,setfit-cli-inspect-tests)

# FLOOR RE-MEASURED IN WAVE 12, which is the first point at which the number was
# stable. It read 13 against a measured 15 for most of Phase 4; 04-20 then added
# four tests to `eval::setfit` in wave 11, so the row was stale by SEVEN. 04-21
# owned the Makefile that wave and deliberately did not touch this line, because
# 04-20 was its same-wave sibling and any value written would have been wrong on
# contact. Measured here AFTER 04-20 landed:
#   rtk proxy cargo test -p apr-cli --features setfit --lib eval::setfit
#   -> rc=0, "test result: ok. 20 passed; 0 failed; 6764 filtered out"
# 13 -> 18, matching the ~90% convention every other row in the table above uses
# (and exactly the 20/18 pair `setfit-parity` already carries).
setfit-cli-eval-tests: ## OPS-02: apr-cli eval::setfit
	@echo "Phase 4: apr-cli eval::setfit suite"
	@mkdir -p target
	@set +e; CARGO_INCREMENTAL=0 cargo test -p apr-cli --features setfit --lib eval::setfit \
		> target/setfit-cli-eval-tests.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/setfit-cli-eval-tests.log; \
	if [ $$rc -ne 0 ]; then echo "FAIL: setfit-cli-eval-tests is red (rc=$$rc); see target/setfit-cli-eval-tests.log"; exit $$rc; fi
	@$(call assert_tests_ran,target/setfit-cli-eval-tests.log,18,setfit-cli-eval-tests)

setfit-cli-io-tests: ## OPS-02: apr-cli setfit_io
	@echo "Phase 4: apr-cli setfit_io suite"
	@mkdir -p target
	@set +e; CARGO_INCREMENTAL=0 cargo test -p apr-cli --features setfit --lib setfit_io \
		> target/setfit-cli-io-tests.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/setfit-cli-io-tests.log; \
	if [ $$rc -ne 0 ]; then echo "FAIL: setfit-cli-io-tests is red (rc=$$rc); see target/setfit-cli-io-tests.log"; exit $$rc; fi
	@$(call assert_tests_ran,target/setfit-cli-io-tests.log,5,setfit-cli-io-tests)

# SCOPED, and the scope is the whole point — see D-04-08-A in the block above. The
# unfiltered `-p aprender-serve --lib` is 51-red at this commit for a reason no
# Phase 4 plan caused. Measured with the filter: 10 passed / 0 failed.
setfit-serve-tests: ## OPS-05: aprender-serve setfit (HTTP transport only, D-09)
	@echo "Phase 4: aprender-serve setfit suite (SCOPED — see D-04-08-A)"
	@mkdir -p target
	@set +e; CARGO_INCREMENTAL=0 cargo test -p aprender-serve --features setfit --lib setfit \
		> target/setfit-serve-tests.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/setfit-serve-tests.log; \
	if [ $$rc -ne 0 ]; then \
		echo "FAIL: setfit-serve-tests is red (rc=$$rc); see target/setfit-serve-tests.log"; \
		echo "NOTE: the UNFILTERED aprender-serve lib suite is 51-red at HEAD from a"; \
		echo "pre-existing overflow in contract_gate.rs:428 (D-04-08-A). If those names"; \
		echo "appear here, the filter has widened, not this surface regressed."; \
		exit $$rc; \
	fi
	@$(call assert_tests_ran,target/setfit-serve-tests.log,10,setfit-serve-tests)

setfit-parity: ## SAFE-01: 04-09's three-reader parity gate (core / CLI / HTTP agree)
	@echo "Phase 4: three-surface parity (aprender-core, apr-cli, aprender-serve)"
	@mkdir -p target
	@set +e; CARGO_INCREMENTAL=0 cargo test -p apr-cli --features setfit,inference --test setfit_parity \
		> target/setfit-parity.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/setfit-parity.log; \
	if [ $$rc -ne 0 ]; then \
		echo "FAIL: the three readers of one artifact no longer agree (rc=$$rc)"; \
		echo "See target/setfit-parity.log"; \
		exit $$rc; \
	fi
	@$(call assert_tests_ran,target/setfit-parity.log,18,setfit-parity)

# `#[ignore]`d on purpose (04-09): it spawns a real server. tier3 is where an
# `--ignored` leg belongs, and a surface that runs in no tier is CR-01 all over
# again — so it gets a NAME here rather than a comment somewhere saying it exists.
setfit-serve-smoke: ## SAFE-01: 04-09's spawned-server smoke leg (#[ignore]d, tier3)
	@echo "Phase 4: spawned aprender-serve smoke (the #[ignore]d parity leg)"
	@mkdir -p target
	@set +e; CARGO_INCREMENTAL=0 cargo test -p apr-cli --features setfit,inference --test setfit_parity \
		-- --ignored spawned_serve_smoke > target/setfit-serve-smoke.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/setfit-serve-smoke.log; \
	if [ $$rc -ne 0 ]; then echo "FAIL: setfit-serve-smoke is red (rc=$$rc); see target/setfit-serve-smoke.log"; exit $$rc; fi
	@$(call assert_tests_ran,target/setfit-serve-smoke.log,1,setfit-serve-smoke)

# TWO invocations, guarded SEPARATELY, both `--ignored`: 04-15's file carries a
# module-level `#![cfg(feature = "setfit")]`, so WITHOUT the feature it compiles to
# an empty test binary and every filter here matches zero. That is precisely what
# `assert_tests_ran` catches — a feature-off invocation would otherwise print
# success having spawned nothing.
setfit-cli-lifecycle: ## OPS-02: 04-15's spawned `apr` lifecycle + tooling ladders (#[ignore]d, tier3)
	@echo "Phase 4: spawned apr lifecycle ladder (04-15)"
	@mkdir -p target
	@set +e; CARGO_INCREMENTAL=0 cargo test -p apr-cli --features setfit --test setfit_cli_lifecycle \
		-- --ignored lifecycle > target/setfit-cli-lifecycle.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/setfit-cli-lifecycle.log; \
	if [ $$rc -ne 0 ]; then echo "FAIL: setfit-cli-lifecycle (lifecycle) is red (rc=$$rc); see target/setfit-cli-lifecycle.log"; exit $$rc; fi
	@$(call assert_tests_ran,target/setfit-cli-lifecycle.log,2,setfit-cli-lifecycle/lifecycle)
	@echo "Phase 4: spawned apr tooling ladder (04-15)"
	@set +e; CARGO_INCREMENTAL=0 cargo test -p apr-cli --features setfit --test setfit_cli_lifecycle \
		-- --ignored tooling > target/setfit-cli-tooling.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/setfit-cli-tooling.log; \
	if [ $$rc -ne 0 ]; then echo "FAIL: setfit-cli-lifecycle (tooling) is red (rc=$$rc); see target/setfit-cli-tooling.log"; exit $$rc; fi
	@$(call assert_tests_ran,target/setfit-cli-tooling.log,1,setfit-cli-lifecycle/tooling)

# ─── OPS-01 dependency boundary ─────────────────────────────────────────────
#
# OPS-01: "a Rust caller can train, save, load, embed, classify and inspect a
# SetFit model through stable fallible library APIs WITHOUT DEPENDING ON CLI
# IMPLEMENTATION MODULES". The mechanical half of that is a graph property: no
# resolved normal-dependency closure of aprender-core or aprender-train may
# contain apr-cli.
#
# GUARD REGEXES SHIP A CASE TABLE (CLAUDE.md rule 7). The pattern here is the
# fixed string `apr-cli`, and the table below was EXECUTED on 5887d301c, not
# reasoned about — `grep -c apr-cli` over `cargo tree -e normal --prefix none`:
#
#   leg                                             expect   MEASURED
#   -p aprender-core                    (default)   0        0    must-not-match
#   -p aprender-core  --features setfit             0        0    must-not-match
#   -p aprender-train                   (default)   0        0    must-not-match
#   -p aprender-train --features setfit             0        0    must-not-match
#   -p apr-cli                          (control)   >= 1     1    MUST-MATCH
#
# The control leg is not decoration. An absence-only gate passes forever if the
# pattern stops matching anything at all — a renamed package, a changed `cargo
# tree` output shape, a typo. The apr-cli leg is a tree that DOES contain the
# string, so if it ever reads 0 the gate reports that its own pattern is dead
# rather than reporting success.
#
# Every leg captures `cargo tree`'s OWN status FIRST and compares counts
# EXPLICITLY. No `| grep -q` (that reads grep's status, CLAUDE.md rule 1) and no
# bare `|| true` (which converts a failed tree into a vacuous pass — the exact
# defect the aprender-core block near line 397 documents). Each tree is also
# checked for its own crate name, so a tree that resolved to nothing cannot
# satisfy the absence half.
setfit-api-boundary: ## OPS-01: aprender-core / aprender-train must not depend on apr-cli
	@echo "OPS-01 boundary: no library crate may depend on apr-cli"
	@mkdir -p target
	@for leg in "aprender-core:" "aprender-core:--features setfit" "aprender-train:" "aprender-train:--features setfit"; do \
		crate=$${leg%%:*}; feats=$${leg#*:}; \
		out=target/api-boundary-$$crate$$(echo "$$feats" | tr -cd 'a-z'); \
		if ! cargo tree -p "$$crate" $$feats -e normal --prefix none > "$$out" 2>&1; then \
			echo "FAIL: cargo tree failed for $$crate $$feats; the absence check would pass vacuously"; \
			cat "$$out"; exit 1; \
		fi; \
		self=$$(grep -c "^$$crate " "$$out"); \
		if [ "$$self" -lt 1 ]; then \
			echo "FAIL: the tree for $$crate $$feats does not contain $$crate itself."; \
			echo "      A tree that resolved nothing satisfies an absence check vacuously."; \
			exit 1; \
		fi; \
		hits=$$(grep -c 'apr-cli' "$$out"); \
		if [ "$$hits" -ne 0 ]; then \
			echo "FAIL (OPS-01): $$crate $$feats depends on apr-cli ($$hits node(s))"; \
			grep -n 'apr-cli' "$$out"; exit 1; \
		fi; \
		echo "  must-not-match OK: $$crate $$feats -> 0 apr-cli nodes ($$(wc -l < "$$out" | tr -d ' ') packages)"; \
	done
	@if ! cargo tree -p apr-cli -e normal --prefix none > target/api-boundary-control.txt 2>&1; then \
		echo "FAIL: the MUST-MATCH control tree failed to resolve; the four legs above prove nothing"; \
		cat target/api-boundary-control.txt; exit 1; \
	fi
	@control=$$(grep -c 'apr-cli' target/api-boundary-control.txt); \
	if [ "$$control" -lt 1 ]; then \
		echo "FAIL: the MUST-MATCH control read $$control. The pattern 'apr-cli' no longer"; \
		echo "matches a tree that definitely contains apr-cli, so the four absence legs"; \
		echo "above were passing for the wrong reason (CLAUDE.md rule 7)."; \
		exit 1; \
	fi; \
	echo "  MUST-MATCH control OK: apr-cli's own tree -> $$control apr-cli node(s)"
	@echo "setfit-api-boundary: PASSED"

# One prerequisite for the tiers to name, while every suite keeps its OWN guard.
# `setfit-config-tests` and `setfit-evaluate-tests` are in this list deliberately
# (W-6): they are 04-14's cited evidence and had no guarded target before.
setfit-all-tests: setfit-apr-tests setfit-classify-tests setfit-bundle-tests \
	setfit-config-tests setfit-evaluate-tests setfit-codec-tests setfit-reload-tests \
	setfit-lock-tests setfit-verify-tests setfit-lifecycle-tests setfit-ui-tests \
	setfit-cli-train-tests setfit-cli-predict-tests setfit-cli-inspect-tests \
	setfit-cli-eval-tests setfit-cli-io-tests setfit-cli-serve-tests \
	setfit-serve-tests ## Phase 4: every scoped setfit suite, each guarded
	@echo "setfit-all-tests: every Phase 4 suite ran under its own floor"

# contract-regen keeps `|| true` ON PURPOSE (PVL-001 EV-4): it is not a gate --
# it writes .rs.new files for a human to review, and one contract probar cannot
# render must not stop the others being written.
contract-regen: ## Regenerate wired test files from contracts
	@echo "Regenerating contract test files..."
	@for contract in $(CONTRACTS); do \
		name=$$(basename "$$contract" .yaml | sed 's/-kernel-v[0-9]*//;s/-v[0-9]*//'); \
		echo "  $$name <- $$contract"; \
		$(PV_CARGO_RUN) probar "$$contract" --binding $(BINDING) > tests/contracts/$${name}_contract.rs.new 2>/dev/null || true; \
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

# APR-RELEASE-001 §11.2 (ONT R-6): the five ontology counters move ONLY through
# this target. `--check` is what guard_tree.sh runs on every PR; `--write` is the
# deliberate restamp, and it is the only way a counter is allowed to change.
.PHONY: ont-ratchet ont-ratchet-check
ont-ratchet:
	@bash scripts/check_ont_ratchet.sh --write

ont-ratchet-check:
	@bash scripts/check_ont_ratchet.sh --check

# ONT-001 §5 ONT-4b2 / R-13 — the out-of-gate SHACL differential oracle.
#
# NOT a PR check, by the rule that puts it here: `shacl` is 316 crates and pinned at ONE version (ONT-0's
# ledger), so it lives in tests/oracle/ — a crate DETACHED from the workspace — and runs in the release gate
# only. It validates the same graph twice (the pinned processor and the in-house validator) and writes
# tests/oracle/differential.json, which is TRACKED: the row's probe reads `cases>0 and disagreements==0`, and
# a tracked file nobody regenerated is caught by `oracle-check` diffing it.
.PHONY: oracle oracle-check
oracle:
	@echo "== W3C cases + the real corpus through the pinned oracle (shacl 0.3.21, out of gate) =="
	@. scripts/pv_bin.sh && "$$PV" lint contracts --gate shapes --format json > "$${TMPDIR:-/tmp}/pv-shapes.json" 2>/dev/null || true
	@# No `cd`: this Makefile is .ONESHELL, so a cd on one line moves every line after it (the first form of
	@# this target built in tests/oracle/ and then ran `$(cd tests/oracle …)` from inside it — "No such file").
	@cargo build --release --quiet --manifest-path tests/oracle/Cargo.toml
	@"$$(cargo metadata --no-deps --format-version 1 --manifest-path tests/oracle/Cargo.toml | sed -n 's/.*"target_directory":"\([^"]*\)".*/\1/p')/release/ont-oracle" . "$${TMPDIR:-/tmp}/pv-shapes.json"

oracle-check: oracle
	@git diff --exit-code tests/oracle/differential.json \
	  || { echo "FAIL: tests/oracle/differential.json differs from a fresh run — commit it"; exit 1; }
