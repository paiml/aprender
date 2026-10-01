---
phase: 05-benchmark-and-claims-gate
plan: 04
subsystem: testing
tags: [scipy, scikit-learn, uv, calibration, ece, brier, student-t, confidence-interval, provable-contracts, fixtures]

# Dependency graph
requires:
  - phase: 01-encoder-conformance
    provides: "the pinned uv fixture environment (uv.lock already resolving scipy 1.18.0 / scikit-learn 1.9.0), the write_manifest SHA-256 + shasum self-verification pattern (D-13), and the contract-resident-frozen-constant pattern (D-14)"
  - phase: 03-faithful-two-stage-trainer-and-head
    provides: "the include_str!-pinned contract-parse convention (thresholds.rs) and the CR-03 lesson that serde_json renders a non-finite f64 as null"
provides:
  - "scripts/setfit_fixtures/claims_stats/ — a self-contained reference fixture set (t critical values, paired-t cases, top-label ECE cases, multiclass Brier cases) with its own manifest.sha256, byte-deterministic on re-run"
  - "aprender_core::calibration::expected_calibration_error_top_label — contract-bound multiclass top-label ECE on ordered labels"
  - "aprender_core::calibration::brier_score_multiclass — contract-bound UNNORMALISED multiclass Brier, codomain [0,2]"
  - "aprender_core::stats::hypothesis::T_CRIT_975_DF9 — frozen scipy.stats.t.ppf(0.975, 9), bit-asserted against the fixture"
  - "aprender_core::stats::hypothesis::{ttest_1samp_f64, ttest_rel_f64, paired_ci, paired_ci95_df9, mean_f64, sample_std_f64, min_max_f64}"
  - "AprenderError::ZeroVarianceDifferences — the typed refusal that keeps a non-finite f64 out of every published claims number"
  - "contracts/calibration-v1.yaml v1.1.0 — two added equations declaring the ECE and Brier formulas, domains and normalization"
  - "resolved assumptions A1 (t critical), A3 (sklearn has no ECE API), A4 (the one-vs-rest Brier identity), each with recorded in-env evidence"
affects: [05-05, 05-06, 05-09, 05-10, benchmark-row-schema, claims-contract, bench-report]

# Actuals — chars/4 over the realized diff (162,753 chars), same scale as an estimateTokens figure.
actuals:
  tokens: 40688
  tasks: 3
  commits: 6

# Tech tracking
tech-stack:
  added: []          # no new pins: scipy/scikit-learn already resolve in the committed uv.lock
  patterns:
    - "Per-directory fixture manifests: a new fixture family gets its OWN directory and OWN manifest.sha256, so it cannot re-baseline an existing corpus even by accident"
    - "Recorded RED values: every fixture case carries what a NAMED wrong implementation produces on that exact input, and the Rust test asserts the result is NOT that value"
    - "Typed degenerate cases: a fixture case with no computable answer records the typed error name and NO numbers, rather than a placeholder"
    - "f64-lossless fixture serialisation via json.dumps (the house jsonfmt writer's %.9g is f32-only and would truncate a frozen f64 constant)"

key-files:
  created:
    - scripts/setfit_fixtures/gen_claims_fixtures.py
    - scripts/setfit_fixtures/claims_stats/t_critical.json
    - scripts/setfit_fixtures/claims_stats/paired_t_cases.json
    - scripts/setfit_fixtures/claims_stats/ece_top_label_cases.json
    - scripts/setfit_fixtures/claims_stats/brier_multiclass_cases.json
    - scripts/setfit_fixtures/claims_stats/manifest.sha256
    - crates/aprender-core/src/stats/tests_claims_stats.rs
  modified:
    - crates/aprender-core/src/calibration.rs
    - crates/aprender-core/src/calibration_tests.rs
    - crates/aprender-core/src/stats/hypothesis.rs
    - crates/aprender-core/src/stats/mod.rs
    - crates/aprender-core/src/error.rs
    - crates/aprender-core/src/generated_contracts.rs
    - contracts/calibration-v1.yaml

key-decisions:
  - "The multiclass Brier is the UNNORMALISED Brier 1950 form; its codomain is [0,2] and at K=2 it is EXACTLY 2 x the binary brier_score. Declared once in the doc comment, the contract invariant and the test, so nobody re-derives it."
  - "The K=2 consistency test asserts the factor of 2 AND asserts the two are NOT equal, so a silent divide-by-K renormalisation turns it red in both directions."
  - "Zero-variance paired inputs are a typed AprenderError::ZeroVarianceDifferences from all three surfaces, never a non-finite f64 — serde_json would render one as null, making it a MISSING number in a published row rather than a visible failure."
  - "Two distinct zero-variance shapes are caught: exactly-constant (checked directly on the values) and numerically-constant (squared deviations underflow). An inferred `std == 0.0` check alone would let ten copies of 0.1 through with a ~1e-34 rounding variance."
  - "The f64 t-tail got its own Lanczos g=7 ln_gamma and continued fraction because the f32 path's six-term series is ~1e-10 accurate and the fixtures are asserted at 1e-9; a test proves the f64 mirror is strictly closer to scipy than the f32 path."
  - "T_CRIT_975_DF9 is asserted by BIT equality against the fixture, not a tolerance — a frozen constant that can drift within a tolerance is not frozen."
  - "paired_ci takes t_crit as a parameter and paired_ci95_df9 refuses n != 10, because the frozen constant is only correct at df = 9."
  - "generated_contracts.rs was extended with the exact bytes `pv codegen` emits for the two new equations, so a future full regeneration reproduces the file instead of fighting a hand edit."

patterns-established:
  - "Fixture generators abort on cross-check disagreement: every quantity is computed three ways (closed form, library, independent hand implementation) and a mismatch beyond 1e-12 is FATAL"
  - "Non-finite floats are unrepresentable in a fixture: json.dumps(allow_nan=False) plus a recursive assert_all_finite walk that names the offending JSON path"
  - "Bin-boundary margin guard: fixture inputs whose top confidence sits within 1e-2 of a bin edge are rejected, so an f32-vs-f64 parity test cannot become a coin flip"
  - "Guard tests scan non-comment source only, so prose can neither satisfy nor trip them"

requirements-completed: [EVAL-01, EVAL-04]

coverage:
  - id: D1
    description: "Reference fixture set generated in the pinned uv env (scipy 1.18.0 / scikit-learn 1.9.0) with its own SHA-256 manifest, self-verified, and byte-deterministic on re-run"
    requirement: EVAL-04
    verification:
      - kind: integration
        ref: "cd scripts/setfit_fixtures && uv run python gen_claims_fixtures.py && shasum -a 256 -c claims_stats/manifest.sha256"
        status: pass
      - kind: integration
        ref: "re-run the generator, then `git status --porcelain` — empty output proves byte-determinism (no RNG)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Assumptions A1 (t_{0.975,9} = 2.262157162798205), A3 (sklearn 1.9.0 exposes no calibration-error API) and A4 (the one-vs-rest Brier identity, max deviation 2.8e-17) resolved with recorded in-env evidence"
    requirement: EVAL-04
    verification:
      - kind: unit
        ref: "crates/aprender-core/src/stats/tests_claims_stats.rs#t_crit_975_df9_is_the_value_the_planning_documents_guessed"
        status: pass
      - kind: integration
        ref: "scripts/setfit_fixtures/claims_stats/ece_top_label_cases.json field `a3_probe`; brier_multiclass_cases.json field `a4_max_abs_deviation`"
        status: pass
    human_judgment: false
  - id: D3
    description: "Top-label ECE and multiclass Brier in aprender-core::calibration, contract-bound to calibration-v1, matching every fixture case within 1e-6 and distinguishable from named wrong implementations"
    requirement: EVAL-01
    verification:
      - kind: unit
        ref: "crates/aprender-core/src/calibration_tests.rs#top_label_ece_matches_pinned_env_fixtures"
        status: pass
      - kind: unit
        ref: "crates/aprender-core/src/calibration_tests.rs#brier_score_multiclass_matches_pinned_env_fixtures"
        status: pass
      - kind: unit
        ref: "cargo test -p aprender-core --lib calibration (71 passed)"
        status: pass
      - kind: integration
        ref: "target/release/pv validate contracts/calibration-v1.yaml (0 errors, 0 warnings)"
        status: pass
    human_judgment: false
  - id: D4
    description: "The multiclass Brier normalization is declared once and asserted at its true value: EXACTLY 2 x binary at K=2, and NOT equal to it"
    requirement: EVAL-01
    verification:
      - kind: unit
        ref: "crates/aprender-core/src/calibration_tests.rs#brier_score_multiclass_is_exactly_twice_binary_at_k2"
        status: pass
      - kind: unit
        ref: "crates/aprender-core/src/calibration_tests.rs#brier_score_multiclass_exceeds_one_for_confidently_wrong_k3"
        status: pass
    human_judgment: false
  - id: D5
    description: "f64 paired statistics (mean, (n-1) std, min/max, paired deltas, 95% CI) with the t critical value frozen and bit-verified against scipy; no RNG and no inverse CDF anywhere in the claims math"
    requirement: EVAL-04
    verification:
      - kind: unit
        ref: "crates/aprender-core/src/stats/tests_claims_stats.rs#t_crit_975_df9_equals_the_pinned_env_fixture_exactly"
        status: pass
      - kind: unit
        ref: "crates/aprender-core/src/stats/tests_claims_stats.rs#ttest_rel_f64_matches_scipy_for_every_finite_case"
        status: pass
      - kind: unit
        ref: "crates/aprender-core/src/stats/tests_claims_stats.rs#paired_ci95_matches_the_fixture_bounds"
        status: pass
      - kind: unit
        ref: "crates/aprender-core/src/stats/tests_claims_stats.rs#no_rng_enters_the_claims_statistics_path"
        status: pass
      - kind: unit
        ref: "cargo test -p aprender-core --lib stats:: (212 passed)"
        status: pass
    human_judgment: false
  - id: D6
    description: "Zero-variance paired input is a typed, finite-policy refusal from ttest_rel_f64, paired_ci and paired_ci95_df9 — never a serialised NaN/Infinity"
    requirement: EVAL-04
    verification:
      - kind: unit
        ref: "crates/aprender-core/src/stats/tests_claims_stats.rs#zero_variance_paired_input_is_a_typed_refusal"
        status: pass
      - kind: unit
        ref: "crates/aprender-core/src/stats/tests_claims_stats.rs#zero_variance_refusal_covers_the_all_zero_shape_specifically"
        status: pass
      - kind: unit
        ref: "crates/aprender-core/src/stats/tests_claims_stats.rs#numerically_constant_input_also_refuses"
        status: pass
      - kind: unit
        ref: "crates/aprender-core/src/stats/tests_claims_stats.rs#no_non_finite_f64_literal_escapes_through_this_module"
        status: pass
    human_judgment: false
  - id: D7
    description: "The Phase 1 fixture corpus, its manifest, generate_fixtures.py, uv.lock and pyproject.toml are byte-untouched"
    requirement: EVAL-04
    verification:
      - kind: integration
        ref: "git diff --stat 350b08575 HEAD -- crates/aprender-core/tests/fixtures/ crates/aprender-contrastive-data/tests/setfit_reference/ scripts/setfit_fixtures/generate_fixtures.py scripts/setfit_fixtures/uv.lock scripts/setfit_fixtures/pyproject.toml (empty)"
        status: pass
    human_judgment: false

# Metrics
duration: 62min
completed: 2026-08-17
status: complete
---

# Phase 5 Plan 04: Claims Numerics Substrate Summary

**Top-label ECE and unnormalised multiclass Brier bound to `calibration-v1` v1.1.0, plus closed-form f64 paired statistics with `T_CRIT_975_DF9` frozen from `scipy.stats.t.ppf(0.975, 9)` — all four metrics falsified against a hash-manifested fixture set generated in the pinned uv env, with zero-variance inputs refused by a typed error instead of a NaN.**

## Performance

- **Duration:** ~62 min
- **Started:** 2026-08-17T03:42Z
- **Completed:** 2026-08-17T04:44Z
- **Tasks:** 3 (two of them TDD, so 6 commits)
- **Files modified:** 15 (8 created, 7 modified)

## Accomplishments

- **The three open assumptions are now resolved facts, measured in the pinned environment rather than recalled.** A1: `scipy.stats.t.ppf(0.975, 9) = 2.262157162798205` — the planning documents' 2.2621571628 was right to every digit it recorded, and the env value is what got frozen. A3: sklearn 1.9.0 exposes **no** calibration-error API at all (`sklearn.metrics` has only `brier_score_loss` and `d2_brier_score`; `calibration_curve` is a binary reliability helper), so the reference ECE is an independent implementation, cross-checked twice. A4: the one-vs-rest Brier identity holds to a worst-case 2.8e-17 across all five cases.
- **The Brier normalization is stated once and pinned at its true value.** The 05-REVIEWS consensus finding was that the plan's original K=2 *equality* criterion is mathematically impossible. The shipped test asserts `multiclass == 2 x binary` **and** asserts the two are not equal, so a future silent divide-by-K fails in both directions; `sharp_wrong` (BS = 1.8479) stands in the fixtures as the permanent counterexample to any `[0,1]` bound.
- **The undefined statistic cannot become a missing number.** Zero-variance paired inputs return `AprenderError::ZeroVarianceDifferences` from all three surfaces. Two distinct shapes are caught — exactly-constant and numerically-constant (underflowed deviations) — and both are tested, because the obvious single check would have let ten copies of `0.1` through with a ~1e-34 rounding variance and returned a 1e17-scale statistic.
- **Every parity assertion is paired with a falsifying one.** Each fixture case records what a *named* wrong implementation produces on that exact input (population std, missing `/√n`, true-label confidence, unweighted bins, divide-by-K, true-class-only), and each test asserts the result is not that value. A test that only confirms the right answer passes for anything that lands nearby.
- **The claims math is provably closed-form.** No RNG symbol and no `f64::NAN`/`f64::INFINITY` literal appears in non-comment source (both acceptance greps return 0), no inverse CDF is implemented, and re-running the generator leaves `git status` empty — the fixtures are byte-deterministic.

## Task Commits

1. **Task 1: Reference fixture generator in the pinned uv env** — `e08e67627` (feat)
2. **Task 2: Multiclass top-label ECE + multiclass Brier** — `a98f8e124` (test, RED) → `ae78e5fa7` (feat, GREEN)
3. **Task 3: f64 paired statistics + frozen t critical constant** — `ca7bd700c` (test, RED) → `a3d766930` (feat, GREEN)

**Out-of-scope log:** `ef990b6dc` (docs: deferred-items.md)

Both TDD tasks show a real RED: `a98f8e124` fails with `E0425: cannot find function
expected_calibration_error_top_label`, `ca7bd700c` with 32 errors including
`E0599: no variant named ZeroVarianceDifferences`. No refactor commit was needed.

## Files Created/Modified

**Created**
- `scripts/setfit_fixtures/gen_claims_fixtures.py` — the reference generator: three-way cross-checks, per-path finiteness assertions, SHA-256 manifest + `shasum -c` self-verification, and a `FATAL` abort on any disagreement beyond 1e-12
- `scripts/setfit_fixtures/claims_stats/t_critical.json` — `t.ppf(0.975, df)` at full f64 repr for df ∈ {5, 9, 19}, with a `t.cdf` inverse round-trip check
- `scripts/setfit_fixtures/claims_stats/paired_t_cases.json` — 4 finite + 2 degenerate cases, n = 10 each
- `scripts/setfit_fixtures/claims_stats/ece_top_label_cases.json` — 5 K=3 cases including a saturated `conf == 1.0` case and the A3 probe result
- `scripts/setfit_fixtures/claims_stats/brier_multiclass_cases.json` — 5 K=3 cases with the A4 identity check
- `scripts/setfit_fixtures/claims_stats/manifest.sha256` — covers those 4 files, and only those
- `crates/aprender-core/src/stats/tests_claims_stats.rs` — 20 tests: fixture parity, the frozen constant, the typed degenerate case, the special functions, and the source guards

**Modified**
- `crates/aprender-core/src/calibration.rs` — `expected_calibration_error_top_label`, `brier_score_multiclass`, and the shared `multiclass_rows` / `top_label` helpers
- `crates/aprender-core/src/calibration_tests.rs` — 12 new tests (fixture parity, K=2 normalization, clamp, precondition refusals)
- `crates/aprender-core/src/stats/hypothesis.rs` — the f64 claims section plus the f64 Lanczos/continued-fraction/t-tail helpers
- `crates/aprender-core/src/stats/mod.rs` — re-exports for the new surface
- `crates/aprender-core/src/error.rs` — the `ZeroVarianceDifferences` variant and its `Display` arm
- `crates/aprender-core/src/generated_contracts.rs` — the four `pv codegen` macro blocks for the two new equations
- `contracts/calibration-v1.yaml` — v1.0.0 → v1.1.0, two added equations

## Decisions Made

- **Fixture serialisation does not use the house `jsonfmt` writer.** `jsonfmt` emits `%.9g` because every Phase 1 fixture records an f32; nine significant digits would truncate `2.262157162798205` in its tenth digit — silently corrupting the one constant this plan exists to freeze. `json.dumps` (shortest round-tripping f64 repr) is used instead, and the module docstring says why.
- **`generated_contracts.rs` was extended with genuine codegen bytes, not a hand edit.** `pv codegen` was run against a temp directory holding only the edited `calibration-v1.yaml`, and the four emitted macro blocks were spliced in verbatim. A future `pv codegen contracts/ -o …` therefore reproduces the file. Regenerating all 1767 contracts to add two equations was rejected as an unreviewable diff.
- **Structural preconditions are hard `assert!`, not `debug_assert!`.** The contract's declared preconditions (non-empty, finite) go through the `contract_pre_*!` macro exactly as the binary pair does. But a label out of range or a ragged matrix would index the wrong row and return a *plausible* number, and these metrics feed published claims — so those are unconditional.
- **`ln_gamma_f64` has no reflection branch.** Every call site passes `df/2`, `1/2` or their sum, all `≥ 0.5`. An unreachable branch is an untested branch, so the precondition is asserted instead of being silently handled.
- **The `WINDOWS.md` ledger entries were waived, not left open.** Both entries are plan-wording corrections where the shipped code is strictly stronger than the plan text, not defects; leaving them `open` would block `/gsd-ship` on non-issues. They are waived with explicit reasons and documented below. **Note for the orchestrator:** `.planning/WINDOWS.md` is a cross-phase file created inside this worktree; if a sibling wave-1 agent also created one, expect a merge conflict and reconcile by concatenating entries and renumbering.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] The plan's named RED mutation for ECE is a no-op**

- **Found during:** Task 2 (Multiclass top-label ECE + multiclass Brier)
- **Issue:** The plan's behavior block specifies that "a deliberately wrong implementation (bin by predicted-class probability instead of max) reproduces the recorded RED value". Those two quantities are the *same number* — the predicted class **is** the argmax, so `p[i][pred_i] == max_k p_ik` identically. That mutation can never turn a test red, so the falsification the criterion asks for would have been theatre.
- **Fix:** Implemented the mutation the wording is reaching for — confidence read from the **true label's** column instead of the maximum, which differs whenever the prediction is wrong — plus a second independent mutation (bin gaps averaged unweighted instead of by occupancy). Both are computed by the generator and recorded per case, and both are asserted against. The reasoning is written into the generator at the mutation's definition so the next reader does not re-derive it.
- **Files modified:** `scripts/setfit_fixtures/gen_claims_fixtures.py`, `crates/aprender-core/src/calibration_tests.rs`
- **Verification:** RED values differ from GREEN by a usable margin on the discriminating cases — `overconfident_sharp` 0.0117 vs 0.5835, `saturated_top_bin` 0.0219 vs 0.3531, `mixed_hard` (unweighted) 0.4104 vs 0.3631. Cases where a mutation degenerates to GREEN (single occupied bin, all-zero ECE) skip the not-equal half rather than weaken it.
- **Committed in:** `e08e67627` (generator) and `ae78e5fa7` (tests)

**2. [Rule 3 - Blocking] The `contract_pre_*!` macros for the new equations did not exist**

- **Found during:** Task 2
- **Issue:** The plan mandates the `contract_pre_*!` precondition convention, but those macros live in the 34,783-line auto-generated `generated_contracts.rs`, which carries a `DO NOT EDIT` banner and is produced by `pv codegen` over the whole `contracts/` tree. Adding two equations to the YAML does not create them.
- **Fix:** Ran `pv codegen` against a temp directory containing only the edited `calibration-v1.yaml` and spliced the four emitted macro blocks in verbatim, adjacent to the existing `calibration-v1` block. The added text is exactly what a full regeneration produces, so the file is not put at odds with its generator.
- **Files modified:** `crates/aprender-core/src/generated_contracts.rs`
- **Verification:** `cargo test -p aprender-core --lib calibration` compiles and passes 71/71; `pv validate` reports 0 errors.
- **Committed in:** `ae78e5fa7`

**3. [Rule 1 - Bug] Two `should_panic` expectations named the wrong message**

- **Found during:** Task 2 (first GREEN run: 69 passed, 2 failed)
- **Issue:** The empty-input tests expected a substring from the local structural `assert!`, but the **contract macro's** `input.len() > 0` precondition fires first. The tests were asserting the wrong mechanism.
- **Fix:** Changed the expected substring to `"precondition violated"`, which names the contract. The ordering is the point — the contract's declared precondition should be what refuses an empty input — so the test now asserts that rather than working around it.
- **Files modified:** `crates/aprender-core/src/calibration_tests.rs`
- **Verification:** 71/71 pass.
- **Committed in:** `ae78e5fa7`

**4. [Rule 2 - Missing Critical] The zero-variance guard needed a second shape**

- **Found during:** Task 3 (f64 paired statistics)
- **Issue:** The plan specifies the guard as "whenever the (n-1) sample std of the differences is exactly 0.0". That check is **not sufficient**: for a difference value with no exact binary representation (`0.1`, say), the computed variance of ten identical copies is a ~1e-34 rounding artefact rather than zero, so a genuinely constant input slips through and returns a meaningless ~1e17 statistic. The plan's own degenerate fixture happens to use exact binary fractions, so the gap would not have shown up in testing.
- **Fix:** `moments_or_zero_variance` checks bit-equality of the values **first** (exact, robust for any value), and keeps the `std == 0.0` check afterwards for the separate numerically-constant case where squared deviations underflow. Both shapes return the typed error; both are tested.
- **Files modified:** `crates/aprender-core/src/stats/hypothesis.rs`, `crates/aprender-core/src/stats/tests_claims_stats.rs`
- **Verification:** `numerically_constant_input_also_refuses` covers the underflow branch with real inputs (`[1e-200, 2e-200, 3e-200]`); `zero_variance_refusal_covers_the_all_zero_shape_specifically` covers 0/0.
- **Committed in:** `a3d766930`

**5. [Rule 1 - Bug] The RNG-guard criterion would have banned the existing API**

- **Found during:** Task 3
- **Issue:** The acceptance criterion lists `sample` as an RNG symbol. In this module `sample` is the statistical noun and is already the parameter name of the pre-existing `ttest_1samp(sample: &[f32], …)`. A literal guard would have failed on shipped, correct code while catching no actual RNG.
- **Fix:** The guard bans the symbols that would really indicate a random stream — `rand::`, `rand_chacha`, `thread_rng`, `SeedableRng`, `StdRng`, `gen_range`, `bootstrap`, `resample`, `shuffle`, `random(` — and the test carries an in-line note explaining the substitution. It scans non-comment source only, so prose can neither satisfy nor trip it.
- **Files modified:** `crates/aprender-core/src/stats/tests_claims_stats.rs`
- **Verification:** `no_rng_enters_the_claims_statistics_path` passes; the criterion's own grep returns 0.
- **Committed in:** `ca7bd700c`

**6. [Rule 2 - Missing Critical] The f32 t-tail is not accurate enough for the asserted band**

- **Found during:** Task 3
- **Issue:** The plan says to reuse the existing closed-form core "in f64". The existing `ln_gamma`/`incomplete_beta` are f32 with a six-term Numerical-Recipes series (~1e-10); the fixtures assert p-values at 1e-9, so a naive widening would have been under-determined.
- **Fix:** Added f64 `ln_gamma_f64` (Lanczos g=7, n=9, ~1e-15), `beta_continued_fraction_f64` (eps 3e-16, 300 iterations) and `t_distribution_pvalue_f64` — the **same closed forms**, not a different method. `ttest_rel_f64` still delegates to `ttest_1samp_f64`, so there is exactly one paired-statistic implementation (OPS-03), and a test asserts the two are bit-identical.
- **Files modified:** `crates/aprender-core/src/stats/hypothesis.rs`
- **Verification:** `ln_gamma_f64_matches_closed_form_values` pins the coefficients against `ln √π`, `ln 1`, `ln 24` and `ln Γ(4.5)`; `incomplete_beta_f64_satisfies_the_symmetry_identity` exercises **both** continued-fraction branches; `t_distribution_pvalue_f64_is_more_accurate_than_the_f32_path` measures the improvement rather than asserting it from principle.
- **Committed in:** `a3d766930`

---

**Total deviations:** 6 auto-fixed (3 bugs, 2 missing-critical, 1 blocking)
**Impact on plan:** No scope creep. Deviations 1, 4 and 5 correct acceptance criteria that were
literally unsatisfiable or vacuous; in each case the shipped guard is strictly stronger than the
written one. Deviations 2 and 6 are mechanism the plan assumed existed. Deviation 3 is a test
correction found by the first GREEN run.

## Issues Encountered

- **`pv` was not built in this worktree** (`.cargo/config.toml` is gitignored, so worktrees do not
  inherit the main checkout's target dir). Built `target/release/pv` from HEAD before using it,
  per the CLAUDE.md binary-pinning rule — a contract validated by a stale `pv` would be a
  confident answer about a schema that is not running.
- **`cargo check --workspace --all-targets` surfaces 12 pre-existing `E0063`s** in
  `aprender-serve`'s `driver_cpu` test target. Proven unrelated: `query_pre_attn_scalar` was added
  to `GGUFConfig` in `366f3c275` (2026-06-19), an ancestor of this plan's base, and the check
  output mentions `AprenderError` **zero** times — so the new error variant broke no downstream
  match. Logged in `deferred-items.md` (D-ITEM-05-01), not fixed. The check was run precisely to
  prove the variant was safe to add, and it did.

## OPS-03 Observation (not fixed here)

Two other f64 incomplete-beta implementations exist downstream —
`aprender-serve/src/bench/welch_t_test.rs::incomplete_beta_approx` and a private method in
`aprender-cbtop/src/profile_compare/comparator.rs`. Neither is reachable from `aprender-core`
(the dependency runs the other way) and neither is the claims path, so this plan could not
consume them. Worth a consolidation ticket if a later phase gives `aprender-core` a public
special-function surface; recording it here so the observation is not lost.

## User Setup Required

None. No new dependency pins — scipy 1.18.0 and scikit-learn 1.9.0 already resolve in the
committed `scripts/setfit_fixtures/uv.lock`, and `pyproject.toml` is byte-untouched. The
generator is a developer-workflow script and is not wired into CI.

## Next Phase Readiness

**Ready for the row/report plans (05-05, 05-06, 05-09, 05-10):**

- EVAL-01's calibration diagnostics have a contract-bound, fixture-verified multiclass pair
  taking ordered labels — call `expected_calibration_error_top_label(probs, n_classes, labels, 10)`
  and `brier_score_multiclass(probs, n_classes, labels)` with row-major probabilities.
- EVAL-04's uncertainty arithmetic is closed-form and exactly recomputable: `paired_ci95_df9`
  for the per-shot-level SetFit − LoRA intervals, `ttest_rel_f64` for the p-values that live in
  the machine-readable detail only (D-08 forbids them in claim language), and
  `mean_f64`/`sample_std_f64`/`min_max_f64` for D-05's headline aggregation.

**Two things the consuming plans must handle:**

1. **`ZeroVarianceDifferences` is a real, reachable outcome.** Ten seeds that all move by the same
   amount is not exotic. The row/report layer must render it as a visible refusal, not fall back
   to a blank cell — which is the failure mode the typed error exists to prevent.
2. **The multiclass Brier is on a `[0, 2]` scale.** Any report column, threshold or plot axis that
   assumes `[0, 1]` will be wrong. The contract invariant and the doc comment both say so.

**Not blocking, but adjacent:** the claims contract (D-15, plan 05-05) should name these core
functions as *the* implementation for rows, so `ClassifyEvalReport`'s unfixtured `ece`/`brier`
cannot drift into the claims path.

## Self-Check: PASSED

- **Created files present on disk:** 9/9 (`gen_claims_fixtures.py`, the four `claims_stats`
  fixtures, `manifest.sha256`, `tests_claims_stats.rs`, this SUMMARY, `deferred-items.md`).
- **Commits present in `git log`:** 6/6 (`e08e67627`, `a98f8e124`, `ae78e5fa7`, `ca7bd700c`,
  `a3d766930`, `ef990b6dc`).
- **Plan verification block:** `cargo test -p aprender-core --lib calibration` → 71 passed,
  exit 0. `cargo test -p aprender-core --lib stats::` → 212 passed, exit 0.
  `pv validate contracts/calibration-v1.yaml` → 0 errors, 0 warnings. Phase 1 fixture corpus,
  `generate_fixtures.py`, `uv.lock` and `pyproject.toml` → empty `git diff` against the base.
- **Working tree:** clean.

---
*Phase: 05-benchmark-and-claims-gate*
*Completed: 2026-08-17*
