---
phase: 03-faithful-two-stage-trainer-and-head
plan: 04
subsystem: classification
tags: [multinomial, softmax, logistic-regression, lbfgs, gradient, sklearn, contracts]
requires:
  - aprender::optim::LbfgsF64 (03-01)
provides:
  - aprender::classification::MultinomialLogisticRegression
  - aprender::classification::{HeadFitError, HeadInputError, HeadFitReport, Regularization}
  - contracts/multinomial-head-v1.yaml@1.0.0 (5 equations, 12 falsification tests)
  - scripts/gen_multinomial_sklearn_fixture.py (PEP 723 self-pinning reference generator)
  - make contract-audit-phase3 (BLOCKING, wired into tier3)
affects:
  - crates/aprender-core/src/classification/mod.rs (two added lines; the binary LogisticRegression is byte-untouched)
  - contracts/aprender/binding.yaml (5 appended entries)
  - Makefile ($(CONTRACTS), PHASE3_CONTRACTS, contract-audit-phase3, tier3, .PHONY)
tech-stack:
  added: []
  patterns:
    - fit in f64, store f32 — the public artifact is the f32 the caller gets, and the falsification compares THAT
    - mutation-by-derivation — the broken-gradient variants are computed FROM the shipped gradient, so the test cannot drift into testing a second implementation
    - contracted tolerance vs regression tripwire as two separate, separately-documented constants
    - the reference generator proves the objective convention before freezing any number
key-files:
  created:
    - crates/aprender-core/src/classification/multinomial.rs
    - crates/aprender-core/src/classification/tests_multinomial_contract.rs
    - scripts/gen_multinomial_sklearn_fixture.py
    - contracts/multinomial-head-v1.yaml
  modified:
    - crates/aprender-core/src/classification/mod.rs
    - contracts/aprender/binding.yaml
    - Makefile
decisions:
  - NonFiniteLogit is UNREACHABLE from finite f32 operands and that is a proof, not a gap — max |f32*f32| is 1.16e77 against an f64 range of 1.8e308
  - HeadFitReport's canonical JSON is stable, but a JSON ROUND TRIP is not bit-exact — serde_json's float parser is one ULP high on the head's own gradient norm
  - predict_proba returns f64; probabilities are derived quantities and downcasting them would inject error into the very sum-to-one property being contracted
  - the fixture design matrix uses EIGHTHS with modulus 25 — exact in f32, and period > n so no row repeats
  - REQUIREMENTS.md deliberately NOT touched (shared file, concurrent peer, outside files_modified)
metrics:
  duration: ~1h17m (first commit 18:17, last 19:17)
  tasks: 3
  files: 7
  completed: 2026-08-09
---

# Phase 03 Plan 04: Multinomial Head Summary

Built TRN-04's general `K >= 2` softmax head — fit in f64 on 03-01's `LbfgsF64`, stored as
f32, failing with typed errors on all sixteen enumerated bad inputs — and, because the
analytic gradient is the one numerical algorithm this phase writes by hand, validated it
pointwise against central differences instead of trusting it because it reached the right
optimum.

## Commits

| Commit | Type | What |
|--------|------|------|
| `a06e6f820` | test | RED behavior suite against a stubbed API — measured rc=101, 27 failed / 26 passed |
| `e474fd608` | feat | The head: objective, analytic gradient, validation gate, fit, predict |
| `aa94d8b1c` | test | Central-difference gradient suite, PEP 723 sklearn fixture generator, factor-2 falsification |
| `56e2e8cd1` | feat | `multinomial-head-v1.yaml`, five binding entries, blocking `contract-audit-phase3` |

Base SHA for every scoped diff: `66ef41495b9505327b7b377f72dcb1d549dcae58`.

## Headline Numbers

| Gate | Result |
|------|--------|
| `cargo test -p aprender-core --lib multinomial` | **rc=0, 62 passed** (18 pre-existing baseline + 44 new) |
| `cargo test -p aprender-core --lib multinomial_contract` | rc=0, 8 passed |
| `cargo test -p aprender-core --lib` (whole crate) | **rc=0, 14181 passed, 0 failed, 2 ignored** |
| `pv validate contracts/multinomial-head-v1.yaml` | rc=0 — "0 error(s), 0 warning(s), Contract is valid." |
| `pv audit --binding` | 5/5 equations bound, 8 obligations, 12 falsification tests |
| `make contract-validate` | rc=0, 45 contracts |
| `make contract-audit-phase3` (standalone) | rc=0 |
| `make contract-audit-phase2` (after my shared binding.yaml edit) | rc=0 — unchanged |
| `cargo fmt -p aprender-core -- --check` | rc=0 |
| `cargo clippy -p aprender-core --lib -- -D warnings` | rc=101 — **PRE-EXISTING, control-measured below** |

## The Gradient Suite Is the Point

The sklearn fixture compares a converged **optimum**, and a subtly wrong gradient can still
reach the right optimum on a given configuration while being wrong everywhere else. So the
derivative is checked pointwise, at a **non-zero** parameter point — zeros is degenerate,
because the penalty gradient `2*lambda*W` vanishes there for *any* factor, which is exactly
the bug the suite exists to catch.

Measured, `|g_analytic - g_central| / max(1, |g_analytic|)` with `h = 1e-6` in f64:

| Configuration | shipped gradient | gradient with `lambda*W` instead of `2*lambda*W` |
|---|---|---|
| K=2, lambda=0 | 1.591549392276903e-10 | **1.591549392276903e-10 — GREEN, bit-identical** |
| K=2, lambda=0.07 | 2.4040597290664323e-10 | **5.949999991876803e-2 — RED** |
| K=3, lambda=0 | 2.2829821366698866e-10 | **2.033087309616377e-10 — GREEN** |
| K=3, lambda=0.07 | 2.2829821366698866e-10 | **5.950000017391088e-2 — RED** |

against a 1e-6 band. Two things worth stating plainly:

1. **The shipped gradient's error sits at 1.6e-10..2.4e-10**, which is the *predicted*
   roundoff floor: central differencing carries `~eps/h = 2.2e-16/1e-6 ≈ 2.2e-10`. The
   measurement landed on the theory rather than merely inside the tolerance.
2. **The RED/GREEN asymmetry is the evidence.** A suite run only at `lambda = 0` would be
   green against the broken gradient and would have proved nothing about regularization.
   That asymmetry is now a permanent test
   (`falsify_multinomial_002_broken_penalty_factor_is_red_only_when_lambda_is_positive`),
   not a line in a log — 03-01's lesson applied.

A second mutation, penalizing the intercept, passes **every** W-block check
(worst 2.4e-10) and is caught **only** by the intercept block. That is its own test.

Both mutations are **derived from the shipped gradient** (`+= (factor - 2) * lambda * W`)
rather than reimplemented, so the file cannot drift into testing a second copy of the NLL.

## The sklearn Relation, Proven Twice

**D-04 as amended: `lambda = 1/(2*C*n)`**, `n` a ROW count. At n=24, C=1 that is
`0.020833333333333332`; the factor-2 RED value `1/(C*n)` is `0.041666666666666664`.

**First proof — before any Rust ran.** The generator evaluates *aprender's* analytic
gradient at *scikit-learn's own converged optimum*:

| lambda | max\|grad\| | verdict |
|---|---|---|
| `1/(2*C*n)` | **2.7934921420329217e-09** | stationary |
| `1/(C*n)` | **0.014778266913130668** | not stationary |

Five million to one. This matters because sklearn 1.9 reports `penalty = "deprecated"` in
`get_params()`, which reads alarmingly like "no penalty at all" — so the penalty being live
is **measured**, not assumed, and the generator aborts if it is not. (A first draft used a
weaker `||W||` norm-comparison heuristic and it failed on this data: 0.62 at C=1 vs 0.75 at
C=1e6, only 21% apart. The stationarity check is both stronger and unambiguous.)

**Second proof — end to end.** Fitting through the public API with
`SklearnEquivalentC { c: 1.0 }`:

| Comparison | measured | contracted tolerance |
|---|---|---|
| worst probability deviation | **7.73997399505788e-9** | 1e-4 |
| worst coefficient deviation | **1.4470913467512503e-8** | 1e-4 |
| worst mean-centred intercept deviation | within 1e-4 | 1e-4 |
| **wrong lambda `1/24`** | **2.8928600974116758e-2** | must exceed 1e-3 |

The wrong lambda is **3.7 million times worse** than the right one. The coefficient
deviation of 1.45e-8 is essentially the **f32 storage floor** (coefficients ~0.35, f32
relative precision ~6e-8): the f64 fit agrees with the reference to within the deliberate
downcast and nothing else.

aprender's own fit: `Converged, 34 iterations, grad norm 7.178489479677525e-11,
objective 1.030720308021935`.

### Two tolerances, on purpose

`PROBA_TOL`/`COEF_TOL` = 1e-4 were frozen **before** the first comparison ran (Ph1 D-14) and
were left alone afterwards — that is the contracted promise. But agreement is four orders of
magnitude better than the promise, so a real regression could degrade the fit ten-thousand-fold
and still sit inside the band unnoticed. Separate, separately-named
`*_MEASURED_TRIPWIRE` constants (1e-6) were therefore added as the alarm. The doc comments say
which is which and what widening each one means.

### SOLVER TOLERANCE — a plan-sanctioned tightening, recorded

The plan said to start at 1e-4 on probabilities and, if that failed, to **tighten the
generator rather than loosen the assertion**. Both sides now run to `tol = 1e-10`
(`max_iter = 5000`; sklearn converged in `n_iter_ = 15`, aprender in 34). The reason is not
that 1e-4 failed but that at 1e-4 the comparison would have been **measuring two stopping
rules** — aprender halts on the L2 norm of the full gradient, scipy on the max-abs projected
gradient — rather than the contracted objective. The assertion was never loosened.

## Fixture Environment (read from the RUNNING interpreter)

`uv run scripts/gen_multinomial_sklearn_fixture.py` — the PEP 723 header is the entire
environment spec, so there is no venv recipe to remember.

| | |
|---|---|
| uv | 0.9.5 |
| python | **3.13.0** (uv-managed; the ambient interpreter is 3.13.7 — the drift is visible precisely because versions are read at runtime, not echoed from the header) |
| scikit-learn | 1.9.0 (pinned) |
| numpy | 2.3.5 (pinned) |
| `n_iter_` | 15 of max_iter 5000 — asserted `< max_iter`, aborts otherwise (Pitfall 7) |

`get_params()`: `C=1.0, class_weight=null, dual=false, fit_intercept=true,
intercept_scaling=1, l1_ratio=0.0, max_iter=5000, n_jobs=null, penalty="deprecated",
random_state=null, solver="lbfgs", tol=1e-10, verbose=0, warm_start=false`.

`multi_class` is deliberately **not** passed (deprecated in 1.5, removed in 1.7).

## The 16 Input-Validation Cases

One test each, one **distinct** variant each. Validation runs in a single
`validate_fit_inputs` gate **before** any solve, so no partially validated state reaches the
optimizer.

| # | Case | Typed variant | Test |
|---|------|---------------|------|
| 1 | `K < 2` | `TooFewClasses { k }` | `invalid_k_less_than_two` |
| 2 | `ordered_labels.len() != K` | `LabelCountMismatch { labels, k }` | `invalid_ordered_label_count_mismatch` |
| 3 | empty string label | `EmptyLabel { index }` | `invalid_empty_ordered_label` |
| 4 | duplicate labels | `DuplicateLabel { first, second, label }` | `invalid_duplicate_ordered_labels` |
| 5 | `n == 0` | `EmptyDataset` | `invalid_empty_dataset` |
| 6 | `d == 0` | `ZeroFeatureDimension` | `invalid_zero_feature_dimension` |
| 7 | ragged rows | `RaggedRow { row, expected, found }` | `invalid_ragged_rows` |
| 8 | NaN feature | `NanFeature { row, col }` | `invalid_nan_feature` |
| 9 | +inf feature | `InfiniteFeature { row, col, value }` | `invalid_infinite_feature` |
| 10 | label index `>= K` | `LabelIndexOutOfRange { row, index, k }` | `invalid_label_index_out_of_range` |
| 11 | class with zero rows | `UnrepresentedClass { class }` | `invalid_unrepresented_class` |
| 12 | `lambda < 0` | `NegativeLambda { lambda }` | `invalid_negative_lambda` |
| 13 | NaN lambda | `NonFiniteLambda { lambda }` | `invalid_nan_lambda` |
| 14 | `C <= 0` | `NonPositiveC { c }` | `invalid_sklearn_c_non_positive` |
| 15 | non-finite `C` | `NonFiniteC { c }` | `invalid_sklearn_c_non_finite` |
| 16 | predict dim mismatch | `FeatureDimMismatch { row, expected, found }` | `invalid_predict_feature_dim_mismatch` |
| +1 | rows vs labels count (beyond the plan) | `RowCountMismatch { rows, class_indices }` | `invalid_row_count_mismatch` |

Finiteness is checked **before** the sign checks in both regularization arms, because
`NaN < 0.0` and `NaN <= 0.0` are both false and a NaN would otherwise slip past.

## The Exact-Tie Construction

The lowest-index tie-break needed an **exact** tie, not a near-tie. On a dataset where every
feature row appears once under every class, the analytic gradient at the zero initialization
is (K=2) exactly zero and (K=3) ~1e-17 — far below any usable tolerance. L-BFGS checks
convergence at the top of the loop, so it returns at iteration 0 with the solution still
**exactly** zeros. Every logit is then exactly `0.0` and every probability exactly `1/K`. The
test asserts the parameters really are bitwise zero before trusting the tie, for both K=2 and
K=3.

## Contract and Gate

`contracts/multinomial-head-v1.yaml` v1.0.0 — five equations
(`softmax_nll_objective`, `analytic_gradient`, `label_order_semantics`,
`convergence_error_mapping`, `logit_finiteness`), 8 proof obligations, **12 falsification
tests each carrying `--lib`** (12 `--lib` occurrences on non-comment lines == 12
`FALSIFY-MH-*` entries — the 02-06 lesson: a bare filter form emits `test result: ok` from a
suite that ran zero matching tests). Two entries are in-band **negative** controls that must
be RED. Kani harnesses are declared **NOT EXECUTED** with the runnable backing named, per the
Phase 2 convention.

### The audit gate was proven falsifiable, not merely observed passing

| Run | Result |
|-----|--------|
| `make contract-audit-phase3` standalone | **rc=0**, 5/5 bound |
| **INDUCED:** delete the `analytic_gradient` binding entry | **rc=2** — `[ERROR] BIND-001: Equation 'analytic_gradient' in multinomial-head-v1.yaml has no binding entry`, `FAIL: unbound equations remain in: contracts/multinomial-head-v1.yaml`, "Bound equations" 5 → 4 |
| **REVERTED** | `binding.yaml` sha256 `dfbce939bdc9a291cbd4a5af775be7074a76c12e4ff4c7c76b2b99e745b94023` — byte-identical to before |
| Re-run after revert | **rc=0** |

The induced status is **rc=2**, not rc=1 — that is `make`'s status for a failed recipe rather
than the recipe's own `exit 1`. Measured, and the Makefile comment says 2.

The loop reads the audit's status on its own line (`status=$$?`), deliberately **not** copying
the repo-wide `contract-audit`, whose loop body ends in `;`, never reads the status, and
therefore reports success while printing 132 BIND-001 errors.

`make -n tier3` confirms the target is reached (after `contract-audit-phase2`).

### Overlap-check verdict (performed BEFORE authoring)

Neither candidate covers any equation here, so nothing was bound into them:

- **`classifier-pipeline-v1.yaml`** — its `linear_probe` is an explicitly **binary sigmoid**
  probe over frozen CodeBERT embeddings (769 parameters, threshold 0.5), plus embedding
  extraction and MCC evaluation. No objective, no regularization, no derivative, no
  convergence semantics. It also belongs to the SSC v11 CodeBERT subsystem, so binding a
  general `aprender-core` capability there would misfile it.
- **`classification-finetune-v1.yaml`** — Poka-Yoke **constructor validators** for the LoRA
  fine-tune path (`logit_shape`, `label_bounds`, `classifier_weight_shape`, `softmax_sum`).
  `softmax_sum` is the closest neighbour but bounds `|sum(softmax(logits)) - 1|` for an
  already-computed logit vector, whereas `logit_finiteness` constrains how logits are
  **accumulated** and what happens when one is not finite. Adjacent, not overlapping.

The verdict and its reasoning are recorded inside the contract's own metadata as well, so it
does not live only here.

### 03-02 hand-off verdict (W-04) — measured, not inherited

Done first, before anything else in Task 3, and by this executor's own commands rather than by
trusting the briefing:

| Check | Result |
|-------|--------|
| resolved absolute path | `/Users/guy/Development/machine-learning/aprender/.claude/worktrees/agent-ac0c3564607b88dea/.planning/phases/03-faithful-two-stage-trainer-and-head/03-02-SUMMARY.md` |
| `test -f` | **rc=0** — file EXISTS, 28303 bytes (so this is NOT the STOP branch) |
| `grep 'CONTINGENCY FIRED'` | **rc=1** — heading **ABSENT** |
| `ls contracts/gemm-partition-determinism-v1.yaml` | rc=1 — does not exist |

**Verdict: 03-02's GEMM gate was GREEN. The contingency did not fire, the contract was
deliberately never authored, and there is nothing for this plan to wire.** 03-02's SUMMARY
says so in its own words at line 48. Recorded explicitly because "I did not see a heading" and
"I did not look" are indistinguishable afterwards. The verdict is also written into the
Makefile beside `PHASE3_CONTRACTS`.

## Deviations from Plan

### 1. [Rule 3 — blocking] Branch precondition satisfied by worktree isolation

Same as 03-01: this executor runs under Claude Code `isolation="worktree"` on
`worktree-agent-ac0c3564607b88dea`, not on `gsd/phase-3-two-stage-trainer`. The orchestrator
merges centrally. No `git checkout -b`, no `git switch`, no `git stash`, no `git clean`;
explicit-pathspec staging only.

### 2. [Design — a plan criterion that is provably unsatisfiable] `NonFiniteLogit` from a near-`f32::MAX` row

The plan requires "a fitted head fed a feature row whose f32 values are near `f32::MAX`
yields a typed `NonFiniteLogit{row, class}`". **That outcome is unreachable, and the reason is
the feature working correctly.** The largest magnitude an `f32 * f32` product can reach is
`f32::MAX^2 ≈ 1.16e77`, against an f64 range of `1.8e308`; you would need ~1e231 terms to
overflow the accumulator. Since the plan's own design mandates f64 accumulation, a finite f32
row **cannot** produce a non-finite logit.

Resolved by splitting the requirement into the two claims it was conflating, both tested:

- `predict_near_f32_max_row_accumulates_in_f64_where_f32_would_overflow` — builds a row of
  `±f32::MAX` sign-aligned with the dominant class, **witnesses that the same accumulation in
  f32 is `inf`** (asserting it, so the test cannot pass vacuously if the fitted weights are
  ever too small — it also asserts `sum |w| > 1` as a precondition), and asserts the head
  returns finite probabilities summing to 1. This is what review fix 3 was actually about.
- `predict_row_with_infinite_feature_yields_nonfinite_logit_error` — asserts the typed
  `NonFiniteLogit { row, class }` on the path where it *is* reachable.

The unreachability is written into the contract's `logit_finiteness` invariants as a measured
consequence, so a later reader does not file it as a coverage gap.

### 3. [Rule 1 — bug, in a dependency] `serde_json`'s f64 parse is one ULP high

Found by the report-stability test going red, and it is **not** what it first looked like. The
two fits were bitwise identical (`assert_eq!(ja, jb)` passed); the failing assertion was the
JSON **round trip**. Measured:

```text
v            = 2.1531120041346774e-5   bits 0x3ef693b74d831429
to_string(v) = "0.000021531120041346774"      (ryu, positional, exact)
from_str(..) = 2.1531120041346778e-5   bits 0x3ef693b74d83142a   (+1 ULP)
```

Verified against a correctly-rounded reference (CPython) that both values are distinct f64s
exactly 1 ULP apart and that the emitted decimal round-trips exactly under a correct parser —
so the **serializer** is fine and the **parser** is not. A second value in the same report
(`0.11346603265462092`) round-trips exactly, so this is value-dependent, not universal.

This matters beyond one test: Phase 3 owns a reload-and-compare boundary (D-07) and a
two-clean-runs reproducibility contract. **Hash the emitted bytes, never a reloaded report** —
otherwise a reloaded artifact compared bitwise against an in-memory one reports a spurious
mismatch. Encoded as `json_roundtrip_of_an_f64_is_not_bit_exact`, which pins both bit patterns
and tells the next reader what to do if a dependency bump makes it green.

The plan's actual requirement — "serializes to canonical JSON stable across two runs" — holds
and is asserted directly (`ja == jb`).

### 4. [Design] `predict_proba` returns `f64`, not `f32`

Probabilities are derived quantities computed in f64; downcasting them would inject rounding
into the very finiteness/sum-to-one property being contracted. The **artifact** (weights,
intercepts) is still f32 per D-03/APR-01, and the sklearn comparison deliberately runs through
the f32 store so it measures what a caller actually gets.

### 5. [Design] Task 1's tests are inline in `multinomial.rs`

The plan's `files_modified` lists exactly one implementation file for the head. A
`#[path]`-registered sibling test file would have been a second one. The contract-level suite
is separate, as the plan specifies.

### 6. [Rule 3 — blocking] Two fixture design-matrix defects, both found and both now guarded

- **Repeated rows.** The first fill used `mod 13` over 24 rows, silently repeating rows 0..10
  as rows 13..23 — the same feature vector carrying two different labels, a noisier and weaker
  reference than the fixture claims to be. Fixed to `mod 25` (coprime to the row coefficient
  7, and > n). The generator now **aborts** on duplicate rows.
- **Not f32-exact.** The divisor was 3, so values were thirds and inexact in binary. Since the
  head takes `f32` features, sklearn and aprender would have been fitting subtly different
  data and the comparison would have silently absorbed a conversion error it was never meant
  to test. Fixed to a power-of-two divisor (eighths), exact in both widths.

### 7. [Finding] `pv lint <single-file>` is vacuous

`pv lint contracts/multinomial-head-v1.yaml` returns rc=0 and **"PASS"** — while reporting
`Gate 1: validate ✓ (0 contracts, ...)`. It scans **zero** contracts. The directory form
`pv lint contracts/` reports 1720 contracts (rc=0, 0 errors, 982 pre-existing warnings). The
plan asked for the single-file form; it was run, and it proves nothing. The meaningful
single-file gate is `pv validate`, which was run and is rc=0 with 0 errors and 0 warnings.
Same failure class as the Phase 2 bare-filter lesson: a green gate that inspected nothing.

### 8. [Rule 3 — blocking] `pv validate` rejected an unquoted formula

`formula: ... argmin { k : p_k = max_c p_c }` is not valid YAML — `{` opens a flow mapping and
`k : p_k` is a mapping value inside it. `pv` said so precisely (`mapping values are not
allowed in this context at line 129 column 83`). Rewritten in prose and quoted. Another
instance of the CLAUDE.md rule to dogfood `pv` rather than hand-check YAML.

### 9. [Design] `pub use multinomial::*;` rather than a named list

The plan's "binary LogisticRegression untouched" probe excludes only a **single-line**
`pub use multinomial::`; an explicit five-name list wraps to three lines under rustfmt and
would have made the probe report 3. The glob matches the house style already in this file
(`pub use gaussian_nb::*;`, `pub use linear_svm::*;`). The new items were also positioned so
rustfmt does not reorder `mod sets;` — the whole diff to `classification/mod.rs` since the
base is **two added lines**, and the probe returns **0**.

### 10. [Cleanup] `weights_f64()` accessor removed; `intercepts_f64()` kept behind `#[cfg(test)]`

Only the intercepts are needed off the f32 path (the gauge's 1e-9 band is tighter than f32
resolution at O(1) magnitudes, so asserting it on the stored values would assert the
rounding). An unused accessor is dead code; it can be re-added by whoever needs it.

## Verification of the Plan's Own Criteria

| Criterion | Measured |
|---|---|
| `--lib multinomial` rc=0, >= 24 tests | rc=0, **62** |
| K=2 test named `k2`/`binary_boundary` | `k2_binary_boundary_fit_predict_proba_and_labels` |
| 16 validation cases, distinct variants | table above (17 variants) |
| `grep -c 'Result<(), String>' multinomial.rs` | **0** |
| log-sum-exp shift asserted at logits ~1000 | `log_sum_exp_is_finite_for_logits_near_1000` (asserts `exp(1000)` really does overflow first, so the test is not vacuous) |
| `NonFiniteLogit` asserted | yes — see deviation 2 for the split |
| intercept mean within 1e-9 of 0 | `fitted_intercept_mean_is_zero_within_1e_9`, K=2 and K=3 |
| binary LogisticRegression untouched probe | **0** (scoped to the base SHA) |
| `grep -c -E 'elapsed\|Duration' multinomial.rs` | **0** |
| contract contains `1/(2` | 7 occurrences |
| `analytic_gradient` names `2*lambda*W` + zero intercept | yes |
| `--lib` count == falsification-test count | **12 == 12** |
| two `#[contract]` annotations | **2** |

### The clippy criterion is unsatisfiable at this plan's own base commit

Measured two-sided rather than asserted: the two new files were moved aside, `mod.rs` restored
to base with `git checkout <BASE> -- <path>`, clippy run, then all three restored and
**sha256-verified byte-identical**.

| | BASE (`66ef41495`) | AFTER (this plan) |
|---|---|---|
| `cargo clippy -p aprender-core --lib --no-deps -- -D warnings` | rc=101 | rc=101 |
| aprender-core citations | 1 — `demo/reliable/performance.rs:126` | 1 — identical |
| citations naming a file this plan wrote | 0 | **0** |

The single error is an `unreachable_code` under `#[cfg(target_arch = "aarch64")]` — the same
aarch64-only class already logged as STATE.md D-ITEM-02, invisible on x86_64 CI, and exactly
what 03-01 measured. Without `--no-deps` the same command adds ~20 `aprender-compute`
citations, also untouched by this plan. With warnings *not* denied,
`cargo clippy -p aprender-core --lib --no-deps --all-targets` is **rc=0** and cites neither of
my files — the two findings it did raise against `multinomial.rs` (an unused binding, an
unused accessor) were both fixed before commit.

## Deferred Issues

- **`bashrs` is not installed on this host** (`which bashrs` → not found), so
  `bashrs make lint Makefile` could not be run against my Makefile edits. Structural validity
  is evidenced instead by `make -n tier3` parsing the whole file and resolving the new target,
  and by both audit targets running green. CI carries a bashrs job.
- **`cargo clippy -p aprender-core --lib -- -D warnings` is red at base on aarch64** — not
  caused by this plan, control-measured above. Same disposition and reasoning as 03-01: not
  written to `deferred-items.md`, because that file is outside this plan's `files_modified`
  and peer 03-05 is running concurrently in its own worktree, so creating it here would hand
  the orchestrator a merge conflict.
- **`REQUIREMENTS.md` was deliberately not updated.** TRN-04's deliverable is complete, but
  the file is outside this plan's `files_modified`, is shared with the concurrently running
  03-05, and no wave-1 executor touched it either (its last commits are from Phase 2).
  Marking TRN-04 complete is left to the orchestrator.

## Self-Check: PASSED

All 7 files exist (`multinomial.rs` 1876 lines, `tests_multinomial_contract.rs` 690,
`classification/mod.rs` 847, `gen_multinomial_sklearn_fixture.py` 252,
`multinomial-head-v1.yaml` 382, `binding.yaml` 1191, `Makefile` 1452). All 4 commits exist
(`a06e6f820`, `e474fd608`, `aa94d8b1c`, `56e2e8cd1`). **Zero deletions in every commit**
(`git diff --diff-filter=D` empty for each, and empty across the whole range).

One note on how this was measured, because the first attempt was wrong: a probe using
`git log --oneline --all | grep "^<hash>"` reported all four commits MISSING while the
deletion check resolved the same hashes fine. The probe was at fault — `git log --oneline` is
rewritten by this environment's tooling and its lines do not begin with a bare hash.
Re-measured with `git log --format='%h %s'`, all four are present. CLAUDE.md rule 1 in
miniature: when a result looks wrong, check how it was measured before believing it.
