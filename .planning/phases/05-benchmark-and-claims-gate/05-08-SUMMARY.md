---
phase: 05-benchmark-and-claims-gate
plan: 08
subsystem: testing
tags: [eval, per-row-predictions, f-avg, tweet-eval, calibration, ece, brier, mcc, confusion-matrix, label-order-evidence, ops-03, credential-gate]

# Dependency graph
requires:
  - phase: 05-benchmark-and-claims-gate
    plan: 04
    provides: "aprender_core::calibration::{expected_calibration_error_top_label, brier_score_multiclass} — the contract-bound, pinned-env-fixtured multiclass pair this assembly calls instead of ClassifyEvalReport's unfixtured ece/brier"
  - phase: 05-benchmark-and-claims-gate
    plan: 05
    provides: "QualityBlock's field list and the CALIBRATION_SPLIT constant — the shape this assembly fills"
  - phase: 04-apr-artifact-and-production-parity
    provides: "evaluate_validation_from_artifact (04-07), the ReloadedSetFitCredential reload door (04-16), the CanonicalTestGrant lock chain, and the in-crate APR-capable fixture estate"
provides:
  - "entrenar::train::setfit::apr_evaluate::{evaluate_rows_from_artifact, RowPredictions, EvaluatedSplit} — the ONE per-row evaluation door a benchmark cell calls"
  - "the extracted check_artifact_identity and predict_rows: one identity re-check and one classify loop, shared by both doors"
  - "entrenar::train::setfit::bench_metrics::{assemble_quality_block, BenchMetricsError, OFFICIAL_F_AVG_CLASSES, ECE_N_BINS, PINNED_TWEET_EVAL_REVISION}"
  - "MultiClassMetrics::from_predictions_with_min_classes — the declared-map-size constructor a zero-support class needs"
  - "the label-order EVIDENCE test: the ordered vector resolved from tweet-eval-stance-benchmark-v1.yaml's own class-label map at its pinned canonical_revision"
affects: [05-09, 05-10, bench-run-adapter, bench-report-renderer]

# Actuals — chars/4 over the realized diff (75,190 added chars), the same scale an estimateTokens figure uses.
actuals:
  tokens: 18797
  tasks: 2
  commits: 4

# Tech tracking
tech-stack:
  added: []          # no new dependencies: aprender, serde_yaml and the eval::classification module are already reachable from aprender-train
  patterns:
    - "Gate-by-variant-payload: the enum arm that names a restricted split CARRIES the credential that unlocks it, so the gate is a compile-time fact rather than a runtime check somebody can forget to write"
    - "Redundant-argument-as-check: a parameter the callee could have derived is accepted anyway and a disagreement is a typed refusal, converting redundancy into a guard"
    - "Falsification by contract-index mutation: flipping the metric's class-index constant and recording WHICH tests turn red, then reverting byte-identically (sha256 quoted)"
    - "Bin-centre fixture design: hand-computed calibration cases place every top confidence 0.05 from a bin edge, so an f32 rounding step cannot turn the assertion into a coin flip"

key-files:
  created:
    - crates/aprender-train/src/train/setfit/apr_evaluate_row_tests.rs
    - crates/aprender-train/src/train/setfit/bench_metrics.rs
    - crates/aprender-train/src/train/setfit/bench_metrics_tests.rs
  modified:
    - crates/aprender-train/src/train/setfit/apr_evaluate.rs
    - crates/aprender-train/src/train/setfit/mod.rs
    - crates/aprender-train/src/eval/classification/metrics.rs

key-decisions:
  - "The split parameter is an enum whose Test arm CARRIES the CanonicalTestGrant. A `split: &str` or an `include_test: bool` would have made the lock chain a runtime check; a caller who has not been through it cannot name this variant's contents."
  - "The eval loop was EXTRACTED, not copied: check_artifact_identity and predict_rows are single functions the scalar door now delegates to. A test asserts the classify-site count is exactly 1 and the identity-check count exactly 1, so a future second loop is red rather than merely discouraged."
  - "assemble_quality_block takes ordered_labels even though both RowPredictions carry it, and refuses a disagreement with either. The redundancy is the point: a declared vector that is not the evidence's own is exactly the state in which every per-class number is published under another class's name."
  - "Label identity is Rust's `==` on `str` — exact UTF-8 byte equality. Case folding or trimming would silently accept a relabelled head, so `Against` and `against ` are each their own refusal, tested."
  - "MultiClassMetrics::from_predictions was NOT sufficient. It infers K from the observed indices, so a class with zero support AND zero predictions vanishes from every per-class vector and f1_average_for_classes([1,2]) returns None — reported as `class 2 not present` for a class that merely did not occur. The declared map size is now passed in."
  - "The confusion matrix is the shipped ConfusionMatrix, not a hand-rolled tally. The matrix a reader inspects and the per-class numbers beside it come from the same constructor, so they cannot describe different class counts."
  - "row_predictions_for_tests is #[cfg(test)] pub(super), on evaluate.rs's evaluation_for_tests precedent. Hand-computed falsification cases are impossible to produce from a real artifact — the door computes the predictions — and a shipped constructor of that shape would be the caller-asserted evidence RowPredictions exists to refuse."

patterns-established:
  - "A new plan's tests go in a NEW sibling test file when the acceptance criteria require the prior plan's test file to stay byte-untouched — an edited assertion is no longer independent evidence about the code it was written for"
  - "Every hand-computed metric carries its derivation in the test comment, so a future convention change has to argue with arithmetic rather than re-record a baseline"

requirements-completed: [EVAL-01]

coverage:
  - id: D1
    description: "A per-row-predictions evaluator door exists BESIDE evaluate_validation_from_artifact, takes the SAME ReloadedSetFitCredential, re-checks artifact-vs-dataset identity the same way, and is the ONE evaluation door (OPS-03)"
    requirement: EVAL-01
    verification:
      - kind: unit
        ref: "apr_evaluate_row_tests.rs#apr_evaluate_rows_takes_the_same_credential_type_as_the_scalar_door"
        status: pass
      - kind: unit
        ref: "apr_evaluate_row_tests.rs#apr_evaluate_rows_shares_one_classify_loop_with_the_scalar_door (classify sites == 1, identity-check sites == 1)"
        status: pass
      - kind: unit
        ref: "apr_evaluate_row_tests.rs#apr_evaluate_rows_refuses_a_dataset_that_is_not_the_artifacts_corpus (asserts the two doors' renderings are byte-equal)"
        status: pass
      - kind: unit
        ref: "apr_evaluate_row_tests.rs#apr_evaluate_rows_accuracy_equals_the_scalar_doors_value"
        status: pass
      - kind: integration
        ref: "cargo test -p aprender-train --lib --features setfit apr_evaluate -> rc=0, 22 passed"
        status: pass
    human_judgment: false
  - id: D2
    description: "EVAL-01 assembly produces F_avg via the contract-bound f1_average_for_classes(&[1,2]), macro-F1, per-class metrics, MCC, confusion matrix, validation-only top-label ECE and multiclass Brier, all bound to the explicit ordered labels"
    requirement: EVAL-01
    verification:
      - kind: unit
        ref: "bench_metrics_tests.rs#bench_metrics_per_class_values_are_the_hand_computed_counts"
        status: pass
      - kind: unit
        ref: "bench_metrics_tests.rs#bench_metrics_f_avg_is_the_official_two_class_average_and_not_the_macro (0.7333333333333334 vs 0.6555555555555556, with a non-vacuity assert that they differ)"
        status: pass
      - kind: unit
        ref: "bench_metrics_tests.rs#bench_metrics_mcc_is_the_hand_computed_coefficient (12/sqrt(528))"
        status: pass
      - kind: unit
        ref: "bench_metrics_tests.rs#bench_metrics_confusion_matrix_is_true_by_predicted_row_major"
        status: pass
      - kind: unit
        ref: "bench_metrics_tests.rs#bench_metrics_calibration_is_the_hand_computed_validation_diagnostics (ECE 0.475 from the bin table, Brier 0.57 from the row sums)"
        status: pass
      - kind: integration
        ref: "cargo test -p aprender-train --lib --features setfit bench_metrics -> rc=0, 14 passed"
        status: pass
    human_judgment: false
  - id: D3
    description: "Calibration diagnostics come from the VALIDATION split only (D-07); the QualityBlock records calibration_split = validation structurally"
    requirement: EVAL-01
    verification:
      - kind: unit
        ref: "bench_metrics_tests.rs#bench_metrics_calibration_split_is_validation_and_test_probabilities_cannot_reach_it (BOTH directions: test rows in the validation position AND validation rows in the test position are typed refusals)"
        status: pass
      - kind: integration
        ref: "the signature itself — assemble_quality_block(test_rows, validation_rows, ordered_labels) with a per-parameter split-tag check; there is no argument order and no boolean that reaches the other outcome"
        status: pass
    human_judgment: false
  - id: D4
    description: "The ordered label vector [none, against, favor] is EVIDENCE: pinned against the TweetEval contract's own ClassLabel order at its pinned canonical_revision, so f1_average_for_classes(&[1,2]) provably selects (against, favor)"
    requirement: EVAL-01
    verification:
      - kind: unit
        ref: "bench_metrics_tests.rs#bench_metrics_label_order_is_evidence_from_the_pinned_dataset_revision (resolves dataset.labels from tweet-eval-stance-benchmark-v1.yaml, asserts indices are exactly 0/1/2, the vector is [none, against, favor], [1]==against, [2]==favor, canonical_revision == 4fbd22cd..., and the official_f_avg formula string)"
        status: pass
      - kind: integration
        ref: "induced RED: OFFICIAL_F_AVG_CLASSES [1,2] -> [0,2] gives rc=101, 12 passed / 2 failed — the arithmetic test (f_avg 0.5833 vs 0.7333) AND the evidence test both go red; reverted byte-identical (sha256 9c7d07bb61acbf50d232a4a1be0a28ffe588b1018c11d1d68fe9f36c7290a169)"
        status: pass
    human_judgment: false
  - id: D5
    description: "A zero-support class yields a DEFINED per-class F1 under the shipped zero-division convention (recorded literally), and a zero-row split is a typed refusal"
    requirement: EVAL-01
    verification:
      - kind: unit
        ref: "bench_metrics_tests.rs#bench_metrics_a_zero_support_class_yields_the_shipped_zero_division_literal (asserts the literal 0.0 for precision, recall and F1, that the vector is still length 3, and that no NaN reaches f_avg or macro_f1)"
        status: pass
      - kind: unit
        ref: "bench_metrics_tests.rs#bench_metrics_a_zero_row_split_is_a_typed_refusal (both positions)"
        status: pass
      - kind: unit
        ref: "bench_metrics_tests.rs#bench_metrics_a_class_index_outside_the_label_map_is_a_typed_error (a two-label map cannot answer F_avg over [1,2] — the None becomes ClassIndexOutsideLabelMap, never a silent 0)"
        status: pass
    human_judgment: false
  - id: D6
    description: "Label identity is exact UTF-8 byte equality — no case folding, no normalization, no trimming"
    requirement: EVAL-01
    verification:
      - kind: unit
        ref: "bench_metrics_tests.rs#bench_metrics_label_identity_is_exact_byte_equality (`Against` and `against ` are each refused)"
        status: pass
      - kind: unit
        ref: "bench_metrics_tests.rs#bench_metrics_label_ordering_changes_attribution_in_both_directions"
        status: pass
    human_judgment: false
  - id: D7
    description: "Import audit: the RNG/bootstrap-capable ClassifyEvalReport is not reachable from the assembly (D-06, T-05-08-03)"
    requirement: EVAL-01
    verification:
      - kind: unit
        ref: "bench_metrics_tests.rs#bench_metrics_does_not_import_the_resampling_evaluator (comment-filtered ban on ClassifyEvalReport/bootstrap/resample/rand::/thread_rng, plus a positive requirement that all five shipped surfaces appear)"
        status: pass
      - kind: integration
        ref: "grep -v '^[[:space:]]*//' crates/aprender-train/src/train/setfit/bench_metrics.rs | grep -c ClassifyEvalReport -> 0"
        status: pass
    human_judgment: false

# Metrics
duration: 71min
completed: 2026-08-17
status: complete
---

# Phase 5 Plan 08: Per-Row Evaluator Door and EVAL-01 Metric Assembly Summary

**`evaluate_rows_from_artifact` opens the ONE per-row evaluation door benchmark cells call — same `ReloadedSetFitCredential`, same identity re-check, a shared single classify loop with the scalar door, and a test-split arm whose enum variant CARRIES the `CanonicalTestGrant` so the lock chain is a compile-time gate — and `assemble_quality_block` fills the row's `QualityBlock` from that evidence using only shipped, fixture-verified surfaces, with calibration structurally validation-only and the `[1, 2]` class selection converted from an assumption into evidence pinned to the TweetEval contract's own `ClassLabel` order at its pinned revision.**

## Performance

- **Duration:** ~71 min
- **Tasks:** 2 (both TDD, so 4 commits)
- **Files:** 6 (3 created, 3 modified); 75,190 added characters

## Accomplishments

- **There is still exactly ONE evaluation door, and that is now asserted rather than intended.** The scalar door's body was EXTRACTED into `check_artifact_identity` and `predict_rows`; the per-row door calls the same two functions. `apr_evaluate_rows_shares_one_classify_loop_with_the_scalar_door` asserts the classify-site count is exactly 1 and the identity-check count exactly 1, so a future second loop turns the suite red rather than merely being discouraged in a comment. The corpus-mismatch test goes further and asserts the two doors' rendered refusals are **byte-equal** — a second wording would be a second policy.

- **The test split is gated at the type level.** `EvaluatedSplit::Test` carries a `&CanonicalTestGrant`. A caller who has not been through lock → token → grant cannot name the variant's contents, so `bench run` cannot reach the test rows by passing a string or flipping a boolean. The arm additionally re-checks the grant's artifact hash against the credential's, because a grant is a value that can be moved into a struct and used against whatever credential is in scope three functions later — a `TestGrantArtifactMismatch`, typed.

- **D-07 is enforced by the signature, not by discipline.** `assemble_quality_block` takes the two splits as separate parameters and checks each against the split tag its `RowPredictions` carries. The test asserts BOTH directions: test rows in the validation position and validation rows in the test position are each a `WrongSplit` naming the parameter. `calibration_split` is then a recorded fact rather than a promise.

- **Gemini's rejected finding was answered with a measurement, not an argument.** The plan's review notes reject changing `f1_average_for_classes(&[1,2])` to `&[0,2]` and require the residual — turning the label-order assumption into evidence. That test now resolves the ordered vector from `tweet-eval-stance-benchmark-v1.yaml`'s own `dataset.labels` map, asserts the indices are exactly 0/1/2, that the vector is `["none","against","favor"]`, that `[1] == "against"` and `[2] == "favor"`, and pins `canonical_revision`. Applying the proposed "fix" was then run as a real mutation: `[1,2] -> [0,2]` gives rc=101 with 2 failures — the arithmetic test (`f_avg = 0.5833` against the official `0.7333`) and the evidence test — and the revert is byte-identical (sha256 `9c7d07bb...`). The indices are unchanged.

- **Every published number is hand-checkable.** The six-row test case and four-row validation case carry their full derivation in the test comments: the confusion matrix, TP/FP/FN per class, `F_avg = (0.8 + 2/3)/2`, `macro_F1 = (0.5 + 0.8 + 2/3)/3`, `MCC = 12/sqrt(528)`, the ECE bin table summing to `0.475`, and the four Brier row sums giving `0.57`. Every validation top confidence sits 0.05 from a bin edge, so an f32 rounding step cannot turn the ECE assertion into a coin flip.

## Task Commits

1. **Task 1: `evaluate_rows_from_artifact` — the per-row predictions door** — `fb9a83b04` (test, RED) → `38d5efee7` (feat, GREEN)
2. **Task 2: `QualityBlock` assembly (`bench_metrics.rs`)** — `e15bd9d45` (test, RED) → `459ff4f35` (feat, GREEN)

Both REDs are real: `fb9a83b04` fails with rc=101 and 19 errors including `E0425: cannot find function evaluate_rows_from_artifact`; `e15bd9d45` with rc=101 and 32 errors including `E0432: unresolved import row_predictions_for_tests`. No refactor commit was needed.

## Files Created/Modified

**Created**

- `crates/aprender-train/src/train/setfit/apr_evaluate_row_tests.rs` — 10 tests. A NEW sibling rather than an edit to `apr_evaluate_tests.rs`, whose byte-untouchedness the plan requires and which is verified below.
- `crates/aprender-train/src/train/setfit/bench_metrics.rs` — the assembly, the five typed refusals, and the contract-resident `OFFICIAL_F_AVG_CLASSES` / `ECE_N_BINS` / `PINNED_TWEET_EVAL_REVISION`.
- `crates/aprender-train/src/train/setfit/bench_metrics_tests.rs` — 14 tests.

**Modified**

- `crates/aprender-train/src/train/setfit/apr_evaluate.rs` — `RowPredictions`, `EvaluatedSplit`, `evaluate_rows_from_artifact`, the two extracted helpers, two new error variants, and the `#[cfg(test)] pub(super)` fixture constructor.
- `crates/aprender-train/src/train/setfit/mod.rs` — `pub mod bench_metrics;` with its siting rationale (rustfmt's `reorder_modules` also moved the adjacent `bench_row` declaration into alphabetical position; no other change).
- `crates/aprender-train/src/eval/classification/metrics.rs` — `MultiClassMetrics::from_predictions_with_min_classes`, purely additive.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `MultiClassMetrics::from_predictions` silently drops a zero-support trailing class**

- **Found during:** Task 2, reading `ConfusionMatrix::from_predictions` before writing the zero-support test.
- **Issue:** The plan names `MultiClassMetrics::from_predictions` as the surface for per-class metrics and the confusion matrix. That constructor delegates to `ConfusionMatrix::from_predictions`, which INFERS the class count as `max(observed) + 1`. On a split where `favor` has neither a true row nor a prediction, `K` collapses to 2 — so `f1` is length 2, `f1_average_for_classes(&f1, &[1, 2])` returns `None`, and the assembly would report "class 2 is outside a 2-label map" for a class that merely did not occur. The plan's own must-have — "a class with zero support yields a DEFINED per-class F1 under the shipped zero-division convention" — is unsatisfiable through the named constructor. On a 64-shot TweetEval cell with 46 `favor` rows this is unlikely; on an 8-shot cell it is not exotic.
- **Fix:** Added `MultiClassMetrics::from_predictions_with_min_classes`, delegating to the already-shipped `ConfusionMatrix::from_predictions_with_min_classes` plus `from_confusion_matrix` — one implementation, one extra parameter. The assembly passes the DECLARED label-map size. The acceptance grep for `MultiClassMetrics::from_predictions` still matches (it is the same constructor family, not a substitute surface), and the confusion matrix uses the same `min_classes` constructor so the matrix and the per-class vectors cannot describe different class counts.
- **Files modified:** `crates/aprender-train/src/eval/classification/metrics.rs`, `crates/aprender-train/src/train/setfit/bench_metrics.rs`
- **Verification:** `bench_metrics_a_zero_support_class_yields_the_shipped_zero_division_literal` asserts the vectors are still length 3, that `precision[2] == recall[2] == f1[2] == 0.0` (the literal, not a description of it), that the confusion matrix keeps its third row and column, and that no NaN reaches `f_avg` or `macro_f1`.
- **Committed in:** `459ff4f35`

**2. [Rule 3 - Blocking] `RowPredictions` had no constructor a hand-computed test could use**

- **Found during:** Task 2.
- **Issue:** The plan requires a hand-computed six-row toy case, a two-sided ordering test, a zero-support case and a zero-row case. All four need `RowPredictions` values at controlled contents, and the type deliberately has no public constructor — producing one from a real artifact is impossible, because the door computes the predictions.
- **Fix:** `row_predictions_for_tests`, `#[cfg(test)]` and `pub(super)`, on the precedent `evaluate.rs::evaluation_for_tests` sets in the same module tree for exactly the same reason. Its doc comment names the exception and says why, so the "no caller-supplied evidence" claim is qualified in the source rather than quietly false.
- **Files modified:** `crates/aprender-train/src/train/setfit/apr_evaluate.rs`
- **Verification:** the four tests exist and pass; the constructor is unreachable from any non-test build.
- **Committed in:** `459ff4f35`

**3. [Rule 2 - Missing Critical] The plan's `split` parameter had no shape that gated the test split**

- **Found during:** Task 1.
- **Issue:** The plan specifies `evaluate_rows_from_artifact(credential, dataset, split)` covering "validation and (token-gated where the existing API demands) test splits", but does not say what `split` is. A `&str` or a `bool` would have made the lock chain a runtime check inside the door — and a door that can be asked for test rows without holding a grant is the substitution TRN-07 exists to block, one refactor away from being written.
- **Fix:** `EvaluatedSplit` is an enum whose `Test` arm carries `&CanonicalTestGrant`. The gate is then a compile-time fact. The arm also re-checks the grant's artifact hash against the credential's (`TestGrantArtifactMismatch`), because grant-time checking happened at one instant and a grant is a movable value.
- **Files modified:** `crates/aprender-train/src/train/setfit/apr_evaluate.rs`, `apr_evaluate_row_tests.rs`
- **Verification:** `apr_evaluate_rows_measures_the_test_split_only_through_a_grant` reaches the test rows through lock → token → grant and asserts they are the canonical test rows;
  `apr_evaluate_rows_refuses_a_grant_that_belongs_to_another_artifact` asserts the re-check mechanism is present and typed.
- **Committed in:** `38d5efee7`

**4. [Rule 2 - Missing Critical] Nothing tied a classify response's probability arity to the head's label count**

- **Found during:** Task 1.
- **Issue:** `ClassifyResult::new` ties `logits` to `probabilities`, and the envelope ties the result count to the request count — but nothing ties either to the number of ordered labels the head carries. A K-vector of the wrong width would be flattened row-major into the calibration functions, mixing classes across row boundaries and producing a plausible number.
- **Fix:** `predict_rows` refuses a row whose probability vector length is not the label map's, as a `ClassifyFailed` naming both widths.
- **Files modified:** `crates/aprender-train/src/train/setfit/apr_evaluate.rs`
- **Verification:** `apr_evaluate_rows_returns_one_prediction_and_one_probability_vector_per_row` asserts the arity, the mass and the argmax agreement on the real fixture artifact.
- **Committed in:** `38d5efee7`

**5. [Rule 2 - Missing Critical] `ordered_labels` as a third parameter needed to be a CHECK, not a trust**

- **Found during:** Task 2.
- **Issue:** The plan's signature passes `ordered_labels` alongside two `RowPredictions` that already carry it. Taken at face value that is redundancy the caller can get wrong: a declared vector disagreeing with the evidence's own is precisely the state in which every per-class number is published under another class's name.
- **Fix:** `check_evidence` compares the declared vector to BOTH evidence vectors by exact byte equality and returns `LabelOrderMismatch` naming the parameter and both vectors. The redundancy becomes a guard.
- **Files modified:** `crates/aprender-train/src/train/setfit/bench_metrics.rs`
- **Verification:** `bench_metrics_label_ordering_changes_attribution_in_both_directions` and `bench_metrics_label_identity_is_exact_byte_equality` (a case-folded and a whitespace-padded label are each refused).
- **Committed in:** `459ff4f35`

**6. [Rule 1 - Bug] The confusion matrix was briefly hand-rolled**

- **Found during:** Task 2, self-review before running.
- **Issue:** The first draft tallied the `[K][K]` counts in a local loop. That is a second confusion-matrix implementation beside the one `MultiClassMetrics` reduces — the RESEARCH document's "Don't Hand-Roll" table names exactly this — and the two could drift so that the matrix a reader inspects and the per-class numbers beside it describe different tallies.
- **Fix:** `confusion_counts` now calls `ConfusionMatrix::from_predictions_with_min_classes` and only widens `usize` to the `u64` the row schema declares.
- **Files modified:** `crates/aprender-train/src/train/setfit/bench_metrics.rs`
- **Verification:** `bench_metrics_confusion_matrix_is_true_by_predicted_row_major` pins the exact `[[1,1,0],[0,2,0],[1,0,1]]`, which is also the matrix the hand-derived per-class numbers were computed from.
- **Committed in:** `459ff4f35`

---

**Total deviations:** 6 auto-fixed (2 bugs, 3 missing-critical, 1 blocking). No Rule 4 (architectural) situations arose. Deviation 1 corrects a named surface that could not satisfy the plan's own must-have; 3, 4 and 5 are guards the plan's threat model already implies (T-05-08-01, T-05-08-02, T-05-08-04); 2 is mechanism the plan assumed existed; 6 is a self-caught regression against the RESEARCH anti-pattern list.

## Import Audit (required by the plan's action block)

`ClassifyEvalReport` is **not** imported, referenced or reachable from `bench_metrics.rs`.

- `grep -v '^[[:space:]]*//' crates/aprender-train/src/train/setfit/bench_metrics.rs | grep -c ClassifyEvalReport` → **0**. The name appears exactly once in the file, in the module header explaining why it is excluded — which the comment filter removes, so the prose can neither satisfy nor trip the gate.
- `bench_metrics_does_not_import_the_resampling_evaluator` extends the ban to `bootstrap`, `resample`, `rand::` and `thread_rng` over the same comment-filtered source, and asserts positively that all five intended surfaces (`f1_average_for_classes`, `MultiClassMetrics::from_predictions`, `matthews_corrcoef`, `expected_calibration_error_top_label`, `brier_score_multiclass`) ARE present — a ban with no positive half passes for an empty file.
- The only calibration imports are `aprender::calibration::{brier_score_multiclass, expected_calibration_error_top_label}`, i.e. 05-04's contract-bound pair.

## Issues Encountered

- **`rustfmt` on `setfit/mod.rs` cascades into the whole module tree.** Formatting `mod.rs` reformatted `apr_reload.rs`, `bench_row.rs` and `bench_row_tests.rs`, none of which this plan may touch (05-05's summary records that `cargo fmt --check` is not clean at this branch's base, so those diffs were pre-existing drift, not this plan's). All three were reverted by path with `git checkout -- <file>`; the remaining `mod.rs` diff is this plan's `pub mod bench_metrics;` plus `reorder_modules` moving the adjacent `bench_row` declaration into alphabetical position. Recording it because the failure mode is silent: the files would have been committed under this plan's name carrying another plan's formatting.

- **The f64 → f32 narrowing into the calibration functions is real and recorded.** `ClassifyResult::probabilities()` is `&[f64]`; `expected_calibration_error_top_label` and `brier_score_multiclass` take `&[f32]`. The conversion happens once, at the call site, with a comment saying why it is acceptable: 05-04's fixtures assert those f32 implementations at 1e-6, and both numbers are published as diagnostics to about four significant figures, so the narrowing is inside the band the metric is trusted to. It is NOT hidden behind a helper.

- **`MultiClassMetrics::from_predictions_with_min_classes` is a change outside this plan's declared `files_modified`.** It is additive (28 lines, no existing behaviour altered) and is the minimum mechanism the plan's zero-support must-have requires. Recorded here rather than deferred because deferring it would have meant shipping an assembly that reports a headline score as "not computable" whenever a class happens not to occur.

## Known Stubs

None. Every symbol this plan ships is fully implemented and exercised against either the real in-crate APR fixture artifact (the door) or hand-derived cases (the assembly). Nothing is wired to a placeholder, and no `QualityBlock` field is left unfilled.

## Threat Flags

None. No new network endpoint, auth path, file-access pattern or trust-boundary schema was introduced. The plan's four mitigate-disposition threats are each answered by a shipped, tested mechanism:

| Threat ID | Mechanism | Test |
|---|---|---|
| T-05-08-01 (evaluator bypass) | `&ReloadedSetFitCredential` on both doors; shared identity check and classify loop | `apr_evaluate_rows_takes_the_same_credential_type_as_the_scalar_door`, `..._shares_one_classify_loop_...` |
| T-05-08-02 (label attribution) | explicit `ordered_labels` checked by exact byte equality against both evidence vectors; `F_avg` pinned to `[1, 2]` with revision-pinned evidence | `bench_metrics_label_ordering_...`, `..._label_identity_is_exact_byte_equality`, `..._label_order_is_evidence_...` |
| T-05-08-03 (RNG in calibration) | 05-04's contract-bound functions only; `ClassifyEvalReport` grep-asserted absent | `bench_metrics_does_not_import_the_resampling_evaluator` |
| T-05-08-04 (split leakage) | two separate split parameters, each split-tag checked; `calibration_split` recorded | `bench_metrics_calibration_split_is_validation_and_test_probabilities_cannot_reach_it` |

An additional surface the plan did not enumerate — a `CanonicalTestGrant` travelling to a different credential — is closed by `TestGrantArtifactMismatch` (deviation 3).

## User Setup Required

None. No new dependencies; `aprender`, `serde_yaml` and `crate::eval::classification` are already reachable from `aprender-train`, and the whole module sits behind the existing `setfit` feature.

## Next Phase Readiness

**Ready for 05-09 (bench-run adapter) and 05-10 (the report gate):**

- `evaluate_rows_from_artifact(&credential, &dataset, EvaluatedSplit::Validation)` and `EvaluatedSplit::Test(&grant)` are the ONLY calls a cell needs to make. `RowPredictions` carries the artifact hash and the split tag, so a row is stamped without re-deriving either.
- `assemble_quality_block(&test_rows, &validation_rows, &ordered_labels)` returns a complete `QualityBlock` with every `*_bits` sibling filled.

**Four things the consuming plans must handle:**

1. **The test split needs a grant, and getting one is the lock chain.** A cell must build the validation evaluation, create the selection lock, mint the token and take the grant before it can measure the test split. That ordering is not a convenience — it is what the enum makes non-optional.
2. **`BenchMetricsError` is reachable and must render as a visible refusal**, not a blank cell — the same warning 05-04 gave for `ZeroVarianceDifferences`. An 8-shot cell in which one class never occurs in the model's predictions AND never occurs in the test truth is the realistic path to `ClassIndexOutsideLabelMap`.
3. **The multiclass Brier is on a `[0, 2]` scale** (05-04's and 05-05's warning, inherited unchanged by `QualityBlock.brier_multiclass_validation`).
4. **`binding.yaml` still lists this plan's equations as `pending`.** 05-05 registered ten entries; nothing here flipped one, because a status flipped on belief rather than a resolvable-symbol check is 04-10's recorded mistake. Whichever plan next touches that registry with such a check in hand owns `bench_row_schema`'s and `official_f_avg`'s status.

## Self-Check: PASSED

- **Created files present on disk:** 3/3 — `apr_evaluate_row_tests.rs`, `bench_metrics.rs`, `bench_metrics_tests.rs`.
- **Commits present in `git log`:** 4/4 — `fb9a83b04`, `38d5efee7`, `e15bd9d45`, `459ff4f35`.
- **Plan verification block, every status captured directly and never through a pipe:**
  - `cargo test -p aprender-train --lib --features setfit apr_evaluate` → rc=0, **22 passed** (12 pre-existing + 10 new).
  - `cargo test -p aprender-train --lib --features setfit bench_metrics` → rc=0, **14 passed**.
  - `cargo test -p aprender-train --lib --features setfit bench_row` → rc=0, **24 passed** (05-05's suite unaffected).
  - `cargo clippy -p aprender-train --lib --features setfit` → rc=0; 29 warnings, **all** in `aprender-compute` (pre-existing, `D-ITEM-05-05-B`), **0** attributable to any file this plan touches.
  - `rustfmt --check` on all five touched files → rc=0.
  - `grep -v '^[[:space:]]*//' bench_metrics.rs | grep -c ClassifyEvalReport` → **0**.
  - `git diff --stat f02a8aadf HEAD -- crates/aprender-train/src/train/setfit/apr_evaluate_tests.rs` → empty (byte-untouched, as the acceptance criterion requires).
  - Induced RED / byte-identical revert recorded for the `F_avg` class-index mutation.
- **Working tree:** clean apart from this SUMMARY.

---
*Phase: 05-benchmark-and-claims-gate*
*Completed: 2026-08-17*
