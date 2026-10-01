---
phase: 03-faithful-two-stage-trainer-and-head
plan: 09
subsystem: training
tags: [setfit, trn-07, selection-lock, canonical-test-access, validation-metric, sha256, contracts, typestate]

requires:
  - phase: 03-08
    provides: SetFitRun<ArtifactReloadedAndVerified> with artifact_hash(), selection_semantic_hash(), evidence().head() and the rest of the read-only reproducibility surface
  - phase: 03-07
    provides: head_input's encode-once path (eval mode, no_grad, detach, checked [B,H] split) and MultinomialLogisticRegression::predict_indices
  - phase: 03-06
    provides: contracts/setfit-train-lifecycle-v1.yaml, reduce.rs's fixed-order f64 reduction door (D-13)
  - phase: 02
    provides: PreparedDataset<Canonical>::validation() / validation_witness(), Split<Validation> vs Split<Test> vs Split<CompatibilityTest>, Selection::semantic_hash()/ledger_hash(), the SelectedId private-constructor token pattern and ledger.rs's canonical-bytes discipline
provides:
  - evaluate::evaluate_validation — the ONLY production path to a ValidationEvaluation; takes the verified run and the canonical dataset and computes the metric itself
  - evaluate::ValidationEvaluation — private fields, no public constructor, Serialize-not-Deserialize, committing {metric_kind, value, artifact_hash, validation_split_fingerprint, dataset_fingerprint, n_rows}
  - evaluate::ValidationMetricKind — a closed enum (Accuracy | MacroF1), no String variant
  - lock::SelectionLock — commits the WHOLE candidate history plus the rule, applies the rule, and binds every committed field to a SHA-256 over canonical bytes
  - lock::SelectionRule::MaxMetricLowestIndexTieBreak — deterministic in both halves (f64::total_cmp, ties to the lowest index)
  - lock::CanonicalTestToken / CanonicalTestAccess::grant / CanonicalTestGrant — minted from the run OBJECT, re-checked at access time, admitting only &Split<Test>
  - SetFitRun<ArtifactReloadedAndVerified>::create_selection_lock — the only public door to a lock, reading provenance off the run
  - head_input::encode_eval_rows — the shared encode-once door at a SHARED borrow
  - reduce::sum_f64_in_index_order / mean_f64_in_index_order
affects:
  - Phase 4 (OPS): the grant is the type a CLI/eval surface must hold before it may read the canonical test split
  - Phase 5 (EVAL): "any post-test-selected cell invalidates the report" hangs off the lock's StaleLock behaviour

tech-stack:
  added: []
  patterns:
    - "Evidence-by-computation: the trusted function takes the artifact AND the typed split and computes the number, so no constructor can accept one"
    - "Identity read off the OBJECT, never from bytes: mint_test_token takes &SetFitRun and calls artifact_hash() itself"
    - "Commit the whole decision, derive the winner: from_candidates takes the candidate set plus a rule and has no `chosen` parameter"
    - "Re-check at the point of ACCESS, not only at issue: a token is a value that can travel, so grant compares identity again"
    - "Float payloads travel as BITS inside canonical bytes — serde_json renders every non-finite f64 as `null`, which would collide NaN with infinity inside a digest"
    - "A guard that must skip an exception NAMES the exception: the float-parameter scan asserts the one #[cfg(test)] door exists and is gated, rather than passing because `pub(super) fn` happens not to match"
    - "Moving a method out of a counted block is only honest if the new home counts it: lock_run_side_door_is_a_single_read_only_method is the counterpart to verify's accessor-exhaustiveness guard"

key-files:
  created:
    - crates/aprender-train/src/train/setfit/evaluate.rs
    - crates/aprender-train/src/train/setfit/evaluate_tests.rs
    - crates/aprender-train/src/train/setfit/lock.rs
    - crates/aprender-train/src/train/setfit/lock_tests.rs
  modified:
    - crates/aprender-train/src/train/setfit/mod.rs
    - crates/aprender-train/src/train/setfit/head_input.rs
    - crates/aprender-train/src/train/setfit/reduce.rs
    - contracts/setfit-train-lifecycle-v1.yaml
    - contracts/aprender/binding.yaml

key-decisions:
  - "SelectionLock::from_candidates is pub(super), not pub: the selection semantic hash and the ledger hash are PROVENANCE and the only honest source is a run that executed, so the public door is create_selection_lock which reads both off the run"
  - "A SelectionCandidate's artifact hash is READ OUT of its evaluation rather than supplied beside it, so the two cannot disagree; config_hash stays caller-supplied and the contract says why (a sweep does not keep N verified runs alive)"
  - "The canonical bytes carry value_bits: u64 rather than a JSON float — a rendering binds the digest to a formatting library, and serde_json writes every non-finite f64 as `null`"
  - "The empty-class macro-F1 convention is 0.0 (sklearn's zero_division default), with both alternatives rejected in the contract: NaN destroys the selection rule's ordering, and excluding the class rewards declaring a class nobody predicts"
  - "create_selection_lock lives in lock.rs, not in mod.rs's reproducibility accessor block, because that block is counted by an exhaustiveness guard whose subject is read-only accessors — and lock_tests counts the new block so neither surface is unguarded"
  - "reduce.rs gained f64-input sum/mean rather than narrowing per-class ratios to the existing f32 door; the door's guarantee is ORDER, and reusing a signature by rounding the input is a silent precision loss"
  - "evaluate_validation takes BOTH the run and the dataset: the dataset is the caller's DECLARATION of which corpus this evaluation is about, and preferring one of the two silently would hide a mismatch"

patterns-established:
  - "Source-assertion tests over include_str! that scan PARAMETER LISTS rather than lines, after a line-wise scan was measured red for the wrong reason"
  - "Two-sided non-vacuity on the headline sequence: assert the two artifact hashes DIFFER before asserting that the lock notices"
  - "The rejected alternatives for a convention are written into the contract, not just the chosen one, so a later change is a visible contract edit"

requirements-completed: []

duration: ~1h20m
completed: 2026-08-11
---

# Phase 3 Plan 09: Selection Lock and Canonical Test Access Summary

TRN-07's two review-blocking gaps are closed: a validation metric is now computed by trusted code from a verified artifact and a typed canonical split, and canonical test access is gated on a hash-committing lock that records the entire candidate history and mints only against the run object itself.

## What Was Built

### Task 1 — `evaluate.rs`: the trusted evaluator (commit `117b0378b`)

`evaluate_validation(run: &SetFitRun<ArtifactReloadedAndVerified>, dataset: &PreparedDataset<Canonical>, metric: ValidationMetricKind)` predicts every row of `dataset.validation()` with the verified model, through the same encode-once path stage two used, and computes the metric itself.

**The shape it replaces** was `ValidationMetric::new(&Split<Validation>, name, value)`. It looked safe — you must hold a `Split<Validation>` to call it, and the compatibility profile cannot produce one — but possessing a split does not prove a number came out of it, and does not prove the number was computed with the artifact whose test access it goes on to unlock. There is now no public function in `evaluate.rs` that takes a float parameter at all.

`ValidationEvaluation` commits six facts: `{metric_kind, value, artifact_hash, validation_split_fingerprint, dataset_fingerprint, n_rows}`. The artifact hash is read off the run inside the evaluator; the two fingerprints come from `validation_witness()`, and a test asserts they DIFFER, so committing one of them twice cannot pass unnoticed.

**Metric definitions** (now contracted, so a change is a `pv diff`-visible edit):

- `accuracy = (sum_i [truth_i == pred_i]) / n`, the indicator vector summed through `reduce::sum_in_index_order`.
- `macro_f1 = (1/K) * sum_{c<K} f1_c` where `f1_c = 2*TP_c / (2*TP_c + FP_c + FN_c)`.
- **Empty-class convention: `f1_c = 0.0`** when `2*TP_c + FP_c + FN_c == 0` (both precision and recall are `0/0`). This matches `sklearn`'s `zero_division` default. Both alternatives are rejected for stated reasons — NaN would make the selection rule's ordering meaningless while every candidate still appeared to have a score, and excluding the class would reward a candidate for declaring a class it never predicts. `evaluate_macro_f1_absent_class_convention_is_not_vacuous` shows the convention BITES: the same rows score `2/3` under a 3-class map and `1.0` under a 2-class one.

### Task 2 — `lock.rs`: the lock, the token and the grant (commit `f65c0a6c4`)

**The lock record's committed field list**, in canonical (wire) order — `lock_hash` is deliberately absent from the bytes it digests:

| # | Field | Source |
|---|-------|--------|
| 1 | `schema_version: u32` | constant `1` |
| 2 | `rule: SelectionRule` | the caller's choice of rule, committed |
| 3 | `candidates: [ {config_hash, artifact_hash, evaluation} ]` | every candidate, in order; the evaluation as its private wire form with `value_bits` |
| 4 | `chosen_index: u64` | DERIVED by applying the rule |
| 5 | `dataset_fingerprint` | the first candidate's, after every other has been required to match |
| 6 | `validation_split_fingerprint` | likewise |
| 7 | `selection_semantic_hash` | read off the creating run |
| 8 | `ledger_hash` | read off the creating run's `Selection` |

`lock_hash = SHA-256(compact JSON of the above)` — the one canonical-bytes-then-SHA-256 convention, no map to iterate and no wall-clock value.

**`SelectionRule::MaxMetricLowestIndexTieBreak` semantics:** the highest `evaluation.value()` wins; comparison is `f64::total_cmp` (a total order over every bit pattern, including NaN, so the outcome cannot depend on arrival order even for values `partial_cmp` refuses to order); a candidate must be STRICTLY greater to displace the incumbent, so an exact tie keeps the LOWEST index.

**Four candidate-consistency rejections**, each naming the offending index and each with a named test: empty list; differing `metric_kind`; differing `validation_split_fingerprint`; differing `dataset_fingerprint`; and a duplicate `artifact_hash`, which also names where it first appeared.

**The StaleLock sequence evidence.** `lock_then_tune_then_test_invalidates_and_a_travelled_token_cannot_be_repaired` walks the attack in order:

1. Run A is the calibrated fixture pipeline through `SerdeJsonCodec`.
2. Run B is the **same configuration, same seed, same selection**, tuned with `TuningProbes::REVERSE_INTRA_BATCH_PULL`. Only the EXECUTION differs, so a lock keyed on configuration would still match and only the artifact hash moves. The test asserts `run_a.artifact_hash() != run_b.artifact_hash()` FIRST, so nothing below can hold vacuously.
3. `lock.mint_test_token(&run_a)` succeeds; `CanonicalTestAccess::grant(token, &run_a, run_a.dataset().test())` returns a grant admitting the real rows.
4. `lock.mint_test_token(&run_b)` returns `LockError::StaleLock { locked, observed }` and the rendered message contains BOTH hashes (asserted individually).
5. The token minted for run A, carried to run B, is refused at the door with `LockError::TokenModelMismatch { token_artifact_hash, model_artifact_hash }`, both hashes again asserted present in the rendering.

**Forgery.** `SelectionLock::forge_chosen_index_for_tests` is `#[cfg(test)] pub(super)` — the lock's analogue of 03-08's `EchoCodec`. Forging `chosen_index` makes `verify_integrity()` return `LockHashMismatch` with both digests, and because `mint_test_token` runs `verify_integrity` FIRST, the forged lock also cannot mint. `lock_hash_binds_every_committed_field` separately varies six single facts — config hash, artifact hash, metric value, candidate ORDER, selection semantic hash, ledger hash — and requires each to move the digest.

### Task 3 — contract growth (commit `c8267f4f0`)

Three equations appended (16 total in the contract, all bound), three proof obligations, three falsification tests (`FALSIFY-STL-014/015/016`), three binding entries:

| Equation | Bound to |
|----------|----------|
| `validation_evaluation_provenance` | `entrenar::train::setfit::evaluate::evaluate_validation` |
| `selection_lock_commitment` | `entrenar::train::setfit::lock::from_candidates` |
| `canonical_test_token_minting` | `entrenar::train::setfit::lock::mint_test_token` |

## Evidence

### The induced-red contract-audit observation

A gate that has only ever been seen passing is not evidence, so the audit's reach to the NEW bindings was measured rather than assumed.

| Step | Command | Result |
|------|---------|--------|
| Baseline | `rtk proxy make contract-audit-phase3` | **rc=0**, `Phase 3 binding audit: every equation is bound`, setfit contract `Total equations: 16 / Bound equations: 16 / Implemented: 16` |
| Induced | renamed the `validation_evaluation_provenance` binding's `equation:` key to `INDUCED_RED_PROBE_validation_evaluation_provenance` | **rc=2** (make's status for a failed recipe, not the recipe's own 1) with `[ERROR] BIND-001: Equation 'validation_evaluation_provenance' in setfit-train-lifecycle-v1.yaml has no binding entry` and `FAIL: unbound equations remain in: contracts/setfit-train-lifecycle-v1.yaml`; `Bound equations` fell **16 → 15** |
| Reverted | restored the key | `shasum -a 256 contracts/aprender/binding.yaml` = `15764990c6696ce60aabc88b469fa500df9c901f00a3b0cd513f0c2a9fa28801`, **byte-identical to the pre-probe digest** |
| Re-run | `rtk proxy make contract-audit-phase3` | **rc=0** |

Both audit runs were captured with `> file 2>&1; rc=$?`, never through a pipe (CLAUDE.md rule 1). The first attempt read the log through the `rtk` hook's rewritten form, which truncated the summarising `FAIL:` line — re-run through `rtk proxy` to read the raw output, which is where the `FAIL: unbound equations remain in:` line was actually observed.

### Test and gate results

| Check | Command | Result |
|-------|---------|--------|
| Task 1 | `CARGO_INCREMENTAL=0 cargo test -p aprender-train --lib --features setfit evaluate_` | rc=0, **45 passed** |
| Task 2 | `CARGO_INCREMENTAL=0 cargo test -p aprender-train --lib --features setfit lock_` | rc=0, **73 passed** |
| Reductions | `cargo test -p aprender-train --lib --features setfit reduce_` | rc=0, 15 passed |
| Contract | `pv validate contracts/setfit-train-lifecycle-v1.yaml` | rc=0, `0 error(s), 0 warning(s)` |
| Binding audit | `make contract-audit-phase3` | rc=0 (induced red observed, see above) |
| Clippy | `cargo clippy -p aprender-train --lib --features setfit --no-deps -- -D warnings` | rc=0 (see Deviations for why `--no-deps`) |
| Whole crate | `cargo test -p aprender-train --lib --features setfit -- --skip gpu:: --skip prune::snapshot_tests` | rc=0, **7625 passed**, 15 ignored |

### Acceptance-criterion greps

Run with `awk` rather than `grep`, because the `rtk` hook rewrites `grep` output into a summary form that cannot be counted:

| Criterion | Measured |
|-----------|----------|
| `pub fn new` inside `impl ValidationEvaluation` | **0** |
| `pub fn` lines taking `value: f64` in `evaluate.rs` | **0** |
| `Duration\|Instant\|SystemTime` in `evaluate.rs` | **0** |
| non-comment `--lib` occurrences in the contract | **32** |
| falsification test command lines in the contract | **32** (equal, as required) |

### RED observed before GREEN, both tasks

| Task | RED | GREEN |
|------|-----|-------|
| 1 | 10 failed / 35 passed — 7 panicking in `todo!("task 1 implementation")`, plus 3 source assertions | 45 passed |
| 2 | 13 failed / 59 passed — all panicking in `todo!("task 2 implementation")` | 73 passed |

The RED was executable, not merely a compile failure: the types and signatures were written first with `todo!()` bodies, so each test proved it reaches the code under test before that code existed.

## Deviations from Plan

### Auto-fixed and adjusted

**1. [Rule 3 — Blocking] `reduce.rs` and `head_input.rs` were modified, though Task 1's `<files>` named only `evaluate.rs` and `mod.rs`.**

- **Found during:** Task 1.
- **Issue:** two capabilities the evaluator needs did not exist. (a) The evaluator holds `&SetFitRun`, so it has only a SHARED borrow of the encoder, while `head_dataset` requires `&mut` to set eval mode — and `verify.rs` records the reason a second encode path is unacceptable ("using a second encode path here would compare the reloaded model against something the trainer never ran"). (b) The plan requires all summation through `reduce::*_in_index_order`, but macro-F1 averages per-class RATIOS already in f64, and the existing door takes `&[f32]`.
- **Fix:** `head_input::encode_eval_rows` — a `pub(crate)` door onto the same `encode_once` at a shared borrow, which OBSERVES the isolation witness (an encoder in training mode is a typed `HeadEncodeNotIsolated`, not a quietly irreproducible metric) rather than setting the mode. `reduce::sum_f64_in_index_order` / `mean_f64_in_index_order` — the same index-order guarantee over f64 inputs, with `reduce_f64_door_preserves_what_narrowing_to_f32_would_discard` asserting both values exactly so the two accumulators are distinguished rather than observed to be close.
- **Commit:** `117b0378b`.

**2. [Rule 2 — Missing critical functionality] `SelectionLock::from_candidates` is `pub(super)` with four parameters, not `pub` with two.**

- **Found during:** Task 2.
- **Issue:** the plan gives `from_candidates(candidates, rule)` while also requiring the lock to commit the selection semantic hash and the ledger hash, and requiring `create_selection_lock` to read those off the run. A public two-argument constructor cannot commit provenance it was never given; a public FOUR-argument one would let a caller supply provenance for a run that never executed — the same class of defect as a caller-supplied metric.
- **Fix:** `from_candidates` is `pub(super)` and takes the two provenance strings; `SetFitRun::create_selection_lock` is the only public door and reads both off the run. The acceptance criterion (no `chosen` parameter, winner derived by the rule) is asserted against the source and holds.
- **Commit:** `f65c0a6c4`.

**3. [Rule 1 — Regression I introduced] `create_selection_lock` was moved out of `mod.rs`'s reproducibility accessor block.**

- **Found during:** the full-suite run after Task 2.
- **Issue:** placing it in `impl SetFitRun<ArtifactReloadedAndVerified>` in `mod.rs` turned `verify_reproducibility_accessors_are_read_only_and_complete` RED — that guard counts the block's `pub fn`s and requires each to be exercised through a shared reference (`declared 12, exercised 11`). The guard is correct and the method does not belong there: it takes arguments, it can fail, and it constructs a new object.
- **Fix:** moved to its own `impl` block in `lock.rs`, with the reason written on the block. Moving a method out of a counted block is only honest if the new home counts it, so `lock_run_side_door_is_a_single_read_only_method` counts THIS block (exactly one `pub fn`, `&self` and not `&mut self`) and additionally calls it through a `&SetFitRun<…>` binding, which is the same compile-time half the `verify` guard uses.
- **Commit:** `f65c0a6c4`.

**4. [Rule 2] A `#[cfg(test)] pub(super) fn evaluation_for_tests` was added to `evaluate.rs`.**

- **Found during:** Task 2.
- **Issue:** the lock's rule, consistency and hash-binding tests need evaluations at CONTROLLED values, fingerprints and metric kinds. Producing those from real runs is impossible by construction — the evaluator computes the value, which is the point.
- **Fix:** `#[cfg(test)]` and nothing weaker, following `test_fixtures`'s precedent and 03-10's rejection of `#[doc(hidden)]` test-support doors. Critically, the float-parameter guard NAMES the exception rather than passing because `pub(super) fn` happens not to match its patterns: it asserts the door exists exactly once and that `#[cfg(test)]` sits immediately above it.
- **Commit:** `f65c0a6c4`.

**5. [Measured, not a deviation in scope] The float-parameter scan and the `pub fn` scan were each wrong once, and the guard now ships the correction.**

- A whole-line scan for `value: f64` was RED for the WRONG reason — it matched the type's own private `value: f64` FIELD. It also could not see a parameter on a wrapped signature. The scan now reads each function's PARAMETER LIST.
- Scanning for `"pub fn "` alone found **5 of the 9** public functions, because `pub const fn` does not contain that substring. Both corrections are written into the test and into the contract's invariant, per CLAUDE.md rule 7 (a guard pattern is re-checked by re-running the case table, not by re-reading the pattern).

**6. [Contract authoring] The `--lib` invariant broke once during authoring and was restored.**

- The criterion is that non-comment `--lib` occurrences equal the number of falsification test command lines. Writing a runnable command inside `FALSIFY-STL-016`'s prose pushed the count to 33 against 32. The prose now names the TEST instead of a command line. Recorded because it is a trap the next contract author will hit.

## Known-Red Baseline — measured, NOT caused by this plan

**`cargo clippy -p aprender-train --lib --features setfit -- -D warnings` cannot exit 0 on this host.** Measured through `rtk proxy` (the hook rewrites clippy's output into a summary that cannot be parsed for locations): **20 errors, all in `aprender-compute` (19) and `aprender-present-terminal` (1)**, zero in `aprender-train`. These are the arch-gated arm64 SIMD errors STATE.md already records as D-ITEM-02. Adding `--no-deps` lints only the selected package and exits **rc=0**; that is the form reported above, and it is the honest one for this criterion on this host.

**`cargo check -p aprender-train --no-default-features --features setfit` exits 101, and this plan is not the cause.** All 8 errors resolve to `presentar_terminal` and its type-inference fallout in `crates/aprender-train/src/monitor/tui/{app,dashboard}.rs` — the `tui` feature is default-on and `monitor/tui/*` is not gated on it, so removing default features removes the dependency but not its users. `git diff --name-only 8b49fa252 -- crates/aprender-train/src/monitor/` is EMPTY: this plan touched none of those files. Not fixed — out of scope per the executor's scope boundary; logged here so a verifier does not read the plan's `<verification>` line as this plan's regression.

**24 pre-existing test failures in the whole-crate run, controlled for.** `gpu::guard`, `gpu::ledger`, `gpu::wait` (21) and `prune::snapshot_tests` (3). Control: both reproduce with `cargo test -p aprender-train --lib gpu::` and `… prune::snapshot_tests` **without `--features setfit`**, so setfit-only changes cannot be their cause. The whole-crate figure reported above skips exactly these two module prefixes and is otherwise unfiltered.

## Environment Notes

- The `rtk` hook rewrites `grep`, `git status`, `cargo clippy` and `make` output into a summary form and writes THAT to a redirect target, so `> file 2>&1` does not capture raw output for hooked commands. Every count in this SUMMARY was taken with `awk` (unhooked) and every raw log with `rtk proxy`. STATE.md already records this for `git status --porcelain`; it is broader than that entry says.
- **`git commit` fails on this host:** `commit.gpgsign=true` with `gpg.program=/opt/homebrew/bin/gpg`, and no `gpg` binary exists anywhere on `PATH`. Every commit in the recent history is unsigned (`git log --format='%G?'` returns `N` for all of them), so the three commits here were made with `git -c commit.gpgsign=false commit`. Hooks ran normally; `--no-verify` was NOT used.
- `CARGO_INCREMENTAL=0` was exported for every cargo invocation, per the Phase 2 host mitigation. Free space at plan end: 102 GB.

## Requirements

TRN-07 is left **UNCHECKED** in REQUIREMENTS.md. The mechanics it names are delivered and defended, but this plan ships no user-reachable surface: nothing in `apr` or any CLI can yet create a lock or exercise a grant, and the phase's "a user can…" tier is 03-10's out-of-crate gate. Marking it complete here would put a claim in the traceability table that the shipped binary does not support — the same policy Phase 2 applied to DATA-01..06 and honoured to the end.

## Known Stubs

None. Every type introduced is fully wired: the evaluator runs a real model over real rows, the lock hashes a real record, and the grant hands out the real `&Split<Test>`.

## Threat Flags

None. Every file created or modified is in-process library code with no network endpoint, no filesystem access and no new deserialization surface reachable from untrusted input — `ValidationEvaluation` is deliberately Serialize-only, and the lock's wire types are constructed only from in-memory records.

## What the Next Plan Inherits

- `CanonicalTestGrant<'a>` is the type Phase 4/5 must hold before reading the canonical test split. It is not yet consumed by anything.
- `SelectionLock` has no persistence path. Its wire types derive `Deserialize` and `deny_unknown_fields` so a persisted lock can be READ, but there is deliberately no `From<Wire>` back into a usable evaluation, so a deserialized lock cannot be handed into minting. A plan that adds persistence must decide what a re-loaded lock is allowed to do.
- `SelectionRule` has one member. A second is a contract edit by construction.

## Self-Check: PASSED

All five claimed files exist on disk (`evaluate.rs` 16.8K, `evaluate_tests.rs` 17.0K,
`lock.rs` 32.0K, `lock_tests.rs` 26.2K, `03-09-SUMMARY.md` 24.9K) and all four claimed
commits resolve in `git log --all`: `117b0378b`, `f65c0a6c4`, `c8267f4f0`, `4959437cc`.
No commit in this plan deleted a tracked file (`git diff --diff-filter=D HEAD~1 HEAD`
was empty after each).
