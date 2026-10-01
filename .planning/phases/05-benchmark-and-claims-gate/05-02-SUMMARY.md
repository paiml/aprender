---
phase: 05-benchmark-and-claims-gate
plan: 02
subsystem: training
tags: [setfit, calibration-regime, thresholds, evidence-gate, refactor, mutation-control]
requires:
  - "Phase 3 D-10 regime mechanism (RegimeCoordinates::parse/covers, CALIBRATED_REGIMES)"
provides:
  - "RegimeThresholds { regime, classes, embedding_delta_floor } — one measured table per calibrated regime"
  - "Thresholds::table_for(regime_id) -> Option<&RegimeThresholds> — THE single membership+table lookup"
  - "Thresholds::is_calibrated reimplemented as table_for(..).is_some() (one source, cannot disagree)"
  - "Thresholds::calibrated_regimes() DERIVED from the tables, so the reported set cannot outrun the measured set"
  - "validate_evidence reads every epsilon, gating flag and floor from ONE resolved regime table"
  - "#[cfg(test)] serde field per_regime_thresholds — the parse capability 05-03 needs, landed before the data"
affects:
  - "05-03 (adds the production regime entry + its contract per_regime_thresholds block; must migrate the #[cfg(test)] sole-regime accessors)"
tech-stack:
  added: []
  patterns:
    - "measurement bound to its coordinates: no table is readable until table_for has resolved one"
    - "capability-before-data (Ph1 D-14 spirit): the per-regime parser lands now, so the plan adding the second regime adds data + assertions only"
    - "fail-loud on ambiguity: the regime-less accessors panic once a second regime exists rather than silently returning the first"
    - "induced-mutation control (Verification Discipline rule 4) rather than an assertion that the gate 'still works'"
key-files:
  created: []
  modified:
    - "crates/aprender-train/src/train/setfit/thresholds.rs"
    - "crates/aprender-train/src/train/setfit/tune.rs"
decisions:
  - "table_for keys on RegimeCoordinates::covers (component-wise), never string equality or prefix-family normalization — Gemini Finding 5's normalization proposal stays REJECTED per the plan's review_notes"
  - "Thresholds::of / embedding_delta_floor became #[cfg(test)] rather than fixture-delegating production API: production code can no longer read a threshold without naming a regime, which is the D-10(c) non-transfer guard made structural"
  - "The sole-regime accessors PANIC once a second regime is calibrated, deliberately, rather than returning the first table — returning the first is exactly how fixture epsilons would come to judge the production encoder"
  - "FIXTURE_REGIME names the fixture id once; CALIBRATED_REGIMES is &[FIXTURE_REGIME] and calibrated_regimes() is derived from the tables, pinned by calibrated_regimes_are_exactly_the_tables_that_exist"
  - "uncalibrated_id_resolves_no_table was added in Task 1 (unconditionally) rather than only as the Task 2 fallback — it asserts on the new lookup itself, and the mutation control was run regardless"
metrics:
  duration_seconds: 1720
  tasks_completed: 2
  files_changed: 2
  completed: 2026-08-16
actuals:
  tokens: 11000
  tasks: 2
  commits: 2
status: complete
---

# Phase 5 Plan 02: Regime-Keyed Thresholds Summary

`Thresholds` now holds one measured table PER calibrated regime instead of a single global table,
and the evidence gate resolves membership and its table in one `table_for` lookup — so D-03's
production epsilons can land as a second regime entry in 05-03 without any code path being able to
apply fixture-scale numbers to a 30522-row vocabulary. No threshold value, no contract byte and no
regime-count test literal changed.

## What Shipped

| Task | Deliverable | Commit |
| ---- | ----------- | ------ |
| 1 | `RegimeThresholds` + `Thresholds::table_for` + derived `calibrated_regimes()` + optional `per_regime_thresholds` contract parsing | `7e9b11ad9` |
| 2 | `validate_evidence` routed through the single regime lookup + induced-mutation control | `6d4828c43` |

## The Shape

Before, `Thresholds` carried `classes`, `embedding_delta_floor` and `calibrated_regimes` side by
side: membership was one question (`is_calibrated`) and the numbers were another (`of`,
`embedding_delta_floor`), and nothing in the type connected an epsilon to the regime it was
measured in. With one regime that reads as a harmless flattening; with two it is the bug — the
table read has to pick, and the only thing it can pick without being told the regime is "the
first".

After:

```rust
pub(crate) struct RegimeThresholds {
    regime: &'static str,                            // an element of CALIBRATED_REGIMES, not a second spelling
    classes: BTreeMap<&'static str, ClassThreshold>,
    embedding_delta_floor: f64,
}
pub(crate) struct Thresholds { regimes: Vec<RegimeThresholds> }

pub(crate) fn table_for(&self, regime_id: &str) -> Option<&RegimeThresholds> {
    let observed = RegimeCoordinates::parse(regime_id).ok()?;
    self.regimes.iter().find(|entry| parse_calibrated(entry.regime).covers(&observed))
}
```

`is_calibrated` is now `table_for(..).is_some()`, so the predicate and the selection are one
computation rather than two that can drift. `calibrated_regimes()` is derived from the tables, so
the list an `UncalibratedRegime` refusal shows a user is by construction the list of regimes that
actually carry measurements — `calibrated_regimes_are_exactly_the_tables_that_exist` holds it
against the declared `CALIBRATED_REGIMES` constant in both directions.

In `tune.rs`, the two-step is gone:

```rust
let Some(regime) = thresholds.table_for(&evidence.calibration_regime_id) else {
    return Err(SetFitTrainError::UncalibratedRegime { observed, calibrated });  // same shape as before
};
```

Every subsequent read — the gated filter, `worst_failing_gated_parameter`'s per-class epsilon, and
the run-level `embedding_delta_floor` — comes out of `regime`. There is no reachable state in which
a number is compared against a run it was not measured on, because obtaining a number requires the
`&RegimeThresholds` this lookup returns.

## A Decision Beyond the Plan's Letter (recorded, not silent)

The plan's file scope is `thresholds.rs` + `tune.rs`, but `Thresholds::of` and
`embedding_delta_floor` have call sites in `evidence.rs` and `mod.rs` test modules. Rather than
leave them as fixture-delegating production API — the exact regime-less read this plan exists to
eliminate — they are now `#[cfg(test)]`, resolved through a private `sole()` that **panics** once a
second regime is calibrated:

```
a regime-less threshold read is ambiguous across N calibrated regimes: resolve the table with
`table_for(regime_id)` and read the epsilon from the regime the run actually executed in
```

Two consequences, both intended:

1. Production code cannot read a threshold without naming a regime — a compile error, not a review
   convention.
2. **05-03 will hit this panic** the moment it adds the production regime entry, and must route the
   `evidence.rs` / `mod.rs` test call sites through `table_for` with the regime they mean. That is
   the work, surfaced loudly at the moment it becomes necessary, instead of those tests silently
   continuing to assert fixture epsilons against whichever table sorted first.

`CALIBRATED_REGIMES` and `is_calibrated` are now dead in a non-test build (both are intra-doc-linked
from always-compiled documentation, so they carry `#[cfg_attr(not(test), allow(dead_code))]` with
the reason stated in-file rather than being deleted or `#[cfg(test)]`-ed into a broken doc link).

## Verification Output

Task 1 — the plan's command, status captured directly into `rc`, never read through a pipe:

```
$ CARGO_INCREMENTAL=0 cargo test -p aprender-train --lib --features setfit thresholds
29 passed, 7932 filtered out
rc=0
```

Task 2 — the wider suite, before and after the induced mutation:

```
$ CARGO_INCREMENTAL=0 cargo test -p aprender-train --lib --features setfit setfit::
313 passed, 1 ignored, 7647 filtered out
rc=0
```

Non-test build and lint, to prove the `#[cfg(test)]`/`allow` decisions above are not hiding real
warnings:

```
$ CARGO_INCREMENTAL=0 cargo check  -p aprender-train --lib --features setfit   # rc=0, 0 warnings from aprender-train
$ CARGO_INCREMENTAL=0 cargo clippy -p aprender-train --lib --features setfit --all-targets  # rc=0, 0 findings in the two edited files
$ cargo fmt -p aprender-train -- --check                                       # 0 diffs in the two edited files
```

### The Induced-Mutation Control (evidence, not assertion)

Verification Discipline rule 4: the gate's proof does not transfer across a restructuring, so the
widening was re-induced against the NEW lookup. One mutation — `table_for` drops the `covers` check
and returns the first table for any parsable id:

```rust
let _observed = RegimeCoordinates::parse(regime_id).ok()?;
self.regimes.first()
```

**RED — 7 named tests failed** (306 passed; 7 failed), including the end-to-end gate negatives that
run through `validate_evidence` rather than only the unit-level membership checks:

| Test | What it caught |
| ---- | -------------- |
| `evidence::tests::negative_uncalibrated_regime_is_refused_before_any_comparison` | `validate_evidence` returned **`Ok(PassedEvidence)`** for a run stamped `minilm-full-h384-l6-a12-i1536-v30522@production` — the exact production-encoder acceptance this plan exists to keep impossible |
| `tests::regime_gate_an_unswept_seed_is_refused` | the gate accepted seed `1131161786113`, never swept |
| `tests::regime_gate_an_unmeasured_cell_is_refused` | the gate accepted cell `s8e1b3`, never measured |
| `tests::regime_id_is_calibrated_only_at_a_measured_seed_and_cell` | `FIXTURE_SEED was never swept` |
| `tests::regime_shots_component_never_names_one_class_of_a_non_uniform_selection` | ``smixed5-8e1b4` must not borrow a measured cell's epsilons` |
| `thresholds::tests::regime_membership_is_component_wise` | seed 2 accepted on a calibrated cell |
| `thresholds::tests::uncalibrated_id_resolves_no_table` | the new lookup itself resolved an uncalibrated id to a table |

The first three are the ones that matter most: they prove the refusal is still enforced at the GATE,
not merely at the predicate. The plan's fallback (add `uncalibrated_id_resolves_no_table` only if
nothing went red) was therefore not needed — the test was added in Task 1 anyway, because it asserts
directly on the newly created `table_for` surface.

**Mutation reverted; GREEN — 313 passed, 1 ignored**, byte-identical to the pre-mutation baseline
(same counts, same ignored test), so the control returned the suite to exactly where it started.

## Acceptance Criteria

| Criterion | Evidence |
| --------- | -------- |
| `table_for` + per-regime struct with regime id, classes, floor | `RegimeThresholds` at `thresholds.rs`; `Thresholds::table_for` |
| `is_calibrated` delegates to the same lookup | `fn is_calibrated(..) -> bool { self.table_for(regime_id).is_some() }` |
| `contracted_regimes.len(), 1` and full-list equality assertions unchanged | both survive verbatim in `thresholds_match_the_contract` (lines 554–555 and the `rust_regimes == contract_regimes` block) |
| Fixture values unedited | the 6 `ClassThreshold` literals and `2.7e-5` moved with the code; `grep -c` returns 7 matches (6 values + the module-header prose mention) and the `git diff` shows no value line altered |
| Contract byte-untouched | `git status --short` lists only `thresholds.rs` and `tune.rs` |
| Single resolution site in `validate_evidence` | one `table_for` call; the old `is_calibrated` + global-table read path is gone |
| `UncalibratedRegime` shape unchanged | same variant, same `observed` + `calibrated` fields, same rendering — and the downstream negatives above still bite |
| Full `--features setfit` lib suite exits 0 | `rc=0`, 313 passed |

## Deviations from Plan

### Auto-fixed / auto-decided

**1. [Rule 2 - correctness] `Thresholds::of` / `embedding_delta_floor` made `#[cfg(test)]` with a panicking `sole()`**
- **Found during:** Task 2, when the non-test build reported them dead after `validate_evidence`
  stopped calling them.
- **Issue:** Keeping them as production API that silently returns the fixture table is precisely the
  regime-less read the restructuring removes; keeping them as always-compiled dead code would have
  left that door open for the next caller.
- **Fix:** `#[cfg(test)]` + `sole()` panicking on ambiguity, documented in-file with the 05-03
  migration named.
- **Files:** `thresholds.rs`. **Commit:** `6d4828c43`.

**2. [Rule 3 - blocking] `calibrated_regimes()` return type `&'static [&'static str]` → `Vec<&'static str>`**
- Mechanically required by deriving the list from the per-regime tables (the plan's own
  "one source" requirement). All call sites compile unchanged: `tune.rs` iterates and clones, the
  contract test uses `.len()` and `.to_vec()`.

**3. [Rule 3 - blocking] `#[cfg_attr(not(test), allow(dead_code))]` on `CALIBRATED_REGIMES` and `is_calibrated`**
- Both are intra-doc-linked from always-compiled documentation (`mod.rs`'s `calibration_regime_id`
  names `Thresholds::is_calibrated`), so `#[cfg(test)]` would have produced a broken intra-doc link.
  The allow carries its justification in-file.

**4. Test added beyond the plan's conditional:** `uncalibrated_id_resolves_no_table` was added in
Task 1 rather than only as the Task 2 fallback (see decision above). It includes a positive control
so it cannot pass by returning `None` for everything.

### Out of scope, logged not fixed

- `crates/aprender-train/src/train/setfit/apr_reload.rs:331` carries a **pre-existing**
  `cargo fmt --check` diff (unrelated to this plan, present at the base commit `350b0857`). Not
  touched: the plan's verification pins the diff to `thresholds.rs` + `tune.rs`. Worth a one-line
  `cargo fmt` in a plan that already owns that file.

## Known Stubs

None. No placeholder values, no unwired data paths, no skipped tests.

## Threat Flags

None. The plan's register (T-05-02-01 tampering, T-05-02-02 elevation, T-05-02-03 repudiation) is
discharged rather than extended:

- **T-05-02-01** — `is_calibrated` IS `table_for`, so there is no second source to weaken; the
  induced mutation turning 7 named tests red is the evidence, not the claim.
- **T-05-02-02** — one resolution site; no threshold is reachable without a resolved regime, now
  enforced by the type system (`#[cfg(test)]` on the regime-less accessors).
- **T-05-02-03** — `calibrated_regimes()` is derived from the tables and pinned to
  `CALIBRATED_REGIMES` by a test that checks both directions.

No new network endpoint, auth path, file access or trust-boundary schema was introduced.

## Handoff to 05-03

1. Add the production regime as a SECOND `RegimeThresholds` in `Thresholds::frozen()`, its id as a
   second `CALIBRATED_REGIMES` entry, and its measured table as a `per_regime_thresholds` block in
   `contracts/setfit-train-lifecycle-v1.yaml` (the parser is already here and deserializes to empty
   today).
2. Flip `contracted_regimes.len(), 1` → `2` at the human checkpoint (D-04), together with the
   contract edit and its `pv diff`.
3. `Thresholds::sole()` will panic — migrate the `evidence.rs` / `mod.rs` test call sites of
   `Thresholds::of` / `embedding_delta_floor` to `table_for(regime_id)` naming the fixture regime.
   That panic is the deliberate signal, not an obstacle.
4. The per-regime loop in `thresholds_match_the_contract` is vacuous today and becomes live with the
   new block — it compares eps, scale_floor, sparse, gated and the floor per class, per regime.

## Self-Check: PASSED

- `crates/aprender-train/src/train/setfit/thresholds.rs` — FOUND (modified)
- `crates/aprender-train/src/train/setfit/tune.rs` — FOUND (modified)
- commit `7e9b11ad9` — FOUND in `git log`
- commit `6d4828c43` — FOUND in `git log`
