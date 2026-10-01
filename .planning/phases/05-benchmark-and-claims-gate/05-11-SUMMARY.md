---
phase: 05-benchmark-and-claims-gate
plan: 11
subsystem: benchmark-claims-gate
status: complete
tags: [claims-contract, expectation-set, fail-closed-gate, descope, tombstone, EVAL-02, EVAL-04, EVAL-05]

requires:
  - "05-06 (selection-manifest door, run_classify_core)"
  - "05-09 (scripts/run_bench_cells.sh, the bench run adapter)"
  - "05-10 (verify_run, the six doctored negatives, the report renderer)"
provides:
  - "setfit-benchmark-claims-v1 2.0.0 — a 40-cell ACTIVE SetFit expectation set with the two-method design retained as an explicitly deferred scope"
  - "ACTIVE_METHODS — the expectation-set domain, split from BENCH_METHODS (the row-validity domain)"
  - "verify_row_evidence — step 4's loop body as one named function, shared by verify_run and the single-cell door"
  - "verify_cell / `apr setfit bench verify-cell` — the single-cell verification door (steps 1+4+6, no statistic)"
  - "ci95_one_sample_df9 — the ACTIVE-scope seed-dispersion interval on the frozen t"
  - "the written tombstone of record for the retired 9B LoRA arm"
affects:
  - "05-12 (can now write a run manifest that means what it says; inherits the pilot-cell door)"
  - "05-13 (the must-not-match literals it gates the committed report on are pinned here)"

tech-stack:
  added: []
  patterns:
    - "one list must never answer two questions — ACTIVE_METHODS vs BENCH_METHODS, each documenting which question it answers"
    - "#[cfg(test)]-gated ENUM VARIANT, so a deferred scope does not exist in a production build rather than merely not being constructed"
    - "behavioural equivalence tables — the same defect fed to two entry points, asserting the same variant tag, never asserting that one calls the other"
    - "retained-as-deferred contract clauses with status/ticket/decision keys, so a descope is a scope amendment rather than a silent loss of guarantees"

key-files:
  created:
    - scripts/setfit_fixtures/claims_stats/seed_dispersion_ci_cases.json
    - .planning/phases/05-benchmark-and-claims-gate/05-11-narrowing-inventory.md
    - .planning/phases/05-benchmark-and-claims-gate/05-11-prepared-edit.patch
  modified:
    - contracts/setfit-benchmark-claims-v1.yaml
    - crates/aprender-train/src/train/setfit/bench_row.rs
    - crates/aprender-train/src/train/setfit/bench_row_tests.rs
    - crates/aprender-train/src/train/setfit/bench_gate.rs
    - crates/aprender-train/src/train/setfit/bench_gate_tests.rs
    - crates/aprender-core/src/stats/hypothesis.rs
    - crates/aprender-core/src/stats/tests_claims_stats.rs
    - crates/apr-cli/src/commands/setfit_bench.rs
    - crates/apr-cli/src/commands/setfit_bench_tests.rs
    - crates/apr-cli/src/setfit_commands.rs
    - crates/apr-cli/src/dispatch_analysis.rs
    - scripts/setfit_fixtures/gen_claims_fixtures.py
    - Makefile

key-decisions:
  - "D-19 narrowing APPROVED by a human at a blocking checkpoint: claims contract 1.0.0 -> 2.0.0, active expectation set 80 two-method cells -> 40 SetFit cells, two-method design RETAINED as deferred (D-ITEM-05-15). pv diff suggested major; major taken."
  - "The narrowing lands on a NEW constant ACTIVE_METHODS, never on BENCH_METHODS, which keeps both methods as the row-validity domain."
  - "No new BenchGateError variant: an out-of-scope cell is refused by the EXISTING step-2 and step-4 refusals. variant_tag arm count 13 -> 13, measured across the plan commit range."
  - "The single-cell door is steps 1+4+6 and returns (), so it cannot emit a statistic."
  - "The deferred two-method scope is NESTED inside expectation_set rather than promoted to a sibling equation — measured: a top-level key produced BIND-001 and would have read as an eleventh implemented equation."

requirements-completed: []

coverage:
  - deliverable: "Claims contract narrowed to a 40-cell active scope with the two-method design retained as deferred"
    verification:
      - kind: command
        ref: "target/release/pv validate contracts/setfit-benchmark-claims-v1.yaml"
        status: pass
      - kind: command
        ref: "target/release/pv diff /tmp/p11/claims-old.yaml contracts/setfit-benchmark-claims-v1.yaml"
        status: pass
      - kind: test
        ref: "crates/aprender-train/src/train/setfit/bench_row_tests.rs#bench_row_expectation_matches_the_contract_as_a_typed_set"
        status: pass
    human_judgment: false
  - deliverable: "The gate verifies the 40-cell active scope with every negative preserved and the four in-scope shapes re-mutated there"
    verification:
      - kind: command
        ref: "cargo test -p aprender-train --lib --features setfit bench_gate"
        status: pass
      - kind: test
        ref: "crates/aprender-train/src/train/setfit/bench_gate_tests.rs#bench_gate_the_six_doctored_negatives_are_six_distinct_variants"
        status: pass
    human_judgment: false
  - deliverable: "Single-cell verification door, behaviourally equivalent to verify_run on every per-row defect"
    verification:
      - kind: test
        ref: "crates/aprender-train/src/train/setfit/bench_gate_tests.rs#bench_gate_the_single_cell_door_yields_the_same_variant_as_verify_run_for_every_row_defect"
        status: pass
      - kind: test
        ref: "crates/aprender-train/src/train/setfit/bench_gate_tests.rs#bench_gate_the_single_cell_door_passes_on_one_complete_cell_among_thirty_nine_pending"
        status: pass
    human_judgment: false
  - deliverable: "Single-method presentation with no cross-method section and the peak-RSS asymmetry surfaced"
    verification:
      - kind: test
        ref: "crates/apr-cli/src/commands/setfit_bench_tests.rs#setfit_bench_report_active_scope_matches_the_case_table"
        status: pass
      - kind: test
        ref: "crates/apr-cli/src/commands/setfit_bench_tests.rs#setfit_bench_report_renders_the_two_peaks_as_separate_labelled_figures"
        status: pass
    human_judgment: false
  - deliverable: "Seed-dispersion 95% interval pinned to a scipy reference case in the pinned uv env"
    verification:
      - kind: test
        ref: "crates/aprender-core/src/stats/tests_claims_stats.rs#ci95_one_sample_df9_matches_scipy_for_every_finite_case"
        status: pass
      - kind: command
        ref: "shasum -a 256 -c manifest.sha256 (scripts/setfit_fixtures/claims_stats/)"
        status: pass
    human_judgment: false
  - deliverable: "Suite banners corrected and every assert_tests_ran floor re-measured and raised"
    verification:
      - kind: command
        ref: "make setfit-bench-tests"
        status: pass
      - kind: command
        ref: "make contract-audit-phase5"
        status: pass
    human_judgment: false
  - deliverable: "The retired 9B LoRA arm has a written tombstone of record"
    human_judgment: true
    rationale: "Whether the record is adequate for a future reader to pick D-ITEM-05-15 up is a judgment no test asserts."

metrics:
  duration: "one session (interrupted twice by the harness watchdog; no work lost)"
  completed: 2026-09-08
  tasks: 3
  files: 17

actuals:
  tokens: 96000
  tasks: 3
  commits: 7
  plan_head_before: b69498736e226fa07f64ba1e5b0fe66c4c2bdc1f
---

# Phase 05 Plan 11: Retarget to a 40-Cell Active SetFit Scope Summary

The declared benchmark matrix goes from 80 two-method cells to 40 SetFit cells — approved by a
human at a blocking checkpoint against a `pv diff` of two filesystem paths — with the entire
two-method design retained as an explicitly deferred scope, the fail-closed gate re-proven at
the new scope rather than inherited, and a single-cell verification door added so 05-12's pilot
cell can be checked before 40 cells are spent.

---

## The two must_haves, measured

The coordinator asked for these two plainly, with commands and rc rather than intent.

### 1. All six of 05-10's doctored negatives still run and refuse under a bare invocation — YES

```
$ CARGO_INCREMENTAL=0 cargo test -p aprender-train --lib --features setfit
rc=101
test result: FAILED. 8041 passed; 24 failed; 17 ignored; 0 measured; 0 filtered out
```

**The rc is 101, and none of it is the bench surface.** Measured rather than asserted — the
failing modules, read out of that same log:

```
gpu::guard::tests
gpu::ledger::tests
gpu::wait::tests
prune::snapshot_tests::tests
```

Bench tests failing in that run: **0**. `bench_gate` tests passing inside it: **38**. Those 24
failures are PRE-EXISTING and outside this plan's blast radius — `git diff --name-only
b69498736..HEAD` touches no file under `gpu/` or `prune/`, and they fail identically with
`--test-threads=1`, so they are not a parallelism artifact. They are recorded as
**D-ITEM-05-16** in `deferred-items.md` with their own reproduction commands, including the one
that looks like a real defect (`test_capacity_invariant_prevents_overallocation` fails
`assert!(result.is_err())` — a capacity invariant meant to REFUSE an overallocation is accepting
one).

The scoped run is clean:

```
$ CARGO_INCREMENTAL=0 cargo test -p aprender-train --lib --features setfit bench_gate
rc=0
test result: ok. 38 passed; 0 failed; 0 ignored; 0 measured; 8044 filtered out
```

All six doctored shapes keep their shapes and their asserted variant tags. The only permitted
change was made: the expectation SCOPE they are constructed under. `bench_gate_the_six_doctored_
negatives_are_six_distinct_variants` still builds all six in one test and still asserts six
DISTINCT variant tags.

| # | Doctored shape | Variant tag | Scope it is built under |
|---|---|---|---|
| 1 | a cell's row file deleted | `incomplete_cell` | active **and** deferred |
| 2 | the evidence block trimmed out of a row | `row_schema_refused` | active **and** deferred |
| 3 | one pair's two rows on different selection manifests | `unpaired_selection` | **deferred only** |
| 4 | a row's payload bytes edited | `row_digest_mismatch` | active **and** deferred |
| 5 | the lock rule is not the committed one (post-test selection) | `post_test_selection` | active **and** deferred |
| 6 | forged provenance — an edited lock file / a ledger carrying two candidate lines against a row claiming one | `provenance_mismatch`, `post_test_selection` | **deferred only** |

Shapes 3 and 6 have no active-scope form and are not faked into one: they need rows production
code cannot construct. That is exactly why `BENCH_METHODS` was left at two methods — narrowing
it would have made those rows fail row validity and silently deleted both negatives while
looking like a tightened gate.

### 2. The four in-scope shapes RE-MUTATED under the 40-cell active scope — YES

CLAUDE.md Verification Discipline rule 4: extending a guard's SCOPE requires re-mutating in the
new scope; the 80-cell proof does not transfer. It was not transferred. `write_valid_run()` now
builds a **40-cell ACTIVE** directory, so each of these tests doctors its shape against the new
expectation set and refuses there, with the same variant tag it always asserted:

| Shape | Active-scope test | Variant tag asserted |
|---|---|---|
| missing cell | `bench_gate_refuses_a_missing_cell_naming_it` | `incomplete_cell` |
| trimmed row | `bench_gate_refuses_a_trimmed_row_whose_evidence_block_was_removed` | `row_schema_refused` |
| bit-flipped row bytes | `bench_gate_refuses_a_row_whose_payload_bytes_were_edited` | `row_digest_mismatch` |
| post-test selection | `bench_gate_refuses_a_setfit_row_whose_lock_rule_is_not_the_committed_one` | `post_test_selection` |

Two more active-scope re-mutations came free and are kept: `bench_gate_refuses_a_setfit_row_
whose_committed_lock_file_was_edited` (`provenance_mismatch`) and
`bench_gate_refuses_a_manifest_whose_expectation_set_is_not_the_contracted_forty`
(`expectation_set_mismatch`, the twelve-cell adversary the contract names in its own words).

---

## Task 1 — the approved narrowing

**Selected: `option-a`**, recorded verbatim from the human at the blocking checkpoint:

> Select: option-a — commit the prepared narrowing, two-method design retained as deferred.

Rationale as given: this was an explicit human selection at a blocking checkpoint, made against
the full bundle (the mechanical coverage enumeration, the retained-vs-narrowed inventory, the
no-deletion counts, the rc=0 verification set, and the nested-vs-sibling design choice with its
measured BIND-001 evidence). Auto-mode was off for this run (`workflow.auto_advance: false`,
`_auto_chain_active: false`), so this was not an auto-approval.

### The RETAINED-vs-NARROWED inventory

Reproduced in full at
`.planning/phases/05-benchmark-and-claims-gate/05-11-narrowing-inventory.md`, which is committed
(`9ed364600`) and is the artifact the human read. Its coverage was **enumerated mechanically,
not by reading**: a script loaded the 1.0.0 YAML and flagged every equation, obligation,
falsification test and Kani harness whose serialized subtree matched
`80 | lora | two-method | second method | pair`. 10/10 equations, 4/9 obligations, 6/10
falsification tests and the 1 Kani harness matched; all 21 appear in the inventory exactly once,
each marked narrowed / retained-as-deferred / unchanged. Nothing unaccounted for.

Disposition at a glance:

| Disposition | Items |
|---|---|
| **Narrowed** | `expectation_set` (80→40, by removing `lora` from `methods`), `completeness_rule`, `pitfall_bindings.pf_007`, `OBLIG-…-EXPECTATION-CLOSED-FORM`, `OBLIG-…-COMPLETENESS-FAIL-CLOSED`, `FALSIFY-001/006/007/008`, `KANI-CLAIMS-001` bound, `qa_gate` description |
| **Retained-as-deferred** (`D-ITEM-05-15`, `D-19`) | the 80-cell product itself (now `expectation_set.deferred_two_method_scope`), `pairing_rule`, `no_selection_attestation`, `selection_safety_evidence.lora_side`, the cross-method half of `model_size_comparability`, the paired-delta half of `claims_statistics`, the two-host framing in `resource_protocol`, the pairing conjunct of the fail-closed obligation, the LoRA halves of `FALSIFY-009` |
| **Unchanged** | `bench_row_schema` (both method tags stay representable), `OBLIG-…-METHOD-TAGGED`, `DIGEST-BEFORE-READ`, `NO-RNG-FIELD`, `CELL-IDENTITY`, `DETERMINISTIC-ORDER`, `FALSIFY-002/003/004/005/010` |
| **Added** (active scope) | `seed_dispersion_ci95`, `not_comparable.within_row_asymmetry`, the no-cross-method-claim prohibition, a third contract-mutation control, the within-row presentation clause on `OBLIG-…-RESOURCE-BOUNDARIES` |

**Nothing was deleted — machine-checked, not asserted:** equations 10→10, obligations 9→9,
falsification tests 10→10, Kani 1→1, `removed=[]` and `added=[]` on every set. `pv diff` renders
each edited obligation as a `-`/`+` pair because it has no "modified" verb; all four `-` lines
have a matching `+` with the identical ID.

### The `pv diff` invocation and its suggested bump

```
$ git show HEAD:contracts/setfit-benchmark-claims-v1.yaml > /tmp/p11/claims-old.yaml
$ target/release/pv diff /tmp/p11/claims-old.yaml contracts/setfit-benchmark-claims-v1.yaml
rc=0
Contract diff: v1.0.0 → v2.0.0
Suggested bump: major
```

Both arguments are filesystem paths, per CLAUDE.md — `pv diff` opens both as files and a
revision passed here reports a misleading "No such file or directory". The tool suggested
**major** and the prepared file already carried `2.0.0`, so there was no bump tension to flag.
The reasoning is recorded in the contract's own metadata header rather than only in a commit
message: a consumer who previously read `expectation_set` as a two-method 80-cell guarantee gets
a *different answer from the same key* after this edit.

```
$ target/release/pv validate contracts/setfit-benchmark-claims-v1.yaml
rc=0
0 error(s), 0 warning(s) — Contract is valid.
```

Byte-level control: `git diff -U0` hunks fall only in `equations` (24), `proof_obligations` (8),
`falsification_tests` (5), `metadata` (3), `qa_gate` (2), `kani_harnesses` (1) — 312 insertions,
68 deletions. `references`, `depends_on`, `contract`, `created`, `author`, `kind` untouched.

**The contract commit is separate from the code commit**: `78a8af6c5` contains
`contracts/setfit-benchmark-claims-v1.yaml` and nothing else, so the diff the human approved is
in history unmixed with the retarget that follows.

### 40 is a product, not a literal

```
ACTIVE   ['setfit'] x [8, 16, 32, 64] x 10 seeds = 40 = expected_cells
DEFERRED ['setfit', 'lora'] -> 80  status=deferred  ticket=D-ITEM-05-15
```

Verified by loading the amended file, not transcribed. The narrowing removed a list element, so
the arithmetic followed; the `FORTY IS A PRODUCT` invariant and the shipped parity assertion
make a hand-typed 40 beside an unchanged two-element list unrepresentable.

---

## Task 2 — the gate at the new scope

### The two contract-mutation controls, RED then reverted

Both were induced against the **committed** contract, with rc captured directly and the revert
proven by sha256 rather than by eye.

**Control 1 — an eleventh seed:**
```
rc=101
test result: FAILED. 24 passed; 3 failed
contract expected_cells 40 != |methods| * |shots| * |seeds| = 44
failures: bench_row_expectation_matches_the_contract_as_a_typed_set
          bench_row_parity_rejects_a_contract_with_a_duplicated_seed
          bench_row_parity_rejects_a_contract_with_an_extra_seed
```
Three failed, not one — the parity test plus both seed-mutation negatives, whose own
"the mutation must actually apply" guard fires once the anchor text has moved. The negatives
cannot go quietly vacuous. Reverted; sha256 `982963b7…` before and after.

**Control 2 — a restored second method (new at 2.0.0, because the method axis is the one the
narrowing moved and the only direction it could be silently undone in):**
```
rc=101
test result: FAILED. 25 passed; 2 failed
contract expected_cells 40 != |methods| * |shots| * |seeds| = 80
failures: bench_row_expectation_matches_the_contract_as_a_typed_set
          bench_row_parity_rejects_a_contract_with_an_extra_method
```
Reverted; sha256 identical, `REVERT BYTE-IDENTICAL`.

### `ACTIVE_METHODS` vs `BENCH_METHODS` — the split that carries the plan

`ACTIVE_METHODS` (**new**, 1 method) is the expectation-set domain; `EXPECTED_CELLS` and
`RunManifest::expectation()` derive from it, so 40 moved as a product. `BENCH_METHODS`
(**unchanged**, 2 methods) stays the row-validity domain that `CellKey::is_contracted` reads at
`bench_row.rs:197`. Both constants document which question they answer, because the defect class
here is one list read as the answer to two.

`bench_row_active_scope_and_row_validity_are_different_questions` asserts both facts at once so
they cannot collapse: a `lora` cell **is contracted as a row** and **is not in the active
expectation set**.

The same collapse was found already shipped in the CLI suite and fixed:
`setfit_bench_accepts_every_contracted_cell` counted all 80 representable cells and asserted the
count equalled `EXPECTED_CELLS`. It now asserts against the row-validity product and separately
that the active set is exactly half of it.

### `verify_run` is unchanged in shape

```
$ grep -n 'fn verify_run' crates/aprender-train/src/train/setfit/bench_gate.rs
pub fn verify_run(manifest: &RunManifest, bench_dir: &Path) -> Result<VerifiedRunSet, BenchGateError>
```

Two arguments, no scope parameter — so no caller can widen the expectation set by passing one.
It delegates to a `pub(crate)` scoped form with `ExpectationScope::Active`. The seven numbered
step comments are in the same order with no check moved between them.

**`ExpectationScope::DeferredTwoMethod` is a `#[cfg(test)]`-gated VARIANT**, quoted from the
diff:

```rust
    #[cfg(test)]
    DeferredTwoMethod,
```

In a production build the variant does not exist and no caller can name it — stronger than a
constructor that merely happens not to be called (T-05-11-07). A search of the non-test sources
finds no caller.

### No new error variant

An out-of-scope cell is refused on the two paths it can take through a production door, by
refusals that already existed:

- **declared in a manifest** → step 2's expectation-set backstop, `expectation_set_mismatch`,
  before any row byte is read — `bench_gate_refuses_a_manifest_declaring_a_second_methods_cell_
  before_reading_a_row`;
- **filed in a declared SetFit slot** → step 4's slot check, `row_slot_mismatch` —
  `bench_gate_refuses_a_second_methods_row_placed_in_a_declared_setfit_slot`.

A third variant would be reachable only from a test-only constructor, which is a guard over a
path production cannot take. Counted over the SHIPPED SOURCE across this plan's own commit
range, not eyeballed in a diff:

```
b69498736  variant_tag arms = 13
f54f5f1b3  variant_tag arms = 13
```

`bench_gate_the_variant_tag_table_gained_no_arm_in_this_plan` pins it at 13 in the suite.

### The extracted step-4 function, and both call sites

`fn verify_row_evidence(entry: &CellEntry, rows_dir: &Path, cell: CellKey) -> Result<BenchRow, BenchGateError>`

It performs exactly what step 4's loop body performed, in that order: file present → `from_bytes`
(schema, then envelope digest) → manifest-digest agreement → slot agreement. It **returns the
parsed row**, because both callers need the value:

- `verify_run`'s step-4 loop: `rows.push((cell, verify_row_evidence(entry, &rows_dir, cell)?));`
  — the value is accumulated into `rows` for steps 5 through 7;
- `verify_cell`: `let row = verify_row_evidence(entry, &rows_dir, cell)?;` then
  `verify_provenance(cell, &row, bench_dir)?;`

`verify_provenance` keeps its single definition and gains the door as a second caller. Steps 1
and 3's per-entry rule are likewise named (`verify_manifest_digest`, `verify_entry_complete`).
No check moved between passes, so refusal order across rows is byte-unchanged.

### The door: which steps, and the behavioural equivalence table

`verify_cell` applies **1, 3'(this entry only), 4, 6** and deliberately **does not** apply step 2,
step 3's sweep, step 5 or step 7. It returns `()`, not the row and not an aggregate — so it
**cannot** emit a statistic (T-05-11-06).

`bench_gate_the_single_cell_door_yields_the_same_variant_as_verify_run_for_every_row_defect` is
the test carrying the whole no-divergence claim. Each defect is built once per entry point in a
fresh directory, fed to both, and the two variant TAGS compared. Enumerated so a reader can see
it is a per-defect comparison rather than a restatement of the door's definition:

| # | Per-row defect | tag from `verify_run` | tag from `verify_cell` |
|---|---|---|---|
| 1 | the row file is absent | `row_file_missing` | `row_file_missing` |
| 2 | the envelope digest no longer covers the payload | `row_digest_mismatch` | `row_digest_mismatch` |
| 3 | the schema no longer parses (a required block trimmed) | `row_schema_refused` | `row_schema_refused` |
| 4 | the row is filed under the wrong slot | `row_slot_mismatch` | `row_slot_mismatch` |
| 5 | the committed lock bytes were tampered with | `provenance_mismatch` | `provenance_mismatch` |
| 6 | the lock role is not one the vocabulary admits | `post_test_selection` | `post_test_selection` |
| 7 | the lock rule is not the committed one | `post_test_selection` | `post_test_selection` |

The test asserts `compared == 7`, so a table that silently shrank to zero rows cannot pass.
A test that merely asserted "the door calls the two functions" would restate the door's own
definition and could never go red — that is the defect class this phase exists to prevent.

`bench_gate_the_single_cell_door_passes_on_one_complete_cell_among_thirty_nine_pending` builds
the pilot state: 40 declared, 39 pending. `verify_run` refuses with `incomplete_cell`; the door
**passes** on the complete cell, and still refuses a pending one. That is precisely why the
set-level steps are excluded.

### Uncertainty, and one definition of the moments

`ci95_one_sample_df9` delegates to `paired_ci` against an all-zero comparator — the exact
one-sample specialisation — so the mean and the (n−1) std come from the same
`moments_or_zero_variance` the paired path uses. There is no second mean and no second std
anywhere in the claims layer.
`ci95_one_sample_df9_is_the_paired_helper_against_zero_not_a_second_definition` asserts BIT
equality, so a re-implementation with its own moments goes red on a single ulp.

The scipy reference case is `scripts/setfit_fixtures/claims_stats/seed_dispersion_ci_cases.json`,
generated in the pinned uv env (python 3.13.7 / numpy 2.5.1 / scipy 1.18.0), cross-checked
against `scipy.stats.t.interval` and `scipy.stats.sem`. Regeneration reproduced the four
pre-existing fixtures **byte-identically** — only `manifest.sha256` gained a line — and:

```
$ shasum -a 256 -c manifest.sha256
rc=0
brier_multiclass_cases.json: OK / ece_top_label_cases.json: OK / paired_t_cases.json: OK
seed_dispersion_ci_cases.json: OK / t_critical.json: OK
```

A zero-variance seed set produces the typed zero-variance shape with an explicit no-interval
reason — never a NaN, never a serde null (CR-03).

---

## Task 3 — presentation, the door, and the floors

### The rendering case table, and where its must-not-match rows come from

Every must-not-match row is a **real prior output**, not an invented near-miss — each is either a
shipped constant of the two-method renderer or a fragment copied out of its own format string.
Provenance recorded, as required:

| Row | Source in the pre-change file |
|---|---|
| `DELTA_TABLE_HEADER` | `render_deltas`, first line of its output |
| `RESOURCE_COMPARISON_HEADER` | `render_resource_comparison`, first line |
| `COMPARISON_ROW_MARKER` (`"  |  lora "`) | `comparison_row`'s own format string, the separator between the two sides of one row |
| `TWO_METHOD_TITLE` (`"SetFit vs LoRA - benchmark claims report"`) | `render_header`'s title line |
| `ESTIMATION_FIRST_NOTE_PAIRED` | the note that advertised paired intervals |
| `PER_HOST_FRAMING_TWO_HOST` | the framing line that described a two-host design |

The table is **non-vacuous in both directions**: every must-not-match row is asserted ABSENT from
the active render and PRESENT in the two-method render of the same fixture, so asserting its
absence cannot be vacuous.

### The two falsified constants, quoted before and after

**Estimation-first note** — it advertised a statistic the active report does not contain:

- before: `"Estimation-first (D-08): point estimates, dispersion and paired 95% CIs only. No binary verdict is printed; p-values live in the --json detail."`
- after: `"Estimation-first (D-08): point estimates, dispersion and 95% seed-dispersion intervals only. No binary verdict is printed."`
- the old wording is **retained** as `ESTIMATION_FIRST_NOTE_PAIRED` and still emitted by the deferred renderer.

**Resource framing line** — it described a two-host design that no longer exists:

- before: `"as-deployed method costs; hosts differ by design and are never averaged together"`
- after: `"as-deployed costs on ONE host for ONE method; every figure carries its measurement boundary, and no figure here is averaged across hosts"`
- the old wording is **retained** as `PER_HOST_FRAMING_TWO_HOST`.

### The single-method note, and the tension resolved in the suite

Rendered line, quoted:

> `SCOPE - ONE METHOD WAS MEASURED. This report covers SetFit alone. A second method was planned for this matrix and was not run; the reason is recorded as decision D-19 and the restoration path as ticket D-ITEM-05-15. Nothing here states or implies any result about a second method. Read the absence as absence.`

`setfit_bench_report_single_method_note_cannot_trip_the_gate_it_is_mandated_beside` pins it
against the **same** must-not-match table 05-13 will gate the committed report on, plus the
constrained vocabulary: no `lora`, no comparative connective (` vs `, `versus`, `compared to`,
`beside`), no `delta`, no `significant`. It must and does carry `D-19`, `D-ITEM-05-15` and
"a second method". So the tension is resolved here rather than discovered in 05-13 Task 1.

### The peak-RSS asymmetry

The two peaks never share a column and are never reduced to one figure. Rendered as
`train peak RSS (training process)` and `inference peak RSS (cold child)`, each printing its own
mechanism string; a `sysinfo_sampled_*` figure carries
`LOWER BOUND (sampled; can only understate)` **beside the value**, in the per-shot detail rather
than only in a methods paragraph; and a group whose two mechanism classes differ carries the
asymmetry note. The test has a non-vacuity control: an all-exact-mechanism fixture must NOT
carry the label, so a label that is always printed would fail.

### Makefile: banners first, then floors

Banners the retarget falsified, corrected in the same pass:

- row banner: `"The row schema and the 80-cell expectation set ARE the claim"` →
  `"… the 40-cell ACTIVE expectation set ARE the claim"`;
- gate banner: it enumerated the unpaired-pair and forged-provenance shapes among what a red
  would mean is no longer detected. It now says which four are RE-MUTATED under the active scope,
  names the other two as DEFERRED-SCOPE negatives, and names the two active-scope out-of-scope
  refusals that reuse existing variants;
- CLI banner: it described a "two-sided incomparability control" the active scope no longer
  renders → now names the rendering case table, the peak-RSS asymmetry assertions and the door's
  no-statistic check.

Floors re-measured from the fresh logs and **raised**. None lowered:

| suite | before | after | measured |
|---|---|---|---|
| `bench_row` | 22 | **27** | 27 |
| `bench_gate` | 29 | **38** | 38 |
| `bench_metrics` | 12 | **14** | 14 |
| `apr-cli` | 55 | **64** | 64 |

**Induced-red control**, because a floor that cannot fail is not a floor. The gate floor was
raised to 39 — one above the measured count:

```
$ make setfit-bench-tests
rc=2
FAIL: setfit-bench-tests/bench_gate reported 38 test(s) passed, expected at least 39.
```

Reverted, and the revert proven **byte-identical** by sha256
(`75236572c9fe6ee80c03d13ee89343bf024df4728893250314387bc42b03a259`) rather than by eye.

### The two gates this plan owns

```
$ make setfit-bench-tests
rc=0
  phase 5 bench surface: row, gate (six negatives: four active-scope, two
  deferred-scope), metrics and the single-method CLI report all ran

$ make contract-audit-phase5
rc=0
Phase 5 binding audit: 1 contract(s) audited, zero BIND- findings
```

---

## Why the deferred scope is nested rather than a sibling equation

Measured, not assumed. Promoting it to `equations.deferred_two_method_scope` produced:

```
$ target/release/pv audit contracts/setfit-benchmark-claims-v1.yaml --binding contracts/aprender/binding.yaml
rc=1
Total equations:    11
[ERROR] BIND-001: Equation 'deferred_two_method_scope' … has no binding entry
```

A top-level key would have needed a binding row in `contracts/aprender/binding.yaml` — a file
outside this plan's declared scope — and would have read as an **eleventh implemented equation**
in `make contract-audit-phase5`: a deferred, never-run scope presented in the binding ledger as
implemented code. Nesting it inside `expectation_set`, which it is the deferred counterpart of,
keeps the equation count at 10 and `binding.yaml` untouched.

---

## Requirements: none completed, and that is the plan's own ruling

`requirements-completed: []`. The plan's `<amended_requirements>` table is explicit — *"Neither
EVAL-02 nor EVAL-04 is satisfied by this plan. Both remain Pending, amended in scope."* EVAL-05
is likewise partial: comparison across shot levels and seeds within SetFit is delivered;
cross-METHOD resource comparison is removed, not repurposed. `requirements.mark-complete` was
deliberately **not** run. Filing a narrower deliverable against an unamended requirement is the
false-completion defect this project has hit six times.

---

## Deviations from Plan

**1. [Rule 3 — blocking issue] Repaired a vacuous non-vacuity guard**

- **Found during:** Task 3, running `cargo test -p apr-cli --lib --features setfit setfit_bench`.
- **Issue:** `driver_never_reads_a_status_through_a_pipe`'s non-vacuity assertion looked only for
  a bare `rc=$?` at line start. The driver moved to the inline `|| rc=$?` form in `9920feae8`
  (it runs under `set -e`, where a bare next-line capture is unreachable), so after
  comment-stripping the predicate matched **zero** lines and the assertion had been unsatisfiable
  ever since. **Measured at the plan base and at HEAD: 0 and 0** — pre-existing, and blocking
  `make setfit-bench-tests`.
- **Fix:** repaired rather than deleted or relaxed, and **strengthened** — the inline form can
  hide the same defect (`cmd | tee log || rc=$?` captures tee's status) and was previously
  unscanned, so it is now scanned too.
- **Files:** `crates/apr-cli/src/commands/setfit_bench_tests.rs`. **Commit:** `d5b6759bc`.

**2. [Rule 1 — bug] The two-questions collapse, already shipped in the CLI suite**

- **Found during:** Task 3. `setfit_bench_accepts_every_contracted_cell` asserted that the count
  of all REPRESENTABLE cells equals `EXPECTED_CELLS` — the exact collapse `ACTIVE_METHODS` exists
  to prevent.
- **Fix:** it now asserts against the row-validity product, and separately that the active
  expectation set is exactly half of it. **Commit:** `d5b6759bc`.

**3. [Rule 3 — process] The stale plan commit ledger**

- `.git/gsd-plan-head-before-05-11` already existed at base `1996d2792`, left by the **retired**
  05-11 run. Corrected to this run's true base `b69498736`. Had it not been, the measured
  `commits:` count would have been inflated by every commit between those two points.

**4. [Rule 3 — blocking] Task 1 committed two artifacts despite `<files>none`**

- Task 1 declares `<files>none (nothing is committed in this task)`. The inventory and a patch
  snapshot were committed anyway (`9ed364600`) because a prior attempt at exactly this
  preparation was destroyed by a harness watchdog kill. This follows 05-03's house form, which
  committed `05-03-prepared-edit.patch`. **The narrowing itself was not committed until approved**
  — `HEAD` carried 1.0.0 throughout the checkpoint.

**Total deviations:** 4 auto-fixed (2 blocking-issue repairs, 1 bug, 1 process). **Impact:** two
pre-existing red/vacuous gates repaired without weakening either; no plan behaviour changed.

---

## A process failure worth recording: a commit message that was briefly false

While working on Task 3, three source edits to `crates/apr-cli/` (`setfit_bench.rs`,
`setfit_commands.rs`, `dispatch_analysis.rs`) were **silently lost between the edit and the
commit** — each write reported success and each subsequent `cargo check` returned rc=0, because
the losses were mutually consistent: nothing referenced the missing module, so nothing failed to
compile. Commit `ccad38fb0` was therefore created carrying its full message and **only the test
file**.

Caught by the next compile, which failed on imports of constants that should have existed. The
edits were re-applied, **each symbol verified on disk by name** rather than trusted from the
writer's exit status, and the commit amended in place (it had never been pushed) so its message
became true. The amended commit is `d5b6759bc` and carries a note recording this.

The general lesson is one CLAUDE.md already states and this run still paid for: **when a result
looks good, check how it was measured.** A successful write call is not evidence that bytes are
on disk, and a green `cargo check` is not evidence that the code you wrote is what compiled.

---

## Issues Encountered

Three pre-existing red gates were found while checking for collateral damage, all **outside this
plan's blast radius** (`git diff --name-only b69498736..HEAD` touches no file under `gpu/`,
`prune/` or the `data` command surface). Recorded as **D-ITEM-05-16** rather than fixed, per the
executor scope boundary:

1. **`gpu::guard` / `gpu::ledger` / `gpu::wait` — 21 failing tests**, failing serially too, so
   not a parallelism artifact. `test_capacity_invariant_prevents_overallocation` fails
   `assert!(result.is_err())`: an invariant meant to REFUSE an overallocation is accepting one.
2. **`prune::snapshot_tests` — 3 failing tests.**
3. **`FALSIFY-CLI-006` red:** the binary ships `data tweet-eval-stance`, `data select` and
   `data pairs`, which `contracts/apr-cli-commands-v1.yaml` does not declare. Fixing it is a
   contract edit — the kind this phase requires a checkpoint for — so it was not slipped into an
   unrelated plan. Note that this plan's own `apr setfit bench verify-cell` is **not** among the
   undeclared: the gate checks depth-2 paths and `verify-cell` sits at depth 3.

Also: 15 `-D warnings` clippy findings, all in `aprender-compute` (14) and
`aprender-present-terminal` (1). Zero in `aprender-train` or `aprender-core`.

## Known Stubs

None. Nothing in this plan renders a placeholder, and the one deliberately unexercised surface —
the deferred two-method scope — is marked `status: deferred` in the contract, gated behind
`#[cfg(test)]` in code, and covered by running tests.

## Threat Flags

None. No new network endpoint, auth path, file-access pattern or schema change at a trust
boundary. The single-cell door reads the same files `verify_run` already reads, through the same
bounded reader.

---

## Next Phase Readiness

**Ready for 05-12.** It can now write a run manifest that means what it says — 40 declared cells,
all buildable — and it inherits the single-cell door for its pilot-cell proof. Two things 05-12
must carry itself:

- **Compute is NOT authorized by this plan.** Approving the narrowing authorized no compute;
  05-03's checkpoint deferred the 40-cell run to wave 7 by explicit human instruction, and 05-12
  asks for it on its own.
- **The selection manifests are not orphaned.** `scripts/run_bench_cells.sh::generate_selections`
  (05-09) generates them; 05-12 inherits the capability by invoking the driver.

**Ready for 05-13.** The must-not-match literals it gates the committed report on are pinned in
`crates/apr-cli/src/commands/setfit_bench_tests.rs`, and the mandated single-method note is
already proven not to trip them.

## Self-Check: PASSED

- `scripts/setfit_fixtures/claims_stats/seed_dispersion_ci_cases.json` — FOUND
- `.planning/phases/05-benchmark-and-claims-gate/05-11-narrowing-inventory.md` — FOUND
- `.planning/phases/05-benchmark-and-claims-gate/05-11-prepared-edit.patch` — FOUND
- commits `9ed364600`, `78a8af6c5`, `bc4c9a65b`, `4a50b88a3`, `d5b6759bc`, `f54f5f1b3` — all FOUND
- `make setfit-bench-tests` rc=0; `make contract-audit-phase5` rc=0, zero BIND- findings
- `cargo fmt --all -- --check` rc=0
