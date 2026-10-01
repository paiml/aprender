---
phase: 05-benchmark-and-claims-gate
plan: 17
subsystem: testing
tags: [eval-01, claims-gate, closed-form-crosscheck, confusion-matrix, bench_gate, bench_metrics, setfit, rust, provable-contracts]

requires:
  - phase: 05-benchmark-and-claims-gate (05-15)
    provides: "resolve_committed_evidence_path, the class invariant, the door probe with its positive control, and the measured-floor discipline"
  - phase: 05-benchmark-and-claims-gate (05-16)
    provides: "the selection-manifest binding at a gate-derived path, the swept-case-table shape with its in-test row-count pin, and finding T-05-16-05"
provides:
  - "quality_from_confusion_matrix + RecomputedQuality: the published accuracy family recomputed in closed form from a row's own counts, routed to the SAME shipped surfaces assemble_quality_block used"
  - "verify_quality_closed_form: a fixed-field-order, to_bits() comparison wired into BOTH shipped doors, plus all five _bits siblings"
  - "BenchGateError::QualityCrossCheckMismatch; BenchMetricsError::ConfusionMatrixShape and ::ConfusionMatrixTooLarge"
  - "the 40-committed-row agreement MEASUREMENT that the exact band was chosen from: 40/40 bit-identical on every field, max deviation 0e0"
  - "a twelve-row swept case table at the ACTIVE 40-cell scope, every mutation observed RED as Ok(40 rows verified) and GREEN as its own variant, plus the same spot-check D through verify_cell"
  - "the RETIRED residual concession, corrected in all three places it is stated (report, gate header, contract) and GATED by must-match / must-not-match rows"
  - "finding T-05-16-05 closed: verify-cell's printed scope note now enumerates every recomputation the door performs"
  - "contract 4.0.0: quality_closed_form_crosscheck, OBLIG-CLAIMS-QUALITY-CLOSED-FORM, FALSIFY-CLAIMS-012, residual_risk.amended_4_0_0"
  - "spot-check D replayed through the shipped apr door (five probe cases), measured rc=0 by the verifier and now rc=5"
affects: [05-VERIFICATION, D-ITEM-05-15, D-ITEM-05-17-A, benchmark claims gate, EVAL-01]

actuals:
  tokens: 35660
  tasks: 3
  commits: 5
plan_head_before: 49832d05c8efe9affea49cd8e011e2b02ce488c1

tech-stack:
  added: []
  patterns:
    - "A cross-check belongs in the module that ROUTES to the metric surfaces, never in the gate — a second computation in the gate is a second definition of the number (OPS-03)"
    - "Measure the agreement over the committed evidence BEFORE choosing the acceptance band, and record the measurement; exactness earned structurally needs no epsilon"
    - "A disclosure is a claim and is gated like a number: a must-match row for what it now says and a must-not-match row for the sentence it retired"
    - "An UNDERSTATED disclosure is a defect of the same class as an overstated one — it teaches a reader to trust real evidence less than it warrants"
    - "Make the fixture a function of its own evidence BEFORE adding the check over that evidence, or the first red is the fixture rather than the gate"
    - "Bound any expansion driven by producer-supplied counts: `u64::MAX` in one JSON cell is a denial of service through a shipped door"

key-files:
  created: []
  modified:
    - "crates/aprender-train/src/train/setfit/bench_metrics.rs"
    - "crates/aprender-train/src/train/setfit/bench_metrics_tests.rs"
    - "crates/aprender-train/src/train/setfit/bench_gate.rs"
    - "crates/aprender-train/src/train/setfit/bench_gate_tests.rs"
    - "crates/apr-cli/src/commands/setfit_bench.rs"
    - "crates/apr-cli/src/commands/setfit_bench_tests.rs"
    - "contracts/setfit-benchmark-claims-v1.yaml"
    - "scripts/setfit_bench_gate_door_probe.sh"
    - "scripts/setfit_bench_gate_doctor.py"
    - "Makefile"
    - "benchmarks/tweeteval-stance/report.md"

key-decisions:
  - "The acceptance band is EXACT IEEE-754 bit equality, and the measurement that chose it was taken first: 40/40 bit-identical on every field over the committed rows, max absolute deviation 0e0. There is no epsilon in this round because none was needed and none was fitted."
  - "The recomputation lives in bench_metrics.rs, not bench_gate.rs. That module's header already says it computes nothing and routes; the gate's header already names a second definition of a number as the failure mode. The gate calls and compares."
  - "pv diff suggested MINOR and the suggestion was RECORDED VERBATIM AND NOT TAKEN — 4.0.0. The acceptance set narrows again: the exact tree spot-check D built verified at 3.0.0 and is refused now. That is the reasoning the 3.0.0 metadata block itself records, and it does not stop applying because the tool scored this edit differently."
  - "ONE new gate variant (17 -> 18). The three degenerate-matrix shapes reuse row_schema_refused, because Vec<Vec<u64>> is the wire type and a shape the schema cannot hold IS a schema refusal. Two new BenchMetricsError variants, one of them a DoS bound the plan's own threat model (T-05-17-07) under-mitigated."
  - "verify_quality_closed_form takes bench_dir, a deliberate departure from the plan's stated (cell, row) signature, so a shape refusal can NAME the row file — symmetric with verify_provenance and verify_selection_binding."
  - "requirements-completed is deliberately empty, consistent with all sixteen prior phase-5 plans: flipping requirement state is the verifier's act."

patterns-established:
  - "A bits sibling is a claim a row makes about ITSELF and costs nothing to hold, even where the VALUE is not recomputable — but holding the encoding is not proving the value, and the disclosure has to keep saying which"
  - "An INVERTING PAIR of mutations pins the field ORDER of a multi-field check: doctor the value alone, then the encoding alone, and assert each reports its own field"

requirements-completed: []

coverage:
  - id: D1
    description: "Every published accuracy figure on a row is recomputed from that row's own confusion_matrix and ordered_labels through the shipped surfaces, and a disagreement is refused by to_bits()"
    requirement: "EVAL-01"
    verification:
      - kind: unit
        ref: "bench_gate_tests.rs#bench_gate_quality_cross_check_case_table_at_the_active_scope (12 rows, ACTIVE 40-cell scope, through public verify_run)"
        status: pass
      - kind: unit
        ref: "bench_metrics_tests.rs#bench_metrics_the_forty_committed_rows_agree_with_their_own_confusion_matrices (40/40 exact, max dev 0e0)"
        status: pass
      - kind: e2e
        ref: "bash scripts/setfit_bench_gate_door_probe.sh -> PASS, spot-check D rc=5"
        status: pass
    human_judgment: false
  - id: D2
    description: "Spot-check D — f_avg 0.4579 -> 0.99 with the row envelope digest, the manifest row_sha256 and the manifest envelope digest all repaired — exits non-zero through the shipped door"
    requirement: "EVAL-01"
    verification:
      - kind: e2e
        ref: "scripts/setfit_bench_gate_door_probe.sh case 5 -> rc=5, asserted to be the cross-check refusal and NOT the row-digest one"
        status: pass
      - kind: unit
        ref: "bench_gate_tests.rs#bench_gate_the_single_cell_door_also_applies_the_quality_cross_check (verify_cell, with a passing control)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Each *_bits field equals to_bits() of the f64 beside it; a doctored decimal with an unrepaired bit pattern, and the reverse, are each refused naming their own field"
    requirement: "EVAL-01"
    verification:
      - kind: unit
        ref: "bench_gate_tests.rs#bench_gate_quality_cross_check_case_table_at_the_active_scope (the f_avg_bits / f_avg inverting pair + the ece bits row)"
        status: pass
      - kind: unit
        ref: "bench_metrics_tests.rs#bench_metrics_the_forty_committed_rows_agree_with_their_own_confusion_matrices (all five siblings, 40/40)"
        status: pass
    human_judgment: false
  - id: D4
    description: "A non-square matrix, one whose dimension disagrees with ordered_labels.len(), and one totalling zero are each refused with a typed error rather than producing NaNs"
    requirement: "EVAL-01"
    verification:
      - kind: unit
        ref: "bench_metrics_tests.rs#bench_metrics_a_degenerate_confusion_matrix_is_refused_rather_than_producing_nans (variant-asserted, 4 shapes + a non-vacuity acceptance)"
        status: pass
      - kind: unit
        ref: "bench_gate_tests.rs#bench_gate_quality_cross_check_case_table_at_the_active_scope (3 degenerate rows -> row_schema_refused)"
        status: pass
    human_judgment: false
  - id: D5
    description: "A class with zero support and zero predictions still occupies its index and scores the shipped zero-division value"
    requirement: "EVAL-01"
    verification:
      - kind: unit
        ref: "bench_metrics_tests.rs#bench_metrics_a_zero_support_class_still_occupies_its_index"
        status: pass
    human_judgment: false
  - id: D6
    description: "The recomputation authors no metric arithmetic of its own and the count expansion is order-independent"
    verification:
      - kind: unit
        ref: "bench_metrics_tests.rs#bench_metrics_the_recomputation_authors_no_metric_arithmetic (source scan of the function body)"
        status: pass
      - kind: unit
        ref: "bench_metrics_tests.rs#bench_metrics_the_recomputation_is_order_independent"
        status: pass
    human_judgment: false
  - id: D7
    description: "The report's residual: line stops conceding what the gate now refuses, still concedes the three that remain, and does not overstate"
    verification:
      - kind: unit
        ref: "setfit_bench_tests.rs#setfit_bench_report_residual_concedes_exactly_what_the_gate_still_cannot_refuse"
        status: pass
      - kind: unit
        ref: "setfit_bench_tests.rs#setfit_bench_report_active_scope_matches_the_case_table (RESIDUAL_DISCLOSURE + PROVENANCE_SOURCES must-match rows)"
        status: pass
    human_judgment: true
    rationale: "The tests prove the retired sentence is gone and the three remaining residuals are each named. Whether the new wording is an ACCURATE account of what the gate enforces — neither overstating nor conceding a refused attack — is a reading of the source against the prose that only a human can confirm, and it is the exact property this plan exists to get right."
  - id: D8
    description: "Finding T-05-16-05 closed: verify-cell's printed scope note enumerates every recomputation the door performs"
    verification:
      - kind: unit
        ref: "setfit_bench_tests.rs#setfit_bench_verify_cell_scope_note_enumerates_every_recomputation_the_door_performs"
        status: pass
    human_judgment: false
  - id: D9
    description: "The committed 40-cell evidence still verifies and no published number moved"
    requirement: "EVAL-01"
    verification:
      - kind: e2e
        ref: "apr setfit bench report --bench-dir benchmarks/tweeteval-stance -> rc=0, 0 cross-check refusals (apr 0.63.0)"
        status: pass
      - kind: e2e
        ref: "apr setfit bench verify-cell --bench-dir ... --method setfit --shots 8 --seed 13 -> rc=0"
        status: pass
      - kind: other
        ref: "regenerated report.json vs committed: cmp -> BYTE-IDENTICAL; report.md diff confined to the verified: and residual: lines"
        status: pass
    human_judgment: false
  - id: D10
    description: "The new rule is falsifiable from the contract alone, validated by pv rather than a bash re-implementation"
    verification:
      - kind: other
        ref: "pv validate contracts/setfit-benchmark-claims-v1.yaml -> 0 errors, 0 warnings (pv 0.63.0)"
        status: pass
      - kind: other
        ref: "pv diff /tmp/claims-old-17.yaml contracts/... -> Suggested bump: minor; RECORDED and NOT taken; applied as 4.0.0"
        status: pass
    human_judgment: true
    rationale: "pv proves the contract validates and scores the bump. Whether quality_closed_form_crosscheck's seven invariants accurately describe what the code does, and whether taking 4.0.0 over the tool's minor is the right call, are readings only a human can confirm."
  - id: D11
    description: "All four Make floors read their suites' measured counts and every banner describes what its leg now carries"
    verification:
      - kind: integration
        ref: "make setfit-bench-tests -> rc=0; row 27, gate 55, metrics 19, apr-cli 67, each against its own raised floor"
        status: pass
      - kind: other
        ref: "grep -n '^\\.SHELLFLAGS' Makefile -> exactly 29 and 57, byte-unchanged; bashrs make lint identical to baseline (1 error / 43 warnings)"
        status: pass
    human_judgment: false

duration: 71min
completed: 2026-09-12
status: complete
---

# Phase 5 Plan 17: Recompute the Published Quality From the Row's Own Confusion Matrix Summary

**Every published accuracy figure a benchmark row carries — `f_avg`, `macro_f1`, `mcc`, the three per-class vectors, `n_test_rows` and all five `_bits` siblings — is now RECOMPUTED in closed form from the counts the row itself records and compared bit for bit, so verifier spot-check D (the headline moved 0.4579 → 0.99 with all three digests repaired, which the verifier measured returning rc=0) is refused at rc=5; and the three places the report's residual was written down stop conceding the attack the gate now catches.**

## Performance

- **Duration:** 71 min
- **Started:** 2026-09-12T00:01:00Z
- **Completed:** 2026-09-12T01:12:40Z
- **Tasks:** 3
- **Files modified:** 11

## Accomplishments

- **Verifier advisory 2 (EVAL-01) is closed, and it cost no new evidence file and no re-run.** The row already carried `confusion_matrix` and `ordered_labels`; nothing read them. Now `quality_from_confusion_matrix` expands the counts back into the index vectors that produced them and routes to the SAME three surfaces `assemble_quality_block` routed to, and both shipped doors compare the result by `to_bits()`.
- **The band was MEASURED before it was chosen, and it is exact.** 40/40 bit-identical on every field over the committed rows, maximum absolute deviation `0e0`. There is no epsilon in this round because none was needed — which is the strongest available statement and the honest one.
- **Eleven mutations were observed RED as `Ok(40 rows verified)`** at the ACTIVE 40-cell scope through the public `verify_run` door — every one of them repairs all three of spot-check D's digests, so the attacks SUCCEEDED against the pre-fix gate, which is the same false green the verifier saw — and GREEN afterwards, each naming its cell AND its field.
- **The disclosure is corrected in all three places and GATED in two of them.** The retired sentence is refused by a must-not-match row; the three residuals that remain true are each required by a must-match row; and `verify-cell`'s printed scope note, which 05-16 flagged as under-claiming (T-05-16-05), now enumerates every recomputation the door performs.
- **Spot-check D refuses through the shipped door.** The probe now runs five cases, each on its own slim copy, with the positive control first.

## Task Commits

1. **Task 1: the synthetic fixtures become closed forms over their own confusion matrices** — `c8f19a333` (test)
2. **Task 2 RED: the quality cross-check case table at the ACTIVE 40-cell scope** — `6c3ae6356` (test)
3. **Task 2 GREEN: the recomputation, the refusal and both call sites** — `692a03ea7` (feat)
4. **Task 2 REFACTOR: the fixture routed through the production recomputation** — `7c030493a` (refactor)
5. **Task 3: the corrected disclosure, the door replay, the contract and the Make floors** — `6cdf44cf6` (test)

**Commits:** 5, MEASURED as `git rev-list --count 49832d05c..HEAD`, not narrated.

## TDD Gate Compliance

| Gate | Commit | Status |
|---|---|---|
| RED | `6c3ae6356` `test(05-17): …` | PASS — the target test failed on an assertion for the planned behaviour |
| GREEN | `692a03ea7` `feat(05-17): …` | PASS |
| REFACTOR | `7c030493a` `refactor(05-17): …` | PASS — tests green before and after; the fixture's own expansion helper and five imports deleted |

The RED is INTENTIONAL, not a nonzero exit: `cargo nextest run … --no-run` was **rc=0** first (the tests compile against the unfixed gate using only symbols that existed pre-fix — they compare `error.variant_tag()` against string literals that simply never match), and the failure was the named target test asserting on the planned behaviour. Not a syntax error, not zero-test discovery, not a fixture crash, not an unrelated assertion.

**`gsd_run check tdd-red-evidence` was NOT used, and that is a measurement rather than an omission.** Its two parsers are Node-TAP-specific: `parseNodeTestSummary` matches `^# tests \d+` and `tapFailedTestNames` matches `^not ok \d+ - `. Counted against this plan's actual RED log, both are **0 and 0**. Feeding it a Rust/nextest run returns `INVALID_RED / no_target_test_failure` on a genuine RED — a false negative — and the only way to get `RED_EVIDENCE_OK` would be to hand it TAP this suite never emits. 05-16 set the same precedent for the same reason.

## THE RED / GREEN EVIDENCE (plan `<output>` requirement)

### 1. THE 40-ROW AGREEMENT MEASUREMENT, taken BEFORE the band was chosen

Verbatim from `bench_metrics_the_forty_committed_rows_agree_with_their_own_confusion_matrices`, `--no-capture`, rc=0:

```
[bench_metrics] COMMITTED_ROW_AGREEMENT rows=40 field=n_test_rows             bit_identical=40/40 max_abs_dev=0 (integer)
[bench_metrics] COMMITTED_ROW_AGREEMENT rows=40 field=f_avg                   bit_identical=40/40 max_abs_dev=0e0
[bench_metrics] COMMITTED_ROW_AGREEMENT rows=40 field=macro_f1                bit_identical=40/40 max_abs_dev=0e0
[bench_metrics] COMMITTED_ROW_AGREEMENT rows=40 field=mcc                     bit_identical=40/40 max_abs_dev=0e0
[bench_metrics] COMMITTED_ROW_AGREEMENT rows=40 field=per_class_precision     bit_identical=40/40 max_abs_dev=0e0
[bench_metrics] COMMITTED_ROW_AGREEMENT rows=40 field=per_class_recall        bit_identical=40/40 max_abs_dev=0e0
[bench_metrics] COMMITTED_ROW_AGREEMENT rows=40 field=per_class_f1            bit_identical=40/40 max_abs_dev=0e0
[bench_metrics] COMMITTED_ROW_AGREEMENT rows=40 field=all_five_bits_siblings  bit_identical=40/40 max_abs_dev=0 (integer)
```

**THE BAND CHOSEN FROM IT: exact IEEE-754 bit equality.** 40/40 agree exactly on every field, so there is no epsilon to justify. The plan's prohibition — "a tolerance must not be chosen because it makes the committed evidence pass" — is satisfied in the strongest available way: the measurement came first, it is a shipped test that reports the per-field agreement count and the maximum deviation on every run, and no committed row was edited. Had any cell disagreed, the recorded instruction was to stop and report the cell rather than widen a band; none did.

Exactness is **structural, not lucky**, and that is why it is safe to gate on: the recomputation expands the counts and feeds the SAME entry points with the same declared class count and the same `f64::from` widening of the same `f32` MCC return. Every metric is a function of the counts alone and all accumulate from integers, so the expansion order cannot change any result — asserted by `bench_metrics_the_recomputation_is_order_independent` rather than argued.

### 2. The twelve-row case table — ONE LINE PER ROW, both observations

Taken through the PUBLIC `verify_run` door over a freshly built valid ACTIVE 40-cell run per row. PRE-FIX is commit `6c3ae6356` with `bench_gate.rs` untouched; POST-FIX is `692a03ea7`. rc=100 then rc=0.

| row | mutation | PRE-FIX (`6c3ae6356`) | POST-FIX (`692a03ea7`) | cell / field named |
|---|---|---|---|---|
| 1 | none (CONTROL) | `Ok(40 rows verified)` | `Ok(40 rows verified)` | — |
| 2 | `quality.f_avg` → 0.99 (**spot-check D**) | **`Ok(40 rows verified)`** | `quality_cross_check_mismatch` | `setfit/s8/seed13` · `f_avg` |
| 3 | `macro_f1` doctored | **`Ok(40 rows verified)`** | `quality_cross_check_mismatch` | `setfit/s16/seed29` · `macro_f1` |
| 4 | `mcc` doctored | **`Ok(40 rows verified)`** | `quality_cross_check_mismatch` | `setfit/s16/seed29` · `mcc` |
| 5 | ONE `per_class_f1` element doctored | **`Ok(40 rows verified)`** | `quality_cross_check_mismatch` | `setfit/s16/seed29` · `per_class_f1[2]` |
| 6 | `n_test_rows` inflated | **`Ok(40 rows verified)`** | `quality_cross_check_mismatch` | `setfit/s16/seed29` · `n_test_rows` |
| 7 | `f_avg_bits` doctored, decimal untouched | **`Ok(40 rows verified)`** | `quality_cross_check_mismatch` | `setfit/s8/seed13` · `f_avg_bits` |
| 8 | THE INVERSION: decimal doctored, bits untouched | **`Ok(40 rows verified)`** | `quality_cross_check_mismatch` | `setfit/s8/seed13` · `f_avg` |
| 9 | `ece_top_label_validation_bits` doctored | **`Ok(40 rows verified)`** | `quality_cross_check_mismatch` | `setfit/s16/seed29` · `…_bits` |
| 10 | ragged (non-square) matrix | **`Ok(40 rows verified)`** | `row_schema_refused` | `setfit/s16/seed29` · `confusion_matrix` |
| 11 | 4×4 matrix against three labels | **`Ok(40 rows verified)`** | `row_schema_refused` | `setfit/s16/seed29` · `confusion_matrix` |
| 12 | all-zero matrix | **`Ok(40 rows verified)`** | `row_schema_refused` | `setfit/s16/seed29` · `confusion_matrix` |

**`Ok` on all eleven is the RIGHT red, and it is the strongest of the three in this class.** Every mutation goes through `reseal_row` (which recomputes the row's own envelope digest) and then `manifest_for` (which records the new `row_sha256` into a freshly derived manifest) — the exact three repairs spot-check D performed. The pre-fix gate therefore had no other reason to refuse and returned `Ok`, which is precisely what the verifier measured through the shipped door. A red that had been `row_digest_mismatch` would have proved the repair was skipped rather than that the cross-check was missing.

Verbatim RED output (`[bench_gate] QUALITY_CROSS_CHECK` lines, `--no-capture`, `why` strings elided for width):

```
case=control                                 observed=Ok(40 rows verified)
case=spot_check_D_f_avg_to_0_99              observed=Ok(40 rows verified)
case=macro_f1_doctored                       observed=Ok(40 rows verified)
case=mcc_doctored                            observed=Ok(40 rows verified)
case=one_per_class_f1_element_doctored       observed=Ok(40 rows verified)
case=n_test_rows_doctored                    observed=Ok(40 rows verified)
case=f_avg_bits_doctored_value_untouched     observed=Ok(40 rows verified)
case=f_avg_decimal_doctored_bits_untouched   observed=Ok(40 rows verified)
case=ece_bits_doctored                       observed=Ok(40 rows verified)
case=non_square_matrix                       observed=Ok(40 rows verified)
case=four_by_four_against_three_labels       observed=Ok(40 rows verified)
case=all_zero_matrix                         observed=Ok(40 rows verified)
```

and the assertion that failed, carrying the same observations independently of the printout:

```
every published quality figure must be recomputable from the row's own confusion matrix
through verify_run at the ACTIVE 40-cell scope, and the control must be ACCEPTED; these
were not: [ "Ok(40 rows verified)|expected=quality_cross_check_mismatch", ×8,
            "Ok(40 rows verified)|expected=row_schema_refused", ×3 ]
```

**Rows 7 and 8 are an INVERTING PAIR and neither is redundant.** Doctoring the bits alone must report `f_avg_bits`; doctoring the decimal alone must report `f_avg`, because the recomputation runs first in the fixed field order. Without row 8, row 7 could pass while the value check was never reached — the same structural argument 05-16's transplant inversion makes.

A second negative carries the SAME spot-check D at the OTHER shipped scope, because scope is part of what a negative proves:

| negative | PRE-FIX | POST-FIX |
|---|---|---|
| `..._the_single_cell_door_also_applies_the_quality_cross_check` | `spot-check D through the single-cell door: ()` — `verify_cell` returned **`Ok(())`** | `quality_cross_check_mismatch` naming `setfit/s16/seed29`, with the untouched cell still ACCEPTED |

### 3. THE CANONICALIZATION, MEASURED AGAIN — AND THIS TIME IT HAS A CONSEQUENCE

05-15 and 05-16 both recorded that `bench_row.rs:37-43`'s digest-scheme comment is false. Re-measured here on the bench rows themselves, before anything depended on it:

| canonicalization of `payload` | reproduces the committed `semantic_hash` |
|---|---|
| compact JSON, **file / declaration order** | **40 / 40** |
| compact JSON, **key-sorted** | **0 / 40** |

**The third measurement found a consequence the first two did not, and it is a real defect.** Writing the 40-row test surfaced it immediately: `BenchRow::from_bytes` REFUSED a committed row inside `cargo test -p aprender-train --lib --features setfit`, computing `fafda648…` where the envelope claims `1c54f3b4…` — and `fafda648…` is exactly the key-sorted digest. Cause, named by measurement rather than inferred: `to_canonical_bytes` serializes through `serde_json::Value`, whose `Map` is an `IndexMap` under `serde_json/preserve_order` and a `BTreeMap` without it; `cargo tree -p apr-cli --features setfit -e features -i serde_json` shows the feature entering through **`pmcp v2.19.3`**, and the same query against `aprender-train` shows it absent.

**So the phase's tamper seal is BUILD-GRAPH DEPENDENT.** The same committed row verifies under the shipped `apr` binary and is refused under the `aprender-train` test binary. Adding or removing a dependency that touches that feature anywhere in a binary's graph — a cargo feature-unification change, with no code change at all — flips the entire committed evidence set between "verifies" and "refused", and an auditor who builds a different binary does not reproduce the guarantee.

**Not fixed here, deliberately and with the reason recorded:** `bench_row.rs` is outside this plan's `files_modified`, and a real fix re-seals 40 rows, 40 selection manifests and the run manifest — a plan of its own. The measurement test routes around it in a way that cannot be lost: it deserializes the `payload` object directly instead of calling `from_bytes`, and its doc comment carries both digests and the `pmcp` attribution. Filed as **D-ITEM-05-17-A** and appended to `WINDOWS.md`.

### 4. The `pv diff` output, and the version bump that was NOT the one it suggested

```
$ git show HEAD:contracts/setfit-benchmark-claims-v1.yaml > /tmp/claims-old-17.yaml
$ target/release/pv diff /tmp/claims-old-17.yaml contracts/setfit-benchmark-claims-v1.yaml
Contract diff: v3.0.0 → v3.0.0
Suggested bump: minor

  equations:
    + quality_closed_form_crosscheck
  proof_obligations:
    + equivalence:OBLIG-CLAIMS-QUALITY-CLOSED-FORM: …
  falsification_tests:
    + FALSIFY-CLAIMS-012
```

`pv 0.63.0`. **The suggestion was recorded verbatim in the file's metadata block and NOT taken:** `metadata.version` 3.0.0 → **4.0.0**. `pv diff` scores the SHAPE of the edit and every element here is additive, which is a minor — but the ACCEPTANCE SET NARROWS AGAIN, and the exact tree spot-check D built VERIFIED at 3.0.0 and is REFUSED now. That is the same reasoning the 3.0.0 metadata block records for taking `pv`'s major when tool and argument agreed, and it does not stop applying because the tool scored this edit differently. Taking the larger bump is also the conservative error: a consumer misled into expecting more change than occurred loses nothing, while one told a narrowing was a minor may not re-verify at all. The disagreement is written into the file rather than smoothed over.

`pv validate contracts/setfit-benchmark-claims-v1.yaml` → **0 error(s), 0 warning(s). Contract is valid.**

### 5. The three-way disclosure agreement

The residual is stated in three places, and all three were changed together.

| where | retired | now says |
|---|---|---|
| `apr setfit bench report`'s `residual:` line | "a producer holding both the rows and those files could still emit a mutually consistent forgery. This report proves consistency, not truth." | the three residuals below, and nothing more |
| `bench_gate.rs` module header, `THE RESIDUAL, STATED RATHER THAN HIDDEN` | the same sentence, unqualified | the same three, with the retired sentence named and marked not-to-be-restored |
| `contracts/…-claims-v1.yaml` `selection_safety_evidence.residual_risk` | its LoRA-ledger statement had become the whole report's residual by 3.0.0 | `amended_4_0_0` scopes the old statement to the deferred LoRA arm (where it stays verbatim and true) and states the same three |

**The three residuals that remain, and which all three now say:** (1) the confusion matrix is itself producer-written, so a producer who edits it and recomputes the metrics from it emits a set nothing distinguishes from a measurement — closing that needs a committed per-row prediction artifact no run writes today; (2) `ece_top_label_validation` and `brier_multiclass_validation` are not recomputable at all, because no committed file carries the per-row probability vectors they need, so those two figures stay self-asserted and only their `_bits` siblings are held; (3) `evidence_table_hash` and `apr_artifact_sha256` remain claims about artifacts the index deliberately does not carry.

**The disclosure is GATED, not merely edited.** `setfit_bench_report_residual_concedes_exactly_what_the_gate_still_cannot_refuse` refuses the retired sentence by its own literal, requires each of the three, and scans for three overclaims. `setfit_bench_verify_cell_scope_note_enumerates_every_recomputation_the_door_performs` closes finding T-05-16-05 in both directions: the note must now name provenance, the selection manifest AND the confusion matrix, and must still not claim the set-level steps it skips.

### 6. The door probe's full five-case PASS output

Run at HEAD `6cdf44cf6` with `apr 0.63.0`, binary pinned through `scripts/apr_bin.sh` (CLAUDE.md rule 3).

```
CONTROL: undoctored slim copy of <repo>/benchmarks/tweeteval-stance verifies (rc=0)
DOCTORED: setfit-s8-seed13 now points at <tmp>/outside/anywhere.json, and the committed lock record is gone
ATTACK: rc=5, refused as a path escape naming <tmp>/outside/anywhere.json
SPOT-CHECK G: rc=5 with selections/ deleted and every row byte untouched, refused as an absent selection manifest
SPOT-CHECK F: rc=5 with setfit-s8-seed13 claiming selection_manifest_hash=0000…0000, refused by the recomputation and NOT at the row digest
SPOT-CHECK D: rc=5 with setfit-s8-seed13 publishing f_avg=0.99 beside an untouched confusion matrix, refused by the cross-check against its own counts and NOT at the row digest
PASS: <repo>/target/release/apr refuses a row-supplied evidence path that leaves the benchmark directory, a deleted selection manifest, a doctored pairing key, and a published metric that does not follow from its own confusion matrix, having first verified the undoctored tree
```

Each case runs on its OWN slim copy, so a later case cannot pass because an earlier one already broke the tree. **The D case leaves the CONFUSION MATRIX alone on purpose** — doctoring it too would produce an internally consistent row that is ACCEPTED, which is the residual the report discloses rather than the attack this case replays.

`bashrs lint scripts/setfit_bench_gate_door_probe.sh` → **0 errors / 9 warnings / 34 infos**; the warning count is byte-identical to the baseline's 9. Three transient IDEM002 hits were the linter matching the substring `rm` inside the word "form" — the same parse artefact 05-16 recorded on "arm" — and were reworded rather than suppressed.

### 7. The measured test-count floors, and where each number came from

Every number was read off its own log's `test result:` line by `make`, which writes those logs itself. None was computed from the plan.

| leg | before | after | the line it was read from |
|---|---|---|---|
| `bench_row` | 27 | **27** (unmoved) | `test result: ok. 27 passed; … 8077 filtered out` |
| `bench_gate` | 52 | **55** | `test result: ok. 55 passed; … 8049 filtered out` |
| `bench_metrics` | 14 | **19** | `test result: ok. 19 passed; … 8085 filtered out` |
| `apr-cli setfit_bench` | 65 | **67** | `test result: ok. 67 passed; … 7241 filtered out` |

`make setfit-bench-tests` → rc=0 against all four. `grep -n '^\.SHELLFLAGS' Makefile` prints exactly `29:.SHELLFLAGS := -o pipefail -c` and `57:.SHELLFLAGS := -e -c`, byte-unchanged and un-moved. `bashrs make lint Makefile` → **1 error / 43 warnings / 0 infos**, byte-identical to the same lint against `git show HEAD:Makefile` before this plan's edit (the one error is the pre-existing `local`-outside-a-function at `dev-setup`).

The gate table's row count is pinned INSIDE the test (`QUALITY_CROSS_CHECK_CASES.len() == 12`, `observations.len() == the table length`, exactly one acceptance row), because `assert_tests_ran` counts test FUNCTIONS and twelve rows collapsing to three would not move the floor by a single test — 05-16's lesson, applied rather than re-derived.

### 8. The `report.md` diff, proving only the disclosure moved

`diff -u benchmarks/tweeteval-stance/report.md <regenerated>` is confined to exactly two blocks: the `verified:` line's source list and the `residual:` paragraph. **No number moved anywhere in the file.** The independent control is stronger than the diff: regenerating `report.json` (`--out`) produces a file **BYTE-IDENTICAL** to the committed one, and that payload is where every statistic lives. The committed `report.md` was refreshed in the same commit — leaving it stale would ship a published report whose disclosure is the retired, now-false sentence, which is the defect this task exists to fix.

`git status --short benchmarks/` is otherwise empty: no row, manifest or selection file was touched.

## Files Created/Modified

- `crates/aprender-train/src/train/setfit/bench_metrics.rs` — `MAX_CROSS_CHECK_ROWS`, `RecomputedQuality`, `quality_from_confusion_matrix`, `expand_confusion_counts`, two new `BenchMetricsError` variants with their `Display` arms; the module's surface table and header extended with the recomputation and the reason it lives here.
- `crates/aprender-train/src/train/setfit/bench_metrics_tests.rs` — the 40-committed-row agreement measurement; the order-independence assertion; the degenerate-matrix variant table; the zero-support-class assertion; the structural scan refusing a second metric definition inside the recomputation.
- `crates/aprender-train/src/train/setfit/bench_gate.rs` — `BenchGateError::QualityCrossCheckMismatch` with its `variant_tag`/`cell()`/`Display` arms; `verify_quality_closed_form`, `cross_check_f64`, `cross_check_vector`, `render_f64`; step 6c wired into both `verify_run_scoped` and `verify_cell`; the module's numbered step list, `verify_cell`'s APPLIES/DOES NOT APPLY enumeration and the residual section all rewritten.
- `crates/aprender-train/src/train/setfit/bench_gate_tests.rs` — `synthetic_confusion`, `synthetic_labels` and a `synthetic_quality` that is a closed form over its own matrix; the dispersion-property test; the twelve-row `QUALITY_CROSS_CHECK_CASES` table and its sweep; the single-cell-door replay; the arm-count guard raised 17 → 18 with its argument; the header inventory extended from twenty to twenty-three entries.
- `crates/apr-cli/src/commands/setfit_bench.rs` — `PROVENANCE_SOURCES` and its two-method sibling extended; `RESIDUAL_DISCLOSURE` introduced and the header's `verified:`/`residual:` block rewritten; `DOOR_SCOPE_NOTE` corrected (T-05-16-05).
- `crates/apr-cli/src/commands/setfit_bench_tests.rs` — the residual must-match row in the ACTIVE case table, plus two dedicated tests gating the disclosure and the door's scope note.
- `contracts/setfit-benchmark-claims-v1.yaml` — 4.0.0; `equations.quality_closed_form_crosscheck` (ACTIVE, seven invariants); `OBLIG-CLAIMS-QUALITY-CLOSED-FORM`; `FALSIFY-CLAIMS-012`; `residual_risk.amended_4_0_0`; the metadata history block recording `pv diff`'s suggestion and why it was not taken.
- `scripts/setfit_bench_gate_door_probe.sh` — the spot-check D replay (five cases), header rewritten.
- `scripts/setfit_bench_gate_doctor.py` — an `f-avg-to-0-99` mode that leaves the confusion matrix alone.
- `Makefile` — four floors raised to measured counts; the gate, metrics and CLI banners rewritten; the door-probe target's prose corrected from one spot-check to four.
- `benchmarks/tweeteval-stance/report.md` — regenerated; the diff is exactly the `verified:` and `residual:` lines.

## Decisions Made

- **The recomputation lives in `bench_metrics`, not in the gate.** That module's header already says "This module computes NOTHING. It routes"; the gate's header already names a second definition of a number as the failure mode ("two definitions of one number disagree eventually and invisibly"). Putting the recomputation in the gate would have made both statements false in the same commit.
- **`verify_quality_closed_form` takes `bench_dir`**, a deliberate departure from the plan's stated `(cell, row)` signature. The shape refusals map to `RowSchemaRefused`, which carries a `path`, and a refusal that could not name the row file an operator has to open would be less actionable than every other refusal in the module. The signature is now symmetric with `verify_provenance` and `verify_selection_binding`.
- **Only ONE gate variant minted (17 → 18).** The three degenerate-matrix shapes reuse `row_schema_refused` — `Vec<Vec<u64>>` is the wire type, so a shape the schema cannot hold IS a schema refusal — and the five `_bits` checks mint nothing, being the same claim about the same row. The arm-count guard carries the argument for the one that was minted and the reasons the rest were not.
- **`MAX_CROSS_CHECK_ROWS` is derived, not fitted.** Two `Vec<usize>` of 10⁶ elements is 16 MB, the same order as the `MAX_EVIDENCE_FILE_BYTES` cap the gate already applies to the bytes the matrix arrives in. The committed rows total 280 observations each — three and a half thousand times below it — so the bound cannot be mistaken for an acceptance band tuned to the evidence.
- **The committed `report.md` was refreshed.** Regenerating a rendering is not editing a row, and the prohibition is about evidence. Leaving it stale would ship a published report whose disclosure is the sentence this task retired.
- **`requirements-completed` is deliberately empty**, consistent with all sixteen prior phase-5 plans: flipping requirement state is the verifier's act, and `05-VERIFICATION.md` calls that "correct process".

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 — Missing Critical] The expansion is bounded; the plan's own threat model under-mitigated T-05-17-07**

- **Found during:** Task 2 (GREEN), writing `quality_from_confusion_matrix`
- **Issue:** The plan's threat register rates "a confusion matrix with enormous counts expanded into index vectors" as `low`/`mitigate`, with the mitigation "the row is bounded by `MAX_EVIDENCE_FILE_BYTES` (16 MB) before it is parsed, and `n_test_rows` is cross-checked against the matrix total; the shape validation rejects a degenerate matrix before any expansion." **None of those three reaches it.** A few hundred bytes of JSON can declare `u64::MAX` in one cell and satisfy the 16 MB cap comfortably; the `n_test_rows` cross-check happens AFTER the expansion in any ordering where the expansion produces the count; and a `u64::MAX` matrix is perfectly square and correctly dimensioned, so shape validation passes it. The expansion would then attempt two allocations of ~10¹⁹ `usize` — a hang or OOM through `apr setfit bench report`, a shipped door.
- **Fix:** the total is accumulated with `saturating_add` (which can neither wrap nor panic) and SHORT-CIRCUITS the moment it crosses `MAX_CROSS_CHECK_ROWS`, so a doctored `u64::MAX` is refused in a few integer comparisons before any allocation. A second `BenchMetricsError` variant, `ConfusionMatrixTooLarge`, carries the lower-bound total and the cap.
- **Files modified:** `crates/aprender-train/src/train/setfit/bench_metrics.rs`
- **Verification:** `bench_metrics_a_degenerate_confusion_matrix_is_refused_rather_than_producing_nans` includes an `above_the_expansion_cap` row asserting the variant by MATCH, not by message substring; the test completes in 0.02s, which is itself the evidence that no allocation was attempted.
- **Committed in:** `692a03ea7`

**2. [Rule 1 — Bug] The plan's `(cell, row)` signature could not name the file a shape refusal is about**

- **Found during:** Task 2 (GREEN)
- **Issue:** `RowSchemaRefused` carries `{cell, path, detail}`, and every other site that produces it passes an absolute path. With the plan's literal signature the only options were a bench-dir-relative spelling (inconsistent with every sibling refusal) or an empty path.
- **Fix:** `verify_quality_closed_form(cell, row, bench_dir)`, matching `verify_provenance` and `verify_selection_binding`. `bench_dir` is used for nothing else, and the doc comment says so.
- **Files modified:** `crates/aprender-train/src/train/setfit/bench_gate.rs`
- **Verification:** the three degenerate rows of the case table assert the refusal names `confusion_matrix`; the refusal's rendered text carries the full row path.
- **Committed in:** `692a03ea7`

**3. [Rule 1 — Bug] The zero-variance test's hardcoded `0.25` delta was a stale fixture artefact**

- **Found during:** Task 1
- **Issue:** `bench_gate_a_zero_variance_delta_set_reports_a_point_estimate_and_no_interval` asserted `degenerate.mean_delta.to_bits() == 0.25_f64.to_bits()`. That `0.25` was the difference between the two hand-written constants the OLD `synthetic_f_avg` returned in its degenerate branch. Once the quality block became a closed form over its own matrix, pinning the literal would have been the same defect this plan closes elsewhere: a published number that no longer follows from the evidence beside it.
- **Fix:** the expected delta is re-derived from the fixture, and the PROPERTY the test is named for — bit-identical across all ten seeds — is now asserted directly in a loop rather than implied by one constant.
- **Files modified:** `crates/aprender-train/src/train/setfit/bench_gate_tests.rs`
- **Verification:** the test is green before and after; `bench_gate_the_synthetic_fixture_keeps_its_two_dispersion_properties` independently asserts both dispersion properties across all methods and shot levels.
- **Committed in:** `c8f19a333`

**4. [Rule 3 — Blocking] The 40-row measurement could not use `BenchRow::from_bytes`**

- **Found during:** Task 2 (GREEN)
- **Issue:** `from_bytes` REFUSED a committed row inside the `aprender-train` test binary. Root cause measured, not guessed: `serde_json/preserve_order` (via `pmcp`) is in `apr-cli`'s graph and absent from `aprender-train`'s, so the same payload canonicalizes in declaration order under one binary and key-sorted under the other. See §3 above — this is the round's most serious finding, and it is `bench_row.rs`'s to fix, not this plan's.
- **Fix:** the measurement deserializes the `payload` object directly into `BenchRowPayload` (which still exercises the schema, including `deny_unknown_fields`) and its doc comment records both digests, the `pmcp` attribution and why the route was taken — so the finding cannot be lost as a silent workaround.
- **Files modified:** `crates/aprender-train/src/train/setfit/bench_metrics_tests.rs`
- **Verification:** the measurement runs and reports 40/40 on every field; the finding is filed as D-ITEM-05-17-A and appended to `WINDOWS.md`.
- **Committed in:** `692a03ea7`

**5. [Rule 1 — Bug] `pv diff` suggested minor and the suggestion was not taken**

- **Found during:** Task 3
- **Issue:** The plan says "Bump `metadata.version` per `pv diff`". The tool suggested **minor**. But the acceptance set narrows: the exact tree spot-check D built verified at 3.0.0 and is refused now — which is the reasoning the 3.0.0 metadata block records for calling an additive-but-narrowing edit a major.
- **Fix:** 4.0.0, with `pv diff`'s output recorded VERBATIM in the metadata block together with the reason the suggestion was not taken. Following the tool here would have contradicted the argument written into this very file two versions ago and understated a narrowing to every consumer.
- **Files modified:** `contracts/setfit-benchmark-claims-v1.yaml`
- **Verification:** `pv validate` → 0 errors, 0 warnings.
- **Committed in:** `6cdf44cf6`

**6. [Rule 1 — Bug] Two of the plan's `<fails_when>` test-count floors could not be satisfied as literally written**

- **Found during:** Task 1 and Task 3
- **Issue:** Task 1's `<fails_when>` requires the `bench_metrics|bench_row` leg to report "fewer than 41" — but Task 1 adds no test to those modules, so the count was exactly 41 and the bar is a floor rather than a growth requirement (satisfied). Task 3's requires the apr-cli leg at ≥66 "for the two disclosure case-table rows" — but rows added INSIDE an existing test function move no count, the same contradiction 05-16 recorded.
- **Fix:** the disclosure assertions were EXTRACTED into their own named test rather than padded into the case table, and a second named test was added for the door's scope note. That is the right shape independently of the count — "the residual says exactly what the gate enforces" is a different property from "the active scope omits the two-method literals", and the failure message now points at the right one. The leg measures 67.
- **Files modified:** `crates/apr-cli/src/commands/setfit_bench_tests.rs`
- **Verification:** 67 passed against a floor of 67; both new tests fail if their literals move.
- **Committed in:** `6cdf44cf6`

**7. [Rule 1 — Bug] Three transient `bashrs` IDEM002 findings in the new probe text**

- **Found during:** Task 3
- **Issue:** `bashrs lint` went 0/9/29 (baseline) → 0/12/34. All three new warnings were IDEM002 "non-idempotent rm" firing on the substring `rm` inside the word **"form"** (in "closed-form") — in a comment and two quoted strings. The same parse artefact 05-16 recorded when it fired on "arm".
- **Fix:** reworded to "counts-based" / "against its own counts" rather than suppressed. Result: **0 errors / 9 warnings / 34 infos** — the warning count byte-identical to baseline.
- **Files modified:** `scripts/setfit_bench_gate_door_probe.sh`
- **Verification:** `bashrs lint` warning count matches the baseline lint of `git show HEAD:scripts/setfit_bench_gate_door_probe.sh`; the probe still PASSes all five cases.
- **Committed in:** `6cdf44cf6`

---

**Total deviations:** 7 auto-fixed (4 bugs, 1 missing critical, 1 blocking, 1 signature correction)
**Impact on plan:** Deviation 1 is a real DoS the plan's own threat model rated as mitigated when it was not. Deviations 5 and 6 are cases where the plan's literal instruction could not be followed without contradicting its own reasoning or its own acceptance criteria. Deviation 4 surfaced the round's most serious finding. No scope creep: the diff is exactly the ten files the plan named plus the regenerated `report.md`.

## Issues Encountered

### 1. FINDING for a future plan: the bench row seal is build-graph dependent

Fully documented in §3 above and in `deferred-items.md` as **D-ITEM-05-17-A**, with the measurement, the named cause (`pmcp v2.19.3` → `serde_json/preserve_order`), and the reason it is not this plan's to fix. It is the most consequential thing this round found, and it was found because the plan required the canonicalization question to be settled by measurement rather than inherited — the third time, and the first with a consequence.

### 2. One workspace failure beyond the 84-cell baseline, proven not ours

`cargo nextest run --profile ci --no-fail-fast --workspace --lib --exclude aprender-gpu --exclude aprender-cuda-edge --exclude aprender-compute` → **81622 tests run, 81537 passed (1 flaky), 85 failed**, against a baseline of 84 at `49832d05c`.

The one beyond baseline is `aprender-test-lib brick::pipeline::tests::test_uuid_v4_generates_unique_ids`. Three independent controls prove it is not ours:

- `git diff --stat 49832d05c..HEAD -- crates/aprender-test-lib/` is **EMPTY** — the file is byte-identical to the commit the baseline was measured at, yet it passed there and fails here. A byte-identical file that changes verdict cannot have been changed by a code change.
- `cargo tree -p aprender-test-lib` declares no dependency on `aprender-train`, `apr-cli` or `aprender-core`. No path exists.
- **Zero** of the 85 failures are in `bench_gate`, `bench_metrics`, `bench_row` or `setfit_bench`, and the per-crate distribution matches the baseline exactly otherwise.

The underlying defect is real and worth a ticket: `uuid_v4()` in that crate is `format!("{:x}{:x}", now.as_nanos(), process::id())` — a timestamp wearing a UUID's name — and the test's verdict is a function of clock resolution. It failed **5/5** on re-run here (deterministic on this box today, not flaky). Filed as **D-ITEM-05-17-B** and appended to `WINDOWS.md`.

The one FLAKY result (`aprender-orchestrate … test_pub_ext_027_git_status_with_staged_files`) passed on retry and inspects the repository's own git staging state, which this execution necessarily perturbs.

### 3. The plan's clippy verification line still CANNOT PASS on this tree — unchanged since 05-15

`cargo clippy -p aprender-train --lib --features setfit -- -D warnings` exits **101**, and zero of the findings are in `aprender-train`; all are in `aprender-compute` and `aprender-present-terminal`, surfaced because `-D warnings` propagates to dependency crates compiled in the same session. The in-scope signal:

```
cargo clippy -p aprender-train --lib --features setfit --no-deps -- -D warnings  →  rc=0
cargo clippy -p apr-cli        --lib --features setfit --no-deps -- -D warnings  →  rc=0
cargo fmt -p aprender-train -- --check                                           →  rc=0
cargo fmt -p apr-cli -- --check                                                   →  rc=0
```

Stated so a reader does not mistake `--no-deps` for the plan's literal command. Widening to `--all-targets` additionally surfaces one PRE-EXISTING finding at `verify_tests.rs:628` (`clippy::search_is_some`), in a file this plan never touched — filed as **D-ITEM-05-17-C**.

### 4. The untouched-neighbour control held

`cargo nextest run -p aprender-core --lib -E 'test(/calibration|claims_stats/)'` → **95 passed**, at the plan's stated bar. This round touched no calibration code, and the control is what would have gone red had the recomputation reached further than intended.

## Known Stubs

None. No file created or modified by this plan contains a hardcoded empty value flowing to output, a placeholder string, or an unwired data source. Two fields — `ece_top_label_validation` and `brier_multiclass_validation` — are deliberately NOT recomputed, which is the opposite of a stub: it is a documented limit, stated in the equation's invariants, in the obligation's "does not cover" list, in the gate's module header and in the report's own `residual:` line.

## Threat Flags

None. Every file touched is covered by the plan's own `<threat_model>`, and the five `mitigate` dispositions are implemented and proven above:

| Threat | Disposition | Evidence |
|---|---|---|
| T-05-17-01 `quality.f_avg` doctored with every digest repaired (spot-check D) | mitigate | table row 2; door probe spot-check D rc=5; verify_cell replay |
| T-05-17-02 `*_bits` doctored either way | mitigate | table rows 7, 8, 9 — the inverting pair plus a calibration sibling |
| T-05-17-03 `n_test_rows` inflated | mitigate | table row 6, compared against the matrix's own total |
| T-05-17-04 the `residual:` line conceding a refused attack, or omitting a real one | mitigate | all three statements corrected together; gated by two dedicated tests |
| T-05-17-07 an enormous confusion matrix expanded | mitigate | **the plan's stated mitigation did not reach it** — see deviation 1; a saturating, short-circuiting cap does |

`T-05-17-05` (the confusion matrix itself doctored) and `T-05-17-06` (the calibration diagnostics doctored) are `accept` and are DISCLOSED in all three places the residual is stated, exactly as the plan required. `T-05-17-08` is `accept` and structurally absent: the cross-check opens no file and spawns no process. No new network endpoint, auth path, file-access pattern or schema change at a trust boundary — the change strictly NARROWS what the gate will accept.

## Next Phase Readiness

- **Phase 5 is complete: 17 of 17 plans.** All three verifier findings the user selected are closed — gap 1 (EVAL-04, path escape) in 05-15, gap 2 (EVAL-02, the selection binding) in 05-16, advisory 2 (EVAL-01, the quality cross-check) here. The four trees verification measured `apr setfit bench report` ACCEPTING are all rc=5 through the shipped door, replayed by one probe with a positive control.
- **Carried forward for a human, in priority order:**
  1. **D-ITEM-05-17-A** — the row seal is build-graph dependent. This is the one that can silently invalidate the whole committed evidence set with no code change, and it needs its own plan (the fix re-seals 40 rows, 40 manifests and the run manifest).
  2. **D-ITEM-05-17-B** — `uuid_v4()` is a timestamp.
  3. **D-ITEM-05-17-C** — one pre-existing clippy finding under `--all-targets`.
  4. The standing workspace clippy/fmt debt (05-15, 05-16), which CLAUDE.md assigns to `toolchain-ceiling.yml`.
- **Both remaining disclosure statements are now load-bearing.** A future plan that strengthens the gate MUST change all three residual statements together; the CLI case table will catch a report that drifts, but nothing yet catches the gate header or the contract drifting from it. That is a candidate for a cross-artifact test — the three are one fact written down three times, and this round is the second time drift between them produced a finding.
- **`D-ITEM-05-15` is unchanged by this round and slightly better positioned.** The cross-check is method-agnostic: it reads a row's own counts and opens nothing, so a restored LoRA arm inherits it with no second implementation and no new evidence file.
- **The next gate to strengthen, if EVAL-01 is reopened,** is the one this round explicitly could not reach: committing the per-row predictions, which would turn residual (1) from `accept` into `mitigate` and is the only thing that would let the report claim the matrix itself was verified.

---
*Phase: 05-benchmark-and-claims-gate*
*Completed: 2026-09-12*

## Self-Check: PASSED

Every file this plan names exists on disk; all six commits (`c8f19a333`, `6c3ae6356`,
`692a03ea7`, `7c030493a`, `6cdf44cf6`, `04e0a91b1`) are reachable from `git log --all`; and
each of the five new symbols — `quality_from_confusion_matrix`, `verify_quality_closed_form`,
`QualityCrossCheckMismatch`, `OBLIG-CLAIMS-QUALITY-CLOSED-FORM`, `FALSIFY-CLAIMS-012` — is
present at HEAD in the file that declares it.
