---
phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
plan: 27
subsystem: gate-verifier
tags: [laya-finetune-gate-v1, numeric_agreement, class-E, V7-a, V7-b, V6-e, fsum, f64, check_f_avg, mutation-proof, OPS-03]

requires:
  - phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
    provides: "08-21 run_field_bindings (5 f_avg rows expected-open for this plan), VerifyPolicy::from_contract_views (the one contract mapping), recompute_nll; 08-15 check_seed_selection / rank_key; 08-14 metrics.py and gate.py"
provides:
  - "aprender-core metrics::fsum: a safe port of CPython math_fsum, bit-identical to Python math.fsum"
  - "aprender-core classification::macro_f1_f64 and mean_f1_over_labels_f64, and calibration::expected_calibration_error_top_label_f64: the gate's f64 exactly-summed paths, added beside the frozen f32 functions"
  - "metrics::f32_bits_tests: a frozen to_bits() snapshot of the pre-plan f32 f1_score (3 modes x 50 inputs) and top-label ECE (30 inputs), committed alone"
  - "metrics.py macro_f1 / f_avg / ece_top_label on math.fsum; metrics.rank_key (gate.rank_key delegates); --write-numeric-cases"
  - "scripts/laya_train/numeric_cases.json (laya-numeric-cases-v1): 23 cases, replayed bit for bit by metrics.py --selftest and verify::tests::gate_numeric_cases_agree_bit_for_bit"
  - "verify: recompute_metrics on the f64 paths (so the margin, the ECE clause and every rank key are f64); recompute_gate (the one Recomputed builder); check_f_avg, f_avg_labels, FAvgSets, F_AVG_STANCE_LABELS; GateContractView.demo.criteria_order to VerifyPolicy.stance_criteria_order"
  - "laya-finetune-gate-v1 4.0.0: numeric_agreement block; ece_top_label, gate_pass and seed_policy.why_quantized text made true; the five f_avg run_field_bindings rows bound; FALSIFY-LAYA-GATE-015"
affects: [08-28, 08-29, 08-30, 08-31, 08-32, aprender-mcp-decide, setfit claims-stats (f32 metrics frozen, not changed)]

actuals:
  tokens: 107600   # chars/4 over the realized diff; ~26400 of it is code/contract/docs, the rest the 325 KB generated numeric_cases.json
  tasks: 3
  commits: 4
plan_head_before: dd67df1bcfe8d5711553a396c43963d66f9cddf2

tech-stack:
  added: []
  patterns:
    - "Exactly-rounded sums as the cross-language agreement mechanism: a correctly rounded sum does not depend on term order or summation tree, so numpy arrays and Rust iterators cannot disagree"
    - "Add, don't swap: a frozen to_bits() snapshot committed alone before the shared file is touched, and the new precision path added beside it"
    - "One frozen case file written by the Python side (byte-reproducible generator) and replayed bit for bit by both languages"
    - "Constructed boundary cases found by exact rational search (Fraction lookups), so an 'exact 1/20 margin' means exactly that"

key-files:
  created:
    - crates/aprender-core/src/metrics/exact_sum.rs
    - crates/aprender-core/src/metrics/f32_bits_tests.rs
    - scripts/laya_train/numeric_cases.json
  modified:
    - crates/aprender-core/src/metrics/mod.rs
    - crates/aprender-core/src/metrics/classification.rs
    - crates/aprender-core/src/calibration.rs
    - crates/aprender-core/src/calibration_tests.rs
    - scripts/laya_train/metrics.py
    - scripts/laya_train/gate.py
    - crates/aprender-decide/src/verify.rs
    - crates/aprender-decide/src/verify/tests.rs
    - contracts/laya-finetune-gate-v1.yaml

key-decisions:
  - "The f64 paths are named functions beside the f32 ones (macro_f1_f64, mean_f1_over_labels_f64, expected_calibration_error_top_label_f64), not a generic over the float type. A generic would route the f32 callers through new code; named functions leave every f32 byte where it was, and the snapshot proves it"
  - "The reviewers' 'exact 1/20 margin' is RATIONAL, not f64-exact. Over every 9-row K=3 input there are 52 pairs whose f64 margin is exactly 0.05, and HEAD's f32 path agrees with all 52. The disagreeing cases are those whose rational margin is 1/20 and whose f64 difference rounds to 0.05000000000000002 (PASS) or 0.049999999999999996 (FAIL). The generator finds them by exact Fraction lookup"
  - "rank_key has one Python implementation: metrics.rank_key. gate.rank_key keeps its non-finite refusal and delegates. metrics.py cannot import gate, because gate imports metrics"
  - "f_avg's null rule is train.py's own: the stance pair (against, favor) exactly when the task's labels equal demo.criteria_order. The verifier reads criteria_order through the typed view into VerifyPolicy.stance_criteria_order"
  - "laya-finetune-gate-v1 is published as 4.0.0. pv diff classifies the amended ece_top_label formula and invariants as MAJOR, the same rule the 3.0.0 text alignment followed. No threshold, seed or recorded outcome moved"
  - "The per-bin confidence-sum mutants are EQUIVALENT, not test gaps. f32-widened confidences of at least 1/K sum exactly in f64, measured on all 287 bin sums of the case file. The cross-bin term sum is where rounding happens, and both term-level mutants are RED. ece_bin_confidence_sums_are_exact_in_f64 keeps the argument checked"

patterns-established:
  - "A gate quantity added to the report needs a numeric_agreement entry, a case-file field and a replay assertion on both sides"
  - "Mutation survivors are measured before being classified: the equivalence claim is itself a test"

requirements-completed: [D-07, D-08]

coverage:
  - id: D1
    description: "metrics::fsum equals Python math.fsum bit for bit, including the half-way case and exact cancellation"
    requirement: D-07
    verification:
      - kind: unit
        ref: "crates/aprender-core/src/metrics/exact_sum.rs#exact_sum_matches_frozen_math_fsum"
        status: pass
      - kind: unit
        ref: "crates/aprender-core/src/metrics/exact_sum.rs#exact_sum_differs_from_a_sequential_sum_where_it_must"
        status: pass
    human_judgment: false
  - id: D2
    description: "The shared f32 f1_score (every Average mode) and top-label ECE return bit-identical results to the pre-plan tree"
    verification:
      - kind: unit
        ref: "crates/aprender-core/src/metrics/f32_bits_tests.rs#f32_metrics_are_bit_identical_to_the_frozen_snapshot"
        status: pass
    human_judgment: false
  - id: D3
    description: "Rust and Python compute macro-F1, ECE, f_avg, margin and rank key bit for bit on 23 frozen cases, and check_gate's margin verdict equals Python's"
    requirement: D-07
    verification:
      - kind: unit
        ref: "crates/aprender-decide/src/verify/tests.rs#gate_numeric_cases_agree_bit_for_bit"
        status: pass
      - kind: other
        ref: "uv run --project scripts/laya_train --frozen python scripts/laya_train/metrics.py --selftest"
        status: pass
    human_judgment: false
  - id: D4
    description: "verify recomputes every f_avg (zero_shot, fine_tuned, each per_seed row, both shift sets) under the trainer's null rule, refusing a forged value or a broken null"
    requirement: D-07
    verification:
      - kind: unit
        ref: "crates/aprender-decide/src/verify/tests.rs#f_avg_forged_refused"
        status: pass
      - kind: unit
        ref: "crates/aprender-decide/src/verify/tests.rs#f_avg_null_rule_enforced"
        status: pass
      - kind: unit
        ref: "crates/aprender-decide/src/verify/tests.rs#every_run_field_is_bound_or_report_only"
        status: pass
    human_judgment: false
  - id: D5
    description: "laya-finetune-gate-v1 4.0.0 declares numeric_agreement per quantity; pv validate 0 errors; the Phase 8 contract audit passes"
    verification:
      - kind: other
        ref: "pv validate contracts/laya-finetune-gate-v1.yaml; make contract-audit-phase8"
        status: pass
    human_judgment: false
  - id: D6
    description: "The deployed artifact still verifies eligible with seed 17 under the f64 recomputation, both fail-closed vectors keep their refusals, and the Python-written run agrees with Rust"
    requirement: D-08
    verification:
      - kind: e2e
        ref: "heavy just laya-verify models/decide/laya-stance-64.apr models/decide/laya-stance-64 data/decide/tweet-stance-64 <snapshot 55cf4c4e>"
        status: pass
      - kind: e2e
        ref: "heavy env LAYA_FAIL_CLOSED_VECTORS=1 LAYA_MODEL_DIR=<snapshot> cargo test -p aprender-decide --release --test fail_closed_vectors"
        status: pass
      - kind: e2e
        ref: "heavy just laya-train-lifecycle (LAYA_LIFECYCLE_KEEP) then heavy cargo test -p aprender-decide --test python_records"
        status: pass
    human_judgment: false

duration: 55min
completed: 2026-09-28
status: complete
---

# Phase 8 Plan 27: Rust-Python Numeric Agreement (Class E) Summary

**Every Laya gate quantity is now one f64 computation with exactly-rounded sums in both languages. These are aprender-core's new `fsum`, `macro_f1_f64`, `mean_f1_over_labels_f64` and `expected_calibration_error_top_label_f64` in Rust, and `math.fsum` in metrics.py. On a frozen 23-case file, including the reviewers' exact-1/20 margins and a rank grid-line ECE, both languages give the same bits for macro-F1, ECE, f_avg, the margin and the rank key. `f_avg` is no longer an unchecked report field. The shared f32 metrics did not move one bit, and the deployed 24a44d7e still verifies eligible with seed 17.**

## Performance

- **Duration:** 55 min
- **Started:** 2026-09-28T20:43:01Z
- **Completed:** 2026-09-28T21:38:08Z
- **Tasks:** 3
- **Files modified:** 12 (3 created, 9 modified)

## Baselines (step 0)

- **BASE:** `dd67df1bcfe8d5711553a396c43963d66f9cddf2`
- **Clippy baseline:** `cargo clippy -p aprender-core --lib --no-deps --message-format=short -- -D warnings` on the unmodified tree gave **1** diagnostic. It is `crates/aprender-core/src/demo/reliable/performance.rs:126:5: error: unreachable expression`, which is pre-existing and outside this plan's files. There were none in classification.rs, calibration.rs or metrics/mod.rs.
- **f32 snapshot commit:** `e5edfef6a`. It changes only `metrics/f32_bits_tests.rs` and the `mod f32_bits_tests;` line in `metrics/mod.rs`, and the file is byte-identical at HEAD. The table covers:
  - 50 f1_score inputs, each in Macro, Micro and Weighted. These are every unit-test input to f1_score, `classification_report` and the evaluator, plus 24 seeded random inputs (N 3..600, K 2..6).
  - 30 top-label ECE inputs: the 5 claims_stats fixture cases, the saturated two-row test and the 24 random inputs (15 bins).

## numeric_agreement: quantity to formula to both implementations to replay test

| Quantity | Formula (f64, exactly-rounded) | Python | Rust | Replayed by |
|---|---|---|---|---|
| macro_f1 | L = sorted(y ∪ pred). F1_c = 2tp/(2tp+fp+fn), or 0 when the denominator is 0. Result = fsum(F1_c) / \|L\| | `metrics.macro_f1` | `aprender::metrics::classification::macro_f1_f64` via `verify::recompute_metrics` | `metrics.py --selftest`, `verify::tests::gate_numeric_cases_agree_bit_for_bit` |
| f_avg | fsum(F1_c for c in S) / \|S\|. S = [against, favor] exactly when the labels equal `demo.criteria_order`, else null | `metrics.f_avg` + train.py `f_avg_labels` | `mean_f1_over_labels_f64` via `verify::check_f_avg` (null rule: `verify::f_avg_labels`) | both of the above, plus `f_avg_forged_refused` and `f_avg_null_rule_enforced` |
| ece_top_label | conf widened f32→f64, first argmax, bin = min(floor(conf × bins), bins−1) in f64. Per bin: acc = fsum(correct)/n_b and conf = fsum(conf)/n_b. ECE = fsum over bins, in bin order, of (n_b/N)·\|acc−conf\| | `metrics.ece_top_label` | `aprender::calibration::expected_calibration_error_top_label_f64` via `recompute_metrics` | both |
| margin | macro_f1(ft) − macro_f1(zs), one f64 subtraction | `gate.evaluate_gate` / train.py | `verify::recompute_gate` (`Recomputed.margin`), judged by `check_gate`; per seed in `check_seed_selection` | both, plus the check_gate verdict |
| rank_key | floor(ece_post × rank_scale), with the integer scale multiplied in | `metrics.rank_key` (`gate.rank_key` delegates) | `verify::rank_key` on the f64 ECE | both |

The case file is `scripts/laya_train/numeric_cases.json`, schema `laya-numeric-cases-v1`, 325 KB, one compact case per line. `metrics.py --write-numeric-cases` reproduces it byte for byte (`cmp` rc 0). It holds:
- `margin_exact_1_20_9row`: counts (3,3,3). Rational margin 1/20, f64 0.05000000000000002, so Python PASSES; HEAD's f32 path gave 0.04999998 and FAILED.
- `margin_exact_1_20_30row`: counts (10,10,10). Rational 1/20, f64 0.049999999999999996, so Python FAILS; HEAD's f32 path PASSED.
- `rank_grid_line_459x3`: search seed 4750. The f64 ECE is 0.03430002953751671 (key 343); HEAD's f32 ECE is 0.034299999475479126 (key 342).
- `random_00`..`random_19`: N 3..600, with case 1 at N = 3. K cycles 2..6 (each value four times). Even-numbered cases carry a zero-shot set.

The Rust replay checks **130 quantities bit for bit**.

## Task Commits

1. **Step 0(b): f32 bit snapshot, committed alone.** `e5edfef6a` (test)
2. **Task 1 (tracer): exact summation and f64 macro-F1 in both languages; the 9-row 1/20 margin agrees through check_gate.** `90a53178c` (feat)
3. **Task 2: f64 ECE and rank key, f_avg recomputed, numeric_agreement and the full case file.** `221556654` (feat)
4. **Task 3: mutation proof on both sides, real evidence, and the ECE bin-sum equivalence guard.** `6aed5c59f` (test)

**Plan metadata:** the docs commit that follows this SUMMARY.

## RED before GREEN

- **Task 1** (`/tmp/p08-27-t1-RED.log`). With HEAD's f32 path in `recompute_metrics`, `gate_numeric_cases_agree_bit_for_bit` failed with:

  ```
  margin_exact_1_20_9row macro_f1: rust 2.1666665375232697e-1 (3fcbbbbba0000000) python 2.1666666666666667e-1 (3fcbbbbbbbbbbbbc)
  margin_exact_1_20_9row margin: rust 4.999998211860657e-2 python 5.000000000000002e-2
  margin_exact_1_20_9row: rust margin pass false, python true
  ```

- **Task 2** (`/tmp/p08-27-t2-RED-ece.log`). With only the verifier's ECE put back on the f32 function, the replay failed on 35 values. On the grid-line case it reported `rank_key: rust 342 python 343`. The file was restored byte-identically.
- **Task 2, f_avg.** `f_avg_forged_refused` and `f_avg_null_rule_enforced` are RED under mutation M6 below. Before this plan the five f_avg leaves were ACCEPTED after mutation: they were the sweep's `EXPECTED-OPEN` rows in 08-21.

## Mutation proof

Each mutation was applied alone by `mutate.py` (scratchpad), tested, then restored and confirmed byte-identical with `filecmp`. The seven rows the plan requires come first; rows 8 and 9 are the term-level ECE mutants that replace the two equivalent ones.

| # | Side | Mutation | Test | Result |
|---|---|---|---|---|
| M1 | Rust | `recompute_metrics` macro-F1 back to f32 `f1_score` | `gate_numeric_cases_agree_bit_for_bit` | **RED** rc 101: 46 values and 2 verdicts disagree |
| M2 | Rust | ECE per-bin **confidence** sums sequential instead of fsum | replay | rc 0: **EQUIVALENT** (see below) |
| M3 | Rust | `metrics::fsum` replaced by a plain left-to-right sum | `metrics::exact_sum` + replay | **RED** rc 101: all 3 exact_sum tests fail, and the replay has 28 values off (e.g. `zero_shot_ece ...aaaa` vs `...aaab`) |
| M4 | Python | `macro_f1` back to `np.mean` | `metrics.py --selftest` | **RED** rc 1: random_04 and random_08 FAIL |
| M5 | Python | ECE per-bin means back to numpy `.mean()` (terms still fsum) | selftest | rc 0: **EQUIVALENT** (see below) |
| M6 | Rust | `check_f_avg` skips the per_seed rows | `f_avg_forged_refused`, `f_avg_null_rule_enforced` | **RED** rc 101: both fail |
| M7 | Rust (shared f32) | f32 `f1_score` Macro routed through `macro_f1_f64` then narrowed to f32 (the "one implementation" refactor) | `f32_metrics_are_bit_identical_to_the_frozen_snapshot` | **RED** rc 101: `f1_score(Macro) on 'report_skewed' moved: got 0x3f38e38e, frozen 0x3f38e38f`. Not bit-identical, so the one-ULP fallback was not needed |
| M8 | Rust | ECE **terms** summed sequentially instead of fsum | replay | **RED** rc 101: 16 values off |
| M9 | Python | ECE back to its full pre-plan numpy form (`.mean()` per bin and `ece += term`) | selftest | **RED** rc 1: margin_exact_1_20_9row, random_00, random_02 and others FAIL |

**Why M2 and M5 are equivalent, measured rather than assumed.** A per-bin confidence sum adds float32 values of at least 1/K, widened to f64. Each carries at most 24 mantissa bits over a narrow exponent range, so any partial sum of fewer than about 2^26 rows is exact in f64. On the case file, all **287 of 287** bin confidence sums are identical under sequential summation and fsum, in Rust and in Python. The rounding happens in the cross-bin term sum instead: it differs in 16 of 35 sets, which is why M8 and M9 are RED. `verify::tests::ece_bin_confidence_sums_are_exact_in_f64` (commit `6aed5c59f`) keeps the argument checked on every case. If a later change feeds inputs where the argument fails, that test fails.

## Real evidence (every leg through `heavy`)

The Task 3 automated verify block ran as written and ended `VERIFY rc=0`.

**Deployed artifact.** Command: `heavy just laya-verify $MAIN/models/decide/laya-stance-64.apr ...`. Result: vrc 0.

```
{"argmax":"459/459","artifact_sha256":"24a44d7e050166c9b64e2716f2bcb3ce91747f7a3b927d03d6eeae5f89b6275a","deploy_eligible":true,"noise":8.33931784072206e-6,"recomputed":{"ece_post":0.04425034660659325,"ft_macro_f1":0.6308225530950676,"margin":0.22166308511207028,"zs_macro_f1":0.40915946798299735},...,"shipped_seed":17,...}
```

- The recomputed numbers moved in the 8th significant digit, because they are f64 now. For comparison, 08-21 printed ece_post 0.04425034672021866 and margin 0.22166302800178528.
- The shipped `gate-report.json` was written by the pre-plan numpy code, so it differs from the new recomputation by 1-2 ULP (zs 0.40915946798299746 vs …735). That is far inside the 1e-5 band.
- Re-running today's `metrics.py` on the deployed run's own probability files gives **exactly Rust's bits** for zs, ft, ECE and margin.
- The per-seed ECEs give rank keys **849 / 442 / 371**, unchanged: no key crossed a grid line. `check_seed_selection` passed because it recomputes the keys and compares them with the reported ones.
- The stance f_avg values (ft 0.6140079160213389, zs 0.37165775401069523, per seed and shift) are recomputed by `check_f_avg` and accepted.

**Fail-closed vectors.** Command: `heavy env LAYA_FAIL_CLOSED_VECTORS=1 ... --test fail_closed_vectors`. Result: crc 0, `FAIL-CLOSED VECTORS REFUSED 2/2 (390 s, ARCH aarch64)`.
- d0f4e40d: `REFUSED RescoreDrift which=fine_tuned row=59 max_abs=0.00004667043685913086 bound=0.00001` (unchanged).
- 3d4b91da: `REFUSED GateFailed clauses=[ece_post] ... margin=0.10561398463002575 ece_post=0.22240783848932813` (the same clause). Its ft_macro_f1 is now 0.4458312929908441, equal to the value gate.py's frozen vector records.

**Python-written run.** Command: `heavy env LAYA_LIFECYCLE_KEEP=$K just laya-train-lifecycle`, then `heavy ... cargo test --test python_records`. Result: prc 0. Output:

```
NOISE which=fine_tuned rust=4.2993464954843574e-8 python=4.2993464954843574e-8
NOISE which=zero_shot rust=3.1062774519252656e-8 python=3.1062774519252656e-8
MEDIAN rust=23 python=23
```

There was no SKIP.

**Clippy.** On aprender-core, the Task 3 check found:
- 0 diagnostics in exact_sum.rs, classification.rs, metrics/mod.rs and calibration.rs;
- `now=1 base=1` outside them.

With the one pre-existing lint allowed (`-A unreachable_code`), aprender-core clippy exits 0. `cargo clippy -p aprender-decide --all-targets --no-deps -- -D warnings` exits 0, as does `cargo fmt -p aprender-decide -p aprender-core -- --check`.

## Other verification

- `cargo test -p aprender-core --lib metrics::` passed 292 (1 ignored: the snapshot printer). `--lib calibration` passed 72.
- `cargo test -p aprender-decide --lib` passed 161. `--tests`: every integration target passes and the armed-only legs SKIP.
- The run-field sweep printed `RUN FIELDS bound=91 report_only=3 expected_open=0 (rows; 121 leaves swept, 0 expected-open leaves listed)`. 08-21 printed bound=86, expected_open=5.
- `pv validate contracts/laya-finetune-gate-v1.yaml` gives `0 error(s), 0 warning(s)`. `pv diff` against the pre-plan file reports v3.0.0 → v4.0.0, suggested bump major (the ece_top_label formula and invariants, the gate_pass invariants, plus FALSIFY-LAYA-GATE-015).
- `make contract-audit-phase8` rc 0.
- `just laya-train-selftest` printed `LAYA TRAIN SELFTEST OK` (metrics, data, gate, lifecycle). `gate.py --selftest` recomputes both fail-closed vectors from their probability files and still decides FAIL.
- `cargo check -p aprender-decide --examples -p aprender-mcp-decide -p aprender-mcp-decide-lambda` rc 0.

## Files Created/Modified

- `crates/aprender-core/src/metrics/exact_sum.rs`: `fsum` and its frozen math.fsum tests.
- `crates/aprender-core/src/metrics/f32_bits_tests.rs`: the frozen f32 snapshot and its printer (ignored).
- `crates/aprender-core/src/metrics/mod.rs`: `pub mod exact_sum; pub use exact_sum::fsum;` and the snapshot `mod`.
- `crates/aprender-core/src/metrics/classification.rs`: `class_f1_f64`, `macro_f1_f64`, `mean_f1_over_labels_f64`, and their tests.
- `crates/aprender-core/src/calibration.rs` and `calibration_tests.rs`: `expected_calibration_error_top_label_f64` and its test.
- `scripts/laya_train/metrics.py`:
  - fsum in `macro_f1`, `f_avg` and `ece_top_label`;
  - `rank_key`;
  - the hex helpers;
  - the rational-exact case search and the f32 emulations (used only to pick cases);
  - `--write-numeric-cases`;
  - the selftest replay.
- `scripts/laya_train/gate.py`: `rank_key` delegates to `metrics.rank_key`.
- `scripts/laya_train/numeric_cases.json`: 23 cases.
- `crates/aprender-decide/src/verify.rs`:
  - the f64 imports and `recompute_gate`;
  - `DemoBlockView` and `VerifyPolicy.stance_criteria_order`;
  - `F_AVG_STANCE_LABELS`, `f_avg_labels`, `FAvgSets`, `check_f_avg` and `f_avg_sets`, wired into `cheap_checks` after `check_shift_probe`.
- `crates/aprender-decide/src/verify/tests.rs`:
  - the replay test;
  - `f_avg_forged_refused` and `f_avg_null_rule_enforced`;
  - `ece_bin_confidence_sums_are_exact_in_f64`;
  - `recompute_matches_house_ece_cases`, now against the f64 house ECE.
- `contracts/laya-finetune-gate-v1.yaml`, now 4.0.0:
  - the metadata paragraph and the house-ECE paragraph;
  - the `numeric_agreement` block;
  - `why_quantized`, the `ece_top_label` formula and invariants, and a `gate_pass` invariant;
  - the 5 f_avg rows bound, with names `field=<path>.f_avg`;
  - FALSIFY-LAYA-GATE-003 wording and the new FALSIFY-LAYA-GATE-015.

## Decisions Made

See key-decisions in the frontmatter.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Correctness of the construction] "Exact 1/20 margin" constructed as a rational margin**
- **Found during:** Task 1, step 4.
- **Issue:** The plan asked for two sets "whose macro-F1s differ by exactly 0.05 in f64-with-fsum" on which HEAD's f32 path disagrees. Exhaustive search over all 9-row K=3 inputs found 52 pairs whose f64 margin is exactly 0.05, and HEAD's f32 path agrees with all 52 (every one has f32 margin > 0.05). The reviewers' disagreement lives where the RATIONAL margin is 1/20 and f64 rounds to one side.
- **Fix:** The generator enumerates confusion matrices and looks up `zs + Fraction(1, 20)` exactly. It picks the first pair on which metrics.py and HEAD's f32 path decide differently: PASS/FAIL for the 9-row case, FAIL/PASS for the 30-row case, matching C2-1 and A5-2.
- **Verification:** Task 1 RED above.
- **Committed in:** `90a53178c`, `221556654`.

**2. [Rule 2 - One implementation] `gate.py` rank_key delegates to `metrics.rank_key`**
- **Found during:** Task 2.
- **Issue:** The case file's rank keys must come from the trainer's own rule. metrics.py cannot import gate (a cycle), and a copy would be a second Python implementation.
- **Fix:** The formula now lives in `metrics.rank_key`. `gate.rank_key` keeps its non-finite refusal and calls it. gate.py was not in the plan's file list.
- **Verification:** `gate.py --selftest` OK; `laya-train-selftest` OK.
- **Committed in:** `221556654`.

**3. [Rule 1 - Test aligned to the new house ECE] `recompute_matches_house_ece_cases`**
- **Found during:** Task 2.
- **Issue:** The test asserted that the verifier's ECE equals the f32 house function bit for bit. The gate's house ECE is now the f64 function.
- **Fix:** The test compares with `expected_calibration_error_top_label_f64`, still within `gate_metric_recompute_abs` of the frozen references. FALSIFY-LAYA-GATE-003's prediction text was changed to match.
- **Committed in:** `221556654`.

**4. [Structural] `recompute_gate` extracted**
- **Found during:** Task 1.
- **Issue:** The replay test must decide the margin exactly as the pipeline does, and the pipeline built `Recomputed` inline.
- **Fix:** `recompute_gate` is now the one builder, used by the pipeline and the test.
- **Committed in:** `90a53178c`.

**5. [Rule 2 - Equivalent mutants replaced, not waved through] M2/M5 → M8/M9 + a guard test**
- **Found during:** Task 3.
- **Issue:** The plan's "ECE bin sums sequential" (Rust) and "ECE back to numpy .mean()" (Python) mutants survived. Measurement showed they are equivalent on every valid input.
- **Fix:** Recorded both as equivalent, with the 287/287 measurement. Added the term-level mutants M8 and M9 (both RED), and the test that keeps the exactness argument checked.
- **Committed in:** `6aed5c59f`.

**6. [Contract versioning] laya-finetune-gate-v1 published as 4.0.0**
- **Found during:** Task 2.
- **Issue:** `pv diff` suggests a MAJOR bump for the amended ece_top_label formula and invariants.
- **Fix:** Bumped to 4.0.0 with a NUMERIC AGREEMENT metadata paragraph, following the 3.0.0 precedent. No consumer pins the version (grep).
- **Committed in:** `221556654`.

**7. [Format] Case file layout and random-case shape**
- **Found during:** Tasks 1-2.
- **Issue:** Pretty-printing would have produced a file of tens of thousands of lines. Also, the first 20 seeded draws never produced K = 2.
- **Fix:** The file has one compact case per line. K now cycles deterministically through 2..6, and case 1 sits at the N = 3 floor. It remains byte-reproducible.
- **Committed in:** `221556654`.

---

**Total deviations:** 7: 2 correctness (rules 1-2), 1 test alignment, 1 structural, 1 mutation-method, 1 versioning, 1 format.
**Impact on plan:** Every must-have truth holds. No recorded gate verdict flipped:
- the s64 demo passes, with the same seed and the same rank keys;
- both vectors refuse with the same kinds and clauses;
- python_records agrees.

The checkpoint condition ("a key lands on the other side of a grid line") did not occur.

## Issues Encountered

- **The seeded search is slow in pure Python.** The rank grid-line case needed search seed 4750, about 1 flip in 4750 sets of 459x3. `--write-numeric-cases` takes about 12 s. The selftest never regenerates; it only replays.
- **Known pre-existing reds, not touched:** the aprender-core `demo/reliable/performance.rs` unreachable-expression lint (the clippy baseline of 1), and the aprender-compute warnings. Both are already in deferred-items.md.

## Known Stubs

None.

## User Setup Required

None. No external service configuration is required.

## Next Phase Readiness

- The numeric half of class E is closed. A new gate quantity needs a `numeric_agreement` entry, a case-file field and a replay assertion on both sides.
- The shipped `gate-report.json` of the deployed run still carries numpy-era values, 1-2 ULP from today's recomputation. That is harmless inside the 1e-5 band. A future re-run of the trainer writes the fsum values Rust recomputes bit for bit.

## Self-Check: PASSED

- FOUND: exact_sum.rs, f32_bits_tests.rs and numeric_cases.json, plus all 9 modified files.
- FOUND: commits `e5edfef6a`, `90a53178c`, `221556654` and `6aed5c59f`.
- `commits: 4` was measured as `git rev-list --count dd67df1bc..HEAD` before this SUMMARY's commit.
- The snapshot file is unchanged since `e5edfef6a` (`git diff --quiet`; clean status).
- `fn fsum`, `fn f32_metrics_are_bit_identical_to_the_frozen_snapshot`, `fn check_f_avg` and `numeric_agreement:` are all present.

---
*Phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server*
*Completed: 2026-09-28*
