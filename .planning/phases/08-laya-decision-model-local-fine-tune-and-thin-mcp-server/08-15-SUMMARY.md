---
phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
plan: 15
subsystem: decision-model-verification
tags: [laya, aprender-decide, verify, rescore, float64-noise, seed-selection, median-ece, shift-probe, fail-closed, gap-closure]
status: complete
gap_closure: true

requires:
  - phase: 08-13
    provides: "laya-parity-v1 2.0.0 (A1 pack_rescore_noise_k 4, pack_rescore_bound_max_abs 1e-3) and laya-finetune-gate-v1 AMENDMENT 1.4.0 / 2.0.0 (A2 shift probe, A3 seed_policy median_ece, demo_s64 pending, rescore_noise_schema)"
  - phase: 08-14
    provides: "the Python writer: rescore-noise.json, seeds/seed-<s>/eval-probs.json + seeds.per_seed, shift probe, LAYA_LIFECYCLE_KEEP run dir"
  - phase: 08-09
    provides: "aprender_decide::verify (cheap_checks, rescore, check_gate, verify_loaded, pack_for_serving, verify_path) and the two fail-closed vectors"
provides:
  - "verify::rescore_bounds: bound(c, s) = max(floor, k x noise) per set, noise RECOMPUTED from the float64 record (bit-for-bit equal to Python's), floor without a record, ceiling refused"
  - "verify::check_seed_selection / select_median_seed / rank_key: the median-ECE seed re-derived from per-seed files; shipped checkpoint and eval file bound by sha256"
  - "verify::check_shift_probe: shift metrics recomputed, never re-scored, never gated"
  - "SeedPolicyMissing decided AFTER the gate (legacy gate-fail keeps GateFailed)"
  - "tests/python_records.rs (cross-language key link) and tests/demo_run.rs (FALSIFY-LAYA-GATE-014, SKIP until 08-16 records demo_s64.outcome)"
  - "D-ITEM-08-14-A/B resolved: gate-report seeds.label 'median-ECE seed of N seeds' under seed selection; GATE-006's legacy multi-seed clause retired with its reason"
affects: [08-16, 08-17, 08-18, 08-12]

actuals:
  tokens: 42011      # chars/4 over the realized diff cc17395a9..HEAD (168,046 chars, 18 files)
  tasks: 3           # plus the orchestrator's extra-scope commit (D-ITEM-08-14-A/B)
  commits: 4         # MEASURED: git rev-list --count cc17395a9..HEAD before this SUMMARY commit
plan_head_before: cc17395a950bca07c167007dc5914043613e20cf

tech-stack:
  added: []
  patterns:
    - "Derived, never reported: a tolerance the run dir could inflate is RECOMPUTED from hash-bound evidence and the reported value is only cross-checked bit-for-bit"
    - "Refusal ORDER is part of the rule: a policy refusal that must not hide a gate failure is decided after the gate"
    - "The fixture writer re-derives what the trainer derives (rehash -> sync_seeds / sync_shift), so a test's induced edit reaches exactly the rule under test"
    - "A cross-language test reads a run dir the OTHER language wrote (LAYA_LIFECYCLE_KEEP) and asserts equality to the bit"

key-files:
  created:
    - crates/aprender-decide/tests/python_records.rs
    - crates/aprender-decide/tests/demo_run.rs
    - crates/aprender-decide/tests/common/mod.rs
  modified:
    - crates/aprender-decide/src/verify.rs
    - crates/aprender-decide/src/verify/tests.rs
    - crates/aprender-decide/src/pack.rs
    - crates/aprender-decide/examples/pack_laya.rs
    - crates/aprender-decide/tests/fail_closed_vectors.rs
    - crates/aprender-decide/README.md
    - crates/aprender-decide/Cargo.toml
    - contracts/laya-finetune-gate-v1.yaml
    - contracts/laya-parity-v1.yaml
    - contracts/aprender/binding.yaml
    - scripts/laya_train/contract.py
    - scripts/laya_train/train.py
    - scripts/laya_train/lifecycle.py
    - scripts/laya_train/README.md
    - .planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/deferred-items.md

key-decisions:
  - "D-ITEM-08-14-A resolved by contract text: gate-report seeds.label is 'median-ECE seed of N seeds' under seed selection, 'single seed' for one seed, and the 1.x 'mean ± sd over N seeds' only for a legacy multi-seed run; variance-report.json keeps 'mean ± sd' (it reports mean and sd). The Rust verifier does not read the label"
  - "D-ITEM-08-14-B resolved by retiring GATE-006's legacy multi-seed clause with its reason in the prediction: no code path writes or can identify a legacy multi-seed run, and every legacy run is refused SeedPolicyMissing once its gate passes"
  - "why_quantized implemented as: refuse when Rust's rank key of its recomputed ECE differs from the rank_key Python reported (not 'within gate_metric_recompute_abs of a grid line', which at 1e-5 against a 1e-4 grid would refuse about 20 % of seeds)"
  - "Report thresholds are checked BEFORE the seed selection (check_thresholds), since per-seed pass is recomputed under the contract's thresholds; ThresholdMismatch therefore now fires before the re-scores"
  - "No contract version bump: pv diff reported both contracts identical after the text fix and after the test:/implemented_by edits (08-09 / 08-14 precedent)"

patterns-established:
  - "rescore_bounds before any model is built: every A1/A2/A3 check is a file check inside cheap_checks, so a forged record costs no re-score"

requirements-completed: [D-07, D-08, D-11, D-17]

coverage:
  - id: D1
    description: "A1 noise-referenced re-score bound: rescore_bounds recomputes noise from the float64 record, derives max(floor, 4 x noise) per set, refuses forged / mismatched / incomplete / argmax-flipping / over-ceiling records, floor without a record; the bound is passed to both re-scores and printed"
    requirement: D-17
    verification:
      - kind: unit
        ref: "cargo test -p aprender-decide --lib verify:: (noise_absent_uses_floor, noise_bound_accepts_what_floor_refuses, forged_noise_value, noise_hash_mismatch, noise_bound_over_ceiling, noise_argmax_flip, noise_rows_incomplete, noise_k_mismatch)"
        status: pass
      - kind: other
        ref: "4 guard mutants (skip recompute, skip ceiling, skip argmax, use reported bound) each turned its named test RED"
        status: pass
    human_judgment: false
  - id: D2
    description: "A3 median-seed re-derivation and the legacy refusal order; A2 shift probe recomputed"
    requirement: D-08
    verification:
      - kind: unit
        ref: "cargo test -p aprender-decide --lib verify:: (seed_median_ships, seed_shipped_not_median, seed_tie_break_smaller_seed, seed_probs_hash_mismatch, seed_metric_forged, seed_checkpoint_not_shipped, seed_policy_mismatch, legacy_policy_refused_after_gate_pass, legacy_gate_fail_keeps_gate_failed, shift_probe_metric_forged, shift_file_missing)"
        status: pass
      - kind: other
        ref: "4 mutants (median index, tie-break direction, legacy refusal before the gate, shift recompute skipped) each turned its named test RED"
        status: pass
    human_judgment: false
  - id: D3
    description: "Cross-language key link: a Python-written 1.4.0 run dir (LAYA_LIFECYCLE_KEEP) parses in Rust; Rust noise == Python max_abs bit-for-bit on both sets; Rust median == Python shipped; shift probe recomputes"
    requirement: D-17
    verification:
      - kind: integration
        ref: "LAYA_PY_RUN_DIR=$K/run LAYA_PY_DATA_DIR=$K/data cargo test -p aprender-decide --test python_records python_run_dir_records_match_rust"
        status: pass
    human_judgment: false
  - id: D4
    description: "Both 1.x fail-closed vectors keep their documented refusals on real weights under the new code, at the 1e-5 floor; full-model parity and the tiny golden unchanged"
    requirement: D-07
    verification:
      - kind: integration
        ref: "LAYA_FAIL_CLOSED_VECTORS=1 LAYA_MODEL_DIR=<snapshot> cargo test -p aprender-decide --release --test fail_closed_vectors demo_vectors_are_refused_fail_closed (2/2, 512 s)"
        status: pass
      - kind: integration
        ref: "LAYA_MODEL_DIR=<snapshot> LAYA_LADDER_BIN=<spike-025 ladder> cargo test -p aprender-decide --release --test laya_parity full_model_reproduces_spike_025_fixture"
        status: pass
      - kind: unit
        ref: "cargo test -p aprender-decide --lib artifact::determinism + verify::tests::fixture_bytes_matches_golden (37d65159)"
        status: pass
    human_judgment: false
  - id: D5
    description: "FALSIFY-LAYA-GATE-014 test (demo_run.rs) compiles, SKIPs unarmed, reports the pending outcome armed, and fails on an out-of-range outcome or a missing declared run dir"
    requirement: D-11
    verification:
      - kind: integration
        ref: "LAYA_DEMO_RUN=1 LAYA_MODEL_DIR=<snapshot> cargo test -p aprender-decide --release --test demo_run (SKIP: demo_s64.outcome is pending)"
        status: pass
    human_judgment: false
  - id: D6
    description: "D-ITEM-08-14-A/B contract text fixes (seeds label, GATE-006 retired clause), bindings, test lines, README wording"
    verification:
      - kind: other
        ref: "pv validate 0 errors x2; make contract-audit-phase8 exit 0; just laya-train-selftest OK; just laya-fixtures byte-identical; strict-binding guard lifted copy 680 refs, only the two pre-existing contracts dangle"
        status: pass
    human_judgment: true
    rationale: "Whether the retired-clause reason and the new label read honestly to a human is a wording judgment no test makes"

duration: 36min
completed: 2026-09-27
---

# Phase 8 Plan 15: Rust verifier for A1/A2/A3 Summary

**`pack` and `verify` now hold each re-score to `max(1e-5, 4 x noise)`. The noise is recomputed in Rust from the float64 record, and on a Python-written run dir it matches the trainer's value to the bit. Both paths re-derive the median-ECE seed from the per-seed files and recompute the shift probe. A legacy run is refused `SeedPolicyMissing` only after its gate. On real weights, both 1.x fail-closed vectors keep their exact refusals at the 1e-5 floor.**

## Performance

- **Duration:** about 36 min, from 2026-09-27T17:27:06Z to 18:03Z.
- **Tasks:** 3 of 3, plus the orchestrator's extra-scope commit for D-ITEM-08-14-A/B.
- **Files:** 18 changed (3 created).
- **Real-weights wall time:** fail-closed vectors 512 s in release; they ran alongside debug builds, against 413 s in 08-09. Full-model parity took 9.7 s of test time.

## Accomplishments

- **A1, `verify::rescore_bounds`.**
  - The contract checks: schema, `float64`, k and floor equal to laya-parity-v1, control exactly 0.0 on rows `0..min(5, n)`, and exactly `fine_tuned` then `zero_shot`.
  - The row checks: every row appears once and in order, with K finite components in [0, 1] summing to 1, and the float64 argmax equals the stored float32 argmax.
  - It recomputes `noise = max |f64(p32) - p64|`, requires the reported `max_abs`, `bound` and `argmax_agree` to equal the recomputation bit-for-bit, and refuses a derived bound above 1e-3.
  - With no record, both sets get the floor.
  - It runs inside `cheap_checks`, so every bound is known before any model is built. Each re-score receives its own bound, and `RescoreDrift` prints the bound used.
- **A3, `check_seed_selection`, `select_median_seed` and `rank_key`.**
  - The declaration must equal the contract's `seed_policy`.
  - The seeds block must be consistent: declared seed, policy, n, and the seeds in order.
  - The shipped row must bind `checkpoint/model.safetensors` and `eval-probs.json` by sha256.
  - Every seed's metrics are recomputed from its hash-bound file. Macro-F1, ECE and margin must agree within `gate_metric_recompute_abs`, and pass and rank key must match exactly.
  - `shipped` must be the median of the RECOMPUTED ECEs.
- **A2, `check_shift_probe`.** `shift.jsonl` must be present and hash-bound, with no normalized-text overlap with train. Both files are validated and every reported metric is recomputed. The probe is never re-scored and never reaches `check_gate`.
- **Refusal order.** `SeedPolicyMissing` (exit 2) comes after the re-scores and the gate. `check_thresholds` now runs before the seed check.
- **CLI.** `pack_laya` reads `pack_rescore_noise_k`, `pack_rescore_bound_max_abs` and `seed_policy` from the contracts. It prints `rescore_bound`, `zs_rescore_bound`, `noise`, `zs_noise` and `shipped_seed` on the `PACKED` line and in the verify JSON.
- **Extra scope (orchestrator).** D-ITEM-08-14-A and -B are resolved by contract text before 08-16 reads anything (commit `c99761825`).

## New VerifyError variants and exit codes

| Variant | Exit |
|---|---|
| `RescoreNoiseInvalid` | 2 |
| `RescoreNoiseMismatch` | 2 |
| `RescoreBoundCeiling` | 2 |
| `NoiseArgmaxFlip` | 2 |
| `SeedPolicyViolated` | 2 |
| `SeedPolicyMismatch` | 2 |
| `SeedPolicyMissing` | 2 |
| `ShiftProbeMismatch` | 2 |
| `GateFailed` (unchanged) | 3 |

- A missing or edited per-seed file surfaces as `Pack(PackError::SeedProbsHashMismatch { seed })`, exit 2.
- `ReportedMetricMismatch` gained `seed: Option<i64>`.
- `RescoreDrift` gained `bound`.

## Evidence

**Task 1: cross-language key link.** The lifecycle ran with `LAYA_LIFECYCLE_KEEP` (three seeds, shift probe, noise record):

```
NOISE which=fine_tuned rust=4.2993464954843574e-8 python=4.2993464954843574e-8 bound=1e-5
NOISE which=zero_shot rust=3.1062774519252656e-8 python=3.1062774519252656e-8 bound=1e-5
```

Both noises are equal to Python's to the bit (asserted with `to_bits`), and so are the bounds.

**Task 2: cross-language, re-run on a fresh keep dir:**

```
MEDIAN rust=23 python=23
SHIFT probe recomputed (gate_clause false)
```

**Task 3: the 1.x vectors on real weights.** Base snapshot `55cf4c4e…`, data `data/decide/tweet-stance-16`, release build:

```
VECTOR d0f4e40d pack: REFUSED RescoreDrift which=fine_tuned row=59 max_abs=0.00004667043685913086 bound=0.00001
VECTOR d0f4e40d bound used: fine_tuned=0.00001 zero_shot=not reached (the A1 floor: no rescore-noise.json)
VECTOR d0f4e40d verify: REFUSED RescoreDrift which=fine_tuned row=59 max_abs=0.00004667043685913086 bound=0.00001
VECTOR 3d4b91da pack: REFUSED GateFailed clauses=[ece_post] zs_macro_f1=0.34021732211112976 ft_macro_f1=0.445831298828125 margin=0.10561397671699524 ece_post=0.22240783274173737 rescore_max_abs=0.000006735324859619141 zs_rescore_max_abs=0.0000068247318267822266 rescore_bound=0.00001 zs_rescore_bound=0.00001 noise=null zs_noise=null shipped_seed=null argmax=280/280 packed_sha256=e397228192960328e974c657baad37daa3401f31beabfec07fb71995e387e217
VECTOR 3d4b91da bound used: fine_tuned=0.00001 zero_shot=0.00001 (the A1 floor: no rescore-noise.json)
VECTOR 3d4b91da verify: REFUSED GateFailed clauses=[ece_post] ... (identical, same packed_sha256)
FAIL-CLOSED VECTORS REFUSED 2/2 (512 s, ARCH aarch64)
```

- Every value is identical to 08-09's record: row 59, 4.667e-5, ece_post 0.2224078, the same packed sha. `SeedPolicyMissing` is never reached, as FALSIFY-LAYA-GATE-010 predicts.
- The `models/decide` listing is unchanged (`cmp`).

**Full-model parity (unchanged):** `ids 14/14; argmax 14/14; max |dp| 3.841e-6 (bar 1e-5); max |dlogit| 2.146e-5 (bar 1e-4); truncated rows 1; ARCH aarch64` and `ladder: 32 blocks within bars`.

**Tiny golden 37d65159 is unchanged.** `artifact::determinism` (4 tests) and `verify::tests::fixture_bytes_matches_golden` pass. Every addition to the report and recipe structs is optional, and the gate report is embedded as bytes, so the artifact bytes do not move.

**Guard mutants.** Each was applied from a backup, run, and restored; `cmp` confirmed the restore.

| # | Mutant | Test that went RED |
|---|---|---|
| M1 | noise := reported `max_abs` (skip recompute) | `forged_noise_value`: an inflated record was accepted with bound 4e-5 |
| M2 | ceiling check skipped | `noise_bound_over_ceiling`: accepted at bound 1.2e-3 |
| M3 | float64 argmax check skipped | `noise_argmax_flip`: refused `RescoreBoundCeiling` instead |
| M4 | bound := reported `bound` | `forged_noise_value`: a zs bound forged to 5e-4 was accepted |
| M5 | median index → `keyed[0]` | `seed_median_ships`: "median-ECE seed ... is seed 13" |
| M6 | tie-break → larger seed first | `seed_tie_break_smaller_seed`: got 23, want 13 |
| M7 | legacy refusal moved before `check_gate` | `legacy_gate_fail_keeps_gate_failed`: exit 2, not 3 |
| M8 | shift recompute comparison skipped | `shift_probe_metric_forged`: a forged ece_post was accepted |

**TDD RED evidence (Task 2).** The tests were written first. `cargo test -p aprender-decide --lib verify::` then failed to compile with 29 errors, each naming a missing piece of the new API:
- `select_median_seed` ×4, `check_seed_selection`, `check_shift_probe`;
- the four `VerifyPolicy` seed fields;
- `SeedPolicyViolated` ×3, `SeedPolicyMismatch`, `SeedPolicyMissing`, `ShiftProbeMismatch` ×2;
- `ReportedMetricMismatch.seed`, `PackError::SeedProbsHashMismatch`, `ProbsWhich::Seed` and `VerifyReport.shipped_seed` ×2.

After implementation all 11 passed with no test edits, and M5 to M8 show that each discriminates.

**Acceptance checks**
- `grep -c 'pack_rescore_noise_k\|pack_rescore_bound_max_abs' examples/pack_laya.rs` = 3.
- `grep -c SeedPolicyMissing verify.rs` = 6. The refusal sits after the gate in `verify_loaded` (line 2280 `let clauses = check_gate(...)?;`, then the GateFailed return, then line 2300 `return Err(VerifyError::SeedPolicyMissing);`):

```rust
    // seed_policy.legacy_rule: AFTER the re-scores and the gate, so a legacy run whose gate
    // fails still reports GateFailed (the 1.x fail-closed vectors keep their refusals).
    if inputs.recipe.seed_selection.is_none() {
        return Err(VerifyError::SeedPolicyMissing);
    }
```

- No file under `models/decide/` or `data/decide/` was written: `git status --porcelain .planning/spikes/ data/ models/` is empty, no file there is newer than the 08-14 SUMMARY, and `models/decide/laya-stance-64` is absent.
- `demo_run.rs`: unarmed prints `SKIP: demo run not armed`. Armed, it prints `SKIP: demo_s64.outcome is pending (plan 08-16 records it)`. Two induced negatives were run on a backed-up contract, which was then restored: `outcome: bogus` fails as "not one of demo_s64.outcome_values", and `outcome: gate_fail` fails as "declared run dir ... missing" and creates nothing.

**Verification run**

| Check | Result |
|---|---|
| `cargo test -p aprender-decide --lib` | 112 passed (93 before this plan) |
| `cargo test -p aprender-decide -p aprender-mcp-decide -p aprender-mcp-decide-lambda --lib` | 112 + 25 + 27 passed |
| Unarmed `demo_run`, `python_records`, `fail_closed_vectors`, `laya_parity`, `ui` | each SKIPs or passes, rc 0 |
| `cargo clippy -p aprender-decide --all-targets --no-deps -- -D warnings` | rc 0 |
| `cargo fmt -p aprender-decide -- --check` | rc 0 |
| `pv validate` on laya-finetune-gate-v1 and laya-parity-v1 | 0 errors, 0 warnings |
| `pv diff` against HEAD copies | "Contracts are identical." on both, so no bump |
| `make contract-audit-phase8` (via `rtk proxy`) | `4 contract(s) audited, every equation is bound` |
| Strict-binding guard | VACUOUS, so the lifted copy ran: `Resolved 680 test references; 44 dangling across 15 contract(s)`. Only the pre-existing chronos-bolt-parity-v1 (9) and setfit-encoder-conformance-v1 (8) FAIL. `.pv/lint-previous.json` is byte-identical |
| `just laya-train-selftest` / `just laya-fixtures` (extra-scope commit) | `LAYA TRAIN SELFTEST OK` / `FIXTURES OK`, no crates/ diff |

## Task Commits

0. **Extra scope, D-ITEM-08-14-A/B:** `c99761825` (fix). Seeds label and GATE-006 retired clause.
1. **Task 1 (tracer), A1 bound end to end:** `3cfa83e1b` (feat).
2. **Task 2, seed re-derivation, shift probe, legacy order:** `6389ec3d5` (feat).
3. **Task 3, real-weights vectors, GATE-014 test, README, bindings:** `3ef728427` (test).

The Task 1 tracer gate was auto-continued: interactive mode, `end-of-phase`, and an automated-only verify, which re-ran green before expansion.

## Files Created/Modified

- `crates/aprender-decide/src/pack.rs`:
  - the noise record structs, the 1.4.0 recipe/report schema, and `ReportInputsSha256`;
  - reads and hash checks for the per-seed and shift files, plus `checkpoint_sha256` and `SeedProbsHashMismatch`.
- `crates/aprender-decide/src/verify.rs`:
  - `rescore_bounds`, `check_thresholds`, `rank_key`, `select_median_seed`, `check_seed_selection`, `check_shift_probe`;
  - the policy fields and variants, `ProbsWhich::{Seed, ShiftFineTuned, ShiftZeroShot}`, and `opt_f64` / `opt_i64`;
  - the per-set bounds and the post-gate `SeedPolicyMissing`.
- `crates/aprender-decide/src/verify/tests.rs`: 19 new tests. `production_copy` is now a three-seed median copy, and `legacy_copy` keeps the 1.x shape. `rehash` re-derives the seed and shift records.
- `crates/aprender-decide/examples/pack_laya.rs`: policy from the contracts; bounds, noises and the shipped seed printed.
- `crates/aprender-decide/tests/{python_records,demo_run}.rs` (new), plus `tests/common/mod.rs` (the contract policy for the new targets).
- `crates/aprender-decide/tests/fail_closed_vectors.rs`: floor bounds asserted and printed; no-record and no-seed-selection asserted.
- `crates/aprender-decide/README.md` and `Cargo.toml` (`serde_json` `float_roundtrip`).
- `contracts/laya-finetune-gate-v1.yaml`: the label, the GATE-006 retirement, and `test:` on GATE-011..014.
- `contracts/laya-parity-v1.yaml`: `test:` on PARITY-006.
- `contracts/aprender/binding.yaml`: three rows implemented; `pack_rescore_probs_abs` notes updated.
- `scripts/laya_train/{contract,train,lifecycle}.py` and `README.md`: the settled label.
- `deferred-items.md`: A and B resolved.

## Decisions Made

See `key-decisions`. The why_quantized reading needs a word:
- The contract invariant says an ECE "within the recompute tolerance of a rank_scale grid line is refused rather than guessed". Read literally with `gate_metric_recompute_abs` (1e-5) against a 1e-4 grid, that refuses about 20 % of seeds.
- `why_quantized` itself says the tolerance is the Python-vs-Rust difference, which is well under 1e-5.
- So Rust refuses exactly when its rank key of the recomputed ECE differs from the `rank_key` Python reported. That is the only case where the two sides would order the seeds differently.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] The 1.4.0 parse-level schema moved into Task 1**
- **Found during:** the Task 1 cross-language run.
- **Issue:** the Python run dir was refused `recipe.json: unknown field seed_selection` (`deny_unknown_fields`), so the tracer's cross-language leg could not run until Task 2's struct fields existed.
- **Fix:** Task 1 added the optional fields (`Recipe.seed_selection`, `GateSeeds.policy/shipped/per_seed`, `GateReport.shift_probe`, `inputs_sha256.shift_jsonl` via the new `ReportInputsSha256`). Task 2 added their checks.
- **Commit:** `3cfa83e1b`.

**2. [Rule 2 - Correctness] `serde_json` `float_roundtrip` declared on aprender-decide**
- **Issue:** the bit-for-bit noise claim needs correctly rounded parsing of the float64 record. The feature was already on, but only by workspace unification.
- **Fix:** declared it explicitly. It changes no build and no Cargo.lock. Cargo.toml was not in `files_modified`.
- **Commit:** `3cfa83e1b`.

**3. [Rule 1 - Ordering] `check_thresholds` runs before the seed check**
- **Issue:** per-seed `pass` is recomputed under the CONTRACT's thresholds. A report with foreign thresholds would otherwise be refused `SeedPolicyViolated per_seed.pass` rather than `ThresholdMismatch`.
- **Fix:** the threshold equality check runs once in `cheap_checks` and is kept in `check_gate`. `ThresholdMismatch` now fires before the re-scores, which is cheaper and still fail-closed. `threshold_mismatch` is unchanged and green.
- **Commit:** `6389ec3d5`.

**4. [Scope] Checks beyond the plan's list, all exit 2**
- In the noise record: `control_rows == 0..min(5, n)` and `argmax_agree == n`.
- In the seeds block: `seeds.declared == recipe.seed == seed_selection.seeds[0]`; a legacy recipe whose report still carries policy, shipped or per_seed is refused; `pass` and `rank_key` are recomputed per seed.
- In the shift probe: `shift.jsonl` must not overlap train (a contract invariant); `shift_probe` and `inputs_sha256.shift_jsonl` must be present together; `gate_clause` must be false.

**5. [Scope] Test and fixture structure**
- `GateFailure` also carries `shipped_seed`.
- GATE-011's `test:` also names `seed_checkpoint_not_shipped`, because its prediction claims that refusal.
- A new `tests/common/mod.rs` is shared by `python_records.rs` and `demo_run.rs`.
- `production_copy` became the three-seed median copy, so every accept path runs the 1.4.0 rule, and `legacy_copy` is the old shape.

**6. [Not done, recorded] Values that are not recomputed**
- Per-seed and shift `f_avg` are not recomputed. It is not a gate metric, and the verifier has never recomputed the top-level `f_avg`.
- The noise record's `t_applied` is not cross-checked. It plays no part in the bound.

---

**Total deviations:** 3 auto-fixed (1 blocking, 1 correctness, 1 ordering), plus scope notes. **Impact:** no threshold, bar, seed list, tie-break, recipe value or demo value moved. The two vectors' refusals are bit-identical to 08-09.

## TDD Gate Compliance

- Task 2 (`tdd="true"`) captured RED as a compile failure against the new API (29 errors, listed above). GREEN required no test edits.
- RED was not committed separately: the per-task commit protocol makes one commit per task, and `workflow.tdd_mode` is false, so this gate is advisory.
- Discrimination is shown by M5 to M8.

## Issues Encountered

- The rtk hook truncated `make contract-audit-phase8` output inside the redirected log itself (`... (66 lines truncated)`). Re-running through `rtk proxy make` gave the real audit line.
- `crates/aprender-decide/src/verify.rs` is now 2454 lines (it was 1451). It is not split here, since that would be a structural refactor outside this plan. Plan 08-12's sweep may want to move the A1/A3 sections into `verify/` submodules.

## Known Stubs

None. `demo_run.rs` SKIPping on `outcome: pending` is the declared staging: plan 08-16 writes `demo_s64.outcome` and arms it.

## Threat Flags

None beyond the plan's register. T-08-15-01, -02, -04 and -05 are mitigated as planned and proven by the named tests and M1 to M8. T-08-15-03 (fabricated non-shipped seed files) stays accepted per `seed_policy.retention_rule`: those files are hash-bound and recomputed, and the shipped seed is fully re-scored.

## User Setup Required

None.

## Next Phase Readiness

- **08-16:** the declared s64 run will be judged by code that already matches the declaration:
  - noise bound per set;
  - median re-derived from its per-seed files;
  - shift probe recomputed;
  - label `median-ECE seed of 3 seeds`.
- After the run, 08-16 writes `demo_s64.outcome` / `outcome_record` (`shipped_seed`; `failed_clauses` for gate_fail; `refusal` for pack_refused) and arms `LAYA_DEMO_RUN=1 LAYA_MODEL_DIR=<snapshot> cargo test -p aprender-decide --release --test demo_run`.
- **08-12:** `python_records`, `demo_run`, `fail_closed_vectors` and `laya_parity` all compile and SKIP in CI. Whether to wire them in is 08-12's call.

## Self-Check: PASSED

- FOUND: `tests/python_records.rs`, `tests/demo_run.rs`, `tests/common/mod.rs`, `src/verify.rs` (`fn rescore_bounds`), `src/pack.rs` (`rescore_noise_sha256`).
- FOUND commits: `c99761825`, `3cfa83e1b`, `6389ec3d5`, `3ef728427` (`git rev-list --count cc17395a9..HEAD` = 4). None deletes a tracked file.
- Every task `<verify>` re-ran green: T1 (lib 34, armed python_records, golden), T2 (lib 45, armed python_records with MEDIAN), T3 (vectors 2/2, unarmed SKIPs, armed demo pending, clippy, fmt, three crates' lib, pv x2, bindings, lifted guard).

---
*Phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server*
*Completed: 2026-09-27*
