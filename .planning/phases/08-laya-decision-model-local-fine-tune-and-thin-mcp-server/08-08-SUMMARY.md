---
phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
plan: 08
subsystem: training
tags: [laya, modernbert, fine-tune, calibration, temperature-scaling, early-stopping, ece, gate, fail-closed, seeds, variance, uv, torch, mps, tweeteval-stance]

requires:
  - phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
    provides: "08-01 laya-finetune-gate-v1 (constants, recipe, base pin, run_dir_layout, schemas); 08-02 pinned scripts/laya_train uv project, metrics.py, laya_tiny fixture; 08-05 packer (Recipe / GateReport readers, deny_unknown_fields)"
provides:
  - "scripts/laya_train/train.py: fine-tune -> F16 save -> reload -> calibrate -> zero-shot/eval/probes -> gate CLI (exit 0 pass / 3 fail / 2 refused), --stopping early_stopping|fixed_epochs, --seeds N"
  - "laya-finetune-gate-v1 1.2.0: demo.outcome gate_fail, demo.fail_closed_vectors (d0f4e40d / 3d4b91da), fail_closed_rule for 08-09, deploy DEFERRED, pass_pending -> the calibration spike; test_harness 'just laya-train-selftest' on FALSIFY-LAYA-GATE-003..007, 009"
  - "gate.verify_report / GateError, gate.EarlyStopper, gate.fit_temperature, gate.evaluate_gate; data/gate/metrics --selftest (numpy + pyyaml only)"
  - "just laya-train-selftest (METRICS / DATA / GATE SELFTEST OK + LIFECYCLE OK) and just laya-train-lifecycle (both stopping rules, --seeds refusals, --seeds 3, separate-reload re-score)"
  - "variance-report.json (laya-variance-report-v1): per-seed rows, mean / sd (ddof 1); only the declared seed's checkpoint is kept"
  - "models/decide/tweet-stance-16/ and models/decide/tweet-stance-16-fixed-epochs/ (gitignored): the two FAIL-CLOSED demo vectors"
  - "models/decide/tweet-stance-16-var/ (gitignored): the 3-seed variance measurement of the FAILING early_stopping recipe"
affects: [08-09, 08-10, 08-11, 08-12]

actuals:
  tokens: 46582    # chars/4 over the realized non-.planning diffs of all six 08-08 code/contract commits (186331 chars)
  tasks: 2         # Task 1 tracer (gate outcome FAIL, recorded under option 3) and Task 2 expansion
  commits: 3       # MEASURED this continuation: git rev-list --count ad5a04bf8..HEAD before the SUMMARY commit
plan_head_before: 06802c0ad0bbddc7b5d9cf33da7988a0a1f45b5c   # first 08-08 session (the range also holds 08-07 and other docs commits)
continuation_head_before: ad5a04bf8587b25ca78dd3ceea699ae2465bfec1
plan_commits_all_sessions: [6afdc73f1, ed19f783a, d787f0e81, 37d66510b, a7657a498, 465aa40d5]

tech-stack:
  added: []
  patterns:
    - "Every threshold / recipe / stopping / seed value read from the contract through contract.py; no literal downstream"
    - "A failed declared run becomes a named FAIL-CLOSED TEST VECTOR (recipe_id + run dir + numbers in the contract), which the gate self-test must decide FAIL on"
    - "Variance seeds run in <out>/.variance-seed-<s>/ and are deleted once their metrics are recorded; only the declared seed ships"
    - "A self-test that needs gitignored local evidence prints an explicit SKIP when it is absent, never a silent pass"

key-files:
  created:
    - scripts/laya_train/contract.py
    - scripts/laya_train/data.py
    - scripts/laya_train/prepare_stance.py
    - scripts/laya_train/train.py
    - scripts/laya_train/gate.py
    - scripts/laya_train/lifecycle.py
  modified:
    - contracts/laya-finetune-gate-v1.yaml
    - contracts/aprender/binding.yaml
    - crates/aprender-decide/src/pack.rs
    - scripts/laya_train/README.md
    - justfile

key-decisions:
  - "OPTION 3 (user, 2026-09-26): the D-19 demo's outcome of record is GATE FAIL under both declared recipes, fixed_epochs d0f4e40d (ece_post 0.3773, T clamped 5.0) and early_stopping 3d4b91da (ece_post 0.2224, T 3.14). The margin passes in both. Both run dirs are FAIL-CLOSED TEST VECTORS that 08-09 pack/verify must refuse. The D-18 live stance deploy is DEFERRED until a declared run passes. Recorded in contract 1.2.0 demo.*; gate_max_ece, T bounds, seeds and data are unchanged."
  - "No further recipe, tuning or gate attempt in this plan. A third attempt waits for the calibration spike (.planning/todos/pending/spike-laya-calibration-slice-and-temperature-cap.md), and whatever it recommends is declared in the contract before that run is read."
  - "MEASURED: the same recipe and seed on MPS is not bitwise reproducible. The seed-13 leg of the variance run (recipe_id 3d4b91da, byte-identical recipe.json) gave macro-F1 0.4437 / ece_post 0.2202 / T 3.2228, against 0.4458 / 0.2224 / 3.1436 recorded. A fail-closed vector is therefore identified by its run-dir files (recipe_id, eval-probs sha256), never by re-running it."
  - "--seeds keeps the data and the calibration split fixed (split at the declared seed). A variance seed varies only the training RNG. The gate is judged on seed 13 and recipe.json does not change with N."
  - "The Python verify_report re-decides pass from REPORTED metrics. It is the decision rule, not the forgery defence: the Rust verifier (08-09) recomputes from verified probabilities. The self-test's run-dir recompute bridges the two locally (max |d| = 0 on both vectors)."

patterns-established:
  - "Back-office CLIs exit 0 = gate pass, 3 = gate fail, 2 = input refused; refusals print `REFUSED <rule>: ...`"
  - "A run dir is written once: train.py refuses a non-empty --out; superseded runs are renamed, never overwritten"

requirements-completed: [D-01, D-02, D-03, D-04, D-05, D-06, D-07, D-08, D-19]   # copied verbatim from the plan. D-07 is delivered as the fail-closed gate, demonstrated FAILING on the demo. D-19 is delivered as "trained and gated", with outcome GATE FAIL under option 3. D-IDs are CONTEXT decisions, not REQUIREMENTS.md rows.

coverage:
  - id: D1
    description: "Tiny-fixture lifecycle: real train -> F16 save -> complete dir -> reload -> calibrate -> gate on CPU for both stopping rules. Proves the digest mapping, the log and mtime order, the --seeds refusals, a --seeds 3 run whose shipped checkpoint is byte-identical to the single-seed run's, and eval-probs re-scored by a separate F16 reload at max |dp| 0"
    requirement: D-01
    verification:
      - kind: integration
        ref: "just laya-train-lifecycle -> rc 0, LIFECYCLE OK"
        status: pass
    human_judgment: false
  - id: D2
    description: "Torch-free self-tests: every data refusal, the group-disjoint split over 50 seeds, reproduction of the demo's recorded slice_ids, the epoch rule, gate decisions and threshold refusals, both fail-closed vectors decided FAIL (and recomputed from their probability files where present), the bounded T fit and the early-stopping trace"
    requirement: D-07
    verification:
      - kind: unit
        ref: "just laya-train-selftest -> METRICS / DATA / GATE SELFTEST OK, LIFECYCLE OK, LAYA TRAIN SELFTEST OK (rc 0)"
        status: pass
    human_judgment: false
  - id: D3
    description: "D-19 demo trained on Laya's code and gated on the F16 reload under two declared recipes. Outcome of record: GATE FAIL, with both run dirs kept as fail-closed vectors (option 3)"
    requirement: D-19
    verification:
      - kind: other
        ref: "gate.py --selftest [fixed_epochs]/[early_stopping]: recipe_id == sha256(recipe.json); probability files match the report sha256s; recomputed metrics within 1e-5 (max |d| 0); gate recomputed from probabilities decides FAIL"
        status: pass
    human_judgment: false
  - id: D4
    description: "Contract 1.2.0 records the outcome, the vectors, the 08-09 refusal rule, the deferred D-18 deploy and the pending spike, with no threshold moved"
    requirement: D-07
    verification:
      - kind: other
        ref: "pv validate contracts/laya-finetune-gate-v1.yaml -> 0 error(s); make contract-audit-phase8 -> rc 0; gate_max_ece still 0.10"
        status: pass
    human_judgment: false
  - id: D5
    description: "--seeds N variance report. Real demo: seeds 13/17/23, one checkpoint kept, label 'mean ± sd over 3 seeds', recipe_id unchanged. It describes a FAILING recipe"
    requirement: D-08
    verification:
      - kind: integration
        ref: "just laya-train data/decide/tweet-stance-16 models/decide/tweet-stance-16-var --seeds 3 -> exit 3; plan verify 2 python assertion ok; find model.safetensors == 1"
        status: pass
    human_judgment: false
  - id: D6
    description: "scripts/laya_train/README.md documents every recipe, run step, output file, seed rule, refusal and the demo outcome"
    verification: []
    human_judgment: true
    rationale: "Documentation adequacy is a reading judgment; no test asserts the README."

duration: 42min   # 14 min first session + 10 min second + 18 min this close-out
completed: 2026-09-27
status: complete
---

# Phase 8 Plan 08: Laya Fine-tune, Calibration and Gate Summary

**`just laya-train` fine-tunes Laya locally on Laya's own code, calibrates on a text-disjoint slice and gates fail-closed on the F16 reload, with a `--seeds N` variance report and a self-tested gate. The TweetEval stance demo FAILED that gate under both declared recipes. By user decision (option 3) both runs are recorded as fail-closed test vectors, and the live stance deploy waits for a passing run.**

## Outcome: gate FAIL recorded as fail-closed vectors (option 3, user 2026-09-26)

The plan halted twice at its tracer gate. The user resolved it with option 3: re-scope D-07 / D-18 for the demo. This close-out applies that decision and changes no gate.

| | fixed_epochs | early_stopping |
|---|---|---|
| recipe_id (= sha256 of recipe.json) | `d0f4e40d39425e68d503f557f4f660eb9a73a4fb258da01f777b6f49362bcf20` | `3d4b91daf86772bcb23e5342c2dff4bb6467f5d9f254f7833ac7f61a8f2f5375` |
| run dir (gitignored) | `models/decide/tweet-stance-16-fixed-epochs/` | `models/decide/tweet-stance-16/` |
| report | `.../tweet-stance-16-fixed-epochs/gate-report.json` | `.../tweet-stance-16/gate-report.json` |
| epochs | 12 of 12 | best 4, run 7 of max 12 (patience) |
| zero-shot / fine-tuned macro-F1 | 0.3402 / 0.4701 | 0.3402 / 0.4458 |
| margin (need >= 0.05) | 0.1299 pass | 0.1056 pass |
| F_avg (information; spike 024: 0.538 ± 0.017) | 0.5151 | 0.4749 |
| ECE pre / **post** (need <= 0.10) | 0.4327 / **0.3773 FAIL** | 0.3546 / **0.2224 FAIL** |
| t_fitted = t_applied, clamp_hit | 5.0, true | 3.1436, false |
| calibration slice | 12 rows `[1, 8, 11, 12, 23, 24, 25, 26, 41, 43, 45, 47]`, sha256 `0640137d…` | identical |
| device / torch / seeds | mps:0 / 2.14.0 / single seed | mps:0 / 2.14.0 / single seed |

What the decision records, and where:

- **Contract** (`laya-finetune-gate-v1` 1.2.0, `demo.*`):
  - `outcome: gate_fail`, and both `fail_closed_vectors` with their numbers.
  - A `fail_closed_rule`: 08-09 pack/verify must refuse both runs, with the Rust-recomputed gate reproducing FAIL on `ece_post`.
  - `deploy: DEFERRED`, and `pass_pending` pointing at the spike.
  - `gate_max_ece` stays 0.10. The T bounds [0.5, 5.0], seeds, data and recipe values are unchanged.
- **Self-test.** `gate.py --selftest` decides FAIL on both vectors' reported numbers, on the `ece_post` clause alone, and refuses each report with `pass` flipped to true. When the gitignored dirs are present, it also recomputes the gate from `eval-probs.json` / `zero-shot-probs.json` plus the eval labels. That check found `recipe_id == sha256(recipe.json)`, all file sha256s matching, the metrics equal to the report (max |d| = 0) and the decision FAIL. `data.py --selftest` reproduces both vectors' recorded `slice_ids` and sha256 from the 16/16/16 label layout alone.
- **Deploy.** Nothing is packed for serving, uploaded or deployed from either run.

**Why calibration fails (measured in the earlier sessions, unchanged).** Two causes, and each blocks a pass on its own:

1. **The T cap binds.** Picking T with the eval labels (an oracle, not usable) gives a best servable ECE (T <= 5) of **0.1126** on the early_stopping checkpoint and 0.3722 on the fixed_epochs one. The ECE only drops under the ceiling above the cap: 0.0515 at T = 7.5 and 0.0389 at T = 10.
2. **The 12-row calibration slice is too small.** The stopping monitor's standard error is **0.22** (per-row NLL sd 0.76 / sqrt 12), against epoch-to-epoch differences of about 0.07. The slice is also easier than eval (accuracy 0.667 vs 0.471), so it fits T = 3.14 where the eval-oracle NLL optimum is about 7.5. A higher cap alone would not have rescued the run.

**Queued next:** `.planning/todos/pending/spike-laya-calibration-slice-and-temperature-cap.md` (commit `ad5a04bf8`). It asks whether a 64-shot cell (48-row slice) and/or a higher T cap can pass, measured over seeds 13/17/23, before any third attempt is declared. A cap change would touch laya-parity-v1 and the Rust clamp, and would diverge from upstream Laya. That is an architectural decision for the user.

## Variance report (evidence, describing a FAILING recipe)

`just laya-train data/decide/tweet-stance-16 models/decide/tweet-stance-16-var --seeds 3`: exit 3 (GATE FAIL), 214 s wall on mps:0.

- recipe_id is `3d4b91da…` (recipe.json byte-identical to the early_stopping vector's).
- Exactly one `model.safetensors` (seed 13) is kept, and the gate label reads `mean ± sd over 3 seeds`.
- **This is a variance measurement of the already-failed recipe.** It is not a gate attempt and it is not a declared vector.

| seed | macro-F1 | F_avg | ece_post | margin | T applied | best epoch | pass |
|---|---|---|---|---|---|---|---|
| 13 (declared, judged) | 0.4437 | 0.4765 | 0.2202 | 0.1034 | 3.2228 | 4 | false |
| 17 | 0.4771 | 0.4675 | 0.3445 | 0.1369 | 2.0141 | 3 | false |
| 23 | 0.4323 | 0.4953 | 0.1141 | 0.0921 | 3.1792 | 4 | false |
| **mean ± sd (ddof 1)** | **0.4510 ± 0.0233** | **0.4798 ± 0.0142** | **0.2263 ± 0.1153** | **0.1108 ± 0.0233** | | | |

- F_avg 0.480 ± 0.014 sits below spike 024's 0.538 ± 0.017. That is expected: 12 fit shots/class instead of 16 (RESEARCH A9).
- No seed passes the ECE ceiling. Seed 23 comes closest at 0.114.
- The ece_post sd of 0.115 supports the spike's slice-size question: with a 12-row slice, calibration quality swings by more than the ceiling itself from seed to seed.
- **MPS is not bitwise reproducible.** Seed 13 here differs from the recorded 3d4b91da run under the same recipe and seed (ECE 0.2202 vs 0.2224; best-epoch monitor 0.7974 vs 0.7866). See key-decisions.

## Performance

- **Duration:** 42 min over three sessions.
  - 2026-09-26T01:25:57Z to 01:39:32Z;
  - 05:30:57Z to 05:41:03Z;
  - this close-out, 2026-09-26T23:53:24Z to about 2026-09-27T00:12Z.
- **Tasks:** 2 of 2. Task 1's gate outcome is FAIL, recorded under option 3. Task 2 is complete.
- **Local compute this session:**
  - `just laya-train-selftest` about 1 min on CPU;
  - one mutation run about 30 s;
  - the 3-seed variance demo 214 s on mps:0.
  - Well under the 1-hour check-in line.

## Accomplishments

- **The trainer (Task 1, first two sessions):**
  - `just laya-train` on Laya's own `Agent._encode_state` and the spike-024 loop;
  - recipe written first;
  - device read back from the parameters;
  - complete F16 checkpoint before any reload;
  - bounded NLL calibration on a text-group-disjoint slice;
  - zero-shot, eval and probes on F16 reloads;
  - fail-closed gate with thresholds from the contract;
  - two declared stopping rules.
- **The re-scope recorded where decisions live:** contract 1.2.0 `demo.*`. This is additive; no threshold moved.
- **`--seeds N` (D-08):**
  - Trains the first N of `variance_seeds` with the declared seed first.
  - The other seeds run in `<out>/.variance-seed-<s>/`, which is deleted after their metrics are recorded.
  - It writes `variance-report.json`, and the gate is judged on seed 13.
  - The lifecycle proves the 3-seed run ships a checkpoint **byte-identical** to the single-seed run's, which is FALSIFY-LAYA-GATE-006's "declared seed ships whatever the others score".
- **`gate.verify_report` / `GateError`:**
  - Re-decides pass under the contract's thresholds.
  - Refuses a changed, missing or extra threshold, and refuses a `pass` that its own metrics contradict.
- **Self-tests (`--selftest`) for data and gate (numpy + pyyaml only), and `just laya-train-selftest`:**
  - The data self-test is characterization of Task 1 code. Two induced breakages turned it RED: no normalization (5 cases fail) and row-disjoint grouping (2 cases fail).
  - The variance temp-dir assertion was also shown RED under a mutation that skips the deletion.
- **FALSIFY-LAYA-GATE-007's second half is now actually asserted.** The lifecycle re-scores every eval row through a separate F16 reload and compares with `eval-probs.json` within `pack_rescore_probs_abs`: max |dp| 0 on all three lifecycle runs.
- **Contract `test_harness: 'just laya-train-selftest'`** on FALSIFY-LAYA-GATE-003, 004, 005, 006, 007 and 009, with `implemented_by` naming the exact self-test. The Rust `test:` lines for 003 and 005 remain 08-09's.
- **README** covers recipes, the 8 run steps, the run dir, seeds, the refusal table and the demo outcome.

## Task Commits

1. **Task 1: Tracer — stance demo end to end.** First session `6afdc73f1` (feat). Early-stopping declaration `ed19f783a` (feat, contract only, before the run). Early-stopping implementation `d787f0e81` (feat).
2. **Option-3 re-scope record:** `37d66510b` (docs, contract 1.2.0).
3. **Task 2: Expansion.** RED `a7657a498` (test), GREEN `465aa40d5` (feat).

Halt docs from the earlier sessions: `04a4fda8c`, `b60927862`, `7ae45bb69`.

**TDD (Task 2, `tdd="true"`; `workflow.tdd_mode` is off, so this is advisory):**

- RED `a7657a498`:
  - the gate self-test failed with `NameError: GateError` (the verify_report feature was absent);
  - the lifecycle failed with `LIFECYCLE FAILED: --seeds 0 was not refused ... unrecognized arguments: --seeds 0` (no seed policy yet);
  - the data self-test was GREEN at RED time because Task 1 already implemented the refusals. It was mutation-checked instead.
- GREEN `465aa40d5`: all four sentinels pass.
- No refactor commit.

## Files Created/Modified

- `contracts/laya-finetune-gate-v1.yaml`:
  - 1.2.0 `demo.outcome` / `outcome_rule` / `fail_closed_vectors` / `fail_closed_rule` / `deploy` / `pass_pending`, plus the AMENDMENT 1.2.0 paragraph;
  - six `test_harness` lines;
  - the KANI-LAYA-GATE-002 evidence text now says "is".
- `scripts/laya_train/train.py`: `--seeds`, `SeedRun`, `variance_row`, `variance_report`, and per-seed log lines (`CHECKPOINT COMPLETE seed=…`, `SCORING START seed=…`, `VARIANCE …`).
- `scripts/laya_train/gate.py`: `GateError`, `verify_report`, `FAIL_CLOSED_VECTORS`, `selftest()` and `--selftest`.
- `scripts/laya_train/data.py`: `selftest()` and `--selftest`, plus the demo label layout and recorded slice constants.
- `scripts/laya_train/contract.py`: `resolve_seeds`, `load_yaml`.
- `scripts/laya_train/lifecycle.py`: `check_seed_refusals`, `check_variance_run`, and the separate-reload re-score.
- `justfile`: `laya-train-selftest` and the `--seeds` help text.
- `scripts/laya_train/README.md`: training docs.
- Earlier sessions: `prepare_stance.py`, `contracts/aprender/binding.yaml`, `crates/aprender-decide/src/pack.rs` (optional strict `early_stopping` in `Recipe`).

## Decisions Made

See `key-decisions` in the frontmatter. The load-bearing one is option 3: FAIL is the recorded outcome, both runs are vectors, the deploy is deferred, and nothing is tuned.

## Deviations from Plan

### Auto-fixed Issues (this session)

**1. [Rule 2 - Missing critical] FALSIFY-LAYA-GATE-007's re-score half was not asserted by anything**
- **Found during:** Task 2, before citing `just laya-train-selftest` as 007's `test_harness`.
- **Issue:** 007 predicts that the eval probabilities equal those of a separate F16 reload. The lifecycle only reloaded and hashed; it never re-scored. Citing the harness would have claimed an unchecked half.
- **Fix:** the lifecycle re-scores every eval row through `train.load_for_scoring` + `train.score_rows` on a fresh reload, and compares against `eval-probs.json` within laya-parity-v1 `pack_rescore_probs_abs` (1e-5) with argmax equality.
- **Files modified:** `scripts/laya_train/lifecycle.py`, `scripts/laya_train/contract.py` (`load_yaml`).
- **Verification:** max |dp| 0 over 9 rows, on 3 runs.
- **Committed in:** `465aa40d5`.

**2. [Rule 2 - Missing critical] `--seeds` bounds refused before any work**
- **Issue:** the plan names seeds as "the first N of variance_seeds" but does not say what happens outside that range.
- **Fix:** `contract.resolve_seeds` refuses `--seeds` outside [1, 3], and a pool whose first seed is not the declared one, inside the data-validation block. That is before `--out` is created or any model loads.
- **Verification:** the lifecycle's `--seeds 0` / `--seeds 4` give exit 2, `REFUSED seeds`, and nothing written.
- **Committed in:** `465aa40d5`.

**3. [Scope, per the resolution] Task 2's variance run describes a FAILING recipe**
- **Issue:** the plan's variance verify assumed a passing demo.
- **Fix:** it was run once (optional, cheap), labelled as evidence, not a gate attempt. It is gitignored at `models/decide/tweet-stance-16-var/` and can be deleted (about 0.84 GB); it is not a declared vector.

Earlier-session deviations (unchanged):
- `lifecycle.py` is a separate file.
- `--base-sha256` for the synthetic variant.
- The epoch rule refuses `--epochs` rather than ignoring it.
- `Recipe.early_stopping` is optional in Rust.
- Eval ordering is enforced in code.
- The first run dir was renamed to `-fixed-epochs`.

---

**Total deviations this session:** 2 missing-critical auto-fixes and 1 scope note. **Impact on plan:** they tighten the evidence behind two FALSIFY entries. The scope change is the user's option-3 decision.

## Issues Encountered

- **`bash scripts/check_contract_test_binding.sh` exits 1 `VACUOUS … SKIPPED`.** This is pre-existing D-ITEM-08-01-A: `contracts/spectral-indices-v1.yaml` from `fdf6b1802` has no `kani_harnesses`.
  - Measured with the 08-01 instrument (a temporary lifted copy, then restored, verified by `git diff --quiet`).
  - With it lifted: 644 references resolve (585 at 08-01). The only FAILs are the pre-existing chronos-bolt-parity-v1 (9) and setfit-encoder-conformance-v1 (8).
  - **No Phase 8 contract is named.** The six new `just` harnesses do not dangle.
  - So plan verify 1's final `grep '^PASS'` is unreachable on this branch for reasons outside the plan. Its selftest half passes.
- `.pv/lint-previous.json` was rewritten by `pv lint` twice and restored from backup each time. It is not in any commit.

## Known Stubs

None.

## Deferred

- The six `contracts/aprender/binding.yaml` rows bound to `justfile::laya-train-selftest` stay `status: pending`. The recipe now exists; plan 08-12 flips every Phase 8 row per its own task.

## Threat Flags

None. There is no new endpoint or trust boundary. T-08-08-02 is strengthened: the self-test refuses a report whose thresholds differ from the contract, or whose `pass` contradicts its metrics.

## User Setup Required

None.

## Next Phase Readiness (downstream notes)

- **08-09 needs a replan before it runs.** Its Task 1 `<precondition>` requires `models/decide/tweet-stance-16/gate-report.json` with `"pass": true`, which is UNMET by design. Under option 3 its demo leg inverts:
  - `just laya-pack` and `just laya-verify` must **refuse** both vectors. Arguments: `data/decide/tweet-stance-16` and the base snapshot `~/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`.
    - `models/decide/tweet-stance-16` (3d4b91da) should refuse on the recomputed `ece_post` 0.2224 > 0.10, with margin 0.1056 passing.
    - `models/decide/tweet-stance-16-fixed-epochs` (d0f4e40d) should refuse on `ece_post` 0.3773, T 5.0 clamped.
  - A replan can keep the real-weight re-score evidence (rescore_max_abs, argmax 280/280) by printing it before the gate refusal. Recording it as a refusal must not produce an `.apr`.
  - `models/decide/laya-stance-16.apr` will not exist. The `PACKED … argmax=280/280` verify, `laya-verify … deploy_eligible: true`, and the stdio real-model leg (`APR_MCP_E2E_DECIDE_MODEL=models/decide/laya-stance-16.apr`) cannot run as written.
  - The spike-025 full-model parity leg (`LAYA_MODEL_DIR`, `tests/laya_parity.rs`) is independent of the demo and can still run.
  - Its FALSIFY `test:` lines for 001, 002, 003, 005 and 008 are unaffected.
  - Optional: the 0.84 GB `models/decide/tweet-stance-16-var/` is a third FAILING run of recipe 3d4b91da (seed 13 ECE 0.2202). It is usable as an extra refusal case, but it is not a declared vector.
- **08-10:** unaffected (the cargo-pmcp wrong-package decision is artifact-independent).
- **08-11 needs a scope decision at its go/no-go checkpoint.**
  - Its truths and Task 2 deploy `models/decide/laya-stance-16.apr`, which cannot exist until a declared run passes.
  - `hold` is the option consistent with option 3: D-18 is not met in this phase, and 08-12 records `accepted_region_cold` as `partial`.
  - The alternative is to wait for the spike and a passing declared run, or to re-scope the deploy target. Deploying any stance artifact would contradict the recorded decision.
- **08-12:**
  - At the CI-edit checkpoint, `data.py` and `gate.py --selftest` now exist and need only numpy + pyyaml (option `approve-ci-edit-with-python`). The gate self-test's run-dir recompute prints `SKIP` in CI.
  - The six `laya-train-selftest`-bound binding rows can flip to `implemented`.
  - `accepted_region_cold` follows 08-11's outcome (expected `hold` -> `partial`).
  - Any CLAUDE.md prose must not claim a deployed or gate-passing stance model.
- **A passing demo later:**
  1. run the spike;
  2. declare its recommendation in `laya-finetune-gate-v1` (a new demo cell and/or a cap change, the latter needing laya-parity-v1 and the user);
  3. then `just laya-train-lifecycle` (about 10 s) and `just laya-train <data> <new run dir>`.
  - `prepare_stance.py` reads s16 only today; a 64-shot cell needs it extended.

---
*Phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server*
*Completed: 2026-09-27 (complete under option 3; demo outcome GATE FAIL recorded as fail-closed vectors)*

## Self-Check: PASSED

- All 6 created scripts, the README, the contract, the justfile and the todo exist.
- Both vector gate reports and `tweet-stance-16-var/variance-report.json` exist and are gitignored.
- Commits `6afdc73f1`, `ed19f783a`, `d787f0e81`, `37d66510b`, `a7657a498` and `465aa40d5` are in history, with no unexpected deletions since `ad5a04bf8`.
- Plan verify 1: `just laya-train-selftest` rc 0 with all four sentinels. Its trailing binding-guard `^PASS` is unreachable because of pre-existing D-ITEM-08-01-A (see Issues).
- Plan verify 2: the python assertion passes and exactly 1 `model.safetensors`.
- `pv validate`: 0 errors. `make contract-audit-phase8`: rc 0.
- `just laya-fixtures`: FIXTURES OK with no `crates/` diff (byte-identical).
- No Rust was touched this session.
