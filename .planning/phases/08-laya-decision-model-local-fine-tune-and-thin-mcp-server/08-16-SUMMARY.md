---
phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
plan: 16
subsystem: decision-model-gate-run
tags: [laya, gate-run, demo_s64, median-ece, seed-selection, rescore-noise, shift-probe, deploy-eligible, stdio-mcp, gap-closure]
status: complete
gap_closure: true
outcome: gate_pass

requires:
  - phase: 08-13
    provides: "the declaration: laya-finetune-gate-v1 AMENDMENT 1.4.0 (published 2.0.0) and laya-parity-v1 2.0.0, committed before this run (fc0c1dd30, a3e155170, 56f888d62; seeds label c99761825)"
  - phase: 08-14
    provides: "the Python writer: three-seed median run, rescore-noise.json, the s64 data by rule, the shift probe"
  - phase: 08-15
    provides: "the Rust verifier (noise-referenced bound, median re-derivation, shift probe recompute) and tests/demo_run.rs (FALSIFY-LAYA-GATE-014)"
provides:
  - "The one declared demo_s64 run (models/decide/laya-stance-64, gitignored): GATE PASS, recipe 6a5489af…, median-ECE seed 17 ships"
  - "The first deploy-eligible decide artifact: models/decide/laya-stance-64.apr (gitignored), sha256 H = 24a44d7e050166c9b64e2716f2bcb3ce91747f7a3b927d03d6eeae5f89b6275a, 846196868 bytes"
  - "demo_s64.outcome gate_pass and outcome_record in laya-finetune-gate-v1 (pv-valid; pv diff identical, no bump)"
  - "08-GATE-RUN-EVIDENCE.json: the machine-checkable record plans 08-17 and 08-12 read"
  - "FALSIFY-LAYA-GATE-014 armed and green on the exact bytes; D-ITEM-08-09-A resolved (stdio real-model leg); D-ITEM-08-11-A re-open condition met"
affects: [08-17, 08-18, 08-12]

actuals:
  tokens: 2737       # chars/4 over the realized diff 39be66265..HEAD (3 files, +182/-3)
  tasks: 2           # Task 1 (tracer) and Task 3; Task 2 (stop-rule checkpoint) skipped by its precondition
  commits: 1         # MEASURED: git rev-list --count 39be66265..HEAD before this SUMMARY commit
plan_head_before: 39be66265055c1fd8de9d1b0b97caf54fc8128b4

tech-stack:
  added: []
  patterns:
    - "Record, never re-decide: the outcome of record is generated from the run's own gate-report / noise record / verify JSON, then re-decided by an armed test on the exact bytes"

key-files:
  created:
    - .planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-GATE-RUN-EVIDENCE.json
  modified:
    - contracts/laya-finetune-gate-v1.yaml
    - .planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/deferred-items.md

key-decisions:
  - "The one declared s64 run PASSED the unchanged gate: shipped median seed 17, margin 0.2217, ece_post 0.0443 on the 459-row in-distribution set; artifact 24a44d7e… is deploy_eligible and is the identity 08-17/08-18 pin"
  - "The fine-tuned re-score max_abs 1.13e-5 exceeds the 1e-5 absent-record floor and sits within the declared A1 bound 3.34e-5; recorded plainly in outcome_record.rescore_note (under the 1.x bar this run would have been refused RescoreDrift)"
  - "No contract version bump: pv diff reports laya-finetune-gate-v1 identical (it does not diff demo_s64), the 08-13/14/15 precedent"

patterns-established:
  - "An env var holding a model path for a cargo integration test must be ABSOLUTE: cargo test runs the test with the crate dir as cwd"

requirements-completed: [D-07, D-08, D-11, D-17, D-19]

coverage:
  - id: D1
    description: "Exactly one declared gate run (just laya-train data/decide/tweet-stance-64 models/decide/laya-stance-64 --seeds 3 --epochs 12), rc 0 GATE PASS, one RECIPE WRITTEN line, one model.safetensors kept"
    requirement: D-19
    verification:
      - kind: other
        ref: "08-16-PLAN Task 1 <verify><automated> (gate pass=True, <=1 checkpoint, laya-verify deploy_eligible true on sha 24a44d7e, stdio leg passed)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Pack, verify and inspect on the exact artifact: PACKED exit 0, deploy_eligible true with artifact_sha256 == shasum, inspect labels none/against/favor, variant production, recipe 6a5489af"
    requirement: D-11
    verification:
      - kind: integration
        ref: "just laya-pack / just laya-verify / just laya-inspect on models/decide/laya-stance-64.apr"
        status: pass
    human_judgment: false
  - id: D3
    description: "Stdio real-model leg on the artifact: one tool, served identity == file sha256 (D-11), K probabilities summing to 1 (lifts D-ITEM-08-09-A)"
    requirement: D-11
    verification:
      - kind: e2e
        ref: "crates/aprender-mcp-decide/tests/e2e_stdio.rs#a_real_decide_model_classifies_over_live_stdio (APR_MCP_E2E_DECIDE_MODEL=<absolute path>)"
        status: pass
    human_judgment: false
  - id: D4
    description: "Outcome of record in the contract (pv-valid) and 08-GATE-RUN-EVIDENCE.json; FALSIFY-LAYA-GATE-014 armed and green without SKIP; thresholds unchanged; no home path; spikes untouched"
    requirement: D-07
    verification:
      - kind: other
        ref: "08-16-PLAN Task 3 <verify><automated> checks (pv 0 errors; yaml/evidence assertions; spikes status empty)"
        status: pass
      - kind: integration
        ref: "LAYA_DEMO_RUN=1 LAYA_MODEL_DIR=<snapshot> target/release/deps/demo_run-* declared_demo_run_matches_its_outcome_of_record --nocapture (DEMO OUTCOME gate_pass, 389 s)"
        status: pass
    human_judgment: false
  - id: D5
    description: "The recorded claim scope matches eval_set.claim, read side by side with the shift probe"
    verification: []
    human_judgment: true
    rationale: "The plan's end-of-phase human-check: a person reads the RESULT and SHIFT PROBE lines side by side and confirms the recorded claim scope (in-distribution calibration only) is stated honestly"

duration: 43min
completed: 2026-09-27
---

# Phase 8 Plan 16: The one declared s64 gate run Summary

**The one declared demo_s64 run passed the unchanged gate. Seed 17 is the median-ECE seed of 13/17/23 and ships, with margin 0.2217 and ece_post 0.0443 on the 459-row in-distribution held-out set. Its packed artifact `24a44d7e…` is deploy-eligible, it answered a real classify over stdio MCP with its own identity, and FALSIFY-LAYA-GATE-014 re-decides `gate_pass` on the exact bytes.**

## Performance

- **Duration:** about 43 min, from 2026-09-27T18:07:34Z to 18:50:44Z.
- **The run:** 684 s wall time, from 18:08:51Z to 18:20:15Z (`total_seconds=682.4` inside train.py).
- **The tail:** pack 232 s, verify 214 s, armed GATE-014 418 s through cargo and 389 s run directly.
- **Tasks:** 2 of 3 executed. Task 2 is the stop-rule checkpoint, and its precondition (no eligible artifact) was false, so it was skipped as the plan specifies.
- **Files:** 3 tracked files changed. The run dir and the `.apr` are gitignored.

## Before the run (Task 1 preconditions)

- **Start sha** (`/tmp/p08-16-start.sha`): `39be66265055c1fd8de9d1b0b97caf54fc8128b4`.
- **Ancestor proof.** `git merge-base --is-ancestor <c> HEAD` held for every one of these, so the rules were committed before this run (D-07):
  - the 08-13 declaration: `fc0c1dd30`, `a3e155170`, `56f888d62`;
  - the settled seeds label: `c99761825`;
  - the 08-14 code: `4f5ada981`, `de58feea6`, `24cea2a63`, `4f4674fbd`;
  - the 08-15 code: `3cfa83e1b`, `6389ec3d5`, `3ef728427`.
- **Disk.** `df -h .` gave `/dev/disk3s5 926Gi 866Gi 26Gi 98%`, above the 10 GB floor.
- **First run.** `test ! -e models/decide/laya-stance-64` held, and the base snapshot `55cf4c4e…` was present.
- **Self-tests.**
  - `just laya-train-selftest` printed `LAYA TRAIN SELFTEST OK`.
  - `cargo test -p aprender-decide --lib` gave 112 passed.
  - `just laya-prepare-stance s64` reported: `unchanged (byte-identical)`, eval [111, 291, 57] (459), shift 280, train [64, 64, 64], `PREPARE OK`.

## The run (Task 1)

`just laya-train data/decide/tweet-stance-64 models/decide/laya-stance-64 --seeds 3 --epochs 12`

The trainer ran exactly once: the log holds exactly **1** `RECIPE WRITTEN` line. It returned **rc 0**, read from a file and not through a pipe. The device was **`mps:0`** on torch 2.14.0, read back from the parameters.

```
RECIPE WRITTEN 6a5489afa3462f566b8b12e932a0e35f93d384bea6f37e3154d06ebc186d6f3b stopping=early_stopping epochs_max=12
STOP seed=13 rule=early_stopping best_epoch=1 best_calib_nll=0.768009 epochs_run=4 reason=patience
STOP seed=17 rule=early_stopping best_epoch=2 best_calib_nll=0.752151 epochs_run=5 reason=patience
STOP seed=23 rule=early_stopping best_epoch=2 best_calib_nll=0.788442 epochs_run=5 reason=patience
SEED seed=13 macro_f1=0.5530 ece_post=0.084974 margin=0.1439 pass=True rank_key=849
SEED seed=17 macro_f1=0.6308 ece_post=0.044250 margin=0.2217 pass=True rank_key=442
SEED seed=23 macro_f1=0.6570 ece_post=0.037151 margin=0.2478 pass=True rank_key=371
MEDIAN seeds=13,17,23 rank_keys=849,442,371 shipped=17 (policy median_ece, tie_break smaller_seed)
NOISE which=fine_tuned max_abs=8.339318e-06 bound=3.335727e-05 argmax=459/459 t_applied=3.427839
NOISE which=zero_shot max_abs=1.495515e-05 bound=5.982062e-05 argmax=459/459 t_applied=1.760152
SHIFT PROBE n=280 ft macro_f1=0.5175 f_avg=0.5464 ece_post=0.1896 | zs macro_f1=0.3402 ece=0.2689 | margin=0.1773 (reported, not a gate clause)
RESULT zero_shot macro_f1=0.4092 f_avg=0.3717 ece=0.1642 | fine_tuned macro_f1=0.6308 f_avg=0.6140 ece_pre=0.2119 ece_post=0.0443 nll=0.7768 | margin=0.2217 (need >= 0.05) ece_post (need <= 0.1)
SEEDS median-ECE seed of 3 seeds | shipped=17 device_used=mps:0 torch=2.14.0 train_seconds=104.5 total_seconds=682.4
GATE PASS
```

| seed | macro_f1 | margin | ece_post | T applied (clamp) | pass | rank_key |
|---|---|---|---|---|---|---|
| 13 | 0.5530 | 0.1439 | 0.0850 | 1.6898 (no) | true | 849 |
| **17 (shipped, median)** | 0.6308 | **0.2217** | **0.0443** | 3.4278 (no) | true | 442 |
| 23 | 0.6570 | 0.2478 | 0.0372 | 4.7807 (no) | true | 371 |

- **All three seeds pass.** The order by rank key is 23 < 17 < 13, so the median is 17. The selection is made **with eval labels** (`seed_policy.honesty`): median-of-3, not best-of-3.
- **Against spike 027's expectation.** It predicted margin 0.27 ± 0.04 and ECE 0.057 ± 0.010, measured on non-gate runs and INFORMATIONAL only. The shipped margin 0.2217 sits just below that band, and its ECE 0.0443 sits just below its band too. The three-seed mean is ECE 0.0555 ± 0.0258 and margin 0.2045 ± 0.0541.
- **Checkpoints.** The run deleted seeds 13 and 23's checkpoints (`DELETED seed=13 checkpoint`, `DELETED seed=23 checkpoint`). The run dir keeps exactly one `checkpoint/model.safetensors` (sha `10264ea0…`, seed 17's) and is 808 MB on disk.

## Pack, verify, inspect and the stdio leg (Task 1, on the gate pass)

```
PACKED models/decide/laya-stance-64.apr sha256=24a44d7e050166c9b64e2716f2bcb3ce91747f7a3b927d03d6eeae5f89b6275a rescore_max_abs=0.000011324882507324219 zs_rescore_max_abs=0.0000073462724685668945 rescore_bound=0.00003335727136288824 zs_rescore_bound=0.00005982061985720932 noise=0.00000833931784072206 zs_noise=0.00001495515496430233 shipped_seed=17 argmax=459/459
{"argmax":"459/459","artifact_sha256":"24a44d7e050166c9b64e2716f2bcb3ce91747f7a3b927d03d6eeae5f89b6275a","deploy_eligible":true,"noise":8.33931784072206e-6,"recomputed":{"ece_post":0.04425034672021866,"ft_macro_f1":0.6308225393295288,"margin":0.22166302800178528,"zs_macro_f1":0.40915951132774353},"rescore_bound":0.00003335727136288824,"rescore_max_abs":0.000011324882507324219,"shipped_seed":17,"zs_noise":0.00001495515496430233,"zs_rescore_bound":0.00005982061985720932,"zs_rescore_max_abs":7.3462724685668945e-6}
{"artifact_sha256":"24a44d7e050166c9b64e2716f2bcb3ce91747f7a3b927d03d6eeae5f89b6275a","base":"laya-en-root@55cf4c4e","embedded_gate":{"ece_post":0.044250346606593244,"margin":0.22166308511207017,"pass":true},"labels":["none","against","favor"],"method":"laya","recipe_id":"6a5489afa3462f566b8b12e932a0e35f93d384bea6f37e3154d06ebc186d6f3b","schema_version":1,"variant":"production"}
E2E-DECIDE-PMCP-001 (real): /…/models/decide/laya-stance-64.apr identity "24a44d7e050166c9b64e2716f2bcb3ce91747f7a3b927d03d6eeae5f89b6275a"
test result: ok. 1 passed; 0 failed
```

- **Pack** (exit 0) printed the first line above, and **verify** (exit 0) printed the JSON line after it. `shasum -a 256` of the file is `24a44d7e…`, equal to both. The Rust gate recomputation agrees with the report.
- **Inspect** printed the second JSON line: labels none/against/favor, variant `production`, and the run's recipe_id.
- **The stdio real-model leg** passed with one tool and the served identity equal to the file's sha256 (D-11). The test asserts that the K probabilities sum to 1 and that the labels come from the task. This lifts D-ITEM-08-09-A.
- **Artifact identity H**, which 08-17/08-18 pin: `24a44d7e050166c9b64e2716f2bcb3ce91747f7a3b927d03d6eeae5f89b6275a`, 846196868 bytes.
- **Read this plainly:** the fine-tuned re-score max_abs **1.13e-5** is ABOVE the 1.0e-5 absent-record floor. It is WITHIN the noise-referenced bound 3.34e-5 (4 x the float64 noise 8.34e-6) that laya-parity-v1 2.0.0 A1 declared in 08-13, before this run. Under the 1.x bar this run would have been refused `RescoreDrift`. It is recorded as `outcome_record.rescore_note`.

**Tracer gate:** interactive, `end-of-phase`, automated-only verify. The Task 1 `<verify>` re-ran green (`gate pass=True`, `deploy_eligible true for 24a44d7e0501`, stdio `1 passed`), so the plan continued to Task 3 with no checkpoint.

## The outcome of record (Task 3)

- **`contracts/laya-finetune-gate-v1.yaml` `demo_s64`:**
  - `outcome: gate_pass`.
  - `outcome_rule` states what the gate decided and the claim scope.
  - `outcome_record` holds: recipe_id, run_dir, data_dir, device_used, torch_version, shipped_seed 17, per_seed ×3, zero_shot/ft macro_f1, margin, ece_post, `failed_clauses: []`, noise and bound for both sets, rescore_max_abs, zs_rescore_max_abs, artifact_sha256, artifact_bytes, shift_probe, `decided_by: gate` and `recorded_on`.
  - No threshold moved: 0.10 / 0.05 are asserted by the verify.
  - `pv validate` gave `0 error(s), 0 warning(s)`. `pv diff` against the HEAD copy printed **"Contracts are identical."** (it does not diff `demo_s64`), so no bump was applied.
- **`08-GATE-RUN-EVIDENCE.json`** was generated from gate-report.json, rescore-noise.json and the verify JSON, never typed by hand. It holds: outcome, decided_by, human_option null, declaration and code commits, run_commit, data sha256s, seeds, gate (including the Rust recomputation), noise, pack, artifact {path, sha256, bytes}, deploy_eligible true, e2e_stdio, shift_probe, device and wall_seconds 684. It contains no absolute home path.
- **FALSIFY-LAYA-GATE-014, armed.** The cargo run exited rc 0, but the rtk hook filters `println!`. The test binary was therefore run directly, and it printed:
  ```
  DEMO gate_pass sha256=24a44d7e050166c9b64e2716f2bcb3ce91747f7a3b927d03d6eeae5f89b6275a shipped_seed=17 rescore_max_abs=0.000011324882507324219 rescore_bound=0.00003335727136288824 zs_rescore_max_abs=0.0000073462724685668945 zs_rescore_bound=0.00005982061985720932 noise=0.00000833931784072206 zs_noise=0.00001495515496430233 ece_post=0.04425034672021866 margin=0.22166302800178528
  DEMO OUTCOME gate_pass decided on the exact bytes (389 s, ARCH aarch64)
  test result: ok. 1 passed; 0 failed
  ```
  The output has no SKIP line. The test's own pack into a TempDir reproduced the same sha `24a44d7e…`, so packing is deterministic on these bytes.
- **deferred-items.md:**
  - D-ITEM-08-09-A is `status: resolved`, naming the identity sha.
  - D-ITEM-08-11-A gained: "re-open condition MET by plan 08-16 (gate_pass, deploy_eligible true); owner: plans 08-17/08-18". Its status stays open, because the deploy itself is 08-17/08-18's.
- **Hygiene:**
  - The three s16 gate-report.json sha256s are **identical before and after**: `2ff6cc2e…` (tweet-stance-16), `fb4980e4…` (-fixed-epochs) and `c0c5cad4…` (-var).
  - `git status --porcelain .planning/spikes/` is empty, and `.pv/` is unchanged.
  - `make contract-audit-phase8` exits 0.
  - The scratch files were removed after this SUMMARY.

## What the pass certifies (eval_set.claim), beside the shift probe

> The gate now certifies margin and calibration on held-out data drawn like the tenant's shots. It does NOT certify robustness to a shifted input population.

| set | rows | ft macro_f1 | margin | ece_post | role |
|---|---|---|---|---|---|
| in-distribution held-out (gate) | 459 | 0.6308 | 0.2217 | **0.0443** | gate clause |
| SemEval-2016 test (shift probe) | 280 | 0.5175 | 0.1773 | **0.1896** | reported, `gate_clause: false` |

The shift probe's ece_post 0.1896 would fail the 0.10 clause if it were a clause. The claim is therefore in-distribution calibration only, and a tenant whose inputs drift from its shots should not read this pass as calibration on the drifted population.

## Task Commits

1. **Task 1 (tracer): the declared run, pack, verify, inspect and the stdio leg.** No commit. Its outputs are gitignored (`models/decide/laya-stance-64/`, `models/decide/laya-stance-64.apr`), and its evidence is committed in Task 3.
2. **Task 2 (checkpoint:decision, blocking-human):** skipped. Its precondition (no eligible artifact) was false.
3. **Task 3: the outcome of record, the evidence file, armed GATE-014 and deferred items.** `820d82732` (feat).

**Plan metadata:** the docs commit that adds this SUMMARY.

## Files Created/Modified

- `contracts/laya-finetune-gate-v1.yaml`: `demo_s64.outcome` gate_pass, `outcome_rule` and `outcome_record`. Records only.
- `.planning/phases/08-…/08-GATE-RUN-EVIDENCE.json` (new): the machine-checkable run record.
- `.planning/phases/08-…/deferred-items.md`: 08-09-A resolved; 08-11-A re-open line added.

## Decisions Made

See `key-decisions`. No threshold, bound, seed, recipe or data value was changed.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] The stdio leg needs an ABSOLUTE model path**
- **Found during:** Task 1, the stdio real-model leg.
- **Issue:** the plan's `APR_MCP_E2E_DECIDE_MODEL=models/decide/laya-stance-64.apr` is relative. `cargo test` runs the test with the crate dir as cwd, so it panicked at `e2e_stdio.rs:327` (`... which is not a file`) before any model code ran. The plan's own `<verify>` uses the same relative path.
- **Fix:** passed `$(pwd)/models/decide/laya-stance-64.apr`, the same file. The leg and the tracer verify then passed. The note is recorded in D-ITEM-08-09-A.
- **Files modified:** none.
- **Why this is not a second attempt:** the failed invocation read no model, and the stop rule governs `just laya-train`, which ran once.

**2. [Rule 3 - Tooling] GATE-014's output was captured by running the test binary directly**
- **Found during:** Task 3.
- **Issue:** the rtk hook reduces `cargo test` output to a summary (`cargo test: 1 passed`), which drops the `DEMO …` and any `SKIP` line. The plan's `! grep -q SKIP` would then pass vacuously.
- **Fix:** ran `target/release/deps/demo_run-b4fc9a406b51b74f … --nocapture` with the same env, from the crate dir. It printed `DEMO OUTCOME gate_pass decided on the exact bytes`, rc 0, and no SKIP. The Task 3 verify script checks that log instead.

---

**Total deviations:** 2 auto-fixed (both Rule 3, invocation and tooling). **Impact:** none on the run or the record. `just laya-train` was invoked exactly once.

## Issues Encountered

- Laya's loader warns that the base checkpoint ships `choice:11+=0.1006` outside [0.5, 5] (clamped to 0.5). This is a pre-existing base-config property. The task uses bucket `choice:3-5`, so it does not touch this run.

## Known Stubs

None.

## Threat Flags

None. T-08-16-01..06 are mitigated as planned:
- one RECIPE WRITTEN line;
- the outcome re-decided by armed GATE-014;
- sha equality across shasum, verify, inspect, stdio and GATE-014;
- numbers and hashes only in the evidence, with no home path;
- the disk check, with one checkpoint kept;
- the s16 hashes unchanged and the spikes tree clean.

## User Setup Required

None.

## Next Phase Readiness

- **08-17 (go/no-go, behind the user):** `08-GATE-RUN-EVIDENCE.json` has `outcome: gate_pass` and `deploy_eligible: true`, so the deploy options open. The identity to pin is `24a44d7e050166c9b64e2716f2bcb3ce91747f7a3b927d03d6eeae5f89b6275a`. No AWS, deploy or S3 action was taken here.
- **Still open for the deploy:** D-ITEM-08-13-A (x86_64 parity is unmeasured; the deploy is arm64). The auth posture and account facts are listed under D-ITEM-08-11-A.
- **08-12:** may claim a declared gate pass with the in-distribution claim scope above. It must not claim shift robustness.

## Self-Check: PASSED

- FOUND: contracts/laya-finetune-gate-v1.yaml, 08-GATE-RUN-EVIDENCE.json, deferred-items.md, models/decide/laya-stance-64.apr, models/decide/laya-stance-64/gate-report.json
- FOUND commit: 820d82732. `git rev-list --count 39be66265..HEAD` = 1 before this SUMMARY, and it deletes no tracked file.
- Task 1 `<verify>` rc 0 and Task 3 `<verify>` rc 0. All acceptance criteria were re-checked: RECIPE WRITTEN count 1, outcome in both files, s16 hashes equal, and the claim quoted beside the shift probe.

---
*Phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server*
*Completed: 2026-09-27*
