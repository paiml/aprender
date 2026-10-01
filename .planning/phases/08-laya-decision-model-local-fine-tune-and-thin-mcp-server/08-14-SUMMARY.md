---
phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
plan: 14
subsystem: training
tags: [laya, python, torch, seed-selection, median-ece, float64, rescore-noise, shift-probe, heldout-eval, gate]
status: complete
gap_closure: true

requires:
  - phase: 08-13
    provides: "laya-finetune-gate-v1 AMENDMENT 1.4.0 (published 2.0.0: A2 eval_set + shift probe, A3 seed_policy median_ece, demo_s64, rescore_noise_schema) and laya-parity-v1 2.0.0 (A1 pack_rescore_noise_k, pack_rescore_bound_max_abs)"
  - phase: 08-08
    provides: "the Laya back office (train.py SeedRun / Scorer / calibrate_and_score, lifecycle.py, gate.py, data.py)"
provides:
  - "A production `just laya-train` trains seeds 13/17/23 in seeds/seed-<s>/, ships the median-ECE seed as checkpoint/, deletes the other checkpoints, and hash-binds every seed's eval-probs.json in gate-report seeds.per_seed (A3)"
  - "Every run writes rescore-noise.json (laya-rescore-noise-v1): float64 probabilities of every eval row for the shipped checkpoint and the base, control 0.0, max_abs over exactly the written float32 values, k and floor read from laya-parity-v1, bound by gate-report rescore_noise_sha256 (A1)"
  - "data/decide/tweet-stance-64 (gitignored) built by eval_set.demo_rule: 192 shots, 459 eval rows [111, 291, 57], 280 shift rows; s16 still byte-identical (A2)"
  - "The optional shift.jsonl probe is scored after the gate, the median and the noise record, reported as shift_probe {gate_clause: false}, and proven not to change the gate"
  - "LAYA_LIFECYCLE_KEEP=<dir> exports a full 1.4.0 three-seed shift run dir and its data dir for plan 08-15's Rust reader test"
affects: [08-15, 08-16, 08-12]

actuals:
  tokens: 34514
  tasks: 3
  commits: 4
plan_head_before: e74c1fa38973ba8e9a61f6e66676b88d4b3a5ea4

tech-stack:
  added: []
  patterns:
    - "Conditional run-dir files and OPTIONAL report keys decided by the contract's own wording (`(only when ...)`, `<s>` templates, `OPTIONAL`), with unknown conditions refused rather than guessed"
    - "A float64 reference is only recorded after a manual fp32 forward reproduces the production Scorer's logits bit for bit (the control); otherwise the run is refused"
    - "Median selection reads eval only after every seed's checkpoint is fixed; the per-seed ordering assertion (checkpoint before that seed's eval probabilities) is kept"

key-files:
  created:
    - .planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-14-SUMMARY.md
  modified:
    - scripts/laya_train/contract.py
    - scripts/laya_train/gate.py
    - scripts/laya_train/train.py
    - scripts/laya_train/lifecycle.py
    - scripts/laya_train/data.py
    - scripts/laya_train/prepare_stance.py
    - scripts/laya_train/README.md
    - justfile
    - contracts/laya-finetune-gate-v1.yaml
    - contracts/laya-parity-v1.yaml
    - contracts/aprender/binding.yaml
    - .planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/deferred-items.md

key-decisions:
  - "gate-report seeds.label stays the contract's literal ('mean ± sd over 3 seeds') for a median run, not the plan's 'median-ECE seed of 3 seeds': seed_policy.rule and gate_report_schema.seeds both fix it (D-ITEM-08-14-A)"
  - "synthetic-fixture accepts --seeds 1 (legacy rule) or 3 (median rule) only; 2 is refused because the median needs an odd N and no legacy multi-seed run is written any more"
  - "The noise record reloads the shipped checkpoint and the base fresh (the float64 cast consumes the model) and the shift probe reloads again, so no fp32 artifact is scored by a model that was cast"
  - "When seed 13 is the median, 'the checkpoint tree equals the single-seed run's' is compared with training.recipe_id removed from rl_agent_config.json: seed_selection changes the recipe_id, which is provenance only. Seed 13's model.safetensors is additionally compared unconditionally"
  - "pv diff reported both contracts identical after the test_harness / implemented_by edits, so no version bump was applied"

patterns-established:
  - "LAYA_LIFECYCLE_KEEP: the lifecycle exports its richest run dir (three seeds + shift probe) for the cross-language reader"
  - "An induced-RED mutation per new self-test family, recorded with its revert"

requirements-completed: [D-01, D-06, D-07, D-08, D-17, D-19]

coverage:
  - id: D1
    description: "A3 median-ECE seed selection in the trainer: per-seed training in seeds/seed-<s>/, median ships as checkpoint/, one model.safetensors, per_seed rows hash-bound, gate = shipped seed's; production --seeds 1 refused before anything is written"
    requirement: D-08
    verification:
      - kind: integration
        ref: "LAYA_LIFECYCLE_KEEP=$K just laya-train-lifecycle (08-14-PLAN Task 1 <verify>) + kept-run-dir assertions"
        status: pass
      - kind: unit
        ref: "scripts/laya_train/gate.py --selftest#median-ECE seed selection (14 cases)"
        status: pass
    human_judgment: false
  - id: D2
    description: "A1 float64 re-score noise record on every run (control 0.0 or exit 2, every row, max_abs bit-for-bit, k/floor from laya-parity-v1, rescore_noise_sha256)"
    requirement: D-17
    verification:
      - kind: integration
        ref: "scripts/laya_train/lifecycle.py#check_noise_record (every lifecycle run)"
        status: pass
      - kind: other
        ref: "induced control mismatch (+1e-6 on the manual fp32 logits): train.py exit 2, REFUSED noise-control, no rescore-noise.json, no gate-report.json"
        status: pass
    human_judgment: false
  - id: D3
    description: "A2 demo_s64 data by rule: data/decide/tweet-stance-64 with 192 shots, 459 eval rows [111, 291, 57], 280 shift rows; s16 cell byte-identical to data/decide/tweet-stance-16"
    requirement: D-19
    verification:
      - kind: integration
        ref: "just laya-prepare-stance s64 + 08-14-PLAN Task 2 <verify> row/class assertions + cmp of the s16 cell in a scratch dir"
        status: pass
      - kind: unit
        ref: "scripts/laya_train/data.py --selftest#in-distribution held-out rule (7 synthetic cases + the pinned-split rebuild)"
        status: pass
    human_judgment: false
  - id: D4
    description: "Shift probe scored after the gate and reported as shift_probe {gate_clause: false}; the gate is identical with and without shift.jsonl; a shift row equal to a train row is refused"
    requirement: D-06
    verification:
      - kind: integration
        ref: "scripts/laya_train/lifecycle.py#check_shift_probe"
        status: pass
    human_judgment: false
  - id: D5
    description: "Self-tests, README 1.4.0 claim text, test_harness bindings on FALSIFY-LAYA-GATE-011/-013 and FALSIFY-LAYA-PARITY-006, eval_set_in_distribution binding implemented; committed fixtures unchanged"
    requirement: D-07
    verification:
      - kind: integration
        ref: "just laya-train-selftest && just laya-fixtures && git diff --quiet crates/ && pv validate (08-14-PLAN Task 3 <verify>)"
        status: pass
    human_judgment: false
  - id: D6
    description: "The seeds label kept at the contract's 'mean ± sd over 3 seeds' instead of the plan's wording, and GATE-006's legacy-variance clause no longer having a Python leg"
    verification: []
    human_judgment: true
    rationale: "A contract-versus-plan wording choice and a claim-scope question (D-ITEM-08-14-A/B); plan 08-15's reader or a human decides whether either needs a contract amendment before 08-16's run"

duration: 18min
completed: 2026-09-27
---

# Phase 8 Plan 14: A1/A2/A3 in the Laya back office Summary

**The trainer now ships the median-ECE seed of 13/17/23, writes a bit-recomputable float64 re-score noise record behind a manual-forward control that refuses the run on any mismatch, builds the 459-row in-distribution s64 eval set by rule, and reports the SemEval test split as a shift probe that is proven not to touch the gate. All of it was proven on the tiny CPU checkpoint, with no production run.**

## Performance

- **Duration:** 18 min
- **Started:** 2026-09-27T17:05:05Z
- **Completed:** 2026-09-27T17:23:13Z
- **Tasks:** 3 of 3
- **Files modified:** 12 (plus this SUMMARY)
- **Lifecycle wall time:** 12 s (Task 1, `just laya-train-lifecycle`); 19 s for the full Task 3 verify (`just laya-train-selftest` + `just laya-fixtures` + pv)

## Accomplishments

- **A3, median-ECE seed selection** (`train.py`, `gate.py`, `contract.py`). Production requires exactly `seed_policy.production_seeds_required` = 3 seeds; `--seeds 1` is refused in the data-validation block, before the base is touched. Each seed trains in `seeds/seed-<s>/`. After every checkpoint is fixed, the seeds are ranked by `(floor(ece_post x rank_scale), seed)`, and the median ships as `checkpoint/`. The other two checkpoints are deleted, and every `seeds/seed-<s>/eval-probs.json` is kept and hash-bound in `seeds.per_seed`. `rank_scale`, the seeds, the policy and the tie-break are all read from the contract.
- **A1, float64 noise record** on every run (`rescore-noise.json`, `laya-rescore-noise-v1`). For the shipped checkpoint and the base, a manual fp32 forward must reproduce the Scorer's logits exactly on the first min(5, n) rows, or the run exits 2 with no record and no gate report. The same model is then cast to float64. The record stores every row's float64 probabilities, `max_abs` over exactly the written float32 values, `bound = max(floor, k x max_abs)` and the argmax agreement. k = 4 and floor = 1e-5 are copied from laya-parity-v1.
- **A2, demo data by rule** (`data.in_distribution_heldout`, `prepare_stance.py --cell s64|s16 --out`). `data/decide/tweet-stance-64` holds 192 verified shots, a **459-row eval set with class counts [111, 291, 57]**, and the 280 test rows as `shift.jsonl`. The s16 cell reproduces `data/decide/tweet-stance-16` byte for byte. The script never overwrites different bytes, and a re-run is a no-op.
- **A2, shift probe outside the gate.** It is scored with fresh reloads only after the gate, the median and the noise record, and reported as `shift_probe {gate_clause: false, ...}` plus `inputs_sha256.shift_jsonl`. The lifecycle proves `pass`, `margin`, `zero_shot`, `fine_tuned`, `calibration` and `seeds` identical with and without `shift.jsonl`.
- **The red lifecycle from 08-13 is fixed by rule, not by allow-list.** `run_dir_files(seeds, has_shift)` and `expected_keys(schema, stopping, n, has_shift)` read the contract's own `(only when ...)`, `<s>` and `OPTIONAL` wording, and refuse any condition they do not know.
- **Self-tests, docs and bindings.** 14 median-rule cases, 8 held-out-rule cases (including a rebuild from the pinned splits), `verify_report` re-deriving `seeds.shipped`, the README's "What the gate certifies (1.4.0)" section, `test_harness` on GATE-011, GATE-013 and PARITY-006, and the `eval_set_in_distribution` binding set to `implemented`.

## Evidence

**Task 1 lifecycle lines** (`/tmp/p08-14-t1.log`, tiny checkpoint, CPU):

```
MEDIAN seeds=13,17,23 rank_keys=94,97,95 shipped=23 (policy median_ece, tie_break smaller_seed)
NOISE which=fine_tuned max_abs=4.299346e-08 bound=1.000000e-05 argmax=9/9 t_applied=1.479579
NOISE which=zero_shot max_abs=3.106277e-08 bound=1.000000e-05 argmax=9/9 t_applied=1.750000
```

The single-seed runs logged `NOISE which=fine_tuned max_abs=3.752272e-08` and `NOISE which=zero_shot max_abs=3.106277e-08`. The control was 0.0 on every run. Seed 23 is the median (keys 94 < 95 < 97 give the order 13, 23, 17). Seed 13's `model.safetensors` in the three-seed run is bit-identical to the single-seed run's. The kept run-dir check printed `kept run dir is a 1.4.0 run dir: shipped 23`.

**Task 2 s64 sha256s** printed by `just laya-prepare-stance s64`:

| file | lines | sha256 |
|------|-------|--------|
| eval.jsonl | 459 | `5250218f30686fe47482dbc1d2785d8eaa0a388a3aaf43ecc2c43b03f790e3bb` |
| shift.jsonl | 280 | `e326952585d57b14ecc45f817efbb6994d6464ae07ec9190e0782547b5b20288` (= the s16 eval.jsonl, the same test split) |
| task.json | 9 | `041c4e38aadae98e3e62ecf77c7ace23616449cbab45439a4cf10cbb4a956b94` (= s16) |
| train.jsonl | 192 | `12a4301e5f0fcd50415e8eb237cbf2e5090f9e01f1edfa591dd5270f6df63351` (= spike 027's recorded s64 train sha) |

**Spike-028 cross-check** (read-only): the parsed multiset of (normalized text, label) equals `.planning/spikes/028-laya-packability-noise-floor/data/indist/eval.jsonl` (459 = 459 rows), and the files are also **byte-identical**. Our eval sha equals spike 028's `results/indist-data.json` `eval_jsonl_sha256`.

**Induced-RED controls** (each reverted from a scratch backup, then confirmed green):

1. `select_median_seed` returning `order[0]`: `gate.py --selftest` exited 1 with 6 named FAILs (distinct ECEs shipped 17, rank-key tie shipped 13, exact tie shipped 13, median-13 case shipped 17, the failing-median report refused, the non-median report accepted). Reverted (0 MUTATION markers): `GATE SELFTEST OK`.
2. `in_distribution_heldout` no longer dropping exclusion-group members: `data.py --selftest` exited 1 with 4 named FAILs, including the real-split rebuild (`eval_set.demo_rule built 460 rows [112, 291, 57]`). Reverted: `DATA SELFTEST OK`.
3. Noise control (extra, for T-08-14-01): +1e-6 on the manual fp32 logits made `train.py` exit 2 with `REFUSED noise-control: ... max |dz| 1.000240445137024e-06 on rows [0, 1, 2, 3, 4]`. The run dir held no `rescore-noise.json` and no `gate-report.json`. Reverted.

**Acceptance checks**
- `grep -c 'rank_scale\|pack_rescore_noise_k' contract.py` = 3. In the added lines of `train.py` / `gate.py`, the only `10000` / `4` / `1e-5` hits are prose: the rank_key docstring naming the contract's value, "steps 4-7" and "3-8". Every governing value comes from `contract.noise_policy()` / `seed_selection_decl()`.
- The README section quotes `eval_set.claim` verbatim. This was checked by comparing the YAML-parsed string: `'> ' + claim + '\n' in README` is True.
- `git status --porcelain .planning/spikes/` is empty. All 31 `gate-report.json` mtimes under `models/decide/tweet-stance-16*` and `models/decide/spike-027` are unchanged (`cmp` of the before and after listings). No file under those dirs or `.planning/spikes/` is newer than the plan start. `ls models/decide` is unchanged (cmp). `models/decide/laya-stance-64` does not exist. `git status --porcelain data/ models/` is empty. The `data/decide/tweet-stance-16` sha256s are unchanged.
- `just laya-fixtures` → `FIXTURES OK`, and `git diff --quiet crates/` holds.
- `pv validate`: 0 errors on both contracts. `pv diff` (old HEAD copy vs new) reported **"Contracts are identical."** for both, so no version bump.

## Task Commits

1. **Task 1 (tracer): three-seed median run, float64 noise record, 1.4.0 report, lifecycle.** `4f5ada981` (feat)
2. **Task 2: s64 data by rule and the shift probe outside the gate.** `de58feea6` (feat)
3. **Task 3: self-test cases, README, test_harness bindings, full selftest.** `24cea2a63` (test)
4. Deferred-items log: `4f4674fbd` (docs)

The tracer gate for Task 1 was auto-continued: interactive mode, `end-of-phase`, and `<verify>` is automated-only. It re-ran green before expansion.

## Files Created/Modified

- `scripts/laya_train/contract.py`: `PARITY_CONTRACT`, `noise_policy()`, `seed_selection_decl()`, `resolve_seeds(n, variant)`, `recipe_json(..., seed_selection)`, rule-driven `run_dir_files` / `expected_keys`.
- `scripts/laya_train/gate.py`: `rank_key`, `select_median_seed`, `verify_report` under `median_ece`, the median self-test cases.
- `scripts/laya_train/train.py`: the three-seed path, `manual_logits` / `rescore_noise_set` / `write_noise_record`, `score_shift_probe`, the 1.4.0 report fields.
- `scripts/laya_train/lifecycle.py`: `check_median_run`, `check_noise_record`, `check_shift_probe`, the seed and production refusals, `keep_run` (`LAYA_LIFECYCLE_KEEP`).
- `scripts/laya_train/data.py`: `in_distribution_heldout`, the `shift` role, `refuse_overlap(..., role)`, the held-out self-test cases and the pinned-split rebuild.
- `scripts/laya_train/prepare_stance.py`: `--cell s64|s16`, `--out`, the demo_s64 assertions, write-once.
- `scripts/laya_train/README.md`: the 1.4.0 recipes, steps, run dir, seeds, refusals, the claim section and the s16 history.
- `justfile`: `laya-prepare-stance cell="s64"` and the 1.4.0 `laya-train` comment.
- `contracts/laya-finetune-gate-v1.yaml`: `test_harness` + `implemented_by` on GATE-011 and GATE-013; GATE-006 `implemented_by` corrected.
- `contracts/laya-parity-v1.yaml`: `test_harness` + `implemented_by` on PARITY-006.
- `contracts/aprender/binding.yaml`: `eval_set_in_distribution` set to implemented; `declared_seed_ships` notes updated (status unchanged).
- `deferred-items.md`: D-ITEM-08-14-A and D-ITEM-08-14-B.

## Decisions Made

See `key-decisions` above. The two that change what the plan literally said are listed under Deviations.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Contract wins] The seeds label for a median run stays "mean ± sd over 3 seeds"**
- **Found during:** Task 1 (contract.py `seeds_label`)
- **Issue:** The plan asked for "median-ECE seed of 3 seeds". The committed contract fixes the literal in `seed_policy.rule` and `gate_report_schema.seeds`, and plan 08-15's Rust reader implements the contract.
- **Fix:** Kept the contract's label. `seeds.policy` / `seeds.shipped` carry the median information. Logged as D-ITEM-08-14-A.
- **Files modified:** scripts/laya_train/contract.py
- **Committed in:** 4f5ada981

**2. [Rule 1 - Bug in the plan's check] The "checkpoint tree equals the single-seed run's" comparison excludes `training.recipe_id`**
- **Found during:** Task 1 (lifecycle `check_median_run`)
- **Issue:** A three-seed recipe.json carries `seed_selection`, so its recipe_id differs. That id is written as provenance into `rl_agent_config.json`, so a byte-identical tree is impossible by construction.
- **Fix:** When 13 is the median, compare the tree with that one provenance field removed. Also compare seed 13's `model.safetensors` sha against the single-seed run unconditionally, which is a stronger check that holds whichever seed ships.
- **Files modified:** scripts/laya_train/lifecycle.py
- **Committed in:** 4f5ada981

**3. [Rule 1 - Interpretation] synthetic `--seeds` accepts {1, 3}, not the interval [1, 3]**
- **Found during:** Task 1 (`resolve_seeds`)
- **Issue:** With the legacy multi-seed path removed, N = 2 has no rule: the median needs an odd N.
- **Fix:** Refuse `--seeds 2` like 0 and 4. The lifecycle asserts all three refusals.
- **Committed in:** 4f5ada981

**4. [Rule 1 - Bug] The pinned-split self-test case aborted the whole data self-test on failure**
- **Found during:** Task 3 (the group-member induced-RED)
- **Issue:** `prepare_stance.fail()` calls `sys.exit`, so a real-split failure exited `data.py --selftest` without a named FAIL or the `DATA SELFTEST FAILED` sentinel.
- **Fix:** Catch the exit, record a named FAIL, and continue. Re-running the mutation now shows 4 named FAILs.
- **Committed in:** 24cea2a63

**5. [Rule 2 - Accuracy] FALSIFY-LAYA-GATE-006 `implemented_by` corrected**
- **Found during:** Task 3
- **Issue:** It named a "--seeds 3 run byte-identical to the single-seed run" legacy leg that the 1.4.0 trainer no longer produces.
- **Fix:** It now names what is exercised: legacy single-seed runs, plus `check_median_run`. The prediction text is unchanged. The orphaned legacy multi-seed clause is D-ITEM-08-14-B.
- **Committed in:** 24cea2a63

**6. [Style] New self-test cases print an `ok` line, like every existing case**
- The plan said the new cases should "print nothing on success". I followed the files' existing `case()` convention instead, so each case prints `ok` on success and `FAIL <name>` on failure. The sentinels are unchanged.

**7. [Scope] The shift-invariance "without" run reuses the lifecycle's three-seed run**
- That run has the same recipe and byte-identical data files, and the plan's shift run adds only `shift.jsonl`. The comparison also covers `recipe_id`, `eval_probs_sha256`, `zero_shot_probs_sha256` and `rescore_noise_sha256`.

---

**Total deviations:** 7 (4 Rule 1, 1 Rule 2, 2 style/scope notes). **Impact:** none of them moves a declared value. Two are logged as deferred items for 08-15 or a human.

## Issues Encountered

- `pv validate contracts/aprender/binding.yaml` fails with `missing field metadata`. The HEAD copy fails identically, because binding.yaml is not a contract. The YAML parses, and the flipped row reads `implemented justfile laya-train-selftest`.

## Known Stubs

None. The one intentionally absent artifact is `models/decide/laya-stance-64`: the one declared run belongs to plan 08-16.

## Threat Flags

None beyond the plan's threat model. T-08-14-01 through T-08-14-06 are mitigated as planned. T-08-14-01 was additionally demonstrated by the induced control mismatch.

## Next Phase Readiness

- **08-15:** `LAYA_LIFECYCLE_KEEP=<dir> just laya-train-lifecycle` exports a complete 1.4.0 run dir: three seeds, the shift probe, `rescore-noise.json`, and the `seeds/` files. The Rust reader and `check_seed_selection` / `check_shift_probe` / the noise recompute can be written against it. Before that, read D-ITEM-08-14-A (label) and D-ITEM-08-14-B (legacy clause).
- **08-16:** `data/decide/tweet-stance-64` exists by rule. The run dir `models/decide/laya-stance-64` does not exist yet and is 08-16's to create.

## User Setup Required

None.

## Self-Check: PASSED

- Files exist: all 11 modified source/contract files and deferred-items.md are present and committed.
- Commits exist: 4f5ada981, de58feea6, 24cea2a63 and 4f4674fbd are on `gsd/phase-2-contract-gate` (`git rev-list e74c1fa38..HEAD` = 4 before this SUMMARY).
- Every task `<verify>` block re-ran green: Task 1 rc 0, Task 2 rc 0, Task 3 rc 0.

---
*Phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server*
*Completed: 2026-09-27*
