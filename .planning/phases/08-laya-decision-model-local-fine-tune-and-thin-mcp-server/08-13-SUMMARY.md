---
phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
plan: 13
subsystem: contracts
tags: [laya, contracts, pv, gate, calibration, seed-policy, parity, rescore, float64-noise, gap-closure]
status: complete
gap_closure: true

requires:
  - phase: 08-09
    provides: "aprender_decide::verify / pack_laya policy() reading pack_rescore_probs_abs; the two fail-closed demo vectors and FALSIFY-LAYA-GATE-010"
  - phase: 08-11
    provides: "the D-18 HOLD that this gap-closure set (08-13..08-18) works to lift"
provides:
  - "laya-parity-v1 2.0.0 (A1): noise-referenced pack/verify re-score bar bound(c, s) = max(1.0e-5, 4 x noise(c, s)), ceiling 1.0e-3, absent record = floor; rescore_noise_reference; FALSIFY-LAYA-PARITY-006 (staged)"
  - "laya-finetune-gate-v1 AMENDMENT 1.4.0, published as 2.0.0 (A2 eval_set + shift probe, A3 median-ECE seed policy, demo_s64 declared with outcome pending, rescore_noise_schema, FALSIFY-LAYA-GATE-011..014 staged)"
  - "decide-apr-v1 deploy_eligibility.verify_checks name the noise-referenced bound and the seed-selection re-derivation"
  - "four status: pending binding rows (rescore_noise_reference, seed_selection_median, eval_set_in_distribution, shift_probe_reported)"
  - "dated amendment notes under D-07, D-08, D-17, D-19 in 08-CONTEXT.md"
affects: [08-14, 08-15, 08-16, 08-17, 08-18, 08-12]

actuals:
  tokens: 17864      # chars/4 over the realized diff 279e13644..HEAD (12111 over added lines only)
  tasks: 3
  commits: 3         # MEASURED: git rev-list --count 279e13644..HEAD before the SUMMARY commit
plan_head_before: 279e1364490e0f46b0f2a1d77546ba85a7a964ea

tech-stack:
  added: []
  patterns:
    - "Declare-before-read amendment: every rule a future run is judged by is committed, pv-valid and version-bumped before its data dir or run exists (test ! -e proof in the plan's final commit)"
    - "Additive amendment beside the 1.x keys readers consume, so no reader breaks; new FALSIFY ids carry implemented_by prose only (staged binding)"

key-files:
  created:
    - .planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-13-SUMMARY.md
  modified:
    - contracts/laya-parity-v1.yaml
    - contracts/laya-finetune-gate-v1.yaml
    - contracts/decide-apr-v1.yaml
    - contracts/aprender/binding.yaml
    - .planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-CONTEXT.md
    - .planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/deferred-items.md

key-decisions:
  - "USER APPROVAL (2026-09-27): option 1 from spikes 027/028 and plans 08-13..08-18 as written (279e13644), including the median tie-break floor(ece_post x 10000) with ties to the smaller seed, stated as selected with eval labels"
  - "USER APPROVAL (2026-09-27): pack_rescore_bound_max_abs = 1.0e-3 (the planner's declared ceiling)"
  - "USER APPROVAL (2026-09-27): both s16 fail-closed vectors kept; non-shipped seeds recomputed but not re-scored; final_norm_abs and logits_abs scoped fixture-only"
  - "laya-parity-v1 1.0.0 -> 2.0.0 exactly as pv diff suggests (major: pack_rescore_probs_abs formula changed)"
  - "laya-finetune-gate-v1 1.3.0 -> 2.0.0 exactly as pv diff suggests (major), while the amendment keeps its plan-set name 1.4.0; the contract states that every 1.4.0 reference means this amendment"
  - "decide-apr-v1 stays 1.0.0: pv diff reports it identical (it does not diff deploy_eligibility), the 08-09 precedent"

patterns-established:
  - "Noise-referenced tolerance: bound = max(floor, k x the reference implementation's own distance from float64), recomputed by the verifier from a hash-bound record, never taken from the run"

requirements-completed: [D-06, D-07, D-08, D-17, D-19]

coverage:
  - id: D1
    description: "laya-parity-v1 A1 declared: constants pack_rescore_noise_k 4 and pack_rescore_bound_max_abs 1.0e-3, floor 1.0e-5 unchanged, rescore_noise_reference, FIXTURE-ONLY scoping, x86_64 risk, FALSIFY-LAYA-PARITY-006 staged, version 2.0.0"
    requirement: "D-17"
    verification:
      - kind: other
        ref: "08-13-PLAN.md Task 1 <verify><automated> (pv validate 0 errors on laya-parity-v1 and decide-apr-v1; yaml assertions; cargo test -p aprender-decide --lib 93 passed; laya-verify on the synthetic fixture REFUSED SyntheticNotDeployable exit 2)"
        status: pass
    human_judgment: false
  - id: D2
    description: "laya-finetune-gate-v1 AMENDMENT 1.4.0 (published 2.0.0): eval_set (A2), seed_policy median rule (A3), demo_s64 pending, s16 demo record intact, schemas extended, GATE-011..014 staged"
    requirement: "D-07"
    verification:
      - kind: other
        ref: "08-13-PLAN.md Task 2 <verify><automated> (pv validate 0 errors; yaml assertions; metrics/data/gate --selftest OK; cargo test -p aprender-decide --lib 93 passed; fail_closed_vectors unarmed 1 passed)"
        status: pass
    human_judgment: false
  - id: D3
    description: "The claim change (in-distribution only, NOT shift robustness) and the median-selected-with-eval-labels honesty are written plainly enough for a reader"
    verification: []
    human_judgment: true
    rationale: "Whether the prose states the claim change honestly and plainly is a judgment the automated grep for NOT/shift/027 cannot make"
  - id: D4
    description: "08-CONTEXT.md amendment notes (D-07, D-08, D-17, D-19), deferred-items statuses, audit, contract-validate, binding guard (lifted copy), no code changed, no s64 data/run dir exists"
    requirement: "D-19"
    verification:
      - kind: other
        ref: "08-13-PLAN.md Task 3 <verify><automated> (4 notes; B superseded, C resolved, 08-13-A present; test ! -e both s64 paths; make contract-audit-phase8 exit 0; make contract-validate passed; git diff start..HEAD -- scripts/ crates/ justfile Makefile .github/ empty; BIND GUARD LIFTED OK)"
        status: pass
    human_judgment: false

duration: 11min
completed: 2026-09-27
---

# Phase 8 Plan 13: Declare the option-1 amendments (A1 noise-referenced bar, A2 in-distribution eval set, A3 median seed) Summary

**laya-parity-v1 2.0.0 now re-scores the packed model within max(1e-5, 4 x torch's own fp32-vs-float64 noise), recomputed in Rust from a hash-bound record. laya-finetune-gate-v1's AMENDMENT 1.4.0 (published as 2.0.0) gates on a 459-row in-distribution held-out set and ships the median-ECE seed of 13/17/23. The SemEval test split becomes a reported shift probe. All of this was declared before any s64 data or run exists.**

## Performance

- **Duration:** about 11 min
- **Started:** 2026-09-27T16:50:46Z
- **Completed:** 2026-09-27T17:01:51Z
- **Tasks:** 3/3
- **Files modified:** 6
- **Start SHA** (`/tmp/p08-13-start.sha`): `279e1364490e0f46b0f2a1d77546ba85a7a964ea`

## User decisions recorded (2026-09-27)

On 2026-09-27 the user approved **Option 1 from spikes 027/028**. They also approved plan set 08-13..08-18 as written (commit `279e13644`), including five baked-in choices, each now in the contracts:

1. **Median-seed tie-break.** The rank key is `floor(ece_post x 10000)`, and ties go to the smaller seed. The contract states plainly that the median is selected **with eval labels**. Evidence: `seed_policy.rank_rule`, `seed_policy.tie_break`, `seed_policy.honesty`.
2. **`pack_rescore_bound_max_abs` is 1.0e-3.** This is the planner's declared value, flagged for review, and it only ever refuses. Its rationale sits next to the laya-parity-v1 constant.
3. **Both s16 fail-closed vectors are kept.** Their refusals are restated under A1/A3 as unchanged, in `demo.fail_closed_rule`, `demo.vectors_scope` and FALSIFY-LAYA-GATE-010.
4. **Non-shipped seeds are recomputed but not re-scored.** See `seed_policy.retention_rule` and the SCOPE invariant of `pack_rescore_probs_abs`.
5. **`final_norm_abs` and `logits_abs` are fixture-only.** Their `domain:` begins `FIXTURE-ONLY:`, and no value moved.

## Accomplishments

- **A1 (laya-parity-v1 2.0.0):**
  - The pack/verify bar is now `bound(c, s) = max(float_tolerance, pack_rescore_noise_k x noise(c, s))`, computed per checkpoint (fine-tuned from the packed bytes, and the base) and per scored set. Argmax stays exact.
  - Rust derives the bound itself. It recomputes the noise from the stored float64 rows and never uses a reported value.
  - A run with no record is held to the 1.0e-5 floor, and a bound above 1.0e-3 is refused.
  - `rescore_noise_reference` carries the control rule (0.0), the argmax rule, the RoPE `inv_freq` caveat and the correlated-error caveat.
  - x86_64 is recorded as UNMEASURED in `qa_gate`.
- **A2 (gate `eval_set`):**
  - The claim is stated plainly. The set is defined by rule: 459 rows, classes [111, 291, 57].
  - The SemEval test split (280 rows, `shift.jsonl`) is reported, recomputed, never re-scored and never a gate clause.
- **A3 (gate `seed_policy`):**
  - Rank key, tie-break, pass rule, report rule, retention rule, honesty text, quantization rationale and legacy rule. Readers still get `declared_seed` and `variance_seeds` unchanged.
- **`demo_s64`:** the amended D-19 cell, with outcome `pending`, exactly one declared run and a stop rule. Every value of the s16 `demo` block is unchanged. The block gains `superseded_by`, a rewritten `fail_closed_rule` (this resolves D-ITEM-08-09-C) and `vectors_scope`.
- **Four pending binding rows**, plus rewritten notes on the `pack_rescore_probs_abs` and `declared_seed_ships` rows.
- **08-CONTEXT.md:** four append-only amendment notes. **deferred-items.md:** 08-09-B superseded, 08-09-C resolved, and a new D-ITEM-08-13-A (x86_64).

The eval_set `claim`, verbatim:

> The gate now certifies margin and calibration on held-out data drawn like the tenant's shots. It does NOT certify robustness to a shifted input population.

## Task Commits

1. **Task 1 (tracer): declare A1 in laya-parity-v1.** `fc0c1dd30` (feat). `git show --name-only` lists exactly `contracts/aprender/binding.yaml`, `contracts/decide-apr-v1.yaml` and `contracts/laya-parity-v1.yaml`.
2. **Task 2: declare A2 and A3 in laya-finetune-gate-v1.** `a3e155170` (feat). It touches only `contracts/aprender/binding.yaml` and `contracts/laya-finetune-gate-v1.yaml`.
3. **Task 3: record the amendments, audit, and prove the declaration came first.** `56f888d62` (docs). It touches only `08-CONTEXT.md` and `deferred-items.md`.

**Plan metadata:** the docs commit that adds this SUMMARY.

## pv diff outputs and applied versions

In-tree pv 0.63.0 (`. scripts/pv_bin.sh`). The old files were materialized with `git show HEAD:<path>` into the session scratchpad.

| Contract | `pv diff` said | Applied |
|---|---|---|
| laya-parity-v1 | `Contract diff: v1.0.0 → v1.0.0` / **`Suggested bump: major`**. Equations: `+ rescore_noise_reference`; `~` domain and invariants of `final_norm_abs` and `logits_abs`; `~` formula, domain, codomain and invariants of `pack_rescore_probs_abs`. `+` the bound proof obligation. `+ FALSIFY-LAYA-PARITY-006` | **1.0.0 → 2.0.0** (re-diff: `v1.0.0 → v2.0.0`) |
| decide-apr-v1 | **`Contracts are identical.`** pv diff does not compare `deploy_eligibility` | **stays 1.0.0**, as in 08-09 |
| laya-finetune-gate-v1 | `Contract diff: v1.3.0 → v1.3.0` / **`Suggested bump: major`**. Equations: `+ eval_set_in_distribution`, `+ seed_selection_median`, `+ shift_probe_reported`; `~` formula, domain and invariants of `declared_seed_ships`; `~` formula and domain of `gate_pass`. Obligations: four `+` and one `-` (the declared-seed invariant, re-scoped to legacy). `+ FALSIFY-LAYA-GATE-011..014` | **1.3.0 → 2.0.0** (re-diff: `v1.3.0 → v2.0.0`) |

`pv validate` reported `0 error(s), 0 warning(s)` on all three contracts.

## Verification evidence

- **Task 1:** `A1 declared`. `cargo test -p aprender-decide --lib` gave 93 passed. On the synthetic fixture, `just laya-pack-fixture` then `just laya-verify` printed `REFUSED SyntheticNotDeployable recipe variant "synthetic-fixture" is not deployable (nothing written)` with exit 2. That proves `pack_laya` `policy()` still parses the amended laya-parity-v1.
- **Task 2:**
  - Python checks: `A2 + A3 declared, s16 record intact`. `METRICS SELFTEST OK (5 frozen cases replayed within 1e-06)`, `DATA SELFTEST OK` and `GATE SELFTEST OK`.
  - Rust checks: lib 93 passed, and the unarmed `fail_closed_vectors` target gave 1 passed.
  - **The expected, recorded exception:** `just laya-train-lifecycle` exits 1 with `LIFECYCLE FAILED: run dir is missing rescore-noise.json`. Its `run_dir_files()` treats every `run_dir_layout.run_dir` entry except variance-report.json as required, and the new 1.4.0 entries are optional or conditional. Plan 08-14 Task 1 teaches it those rules. It was not patched here (no code in this plan), and CI never runs the lifecycle.
- **Task 3:**
  - Proof that the declaration came first: `test ! -e models/decide/laya-stance-64 && test ! -e data/decide/tweet-stance-64`. Both are absent at the final task commit `56f888d62`.
  - `make contract-audit-phase8` printed `Phase 8 binding audit: 4 contract(s) audited, every equation is bound` (exit 0). `make contract-validate` printed `Contract validation passed`.
  - `git diff 279e13644 -- scripts/ crates/ justfile Makefile .github/` is empty.
  - The strict-binding guard is VACUOUS (D-ITEM-08-01-A), so the lifted-copy fallback ran: `Resolved 661 test references; 44 dangling across 15 contract(s).` Only `FAIL contracts/chronos-bolt-parity-v1.yaml` (9) and `FAIL contracts/setfit-encoder-conformance-v1.yaml` (8) remain, both pre-existing, so the result is `BIND GUARD LIFTED OK`. `.pv/lint-previous.json` is byte-identical to its state before the run (`cmp`).
  - `git diff 279e13644 HEAD -- 08-CONTEXT.md` has no removed lines, so every original D-07, D-08, D-17 and D-19 sentence is kept.

## Files Created/Modified

- `contracts/laya-parity-v1.yaml`: A1. The constants, the rewritten `pack_rescore_probs_abs`, the new `rescore_noise_reference`, FIXTURE-ONLY scoping, the bound obligation, PARITY-006, the reworded PARITY-005 prediction and the qa_gate x86_64 entry.
- `contracts/laya-finetune-gate-v1.yaml`: AMENDMENT 1.4.0 (version 2.0.0). The `eval_set`, `demo_s64` and `rescore_noise_schema` blocks, the extended `seed_policy`, the schema and run-dir additions, the three new equations, the two amended ones, the obligations, GATE-006/010 amended and GATE-011..014 added.
- `contracts/decide-apr-v1.yaml`: the `deploy_eligibility.verify_checks` wording, and one added seed-selection item.
- `contracts/aprender/binding.yaml`: four pending rows and two rewritten notes.
- `08-CONTEXT.md`: amendment notes under D-07, D-08, D-17 and D-19.
- `deferred-items.md`: 08-09-B superseded, 08-09-C resolved, 08-13-A added.

## Decisions Made

- Every version follows `pv diff`, including the major bump of the gate contract, which the plan called "1.4.0". The name is kept because downstream plans (08-14 mentions it 10 times) and the committed Task-1 text use it. No downstream plan asserts a version string.
- The ceiling of 1.0e-3 is recorded as the planner's value and marked approved by the user on 2026-09-27.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] The gate contract's pv-suggested version is 2.0.0, not the 1.4.0 the plan named**
- **Found during:** Task 2.
- **Issue:** `pv diff` suggests **major**, because the `gate_pass` and `declared_seed_ships` formulas were amended as the plan requires. The plan's hard rule is "at the version `pv diff` suggests", but its text, the committed Task-1 text and plan 08-14 all say "1.4.0".
- **Fix:** `metadata.version` is set to 2.0.0. A NAMING sentence was added to the amendment paragraph: every "1.4.0" in the three contracts means this amendment, published as 2.0.0. The binding notes and CONTEXT notes say "1.4.0 (published 2.0.0)".
- **Files modified:** contracts/laya-finetune-gate-v1.yaml.
- **Verification:** the re-diff reads `v1.3.0 → v2.0.0`. The Task 2 assertion (`version != '1.3.0'`) passes.
- **Committed in:** `a3e155170`.

**2. [Rule 1 - Correctness] `run_dir_layout`'s variance-report entry was kept byte-identical**
- **Found during:** Task 2, item 7.
- **Issue:** Item 7 asks both for the variance-report note to become "always for production (three seeds)" and for every existing entry to stay byte-identical.
- **Fix:** The entry is kept byte-identical and the new wording went into `run_dir_layout.run_dir_notes.variance_report`, a sibling key no reader iterates. The compatibility note for `laya-gate-report-v1` was also folded into the `schema` field's value rather than a new key. A new key would have been one more non-field key that 08-14's lifecycle key-set comparison must skip.
- **Committed in:** `a3e155170`.

**3. [Rule 2 - Honesty] Scope notes the plan did not list**
- **What was added:**
  - FALSIFY-LAYA-GATE-006's `implemented_by` now says the median leg is STAGED to 08-14. The legacy legs remain under the existing `test_harness`.
  - The existing declared-seed proof obligation is re-scoped to the legacy rule. pv diff shows it as a removal plus an addition.
  - One `bound` proof obligation was added to laya-parity-v1 for A1.
  - The qa_gate lines on top-level keys and declared seeds were brought up to date.
- **Why:** Without these, an unchanged `test_harness` would appear to cover the new median rule, and the old invariant would contradict A3.
- **Committed in:** `fc0c1dd30` and `a3e155170`.

---

**Total deviations:** 3 auto-fixed (1 blocking, 1 correctness, 1 honesty).
**Impact on plan:** No threshold, tolerance value, recipe value, s16 demo value or reader key moved. No code changed.

## Issues Encountered

- `just laya-train-lifecycle` is red, as the plan predicted and recorded above. Plan 08-14 Task 1 owns it.
- pv diff cannot see changes to non-equation top-level keys such as `deploy_eligibility`, `seed_policy`, `eval_set` or `demo_s64`. The version bumps therefore rest on the formula changes. The decide-apr-v1 wording change carries no bump, following the 08-09 precedent.

## Known Stubs

None. The FALSIFY entries without `test:` lines (PARITY-006 and GATE-011..014) and the four pending binding rows are declared staged bindings. Plans 08-14 and 08-15 implement them and 08-16 arms GATE-014.

## Threat Flags

None. The plan's threat register covers this plan's changes. T-08-13-01 is mitigated by the `test ! -e` proof and by the pv versioning. T-08-13-06 (the ceiling) was accepted and then approved by the user.

## User Setup Required

None.

## Next Phase Readiness

- **Plan 08-14** (the Python writer): `rescore-noise.json`, `seed_selection`, the median selection, the eval-set builder and the shift probe. It must also teach `lifecycle.py` the optional and conditional 1.4.0 entries, since the lifecycle is red until then.
- **Plan 08-15** (the Rust verifier): `rescore_bounds`, `check_seed_selection`, `check_shift_probe`, `SeedPolicyMissing` and `demo_run.rs`.
- No `data/decide/tweet-stance-64` or `models/decide/laya-stance-64` exists yet. The one declared run (08-16) will be judged by rules committed before it.

## Self-Check: PASSED

- FOUND: contracts/laya-parity-v1.yaml, contracts/laya-finetune-gate-v1.yaml, contracts/decide-apr-v1.yaml, contracts/aprender/binding.yaml
- FOUND commits: fc0c1dd30, a3e155170, 56f888d62
- Artifact `contains` patterns: `rescore_noise_reference` (laya-parity-v1), `demo_s64` (gate), `rescore-noise.json` (decide-apr-v1), `seed_selection_median` (binding), and `Amended 2026-09-27` 4x (CONTEXT)

---
*Phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server*
*Completed: 2026-09-27*
