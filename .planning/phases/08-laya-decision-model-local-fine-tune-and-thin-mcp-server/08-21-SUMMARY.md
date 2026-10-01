---
phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
plan: 21
subsystem: gate-verifier
tags: [laya-finetune-gate-v1, verify, provenance, class-A, run_field_bindings, CR-01, V6-b, V6-c, V6-d, WR-08, WR-02, V8-c, V8-d, AL2, S4, mutation-proof]

requires:
  - phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
    provides: "08-09 verify pipeline, 08-12 contract audit, 08-15 seed/shift/noise checks, 08-19 manifest bindings (load side), ea940faec verify-side ArtifactNotFromRun"
provides:
  - "GateContractView / ParityContractView serde views and VerifyPolicy::from_contract_views: the ONE contract-to-policy mapping (CLI, tests/common, verify::tests)"
  - "check_base over family, checkpoint, repo, revision, then sha256 (BaseWhich::{Family, Checkpoint, Repo, Revision})"
  - "base-dir pins encoder_config_sha256 / rl_agent_config_sha256 / tokenizer_json_sha256, checked on the exact bytes the zero-shot baseline is built from (BaseWhich::{EncoderConfig, AgentConfig})"
  - "check_recipe_block (recipe values, schedule literal, seed, epoch rule, early_stopping) and check_recipe_shots (shots_per_class == largest train.jsonl class count); VerifyError::RecipeMismatch { field }"
  - "slice fraction rule max(min_per_class, ceil(fraction x n_class)) in check_split"
  - "check_report_record (t_applied == clamp(t_fitted), clamp_hit, device_is_cpu), fine_tuned.nll recomputed, seeds.label literal and shipped per_seed.t_applied bound; VerifyError::RecordMismatch"
  - "write_atomic: create_new under a pid+counter unique name; ProbsRowCoverage names its file; verify_run cfg(test) pub(crate)"
  - "laya-finetune-gate-v1 run_field_bindings (94 rows) and the sweep every_run_field_is_bound_or_report_only + run_field_bindings_table_matches_fixture_leaves"
affects: [08-26, 08-27, 08-31, aprender-mcp-decide, aprender-mcp-decide-lambda]

actuals:
  tokens: 30800
  tasks: 3
  commits: 3
plan_head_before: 23daf2f2a60dc35bf2face534f0472387cfe3a37

tech-stack:
  added: []
  patterns:
    - "Typed contract views: the library declares serde views (not deny_unknown_fields) and the only mapping; callers parse YAML into them, so a malformed contract value fails identically everywhere"
    - "Run-field sweep: each leaf of the fullest run copy mutated alone, recorded hashes resealed, verified; the contract table decides bound (refusal naming it) vs report_only (verdict unchanged) vs expected-open (listed)"
    - "Split rows: `only: shipped_seed_row` and `membership_report_only` let one leaf pattern be bound where a file can bind it and report-only where none can, each half proved by the sweep"

key-files:
  created: []
  modified:
    - crates/aprender-decide/src/verify.rs
    - crates/aprender-decide/src/verify/tests.rs
    - crates/aprender-decide/examples/pack_laya.rs
    - crates/aprender-decide/tests/common/mod.rs
    - crates/aprender-decide/tests/fail_closed_vectors.rs
    - contracts/laya-finetune-gate-v1.yaml
    - contracts/decide-apr-v1.yaml

key-decisions:
  - "verify_run is cfg(test) pub(crate), not just pub(crate): once private it had no production caller and clippy -D warnings refused it as dead code; the two eligibility doors, pack_for_serving and verify_path, keep their own order (cheap checks before packing)"
  - "Leaves the sweep found unbound were bound where the contract states a machine rule (fine_tuned.nll, the calibration_fit_bounded clamp relation, device_recorded, the seeds.label literal, the shipped per_seed.t_applied), and left report_only only where no file verify holds can decide them (ece_pre, device_used, torch_version, non-shipped per_seed t_applied / model hash, slice membership)"
  - "The base-dir file pins are checked once, in load_declared_base, on the bytes the scorer is built from, not also in check_base: one site per pin keeps each mutation-provable, at the cost of refusing a bad base config only after the fine-tuned re-score"
  - "The calibration_fit_bounded invariant text was NOT edited: it never claimed the fraction was un-derived, and pv diff would have required a major bump for an invariant-text change; the contract additions (base pins, schedule_literal, run_field_bindings) leave pv diff at identical claims"
  - "decide-apr-v1 verify_checks[2] rewritten to state what check_base and the zero-shot load now bind (it said plan 08-21, NOT checked yet)"

patterns-established:
  - "A new recipe.json or gate-report.json field needs a run_field_bindings row before a run carrying it verifies: the table-match test fails on an unlisted leaf or a stale row"
  - "Mutation proof per single comparison: disable one, run the sweep plus the named negative, record RED, restore by byte compare"

requirements-completed: [D-04, D-06, D-07, D-17]

coverage:
  - id: D1
    description: "One typed contract-to-policy mapping used by pack_laya, tests/common and verify::tests; a non-integer ece_bins fails to deserialize for every caller"
    requirement: D-07
    verification:
      - kind: unit
        ref: "crates/aprender-decide/src/verify/tests.rs#policy_views_refuse_non_integer_ece_bins"
        status: pass
      - kind: integration
        ref: "cargo test -p aprender-decide --tests (every target compiles through common::policy())"
        status: pass
    human_judgment: false
  - id: D2
    description: "verify refuses a run whose declared base differs from the contract in any identity field, and a base dir whose encoder config, agent config or tokenizer does not hash to the contract pins"
    requirement: D-04
    verification:
      - kind: unit
        ref: "crates/aprender-decide/src/verify/tests.rs#base_identity_mismatch_revision"
        status: pass
      - kind: unit
        ref: "crates/aprender-decide/src/verify/tests.rs#base_encoder_config_unpinned_refused"
        status: pass
      - kind: unit
        ref: "crates/aprender-decide/src/verify/tests.rs#base_agent_config_unpinned_refused"
        status: pass
      - kind: unit
        ref: "crates/aprender-decide/src/verify/tests.rs#base_mismatch_tokenizer"
        status: pass
    human_judgment: false
  - id: D3
    description: "verify refuses a recipe.json that differs from the contract recipe block (values, schedule, seed, epoch rule, early_stopping, shots) and a calibration slice below the fraction rule"
    requirement: D-06
    verification:
      - kind: unit
        ref: "crates/aprender-decide/src/verify/tests.rs#recipe_block_differs_from_contract_refused"
        status: pass
      - kind: unit
        ref: "crates/aprender-decide/src/verify/tests.rs#recipe_epochs_break_the_rule_refused"
        status: pass
      - kind: unit
        ref: "crates/aprender-decide/src/verify/tests.rs#slice_below_fraction_refused"
        status: pass
      - kind: unit
        ref: "crates/aprender-decide/src/verify/tests.rs#slice_at_fraction_accepted"
        status: pass
    human_judgment: false
  - id: D4
    description: "Every recipe.json and gate-report.json leaf is bound or report-only per run_field_bindings, proved by the sweep; the table equals the fixture's leaf patterns"
    requirement: D-17
    verification:
      - kind: unit
        ref: "crates/aprender-decide/src/verify/tests.rs#every_run_field_is_bound_or_report_only"
        status: pass
      - kind: unit
        ref: "crates/aprender-decide/src/verify/tests.rs#run_field_bindings_table_matches_fixture_leaves"
        status: pass
      - kind: other
        ref: "scratch mutate.py: 22 single-check disables, each RED on a named test (table below)"
        status: pass
    human_judgment: false
  - id: D5
    description: "Small verify defects closed: write_atomic never follows a planted symlink (V8-c), ProbsRowCoverage names its file (V8-d)"
    verification:
      - kind: unit
        ref: "crates/aprender-decide/src/verify/tests.rs#write_atomic_does_not_follow_a_planted_symlink"
        status: pass
      - kind: unit
        ref: "crates/aprender-decide/src/verify/tests.rs#probs_row_coverage_names_the_file"
        status: pass
    human_judgment: false
  - id: D6
    description: "The deployed artifact 24a44d7e still verifies deploy_eligible with shipped seed 17, and both 1.x fail-closed vectors keep their documented refusals"
    requirement: D-07
    verification:
      - kind: e2e
        ref: "heavy just laya-verify models/decide/laya-stance-64.apr models/decide/laya-stance-64 data/decide/tweet-stance-64 <snapshot 55cf4c4e>"
        status: pass
      - kind: e2e
        ref: "heavy env LAYA_FAIL_CLOSED_VECTORS=1 LAYA_MODEL_DIR=<snapshot> cargo test -p aprender-decide --release --test fail_closed_vectors"
        status: pass
    human_judgment: false

duration: 50min
completed: 2026-09-28
status: complete
---

# Phase 8 Plan 21: Class-A Run-Dir Binding at Verify Summary

**`pack_laya verify` now binds every leaf of a run dir's recipe.json and gate-report.json.** Each leaf is bound to the laya-finetune-gate-v1 contract, to a file hash, or to a Rust recomputation. The exceptions are listed report-only with a reason. The contract's new `run_field_bindings` table (94 leaf patterns) drives a sweep that mutates each of the 121 leaves alone. The contract is read through one typed mapping, `VerifyPolicy::from_contract_views`, everywhere. The deployed artifact 24a44d7e still verifies eligible, and both fail-closed vectors keep their documented refusals.

## Performance

- **Duration:** 50 min
- **Started:** 2026-09-28T16:19:22Z
- **Completed:** 2026-09-28T17:09:22Z
- **Tasks:** 3
- **Files modified:** 7

## Accomplishments

- **One typed mapping.** `GateContractView` / `ParityContractView` are serde views (not `deny_unknown_fields`), and `VerifyPolicy::from_contract_views` is the only mapping. `examples/pack_laya.rs`, `tests/common/mod.rs` and `verify::tests` all use it.
  - The mapping existed in four copies, which disagreed on `ece_bins`: `as_u64` in the CLI, `as_f64 as u64` in the tests. A non-integer `ece_bins: 15.0` now fails to deserialize for every caller (AL2, V12-b).
  - `tests/fail_closed_vectors.rs` declares `mod common;`. It has dropped its copies of `workspace_root`, `contract`, `f64_at`, `policy` and `sha256_file`, and now hashes with the library's `artifact_sha256_hex` (S4).
- **Base binding.** `VerifyPolicy.base: BasePins` replaces `base_sha256`.
  - `check_base` compares family, checkpoint, repo and revision, then the sha256 (V6-b).
  - `load_declared_base` reads `model.safetensors`, `encoder/config.json`, `rl_agent_config.json` and `tokenizer/tokenizer.json` once each, and hashes those bytes against the contract pins. The tokenizer is also hashed against the run's `inputs_sha256.tokenizer_json`. It then builds the scorer from exactly those bytes (V6-c).
- **Recipe binding.**
  - `check_recipe_block` compares the recipe values (floats bit for bit), the `schedule` literal, `seed == declared_seed`, the epoch rule and every `early_stopping` field. An absent `early_stopping`, the declared fixed_epochs rule, is accepted.
  - `check_recipe_shots` binds `shots_per_class` to the largest class count in train.jsonl, the value `train.py` writes (V6-d).
- **Slice fraction.** `check_split` requires `max(calibration_slice_min_per_class, ceil(calibration_slice_fraction x n_class))` slice rows per class, as `data.py calibration_split` does (WR-08).
- **Small verify defects.**
  - `write_atomic` opens its temp file `create_new` under a unique pid+counter name. It skips an occupied name, and on failure removes only a file it created (V8-c).
  - `ProbsRowCoverage` Display names its file (V8-d).
  - `verify_run` is `cfg(test) pub(crate)` (WR-02).
- **Sweep-found bindings (see Deviations).**
  - `fine_tuned.nll` is recomputed with aprender-core `log_loss`.
  - `t_applied == clamp(t_fitted)`, `clamp_hit`, and `device_is_cpu == (device_used == "cpu")` are checked in `check_report_record`.
  - The `seeds.label` literal and the shipped row's `per_seed.t_applied` are checked in `check_seeds_block`.
- **Contract additions**, which pv diff reports as identical claims:
  - the `base` pins;
  - `recipe.schedule_literal`;
  - the `run_field_bindings` table.
- **decide-apr-v1:** `verify_checks[2]` now states what is bound.

## Task Commits

1. **Task 1 (tracer): one typed contract-to-policy mapping and check_base over every base identity field**, `25d9ead18` (feat)
2. **Task 2: pin the base dir's config files, bind the recipe block, enforce the slice fraction**, `2b7149fa0` (feat)
3. **Task 3: run_field_bindings table and the class-A verify-side sweep; mutation-proved; real evidence unchanged**, `78a3683ec` (test)

**Plan metadata:** the docs commit that follows this SUMMARY.

## Class-A verify-side surface: `run_field_bindings`, one row per leaf pattern

Totals: 94 rows over 121 concrete leaves.
- 86 bound. Two of these are bound only on the shipped seed's row, and one binds only the structure of its list.
- 3 report_only.
- 5 expected-open: the f_avg rows, owned by plan 08-27.

The sweep printed: `RUN FIELDS bound=86 report_only=3 expected_open=5 (rows; 114 leaves swept, 7 expected-open leaves listed)`. Every expected-open leaf is currently ACCEPTED after mutation. That is the V-class gap plan 08-27 closes.

| # | File | Leaf pattern | Class | Bound by / reason | Refusal names |
|---|---|---|---|---|---|
| 1 | recipe.json | `/variant` | bound | check_variant | `is not deployable` |
| 2 | recipe.json | `/optimizer` | bound | check_recipe_block | `recipe.json optimizer is` |
| 3 | recipe.json | `/encoder_lr` | bound | check_recipe_block | `recipe.json encoder_lr is` |
| 4 | recipe.json | `/head_lr` | bound | check_recipe_block | `recipe.json head_lr is` |
| 5 | recipe.json | `/eta_min` | bound | check_recipe_block | `recipe.json eta_min is` |
| 6 | recipe.json | `/weight_decay` | bound | check_recipe_block | `recipe.json weight_decay is` |
| 7 | recipe.json | `/grad_clip` | bound | check_recipe_block | `recipe.json grad_clip is` |
| 8 | recipe.json | `/batch_size` | bound | check_recipe_block | `recipe.json batch_size is` |
| 9 | recipe.json | `/proper_reward_w_sph` | bound | check_recipe_block | `recipe.json proper_reward_w_sph is` |
| 10 | recipe.json | `/proper_reward_w_rps` | bound | check_recipe_block | `recipe.json proper_reward_w_rps is` |
| 11 | recipe.json | `/schedule` | bound | check_recipe_block | `recipe.json schedule is` |
| 12 | recipe.json | `/shots_per_class` | bound | check_recipe_shots (the largest train.jsonl class count) | `recipe.json shots_per_class is` |
| 13 | recipe.json | `/epochs` | bound | check_recipe_block (recipe.epoch_rule) | `recipe.json epochs is` |
| 14 | recipe.json | `/seed` | bound | check_recipe_block (seed_policy.declared_seed) | `recipe.json seed is` |
| 15 | recipe.json | `/base/family` | bound | check_base | `which=family` |
| 16 | recipe.json | `/base/checkpoint` | bound | check_base | `which=checkpoint` |
| 17 | recipe.json | `/base/repo` | bound | check_base | `which=repo` |
| 18 | recipe.json | `/base/revision` | bound | check_base | `which=revision` |
| 19 | recipe.json | `/base/sha256` | bound | PackInputs::from_run_dir (== inputs_sha256.base_model), then check_base (== base.model_safetensors_sha256) | `file=base_model` |
| 20 | recipe.json | `/early_stopping/monitor` | bound | check_recipe_block | `recipe.json early_stopping.monitor is` |
| 21 | recipe.json | `/early_stopping/mode` | bound | check_recipe_block | `recipe.json early_stopping.mode is` |
| 22 | recipe.json | `/early_stopping/eval_every_epochs` | bound | check_recipe_block | `recipe.json early_stopping.eval_every_epochs is` |
| 23 | recipe.json | `/early_stopping/first_candidate_epoch` | bound | check_recipe_block | `recipe.json early_stopping.first_candidate_epoch is` |
| 24 | recipe.json | `/early_stopping/patience_epochs` | bound | check_recipe_block | `recipe.json early_stopping.patience_epochs is` |
| 25 | recipe.json | `/early_stopping/min_delta` | bound | check_recipe_block | `recipe.json early_stopping.min_delta is` |
| 26 | recipe.json | `/early_stopping/restore` | bound | check_recipe_block | `recipe.json early_stopping.restore is` |
| 27 | recipe.json | `/early_stopping/tie_break` | bound | check_recipe_block | `recipe.json early_stopping.tie_break is` |
| 28 | recipe.json | `/seed_selection/policy` | bound | check_seed_decl | `seed_selection.policy is` |
| 29 | recipe.json | `/seed_selection/seeds/[*]` | bound | check_seed_decl | `seed_selection.seeds is` |
| 30 | recipe.json | `/seed_selection/rank_scale` | bound | check_seed_decl | `seed_selection.rank_scale is` |
| 31 | recipe.json | `/seed_selection/tie_break` | bound | check_seed_decl | `seed_selection.tie_break is` |
| 32 | gate-report.json | `/schema` | bound | PackInputs::from_run_dir (GATE_REPORT_SCHEMA) | `schema is` |
| 33 | gate-report.json | `/pass` | bound | check_gate (recomputed pass) | `reported pass=` |
| 34 | gate-report.json | `/thresholds/min_macro_f1_margin` | bound | check_thresholds | `field=min_macro_f1_margin` |
| 35 | gate-report.json | `/thresholds/max_ece` | bound | check_thresholds | `field=max_ece` |
| 36 | gate-report.json | `/thresholds/ece_bins` | bound | check_thresholds | `field=ece_bins` |
| 37 | gate-report.json | `/zero_shot/macro_f1` | bound | check_gate (recomputed) | `field=zero_shot.macro_f1` |
| 38 | gate-report.json | `/zero_shot/ece` | bound | check_gate (recomputed) | `field=zero_shot.ece` |
| 39 | gate-report.json | `/zero_shot/n` | bound | check_gate (recomputed) | `field=zero_shot.n` |
| 40 | gate-report.json | `/zero_shot/f_avg` | EXPECTED-OPEN | check_f_avg (plan 08-27) | `f_avg` |
| 41 | gate-report.json | `/fine_tuned/macro_f1` | bound | check_gate (recomputed) | `field=fine_tuned.macro_f1` |
| 42 | gate-report.json | `/fine_tuned/ece_post` | bound | check_gate (recomputed) | `field=fine_tuned.ece_post` |
| 43 | gate-report.json | `/fine_tuned/nll` | bound | check_gate (recomputed) | `field=fine_tuned.nll` |
| 44 | gate-report.json | `/fine_tuned/n` | bound | check_gate (recomputed) | `field=fine_tuned.n` |
| 45 | gate-report.json | `/fine_tuned/ece_pre` | report_only | train.py computes the pre-calibration ECE from the T = 1 probabilities, which the run dir does not hold. Rebuilding them from the calibrated float32 file as softmax(T x ln p) is not exact, and ECE jumps at bin edges, so a recomputation could refuse an honest run. It is in no gate clause and not in the artifact manifest | - |
| 46 | gate-report.json | `/fine_tuned/f_avg` | EXPECTED-OPEN | check_f_avg (plan 08-27) | `f_avg` |
| 47 | gate-report.json | `/margin` | bound | check_gate (recomputed) | `field=margin` |
| 48 | gate-report.json | `/calibration/bucket` | bound | decide-apr-v1 rung 4 (check_manifest_bindings: bucket_key of the task) | `calibration.bucket` |
| 49 | gate-report.json | `/calibration/t_fitted` | bound | check_report_record (t_applied == clamp(t_fitted), calibration_fit_bounded) | `calibration_fit_bounded` |
| 50 | gate-report.json | `/calibration/t_applied` | bound | check_report_record (clamp relation), and decide-apr-v1 rung 4 (the checkpoint agent config) | `calibration.t_applied` |
| 51 | gate-report.json | `/calibration/clamp_hit` | bound | check_report_record (t_fitted at a bound) | `calibration.clamp_hit` |
| 52 | gate-report.json | `/calibration/slice_size` | bound | check_split (check_slice_ids) | `slice_size` |
| 53 | gate-report.json | `/calibration/slice_ids/[*]` | bound (structure); membership report_only | check_split binds the structure: sorted, unique, in range, per-class need, group-disjoint, hash. Membership is report-only. data.py draws the slice with numpy `RandomState(declared_seed).permutation`, which verify does not re-derive, and verify does not re-fit T from the slice (the slice logits are not in the run dir). So which valid rows form the slice decides nothing verify recomputes. The sweep proves both halves: a duplicate id is refused, a same-class swap is accepted | `slice_ids` |
| 54 | gate-report.json | `/calibration/slice_ids_sha256` | bound | check_split (check_slice_ids) | `slice_ids hash to` |
| 55 | gate-report.json | `/seeds/declared` | bound | check_seeds_block | `field=declared` |
| 56 | gate-report.json | `/seeds/n` | bound | check_seeds_block | `field=per_seed` |
| 57 | gate-report.json | `/seeds/label` | bound | check_seeds_block (the seed_policy.rule literal) | `field=seeds.label` |
| 58 | gate-report.json | `/seeds/policy` | bound | check_seeds_block | `field=seeds.policy` |
| 59 | gate-report.json | `/seeds/shipped` | bound | check_seeds_block, check_seed_selection (the median) | `field=shipped` |
| 60 | gate-report.json | `/seeds/per_seed/[*]/seed` | bound | PackInputs::from_run_dir (reads seeds/seed-<s>/), check_seeds_block | `seeds/seed-` |
| 61 | gate-report.json | `/seeds/per_seed/[*]/macro_f1` | bound | check_seed_selection (recomputed) | `field=per_seed.macro_f1` |
| 62 | gate-report.json | `/seeds/per_seed/[*]/ece_post` | bound | check_seed_selection (recomputed) | `field=per_seed.ece_post` |
| 63 | gate-report.json | `/seeds/per_seed/[*]/margin` | bound | check_seed_selection (recomputed) | `field=per_seed.margin` |
| 64 | gate-report.json | `/seeds/per_seed/[*]/f_avg` | EXPECTED-OPEN | check_f_avg (plan 08-27) | `f_avg` |
| 65 | gate-report.json | `/seeds/per_seed/[*]/pass` | bound | check_seed_selection (recomputed) | `field=per_seed.pass` |
| 66 | gate-report.json | `/seeds/per_seed/[*]/t_applied` | bound (shipped row); report_only otherwise | check_seeds_block binds the shipped row to calibration.t_applied. The other rows are report-only: a non-shipped seed's checkpoint is deleted after selection (seed_policy.retention_rule) and its T is recorded nowhere else, so no file verify holds can bind it. It enters no verdict | `field=per_seed.t_applied` |
| 67 | gate-report.json | `/seeds/per_seed/[*]/rank_key` | bound | check_seed_selection (recomputed) | `field=per_seed.rank_key` |
| 68 | gate-report.json | `/seeds/per_seed/[*]/eval_probs_sha256` | bound | PackInputs::from_run_dir (the seed file hash) | `its per_seed row records` |
| 69 | gate-report.json | `/seeds/per_seed/[*]/model_safetensors_sha256` | bound (shipped row); report_only otherwise | check_seeds_block binds the shipped row to checkpoint/model.safetensors. The other rows are report-only: a non-shipped seed's checkpoint is deleted after selection (seed_policy.retention_rule), so its hash names bytes no run dir holds. It enters no verdict | `field=model_safetensors_sha256` |
| 70 | gate-report.json | `/device_used` | report_only | Provenance (D-03): the trainer reads it back from the parameters, and verify cannot re-observe it. Its CPU-ness is bound through device_is_cpu (device_recorded). The string enters no verdict; the artifact manifest copies it from this report (decide-apr-v1 rung 4) | - |
| 71 | gate-report.json | `/device_is_cpu` | bound | check_report_record (device_recorded) | `field=device_is_cpu` |
| 72 | gate-report.json | `/torch_version` | report_only | Provenance (D-03): the trainer's torch version, which verify cannot re-observe. It enters no verdict | - |
| 73 | gate-report.json | `/recipe_id` | bound | PackInputs::from_run_dir (sha256 of recipe.json) | `file=recipe_json` |
| 74 | gate-report.json | `/inputs_sha256/task_json` | bound | check_inputs (the data dir) | `file=task_json` |
| 75 | gate-report.json | `/inputs_sha256/train_jsonl` | bound | check_inputs (the data dir) | `file=train_jsonl` |
| 76 | gate-report.json | `/inputs_sha256/eval_jsonl` | bound | check_inputs (the data dir) | `file=eval_jsonl` |
| 77 | gate-report.json | `/inputs_sha256/base_model` | bound | PackInputs::from_run_dir (== recipe base.sha256, itself == the contract pin and the base dir) | `file=base_model` |
| 78 | gate-report.json | `/inputs_sha256/tokenizer_json` | bound | PackInputs::from_run_dir (the checkpoint tokenizer), and the base tokenizer at the zero-shot load | `file=tokenizer_json` |
| 79 | gate-report.json | `/inputs_sha256/shift_jsonl` | bound | check_shift_probe (the data dir shift.jsonl) | `field=shift_jsonl` |
| 80 | gate-report.json | `/eval_probs_sha256` | bound | check_inputs | `file=eval_probs_json` |
| 81 | gate-report.json | `/zero_shot_probs_sha256` | bound | check_inputs | `file=zero_shot_probs_json` |
| 82 | gate-report.json | `/probes_sha256` | bound | PackInputs::from_run_dir | `file=probes_json` |
| 83 | gate-report.json | `/rescore_noise_sha256` | bound | check_inputs | `file=rescore_noise_json` |
| 84 | gate-report.json | `/shift_probe/gate_clause` | bound | check_shift_probe | `field=shift_probe.gate_clause` |
| 85 | gate-report.json | `/shift_probe/n` | bound | check_shift_probe | `field=shift_probe.n` |
| 86 | gate-report.json | `/shift_probe/zero_shot/macro_f1` | bound | check_shift_probe (recomputed) | `field=shift_probe.zero_shot.macro_f1` |
| 87 | gate-report.json | `/shift_probe/zero_shot/ece` | bound | check_shift_probe (recomputed) | `field=shift_probe.zero_shot.ece` |
| 88 | gate-report.json | `/shift_probe/zero_shot/f_avg` | EXPECTED-OPEN | check_f_avg (plan 08-27) | `f_avg` |
| 89 | gate-report.json | `/shift_probe/fine_tuned/macro_f1` | bound | check_shift_probe (recomputed) | `field=shift_probe.fine_tuned.macro_f1` |
| 90 | gate-report.json | `/shift_probe/fine_tuned/ece_post` | bound | check_shift_probe (recomputed) | `field=shift_probe.fine_tuned.ece_post` |
| 91 | gate-report.json | `/shift_probe/fine_tuned/f_avg` | EXPECTED-OPEN | check_f_avg (plan 08-27) | `f_avg` |
| 92 | gate-report.json | `/shift_probe/margin` | bound | check_shift_probe (recomputed) | `field=shift_probe.margin` |
| 93 | gate-report.json | `/shift_probe/probs_sha256` | bound | PackInputs::from_run_dir | `file=shift_probs_json` |
| 94 | gate-report.json | `/shift_probe/zero_shot_probs_sha256` | bound | PackInputs::from_run_dir | `file=shift_zero_shot_probs_json` |

How the sweep works:
- **Fixture.** The fullest run copy is the three-seed median production copy with early_stopping, plus a float64 noise record, a shift probe, and the deployed run's device record (`mps:0`).
- **Mutations.** A 64-hex string has its first digit flipped, an integer gets +1, a float becomes x2+1, a bool is flipped, a string gets `x` appended, and a null becomes 0.5.
- **Reseal.** recipe.json's sha is resealed as the report's `recipe_id`. When a slice id moves, `slice_ids_sha256` is resealed too.
- **Pass condition.** A bound row must be refused, with a Display containing `names`. A report_only row must still verify, with recomputed metrics and shipped seed equal to the control.

## Mutation proof

Each check added in this plan was disabled alone. The script (`mutate.py`) then ran `cargo test -p aprender-decide --lib -- <sweep> <named negative>` and restored the file. Every restore was confirmed byte-identical with `filecmp`, and no mutation leftovers remain in verify.rs.

| # | Check disabled | rc | RED tests | Evidence |
|---|---|---|---|---|
| 0 | base.family (check_base) | 101 | every_run_field_is_bound_or_report_only | `recipe.json/base/family (bound): expected a refusal naming "which=family", got ACCEPTED` |
| 1 | base.checkpoint | 101 | sweep | `/base/checkpoint ... got ACCEPTED` |
| 2 | base.repo | 101 | sweep | `/base/repo ... got ACCEPTED` |
| 3 | base.revision | 101 | base_identity_mismatch_revision, sweep | `/base/revision ... got ACCEPTED` |
| 4 | base pin encoder_config_sha256 | 101 | base_encoder_config_unpinned_refused | verified instead of refusing |
| 5 | base pin rl_agent_config_sha256 | 101 | base_agent_config_unpinned_refused | verified instead of refusing |
| 6 | base pin tokenizer_json_sha256 | 101 (after the fix below; first run rc 0) | base_mismatch_tokenizer | the new "run and base agree on an unpinned tokenizer" case verified |
| 7 | recipe float encoder_lr | 101 | recipe_block_differs_from_contract_refused, sweep | `/encoder_lr ... got ACCEPTED` |
| 8 | recipe literal schedule | 101 | recipe_block…, sweep | `/schedule ... got ACCEPTED` |
| 9 | recipe seed == declared_seed | 101 | recipe_block…, sweep | the leaf is still refused, but later: `SeedPolicyViolated field=declared` (check_seeds_block), not the row's name |
| 10 | recipe epoch rule | 101 | recipe_epochs_break_the_rule_refused, sweep | `/epochs ... got ACCEPTED` |
| 11 | recipe early_stopping.min_delta | 101 | recipe_block…, sweep | `/early_stopping/min_delta ... got ACCEPTED` |
| 12 | shots_per_class == train (check_recipe_shots) | 101 | recipe_block…, sweep | `/shots_per_class ... got ACCEPTED` |
| 13 | slice fraction (slice_need) | 101 | slice_below_fraction_refused | not caught by the sweep: at 4 rows per class the fixture's minimum (2) dominates the fraction (ceil(0.25 x 4) = 1) |
| 14 | device_is_cpu relation | 101 | sweep | `/device_is_cpu ... got ACCEPTED` |
| 15 | t_applied == clamp(t_fitted) | 101 | sweep | `/calibration/t_fitted ... got ACCEPTED` (`/calibration/t_applied` stays refused by rung 4, naming the same field) |
| 16 | clamp_hit at a bound | 101 | sweep | `/calibration/clamp_hit ... got ACCEPTED` |
| 17 | fine_tuned.nll recomputed | 101 | sweep | `/fine_tuned/nll ... got ACCEPTED` |
| 18 | seeds.label literal | 101 | sweep | `/seeds/label ... got ACCEPTED` |
| 19 | shipped per_seed.t_applied | 101 | sweep | `/seeds/per_seed/2/t_applied (the shipped seed's row) ... got ACCEPTED` |
| 20 | write_atomic create_new + unique name (reverted to `File::create` on `.v.apr.tmp-<pid>`) | 101 | write_atomic_does_not_follow_a_planted_symlink | the victim file was overwritten |
| 21 | ProbsRowCoverage Display names the file | 101 | probs_row_coverage_names_the_file | the Display lacks `fine_tuned: ` |

Task 1 RED was measured separately, before its commit: without the revision comparison, `base_identity_mismatch_revision` got `VerifyReport { deploy_eligible: true, .. }`.

## Real evidence (every leg through `heavy`)

- **Deployed artifact.** Command: `heavy just laya-verify $MAIN/models/decide/laya-stance-64.apr $MAIN/models/decide/laya-stance-64 $MAIN/data/decide/tweet-stance-64 <snapshot 55cf4c4e>`. Result: `vrc=0`. The build log shows `Compiling aprender-decide` from the Task-3 tree. Output:
  ```
  {"argmax":"459/459","artifact_sha256":"24a44d7e050166c9b64e2716f2bcb3ce91747f7a3b927d03d6eeae5f89b6275a","deploy_eligible":true,"noise":8.33931784072206e-6,"recomputed":{"ece_post":0.04425034672021866,"ft_macro_f1":0.6308225393295288,"margin":0.22166302800178528,"zs_macro_f1":0.40915951132774353},"rescore_bound":0.00003335727136288824,"rescore_max_abs":0.000011324882507324219,"shipped_seed":17,"zs_noise":0.00001495515496430233,"zs_rescore_bound":0.00005982061985720932,"zs_rescore_max_abs":7.3462724685668945e-6}
  ```
  The same leg after Task 1 (typed mapping and identity fields) printed the identical line.
- **Fail-closed vectors.** Command: `heavy env CARGO_INCREMENTAL=0 LAYA_FAIL_CLOSED_VECTORS=1 LAYA_MODEL_DIR=<snapshot> cargo test -p aprender-decide --release --test fail_closed_vectors -- --nocapture`. Result: `crc=0` and `FAIL-CLOSED VECTORS REFUSED 2/2 (388 s, ARCH aarch64)`.
  - d0f4e40d (pack and verify): `REFUSED RescoreDrift which=fine_tuned row=59 max_abs=0.00004667043685913086 bound=0.00001` (exit 2, unchanged).
  - 3d4b91da (pack and verify): `REFUSED GateFailed clauses=[ece_post] ... margin=0.10561397671699524 ece_post=0.22240783274173737 ... argmax=280/280` (exit 3, unchanged).
  - No new check refused either vector earlier, so the STOP condition did not trigger. Both recipes equal the contract block, both slices are 4 per class (need 4), and their t_fitted/clamp_hit/device/nll records satisfy the new relations.

## Other verification

- `cargo test -p aprender-decide --tests`: lib 146 passed; demo_run, fail_closed_vectors, laya_parity and python_records SKIP unarmed as before; ui passed.
- `cargo clippy -p aprender-decide --all-targets --no-deps -- -D warnings`: rc 0. `cargo fmt -p aprender-decide -- --check`: rc 0.
- `cargo check -p aprender-mcp-decide -p aprender-mcp-decide-lambda`: rc 0.
- `pv validate`: `0 error(s), 0 warning(s)` on laya-finetune-gate-v1 and on decide-apr-v1. `pv diff` against HEAD~ for both: `Contracts are identical.`
- `make contract-audit-phase8`: rc 0.
- The Python torch-free self-tests (`metrics.py`, `data.py`, `gate.py --selftest`) pass against the extended contract.

## Files Created/Modified

- `crates/aprender-decide/src/verify.rs`:
  - the views and `from_contract_views`;
  - `BasePins` and `RecipePins`;
  - `check_base` over every identity field;
  - `check_recipe_block` and `check_recipe_shots`;
  - `base_file_matches` and the rewritten `load_declared_base`;
  - `slice_need` and the fraction rule;
  - `check_report_record` and `recompute_nll`;
  - the label and shipped-t_applied checks;
  - `write_atomic` hardening;
  - the `RecipeMismatch` and `RecordMismatch` variants;
  - `verify_run` made `cfg(test) pub(crate)`;
  - module docs.
- `crates/aprender-decide/src/verify/tests.rs`:
  - the policy built from the views;
  - `tiny_base_pins`;
  - trainer-consistent production copies and `Run::append_train`;
  - 12 new tests (2 in Task 1, 8 in Task 2, the 2 sweep tests);
  - `base_mismatch_tokenizer` extended.
- `crates/aprender-decide/examples/pack_laya.rs`: `policy()` goes through the views; the hand mapping is deleted.
- `crates/aprender-decide/tests/common/mod.rs`: `contract_view` and `policy()` go through the views.
- `crates/aprender-decide/tests/fail_closed_vectors.rs`: uses `mod common;` and the library's `artifact_sha256_hex`; the duplicated helpers are gone.
- `contracts/laya-finetune-gate-v1.yaml`: the `base` pins, `recipe.schedule_literal` and `run_field_bindings`.
- `contracts/decide-apr-v1.yaml`: `verify_checks[2]` wording.

## Decisions Made

See key-decisions in the frontmatter.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical] Leaves the sweep found unbound are now bound**
- **Found during:** Task 3. A discovery sweep ran before the table was written.
- **Issue:** Seven leaves verified after mutation:
  - `fine_tuned.nll`;
  - `calibration.t_fitted` and `calibration.clamp_hit`;
  - `device_is_cpu` and `device_used`;
  - `seeds.label`;
  - every `per_seed.t_applied` and non-shipped `per_seed.model_safetensors_sha256`.
- **Fix:** Each leaf the contract gives a machine rule for is now bound:
  - nll is recomputed with aprender-core `log_loss`, and is exactly recomputable (Python diff about 1e-16 on the deployed run and both vectors);
  - the `calibration_fit_bounded` clamp relation;
  - `device_recorded`;
  - the `seed_policy.rule` label literal;
  - the shipped row's t_applied.

  Leaves no file can bind are report_only with a reason. New: `VerifyError::RecordMismatch`, `VerifyPolicy.calibration_temp_min/max` (read through the views), and `Recomputed.ft_nll`.
- **Verification:** mutations 14 to 19 are RED. Both real legs are unchanged.
- **Committed in:** 78a3683ec

**2. [Rule 2 - Missing critical] shots_per_class bound to the data dir**
- **Found during:** Task 2.
- **Issue:** The epoch rule keys on `shots_per_class`. Unbound, a recipe could declare 17 shots to reach the `[4, 12]` branch.
- **Fix:** `check_recipe_shots` requires the largest train.jsonl class count, which is what `train.py` writes. Three tests that append train rows (`split_overlap`, `conflicting_labels`, `slice_splits_group`) now re-derive shots through `Run::append_train`, so only their own rule refuses.
- **Committed in:** 2b7149fa0

**3. [Rule 3 - Blocking] verify_run is `cfg(test) pub(crate)`**
- **Found during:** Task 2.
- **Issue:** Once it was `pub(crate)`, no production code called it, and `clippy -D warnings` refused it as dead code.
- **Fix:** Gated to tests. The pipeline docs no longer link it.
- **Committed in:** 2b7149fa0

**4. [Rule 1 - Honesty] decide-apr-v1 verify_checks[2] rewritten**
- **Found during:** Task 2.
- **Issue:** The row said the recipe's other base fields and the base-dir files were "plan 08-21, NOT checked yet". After this plan that is false. The file was not in the plan's list.
- **Fix:** The row now states what check_base and the zero-shot load bind. `pv diff` reports identical claims.
- **Committed in:** 2b7149fa0

**5. [Rule 1 - Test gap] base_mismatch_tokenizer could not see the tokenizer pin**
- **Found during:** Task 3 mutation proof (mutation 6, first run rc 0).
- **Issue:** On the tiny fixture the pin equals the run's tokenizer hash, so the run binding alone refused every case the test built.
- **Fix:** Added the case where the run and its base dir agree on a tokenizer the contract does not pin. Only the pin refuses it. Re-run: rc 101.
- **Committed in:** 78a3683ec

**6. [Structural] Table schema extensions**
- **Found during:** Task 3.
- **Issue:** Three leaf patterns are bindable in one row or part but not in others: `per_seed.t_applied`, `per_seed.model_safetensors_sha256` and `slice_ids`.
- **Fix:** Added the row keys `only: shipped_seed_row` with `report_only_otherwise`, and `membership_report_only`. The sweep proves both halves of each. The fullest copy's device is `mps:0` (the deployed run's), so the report_only `device_used` row is exercised off the CPU relation.
- **Committed in:** 78a3683ec

**7. [Process] TDD RED for Task 2 measured by reversion**
- **Found during:** Task 2.
- **Issue:** The Task-2 tests were written after their checks. Against the Task-1 code they do not compile, because the variants did not exist, so a compile failure would have been the only RED.
- **Fix:** RED was measured per check with the Task-3 mutation loop: each check disabled alone turns its named test RED (table above).

---

**Total deviations:** 7 (2 missing-critical bindings, 1 blocking lint, 1 honesty wording, 1 test gap found by mutation, 1 table schema, 1 process).
**Impact on plan:** Every must-have truth holds. The additions widen the bound set beyond the plan's list, which the class-A invariant requires. Nothing the phase shipped changed its verdict.

## Issues Encountered

- **Base config pins refuse late.** The base-dir config pins are checked only at the zero-shot load, which runs after the fine-tuned re-score (minutes on the real model). A base dir with a wrong config therefore refuses late. That is correct, but slow; see the key-decision on one check site per pin.
- **The sweep cannot see the slice fraction.** The tiny fixture's classes are too small for the fraction to exceed the per-class minimum, so the sweep cannot detect the slice-fraction rule being disabled. Only `slice_below_fraction_refused` (a direct `check_split` case at 16 rows per class) covers it.

## Known Stubs

None. The five f_avg rows are EXPECTED-OPEN by design and owned by plan 08-27. They are listed, never counted as bound.

## User Setup Required

None. No external service configuration is required.

## Next Phase Readiness

- The verify side of class A is closed for the run dir. Plan 08-27 turns the five `check_f_avg (plan 08-27)` rows from EXPECTED-OPEN to bound, and the sweep will then assert them.
- Any new recipe.json or gate-report.json field needs a `run_field_bindings` row first, or `run_field_bindings_table_matches_fixture_leaves` fails.

## Self-Check: PASSED

- FOUND: all 7 modified files.
- FOUND: commits 25d9ead18, 2b7149fa0 and 78a3683ec.
- `fn from_contract_views`, `fn check_recipe_block` and `fn every_run_field_is_bound_or_report_only` are present. `run_field_bindings:` appears once in the contract.
- `commits: 3` was measured as `git rev-list --count 23daf2f2a..HEAD` before this SUMMARY's commit.

---
*Phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server*
*Completed: 2026-09-28*
