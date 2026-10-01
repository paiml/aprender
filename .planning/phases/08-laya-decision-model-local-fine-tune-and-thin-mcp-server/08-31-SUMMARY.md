---
phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
plan: 31
subsystem: claim-honesty
tags: [class-D, claims-ledger, laya-claims-check, D-14, publish-false, release-cascade, argmax, digest, safetensors, deferred-items, mutation-proof]
status: complete
gap_closure: true

requires:
  - phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
    provides: "08-19..08-30: the corrected contract clauses, sweeps and tests the ledger names; 08-22 laya_gates.tsv and laya-gates-selftest; 08-28 / 08-30 owner decisions"
provides:
  - "crate::digest::sha256_hex, the crate's one sha256 helper; the load path no longer calls into pack; safetensors_is_read_only_by_pack (Task 1, c4a6db7c5)"
  - "scripts/laya_claims.tsv: 133 rows (rust 104, python 13, evidence 7, just 6, pv 3) covering every FALSIFY id this round added or changed, every row of both untrusted_input_bounds tables, and every doc / contract / CONTEXT claim the round corrected"
  - "just laya-claims-check: rust and just rows take a comma list matched exactly; python kind exercised for real; 3 built-in must-fail self-cases; mutation-proven (5 mutants RED)"
  - "D-14 owner decision publish-false: aprender-decide is publish = false (dated comment), publish_is_false_until_the_crate_name_is_confirmed, gate row cascade-guard now runs a case"
  - "argmax folds from the first non-NaN element (IN-01) with argmax_nan_never_wins; tests/ui.rs header says it runs in CI (IN-04)"
  - "deferred-items.md: gap-round final status of every 08-REVIEW / 08-CODE-REVIEW-FINDINGS finding, statuses and owners on the planning-time deferrals, the non-Phase-8 guard offenders"
affects: [08-32, phase-8-verification, release-cascade, aprender-decide]

actuals:
  tokens: 7949       # chars/4 over the realized diff (git diff 17dae0661..23a640e6a | wc -c = 31795)
  tasks: 3
  commits: 2         # MEASURED: git rev-list --count 17dae0661..HEAD before this SUMMARY's commit
plan_head_before: 17dae066151a33aa8896a93756bce381c20f9c37

tech-stack:
  added: []
  patterns:
    - "Ledger rows derived from the contracts' own cargo commands, one row per (contract row, owner target), each named test matched exactly against `cargo test -p <owner> [--lib | --test T] -- --list` (never cargo's substring filter, which runs zero tests on a stale name and passes)"
    - "An owner decision that only a red-at-base guard would protect gets its own test (publish_is_false_until_the_crate_name_is_confirmed) and a gate case"

key-files:
  created:
    - crates/aprender-decide/src/digest.rs
    - scripts/laya_claims.tsv
  modified:
    - crates/aprender-decide/Cargo.toml
    - crates/aprender-decide/src/lib.rs
    - crates/aprender-decide/src/laya/mod.rs
    - crates/aprender-decide/src/laya/tests.rs
    - crates/aprender-decide/tests/ui.rs
    - crates/aprender-decide/src/pack.rs
    - crates/aprender-decide/src/artifact.rs
    - crates/aprender-decide/src/task.rs
    - crates/aprender-decide/src/verify.rs
    - justfile
    - scripts/laya_gates.tsv
    - .planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-CONTEXT.md
    - .planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/deferred-items.md

key-decisions:
  - "D-14: owner chose publish-false (2026-09-29). aprender-decide is publish = false until the crate name is confirmed; it is not added to TIERS; nothing was published"
  - "The ledger covers all 39 untrusted_input_bounds rows (both tables), not only the rows owned by another crate: the CI strict-binding guard is vacuous here (D-ITEM-08-01-A), so the ledger is where a dangling test name is seen"
  - "argmax deliberately differs from numpy on NaN input (numpy returns the first NaN); the doc no longer cites numpy. Every caller refuses non-finite probabilities first, so no served or verified decision changes"
  - "The D-09 and D-14 CONTEXT amendments are ledgered as rust rows naming their enforcing test, not as evidence rows: no JSON record exists for them. The D-18 amendments are evidence rows on 08-LIVE-REDEPLOY-EVIDENCE.json"

patterns-established:
  - "A claim a plan corrects gets a scripts/laya_claims.tsv row in the same commit; `just laya-claims-check` fails on a lost anchor, a renamed test, an unknown kind, a missing python case or a missing evidence key"

requirements-completed: [D-14, D-16, D-17]

coverage:
  - id: D1
    description: "The load path never calls into pack; production code reads SafeTensors only in src/pack.rs (Task 1)"
    requirement: D-17
    verification:
      - kind: unit
        ref: "crates/aprender-decide/src/lib.rs#tests::safetensors_is_read_only_by_pack"
        status: pass
      - kind: unit
        ref: "crates/aprender-decide/src/artifact/determinism.rs#golden_sha"
        status: pass
    human_judgment: false
  - id: D2
    description: "D-14 publish-false implemented: publish = false, the cascade guard no longer lists aprender-decide, the decision is pinned by a test and a gate case"
    requirement: D-14
    verification:
      - kind: unit
        ref: "crates/aprender-decide/src/lib.rs#tests::publish_is_false_until_the_crate_name_is_confirmed"
        status: pass
      - kind: other
        ref: "bash scripts/check_cascade_covers_all_crates.sh (R1 lists only aprender-contrastive-data); LAYA_GATES_ONLY=cascade-guard just laya-gates-selftest (GREEN; RED with the key removed)"
        status: pass
    human_judgment: false
  - id: D3
    description: "argmax's 'NaN never wins' is true and pinned; tests/ui.rs says it runs in CI"
    requirement: D-17
    verification:
      - kind: unit
        ref: "crates/aprender-decide/src/laya/tests.rs#argmax_nan_never_wins"
        status: pass
      - kind: integration
        ref: "cargo test -p aprender-decide --test ui (decider_has_no_second_minting_path)"
        status: pass
    human_judgment: false
  - id: D4
    description: "Class D is a gate: 133 ledger rows, each anchor present and each named test / recipe / case / key existing; five mutants each RED"
    requirement: D-16
    verification:
      - kind: other
        ref: "just laya-claims-check -> LAYA CLAIMS OK 133 rows; LAYA_GATES_ONLY=claims-check just laya-gates-selftest -> GREEN"
        status: pass
    human_judgment: false
  - id: D5
    description: "deferred-items.md gives every 08-REVIEW / 08-CODE-REVIEW-FINDINGS finding a final status, reason and re-open trigger, and names the non-Phase-8 guard offenders"
    verification: []
    human_judgment: true
    rationale: "Completeness of a finding-by-finding accounting is an editorial judgment; the plan's grep only proves the section exists"

duration: 2h14m wall (2026-09-29T02:56:20Z to 05:10Z, including the Task 2 owner checkpoint)
completed: 2026-09-29
---

# Phase 8 Plan 31: Class D Claim Honesty, D-14 Publication and the Round's Deferrals Summary

**Every claim the 08-19..08-30 round corrected or enforced is now a row of `scripts/laya_claims.tsv`, 133 rows in all, and `just laya-claims-check` fails when a row's anchor text disappears or a test it names stops existing. The FALSIFY ids and both `untrusted_input_bounds` tables are ledgered straight from the contracts' own `cargo test` commands. aprender-decide is `publish = false` by the owner's D-14 decision `publish-false`, and a test and a gate case keep it that way. `argmax` really never lets a NaN win. deferred-items.md now gives every review finding a final status.**

## Performance

- **Duration:** 2h14m wall, 2026-09-29T02:56:20Z to about 05:10Z. That includes the Task 2 owner checkpoint and about 1h10m of claims-check runs. Each run takes about 2:40, because aprender-core/-decide rebuild on every cargo invocation (the 08-22 deferred item).
- **Tasks:** 3 of 3 (Task 1 in the prior session, c4a6db7c5; Task 2 owner decision; Task 3 here).
- **Files modified:** 10 in Task 3 (13 across the plan, plus digest.rs and the ledger created in Task 1).

## Owner decision (Task 2, verbatim)

**`publish-false`**, recorded 2026-09-29:
- as the dated D-14 amendment in 08-CONTEXT.md: "— **Amended 2026-09-29 (user, plan 08-31 Task 2 decision `publish-false`):** …";
- in the `crates/aprender-decide/Cargo.toml` comment;
- here.

Evidence read before the decision (read-only):
- `check_cascade_covers_all_crates.sh` exited 1 listing aprender-contrastive-data and aprender-decide.
- aprender-decide had no `publish` key.
- Its dependents aprender-mcp-decide and aprender-mcp-decide-lambda are `publish = false`.

Nothing was published.

## Accomplishments

- **Task 1 (tracer, c4a6db7c5):**
  - `crate::digest::sha256_hex` is the one helper. `pack::sha256_hex` is gone, and the load path imports no hash function from pack.
  - `safetensors_is_read_only_by_pack` makes the claim true as worded: production code reads SafeTensors only in `src/pack.rs`. `src/test_support.rs` is the one allowlisted `#[cfg(test)]` reader.
  - The ledger and gate were created with their first 3 rows.
- **D-14, `publish-false`:**
  - `publish = false` in `crates/aprender-decide/Cargo.toml`, with a comment citing D-14, the decision id, the date and how to reverse it.
  - `check_cascade_covers_all_crates.sh` now reports 74 publishable crates, and R1 lists only aprender-contrastive-data (exit 1, not Phase 8's).
  - `check_publishable_deps_publishable.sh` still passes (93 members, exit 0).
  - Two guards keep the decision, and both go RED when the line is deleted:
    - `tests::publish_is_false_until_the_crate_name_is_confirmed` (lib, so CI's workspace-test runs it);
    - the `cascade-guard` row of `scripts/laya_gates.tsv`. It was a `-` placeholder and now runs a case: the guard ran, and it does not list aprender-decide.
- **IN-01:** `argmax` starts its fold at the first element equal to itself, so a NaN never holds the running maximum. All-NaN and empty inputs still return 0, and ties keep the first index. `argmax_nan_never_wins` covers `[NaN, 0.9]` → 1, `[0.2, NaN, 0.9]` → 2, ties, and the all-NaN / empty / ordinary cases. Under the old fold it is RED ("a NaN at index 0 must not win").
- **IN-04:** the `tests/ui.rs` header says the target runs in CI on ci.yml's integration line (plan 08-12). Two ledger rows pin the header and the ci.yml line.
- **The ledger (133 rows).** `laya-claims-check` now accepts a comma-separated list for `rust` and `just` rows, and matches each name exactly. The rows come in five groups:
  - Rows generated from the contracts' own commands, one row per (contract row, owner target):
    - FALSIFY-DECIDE-APR-013;
    - FALSIFY-DECIDE-TOOL-001/005/006/008;
    - FALSIFY-LAYA-GATE-003/015;
    - all 30 decide-apr-v1 and all 9 decide-tool-boundary-v1 `untrusted_input_bounds` rows. This includes every row owned by apr-format, aprender-core or aprender-mcp-decide-lambda, so those crates' tests are proven to be listed.
  - FALSIFY ids with no `cargo test` command, ledgered through another kind:
    - FALSIFY-LAYA-GATE-009 (python);
    - FALSIFY-DECIDE-TOOL-009 (evidence plus just).
  - Contract clauses:
    - decide-apr-v1: verify_checks[2], load_ladder rungs 2/4/7, `manifest.bindings`, `max_criteria`;
    - decide-tool-boundary-v1: `classify_count_bound`, `classify_admission` transport_dispatch, the refusal_names_bound formula and wire code, `refusal_echo_min_chars`, `served_fields`, the tier constants 8 / 800 / 10240, the 648-vs-800 statement, the `accepted_region_cold` live pass;
    - laya-finetune-gate-v1: base pins, `schedule_literal`, `run_field_bindings`, `numeric_agreement` (Rust, Python, aprender-core f64), `early_stopping_train_side`, `tie_break_rule`, `python_refusals`, rescore-noise k.
  - Docs:
    - the served truncation sentence;
    - the aprender-mcp-decide README (truncation, admission, isError);
    - the Lambda README and OnceCell docs;
    - `DOWNLOAD_DEADLINE`;
    - the deploy.toml.template tell and memory;
    - `LABELS_SEGMENT`;
    - the three COVERAGE.md IAM rows;
    - the CLAUDE.md decide row (10,240 MB; 8 texts / 800 tokens).
  - The rest:
    - the CONTEXT amendments D-09, D-14 and D-18 (x3);
    - the pack.rs / Cargo.toml / CLAUDE.md SafeTensors claims (Task 1);
    - argmax, ui.rs, digest;
    - the verifier's write_atomic / ProbsRowCoverage / slice fraction;
    - the laya_parity NaN-visible max and MEASURED lines;
    - the Python back office: the gate.py margin comment, jsonl_lines, row-encoding, early-stopping refusal, write_once.
- **deferred-items.md:**
  - A final-status section covers every finding in 08-REVIEW.md (18) and 08-CODE-REVIEW-FINDINGS.md: closed, decided, refuted or open.
  - Each planning-time deferral gets a `status:` line and an owner.
  - Owner decisions are recorded with their dates.
  - New open entries have a reason, owner and re-open trigger. They are tokenizer_pipeline, the aprender-compute clippy warnings, aprender-core's clippy baseline, the four hand-rolled parser offenders and aprender-contrastive-data.
  - The 08-24 frame_http / probe_id_header entry is marked resolved by 08-28 (890b769e9). It was still `status: open` although the contract had been repointed.

## The python kind

13 rows use it. The first full check ran `just laya-train-selftest` inside the gate, and it printed `LAYA TRAIN SELFTEST OK` in 19 s. Every python row's case was found in that output. Before this plan the kind had never run. Mutation M4 (below) shows it bites.

## Gate mutation record (Task 3 step 4)

Each mutant was applied alone, `just laya-claims-check` was run, and the file was restored and `cmp`-checked. The self-cases passed in every run, so each failure below is the ledger's own.

| # | Mutant | Result |
|---|---|---|
| M1 | `fn argmax_nan_never_wins` renamed in `laya/tests.rs` | exit 1: `TEST MISSING decide.argmax.nan-never-wins: laya::tests::argmax_nan_never_wins not in the test list of aprender-decide:lib (165 tests)`. Restored byte-identical |
| M2 | `classify_max_total_tokens: 800` → `801` in decide-tool-boundary-v1 | exit 1: `ANCHOR MISSING tool.tier.max_total_tokens: contracts/decide-tool-boundary-v1.yaml no longer contains 'classify_max_total_tokens: 800'`. Restored byte-identical |
| M3 | a row with kind `prose` appended to a copy of the ledger | exit 1: `UNKNOWN KIND mutant-unknown-kind: 'prose' is not one of rust, just, python, pv, evidence` |
| M4 (extra) | a python row whose case the selftest never prints | exit 1: `TEST MISSING mutant-python: 'no such selftest case planted by M4' is not in the laya-train-selftest output` |
| M5 (extra) | an evidence row naming `live.no_such_key` | exit 1 (same run as M4): `TEST MISSING mutant-evidence: .../08-LIVE-REDEPLOY-EVIDENCE.json has no key live.no_such_key` |

Other mutants run in this plan:

| Mutant | Guard | Result |
|---|---|---|
| argmax back to `(0..len).fold(0, ..)` | `argmax_nan_never_wins` | RED: "a NaN at index 0 must not win". Restored |
| `publish = false` deleted | `publish_is_false_until_the_crate_name_is_confirmed` | RED (rc 101). The cascade guard lists aprender-decide again |
| `publish = false` deleted | gate row `cascade-guard` (`LAYA_GATES_ONLY=cascade-guard`) | `LAYA GATES SELFTEST FAILED: 1 row(s) RED: cascade-guard`. Restored |

## The full claims ledger (scripts/laya_claims.tsv)

`<phase>` = `.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server`. The anchors are omitted here for width; they are in the TSV. Rust test lists are joined with `, ` here and with `,` in the file.

| # | claim_id | file | kind | owner | test(s) | finding |
|---|---|---|---|---|---|---|
| 1 | pack-safetensors-scope | crates/aprender-decide/src/pack.rs | rust | aprender-decide | safetensors_is_read_only_by_pack | V14-d,CV5 |
| 2 | cargo-safetensors-scope | crates/aprender-decide/Cargo.toml | rust | aprender-decide | safetensors_is_read_only_by_pack | V14-d,CV5 |
| 3 | claude-md-safetensors-never-served | CLAUDE.md | rust | aprender-decide | safetensors_is_read_only_by_pack | V14-d,CV5 |
| 4 | FALSIFY-DECIDE-APR-013 | contracts/decide-apr-v1.yaml | rust | aprender-decide:lib | artifact::ladder::every_manifest_leaf_is_bound, artifact::ladder::manifest_bindings_table_matches_manifest_leaves, artifact::ladder::manifest_bound_at_every_load_door | CR-01 (08-19) |
| 5 | FALSIFY-DECIDE-TOOL-001 | contracts/decide-tool-boundary-v1.yaml | rust | aprender-mcp-decide:lib | tests::empty_texts_refused_naming_min_texts, tests::one_over_max_texts_refused_naming_max_texts, tests::max_texts_accepted_and_classified, tests::count_is_checked_before_element_shapes | V5-c (08-23) |
| 6 | FALSIFY-DECIDE-TOOL-005 | contracts/decide-tool-boundary-v1.yaml | rust | aprender-mcp-decide:lib | tests::truncation_flag_flips_one_past_the_room, tests::long_text_truncated_short_text_not, tests::description_truncation_sentence_matches_tier | WR-03 (08-28 A-derive) |
| 7 | FALSIFY-DECIDE-TOOL-006@aprender-mcp-decide:lib | contracts/decide-tool-boundary-v1.yaml | rust | aprender-mcp-decide:lib | tests::every_refusal_names_key_without_text, tests::oversized_text_refused_without_echo, tests::error_taxonomy_budget_rejected_model_internal | V5-a,A4-7 (08-28 B-iserror) |
| 8 | FALSIFY-DECIDE-TOOL-006@aprender-mcp-decide:test=e2e_stdio | contracts/decide-tool-boundary-v1.yaml | rust | aprender-mcp-decide:test=e2e_stdio | bound_refusals_are_iserror_results_over_live_stdio | V5-a,A4-7 (08-28 B-iserror) |
| 9 | FALSIFY-DECIDE-TOOL-006@aprender-mcp-decide-lambda:lib | contracts/decide-tool-boundary-v1.yaml | rust | aprender-mcp-decide-lambda:lib | tests::loopback_bound_refusal_is_an_iserror_result | V5-a,A4-7 (08-28 B-iserror) |
| 10 | FALSIFY-DECIDE-TOOL-008@aprender-mcp-decide:lib | contracts/decide-tool-boundary-v1.yaml | rust | aprender-mcp-decide:lib | tests::admission_refuses_over_pending, tests::admission_slot_held_until_blocking_ends, tests::admission_waiting_cancel_frees_pending, tests::admission_limits_match_contract | V5-b (08-28) |
| 11 | FALSIFY-DECIDE-TOOL-008@aprender-mcp-decide:test=e2e_stdio | contracts/decide-tool-boundary-v1.yaml | rust | aprender-mcp-decide:test=e2e_stdio | pipelined_calls_are_serialized_not_refused | V5-b (08-28) |
| 12 | FALSIFY-LAYA-GATE-003 | contracts/laya-finetune-gate-v1.yaml | rust | aprender-decide:lib | verify::tests::recompute_matches_house_ece_cases | V7-b (08-27) |
| 13 | FALSIFY-LAYA-GATE-015 | contracts/laya-finetune-gate-v1.yaml | rust | aprender-decide:lib | verify::tests::gate_numeric_cases_agree_bit_for_bit, verify::tests::f_avg_forged_refused, verify::tests::f_avg_null_rule_enforced | V7-a,V7-b,V6-e (08-27) |
| 14 | apr-bound.file_length | contracts/decide-apr-v1.yaml | rust | aprender-decide:lib | artifact::ladder::declared_length_over_cap, artifact::ladder::read_over_cap, artifact::ladder::in_memory_over_cap | class B artifact half (08-20, 08-26); owner_crate aprender-decide, enforced |
| 15 | apr-bound.header_version | contracts/decide-apr-v1.yaml | rust | aprender-decide:lib | artifact::ladder::header_version_refused | class B artifact half (08-20, 08-26); owner_crate aprender-decide, enforced |
| 16 | apr-bound.metadata_size | contracts/decide-apr-v1.yaml | rust | aprender-decide:lib | artifact::ladder::metadata_over_cap | class B artifact half (08-20, 08-26); owner_crate aprender-decide, enforced |
| 17 | apr-bound.tensor_count | contracts/decide-apr-v1.yaml | rust | aprender-decide:lib | artifact::ladder::tensor_count_over_cap | class B artifact half (08-20, 08-26); owner_crate aprender-decide, enforced |
| 18 | apr-bound.index_extent | contracts/decide-apr-v1.yaml | rust | aprender-decide:lib | artifact::ladder::index_extent_too_small, artifact::ladder::index_past_end, artifact::ladder::rung2_predicate_exhaustive | class B artifact half (08-20, 08-26); owner_crate aprender-decide, enforced |
| 19 | apr-bound.index_reservation | contracts/decide-apr-v1.yaml | rust | apr-format:lib | v2::tests::forged_tensor_count_reserves_nothing_proportional, v2::tests::index_capacity_is_bounded_by_the_index_bytes | class B artifact half (08-20, 08-26); owner_crate apr-format, enforced |
| 20 | apr-bound.duplicate_names_reader | contracts/decide-apr-v1.yaml | rust | apr-format:lib | v2::tests::duplicate_tensor_names_are_refused_by_both_readers | class B artifact half (08-20, 08-26); owner_crate apr-format, enforced |
| 21 | apr-bound.duplicate_names_ladder | contracts/decide-apr-v1.yaml | rust | aprender-decide:lib | artifact::ladder::repeated_name_walk_names_the_first_repeat, artifact::ladder::duplicate_tensor_name_is_refused_at_load | class B artifact half (08-20, 08-26); owner_crate aprender-decide, enforced |
| 22 | apr-bound.num_hidden_layers@aprender-decide:lib | contracts/decide-apr-v1.yaml | rust | aprender-decide:lib | artifact::ladder::untrusted_layer_counts_are_bounded_before_derivation | class B artifact half (08-20, 08-26); owner_crate aprender-decide, enforced |
| 23 | apr-bound.num_hidden_layers@aprender-core:lib | contracts/decide-apr-v1.yaml | rust | aprender-core:lib | models::modernbert::config::tests::layer_count_is_capped_before_allocation | class B artifact half (08-20, 08-26); owner_crate aprender-decide, enforced |
| 24 | apr-bound.head_layers | contracts/decide-apr-v1.yaml | rust | aprender-decide:lib | artifact::ladder::untrusted_layer_counts_are_bounded_before_derivation | class B artifact half (08-20, 08-26); owner_crate aprender-decide, enforced |
| 25 | apr-bound.activation_and_rope@aprender-decide:lib | contracts/decide-apr-v1.yaml | rust | aprender-decide:lib | artifact::ladder::artifact_bounds_table_is_swept | class B artifact half (08-20, 08-26); owner_crate aprender-decide, enforced |
| 26 | apr-bound.activation_and_rope@aprender-core:lib | contracts/decide-apr-v1.yaml | rust | aprender-core:lib | models::modernbert::config::tests::unsupported_forward_semantics_are_refused | class B artifact half (08-20, 08-26); owner_crate aprender-decide, enforced |
| 27 | apr-bound.encoder_dims | contracts/decide-apr-v1.yaml | rust | aprender-core:lib | models::modernbert::config::tests::config_domain, models::modernbert::load::tests::refuses_shape_mismatch | class B artifact half (08-20, 08-26); owner_crate aprender-core, enforced |
| 28 | apr-bound.tensor_dtype | contracts/decide-apr-v1.yaml | rust | aprender-decide:lib | artifact::ladder::artifact_bounds_table_is_swept | class B artifact half (08-20, 08-26); owner_crate aprender-decide, enforced |
| 29 | apr-bound.load_tensor_dtype | contracts/decide-apr-v1.yaml | rust | aprender-core:lib | models::modernbert::load::tests::refuses_foreign_dtype_and_truncated_payload | class B artifact half (08-20, 08-26); owner_crate aprender-core, enforced |
| 30 | apr-bound.tensor_entry_size | contracts/decide-apr-v1.yaml | rust | aprender-decide:lib | artifact::ladder::size_mismatch, artifact::ladder::size_rule_exhaustive | class B artifact half (08-20, 08-26); owner_crate aprender-decide, enforced |
| 31 | apr-bound.tensor_data_range | contracts/decide-apr-v1.yaml | rust | aprender-decide:lib | artifact::ladder::artifact_bounds_table_is_swept | class B artifact half (08-20, 08-26); owner_crate aprender-decide, enforced |
| 32 | apr-bound.tensor_name_set | contracts/decide-apr-v1.yaml | rust | aprender-decide:lib | artifact::ladder::missing_tensor, artifact::ladder::extra_tensor | class B artifact half (08-20, 08-26); owner_crate aprender-decide, enforced |
| 33 | apr-bound.non_finite | contracts/decide-apr-v1.yaml | rust | aprender-decide:lib | artifact::ladder::nan_weight | class B artifact half (08-20, 08-26); owner_crate aprender-decide, enforced |
| 34 | apr-bound.criteria_count | contracts/decide-apr-v1.yaml | rust | aprender-decide:lib | task::tests::too_many_criteria_refused_while_reading, task::tests::max_criteria_matches_contract | class B artifact half (08-20, 08-26); owner_crate aprender-decide, enforced |
| 35 | apr-bound.probe_row_tokens | contracts/decide-apr-v1.yaml | rust | aprender-decide:lib | artifact::ladder::probe_row_budget_checked_before_replay_forward, artifact::ladder::probe_row_over_budget | class B artifact half (08-20, 08-26); owner_crate aprender-decide, enforced |
| 36 | apr-bound.tokenizer_truncation_padding | contracts/decide-apr-v1.yaml | rust | aprender-decide:lib | laya::tests::tokenizer_truncation_and_padding_are_disabled, artifact::ladder::artifact_bounds_table_is_swept | class B artifact half (08-20, 08-26); owner_crate aprender-decide, enforced |
| 37 | apr-bound.inspect_read | contracts/decide-apr-v1.yaml | rust | aprender-decide:lib | artifact::ladder::declared_length_over_cap, artifact::ladder::artifact_bounds_table_is_swept | class B artifact half (08-20, 08-26); owner_crate aprender-decide, enforced |
| 38 | apr-bound.agent_max_len | contracts/decide-apr-v1.yaml | rust | aprender-decide:lib | artifact::ladder::artifact_bounds_table_is_swept | class B artifact half (08-20, 08-26); owner_crate aprender-decide, accepted |
| 39 | apr-bound.agent_head_max_len | contracts/decide-apr-v1.yaml | rust | aprender-decide:lib | artifact::ladder::artifact_bounds_table_is_swept | class B artifact half (08-20, 08-26); owner_crate aprender-decide, accepted |
| 40 | apr-bound.encoder_row_length | contracts/decide-apr-v1.yaml | rust | aprender-decide:lib | artifact::ladder::artifact_bounds_table_is_swept | class B artifact half (08-20, 08-26); owner_crate aprender-decide, accepted |
| 41 | apr-bound.layer_row_length | contracts/decide-apr-v1.yaml | rust | aprender-core:lib | models::modernbert::layer::tests::layer_forward_empty_row_is_refused, models::modernbert::layer::tests::layer_forward_huge_l_is_refused_before_rope, models::modernbert::layer::tests::layer_forward_with_rope_empty_row_is_refused | class B artifact half (08-20, 08-26); owner_crate aprender-core, enforced |
| 42 | apr-bound.token_ids@aprender-core:lib | contracts/decide-apr-v1.yaml | rust | aprender-core:lib | models::modernbert::tests::forward_refuses_out_of_vocab_and_empty | class B artifact half (08-20, 08-26); owner_crate aprender-core, enforced |
| 43 | apr-bound.token_ids@aprender-decide:lib | contracts/decide-apr-v1.yaml | rust | aprender-decide:lib | artifact::ladder::probe_replay_failure_is_rung_7 | class B artifact half (08-20, 08-26); owner_crate aprender-core, enforced |
| 44 | apr-bound.config_blob_parse | contracts/decide-apr-v1.yaml | rust | aprender-decide:lib | artifact::ladder::artifact_bounds_table_is_swept | class B artifact half (08-20, 08-26); owner_crate aprender-decide, accepted |
| 45 | apr-bound.tokenizer_pipeline | contracts/decide-apr-v1.yaml | rust | aprender-decide:lib | artifact::ladder::artifact_bounds_table_is_swept | class B artifact half (08-20, 08-26); owner_crate aprender-decide, accepted |
| 46 | apr-bound.run_dir_files | contracts/decide-apr-v1.yaml | rust | aprender-decide:lib | artifact::ladder::artifact_bounds_table_is_swept | class B artifact half (08-20, 08-26); owner_crate aprender-decide, accepted |
| 47 | tool-bound.frame_stdio | contracts/decide-tool-boundary-v1.yaml | rust | aprender-mcp-decide:lib | tests::request_bounds_table_is_swept | class B request half (08-23, 08-24); owner_crate aprender-mcp-decide, accepted |
| 48 | tool-bound.frame_http | contracts/decide-tool-boundary-v1.yaml | rust | aprender-mcp-decide-lambda:lib | tests::lambda_request_rows_are_swept, tests::server_config_is_stateless | class B request half (08-23, 08-24); owner_crate aprender-mcp-decide-lambda, enforced |
| 49 | tool-bound.args_shape | contracts/decide-tool-boundary-v1.yaml | rust | aprender-mcp-decide:lib | tests::request_bounds_table_is_swept, tests::malformed_arguments_are_refused_without_echo | class B request half (08-23, 08-24); owner_crate aprender-mcp-decide, enforced |
| 50 | tool-bound.texts_count_min | contracts/decide-tool-boundary-v1.yaml | rust | aprender-mcp-decide:lib | tests::request_bounds_table_is_swept | class B request half (08-23, 08-24); owner_crate aprender-mcp-decide, enforced |
| 51 | tool-bound.texts_count_max | contracts/decide-tool-boundary-v1.yaml | rust | aprender-mcp-decide:lib | tests::request_bounds_table_is_swept, tests::count_is_checked_before_element_shapes | class B request half (08-23, 08-24); owner_crate aprender-mcp-decide, enforced |
| 52 | tool-bound.text_bytes | contracts/decide-tool-boundary-v1.yaml | rust | aprender-mcp-decide:lib | tests::request_bounds_table_is_swept | class B request half (08-23, 08-24); owner_crate aprender-mcp-decide, enforced |
| 53 | tool-bound.built_tokens_total | contracts/decide-tool-boundary-v1.yaml | rust | aprender-mcp-decide:lib | tests::request_bounds_table_is_swept | class B request half (08-23, 08-24); owner_crate aprender-mcp-decide, enforced |
| 54 | tool-bound.served_task_min_row | contracts/decide-tool-boundary-v1.yaml | rust | aprender-mcp-decide:lib | tests::request_bounds_table_is_swept, tests::served_task_must_fit_the_contracted_count | class B request half (08-23, 08-24); owner_crate aprender-mcp-decide, enforced |
| 55 | tool-bound.probe_id_header | contracts/decide-tool-boundary-v1.yaml | rust | aprender-mcp-decide-lambda:lib | tests::lambda_request_rows_are_swept, tests::probe_id_accepts_only_short_safe_ids | class B request half (08-23, 08-24); owner_crate aprender-mcp-decide-lambda, enforced |
| 56 | apr.verify_checks.2 | contracts/decide-apr-v1.yaml | rust | aprender-decide:lib | verify::tests::base_identity_mismatch_revision, verify::tests::base_encoder_config_unpinned_refused, verify::tests::base_agent_config_unpinned_refused, verify::tests::base_mismatch_tokenizer, artifact::ladder::manifest_base_disagrees_with_recipe_blob | CR-01,V6-b,V6-c (08-19 honest wording, 08-21 bound) |
| 57 | apr.load_ladder.rung2-version | contracts/decide-apr-v1.yaml | rust | aprender-decide:lib | artifact::ladder::header_version_refused | V9-d,A2-8 (08-26) |
| 58 | apr.load_ladder.rung4-unique-and-bound | contracts/decide-apr-v1.yaml | rust | aprender-decide:lib | artifact::ladder::repeated_name_walk_names_the_first_repeat, artifact::ladder::duplicate_tensor_name_is_refused_at_load, artifact::ladder::manifest_bound_at_every_load_door | WR-01,CR-01 (08-19, 08-26) |
| 59 | apr.load_ladder.rung7-replay | contracts/decide-apr-v1.yaml | rust | aprender-decide:lib | artifact::ladder::probe_replay_failure_is_rung_7, artifact::ladder::probe_row_budget_checked_before_replay_forward | V9-b (08-26) |
| 60 | apr.manifest.bindings | contracts/decide-apr-v1.yaml | rust | aprender-decide:lib | artifact::ladder::every_manifest_leaf_is_bound, artifact::ladder::manifest_bindings_table_matches_manifest_leaves | CR-01 load side (08-19) |
| 61 | apr.untrusted_input_bounds.table | contracts/decide-apr-v1.yaml | rust | aprender-decide:lib | artifact::ladder::artifact_bounds_table_is_swept | class B artifact half (08-26) |
| 62 | apr.max_criteria | contracts/decide-apr-v1.yaml | rust | aprender-decide:lib | task::tests::max_criteria_matches_contract, task::tests::too_many_criteria_refused_while_reading | V1-b (08-26: 510, not the plan's 128) |
| 63 | apr.pv-valid | contracts/decide-apr-v1.yaml | pv | aprender-contracts-cli | FALSIFY-DECIDE-APR-013 | decide-apr-v1 3.0.0 validates |
| 64 | tool.classify_count_bound | contracts/decide-tool-boundary-v1.yaml | rust | aprender-mcp-decide:lib | tests::count_is_checked_before_element_shapes, tests::request_bounds_table_is_swept | V5-c,A4-8 (08-23) |
| 65 | tool.classify_count_bound.live-stdio | contracts/decide-tool-boundary-v1.yaml | rust | aprender-mcp-decide:test=e2e_stdio | the_tiny_decide_server_classifies_over_live_stdio | V5-c (08-23 live leg) |
| 66 | tool.classify_admission.transport_dispatch | contracts/decide-tool-boundary-v1.yaml | rust | aprender-mcp-decide:test=e2e_stdio | pipelined_calls_are_serialized_not_refused | V5-b (08-28) |
| 67 | tool.refusal_names_bound.formula | contracts/decide-tool-boundary-v1.yaml | rust | aprender-mcp-decide:lib | tests::every_refusal_names_key_without_text, tests::error_taxonomy_budget_rejected_model_internal | A4-7,V5-a (08-28) |
| 68 | tool.refusal_names_bound.wire-stdio | contracts/decide-tool-boundary-v1.yaml | rust | aprender-mcp-decide:test=e2e_stdio | bound_refusals_are_iserror_results_over_live_stdio | V5-a wire code, B-iserror (08-28) |
| 69 | tool.refusal_names_bound.wire-http | contracts/decide-tool-boundary-v1.yaml | rust | aprender-mcp-decide-lambda:lib | tests::loopback_bound_refusal_is_an_iserror_result | V5-a wire code, B-iserror (08-28) |
| 70 | tool.refusal_echo_min_chars | contracts/decide-tool-boundary-v1.yaml | rust | aprender-mcp-decide:lib | tests::every_refusal_names_key_without_text | A4-7 (08-28) |
| 71 | tool.served_fields | contracts/decide-tool-boundary-v1.yaml | rust | aprender-mcp-decide:lib | tests::every_served_field_has_a_bound_source | class A served half (08-23, 08-28 description.truncation) |
| 72 | tool.untrusted_input_bounds.table | contracts/decide-tool-boundary-v1.yaml | rust | aprender-mcp-decide:lib | tests::request_bounds_table_is_swept | class B request half (08-23) |
| 73 | tool.untrusted_input_bounds.table-lambda | contracts/decide-tool-boundary-v1.yaml | rust | aprender-mcp-decide-lambda:lib | tests::lambda_request_rows_are_swept | class B request half, Lambda rows (08-24) |
| 74 | tool.tier.max_texts | contracts/decide-tool-boundary-v1.yaml | rust | aprender-mcp-decide:lib | tests::bounds_match_contract | D-18 10,240 MB tier (08-30) |
| 75 | tool.tier.max_total_tokens | contracts/decide-tool-boundary-v1.yaml | rust | aprender-mcp-decide:lib | tests::bounds_match_contract | D-18 10,240 MB tier (08-30 keep-800-apply-10739) |
| 76 | tool.tier.memory | contracts/decide-tool-boundary-v1.yaml | rust | aprender-mcp-decide-lambda:lib | tests::deploy_memory_is_the_contract_tier | D-18 restore-10240-tier (08-30) |
| 77 | tool.648-vs-800 | contracts/decide-tool-boundary-v1.yaml | evidence | - | <phase>/08-LIVE-REDEPLOY-EVIDENCE.json#deadline_decision.choice | D-ITEM-08-30-B (08-30) |
| 78 | tool.accepted_region_cold.live | contracts/decide-tool-boundary-v1.yaml | evidence | - | <phase>/08-LIVE-REDEPLOY-EVIDENCE.json#live.final | V4-b,D-18 (08-30) |
| 79 | tool.FALSIFY-DECIDE-TOOL-009.evidence | contracts/decide-tool-boundary-v1.yaml | evidence | - | <phase>/08-LIVE-REDEPLOY-EVIDENCE.json#live.accepted_region_cold | D-18 live (08-30) |
| 80 | tool.FALSIFY-DECIDE-TOOL-009.harness | contracts/decide-tool-boundary-v1.yaml | just | - | laya-deploy-verify | D-18 live harness (08-30) |
| 81 | tool.pv-valid | contracts/decide-tool-boundary-v1.yaml | pv | aprender-contracts-cli | FALSIFY-DECIDE-TOOL-009 | decide-tool-boundary-v1 9.0.0 validates |
| 82 | gate.base-pins | contracts/laya-finetune-gate-v1.yaml | rust | aprender-decide:lib | verify::tests::base_encoder_config_unpinned_refused, verify::tests::base_agent_config_unpinned_refused, verify::tests::base_mismatch_tokenizer | V6-c (08-21) |
| 83 | gate.recipe.schedule_literal | contracts/laya-finetune-gate-v1.yaml | rust | aprender-decide:lib | verify::tests::recipe_block_differs_from_contract_refused, verify::tests::recipe_epochs_break_the_rule_refused | V6-d (08-21) |
| 84 | gate.run_field_bindings | contracts/laya-finetune-gate-v1.yaml | rust | aprender-decide:lib | verify::tests::every_run_field_is_bound_or_report_only, verify::tests::run_field_bindings_table_matches_fixture_leaves, verify::tests::f_avg_forged_refused | CR-01 verify side (08-21), V6-e f_avg rows (08-27) |
| 85 | gate.numeric_agreement.rust | contracts/laya-finetune-gate-v1.yaml | rust | aprender-decide:lib | verify::tests::gate_numeric_cases_agree_bit_for_bit, verify::tests::ece_bin_confidence_sums_are_exact_in_f64 | V7-a,V7-b,C2-1 (08-27) |
| 86 | gate.numeric_agreement.python | contracts/laya-finetune-gate-v1.yaml | python | - | 23 numeric cases bit for bit | V7-a,V7-b,C2-1 (08-27) |
| 87 | gate.numeric_agreement.core-f64 | crates/aprender-core/src/metrics/exact_sum.rs | rust | aprender-core:lib | metrics::exact_sum::tests::exact_sum_matches_frozen_math_fsum, metrics::f32_bits_tests::f32_metrics_are_bit_identical_to_the_frozen_snapshot | V7-a (08-27: f64 added beside the frozen f32) |
| 88 | gate.early_stopping_train_side | contracts/laya-finetune-gate-v1.yaml | python | - | best-anchored: the equation, FALSIFY-LAYA-GATE-009 and tie_break_rule all state the best-anchored rule | V13-c (08-29) |
| 89 | gate.early_stopping.tie_break_rule | contracts/laya-finetune-gate-v1.yaml | python | - | best-anchored: trace [1.0, 0.9993, 0.9988] (min_delta 0.001) -> epoch 3 | V13-c (08-29) |
| 90 | FALSIFY-LAYA-GATE-009 | contracts/laya-finetune-gate-v1.yaml | python | - | best-anchored: the run of record's seed-17 trace replays to best_epoch 2, epochs_run 5, patience | V13-c (08-29) |
| 91 | FALSIFY-LAYA-GATE-003.python | contracts/laya-finetune-gate-v1.yaml | python | - | frozen house cases (scripts/setfit_fixtures/claims_stats/ece_top_label_cases.json) | V7-b (08-27) |
| 92 | FALSIFY-LAYA-GATE-015.python | contracts/laya-finetune-gate-v1.yaml | python | - | margin_exact_1_20_9row | V7-a (08-27) |
| 93 | gate.python_refusals | contracts/laya-finetune-gate-v1.yaml | python | - | PYTHON REFUSALS swept=42 lifecycle=5 | V13-a,IN-08,WR-09 (08-29) |
| 94 | gate.rescore_noise.k-float | contracts/laya-finetune-gate-v1.yaml | python | - | contract-value k = 4 (the committed integer) is read as the float 4.0 | WR-09,V13-d (08-29) |
| 95 | gate.pv-valid | contracts/laya-finetune-gate-v1.yaml | pv | aprender-contracts-cli | FALSIFY-LAYA-GATE-015 | laya-finetune-gate-v1 5.0.0 validates |
| 96 | mcp.description.truncation-sentence | crates/aprender-mcp-decide/src/lib.rs | rust | aprender-mcp-decide:lib | tests::description_truncation_sentence_matches_tier, tests::every_served_field_has_a_bound_source | WR-03 (08-28 A-derive) |
| 97 | mcp.readme.truncation | crates/aprender-mcp-decide/README.md | rust | aprender-mcp-decide:lib | tests::description_truncation_sentence_matches_tier | WR-03 (08-28) |
| 98 | mcp.readme.admission | crates/aprender-mcp-decide/README.md | rust | aprender-mcp-decide:test=e2e_stdio | pipelined_calls_are_serialized_not_refused | V5-b (08-28) |
| 99 | mcp.readme.iserror | crates/aprender-mcp-decide/README.md | rust | aprender-mcp-decide:test=e2e_stdio | bound_refusals_are_iserror_results_over_live_stdio | V5-a wire code, B-iserror (08-28) |
| 100 | lambda.readme.lazy-load | crates/aprender-mcp-decide-lambda/README.md | rust | aprender-mcp-decide-lambda:lib | tests::non_post_methods_do_not_load, tests::health_is_not_ok_on_invalid_config, tests::loopback_end_exits_the_process, s3::tests::failed_load_leaves_the_cell_empty_and_the_next_call_loads | V3-c,S6,V4-c,A4-6,WR-06 (08-24) |
| 101 | lambda.main.oncecell-doc | crates/aprender-mcp-decide-lambda/src/main.rs | rust | aprender-mcp-decide-lambda:lib | s3::tests::failed_load_leaves_the_cell_empty_and_the_next_call_loads | V3-c,S6 (08-24) |
| 102 | lambda.s3.oncecell-doc | crates/aprender-mcp-decide-lambda/src/s3.rs | rust | aprender-mcp-decide-lambda:lib | s3::tests::failed_load_leaves_the_cell_empty_and_the_next_call_loads | V3-c,S6 (08-24) |
| 103 | lambda.s3.download-deadline | crates/aprender-mcp-decide-lambda/src/s3.rs | rust | aprender-mcp-decide-lambda:lib | tests::download_deadline_is_derived_from_the_contract_and_the_samples, tests::download_budget_is_the_smaller_of_the_deadline_and_what_the_invocation_leaves, s3::tests::deadline_abandons_stalled_parts | V4-b,A4-4,D2-4 (08-30) |
| 104 | lambda.template.tell | crates/aprender-mcp-decide-lambda/.pmcp/deploy.toml.template | rust | aprender-mcp-decide-lambda:lib | probe::tests::probe_fails_when_identity_differs, probe::tests::probe_fails_when_response_labels_differ, probe::tests::ok_requires_every_check | IN-04,V12-a (08-25) |
| 105 | lambda.template.memory | crates/aprender-mcp-decide-lambda/.pmcp/deploy.toml.template | rust | aprender-mcp-decide-lambda:lib | tests::deploy_memory_is_the_contract_tier | D-18 restore-10240-tier (08-30) |
| 106 | lambda.probe.labels-segment | crates/aprender-mcp-decide-lambda/src/probe.rs | rust | aprender-mcp-decide-lambda:lib | probe::tests::labels_segment_is_parsed_exactly, probe::tests::maximal_request_uses_the_server_budget | IN-03,R6 (08-25) |
| 107 | coverage.iam-read-only | <phase>/COVERAGE.md | just | - | laya-grant, _laya-grant-check, laya-gates-selftest | WR-07,V11-b (08-22; gate rows grant-check, grant-listing-failure) |
| 108 | coverage.no-listbucket | <phase>/COVERAGE.md | just | - | _laya-iam-check | 08-22 COVERAGE correction (gate row iam-check) |
| 109 | coverage.teardown | <phase>/COVERAGE.md | just | - | laya-teardown | V11-b,CV4 (08-22; gate row teardown-classify) |
| 110 | claude-md.decide-row.tier | CLAUDE.md | evidence | - | <phase>/08-LIVE-REDEPLOY-EVIDENCE.json#live.final | D-18 (08-30) |
| 111 | claude-md.decide-row.bounds | CLAUDE.md | rust | aprender-mcp-decide:lib | tests::bounds_match_contract | D-18 (08-30) |
| 112 | context.D-09.amendment-08-28 | <phase>/08-CONTEXT.md | rust | aprender-mcp-decide:test=e2e_stdio | bound_refusals_are_iserror_results_over_live_stdio | WR-03,V5-a owner decision (08-28) |
| 113 | context.D-14.amendment-08-31 | <phase>/08-CONTEXT.md | rust | aprender-decide:lib | tests::publish_is_false_until_the_crate_name_is_confirmed | V14-c,D-14 owner decision (08-31) |
| 114 | context.D-18.restore-10240-tier | <phase>/08-CONTEXT.md | evidence | - | <phase>/08-LIVE-REDEPLOY-EVIDENCE.json#tier_decision | D-18 owner decision (08-30) |
| 115 | context.D-18.redeploy-and-measure | <phase>/08-CONTEXT.md | evidence | - | <phase>/08-LIVE-REDEPLOY-EVIDENCE.json#decision | D-18 owner decision (08-30) |
| 116 | context.D-18.keep-800-apply-10739 | <phase>/08-CONTEXT.md | evidence | - | <phase>/08-LIVE-REDEPLOY-EVIDENCE.json#deadline_decision.stop_guard_reference_set_correction | D-18 owner decision; stop-guard reference set corrected (08-30) |
| 117 | decide.cargo.publish-false | crates/aprender-decide/Cargo.toml | rust | aprender-decide:lib | tests::publish_is_false_until_the_crate_name_is_confirmed | V14-c,CV2 Phase 8 share (08-31) |
| 118 | gates.cascade-guard | scripts/laya_gates.tsv | just | - | laya-gates-selftest | V14-c (08-31: the external row now runs a case) |
| 119 | decide.argmax.nan-never-wins | crates/aprender-decide/src/laya/mod.rs | rust | aprender-decide:lib | laya::tests::argmax_nan_never_wins | IN-01 (08-31) |
| 120 | decide.ui.runs-in-ci | crates/aprender-decide/tests/ui.rs | rust | aprender-decide:test=ui | decider_has_no_second_minting_path | IN-04 (08-31) |
| 121 | ci.integration-line.ui | .github/workflows/ci.yml | rust | aprender-decide:test=ui | decider_has_no_second_minting_path | IN-04, CI wiring (08-12) |
| 122 | ci.integration-line.e2e-stdio | .github/workflows/ci.yml | rust | aprender-mcp-decide:test=e2e_stdio | pipelined_calls_are_serialized_not_refused, bound_refusals_are_iserror_results_over_live_stdio, the_tiny_decide_server_classifies_over_live_stdio | CI wiring (08-12); 08-23/08-28 live legs |
| 123 | decide.digest.one-helper | crates/aprender-decide/src/digest.rs | rust | aprender-decide:lib | artifact::determinism::golden_sha | S1,V14-d (08-31 Task 1) |
| 124 | decide.verify.write-atomic | crates/aprender-decide/src/verify.rs | rust | aprender-decide:lib | verify::tests::write_atomic_does_not_follow_a_planted_symlink | V8-c (08-21) |
| 125 | decide.verify.probs-row-coverage | crates/aprender-decide/src/verify.rs | rust | aprender-decide:lib | verify::tests::probs_row_coverage_names_the_file | V8-d (08-21) |
| 126 | decide.verify.slice-fraction | crates/aprender-decide/src/verify.rs | rust | aprender-decide:lib | verify::tests::slice_below_fraction_refused, verify::tests::slice_at_fraction_accepted | WR-08 (08-21) |
| 127 | decide.laya_parity.nan-visible-max | crates/aprender-decide/tests/laya_parity.rs | rust | aprender-decide:test=laya_parity | nan_max_keeps_an_earlier_nan | A2-6 (08-22) |
| 128 | decide.laya_parity.measured-lines | crates/aprender-decide/tests/laya_parity.rs | just | - | _laya-leg-verdict, laya-verify-suite | V11-a,AL7,D3-1 (08-22) |
| 129 | py.gate.margin-comment | scripts/laya_train/gate.py | python | - | margin_exact_1_20_30row | C2-1 (08-27, 08-29) |
| 130 | py.data.jsonl-lines | scripts/laya_train/data.py | python | - | row-split: jsonl_lines == Rust str::lines on the rustc 1.98 case table (9 cases) | WR-05 (08-29) |
| 131 | py.data.row-encoding | scripts/laya_train/data.py | python | - | train.jsonl with an invalid UTF-8 byte -> train-row-encoding (not UnicodeDecodeError) | V13-a,C2-6 (08-29) |
| 132 | py.train.early-stopping-refusal | scripts/laya_train/train.py | python | - | python_refusals early-stopping via train.stopping_record_or_refuse | IN-08 (08-29) |
| 133 | py.prepare_stance.write-once | scripts/laya_train/prepare_stance.py | python | - | python_refusals prepare-out-dir via prepare_stance.write_once | D3-5 (08-29) |

## Task Commits

1. **Task 1 (tracer): one digest module, pack-free load path, ledger gate on its first rows**: `c4a6db7c5` (feat, prior session)
2. **Task 2: owner decision `publish-false`**: no separate commit. It is recorded in Task 3's commit (CONTEXT amendment, Cargo.toml).
3. **Task 3: D-14 implemented, 133-row ledger, argmax / ui.rs, deferrals, gate mutation proof**: `23a640e6a` (feat)

**Plan metadata:** the docs commit that follows this SUMMARY.

## Files Created/Modified

- `crates/aprender-decide/Cargo.toml`: `publish = false` with the D-14 comment.
- `crates/aprender-decide/src/lib.rs`: `publish_is_false_until_the_crate_name_is_confirmed`.
- `crates/aprender-decide/src/laya/mod.rs`: argmax and its doc.
- `crates/aprender-decide/src/laya/tests.rs`: `argmax_nan_never_wins`.
- `crates/aprender-decide/tests/ui.rs`: header.
- `justfile`: `laya-claims-check` takes comma lists for rust and just rows; its docs are updated.
- `scripts/laya_claims.tsv`: 3 → 133 rows.
- `scripts/laya_gates.tsv`: `cascade-guard` runs a case.
- `08-CONTEXT.md`: the D-14 amendment.
- `deferred-items.md`: the final-status section, statuses and owners, and the 08-24 entry resolved.
- Task 1's files are listed in c4a6db7c5.

## Decisions Made

See key-decisions in the frontmatter.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical] `rust` and `just` rows take a comma-separated test list**
- **Found during:** Task 3 (ledgering the FALSIFY commands).
- **Issue:** A FALSIFY or bounds-table command names up to five tests. With one test per row, a ledger row could prove only one of them, and the others could dangle unseen, which is the D-ITEM-08-01-A defect the ledger exists to close.
- **Fix:** each comma-separated name must match a listed test exactly or as its `::`-suffix. The first full run also caught two `just` rows written with lists before `just` supported them.
- **Files modified:** justfile.
- **Committed in:** 23a640e6a.

**2. [Rule 2 - Missing critical] A test and a gate case keep the D-14 decision**
- **Found during:** Task 3.
- **Issue:** The only existing guard is `check_cascade_covers_all_crates.sh`, and it is already red at base (aprender-contrastive-data), so a regression of the decision would be invisible. The ledger row for the D-14 amendment also needed something to enforce it.
- **Fix:** added `publish_is_false_until_the_crate_name_is_confirmed`, and turned the `cascade-guard` gate row (08-22's placeholder, `external:08-31`, `-`) into a case. Both are mutation-proven.
- **Files modified:** crates/aprender-decide/src/lib.rs, scripts/laya_gates.tsv. laya_gates.tsv is not in this plan's `files_modified`; Task 1 had already edited it.
- **Committed in:** 23a640e6a.

**3. [Rule 1 - Stale record] The 08-24 frame_http / probe_id_header deferral was open although 08-28 closed it**
- **Fix:** marked `status: resolved` with the commit (890b769e9) and its ledger rows.
- **Committed in:** 23a640e6a.

**4. [Scope] The CONTEXT D-09 / D-14 rows are `rust`, not `evidence`**
- The plan says "kind evidence: the dated line". There is no JSON record of those two decisions to point at, so each row names the test that enforces the decision. The three D-18 rows are `evidence` rows on 08-LIVE-REDEPLOY-EVIDENCE.json.

**5. [Scope] Every `untrusted_input_bounds` row is ledgered, not only the other-crate ones**
- This is a superset of the plan's list, for the reason in key-decisions.

**6. [Format] rustfmt re-wrapped `argmax_nan_never_wins`**
- Formatting only. Caught by `cargo fmt -p aprender-decide -- --check` before the commit.

**Total deviations:** 3 auto-fixed (2 missing-critical, 1 stale record) and 3 scope/format notes.
**Impact on plan:** Every must-have truth holds. The additions strengthen the gate the plan asked for, and nothing outside the claim/publish surface changed.

## Issues Encountered

- **Every claims check takes about 2:40.** Each `cargo test -p <owner> -- --list` rebuilds aprender-core/-compute/-decide, which is the 08-22 deferred item. A full `laya-claims-check` with the python selftest therefore takes 2:40-2:45, and the gates-selftest `claims-check` row, which runs it twice, took 316 s.
- **The previous session's User-Agent.** The owner's corrections note that the previous executor sent the user's email address in a User-Agent header to crates.io. This session made no outbound network call.

## Verification

- The plan's Task 3 `<automated>` block, run through its own `r()` wrapper, printed `VERIFY rc=0`:
  - the cascade guard does not list aprender-decide;
  - `cargo test -p aprender-decide --lib` gave 165 passed, including `argmax_nan_never_wins`, `publish_is_false_until_the_crate_name_is_confirmed` and `safetensors_is_read_only_by_pack`;
  - `LAYA CLAIMS OK 133 rows`, with n >= 30;
  - the gap-round section and the D-14 line are present;
  - `--test ui` gave 1 passed.
- `cargo test -p aprender-mcp-decide -p aprender-mcp-decide-lambda --lib`: 79 passed.
- `cargo clippy -p aprender-decide --all-targets --no-deps -- -D warnings`: rc 0. `cargo fmt -p aprender-decide -- --check`: rc 0.
- `LAYA_GATES_ONLY=claims-check,cascade-guard just laya-gates-selftest` printed `LAYA GATES ROWS OK 2 of 28`.
  - The drift check reported 29 recipes, 27 gates and 28 rows, all covered.
  - `claims-check` must-fail: `ANCHOR MISSING gates-selftest-anchor`. Must-pass: `LAYA CLAIMS OK 133 rows`.
  - `cascade-guard` was GREEN.
  - AWS CALLS: 0.
- `grep -c dark crates/aprender-decide/tests/ui.rs` gives 0.

## Known Stubs

None.

## Threat Flags

None. No new endpoint, file access or trust boundary was added. The new test reads the crate's own Cargo.toml at test time.

## User Setup Required

None.

## Next Phase Readiness

- Plan 08-32 (`just laya-gap-regression`, the first CI run of Phase 8 code) can call `just laya-claims-check` as class D's sweep. It should expect about 2:40 per run until the rebuild item is fixed.
- The next CI run will be red on two guards that are no longer Phase 8's, and both are itemised in deferred-items.md:
  - the hand-rolled parser guard, over 4 SetFit / forecasting crates;
  - the cascade guard, over aprender-contrastive-data.
- Open and owned elsewhere: tokenizer_pipeline (an owner decision on a pipeline allowlist), D-ITEM-08-30-A/-B, the laya_tiny fixture drift, apr-format golden_v2, and examples/probe.rs's crate-wide allow.

---
*Phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server*
*Completed: 2026-09-29*

## Self-Check: PASSED

- FOUND: crates/aprender-decide/src/digest.rs, scripts/laya_claims.tsv (133 data rows) and this SUMMARY.
- FOUND: commits c4a6db7c5 and 23a640e6a (`git cat-file -e`).
- `commits: 2` was measured as `git rev-list --count 17dae0661..HEAD` before this SUMMARY's commit.
- The ledger table above has 133 rows, the same as the TSV.
