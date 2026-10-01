---
phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
plan: 26
subsystem: artifact-load-ladder
tags: [decide-apr-v1, class-B, untrusted-input-bounds, WR-01, V1-b, V9-a, V9-b, V9-d, IN-02, mutation-proof]

requires:
  - phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
    provides: "08-19 manifest.bindings + check_manifest_bindings (rung 4 e); 08-20 shared-reader duplicate refusal and index reservation; 08-21 typed contract views; 08-22 _laya-leg-verdict; ea940faec layer / head / metadata / activation / load_tensor bounds"
provides:
  - "ArtifactError::DuplicateTensor (rung 4) via first_repeated_tensor_name, a set walk before any lookup by name"
  - "inspect_manifest runs rungs 1-4; pack_laya inspect reads through read_decide_apr_bytes_bounded; non-UTF-8 argv is a usage refusal (exit 2)"
  - "task::MAX_CRITERIA = decide-apr-v1 constants.max_criteria = 510, refused while reading (TaskError::TooManyCriteria); linear HashSet duplicate scan"
  - "Builder::from_bytes disables the tokenizer file's truncation and padding"
  - "rung 2 refuses header.version != VERSION_V2; ArtifactError::ProbeReplay (rung 7) for classify failures during load-time replay"
  - "run_probes checks every probe row against probe_max_row_tokens BEFORE any forward pass (cfg(test) laya::FORWARD_ROWS observes it)"
  - "decide-apr-v1 3.0.0 untrusted_input_bounds (30 rows) swept by artifact::ladder::artifact_bounds_table_is_swept"
affects: [08-27, 08-31, aprender-mcp-decide, aprender-mcp-decide-lambda]

actuals:
  tokens: 18179
  tasks: 3
  commits: 3
plan_head_before: 6eb5453c1d3f8e5377c42a8cef2d64a737a1c2f3

tech-stack:
  added: []
  patterns:
    - "Contract-driven class-B sweep: every row id dispatches to a hostile case built from the packed tiny artifact; the refusal's variant and rung must be the row's; accepted rows assert their documented behaviour; external rows must name a fn that exists in their crate"
    - "Test-only forward counter (thread_local) proves a bound bites BEFORE the forward, where the refusal alone is identical before or after"
    - "Timing assertion placed before the result assertion, so a quadratic scan is caught by the clock and not only by the result"

key-files:
  created: []
  modified:
    - crates/aprender-decide/src/artifact.rs
    - crates/aprender-decide/src/artifact/ladder.rs
    - crates/aprender-decide/src/task.rs
    - crates/aprender-decide/src/laya/builder.rs
    - crates/aprender-decide/src/laya/mod.rs
    - crates/aprender-decide/src/laya/tests.rs
    - crates/aprender-decide/examples/pack_laya.rs
    - contracts/decide-apr-v1.yaml
    - justfile

key-decisions:
  - "max_criteria is 510 (max_len 512 - 2), NOT the plan's max_len / 4 = 128: measured on Laya-en's tokenizer, one-token names keep every marker up to K = 253, so 128 would refuse tasks Laya serves"
  - "Load-time probe rows are checked against probe_max_row_tokens before the forward: the replay was the one forward sized by the artifact's max_len and tokenizer"
  - "The rung-7 test forges an out-of-vocabulary tokenizer id (a real artifact), because on the tiny tokenizer the served and probe prefixes are the same length and a max_len split cannot separate them"
  - "decide-apr-v1 bumped 2.0.0 -> 3.0.0 on pv diff's major suggestion (the task_order_is_label_index invariant gained a refusal), the 08-19 precedent"
  - "tokenizer_pipeline is ACCEPTED: the ladder bounds the blob's bytes and pins its digest, but not what a normalizer can expand; deployability binds it to the pinned Laya-en tokenizer through verify"

patterns-established:
  - "Class-B artifact table: bound + checked_by + rung + refusal + owner_crate + test per row, and one sweep that fails on an unknown id, a missing case or a missing test"

requirements-completed: [D-05, D-12, D-17]

coverage:
  - id: D1
    description: "A repeated tensor name is refused at load naming it (rung 3 today by the reader; rung 4 DuplicateTensor as defence in depth), and the rung-4 walk is proven on its own"
    requirement: D-17
    verification:
      - kind: unit
        ref: "crates/aprender-decide/src/artifact/ladder.rs#duplicate_tensor_name_is_refused_at_load"
        status: pass
      - kind: unit
        ref: "crates/aprender-decide/src/artifact/ladder.rs#repeated_name_walk_names_the_first_repeat"
        status: pass
    human_judgment: false
  - id: D2
    description: "inspect prints identity only after rungs 1-4 on a bounded read, and the deployed file inspects to 24a44d7e... / laya-en-root@55cf4c4e; a non-UTF-8 argument exits 2"
    requirement: D-17
    verification:
      - kind: unit
        ref: "crates/aprender-decide/src/artifact/ladder.rs#inspect_refuses_a_manifest_its_blobs_contradict"
        status: pass
      - kind: e2e
        ref: "just laya-inspect models/decide/laya-stance-64.apr; target/release/examples/pack_laya inspect $'x\\377y' -> exit 2"
        status: pass
    human_judgment: false
  - id: D3
    description: "A task with more than max_criteria criteria is refused while reading; 100 000 criteria refused in under a second; duplicates found linearly"
    requirement: D-05
    verification:
      - kind: unit
        ref: "crates/aprender-decide/src/task.rs#too_many_criteria_refused_while_reading"
        status: pass
      - kind: unit
        ref: "crates/aprender-decide/src/task.rs#max_criteria_matches_contract"
        status: pass
    human_judgment: false
  - id: D4
    description: "A tokenizer.json truncation / padding block cannot change a built row or a served decision"
    requirement: D-12
    verification:
      - kind: unit
        ref: "crates/aprender-decide/src/laya/tests.rs#tokenizer_truncation_and_padding_are_disabled"
        status: pass
    human_judgment: false
  - id: D5
    description: "Rung 2 refuses a non-v2 header version; a load-time replay failure is a rung-7 refusal; over-budget probe rows are refused before any forward"
    requirement: D-17
    verification:
      - kind: unit
        ref: "crates/aprender-decide/src/artifact/ladder.rs#header_version_refused"
        status: pass
      - kind: unit
        ref: "crates/aprender-decide/src/artifact/ladder.rs#probe_replay_failure_is_rung_7"
        status: pass
      - kind: unit
        ref: "crates/aprender-decide/src/artifact/ladder.rs#probe_row_budget_checked_before_replay_forward"
        status: pass
    human_judgment: false
  - id: D6
    description: "decide-apr-v1 untrusted_input_bounds enumerates the artifact half of class B and one test sweeps it"
    requirement: D-17
    verification:
      - kind: unit
        ref: "crates/aprender-decide/src/artifact/ladder.rs#artifact_bounds_table_is_swept (ARTIFACT BOUNDS swept=18 accepted=6 external=6)"
        status: pass
      - kind: other
        ref: "pv validate contracts/decide-apr-v1.yaml (0 errors); make contract-audit-phase8 (rc 0)"
        status: pass
    human_judgment: false
  - id: D7
    description: "The deployed artifact still verifies eligible with shipped_seed 17, real-weights parity is unchanged, the tiny golden is unchanged"
    requirement: D-17
    verification:
      - kind: e2e
        ref: "heavy just laya-verify <deployed 24a44d7e> <run> <data> <base 55cf4c4e> -> deploy_eligible true, shipped_seed 17, argmax 459/459"
        status: pass
      - kind: integration
        ref: "heavy env LAYA_MODEL_DIR=<55cf4c4e> LAYA_LADDER_BIN=<main ladder> cargo test -p aprender-decide --release --test laya_parity | just _laya-leg-verdict laya_parity -> LEG OK"
        status: pass
      - kind: unit
        ref: "crates/aprender-decide/src/artifact/determinism.rs#golden_sha (both serde backings)"
        status: pass
    human_judgment: false

duration: 41min
completed: 2026-09-28
status: complete
---

# Phase 8 Plan 26: Class B, Artifact Half Summary

**The decide ladder now bounds every size and config value an artifact supplies, before that value sizes any work.** New refusals: a repeated tensor name (rung 4), a non-v2 header version (rung 2), more than 510 criteria (refused while reading, linear duplicate scan), and a probe row over 48 tokens (refused before the replay forward). A tokenizer file's own truncation and padding are ignored. A classify failure during replay is reported at rung 7, and `inspect` prints only identity bound to the blobs, after rungs 1-4 on a bounded read. All of it is recorded in the 30-row `untrusted_input_bounds` table in decide-apr-v1 3.0.0, and one test sweeps that table. Every new bound was mutated and went RED. The deployed 24a44d7e artifact still verifies as eligible, and real-weights parity is still 3.841e-6.

## Performance

- **Duration:** about 41 min
- **Started:** 2026-09-28T19:58:56Z
- **Completed:** 2026-09-28T20:40Z
- **Tasks:** 3
- **Files modified:** 9

## Accomplishments

- **WR-01 (decide side).** `first_repeated_tensor_name` walks the index names with a set, so it does not depend on the sort order. It runs at the start of rung 4, before any lookup by name, and returns `DuplicateTensor { name }`. Loading a repacked artifact that names `encoder.embeddings.norm.weight` twice is refused at **rung 3** by plan 08-20's reader (`duplicate tensor name "encoder.embeddings.norm.weight" in tensor index`). The rung-4 check is defence in depth, and its logic is proven directly by its own unit test.
- **IN-02 / A2-3.** `inspect_manifest` now runs rungs 1-4, including 08-19's manifest bindings. Before this change `inspect_refuses_a_manifest_its_blobs_contradict` went RED: inspect accepted a base the recipe contradicts. `pack_laya inspect` opens the file and reads it with `read_decide_apr_bytes_bounded(file, Some(metadata.len()))`. `main` uses `args_os`. The `justfile` recipe comment now says rungs 1-4.
- **V1-b.** `MAX_CRITERIA = 510` is checked in `visit_map` before the entry is stored. The refusal is a custom serde error mapped back to the typed `TooManyCriteria`, and parsing stops there. Duplicates are found with a `HashSet`. A 100 000-criteria task is refused in milliseconds in a debug build.
- **V9-a.** `Builder::from_bytes` calls `with_truncation(None)` and `with_padding(None)`. Before the fix, an injected `max_length 4` + `Fixed(40)` block produced a 64-token row padded with zeros that was marked `truncated: true`. The same text builds a 25-token row without the block.
- **V9-d.** Rung 2 refuses `header.version != VERSION_V2` and names the version (tested with 3.0, 2.1 and 1.0). The container reader alone accepts those headers.
- **V9-b.** A classify failure during load-time replay is now `ProbeReplay` at rung "7 probe_replay". Before the fix, a forged out-of-vocab tokenizer id was reported as `rung 6 rebuild: ... token id 1512 ... out of vocabulary`.
- **New bound found by the enumeration.** The load-time probe replay was the only forward pass a load runs. Its row length came from the artifact's `max_len` and tokenizer, and was checked against `probe_max_row_tokens` only after the forward. `run_probes` now builds and checks every probe row first. The proof is a merges-free tokenizer artifact (probe row 0 has 64 tokens): it is refused as `ProbeMismatch{0, tokens}` while a `#[cfg(test)]` forward counter stays at 0.
- **The table:** `untrusted_input_bounds { see_also, rows }` in decide-apr-v1, swept by `artifact_bounds_table_is_swept`, which prints `ARTIFACT BOUNDS swept=18 accepted=6 external=6`.

## Class-B artifact-side surface (decide-apr-v1 `untrusted_input_bounds`)

| id | bound | rung | disposition | owner | refusal / behaviour | test |
|---|---|---|---|---|---|---|
| file_length | max_artifact_bytes | 1 | enforced | aprender-decide | ArtifactTooLarge | declared_length_over_cap, read_over_cap, in_memory_over_cap |
| header_version | APR v2 (2.0) | 2 | enforced | aprender-decide | Header | header_version_refused |
| metadata_size | max_metadata_bytes | 2 | enforced | aprender-decide | MetadataOverCap | metadata_over_cap |
| tensor_count | max_tensor_count | 2 | enforced | aprender-decide | TensorCountOverCap | tensor_count_over_cap |
| index_extent | count x 20 <= extent <= file | 2 | enforced | aprender-decide | IndexExtentTooSmall / IndexPastEnd | index_extent_too_small, index_past_end, rung2_predicate_exhaustive |
| index_reservation | min(count, index bytes / unit) | 3 | enforced | apr-format | (08-20) | forged_tensor_count_reserves_nothing_proportional, index_capacity_is_bounded_by_the_index_bytes |
| duplicate_names_reader | names strictly increasing | 3 | enforced | apr-format | (08-20) | duplicate_tensor_names_are_refused_by_both_readers |
| duplicate_names_ladder | names unique (set walk) | 4 | enforced, reachable_via_load: false | aprender-decide | DuplicateTensor | repeated_name_walk_names_the_first_repeat, duplicate_tensor_name_is_refused_at_load |
| num_hidden_layers | 1024 | 4 | enforced | aprender-decide (core check) | ConfigBlob | untrusted_layer_counts_are_bounded_before_derivation, layer_count_is_capped_before_allocation |
| head_layers | 64 | 4 | enforced | aprender-decide | ConfigBlob | untrusted_layer_counts_are_bounded_before_derivation |
| activation_and_rope | gelu / default rope / no scaling | 4 | enforced | aprender-decide (core check) | ConfigBlob | sweep, unsupported_forward_semantics_are_refused |
| encoder_dims | > 0, products fit usize, exact stored shapes | 6 | enforced | aprender-core | (core) | config_domain, refuses_shape_mismatch |
| tensor_dtype | F16 / F32 temperature / U8 blobs | 4 | enforced | aprender-decide | DtypeMismatch | sweep |
| load_tensor_dtype | F16/F32, exact payload | 6 | enforced | aprender-core | (ea940faec) | refuses_foreign_dtype_and_truncated_payload |
| tensor_entry_size | size == product(shape) x width | 4 | enforced | aprender-decide | SizeMismatch | size_mismatch, size_rule_exhaustive |
| tensor_data_range | data inside the file | 4 | enforced | aprender-decide | DataOutOfBounds | sweep (index offset patched in place) |
| tensor_name_set | derived set; O(expected x observed) bounded by both layer caps and max_tensor_count | 4 | enforced | aprender-decide | MissingTensor / UnexpectedTensor | missing_tensor, extra_tensor |
| non_finite | finite weights | 5 | enforced | aprender-decide | NonFiniteWeight | nan_weight |
| criteria_count | max_criteria (510) | 4 | enforced | aprender-decide | Task(TooManyCriteria) | too_many_criteria_refused_while_reading, max_criteria_matches_contract |
| probe_row_tokens | probe_max_row_tokens (48), before any forward | 7 | enforced | aprender-decide | ProbeMismatch (load) / ProbeRowOverBudget (pack) | probe_row_budget_checked_before_replay_forward, probe_row_over_budget |
| tokenizer_truncation_padding | none honoured | 6 | enforced (override) | aprender-decide | none: decisions identical | tokenizer_truncation_and_padding_are_disabled, sweep |
| inspect_read | max_artifact_bytes | pack_laya inspect | enforced | aprender-decide | ArtifactTooLarge | declared_length_over_cap, sweep (source check of cmd_inspect) |
| agent_max_len | >= 1, no upper bound | 4 | accepted | aprender-decide | a 2^40 max_len loads; rows bounded by probe_row_tokens / built_tokens_total | sweep |
| agent_head_max_len | none | 4 | accepted | aprender-decide | a 2^40 head_max_len loads; saturating arithmetic only | sweep |
| encoder_row_length | the caller's row (core API) | 7 | accepted | aprender-decide | every built row cut at max_len | sweep |
| layer_row_length | l >= 1, x.len() == l x d before RoPE | 7 | enforced | aprender-core | (08-20) | layer_forward_empty_row_is_refused, layer_forward_huge_l_is_refused_before_rope, layer_forward_with_rope_empty_row_is_refused |
| token_ids | id < vocab_size | 7 | enforced | aprender-core | OutOfVocab (ProbeReplay at load) | forward_refuses_out_of_vocab_and_empty, probe_replay_failure_is_rung_7 |
| config_blob_parse | blob bytes; recursion limit 128 | 4 | accepted | aprender-decide | 200-deep value in rope_scaling -> ConfigBlob "recursion limit"; under an ignored key it is skipped without recursion and loads | sweep |
| tokenizer_pipeline | blob bytes; pipeline unbounded by the ladder | 6 | accepted | aprender-decide | the gate pins the base tokenizer; rung 4 binds inputs_sha256.tokenizer_json to sha256:tokenizer.blob | sweep |
| run_dir_files | none on inputs; packed <= cap | pack | accepted | aprender-decide | pack refuses `packed` over the cap | sweep |

The request half (frame, argument shape, text count and bytes, built-token budget) is in decide-tool-boundary-v1 `untrusted_input_bounds`, from plans 08-23 and 08-24. The `see_also` key points to it.

## Mutation table

Each mutant was applied alone by `scratchpad/mutate.py`, restored from a byte-exact backup, and checked with `filecmp` (`restored byte-identical: True` for every mutant).

| # | Mutant | Test that went RED | How |
|---|---|---|---|
| M1 | `first_repeated_tensor_name` returns `None` | repeated_name_walk_names_the_first_repeat (+ sweep) | `adjacent repeat: left: None` |
| M2 | MAX_CRITERIA check removed AND HashSet reverted to the linear scan | too_many_criteria_refused_while_reading | **timing**: `a 100 000-criteria task took 72.011s to refuse` (bound 1 s) |
| M2a | MAX_CRITERIA check removed, HashSet kept | too_many_criteria_refused_while_reading (+ sweep) | `100 000 criteria refused` got Ok; sweep: criteria_count refused as LabelsDisagreeWithTask |
| M3 | truncation / padding disable removed | tokenizer_truncation_and_padding_are_disabled (+ sweep) | padded 64-token row vs 25; sweep: the forged artifact failed replay (ProbeMismatch tokens) |
| M4 | rung-2 version check removed | header_version_refused (+ sweep) | the ladder accepted a (3,0) header |
| M5 | load replay failure mapped back to Rebuild | probe_replay_failure_is_rung_7 | `rung 6 rebuild: ... out of vocabulary` |
| M6 | pre-forward probe row budget removed | probe_row_budget_checked_before_replay_forward (+ sweep) | `reached 2 forward pass(es) before the budget refused it` |
| X1 (not a row) | rung-4 duplicate CALL removed | duplicate_tensor_name_is_refused_at_load **stays green** | Expected. 08-20's reader refuses at rung 3 first, so the rung-4 call is defence in depth and is not observable at load. The helper's logic is covered by M1 |
| X2 | `args_os` back to `std::env::args` | non-UTF-8 argv on the release binary | exit **101** (panic `called Result::unwrap() on an Err value: "x\xFFy"`); restored build exits 2 |
| S1 | a row's test renamed to a missing fn | artifact_bounds_table_is_swept | `header_version: aprender-decide: no fn header_version_refused_nowhere` |
| S2 | an owned row id renamed | artifact_bounds_table_is_swept | `tensor_dtype_renamed: owned row without a hostile case` |

## Task Commits

1. **Task 1 (tracer): repeated tensor name refused; inspect bound and bounded:** `38824c6e1` (fix)
2. **Task 2: criteria bound, tokenizer disable, rung-2 version, rung-7 replay:** `a986f2230` (fix)
3. **Task 3: bounds table and sweep, pre-forward probe budget, mutation proof:** `161909525` (test)

**Plan metadata:** the docs commit that follows this SUMMARY.

## Verification evidence

- `cargo test -p aprender-decide --lib`: 157 passed. Under `--features serde-preserve-order`: 157 passed. `task::tests::stance_order_none_against_favor` passes under both backings; the preserve_order leg printed `preserve_order=ON`.
- `cargo test -p aprender-mcp-decide -p aprender-mcp-decide-lambda --lib`: 31 + 42 passed. `cargo check` of both servers `--all-targets`: rc 0.
- `pv validate contracts/decide-apr-v1.yaml`: `0 error(s), 0 warning(s)`. `make contract-audit-phase8`: rc 0.
- `cargo clippy -p aprender-decide --all-targets --no-deps -- -D warnings`: rc 0. `cargo fmt -p aprender-decide -- --check`: rc 0.
- `just laya-inspect` on the deployed file ran on a release binary rebuilt from this tree (the log shows `Compiling aprender-decide`). Output: rc 0, `artifact_sha256` 24a44d7e050166c9b64e2716f2bcb3ce91747f7a3b927d03d6eeae5f89b6275a, `base` laya-en-root@55cf4c4e. A non-UTF-8 argument exits 2.
- `heavy just laya-verify` on the deployed file with its run dir, data dir and base 55cf4c4e: rc 0, `"deploy_eligible":true`, `"shipped_seed":17`, argmax 459/459, recomputed margin 0.2217, ece_post 0.0443. The file's shasum is unchanged at 24a44d7e.
- Armed real-weights `laya_parity` under `heavy`, with `LAYA_LADDER_BIN` set to the main checkout's spike-025 ladder: rc 0.
  - `MEASURED ids 14/14 argmax 14/14 truncated 1`
  - `MEASURED probs max_abs 3.841e-6 bar 1e-5`
  - the ladder lines
  - 0 lines containing SKIP
  - `just _laya-leg-verdict laya_parity` printed `LEG OK: laya_parity`
- The golden `laya_tiny.apr.sha256` is unchanged (`git diff --quiet`), and `golden_sha` passes under both backings.
- Tracer gate (interactive, end-of-phase, automated-only verify): re-ran green before expanding to Task 2.

## Decisions Made

See `key-decisions`. The main ones are the value of max_criteria (derived and measured, not the plan's 128) and the pre-forward probe budget, which the enumeration turned up.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Claim honesty] max_criteria is 510, not max_len / 4 = 128**
- **Found during:** Task 2
- **Issue:** The plan derived "every option keeps at least 4 tokens after the shrink, so K > 512/4 = 128 always loses a marker". The shrink cuts options DOWN to `max(4, ...)` but never lengthens them, so a one-token name keeps 2 tokens (`[MASK]` plus the name). I replicated the builder in Python on Laya-en's tokenizer (snapshot 55cf4c4e, max_len 512, head_max_len 192):
  - one-token names keep every marker up to K = 253 and lose one from K = 254;
  - options of 4 or more tokens keep every marker up to K = 127.
  A bound of 128 would therefore refuse K = 129..253 tasks that Laya serves.
- **Fix:** The bound is `max_len - 2 = 510`. Every marker occupies its own row position after `[CLS]` and the head's `[SEP]`, so K > 510 loses a marker under any tokenizer. The derivation and the measurements are in the contract comment.
- **Files modified:** crates/aprender-decide/src/task.rs, contracts/decide-apr-v1.yaml
- **Committed in:** a986f2230

**2. [Rule 2 - Missing bound] Probe rows are checked before the load-time forward**
- **Found during:** Task 3 (enumerating `agent_max_len`)
- **Issue:** `max_len` is bounded only below. The load-time replay, the only forward a load runs, was sized by `min(max_len, tokenized length)`. The artifact's own tokenizer can make the tokenized length large, and `probe_max_row_tokens` was checked only after the forward.
- **Fix:** `run_probes` builds every probe row and refuses an over-budget one before `classify_for_task` (pack: `ProbeRowOverBudget`; load: `ProbeMismatch{tokens}`). A `#[cfg(test)]` `laya::FORWARD_ROWS` counter proves it; mutant M6 goes RED. `check_probe_budget` was folded into `run_probes`.
- **Files modified:** crates/aprender-decide/src/artifact.rs, crates/aprender-decide/src/laya/mod.rs, crates/aprender-decide/src/artifact/ladder.rs
- **Committed in:** 161909525

**3. [Rule 3 - Test mechanism] The rung-7 test uses an out-of-vocabulary tokenizer forgery**
- **Found during:** Task 2
- **Issue:** On the tiny tokenizer, `"choice question: "` and `"choice question: probe"` are both 4 tokens (`Ġ` versus `Ġprobe`), so no max_len can split the served task from the probe task.
- **Fix:** A coherent forgery remaps one token of probe input 0 to `vocab_size + 1000`, with the tokenizer digests re-pinned. Rung 6 only tokenizes the served prefix, so it rebuilds; the replay is the first forward, and core refuses `OutOfVocab`. This is still a real crafted artifact, not a unit test of the mapping function, and a control with the id left alone loads.
- **Committed in:** a986f2230

**4. [Rule 3 - Tool verdict] decide-apr-v1 3.0.0**
- `pv diff` against the plan base suggested `major`: the `task_order_is_label_index` invariants gained the max_criteria refusal. The bump follows the 08-19 precedent. Nothing pins the version.

**5. [Scope - small] Files outside `files_modified`**
- `justfile`: the laya-inspect comment said "bounded/header/manifest rungs", which would now be false.
- `laya/mod.rs`: the test-only forward counter.

**6. [Test theater avoided] Timing asserted before the result**
- In the first M2 run, the 100k timing check sat after the MAX+1 assertion, so the mutant went RED on the result and the clock was never exercised. The 100k block now runs first. Under M2 the clock fires at 72 s.
- **Committed in:** 161909525

---

**Total deviations:** 6 (1 claim correction, 1 missing bound added, 1 test mechanism, 1 tool-directed version, 1 minor scope, 1 test-order fix).
**Impact on plan:** Every must-have truth holds. The criteria value and the version number differ from the plan text because a measurement and `pv`, respectively, say otherwise.

## Issues Encountered

- The first contract edit contained an unescaped `'` inside a single-quoted YAML string. `pv validate` had run before that edit; the sweep's YAML parse caught it, and the quote was escaped.
- serde_json skips an ignored value iteratively, so a 200-deep value under an unknown key does not hit the recursion limit: the artifact loads. The `config_blob_parse` row asserts both behaviours: skipped without recursion, and a typed recursion-limit refusal inside `rope_scaling`.

## Threat Flags

| Flag | File | Description |
|------|------|-------------|
| threat_flag: accepted-surface | contracts/decide-apr-v1.yaml (tokenizer_pipeline) | The ladder does not bound what a tokenizer.json pipeline does: a normalizer can declare an expanding replacement. Only a VERIFIED artifact is bound to Laya-en's pinned tokenizer. A server that loads an unverified artifact runs whatever tokenizer it carries. Bounding it would need an allowlist of pipeline components, which is a design decision. |

## Known Stubs

None.

## User Setup Required

None.

## Next Phase Readiness

- Plan 08-31's ledger should add rows for the six external `untrusted_input_bounds` rows (apr-format x2, aprender-core x4) so that `cargo test -- --list` proves their tests run. This plan's sweep checks only that the fn exists.
- `tokenizer_pipeline` is accepted and flagged above. If unverified artifacts are ever served, it needs an owner decision (a pipeline allowlist).

## Self-Check: PASSED

- FOUND: every modified file listed in key-files exists.
- FOUND commits 38824c6e1, a986f2230 and 161909525 in `git log`. `git rev-list --count 6eb5453c1..HEAD` was 3 before this SUMMARY's commit.
- `DuplicateTensor`, `MAX_CRITERIA`, `with_truncation`, `fn artifact_bounds_table_is_swept` and `untrusted_input_bounds` are each present in their files.

---
*Phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server*
*Completed: 2026-09-28*
