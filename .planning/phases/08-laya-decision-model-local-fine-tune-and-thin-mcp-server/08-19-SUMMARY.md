---
phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
plan: 19
subsystem: artifact-load-ladder
tags: [decide-apr-v1, provenance, rung-4, manifest-bindings, CR-01, class-A, mutation-proof]

requires:
  - phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
    provides: "08-05 load ladder (rungs 1-8), 08-09 verify_path, 08-12 contract audit, ea940faec verify-side ArtifactNotFromRun"
provides:
  - "ArtifactError::ManifestDisagreesWithBlob { field } (rung 4 structural)"
  - "check_manifest_bindings: rung 4 (e), 23 comparisons binding every manifest leaf to a sha-bound source on every load door"
  - "laya::temperature::applied_temperature_f64: one lookup-and-clamp shared by the forward and rung 4"
  - "decide-apr-v1 2.0.0 manifest.bindings table (37 leaf patterns, 53 concrete leaves, 59 bindings) + FALSIFY-DECIDE-APR-013"
  - "a coherent-forgery test helper and a leaf sweep that breaks exactly one binding per forgery"
affects: [08-21, 08-26, 08-27, 08-31, aprender-mcp-decide, aprender-mcp-decide-lambda]

actuals:
  tokens: 12889
  tasks: 3
  commits: 2
plan_head_before: d6d3157c7b7977c3a34ccbf0c23d78bea91f7cfe

tech-stack:
  added: []
  patterns:
    - "Binding graph: manifest leaves and blob fields are settable nodes, blob digests and derived values are fixed; a forgery that breaks exactly one edge moves the leaf's component as one"
    - "One comparison per checks-array entry, so each can be disabled on its own for mutation proof"

key-files:
  created: []
  modified:
    - crates/aprender-decide/src/artifact.rs
    - crates/aprender-decide/src/artifact/ladder.rs
    - crates/aprender-decide/src/laya/temperature.rs
    - contracts/decide-apr-v1.yaml

key-decisions:
  - "decide-apr-v1 bumped 1.0.0 -> 2.0.0, not the plan's 1.1.0: pv diff reports Suggested bump major because the blob_integrity formula changed, and 08-13 set the precedent of applying pv's suggestion"
  - "verify_checks[2] states what check_base enforces TODAY (recipe base.sha256 vs the gate pin, base dir model hash, base tokenizer hash) and marks field-by-field base equality as plan 08-21, NOT checked yet"
  - "The sweep breaks exactly ONE binding per forgery (not just the manifest leaf), because a leaf bound to two sources would otherwise let either comparison be deleted with the sweep still green"
  - "check_manifest_bindings runs at the END of rung 4, after the labels check (d), because it needs the parsed task (K) and agent config. It is still inside rung 4 and still before rung 5"

patterns-established:
  - "Coherent forgery: a forger controls every blob and hash, so a negative re-pins the recipe digest (blobs, recipe_id, report recipe_id) and the report digest (blobs, gate.report_sha256) before the explicit manifest edit"
  - "Contract-driven sweep: the contract table is the list of leaves, and the test fails on an unbound leaf, a stale row, an unknown source kind or an unresolvable recipe/report pointer"

requirements-completed: [D-04, D-11, D-17]

coverage:
  - id: D1
    description: "A manifest whose base or variant disagrees with its embedded recipe blob is refused at rung 4 by load_bytes, load_hashed (Lambda) and load_path (stdio)"
    requirement: D-11
    verification:
      - kind: unit
        ref: "crates/aprender-decide/src/artifact/ladder.rs#manifest_base_disagrees_with_recipe_blob"
        status: pass
      - kind: unit
        ref: "crates/aprender-decide/src/artifact/ladder.rs#manifest_bound_at_every_load_door"
        status: pass
    human_judgment: false
  - id: D2
    description: "Every manifest leaf is bound at load: for all 59 (leaf, source) bindings over 53 leaves, a forgery breaking exactly that one is refused at the table's rung, and the table's leaf set equals the manifest's"
    requirement: D-17
    verification:
      - kind: unit
        ref: "crates/aprender-decide/src/artifact/ladder.rs#every_manifest_leaf_is_bound"
        status: pass
      - kind: unit
        ref: "crates/aprender-decide/src/artifact/ladder.rs#manifest_bindings_table_matches_manifest_leaves"
        status: pass
    human_judgment: false
  - id: D3
    description: "calibration.t_applied equals, bit for bit, the temperature the loaded model applies (applied_temperature_f64 shared by rung 4 and temperature_for)"
    requirement: D-04
    verification:
      - kind: unit
        ref: "crates/aprender-decide/src/artifact/ladder.rs#manifest_t_applied_disagrees_with_agent"
        status: pass
      - kind: unit
        ref: "crates/aprender-decide/src/laya/temperature.rs#lookup_falls_back_in_laya_order"
        status: pass
    human_judgment: false
  - id: D4
    description: "Each of the 23 comparisons is mutation-verified: disabling it turns the sweep RED naming exactly its (leaf, source)"
    verification:
      - kind: other
        ref: "scratch mutate.py: 23 single-comparison disables, each run of cargo test -p aprender-decide --lib artifact::ladder::every_manifest_leaf_is_bound (table below)"
        status: pass
    human_judgment: false
  - id: D5
    description: "The deployed artifact 24a44d7e... still loads (real stdio leg) and still verifies deploy_eligible true with shipped_seed 17; the golden and the ea940faec verify-side tests hold"
    requirement: D-11
    verification:
      - kind: e2e
        ref: "crates/aprender-mcp-decide/tests/e2e_stdio.rs#a_real_decide_model_classifies_over_live_stdio (APR_MCP_E2E_DECIDE_MODEL=models/decide/laya-stance-64.apr, release)"
        status: pass
      - kind: e2e
        ref: "just laya-verify models/decide/laya-stance-64.apr models/decide/laya-stance-64 data/decide/tweet-stance-64 <base snapshot 55cf4c4e>"
        status: pass
      - kind: unit
        ref: "crates/aprender-decide/src/verify/tests.rs#verify_path_refuses_bytes_the_run_does_not_pack_to"
        status: pass
      - kind: unit
        ref: "crates/aprender-decide/src/artifact/determinism.rs#golden_sha"
        status: pass
    human_judgment: false

duration: 24min
completed: 2026-09-28
status: complete
---

# Phase 8 Plan 19: Manifest-to-Blob Provenance Binding (CR-01 load side) Summary

**Rung 4 now binds all 53 leaves of the decide-apr-v1 manifest to their sha-bound sources on every load door.** The sources are the recipe blob, the gate-report blob, the blob digests, the task's bucket and the agent config's applied temperature. A forged `model.base` or calibration record can no longer load through `load_bytes`, `load_hashed` (the Lambda) or `load_path` (stdio). The contract table drives a sweep that forges each of the 59 bindings alone. All 23 comparisons are mutation-verified, and the deployed 24a44d7e artifact still loads and still verifies as eligible.

## Performance

- **Duration:** 24 min
- **Started:** 2026-09-28T15:33:27Z
- **Completed:** 2026-09-28T15:57:03Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- `ArtifactError::ManifestDisagreesWithBlob { field }` (rung "4 structural") and `check_manifest_bindings`. The function runs as rung 4 (e) and makes 23 comparisons.
  - It reuses the blob digests that rung 4 (a) computed. Each blob is hashed once; the recipe was hashed twice before this plan.
- `laya::temperature::applied_temperature_f64` is the single lookup-and-clamp. `temperature_for` now just casts it, so the forward and rung 4 share one definition. The temperature tests are unchanged and green.
- decide-apr-v1 2.0.0 gains:
  - a full `manifest.bindings` table (below);
  - a rung-4 rule that names the bindings;
  - the `blob_integrity` formula extended with `forall leaf f in manifest.bindings: manifest[f] == source(f)`;
  - `verify_checks[2]` rewritten to what the code enforces;
  - FALSIFY-DECIDE-APR-013.
- The ea940faec verify-side binding still holds: `verify_path_refuses_bytes_the_run_does_not_pack_to` and `verify_path_accepts_exact_file` pass.

## Task Commits

1. **Task 1 (tracer): manifest.base and manifest.variant bound to the recipe blob, refused at every load door**, `b2e66dc7f` (feat)
2. **Task 2: bind every remaining manifest leaf, and sweep all leaves against the contract table**, `602db2eda` (feat, TDD: RED observed before the comparisons were added)
3. **Task 3: mutation proof per cross-check, and the deployed artifact still loads and verifies eligible.** No commit: verification only. `artifact.rs` is byte-identical to its Task-2 state (`git diff --quiet HEAD -- crates/ contracts/`).

**Plan metadata:** the docs commit that follows this SUMMARY.

## Class-A load-side surface: `manifest.bindings`, one row per leaf pattern

The sweep ran over 53 concrete leaves (arrays expanded: 3 labels, 6 blobs, 2 probes with 3 probabilities each) and 59 bindings.

| Leaf pattern | Rung | Sources (each one a separately forged binding) |
|---|---|---|
| `/schema` | 3 | rung-3 constant |
| `/schema_version` | 3 | rung-3 constant |
| `/method` | 3 | rung-3 constant |
| `/variant` | 4 | `recipe:/variant` |
| `/base/family` | 4 | `recipe:/base/family` |
| `/base/checkpoint` | 4 | `recipe:/base/checkpoint` |
| `/base/repo` | 4 | `recipe:/base/repo` |
| `/base/revision` | 4 | `recipe:/base/revision` |
| `/base/sha256` | 4 | `recipe:/base/sha256`, `report:/inputs_sha256/base_model` |
| `/labels/[*]` | 4 | task blob criteria in document order (existing check (d)) |
| `/agent/head_layers` | 4 | agent config blob (existing AgentMismatch) |
| `/agent/max_len` | 4 | agent config blob |
| `/agent/head_max_len` | 4 | agent config blob |
| `/blobs/[*]/name` | 4 | blob_tensors order (existing ManifestBlobs) |
| `/blobs/[*]/sha256` | 4 | sha256 of the named blob (existing BlobHashMismatch) |
| `/recipe_id` | 4 | `sha256:decide.recipe_json` (existing RecipeIdMismatch), `report:/recipe_id` |
| `/calibration/bucket` | 4 | `report:/calibration/bucket`, `bucket_key(choice, K of the task blob)` |
| `/calibration/t_fitted` | 4 | `report:/calibration/t_fitted` |
| `/calibration/t_applied` | 4 | `report:/calibration/t_applied`, `applied_temperature_f64(agent config, choice, K)` (bits) |
| `/calibration/clamp_hit` | 4 | `report:/calibration/clamp_hit` |
| `/calibration/slice_ids_sha256` | 4 | `report:/calibration/slice_ids_sha256` |
| `/gate/pass` | 4 | `report:/pass` |
| `/gate/margin` | 4 | `report:/margin` (bits) |
| `/gate/ece_post` | 4 | `report:/fine_tuned/ece_post` (bits) |
| `/gate/report_sha256` | 4 | `sha256:decide.gate_report_json` |
| `/inputs_sha256/task_json` | 4 | `sha256:decide.task_json`, `report:/inputs_sha256/task_json` |
| `/inputs_sha256/train_jsonl` | 4 | `report:/inputs_sha256/train_jsonl` |
| `/inputs_sha256/eval_jsonl` | 4 | `report:/inputs_sha256/eval_jsonl` |
| `/inputs_sha256/base_model` | 4 | `report:/inputs_sha256/base_model` |
| `/inputs_sha256/tokenizer_json` | 4 | `sha256:tokenizer.blob`, `report:/inputs_sha256/tokenizer_json` |
| `/device_used` | 4 | `report:/device_used` |
| `/probes/[*]/input_index` | 7 | rung-7 replay |
| `/probes/[*]/tokens` | 7 | rung-7 replay |
| `/probes/[*]/label` | 7 | rung-7 replay |
| `/probes/[*]/probabilities_f32_hex/[*]` | 7 | rung-7 replay (mutation moves the probability by at least 0.25) |

## TDD RED evidence (Task 2)

Before any Task-2 comparison existed, with the full table already in the contract:
- 11 of the new `manifest_*` negatives FAILED.
- The three that passed are the controls and Task 1's tests: `manifest_forge_control_loads`, `manifest_base_disagrees_with_recipe_blob` and `manifest_bound_at_every_load_door`. `manifest_bindings_table_matches_manifest_leaves` also passed.
- `every_manifest_leaf_is_bound` FAILED and listed exactly the 21 unbound bindings, each as `<leaf> <-> <source>: LOADED after the binding was broken`. These were:
  - `/base/sha256 <-> report`
  - all five calibration-vs-report bindings, plus the two derived bucket and t_applied bindings
  - `/device_used`
  - the gate triple
  - `/gate/report_sha256`
  - the 7 `inputs_sha256` bindings
  - `/recipe_id <-> report`

Every pre-existing binding (rung 3, base and variant, labels, agent, blobs, recipe digest, probes) was already refused at its table rung. After the comparisons were added, `cargo test -p aprender-decide --lib` gave 134 passed.

## Mutation proof (Task 3)

Each comparison in `check_manifest_bindings` was replaced with `true` on its own. The script then ran `cargo test -p aprender-decide --lib artifact::ladder::every_manifest_leaf_is_bound` and restored the file.

| # | Comparison disabled | Sweep result | Binding the sweep named (LOADED after the binding was broken) | Restored green |
|---|---|---|---|---|
| 0 | variant == recipe.variant | FAILED rc 101 | `/variant <-> recipe:/variant` | yes |
| 1 | base == recipe.base | FAILED rc 101 | `/base/{checkpoint,family,repo,revision,sha256} <-> recipe:/base/*` (5) | yes |
| 2 | gate.report_sha256 == report blob digest | FAILED rc 101 | `/gate/report_sha256 <-> sha256:decide.gate_report_json` | yes |
| 3 | inputs.task_json == task blob digest | FAILED rc 101 | `/inputs_sha256/task_json <-> sha256:decide.task_json` | yes |
| 4 | inputs.tokenizer_json == tokenizer blob digest | FAILED rc 101 | `/inputs_sha256/tokenizer_json <-> sha256:tokenizer.blob` | yes |
| 5 | inputs.task_json == report | FAILED rc 101 | `/inputs_sha256/task_json <-> report:/inputs_sha256/task_json` | yes |
| 6 | inputs.train_jsonl == report | FAILED rc 101 | `/inputs_sha256/train_jsonl <-> report:/inputs_sha256/train_jsonl` | yes |
| 7 | inputs.eval_jsonl == report | FAILED rc 101 | `/inputs_sha256/eval_jsonl <-> report:/inputs_sha256/eval_jsonl` | yes |
| 8 | inputs.base_model == report | FAILED rc 101 | `/inputs_sha256/base_model <-> report:/inputs_sha256/base_model` | yes |
| 9 | inputs.tokenizer_json == report | FAILED rc 101 | `/inputs_sha256/tokenizer_json <-> report:/inputs_sha256/tokenizer_json` | yes |
| 10 | base.sha256 == report inputs base_model | FAILED rc 101 | `/base/sha256 <-> report:/inputs_sha256/base_model` | yes |
| 11 | recipe_id == report.recipe_id | FAILED rc 101 | `/recipe_id <-> report:/recipe_id` | yes |
| 12 | gate.pass == report.pass | FAILED rc 101 | `/gate/pass <-> report:/pass` | yes |
| 13 | gate.margin bits == report.margin | FAILED rc 101 | `/gate/margin <-> report:/margin` | yes |
| 14 | gate.ece_post bits == report.fine_tuned.ece_post | FAILED rc 101 | `/gate/ece_post <-> report:/fine_tuned/ece_post` | yes |
| 15 | calibration.bucket == report | FAILED rc 101 | `/calibration/bucket <-> report:/calibration/bucket` | yes |
| 16 | calibration.t_fitted bits == report | FAILED rc 101 | `/calibration/t_fitted <-> report:/calibration/t_fitted` | yes |
| 17 | calibration.t_applied bits == report | FAILED rc 101 | `/calibration/t_applied <-> report:/calibration/t_applied` | yes |
| 18 | calibration.clamp_hit == report | FAILED rc 101 | `/calibration/clamp_hit <-> report:/calibration/clamp_hit` | yes |
| 19 | calibration.slice_ids_sha256 == report | FAILED rc 101 | `/calibration/slice_ids_sha256 <-> report:/calibration/slice_ids_sha256` | yes |
| 20 | calibration.bucket == bucket_key(task) | FAILED rc 101 | `/calibration/bucket <-> derived:bucket_key(choice, K ...)` | yes |
| 21 | calibration.t_applied bits == applied_temperature_f64(agent) | FAILED rc 101 | `/calibration/t_applied <-> derived:applied_temperature_f64(...)` | yes |
| 22 | device_used == report.device_used | FAILED rc 101 | `/device_used <-> report:/device_used` | yes |

- Every disable went RED and named exactly its own binding and no other. No comparison is redundant under the sweep, so no sweep fix was needed.
- After the loop, `git diff --quiet -- crates/aprender-decide/src/artifact.rs` held, and the sweep re-ran `ok`.

## Deployed artifact (unchanged, no redeploy)

- **Real stdio leg.** Command: `heavy env APR_MCP_E2E_DECIDE_MODEL=$MAIN/models/decide/laya-stance-64.apr cargo test -p aprender-mcp-decide --release --test e2e_stdio a_real_decide_model_classifies_over_live_stdio -- --nocapture`. Result: rc 0 and 0 SKIP lines. It printed `E2E-DECIDE-PMCP-001 (real): .../laya-stance-64.apr identity "24a44d7e050166c9b64e2716f2bcb3ce91747f7a3b927d03d6eeae5f89b6275a"`.
  - Mechanism engaged: the server binary is pinned by `CARGO_BIN_EXE_aprender-mcp-decide`, and that run's build log shows `Compiling aprender-decide` and `Compiling aprender-mcp-decide` from the tree at `602db2eda`.
- **Verify.** Command: `heavy just laya-verify` on the exact file, with its run dir, data dir and base snapshot 55cf4c4e. Result: rc 0 after 209 s. Output: `"artifact_sha256":"24a44d7e050166c9b64e2716f2bcb3ce91747f7a3b927d03d6eeae5f89b6275a","deploy_eligible":true`, `"shipped_seed":17`, `argmax 459/459`, recomputed margin 0.2217, ece_post 0.0443.
- **File hash.** `shasum -a 256 models/decide/laya-stance-64.apr` = `24a44d7e050166c9b64e2716f2bcb3ce91747f7a3b927d03d6eeae5f89b6275a`.
- **Golden.** `laya_tiny.apr.sha256` is unchanged (`37d65159...`), and `golden_sha` passes under both serde_json backings (the `--features serde-preserve-order` artifact:: leg gave 52 passed).

## Other verification

- `cargo test -p aprender-mcp-decide -p aprender-mcp-decide-lambda --lib`: 28 + 31 passed (the server doors).
- `cargo clippy -p aprender-decide --all-targets --no-deps -- -D warnings`: rc 0. `cargo fmt -p aprender-decide -- --check`: rc 0.
- `pv validate contracts/decide-apr-v1.yaml`: `0 error(s), 0 warning(s)`. `make contract-audit-phase8`: rc 0.

## Files Created/Modified

- `crates/aprender-decide/src/artifact.rs`:
  - adds the error variant, the `BoundSources` struct, `same_f64` (bit equality, NaN never equal) and `check_manifest_bindings`;
  - rung 4 (a) keeps each blob digest;
  - updates the module docs.
- `crates/aprender-decide/src/laya/temperature.rs`: `applied_temperature_f64`; `temperature_for` casts it.
- `crates/aprender-decide/src/artifact/ladder.rs` adds:
  - the `forge` helper, which re-pins every author-controlled hash;
  - the binding-graph `break_only` forgery;
  - 12 new `manifest_*` tests (14 in all with Task 1's);
  - `every_manifest_leaf_is_bound` and `manifest_bindings_table_matches_manifest_leaves`.
- `contracts/decide-apr-v1.yaml`:
  - version 2.0.0;
  - `manifest.bindings`;
  - the rung-4 rule;
  - `verify_checks[2]`;
  - the `blob_integrity` formula and codomain;
  - FALSIFY-DECIDE-APR-013;
  - the qa_gate checks and `pass_criteria` 001..013.

## Decisions Made

See key-decisions in the frontmatter. The central design choice is that the sweep forges one binding at a time, not one leaf at a time. Five leaves have two sources:
- `base.sha256`
- `recipe_id`
- `calibration.bucket`
- `calibration.t_applied`
- `inputs_sha256.task_json` and `.tokenizer_json`

Mutating only the manifest leaf would let either source's comparison be deleted with the sweep still green. The plan's own mutation criterion rules this out.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Contract version 2.0.0 instead of 1.1.0**
- **Found during:** Task 2, step 4 ("confirm the bump with `pv diff`")
- **Issue:** `pv diff` of the HEAD copy against the edited contract reported `Suggested bump: major`. The plan-mandated `blob_integrity` formula extension counts as an equation change, which `pv` treats as major.
- **Fix:** Applied `pv`'s suggestion, as 08-13 did for laya-parity-v1 and laya-finetune-gate-v1. The re-diff shows `v1.0.0 → v2.0.0`. No code, test or later plan pins the decide-apr-v1 version. The artifact `schema_version` stays 1.
- **Files modified:** contracts/decide-apr-v1.yaml
- **Committed in:** 602db2eda

**2. [Rule 1 - Honesty, class D] verify_checks[2] wording**
- **Found during:** Task 2
- **Issue:** The plan's prescribed text said verify checks the recipe's base "field by field" against the gate contract's base block, and all base-dir files against its pins. Today `check_base` compares only `base.sha256` with the pin and hashes `model.safetensors`; `load_declared_base` also hashes the tokenizer. The prescribed text would have over-claimed.
- **Fix:** The row now states exactly those checks and marks the rest as plan 08-21, "NOT checked yet".
- **Files modified:** contracts/decide-apr-v1.yaml
- **Committed in:** 602db2eda

**3. [Minor] Bindings call placed at the end of rung 4**
- **Found during:** Task 1
- **Issue:** The plan said to call the bindings right after the recipe_id check. Task 2's comparisons need the parsed task (K) and agent config, which rung 4 (b) and (d) produce later.
- **Fix:** The call is rung 4 (e), after the labels check. It is still inside rung 4, after the blob hashes and before rung 5, which is the plan's key_link.
- **Committed in:** b2e66dc7f

**4. [Minor] Base rows per leaf from the start**
- **Found during:** Task 1
- **Issue:** The plan named a single `/base/*` row. The final table must list one row per leaf pattern.
- **Fix:** Task 1 wrote the five per-leaf base rows straight away, so Task 2 did not have to rewrite them.
- **Committed in:** b2e66dc7f

---

**Total deviations:** 4 (1 blocking tool verdict, 1 honesty correction, 2 structural).
**Impact on plan:** No scope creep. Every must-have truth holds. The version number and the verify_checks wording differ from the plan text because the tool and the code, respectively, say otherwise.

## Issues Encountered

- The rtk hook compacts `cargo test` output even when redirected to a file, which hides the per-test `... ok` lines. The verify blocks were re-run through `rtk proxy sh -c '...'`. The mutation script calls cargo from Python `subprocess`, which is unfiltered.

## Known Stubs

None.

## User Setup Required

None. No external service configuration is required.

## Next Phase Readiness

- The load side of CR-01 is closed for every consumer of a `Decider`.
- Plan 08-21 still owns field-by-field base equality at verify time. The contract text says so explicitly.
- Plans that add a manifest field must add a `manifest.bindings` row, or `manifest_bindings_table_matches_manifest_leaves` fails.

## Self-Check: PASSED

- FOUND crates/aprender-decide/src/artifact.rs, crates/aprender-decide/src/artifact/ladder.rs, crates/aprender-decide/src/laya/temperature.rs, contracts/decide-apr-v1.yaml
- FOUND commits b2e66dc7f, 602db2eda
- `ManifestDisagreesWithBlob` occurs 4 times in artifact.rs (variant, rung arm, Display, constructor); `fn applied_temperature_f64`, `fn every_manifest_leaf_is_bound` and FALSIFY-DECIDE-APR-013 are present
- `commits: 2` measured as `git rev-list --count d6d3157c7..HEAD` before this SUMMARY's commit

---
*Phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server*
*Completed: 2026-09-28*
