---
phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
plan: 05
subsystem: ml-models
tags: [aprender-decide, decide-apr-v1, apr-format, laya, artifact, load-ladder, determinism, trybuild, provable-contracts]

requires:
  - phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
    provides: "08-01 decide-apr-v1 (manifest, blob set, load ladder, probe policy, identity, determinism) and laya-finetune-gate-v1 (run_dir_layout, recipe/gate-report/probes schemas)"
  - phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
    provides: "08-02 laya_tiny run dir + data dir (F16 checkpoint, probes.json with Laya's own probe values, synthetic-fixture recipe, pass=false report)"
  - phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
    provides: "08-04 aprender-decide: DecisionMethod, Task::from_slice, Laya::from_parts, forward_row, temperature_for"
provides:
  - "aprender_decide::artifact: decide-apr-v1 constants (each asserted equal to the contract), Manifest/BaseDecl/AgentDecl/BlobHash/CalibrationDecl/GateSummary/InputsSha256/ProbeRecord, write_decide_apr, read_decide_apr_bytes_bounded, check_index_extent, inspect_manifest, artifact_sha256_hex, ArtifactLimits::CONTRACTED, ArtifactError (rung-named)"
  - "aprender_decide::pack: PackInputs::from_run_dir(run_dir, data_dir), pack_run_dir -> Vec<u8>, PackError, typed Recipe/GateReport/ProbesFile (deny_unknown_fields), sha256_hex"
  - "aprender_decide::{Decider, ModelIdentity}: private-field Decider minted only by the ladder (load_bytes, load_path); identity = whole-file sha256 + recipe_id + method + base display"
  - "Laya::classify_for_task: score another choice task (the probe task) with the loaded weights"
  - "apr-format v2 reader: index_capacity caps the tensor-index reservation at remaining/20 for every APR consumer"
  - "tests/fixtures/laya_tiny.apr.sha256 golden (37d65159...), blessed on aarch64"
  - "tests/ui.rs trybuild harness + decider_struct_literal / decider_no_constructor compile-fail cases"
affects: [08-06, 08-07, 08-09, 08-10, 08-12]

actuals:
  tokens: 35561   # chars/4 over the realized diff (142245 chars, git diff be9934028..HEAD)
  tasks: 3
  commits: 5
plan_head_before: be99340288cb8863973002c21ddcbb99cf78929d

tech-stack:
  added:
    - "safetensors (workspace 0.4) promoted from dev- to normal dependency of aprender-decide (pack path only)"
    - "trybuild (workspace 1) as an aprender-decide dev-dependency"
  patterns:
    - "One manifest string under one custom key, serialized from typed structs with arrays (never maps), so its bytes do not depend on the serde_json map backing"
    - "Probe expectations stored from probes.json (Python); the Rust probe is checked, never serialized"
    - "Rung 2 parses the 64-byte header and bounds tensor_count / index extent BEFORE AprV2ReaderRef::from_bytes"
    - "Negatives induced on packed bytes through a repack helper whose no-edit control is byte-identical and loads"
    - "Pack step and load rung share one implementation of each structural rule (expected_weights, check_name_set, check_size, all_finite, compare_probes)"

key-files:
  created:
    - crates/aprender-decide/src/artifact.rs
    - crates/aprender-decide/src/artifact/tests.rs
    - crates/aprender-decide/src/artifact/ladder.rs
    - crates/aprender-decide/src/artifact/determinism.rs
    - crates/aprender-decide/src/pack.rs
    - crates/aprender-decide/tests/fixtures/laya_tiny.apr.sha256
    - crates/aprender-decide/tests/ui.rs
    - crates/aprender-decide/tests/ui/decider_struct_literal.rs
    - crates/aprender-decide/tests/ui/decider_struct_literal.stderr
    - crates/aprender-decide/tests/ui/decider_no_constructor.rs
    - crates/aprender-decide/tests/ui/decider_no_constructor.stderr
  modified:
    - crates/aprender-decide/src/lib.rs
    - crates/aprender-decide/src/laya/mod.rs
    - crates/aprender-decide/src/test_support.rs
    - crates/aprender-decide/Cargo.toml
    - Cargo.lock
    - crates/apr-format/src/v2/reader_impl.rs
    - crates/apr-format/src/v2/tests.rs
    - contracts/decide-apr-v1.yaml

key-decisions:
  - "Probes run through Laya::classify_for_task on the ALREADY-LOADED weights, not a second Laya::from_parts: a second load would widen the ~0.85 GB F16 checkpoint to f32 twice (about 3.4 GB) at every server cold start"
  - "Pack step 4 builds Laya over a weights-only in-memory .apr before the container is written (the plan's order); nothing is returned until every step passes"
  - "manifest.blobs is an ARRAY of {name, sha256} and the manifest gains a `variant` field; decide-apr-v1 manifest.fields updated to match (contract_mirror asserts the field SET equals the contract's)"
  - "The rung-5 non-finite scan reads raw F16/F32 bit patterns (exponent all ones) with no widening; rung 6 widens once, through core"
  - "expected_bytes is exact: Some(0) whenever any dimension is 0, even after a dimension whose running product would overflow"
  - "The trybuild proof is TWO cases: in one file rustc stops at the E0599 type error and never runs the privacy pass, so E0451 was missing from the snapshot"
  - "binding.yaml is untouched (plan 08-12 flips Phase 8 rows); its decide-apr-v1 signatures still say VerifiedDecider/LoadError, while the code names are Decider/ArtifactError and read_decide_apr_bytes_bounded takes declared_len: Option<u64>"

patterns-established:
  - "Rung-named errors: ArtifactError::rung() returns '1 bounded_read' .. '7 probe_replay' or 'pack', and every negative test asserts the variant, not a string"
  - "Golden bless fails the run by design (DECIDE_BLESS_GOLDEN=1 exits 101), so a bless can never pass silently"

requirements-completed: [D-04, D-11, D-14, D-17]

coverage:
  - id: D1
    description: "A Laya run dir packs into one decide-apr-v1 .apr (F16 raw bytes, F32 temperature, six U8 blobs, one `decide` custom key, declared base and variant in the manifest) that loads only through the ladder and classifies the oracle's task rows within 2.98e-8 (bar 1e-5); identity == whole-file sha256, recipe_id == sha256(recipe.json), labels in task.json order"
    requirement: "D-17"
    verification:
      - kind: unit
        ref: "crates/aprender-decide/src/artifact/tests.rs#tiny_roundtrip"
        status: pass
      - kind: unit
        ref: "crates/aprender-decide/src/artifact/tests.rs#pack_unpack_closure"
        status: pass
      - kind: unit
        ref: "crates/aprender-decide/src/artifact/tests.rs#load_path_door"
        status: pass
    human_judgment: false
  - id: D2
    description: "Every ladder rung refuses its own induced negative, naming the rung (26 artifact::ladder tests: bounded read x3, header CRC, tensor_count cap, index extent, index past end, rung-2 predicate exhaustive, model_type, custom keys, unknown manifest key, schema_version, method, missing/extra tensor, size, size rule exhaustive, tokenizer/task/recipe blob hash, labels, NaN weight, probe replay, pack-time probe budget and oracle refusals, repack control)"
    requirement: "D-17"
    verification:
      - kind: unit
        ref: "cargo test -p aprender-decide --lib artifact::ladder (26 ok)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Byte determinism: two packs in one process identical; golden sha 37d65159b2be0fa091aa840cd56c1a84b73c0bcd9e2df5906d1f8218f5448561 reproduced standalone (preserve_order=OFF observed) and with -p aprender-mcp-setfit (preserve_order=ON observed); manifest probe values == probes.json byte for byte"
    requirement: "D-11"
    verification:
      - kind: unit
        ref: "cargo test -p aprender-decide --lib artifact::determinism -- --nocapture (OFF)"
        status: pass
      - kind: unit
        ref: "cargo test -p aprender-decide -p aprender-mcp-setfit --lib artifact::determinism -- --nocapture (ON)"
        status: pass
    human_judgment: false
  - id: D4
    description: "A forged header cannot make any APR reader allocate for its declared tensor_count: apr-format index_capacity = min(count, remaining/20); both readers refuse a u32::MAX count over a small file with InvalidTensorIndex; decide rung 2 refuses 4097 and an over-extent count before the reader runs"
    requirement: "D-17"
    verification:
      - kind: unit
        ref: "crates/apr-format/src/v2/tests.rs#index_capacity_is_bounded_by_the_index_bytes"
        status: pass
      - kind: unit
        ref: "crates/apr-format/src/v2/tests.rs#forged_tensor_count_is_refused_by_both_readers"
        status: pass
      - kind: unit
        ref: "crates/aprender-decide/src/artifact/ladder.rs#tensor_count_over_cap, #index_extent_too_small"
        status: pass
    human_judgment: false
  - id: D5
    description: "Decider has no second minting path: a struct literal fails with E0451 on all three private fields, Decider::new fails with E0599 pointing at load_bytes/load_path; making the fields pub turns the harness red"
    requirement: "D-14"
    verification:
      - kind: unit
        ref: "cargo test -p aprender-decide --test ui (decider_has_no_second_minting_path)"
        status: pass
      - kind: other
        ref: "cargo clippy -p aprender-decide --all-targets --no-deps -- -D warnings"
        status: pass
    human_judgment: false
  - id: D6
    description: "Contract mirror: every artifact constant, the probe inputs and probe task, the blob set, model_type, the single key and the manifest field set equal decide-apr-v1; the contract binds FALSIFY-001..006/008/009/011 to concrete test fns (0 dangling on the lifted guard, 622 refs)"
    requirement: "D-04"
    verification:
      - kind: unit
        ref: "crates/aprender-decide/src/artifact/tests.rs#contract_mirror"
        status: pass
      - kind: other
        ref: "pv validate contracts/decide-apr-v1.yaml -> 0 errors, 0 warnings"
        status: pass
      - kind: other
        ref: "bash scripts/check_contract_test_binding.sh -> VACUOUS (pre-existing D-ITEM-08-01-A); measured on a lifted copy instead"
        status: unknown
    human_judgment: true
    rationale: "The plan's strict-binding PASS line cannot be printed while spectral-indices-v1 (another phase) makes the guard skip; the lifted-copy measurement is evidence, not the gate itself"

duration: 38min
completed: 2026-09-26
status: complete
---

# Phase 8 Plan 05: decide-apr-v1 Artifact, Packer and Ladder-only Decider Summary

**A Laya run dir now packs into one deterministic `decide-apr-v1` `.apr`: raw F16 weights, six hashed U8 blobs, and one typed manifest string that carries the declared base, the variant and the Python-recorded probe values. The only way to a `Decider` is an eight-rung load ladder, and every rung is proven to refuse its own induced negative. The golden hash `37d65159…` reproduces under both serde_json backings, and apr-format no longer reserves index memory for a forged `tensor_count`. The same code also packs and loads the real 846 MB Laya-en run dir: the derived tensor set is an exact bijection with its 206 tensors, and the probes agree within 1.26e-6.**

## Performance

- **Duration:** 38 min
- **Started:** 2026-09-26T02:09:41Z
- **Completed:** 2026-09-26T02:48:01Z
- **Tasks:** 3 (Task 1 tracer, Task 2 TDD expansion, Task 3 trybuild + lint)
- **Files modified:** 19 (11 created, 8 modified)

## Accomplishments

- **The tracer passed on its first run** (aarch64, Apple M4). The path is `pack_run_dir(laya_tiny, laya_tiny/data)`, then 201,412 bytes, then `Decider::load_bytes`, then `classify`. On the 5 oracle task rows the maximum probability difference is **2.98e-8** (bar 1e-5), and every argmax matches. The identity equals `sha256` of the bytes, `recipe_id` equals `sha256(recipe.json)`, `base` is `laya-tiny-synthetic@fixtures`, the labels are `[shipping, billing, account]` (document order, not sorted), the variant is `synthetic-fixture`, and the stored probes equal `probes.json`.
- **The ladder has 8 rungs, each with its own typed refusal.** `ArtifactError::rung()` names the rung for every variant. There are 26 `artifact::ladder` tests, 7 more than the ≥ 19 the plan requires. They include a **repack control**: an unedited re-write of the artifact is byte-identical and loads, so every negative is caused by its own edit.
- **Rung 2 runs before the reader.** It parses the header itself, checks the CRC and the column-major flag, and then applies the pure `check_index_extent`. `tensor_count_over_cap` shows that `AprV2ReaderRef::from_bytes` would have failed differently on the same bytes, which proves the refusal came from rung 2.
- **The apr-format hardening applies to every APR consumer.** `index_capacity(count, remaining) = min(count, remaining / 20)` is now the only capacity the index vector gets, and `with_capacity(tensor_count as usize)` no longer appears anywhere. Both readers refuse a u32::MAX count over a small file with a valid CRC.
- **Determinism was observed, not assumed.** Two packs in one process are identical. The golden `37d65159b2be0fa091aa840cd56c1a84b73c0bcd9e2df5906d1f8218f5448561` (blessed with ARCH=aarch64) passes in the standalone run, which printed `preserve_order=OFF`, and in the `-p aprender-mcp-setfit` run, which printed `preserve_order=ON`.
- **Bless record:** `DECIDE_BLESS_GOLDEN=1 cargo test … golden_sha` exited **101 by design** and printed `blessed … = 37d65159… on ARCH=aarch64; re-run without DECIDE_BLESS_GOLDEN`.
- **The probes are Python's values.** The manifest stores `probes.json` byte for byte. On this aarch64 box the Rust probe bits happen to equal Python's in all 4 values, so here the test proves storage but not divergence. x86_64 CI is where the two can differ, and the golden is built so that such a difference cannot move it.
- **The private mint is proven by trybuild, and the proof is shown to bite.** A struct literal fails with E0451 on `method`, `identity` and `manifest`. `Decider::new` fails with E0599, and rustc's note points at `load_bytes` and `load_path`. When the fields are made `pub`, the harness goes red with "Expected test case to fail to compile, but it succeeded".
- **Real-weights smoke (read-only, not committed).** A temporary integration test packed `models/decide/tweet-stance-16`, the halted 08-08 run dir:
  - The derived set is an exact bijection with all 206 checkpoint tensors (28 encoder layers, 2 head layers, `act_head`, and the F32 `temperature`).
  - The artifact is **846,192,964 bytes**, 63% of the 1.25 GiB cap.
  - The Rust CPU probes agree with the Python probes recorded on **mps:0**: max|d| **1.26e-6** and **1.21e-6** (bar 1e-5). Tokens (21 and 28) and labels match.
  - All 8 rungs passed. The identity is `a3da497a…` and the base is `laya-en-root@55cf4c4e`.
  - Debug-build timings: pack 55 s, load 64 s.
  - This is evidence for 08-09's pack and verify; it is not a deployability claim.

## Task Commits

1. **Task 1: tracer (artifact, packer, ladder, Decider)**: `c44397cfb` (feat)
2. **Task 2 RED: ladder, determinism and forged-header tests**: `73e0be235` (test)
3. **Task 2 GREEN: index cap, exact size rule, golden and contract bindings**: `224366468` (feat)
4. **Task 3: trybuild private-mint proof**: `392aa131b` (test)
5. **Formatting pass (rustfmt on the two touched packages, no behaviour change)**: `f8a12af61` (style)

**Plan metadata:** recorded in the docs commit that carries this SUMMARY.

## Files Created/Modified

- `crates/aprender-decide/src/artifact.rs`: constants, manifest types, `ArtifactLimits`, `ArtifactError`, the writer (steps 1-6), the bounded read, `check_index_extent`, rungs 2-7, `load_verified` (the mint) and `inspect_manifest`.
- `crates/aprender-decide/src/artifact/{tests,ladder,determinism}.rs`: the tracer, closure, door and mirror tests, the 26 ladder negatives, and the 4 determinism tests.
- `crates/aprender-decide/src/pack.rs`: `PackInputs::from_run_dir` with its consistency refusals, `pack_run_dir`, the typed run-dir schemas, and 3 refusal tests.
- `crates/aprender-decide/src/lib.rs`: `Decider` (private fields, `load_bytes`/`load_path`, delegating accessors) and `ModelIdentity`.
- `crates/aprender-decide/src/laya/mod.rs`: `classify_for_task`, with the row scoring shared with `classify_prepared`.
- `crates/aprender-decide/src/test_support.rs`: `contract_yaml` is now `pub(crate)`.
- `crates/aprender-decide/tests/ui.rs` + `tests/ui/*.{rs,stderr}`: the trybuild harness and its two cases.
- `crates/aprender-decide/tests/fixtures/laya_tiny.apr.sha256`: the golden.
- `crates/apr-format/src/v2/reader_impl.rs`, `tests.rs`: `index_capacity` and 2 tests.
- `contracts/decide-apr-v1.yaml`: the `variant` field, `blobs` as an array, `Decider` in `private_mint`, the `test:` lines, and KANI evidence names.
- `crates/aprender-decide/Cargo.toml`, `Cargo.lock`: `safetensors` is now a normal dependency, `trybuild` is a dev-dependency, and the lock adds only the trybuild edge.

## Decisions Made

See `key-decisions` in the frontmatter. These are the ones later plans need:
- **08-06/07 (servers):** load with `Decider::load_path(path)` or `Decider::load_bytes(&bytes)`. Put `decider.identity()` (artifact_sha256, recipe_id, method, base) in every classify response, and price requests with `prepare` → `PreparedRow::tokens()`. The Lambda local-file path should read through `artifact::read_decide_apr_bytes_bounded(file, Some(len))`.
- **08-09 (pack/verify CLI):** `PackInputs::from_run_dir` already refuses a task-copy mismatch and any report hash (recipe_id, inputs, eval-probs, zero-shot-probs, probes) that disagrees with the files. Gate and variant POLICY is still yours. Use `inspect_manifest` for identity only. `artifact_sha256_hex` is the one hash implementation.
- **08-12 (CI/bindings):**
  - Add `-p aprender-decide --test ui` to the ci.yml `--test` line, because the target is dark until then.
  - The binding.yaml signatures for decide-apr-v1 need `Decider`/`ArtifactError` and `declared_len: Option<u64>`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] The planned single trybuild case could not prove the struct-literal half**
- **Found during:** Task 3
- **Issue:** The plan put the struct literal and `Decider::new` in one file. rustc reports the E0599 type error and never runs the privacy pass, so the generated `.stderr` held no E0451. The struct literal was therefore proven by nothing.
- **Fix:** I split it into `decider_struct_literal.rs` (E0451 on all three fields) and `decider_no_constructor.rs` (E0599). A `fn any<T>() -> T { loop {} }` helper avoids the `unreachable expression` warning noise that `unimplemented!()` produced.
- **Verification:** Both snapshots name their errors. A `pub`-fields mutant turns the harness red.
- **Committed in:** `392aa131b`

**2. [Rule 1 - Bug] The size rule was wrong for a zero dimension after an overflowing one**
- **Found during:** Task 2 RED (`size_rule_exhaustive`: `[usize::MAX, 0]` F16 gave `None`, but the exact size is `Some(0)`)
- **Fix:** `expected_bytes` returns `Some(0)` when any dimension is 0, before the checked fold. The old behaviour failed closed, but it misstated the size.
- **Committed in:** `224366468`

**3. [Rule 3 - Blocking] The contract's manifest schema did not match the plan's manifest**
- **Found during:** Task 2 RED (`contract_mirror`: the manifest has `variant`, and the contract did not)
- **Fix:** decide-apr-v1 `manifest.fields` gained `variant`, and `blobs` changed from "object" to "array of {name, sha256}" (the plan's typed-struct rule). `pv validate` reports 0 errors and 0 warnings. Nothing had shipped, so `schema_version` stays 1.
- **Committed in:** `224366468`

**4. [Rule 2 - Missing critical] Probes on the loaded weights instead of a second model load**
- **Issue:** The contract's probe task differs from the served task, and `Laya` binds one task at load. A second `Laya::from_parts` per load would widen the full checkpoint twice at every cold start (about 3.4 GB for Laya-en).
- **Fix:** Added `Laya::classify_for_task(&Task, &[String])` (laya/mod.rs, a file outside the plan's list). It shares its row scoring with `classify_prepared`.
- **Committed in:** `c44397cfb`

**5. [Rule 2 - Missing critical] Tests beyond the named set**
- **Ladder:** `repack_control_loads`, `index_past_end`, `in_memory_over_cap`, `recipe_blob_hash` (the contract's qa_gate falsification), and the exhaustive `rung2_predicate_exhaustive` and `size_rule_exhaustive`. The contract's KANI-DECIDE-APR-001/-003 harness texts now name these two tests as their evidence.
- **Artifact tests:** `pack_unpack_closure` (FALSIFY-004), `load_path_door` and `contract_mirror`.
- **Pack tests:** `task_copy_must_match`, `report_hash_must_match` and `unknown_recipe_key_is_refused`.
- **Committed in:** `73e0be235`

**6. [Rule 3 - Blocking] Files beyond `files_modified`**
- The ladder, determinism and tests modules live in `src/artifact/{ladder,determinism,tests}.rs` rather than inline, to keep `artifact.rs` reviewable. I also touched `laya/mod.rs` and `test_support.rs`, and added a second UI case with its `.stderr`. None of these files is gitignored (they committed without `-f`).

**7. [Rule 3 - Blocking, not fixable in scope] The Task 2 verify-2 PASS line is unreachable**
- `scripts/check_contract_test_binding.sh` still exits rc 1 with `VACUOUS`, caused by the pre-existing `spectral-indices-v1` (D-ITEM-08-01-A). Every other clause of that verify passed.
- I measured on a lifted copy instead: 622 refs, 0 dangling in decide-apr-v1 and laya-parity-v1, and the 44 pre-existing dangling refs unchanged. Two mutated names were flagged. Recorded as D-ITEM-08-05-A.

---

**Total deviations:** 2 auto-fixed bugs (Rule 1), 2 auto-added items (Rule 2), 3 blocking notes (Rule 3, one of them pre-existing and out of scope).
**Impact on plan:** Every must-have truth holds. The only acceptance clause not met is the guard's PASS line, which is blocked by another phase's contract.

## TDD Gate Compliance

- **RED** is `73e0be235 test(08-05)`. Four target tests failed on assertions for the planned behaviour, and each record was classified `RED_EVIDENCE_OK` by `gsd check tdd-red-evidence`:
  - `v2::tests::index_capacity_is_bounded_by_the_index_bytes`: `left: 4294967295, right: 5`, from a behaviour-preserving extraction of the old reservation.
  - `artifact::ladder::size_rule_exhaustive`: `[usize::MAX, 0]` gave `None`, but `Some(0)` was wanted.
  - `artifact::determinism::golden_sha`: there was no golden yet.
  - `artifact::tests::contract_mirror`: the contract had no `variant` field.
- The forged-header test **passed against the old reservation** on macOS, whose allocator reserves the ~309 GB lazily, as the plan predicted. The pure `index_capacity` test is therefore the observable RED.
- **Substitute RED evidence** covers the tests that passed on first run, because Task 1's tracer already implemented their rungs. I made one induced mutation of production code per target test, ran the test with `--exact`, and restored the file from HEAD (`git status` was empty afterwards). All **33/33 records are `RED_EVIDENCE_OK`**:
  - declared-length check removed
  - stream and in-memory caps each one byte loose
  - CRC check removed
  - tensor_count cap one loose
  - extent comparison removed, and `<=` weakened to `<` (for the exhaustive predicate)
  - past-end check loosened
  - model_type, custom-key count, schema_version, method and labels each unchecked
  - `deny_unknown_fields` removed from `Manifest` and from `Recipe`
  - missing-name and unexpected-name checks removed
  - size equality unchecked
  - blob hash unchecked, for tokenizer, task and recipe
  - rung-5 scan removed
  - probe tolerance widened to 1.0 (both at load and at pack)
  - row budget loosened by 100
  - a `created_at` timestamp written into the metadata (`same_process_twice`, `golden_sha`)
  - stored probes reversed
  - one weight bit flipped on the way into the container (`pack_unpack_closure`)
  - identity set to the recipe_id (`load_path_door`, `tiny_roundtrip`)
  - task copy unchecked
  - report hashes unchecked
- `backing_canary` cannot go RED by design. Its evidence is the dual run, which printed OFF in one run and ON in the other.
- **GREEN** is `224366468 feat(08-05)`. **REFACTOR:** the `f8a12af61 style(08-05)` rustfmt pass, with all tests re-run green afterwards.

## Issues Encountered

- **rtk mis-reports `wc -c < file` as 0.** Through the hook, the golden file measured 0 bytes. `rtk proxy wc -c` and Python both give 65. The plan's verify wraps the whole command in `rtk run`, and there the size clause passed.
- **`pv lint` rewrote `.pv/lint-previous.json`** on every guard run. I restored it each time.
- **The windows ledger still refuses appends** (pre-existing, entry 24). The unrun-verify item is in deferred-items.md as D-ITEM-08-05-A.
- `cargo fmt -p aprender-decide -p apr-format -- --check` flagged formatting in my own Task 1 and 2 code. I fixed it in `f8a12af61`. The workspace-wide `--all` failure from D-ITEM-08-04-B is untouched.

## Known Stubs

None. A scan of every created or modified source file for TODO, FIXME, placeholder, `todo!` and `unimplemented!` found nothing (rc 1).

## Threat Flags

None beyond the plan's register. Each threat is covered by a test:
- **T-08-05-01:** `declared_length_over_cap`, `read_over_cap` (including an endless stream) and `in_memory_over_cap`.
- **T-08-05-07:** rung 2 plus apr-format `index_capacity`, the forged-header tests on both readers, and the exhaustive predicate.
- **T-08-05-08:** `no_rust_floats_in_manifest`.
- **T-08-05-02:** the blob hash, NaN and probe tests.
- **T-08-05-03:** trybuild.
- **T-08-05-04:** `labels_disagree_with_task`.
- **T-08-05-05:** the golden in two processes and two backings.
- **T-08-05-06:** only hashes and the contract's synthetic probe strings enter the manifest.

The new file-read surface (`PackInputs::from_run_dir`) is a pack-time library path that reads a local run dir. No server calls it.

## User Setup Required

None. No external service configuration is required.

## Next Phase Readiness

- **08-06/08-07 can load artifacts now.** The thin servers can build on `Decider` and `ModelIdentity`. The real Laya-en artifact loads through all 8 rungs, but at debug-build speed; the release-build cold start is still to be measured in 08-07/08-10.
- **08-09** can wrap `pack_run_dir` with gate and variant policy. The real run dir's probes already pass the 1e-5 replay, so an mps-recorded `probes.json` is not a blocker.
- **08-12** must:
  - add `--test ui` for aprender-decide to CI;
  - reconcile the decide-apr-v1 binding.yaml signatures;
  - note that x86_64 CI is the first run of `golden_sha` on AVX (this is expected to hold, because only Python probe bits are stored).
- **Carried blockers:** 08-08 is still HALTED at its human decision (the ECE ceiling). This plan did not touch `scripts/laya_train/` or the laya-finetune-gate-v1 thresholds. D-ITEM-08-01-A still keeps the strict-binding guard from printing PASS.

## Self-Check: PASSED

- All 11 created and 8 modified files are present on disk.
- Commits `c44397cfb`, `73e0be235`, `224366468`, `392aa131b` and `f8a12af61` exist, and `git rev-list --count be9934028..HEAD` = 5.
- Plan verification was re-run on the final HEAD:
  - `cargo test -p aprender-decide --lib` gave 66 passed and 0 failed, with 26 ladder tests ok.
  - `--test ui` returned rc 0 and names `decider_struct_literal`.
  - The determinism runs returned rc 0 for OFF and rc 0 for ON, with golden_sha ok in each.
  - `cargo test -p apr-format --lib` gave 90 passed.
  - `clippy -p aprender-decide -p apr-format --all-targets --no-deps -D warnings` returned rc 0.
  - fmt `--check` returned rc 0.
  - `grep -c 'with_capacity(tensor_count as usize)'` returned 0.
  - `pv validate decide-apr-v1` gave 0 errors.
  - The one exception is the strict-binding guard, which returned rc 1 VACUOUS. That is pre-existing (Deviation 7).

---
*Phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server*
*Completed: 2026-09-26*
