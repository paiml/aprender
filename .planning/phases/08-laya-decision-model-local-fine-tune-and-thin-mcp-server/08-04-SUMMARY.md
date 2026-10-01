---
phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
plan: 04
subsystem: ml-models
tags: [aprender-decide, laya, modernbert, decision-method, tokenizers, serde, preserve_order, parity, provable-contracts]

requires:
  - phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
    provides: "08-01 contracts: laya-parity-v1 (logits_abs, probs_abs, ids/markers exact), decide-apr-v1 (task_json_schema, marker_rule), laya-finetune-gate-v1 (calibration_temp_min/max)"
  - phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
    provides: "08-02 laya_tiny fixture: F16 checkpoint, tokenizer.json, rl_agent_config.json, oracle.json (9 rows + the `many` marker-loss question)"
  - phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
    provides: "08-03 aprender::models::modernbert: ModernBertEncoder::from_apr and the Result-returning Linear / layer_norm / gelu_exact / attention"
provides:
  - "crates/aprender-decide: lib-only, method-neutral crate (no [[bin]]), workspace member #88"
  - "DecisionMethod trait (task, prepare, classify_prepared, provided classify); Decision, PreparedRow (private-constructed), DecideError"
  - "task::Task::from_slice: order-preserving MapAccess visitor over the raw bytes; typed TaskError refusals; labels(), render_options(), sha256()"
  - "laya::Laya::from_parts(reader, prefix, encoder_config, agent_config, tokenizer, task) with MarkersLost checked once at load; forward_row(ids, markers, qtype, tap)"
  - "laya::builder::Builder (Tokenizer::from_bytes; port of build_sequence with the shrink and the D-12 truncated flag), laya::head (nhead = max(1, d/64), HeadDoesNotDivide), laya::scorer, laya::temperature (bucket_key, temperature_for, clamp to [0.5, 5.0], softmax_t)"
  - "29 lib tests incl. laya::tests::tiny_parity; bound in laya-parity-v1 (FALSIFY-001/-003, KANI-001 evidence) and decide-apr-v1 (FALSIFY-007/-010)"
affects: [08-05, 08-06, 08-07, 08-09, 08-10, 08-12]

actuals:
  tokens: 22763   # chars/4 over the realized diff (91053 chars, rtk proxy git diff b60927862..HEAD)
  tasks: 2
  commits: 3
plan_head_before: b60927862fa5261e35ecb1fb62403db55d4b17e9

tech-stack:
  added:
    - "aprender-decide crate (deps: aprender-core, tokenizers 0.23.1, serde, serde_json WITHOUT preserve_order, sha2, rayon, provable-contracts-macros; dev: safetensors 0.4, serde_yaml 0.9, base64 0.22 — all already workspace deps)"
  patterns:
    - "Label order from raw bytes via a hand-written Visitor::visit_map, never serde_json::Value/Map; a canary test prints the map backing each run compiled"
    - "Marker loss decided once at task load by building the row with an empty state (markers precede the state)"
    - "Head/scorer tensors loaded through the same F16 widening as core, refused by name (missing, shape, undecodable, non-finite)"
    - "Contract test: lines cite concrete test fns (compound && legs) — pv's resolver cannot see a module-prefix filter"

key-files:
  created:
    - crates/aprender-decide/Cargo.toml
    - crates/aprender-decide/README.md
    - crates/aprender-decide/src/lib.rs
    - crates/aprender-decide/src/task.rs
    - crates/aprender-decide/src/test_support.rs
    - crates/aprender-decide/src/laya/mod.rs
    - crates/aprender-decide/src/laya/head.rs
    - crates/aprender-decide/src/laya/scorer.rs
    - crates/aprender-decide/src/laya/builder.rs
    - crates/aprender-decide/src/laya/temperature.rs
    - crates/aprender-decide/src/laya/tests.rs
  modified:
    - Cargo.toml
    - Cargo.lock
    - README.md
    - contracts/laya-parity-v1.yaml
    - contracts/decide-apr-v1.yaml

key-decisions:
  - "Laya::from_parts prefixes EVERY tensor with `prefix` and loads the encoder at `{prefix}encoder.`; the fixture and Laya checkpoints use prefix \"\""
  - "Decision.truncated is `untruncated_len > max_len` (the oracle's own definition); it equals 'state tokens exceeded the room' whenever the head fits"
  - "Head and scorer LayerNorms use torch's default eps 1e-5 (TORCH_LAYER_NORM_EPS), not the encoder's norm_eps"
  - "Duplicate criteria are detected inside the visitor but refused as a typed TaskError::DuplicateCriterion after parse, so the refusal is not a stringly serde error"
  - "PreparedRow fields are pub(crate): only a method's own prepare() builds one, and classify_prepared still refuses a row whose marker count is not the task's"
  - "decide-apr-v1 FALSIFY-007/-010 cite named tests, not the plan's `--lib task::` / `--lib laya::`: pv's resolver reads the last `::` segment, so a module filter is invisible to the binding guard"

patterns-established:
  - "Contract-mirror test: a Rust constant that restates a contract constant is asserted equal to the YAML at test time (TEMP_MIN/MAX vs laya-finetune-gate-v1)"
  - "Substitute RED evidence for an unexpected GREEN: one induced mutation per target test, classified by gsd check tdd-red-evidence, file restored by git checkout"

requirements-completed: [D-05, D-12, D-13, D-14]

coverage:
  - id: D1
    description: "Laya on core ModernBERT reproduces Laya's oracle on the tiny fixture: ids and markers exact on all 9 rows (shrink, truncation, [MASK]/[SEP] injection, score and noul rows), logits max|d| 1.04e-7 (bar 1e-4), probabilities 5.96e-8 (bar 1e-5), argmax exact, bucket temperatures equal; classify() through the seam equals the per-row probabilities"
    requirement: "D-12"
    verification:
      - kind: unit
        ref: "crates/aprender-decide/src/laya/tests.rs#tiny_parity"
        status: pass
    human_judgment: false
  - id: D2
    description: "Method-neutral crate with the DecisionMethod seam and Laya as the only implementation; lib only (no bin target), README links paiml/aprender, README crate count 88 matches cargo metadata"
    requirement: "D-14"
    verification:
      - kind: integration
        ref: "cargo test -p aprender-core --test monorepo_invariants --test readme_contract (FALSIFY-MONO-011, FALSIFY-README-005, FALSIFY-README-CRATE-001/002)"
        status: pass
      - kind: other
        ref: "cargo metadata --no-deps: aprender-decide targets == [lib]; test ! -e crates/aprender-decide/src/main.rs"
        status: pass
    human_judgment: false
  - id: D3
    description: "Criteria document order is the label index under BOTH serde_json backings, observed: the standalone run prints preserve_order=OFF, the -p aprender-mcp-setfit run prints preserve_order=ON, 11/11 task tests pass in each; refusals typed (score type, one criterion, duplicate, empty name, unknown key, non-string description)"
    requirement: "D-05"
    verification:
      - kind: unit
        ref: "cargo test -p aprender-decide --lib task:: -- --nocapture (preserve_order=OFF)"
        status: pass
      - kind: unit
        ref: "cargo test -p aprender-decide -p aprender-mcp-setfit --lib task:: -- --nocapture (preserve_order=ON)"
        status: pass
    human_judgment: false
  - id: D4
    description: "Marker rule: a task whose options exceed head_max_len is shrunk as Laya does, loads and builds Laya's ids; the 16-criteria task is refused at load with MarkersLost { criteria: 16, markers: 14 }"
    requirement: "D-05"
    verification:
      - kind: unit
        ref: "crates/aprender-decide/src/laya/tests.rs#shrunk_task_is_served"
        status: pass
      - kind: unit
        ref: "crates/aprender-decide/src/laya/tests.rs#marker_loss_is_refused_at_load"
        status: pass
    human_judgment: false
  - id: D5
    description: "Head, scorer and builder reuse core's ModernBERT primitives (D-13): nhead = max(1, d/64) with HeadDoesNotDivide for d=129; builder budget (even shrink, head >= 8, markers < max_len, D-12 flag, [MASK] replacement); temperature buckets, clamp and Laya fallback order, bounds mirrored from laya-finetune-gate-v1"
    requirement: "D-13"
    verification:
      - kind: unit
        ref: "crates/aprender-decide/src/laya/head.rs#tests (nhead_rule, head_does_not_divide)"
        status: pass
      - kind: unit
        ref: "crates/aprender-decide/src/laya/builder.rs#tests (6 tests)"
        status: pass
      - kind: unit
        ref: "crates/aprender-decide/src/laya/temperature.rs#tests (5 tests)"
        status: pass
      - kind: other
        ref: "cargo clippy -p aprender-decide --all-targets --no-deps -- -D warnings"
        status: pass
    human_judgment: false

duration: 24min
completed: 2026-09-26
status: complete
---

# Phase 8 Plan 04: aprender-decide, Laya on core ModernBERT Summary

**`aprender-decide` is a lib-only, method-neutral crate. Its `DecisionMethod` seam has one implementation, Laya (type embedding, `nhead = max(1, d/64)` head, scorer, `build_sequence` port and clamped temperature buckets) on top of core's ModernBERT. On the tiny fixture it matches Laya's own oracle on the first run: ids and markers exact on all 9 rows, logits within 1.04e-7 and probabilities within 5.96e-8. Criteria order is proven to be the label index under both serde_json backings, and each of those two backings was observed in its own run.**

## Performance

- **Duration:** 24 min
- **Started:** 2026-09-26T01:41:59Z
- **Completed:** 2026-09-26T02:06:20Z
- **Tasks:** 2 (Task 1 tracer, Task 2 tdd expansion)
- **Files modified:** 16 (11 created, 5 modified)

## Accomplishments

- **Tracer passed on the first run** (aarch64, Apple M4). The path is the fixture's F16 safetensors, then an in-memory `.apr` (the F32 `temperature` buffer stays F32), then `Laya::from_parts`, then the builder and `forward_row`. Bars are read from `laya-parity-v1`:

  | row | qid | tokens | final | head1 | logits max\|Δ\| (bar 1e-4) | probs max\|Δ\| (bar 1e-5) |
  |---|---|---|---|---|---|---|
  | 0-3 | team (shrunk) | 57/55/50/52 | ≤1.07e-6 | ≤1.43e-6 | ≤1.04e-7 | 2.98e-8 |
  | 4 | urgent (noul) | 53 | 1.05e-6 | 1.43e-6 | 5.22e-8 | 2.98e-8 |
  | 5 | mood (score) | 61 | 8.34e-7 | 1.19e-6 | 4.47e-8 | 2.98e-8 |
  | 6 | safe ([MASK]/[SEP] injection) | 58 | 8.34e-7 | 1.43e-6 | 6.71e-8 | 5.96e-8 |
  | 7 | lang (over-window, truncated) | 64 | 1.43e-6 | 1.43e-6 | 3.35e-8 | 1.49e-8 |
  | 8 | team (419-token state, truncated) | 64 | 1.13e-6 | 1.07e-6 | 1.04e-7 | 2.98e-8 |

  Every row's argmax, bucket temperature and `truncated` flag match. For `classify(texts)` on the team rows, the probabilities are bit-equal to the per-row ones, and the over-window row comes back with `truncated == true`. The `many` question reproduces Laya's 64 ids and its shorter 14-marker list, and the 16-criteria task is refused at load with `MarkersLost { criteria: 16, markers: 14 }`.
- **Both serde_json backings were observed, not assumed.** The two runs printed:
  - `cargo test -p aprender-decide --lib task:: -- --nocapture` printed `serde_json backing: preserve_order=OFF`
  - `cargo test -p aprender-decide -p aprender-mcp-setfit --lib task:: -- --nocapture` printed `serde_json backing: preserve_order=ON`

  Each run passed all 11 task tests, and the crate does not enable the feature itself.
- **The refusal set is typed:** a non-`choice` type, fewer than 2 criteria, a duplicate name, an empty name, an unknown top-level key, a non-string description, `HeadDoesNotDivide` (d=129 gives nhead 2), `MarkersLost`, `MissingSpecialToken`, `RowMarkerCount` and `MarkerOutOfRange`. The crate contains no `unwrap()`, and clippy passes with `-D warnings` on all targets.
- **Contracts are bound.** In laya-parity-v1, FALSIFY-001 and -003 are bound to `tiny_parity`, and KANI-LAYA-PARITY-001 now names the `within` NaN-position test as its evidence. In decide-apr-v1, FALSIFY-007 and -010 are bound to named tests. `pv validate` reports 0 errors and 0 warnings for both contracts.
- The README crate count is now 88 (derived with `cargo metadata --no-deps`), and `monorepo_invariants` and `readme_contract` both pass.

## Task Commits

1. **Task 1: Tracer, from the tiny checkpoint to Laya::from_parts to parity**: `08dca8371` (feat)
2. **Task 2: Expansion, covering task order under both backings, refusals, buckets, nhead and the builder budget**: `70cd1d4fa` (test)
3. **Deviation fix: decide-apr-v1 bindings the guard can resolve**: `7e9a93fe0` (fix)

**Plan metadata:** recorded in the docs commit that carries this SUMMARY.

## Files Created/Modified

- `crates/aprender-decide/Cargo.toml`: the lib-only package. serde_json is declared without `preserve_order`, and a comment says why.
- `crates/aprender-decide/README.md`: describes the seam, Laya, the task.json order rule and the contracts, and links paiml/aprender.
- `crates/aprender-decide/src/lib.rs`: `DecisionMethod`, `Decision`, `PreparedRow` and `DecideError`.
- `crates/aprender-decide/src/task.rs`: `Task`, `Criterion`, `TaskError`, the order-preserving visitor, and 11 tests including the canary.
- `crates/aprender-decide/src/laya/mod.rs`: `QType`, `AgentConfig` (with Laya's defaults), `LayaError`, `Laya::from_parts`, `forward_row`, and `impl DecisionMethod`.
- `crates/aprender-decide/src/laya/{head,scorer,builder,temperature}.rs`: one struct per file, each with its own tests.
- `crates/aprender-decide/src/laya/tests.rs`: `tiny_parity`, `shrunk_task_is_served`, `marker_loss_is_refused_at_load` and `foreign_rows_are_refused`.
- `crates/aprender-decide/src/test_support.rs`: fixture plumbing, YAML bar and constant readers, the NaN-visible `within`, and its test.
- `Cargo.toml`, `Cargo.lock`: the workspace member (the lock adds only the new package), `README.md` (count 88), `contracts/laya-parity-v1.yaml`, and `contracts/decide-apr-v1.yaml`.

## Decisions Made

See `key-decisions` in the frontmatter. These are the ones later plans need:
- **08-05 (artifact):** use `Laya::from_parts(&reader, "", encoder_config, agent_config, tokenizer, Task::from_slice(task_json)?)`. `Task::sha256()` is the sha256 of the exact task bytes, and `task.labels()` gives the ordered array for the manifest.
- **08-06/07 (servers):** call `prepare(texts)` once to price the token budget from `PreparedRow::tokens()`, then call `classify_prepared(&rows)`. Probabilities are arrays in label order.
- **08-09 (verify/pack):** `forward_row` and `temperature::{temperature_for, softmax_t}` are public, so the pack re-score can reuse them.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] The plan's `test:` filters for decide-apr-v1 were invisible to the binding guard**
- **Found during:** Task 2 (lifted strict-binding measurement)
- **Issue:** The plan specified `cargo test -p aprender-decide --lib task::` and `--lib laya::`. pv's resolver (`strict_test_binding.rs::extract_one_fn_name`) takes the last `::` segment of the filter. For those filters that segment is empty, so the reference is treated as "no binding". The resolved count did not move after I added them (589 → 589), and when I mutated them to `task_mutant_zz::` they were **not** flagged.
- **Fix:** I cited concrete tests as compound `&&` legs. FALSIFY-007 names `task::tests::stance_order_none_against_favor` in both the standalone and the pmcp package-set shapes. FALSIFY-010 names `laya::tests::shrunk_task_is_served` and `laya::tests::marker_loss_is_refused_at_load`.
- **Verification:** On the lifted copy, 593 references resolve (+4), with 0 dangling in decide-apr-v1. Mutating two of the names produced `FAIL contracts/decide-apr-v1.yaml: 2 dangling`. All four bound invocations pass as written.
- **Committed in:** `7e9a93fe0`

**2. [Rule 2 - Missing critical] Added tests beyond the plan's named set**
- **Found during:** Tasks 1 and 2
- **Issue:** KANI-LAYA-PARITY-001 cited "the exhaustive NaN-position unit test of aprender-decide's within helper (plan 08-04)", but the plan listed no such test. `RowMarkerCount` and `MarkerOutOfRange` (T-08-04-02: markers come only from the builder) also had no test.
- **Fix:** I added `test_support::tests::within_is_nan_visible`, which covers 10 special values in both argument positions, and updated the KANI harness text to name it. I also added `foreign_rows_are_refused`, `refuses_non_string_description`, `sha256_is_of_the_raw_bytes`, `mask_in_state_is_replaced` and `refuses_a_tokenizer_without_mask`.
- **Committed in:** `08dca8371`, `70cd1d4fa`

**3. [Rule 3 - Blocking] Two small additions to the plan's file and dependency lists**
- `base64` (workspace 0.22) is now a dev-dependency, because the oracle's ladder blocks are `f32le_base64`, and `rayon` (workspace) handles the head's residual adds as core does. Both were already workspace dependencies, and `Cargo.lock` adds only the new package.
- `src/test_support.rs` and `src/laya/tests.rs` are new files that were not in `files_modified`. Neither is gitignored: `git check-ignore -q --no-index` returned rc 1 for both.

**4. [Rule 3 - Blocking, not fixable in scope] Task 1 verify 3 cannot print its PASS line**
- `scripts/check_contract_test_binding.sh` still exits with rc 1 `VACUOUS`, because of the pre-existing `spectral-indices-v1` (D-ITEM-08-01-A). The gitignore half of that verify passed for all 9 listed paths. I measured the guard on a lifted copy instead (see Deviation 1 and D-ITEM-08-04-A). The laya-parity-v1 bindings resolve with 0 dangling, and a mutated name is flagged.

---

**Total deviations:** 1 auto-fixed bug (Rule 1), 1 auto-added test set (Rule 2), 2 blocking notes (Rule 3, one of them pre-existing and out of scope).
**Impact on plan:** The code matches the plan. The only acceptance clause I could not meet is the guard's PASS line, which is blocked by another phase's contract.

## TDD Gate Compliance

- **RED (Task 2):** `70cd1d4fa test(08-04)`. All 29 tests passed on their first run, so RED was an **unexpected GREEN** (fail-fast rule 1). The cause is structural, as it was in 08-03. Task 1's `<action>` required the full parser, refusal set, builder, head rule and temperature module for the tracer to be production-quality, so Task 2 had nothing left to go RED against.
- **Substitute RED evidence:** I made one induced mutation per target test in committed production code, ran the test with `--exact`, and restored the file with `git checkout -- <file>`. Afterwards `git status crates/aprender-decide` was empty. All **23 records** (cargo output converted to TAP) were classified `RED_EVIDENCE_OK` by `gsd check tdd-red-evidence`. The mutations, and the tests they turned red:
  - task (8): criteria sorted (`order_is_document_order`, which got `["alpha","mid","zeta"]`, and `stance_order_none_against_favor`, which got `["against","favor","none"]`), duplicate unchecked, type unchecked, empty description kept (`"a: "`), `deny_unknown_fields` removed, empty name allowed, one criterion allowed
  - head (2): `d/32` heads (`nhead_rule` got `(2, 32)`), divisibility unchecked (`head_does_not_divide` got `Ok((2, 64))`)
  - builder (6): shrunk option 3 (3 ≠ 4), head minimum 4, `markers.retain` removed, `truncated = false`, `[MASK]` kept in state (4 masks ≠ 2), shrink disabled (`shrunk_task_is_served`, where the ids diverge from Laya's)
  - laya (2): the MarkersLost check weakened (`marker_loss_is_refused_at_load`), the row-count check weakened (`foreign_rows_are_refused`)
  - temperature (4): `inf` clamped instead of mapped to 1.0 (got 5.0), `TEMP_MAX` 5.5 (contract mirror), bucket `3..=4` (got `choice:11+`), `temperature[qtype]` fallback skipped (got 1.0)
  - within (1): `!(delta > bound)` (`within(NaN, NaN)` returned true)
- The canary cannot go RED by design. Its evidence is the dual-run verify, which requires OFF in one run and ON in the other.
- **GREEN:** there is no `feat` commit for Task 2, because no implementation change was needed. Task 1's `feat` commit `08dca8371` comes before the tests.
- **REFACTOR:** none.

## Issues Encountered

- **rtk filtered redirected cargo output.** A plain `cargo test ... > log` through the hook wrote rtk's one-line summary instead of cargo's output. From then on, every verify ran through the plan's own `rtk run` / `rtk proxy` forms.
- **`serde_json::json!` expands to `unwrap()`,** which trips clippy's `disallowed_methods` in tests. I replaced it with a `quote()` helper that uses `to_string(..).expect(..)`.
- **`pv lint` rewrote `.pv/lint-previous.json`** on every guard run. I restored it each time with `git checkout -- .pv/lint-previous.json`.
- **`cargo fmt --all -- --check` fails** on `aprender-image` and `aprender-mcp-chronos` files from `fdf6b1802` (D-ITEM-08-04-B), which is pre-existing. `cargo fmt -p aprender-decide -- --check` exits 0.
- **The windows ledger refuses every append** (`Ledger entry 24 has invalid status: "resolved"`), which is pre-existing. The unrun-verify item is therefore recorded in deferred-items.md.
- `pv validate` ran on `target/debug/pv` (0.63.0), because `pv` is not on PATH.

## Known Stubs

None. A scan for TODO/FIXME/placeholder/`todo!`/`unimplemented!` in `crates/aprender-decide/src` found nothing (rc 1).

## Threat Flags

None. The trust boundaries are task.json bytes to label order, caller text to the tokenizer and builder, and .apr bytes to the head/scorer loader. The plan's threat model covers all of them, and each is tested: T-08-04-01 by the order tests and the dual-backing canary; T-08-04-02 by `mask_in_state_is_replaced`, the injection row and `foreign_rows_are_refused`; T-08-04-03 by the shrink and marker-loss tests; T-08-04-04 by the typed errors and the unwrap ban under `-D warnings`; T-08-04-05 by the OFF/ON verify.

## User Setup Required

None. No external service configuration is required.

## Next Phase Readiness

- 08-05 (the decide-apr-v1 artifact and its load ladder) can build on `Laya::from_parts` and `Task::from_slice`/`sha256`/`labels`. The probe replay can reuse `prepare` and `classify_prepared`.
- The x86_64 value for the laya-parity-v1 provisional bars is still unrecorded. CI's first `workspace-test` run prints it through `tiny_parity`, which runs as part of `--workspace --lib`.
- D-ITEM-08-01-A / 08-04-A still keep the strict-binding guard from printing PASS.
- 08-08 remains HALTED at its human decision (the ECE ceiling). This plan did not touch `scripts/laya_train/` or the gate thresholds.

## Self-Check: PASSED

- All 11 created files and 5 modified files are present on disk. Commits `08dca8371`, `70cd1d4fa` and `7e9a93fe0` exist, and `git rev-list --count b60927862..HEAD` = 3.
- Plan verification was re-run on HEAD:
  - The tiny_parity verify returned rc 0.
  - monorepo_invariants + readme_contract returned rc 0, with 2 `test result: ok` lines.
  - The dual-backing verify returned rc 0 (OFF, then ON).
  - The full lib verify showed 29 ok (≥ 14), and clippy `--all-targets --no-deps -D warnings` returned rc 0.
  - The gitignore check returned rc 1 (not ignored) for all 11 created paths.
  - The one exception is the strict-binding guard, which returned rc 1 VACUOUS. That is pre-existing (Deviation 4).

---
*Phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server*
*Completed: 2026-09-26*
