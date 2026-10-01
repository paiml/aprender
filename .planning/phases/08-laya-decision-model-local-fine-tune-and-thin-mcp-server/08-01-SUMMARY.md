---
phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
plan: 01
subsystem: contracts
tags: [provable-contracts, pv, binding-audit, laya, modernbert, apr, mcp, calibration]

requires:
  - phase: 06-native-time-series-forecasting-stack
    provides: forecast-tool-boundary-v1 thin-server boundary precedent, contract-audit-phase4/6 audit targets
  - phase: 04 (setfit APR)
    provides: setfit-apr-v1 artifact-schema precedent (one custom key, bounded read, load ladder, probe replay)
provides:
  - contracts/decide-tool-boundary-v1.yaml (classify bounds priced to the 30 s API Gateway cap, admission, response shape, cold accepted region)
  - contracts/laya-finetune-gate-v1.yaml (gate thresholds, recipe, calibration, seed policy, demo cell, run-dir layout and file schemas)
  - contracts/laya-parity-v1.yaml (torch -> .apr -> Rust tolerances, window mutation rung, pack re-score bar)
  - contracts/decide-apr-v1.yaml (artifact schema, load ladder, probe policy, identity, size cap, deploy eligibility, D-05 task schema)
  - Makefile PHASE8_CONTRACTS + blocking pending-tolerant contract-audit-phase8 wired into tier3
  - 39 binding rows in contracts/aprender/binding.yaml (38 pending, ece_top_label implemented)
affects: [08-02, 08-03, 08-04, 08-05, 08-06, 08-07, 08-08, 08-09, 08-10, 08-11, 08-12]

actuals:
  tokens: 31929
  tasks: 3
  commits: 3
plan_head_before: 59d9ed07fc7b79b331e423d3091d0638fdc14dd0

tech-stack:
  added: []
  patterns:
    - "Flat-scalar constants: blocks with every structured declaration as a top-level key"
    - "Staged falsification binding: implemented_by prose until the implementing plan adds test:, LIVE-PENDING for live-only claims"
    - "Pending-tolerant phase audit (BIND-004 passes, BIND-001 fails) copied from contract-audit-phase4"

key-files:
  created:
    - contracts/decide-tool-boundary-v1.yaml
    - contracts/laya-finetune-gate-v1.yaml
    - contracts/laya-parity-v1.yaml
    - contracts/decide-apr-v1.yaml
    - .planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/deferred-items.md
  modified:
    - contracts/aprender/binding.yaml
    - Makefile
    - README.md

key-decisions:
  - "classify token budget declared at 1024 built-row tokens, DOWN from the derived (30000 - 12000 - 1100 - 4000) / 11 = 1172; probe_budget_ms 1100 derived from 2 x 48 x 11 ms"
  - "Gate pass is decided on Rust-recomputed metrics from probabilities verified against Rust re-scores of the artifact and declared base; reported metrics and reported pass are only checked, never trusted"
  - "Gate ECE is the house top-label ECE (aprender::calibration, floor binning); the spike's right-closed bins are a recorded deviation, not matched"
  - "Probes run through a contract-resident synthetic K = 2 task and store Python's F16-reload values, so probe cost is bounded and artifact bytes do not depend on the packing machine"
  - "Task refusal is marker LOSS after Laya's option shrink (spike-026 rule), never raw option length"
  - "Rust-reverifiable training equations bind to aprender_decide::verify (what CI runs); Python-only ones bind to the laya-train-selftest recipe, marked local-only"

patterns-established:
  - "Phase contract committed before any run: thresholds read from YAML at run time, never literals"
  - "Every constants: value is a flat int/float scalar (python check in verify)"

requirements-completed: [D-03, D-04, D-05, D-06, D-07, D-08, D-09, D-10, D-11, D-12, D-17, D-19]

coverage:
  - id: D1
    description: "decide-tool-boundary-v1 declared (D-09..D-12): classify bounds, admission, response shape, truncation flag, refusal rule, cold accepted region"
    requirement: D-10
    verification:
      - kind: other
        ref: "cargo run -p aprender-contracts-cli --bin pv -- validate contracts/decide-tool-boundary-v1.yaml (0 error(s))"
        status: pass
      - kind: other
        ref: "python3 flat-constants check on decide-tool-boundary-v1 constants"
        status: pass
    human_judgment: false
  - id: D2
    description: "laya-finetune-gate-v1 and laya-parity-v1 declared before any run (D-03..D-08, D-17, D-19)"
    requirement: D-07
    verification:
      - kind: other
        ref: "pv validate contracts/laya-finetune-gate-v1.yaml and contracts/laya-parity-v1.yaml (0 error(s) each)"
        status: pass
      - kind: other
        ref: "python3 structure check: flat constants, seed_policy.variance_seeds == [13,17,23], recipe_variants, run_dir_layout incl. probes.json / zero-shot-probs.json"
        status: pass
    human_judgment: false
  - id: D3
    description: "decide-apr-v1 declared (D-17 artifact schema, D-05 task schema, D-11 identity)"
    requirement: D-17
    verification:
      - kind: other
        ref: "pv validate contracts/decide-apr-v1.yaml (0 error(s)); pv status: 12 equations, 6 obligations, 12 falsification tests, 3 kani harnesses"
        status: pass
    human_judgment: false
  - id: D4
    description: "Blocking contract-audit-phase8 wired into tier3; all four contracts in CONTRACTS and PHASE8_CONTRACTS"
    verification:
      - kind: other
        ref: "make contract-audit-phase8 (rc 0, 'Phase 8 binding audit: 4 contract(s) audited'); negative control rc 2 -> restored rc 0"
        status: pass
      - kind: other
        ref: "make contract-validate (rc 0)"
        status: pass
    human_judgment: false
  - id: D5
    description: "README contract count moved with the tree (1795)"
    verification:
      - kind: unit
        ref: "crates/aprender-core/tests/readme_contract.rs (15 passed incl. test_readme_contract_count_matches_workspace)"
        status: pass
    human_judgment: false
  - id: D6
    description: "No Phase 8 contract cites a nonexistent test (CI strict-binding gate)"
    verification:
      - kind: other
        ref: "bash scripts/check_contract_test_binding.sh — rc 1 for PRE-EXISTING reasons (D-ITEM-08-01-A); no Phase 8 contract named in either the as-is or the lifted run"
        status: fail
    human_judgment: true
    rationale: "The guard's PASS line is unreachable on this branch for reasons outside this plan (VACUOUS skip from spectral-indices-v1, 17 pre-existing dangling refs underneath). The Phase 8 half of the claim was verified; the verifier must decide whether the pre-existing red blocks the phase."

duration: 17min
completed: 2026-09-26
status: complete
---

# Phase 8 Plan 01: Phase 8 Contracts Declared Before Any Run Summary

**Four pv-validated Phase 8 contracts — classify tool boundary priced to the 30 s API Gateway cap, Laya fine-tune gate/recipe/calibration/seed policy, torch -> .apr -> Rust parity bars, and the decide-apr-v1 artifact + D-05 task schema — committed before any training, parity or deploy run, with a blocking pending-tolerant `contract-audit-phase8` in tier3.**

## Performance

- **Duration:** ~17 min
- **Started:** 2026-09-26T00:07:49Z
- **Completed:** 2026-09-26T00:25Z
- **Tasks:** 3 of 3
- **Files modified:** 7 (4 created contracts, binding.yaml, Makefile, README.md) + deferred-items.md

## Accomplishments

- `decide-tool-boundary-v1`: 12 flat integer constants (8 texts, 16384 bytes, 1024 built-row tokens, in-flight 1 / pending 4, envelope constants), 8 equations incl. `classify_admission` and `accepted_region_cold` (one HTTP request, first POST at a cold instance, CONCENTRATED 2 x 512 and DISTRIBUTED 8-text shapes, every cold sample, cold proven by load-log correlation). FALSIFY-DECIDE-TOOL-001..009 staged; 009 is `LIVE-PENDING` for plan 08-11.
- `laya-finetune-gate-v1`: margin 0.05, ECE ceiling 0.10 (house top-label ECE, 15 bins), recompute tolerance 1e-5, calibration [0.5, 5.0] on a 25 % stratified slice (>= 2/class), declared seed 13; `seed_policy`, `recipe`, `recipe_variants` (production / synthetic-fixture), pinned `base`, `device_order`, D-19 `demo` cell, `run_dir_layout` and the recipe / gate-report / eval-probs / probes schemas as top-level keys; 9 equations.
- `laya-parity-v1`: ids/markers/argmax exact, probs 1e-5, logits 1e-4, final norm 1e-4, embeddings 1e-5, per-layer rel rms 1e-3, pack re-score 1e-5, window-mutation rung; spike-025 fixture sha256 `ebf9d94e...706f` recorded.
- `decide-apr-v1`: task schema with order-preserving parse and marker-loss refusal, manifest in one `decide` key, six blobs, 8-rung load ladder with the index-extent bound before any index allocation, synthetic K = 2 probe task and two synthetic inputs (48-token row cap), identity, 1.25 GiB cap with derivation, `deploy_eligibility`; 12 equations.
- Makefile: 4 entries in `CONTRACTS`, `PHASE8_CONTRACTS`, `contract-audit-phase8` (copied from phase4: `set +e`, `status=$$?` on its own line, `audited` counter, empty-list FAIL), wired into tier3 after phase6, added to `.PHONY`.

## Task Commits

1. **Task 1 (tracer): decide-tool-boundary-v1 + contract-audit-phase8** - `12ede1872` (feat)
2. **Task 2: laya-finetune-gate-v1 + laya-parity-v1** - `b9402f825` (feat)
3. **Task 3: decide-apr-v1** - `e6c11902b` (feat)

## Verification evidence (status read directly, never through a pipe)

| Check | Result |
|---|---|
| `pv validate` x4 | rc 0, `0 error(s), 0 warning(s)` each |
| `pv status` x4 | obligations 6/6/4/6, falsification 9/8/5/12, kani 1/2/2/3 — none zero |
| `make contract-audit-phase8` before any row (Task 1) | rc 2, 8 BIND-001 |
| after rows | rc 0, `Phase 8 binding audit: 1 / 3 / 4 contract(s) audited` per task |
| **Negative control** (Task 1): deleted the `truncation_flag` row | **rc 2**, `BIND-001 ... truncation_flag`, `FAIL: unbound equations remain in: contracts/decide-tool-boundary-v1.yaml` |
| restored the row | **rc 0** |
| decide-apr-v1 against pre-row binding.yaml | rc 1, 12 BIND-001 |
| `make contract-validate` | rc 0 |
| `readme_contract` (test binary run directly; rtk filters cargo output) | 15 passed, incl. contract count at 1795 |
| `check_readme_claims.sh` FALSIFY-README-002 | PASS 1795 |
| flat-constants python checks, structure check | pass |
| `ls models/decide` at each contract commit | absent (D-07 ordering evidence) |
| `check_contract_test_binding.sh` | rc 1, pre-existing — see Deviations |

## Decisions Made

See `key-decisions` in the frontmatter. Also: parity `constants:` hold only the structural numbers (512, 192, 128, 64, 28, 2, 14, 158); tolerances live in each equation's `float_tolerance`, the chronos-bolt precedent. Binding function names for not-yet-written code (`precheck`, `try_admit`, `classify_response`, `check_gate`, `check_split`, `verify_run`, `parse_task`, `check_markers`, `load_verified`, `read_decide_apr_bytes_bounded`, `pack_run_dir`, `build_row`, `score_rows`, `ModernBertEncoder::forward`) are the intended symbols; the implementing plans correct a row's `function:` when they flip it.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Two stale README contract counts (1778) fixed alongside the table cell**
- **Found during:** Task 1
- **Issue:** `scripts/check_readme_claims.sh` FALSIFY-README-002 was already failing: README lines 225 and 256 said 1778 against 1791 on disk. The plan names only the table cell, but the same number moves with every contract this plan adds.
- **Fix:** all three occurrences set from `find contracts -name '*.yaml' | wc -l` at each task (1792 -> 1794 -> 1795).
- **Files modified:** README.md
- **Verification:** FALSIFY-README-002 PASS 1795; readme_contract 15 passed.
- **Committed in:** 12ede1872, b9402f825, e6c11902b

### Not satisfied (pre-existing, out of scope)

**2. strict-test-binding guard PASS line unreachable (D-ITEM-08-01-A)**
- **Found during:** Task 1 verify.
- **Issue:** `bash scripts/check_contract_test_binding.sh` exits 1 with `VACUOUS: ... SKIPPED (contract validation failed)` because `contracts/spectral-indices-v1.yaml` (commit fdf6b1802, on this branch, not on origin/main) has no `kani_harnesses:`. Lifting that in a temporary copy makes the guard measure (585 refs resolved) and it then fails on 17 pre-existing dangling references in chronos-bolt-parity-v1 (9) and setfit-encoder-conformance-v1 (8).
- **What was verified instead:** in both the as-is and the lifted runs no Phase 8 contract is named; a python check confirms the only test reference in all four contracts is the `LIVE-PENDING` FALSIFY-DECIDE-TOOL-009. The temporary spectral edit was reverted each time (`git diff --quiet` confirmed).
- **Why not fixed:** a partial repair of two other phases' contracts that still would not turn the guard green; logged in `deferred-items.md`. The WINDOWS.md ledger append was attempted and refused by the tool (`Ledger entry 24 has invalid status: "resolved"` — a pre-existing ledger defect), so the item lives in deferred-items.md only.

**3. Makefile edit anchor collision (Task 3), self-corrected before commit**
- The first Task 3 Makefile substitution asserted a unique anchor that also matched the 20-space PHASE8 line, aborted, and the binding rows were inserted before decide-apr-v1 was in the list, so the pre-row RED state was measured separately with `pv audit` against `git show HEAD:contracts/aprender/binding.yaml` (rc 1, 12 BIND-001). No committed state was affected.

**Total deviations:** 1 auto-fixed (Rule 1), 1 acceptance clause not satisfiable for pre-existing reasons, 1 process self-correction.
**Impact on plan:** every contract, number, schema and gate the plan specifies is committed and validated; the only unmet clause is the repo-wide strict-binding PASS line, blocked outside Phase 8.

## Issues Encountered

- `rtk` filters `cargo test` output, so the readme_contract verify's `grep -c '^test .* ... '` count reads 0 through the hook; the test binary was run directly (15 `test ... ok` lines).
- `pv lint` rewrites the tracked `.pv/lint-previous.json`; restored with a file-scoped `git checkout --` after each guard run.
- FALSIFY-README-003 (CLI count 110 vs 111) fails, pre-existing and unrelated (D-ITEM-08-01-B).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Ready for 08-02 (synthetic fixtures): `laya-finetune-gate-v1` `run_dir_layout`, `recipe_variants.synthetic-fixture` and the schemas are the layout 08-02 generates, and `laya-parity-v1` expects 08-02 to record its two tiny-fixture sha256 values. Plan 08-12 tightens `contract-audit-phase8` to refuse any BIND- line once every row is flipped.

## Self-Check: PASSED

- FOUND: contracts/decide-tool-boundary-v1.yaml, contracts/laya-finetune-gate-v1.yaml, contracts/laya-parity-v1.yaml, contracts/decide-apr-v1.yaml
- FOUND commits: 12ede1872, b9402f825, e6c11902b
