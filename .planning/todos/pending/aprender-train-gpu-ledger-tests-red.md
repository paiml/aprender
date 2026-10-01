---
type: defect
severity: medium
created: 2026-08-17
found_in: phase-05-wave-3-post-merge-gate
resolves_phase:
---

# 21 `aprender-train` GPU ledger/guard tests fail on the dev box

## What

`cargo test -p aprender-train --lib` reports:

```
test result: FAILED. 7626 passed; 21 failed; 14 ignored
```

All 21 are confined to three modules:

- `gpu::guard::tests::*` (8)
- `gpu::ledger::tests::*` (12)
- `gpu::wait::tests::test_timeout_when_full` (1)

## Evidence it is pre-existing

Ran the same filtered subset at commit `f02a8aadfec04bdc5d175c977c0c2be01cb3c051`
in a detached control worktree:

```
BASE_COMMIT_GPU_TESTS_EXIT=101
test result: FAILED. 196 passed; 21 failed; 0 ignored; 7444 filtered out
```

The failing test-name sets at base and at Wave 3 HEAD are **byte-identical**
(`diff` of the sorted name lists is empty). Phase 5 Wave 3 (05-07, 05-08) touched
no file under `crates/aprender-train/src/gpu/`.

## Open question — do NOT assume the cause

These are reservation-ledger tests, and the likely candidates are (a) genuine
logic breakage, (b) dependence on a real GPU absent on this darwin/aarch64 dev
box, or (c) cross-test interference on a shared on-disk ledger under parallel
execution. These have **not** been distinguished. Per CLAUDE.md Verification
Discipline §2/§6, pick the cause by measurement — run single-threaded
(`--test-threads=1`) and on a GPU host — rather than by inspection of the names.

If the cause turns out to be (b) or (c), the fix is to make the tests declare
their environment requirement (skip/`#[ignore]` with a reason, or serialize on
the ledger), so a red result means a real defect instead of an unmet precondition.

## Reproduce

```bash
cargo test -p aprender-train --lib gpu::
cargo test -p aprender-train --lib gpu:: -- --test-threads=1   # control for (c)
```
