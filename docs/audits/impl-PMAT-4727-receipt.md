# PMAT-4727 implementation receipt — FLAKE-0, modality tests' backend env race

## Defect
`crates/aprender-serve/tests/modality_matrix/common.rs` `force_backend` / `clear_backend_forcing`
set and remove the process-wide `REALIZAR_BACKEND`, `REALIZAR_FORCE_SCALAR`, `REALIZAR_FORCE_SIMD`.
libtest runs the `e2e_modality_parity` target's tests on parallel threads, so one test's
`clear_backend_forcing` / `force_backend` lands between another test's `force_backend` and its
assertion. Seen as `test_wgpu_in_common_infrastructure` failing at `wgpu_vulkan.rs:259`.

## Change
1. Commit 0055d4a41f (FLAKE-0 step 1): `#[ignore]` on the flaky test, its own commit.
2. Commit ed46d951f0 (fix): a process-wide `Mutex<()>`. `force_backend` takes it and parks the
   guard in a thread-local (re-entrant: `e2e_modality_parity.rs` calls `force_backend` in a loop
   on one thread); `clear_backend_forcing` takes it if not held, clears the vars, then drops the
   guard. Poison is recovered with `PoisonError::into_inner`, so one panicking test cannot fail
   the rest. The `#[ignore]` is removed. No production code and no gate changes.

## Evidence (intel, RED/GREEN, same harness)
The `e2e_modality_parity` test binary was built at base (316dee2cd4) and at head (with the fix)
and each was run 30 times in full, with no filter:

| binary | failed runs |
|--------|-------------|
| base   | 8 / 30      |
| head   | 0 / 30      |

The disabled test is re-enabled at head, so the 0 / 30 includes it.
