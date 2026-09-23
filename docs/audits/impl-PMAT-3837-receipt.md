# Receipt — PMAT-3837: the `--features cuda` lint surface, cleared and gated

Branch `fix/3837-cuda-feature-lint-gate` off `main@49fe19c28`; author aprender-dd (claude-opus-5-5). Two commits:
`b56baf78d` (the lint sweep) and `f18503020` (the gate).

## Claim 1: the surface is clean (b56baf78d)
- Before, on main: `cargo clippy -p apr-cli --lib --features cuda -- -D warnings` compiled 1 finding in
  aprender-compute (`unused import GemmOp`), then 61 in aprender-train once that was cleared. The ticket's "60" was
  measured on batch-2's base.
- After: 0. The default axis stays at 0. `cargo check -p aprender-train --lib --profile test`, with and without
  `--features cuda`: rc 0.
- How, by kind:
  - 38 redundant casts and unused imports/variables: `clippy --fix`. Its renames of three unused `*_bits` locals to
    `_x` were replaced by DELETING those lines.
  - 6 `unnecessary_unwrap`: `if let Some(w) = x.as_ref().filter(|_| cond)`. The else-if chains are unchanged, and
    line 3237's identical check (no unwrap) is untouched.
  - Dead functions with zero callers workspace-wide, tests included (verified by grep and `git log -S`; all dead
    since the April monorepo import): deleted. A diff-shape check shows each deletion is exactly one fn.
  - Test-only hooks (`forward_jit_compiles`, `reset_forward_jit_counter` and the forward cache's counter methods,
    used only by the R-3 tests under `#[cfg(all(test, feature = "cuda"))]`): `#[cfg(test)]`. The backward
    cache's counter methods had no user at all: deleted.
  - Fields BUILT at init but never read (`embed_transposed`, `profiler_op_*`, `fused_clip`): a reasoned
    `#[allow(dead_code)]`. Removing them changes GPU allocation, which needs a GPU-verified run; no GPU was
    available during the 0.69.1 sweep. FINDING: `fused_clip` means the ALB-078 fused-clip pipeline is allocated
    and never runs.
  - `CudaBlock` `large_enum_variant`: allowed, with the measured sizes (2288 vs 2040 bytes, one per layer).

## Claim 2: the axis is gated (f18503020)
- `scripts/check_clippy_feature_matrix.sh`: axes `default` and `cuda`. The flags are a bash array, passed as
  separate words. `--no-default-features` is EXCLUDED by name and #4041 (it does not compile: 8 errors), not
  silently.
- No CUDA toolkit is needed: aprender-gpu dlopens the driver (`libloading`); no `build.rs` links CUDA.
- Measured:
  - `--self-test` plants `x as u32` in cuda-only code, with the restore trapped first (`git checkout --`) → cuda
    FAIL (3 error lines), default ok. PASS.
  - A planted `F=("--features cuda")` (the ticket's single-argument trap) → `BROKE cuda: cargo rejected the flags`,
    rc 2, "could not check". It is not reported as a lint result.
  - The real run: default ok, cuda ok → PASS.
- Workflow: `toolchain-ceiling.yml` job `clippy-feature-matrix`, daily, clean-room X64, self-test step first.
  actionlint OK; `check_guards_are_wired.sh` PASS; `check_runner_labels.sh` OK; bashrs 0 errors.

## Not done / out of scope
- The `--no-default-features` axis: #4041 (is it a supported configuration?).
- Single-feature axes (`training`, `visualization`, `zram`, `hf-hub`, …) are not measured.
- The workflow edit needs the operator's OK at merge (batched by the cop with #3810/#3658). NOT ARMED.
