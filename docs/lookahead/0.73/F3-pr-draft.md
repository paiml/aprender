# F3 PR draft — open AFTER the 0.70 cut (cop ruling 2026-09-28 10:20Z)

Branch: `la-73/4575-wgpu-backend-label` @ `fd02954a8e` (2 commits over origin/main `c115c5ed02`). Refs #4575 (OBS-18). Not folded into #4575.

## Title
fix(wgpu): the backend line said Vulkan on every host — build it from AdapterInfo (Refs #4575)

## Body
`apr run` printed the literal `Backend: wgpu (Vulkan)` after `GpuDevice::new()`, whatever adapter wgpu picked.
`gpu_backends()` is `Backends::PRIMARY` (Vulkan, Metal, DX12, WebGPU), so on a Mac the line named a backend
that never ran.

- `GpuDevice` keeps the adapter's `AdapterInfo`. It is set at all three construction sites: `device/mod.rs`
  `new_async`, `new_with_adapter_index_async`, and `pool.rs`.
- `backend_line()` prints `Backend: wgpu (<AdapterInfo.backend>) adapter=<name>`. On Vulkan the prefix is
  unchanged, so FALSIFY-CPU-GPU-005's grep still matches.
- The three `eprintln!` sites in aprender-serve (`gguf_gpu_generate.rs` ×2, `batch_wgpu.rs`) use it.
- The remaining `Backend: wgpu (Vulkan)` strings (`dispatch.rs:155`, `cli_commands.rs:416`,
  `inference_result.rs:1143`, the contract) quote a measured RTX 4090 Vulkan run. They are history and are left unchanged.

## Evidence (intel, private CARGO_TARGET_DIR, 2026-09-28)
- `cargo test -p aprender-compute --lib --features gpu -- backend_line test_gpu_backends`: 2 passed.
- Mutant: hardcode `"Backend: wgpu (Vulkan)"` in `backend_line` → `backend_line_names_the_reported_backend` FAILS (rc 101). Restored, porcelain clean.
- `cargo clippy -p aprender-compute --lib --features gpu -- -D warnings`: 0.
- `cargo check -p aprender-serve --lib` (default features include `gpu`): 0.

## Before opening (checklist)
- [ ] Rebase or merge onto post-cut main only if it conflicts (L17: no bulk update-branch).
- [ ] Re-run the four commands above on intel at the PR head.
- [ ] Run the aprender-serve lib suite (not yet run).
- [ ] Darwin live line on mini when the train is inactive: expect `Backend: wgpu (Metal) adapter=Apple …`.
- [ ] Quorum lanes: Sonnet 5 + non-Claude (author is Opus 5.5, never an Opus lane).
