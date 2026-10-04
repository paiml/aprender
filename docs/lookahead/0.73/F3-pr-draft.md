# F3 PR draft — open AFTER the 0.70 cut (cop ruling 2026-09-28 10:20Z)

Branch: `la-73/4575-wgpu-backend-label` @ `fd02954a8e` (2 commits over origin/main `c115c5ed02`). Refs #4575 (OBS-18). Not folded into #4575. Re-checked against origin/main `316dee2cd4` on 2026-10-04
(static, no cargo): see Readiness below.

## Title
fix(wgpu): the backend line said Vulkan on every host — build it from AdapterInfo (Refs #4575)

## Body
`apr run` printed the literal `Backend: wgpu (Vulkan)` after `GpuDevice::new()`, whatever adapter wgpu picked.
`gpu_backends()` is `Backends::PRIMARY` (Vulkan, Metal, DX12, WebGPU), so on a Mac the line named a backend
that never ran. A release receipt shows it: on mini (Darwin 25.6.0, arm64, accelerator `Apple M4 (Metal)`) the 0.68.2
receipt recorded `Backend: wgpu (Vulkan)` (`evidence/dogfood/0.68.2/mini.json:176` at 316dee2cd4).

- `GpuDevice` keeps the adapter's `AdapterInfo`. It is set at all three construction sites: `device/mod.rs`
  `new_async`, `new_with_adapter_index_async`, and `pool.rs`.
- `backend_line()` prints `Backend: wgpu (<AdapterInfo.backend>) adapter=<name>`, with the backend in `{:?}` form
  (`Vulkan`, `Metal`). On Vulkan the prefix is unchanged, so FALSIFY-CPU-GPU-005's grep still matches.
- The three `eprintln!` sites in aprender-serve (`gguf_gpu_generate.rs` ×2, `batch_wgpu.rs`) use it.
- The remaining `Backend: wgpu (Vulkan)` strings (`crates/apr-cli/src/dispatch.rs:181`, `cli_commands.rs:417`,
  `crates/aprender-serve/src/infer/inference_result.rs:1143` at 316dee2cd4, and the contract) quote a measured RTX 4090
  Vulkan run. They are history and are left unchanged.
- Readers of the line at 316dee2cd4: `scripts/release/host_receipt.sh:268` keeps the first stderr line that starts
  with `Backend:`, so it still records the line. The a9 case at 316dee2cd4
  (`scripts/check_release_host_receipts.sh:597-599`) expects `Backend: wgpu (Vulkan)`, but its fixture prints that
  line itself (`:313`), so it never reads apr's output. `cli_commands.rs:470` at 316dee2cd4 asserts that a run does
  not print `Backend: wgpu`, and that prefix is kept.

## Evidence (intel, private CARGO_TARGET_DIR, 2026-09-28)
- `cargo test -p aprender-compute --lib --features gpu -- backend_line test_gpu_backends`: 2 passed.
- Mutant: hardcode `"Backend: wgpu (Vulkan)"` in `backend_line` → `backend_line_names_the_reported_backend` FAILS (rc 101). Restored, porcelain clean.
- `cargo clippy -p aprender-compute --lib --features gpu -- -D warnings`: 0.
- `cargo check -p aprender-serve --lib` (default features include `gpu`): 0.
- `cargo test -p aprender-serve --lib -j 8` at fd02954a8 (intel, 2026-09-28 14:18Z): STOPPED by PID at 14:48Z because intel load hit 199 with release CI running. Partial: 16143 `... ok`, 0 FAILED, 0 panicked. This is not a full-suite result.
- `cargo test -p aprender-serve --lib -j 4 -- --test-threads=4` at fd02954a8 (intel, nice 19, started 2026-09-29 03:18Z at load 11): **16149 passed, 0 failed, 62 ignored, rc 0**, finished in 1673 s.

## Readiness at 316dee2cd4 (2026-10-04, item q; static, no cargo, no PR)
- Merge: `git merge-tree c115c5ed02 316dee2cd4 fd02954a8e` (the old form; git 2.34.1 has no `--write-tree`)
  prints 0 conflict markers. Two files change on both sides, `batch_wgpu.rs` and `gguf_gpu_generate.rs`. Main's
  side is three 2-line inserts from #4056 (folded in 989cb012e5), `fwd.set_rms_norm_eps(...)`, each below an F3
  `eprintln!` site. So no rebase is needed now; re-run the merge-tree at open time.
- The F3 sites at 316dee2cd4: `crates/aprender-serve/src/infer/batch_wgpu.rs:144`, and
  `crates/aprender-serve/src/infer/gguf_gpu_generate.rs:178` and `:655`.
- Not changed here: the serve banners print fixed text, `WGPU device ready (Vulkan/Metal)`
  (`crates/apr-cli/src/commands/serve/handlers.rs:756`) and `Backend: WGPU (Vulkan/Metal/WebGPU)` (`:911`). They
  name no adapter. A follow-up could print `backend_line()` there.
- Checked against item (p): the draft claims nothing about the fallback or the exit code, so nothing changes. F3
  changes only the label; a forced `--backend wgpu` run that falls back to the CPU still exits 14.

## Before opening (checklist)
- [ ] Rebase or merge onto post-cut main only if it conflicts (L17: no bulk update-branch). 2026-10-04: no conflict
  against 316dee2cd4 (Readiness above); re-run the merge-tree at open time.
- [ ] Open only when the repo is under its open-PR cap, checked at open time.
- [ ] Re-run the four commands above on intel at the PR head.
- [x] Run the aprender-serve lib suite to completion (2026-09-29 03:18Z: 16149 passed / 0 failed / 62 ignored at fd02954a8). Re-run at the PR head only if the branch moves.
- [ ] Darwin live line on mini when the train is inactive: expect `Backend: wgpu (Metal) adapter=Apple …`.
- [ ] Quorum lanes: Sonnet 5 + non-Claude (author is Opus 5.5, never an Opus lane).
