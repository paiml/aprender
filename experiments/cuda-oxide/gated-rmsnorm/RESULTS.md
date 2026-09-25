# #3522 OXIDE-001 O-1: cuda-oxide gated RMSNorm, pilot results

**VERDICT: GO on lambda, yoga and gx10.** Parity and timing pass for both variants
on every CUDA host, from a clean tree. yoga and gx10 were measured at `5630ddc8b`; lambda was
re-measured at `5b29ed2cd`, which changes only the harness exit status (kernels identical).
lambda's earlier 5630ddc8b run read 0.914 / 0.980.

A safe-Rust `#[kernel]` port of the hand-PTX `GatedRmsNormKernel`
(`crates/aprender-gpu/src/kernels/gdn/gated_rmsnorm.rs`, entry `gdn_gated_rmsnorm`),
the GDN output norm of Qwen3.5: head_dim 128, 16 heads, eps 1e-6.

- Source: `src/main.rs`. Run it with `./receipt.sh` (wraps `cargo oxide run`).
- cuda-oxide pinned at `b9847e9515ed3a23096f22567d3eaf0a6e3e440c`, with `cuda-core = 0.3.1` and nightly-2026-08-28.
- Receipts (`apr-kernel-receipt/v1`) are in `evidence/kernels/gdn_gated_rmsnorm/{noah-Lambda-Vector,yoga,gx10-a5b5}.json`.
- The hand-PTX baselines are `baseline-ptx/gdn_gated_rmsnorm_h128.{sm_89,sm_121}.ptx`. They are pinned to the
  shipped emitter by `gdn_gated_rmsnorm_ptx_golden` (aprender-gpu, `--features cuda`; runs with no GPU,
  so this is also mini's PTX-golden half).

## Safety shape (kernel-safety)

Both kernels are safe Rust: they contain no `unsafe` block and no raw pointer.
- Reads use bounds-checked slice indexing.
- The only write goes through `DisjointSlice<f32, LinearTiles<4>>::thread_run32`, so each lane owns 4 contiguous
  outputs, and the type system proves they are disjoint.
- The single `unsafe` in the crate is on the host side. It is the raw `launch_kernel_on_stream` call for the
  hand-PTX baseline.

O-2 (#3522) makes this a checked fact rather than prose. `evidence/kernels/gdn_gated_rmsnorm/kernel.json` declares
the kernel, and `contracts/kernel-receipt-v1.yaml` grades it:
- The `kernel-safety` shape reads the device module through a `syn` walk: helpers, impl and trait methods,
  extern blocks and macro tokens included.
- The `kernel-parity` and `kernel-timing` shapes read these receipts.
- The shapes are reported, not armed. See `pv lint contracts --gate shapes`.

## Results (worst ratio over heads 16/32/48; parity over heads 1/16/32/48)

| host | GPU | cc | variant | cos min | max\|Δ\| | oxide µs | hand µs | ratio | regs oxide/hand |
|---|---|---|---|---|---|---|---|---|---|
| lambda | RTX 4090 | sm_89 | exp | 1.0000000 | 1.43e-6 | 2.35 | 2.59 | 0.909 | 31 / 27 |
| lambda | RTX 4090 | sm_89 | ex2 | 1.0000000 | 1.43e-6 | 2.38 | 2.60 | 0.913 | 31 / 27 |
| yoga | RTX 4060 Laptop | sm_89 | exp | 1.0000000 | 1.43e-6 | 2.95 | 3.12 | 0.944 | 30 / 27 |
| yoga | RTX 4060 Laptop | sm_89 | ex2 | 1.0000000 | 1.43e-6 | 2.96 | 3.12 | 0.948 | 30 / 27 |
| gx10 | GB10 | sm_121 | exp | 1.0000000 | 1.43e-6 | 4.10 | 4.10 | 1.000 | 32 / 29 |
| gx10 | GB10 | sm_121 | ex2 | 1.0000000 | 1.43e-6 | 4.09 | 4.10 | 0.998 | 32 / 29 |

Gates: parity requires cos ≥ 0.9999 and max|Δ| < 1e-3 against an f64 CPU reference. Timing requires
oxide/hand ≤ 1.2, measured as the GPU-event median of 5 × 100 launches after 20 warmups. Both sides use the
same launch geometry: grid = heads, block = 32.

## What the timing does and does not show

At the production shape, the whole kernel is 2 KiB of input. Both implementations therefore sit on the
per-launch floor: about 2.4–2.6 µs on sm_89, and a quantised ~4.09 µs step on GB10. The ratio shows the
oxide kernel adds no launch or occupancy cost relative to the hand PTX. It cannot separate their arithmetic,
because that difference is below this measurement's resolution.

Registers differ by only 3–4 per thread: 30–32 for oxide against 27–29 for the hand PTX. At one warp per
block, neither count limits occupancy.

## Contaminated gx10 run (recorded, not used)

The first gx10 run reported ex2 NO-GO at heads 48 (ratio 1.50). During that run, the hand PTX itself moved
between 4.1 and 6.2 µs. `nvidia-smi` showed a foreign `apr v0.69.3` process on the GB10 that did not hold
`/tmp/apr-gpu.lock`. Three re-runs followed:
- Re-run 1: the process was still active. Rows jumped again, to 6.2 and 7.8 µs, and on both sides.
- Re-run 2: the process was exiting. Every row was at ~4.1 µs, with ratios of 0.997–1.002.
- Re-run 3: no foreign process. This is the committed receipt.

`receipt.sh` records `foreign_gpu_procs` so a contaminated receipt is visible on its face.

## Mutation proofs (restored by trap)

- **Parity gate.** Deleting the last warp-butterfly step (`shuffle_down … 1`) made all 8 oxide parity rows FAIL,
  and the harness exited 1.
- **Golden gate.** Changing one token in the sm_89 baseline (`ex2.approx.f32` → `ex2.approx.ftz.f32`) failed
  `gdn_gated_rmsnorm_ptx_golden` with "drifted from the emitter".
- **Timing gate** (at 5b29ed2cd): the sonnet quorum lane found that `timing_ok` never reached the exit status,
  so a NO-GO exited 0. It now exits 4. With `TIMING_RATIO_MAX` mutated to 0.5, `receipt.sh` exited 4 and still
  wrote the receipt with `"pass": false` on both variants. The unmutated run exited 0.
