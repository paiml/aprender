# perf(compute): NEON 8×6 microkernel for `gemm_blis` on aarch64

## Problem

On aarch64 every fast path in `gemm_blis` is `#[cfg(target_arch = "x86_64")]`, so the generic
five-loop BLIS path runs `microkernel_scalar` for every shape. The one NEON kernel in the tree,
`microkernel_8x8_neon`, is exported but never called: it reads the B panel with stride 8 while
`pack_b_block` packs with stride `NR = 6`, so it cannot be dispatched without changing the
packing. Measured on an Apple M4 Pro (single thread), `gemm_blis` ran at **7.4 GFLOP/s** on every
shape from 64³ to 1024³ — 1.3× *slower* than a plain 8-accumulator loop and 14× slower than
`faer`, which the crate's own `benches/gemm_comparison.rs` compares against.

## Change

- `blis/microkernels/neon.rs`: `microkernel_8x6_neon` — the aarch64 twin of the AVX2 8×6
  kernels, honouring the existing packing contract (A column-major stride `MR`, B row-major
  stride `NR`, C column-major with `ldc`). Twelve `float32x4` accumulators; per K step two A
  loads, one q-load plus one d-load of B, and twelve lane-indexed FMAs (`fmla.4s v, v, v[lane]`),
  so B is never broadcast through memory. K = 0 leaves C untouched.
- `blis/compute.rs`: `dispatch_microkernel` uses it under `cfg(target_arch = "aarch64")`.
  Packed panels and the C micro-tile are already zero-padded to a full `MR × NR` tile and
  `store_c_tile` writes back only the live part, so no remainder fallback is needed. No packing,
  blocking constant or x86 path changes.
- `contracts/neon-blis-v1.yaml` (`C-NEON-BLIS-001..003`, `FALSIFY-NEON-BLIS-001..003`) and three
  tests in `blis/tests/microkernel.rs` (kernel vs scalar for K ∈ {0,1,2,3,4,5,7,9,255,256,257}
  accumulating into non-zero C; `gemm_blis` vs `gemm_reference` on nine odd shapes including
  remainders in M, N and K and K < 4).
- Removes the two aarch64-only `unused variable` warnings on `mr_block` / `nr_block`.

## Before / after (Apple M4 Pro, single thread, same driver, median of repeated runs)

| shape | before GFLOP/s | after GFLOP/s | speed-up | faer GFLOP/s | after / faer |
|---|---|---|---|---|---|
| 129×256×256 (Chronos-Bolt-tiny projection) | 7.4 | 64.5 | 8.8× | 96.1 | 0.67 |
| 129×1024×256 (Bolt-tiny FF wi) | 7.4 | 68.9 | 9.3× | 97.4 | 0.71 |
| 129×256×1024 (Bolt-tiny FF wo) | 7.3 | 65.5 | 8.9× | 94.3 | 0.69 |
| 129×2048×512 (Bolt-small FF wi) | 7.4 | 67.7 | 9.1× | 95.3 | 0.71 |
| 256³ | 7.8 | 72.3 | 9.3× | 105.5 | 0.69 |
| 512³ | 7.8 | 75.6 | 9.7× | 108.3 | 0.70 |
| 1024³ | 7.8 | 77.1 | 9.8× | 109.1 | 0.71 |
| 129×131×67 (remainders in M, N, K) | 7.1 | 52.2 | 7.4× | 88.9 | 0.59 |

Max relative error vs an f64 reference is unchanged (1.8e-6 before, 1.8e-6 after; bar 1e-5).

`benches/gemm_comparison.rs`, `gemm/trueno` vs `gemm/faer` (criterion mean; faer column from the *after* run, the before run had it within 4 %):

| n | trueno before | trueno after | faer |
|---|---|---|---|
| 64 | 75.6 µs | 10.4 µs (7.3×) | 5.26 µs |
| 128 | 584 µs | 64.2 µs (9.1×) | 39.7 µs |
| 256 | 4.47 ms | 443 µs (10.1×) | 303 µs |
| 512 | 35.6 ms | 3.36 ms (10.6×) | 2.38 ms |
| 1024 | 283 ms | 26.4 ms (10.7×) | 19.6 ms |

Downstream effect (Chronos-Bolt-tiny port, 2048-point context, 129 tokens, same parity to 1e-6): encoder forward 137 ms → **21 ms** through `gemm_blis` (plain loops 36 ms; torch 5.9 ms), 365-step rollout 6.4 s → 0.97 s.

## Gates

- `cargo test -p aprender-compute --lib` on this branch (aarch64, M4 Pro) — 3327 passed, 1 failed, 4 ignored. The failure is `brick::tests::profiler::test_brick_profiler_reset_v2`, a pre-existing timer-resolution flake: it starts and stops a brick timer with nothing in between and asserts `total_ns() > 0`; on untouched `upstream/main` it fails 4 of 8 solo runs on the same machine, because back-to-back `Instant::now()` reads are equal about half the time on Apple Silicon (24 MHz timer, 41.67 ns ticks). No GEMM is involved; reported separately.
- `cargo clippy -p aprender-compute --lib` — 19 → 18 warnings on aarch64, none in the new code
  (the two removed are the `mr_block` / `nr_block` unused-variable warnings); x86 output unchanged
  since every new line is `cfg(aarch64)`.
- `cargo fmt --all -- --check` — clean.
- `pv validate contracts/neon-blis-v1.yaml` — `Contract is valid` (0 errors, 0 warnings)

## Not in this PR (measured headroom)

The 8×6 tile keeps 12 FMA chains in flight where Apple's four FMA pipes with 4-cycle latency want
16, which is consistent with landing at ~0.7× faer. An 8×12 or 12×8 tile (24 accumulators) needs
a per-arch `NR` and its own packing, and is the next step if the remaining 1.4× matters. Shapes
with K ≤ 3 stay at the reference cutoff's mercy (packing dominates); raising the
`m * n * k < 4096` cutoff for tiny K is a separate change.

Reproduction: `.planning/spikes/008-neon-gemm-microkernel-upstream/` on the fork (driver,
`RUN-OUTPUT-before.md`, `RUN-OUTPUT-after.md`, criterion logs).
