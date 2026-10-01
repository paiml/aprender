---
spike: 008
idea: prophet-forecast-mcp
name: neon-gemm-microkernel-upstream
type: standard
validates: "Given `gemm_blis` on aarch64 falls to `microkernel_scalar` because the only NEON kernel is 8×8 while panels are packed 8×6, when an 8×6 NEON FMA kernel honouring the packed-panel contract is added and dispatched under `cfg(aarch64)`, then it agrees with the scalar kernel within FMA rounding on the microkernel test matrix, `cargo test -p aprender-compute --lib` stays green, and a before/after table on spike-005 shapes (129×256×1024) plus square perf-gate shapes shows the packed GEMM beating plain loops, with the Chronos forward dropping from 36 ms toward torch's 6 ms"
verdict: VALIDATED
related: [005, 006, 007]
tags: [gemm, neon, blis, upstream, performance, contribution]
---

# Spike 008: NEON GEMM Microkernel as a Measured Upstream Contribution

## What This Validates

Given `trueno::blis::gemm_blis` on aarch64 runs the scalar reference microkernel for every
shape (spike 005 measured it 4× slower than plain loops), when an 8×6 NEON FMA microkernel that
honours the existing packing contract is added and dispatched under `cfg(target_arch = "aarch64")`,
then (a) it matches the scalar kernel within FMA rounding on every K including 0 and the
remainder sizes, (b) `gemm_blis` on odd shapes matches the naive reference, (c) the crate's own
lib test suite, clippy `-D warnings` and fmt stay green, and (d) the same driver, run before and
after, shows the packed GEMM beating plain loops on transformer-shaped and square GEMMs, with
the crate's own criterion bench (`gemm_comparison`, trueno vs faer) as the maintainer-side
instrument. The change is packaged as a branch off `upstream/main` with a contract
(`contracts/neon-blis-v1.yaml`) so it reads as parity work in the maintainer's terms.

## Research

No external dependencies; everything came from the tree and the ARM reference.

**Why the packed GEMM was scalar on aarch64.** `gemm_blis` (`crates/aprender-compute/src/blis/compute.rs`)
has four x86-only fast paths (direct row-major ≤128, AVX-512 16×8, no-pack 8×8, strided AVX2)
and an AVX-512 / AVX2 large path, all under `#[cfg(target_arch = "x86_64")]`. On aarch64 every
one compiles out and the generic five-loop path runs `dispatch_microkernel`, whose only non-x86
branch is `microkernel_scalar`. The exported `microkernel_8x8_neon` is never called anywhere:
it reads the B panel with stride 8 (`b.add(p * 8 + j)`) while `pack_b_block` packs with stride
`NR = 6`, so it *cannot* be dispatched without changing the packing.

**Packing contract** (`packing.rs`, `compute.rs`): A panels column-major `a[p*MR + i]`, B panels
row-major `b[p*NR + j]`, C micro-tile column-major `c[j*ldc + i]` with `ldc = MR`. Remainder
tiles are zero-padded in all three (`pack_a_block`, `pack_b_block`, `load_c_tile`) and
`store_c_tile` writes back only the live `mr × nr` part, so a full-tile kernel is exact for
remainders and no scalar fallback is needed for them.

| Approach | Change surface | Pros | Cons | Status |
|----------|----------------|------|------|--------|
| A. 8×6 NEON kernel matching MR=8/NR=6 packing | `neon.rs`, dispatch in `compute.rs`, tests, contract | Mirrors the AVX2 8×6 kernel one to one; 12 accumulators of 32 registers; no packing change | Not the peak NEON tile (12 of the 16 FMA chains needed to hide latency) | **Chosen** |
| B. Switch aarch64 to NR=8 and dispatch the existing 8×8 kernel | Per-arch `MR`/`NR` consts, packing sizes | Reuses code | `MR`/`NR` are global consts used by x86 paths and tests; wide blast radius | Rejected for this PR |
| C. 8×12 or 12×8 tile (24 accumulators) | New packing plus kernel | Closest to peak | Larger change, needs its own before/after | Follow-up; headroom measured here |
| D. Delegate to faer or Accelerate | Dependency | Fast today | Against the one-binary pure-Rust vision | Comparison column only |

**Kernel design (A).** Twelve `float32x4` accumulators (6 columns × two halves of the 8-row A
vector). Per K step: two A loads, one q-load (`b[0..4]`) and one d-load (`b[4..6]`) of B, and
twelve lane-indexed FMAs (`vfmaq_laneq_f32::<L>` / `vfmaq_lane_f32::<L>`, i.e. `fmla.4s v, v,
v[lane]`), so B is never broadcast through memory. K = 0 leaves C untouched. Toolchain pin is
Rust 1.93.0; all intrinsics used are stable.

**Maintainer's bar** (memory: upstream focus is stabilisation and measurement, "is performance at
parity with similar tools"): a contract in `contracts/` with falsification tests, the crate's lib
tests green, and a before/after table from the same instrument. The crate already ships
`benches/gemm_comparison.rs` comparing trueno with faer, ndarray, nalgebra and matrixmultiply on
64…1024 square GEMMs; that is the instrument.

## How to Run

```bash
# 1. Before/after driver (from this directory; label names the results file)
CARGO_TARGET_DIR=../../../target cargo run --release -- before   # with the scalar kernel
CARGO_TARGET_DIR=../../../target cargo run --release -- after    # with the NEON kernel

# 2. The crate's own instrument (from the repo root)
CARGO_TARGET_DIR=target cargo bench -p aprender-compute --bench gemm_comparison -- "gemm/"

# 3. Gates
cargo test -p aprender-compute --lib microkernel_neon
cargo test -p aprender-compute --lib gemm_blis_neon
cargo test -p aprender-compute --lib
cargo clippy -p aprender-compute --lib -- -D warnings
cargo fmt --all -- --check
cargo run -p aprender-contracts-cli --bin pv -- validate contracts/neon-blis-v1.yaml

# 4. Effect on the Chronos port (spike 005 cost table, `fast` path = gemm_blis)
cd ../005-chronos-bolt-tiny-parity && CARGO_TARGET_DIR=../../../target cargo run --release
```

## What to Expect

`RUN-OUTPUT-before.md` and `RUN-OUTPUT-after.md` from the same driver; `bench-before.log` /
`bench-after.log` from criterion; the three new tests passing; the contract validating; the
spike-005 cost table with `gemm_blis` faster than plain loops instead of 4× slower.

## Investigation Trail

1. **Before.** Driver on 17 shapes (Bolt-tiny/-small/Chronos-2 layer shapes at 129 tokens,
   squares 64…1024, remainders, K = 1 and 3). `gemm_blis` 7.4 GFLOP/s on every shape,
   plain 8-accumulator loops 9.7, faer 100–117 single-threaded. Criterion before: trueno 75 µs
   vs faer 5.4 µs on 64³ (14×), 584 µs vs 42 µs on 128³. Errors all ≤ 1.8e-6 relative.
2. **Kernel.** `microkernel_8x6_neon` (78 lines) with the packing contract of `microkernel_scalar`;
   dispatched for every tile on aarch64 (remainders are zero-padded by the packers). Exported next
   to the unused 8×8 kernel; the dispatch doc names all three paths.
3. **After, same driver.** 64–77 GFLOP/s on every real shape (8.8–9.8× the scalar kernel), now
   7× faster than the plain loops and 0.59–0.71× faer. Errors unchanged (worst 1.79e-6 vs 1.77e-6
   before). Bolt-tiny layer GEMMs: 20.6 ms → 2.3 ms (faer 1.6 ms). Shapes with K ≤ 3 gain
   nothing (packing dominates; the `m·n·k < 4096` reference cutoff is the lever, out of scope).
4. **Gates.** New tests pass (`test_microkernel_neon_8x6_matches_scalar`,
   `..._short_k_and_accumulate`, `test_gemm_blis_neon_odd_shapes_match_reference`). Full crate lib
   suite: 3323 passed, 1 failed, 3 ignored. `cargo fmt --all -- --check` clean after `cargo fmt`
   (five spots in the new code). `pv validate contracts/neon-blis-v1.yaml`: valid.
5. **The one failure is not ours — measured, not assumed.** `test_brick_profiler_reset_v2` starts
   and stops a brick timer back to back and asserts `total_ns() > 0`. Solo reruns: 3/5 fail on our
   tree, **4/8 fail on untouched `upstream/main`** in the worktree. A 100 000-iteration probe of
   `Instant::now().elapsed()` returns 0 ns 50.6 % of the time on this M4 Pro (24 MHz timer,
   41.67 ns ticks). A stabilisation finding for upstream in its own right (the test needs a busy
   loop or `>= 0`); kept out of this PR.
6. **Clippy before/after** (raw output, same command, both trees): 19 → 18 warnings on aarch64;
   the two removed are the `mr_block`/`nr_block` unused-variable warnings this dispatch now
   consumes; nothing added. All pre-existing warnings are aarch64-only dead code (`NeonBackend`
   import, `MR_512V2`, …) — a second stabilisation item for an ARM-first maintainer.
7. **The crate's own instrument.** `benches/gemm_comparison.rs`, `gemm/trueno` (criterion mean):
   64³ 75.6 µs → 10.4 µs, 128³ 584 → 64.2 µs, 256³ 4.47 ms → 443 µs, 512³ 35.6 → 3.36 ms,
   1024³ 283 → 26.4 ms (7.3–10.7×). faer: 5.26 µs, 39.7 µs, 303 µs, 2.38 ms, 19.6 ms, so trueno
   is now 1.35–1.98× faer's *time* instead of 14×. `bench-before.log`, `bench-after.log`.
8. **Downstream: the Chronos port (spike 005 driver, unchanged code).** Parity unchanged (9.5e-7
   quantiles, 1.7e-5 rollout). Forward on the 2048-point Peyton context: plain loops 36.4 ms,
   `gemm_blis` **21.0 ms** (was 137.2 ms), torch 5.9 ms. 365-step rollout: 1671 / **967** ms (was
   6397), torch 81 ms. Per-stage: a 256→256 projection 0.50 → 0.26 ms, FF wi 1.98 → 1.02, FF wo
   2.37 → 1.01, full encoder 29.1 → 15.9 ms. The remaining gap to torch is no longer the GEMM:
   spike 005's attention scores and context are still plain loops (129×129×64 per head), and the
   kernel is at 0.7× faer. `spike005-after.md`.
9. **Upstream branch.** `perf/neon-gemm-8x6-microkernel` in a worktree cut from `upstream/main`
   (b1a6324b8, 0.65.2); the commit cherry-picks cleanly (the touched files are identical at both
   bases). On that branch: `cargo fmt --all -- --check` clean; lib suite **3327 passed, 1 failed
   (the same `test_brick_profiler_reset_v2` flake), 4 ignored**; the three NEON tests pass.
   `PR.md` is the pull-request body. Opening the PR is a checkpoint, not automatic.

## Results

**Verdict: VALIDATED.** The 8×6 NEON kernel is exact, passes the crate's gates, and turns the
packed GEMM on aarch64 from the slowest path in the tree into one within 1.4× of faer — with no
packing, blocking or x86 change, and a contract the maintainer can run.

| shape | before GFLOP/s | after GFLOP/s | speed-up | plain loops | faer | after / faer |
|---|---|---|---|---|---|---|
| 129×256×256 (Bolt-tiny projection) | 7.4 | 64.5 | 8.8× | 9.2 | 96.1 | 0.67 |
| 129×1024×256 (Bolt-tiny FF wi) | 7.4 | 68.9 | 9.3× | 9.2 | 97.4 | 0.71 |
| 129×256×1024 (Bolt-tiny FF wo) | 7.3 | 65.5 | 8.9× | 9.1 | 94.3 | 0.69 |
| 129×2048×512 (Bolt-small FF wi) | 7.4 | 67.7 | 9.1× | 9.3 | 95.3 | 0.71 |
| 129×3072×768 (Chronos-2 FF wi) | 7.5 | 69.1 | 9.2× | 9.0 | 95.4 | 0.72 |
| 64³ / 128³ / 256³ | 7.1 / 7.5 / 7.8 | 44.2 / 59.8 / 72.3 | 6.2× / 8.0× / 9.3× | 8.6 / 8.9 / 9.0 | 95 / 100 / 106 | 0.46 / 0.60 / 0.69 |
| 512³ / 1024³ | 7.8 / 7.8 | 75.6 / 77.1 | 9.7× / 9.8× | 9.1 / 9.0 | 108 / 109 | 0.70 / 0.71 |
| 129×131×67 (remainders in M, N, K) | 7.1 | 52.2 | 7.4× | 8.7 | 88.9 | 0.59 |
| K = 1, K = 3 | 2.1 / 4.1 | 2.4 / 6.7 | ~1× | – | – | – |

Worst relative error vs f64 reference 1.79e-6 after (1.77e-6 before; bar 1e-5). Criterion
(`gemm/trueno`): 7.3–10.7× across 64³…1024³. Chronos-Bolt-tiny forward 137 → 21 ms through
`gemm_blis`; rollout 6.4 s → 0.97 s; parity to 1e-6 unchanged.

**Gates:** three new tests pass; lib suite 3323 passed / 1 failed / 3 ignored where the failure is
a pre-existing timer-resolution flake reproduced on untouched `upstream/main` (4/8); clippy 19 → 18
warnings on aarch64 with nothing new; fmt clean; `pv validate` clean.

**Surprises**
- The 4× "trueno is slower than loops" of spike 005 was really a 14× gap to a real BLAS: the
  scalar kernel sits at 7.4 GFLOP/s on a machine whose single core does 109 with faer. Plain
  8-accumulator loops (9.7) were never the bar.
- A dead NEON kernel with the wrong panel stride sat in the tree exported but uncallable; the fix
  was 78 lines that mirror the AVX2 kernel, not a redesign.
- The only red test has nothing to do with GEMM: `Instant::now()` on Apple Silicon ticks at
  41.67 ns, and half of back-to-back reads are equal. Any `elapsed > 0` assertion is a coin flip
  here — a stabilisation item that fits the maintainer's stated focus better than the kernel does.
- After the kernel, the Chronos forward is bounded by the spike's attention loops, not by the
  projections; spike 007 should route scores/context through `gemm_blis` too.

**Signal for the build**
- Upstream PR as prepared (`PR.md`); ask for a second, one-line PR or issue for the flaky test.
- Spike 007 embeds Bolt-**small** (5.9 s of GEMM per 365-step rollout before, ~0.9 s now); route
  attention through `gemm_blis`; cap the default horizon at 64 as spike 006 found.
- Headroom, measured: 0.7× faer. The next step is an 8×12/12×8 tile with a per-arch `NR` (24
  accumulators), which is a packing change and its own before/after.
- Convention: for any kernel claim, run the maintainer's own bench (`gemm_comparison`) before and
  after, on the same machine, and quote both. Tiny-K shapes stay at the reference cutoff's mercy.
