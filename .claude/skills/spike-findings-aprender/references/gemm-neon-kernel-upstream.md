# NEON GEMM Microkernel (aarch64) and the Parity-Work Protocol for Core Changes

`trueno::blis::gemm_blis` ran the scalar reference microkernel for every shape on aarch64
(7.4 GFLOP/s; 14× slower than faer). An 8×6 NEON FMA kernel honouring the existing packing
contract takes it to 65–77 GFLOP/s (0.7× faer) with no packing, blocking or x86 change, and the
Chronos-Bolt forward from 137 to 21 ms (spike 008). Packaged as an upstream PR with a contract.

## Requirements

From the `prophet-forecast-mcp` idea (MANIFEST.md):

- **Build order 008 → 007 → 009 → 010.** The 008 kernel lives in `crates/aprender-compute` on a
  branch cut from `upstream/main`; **opening the PR is a checkpoint, not automatic.**
- Every Chronos port routes its GEMMs through this kernel; the server sizing in
  `chronos-mcp-server.md` assumes it.

## How to Build It

**State of the tree (verified 2026-09-05):** the kernel is already present in this checkout —
`crates/aprender-compute/src/blis/microkernels/neon.rs` (`microkernel_8x6_neon`), dispatched in
`crates/aprender-compute/src/blis/compute.rs` under `cfg(target_arch = "aarch64")`, tests in
`crates/aprender-compute/src/blis/tests/microkernel.rs`, contract `contracts/neon-blis-v1.yaml`.
The upstream PR branch is `perf/neon-gemm-8x6-microkernel` in the worktree
`~/Development/machine-learning/aprender-neon-upstream` (cut from `upstream/main` b1a6324b8,
0.65.2); PR body is `sources/008-neon-gemm-microkernel-upstream/PR.md`.

**1. The kernel** — the aarch64 twin of the AVX2 8×6 kernels, 78 lines:

```rust
#[cfg(target_arch = "aarch64")]
pub unsafe fn microkernel_8x6_neon(k: usize, a: *const f32, b: *const f32, c: *mut f32, ldc: usize) {
    use std::arch::aarch64::*;
    // 12 float32x4 accumulators: 6 columns × two halves of the 8-row A vector
    let mut c00 = vld1q_f32(c); let mut c01 = vld1q_f32(c.add(4));
    let mut c10 = vld1q_f32(c.add(ldc)); /* … c50, c51 */
    for p in 0..k {
        let a0 = vld1q_f32(a.add(p * 8));      // A panel: column-major, stride MR = 8
        let a1 = vld1q_f32(a.add(p * 8 + 4));
        let bq = vld1q_f32(b.add(p * 6));      // B panel: row-major, stride NR = 6 → b[p][0..4]
        let bd = vld1_f32(b.add(p * 6 + 4));   //                                   b[p][4..6]
        c00 = vfmaq_laneq_f32::<0>(c00, a0, bq); c01 = vfmaq_laneq_f32::<0>(c01, a1, bq);
        c10 = vfmaq_laneq_f32::<1>(c10, a0, bq); /* … lanes 2,3 from bq; lanes 0,1 from bd via vfmaq_lane_f32 */
    }
    // store back the 12 accumulators; K = 0 leaves C untouched
}
```

Packing contract it honours (`packing.rs`, `compute.rs`): A panels column-major `a[p*MR + i]`,
B panels row-major `b[p*NR + j]`, C micro-tile column-major `c[j*ldc + i]` with `ldc = MR`.
Remainder tiles are zero-padded by `pack_a_block` / `pack_b_block` / `load_c_tile` and
`store_c_tile` writes back only the live `mr × nr` part — so a full-tile kernel is exact for
remainders and **no scalar fallback is needed**. Toolchain pin Rust 1.93.0; all intrinsics stable.

**2. Gates** (all green except one pre-existing flake):

```bash
cargo test -p aprender-compute --lib microkernel_neon      # kernel vs scalar, K ∈ {0,1,2,3,4,5,7,9,255,256,257}, non-zero C
cargo test -p aprender-compute --lib gemm_blis_neon        # gemm_blis vs gemm_reference on 9 odd shapes
cargo test -p aprender-compute --lib                       # 3327 passed / 1 failed (flake, see below) / 4 ignored
cargo clippy -p aprender-compute --lib -- -D warnings      # 19 → 18 aarch64 warnings, none new
cargo fmt --all -- --check
cargo run -p aprender-contracts-cli --bin pv -- validate contracts/neon-blis-v1.yaml
```

**3. The parity-work protocol — how any kernel or core change is presented to the maintainer**
(upstream focus: stabilisation and measurement, "is performance at parity with similar tools"):

1. **Measure with the maintainer's own instrument** on the same machine, before and after:
   `cargo bench -p aprender-compute --bench gemm_comparison -- "gemm/"` (trueno vs faer, ndarray,
   nalgebra, matrixmultiply on 64…1024 squares). Quote both columns.
2. **Run the control on untouched `upstream/main`** in a git worktree
   (`git worktree add ../aprender-<topic> -b <branch> upstream/main`); cherry-pick the commit there.
3. **Diff raw clippy output between the trees** (`rtk proxy cargo clippy …` to see the full list).
4. **Ship a `contracts/*.yaml` with falsification tests** (`C-NEON-BLIS-001..003`,
   `FALSIFY-NEON-BLIS-001..003`) and tests that exercise them.
5. **A flaky test is reported with its solo-run failure rate on both trees**, never attributed to
   the change, and kept out of the PR.
6. Write the PR body as problem → change → before/after table → gates → measured headroom
   (`PR.md` is the template).

**4. Where the time goes after the kernel** (Bolt-tiny, 2048-point context): plain loops 36 ms →
`gemm_blis` 21 ms (was 137 ms); the encoder is 80 % of the forward and sits at the kernel's rate;
attention scores/context via `gemm_blis` bought 1 ms. The remaining gap to torch (5.9 ms) is the
kernel's 0.7× faer, not routing.

## What to Avoid

- **Do not dispatch the pre-existing `microkernel_8x8_neon`** — it reads B with stride 8 while
  packing uses `NR = 6`; it is exported but uncallable without a packing change (dead code).
- **Do not change `MR`/`NR` per arch in this PR** — they are global consts used by x86 paths and
  tests; wide blast radius. That is the 8×12 / 12×8 follow-up (24 accumulators, needs its own
  packing and its own before/after).
- **Do not attribute `test_brick_profiler_reset_v2` to the kernel.** It asserts `total_ns() > 0`
  after a back-to-back start/stop; `Instant::now()` on Apple Silicon ticks at 41.67 ns (24 MHz) and
  back-to-back reads are equal 50.6 % of the time. Fails 4/8 solo on untouched `upstream/main`.
  Separate stabilisation issue: busy loop or `>= 0`.
- **Do not benchmark plain loops as "the bar"** — they were 9.7 GFLOP/s on a core that does 109
  with faer; spike 005's "trueno 4× slower than loops" was a 14× gap to a real BLAS.
- **Do not expect gains on K ≤ 3** — packing dominates; the `m·n·k < 4096` reference cutoff is a
  separate lever.
- **Do not open the upstream PR without the checkpoint** — MANIFEST requirement.

## Constraints

| shape | before GFLOP/s | after | speed-up | plain loops | faer | after / faer |
|---|---|---|---|---|---|---|
| 129×256×256 (Bolt-tiny projection) | 7.4 | 64.5 | 8.8× | 9.2 | 96.1 | 0.67 |
| 129×1024×256 (Bolt-tiny FF wi) | 7.4 | 68.9 | 9.3× | 9.2 | 97.4 | 0.71 |
| 129×3072×768 (Chronos-2 FF wi) | 7.5 | 69.1 | 9.2× | 9.0 | 95.4 | 0.72 |
| 64³ / 256³ / 1024³ | 7.1 / 7.8 / 7.8 | 44.2 / 72.3 / 77.1 | 6.2× / 9.3× / 9.8× | 8.6 / 9.0 / 9.0 | 95 / 106 / 109 | 0.46 / 0.69 / 0.71 |
| 129×131×67 (remainders M, N, K) | 7.1 | 52.2 | 7.4× | 8.7 | 88.9 | 0.59 |
| K = 1 / K = 3 | 2.1 / 4.1 | 2.4 / 6.7 | ~1× | – | – | – |

- Criterion `gemm/trueno`: 64³ 75.6 → 10.4 µs, 256³ 4.47 ms → 443 µs, 1024³ 283 → 26.4 ms
  (7.3–10.7×); faer 5.26 µs / 303 µs / 19.6 ms. Worst relative error vs f64 reference 1.79e-6
  (bar 1e-5), unchanged.
- The rayon `blis::gemm` (feature `parallel`) splits M only in `MC = 128`-row blocks
  (`blis/parallel.rs`); ≤ 1.4× on transformer shapes below ~500 tokens, 2.1× at 517. Partitioning
  along N for M ≤ 256 (LLM prefill shapes) is the next upstream issue, with spike 009's table.
- Measured on Apple M4 Pro, single thread. Logs: `bench-before.log` / `bench-after.log`,
  `test-*-raw.log` in `.planning/spikes/008-neon-gemm-microkernel-upstream/` (not copied).

## Origin

Synthesized from spike: 008 (downstream effect measured on 005/007/009)
Source files available in: `sources/008-neon-gemm-microkernel-upstream/` (main.rs driver, PR.md, tools/report.py,
RUN-OUTPUT-before.md, RUN-OUTPUT-after.md, spike005-after.md, results-*.json)
In-tree: `crates/aprender-compute/src/blis/microkernels/neon.rs`, `contracts/neon-blis-v1.yaml`
