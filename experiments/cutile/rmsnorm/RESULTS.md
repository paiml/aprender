# T3 — cutile-rs RMSNorm vs hand-PTX on GB10 (sm_121)

**VERDICT: PARITY-CORRECT, NOT FASTER. No adoption recommended for 0.67.**
cutile is **1.03–1.31× SLOWER** than the equivalent hand-PTX kernel at every shape,
converging to parity at the largest hidden size. It is bit-parity-correct everywhere.

Run 2026-09-09 on `gx10-a5b5` — NVIDIA GB10, compute_cap 12.1 (sm_121), driver 590.48.01,
CUDA **13.3** (installed the same day for this experiment; the fleet max had been 13.0,
below cutile's 13.1 floor), stable rustc 1.95, `cutile` 0.3.1 from crates.io.

## Headline

| shape | comparand | cutile | hand-PTX | ratio |
|---|---|---:|---:|---|
| rows=1 h=2048 | `VectorizedRmsNormKernel` | 7.7 µs | 5.9 µs | **1.31× slower** |
| rows=1 h=4096 | `VectorizedRmsNormKernel` | 10.1 µs | 8.3 µs | **1.22× slower** |
| rows=1 h=8192 | `VectorizedRmsNormKernel` | 12.9 µs | 12.5 µs | **1.03× slower** |
| rows=8 h=2048 | `BatchedVectorizedRmsNormKernel` | 8.1 µs | 6.1 µs | **1.32× slower** |
| rows=8 h=4096 | `BatchedVectorizedRmsNormKernel` | 10.2 µs | 8.5 µs | **1.20× slower** |
| rows=8 h=8192 | `BatchedVectorizedRmsNormKernel` | 13.2 µs | 12.7 µs | **1.04× slower** |

Parity gate — cos ≥ 0.9999 AND maxdiff < 1e-4 vs an **f64 CPU reference**, all paths, both
shapes: **PASS**. Every path reported `cos = 1.0000000`, maxdiff 1.19e-7 … 2.38e-7.

## Two false wins this experiment produced before the comparand was fixed

Both were real measurements. Both were wrong about what they measured. They are recorded
because the retracted "cuda-oxide is 1.45× faster" claim (PR #2045, it raced
`TiledQ4KGemv` instead of the production `HwDp4a`) came from exactly this mistake, and it
recurred **twice more** here inside one afternoon.

| claim measured | why it was wrong |
|---|---|
| "cutile is **5.9× faster**" | raced `RmsNormKernel`, a **32-thread** kernel, with a 256-thread Tile kernel. That measures occupancy, not Tile IR. Against the 256-thread `VectorizedRmsNormKernel` the same code is 1.03–1.31× *slower*. |
| "cutile is **7.7× faster** batched" | raced cutile's single 8-row launch against **8 sequential** single-row launches. `BatchedVectorizedRmsNormKernel` exists, does all 8 rows in one launch via `%ctaid.y`, and is 1.04–1.32× *faster* than cutile. |

A third error was caught before it became a claim: the first run reported the hand-PTX
side at `cos = 0.35` and it would have been easy to call that a hand-PTX defect. `0.35 ≈
1/√8` with 8 rows is the signature of **one row written and seven left zero** — the
committed baseline uses `%tid.x` only, with no `%ctaid`, so it is a single-row kernel and
`grid=(8,1,1)` had eight blocks racing on row 0. The kernel was fine; the harness was not.

## What was actually compared

| id | kernel | threads/row | row dispatch | source |
|---|---|---|---|---|
| `cutile` | `#[cutile::entry] rms_norm<N, BLOCK>` | 256 (BLOCK) | partition `[1, n]` | `src/main.rs`, adapted from upstream `cutile-examples/examples/rms_norm.rs` |
| `vec256` | `VectorizedRmsNormKernel` | 256 | none (`%tid.x`) | `baseline-ptx/rmsnorm_vec_h*.sm121.ptx` |
| `batch8` | `BatchedVectorizedRmsNormKernel` (batch=8) | 256 | `%ctaid.y`, grid (1,M,1) | `baseline-ptx/rmsnorm_batch8_h*.sm121.ptx` |
| `warp32` | `RmsNormKernel` | 32 | none (`%tid.x`) | `../../cuda-oxide/rmsnorm/baseline-ptx/` — **shared with the cuda-oxide run**, by path not by copy, so both experiments have one comparand in common |

Cross-check against the cuda-oxide RMSNorm run (PMAT-893), which used `warp32`: it measured
that kernel at 22.55 / 38.95 / 75.82 µs; this harness measures 20.6 / 40.0 / 77.4 µs. The
two independent harnesses agree to within ~9%, which is what makes the shared comparand
worth keeping.

## Methodology

- Deterministic LCG inputs — no `rand` dependency, identical bytes on every run and host,
  so a parity difference is never the data's fault.
- Oracle is **f64 on the CPU**, a different precision from both GPU paths, so it is an
  independent reference rather than a re-run of one of them.
- Timing: wall-clock around `INNER=50` launches with one sync, `OUTER=5` repeats, median.
  Both sides upload once **outside** the loop and re-launch in place — kernel + launch
  overhead, not PCIe. The oxide run used CUDA-event medians; cutile drives its own stream
  through `sync_on`, so one event pair cannot bracket both paths identically. The same
  method is applied to both sides, which matters more than which method it is.
- **Not corrected for:** cutile's per-launch cost includes its own host-side builder; the
  hand-PTX side is a bare `cuLaunchKernel`. That asymmetry is part of what a Tile-track
  launch costs today, so it is reported rather than subtracted.

## What this says about the Tile track

- **Correctness is not the question.** cutile matched an f64 reference to 2.4e-7 at every
  shape, first try, with no tuning — and with no `%ctaid` arithmetic, no shared-memory
  reduction, and no GH-480 rewriter, in ~25 lines of safe Rust the compiler owns.
- **Performance is a wash at this kernel.** 1.03–1.31× slower, converging to ~parity as
  hidden grows. RMSNorm is memory-bound and trivially parallel, so this is close to the
  best case for a high-level abstraction; it is *not* evidence about GEMM or quantized
  matvec, which is where aprender's hand-PTX actually earns its keep (and where cuda-oxide
  measured **4× slower** than `HwDp4a`).
- **The toolchain story is the real difference from cuda-oxide.** Everything here is
  published crates.io releases on **stable** Rust — no nightly, no LLVM-21, no
  `cargo-oxide`, no git dependencies. `experiments/cuda-oxide/*` can never be a dependency
  of a published crate; this crate could be, if the numbers ever justified it. They do not
  yet.

## Reproduce

```bash
# on gx10 (needs CUDA >= 13.1 for sm_121; 13.3 installed 2026-09-09)
cd experiments/cutile/rmsnorm && cargo run --release
```

Regenerate the hand-PTX baselines (any host with the aprender workspace):

```rust
VectorizedRmsNormKernel::new(h).with_epsilon(1e-5).emit_ptx_for_target("sm_121")
BatchedVectorizedRmsNormKernel::new(h, 8).with_epsilon(1e-5).emit_ptx_for_target("sm_121")
```

## Incidental finding — PTX emitter output is not byte-stable

`rmsnorm_vec_h2048` and `rmsnorm_vec_h8192` differ in the **order of their `.reg`
declarations** (`.reg .u32 %r<17>` before vs after `.reg .f32 %f<26>`), not only in the
baked-in hidden size. Register *declaration* order is semantically irrelevant, so this is
not a correctness bug — but it means the emitter's output is not a pure function of its
inputs in a byte-comparable sense, which defeats a `sha256`-based "did the PTX change?"
check and perturbs the cubin cache key (`sha256(patched_ptx ‖ target ‖ driver)`), causing
avoidable JIT recompiles. Worth its own issue; not chased here.
