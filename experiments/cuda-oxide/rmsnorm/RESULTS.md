# PMAT-893 — cuda-oxide RMSNorm port: gx10 sm_121 A/B vs hand-PTX

**VERDICT: GO** (F-OXIDE-RMSNORM-PARITY-001 PASS at every hidden size, fair single-row gate.)

A pure-Rust `#[kernel]` port of the hand-PTX `RmsNormKernel`
(`crates/aprender-gpu/src/kernels/layernorm/rmsnorm.rs`, entry `rmsnorm`) was
authored, built, and run on the GB10 Blackwell (sm_121 / compute_cap 12.1) via
cuda-oxide. It is **bit-parity-correct** (cos = 1.0000000) and **matches-or-beats**
the hand-PTX at every hidden size, with NO hand-PTX and NO GH-480 JIT workaround.

- Source: `experiments/cuda-oxide/rmsnorm/src/main.rs`
- Build/run on gx10: `cargo oxide run` (nightly-2026-04-03 + LLVM-21.1.8 + cargo-oxide 0.2.1)
- GPU: NVIDIA GB10, compute_cap 12.1 (sm_121a), CUDA 12.1
- Hand-PTX baselines: `experiments/cuda-oxide/rmsnorm/baseline-ptx/rmsnorm_h{2048,4096,8192}.sm121.ptx`
  (emitted via `RmsNormKernel::new(h).with_epsilon(1e-5).emit_ptx_for_target("sm_121")`)

## What it computes

For each row r of width `hidden`:

```
mean_sq   = mean_i( x[r,i]^2 )
rms_inv   = 1 / sqrt(mean_sq + eps)          (eps = 1e-5)
out[r,i]  = x[r,i] * rms_inv * gamma[i]
```

This is the exact serve `rmsnorm` math (single-warp `RmsNormKernel` dispatched by
`crates/aprender-serve/src/cuda/executor/layers/rmsnorm.rs`). RMSNorm is pure f32
FMA + a warp-shuffle reduce + rsqrt — **ZERO DP4A** — so it is squarely the
PMAT-882 GO class (FMA/softmax wins; DP4A-bound Q4K GEMV/FFN is the NO-GO class).

## Two oxide kernel variants

| | structure | role |
|---|---|---|
| **(A) `rmsnorm_warp`** | 1 warp (32 threads) / row, shfl.down reduce | faithful hand-PTX analog (matched 1-warp/row) |
| **(B) `rmsnorm_block`** | 256 threads (8 warps) / row, SMEM cross-warp reduce | occupancy analog of `VectorizedRmsNormKernel`; the GO candidate at large hidden |

## Falsifiable target 1 — PARITY: PASS (both kernels, all sizes)

rows=8, hidden in {2048, 4096, 8192}, vs an f64 CPU reference. Required: cos >= 0.9999
AND maxdiff < 1e-4. Result (every config): **cos = 1.0000000, maxdiff 2.4e-7 .. 4.8e-7
=> PASS**.

## Falsifiable target 2 — PERF: GO (FAIR single-row hand-PTX A/B on GB10)

**Methodology honesty.** The hand-PTX `rmsnorm` is a SINGLE-ROW, SINGLE-BLOCK (1
warp) kernel — exactly one row per launch (how the serve executor calls it per
decode token). The PRIMARY gate is a like-for-like **rows=1** A/B: each side does
exactly ONE grid launch of ONE block, so the ratio is per-row kernel throughput
with no launch-count confound. Both verified parity-correct vs the CPU reference
inside the harness. GPU-event timing, median of 5x100 warm launches.

`ratio = oxide_us / handPTX_us <= 1.2 = GO`:

| hidden | oxide A (1 warp) µs | ratio A | oxide B (256 thr) µs | ratio B | hand-PTX (1 warp) µs | best | parity |
|---|---|---|---|---|---|---|---|
| 2048 | 14.35 | 0.636x | **4.11** | **0.182x** | 22.55 | B 0.182x | cos=1.0000000 md=2.4e-7 |
| 4096 | 27.31 | 0.701x | **6.15** | **0.158x** | 38.95 | B 0.158x | cos=1.0000000 md=2.4e-7 |
| 8192 | 49.98 | 0.659x | **8.55** | **0.113x** | 75.82 | B 0.113x | cos=1.0000000 md=4.8e-7 |

- The **matched** oxide kernel (A, same 1-warp/row structure as the hand-PTX) is
  **0.64–0.70x (1.4–1.5x faster)** at every hidden — a real per-kernel win, not a
  launch-count artifact (the hand-PTX's `div.f32` mean + `rsqrt.approx` loop is
  tighter in the oxide lowering).
- The **256-thread** oxide kernel (B) wins **5.5–8.9x** by retiring the single-warp
  occupancy starvation at large hidden — the GO candidate to wire into serve.
- hand-PTX parity PASS (cos=1.0000000) at every size.

**Perf gate (<= 1.2x): PASS at every size. GO.**

### Secondary (context only, NOT the gate) — 8-row throughput

The oxide kernel does all 8 rows in ONE grid launch; the hand-PTX must relaunch
8x (single-row ABI, no blockIdx row dispatch). This shows a 40–65x batched-decode
"fewer-launch" win, but it is launch-overhead-dominated and we do **NOT** claim it
as a per-kernel speedup. It is reported only to quantify the real-world benefit of
a single grid launch over N per-token relaunches.

| hidden | oxide best (1 launch) µs | hand-PTX (8 launches) µs | fewer-launch win |
|---|---|---|---|
| 2048 | 4.10 | 163.93 | 40.0x |
| 4096 | 6.15 | 312.11 | 50.8x |
| 8192 | 9.28 | 606.52 | 65.4x |

## Why this is a GO (consistent with PMAT-882, opposite of PMAT-881)

PMAT-881 (FFN-fusion) was a NO-GO because the production FFN is Q8_1 **DP4A**
integer math (4 MACs/instr on dedicated hardware) that f32 oxide can't match.
RMSNorm — like attention (PMAT-882 GO) — is **f32 FMA + warp-reduce + rsqrt**, NOT
DP4A-bound, and additionally memory-bandwidth-bound. The oxide port competes on
equal terms and wins, exactly as predicted. The value is twofold: (1) a genuine
per-kernel speedup at the matched structure, plus a large occupancy win at the
256-thread structure; (2) **retiring the hand-PTX and the GH-480 Blackwell-JIT
workaround** for the norm hot path.

## VERDICT: GO — promote into the serve CUDA executor next (follow-up only)

The pure-Rust cuda-oxide RMSNorm kernel is bit-parity-correct on Blackwell sm_121
and matches-or-beats the hand-PTX `RmsNormKernel` at every size. This spike does
NOT migrate production trueno kernels — that is the follow-up after this GO.

## Exact next step (follow-up ticket)

1. Emit standalone embeddable PTX for `rmsnorm_block` via `cargo oxide pipeline`
   (as q4k-matvec did) -> `include_str!` -> `CudaModule::from_ptx`
   (`crates/aprender-gpu/src/driver/module.rs`), raw-pointer ABI
   `(x, gamma, out: *…, hidden, eps)` (single grid launch, blockIdx = row).
2. Add a 3-way parity gate (oxide PTX vs hand-PTX vs CPU) on gx10 (no sm_121 CI
   runner; gx10-manual like PMAT-734/882).
3. Confirm the live-serve per-row RMSNorm dispatch maps directly (it does — the
   kernel uses the same `[rows, hidden]` layout + shared gamma).
4. Measure end-to-end decode tok/s with the oxide RMSNorm kernel vs default on a
   real model on Blackwell.

## Regenerate the hand-PTX baselines (for the A/B)

Pure string-gen, no GPU needed (run on any CUDA host):

```rust
// RmsNormKernel::new(hidden).with_epsilon(1e-5).emit_ptx_for_target("sm_121")
// for hidden in {2048, 4096, 8192} -> baseline-ptx/rmsnorm_h{h}.sm121.ptx
```

The A/B harness auto-loads `baseline-ptx/*.sm121.ptx` (or `/tmp/rmsnorm_spike/*`).

## Reproduce on gx10

```bash
ssh gx10
export PATH="$HOME/.cargo/bin:/usr/lib/llvm-21/bin:$PATH"
export LLVM_SYS_211_PREFIX=/usr/lib/llvm-21
cd /tmp/rmsnorm_spike        # or rsync experiments/cuda-oxide/rmsnorm/
cargo oxide run              # parity (2 kernels x 3 sizes) + fair single-row A/B + secondary
```
