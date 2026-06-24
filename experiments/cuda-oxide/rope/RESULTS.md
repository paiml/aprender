# PMAT-921 — cuda-oxide RoPE (adjacent-pair) port: gx10 sm_121 A/B vs hand-PTX

**VERDICT: GO** (F-OXIDE-ROPE-PARITY-001 PASS at every shape, fair matched single-launch gate.)

A pure-Rust cuda-oxide `#[kernel]` port of the hand-PTX **`RopeKernel`**
(`crates/aprender-gpu/src/kernels/elementwise/rope/standard.rs`, entry `rope`) was
authored, built, and run on the GB10 Blackwell (sm_121 / compute_cap 12.1) via
cuda-oxide. It is **bit-parity-correct** (cos = 1.0000000 vs an f64 CPU reference)
and **matches** the hand-PTX at every decode-path shape, with NO hand-PTX and NO
GH-480 Blackwell-JIT workaround.

- Source: `experiments/cuda-oxide/rope/src/main.rs`
- Build/run on gx10: `cargo oxide run` (nightly-2026-04-03 + LLVM-21.1.8 + cargo-oxide 0.2.1)
- GPU: NVIDIA GB10, compute_cap 12.1 (sm_121a auto-detected by `cargo oxide`)
- Hand-PTX baselines: `experiments/cuda-oxide/rope/baseline-ptx/rope_hd128_t{10000,1000000}.sm121.ptx`
  (emitted via `RopeKernel::new(128,128,theta).emit_ptx_for_target("sm_121")`)

## What it computes

For each head `h` and rotation pair `p` (p < head_dim/2) of the adjacent-pair RoPE:

```
freq_base = theta^(-2p/head_dim)             [ex2((-2p/head_dim)*log2(theta))]
angle     = pos * freq_base
out[h,2p]   = x[h,2p]*cos(angle) - x[h,2p+1]*sin(angle)
out[h,2p+1] = x[h,2p]*sin(angle) + x[h,2p+1]*cos(angle)
```

This is the exact serve `rope` math (the hand-PTX `RopeKernel`, grid = num_heads
blocks x block = head_dim/2 threads, one rotation pair per thread). RoPE is pure
f32 FMA + sin/cos/ex2 transcendentals — **ZERO DP4A** — so it is squarely the
PMAT-882 GO class (attention softmax, RMSNorm, SwiGLU all GO; only the DP4A-bound
Q4K GEMV/FFN is the NO-GO class). It is applied to Q and K every layer, every
decode token, so it is a genuine decode-hot kernel.

## Two oxide kernel variants (both bit-parity-correct on GB10)

| | freq base | trig | note |
|---|---|---|---|
| **(A) `rope_approx`** | `ex2(power)` (exact hand-PTX mirror) | `.sin()/.cos()` | bit-closest to the baseline |
| **(B) `rope_libdev`** | `exp(power*ln2)` (libdevice) | `.sin()/.cos()` | second lowering datapoint |

## Falsifiable target 1 — PARITY: PASS (both variants, all shapes)

vs an f64 CPU reference. Required: cos >= 0.9999 AND maxdiff < 1e-3.

| heads | head_dim | theta | pos | oxide cos | oxide maxdiff (A / B) |
|---|---|---|---|---|---|
| 32 | 128 | 10000 | 17 | 1.0000000 | 1.07e-6 / 6.86e-7 |
| 14 | 128 | 1000000 (Qwen2.5) | 53 | 1.0000000 | 2.83e-6 / 2.83e-6 |
| 128 | 128 | 10000 | 256 | 1.0000000 | 1.65e-5 / 7.47e-6 |

**PARITY: PASS** (cos = 1.0000000 at every shape, incl. the theta=1M high-freq case).

## Falsifiable target 2 — PERF: GO (true hand-PTX A/B on GB10)

A **true on-device A/B** was run: the actual hand-PTX `rope` was emitted for
sm_121 via aprender-gpu's `RopeKernel::emit_ptx_for_target("sm_121")` (committed
in `baseline-ptx/`) and launched on the same GB10 with the same data and the same
GPU-event timing (median of 5x100 launches). **Fair matched launch:** the hand-PTX
is grid = num_heads blocks x block = head_dim/2 threads, ONE launch; the oxide
kernel uses the IDENTICAL grid/block and ALSO one launch. There is NO launch-count
confound; the ratio is a true per-kernel compare.

`ratio = best(oxide_A,oxide_B)_us / handPTX_us`, gate `<= 1.2 = GO`
(representative run; stable GO across 4 runs):

| heads | head_dim | theta | oxA us | oxB us | hand-PTX us | best ratio | verdict |
|---|---|---|---|---|---|---|---|
| 32 | 128 | 10000 | 2.070 | 2.073 | 1.863 | A 1.111x | GO |
| 14 | 128 | 1000000 | 2.032 | 2.052 | 2.030 | A 1.001x | GO |
| 128 | 128 | 10000 | 2.215 | 2.159 | 2.041 | B 1.058x | GO |

Across 4 runs the best ratio swings 0.984x-1.116x, centered ~1.00-1.05x — a clean
**statistical tie at the DRAM-bandwidth roofline**, exactly as predicted for an
elementwise trig kernel (2 reads + 2 writes per pair, ~10 FLOP + 2 transcendentals).
hand-PTX parity PASS (cos=1.00000) at every shape.

**Perf gate (<= 1.2x): PASS at every shape. VERDICT: GO (tie).**

## Honest framing

This is a **tie**, not a speedup — and it is reported as a tie. Adjacent-pair RoPE
is DRAM-bandwidth-bound (one global load + one global store per element), so neither
the hand-PTX nor the oxide port can beat the memory wall; both sit at the roofline
(~2us both sides). The value of the GO is that the pure-Rust `#[kernel]` **matches**
the hand-PTX with no perf loss, so this decode-hot kernel can be migrated off
hand-PTX (and off the GH-480 Blackwell-JIT path) for free — another north-star
datapoint where pure-Rust->PTX replaces hand-PTX.

## Why GO where PMAT-881 (FFN Q4K matmul) was NO-GO

PMAT-881 lost (~1.58x) because the production FFN gate+up kernel is Q4K **DP4A**
integer math (4 MACs/instr on dedicated hardware) that f32 oxide cannot match.
RoPE is **f32 FMA + sin/cos/ex2** applied to f32 Q/K — zero DP4A — so the oxide
port competes on equal terms and ties. Exactly the prediction from the rule:
port FMA/softmax/elementwise/transcendental kernels; keep DP4A-bound kernels on
hand-PTX.

## VERDICT: GO — safe to migrate the RoPE apply off hand-PTX (follow-up only)

The pure-Rust cuda-oxide RoPE kernel is bit-parity-correct on Blackwell sm_121 and
matches the hand-PTX `RopeKernel` at every decode shape (incl. Qwen2.5 theta=1M),
with NO hand-PTX and NO GH-480 JIT workaround. This spike does NOT migrate the
production trueno kernel — that is the follow-up after this GO.

## Reproduce on gx10

```bash
ssh gx10
export PATH="$HOME/.cargo/bin:/usr/lib/llvm-21/bin:$PATH"
export LLVM_SYS_211_PREFIX=/usr/lib/llvm-21
# (one-off) emit the sm_121 hand-PTX baselines via RopeKernel::emit_ptx_for_target
#   -> baseline-ptx/rope_hd128_t{10000,1000000}.sm121.ptx (committed)
cd experiments/cuda-oxide/rope
cargo oxide run   # parity (2 variants x 3 shapes) + fair matched-launch A/B
```
