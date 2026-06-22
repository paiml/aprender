# PMAT-894 — cuda-oxide SwiGLU/SiLU activation port: gx10 sm_121 A/B

**Status: COMPLETE (kernel works, parity PASS, true hand-PTX A/B measured) — VERDICT: GO (statistical tie).**

A pure-Rust cuda-oxide `#[kernel]` port of the hand-PTX **`FusedSwigluKernel`**
(`crates/aprender-gpu/src/kernels/elementwise/swiglu.rs`, entry `fused_swiglu`)
was authored, built, and run on the GB10 Blackwell (sm_121 / compute_cap 12.1)
via cuda-oxide. The kernel computes the elementwise SwiGLU activation

```
out[i] = silu(gate[i]) * up[i]
silu(x) = x * sigmoid(x)
sigmoid(x) = 1 / (1 + exp(-x))   [hand-PTX: 1/(1 + exp2(-x * log2e)), ex2.approx]
```

over already-dequantized f32 `gate`/`up` vectors.

- Source: `experiments/cuda-oxide/swiglu/src/main.rs`
- Build/run on gx10: `cargo oxide run` (nightly-2026-04-03 + LLVM-21.1.8 + cargo-oxide 0.2.1)
- GPU: NVIDIA GB10, compute_cap 12.1 (sm_121a auto-detected by `cargo oxide`)
- Hand-PTX baseline: `experiments/cuda-oxide/swiglu/baseline-ptx/fused_swiglu.sm121.ptx`

## This is the GO class — NOT the PMAT-881 NO-GO

This is the elementwise SwiGLU **ACTIVATION** (`silu(gate) * up`, transcendental
ex2 class), applied to already-dequantized f32 — the same GO class as PMAT-882
softmax and PMAT-893 RMSNorm (both BEAT/tied hand-PTX on Blackwell). It is **NOT**
the PMAT-881 FFN gate+up **Q4K matmul** (DP4A-bound, NO-GO ~1.58× slower). There
is zero DP4A here; the oxide port competes on equal f32 + ex2 terms.

The kernel is **purely DRAM-bandwidth-bound** (2 reads + 1 write per element,
~6 FLOP + 1 transcendental), so the a-priori prediction was a **tie** — and a tie
that passes the `<=1.2×` gate is still a GO (it retires the hand-PTX + the GH-480
Blackwell-JIT workaround with no perf loss). The measurement confirms the tie.

## Two oxide variants (both bit-parity-correct on GB10)

| | sigmoid | note |
|---|---|---|
| (A) `swiglu_libdev` | `1/(1+exp(-g))` via libdevice `f32::exp` | proven 882/893 path; slightly higher precision than ex2.approx |
| (B) `swiglu_ex2` | `1/(1+exp2(-g·log2e))` via `f32::exp2` | mirrors the hand-PTX `ex2.approx` form exactly |

## Falsifiable target 1 — PARITY: ✅ PASS (both variants, all sizes)

n ∈ {4096, 11008, 14336} (Llama/Mistral FFN intermediate widths), `gate` in
~[-4, 4] (≈ half the elements negative — the sigmoid is exercised on both sides
of 0), `up` in ~[-2, 2], vs an f64 CPU reference. Required: cos ≥ 0.9999 AND
maxdiff < 1e-4.

| n | neg gate | oxide cos | oxide maxdiff |
|---|---|---|---|
| 4096 | 2048 | 1.0000000 | 4.77e-7 |
| 11008 | 5503 | 1.0000000 | 4.77e-7 |
| 14336 | 7168 | 1.0000000 | 7.15e-7 |

(Both variants A and B identical to f32 rounding.) **PARITY: PASS.**

## Falsifiable target 2 — PERF: ✅ GO (true hand-PTX A/B on GB10)

A **true on-device A/B** was run: the actual hand-PTX `fused_swiglu` was emitted
for sm_121 via aprender-gpu's `FusedSwigluKernel::emit_ptx_for_target("sm_121")`
(committed in `baseline-ptx/`) and launched on the same GB10 with the same data
and the same GPU-event timing (median of 5×100 launches). **Fair matched launch:**
the hand-PTX is a flat 1-D grid kernel that processes the whole length `n` in ONE
launch (`ceil(n/256)` blocks × 256 threads); the oxide kernel uses the IDENTICAL
grid/block and ALSO one launch — so, unlike the PMAT-893 RMSNorm single-row ABI,
there is **no launch-count confound** here. The ratio is a true per-kernel compare.

`ratio = oxide_us / handPTX_us` (best of A/B), gate `<= 1.2 = GO`:

| n | oxide A (µs) | oxide B (µs) | hand-PTX (µs) | best ratio | verdict |
|---|---|---|---|---|---|
| 4096 | 2.020 | 2.041 | 1.927 | **A 1.048×** | GO |
| 11008 | 2.014 | 1.898 | 1.873 | **B 1.013×** | GO |
| 14336 | 2.041 | 2.049 | 2.059 | **A 0.991×** | GO |

Run-to-run (3 runs) the best ratio swings **0.970×–1.048×**, centered at ~1.00× —
a clean **statistical tie**. ~2.0 µs both sides ≈ DRAM-bandwidth-limited, exactly
as predicted. hand-PTX parity PASS (cos=1.00000, maxdiff=7.15e-7) at every size.

**Perf gate (≤ 1.2×): PASS at every size. VERDICT: GO (tie).**

## Honest framing

This is a **tie**, not a speedup — and it is reported as a tie. SwiGLU activation
is pure DRAM-bandwidth-bound, so neither the hand-PTX nor the oxide port can beat
the memory wall; both sit at the roofline. The value of the GO is that the
pure-Rust `#[kernel]` **matches** the hand-PTX with no perf loss, so this kernel
can be migrated off hand-PTX (and off the GH-480 Blackwell-JIT path) for free.
No launch-overhead-dominated context numbers are used to inflate the result — the
A/B is a single matched launch on each side.

## Why GO where PMAT-881 (FFN Q4K matmul) was NO-GO

PMAT-881 lost (~1.58×) because the production FFN gate+up kernel is **Q4K DP4A**
integer math (4 MACs/instr on dedicated hardware) and an f32 oxide port can't
match it. The SwiGLU **activation** is **f32 + ex2 (no DP4A)** applied to
already-dequantized data — so the oxide port competes on equal terms and ties.
Exactly the prediction in the ticket.

## VERDICT: GO — safe to migrate the SwiGLU activation off hand-PTX

The pure-Rust cuda-oxide SwiGLU activation kernel is bit-parity-correct on
Blackwell sm_121 and **matches** the hand-PTX `fused_swiglu` at every FFN width,
with NO hand-PTX and NO GH-480 JIT workaround. A genuine (if quiet) north-star
datapoint: another decode-path elementwise kernel where pure-Rust→PTX replaces
hand-PTX with no perf loss.

## Reproduce

```bash
# 1. (one-off, on a CUDA host) regenerate the sm_121 hand-PTX baseline — pure
#    string gen, no GPU needed:
#      trueno_gpu::kernels::FusedSwigluKernel::new(n)
#          .emit_ptx_for_target("sm_121")
#    -> baseline-ptx/fused_swiglu.sm121.ptx  (committed; one emit covers all n,
#       the kernel bounds-checks the runtime `n` param)
#
# 2. on gx10 (GB10 sm_121):
ssh gx10
mkdir -p /tmp/swiglu_spike && rsync -az experiments/cuda-oxide/swiglu/ gx10:/tmp/swiglu_spike/
cp /tmp/swiglu_spike/baseline-ptx/fused_swiglu.sm121.ptx /tmp/swiglu_spike/fused_swiglu.ptx
cd /tmp/swiglu_spike && cargo oxide run   # arch auto-detected = sm_121
```
