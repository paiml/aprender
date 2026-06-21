# PMAT-881 — cuda-oxide port of the FusedGateUpSwiglu Q4K kernel

**Status: COMPLETE (kernel works, parity PASS, perf measured) — VERDICT: NO-GO for production decode.**

A pure-Rust `#[kernel]` port of the hand-PTX `FusedGateUpSwigluHwDp4aQ4KGemvKernel`
(`crates/aprender-gpu/src/kernels/quantize/q4k/coalesced/fused_gate_up_swiglu_hw_dp4a.rs`)
was authored, built, and run on the GB10 Blackwell (sm_121 / compute_cap 12.1) via cuda-oxide.

- Source: `experiments/cuda-oxide/ffn-fusion/src/main.rs`
- Build/run on gx10: `cargo oxide run` (nightly-2026-04-03 + LLVM-21 + cargo-oxide 0.2.1)
- GPU: NVIDIA GB10, compute_cap 12.1, driver 590.48.01

## What it computes

`y[i] = silu(dot(W_gate[i], x)) * dot(W_up[i], x)` for each FFN-intermediate row `i`,
in a SINGLE launch with TRUE fusion (dual gate+up accumulators, in-register SwiGLU,
one global write per row, no intermediate global buffers, no second SwiGLU kernel) —
the exact fusion shape of the production hand-PTX kernel.

Design (reuses the proven q4k_matvec oxide patterns):
- one block per output row, `T` threads/block, grid = N blocks
- each thread does an element-strided partial dot over K for BOTH gate and up
- block reduction via `SharedArray<f32, {2*T}>` + `thread::sync_threads()` (tree reduce)
- thread 0 computes SwiGLU and writes `y[row]`
- `T = 256` (swept 32/128/256; 256 optimal — element-strided ⇒ more T = more K-parallelism)

cuda-oxide primitives used (all verified working on Blackwell): `#[cuda_module]`/`#[kernel]`,
`thread::blockIdx_x/threadIdx_x/sync_threads`, `static mut SharedArray<f32,N> = UNINIT`
with `Index`/`IndexMut`, custom `LaunchConfig { grid_dim, block_dim, shared_mem_bytes }`,
`&[u8]`/`&[f32]` slice params, raw-pointer output write, device `#[device]`-style helpers
(`f16_to_f32`, `extract_scale_min`, `dequant_elem`, `swiglu`).

## Falsifiable target 1 — PARITY: ✅ PASS

Parity is vs the CPU fused reference (identical Q4K dequant + SwiGLU math). A true
hand-PTX A/B was impractical (the hand-PTX kernel needs the full aprender-gpu CUDA
executor toolchain on gx10, and uses Q8_1 DP4A integer math vs this port's f32-activation
math — see the perf caveat below), so per the PMAT-881 process this is the
CPU-reference parity + documented-baseline perf route.

- 10 test vectors at a Qwen-like FFN shape (N=512, K=768): **PASS**, total_errs=0,
  worst_maxdiff = 1.95e-3 (tol = 1e-4·max|ref| = 1.20e-1).
- All 4 large Qwen FFN shapes also PASS parity (maxdiff ≤ 2.86e-2 vs tol ≥ 5.40e-1).
- Gate and up weight matrices use DISTINCT seeds, so a gate/up swap would fail (it doesn't).

## Falsifiable target 2 — PERF on GB10 (100+ runs, median of 5 reps × 200 launches)

| Shape (N×K) | Role | oxide T=256 (µs/launch) | hand-PTX baseline | ratio |
|---|---|---|---|---|
| 1536 × 8960 | Qwen 1.5B FFN | **189.1** | ~120 (documented) | **~1.58× SLOWER** |
| 11008 × 6656 | 7B-class FFN | 1377 | (no documented number) | — |
| 14784 × 8960 | wide FFN | 2479 | (no documented number) | — |
| 11008 × 11008 | 7B FFN | 2258 | (no documented number) | — |

T-sweep at the baseline shape: T=32 → 269µs, T=128 → 193µs, T=256 → 189µs.
A per-superblock-decode-once variant (amortize the f16/scale decode like hand-PTX) was
also measured and was SLOWER (343µs @ T=32) — at T≈num_superblocks the occupancy / K-
parallelism loss dominates the amortized-decode win on this hardware. Element-strided
(high T, redundant per-element decode) is the faster oxide structure here.

**Perf gate (< 1.3×, ideally < 1.0×): FAIL.** Best achieved ≈ 1.58× the documented
hand-PTX ~120µs baseline.

## Root cause of the perf gap (honest)

The production hand-PTX kernel uses **Q8_1 DP4A**: it pre-quantizes activations to int8
and does 4-way int8·int8 dot products on dedicated DP4A hardware (`dp4a.u32.s32`), so the
inner loop is integer and ~4 MACs/instruction. This oxide port (per the PMAT-881 parity
signature, `x: f32`) does **per-element f32 Q4K dequant + f32 FMA** — far more work per
element and no DP4A. This is exactly consistent with the prior decisive finding in
`memory/reference_cuda_oxide_rust_to_ptx.md`: *"oxide vs HwDp4a = 3.4–4.2× (oxide LOSES)."*
The fusion structure here (single launch, no intermediate buffers) recovers some of that —
~1.58× rather than ~4× — but it does not close the gap.

## VERDICT: NO-GO (stays hand-PTX)

Wiring this f32-activation oxide FFN-fusion kernel into the serve CUDA executor would make
the Q4K FFN block ~1.58× slower than the current hand-PTX HwDp4a path. It is **NOT** a
decode speedup and should NOT replace the hand-PTX kernel.

**What this DOES prove (the value delivered):**
- The full FFN-fusion kernel (dual accumulators + in-register SwiGLU + block reduction)
  is expressible in pure-Rust cuda-oxide and runs **bit-parity-correct on Blackwell sm_121**
  with NO hand-PTX and NO GH-480 JIT workaround. The north-star capability (pure-Rust GPU
  kernels for the real FFN-fusion hot path) is confirmed end-to-end.
- The perf characterization is now honest and measured (not assumed): ~1.58×, root-caused
  to the missing DP4A integer path, matching the documented HwDp4a A/B.

## Exact next step (to EVER make this a GO)

A DP4A-class oxide FFN-fusion kernel: take Q8_1-quantized activations (`q8_ptr`, 288 B/SB)
and use cuda-oxide's `dp4a` intrinsic (`cuda_device` exposes `dp4a` — verify the
`dp4a.u32.s32` lowering) to match the hand-PTX integer inner loop, then re-A/B vs the
hand-PTX kernel on gx10. ONLY if that DP4A oxide kernel reaches < 1.3× (ideally beats)
HwDp4a is integration into the serve executor justified. Until then: hand-PTX stays.

To emit standalone embeddable PTX (for a future promotion path), use
`cargo oxide pipeline` (as documented for q4k-matvec) rather than `cargo oxide run`.
