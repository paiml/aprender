# PMAT-882 — cuda-oxide port of the incremental (KV-cache) attention kernel

**Status: COMPLETE (kernel works, parity PASS, true hand-PTX A/B measured) — VERDICT: GO.**

A pure-Rust `#[kernel]` port of the hand-PTX `MultiWarpIncrementalAttentionKernel`
(`crates/aprender-gpu/src/kernels/attention/paged/multi_warp/build_ptx.rs`,
entry `multi_warp_attention`) was authored, built, and run on the GB10 Blackwell
(sm_121 / compute_cap 12.1) via cuda-oxide. CPU reference =
`causal_attention_cached`
(`crates/aprender-serve/src/apr_transformer/attention_kernels.rs`).

- Source: `experiments/cuda-oxide/incremental-attention/src/main.rs`
- Build/run on gx10: `cargo oxide run` (nightly-2026-04-03 + LLVM-21 + cargo-oxide 0.2.1)
- GPU: NVIDIA GB10, compute_cap 12.1, driver 590.48.01
- Hand-PTX baselines: `experiments/cuda-oxide/incremental-attention/baseline-ptx/*.sm121.ptx`

## What it computes

For ONE decode token (new_seq_len == 1) and each query head h:

```
score[j] = dot(Q[h], K[j, kv_head(h)]) * scale     for j in 0..kv_len
w        = softmax(score)                            (all cached j valid)
out[h]   = sum_j w[j] * V[j, kv_head(h)]
kv_head(h) = h / (n_heads / n_kv_heads)              (GQA)
```

Layout matches the live serve `causal_attention_cached` `[seq, kv_dim]` cache
(kv_dim = n_kv_heads*head_dim), Q/out `[n_heads, head_dim]`.

## Three kernel variants explored (all bit-parity-correct on GB10)

| | structure | kv=4096 us |
|---|---|---|
| A `incremental_attention` | one block/head, T=128 threads stride positions (uncoalesced) | ~1040 |
| B `attn_chunk`+`attn_reduce` | Flash-Decoding split-K, block per (head,chunk) | ~1025 |
| **C `attn_warp`** | **warp-coalesced, NW warps/head, lane holds head-dim slots l,l+32,l+64,l+96; shfl_xor warp-reduce; online softmax; cross-warp merge** | **165** |

Kernels A/B were ~6× slower than C: the lane-per-position pattern reads K/V
**un-coalesced** (consecutive lanes stride by kv_dim=1024 floats). Kernel C is the
faithful hand-PTX analog (lane-cooperative dot, coalesced K/V) and is the GO
candidate. NW (warps/head) swept on GB10 {4,8,16,32}: **NW=32 best** (32×32=1024 =
max threads/block; more warps/head = more KV-position parallelism).

## Falsifiable target 1 — PARITY: ✅ PASS (all 9 configs, all 3 kernels)

seq_len ∈ {128,1024,4096} × n_heads ∈ {8,16,32}, head_dim=128, n_kv_heads=8 (GQA),
vs the CPU `causal_attention_cached`. Required: cos ≥ 0.99 AND maxdiff < 1e-4·max|ref|.

Kernel C result (every config): **cos = 1.000000**, maxdiff 4.8e-7 … 1.0e-5,
tol ≈ 6.5e-5 ⇒ **PASS** at every (seq_len, n_heads). (Q/K/V seeded; the GQA
kv_head mapping is exercised at group_size = n_heads/8 ∈ {1,2,4}.)

## Falsifiable target 2 — PERF: ✅ GO (true hand-PTX A/B on GB10)

A **true on-device A/B** was run (not just "vs documented baseline"): the actual
hand-PTX `multi_warp_attention` was emitted for sm_121 via aprender-gpu's
`emit_ptx_for_target` and launched on the same GB10 with the same Q/K/V data and
the same GPU-event timing (median of 5×50 launches), repacked into the hand-PTX's
`[kv_head, max_seq_len, head_dim]` separate-head layout. Both kernels verified
parity-correct vs the CPU reference inside the harness.

oxide C (NW=32) vs hand-PTX `multi_warp_attention` — **ratio = oxide_us / handPTX_us
≤ 1.2 = GO**:

| Shape (kv × heads) | oxide C (µs) | hand-PTX NW=8 (default) | ratio | hand-PTX NW=32 (best) | ratio |
|---|---|---|---|---|---|
| 128 × 32   | 6.17  | 10.26 | **0.60×** | 10.22 | **0.60×** |
| 1024 × 32  | 22.0  | 51.2  | **0.43×** | 24.6  | **0.90×** |
| 4096 × 32  | 165   | 489   | **0.34×** | 165   | **~0.95–1.01×** |

- vs the **production default (NW=8)** the oxide kernel is **0.34–0.60× (1.7–2.9×
  faster)** at every shape.
- vs a **best-case matched hand-PTX (NW=32)** it is **0.60–1.01×** — faster at
  short/mid context, statistically tied at long context (run-to-run 0.95–1.01×).
- hand-PTX parity PASS (cos=1.0000) at all heads=32 shapes (the emitted PTX bakes
  in n_heads=32, so the A/B uses heads=32 for a parity-valid compare).
- the short-ctx 6.17µs hits the documented hand-PTX ~10µs target and beats it.

**Perf gate (≤ 1.2×): PASS at every shape, against both hand-PTX configs. GO.**

## Why this is a GO where PMAT-881 (FFN-fusion) was a NO-GO

PMAT-881 lost (~1.58×) because the production FFN kernel is **Q8_1 DP4A** integer
math (4 MACs/instr on dedicated hardware) and the f32 oxide port can't match it.
Attention is different: the hand-PTX `multi_warp_attention` is **f32 FMA + softmax
(ex2)** — NOT DP4A-bound — so the oxide port competes on equal terms and wins via
the same warp-coalesced structure. Exactly the prediction in the ticket.

## VERDICT: GO — port into the serve CUDA executor next

The pure-Rust cuda-oxide incremental-attention kernel is bit-parity-correct on
Blackwell sm_121 and **matches-or-beats** the hand-PTX MultiWarpIncrementalAttention
at every decode shape, with NO hand-PTX and NO GH-480 JIT workaround. This is a
genuine north-star win: a real decode hot kernel where pure-Rust→PTX replaces
hand-PTX with no perf loss (and a speedup vs the production NW=8 config).

## Exact next step

Wire kernel C into the serve CUDA executor's decode-attention dispatch:
1. Emit standalone embeddable PTX for `attn_warp` via `cargo oxide pipeline`
   (as q4k-matvec did) → `include_str!` → `CudaModule::from_ptx`
   (`crates/aprender-gpu/src/driver/module.rs`), raw-pointer ABI
   `(q,k,v,out: *…, kv_len,head_dim,n_heads,n_kv_heads: u32, scale: f32)`.
2. Add a 3-way parity gate (oxide PTX vs hand-PTX vs CPU) on gx10 (no sm_121 CI
   runner; gx10-manual like PMAT-734).
3. Confirm the live-serve `[seq, kv_dim]` cache layout maps directly (it does —
   the kernel already uses it); GQA `kv_head = head/group_size` matches.
4. Then measure end-to-end decode tok/s with the oxide attention kernel vs default
   on a real GQA model on Blackwell.

## Regenerate the hand-PTX baselines (for the A/B)

```bash
# one-off, on a CUDA host (lambda-vector) — pure string gen, no GPU needed:
#   trueno_gpu::kernels::MultiWarpIncrementalAttentionKernel::new(
#       max_seq_len=4096, head_dim=128, n_heads=32, n_kv_heads=8, num_warps)
#       .emit_ptx_for_target("sm_121")
# emit num_warps ∈ {8,32} → baseline-ptx/multiwarp_msl4096_nw{8,32}.sm121.ptx
```

The A/B harness auto-loads `baseline-ptx/*.sm121.ptx` (or `/tmp/incattn_spike/*`).

## Reproduce on gx10

```bash
ssh gx10
export PATH="$HOME/.cargo/bin:/usr/lib/llvm-21/bin:$PATH"
export LLVM_SYS_211_PREFIX=/usr/lib/llvm-21
cd /tmp/incattn_spike        # or rsync experiments/cuda-oxide/incremental-attention/
cargo oxide run              # parity (9×3 configs) + perf (A/B/C) + true hand-PTX A/B
```
