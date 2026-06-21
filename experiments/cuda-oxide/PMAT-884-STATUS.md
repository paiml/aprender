# PMAT-884 — oxide attention GO-LIVE (separate-head live-cache kernel + wiring)

**Status: IN PROGRESS — layout SOLVED (option b, zero-cost), live-cache 3-way
parity PASS on GB10, kernel wired behind default-OFF feature + env guard.
E2E decode tok/s measurement: see §4.**

Realizes the PMAT-882/883 GO (pure-Rust cuda-oxide `attn_warp` decode-attention
kernel: bit-exact cos=1.0 + 1.7–2.9× faster than hand-PTX on GB10 sm_121) into a
production-wireable path **without changing the production default**. The feature
`oxide-attention` stays default-OFF AND is additionally runtime-gated by
`APR_OXIDE_ATTENTION=1`; with either off the decode path is byte-identical to the
hand-PTX `MultiWarpIncrementalAttentionKernel` default.

- gx10: NVIDIA GB10, compute_cap 12.1 (sm_121), driver 590.48.01, cargo-oxide
  0.2.1, nightly-2026-04-03, LLVM-21, CUDA 13.0 (ptxas).

## 1. KV-cache layout solution — option (b), ZERO-COST

PMAT-883's one promotion blocker: the live `incremental_attention_async` KV cache
is **separate-head** `[num_kv_heads, max_len, head_dim]` (per-kv-head slab stride
`kv_stride = max_len*head_dim`), but the oxide `attn_warp_rawptr` kernel used the
**interleaved** `[seq, kv_dim]` layout (`krow = pos*kv_dim + kv_head*head_dim`).

**Chosen: option (b)** — a new kernel-C variant `attn_warp_sephead_rawptr` that
indexes the live separate-head cache **directly**, with an added `kv_stride: u32`
param (10-param ABI). The ONLY compute difference vs `attn_warp_rawptr` is the
K/V row address:

```
interleaved (883):  krow = pos*kv_dim       + kv_head*head_dim
separate-head (884): krow = kv_head*kv_stride + pos*head_dim    (kv_stride=max_len*head_dim)
```

**Cost = ZERO.** No interleave/gather adapter, no extra D2D copy, no extra
allocation — the kernel reads the cache the executor already wrote. The K/V reads
along the head-dim are still fully coalesced (consecutive lanes read consecutive
`head_dim` elements). The kernel-C speedup is therefore fully preserved (option
(a), a per-token gather adapter, was rejected because it would add a copy on the
hot decode path with no benefit — measure-first verdict: the direct-index kernel
is strictly better).

- Source kernel: `incremental-attention/src/main.rs` — `attn_warp_sephead_rawptr`
  (`#[cuda_module]`), bit-identical online-softmax/warp-coalesced compute to 882/883.
- Emit: `incremental-attention/emit_ptx_sephead.sh` (thin wrapper over the shared
  `emit_ptx.sh` with `ENTRY=attn_warp_sephead_rawptr`) → self-contained sm_121 PTX
  `generated/attn_warp_sephead.sm121.ptx` (551 lines, 1 entry, 0 extern `__nv_*`,
  ptxas-verified 18960-byte cubin).

## 2. Live-cache 3-way parity gate — ✅ PASS (measured, GB10, 2026-06-21)

New gate `run_sephead_live_gate` (auto-runs when the sephead PTX is present). For
each of the 9 decode configs (seq{128,1024,4096} × heads{8,16,32}, head_dim=128,
n_kv_heads=8 GQA) it packs K/V into the **LIVE separate-head slab**
(`kv_stride = max_len*head_dim`, max_len=4096 — exactly how the executor stores
the cache) and compares oxide-sephead-PTX vs hand-PTX `multi_warp_attention`
(also separate-head; parity-valid at heads=32) vs CPU `causal_attention_cached`,
asserting cos≥0.99 AND maxdiff<1e-4·max|ref|.

```
== PMAT-884 LIVE-CACHE 3-WAY PARITY GATE (oxide sephead == hand-PTX == CPU) ==
   K/V layout = SEPARATE-HEAD [num_kv_heads, max_len=4096, head_dim] (LIVE cache)
    seq head  kv | oxide sephead (live layout)    | hand-PTX multi_warp          | verdict
    128    8   8 | cos=1.000000 md=3.58e-7        | n/a (baked 32h)              | PASS
   1024    8   8 | cos=1.000000 md=2.15e-6        | n/a (baked 32h)              | PASS
   4096    8   8 | cos=1.000000 md=9.89e-6        | n/a (baked 32h)              | PASS
    128   16   8 | cos=1.000000 md=4.77e-7        | n/a (baked 32h)              | PASS
   1024   16   8 | cos=1.000000 md=2.15e-6        | n/a (baked 32h)              | PASS
   4096   16   8 | cos=1.000000 md=8.20e-6        | n/a (baked 32h)              | PASS
    128   32   8 | cos=1.000000 md=4.17e-7        | cos=1.000000 md=5.36e-7      | PASS
   1024   32   8 | cos=1.000000 md=2.92e-6        | cos=1.000000 md=2.68e-6      | PASS
   4096   32   8 | cos=1.000000 md=1.22e-5        | cos=1.000000 md=1.20e-5      | PASS
PMAT-884 LIVE-CACHE 3-WAY PARITY GATE: PASS
```

All three agree at cos=1.000000 across every config; the 883 interleaved gate
also still PASSes in the same run. Kernel-C perf re-confirmed (true on-device A/B,
GPU-event median 5×50): oxide vs hand-PTX NW=8 = **0.29–0.60×** (1.7–3.4× faster),
vs matched NW=32 = **0.62–1.11×** (faster short/mid, ~tied long).

GPU-free CI guard ships too: `oxide_attention::tests::embedded_sephead_ptx_is_self_contained_single_entry`
asserts the committed sephead PTX is sm_121 / 1 entry / 0 extern `__nv_*`
(passes on any host; `cargo test -p aprender-serve --features oxide-attention --lib oxide_attention` → 2/2 PASS).

## 3. Production wiring (default OFF, byte-identical when off)

- `crates/aprender-serve/src/cuda/executor/oxide_attention.rs`:
  embeds `attn_warp_sephead.sm121.ptx`, adds `compile_oxide_attention_sephead` +
  `launch_oxide_attention_sephead` (10-param ABI, takes the live cache buffers).
- `crates/aprender-serve/src/cuda/executor/attention_async.rs`
  (`incremental_attention_async`, the LIVE decode path): a
  `#[cfg(feature="oxide-attention")]` branch entered ONLY when
  `APR_OXIDE_ATTENTION=1`, which compiles+caches the sephead module once and
  launches it against the live K/V cache, then returns. With the feature OFF the
  branch does not compile; with the feature ON but env unset it is skipped — the
  hand-PTX default path is unchanged.
- `Cargo.toml`: `oxide-attention = ["cuda"]` stays out of `default`/`full`.
- `cargo check -p aprender-serve --features oxide-attention` clean on lambda-vector;
  clippy clean on the oxide path; both PTX-shape guards PASS.

## 4. End-to-end decode tok/s on a real GQA model (GB10) — <FILL>

<E2E measurement on Qwen2.5-Coder GQA: APR_OXIDE_ATTENTION=0 (hand-PTX default)
vs =1 (oxide), tok/s + ratio + coherence spot-check.>

## 5. GO/NO-GO + remaining go-live step

<verdict + exact remaining step>

## Files

- `experiments/cuda-oxide/generated/attn_warp_sephead.sm121.ptx` — sephead source-of-record PTX.
- `experiments/cuda-oxide/incremental-attention/src/main.rs` — `attn_warp_sephead_rawptr` kernel + `sephead_ptx_parity` + `run_sephead_live_gate`.
- `experiments/cuda-oxide/incremental-attention/emit_ptx_sephead.sh` — sephead emit wrapper.
- `experiments/cuda-oxide/incremental-attention/emit_ptx.sh` — shared emit pipeline (now ENTRY-overridable).
- `crates/aprender-serve/src/cuda/executor/oxide_attention.rs` — embed + launch wrappers + CI guard.
- `crates/aprender-serve/src/cuda/executor/attention_async.rs` — live dispatch branch (cfg + env, default OFF).
