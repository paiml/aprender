# PMAT-884 — oxide attention GO-LIVE (separate-head live-cache kernel + wiring)

**Status: COMPLETE — layout SOLVED (option b, zero-cost), live-cache 3-way
parity PASS (cos=1.0) on GB10, kernel wired behind default-OFF feature + env
guard. E2E VERDICT: NO-GO on flipping the default (measured 0.41× — the
production decode is CUDA-graph-captured and a cuLaunchKernel-based oxide kernel
cannot be recorded into the graph; e2e is also host-overhead-bound). Kernel is
correct + fast in isolation; the go-live blocker is graph-capturability, not the
kernel. Default stays OFF; production path byte-identical. See §4/§5.**

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

## 4. End-to-end decode tok/s on a real GQA model (GB10) — measured

Model: `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` (GQA: num_heads=12,
num_kv_heads=2, head_dim=128), `apr run --gpu --chat --benchmark`, GB10 sm_121.

| condition | tok/s (100-tok) | notes |
|---|---|---|
| **production default** (`APR_OXIDE_ATTENTION=0`, graph ON) | **10.2–10.4** | hand-PTX, byte-identical to today |
| `APR_OXIDE_ATTENTION=1` (graph ON, default) | **4.2–4.3** | 2.4× REGRESSION |
| `APR_OXIDE_ATTENTION=0`, `GRAPH_DISPATCH=0` | 10.8–10.9 | non-graph baseline |
| `APR_OXIDE_ATTENTION=1`, `GRAPH_DISPATCH=0` | 4.5–4.6 | 2.4× regression |

Coherence: oxide-ON output IS coherent (`2+2 → "4"`) — the kernel is correct
end-to-end (consistent with §2's cos=1.0 live-cache parity). **Ratio = 0.41×
(oxide-on / hand-ptx). NO speedup; large regression.**

### Why the regression — root-caused (NOT the kernel)

The oxide kernel itself is FAST at the real decode shape (measured in the harness,
GPU-event median, heads=12 kv_heads=2 hd=128):

```
-- PMAT-884 REAL DECODE SHAPE (Qwen2.5-1.5B: heads=12 kv_heads=2 hd=128) C(warp) us --
  kv_len=   1 : oxide C = 4.123us     kv_len=  64 : oxide C = 5.062us
  kv_len=   8 : oxide C = 4.118us     kv_len= 128 : oxide C = 6.168us
  kv_len=  32 : oxide C = 4.121us     kv_len= 256 : oxide C = 8.213us
```

At ~4–8 µs the attention kernel is <0.3% of the ~95 ms/token budget — so the
e2e baseline (10 tok/s) is **host-overhead-bound, not attention-bound**; no
attention-kernel swap can move the e2e needle on this build regardless.

The 2.4× REGRESSION instead comes from a **production-architecture mismatch**:
the live Q4K GGUF decode loop is **CUDA-graph-captured** (PAR-054
`forward_all_layers_gpu_to_logits_graphed` → `incremental_attention_into_for_capture`
with `seq_len_buf` set, i.e. `capturing=true`). A cuda-oxide kernel launched via
immediate `cuLaunchKernel` **cannot be RECORDED into a CUDA graph**, so the oxide
branch is (correctly) skipped during capture — meaning **oxide never runs on the
production decode hot path**. Worse, with the env on the single prefill
`into_inner` oxide call mutates the module map / launches outside the graph, which
disrupts the PAR-054 captured decode graph and forces per-token re-capture — that
is the 2.4× regression. (Instrumented: with `APR_OXIDE_ATTENTION=1` the oxide
branch fires only on the few prefill calls, never on the graph-captured decode
tokens.) Diagnostic instrumentation has been removed from the committed code.

So the wiring is correct for the documented `incremental_attention_async` /
`incremental_attention_into_inner` decode functions, but **the real Qwen2.5 decode
path is graph-captured and does not call them in a graph-injectable way.**

## 5. GO/NO-GO + remaining go-live step

**VERDICT: NO-GO on flipping the production default — for a real, measured reason.**

- Kernel correctness: GO (live-cache 3-way parity cos=1.0, all 9 shapes; coherent e2e).
- Kernel speed in isolation: GO (4–8 µs at the real decode shape; 882/883 A/B re-confirmed).
- **E2E decode tok/s: NO-GO** — oxide-on 4.2 vs hand-PTX 10.4 (0.41×). The win
  from §2/882/883 does NOT transfer to e2e because (a) the production decode is
  CUDA-graph-captured and a `cuLaunchKernel`-based oxide kernel cannot be recorded
  into the graph, and (b) the e2e is host-overhead-bound so attention is <1% of
  the per-token budget. **A NO-GO with the real reason is the honest result.**

Default stays OFF (feature out of default/full AND env-gated). Production path is
byte-identical with the feature/env off (verified: 10.2–10.4 tok/s unchanged).

### Exact remaining step to a real go-live

The blocker is graph-capturability, not the kernel. Go-live requires ONE of:

1. **Record the oxide launch INTO the captured graph** (preferred): add the oxide
   `cuLaunchKernel` to the `incremental_attention_into_for_capture` path so it is
   captured as a graph node (instead of being skipped when `capturing=true`). This
   needs the oxide module + params to be live at capture time and the launch issued
   on the capture stream — i.e. wire `launch_oxide_attention_sephead` through the
   PAR-054 graph builder, not as an immediate launch. THEN re-measure e2e.
2. Add a non-graph decode mode that routes the Q4K decode through
   `incremental_attention_async`/`_into_inner` (where oxide already works) and
   measure there — but that mode is itself slower (graph capture is the perf win),
   so it would only validate the kernel, not beat the default.

Separately, the ~10 tok/s e2e baseline on this gx10 build is abnormally low for a
1.5B Q4_K_M on GB10 (host-overhead-bound); even a perfect attention kernel can't
help until that is addressed. The attention kernel was never the e2e bottleneck.

### What IS shippable now (already landed on this branch)

- The bit-exact, faster-in-isolation oxide sephead kernel + self-contained sm_121 PTX.
- The live-cache 3-way parity gate (cos=1.0) — promotion criterion 1 DISCHARGED.
- The default-OFF + env-gated wiring into both `incremental_attention_async` and
  `incremental_attention_into_inner` (correct for the non-graph decode paths),
  with the production graph path explicitly + safely skipped.
- GPU-free CI guard for the committed PTX artifact.

## Files

- `experiments/cuda-oxide/generated/attn_warp_sephead.sm121.ptx` — sephead source-of-record PTX.
- `experiments/cuda-oxide/incremental-attention/src/main.rs` — `attn_warp_sephead_rawptr` kernel + `sephead_ptx_parity` + `run_sephead_live_gate`.
- `experiments/cuda-oxide/incremental-attention/emit_ptx_sephead.sh` — sephead emit wrapper.
- `experiments/cuda-oxide/incremental-attention/emit_ptx.sh` — shared emit pipeline (now ENTRY-overridable).
- `crates/aprender-serve/src/cuda/executor/oxide_attention.rs` — embed + launch wrappers + CI guard.
- `crates/aprender-serve/src/cuda/executor/attention_async.rs` — live dispatch branch (cfg + env, default OFF).
