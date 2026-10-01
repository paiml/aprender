---
spike: 009
idea: prophet-forecast-mcp
name: chronos-2-parity
type: standard
validates: "Given `amazon/chronos-2` safetensors (120M, RoPE, arcsinh scaling, 21 quantiles, covariates), when ported on the spike-005 ladder (scaling → embeddings → hidden states → quantiles), then Python parity holds on Peyton, air and the edge probes, forward cost is measured with and without 008, and a verdict is given on whether 120M is servable in one binary"
verdict: VALIDATED
related: [005, 007, 008]
tags: [chronos-2, t5, rope, parity, group-attention, f16]
---

# Spike 009: Chronos-2 Parity

## What This Validates

Given the `amazon/chronos-2` checkpoint (119.5M parameters, f32), when its architecture is ported
to Rust on the spike-005 ladder and every product goes through trueno's packed GEMM with the
spike-008 NEON kernel, then (a) scaling, patch features, embeddings, hidden states and the 21
quantiles match `chronos-forecasting` 2.3.1 on Peyton (64, 365 and 1024 steps), air passengers
and seven edge probes; (b) the forward cost is measured single-threaded, with the crate's parallel
GEMM, and with f16 weights; and (c) memory and binary size say whether a 120M model belongs in
one thin-server binary.

## Research

Read from the installed package (`chronos/chronos2/{config,layers,model,pipeline}.py`,
`chronos_bolt.py` for `InstanceNorm`/`Patch`) and the checkpoint header. Nothing external.

**Architecture** (differs from Bolt in every layer): encoder-only. Context → last 8192 → instance
norm (nanmean, population std, 0 → 1e-5) → **arcsinh** → left-NaN-pad to 16 → patches of
`[time(16) | values(16) | mask(16)]` where time is `[-L, …, -1] / 8192` → residual patch embedding
(48 → 3072 → 768) → REG token (`shared[1]`) → *future* patches `[t/8192 | 0 | 0]` through the same
embedding, one per output patch (`ceil(h/16)`, at most 64 ⇒ 1024 steps direct). Twelve blocks of:
**time self-attention with RoPE** (θ = 10 000, d_kv = 64, `rotate_half`, positions 0…L−1 over the
whole sequence, no score scaling, all-NaN patches masked with `finfo.min`); **group self-attention**
across the batch (no RoPE) — for a single series the softmax is over one key, so the layer is
exactly `x += o(v(ln(x)))`; T5 feed-forward (ReLU, no bias). Final RMS norm, then the residual head
(768 → 3072 → 21×16) on the future positions, `sinh` and unscale. Beyond 64 patches the pipeline
unrolls autoregressively over quantile paths with probability-mass weights — not ported (refused
with a message).

**Checkpoint**: 170 f32 tensors, 477.9 MB; names `encoder.block.N.layer.{0,1,2}.*`,
`input_patch_embedding.*`, `output_patch_embedding.*` (bias 336 = 21×16), `shared.weight` [2, 768].

| Approach | Pros | Cons | Status |
|---|---|---|---|
| Port on the 005/007 primitives (GEMM everywhere, per-head attention through `gemm_blis`) | 200 lines; parity ladder ready | single-thread | **Chosen** |
| Keep both weight layouts for a loop A/B (as 007) | debuggable | doubles 478 MB | Rejected: transposed copies only |
| Port the batch/group machinery (multivariate, covariates) | full feature set | not needed for one series; group mask logic | Follow-up |

## How to Run

```bash
# weights: models/chronos-2 -> ~/.cache/huggingface/hub/models--amazon--chronos-2/snapshots/<sha>/
uv run --python 3.12 --with huggingface_hub python -c "from huggingface_hub import snapshot_download; print(snapshot_download('amazon/chronos-2'))"
uv run --python 3.12 --with chronos-forecasting --with pandas --with safetensors tools/oracle.py   # fixture (2.2 MB, committed)
uv run --python 3.12 --with safetensors --with numpy tools/to_f16.py models/chronos-2 models/chronos-2-f16
CARGO_TARGET_DIR=../../../target cargo run --release                     # f32: RUN-OUTPUT.md
CARGO_TARGET_DIR=../../../target cargo run --release models/chronos-2-f16 # f16: RUN-OUTPUT-f16.md
```

## What to Expect

`RUN-OUTPUT.md`: the ladder table (max |Δ| per rung), the cost table, the thread comparison and
the long-horizon refusal; `results.json` the same numbers.

## Investigation Trail

1. **Oracle.** Eleven cases (Peyton at 64/365/1024, air 24, short100, NaN gaps, a leading all-NaN
   patch, constant, ×1e6, negated, five points) with ladder tensors from the model's own methods
   (`_prepare_patched_context`, `input_patch_embedding`, `encode`) and both `model(...)` and
   `pipeline.predict(...)` outputs. torch on 10 threads: 47–73 ms per Peyton forward.
2. **Port, first run, every rung green.** Peyton 64: loc 4.8e-6, scale 1.8e-7, patch features
   5.6e-6, embeddings 1.1e-6, hidden states (first / REG / last future) 7.7e-6, quantiles
   **1.6e-5** (1.9e-5 of the series scale); 365 steps 2.7e-5; 1024 steps 4.0e-5. Attention masks
   identical on every case. The group-attention collapse to `o(v(ln x))` is confirmed by the hidden
   states, not assumed.
3. **Edge probes.** NaN gaps 5.7e-6; leading all-NaN patch 9.5e-6 (its mask row is 0 and the
   output is unchanged); constant series exact; ×1e6 series 2.2e-5 of scale (the loc differs by 1.5
   on 7.8e6 — f32 summation order in the mean); negated 1.1e-5; five points (3 tokens) 2.3e-4 of
   scale, the worst case — the shortest sequences amplify f32 rounding through 12 layers.
4. **Cost, single thread, spike-008 kernel.** 12 tokens 90 ms (26 GFLOP/s — skinny GEMMs);
   37 tokens 156 ms; 133 tokens (2048 pts) 423 ms at 64 GFLOP/s; 187 tokens (Peyton) 587 ms;
   1024 steps 613 ms (193 tokens); 8192-point context (517 tokens) 1.67 s at 67 GFLOP/s. With the
   scalar kernel (7.4 GFLOP/s, spike 008 before) the same forward would take ~5 s. Load 0.46 s.
5. **Threads.** `blis::gemm` (rayon, 14 logical cores): 1.3× at 37 tokens, **1.1× at 133**,
   1.4× at 187, 2.1× at 517; outputs bit-identical. Cause, from `blis/parallel.rs`: partitions are
   along M only in `MC = 128`-row blocks (133 rows ⇒ one block of 128 and one of 5) and the thread
   cap ladder was tuned on square shapes on a Threadripper. A 133×768×3072 product is exactly an
   LLM prefill shape; partitioning along N (or 2-D) is the upstream fix.
6. **f16 weights.** Quantile |Δ| vs Python: Peyton 2.8e-3 / 4.8e-3 / 6.2e-3 of scale at 64 / 365 /
   1024 steps, air 1.7e-3, five points **3.1e-2**; hidden states already differ by 2.3e-3 after 12
   layers. Thirty times Bolt-tiny's f16 penalty (spike 007: 1.1e-3). File 228 MB vs 478 MB.
7. **Memory.** Peak RSS 1.45 GB (f32) / 1.21 GB (f16): the loader holds the raw bytes, the decoded
   f32 and the transposed copy at once; steady state is the 478 MB of transposed weights. A
   streaming decode-and-transpose per tensor would bring the peak to ~0.5 GB (f32) or ~0.75 GB with
   the f16 bytes embedded.
8. **Long horizon.** `predict(h = 1100)` is refused with the reason (69 patches > 64).

## Results

**Verdict: VALIDATED.** The port is exact, the cost is known, and 120M is servable in one binary
at a price the smaller models do not pay.

| | Chronos-2 (this spike) | Bolt-small (007) | Bolt-tiny (007) |
|---|---|---|---|
| params / f16 file | 119.5M / 228 MB | 47.7M / 95.5 MB | 8.7M / 17.3 MB |
| forward, 2048 points, 1 thread | 423 ms | 98 ms | 18.5 ms |
| forward, 8192 points | 1.67 s | – (2048 max) | – |
| quantiles / direct horizon / context | 21 / 1024 / 8192 | 9 / 64 / 2048 | 9 / 64 / 2048 |
| f16 penalty (of series scale) | 0.3–0.6 % (3 % on 5 pts) | – | 0.1 % |
| parity vs Python (f32) | 2e-5 of scale | 1e-6 abs | 1e-6 abs |
| peak RSS at load | 1.45 GB (0.5 GB steady) | ~0.3 GB | ~0.1 GB |

**Surprises**
- The port matched on the first run because the ladder made every assumption (arcsinh before
  patching, time encoding over the padded length, RoPE over the *whole* sequence including future
  patches, group attention collapsing) checkable at its own rung.
- f16 is not free at 12 layers: 0.3–0.6 % of scale on ordinary series and 3 % on a five-point
  series, versus 0.1 % for Bolt-tiny. bf16 was not tried; the checkpoint is f32.
- Threading buys 1.1× on the shape that matters because the parallel GEMM only splits rows.

**Signal for the build**
- Chronos-2 as a third, "large" tier of the same server (one model per binary): f32 or f16 is a
  build choice with a documented accuracy cost; horizon ≤ 1024 direct, refuse beyond (or port the
  unrolling later); expose the 21 quantiles; 8192-point context.
- Sizing: ~0.5 s per forecast on one core for 2–3k daily points, ~1.7 s at 8192; ~0.5 GB RAM after
  a streaming loader (this spike peaks at 1.45 GB); 228 MB f16 binary (container image on Lambda).
- Upstream: partition the parallel GEMM along N for M ≤ 256 (prefill shapes) — a measured issue
  with this spike's table; and a streaming safetensors loader would serve every model here.
- Not ported: multivariate/covariate groups, long-horizon unrolling. Both are pipeline-level
  additions on top of the same forward.
