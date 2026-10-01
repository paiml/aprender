---
spike: 005
idea: prophet-forecast-mcp
name: chronos-bolt-tiny-parity
type: standard
validates: "Given `amazon/chronos-bolt-tiny` safetensors, when its T5 encoder-decoder (patch embedding, instance scaling, REG token, relative position bias, quantile head) is run in Rust, then the 9 quantiles match the Python `chronos-forecasting` pipeline on Peyton and air passengers within a committed tolerance, including the autoregressive rollout past 64 steps"
verdict: VALIDATED
related: [001, 002, 004]
tags: [chronos, t5, zero-shot, safetensors, parity, gemm]
---

# Spike 005: Chronos-Bolt-tiny, zero-shot, in Rust

## What This Validates

Given the 8.65M-parameter `amazon/chronos-bolt-tiny` weights (float32 safetensors, Apache-2.0),
when the model is re-implemented from scratch in Rust (no torch, no transformers), then its nine
quantile forecasts equal the Python pipeline's to float32 rounding on three series, on six edge
cases, and through the 365-step rollout — and the cost of a forward pass is known.

## Research

Grounded in the `main` sources of `amazon-science/chronos-forecasting` (`chronos_bolt.py`) and
`huggingface/transformers` (`modeling_t5.py`), the Hub `config.json`, and an installed
`chronos-forecasting==2.3.1` oracle (`tools/oracle.py`, fixtures in `fixtures/`).

- **Model** (`config.json`): T5, `d_model 256`, `d_ff 1024`, 4 heads × `d_kv 64`, 4 encoder + 4 decoder
  layers, ReLU FFN (not gated), `layer_norm_epsilon 1e-6`, 32 relative buckets / max distance 128,
  `vocab_size 2` (pad + REG token), patch 16 / stride 16, context 2048, horizon 64, quantiles 0.1…0.9.
- **Pipeline** (`chronos_bolt.py`): instance norm `loc = nanmean`, `scale = sqrt(nanmean((x−loc)²))`
  (population std; 0 → 1e-5); truncate to the last 2048; left-pad with NaN to a multiple of 16;
  per patch `[16 values (NaN→0), 16 mask]` → `ResidualBlock(32→1024→256)`
  (`out(relu(hidden(x))) + residual(x)`); append `shared[1]` as REG; encoder attention mask = patch
  has any observed value, REG always on; decoder input = `shared[0]`; decoder last hidden →
  `ResidualBlock(256→1024→576)` → `[9][64]` → `× scale + loc`.
- **Rollout past 64** (2.3.1 = `main`): after the direct block, the 9 quantile paths are extended
  with their own quantile and run as a batch; the 81 values per step are re-quantiled at the 9
  levels with `torch.quantile` (linear interpolation). The older median-only scheme differs by 0.5.
- **T5** (`modeling_t5.py`): `T5LayerNorm` = RMS without mean subtraction; attention scores are
  **not** scaled (`scaling = 1.0`); relative position bias computed in layer 0 only and reused;
  bidirectional buckets for the encoder, causal for the decoder self-attention; cross-attention has
  zero bias; masked keys get `finfo.min` added; blocks are pre-norm residual `[self, (cross), FF]`;
  final layer norm; dropout is inference-inert.

| Approach | Pros | Cons | Status |
|---|---|---|---|
| From-scratch Rust (this spike) | Every op is visible and testable against the ladder | No SIMD; ~36 ms per forward | **Chosen** for parity |
| `trueno::Matrix::matmul` / `blis::gemm_blis` | Existing kernels | Both ran at ~3.6 GFLOP/s on 129×256×{256,1024}, 4× slower than plain unrolled loops | Measured, not used |
| Realizar's transformer path (`is_encoder_decoder`, relative positions) | Proven kernels, GGUF/APR loaders | Not exercised on T5 weights in this spike | Build-time path |

## How to Run

```bash
cd .planning/spikes/005-chronos-bolt-tiny-parity
# weights are gitignored (34.6 MB); the oracle downloads them and writes the fixtures:
uv run --python 3.12 --with chronos-forecasting --with pandas --with safetensors python tools/oracle.py /tmp/chronos-out
cp "$(grep MODEL_DIR /tmp/chronos-out/../oracle.log | awk '{print $2}')"/{model.safetensors,config.json} models/   # or copy from ~/.cache/huggingface/hub/models--amazon--chronos-bolt-tiny/snapshots/*/
CARGO_TARGET_DIR=../../../target cargo run --release      # ~12 s; writes report.html
```

## What to Expect

Section 2 prints a parity ladder per series (loc, scale, patch embeddings, encoder hidden states,
decoder hidden state, quantiles) with max abs diffs at 1e-7…1e-6; section 3 the 365-step rollout
at 1.5e-5 with a control showing the median-only scheme at 0.5; section 4 six edge probes; section 5
timings and a stage profile. `report.html` shows y, the q10–q90 fan, and Rust vs Python q50.

## Investigation Trail

1. **Ladder before quantiles.** Instance-norm `loc`/`scale`, the first/last patch embeddings and
   the REG embedding, encoder hidden states (first token, REG), and the decoder hidden state each
   match to 1e-7…5e-6 on all three series. Attention masks and token counts identical.
2. **Quantiles:** Peyton 9.5e-7, air 1.8e-4 (scale 119.5, i.e. 0.0002 %), short100 9.5e-7 — float32
   rounding, nothing else. `predict_quantiles([0.1,0.5,0.9])` and `mean` reproduce too.
3. **Rollout:** 365 steps within 1.5e-5 end to end. The control (median-only rollout) is off by
   0.51, which pins the installed pipeline to the 9-path scheme and confirms the port follows it.
4. **Edge probes** (fixtures from `tools/probes.py`): NaN gaps inside the context (mask path),
   a 5-point series (11 NaN left-pad, one patch), a constant series (`scale → 1e-5`, diff 9.5e-7),
   values × 1e6 (diff 1.0 on values ≈ 8e6 → float32), a 130-point series rolled to 130 steps,
   air passengers rolled to 24 — all match; masks equal on every case.
5. **First cost: 180 ms per forward** (torch 5.9 ms). Adding trueno's `Matrix::matmul` gave 151 ms.
   A stage profile showed every GEMM at ~3.5 GFLOP/s: the scalar dot product is a strict-order float
   reduction LLVM will not vectorise.
6. **8-accumulator dot product → 36 ms** (5×), parity unchanged. Same profile through
   `blis::gemm_blis`: 137 ms — trueno's packed GEMM is *slower* than the plain loops on these
   skinny shapes. Cause, by reading not by profiling: the inner tile loop at
   `crates/aprender-compute/src/blis/compute.rs:117` calls `microkernel_scalar`; a `microkernel_8x8_neon`
   exists in the same module but this path does not use it.
   Short series already beat torch: air 3.0 ms vs 2.6 ms, short100 2.3 ms vs 2.6 ms.

## Results

**Verdict: VALIDATED.** The zero-shot model runs in Rust with no torch and reproduces Python to
float32 rounding, including rollout and edge cases. Latency is 6× torch on a 2048-point context
with plain loops and already at parity on short series.

| series | context | tokens | quantiles max abs vs Python | Rust ms | torch ms |
|---|---|---|---|---|---|
| Peyton | 2905 → 2048 | 129 | 9.5e-7 | 36 | 5.9 |
| air passengers | 144 | 10 | 1.8e-4 (0.0002 % of scale) | 3.0 | 2.6 |
| short100 | 100 | 8 | 9.5e-7 | 2.3 | 2.6 |
| Peyton, 365-step rollout (46 forwards) | | | 1.5e-5 | 1640 | 81 |

**Surprises**
- The rollout scheme changed in 2025 (9 paths, re-quantiled); anyone porting from the paper or an
  old notebook gets a forecast that is off by half a unit past step 64.
- Both trueno GEMM entry points lost to hand-written loops by 4× here because `gemm_blis` runs the
  scalar microkernel (`blis/compute.rs:117`). Wiring `microkernel_8x8_neon` in is the core fix; verify
  on transformer-shaped GEMMs (129×256×1024) with a before/after table.
- A REG token, a 2-row "vocabulary" and a bias-free T5 are all the model needs; there is no tokenizer.

**Signal for the build**
- `src/bolt.rs` is a complete reference implementation: 240 lines, contract-shaped (ladder tensors
  exposed). Port it behind realizar's loaders, or keep it as the Chronos backend if realizar's T5
  path proves slower to adapt.
- Serve it exactly like SetFit: `aprender-mcp-chronos`, model embedded (34.6 MB f32; f16 halves it),
  one `forecast` tool sharing the spike-004 request/response shape, quantiles native (no simulation).
- Cost model for Lambda: 36 ms per 2048-point forward now; a NEON-vectorised attention and GEMM
  should reach torch's 6 ms. The 365-step rollout is 46 forwards — cap the horizon or document it.
- Chronos-2 (120M, RoPE, arcsinh, covariates) is the follow-on spike; Bolt-small/base are the same
  code with bigger configs.
