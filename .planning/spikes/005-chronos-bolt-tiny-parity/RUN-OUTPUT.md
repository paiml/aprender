# Spike 005 — Chronos-Bolt-tiny in Rust vs `chronos-forecasting`

Loaded 101 tensors, 8652672 parameters, in 0.030s. Config: d_model 256, d_ff 1024, 4 heads × d_kv 64, 4+4 layers, patch 16, context 2048, horizon 64, 9 quantiles, REG token true.

## 1. Relative position buckets (bidirectional / causal) for sampled offsets

-200:15/31  -128:15/31  -100:15/30  -17:10/16  -16:10/16  -8:8/8  -1:1/1  0:0/0  1:17/0  7:23/0  8:24/0  15:25/0  16:26/0  31:27/0  64:30/0  127:31/0  128:31/0  500:31/0

## 2. Parity ladder (max abs diff vs Python, float32 both sides)

| series | n | patches+REG | loc | scale | input_embeds first/last/REG | encoder first/REG | decoder hidden | quantiles_64 max abs | as % of scale | Rust ms | torch ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| peyton | 2905 | 129 | 9.5e-7 | 1.8e-7 | 6.0e-7 / 3.6e-7 / 0.0e0 | 1.4e-6 / 5.2e-7 | 5.0e-6 | **9.54e-7** | 0.0001% | 38.0 | 5.9 |
| air | 144 | 10 | 0.0e0 | 1.5e-5 | 3.3e-7 / 4.5e-7 / 0.0e0 | 5.7e-7 / 2.7e-7 | 2.0e-6 | **1.83e-4** | 0.0002% | 3.0 | 2.6 |
| short100 | 100 | 8 | 0.0e0 | 6.0e-8 | 2.4e-7 / 3.6e-7 / 0.0e0 | 3.6e-7 / 3.0e-7 | 7.6e-6 | **9.54e-7** | 0.0001% | 2.4 | 2.6 |

## 3. Rollout past the native horizon (Peyton, 365 steps: 1 direct block + 5 batched 9-path blocks, re-quantiled over 81 values)

- max abs diff: first 64 steps 9.54e-7, all 365 steps **1.72e-5**, steps 300–365 1.72e-5 (y range ≈ 7.6); Rust 1640 ms vs torch 81 ms
- control, median-only rollout (pre-2025 pipeline): max abs diff vs Python all steps 5.11e-1
- `predict_quantiles([0.1,0.5,0.9])` first 5 steps: max abs diff 9.54e-7; `mean` (= q50) 9.54e-7

## 4. Edge probes vs Python (chronos-forecasting "2.3.1")

| case | n | loc diff | scale diff | mask equal | quantiles max abs | as % of scale |
|---|---|---|---|---|---|---|
| air_rollout_24 | 144 → 24 | 0.0e0 | 1.5e-5 | true | 1.22e-4 | 0.0001% |
| constant | 64 → 64 | 0.0e0 | 0.0e0 | true | 9.54e-7 | 9.5367% |
| five_points | 5 → 64 | 0.0e0 | 0.0e0 | true | 3.81e-6 | 0.0005% |
| huge_scale | 200 → 64 | 0.0e0 | 1.2e-1 | true | 1.00e0 | 0.0001% |
| nan_gaps | 300 → 64 | 2.9e-6 | 0.0e0 | true | 1.91e-6 | 0.0003% |
| tiny_len_130_rollout | 130 → 130 | 1.9e-6 | 1.2e-7 | true | 2.86e-6 | 0.0004% |

## 5. Cost

| path | forward ms (Peyton, 129 tokens) | quantiles_64 max abs vs Python | 365-step rollout ms | rollout max abs |
|---|---|---|---|---|
| plain Rust loops, 8-accumulator dot | 36.2 | 9.5e-7 | 1640 | 1.7e-5 |
| trueno `blis::gemm_blis` packed GEMM, weights pre-transposed | **137.2** | 9.5e-7 | **6397** | 1.5e-5 |
| torch (Python, 1 thread default) | 5.9 | – | 81 | – |

Stage profile (Peyton, 129 tokens), plain loops vs gemm_blis:

| stage | plain ms | gemm_blis ms |
|---|---|---|
| position bias (129×129×4 buckets) | 0.08 | 0.08 |
| layer norm | 0.02 | 0.02 |
| one 256→256 projection | 0.51 | 2.38 |
| self-attention (4 projections + scores + context) | 2.81 | 10.43 |
| FF wi 256→1024 | 2.03 | 9.54 |
| FF wo 1024→256 | 2.34 | 9.58 |
| full encoder (4 layers) | 29.66 | 117.41 |
- weights: 8652672 params = 34.6 MB f32; load 0 ms

Wrote report.html.
