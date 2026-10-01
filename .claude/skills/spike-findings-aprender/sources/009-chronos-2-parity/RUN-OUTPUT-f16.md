# Spike 009 — Chronos-2 parity (Rust port vs chronos-forecasting 2.3.1)

Model: models/chronos-2-f16 — 119477664 params, weights F16, load 0.42 s (decode + transposes); torch threads in oracle: 10

## 1. Parity ladder (max |Δ| vs oracle)

| series | h | n | patches | tokens | loc | scale | patch feats | embed first/last | hidden first / REG / last | quantiles (21×h) | pipeline (trunc.) | Rust s | torch s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| peyton | 64 | 2905 | 182 | 187 | 4.8e-6 | 1.8e-7 | 5.6e-6 | 1.5e-4 | 2.3e-3 | 2.4e-3 (2.8e-3 of scale) | 2.4e-3 | 0.59 | 0.07 |
| peyton | 365 | 2905 | 182 | 206 | 4.8e-6 | 1.8e-7 | 5.6e-6 | 1.5e-4 | 2.4e-3 | 4.1e-3 (4.8e-3 of scale) | 4.1e-3 | 0.63 | 0.05 |
| peyton | 1024 | 2905 | 182 | 247 | 4.8e-6 | 1.8e-7 | 5.6e-6 | 1.5e-4 | 2.4e-3 | 5.3e-3 (6.2e-3 of scale) | 5.3e-3 | 0.77 | 0.05 |
| air | 24 | 144 | 9 | 12 | 0.0e0 | 1.5e-5 | 2.4e-7 | 1.8e-4 | 2.3e-3 | 2.0e-1 (1.7e-3 of scale) | 1.2e-1 | 0.09 | 0.02 |
| short100 | 64 | 100 | 7 | 12 | 0.0e0 | 6.0e-8 | 1.2e-7 | 1.5e-4 | 2.4e-3 | 4.1e-3 (5.4e-3 of scale) | 4.1e-3 | 0.09 | 0.02 |
| nan_gaps | 16 | 300 | 19 | 21 | 4.8e-7 | 0.0e0 | 6.1e-7 | 1.6e-4 | 2.3e-3 | 2.0e-3 (2.6e-3 of scale) | 2.0e-3 | 0.11 | 0.02 |
| leading_nan_patch | 16 | 116 | 8 | 10 | 0.0e0 | 6.0e-8 | 1.2e-7 | 1.5e-4 | 2.3e-3 | 1.3e-3 (1.8e-3 of scale) | 1.3e-3 | 0.09 | 0.02 |
| constant | 16 | 100 | 7 | 9 | 0.0e0 | 0.0e0 | 0.0e0 | 1.5e-4 | 2.4e-3 | 0.0e0 (0.0e0 of scale) | 0.0e0 | 0.09 | 0.02 |
| huge_scale | 40 | 256 | 16 | 20 | 1.5e0 | 0.0e0 | 2.0e-6 | 1.6e-4 | 2.3e-3 | 2.2e3 (2.9e-3 of scale) | 1.8e3 | 0.11 | 0.02 |
| negative | 40 | 256 | 16 | 20 | 1.9e-6 | 0.0e0 | 2.5e-6 | 1.5e-4 | 2.3e-3 | 1.4e-3 (1.8e-3 of scale) | 1.2e-3 | 0.11 | 0.04 |
| short5 | 16 | 5 | 1 | 3 | 0.0e0 | 0.0e0 | 1.2e-7 | 1.3e-4 | 2.6e-3 | 2.5e-2 (3.1e-2 of scale) | 2.5e-2 | 0.07 | 0.02 |

Worst quantile |Δ| relative to the series scale: 3.14e-2

## 2. Cost (Apple M4 Pro, single thread, packed GEMM with the spike-008 NEON kernel)

| context | h | patches | tokens | forward ms (median of 3) | GFLOP | GFLOP/s |
|---|---|---|---|---|---|---|
| 100 | 64 | 4 | 12 | 91 | 2.4 | 26.3 |
| 512 | 64 | 4 | 37 | 155 | 7.4 | 47.7 |
| 2048 | 64 | 4 | 133 | 423 | 27.0 | 63.9 |
| 2905 | 64 | 4 | 187 | 585 | 38.3 | 65.6 |
| 2048 | 365 | 23 | 152 | 474 | 31.0 | 65.3 |
| 2048 | 1024 | 64 | 193 | 613 | 39.6 | 64.6 |
| 8192 | 64 | 4 | 517 | 1666 | 112.3 | 67.4 |

predict(h=1100): refused — horizon 1100 needs 69 output patches > max_output_patches 64; the pipeline's long-horizon unrolling is not ported
