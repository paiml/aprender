# Spike 009 — Chronos-2 parity (Rust port vs chronos-forecasting 2.3.1)

Model: models/chronos-2 — 119477664 params, weights F32, load 0.44 s (decode + transposes); torch threads in oracle: 10

## 1. Parity ladder (max |Δ| vs oracle)

| series | h | n | patches | tokens | loc | scale | patch feats | embed first/last | hidden first / REG / last | quantiles (21×h) | pipeline (trunc.) | Rust s | torch s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| peyton | 64 | 2905 | 182 | 187 | 4.8e-6 | 1.8e-7 | 5.6e-6 | 1.1e-6 | 7.7e-6 | 1.6e-5 (1.9e-5 of scale) | 1.6e-5 | 0.59 | 0.07 |
| peyton | 365 | 2905 | 182 | 206 | 4.8e-6 | 1.8e-7 | 5.6e-6 | 1.1e-6 | 7.5e-6 | 2.7e-5 (3.2e-5 of scale) | 2.7e-5 | 0.64 | 0.05 |
| peyton | 1024 | 2905 | 182 | 247 | 4.8e-6 | 1.8e-7 | 5.6e-6 | 1.1e-6 | 9.7e-6 | 4.0e-5 (4.7e-5 of scale) | 4.0e-5 | 0.76 | 0.05 |
| air | 24 | 144 | 9 | 12 | 0.0e0 | 1.5e-5 | 2.4e-7 | 1.4e-7 | 2.9e-6 | 1.1e-3 (9.2e-6 of scale) | 9.8e-4 | 0.10 | 0.02 |
| short100 | 64 | 100 | 7 | 12 | 0.0e0 | 6.0e-8 | 1.2e-7 | 1.2e-7 | 1.4e-6 | 1.9e-5 (2.5e-5 of scale) | 1.9e-5 | 0.09 | 0.02 |
| nan_gaps | 16 | 300 | 19 | 21 | 4.8e-7 | 0.0e0 | 6.1e-7 | 1.6e-7 | 1.7e-6 | 5.7e-6 (7.4e-6 of scale) | 5.7e-6 | 0.11 | 0.02 |
| leading_nan_patch | 16 | 116 | 8 | 10 | 0.0e0 | 6.0e-8 | 1.2e-7 | 1.8e-7 | 1.9e-6 | 9.5e-6 (1.3e-5 of scale) | 9.5e-6 | 0.10 | 0.02 |
| constant | 16 | 100 | 7 | 9 | 0.0e0 | 0.0e0 | 0.0e0 | 2.0e-7 | 9.5e-7 | 0.0e0 (0.0e0 of scale) | 0.0e0 | 0.10 | 0.02 |
| huge_scale | 40 | 256 | 16 | 20 | 1.5e0 | 0.0e0 | 2.0e-6 | 3.4e-7 | 2.5e-6 | 1.7e1 (2.2e-5 of scale) | 1.7e1 | 0.12 | 0.02 |
| negative | 40 | 256 | 16 | 20 | 1.9e-6 | 0.0e0 | 2.5e-6 | 4.5e-7 | 3.8e-6 | 1.1e-5 (1.5e-5 of scale) | 1.1e-5 | 0.11 | 0.04 |
| short5 | 16 | 5 | 1 | 3 | 0.0e0 | 0.0e0 | 1.2e-7 | 7.6e-8 | 3.3e-6 | 1.8e-4 (2.3e-4 of scale) | 1.8e-4 | 0.07 | 0.02 |

Worst quantile |Δ| relative to the series scale: 2.27e-4

## 2. Cost (Apple M4 Pro, single thread, packed GEMM with the spike-008 NEON kernel)

| context | h | patches | tokens | forward ms (median of 3) | GFLOP | GFLOP/s |
|---|---|---|---|---|---|---|
| 100 | 64 | 4 | 12 | 90 | 2.4 | 26.4 |
| 512 | 64 | 4 | 37 | 156 | 7.4 | 47.2 |
| 2048 | 64 | 4 | 133 | 423 | 27.0 | 63.8 |
| 2905 | 64 | 4 | 187 | 586 | 38.3 | 65.4 |
| 2048 | 365 | 23 | 152 | 471 | 31.0 | 65.7 |
| 2048 | 1024 | 64 | 193 | 615 | 39.6 | 64.4 |
| 8192 | 64 | 4 | 517 | 1660 | 112.3 | 67.7 |

## 3. Threads: rayon-parallel `blis::gemm` (crate feature `parallel`, 14 cores) vs single-threaded `gemm_blis`

| context | h | tokens | serial ms | parallel ms | speed-up | max abs Δ |
|---|---|---|---|---|---|---|
| 512 | 64 | 37 | 157 | 122 | 1.3× | 0.0e0 |
| 2048 | 64 | 133 | 423 | 391 | 1.1× | 0.0e0 |
| 2905 | 64 | 187 | 586 | 413 | 1.4× | 0.0e0 |
| 8192 | 64 | 517 | 1651 | 771 | 2.1× | 0.0e0 |

predict(h=1100): refused — horizon 1100 needs 69 output patches > max_output_patches 64; the pipeline's long-horizon unrolling is not ported
