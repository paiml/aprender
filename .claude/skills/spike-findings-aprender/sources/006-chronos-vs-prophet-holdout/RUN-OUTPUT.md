# Spike 006 — zero-shot Chronos-Bolt vs fitted Prophet / NeuralProphet-lite, rolling-origin holdouts

Bolt-small parity on Peyton (same code as spike 005, bigger config): quantiles_64 max abs diff vs Python **1.9e-6**.

## Per-split results

MASE = MAE / in-sample seasonal-naive MAE (period 7 daily, 12 monthly). cov80 / width: 80 % band (Prophet: simulated; Chronos: q10–q90; NP-lite: residual sd). WQL3: weighted quantile loss over q10/q50/q90.

| series | train → h | model | MAE | MASE | RMSE | cov80 | width/σ | WQL3 | secs |
|---|---|---|---|---|---|---|---|---|---|
| peyton | 2540 → 365 | naive (last value) | 1.2421 | 3.263 | 1.3654 | – | – | – | 0.00 |
| peyton | 2540 → 365 | seasonal naive | 1.5846 | 4.162 | 1.7982 | – | – | – | 0.00 |
| peyton | 2540 → 365 | Prophet (Rust port) | 0.4131 | 1.085 | 0.5363 | 0.83 | 1.55 | 0.0333 | 0.65 |
| peyton | 2540 → 365 | NeuralProphet-lite (Rust) | 0.4510 | 1.185 | 0.5624 | 0.77 | 1.53 | 0.0354 | 0.14 |
| peyton | 2540 → 365 | chronos-bolt-tiny (zero-shot) | 0.4800 | 1.261 | 0.6195 | 0.44 | 1.12 | 0.0440 | 1.67 |
| peyton | 2540 → 365 | chronos-bolt-small (zero-shot) | 0.5007 | 1.315 | 0.6604 | 0.55 | 1.34 | 0.0449 | 10.09 |
| peyton | 2200 → 90 | naive (last value) | 0.6529 | 1.681 | 0.7038 | – | – | – | 0.00 |
| peyton | 2200 → 90 | seasonal naive | 1.2485 | 3.214 | 1.3595 | – | – | – | 0.00 |
| peyton | 2200 → 90 | Prophet (Rust port) | 0.5368 | 1.382 | 0.5953 | 0.66 | 1.47 | 0.0403 | 0.51 |
| peyton | 2200 → 90 | NeuralProphet-lite (Rust) | 0.3676 | 0.947 | 0.4442 | 0.89 | 1.55 | 0.0294 | 0.13 |
| peyton | 2200 → 90 | chronos-bolt-tiny (zero-shot) | 0.4739 | 1.220 | 0.5264 | 0.51 | 1.54 | 0.0392 | 0.37 |
| peyton | 2200 → 90 | chronos-bolt-small (zero-shot) | 0.2549 | 0.656 | 0.3075 | 0.83 | 1.32 | 0.0226 | 2.17 |
| peyton | 2400 → 90 | naive (last value) | 0.7025 | 1.854 | 0.9325 | – | – | – | 0.00 |
| peyton | 2400 → 90 | seasonal naive | 0.5667 | 1.496 | 0.7560 | – | – | – | 0.00 |
| peyton | 2400 → 90 | Prophet (Rust port) | 0.3934 | 1.038 | 0.5338 | 0.86 | 1.48 | 0.0296 | 0.35 |
| peyton | 2400 → 90 | NeuralProphet-lite (Rust) | 0.4345 | 1.147 | 0.5725 | 0.82 | 1.51 | 0.0318 | 0.14 |
| peyton | 2400 → 90 | chronos-bolt-tiny (zero-shot) | 0.5680 | 1.499 | 0.7492 | 0.59 | 1.49 | 0.0433 | 0.37 |
| peyton | 2400 → 90 | chronos-bolt-small (zero-shot) | 0.5775 | 1.524 | 0.7711 | 0.69 | 1.57 | 0.0429 | 2.18 |
| peyton | 2600 → 90 | naive (last value) | 0.2664 | 0.696 | 0.3632 | – | – | – | 0.00 |
| peyton | 2600 → 90 | seasonal naive | 0.2544 | 0.665 | 0.3354 | – | – | – | 0.00 |
| peyton | 2600 → 90 | Prophet (Rust port) | 0.2630 | 0.687 | 0.3770 | 0.91 | 1.48 | 0.0258 | 0.31 |
| peyton | 2600 → 90 | NeuralProphet-lite (Rust) | 0.3112 | 0.813 | 0.4005 | 0.91 | 1.53 | 0.0282 | 0.14 |
| peyton | 2600 → 90 | chronos-bolt-tiny (zero-shot) | 0.2947 | 0.770 | 0.3747 | 0.60 | 0.91 | 0.0291 | 0.37 |
| peyton | 2600 → 90 | chronos-bolt-small (zero-shot) | 0.4140 | 1.082 | 0.4830 | 0.54 | 1.23 | 0.0383 | 2.17 |
| peyton | 2815 → 90 | naive (last value) | 0.5136 | 1.367 | 0.8431 | – | – | – | 0.00 |
| peyton | 2815 → 90 | seasonal naive | 0.3942 | 1.049 | 0.6456 | – | – | – | 0.00 |
| peyton | 2815 → 90 | Prophet (Rust port) | 0.3875 | 1.031 | 0.6903 | 0.83 | 1.45 | 0.0351 | 0.48 |
| peyton | 2815 → 90 | NeuralProphet-lite (Rust) | 0.3810 | 1.014 | 0.6782 | 0.84 | 1.50 | 0.0335 | 0.15 |
| peyton | 2815 → 90 | chronos-bolt-tiny (zero-shot) | 0.5493 | 1.462 | 0.8350 | 0.64 | 1.42 | 0.0454 | 0.37 |
| peyton | 2815 → 90 | chronos-bolt-small (zero-shot) | 0.5520 | 1.469 | 0.8895 | 0.77 | 1.56 | 0.0468 | 2.17 |
| wp_log_r | 2498 → 365 | naive (last value) | 0.3830 | 2.500 | 0.4286 | – | – | – | 0.00 |
| wp_log_r | 2498 → 365 | seasonal naive | 0.5342 | 3.487 | 0.5979 | – | – | – | 0.00 |
| wp_log_r | 2498 → 365 | Prophet (Rust port) | 0.2699 | 1.762 | 0.3509 | 0.69 | 0.98 | 0.0221 | 0.25 |
| wp_log_r | 2498 → 365 | NeuralProphet-lite (Rust) | 0.1568 | 1.024 | 0.2369 | 0.88 | 0.94 | 0.0140 | 0.15 |
| wp_log_r | 2498 → 365 | chronos-bolt-tiny (zero-shot) | 0.3254 | 2.124 | 0.3900 | 0.49 | 0.81 | 0.0288 | 1.66 |
| wp_log_r | 2498 → 365 | chronos-bolt-small (zero-shot) | 0.3878 | 2.532 | 0.4476 | 0.30 | 0.81 | 0.0355 | 10.01 |
| wp_log_r | 2100 → 90 | naive (last value) | 0.4491 | 2.917 | 0.5264 | – | – | – | 0.00 |
| wp_log_r | 2100 → 90 | seasonal naive | 0.2536 | 1.647 | 0.3450 | – | – | – | 0.00 |
| wp_log_r | 2100 → 90 | Prophet (Rust port) | 0.1246 | 0.810 | 0.2323 | 0.93 | 0.94 | 0.0126 | 0.71 |
| wp_log_r | 2100 → 90 | NeuralProphet-lite (Rust) | 0.1565 | 1.017 | 0.2692 | 0.92 | 0.99 | 0.0149 | 0.23 |
| wp_log_r | 2100 → 90 | chronos-bolt-tiny (zero-shot) | 0.2488 | 1.616 | 0.3506 | 0.50 | 0.58 | 0.0238 | 0.36 |
| wp_log_r | 2100 → 90 | chronos-bolt-small (zero-shot) | 0.2799 | 1.818 | 0.3807 | 0.34 | 0.54 | 0.0283 | 2.17 |
| wp_log_r | 2400 → 90 | naive (last value) | 0.2418 | 1.591 | 0.3441 | – | – | – | 0.00 |
| wp_log_r | 2400 → 90 | seasonal naive | 0.2206 | 1.451 | 0.2799 | – | – | – | 0.00 |
| wp_log_r | 2400 → 90 | Prophet (Rust port) | 0.1022 | 0.672 | 0.1905 | 0.94 | 0.89 | 0.0107 | 0.68 |
| wp_log_r | 2400 → 90 | NeuralProphet-lite (Rust) | 0.1401 | 0.921 | 0.2300 | 0.94 | 0.94 | 0.0128 | 0.14 |
| wp_log_r | 2400 → 90 | chronos-bolt-tiny (zero-shot) | 0.2657 | 1.748 | 0.3227 | 0.42 | 0.73 | 0.0234 | 0.36 |
| wp_log_r | 2400 → 90 | chronos-bolt-small (zero-shot) | 0.2610 | 1.717 | 0.3199 | 0.37 | 0.65 | 0.0233 | 2.17 |
| wp_log_r | 2773 → 90 | naive (last value) | 0.3094 | 1.977 | 0.4268 | – | – | – | 0.00 |
| wp_log_r | 2773 → 90 | seasonal naive | 0.2681 | 1.713 | 0.3730 | – | – | – | 0.00 |
| wp_log_r | 2773 → 90 | Prophet (Rust port) | 0.1372 | 0.877 | 0.1801 | 0.89 | 0.85 | 0.0119 | 0.52 |
| wp_log_r | 2773 → 90 | NeuralProphet-lite (Rust) | 0.1790 | 1.144 | 0.2370 | 0.78 | 0.90 | 0.0149 | 0.15 |
| wp_log_r | 2773 → 90 | chronos-bolt-tiny (zero-shot) | 0.2690 | 1.719 | 0.3394 | 0.58 | 0.79 | 0.0233 | 0.36 |
| wp_log_r | 2773 → 90 | chronos-bolt-small (zero-shot) | 0.2815 | 1.799 | 0.3698 | 0.53 | 0.67 | 0.0261 | 2.17 |
| air | 120 → 24 | naive (last value) | 115.2500 | 4.033 | 137.3290 | – | – | – | 0.00 |
| air | 120 → 24 | seasonal naive | 71.2500 | 2.494 | 76.9946 | – | – | – | 0.00 |
| air | 120 → 24 | Prophet (Rust port) | 31.1555 | 1.090 | 40.3576 | 0.50 | 0.50 | 0.0492 | 0.01 |
| air | 120 → 24 | NeuralProphet-lite (Rust) | 32.9116 | 1.152 | 45.5826 | 0.50 | 0.48 | 0.0560 | 0.06 |
| air | 120 → 24 | chronos-bolt-tiny (zero-shot) | 36.8971 | 1.291 | 48.0896 | 0.75 | 1.04 | 0.0504 | 0.00 |
| air | 120 → 24 | chronos-bolt-small (zero-shot) | 29.5631 | 1.035 | 38.6070 | 0.79 | 1.04 | 0.0406 | 0.02 |
| air | 96 → 12 | naive (last value) | 63.4167 | 2.172 | 83.4740 | – | – | – | 0.00 |
| air | 96 → 12 | seasonal naive | 40.1667 | 1.375 | 41.4749 | – | – | – | 0.00 |
| air | 96 → 12 | Prophet (Rust port) | 24.3569 | 0.834 | 28.9389 | 0.33 | 0.46 | 0.0479 | 0.03 |
| air | 96 → 12 | NeuralProphet-lite (Rust) | 23.2661 | 0.797 | 26.0841 | 0.33 | 0.42 | 0.0443 | 0.08 |
| air | 96 → 12 | chronos-bolt-tiny (zero-shot) | 22.3539 | 0.765 | 30.7832 | 0.67 | 1.04 | 0.0417 | 0.00 |
| air | 96 → 12 | chronos-bolt-small (zero-shot) | 19.3577 | 0.663 | 25.7617 | 0.75 | 1.11 | 0.0343 | 0.01 |
| air | 108 → 12 | naive (last value) | 52.3333 | 1.712 | 76.4341 | – | – | – | 0.00 |
| air | 108 → 12 | seasonal naive | 12.5833 | 0.412 | 17.0123 | – | – | – | 0.00 |
| air | 108 → 12 | Prophet (Rust port) | 40.8235 | 1.335 | 44.0166 | 0.08 | 0.45 | 0.0824 | 0.01 |
| air | 108 → 12 | NeuralProphet-lite (Rust) | 37.8557 | 1.238 | 41.1643 | 0.08 | 0.44 | 0.0756 | 0.05 |
| air | 108 → 12 | chronos-bolt-tiny (zero-shot) | 29.7046 | 0.972 | 33.1446 | 0.83 | 1.01 | 0.0439 | 0.00 |
| air | 108 → 12 | chronos-bolt-small (zero-shot) | 27.0935 | 0.886 | 30.2411 | 0.83 | 0.96 | 0.0415 | 0.01 |
| air | 132 → 12 | naive (last value) | 76.0000 | 2.496 | 102.9765 | – | – | – | 0.00 |
| air | 132 → 12 | seasonal naive | 47.8333 | 1.571 | 50.7083 | – | – | – | 0.00 |
| air | 132 → 12 | Prophet (Rust port) | 33.7679 | 1.109 | 44.0190 | 0.50 | 0.48 | 0.0510 | 0.01 |
| air | 132 → 12 | NeuralProphet-lite (Rust) | 31.4701 | 1.034 | 40.6249 | 0.50 | 0.49 | 0.0454 | 0.06 |
| air | 132 → 12 | chronos-bolt-tiny (zero-shot) | 27.1009 | 0.890 | 36.0224 | 0.75 | 0.77 | 0.0371 | 0.00 |
| air | 132 → 12 | chronos-bolt-small (zero-shot) | 18.8824 | 0.620 | 22.2980 | 0.83 | 0.65 | 0.0251 | 0.02 |
| retail | 269 → 24 | naive (last value) | 24957.3333 | 1.602 | 32377.2285 | – | – | – | 0.00 |
| retail | 269 → 24 | seasonal naive | 21196.6667 | 1.361 | 23298.0355 | – | – | – | 0.00 |
| retail | 269 → 24 | Prophet (Rust port) | 12834.7340 | 0.824 | 15979.9608 | 0.46 | 0.26 | 0.0201 | 0.04 |
| retail | 269 → 24 | NeuralProphet-lite (Rust) | 10076.3736 | 0.647 | 11638.7580 | 0.75 | 0.35 | 0.0134 | 0.08 |
| retail | 269 → 24 | chronos-bolt-tiny (zero-shot) | 14189.5039 | 0.911 | 19042.7246 | 0.83 | 0.77 | 0.0214 | 0.00 |
| retail | 269 → 24 | chronos-bolt-small (zero-shot) | 10891.2031 | 0.699 | 13939.8911 | 0.96 | 0.75 | 0.0173 | 0.03 |
| retail | 221 → 12 | naive (last value) | 18840.0000 | 1.274 | 24775.0886 | – | – | – | 0.00 |
| retail | 221 → 12 | seasonal naive | 21020.3333 | 1.422 | 21813.5100 | – | – | – | 0.00 |
| retail | 221 → 12 | Prophet (Rust port) | 22377.6294 | 1.514 | 26182.3003 | 0.17 | 0.31 | 0.0472 | 0.02 |
| retail | 221 → 12 | NeuralProphet-lite (Rust) | 13874.7946 | 0.938 | 17816.3442 | 0.50 | 0.35 | 0.0276 | 0.08 |
| retail | 221 → 12 | chronos-bolt-tiny (zero-shot) | 16908.3073 | 1.144 | 23319.6308 | 0.75 | 0.74 | 0.0301 | 0.00 |
| retail | 221 → 12 | chronos-bolt-small (zero-shot) | 12394.6901 | 0.838 | 16331.4560 | 0.83 | 0.55 | 0.0214 | 0.03 |
| retail | 245 → 12 | naive (last value) | 21982.6667 | 1.403 | 26669.1472 | – | – | – | 0.00 |
| retail | 245 → 12 | seasonal naive | 14249.8333 | 0.909 | 15526.9480 | – | – | – | 0.00 |
| retail | 245 → 12 | Prophet (Rust port) | 27196.6560 | 1.736 | 28470.1491 | 0.00 | 0.39 | 0.0484 | 0.02 |
| retail | 245 → 12 | NeuralProphet-lite (Rust) | 29347.2371 | 1.873 | 30566.4682 | 0.08 | 0.39 | 0.0538 | 0.08 |
| retail | 245 → 12 | chronos-bolt-tiny (zero-shot) | 12142.9453 | 0.775 | 15532.0667 | 1.00 | 0.78 | 0.0192 | 0.00 |
| retail | 245 → 12 | chronos-bolt-small (zero-shot) | 10430.5026 | 0.666 | 12061.9197 | 0.92 | 0.51 | 0.0151 | 0.03 |
| retail | 281 → 12 | naive (last value) | 23535.1667 | 1.512 | 30880.1070 | – | – | – | 0.00 |
| retail | 281 → 12 | seasonal naive | 11994.3333 | 0.771 | 13468.0483 | – | – | – | 0.00 |
| retail | 281 → 12 | Prophet (Rust port) | 11810.6695 | 0.759 | 13695.1549 | 0.58 | 0.24 | 0.0165 | 0.05 |
| retail | 281 → 12 | NeuralProphet-lite (Rust) | 9243.1838 | 0.594 | 10793.4657 | 0.75 | 0.25 | 0.0122 | 0.09 |
| retail | 281 → 12 | chronos-bolt-tiny (zero-shot) | 13709.3854 | 0.881 | 18776.6189 | 0.75 | 0.63 | 0.0208 | 0.01 |
| retail | 281 → 12 | chronos-bolt-small (zero-shot) | 11444.0260 | 0.735 | 15094.8104 | 0.83 | 0.53 | 0.0163 | 0.03 |

## Summary — mean MASE over the rolling origins (lower is better; 1.0 = seasonal naive in-sample)

| model | peyton | wp_log_r | air | retail | mean | MASE steps 1–64 | MASE steps 65+ (daily 90/365 windows) | mean cov80 | mean width/σ | mean WQL3 | total secs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| naive (last value) | 1.772 | 2.246 | 2.603 | 1.448 | **2.017** | 1.963 | 2.094 | – | – | – | 0.0 |
| seasonal naive | 2.117 | 2.075 | 1.463 | 1.116 | **1.693** | 1.672 | 2.211 | – | – | – | 0.0 |
| Prophet (Rust port) | 1.045 | 1.030 | 1.092 | 1.208 | **1.094** | 1.070 | 1.058 | 0.60 | 0.83 | 0.0343 | 4.6 |
| NeuralProphet-lite (Rust) | 1.021 | 1.026 | 1.055 | 1.013 | **1.029** | 1.011 | 1.083 | 0.66 | 0.86 | 0.0320 | 1.9 |
| chronos-bolt-tiny (zero-shot) | 1.242 | 1.802 | 0.980 | 0.928 | **1.238** | 1.193 | 1.640 | 0.65 | 0.95 | 0.0332 | 5.9 |
| chronos-bolt-small (zero-shot) | 1.209 | 1.966 | 0.801 | 0.735 | **1.178** | 1.120 | 1.733 | 0.69 | 0.93 | 0.0306 | 35.5 |

## Cross-check against the Python Chronos oracle (same splits)

- 34 Chronos forecasts compared: max |MAE(Rust) − MAE(Python)| = **1.0e-2** (the metric plumbing and the port agree); total Chronos time Rust 41.4 s vs torch 1.5 s

Total wall time 48 s.
Wrote results.json and report.html.
