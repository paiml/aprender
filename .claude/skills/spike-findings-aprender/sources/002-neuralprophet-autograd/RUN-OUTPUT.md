# Spike 002 — NeuralProphet-lite on aprender's f32 autograd

Peyton Manning: 2905 rows, train 2540, test = last 365 rows (2015-01-19 … 2016-01-20).

## 0. Graph connectivity of core's losses (a Huber loss that cannot backprop cannot train NeuralProphet)

| loss | param grad after backward() |
|---|---|
| MSELoss | Some([-0.75, 3.75]) |
| SmoothL1Loss (Huber) | **None — detached** (builds output with `Tensor::new` from raw data) |

This spike therefore builds Huber from graph-connected ops (`sub`, `abs`, `pow`, `mul` by a constant 0/1 mask) — exact value and exact gradient.

## 1. Data preparation (NeuralProphet defaults)

- soft normalisation: shift (min) = 5.262690, scale (q95 − min) = 4.582609  — NP: shift "5.26269018890489", scale "4.582609246193553"
- time: t = (ds − 2007-12-10) / 2596 days  — NP: "2007-12-10 00:00:00" / "2596 days 00:00:00"
- changepoints_t (11 segments): ["0.0000", "0.0727", "0.1455", "0.2182", "0.2909", "0.3636", "0.4364", "0.5091", "0.5818", "0.6545", "0.7273"]
  NP changepoints max abs diff: 1.1e-16
- seasonalities: ["yearly P=365.25 R=6", "weekly P=7 R=3"]
- daily grid 2964 days (2540 observed in train, 57 imputed by linear interpolation); train grid 2597 days
- auto batch 64 / epochs 80 for 2540 lag-free samples (NP: batch 64, epochs 80)

## 2. Trend + seasonality (n_lags = 0): AdamW + one-cycle, learning-rate sweep (NP picks lr by a range test)

| max_lr | epochs | batch | steps | params | train secs | tape len/step | final train loss (Huber) | test MAE (365-ahead) | test RMSE |
|---|---|---|---|---|---|---|---|---|---|
| 0.003 | 80 | 64 | 3200 | 31 | 0.05 | 20 | 0.18774 | 1.1032 | 1.2006 |
| 0.01 | 80 | 64 | 3200 | 31 | 0.04 | 20 | 0.02462 | 0.4112 | 0.5296 |
| 0.03 | 80 | 64 | 3200 | 31 | 0.04 | 20 | 0.01542 | 0.4354 | 0.5498 |
| 0.1 | 80 | 64 | 3200 | 31 | 0.04 | 20 | 0.01498 | 0.4510 | 0.5624 |
| 0.3 | 80 | 64 | 3200 | 31 | 0.04 | 20 | 0.01504 | 0.4506 | 0.5609 |
| 1 | 80 | 64 | 3200 | 31 | 0.04 | 20 | 0.01540 | 0.4976 | 0.5989 |

Selected by lowest final TRAIN loss (never by test): max_lr = 0.1 (loss 0.01498) → test MAE **0.4510**.

## 3. Same split, other forecasters

| forecaster | fit secs | test MAE (365-ahead) | test RMSE |
|---|---|---|---|
| Rust NeuralProphet-lite (best train-loss lr 0.1) | 0.04 | **0.4510** | 0.5624 |
| Python NeuralProphet 0.9.0 (torch, auto lr-finder; 363 test rows covered) | 20.9 | **0.4611** | – |
| Rust Prophet (spike 001 config; Stalled after 628 iters) | 0.20 | **0.4125** | 0.5358 |
| last-year mean | 0 | 0.6894 | – |

## 4. AR-Net on 30 stationarised lags, one-step-ahead on the test year (true lags)

| model | epochs | steps | params | train secs | final train loss | test MAE (1-step) |
|---|---|---|---|---|---|---|
| Rust AR-Net linear (30 → 1) (lr 0.1 by train loss) | 80 | 3280 | 61 | 0.95 | 0.01141 | **0.3216** |
| Python NP AR-Net linear (30 → 1) | 80 | – | – | 23.5 | 0.00854 | **0.2557** |
| Rust AR-Net 30 → 32 → 1 (ReLU) (lr 0.3 by train loss) | 80 | 3280 | 1055 | 1.18 | 0.00805 | **0.2509** |
| Python NP AR-Net 30 → 32 → 1 (ReLU) | 80 | – | – | 24.1 | 0.01044 | **0.3933** |
| naive (previous observed row) | – | – | – | 0 | – | 0.3461 |

## 5. Robustness probes (trend + seasonality, lr 0.1)

- seed 1: final train loss 0.01494, test MAE 0.4549, all epoch losses finite: true
- seed 2: final train loss 0.01489, test MAE 0.4515, all epoch losses finite: true
- seed 3: final train loss 0.01494, test MAE 0.4508, all epoch losses finite: true
- half the epochs (40): final train loss 0.01519, test MAE 0.4535, 0.02s
- double the epochs (160): final train loss 0.01491, test MAE 0.4534, 0.09s
- full-batch (2540 rows, 80 steps): final train loss 0.34177, test MAE 2.6168, 0.04s
- no weight decay, no newer-sample weighting: final train loss 0.01499, test MAE 0.4511
- 60-row series: auto batch 8 / epochs 300, 0.02s, seasonalities ["weekly"], test MAE (next 30 rows) 3.0389

## 6. Autograd ops this model needed

`matmul` (2-D), `transpose`, `broadcast_add` (bias), `add`, `sub`, `mul` (same shape), `mul_scalar`, `abs`, `pow`, `mean`, `relu`, `view`, `backward`, `no_grad`, `clear_graph`. Missing but worked around: a graph-connected Huber loss (core's `SmoothL1Loss` is detached), a `where`/`clamp` op (constant mask instead), `cat` (avoided by reshaping the lag-time seasonality through `view`).

Wrote results.json and report.html.
