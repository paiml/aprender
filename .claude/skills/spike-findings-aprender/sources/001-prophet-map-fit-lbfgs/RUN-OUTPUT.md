# Spike 001 — Prophet MAP via aprender `LbfgsF64` vs Python Prophet (Stan L-BFGS)

---

# Dataset `peyton_manning` — 2905 rows, Python Prophet 1.4.0 fit 0.33s
Auto seasonalities: Rust [("yearly", 365.25, 10), ("weekly", 7.0, 3)] vs Python [("yearly", 365.25, 10), ("weekly", 7.0, 3)]

## 1. Data preparation parity (Rust from raw ds/y vs Prophet's Stan data)

| quantity | max abs diff |
|---|---|
| t (scaled time) | 0.00e0 |
| y_scaled (y_scale 12.846746888829 vs 12.846746888829) | 0.00e0 |
| changepoints_t (25) | 0.00e0 |
| Fourier X first/last 3 rows (K=26) | 0.00e0 |

## 2. Objective parity at Python's MAP

Rust f(θ_py) = -8004.797953, Python −lp = -8004.797953, abs diff 9.1e-13. Gradient norm where Stan stopped: 58.936 (without the 25 δ subgradients: 4.258). Python δ active: 13 of 25.

## 3. Analytic gradient vs central finite differences (h = 1e-6)

| point | L1 mode | worst abs | worst rel |
|---|---|---|---|
| init | exact | 7.70e-8 | 4.49e-8 |
| init | smooth 1e-4 | 7.70e-8 | 4.49e-8 |
| python MAP | exact | 1.97e1 | 1.70e0 |
| python MAP | smooth 1e-4 | 1.02e-4 | 5.97e-5 |
| perturbed MAP (no δ at 0) | exact | 6.34e-6 | 2.14e-9 |

## 4. L-BFGS fits from Prophet's init (k, m through the endpoints; δ = β = 0; σ = 1)

Python f* = -8004.7980

| variant | status | iters | f evals | secs | f (exact L1) | Δf vs Python | grad norm | σ | active δ |
|---|---|---|---|---|---|---|---|---|---|
| exact, raw, m=5, tol 1e-5 | NumericalError | 0 | 3 | 0.000 | 19.4685 | +8024.2664 | 2899.293 | 1.00000 | 0 |
| exact, f/T, guard, m=20, tol 1e-7 | Stalled | 726 | 2677 | 0.259 | -8004.9295 | -0.1315 | 0.022 | 0.03763 | 11 |
| exact, f/T, guard, m=5, tol 1e-7 | Stalled | 815 | 3324 | 0.305 | -8004.4801 | +0.3178 | 0.022 | 0.03768 | 14 |
| smooth 1e-4, raw, guard, m=5, tol 1e-5 | Stalled | 8321 | 146220 | 13.504 | -8005.1536 | -0.3556 | 0.001 | 0.03761 | 10 |

## 5. Forecast parity vs Python (history 2905 rows + 365 daily future; y range 7.584)

| variant | yhat hist max abs | yhat hist RMSE | yhat future max abs | yhat future RMSE | trend future max abs | weekly max abs | yearly max abs |
|---|---|---|---|---|---|---|---||
| exact, raw, m=5, tol 1e-5 | 3.0356 | 1.31823 | 2.6840 | 1.58236 | 1.6130 | 0.3523 | 1.1153 |
| exact, f/T, guard, m=20, tol 1e-7 | 0.0082 | 0.00307 | 0.0049 | 0.00357 | 0.0047 | 0.0000 | 0.0005 |
| exact, f/T, guard, m=5, tol 1e-7 | 0.0130 | 0.00523 | 0.0131 | 0.01018 | 0.0136 | 0.0000 | 0.0007 |
| smooth 1e-4, raw, guard, m=5, tol 1e-5 | 0.0224 | 0.00689 | 0.0080 | 0.00532 | 0.0068 | 0.0000 | 0.0021 |

Python params through Rust predict(): yhat max abs diff 5.3e-15, trend 3.6e-15 (predict-path parity).

## 6. Robustness probes (recommended config)

- (a) restart from a perturbed point: Stalled, 243 iters, f = -8004.3913 (Δ vs recommended +5.4e-1); yhat max abs diff vs recommended 2.4e-2
- (b) y × 1e6: Stalled, yhat/1e6 max abs diff vs unscaled 2.1e-2
- (c) first 60 rows only (Python would use Newton; seasonalities ["weekly"]; 25 changepoints): Stalled, 73 iters, 0.000s, σ = 0.0396, active δ 5
- (c) first 30 rows only (Python would use Newton; seasonalities ["weekly"]; 23 changepoints): Stalled, 68 iters, 0.000s, σ = 0.0251, active δ 11
- (d) n_changepoints = 0: Stalled, 25 iters, k = 0.0114, σ = 0.0458
- (e) constant y (200 rows): MaxIterations, 10000 iters, σ = 4.3e-8, f = -3356.5 — unbounded below as σ → 0; Prophet special-cases y.min()==y.max() (σ = 1e-9, no fit) and so must we
- (f) cost model: 2677 objective evals in 0.259s = 97 µs per eval at T×K = 2905×26

---

# Dataset `air_passengers` — 144 rows, Python Prophet 1.4.0 fit 0.02s
Auto seasonalities: Rust [("yearly", 365.25, 10)] vs Python [("yearly", 365.25, 10)]

## 1. Data preparation parity (Rust from raw ds/y vs Prophet's Stan data)

| quantity | max abs diff |
|---|---|
| t (scaled time) | 0.00e0 |
| y_scaled (y_scale 622 vs 622) | 0.00e0 |
| changepoints_t (25) | 0.00e0 |
| Fourier X first/last 3 rows (K=20) | 0.00e0 |

## 2. Objective parity at Python's MAP

Rust f(θ_py) = -401.979952, Python −lp = -401.979952, abs diff 5.7e-14. Gradient norm where Stan stopped: 61.626 (without the 25 δ subgradients: 8.813). Python δ active: 5 of 25.

## 4. L-BFGS fits from Prophet's init (k, m through the endpoints; δ = β = 0; σ = 1)

Python f* = -401.9800

| variant | status | iters | f evals | secs | f (exact L1) | Δf vs Python | grad norm | σ | active δ |
|---|---|---|---|---|---|---|---|---|---|
| exact, raw, m=5, tol 1e-5 | Stalled | 141 | 633 | 0.002 | -401.9291 | +0.0508 | 71.739 | 0.03600 | 5 |
| exact, f/T, guard, m=20, tol 1e-7 | Stalled | 55 | 298 | 0.001 | -401.5666 | +0.4134 | 0.575 | 0.03622 | 10 |
| exact, f/T, guard, m=5, tol 1e-7 | Stalled | 226 | 931 | 0.003 | -401.9678 | +0.0121 | 0.660 | 0.03579 | 5 |
| smooth 1e-4, raw, guard, m=5, tol 1e-5 | Converged | 2215 | 18564 | 0.057 | -402.3859 | -0.4059 | 0.000 | 0.03597 | 1 |

## 5. Forecast parity vs Python (history 144 rows + 365 daily future; y range 518.000)

| variant | yhat hist max abs | yhat hist RMSE | yhat future max abs | yhat future RMSE | trend future max abs | yearly max abs |
|---|---|---|---|---|---|---||
| exact, raw, m=5, tol 1e-5 | 1.7584 | 0.61002 | 4.5327 | 1.56830 | 0.1612 | 4.4356 |
| exact, f/T, guard, m=20, tol 1e-7 | 2.1145 | 0.79228 | 35.3831 | 11.45289 | 0.5705 | 35.9422 |
| exact, f/T, guard, m=5, tol 1e-7 | 0.7628 | 0.30943 | 2.2127 | 0.79726 | 0.1941 | 2.0916 |
| smooth 1e-4, raw, guard, m=5, tol 1e-5 | 2.9998 | 0.95038 | 58.2810 | 21.17237 | 1.0791 | 59.2923 |

Python params through Rust predict(): yhat max abs diff 2.3e-13, trend 2.3e-13 (predict-path parity).

---

# Dataset `retail_sales` — 293 rows, Python Prophet 1.4.0 fit 0.05s
Auto seasonalities: Rust [("yearly", 365.25, 10)] vs Python [("yearly", 365.25, 10)]

## 1. Data preparation parity (Rust from raw ds/y vs Prophet's Stan data)

| quantity | max abs diff |
|---|---|
| t (scaled time) | 0.00e0 |
| y_scaled (y_scale 518253 vs 518253) | 0.00e0 |
| changepoints_t (25) | 0.00e0 |
| Fourier X first/last 3 rows (K=20) | 0.00e0 |

## 2. Objective parity at Python's MAP

Rust f(θ_py) = -1020.946964, Python −lp = -1020.946964, abs diff 2.3e-13. Gradient norm where Stan stopped: 49.169 (without the 25 δ subgradients: 2.864). Python δ active: 9 of 25.

## 4. L-BFGS fits from Prophet's init (k, m through the endpoints; δ = β = 0; σ = 1)

Python f* = -1020.9470

| variant | status | iters | f evals | secs | f (exact L1) | Δf vs Python | grad norm | σ | active δ |
|---|---|---|---|---|---|---|---|---|---|
| exact, raw, m=5, tol 1e-5 | Stalled | 387 | 1433 | 0.009 | -1019.7275 | +1.2195 | 234.198 | 0.01560 | 20 |
| exact, f/T, guard, m=20, tol 1e-7 | Stalled | 413 | 1510 | 0.011 | -1020.9277 | +0.0193 | 0.366 | 0.01492 | 11 |
| exact, f/T, guard, m=5, tol 1e-7 | Stalled | 318 | 1185 | 0.008 | -1016.1394 | +4.8076 | 2.362 | 0.01467 | 23 |
| smooth 1e-4, raw, guard, m=5, tol 1e-5 | MaxIterations | 10000 | 39968 | 0.255 | -1023.4372 | -2.4902 | 0.004 | 0.01474 | 9 |

## 5. Forecast parity vs Python (history 293 rows + 365 daily future; y range 371877.000)

| variant | yhat hist max abs | yhat hist RMSE | yhat future max abs | yhat future RMSE | trend future max abs | yearly max abs |
|---|---|---|---|---|---|---||
| exact, raw, m=5, tol 1e-5 | 603.4715 | 192.12658 | 5565.8037 | 1607.59867 | 475.2353 | 6021.5212 |
| exact, f/T, guard, m=20, tol 1e-7 | 2139.1920 | 684.15516 | 1702.2072 | 1083.87038 | 1260.7032 | 756.6119 |
| exact, f/T, guard, m=5, tol 1e-7 | 2220.5243 | 736.23192 | 2855.0579 | 1073.60349 | 650.8608 | 3158.5840 |
| smooth 1e-4, raw, guard, m=5, tol 1e-5 | 4417.8961 | 1145.36415 | 5703.1171 | 2166.33578 | 1319.9463 | 6926.5314 |

Python params through Rust predict(): yhat max abs diff 2.3e-10, trend 1.2e-10 (predict-path parity).

---

# Summary (recommended config: exact, f/T, guard, m=20, tol 1e-7)

| dataset | rows | Rust fit secs | Python fit secs | status | Δf vs Python | yhat hist max abs | yhat future max abs | y range |
|---|---|---|---|---|---|---|---|---|
| peyton_manning | 2905 | 0.259 | 0.33 | Stalled | -0.1315 | 0.0082 | 0.0049 | 7.6 |
| air_passengers | 144 | 0.001 | 0.02 | Stalled | +0.4134 | 2.1145 | 35.3831 | 518.0 |
| retail_sales | 293 | 0.011 | 0.05 | Stalled | +0.0193 | 2139.1920 | 1702.2072 | 371877.0 |

Wrote results.json and report-<dataset>.html (Rust vs Python yhat overlay, residual panel, δ table).
