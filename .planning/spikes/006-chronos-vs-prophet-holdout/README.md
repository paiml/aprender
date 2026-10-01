---
spike: 006
idea: prophet-forecast-mcp
name: chronos-vs-prophet-holdout
type: standard
validates: "Given the spike-001/003 series with rolling-origin holdouts (Peyton daily; air, retail monthly; wp_log_R daily), when zero-shot Chronos-Bolt (tiny and small) and the fitted Prophet / NeuralProphet-lite ports forecast the same windows, then MAE/MASE, 80% coverage/width and latency are compared per series with naive and seasonal-naive baselines, so the value of a Chronos server versus the fitted models is measured, not assumed"
verdict: VALIDATED
related: [001, 002, 003, 004, 005]
tags: [chronos, prophet, benchmark, holdout, coverage, latency]
---

# Spike 006: zero-shot Chronos-Bolt vs fitted Prophet / NeuralProphet-lite

## What This Validates

Given four real series and 17 rolling-origin windows (Peyton 5, wp_log_R 4, air passengers 4,
retail sales 4; horizons 12/24 monthly, 90/365 daily), when every model forecasts the same windows
from the same training prefix, then a per-series and per-horizon comparison says where a zero-shot
Chronos server beats the fitted models, where it loses, and what each costs.

## Research

No new sources: the models are spikes 001–005's ports (`prophet.rs` + `fit.rs` from 004,
`np.rs` from 002/004, `bolt.rs` from 005). The Python `chronos-forecasting==2.3.1` oracle
(`tools/oracle6.py`) forecasts the same windows with tiny and small as a cross-check of the metric
plumbing and to fetch `amazon/chronos-bolt-small` (47.7M params, 182 MB f32, gitignored).

- **MASE** = MAE / in-sample seasonal-naive MAE (period 7 daily, 12 monthly; Hyndman), so series
  of different scale are comparable and 1.0 means "as good as repeating last week/year".
- **cov80 / width/σ**: Prophet's simulated 80 % band, Chronos q10–q90, NP-lite residual-sd band.
- **WQL3**: weighted quantile loss over q10/q50/q90, the three levels all models can produce.
- **Horizon slice:** MASE on steps 1–64 (Chronos's native direct horizon) vs steps 65+ (its
  batched 9-path rollout).

## How to Run

```bash
cd .planning/spikes/006-chronos-vs-prophet-holdout
uv run --python 3.12 --with chronos-forecasting --with pandas --with safetensors python tools/oracle6.py /tmp/c6   # downloads tiny+small, writes the oracle JSON
cp ~/.cache/huggingface/hub/models--amazon--chronos-bolt-tiny/snapshots/*/{model.safetensors,config.json} models/tiny/
cp ~/.cache/huggingface/hub/models--amazon--chronos-bolt-small/snapshots/*/{model.safetensors,config.json} models/small/
cp /tmp/c6/chronos_holdout_oracle.json fixtures/
CARGO_TARGET_DIR=../../../target cargo run --release      # ~50 s → RUN-OUTPUT.md-equivalent on stdout, results.json, report.html
```

## What to Expect

A Bolt-small parity line (1.9e-6 vs Python on Peyton), a 102-row per-split table, the summary
table below, and a cross-check line showing Rust and Python Chronos MAEs agree to float32
(1e-2 on retail values ≈ 1e4). `report.html` overlays the long window of each series.

## Investigation Trail

1. **Data first.** `wp_log_R.csv` is not in date order (1396 descending steps). Rust sorted it; the
   first oracle run did not, and the cross-check flagged exactly those eight forecasts (diff up to
   8.5e-2). Sorting the oracle the same way closed it. Spike 004's tool already refuses unsorted
   `ds`; this is the case it protects against.
2. **Bolt-small** runs through the spike-005 code unchanged (bigger config): quantiles within
   1.9e-6 of Python on the full Peyton context.
3. **Overall (mean MASE, 17 windows):** NeuralProphet-lite 1.029, Prophet 1.094, Chronos-small
   1.178, Chronos-tiny 1.238, seasonal naive 1.693, naive 2.017. Every model beats the baselines.
4. **By series, the story splits.** Monthly: Chronos-small 0.80 (air) / 0.74 (retail) vs Prophet
   1.09 / 1.21 and NP-lite 1.06 / 1.01 — zero-shot wins clearly. Daily Wikipedia: Chronos-small
   1.21 (Peyton) / 1.97 (wp_log_R) vs Prophet 1.05 / 1.03 and NP-lite 1.02 / 1.03 — zero-shot loses,
   badly on the 365-day windows.
5. **Horizon slice explains the loss.** Steps 1–64: Chronos-small 1.12, Prophet 1.07, NP-lite 1.01
   — a close race. Steps 65+: Chronos-small **1.73**, tiny 1.64, Prophet 1.06, NP-lite 1.08. The
   degradation is the rollout past the native horizon (the pipeline's own warning), not the domain.
6. **Intervals under-cover everywhere on rolling origins:** Prophet 0.60, NP-lite 0.66, tiny 0.65,
   small 0.69 at nominal 0.80. Chronos-small has the best WQL3 (0.0306) and Prophet the worst
   (0.0343); Prophet's band is the narrowest (0.83 σ). Spike 003's 0.82 coverage on the single
   Peyton year was the favourable case.
7. **Latency** (this Rust build, plain loops): Chronos-tiny 5.9 s for all 34 forecasts, small
   35.5 s (a 365-day rollout costs 10 s on small: 46 forwards); torch does all of it in 1.5 s.
   Prophet 4.6 s for 17 fits, NP-lite 1.9 s. The NEON GEMM fix from spike 005 is the difference
   between a viable and a slow Chronos-small server.

## Results

**Verdict: VALIDATED** — the comparison is measured and it is not one-sided.

| model | peyton | wp_log_r | air | retail | mean MASE | steps 1–64 | steps 65+ | cov80 | WQL3 | secs |
|---|---|---|---|---|---|---|---|---|---|---|
| naive (last value) | 1.772 | 2.246 | 2.603 | 1.448 | 2.017 | 1.963 | 2.094 | – | – | 0 |
| seasonal naive | 2.117 | 2.075 | 1.463 | 1.116 | 1.693 | 1.672 | 2.211 | – | – | 0 |
| Prophet (Rust) | 1.045 | 1.030 | 1.092 | 1.208 | 1.094 | 1.070 | 1.058 | 0.60 | 0.0343 | 4.6 |
| NeuralProphet-lite (Rust) | **1.021** | **1.026** | 1.055 | 1.013 | **1.029** | **1.011** | 1.083 | 0.66 | 0.0320 | 1.9 |
| Chronos-Bolt-tiny (zero-shot) | 1.242 | 1.802 | 0.980 | 0.928 | 1.238 | 1.193 | 1.640 | 0.65 | 0.0332 | 5.9 |
| Chronos-Bolt-small (zero-shot) | 1.209 | 1.966 | **0.801** | **0.735** | 1.178 | 1.120 | 1.733 | **0.69** | **0.0306** | 35.5 |

**Surprises**
- The fitted models' advantage on the daily series is almost entirely a long-horizon effect;
  inside 64 steps the three are within 0.1 MASE of each other.
- NeuralProphet-lite, 31 parameters trained in 0.1 s, is the best point forecaster on average.
- Nominal 80 % bands cover 60–69 % out of sample for every method; Prophet's is the narrowest and
  worst. A forecasting server should report empirical coverage, not nominal.

**Signal for the build**
- Ship Chronos as a **short-horizon** forecaster: accept `horizon ≤ 64` by default (one direct
  block), allow more with an explicit flag and a warning, and route long horizons to Prophet /
  NP-lite. That is exactly what the data supports.
- Prefer Bolt-**small** over tiny for accuracy (0.06 MASE, better calibration) once the NEON GEMM
  path lands; until then tiny is the deployable one (36 ms per forward vs ~150 ms).
- Keep all three behind one request/response shape; a `model: auto` policy can be as simple as
  "monthly or ≤ 64 steps → Chronos, else → NP-lite/Prophet".
- Add a rolling-origin evaluation like this one to the release gate for the forecasting servers;
  a single split flatters every model (spike 003's Prophet coverage was 0.82; here it is 0.60).
