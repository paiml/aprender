# Forecaster Evaluation and Model Routing

The rolling-origin benchmark that measured zero-shot Chronos-Bolt against the fitted Prophet and
NeuralProphet-lite ports on four real series and 17 windows (spike 006). It is the evidence behind
the horizon policy, the `model: auto` routing rule, and the "report empirical coverage" rule.

## Requirements

From the `prophet-forecast-mcp` idea (MANIFEST.md):

- Chronos joins as a THIRD forecaster (own thin server); all three sit behind one request/response
  shape, so a routing policy across them is a product decision this evidence informs.
- Correctness bars are parity with the Python originals, not self-consistency — the benchmark
  cross-checks Rust and Python Chronos MAEs to float32 on every window.

## How to Build It

**1. The harness** (`sources/006-chronos-vs-prophet-holdout/src/main.rs`): per series, a list of
`(n_train, horizon)` splits; every model forecasts the same windows from the same prefix.

```rust
struct Series { name, ds: Vec<i64>, y: Vec<f64>, period: usize, splits: Vec<(usize, usize)> }
// peyton   period 7  splits (2540,365) (2200,90) (2400,90) (2600,90) (2815,90)
// wp_log_r period 7  splits (2498,365) (2100,90) (2400,90) (2773,90)
// air      period 12 splits (120,24) (96,12) (108,12) (132,12)
// retail   period 12 splits (269,24) (221,12) (245,12) (281,12)

// MASE denominator: in-sample seasonal-naive MAE on the training prefix
let denom = (s.period..n_train).map(|i| (tr_y[i] - tr_y[i - s.period]).abs()).sum::<f64>() / (n_train - s.period) as f64;
// seasonal-naive forecast row
let seasonal_naive: Vec<f64> = (0..h).map(|t| tr_y[n_train - s.period + t % s.period]).collect();
```

Metrics per window: MAE, **MASE** (period 7 daily / 12 monthly; 1.0 = "as good as repeating last
week/year"), 80 % coverage and width/σ (Prophet simulated band, Chronos q10–q90, NP residual-sd),
**WQL3** (weighted quantile loss over q10/q50/q90 — the levels all models produce), latency.
Always include `naive` and `seasonal naive` rows and a **horizon slice** (steps 1–64 vs 65+).

**2. Make it a release gate** for the forecasting servers. A single split flatters every model:
spike 003's Prophet coverage was 0.82 on one Peyton year; on rolling origins it is 0.60.

**3. The routing rule the data supports** (`model: auto`):

- Monthly series, or horizon ≤ 64 steps → **Chronos** (Bolt-small if available; tiny on Lambda zip).
- Otherwise (daily, long horizon) → **NeuralProphet-lite** (best mean MASE) or Prophet (bands + components).
- Chronos beyond 64 steps only with `allow_long_horizon` and a warning (see `chronos-mcp-server.md`).

**4. Data hygiene the harness enforces:** sort and de-duplicate by `ds` at load in **both** the Rust
driver and the Python oracle — `wp_log_R.csv` is not chronological, and the cross-check flagged
exactly the eight unsorted forecasts (diff 8.5e-2) until the oracle sorted too.

## What to Avoid

- **Do not claim accuracy from one split or one series.** Use rolling origins, MASE, baselines, and
  a horizon slice.
- **Do not present nominal 80 % bands as 80 %.** Every method covered 0.60–0.69 out of sample.
  Prophet's band is the narrowest (0.83 σ) and worst (0.60); Chronos-small the best (0.69, WQL3 0.0306).
- **Do not route long daily horizons to Chronos.** Its loss on Peyton / wp_log_R is almost entirely
  a past-64-steps effect (MASE 1.73 vs 1.06 for Prophet); inside 64 the three are within 0.1.
- **Do not assume NeuralProphet needs to be "neural" to win.** NP-lite here is 31 parameters
  trained in 0.1 s and is the best point forecaster on average (mean MASE 1.029).
- Do not compare Rust and Python Chronos MAEs at tighter than float32 on large-scale series
  (1e-2 on retail values ≈ 1e4 is rounding).

## Constraints

| model | peyton | wp_log_r | air | retail | mean MASE | steps 1–64 | steps 65+ | cov80 | WQL3 | secs (17–34 fc) |
|---|---|---|---|---|---|---|---|---|---|---|
| naive | 1.772 | 2.246 | 2.603 | 1.448 | 2.017 | 1.963 | 2.094 | – | – | 0 |
| seasonal naive | 2.117 | 2.075 | 1.463 | 1.116 | 1.693 | 1.672 | 2.211 | – | – | 0 |
| Prophet (Rust) | 1.045 | 1.030 | 1.092 | 1.208 | 1.094 | 1.070 | 1.058 | 0.60 | 0.0343 | 4.6 |
| NeuralProphet-lite (Rust) | **1.021** | **1.026** | 1.055 | 1.013 | **1.029** | **1.011** | 1.083 | 0.66 | 0.0320 | 1.9 |
| Chronos-Bolt-tiny | 1.242 | 1.802 | 0.980 | 0.928 | 1.238 | 1.193 | 1.640 | 0.65 | 0.0332 | 5.9* |
| Chronos-Bolt-small | 1.209 | 1.966 | **0.801** | **0.735** | 1.178 | 1.120 | 1.733 | **0.69** | **0.0306** | 35.5* |

\* plain-loop build before the spike-008 kernel; with it Bolt-tiny forwards are 6.5× faster
(137 → 21 ms through `gemm_blis`), Bolt-small ~2.3× (229 → 98 ms).

- Series: Peyton Manning (Wikipedia daily, 2905), `wp_log_R` (daily, 2863), air passengers
  (monthly, 144), retail sales (monthly, 293). CSVs in `.planning/spikes/006-chronos-vs-prophet-holdout/fixtures/`.
- Oracle: `tools/oracle6.py` under `uv run --python 3.12 --with chronos-forecasting --with pandas --with safetensors`
  (downloads tiny + small; writes `chronos_holdout_oracle.json`, 0.5 MB, committed).

## Origin

Synthesized from spike: 006 (models from 001–005)
Source files available in: `sources/006-chronos-vs-prophet-holdout/` (main.rs, fit.rs, tools/oracle6.py, RUN-OUTPUT.md, results.json)
Fixtures (not copied): `.planning/spikes/006-chronos-vs-prophet-holdout/fixtures/`
