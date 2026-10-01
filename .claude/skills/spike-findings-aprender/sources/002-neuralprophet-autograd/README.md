---
spike: 002
idea: prophet-forecast-mcp
name: neuralprophet-autograd
type: standard
validates: "Given the Peyton Manning series split into 2540 train / 365 test rows, when NeuralProphet's model (11-segment trend + Fourier seasonality + optional AR-Net on stationarised lags) is built from aprender's `nn::Linear` on the f32 autograd and trained with AdamW + one-cycle + weighted Huber, then it reaches holdout error comparable to spike 001's Prophet and to Python NeuralProphet using only existing ops, or names exactly which ops are missing"
verdict: VALIDATED
related: [001, 004]
tags: [neuralprophet, autograd, ar-net, adamw, huber]
---

# Spike 002: NeuralProphet-lite on aprender's autograd

## What This Validates

Given Peyton Manning (train = first 2540 rows, test = last 365), when NeuralProphet's default model is
re-implemented on `aprender::autograd` / `aprender::nn` and trained the way NeuralProphet trains it,
then (a) data preparation matches NeuralProphet 0.9.0 exactly, (b) the 365-day-ahead holdout error
is comparable to Python NeuralProphet and to the Prophet port from spike 001, (c) the AR-Net variants
beat the naive one-step baseline, and (d) every op needed already exists — with the exceptions named.

## Research

Grounded in the NeuralProphet `main` sources (`configure.py`, `forecaster.py`, `time_net.py`,
`time_dataset.py`, `df_utils.py`, `components/trend/piecewise_linear.py`, `utils_torch.py`) and
an installed `neuralprophet==0.9.0` oracle (`tools/np_oracle.py`, fixture
`fixtures/np_oracle_peyton.json`).

- **Normalisation:** `soft` — shift = min(y), scale = q95(y) − min(y); `t = (ds − ds_min)/(ds_max − ds_min)`.
- **Trend:** `n_changepoints = 10`, `changepoints_range = 0.8` → 11 segments at `0.8·i/11`; params
  `k0`, `deltas[11]` (segment-wise slopes `k0 + δ_i`), continuity via `−cp_j·(δ_j − δ_{j−1})`, plus a
  global `bias`. This is linear in the parameters, so it is a bias-free `Linear(12 → 1)` over a
  precomputed feature row.
- **Seasonality:** Fourier on **days since 1900-01-01** (not 1970), all `sin` then all `cos` per
  seasonality; auto resolutions yearly 6 / weekly 3 / daily 6 with Prophet's disable rules.
- **AR-Net:** `Linear(n_lags → …)` with hidden `ar_layers` + ReLU, final layer bias-free, kaiming init,
  fed **stationarised** lags: raw lag − trend(lag time, detached) − seasonality(lag time, in-graph).
- **Loss:** `SmoothL1Loss(β = 0.3)` per element × newer-sample weight `(1 + ½(cos(π(t−1))+1))/2`, mean.
- **Optimiser:** AdamW (`weight_decay = 1e-3`), OneCycleLR three-phase cosine
  (`div_factor 10`, `final_div_factor 10`, `pct_start 0.3`) stepped per batch by fractional epoch.
  Auto batch `2^(1+⌊1.5·log10 n⌋)` ∈ [8, 2048], auto epochs `10·⌈100/n · 2^(2.25·log10(10+n))⌉` ∈ [20, 500].
  Learning rate from a range test (Lightning's LR finder, 1e-6 … 10).
- **Init:** `init_parameter` = xavier-normal (k0 std 1, bias std 1, δ std √(1/11), season std √(1/(2R))).

| Approach | Tool | Pros | Cons | Status |
|----------|------|------|------|--------|
| Core `SmoothL1Loss` | `aprender::nn::loss` | Already there | **Detached from the graph** — builds output with `Tensor::new` from raw data; `backward()` leaves the parameter with no gradient | Rejected (defect) |
| Huber from ops + constant 0/1 mask | `sub/abs/pow/mul/mean` | Exact value and exact gradient (mask is piecewise constant) | Six ops instead of one | **Chosen** |
| Pseudo-Huber `β²(√(1+(d/β)²)−1)` | `pow/sqrt` | One smooth expression | Not NeuralProphet's loss | Not needed |
| Lag-time seasonality through `view` | `matmul` + `view` | Keeps the in-graph stationarisation without a `cat` op | Reshape trick | **Chosen** |

**Oracle gotchas:** NeuralProphet 0.9.0 needs `pandas<3` (`Series.view` removed) and `torch<2.6`
(Lightning's LR finder restores a checkpoint that the new `weights_only` default refuses).

## How to Run

```bash
cd .planning/spikes/002-neuralprophet-autograd
CARGO_TARGET_DIR=../../../target cargo run --release      # ~10 s; stderr carries the sweep detail
open report.html
# regenerate the oracle (about 80 s):
uv run --python 3.12 --with neuralprophet --with "pandas<3" --with "numpy<2.3" --with "torch<2.6" \
  python tools/np_oracle.py fixtures/peyton_manning.csv fixtures/np_oracle_peyton.json
```

## What to Expect

Section 0 shows `MSELoss` producing a gradient and `SmoothL1Loss` producing none. Section 1 prints
normalisation, time scaling, changepoints, seasonalities and auto batch/epochs identical to the
oracle's. Sections 2–4 are the accuracy tables; section 5 the probes. `RUN-OUTPUT.md` is the verdict
run; `train.log` holds the per-model learning-rate sweeps and the budget probe.

## Investigation Trail

1. **Read the API before assuming.** `SmoothL1Loss::forward` maps over `diff.data()` and wraps the
   result in `Tensor::new(...)` — a fresh leaf. `MSELoss` and `L1Loss` compose graph ops. A two-line
   probe confirmed it: after `backward()`, the parameter's gradient is `Some` for MSE and `None` for
   Huber. Built Huber from ops with a constant mask instead.
2. **No `cat`, no `where`, 2-D `matmul` only.** The AR path needs seasonality at 30 lag times per
   sample; computing it as `[b·30, F] @ [F, 1]` then `view([b, 30])` keeps it in-graph without
   concatenation. Trend at lag times is detached in NeuralProphet too, so it is a constant tensor.
3. **Data prep parity first.** Shift/scale, time scaling and the 11 changepoints match the oracle to
   1e-9 / 1e-16; auto batch 64 and epochs 80 match its formula exactly.
4. **Lag-free model, learning-rate sweep** (NeuralProphet picks lr with a range test; the spike sweeps
   1e-3 … 1 and selects by *train* loss only): lr 0.1 → test MAE **0.4510** in **0.04 s**. Python
   NeuralProphet: **0.4611** in **20.9 s**. Spike 001's Prophet on the same split: **0.4125** in 0.20 s.
   Last-year mean: 0.689. Seeds 1/2/3: 0.4549 / 0.4515 / 0.4508. Half or double the epochs: ±0.003.
   Removing weight decay and sample weighting: 0.4511 (they do not matter here).
5. **Full-batch training collapses** (MAE 2.6): 80 epochs × 1 batch = 80 steps. The auto-epoch
   formula assumes mini-batches; a wrapper must not "optimise" batch size upward.
6. **AR-Net, one-step-ahead with true lags** (per-model lr sweep, selected by train loss):
   hidden-32 ReLU **0.2509** vs Python NeuralProphet 0.3933 vs naive 0.3461. Linear 30→1: **0.3216**
   vs NeuralProphet **0.2557** — worse, with train loss 0.0114 vs 0.0085 on a convex model.
7. **Control before naming a cause:** four times the epochs (320) takes the linear model to train
   loss **0.0078** and test MAE **0.2432** in 3.8 s; 800 epochs adds nothing (0.0078 / 0.2425). The
   gap was optimisation budget: NeuralProphet's range-tested lr converges faster per step than a
   coarse grid, and its 80-epoch default is tuned to that. Same lesson as spike 001: parity between
   optimisers is a band, not a point.
8. **Tape hygiene:** 20 graph entries per step, `clear_graph()` after every optimiser step; 3200 steps
   in 0.04 s, so a fit-at-request MCP tool is not threatened by training cost.
9. **60-row series** trains (batch 8, epochs 300, 0.02 s) but forecasts badly (MAE 3.0 on the next 30
   rows) — NeuralProphet's own defaults are not meant for 60 points; a wrapper should say so, not fit.

## Results

**Verdict: VALIDATED.** NeuralProphet's default model trains on aprender's existing autograd to
NeuralProphet-level holdout error, hundreds of times faster than the torch/Lightning original.

| model | Rust (this spike) | Python NeuralProphet 0.9.0 | Rust Prophet (001) | baseline |
|---|---|---|---|---|
| trend + seasonality, 365-day-ahead MAE | **0.4510** (0.04 s) | 0.4611 (20.9 s) | 0.4125 (0.20 s) | last-year mean 0.689 |
| AR-Net linear, 1-step MAE | 0.3216 @ 80 ep · **0.2432 @ 320 ep** (3.8 s) | 0.2557 (23.5 s) | – | naive 0.346 |
| AR-Net 30→32→1, 1-step MAE | **0.2509** (1.2 s) | 0.3933 (24.1 s) | – | naive 0.346 |

**Ops used:** `matmul` (2-D), `transpose`, `broadcast_add`, `add`, `sub`, `mul`, `mul_scalar`,
`abs`, `pow`, `mean`, `relu`, `view`, `backward`, `no_grad`, `clear_graph`, `Linear`, `AdamW`.
**Missing:** a graph-connected Huber loss (core defect), `where`/`clamp`, `cat`. All three had cheap
workarounds; none blocks the build.

**Surprises**
- Core's `SmoothL1Loss` cannot train anything — it silently detaches. Worth a core fix + a
  connectivity test for every loss in `nn::loss`.
- NeuralProphet is *slower and no more accurate* than Prophet on this series; its value is the AR-Net
  (0.25 vs naive 0.35 one step ahead), which is where a "NeuralProphet MCP tool" earns its keep.
- Training this model is ~0.04 s; the whole "training is expensive" premise of a fit→artifact
  workflow does not hold for these two forecasters.

**Signal for the build**
- Port `np.rs` as-is: feature builders, `NpModel`, `weighted_huber`, `one_cycle_lr`, auto batch/epochs.
- Replace the lr grid with NeuralProphet's range test (or a short geometric sweep selected by train
  loss) and budget ~4× NeuralProphet's auto epochs for the linear AR case; it is still seconds.
- Fix core `SmoothL1Loss` to compose graph ops; add a `where`/`clamp` op and `cat` to the autograd.
- Keep `clear_graph()` per step; the tape is global per thread — an MCP server must train on a
  blocking thread and never interleave two fits on one thread.
- The stateless `forecast` tool can offer `model: prophet | neuralprophet` with `n_lags` for
  one-step-ahead nowcasting; multi-step AR needs `n_forecasts > 1` (not spiked).
