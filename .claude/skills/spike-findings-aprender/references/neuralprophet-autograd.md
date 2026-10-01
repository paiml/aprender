# NeuralProphet-lite on aprender's f32 Autograd

NeuralProphet 0.9.0's default model (11-segment trend + Fourier seasonality + optional AR-Net)
built from `aprender::nn::Linear` on the autograd and trained with AdamW + one-cycle + weighted
Huber. Reaches NeuralProphet's holdout error ~500× faster (spike 002); the best mean-MASE point
forecaster in the spike-006 benchmark.

## Requirements

From the `prophet-forecast-mcp` idea (MANIFEST.md):

- Both Prophet (MAP / L-BFGS) and **NeuralProphet (autograd / AdamW)** are in scope; NeuralProphet
  is spiked, not deferred.
- The serving shape is a STATELESS `forecast` tool: the fit runs inside the call (lag-free: 0.04–0.16 s;
  with 30 lags and a hidden layer: 1.2–2.75 s at 1k–3k points).

## How to Build It

**1. Port `np.rs` as-is** — `sources/004-forecast-mcp-thin-server/src/np.rs` (19 KB) is the
evolved copy. It contains the feature builders, `NpModel`, `weighted_huber`, `one_cycle_lr`,
`auto_batch`/`auto_epochs`, `train`, `predict_ts`, `predict_ar_1step`, `predict_ar_recursive`.

Data preparation that MUST match NeuralProphet exactly (verified 1e-9 / 1e-16):

- `soft` normalisation: shift = min(y), scale = q95(y) − min(y); `t = (ds − ds_min)/(ds_max − ds_min)`.
- Trend: 10 changepoints, range 0.8 → 11 segments at `0.8·i/11`; params `k0`, `deltas[11]`, `bias`.
  Linear in the parameters, so it is a bias-free `Linear(12 → 1)` over a precomputed feature row.
- Fourier on **days since 1900-01-01** (not 1970!), all `sin` then all `cos` per seasonality;
  yearly 6 / weekly 3 / daily 6 with Prophet's disable rules.
- AR-Net: `Linear(n_lags → hidden…)` + ReLU, final layer bias-free, kaiming init, fed **stationarised**
  lags: raw lag − trend(lag time, detached) − seasonality(lag time, in-graph).
- Init: xavier-normal (k0 std 1, bias std 1, δ std √(1/11), season std √(1/(2R))).

**2. Loss — build Huber from ops, never from core `SmoothL1Loss`:**

```rust
pub fn weighted_huber(pred: &Tensor, target: &Tensor, w: &Tensor, beta: f32) -> Tensor {
    let d = pred.sub(target);
    let a = d.abs();
    let mask: Vec<f32> = a.data().iter().map(|v| if *v < beta { 1.0 } else { 0.0 }).collect(); // piecewise constant → exact gradient
    let inv: Vec<f32> = mask.iter().map(|m| 1.0 - m).collect();
    let m = Tensor::from_vec(mask, a.shape());
    let im = Tensor::from_vec(inv, a.shape());
    let half_beta = Tensor::from_vec(vec![0.5 * beta; a.numel()], a.shape());
    let quad = d.pow(2.0).mul_scalar(0.5 / beta).mul(&m);
    let lin = a.sub(&half_beta).mul(&im);
    quad.add(&lin).mul(w).mean()
}
```

`beta = 0.3`; sample weight `(1 + ½(cos(π(t−1))+1))/2` favours newer rows.

**3. Schedule and budget:**

```rust
pub fn one_cycle_lr(p: f64, max_lr: f64) -> f64 {       // three-phase cosine, div 10 / final div 10 / pct_start 0.3
    let (init, fin) = (max_lr / 10.0, max_lr / 100.0);
    let cosine = |s: f64, e: f64, f: f64| e + (s - e) / 2.0 * (1.0 + (std::f64::consts::PI * f).cos());
    if p < 0.3 { cosine(init, max_lr, p / 0.3) } else if p < 0.6 { cosine(max_lr, init, (p - 0.3) / 0.3) } else { cosine(init, fin, (p - 0.6) / 0.4) }
}
pub fn auto_batch(n: usize) -> usize { (2usize.pow(1 + (1.5 * (n as f64).log10()) as u32)).clamp(8, 2048).min(n) }
pub fn auto_epochs(n: usize) -> usize { (10.0 * (100.0 / n as f64 * 2f64.powf(2.25 * (10.0 + n as f64).log10())).ceil()).clamp(20.0, 500.0) as usize }
```

AdamW `weight_decay = 1e-3`, stepped per batch by fractional epoch. Learning rate: a short
geometric sweep **selected by train loss only** — {0.01, 0.03, 0.1} lag-free, {0.03, 0.1} with lags
(0.1 won every time). Budget ~4× NeuralProphet's auto epochs for the linear AR case (80 → 320
epochs took 1-step MAE 0.3216 → 0.2432, matching NeuralProphet's 0.2557; 800 adds nothing).

**4. Tape hygiene.** `clear_graph()` after every optimiser step (20 graph entries per step). The tape
is global per thread: train on a `spawn_blocking` thread and never interleave two fits on one
thread. Spike 010 proved 16 concurrent fits on reused blocking threads stay bit-identical.

**5. In-graph stationarisation without `cat`:** seasonality at the 30 lag times per sample is
`[b·30, F] @ [F, 1]` then `view([b, 30])`; trend at lag times is a detached constant (NeuralProphet
detaches it too).

Ops used and proven: `matmul` (2-D), `transpose`, `broadcast_add`, `add`, `sub`, `mul`, `mul_scalar`,
`abs`, `pow`, `mean`, `relu`, `view`, `backward`, `no_grad`, `clear_graph`, `Linear`, `AdamW`.

## What to Avoid

- **`aprender::nn::loss::SmoothL1Loss` cannot train anything.** Its `forward` maps over `diff.data()`
  and wraps the result in `Tensor::new(...)` — a fresh leaf. After `backward()` the parameter's
  gradient is `None`. Core defect; fix it to compose graph ops and add a connectivity test for every
  loss in `nn::loss`.
- **Do not "optimise" batch size upward.** Full-batch training collapses (MAE 2.6): 80 epochs × 1
  batch = 80 steps. The auto-epoch formula assumes mini-batches.
- **Never select the learning rate or epochs by test error.** Select by train loss; report test.
- **Do not fit 60-row series with NeuralProphet defaults** — trains fine (0.02 s) but MAE 3.0 on the
  next 30 rows. Say so rather than fit.
- Missing autograd ops: `where`/`clamp`, `cat`. Cheap workarounds exist (constant mask, `view`);
  adding them to core is a follow-up, not a blocker.

## Constraints

- Lag-free model, Peyton 365-day-ahead: test MAE 0.4510 in 0.04 s (Python NP 0.4611 in 20.9 s;
  Rust Prophet 0.4125). AR-Net 30→32→1 one-step: 0.2509 in 1.2 s (NP 0.3933, naive 0.346).
- NeuralProphet is *slower and no more accurate* than Prophet on trend+seasonality; its value is the
  AR-Net for short-horizon nowcasting. `n_forecasts > 1` (multi-step AR) is not spiked.
- Bands in the port are residual-sd based (nominal 80 % covered 0.66 on rolling origins); NP's
  quantile regression is a follow-up.
- Oracle: `neuralprophet==0.9.0` needs `pandas<3`, `numpy<2.3`, `torch<2.6`
  (`uv run --python 3.12 --with neuralprophet --with "pandas<3" --with "numpy<2.3" --with "torch<2.6"`).

## Origin

Synthesized from spike: 002 (evolved `np.rs` from 004; benchmark evidence from 006; concurrency from 010)
Source files available in: `sources/002-neuralprophet-autograd/`, `sources/004-forecast-mcp-thin-server/src/np.rs`
Oracle fixture (not copied): `.planning/spikes/002-neuralprophet-autograd/fixtures/np_oracle_peyton.json`
