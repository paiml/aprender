# NeuralProphet Exogenous Inputs — Events and Regressors

Adding caller-supplied events (holiday-shaped indicators) and numeric regressors to the
NeuralProphet-lite arm of `aprender-forecast`, on the f32 autograd.

**Two findings decide the delivery, and both are blockers rather than polish:**

1. **The door's train-cost bound has no term for event columns.** At
   `MAX_HOLIDAY_COLUMNS` (1 000) a request buys **7.6× the work it was priced at** (spike 013).
2. **The change request's `RegressorArg` contract does not cover NeuralProphet with lags.** NP
   trains on an imputed daily grid *denser* than `ds`; with `n_lags > 0` it reads every imputed
   day, and the fill rule is worth ~30 % of the series scale (spike 014).

## Requirements

From the `forecast-exogenous-inputs` idea (`.planning/spikes/MANIFEST.md`):

- **One `RegressorArg` shape serves both models**, so the caller sends one argument to `prophet`
  and `neuralprophet` alike. (This is the requirement finding 2 puts under pressure — see
  *Constraints*.)
- **No new required arguments.** Every new field is `Option<_>` with a serde default;
  `#[serde(deny_unknown_fields)]` stays as it is.
- **Errors, not silence.** An unsupported combination (a driver on a model that cannot use it) is
  refused at the door with a message naming the limitation — the existing D-11 pattern.
- **Byte-identical when the new arguments are absent** — see
  `references/no-argument-invariance-gate.md`.
- **One tag per release**; the consumer bumps one line plus the lock entry and builds `--locked`.
- The public entry point is **`forecast(&ForecastArgs)`**; `fitted_forecast` does not exist.

## How to Build It

### 1. Events: one additive `Linear(E, 1)` block, composed beside `NpModel`

`NpModel::forward` returns a `Tensor`, so the event block is **one `.add()` away**. Do not fork
`NpModel`. Hand both parameter sets to one `AdamW`:

```rust
let mut model = NpModel::new(d, n_lags, &[], &mut rng);
let mut block = EventBlock::new(event_cols.len(), &mut rng);

let mut opt = {
    let mut p = model.parameters_mut();
    p.extend(block.parameters_mut());
    AdamW::new(p, max_lr).weight_decay(0.0)
};
// …
let pred = model.forward(&xt, &xs, ar).add(&block.forward(&xe));
let loss = weighted_huber(&pred, &yt, &wt, HUBER_BETA);
loss.backward();
{ let mut p = model.parameters_mut(); p.extend(block.parameters_mut()); opt.step_with_params(&mut p); }
opt.zero_grad();
clear_graph();                     // D-10: after EVERY step
```

Block init mirrors the seasonality block's scaling for a block of this width:

```rust
let mut lin = Linear::without_bias(dim, 1);
let std = (1.0 / dim as f64).sqrt();
lin.set_weight(Tensor::from_vec((0..dim).map(|_| (rng.normal() * std) as f32).collect(), &[1, dim])
               .requires_grad());
```

### 2. Event indicator columns — the Prophet expansion, reused

One column per `(event, window offset)`, events in insertion order, offsets ascending within an
event — **the same expansion the Prophet arm already applies to `HolidayArg`**, which is how one
argument serves both models:

```rust
pub fn event_columns(events: &[EventSpec]) -> Vec<(usize, i64)> {
    let mut cols = Vec::new();
    for (ei, e) in events.iter().enumerate() {
        for off in e.lower_window..=e.upper_window { cols.push((ei, off)); }
    }
    cols
}

pub fn event_row(day: i64, cols: &[(usize, i64)], sets: &[HashSet<i64>], out: &mut Vec<f32>) {
    for &(ei, off) in cols {
        // `d + off == day` iff `d == day - off` — the same rearrangement `feature_row` uses.
        out.push(f32::from(u8::from(sets[ei].contains(&(day - off)))));
    }
}
```

Hoist one membership set per event out of the row loop, the `holiday_day_sets` pattern.

### 3. Fix the train-cost bound BEFORE events ship

`np.rs:464` today:

```rust
pub fn train_cost(n_samples: usize, epochs: usize, n_lags: usize) -> u64 {
    (epochs as u64).saturating_mul(n_samples as u64).saturating_mul(n_lags as u64 + 1)
}
```

There is **no event term**, so every row of the sweep below is priced identically at 132 000 while
the real work rises linearly in E. `request_train_cost` (`np.rs:508`) must take `n_event_cols`,
and the invariant `a_neuralprophet_request_over_the_train_cost_bound_is_refused`
(`forecast.rs:888`) must be re-derived against the new bound. Cost axis **C-08** in
`contracts/forecast-tool-boundary-v1.yaml` is the one this changes.

Measured shape: `µs/step ≈ 12.9 + 0.085·E` on an Apple-Silicon dev box, 1 200 points, lag-free.
**The shape (linear in E) transfers; the constant does not** — calibrate `0.085` on the deployment
target, and **round the bound up: a door must never under-price.**

### 4. Regressors on NeuralProphet: the rule depends on `n_lags`

`NpData::new` builds a daily grid over `[first, last]` and **linearly imputes `y`** on missing days
(`np.rs:183-196`). A regressor is supplied on the caller's rows, so imputed days have no value of
their own. The training-sample rule splits (`np.rs:597`):

```rust
let samples: Vec<usize> = if n_lags == 0 {
    (0..n_grid).filter(|&i| d.grid_observed[i]).collect()   // observed rows ONLY
} else {
    (n_lags..n_grid).collect()                              // EVERY grid row, imputed included
};
```

- **Lag-free — the imputed-day value is never read.** All four fill rules, a deliberate garbage
  probe (`1e3`) included, give bit-identical predictions and identical learned weights. There is
  no rule to choose and none to document.
- **Lagged — the rule is load-bearing.** All 109 imputed days (12.1 % of the grid) carry a
  regressor value into the loss.

Standardise on the **observed train rows** with ddof = 1 and leave binary `{0,1}` at `mu=0, std=1`
— the same Prophet rule measured in spike 011 (`references/prophet-external-regressors.md`).

## What to Avoid

- **Shipping events against the current `train_cost`.** This is the change request's own stated
  concern — *"the door's budget is a promise we rely on under Lambda's timeout"* — and it is real,
  quantified, and larger than the 3-day estimate implies.
- **Sweeping a cost claim only to a typical value.** Events cost 1.3× at 6 columns and 7.6× at
  `MAX_HOLIDAY_COLUMNS`. Only the second number decides whether the budget promise survives.
  Sweep to the **door's own ceiling**.
- **`RegressorArg { values: len == ds.len() + horizon }` on the NP arm with lags.** Correct for
  Prophet; on a gappy daily series it leaves 109 values in this fixture undefined, and that choice
  is worth ~30 % of the series scale.
- **Linear interpolation of a binary driver.** It produces a "0.5 promo" day. If one rule is
  chosen for all columns, **carry-forward is the defensible one**.
- **Reading "bit-identical under every rule" as proof without a falsification probe.** The
  `garbage (1e3)` rule exists so a green column cannot be mistaken for a broken harness — and it
  is shown moving the lagged forecast by 192 units.
- **Assuming weekly / month-start series need a grid mapping.** They do not: `forecast.rs:421-423`
  refuses `freq != "D"` on the neuralprophet arm, pinned by `neuralprophet_refuses_non_daily_freq`.
  That refusal answers the change request's P5 Q3 and voided spike 014's original premise.
- **Trusting `core`'s `SmoothL1Loss`.** It is detached from the graph — build Huber from ops
  (`weighted_huber`, already public). Carried over from spike 002.

## Constraints

### Events (spike 013) — synthetic 1 200-point daily series, scale 30.43, `n_lags = 0`

| run | params | final train loss | MAE vs truth |
|---|---|---|---|
| events ON | 37 | 0.00035 | 0.403 |
| events OFF | 31 | 0.00237 | 0.697 |

All six learned weights recover the planted `+8.0`: `+7.769, +8.293, +7.990, +8.043, +8.249,
+8.462` — worst error **0.462 (5.8 %)**. The effect is learned by the block, not absorbed by trend
or seasonality.

Determinism: seed 42 twice bit-identical (predictions and event weights); **seed 43 differs**, so
the check can fail. Tape length per step **25 ON / 20 OFF** — a fixed +5 that does **not** grow
with E, so there is no leak and `clear_graph()` still empties it.

Train cost, priced at 132 000 on every row:

| event columns E | µs/step | measured / priced |
|---|---|---|
| 0 | 12.6 | 1.00× |
| 7 | 15.6 | 1.24× |
| 28 | 18.9 | 1.50× |
| 84 | 21.6 | 1.71× |
| 210 | 31.9 | 2.53× |
| 504 | 58.0 | 4.59× |
| **1001** | **100.1** | **7.93×** |

**Scope not covered:** lag-free only. The AR paths (`predict_ar_recursive`, `predict_ar_1step`) and
the interaction between an event block and stationarised lags are **unmeasured**; the change
request lists them in the same 3-day item. Multiplicative event mode was not built (CR marks it
optional).

### Regressors on the imputed grid (spike 014) — 790 observed rows / 899-day grid, 109 imputed (12.1 %), scale 35.32

| n_lags | fill rule | bit-identical to linear | max abs Δ pred | promo w | price w |
|---|---|---|---|---|---|
| 0 | linear interp | yes | 0.0 | +0.2546 | −0.0493 |
| 0 | zero fill | **yes** | 0.0 | +0.2546 | −0.0493 |
| 0 | carry forward | **yes** | 0.0 | +0.2546 | −0.0493 |
| 0 | garbage (1e3) | **yes** | 0.0 | +0.2546 | −0.0493 |
| 7 | linear interp | yes | 0.0 | +0.2640 | −0.0048 |
| 7 | zero fill | no | **7.98** | +0.2707 | −0.0018 |
| 7 | carry forward | no | **10.48** | +0.2453 | −0.0086 |
| 7 | garbage (1e3) | no | **192.2** | **−0.1920** | **+0.5711** |

Two *defensible* rules differ by 10.48 on scale 35.32, ~30 %. The garbage probe moves it by 192 and
**flips the sign of both coefficients**: a wrong value on 12 % of grid days does not degrade the
model, it inverts it.

**Three ways out, ranked:**

1. **Refuse.** On the neuralprophet arm with `n_lags > 0`, refuse a regressor when the series has
   missing days. Consistent with D-11 (refuse rather than default), needs no new semantics, costs
   the caller a gap-filled series.
2. **Require grid-complete values.** Contract becomes `len == (last - first + 1) + horizon` on the
   NP arm — honest, but the argument shape then differs between the two models, which is exactly
   what the change request asked to avoid.
3. **Impute and disclose.** Carry forward for binary, linear for continuous, stated in the response
   diagnostics. Cheapest for the caller; also the only option where the operator's number was
   invented by us, so it **must** be visible.

**Second identifiability caution, same class as spike 011's:** at `n_lags = 7` the `price` weight
collapses to −0.0048 from −0.0493 lag-free — the AR term absorbs a smooth continuous driver.

**Scope not covered:** one series, one gap pattern, `n_lags ∈ {0, 7}`, additive mode. The
sensitivity number (10.48, 30 % of scale) is illustrative of the magnitude, **not a bound** — it
moves with gap fraction, gap run length and driver volatility. The finding that lag-free never
reads the value is structural and holds for any series.

## Origin

Synthesized from spikes: 013, 014.
Source files available in: `sources/013-np-events-autograd/` (README.md, `src/events.rs` — the
`Linear(E,1)` block, `src/main.rs` — the A/B/C/D driver with the cost sweep, RUN-OUTPUT.md),
`sources/014-np-gap-imputation-regressors/` (README.md, `src/main.rs` — `reg_grid` fill rules plus
the lag-split training loop, RUN-OUTPUT.md).
