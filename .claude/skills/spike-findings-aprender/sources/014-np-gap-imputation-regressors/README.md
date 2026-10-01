---
spike: 014
idea: forecast-exogenous-inputs
name: np-gap-imputation-regressors
type: standard
validates: "Given a daily series with gaps, when a future regressor is supplied on the caller's rows and NeuralProphet trains on its imputed daily grid, then whether the imputed-day value is read at all is determined per n_lags, and the forecast's sensitivity to the fill rule is measured"
verdict: VALIDATED
related: [012, 013, 002]
tags: [neuralprophet, regressors, imputation, gaps, api-contract]
---

# Spike 014: Regressor values on NeuralProphet's imputed grid days

## What This Validates

**Given** a daily series with block gaps and two drivers supplied on the observed rows,
**when** NeuralProphet trains on the daily grid `NpData::new` imputes, **then** whether the
imputed-day regressor value is read is determined for each `n_lags`, and the forecast's
sensitivity to the fill rule is measured.

## Research — why this spike was re-scoped

As planned, 014 asked how weekly (`W`) and month-start (`MS`) series map onto NeuralProphet's
daily grid — the change request's P5 Q3. **Spike 012 refuted the premise**: `forecast.rs:421-423`
refuses `freq != "D"` on the neuralprophet arm, pinned by its own test
`neuralprophet_refuses_non_daily_freq`. There is no W/MS-on-daily-grid mapping to describe,
because those requests never reach the model.

What survives is the daily case. `NpData::new` builds a grid over `[first, last]` and **linearly
imputes `y`** on missing days (`np.rs:183-196`). A regressor is supplied on the caller's rows, so
the imputed days have no value of their own — and the training-sample rule differs by `n_lags`
(`np.rs:597`): lag-free trains on **observed rows only**; lagged trains on `(l..n_train_grid)`,
every grid row, imputed ones included.

## How to Run

```bash
CARGO_TARGET_DIR=../../../target cargo run --release --quiet > RUN-OUTPUT.md
```

## What to Expect

Four fill rules per `n_lags`, compared bit-for-bit against linear interpolation. `garbage (1e3)`
is the falsification probe: a rule that changes nothing proves the value is unread, but only if
the probe is shown to change something somewhere.

## Investigation Trail

1. **The premise had to be re-checked before building**, because 012 had already refuted the
   original framing. Reading `np::train`'s sample rule showed the surviving question is not
   "which cadence?" but "which rows does training actually touch?".
2. **A falsification probe was added rather than assumed.** `garbage (1e3)` exists so that
   "bit-identical under every rule" cannot be mistaken for a broken harness — the lagged rows
   show the same probe moving the forecast by 192 units.
3. **Block gaps, not just scattered dropout.** Three 15-day blackouts plus 8 % random dropout, so
   12.1 % of grid days are imputed and some runs of imputed days are long enough that carry-forward
   and linear interpolation genuinely diverge.

## Results

**VALIDATED.** 790 observed rows over an 899-day grid → **109 imputed days (12.1 %)**, series
scale 35.32.

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

### The answer, and it is conditional on `n_lags`

**Lag-free: the imputed-day value is never read.** All four rules — garbage included — give
bit-identical predictions and identical learned weights. Training samples are the observed rows
only, so an imputed day never enters a batch. There is no rule to choose and none to document.

**Lagged: the rule is load-bearing.** Every grid row from `l` onward is a training sample, so all
109 imputed days carry a regressor value into the loss. Two *defensible* rules — zero fill and
carry forward — move the forecast by up to **10.48 units on a series of scale 35.32, about 30 %**.
The garbage probe moves it by 192 and **flips the sign of both coefficients** (`promo` +0.264 →
−0.192, `price` −0.0048 → +0.571): a wrong value on 12 % of grid days does not degrade the model,
it inverts it.

Note also that at `n_lags = 7` the `price` weight collapses to −0.0048 from −0.0493 lag-free: the
AR term absorbs a smooth continuous driver. That is a separate identifiability caution, the same
class spike 011 found for Prophet.

### The change request's proposed API does not cover this case

`RegressorArg` as proposed carries `values` with `len == ds.len() + horizon` — one value per
caller row, plus one per future step. For Prophet that is exactly right. **For NeuralProphet with
a gappy series it is not enough**: the model trains on a daily grid that is *denser* than `ds`,
and with lags on it reads the grid rows `ds` does not cover. The contract as written leaves 109
values undefined in this fixture, and the measurement above says that choice is worth 30 % of the
series scale.

Three ways out, in the order I would rank them:

1. **Refuse.** On the neuralprophet arm with `n_lags > 0`, refuse a regressor when the series has
   missing days. Consistent with the door's existing habit of refusing rather than defaulting
   (D-11), and it needs no new semantics. Costs the caller a gap-filled series.
2. **Require grid-complete values.** Make the contract `len == (last - first + 1) + horizon` on the
   NP arm, so the caller supplies the value for every day. Honest, but it makes the argument shape
   differ between the two models — which is exactly what the change request asked to avoid.
3. **Impute and disclose.** Carry forward for binary columns, linear for continuous, stated in the
   response diagnostics. Cheapest for the caller; it is also the only option where the operator's
   number was invented by us, so it must be visible.

Linear interpolation of a **binary** driver is the trap inside option 3: it produces a "0.5 promo"
day, which is not a thing that happens. If a single rule is chosen for all columns, carry-forward
is the defensible one.

### Scope

One series, one gap pattern, `n_lags ∈ {0, 7}`, additive mode. The sensitivity number (10.48, 30 %
of scale) is illustrative of the magnitude, not a bound — it will move with gap fraction, gap run
length and driver volatility. The finding that lag-free never reads the value is structural and
holds for any series.
