---
spike: 013
idea: forecast-exogenous-inputs
name: np-events-autograd
type: standard
validates: "Given NeuralProphet-lite's forward pass, when an additive Linear(E,1) event block is trained jointly on the f32 autograd, then a known synthetic event effect is recovered, fixed-seed runs are bit-identical, and the D-10 train-cost bound is checked against measured work"
verdict: VALIDATED
related: [002, 012]
tags: [neuralprophet, events, autograd, train-cost, determinism]
---

# Spike 013: NeuralProphet events on the autograd

## What This Validates

**Given** the shipped `NpModel::forward`, **when** an additive `Linear(E, 1)` block over event
indicator columns is trained jointly with it, **then** (A) it trains, (B) a known synthetic event
effect is recovered rather than absorbed by trend and seasonality, (C) fixed-seed runs stay
bit-identical, and (D) the D-10 train-cost bound is measured against real work.

## Research

`NpModel`, `Linear`, `rows_for`, `weighted_huber`, `one_cycle_lr`, `auto_batch`, `auto_epochs` and
`train_cost` are all public, so the event block is **composed beside** the shipped model rather
than forked into it: `pred = model.forward(xt, xs, None).add(block.forward(xe))`, with both
parameter sets handed to one `AdamW`. The spike-local loop mirrors `np::train` exactly — mini
batches, one-cycle lr, weighted Huber, `clear_graph()` after every step (D-10).

Indicator columns are one per (event, window offset), the same expansion the Prophet arm already
applies to `HolidayArg` — which is the change request's requirement that one argument serve both
models.

## How to Run

```bash
CARGO_TARGET_DIR=../../../target cargo run --release --quiet > RUN-OUTPUT.md
```

## What to Expect

The six learned event weights, denormalised, should land near the planted `+8.0`. Seed 42 twice
must be bit-identical and seed 43 must differ. The cost table's `priced cost` column must be
constant while `us/step` rises — that is the finding.

## Investigation Trail

1. **Composed, not forked.** The first design question was whether an event block needs `NpModel`
   to change. It does not: the public `forward` returns a `Tensor`, so an additive block is one
   `.add()` away, and `AdamW` takes the concatenated parameter list.
2. **A synthetic series with a planted effect**, rather than a real one, because "was the effect
   recovered?" needs a known answer. Trend + yearly + weekly + `+8.0` per active indicator + noise.
3. **The determinism check was made falsifiable.** Seed 42 twice bit-identical is only meaningful
   alongside seed 43 differing; both are asserted.
4. **Question D started as a yes/no and became a sweep.** Events ON at 6 columns cost 1.31× per
   step against an unchanged priced cost. Since the Prophet arm's `MAX_HOLIDAY_COLUMNS` is 1000 and
   the change request wants one argument for both models, the interesting number is not 6 but 1000
   — so E was swept to 1001.

## Results

**VALIDATED.** Measured (`RUN-OUTPUT.md`), 1200-point synthetic daily series, scale 30.43.

**A/B — the block trains and the effect is recovered**

| run | params | final train loss | MAE vs truth |
|---|---|---|---|
| events ON | 37 | 0.00035 | 0.403 |
| events OFF | 31 | 0.00237 | 0.697 |

All six learned weights land on the planted effect: `+7.769, +8.293, +7.990, +8.043, +8.249,
+8.462` against a truth of `+8.0` — worst error **0.462, 5.8 %**. Train loss falls 6.8× and MAE
against the noiseless truth falls 42 %. The effect is learned by the block, not absorbed by trend
or seasonality.

**C — determinism holds**

| check | result |
|---|---|
| seed 42 twice, predictions bit-identical | yes |
| seed 42 twice, event weights bit-identical | yes |
| seed 43 differs (the check can fail) | yes |
| tape length per step, ON / OFF | 25 / 20 |

The tape grows by a fixed 5 entries for the block and does not grow with E, so there is no leak —
`clear_graph()` after every step still empties it.

### D — the finding: the door's budget promise does not survive events

`train_cost(n_samples, epochs, n_lags) = epochs * n_samples * (n_lags + 1)` has **no term for
event columns**, so every row below is priced identically at 132 000 while the real work rises:

| event columns E | priced cost | µs/step | measured / priced |
|---|---|---|---|
| 0 | 132 000 | 12.9 | 1.00× |
| 7 | 132 000 | 15.7 | 1.22× |
| 28 | 132 000 | 18.9 | 1.47× |
| 84 | 132 000 | 21.7 | 1.68× |
| 210 | 132 000 | 32.9 | 2.55× |
| 504 | 132 000 | 58.2 | 4.50× |
| **1001** | 132 000 | 98.2 | **7.60×** |

Cost is **linear in E**, as a `Linear(E, 1)` over the batch should be: `µs/step ≈ 12.9 + 0.085·E`
on this box. At `E = MAX_HOLIDAY_COLUMNS` (1000, the ceiling the Prophet arm already allows, and
the one events would inherit by reusing `HolidayArg`) a request buys **7.6× the work the door
priced it at**.

This is precisely the change request's own concern — *"the training-cost bound extended for the new
input dimensions … the door's budget is a promise we rely on under Lambda's timeout"* — and it is
real, quantified, and larger than the 3-day estimate implies. The bound needs a term in the event
column count before events ship, not after; `request_train_cost` must take `n_event_cols` and the
`a_neuralprophet_request_over_the_train_cost_bound_is_refused` invariant re-derived against it.

The **shape** (linear in E) transfers; the **constant** (0.085 µs/column/step) is one machine,
lag-free, at one series length. Calibrate it on the deployment target before writing it into the
bound, and round the bound up — a door must never under-price.

### Scope

Lag-free (`n_lags = 0`) only. The AR paths (`predict_ar_recursive`, `predict_ar_1step`) and the
interaction between an event block and stationarised lags were not exercised; the change request
lists those as part of the same 3-day item and they remain unmeasured. Multiplicative event mode
was not built — the CR marks it optional.
