---
spike: 012
idea: forecast-exogenous-inputs
name: no-arg-bitwise-invariance
type: standard
validates: "Given the door's fixture matrix, when the regressor plumbing is present but no new argument is passed, then every deterministic field of ForecastResponse is byte-identical to the pre-change build — and the signature that says so is proven able to fail"
verdict: VALIDATED
related: [011, 010]
tags: [invariance, determinism, delivery-gate, mutation-proof]
---

# Spike 012: No-argument bitwise invariance

## What This Validates

**Given** eight cases through the public door (`forecast`), spanning both models, linear /
logistic / multiplicative growth, holidays with windows, and AR lags, **when** the spike-011
regressor plumbing is present but handed no regressors, **then** every deterministic field of
`ForecastResponse` is bit-identical to the untouched path.

This is Forecast Coach's stated acceptance gate: *"Our deployed results must reproduce to the last
bit after the bump with no new arguments passed."*

## Research

The signature must exclude `fit_seconds` and `predict_seconds` — they are wall-clock, and a
signature containing them can never be stable, which would make the gate vacuous in the opposite
direction. Everything else is included. `diagnostics` was checked and is fully deterministic
(`forecast.rs:395-414`): no timings, and `serde_json`'s default `Map` is a `BTreeMap`, so key
order does not depend on insertion.

f64s are hashed by **bits** (`to_bits`), not by a printed decimal — a formatted comparison
silently accepts any change below the print precision. `-0.0` is normalised to `0.0`; every other
bit pattern is significant.

## How to Run

```bash
CARGO_TARGET_DIR=../../../target cargo run --release --quiet > RUN-OUTPUT.md
```

## What to Expect

Part A: all eight repeat-call pairs identical. Part B: every mutation **detected**, and the clean
signature restored afterwards. Part C: every dataset `INERT ✓`.

## Investigation Trail

1. **The door refused a case, and the refusal was the finding.** `retail/neuralprophet/MS` panicked
   with `Validation("neuralprophet supports freq D only")` — `forecast.rs:421-423`, pinned by its
   own test `neuralprophet_refuses_non_daily_freq`. This **answers the change request's P5 Q3**
   ("how do weekly and month-start series map onto NeuralProphet's daily imputed grid?"): they do
   not, because the door refuses them. The case was swapped for a second daily series.
2. **A baseline alone proves nothing**, so part B mutates the response and requires the signature
   to change. Without it, a signature function that returned a constant would have produced the
   same green table.
3. **Part C is the mechanism test, not a proxy.** Rather than assert that appending columns is
   harmless, it runs the spike-011 `splice` with an empty regressor list and compares the resulting
   `Design` and forecast against the crate's own untouched `make_design` / `predict`, field by
   field, bit by bit.

## Results

**VALIDATED.** Measured (`RUN-OUTPUT.md`, signatures in `baseline.json`):

**A — determinism, 8/8 identical on repeat**

| case | signature | fit s |
|---|---|---|
| peyton/prophet/default | `aa669c2352dd376a` | 1.39 |
| air/prophet/multiplicative | `3053244dcb27492c` | 0.01 |
| retail/prophet/default | `73b171523eb3fa3b` | 0.19 |
| wp_log_R/prophet/logistic | `331d67a8f8c924bb` | 0.76 |
| peyton/prophet/holidays+windows | `e0a13f9a10e3c3cc` | 1.29 |
| peyton/neuralprophet/lag0 | `d5a583795591a3db` | 0.16 |
| peyton/neuralprophet/lag7 | `726043759dde45dc` | 0.47 |
| wp_log_R/neuralprophet/lag0 | `72528357591ad4d6` | 0.15 |

**B — the gate can fail.** A 1-ULP bump in `yhat[0]` (`8.56059168341344368e0` →
`...545e0`), a 1-ULP bump in `trend[last]`, and one extra component key were each detected;
reverting all three restored `aa669c2352dd376a` exactly.

**C — the plumbing is inert at zero regressors.** On `peyton_manning`, `retail_sales` and
`air_passengers`: `Design` identical (`k`, `cols`, `x`, `s_a`, `s_m`, `prior_scales` all bit-equal),
`yhat` and `trend` bit-identical, all 3 shared components bit-identical.

### Why it is inert, mechanically

Spike 011 measured Prophet's column order as seasonalities → holidays (name-sorted) → regressors
(insertion order). Regressor columns therefore **append**; with zero regressors the append is
empty, `k` is unchanged, and no existing column index moves. Nothing downstream — the objective,
the gradient, the L-BFGS path, the component roll-ups — sees a different matrix.

### What this does NOT prove

- **The uncertainty bands were not compared in part C.** The spike-011 prototype computes point
  estimates only, so `yhat_lower` / `yhat_upper` are outside the mechanism test. They *are* covered
  by parts A and B, which go through the real door. The band path resamples changepoints around a
  yhat that depends on `beta` and `X`, so it should be added to the comparison when the feature
  lands in-crate.
- **The real re-check is a build step, not a spike step.** What this spike delivers is the harness,
  a signature proven able to fail, and a captured baseline at the current commit. Re-running it
  after the regressor work lands in `crates/aprender-forecast` is the actual gate.
- `budget_hit` inside `diagnostics` is wall-clock dependent. It did not flip across these runs, but
  on a loaded machine it could, and it would read as a signature change. Treat a lone `budget_hit`
  difference as an environment result, not a regression.

### Signal for the delivery

The constraint Forecast Coach made non-negotiable is satisfiable, and cheaply: because regressor
columns append rather than interleave, no-argument invariance falls out of the column-order
decision rather than needing to be engineered. Ship this harness alongside the feature and run it
as the release gate for every tag.
