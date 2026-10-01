---
spike: 011
idea: forecast-exogenous-inputs
name: prophet-regressor-parity
type: standard
validates: "Given Python Prophet 1.4.0 with add_regressor on retail_sales plus binary, continuous and multiplicative regressors, when the shipped design-matrix path grows regressor columns, then column order, standardisation constants, s_a/s_m, X, predict-at-Python-MAP and every named component match the oracle, and the fit stays inside the contract's objective-slack bar"
verdict: VALIDATED
related: [001, 003]
tags: [prophet, regressors, parity, design-matrix, identifiability]
---

# Spike 011: Prophet external regressors

## What This Validates

**Given** Python Prophet 1.4.0 fitted on `retail_sales` with four `add_regressor` columns
(binary additive, continuous additive, continuous multiplicative, and an orthogonal control),
**when** the shipped `aprender-forecast` design-matrix path is extended with regressor columns,
**then** the parity ladder holds: column order identical, standardisation constants exact,
`prior_scales`/`s_a`/`s_m` exact, predict-at-Python-MAP within the contract bar, every named
component (including `extra_regressors_additive` / `extra_regressors_multiplicative`) matching,
and the Rust MAP fit inside `fitted_objective_slack`.

Run twice: regressors alone, and regressors **plus two holidays with non-zero windows**, because
the relative column order of holidays and regressors is exactly where a port drifts.

## Research

The change request's four premises were checked against the crate at the consumer's own pinned
tag `aprender-forecast-v0.63.0` (= `fdf6b1802`, matching their `fdf6b18`) before any code:

| CR premise | Verified |
|---|---|
| Prophet has no external regressors | **True.** No `regressors` on `ForecastArgs`, no regressor columns. |
| NeuralProphet silently ignores `holidays` | **False.** `forecast.rs:117-130` already refuses: `"holidays is prophet-only; set model to \"prophet\""`. Present at their pinned tag. The CR's 0.5-day refusal is already shipped. |
| No parity fixture carries holidays | **False.** `peyton_holidays_prophet140.json` is committed and present at their pinned tag: two holidays with `upper_window: 1`, 4 rungs, measured `X 0.00e0 (K=30)` and `playoff 0.0e0, superbowl 0.0e0, holidays 0.0e0`. |
| The entry point is `fitted_forecast(&ForecastArgs)` | **Does not exist** at HEAD or at the pinned tag. The public door is `forecast(&ForecastArgs)`. |

Oracle: `uv run --python 3.12 --with "prophet==1.4.0" --with pandas python tools/oracle.py`.
`make_all_seasonality_features` returns `(features, prior_scales, component_cols, modes)` — `s_a`
and `s_m` are **derived in `fit()`** from `component_cols`, not returned; the first oracle run
crashed on that.

## How to Run

```bash
# fixtures (committed; regenerate only if the oracle changes)
uv run --python 3.12 --with "prophet==1.4.0" --with pandas python tools/oracle.py
uv run --python 3.12 --with "prophet==1.4.0" --with pandas python tools/oracle_holidays.py

# the ladder
CARGO_TARGET_DIR=../../../target cargo run --release --quiet > RUN-OUTPUT.md
python3 tools/report.py && open report.html
```

## What to Expect

`RUN-OUTPUT.md` prints rungs 0/1/3/4 per fixture. Column order must read `yes`; the
standardisation, `prior_scales`, `s_a`, `s_m` diffs must be at float-noise; the rung-4 slack must
be **≤ 0.5** (negative is better-than-Python). `report.html` charts the three yhat curves, the two
`extra_regressors_*` components, and the identifiability table.

## Investigation Trail

1. **Read the shipped types before designing.** `Design` has every field `pub`, and `Model`,
   `feature_row`, `holiday_day_sets`, `make_design`, `columns` and `fit_prophet` are all public.
   The prototype therefore composes with the real crate instead of forking `prophet.rs` — which is
   itself the answer to the CR's architectural worry.
2. **`predict` is the actual constraint, not `feature_row`.** `predict(d, p, ds_days, seed)`
   rebuilds X from `ds_days` alone via `feature_row(day, …)`. A regressor value is not a function
   of the day, so **both** `make_design` and `predict` need a per-row value channel. `feature_row`
   itself needs no change: it keeps filling the seasonality and holiday cells, and the regressor
   cells are written beside it from the caller's array.
3. **Oracle run 1 crashed** unpacking `s_a`/`s_m` from `make_all_seasonality_features`. Fixed by
   deriving them from `component_cols`, which also gave the per-component membership matrix.
4. **Column order discovered, not guessed.** Regressors-only gave `yearly_delim_1..20, promo,
   price, discount` — insertion order, not alphabetical. A second oracle with holidays pinned the
   relative order: **seasonalities → holidays (name-sorted) → regressors (insertion order)**.
   Holiday names sort as `+0, +1, +2, -1` because `'+' (0x2B) < '-' (0x2D)`, which is what the
   crate's existing `hcols.sort_by(name)` already produces.
5. **The standardisation trap, found before writing Rust.** `mu` is the mean over history rows
   only; `std` is pandas `Series.std()` — **ddof = 1**. Numpy's default `ddof=0` gives `6.500320`
   where Prophet stores `6.511442`, a silent 0.17 % shift in every regressor coefficient. Binary
   `{0,1}` columns take `mu=0, std=1` under `standardize="auto"`.
6. **Rung 4 looked like a failure and was not.** Rust's `price` beta came out `+0.179538` against
   Python's `+0.038540` — 4.7×. The control (objective at both parameter vectors) showed slack
   **−3.42**: Rust found a *better* optimum, well inside the contract's one-sided 0.5 bar. The
   disagreement is a flat ridge, not a defect.
7. **Followed that to the cause.** A collinearity diagnostic against `t` and every other design
   column showed `discount` (a cosine of period 6 months) has **r = +0.999 with `yearly_delim_4`** —
   it *is* a harmonic of the yearly seasonality, so its coefficient is unidentifiable.
8. **Added an identifiable control** (`weather`, deterministic LCG noise, no trend or seasonal
   structure) and re-ran. Beta agreement now tracks identifiability exactly: the orthogonal
   regressors agree, the collinear ones do not.

## Results

**VALIDATED.** Both fixtures, measured (`RUN-OUTPUT.md`, `results.json`):

| Rung | retail + 4 regressors | + 2 holidays with windows |
|---|---|---|
| 0 standardisation constants | 3.6e-15 | 3.6e-15 |
| 1 column order identical to Python | **yes** (24 cols) | **yes** (29 cols) |
| 1 `prior_scales` / `s_a` / `s_m` | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| 1 X (first3 + last3 rows) | 3.3e-15 | 3.3e-15 |
| 3 yhat at Python MAP | 2.3e-10 (**4.5e-16** of y_scale) | 1.7e-10 (3.4e-16) |
| 3 components | 9 compared, worst 3.6e-11 | 11 compared, worst 8.7e-11 |
| 4 fit objective slack | **−3.42** (bar 0.5) | **−0.16** (bar 0.5) |

### The architecture question is answered: the change is additive, not a restructure

The prototype is 219 lines and touches nothing in `crates/`. The shipped `Design`, `feature_row`,
`holiday_day_sets`, `Model`, `columns` and `fit_prophet` were sufficient. Concretely:

- `columns()` **appends** regressor columns after the holiday block — no existing column moves.
- `Design` grows `cols`, `x`, `k`, `s_a`, `s_m`, `prior_scales` by `R` entries.
- `make_design` needs the history values; `predict` needs the values for the rows it is predicting.
  That second one is the only real signature change, and it is the CR's own observation.
- The optimiser and gradient needed **no** change, as the CR predicted — rung 4 fits 24 and 29
  columns with the unmodified `fit_prophet`.

### Surprise: a collinear driver has no reproducible lift

| regressor | python β | rust β | r vs trend | max r vs another column | identifiable |
|---|---|---|---|---|---|
| promo (binary) | +0.001404 | +0.001427 | −0.006 | +0.022 `weather` | yes |
| price | +0.036035 | +0.183202 | +0.647 | +0.759 `yearly_delim_1` | marginal |
| discount (mult.) | −0.004205 | −0.009709 | −0.013 | **+0.999** `yearly_delim_4` | **no** |
| weather (control) | −0.001322 | −0.001268 | −0.021 | +0.169 `yearly_delim_1` | yes |

This lands directly on the change request's own delivery plan. Its verification step 4 backtests
"each algorithm that can use the information … with and without it" and reports the lift per
algorithm. **On a driver that is collinear with trend or seasonality that lift is not
reproducible** — two optimisers at equal-or-better objective report coefficients 2–5× apart, and
the seasonality absorbs the effect. A promotions flag is fine; a price series with a trend, or any
driver with a 6- or 12-month cycle, is not. The port should surface a collinearity diagnostic
beside each regressor rather than let an operator read a lift number that will not hold.

My own 0.9 threshold is too loose: `price` at r = 0.759 was flagged "identifiable" and its beta is
still 5× off. A real implementation should use the design matrix's condition number or per-column
VIF, not a pairwise correlation cutoff.

### The components contract bar is scale-dependent

`components_via_python_params_abs` is an **absolute** 1.0e-10. On `retail_sales`
(y_scale 518 253) the measured worst is 3.6e-11 — it passes with only 3× headroom, where the same
relative accuracy on Peyton (y_scale 12.8) sits at ~1e-15. A regressor fixture on a large-scale
series should bind a bar relative to `y_scale`, as `predict_path_rel_yscale` already does, or the
rung will start failing for arithmetic reasons rather than parity ones.

### Effort signal for the change request's P1

The CR's 4–6 days stands, but the architectural risk it flagged is retired: no restructure. The
remaining work is door validation, the `predict` value channel, the response components, and a
committed fixture with new contract rungs — the fixture and oracle from this spike are reusable
as-is.
