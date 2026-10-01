# Prophet External Regressors (`add_regressor`)

Extending the shipped `aprender-forecast` Prophet design matrix with caller-supplied numeric
drivers — binary, continuous, additive and multiplicative — at parity with Python Prophet 1.4.0.

**The architecture question is already answered: this is an additive change, not a restructure.**
Spike 011's prototype is 219 lines (`sources/011-prophet-regressor-parity/src/regressors.rs`),
touches nothing under `crates/`, and reaches full parity using only the crate's existing public
types. Do not plan a `prophet.rs` rewrite.

## Requirements

From the `forecast-exogenous-inputs` idea (`.planning/spikes/MANIFEST.md`):

- **One `RegressorArg` shape serves both models**, so the caller sends one argument to `prophet`
  and `neuralprophet` alike.
- **Correctness bar for regressors is parity with Python Prophet 1.4.0**, measured by extending
  the existing rung ladder (data prep → predict-at-Python-params → components), not
  self-consistency.
- **No new required arguments.** Every new field is `Option<_>` with a serde default;
  `#[serde(deny_unknown_fields)]` stays as it is.
- **Errors, not silence.** An unsupported combination is refused at the door with a message
  naming the limitation — the existing D-11 pattern.
- **Byte-identical when the new arguments are absent** — see
  `references/no-argument-invariance-gate.md`; regressor columns *append*, which is what makes
  this free.
- The public entry point is **`forecast(&ForecastArgs)`**. `fitted_forecast` does not exist at
  HEAD or at the consumer's pinned tag `aprender-forecast-v0.63.0` (`fdf6b1802`).

## How to Build It

### 1. Column order — measured, not guessed

Prophet 1.4.0's design matrix is ordered:

```
seasonalities  →  holidays (sorted by the generated `{name}_delim_{±off}` string)  →  extra regressors (INSERTION order)
```

Holiday names sort as `+0, +1, +2, -1` because `'+' (0x2B) < '-' (0x2D)` — which is exactly what
the crate's existing `hcols.sort_by(name)` already produces. **Regressors append; no existing
column index moves.** Verified at 24 columns (regressors only) and 29/30 columns (regressors plus
two holidays with `upper_window: 1`), column order `yes` against the oracle in both.

### 2. Standardisation — the ddof trap

Prophet's `initialize_scales`, over the **history rows only**:

```rust
// `standardize = "auto"` ⇒ do NOT standardize a column whose unique values are exactly {0, 1}
let do_std = spec.standardize.unwrap_or(!binary);
let (mu, std) = if do_std {
    let n = history.len() as f64;
    let mu = history.iter().sum::<f64>() / n;
    let var = history.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / (n - 1.0); // ddof = 1
    (mu, var.sqrt())
} else {
    (0.0, 1.0)   // binary {0,1} under "auto"
};
```

`std` is pandas `Series.std()` — **ddof = 1**. Numpy's default `ddof = 0` gives `6.500320` where
Prophet stores `6.511442`: a silent 0.17 % shift in every regressor coefficient, and a parity
ladder that fails at rung 0 for a reason that looks like arithmetic noise.

### 3. Splice the columns onto the shipped `Design`

`Design` has every field `pub`. Append one column per regressor after the base block; the
optimiser and gradient need **no** change (rung 4 fits 24 and 29 columns with the unmodified
`fit_prophet`):

```rust
pub fn splice(d: &mut Design, regs: &[Standardized], values_history: &[Vec<f64>]) {
    let (base_k, t, r) = (d.k, d.t.len(), regs.len());
    if r == 0 { return; }                    // the no-argument invariance case: a no-op
    let new_k = base_k + r;
    let mut x = vec![0.0f64; t * new_k];
    for i in 0..t {
        x[i * new_k..i * new_k + base_k].copy_from_slice(&d.x[i * base_k..(i + 1) * base_k]);
        for (j, reg) in regs.iter().enumerate() {
            x[i * new_k + base_k + j] = (values_history[j][i] - reg.mu) / reg.std;
        }
    }
    for reg in regs {
        d.cols.push(Column { name: reg.name.clone(), component: reg.name.clone(),
                             mode: reg.mode, prior_scale: reg.prior_scale, holiday: None });
        d.prior_scales.push(reg.prior_scale);
        d.s_a.push(if reg.mode == Mode::Additive { 1.0 } else { 0.0 });
        d.s_m.push(if reg.mode == Mode::Multiplicative { 1.0 } else { 0.0 });
    }
    d.x = x;
    d.k = new_k;
}
```

`s_a` / `s_m` / `prior_scales` come out exact `0.0` against the oracle.

### 4. `predict` is the real signature change, not `feature_row`

`predict(d, p, ds_days, seed)` rebuilds X from `ds_days` alone via `feature_row(day, …)`.
**A regressor value is not a function of the day**, so `predict` needs a per-row value channel:

- `feature_row` itself needs **no** change — it keeps filling the seasonality and holiday cells.
- The regressor cells are written *beside* the row from the caller's array.

```rust
for (i, &day) in ds_days.iter().enumerate() {
    row.clear();
    feature_row(day, &d.spec, base_cols, &hol_sets, &mut row);   // base_cols = &d.cols[..d.k - regs.len()]
    x[i * d.k .. i * d.k + base_k].copy_from_slice(&row);
    for (j, reg) in regs.iter().enumerate() {
        x[i * d.k + base_k + j] = (values_all[j][i] - reg.mu) / reg.std;
    }
}
```

So `values` must cover history **and** the horizon rows: `len == ds.len() + horizon` on the
Prophet arm. (That contract is **not** sufficient on the NeuralProphet arm — see
`references/neuralprophet-exogenous-inputs.md`.)

### 5. Components Prophet publishes by name

Beyond one component per regressor, Prophet publishes two roll-ups that the response must carry:

- `extra_regressors_additive` — sum over additive regressor columns, scaled by `y_scale`
- `extra_regressors_multiplicative` — sum over multiplicative regressor columns, **not** scaled

### 6. Oracle and fixtures (reusable as-is)

```bash
uv run --python 3.12 --with "prophet==1.4.0" --with pandas python tools/oracle.py
uv run --python 3.12 --with "prophet==1.4.0" --with pandas python tools/oracle_holidays.py
```

Fixtures are committed at `.planning/spikes/011-prophet-regressor-parity/fixtures/`
(`retail_regressors_prophet140.json`, `retail_regressors_holidays_prophet140.json`) — reuse them
for the in-crate contract rungs rather than regenerating.

**Oracle fact that crashed run 1:** `make_all_seasonality_features(df)` returns
`(features, prior_scales, component_cols, modes)`. `s_a` / `s_m` are **derived in `fit()`** from
`component_cols`, not returned. Derive them the same way — it also yields the per-component
membership matrix the rung-3 comparison needs.

## What to Avoid

- **Forking `prophet.rs` into a prototype.** The whole architectural question is "do the shipped
  types admit this?", and forking makes it unanswerable. Compose against the public API by path
  dependency; if it compiles, the answer is yes.
- **Numpy `ddof = 0` for the regressor spread.** See above — 0.17 % on every coefficient.
- **Linearly interpolating a binary driver** anywhere in the pipeline: it produces a "0.5 promo"
  day, which is not a thing that happens.
- **Reading a coefficient disagreement as a defect.** Rust's `price` beta came out `+0.179538`
  against Python's `+0.038540` — 4.7×. Evaluating the objective at both parameter vectors gave
  slack **−3.42** against the contract's one-sided 0.5 bar: Rust found a *better* optimum. Two
  optimisers on a flat ridge are not a parity failure. Always run the objective control before
  calling a beta wrong.
- **Reporting "lift" for a collinear driver.** The change request's verification plan backtests
  each algorithm with and without the driver and reports lift per algorithm. On a driver collinear
  with trend or seasonality **that lift is not reproducible** — two optimisers at equal-or-better
  objective report coefficients 2–5× apart, and the seasonality absorbs the effect.
- **A pairwise correlation cutoff as the identifiability test.** A 0.9 threshold flagged `price`
  (r = 0.759 against `yearly_delim_1`) as identifiable, and its beta is still 5× off. Use the
  design matrix's **condition number or per-column VIF**.
- **An absolute components bar on a large-scale series.** `components_via_python_params_abs` is an
  absolute `1.0e-10`. On `retail_sales` (y_scale 518 253) the measured worst is 3.6e-11 — passing
  with only 3× headroom, where the same *relative* accuracy on Peyton (y_scale 12.85) sits at
  ~1e-15. A regressor fixture on a large-scale series should bind a bar **relative to `y_scale`**,
  as `predict_path_rel_yscale` already does, or the rung starts failing for arithmetic reasons.
- **Trusting the change request's premises.** Three of four were refuted against the consumer's
  own pinned tag before any code: NeuralProphet does *not* silently ignore `holidays`
  (`forecast.rs:117-130` already refuses with `"holidays is prophet-only; set model to
  \"prophet\""`); a holidays parity fixture *is* committed (`peyton_holidays_prophet140.json`,
  two holidays with `upper_window: 1`, measured `X 0.00e0 (K=30)`); and `fitted_forecast` does not
  exist. Check with `git show <tag>:path`, not HEAD — the consumer is on the tag.

## Constraints

Measured, `sources/011-prophet-regressor-parity/RUN-OUTPUT.md`:

| Rung | retail + 4 regressors | + 2 holidays with windows |
|---|---|---|
| 0 standardisation constants | 3.6e-15 | 3.6e-15 |
| 1 column order identical to Python | **yes** (24 cols) | **yes** (30 cols) |
| 1 `prior_scales` / `s_a` / `s_m` | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| 1 X (first3 + last3 rows) | 3.3e-15 | 3.3e-15 |
| 3 yhat at Python MAP | 2.3e-10 (**4.5e-16** of y_scale) | 1.7e-10 (3.4e-16) |
| 3 components | 9 compared, worst 3.6e-11 | 12 compared, worst 2.9e-11 |
| 4 fit objective slack | **−3.42** (bar 0.5) | **−2.22** (bar 0.5) |

Identifiability, same run:

| regressor | python β | rust β | r vs trend | max r vs another column | identifiable |
|---|---|---|---|---|---|
| promo (binary) | +0.001404 | +0.001427 | −0.006 | +0.022 `weather` | yes |
| price (continuous) | +0.036035 | +0.183202 | +0.647 | +0.759 `yearly_delim_1` | marginal |
| discount (multiplicative) | −0.004205 | −0.009709 | −0.013 | **+0.999** `yearly_delim_4` | **no** |
| weather (control) | −0.001322 | −0.001268 | −0.021 | +0.169 `yearly_delim_1` | yes |

- `discount` is a cosine of period 6 months — it **is** a harmonic of the yearly seasonality.
  A promotions flag is fine; a price series with a trend, or any driver with a 6- or 12-month
  cycle, is not. **Surface a collinearity diagnostic beside each regressor** rather than letting
  an operator read a lift number that will not hold.
- The fixture CSVs under `crates/aprender-forecast/tests/fixtures/` **quote their fields**
  (`"2007-12-10"`); strip quotes before parsing or the date validator refuses the row.
- Effort: the change request's 4–6 days stands, but the architectural risk it flagged is retired.
  Remaining work is door validation, the `predict` value channel, the response components, and a
  committed fixture with new contract rungs.

## Origin

Synthesized from spike: 011.
Source files available in: `sources/011-prophet-regressor-parity/`
(README.md, `src/regressors.rs` — the 219-line prototype, `src/main.rs` — the ladder driver,
`tools/oracle.py`, `tools/oracle_holidays.py`, `tools/report.py`, RUN-OUTPUT.md).
Fixtures (263 KB) and `results.json` (256 KB, chart-feeding data for `report.py`) stay in
`.planning/spikes/011-prophet-regressor-parity/`.
