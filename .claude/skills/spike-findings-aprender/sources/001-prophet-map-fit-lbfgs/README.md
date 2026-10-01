---
spike: 001
idea: prophet-forecast-mcp
name: prophet-map-fit-lbfgs
type: standard
validates: "Given Prophet's Stan model (piecewise-linear trend + 25 Laplace-prior changepoints + Fourier seasonality with Normal priors + σ), when it is written as an f64 objective with an analytic gradient and minimized with aprender's `LbfgsF64`, then it converges and the forecast matches Python Prophet 1.4.0 within Prophet's own optimizer-to-optimizer band"
verdict: VALIDATED
related: [003, 004]
tags: [prophet, lbfgs, changepoints, fourier, parity]
---

# Spike 001: Prophet MAP fit via aprender `LbfgsF64`

## What This Validates

Given Peyton Manning (2905 daily rows) plus Prophet's two monthly example series (air passengers,
144 rows; retail sales, 293 rows), when Prophet's Stan objective is re-implemented in Rust and
minimized with the L-BFGS that already lives in `aprender::optim`, then (a) the data preparation,
objective and predict path are bit-for-bit Prophet's, (b) the optimizer reaches an objective at
least as good as Stan's, and (c) the resulting forecast lies inside the band Prophet's own two
optimizers (L-BFGS vs Newton) span on the same data.

## Research

Grounded in the installed `prophet==1.4.0` source, not memory (`prophet.stan`, `forecaster.py`,
`models.py`):

- **Model** (`prophet.stan`): `k ~ N(0,5)`, `m ~ N(0,5)`, `delta ~ DoubleExponential(0, τ=0.05)`,
  `sigma_obs ~ N(0, 0.5)` (half-normal), `beta ~ N(0, 10)`,
  `y ~ N(trend·(1 + X·(β∘s_m)) + X·(β∘s_a), σ)` with
  `trend = (k + A·δ)·t + (m + A·(−t_change∘δ))` and `A[i,j] = 1{t_i ≥ t_change_j}`.
- **Data prep** (`setup_dataframe`, `initialize_scales`, `set_changepoints`): `t = (ds − start)/(end − start)`,
  `y_scaled = y / max|y|`, changepoints at `linspace(0, floor(0.8·T) − 1, 26).round()[1:]`
  (numpy rounds half-to-even → Rust `round_ties_even`). Fourier features use **days since epoch**,
  columns interleaved `sin, cos` per order; yearly 365.25 / order 10, weekly 7 / order 3, daily 1 / 4.
- **Auto seasonality** (`set_auto_seasonalities`): yearly if span ≥ 730 days; weekly if span ≥ 14 days
  and the minimum row gap < 7 days; daily if span ≥ 2 days and min gap < 1 day.
- **Fit** (`models.py`): cmdstan `optimize(algorithm='LBFGS' if T ≥ 100 else 'Newton', iter=1e4)`,
  init `k, m` through the first/last points, `δ = β = 0`, `σ = 1`. cmdstan optimizes on the
  unconstrained scale (`log σ`) **without** the Jacobian, so the MAP is in constrained space.
  Stan's L-BFGS starts with `init_alpha = 0.001` and stops on *relative objective* change.
- **Predict**: `piecewise_linear` on future `t`; components are `X_block·β_block·y_scale`.

| Approach | Tool | Pros | Cons | Status |
|----------|------|------|------|--------|
| Exact L1 (sign subgradient), raw Stan objective | `LbfgsF64` | Literally Stan's function | First trial step overshoots `log σ` by ‖g‖ ≈ 2899 → non-finite → `NumericalError` at iteration 0 | **Fails on Peyton**, works on the small series |
| Exact L1 + objective ÷ T + finite guard | `LbfgsF64` | O(1) gradients, no overshoot; 0.26 s | Terminates `Stalled` at a kink, never `Converged`; landing point varies by ±0.5 in f | **Chosen** (m = 20, tol 1e-7) |
| Smoothed L1 `sqrt(δ² + ε²)` | `LbfgsF64` | Reaches the lowest objective (−8005.15) | 4–13 s (10 000 iters); loses exact sparsity | Objective bound only |
| Proximal (soft-threshold on δ) | `FISTA` | Exact sparsity, true convergence | core's `FISTA` is f32 with a fixed step; no f64 variant | Not built — build-time option |

**Chosen approach:** exact L1, objective divided by T, non-finite guard, history 20, tol 1e-7.

**Gotchas found:** Prophet 1.4 stores scalar params as 2-D arrays (`np.ravel` them);
`example_air_passengers.csv` ships with CR-only line endings; `set --` does not word-split in zsh.

## How to Run

```bash
cd .planning/spikes/001-prophet-map-fit-lbfgs
CARGO_TARGET_DIR=../../../target cargo run --release            # all three fixtures
CARGO_TARGET_DIR=../../../target cargo run --release -- fixtures/peyton_manning_prophet140.json
open report-peyton_manning.html                                  # y / Rust yhat / Python yhat overlay
```

Regenerate the oracle fixtures (needs `uv`; installs prophet 1.4.0 into an ephemeral env):

```bash
uv run --python 3.12 --with prophet --with pandas python tools/make_fixture.py data/peyton_manning.csv fixtures/peyton_manning_prophet140.json peyton_manning
uv run --python 3.12 --with prophet --with pandas python tools/control_newton.py air_passengers retail_sales peyton_manning
```

## What to Expect

Sections 1–2 print zero-diff data-prep parity and an objective match at Python's MAP to ~1e-12.
Section 4 shows the raw-objective baseline dying at iteration 0 on Peyton and the recommended
config reaching f ≤ Python's in ~0.26 s. Section 5 shows forecast diffs; the summary table at the
end is the verdict. Full output of the verdict run: `RUN-OUTPUT.md`.

## Investigation Trail

1. **Port and check the function first, not the optimizer.** Rebuilt `t`, `y_scaled`,
   `changepoints_t`, the Fourier matrix and auto-seasonalities from raw `ds/y` with no date
   library (Hinnant's civil-date arithmetic): all **0.00e0** diff vs Prophet's Stan data on all
   three datasets. The Rust objective at Python's MAP equals Python's `−lp` to **9e-13**. Python's
   params through Rust `predict()` reproduce Python's `yhat`, `trend`, `yearly`, `weekly` to 1e-13.
   So every later disagreement is the optimizer, nothing else.
2. **Gradient.** Central finite differences: relative error 4e-8 at init, **2e-9** at a perturbed
   MAP. At Python's exact MAP the exact-L1 check "fails" by 19.7 — that is the kink (12 δ sit at
   ±0.0, FD sees ±1/τ = ±20). Expected, not a bug; the smooth variant passes there (6e-5).
3. **First fit: every variant `NumericalError` at iteration 0 after 3 evals.** ‖g‖ = 2899 at
   init; the Wolfe search's first trial is `x − 1·g`, which sends `log σ` to −2899, the objective to
   `inf`, and the line search returns its NaN *signal* instead of backtracking. Stan never hits this
   because its first step is `init_alpha = 0.001`. On T = 144 the same raw objective survives
   (‖g‖ ≈ 140 → `exp(−140)` is finite), which is why the baseline only fails on the long series.
4. **Two fixes on the objective side, tested as variants:** divide the objective by T (gradients
   become O(1)), and/or return `1e300` with a zero gradient when non-finite so Armijo backtracks.
   Either alone repairs it; together they are the recommended config. With them, all variants land
   within **−0.36 … +0.32** of Python's objective.
5. **Termination semantics.** With exact L1 the gradient norm can never reach `tol` (each near-zero δ
   contributes ±1/τ), so every run ends `Stalled` (step < ε), never `Converged`. Different starts
   stall at slightly different points (restart from a perturbed point: Δf = +0.54, yhat Δ 0.024).
   The smoothed objective finds the lowest f (−8005.15) but takes 13 s. **Python's own answer is an
   early stop too:** Stan quit at f = −8004.80 with ‖g‖ = 58.9.
6. **One dataset is an anecdote.** Added Prophet's two monthly example series. Peyton: future yhat
   max |Δ| **0.0049** on a range of 7.6. Air passengers: 35 on 518; retail: 1702 on 371 877 — and
   the whole disagreement is in `yearly` *between* the monthly samples (20 Fourier columns fit 12
   points per cycle under a σ = 10 prior; the posterior is flat along high-order terms).
7. **Control before naming a cause:** what does *Prophet* do when Stan switches L-BFGS → Newton on
   the same data? Air passengers: future |Δ| **58**, yearly 59. Retail: **5252**, yearly 6491. Peyton:
   0.008. Even `init_alpha=0.1` alone moves retail's future yhat by 7329. Our recommended fit is
   inside Prophet's own band on all three (see Results). The right notion of "parity" here is
   *objective within Stan's stopping slack*, not parameter equality.
8. **Edge probes** (recommended config): `y × 1e6` → identical up to the same slack (scale
   invariance holds); first 60 / 30 rows (Newton territory in Python; weekly-only, 25 / 23 clipped
   changepoints) fit in < 1 ms; `n_changepoints = 0` fits; **constant `y` diverges** (σ → 4e-8,
   f → −∞, 10 000 iters) — the objective is unbounded below there and Prophet special-cases
   `y.min() == y.max()` (σ = 1e-9, no fit); the build must too.
9. **Cost model:** 97 µs per objective+gradient eval at T×K = 2905×26, ~2700 evals → 0.26 s.
   Python's 0.33 s includes spawning cmdstan. Small series fit in ~1–10 ms.

## Results

**Verdict: VALIDATED.** `LbfgsF64` fits Prophet's MAP to parity, *provided the wrapper scales the
objective per observation and guards non-finite trial points* (or core's L-BFGS gains Stan-style
initial-step scaling / non-finite backtracking).

| dataset | rows | Rust fit | Python fit | Δf vs Stan | yhat hist max abs | yhat future max abs | Prophet's own Newton-vs-LBFGS future band | y range |
|---|---|---|---|---|---|---|---|---|
| peyton_manning | 2905 | 0.259 s | 0.33 s | −0.13 (better) | 0.0082 | **0.0049** | 0.0079 | 7.6 |
| air_passengers | 144 | 0.001 s | 0.02 s | +0.41 | 2.11 | 35.4 | 58.0 | 518 |
| retail_sales | 293 | 0.011 s | 0.05 s | +0.02 | 2139 | 1702 | 5252 | 371 877 |

Weekly component: identical (0.0000) on Peyton. Predict path: 1e-13. Data prep: 0.

**Surprises**
- The failure was not the L1 kink I expected; it was the *first step*. aprender's Wolfe search treats
  a non-finite trial as fatal (`NAN_SIGNAL`) and `compute_direction` uses `γ = 1` with empty
  history, so the first move is the raw negative gradient. Stan's `init_alpha = 0.001` is what
  makes Prophet robust, not anything in the model.
- `Stalled` is the *success* status for this objective. A wrapper that treats it as failure will
  reject every good fit.
- On monthly data the yearly Fourier block is nearly unidentified; "parity" at daily resolution
  is not a meaningful test there and Prophet itself fails it.

**Signal for the build**
- Port `Design`/`Model`/`predict` from `src/prophet.rs` as-is into core (it is already
  contract-shaped: zero-diff parity with Stan's data on three datasets).
- Fit config: objective ÷ T, finite guard, `LbfgsF64::new(10_000, 1e-7, 20)`; accept `Stalled`;
  judge success by finite objective ≤ init objective. Special-case constant `y` before fitting.
- Consider a core PR: initial step `α₀ = min(1, 1/‖g‖)` (Nocedal–Wright) or backtracking on
  non-finite trials in `WolfeSearch`. Then the raw Stan objective works unmodified.
- An f64 proximal solver (soft-threshold on δ) would give exact sparsity and a real convergence
  criterion; core's `FISTA` is f32/fixed-step, so this is new work, not a drop-in.
- No date library is needed for daily/monthly data: 60 lines of civil-date arithmetic gave exact
  parity. Sub-daily (`daily` seasonality) needs seconds, which the same arithmetic extends to.
