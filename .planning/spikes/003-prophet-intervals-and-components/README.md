---
spike: 003
idea: prophet-forecast-mcp
name: prophet-intervals-and-components
type: standard
validates: "Given the spike-001 fit, when future changepoints are simulated (Prophet 1.4's vectorised trend-shift matrix, 1000 draws) and trend/seasonality/holiday components are decomposed, then the 80% band matches Python's width and covers ~80% of a holdout, components reconstruct yhat, and logistic growth, multiplicative seasonality and holidays fit with analytic gradients"
verdict: VALIDATED
related: [001, 004]
tags: [prophet, uncertainty, holidays, logistic, multiplicative, components]
---

# Spike 003: Prophet intervals, components, holidays, logistic, multiplicative

## What This Validates

Given four Prophet configurations fitted by Python Prophet 1.4.0 (Peyton default, Peyton with
playoff/Super Bowl holidays, `wp_log_R` with logistic growth and cap 8.5, air passengers with
multiplicative seasonality), when the same models are built in Rust on top of spike 001, then
design matrices, objectives, gradients, forecasts, named components and 80% intervals match, and a
holdout test shows the band covering what Python's band covers.

## Research

Same installed `prophet==1.4.0` source as spike 001, now the parts spike 001 skipped:

- **Multiplicative terms** (`prophet.stan`): `y ~ N(trend·(1 + X·(β∘s_m)) + X·(β∘s_a), σ)`.
- **Holidays** (`make_holiday_features`): one indicator column per `(holiday, offset)` in
  `lower_window..=upper_window`, named `{holiday}_delim_{±offset}`, columns **sorted by name**
  (`+0`, `+1` before `-1`), prior `holidays_prior_scale = 10`, mode = `seasonality_mode`.
- **Logistic growth** (`logistic_gamma` / `piecewise_logistic`): `trend = cap·σ(k_t(t − m_t))` with
  continuity offsets `γ_j = (cp_j − m − Σ_{i<j}γ_i)(1 − k_s[j]/k_s[j+1])`; init from
  `logistic_growth_init`. Its gradient is a reverse pass through the γ recursion (about 40 lines).
- **Uncertainty** (`_sample_uncertainty`, `_make_trend_shift_matrix`, `sample_model_vectorized`):
  for future rows, per draw, a Laplace(0, mean|δ| + 1e-8) shift where `U < S·single_diff`, averaged
  with the previous step, cumulated twice, scaled by `single_diff`; trend draws
  `(expected + unc)·y_scale`; `yhat = trend·(1 + Xb_m) + Xb_a + N(0, σ)·y_scale`; percentiles
  10/90 (numpy linear interpolation). Logistic uncertainty in 1.4 is a separate vectorised routine;
  the spike uses Prophet's pre-vectorised algorithm for it (Poisson-many new changepoints on
  `(1, T]`, Laplace deltas, full `piecewise_logistic`), which gives the same band (see Results).
- **Components** (`predict_seasonal_components`): `X·(β∘cols)`, times `y_scale` if additive;
  groups `additive_terms`, `multiplicative_terms`, `holidays`.

| Approach | Pros | Cons | Status |
|---|---|---|---|
| Analytic logistic gradient (reverse γ recursion) | Exact; FD-verified to 4.5e-8 | 40 lines of adjoint code | **Chosen** |
| Finite-difference trend gradient | Trivial | 27 extra objective evals per gradient | Not needed |
| Single L-BFGS run (spike 001 config) | Fast (0.17 s) | Stalls early at a kink on 2 of 4 fixtures (Δf +5.2, +15.5 vs Python) | Superseded |
| **L-BFGS with restarts from the stall point** | Reaches f below Python's on all 4 | 4–8 rounds, 0.04–3.1 s | **Chosen** |

## How to Run

```bash
cd .planning/spikes/003-prophet-intervals-and-components
CARGO_TARGET_DIR=../../../target cargo run --release        # ~6 s, writes RUN-OUTPUT.md-equivalent to stdout
open report.html                                            # four panels: y, Rust yhat, Rust band, Python band edges
# regenerate a fixture (needs uv):
uv run --python 3.12 --with prophet --with pandas python tools/make_fixture3.py holidays ../001-prophet-map-fit-lbfgs/fixtures/../../../.planning/spikes/002-neuralprophet-autograd/fixtures/peyton_manning.csv fixtures/peyton_holidays_prophet140.json
```

## What to Expect

Per fixture: identical column names/order and indicator matrices (1), gradient check ≤ 1.5e-7 (2),
a fit with restarts landing at or below Python's objective (3), components matching to 1e-16 and
reconstructing yhat exactly (4), band widths within ~1 % of Python's (5), and for Peyton a holdout
coverage table (6). The verdict run is `RUN-OUTPUT.md`.

## Investigation Trail

1. **Columns first.** Prophet sorts holiday columns lexically, which puts `playoff_delim_+0`,
   `playoff_delim_+1`, `superbowl_delim_+0`, `superbowl_delim_+1` after the Fourier block; the
   `s_a`/`s_m` indicator vectors and prior scales must line up with that order. All four fixtures:
   column lists identical, X rows 0.0 diff, 25 holiday ones in X on the holiday fixture.
2. **Gradients.** Central finite differences at a perturbed MAP: worst relative error 1.5e-7
   (linear), 4.4e-8 (holidays), **4.5e-8 (logistic, through the γ recursion)**, 2.1e-9
   (multiplicative). The logistic adjoint was right first time because it was checked, not
   because it was obvious.
3. **Predict path.** Python's parameters through Rust `predict()` reproduce yhat/trend to 1e-13
   and every named component (`yearly`, `weekly`, `playoff`, `superbowl`, `holidays`,
   `additive_terms`, `multiplicative_terms`) to 1e-16; `trend·(1+mult)+add` rebuilds yhat exactly.
4. **Intervals match to ~1 %.** Mean 80 % band width, future rows: Peyton 1.2907 vs 1.2936; holidays
   1.2352 vs 1.2353; logistic 0.5980 vs 0.6004; multiplicative 26.31 vs 26.38. History rows and the
   trend-only band match as well (0.2610 vs 0.2716 trend band on Peyton — seed-level noise, a
   second seed gives 1.3092 vs 1.2907 on the same yhat band). The pre-vectorised logistic
   algorithm gives the same trend band as 1.4's vectorised one (0.0024 vs 0.0024).
5. **Holdout coverage (Peyton, train 2540 / test 365, 3 seeds):** Rust 0.816 / 0.833 / 0.811 with
   mean width 1.31; Python 0.827 / 0.825 / 0.814 with mean width 1.31–1.32. Same band, same
   slight over-coverage. Test MAE 0.4131 vs 0.4125.
6. **Single-run fits stalled short on two fixtures** (Peyton Δf +5.2 after 203 iters; logistic
   +15.5 after 38) — the kink fragility spike 001 flagged, but at a worse landing point than any
   001 run. Restarting `LbfgsF64` from the stall point with fresh history until the objective stops
   improving fixed all four: Δf **−0.13, −0.54, −0.20, −0.04** vs Python (all better), Peyton
   future yhat max|Δ| **0.031 → 0.0030**, logistic **0.0107 → 0.0018**. Rounds: 6, 4, 8, 8.
7. **Cost of restarts:** Peyton 1312 iterations / 3.1 s (spike 001's single run: 726 / 0.26 s).
   The objective here is a generic additive+multiplicative loop; the L-BFGS line search also
   re-evaluates f and ∇f at x on every call. Both are build-time optimisations, not blockers.
8. **Multiplicative on monthly data** again disagrees with Python at daily resolution (future
   max|Δ| 26 on range 518) while the objective is *better* than Python's — the unidentified
   high-order yearly terms from spike 001, not a defect.

## Results

**Verdict: VALIDATED.** Intervals, components, holidays, logistic growth and multiplicative
seasonality all reproduce Python Prophet 1.4.0; the fit needs L-BFGS restarts to be robust.

| fixture | Δf vs Python | yhat future max abs (y range) | band width future Rust / Python | trend band Rust / Python | fit |
|---|---|---|---|---|---|
| Peyton default | −0.131 | 0.0030 (7.6) | 1.291 / 1.294 | 0.261 / 0.272 | 6 rounds, 3.1 s |
| Peyton + holidays | −0.535 | 0.0207 (7.6) | 1.235 / 1.235 | 0.278 / 0.276 | 4 rounds, 1.8 s |
| wp_log_R logistic (cap 8.5) | −0.196 | 0.0018 (5.0) | 0.598 / 0.600 | 0.0024 / 0.0024 | 8 rounds, 0.9 s |
| air passengers multiplicative | −0.044 | 26.3 (518) | 26.31 / 26.38 | 1.43 / 1.42 | 8 rounds, 0.04 s |

Holdout 80 % band coverage on Peyton: Rust 0.81–0.83, Python 0.81–0.83.

**Surprises**
- Prophet's "80 %" band covers 81–83 % out of sample here and 87–92 % in sample; both
  implementations agree, so this is Prophet's calibration, not a port error.
- The single-run L-BFGS fragility is worse than spike 001 suggested (Δf +15 on logistic). Restarts
  are cheap insurance and should be the default.
- Logistic uncertainty via the old Poisson-changepoint sampler matches 1.4's vectorised routine.

**Signal for the build**
- `prophet3.rs` is the model to port: `Spec` → `Design` → `Model` (objective/gradient) →
  `predict` (forecast + band + components). Country-holiday calendars are the one Prophet feature
  not covered (user-supplied holiday lists are).
- Fit = spike 001 config + restart loop (stop when the objective stops improving or after ~8
  rounds). Speed up the objective (SIMD `X·β`, avoid recomputing `f`/`∇f` at `x` in the line search).
- Seed the interval RNG per request for reproducible MCP responses; 1000 draws cost ~30 ms.
- Component output should be the Prophet column set (`trend`, `yhat`, `*_lower/upper`, per
  seasonality, per holiday, `holidays`, `additive_terms`, `multiplicative_terms`).
