# Prophet Port: MAP Fit, Intervals, Components, Holidays, Logistic, Multiplicative

Rust re-implementation of Facebook Prophet 1.4.0's Stan model on `aprender::optim::LbfgsF64`,
proven to parity on Peyton Manning, air passengers, retail sales and `wp_log_R` (spikes 001, 003).

## Requirements

From the `prophet-forecast-mcp` idea (MANIFEST.md):

- Correctness bar for the Prophet port is **parity with Python Prophet 1.4.0 on the Peyton Manning
  dataset** (fixture: `.planning/spikes/001-prophet-map-fit-lbfgs/fixtures/peyton_manning_prophet140.json`),
  not self-consistency.
- Both Prophet (MAP / L-BFGS) and NeuralProphet (autograd / AdamW) are in scope.
- The serving shape is a STATELESS `forecast` tool: fit and forecast happen inside one call, so the
  fit must be fast enough to run per request (Peyton: 1.4 s after the spike-004 optimisations).

## How to Build It

**1. Port the model as-is.** `sources/004-forecast-mcp-thin-server/src/prophet.rs` (26 KB) is the
most evolved copy: `Spec` → `columns()` → `make_design()` → `Model` (objective + analytic gradient)
→ `predict()` (yhat, band, named components). It is contract-shaped: zero-diff parity with Stan's
data on four datasets, objective at Python's MAP to 9e-13, predict path to 1e-13. Do not rewrite it.

Data preparation that MUST match Prophet exactly (all verified at 0.0 diff):

- `t = (ds − start)/(end − start)`, `y_scaled = y / max|y|`.
- Changepoints at `linspace(0, floor(0.8·T) − 1, 26).round()[1:]` — numpy rounds half-to-even, so
  use `f64::round_ties_even`.
- Fourier features on **days since 1970-01-01** (civil-date arithmetic, no `chrono`), columns
  interleaved `sin, cos` per order; yearly 365.25/10, weekly 7/3, daily 1/4.
- Auto seasonality: yearly if span ≥ 730 d; weekly if span ≥ 14 d and min gap < 7 d; daily if
  span ≥ 2 d and min gap < 1 d.
- Holiday columns: one per `(holiday, offset)` named `{holiday}_delim_{±offset}`, **sorted by name**
  (`+0`, `+1` sort before `-1`), prior scale 10, mode = `seasonality_mode`.
- Logistic trend `cap·σ(k_t(t − m_t))` with the γ continuity recursion; its gradient is a reverse
  pass through that recursion (FD-verified to 4.5e-8).

**2. Fit config — every piece is load-bearing.** From `sources/004-forecast-mcp-thin-server/src/lib.rs`
(`fit_prophet`, `Cached`) and `sources/006-chronos-vs-prophet-holdout/src/fit.rs`:

```rust
pub const MAX_ITERS_PER_ROUND: usize = 2_000;
pub const FIT_BUDGET_SECS: f64 = 15.0;

pub fn fit_prophet(design: &Design, max_rounds: usize /* 8 */) -> (Params, FitInfo) {
    let model = Model::new(design);              // objective ÷ T, non-finite guard → 1e300 + zero grad
    let mut x = Vector::from_vec(model.pack(&model.init()));
    let cache = Cached { model, last: RefCell::new(None), evals: RefCell::new(0) }; // x-keyed value+grad cache
    let mut best_f = cache.f(x.as_slice());
    let mut opt = LbfgsF64::new(MAX_ITERS_PER_ROUND, 1e-7, 20);
    let t0 = Instant::now();
    for _ in 0..max_rounds {
        let r = opt.minimize(|v| cache.f(v.as_slice()), |v| Vector::from_vec(cache.g(v.as_slice())), &x);
        let improved = r.objective_value < best_f - 1e-6 * best_f.abs().max(1.0);
        if improved { best_f = r.objective_value; x = r.solution; }
        if !improved || r.status == ConvergenceStatus::Converged { break; }
        if t0.elapsed().as_secs_f64() > FIT_BUDGET_SECS { /* budget_hit = true; */ break; }
    }
    // ...
}
```

Why each piece exists:

| Piece | Without it |
|---|---|
| Objective ÷ T | ‖g‖ ≈ 2899 at init on Peyton; the Wolfe search's first trial `x − g` sends `log σ` to −2899 → `NumericalError` at iteration 0 (Stan survives via `init_alpha = 0.001`) |
| Non-finite guard (return `1e300`, zero gradient) | Same failure; either fix alone works, both together are the recommended config |
| Exact L1 on δ (sign subgradient) | Smoothed L1 reaches a lower objective but takes 4–13 s and loses sparsity |
| `Stalled` accepted as success | With exact L1 the gradient norm can never reach `tol` (each near-zero δ contributes ±1/τ = ±20); every good fit ends `Stalled` |
| Restart loop from the stall point, fresh history, stop at < 1e-6 relative improvement or 8 rounds | Single runs stall short by Δf +5.2 (Peyton) and +15.5 (logistic) vs Python; restarts land **below** Python on all four fixtures |
| x-keyed `value_and_grad` cache | L-BFGS asks for `f` and `∇f` separately at the same x and re-asks at the accepted point; 11 312 evals for 1312 iterations. One-pass + cache: 2.55 s → 1.40 s on Peyton |
| Per-round iteration cap 2000 + 15 s wall-clock budget, reported in diagnostics | A 20 000-point fit ran 66 s with two rounds at the 10 000-iteration cap; capped it is 8.2 s |
| Special-case constant `y` **before** fitting | Objective is unbounded below (σ → 4e-8, f → −∞, 10 000 iters); Prophet special-cases `y.min() == y.max()` |

**3. Uncertainty and components** (`prophet3.rs` in spike 003, folded into 004's `prophet.rs`):
Prophet 1.4's vectorised trend-shift matrix, 1000 draws, Laplace(0, mean|δ| + 1e-8) future
changepoints, `yhat = trend·(1 + Xb_m) + Xb_a + N(0, σ)·y_scale`, percentiles by numpy linear
interpolation. **Seed the RNG per request** (`Rng(u64)` in `prophet.rs`) so responses are
reproducible; 1000 draws cost ~15–30 ms. Components: `X·(β∘cols)` × `y_scale` if additive; output
the Prophet column set (`trend`, `yhat`, `*_lower/upper`, per seasonality, per holiday, `holidays`,
`additive_terms`, `multiplicative_terms`). `trend·(1+mult)+add` must rebuild `yhat` exactly (1e-16).

**4. Parity test to keep.** Load the committed Prophet 1.4.0 fixtures
(`.planning/spikes/001-prophet-map-fit-lbfgs/fixtures/*.json`,
`.planning/spikes/003-prophet-intervals-and-components/fixtures/*.json`), assert: data prep 0.0
diff; objective at Python's MAP ≤ 1e-9; Python's params through Rust `predict` ≤ 1e-10; Rust fit
objective ≤ Python's + 0.5; future yhat max |Δ| inside Prophet's own Newton-vs-L-BFGS band
(Peyton 0.0079 on a range of 7.6). Regenerate fixtures with `tools/make_fixture.py` /
`tools/make_fixture3.py` under `uv run --python 3.12 --with prophet --with pandas`.

## What to Avoid

- **Do not judge parity by parameter equality or by daily-resolution yhat on monthly data.** On air
  passengers and retail sales the yearly Fourier block (20 columns fitting 12 points per cycle) is
  nearly unidentified; Prophet's own two optimisers disagree by 58 / 5252 there. The right bar is
  "objective within Stan's stopping slack" plus the Newton-vs-L-BFGS band.
- **Do not treat `Stalled` as failure.** A wrapper that does will reject every good fit.
- **Do not use core's `FISTA` for a proximal variant** — it is f32 with a fixed step. An f64
  proximal solver with soft-thresholding on δ is new work.
- **Do not check the exact-L1 gradient by finite differences at Python's exact MAP** — 12 δ sit at
  ±0.0 and FD sees ±1/τ; check at a perturbed point (2e-9 there).
- **Do not run the raw Stan objective through `LbfgsF64` unmodified** unless core gains Stan-style
  initial-step scaling (`α₀ = min(1, 1/‖g‖)`) or non-finite backtracking in `WolfeSearch`. Both are
  candidate core PRs; until then the ÷T + guard wrapper is required.
- Prophet 1.4 stores scalar params as 2-D arrays (`np.ravel` them in oracles);
  `example_air_passengers.csv` ships with CR-only line endings.
- Sort and de-duplicate by `ds` at load — `wp_log_R.csv` is not chronological (spike 006 caught it).

## Constraints

- Prophet 1.4.0 semantics only; country-holiday calendars are NOT covered (user-supplied holiday
  lists are). Sub-daily (`freq` H) needs fractional days throughout — not spiked; the tool refuses it.
- Fit cost: ~97 µs per objective+gradient eval at 2905×26; Peyton 1.4 s with restarts, 3k synthetic
  daily 0.21 s, 10k 1.19 s, 20k 8.2 s capped. Small series (≤ 300 rows) fit in 1–40 ms.
- Prophet's nominal 80 % band covers 81–83 % on a single Peyton holdout year but only **0.60** on
  rolling origins (spike 006) — report empirical coverage, not nominal.
- Core defects observed: `LbfgsF64` re-evaluates `f` and `∇f` at `x` inside every line search; the
  Wolfe search treats a non-finite trial as fatal (`NAN_SIGNAL`) instead of backtracking.

## Origin

Synthesized from spikes: 001, 003 (fit refinements from 004, fit.rs from 006)
Source files available in: `sources/001-prophet-map-fit-lbfgs/`, `sources/003-prophet-intervals-and-components/`,
`sources/004-forecast-mcp-thin-server/src/prophet.rs`, `sources/006-chronos-vs-prophet-holdout/src/fit.rs`
Parity fixtures (not copied, 5 MB): `.planning/spikes/001-prophet-map-fit-lbfgs/fixtures/`,
`.planning/spikes/003-prophet-intervals-and-components/fixtures/`
