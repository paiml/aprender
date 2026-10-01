//! The tool boundary's argument and response types, and the DoS bounds that make
//! [`crate::forecast::forecast`] safe to expose to untrusted JSON-RPC arguments (D-11).
//!
//! Ported verbatim from `sources/004-forecast-mcp-thin-server/src/lib.rs:26-103` (D-08).
//! The doc comments on [`ForecastArgs`] are load-bearing: `schemars` lifts them into the
//! advertised MCP tool schema.

// schemars' JsonSchema derive expands to .unwrap() internally, and the derive's
// generated impl lands at file scope where a struct-level allow cannot reach it.
// Same precedent as aprender-mcp-setfit/src/lib.rs:29-32.
#![allow(clippy::disallowed_methods)]

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Hard upper bound on history length (T-06-02: this is what actually bounds the work).
pub const MAX_POINTS: usize = 20_000;
/// Hard upper bound on the number of future periods.
pub const MAX_HORIZON: usize = 3_650;
/// Fewer points than this and there is nothing to fit.
pub const MIN_POINTS: usize = 10;

/// Hard upper bound on the DAILY SPAN (`last - first + 1`) a fit may cover.
///
/// `MAX_POINTS` bounds how many points arrive; it does NOT bound how far apart they are.
/// `np::NpData::new` materialises an imputed DAILY GRID over `first..=last`, so ten points
/// a millennium apart bought a 3.6-million-row grid and hundreds of MB of feature matrices
/// from a ~200-byte request. Equal to `MAX_POINTS` so the grid can never exceed the rows a
/// fully dense series at the point ceiling would produce (D-11).
pub const MAX_SPAN_DAYS: i64 = MAX_POINTS as i64;

/// Hard upper bound on `|lower_window|` and `upper_window` for a single holiday.
///
/// `prophet::columns` emits one design column per offset in `lower_window..=upper_window`,
/// so the two `i64` fields were an unbounded column multiplier: only their SIGN was
/// checked. A year of window either side is past anything Prophet's own users write.
pub const MAX_HOLIDAY_WINDOW: i64 = 365;

/// Hard upper bound on the total number of holiday design columns across all holidays.
///
/// Bounds `sum(window_width)` the way [`MAX_HOLIDAY_WINDOW`] bounds one term of it, so a
/// long list of individually legal holidays cannot multiply back into the same blow-up.
pub const MAX_HOLIDAY_COLUMNS: usize = 1_000;

/// Hard upper bound on the number of dates one holiday may carry.
///
/// `prophet::feature_row` used to SCAN this list per row per holiday column, which made it
/// a second multiplier on the design build. Since 06-11 membership is a prebuilt
/// `HashSet` lookup (`prophet::holiday_day_sets`), so the list is an additive
/// set-construction cost rather than a multiplier — but it is still per-holiday only, and
/// [`MAX_HOLIDAY_DATES_TOTAL`] is what bounds the aggregate.
pub const MAX_HOLIDAY_DATES: usize = 1_000;

/// Hard upper bound on `(points + horizon) * holiday_columns` — the design feature cells
/// ONE request buys across `prophet::make_design` (points rows) AND `prophet::predict`
/// (horizon rows).
///
/// **Why the existing bounds did not cover this.** [`MAX_POINTS`], [`MAX_HORIZON`] and
/// [`MAX_HOLIDAY_COLUMNS`] are each checked in ISOLATION and their product is not, so a
/// request inside all three bought `(20_000 + 3_650) * 1_000` = 23 650 000 feature cells —
/// and every fit iteration is `O(rows * K)` over exactly that matrix.
/// `fit::FIT_BUDGET_SECS` cannot substitute: it is a COOPERATIVE ROUND-BOUNDARY budget, so
/// it is blind to `make_design` (which runs before `fit_prophet` is entered) and it
/// overshoots by a whole round inside the fit. That overshoot is measured, not argued — a
/// 20 000-point, 1 000-column request walled at **70.089 s** against a 15 s budget.
///
/// **The measured number that motivated the value.** A 3 000-point daily series with 181
/// holiday columns and 84 dates — inside every bound above — walled at **16.081 s**
/// against SC1's 2 s bar (`just forecast-holiday-bench 3000 181 84 365`, release,
/// aarch64). The same series with no holidays walls at 0.212 s.
///
/// **50 000 is the largest round value whose at-the-bound walls all clear 2 s** on that
/// host, measured at three compositions of the SAME product: many rows / few columns
/// (9 500 x 5) 1.174 s, balanced (800 x 50) 1.692 s, few rows / many columns (50 x 500)
/// 0.106 s. At 100 000 two of the three compositions exceed the bar (2.771 s, 2.305 s).
///
/// **This bounds WORK, not WALL, and the distinction is measured.** The fit's iteration
/// count is data-dependent and no payload statistic predicts it: a 4 700-point, 5-column
/// request is only 25 000 cells — half this bound — and still walls at 4.2 s, reproducibly.
/// What this constant guarantees is the arithmetic ceiling per iteration, which is what
/// turns the ~85-minute in-bounds request the verifier extrapolated into a refusal.
pub const MAX_HOLIDAY_DESIGN_COST: usize = 50_000;

/// Hard upper bound on `sum(len(dates))` across ALL holidays in one request.
///
/// **Why the existing bounds did not cover this.** [`MAX_HOLIDAY_DATES`] bounds ONE
/// holiday's list; nothing bounded the sum, so 1 000 holidays each carrying 1 000 dates
/// put 1 000 000 dates through `parse_date` and into the membership sets from a single
/// request. This is the third factor of the product [`MAX_HOLIDAY_DESIGN_COST`] closes
/// the first two of.
///
/// **The derivation.** The largest total any committed fixture, example or test sends is
/// **17** — Prophet 1.4.0's own canonical `peyton_holidays` frame (14 `playoff` + 3
/// `superbowl`). 10 000 is 588x that, and building the two membership sets for 10 000
/// dates measures **0.115 ms**, three orders of magnitude under the 50 ms this bound was
/// derived against (the structural maximum of 1 000 000 measures 18.049 ms).
pub const MAX_HOLIDAY_DATES_TOTAL: usize = 10_000;

/// Hard upper bound on the logistic uncertainty simulation's Poisson mean
/// `lambda = changepoint_count(len(ds)) * (t_max - 1)` — the count of NEW changepoints
/// each of the 1 000 simulation rows draws in `prophet::predict`'s `Growth::Logistic` arm.
///
/// **Why the existing bounds did not cover this.** [`MAX_POINTS`], [`MAX_SPAN_DAYS`] and
/// [`MAX_HORIZON`] are each checked in ISOLATION and it is their RATIO that sizes this
/// cost: `t_max = (future_span_days) / (history_span_days)`. Worse, [`MAX_HORIZON`] bounds
/// the COUNT of future steps while `dates::future_days` multiplies that count by 7 for
/// `"W"` and by ~30.44 for `"MS"`, so the same legal horizon buys a 30x longer span on one
/// frequency than on another. A high-lambda request is therefore necessarily a SHORT-history
/// request, which is precisely why every point-count bound is satisfied while the product is
/// not — the same shape as [`MAX_HOLIDAY_DESIGN_COST`].
///
/// **Why `fit::FIT_BUDGET_SECS` cannot substitute.** That budget is entered inside
/// `fit::fit_prophet`. This cost is spent in `prophet::predict`, which `forecast::forecast`
/// calls AFTER the fit returns, with no budget of any kind. No setting of `FIT_BUDGET_SECS`
/// could ever have seen it.
///
/// **The measured breach (`06-REVIEW.md` CR-01, reproduced three times).** Release build,
/// `--http --pool 1`, min of 2 runs, same host. A **1 132-byte** accepted request — 33 daily
/// points, `horizon: 3650`, `freq: "MS"`, `growth: "logistic"`, `cap: 50`, inside every door
/// bound — went from **0.157 s** before 06-13 to **2.334 s** at HEAD (14.8x), over SC1's 2 s
/// bar. Its controls pin the mechanism rather than argue it: the same request at 30-day point
/// spacing (lambda / 30) is 0.209 s, at `freq: "W"` 0.634 s, at `freq: "D"` 0.231 s, and on
/// the `"linear"` arm — the one that never calls `poisson` — flat at 0.149 s. The two logistic
/// rows differ ONLY in history spacing, so the 11x spread between them is lambda and nothing
/// else. 06-13 did not cause this by being wrong: it correctly deleted Knuth's accidental
/// saturation near 745, after which the drawn count tracks lambda — which nothing bounded.
///
/// **The structural maximum is 86 789.8**, and it is exactly the review's configuration:
/// `changepoint_count` peaks at 25 (`Spec::default_linear`) and needs
/// `floor(n * 0.8) - 1 >= 25`, i.e. `n >= 33` points; the tightest legal span for 33 daily
/// points is 32 days; and the widest legal future span is 3 650 `MS` steps = 111 123 days.
/// `25 * (111123 / 32 - 1)` = 86 789.8.
///
/// **20 000 is the value, and it is derived from measurement.** Interpolating the review's
/// four points (lambda 2 852 -> 0.231 s, 2 866 -> 0.209 s, 19 925 -> 0.634 s, 86 790 ->
/// 2.334 s) puts the 2 s bar near lambda 74 000; 20 000 is the largest round value with
/// roughly 3x measured headroom. Walled at three compositions on a release build, at the
/// tightest legal history span (33 daily points). Originally measured by
/// `prophet::sampler::logistic_band_wall` under `LOGISTIC_BENCH_FREQ`; that harness was
/// DELETED by plan 06-16 because it was `#[ignore]`d with no recipe and no bar, and its
/// geometry is now three rows of [`crate::sc1_wall`]'s cross product, which
/// `just forecast-sc1-sweep` runs on release with the 2 s bar asserted:
///
/// | freq | points | horizon | lambda | total_s (06-14) | total_s (06-16 sweep) |
/// |------|--------|---------|--------|-----------------|-----------------------|
/// | `D`  | 33 | 3 650 | 2 851.6 (freq `D`'s OWN structural maximum) | **0.244 s** | 0.224 s |
/// | `W`  | 33 | 3 650 | 19 960.9 | **0.711 s** | 0.714 s |
/// | `MS` | 33 | 840 / 841 | 19 974.2 / 19 996.1 | **0.582 s** | 0.620 s |
///
/// The two columns are the SAME compositions measured by two harnesses on the same host and
/// they agree; the `MS` row differs in the last horizon step because the sweep DERIVES the
/// widest legal horizon from the bound rather than being told one.
///
/// The worst of the three is ~0.71 s against the 2 s bar — roughly 2.8x headroom — and the
/// cost is almost entirely in `predict_s`, which is the half `FIT_BUDGET_SECS` does not
/// cover. The band still widens with lambda (the sweep prints `band_width=` on every line:
/// 21.07 on both at-the-bound rows against 19.74 on the `D` row), so the bound refuses cost;
/// it does not quietly truncate the simulation the way the pre-06-13 saturation did. The
/// absolute widths are smaller than the 49.37 / 49.39 / 45.51 the deleted harness recorded
/// because it hardcoded `cap: 50.0` while the sweep derives the cap from `max(y)`, which is
/// the change that stops a future edit to the synthetic series silently invalidating the
/// composition.
///
/// `freq: "D"` cannot reach this bound at all: at 33 points its largest attainable lambda is
/// 2 851.6, so its row is that maximum rather than an at-the-bound point. That is a fact about
/// the frequency multiplier, not a gap in the measurement — `MS` is 30.44x `D` on the same
/// horizon, which is why the bound has to be on the PRODUCT and not on the horizon.
pub const MAX_LOGISTIC_CHANGEPOINT_LAMBDA: f64 = 20_000.0;

/// Lower end of the REPRESENTABLE domain for `RegressorArg.prior_scale`.
///
/// Mirrors `constants.regressor_prior_scale_min` in `forecast-tool-boundary-v1`, asserted
/// equal by `cost_bounds_match_contract`.
///
/// **Why "greater than zero" is not the bound.** `Model::objective` computes
/// `b * b / (2.0 * sc * sc)` (`prophet.rs:536`) and the gradient computes
/// `beta[c] / (sc * sc)` (`prophet.rs:685`): the prior scale is SQUARED into a denominator,
/// and `f64::MIN_POSITIVE` (2.2250738585072014e-308) squares to EXACTLY `0.0`, so the
/// initial zero coefficients meet `0.0 / 0.0`.
///
/// **What that 0/0 actually does — measured, not assumed.** This bound was introduced on
/// the theory that the NaN "propagates through L-BFGS into every response field and
/// serialises as JSON null". That is REFUTED: across 18 configurations spanning three
/// series shapes, no response field is ever non-finite, because `Model::objective`'s
/// existing `if self.guard && !f.is_finite() { return 1e300 }` absorbs it. The real
/// outcome is quieter and arguably worse — L-BFGS runs ZERO iterations and
/// `extra_regressors_additive` is exactly `0.0` everywhere, while the door returns a
/// normal-looking forecast. The regressor is silently ignored (the D-21 class).
///
/// **The residual, stated so it is not mistaken for covered.** The degenerate region is far
/// wider than this floor: every `prior_scale <= 1e-9` measured 0-1 iterations with a zero
/// regressor contribution, while `>= 1e-7` fits normally. `1e-153` therefore sits roughly
/// 145 decades INSIDE the collapse region. This constant guarantees only that the
/// arithmetic stays REPRESENTABLE and the objective finite — which is what this plan's
/// acceptance criterion asks — never that the regressor is meaningfully fitted. A
/// usability floor needs its own measurement campaign (plan 06.1-03's territory).
///
/// DERIVED, not chosen: the smallest value whose square is still normal, rounded up to a
/// round decade. `sqrt(f64::MIN_POSITIVE) = 1.4916681462400413e-154`, so the decade above
/// is `1e-153`, whose square `1e-306` IS normal; the decade below squares to `1e-308`,
/// which is SUBNORMAL.
pub const REGRESSOR_PRIOR_SCALE_MIN: f64 = 1.0e-153;

/// Upper end of the same range. Mirrors `constants.regressor_prior_scale_max`.
///
/// DERIVED: the largest value whose square stays well inside `f64::MAX`.
/// `sqrt(f64::MAX) = 1.3407807929942596e154`, backed off to the round decade `1e150`, whose
/// square is ~`1e300` — about 1.8e8 below `f64::MAX`.
pub const REGRESSOR_PRIOR_SCALE_MAX: f64 = 1.0e150;

/// Hard upper bound on the NUMBER of `RegressorArg` entries in one request.
///
/// FINALISED by plan 06.1-03 Task 2 step 5, against a door that CONTAINS the identifiability
/// diagnostic. Task 1 could not decide this value: the only cost that is a function of the
/// count alone is the diagnostic's `O(K^3)` factorisation, and it did not exist yet.
///
/// Replaces plan 06.1-01's interim literal of 50, which was deliberately temporary and
/// carried no measurement at all.
///
/// **Why [`MAX_REGRESSOR_DESIGN_COST`] does not cover it.** The product ceiling is
/// satisfiable by trading rows for columns — at the tightest legal history it would alone
/// admit `25_000 / 11` = 2 272 regressors — and it is measured NOT to bound the count:
/// against the stage-1 door (no diagnostic) a request at 800 regressors walled at
/// **0.063 s**, so neither the fit, the design build nor `predict` constrains it.
///
/// **THE COUNT LADDER, measured with the diagnostic LIVE** (release, aarch64,
/// `just forecast-regressor-bench 25000 <R>`). `diag_s` is `total_s - fit_s - predict_s`,
/// which is where this routine's work lands:
///
/// | R | K | K^3 | diag_s | worst total_s | verdict |
/// |---|---|---|---|---|---|
/// | 200 | 207 | 8.9e6 | 0.003-0.004 | 1.157 | **ACCEPTED** |
/// | 400 | 407 | 6.7e7 | 0.017-0.018 | 1.203 | clears |
/// | 800 | 807 | 5.3e8 | 0.127-0.132 | 1.262 | clears |
/// | 2 000 | 2 001 | 8.0e9 | 1.902-1.951 | 2.038 | **REJECTED** — 3 of 5 at or over the bar |
///
/// **The cubic law is CONFIRMED by the ladder rather than asserted by the formula**: the
/// `diag_s` ratios are 5.0x, 7.4x and 15x against `K^3` ratios of 8.0x, 7.9x and 15x, so
/// the last two track within 7 %. At R = 2 000 the diagnostic is 96 % of the whole request.
///
/// **200 is the value, and it is deliberately NOT the largest that clears.** 2 000 is the
/// measured rejection. 800 CLEARS and was still not taken, for two stated reasons:
///
/// 1. At 800 the diagnostic (0.130 s) costs MORE than the fit it describes (0.040-0.124 s).
///    A diagnostic that dominates the thing it diagnoses is the wrong trade.
/// 2. This is published refusal surface and the door swings one way: raising a ceiling later
///    is cheap, lowering one refuses requests that previously succeeded. 6.5 % of the SC1
///    budget on a host with headroom is not a margin Phase 7's tighter tiers can be assumed
///    to have.
///
/// 200 is 50x the largest regressor count in any committed fixture, example or test (4, the
/// spike-011 retail frame). Phase 7 can raise it per tier against the cubic law above, which
/// is now measured rather than projected — that is the point of recording the whole ladder
/// instead of only the chosen rung.
///
/// Mirrors `constants.fit_max_regressors`, asserted equal by `cost_bounds_match_contract`.
pub const MAX_REGRESSORS: usize = 200;

/// Hard upper bound on `(points + horizon) * n_regressors` — the regressor design feature
/// cells ONE request buys across `regressors::splice` (history rows) and `prophet::predict`
/// (horizon rows).
///
/// FINALISED by plan 06.1-03 Task 2 step 5. Cost axis C-17 prices these design cells AND the
/// identifiability Gram plus its factorisation, `len(ds) * K^2 + K^3`; the second did not
/// exist when Task 1 first measured, so that sweep was re-run in full against the completed
/// door. Both sweeps are below, because a ceiling certified against incomplete work is
/// certified against the wrong request.
///
/// **Why the existing bounds did not cover this.** Exactly the [`MAX_HOLIDAY_DESIGN_COST`]
/// shape, one column family over: [`MAX_POINTS`], [`MAX_HORIZON`] and [`MAX_REGRESSORS`] are
/// each checked in ISOLATION and their product is not, so a request inside all three would
/// buy `(20_000 + 3_650) * 200` = 4 730 000 regressor design cells — and every fit iteration
/// is `O(rows * K)` over exactly that matrix, `predict` re-sweeps it per distinct component,
/// and the diagnostic builds an `N x K^2` Gram on top. `fit::FIT_BUDGET_SECS` cannot
/// substitute for the same structural reason it could not substitute for the holiday
/// product: it is a COOPERATIVE ROUND-BOUNDARY budget entered inside `fit::fit_prophet`, so
/// it is blind to `splice` (before) and to `predict` and the diagnostic (after), and it
/// overshoots by a whole round inside the fit.
///
/// **SWEEP 1 — the design build alone, no diagnostic (Task 1).** `total_s` through the
/// public door at five compositions of the SAME product differing in every factor:
///
/// | cost | many/4 | balanced/20 | few/200 | max width/200 | combined/200 | verdict |
/// |---|---|---|---|---|---|---|
/// | 50 000 | 1.391 | **2.817** | 0.070 | 0.199 | 0.360 | **REJECTED** |
/// | 40 000 | 1.132 / 1.369 | **2.182 / 3.021** | 0.055 | **4.735 / 6.024** | 0.190 | **REJECTED** |
/// | 30 000 | 1.695 / 1.836 | 0.254 | 0.070 | 0.084 | 0.382 | clears, 1.09x headroom |
/// | 25 000 | 0.681 / 0.683 | 1.268 / 1.226 | 0.036 | 0.051 | 0.145 | ACCEPTED, 1.58x |
///
/// **SWEEP 2 — the WHOLE REQUEST, diagnostic live (Task 2 step 5).** The identical five
/// compositions, re-measured:
///
/// | cost | many/4 | balanced/20 | few/200 | max width/200 | combined/200 | verdict |
/// |---|---|---|---|---|---|---|
/// | 40 000 | 1.130 | **2.164** | 0.050 | **4.242** | 0.205 | **REJECTED**, re-confirmed |
/// | 25 000 | 0.665 | 1.157 | 0.032 | 0.053 | 0.144 | **ACCEPTED**, 1.73x headroom |
///
/// **THE DELTA, which is the number Phase 7 needs.** Sweep 1 minus sweep 2 on `total_s` is
/// within run-to-run noise on the fit-dominated rows (-0.019, -0.031, +0.001, +0.002,
/// -0.002), so the diagnostic's share is NOT visible in the total. It IS visible in
/// `total_s - fit_s - predict_s`, which rose from **<= 1 ms** to **3-4 ms** at R = 200 — a
/// measured 0.2 % of the 2 s bar, not an unmeasured one. That is the evidence the
/// `N*K^2 + K^3` term in C-17's formula is bounded rather than asserted; the full cubic
/// ladder behind it is in [`MAX_REGRESSORS`].
///
/// **25 000 is the value.** 50 000 and 40 000 are the REJECTED candidates, each re-run so
/// the failures are known to be reproducible rather than scheduler noise, and 40 000 was
/// re-rejected against the completed door. 30 000 CLEARS and was still not taken: its worst
/// composition is 1.836 s against a 2.0 s bar, an 8 % margin, and a bound that clears by
/// 8 % is a coin flip rather than a ceiling.
///
/// **This bounds WORK, not WALL, and here the distinction is the dominant effect, not a
/// caveat.** Every wall above is >= 97 % `fit_s`, and the L-BFGS iteration count is
/// data-dependent and NON-MONOTONIC in the payload: the max-width composition at 199 points
/// measures **4.735 s** while the SAME shape at 249 points measures **0.199 s**, a 24x swing
/// across a 50-point change with the product held constant. No payload statistic predicts
/// that, which is why the sweep is five compositions rather than one. What this constant
/// guarantees is the arithmetic ceiling per iteration, per component sweep and per Gram.
///
/// Mirrors `constants.fit_max_regressor_design_cost`, asserted equal by
/// `cost_bounds_match_contract`.
pub const MAX_REGRESSOR_DESIGN_COST: usize = 25_000;

/// WARN-above threshold for a regressor's variance inflation factor (D-35).
///
/// **NOT a cost ceiling, and the name is load-bearing.** A warning threshold bounds no work
/// and refuses nothing, so it must NOT match the `fit_max_` / `chronos_max_` prefix filter
/// that [`tests::every_cost_ceiling_constant_is_named_by_an_axis`] derives its ceiling list
/// from — pulling it in would demand a cost axis for a number that prices nothing. Same
/// reasoning as [`REGRESSOR_PRIOR_SCALE_MIN`], which is a numeric DOMAIN bound.
///
/// **DERIVED from the committed spike-011 fixture, not chosen.** Running the diagnostic over
/// `retail_regressors_prophet140.json`'s four regressor columns against the 24-column design
/// gives the VIFs recorded in the 06.1-03 SUMMARY. The value is pinned so that the KNOWN-BAD
/// column warns and the two KNOWN-GOOD ones do not:
///
/// - `discount` — a cosine of period six months, a harmonic of the yearly seasonality at
///   r = 0.999 against `yearly_delim_4`. Must WARN.
/// - `promo`, `weather` — the controls. Must NOT warn; a threshold that warned on everything
///   would prove nothing.
/// - `price` — labelled MARGINAL by the spike (r = 0.759, beta still 5x off). Deliberately
///   UNASSERTED: it is the column that proves a pairwise cutoff is the wrong test, not a
///   control for this threshold.
///
/// 10.0 is also the conventional VIF rule of thumb, which is why the fixture's separation
/// lands on a round number rather than on a fixture-specific one.
pub const REGRESSOR_VIF_WARN: f64 = 10.0;

/// WARN-above threshold for the DESIGN condition number (D-35). Not a cost ceiling either —
/// see [`REGRESSOR_VIF_WARN`].
///
/// The reported number is the condition number of the DESIGN, i.e. the square root of the
/// correlation matrix's, because the correlation matrix is the Gram of the scaled design.
/// That is the scale every published rule of thumb is stated on, and 30 is the conventional
/// one. Measured against the committed fixture in the 06.1-03 SUMMARY.
pub const REGRESSOR_CONDITION_NUMBER_WARN: f64 = 30.0;

/// Hard upper bound on `len(holidays[].name)` in BYTES.
///
/// **Why the existing bounds did not cover this.** They did not cover it at all: at the close
/// of plan 06-14 this was the ONE caller-settable field on the whole door surface with
/// literally NO enforcement (`door_surface.knobs`, `field: name`, `enforced_by: NOTHING`). It
/// was found by 06-14's enumeration, not by any review finding.
///
/// **The amplification.** `prophet::columns` turns ONE payload occurrence of the name into
/// TWO owned `String`s per design column — the column `name`
/// (`format!("{}_delim_{}{}", h.name, sign, off)`) and the `component`
/// (`h.name.clone()`) — and then makes it the key of `hcols.sort_by(|a, b|
/// a.name.cmp(&b.name))`, an `O(C log C)` comparison sort whose comparisons are byte-wise
/// over those names. `prophet::predict`'s component-name dedup compares them again,
/// `O(C * distinct_components)` times. At [`MAX_HOLIDAY_COLUMNS`] (1 000) that is 2 000
/// copies plus roughly 10 000 byte-wise comparisons per request, each LINEAR in the name
/// length — a ~2 000:1 amplification off a single occurrence in the payload. The name also
/// reaches the serialized reply as a key of the `components` map.
///
/// **Why a bound and not a measurement.** This axis has no structural maximum to measure
/// against, so no `measured_at_structural_maximum` disposition was ever available for it:
/// `crates/aprender-mcp-forecast`'s router construction (`http_app` / `pooled_app`) applies
/// no `DefaultBodyLimit`, no `max_body` and no content-length layer, and the stdio transport
/// has no framing cap, so any wall written here would describe an arbitrarily CHOSEN name
/// length rather than a maximum. That is why 06-14 shipped C-07 with
/// `no_structural_maximum: true` and held it open with a red test instead of measuring it.
///
/// **The derivation.** The longest holiday name in ANY committed fixture, example or test is
/// **9 bytes** — `superbowl`, from Prophet 1.4.0's own canonical `peyton_holidays` frame
/// (whose other label is `playoff`, 7 bytes). 200 is 22x that, two orders of magnitude past
/// any human label, and it bounds the worst case at 1 000 x 2 x 200 = ~400 KB of `String`
/// plus ~10 000 comparisons of <= 200 bytes. The same arithmetic on the UNBOUNDED field with
/// a 1 MB name — which nothing at HEAD refused — is ~2 GB of `String` from a single request.
///
/// **BYTES, not characters.** `String::len` is bytes, and bytes are what the clone and the
/// comparison actually cost. A rewrite to `chars().count()` would let a 3-byte-per-char UTF-8
/// name buy 3x the bounded work; [`crate::forecast`]'s test
/// `a_holiday_name_whose_char_count_fits_but_whose_byte_length_does_not_is_refused` is the
/// case that catches exactly that rewrite.
pub const MAX_HOLIDAY_NAME_LEN: usize = 200;

/// Hard upper bound on the NeuralProphet TRAINING work one request may buy, as the
/// door-computable proxy `lr_sweep_width * epochs * n_samples * (n_lags + 1)`
/// ([`crate::np::request_train_cost`]).
///
/// **Why the existing bounds did not cover this.** Nothing did. This path has no budget of
/// ANY kind: `fit::FIT_BUDGET_SECS` is read at exactly one place, inside `fit::fit_prophet`,
/// and the `"neuralprophet"` arm never enters that function. Every factor is individually
/// bounded — `n_lags <= 365`, `n_samples <= n_train_grid <= [`MAX_SPAN_DAYS`]`,
/// `epochs <= 500` — and their PRODUCT was not, with the door's learning-rate sweep
/// multiplying the whole thing by 2 (lags on) or 3 (lag-free). Note it is NOT that
/// `n_train_grid` is bounded by a larger constant than [`MAX_POINTS`]: `MAX_SPAN_DAYS ==
/// MAX_POINTS as i64`, so the two are numerically identical. The drivers are the missing
/// budget, the 2-3x sweep and `n_lags`.
///
/// **The measured breach.** The worst LEGAL request — 20 000 contiguous daily points (so the
/// span is also at `MAX_SPAN_DAYS` and `n_train_grid` is at its ceiling), `n_lags: 365`,
/// `horizon: 3650`, `freq: "D"` — prices at 718 641 000 and walls at **47.924 s** on a
/// release build, 24x over SC1's 2 s bar. Measured by `np::wall::np_train_wall` under
/// `NP_WALL_MODE=structural_max`.
///
/// **15 000 000 is the value, and the measurement chose it.** A first candidate of
/// 20 000 000 was REJECTED by its own wall: the long-history composition at 19 991 000
/// measured **2.089 s**, over the bar. At 15 000 000 three compositions differing in every
/// factor the proxy multiplies all clear it, on a release build, via `NP_WALL_MODE=at_bound_*`:
///
/// | composition | points | n_lags | horizon | train_cost | total_s |
/// |---|---|---|---|---|---|
/// | long history  | 20 000 |  6 | 3 650 | 13 995 800 | **1.642 s** |
/// | mid history   | 10 000 | 11 | 3 650 | 14 384 160 | **1.541 s** |
/// | short history |  2 000 | 41 |   365 | 14 810 040 | **1.402 s** |
///
/// **It is also the smallest round value that does not refuse the parity ladder's own
/// geometry.** `np::parity` validates correctness on Peyton Manning with `n_lags` 0 and 30;
/// through the door that geometry prices at 697 200 and **14 552 640** respectively, so
/// 14 000 000 would refuse the very request the ladder proves the model correct on.
/// `forecast::tests::the_np_parity_ladder_geometry_prices_under_the_train_cost_bound` pins that.
///
/// **The lag-free arm can never reach this bound.** Its structural maximum is
/// `3 * auto_epochs(20 000) * 20 000 * 1` = 3 000 000, measured at **0.508 s**. C-08 is
/// entirely about the lagged arm.
///
/// **Why the at-the-bound positive control lives in the `#[ignore]`d release harness rather
/// than in the always-run suite:** at the bound one request costs **45.619 s on a debug
/// profile** (measured, `at_bound_short_history`). Paying that in every `cargo test` run
/// would be a 60x regression on this crate's suite. The always-run controls are the
/// parity-geometry arithmetic above, a cheap accepted request through the door, and
/// `the_np_train_cost_bound_is_exclusive_not_inclusive`, which pins the comparison at the
/// boundary without paying for it.
pub const MAX_NP_TRAIN_COST: u64 = 15_000_000;

/// The event-column term of cost axis C-08, in PROXY WIDTH UNITS per event column
/// (`constants.fit_np_event_cost_per_column`, SC4 / D-32 / D-34).
///
/// # It is a COEFFICIENT, not a ceiling, and that distinction is checked
///
/// Nothing is ever refused for exceeding this number. It is multiplied into the priced
/// quantity, which is THEN compared against [`MAX_NP_TRAIN_COST`]. That is why it is
/// deliberately NOT named `fit_max_*`: `every_cost_ceiling_constant_is_named_by_an_axis`
/// enumerates cost CEILINGS, and capturing a coefficient there would make that test assert
/// that a coefficient is a bound and would require inventing an axis whose `bound:` no door
/// check ever compares against. The obligation a coefficient DOES carry — that some priced
/// formula reads it — is enforced by the sibling
/// `every_cost_coefficient_constant_is_named_by_an_axis_formula`.
///
/// # Where it sits in the formula, and why not beside the product
///
/// ```text
/// request_train_cost = n_lrs * epochs * n_samples
///                      * ( (n_lags + 1) + FIT_NP_EVENT_COST_PER_COLUMN * E )
/// ```
///
/// `(n_lags + 1)` is the PER-SAMPLE FEATURE WIDTH the optimiser sweeps, and the event block
/// adds work to that width — the measured growth is a PER-STEP slope, so the added work is
/// paid once per sample per epoch exactly like the width beside it. The alternative reading,
/// `product + c * E`, scales with NEITHER epochs NOR samples: it can be tuned to pass an
/// observation at one geometry and under-price every larger series, which is the failure
/// SC4's "observed closed, not asserted closed" exists to prevent.
/// `np::tests::the_event_cost_shape_scales_with_geometry` is the discriminator.
///
/// # What the number IS, dimensionally
///
/// The calibration sweep measures, at ONE named geometry, an intercept `a` (microseconds per
/// step at `E = 0`) and a slope `b` (microseconds per step per event column). The proxy's
/// width at `E = 0` is `(n_lags_cal + 1)` and corresponds to `a`, so one proxy width unit is
/// `a / (n_lags_cal + 1)` microseconds per step, and the coefficient is
///
/// ```text
/// FIT_NP_EVENT_COST_PER_COLUMN = b * (n_lags_cal + 1) / a     [rounded UP, then x safety]
/// ```
///
/// which reduces to `b / a` only when the calibration is LAG-FREE. `n_lags_cal` is recorded
/// in the C-08 `calibration:` mapping so that reduction is checkable rather than assumed.
/// The batch size divides `a` and `b` identically and cancels out of the ratio, which is
/// exactly why the coefficient is defined as a ratio of two measurements at one geometry and
/// never as an absolute microsecond figure — an absolute number would need a batch, a step
/// count and a sample count to convert, none of which the door knows at check time.
///
/// # The value in force, and its provenance
///
/// The measurement, the host, the architecture, the date, the commit, the method and the
/// safety factor with its reason all live in the C-08 `calibration:` mapping in
/// `contracts/forecast-tool-boundary-v1.yaml`, which is the single source. `constant_f64`
/// asserts this constant equals the YAML value in `cost_bounds_match_contract`. Phase 7
/// re-prices the CONSTANT per tier without re-deriving the SHAPE, which is what D-34 asked
/// for.
pub const FIT_NP_EVENT_COST_PER_COLUMN: f64 = 0.0126;

/// The NUMERIC-REGRESSOR term of cost axis C-08, in PROXY WIDTH UNITS per regressor column
/// (`constants.fit_np_regressor_cost_per_column`, SC4 / D-34).
///
/// # Why a SECOND coefficient and not a reuse of the event one
///
/// Plan 06.1-03's `fit_max_regressor_design_cost` bounds `(rows + horizon) x n_regressors` —
/// the DESIGN CELLS, built once. On the neuralprophet arm a regressor column is also read on
/// EVERY batch of EVERY epoch of EVERY learning rate in the sweep, exactly as an event column
/// is, and plan 06.1-05 priced event columns and only event columns. So a request AT the
/// regressor design-cost ceiling could buy training work cost axis C-08 never saw — the same
/// under-pricing that plan closed, one column family over.
///
/// A regressor column is a dense `f32` multiply-accumulate and an event column is a 0/1
/// indicator through the same `Linear`, so the two slopes are plausibly close. "Plausibly
/// close" is the kind of claim this project withdraws, so it is MEASURED on its own sweep
/// rather than inherited, and both measurements are recorded in the C-08 `calibration:`
/// mapping so the comparison is checkable rather than asserted.
///
/// # It is a COEFFICIENT, not a ceiling
///
/// Nothing is ever refused for exceeding this number; it is multiplied into the priced
/// quantity, which is THEN compared against [`MAX_NP_TRAIN_COST`]. That is why it is not
/// named `fit_max_*`. Its name contains `_cost_per_`, so it is captured by
/// `every_cost_coefficient_constant_is_named_by_an_axis_formula` and must NOT enter the
/// `fit_max_` ceiling enumeration.
///
/// # Where it sits in the formula
///
/// ```text
/// request_train_cost = n_lrs * epochs * n_samples
///                      * ( (n_lags + 1) + FIT_NP_EVENT_COST_PER_COLUMN * E
///                                       + FIT_NP_REGRESSOR_COST_PER_COLUMN * R )
/// ```
///
/// The SAME width term plan 06.1-05 pinned, for the same measured reason: the growth is a
/// PER-STEP slope, so the added work is paid once per sample per epoch. The rejected reading,
/// `product + c * R`, scales with neither epochs nor samples.
/// [`crate::np::tests::the_regressor_cost_shape_scales_with_geometry`] is the discriminator,
/// and it is a separate test from the event one because a shape argument that is only ever
/// checked on one of two terms is an argument about the other by analogy.
///
/// # What the number IS, dimensionally, and its provenance
///
/// Identical in form to the event coefficient: the sweep measures an intercept `a`
/// (microseconds per step at `R = 0`) and a slope `b` (microseconds per step per regressor
/// column) at ONE geometry, and
///
/// ```text
/// FIT_NP_REGRESSOR_COST_PER_COLUMN = b * (n_lags_cal + 1) / a   [rounded UP, then x safety]
/// ```
///
/// The measurement, the host, the architecture, the date, the commit, the method and the
/// safety factor with its reason live in the C-08 `calibration:` mapping in
/// `contracts/forecast-tool-boundary-v1.yaml`, which is the single source. `constant_f64`
/// asserts this constant equals the YAML value in `cost_bounds_match_contract`.
pub const FIT_NP_REGRESSOR_COST_PER_COLUMN: f64 = 0.0106;

/// Default router-pool size for the streamable-HTTP server (`constants.pool_default`).
///
/// Lives here, beside the other contract-mirrored bounds, because this is the crate that
/// owns the memoized contract reader — so the value is ASSERTED equal to the YAML by
/// `cost_bounds_match_contract` below. It previously sat in `aprender-mcp-forecast` as two
/// separate literal `8`s whose only "mirrors the contract" evidence was a doc comment and
/// a test that compared the constant to ITSELF.
pub const DEFAULT_POOL: usize = 8;

#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HolidayArg {
    /// Holiday name (becomes a component). At most 200 bytes.
    pub name: String,
    /// Dates the holiday occurs on, YYYY-MM-DD (past AND future occurrences).
    pub dates: Vec<String>,
    /// Days before the date to include (≤ 0). Default 0.
    #[serde(default)]
    pub lower_window: i64,
    /// Days after the date to include (≥ 0). Default 0.
    #[serde(default)]
    pub upper_window: i64,
}

/// One external regressor: a caller-supplied covariate spliced onto the design matrix as a
/// trailing column (D-22).
///
/// # Why this field set, and why it is not revisited cheaply
///
/// `ForecastArgs` IS the published MCP tool schema — `schemars::schema_for!` derives it at
/// `aprender-mcp-forecast`'s tool registration — and `deny_unknown_fields` makes a renamed
/// field a hard transport-layer refusal for every existing caller, with no migration path.
/// So the shape was frozen deliberately before any of it was written:
///
/// - It matches the parity-proven spike 011 `RegressorSpec`, the prototype that actually
///   produced the committed Python Prophet 1.4.0 oracle.
/// - It matches [`HolidayArg`]'s conventions in this same file: `name` first, optionals
///   carrying `#[serde(default)]`, and an ARRAY of objects rather than a map.
/// - `standardize` is `Option<bool>` with `None` meaning Prophet's "auto". The oracle
///   records the literal string `"auto"`, but that is its record of the PYTHON call, not a
///   requirement on this wire; all four oracle regressors are `"auto"`, so nothing is lost.
///
/// # Rejected alternative: a map keyed by regressor name
///
/// Recorded here rather than only in the plan, so it is not re-proposed from the plan text.
/// A map would make duplicate names structurally impossible, which is attractive — but
/// `serde_json::Map` is a `BTreeMap`, so iteration is KEY-SORTED. The oracle's trailing
/// four design columns are `promo, price, discount, weather`, which is INSERTION order;
/// sorted order would be `discount, price, promo, weather`. A map-keyed wire shape would
/// hand `splice` the wrong column order and break the 24-column parity assertion outright.
/// Insertion order is load-bearing, so the array stays.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RegressorArg {
    /// Regressor name. Becomes a response component key, so it must be unique within one
    /// request. At most 200 bytes.
    pub name: String,
    /// One value per HISTORY row AND per HORIZON row, in `ds` order then future order — so
    /// its length is exactly `ds.len() + horizon`. A future value is required because the
    /// model needs the covariate over the period it is forecasting.
    ///
    /// CAUTION on `model: "neuralprophet"` with `n_lags > 0`: a continuous driver competes
    /// with the AR term, which can ABSORB its contribution, so the fitted coefficient is not
    /// a clean effect estimate. Measured at `n_lags = 7`: the `price` weight collapsed to
    /// -0.0048 from -0.0493 lag-free — illustrative of the MAGNITUDE, explicitly NOT a
    /// bound (one series, one gap pattern). No VIF or condition number is computed on that
    /// arm: AR absorption is a training dynamic, not column collinearity. Read coefficients
    /// off the `prophet` arm, which publishes `diagnostics.regressors`, or set `n_lags = 0`.
    pub values: Vec<f64>,
    /// "additive" (default) or "multiplicative". Multiplicative is Prophet-only.
    #[serde(default)]
    pub mode: Option<String>,
    /// Prior scale for this regressor's coefficient. Defaults to 10.0, which is Prophet's
    /// `holidays_prior_scale`.
    #[serde(default)]
    pub prior_scale: Option<f64>,
    /// Whether to standardise the column. Absent means Prophet's "auto" rule: standardise
    /// unless the history values are exactly the two-element set {0, 1}.
    #[serde(default)]
    pub standardize: Option<bool>,
}

#[derive(Debug, Clone, Default, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ForecastArgs {
    /// Timestamps, YYYY-MM-DD (a time part is ignored), ascending, unique.
    pub ds: Vec<String>,
    /// Observed values, same length as `ds`.
    pub y: Vec<f64>,
    /// Number of future periods to forecast (1 … 3650).
    pub horizon: usize,
    /// Period of the future steps: "D" (default), "W", or "MS" (month start).
    #[serde(default)]
    pub freq: Option<String>,
    /// "prophet" (default) or "neuralprophet".
    #[serde(default)]
    pub model: Option<String>,
    /// Prophet growth: "linear" (default), "logistic" (needs `cap`) or "flat".
    #[serde(default)]
    pub growth: Option<String>,
    /// Carrying capacity for logistic growth (original units).
    #[serde(default)]
    pub cap: Option<f64>,
    /// "additive" (default) or "multiplicative" seasonality.
    #[serde(default)]
    pub seasonality_mode: Option<String>,
    /// Width of the uncertainty band, default 0.8.
    #[serde(default)]
    pub interval_width: Option<f64>,
    /// Holidays / events with optional windows. Accepted on BOTH models (D-29): on
    /// `neuralprophet` they require `n_lags = 0` and are published as per-event
    /// response components plus a `holidays` roll-up. A name that is already a
    /// reserved response key (`trend`, `holidays`, `additive_terms`, ...) is refused.
    #[serde(default)]
    pub holidays: Option<Vec<HolidayArg>>,
    /// NeuralProphet only: number of autoregressive lags (0 = trend + seasonality only).
    ///
    /// CAUTION with `regressors`: at `n_lags > 0` the AR term competes with a continuous
    /// driver and can ABSORB it, so a regressor coefficient on this arm is not a clean
    /// effect estimate. Measured at `n_lags = 7`: the `price` weight collapsed to -0.0048
    /// from -0.0493 lag-free — illustrative of the MAGNITUDE, explicitly NOT a bound. No
    /// identifiability number is computed here (AR absorption is a training dynamic, not
    /// column collinearity, so a green VIF would reassure about the wrong thing). `holidays`
    /// require `n_lags = 0`, and a regressor on a GAPPY series is refused at `n_lags > 0`.
    #[serde(default)]
    pub n_lags: Option<usize>,
    /// Random seed for the uncertainty simulation / training (default 42).
    #[serde(default)]
    pub seed: Option<u64>,
    /// External regressors (covariates) spliced onto the design as trailing columns.
    /// Accepted on BOTH models (D-26/D-28). On `prophet` each one is published as its
    /// own response component with an identifiability diagnostic; on `neuralprophet`
    /// they enter the additive regressor block. `neuralprophet` requires freq D.
    #[serde(default)]
    pub regressors: Option<Vec<RegressorArg>>,
}

#[derive(Debug, Serialize)]
pub struct ForecastResponse {
    pub model: String,
    pub freq: String,
    pub n_history: usize,
    pub fit_seconds: f64,
    pub predict_seconds: f64,
    pub ds: Vec<String>,
    pub yhat: Vec<f64>,
    pub yhat_lower: Vec<f64>,
    pub yhat_upper: Vec<f64>,
    pub trend: Vec<f64>,
    pub components: serde_json::Map<String, serde_json::Value>,
    pub diagnostics: serde_json::Value,
}

/// Every refusal the door can produce. `Validation` is the caller's fault and maps to
/// `pmcp::Error::validation`; `Internal` maps to `pmcp::Error::internal`.
#[derive(Debug)]
pub enum ForecastError {
    Validation(String),
    Internal(String),
}

impl std::fmt::Display for ForecastError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Validation(s) | Self::Internal(s) => f.write_str(s),
        }
    }
}

impl std::error::Error for ForecastError {}

#[cfg(test)]
mod tests {
    use super::{
        ForecastArgs, ForecastError, DEFAULT_POOL, FIT_NP_EVENT_COST_PER_COLUMN,
        FIT_NP_REGRESSOR_COST_PER_COLUMN, MAX_HOLIDAY_COLUMNS, MAX_HOLIDAY_DATES,
        MAX_HOLIDAY_DATES_TOTAL, MAX_HOLIDAY_DESIGN_COST, MAX_HOLIDAY_NAME_LEN, MAX_HOLIDAY_WINDOW,
        MAX_HORIZON, MAX_LOGISTIC_CHANGEPOINT_LAMBDA, MAX_NP_TRAIN_COST, MAX_POINTS,
        MAX_REGRESSORS, MAX_REGRESSOR_DESIGN_COST, MAX_SPAN_DAYS, MIN_POINTS,
        REGRESSOR_CONDITION_NUMBER_WARN, REGRESSOR_PRIOR_SCALE_MAX, REGRESSOR_PRIOR_SCALE_MIN,
        REGRESSOR_VIF_WARN,
    };
    use crate::test_support::{constant_f64, constant_u64};

    /// The fit server's three bounds are EQUAL to the contract, not merely similar.
    ///
    /// D-15: a bound written twice can be loosened in one place. The contract is the
    /// source and this test is what makes the Rust constant a mirror of it —
    /// `constants.fit_max_horizon: 3651` in the YAML alone turns this red.
    #[test]
    fn bounds_match_contract() {
        assert_eq!(
            MIN_POINTS as u64,
            constant_u64("forecast-tool-boundary-v1", "fit_min_points"),
            "types::MIN_POINTS must equal constants.fit_min_points in forecast-tool-boundary-v1"
        );
        assert_eq!(
            MAX_POINTS as u64,
            constant_u64("forecast-tool-boundary-v1", "fit_max_points"),
            "types::MAX_POINTS must equal constants.fit_max_points in forecast-tool-boundary-v1"
        );
        assert_eq!(
            MAX_HORIZON as u64,
            constant_u64("forecast-tool-boundary-v1", "fit_max_horizon"),
            "types::MAX_HORIZON must equal constants.fit_max_horizon in forecast-tool-boundary-v1"
        );
    }

    /// The PER-REQUEST COST CEILINGS against the SAME contract, for the SAME reason.
    ///
    /// `fit_max_points` bounds how many points arrive; every bound in this table bounds the
    /// WORK each one can buy. They are the bounds a request can be inside all three headline
    /// limits and still blow past — a span-based daily grid, a holiday window that is a
    /// design-column multiplier, and a future-span-over-history-span ratio that sizes the
    /// logistic uncertainty simulation — so they are contract-owned exactly like the headline
    /// three.
    ///
    /// IN-03: this doc used to state a COUNT of rows, which drifted the moment a row was
    /// added, and the table used to carry `DEFAULT_POOL` — a router-pool size, which is a
    /// deployment default and not a per-request cost ceiling at all. That assertion now lives
    /// in [`pool_default_matches_contract`], so this table describes one kind of thing.
    #[test]
    fn cost_bounds_match_contract() {
        // The REAL-valued ceilings. `constant_u64` cannot express a threshold that is
        // conceptually a real number, and rounding one to fit would make the mirror assert
        // something weaker than the constant it mirrors.
        for (name, key, value) in [
            (
                "MAX_LOGISTIC_CHANGEPOINT_LAMBDA",
                "fit_max_logistic_changepoint_lambda",
                MAX_LOGISTIC_CHANGEPOINT_LAMBDA,
            ),
            (
                "prophet::POISSON_NORMAL_BRANCH_LAMBDA",
                "poisson_normal_branch_lambda",
                crate::prophet::POISSON_NORMAL_BRANCH_LAMBDA,
            ),
            // A per-column VIF and a condition number are conceptually REAL, so
            // `constant_f64` is the right reader: rounding either to fit `constant_u64`
            // would make the mirror assert something weaker than the constant it mirrors.
            (
                "REGRESSOR_VIF_WARN",
                "regressor_vif_warn",
                REGRESSOR_VIF_WARN,
            ),
            (
                "REGRESSOR_CONDITION_NUMBER_WARN",
                "regressor_condition_number_warn",
                REGRESSOR_CONDITION_NUMBER_WARN,
            ),
            // A per-column cost SLOPE is conceptually real for the same reason: rounding it
            // to fit `constant_u64` would round 0.0126 to 0 and make the mirror assert
            // nothing at all. It is a COEFFICIENT and not a ceiling — see
            // `every_cost_coefficient_constant_is_named_by_an_axis_formula` below for the
            // obligation it does carry.
            (
                "FIT_NP_EVENT_COST_PER_COLUMN",
                "fit_np_event_cost_per_column",
                FIT_NP_EVENT_COST_PER_COLUMN,
            ),
            // The SECOND cost coefficient, added by plan 06.1-07 when numeric regressors
            // were opened on the neuralprophet arm. Separately MEASURED, not inherited:
            // the regressor basis came in 15.3 per cent below the event basis on the same
            // host and geometry.
            (
                "FIT_NP_REGRESSOR_COST_PER_COLUMN",
                "fit_np_regressor_cost_per_column",
                FIT_NP_REGRESSOR_COST_PER_COLUMN,
            ),
        ] {
            let from_contract = constant_f64("forecast-tool-boundary-v1", key);
            assert!(
                (value - from_contract).abs() < f64::EPSILON,
                "types::{name} ({value}) must equal constants.{key} ({from_contract}) in \
                 forecast-tool-boundary-v1"
            );
        }
        // The two prior-scale domain bounds are compared BIT FOR BIT, not within
        // `f64::EPSILON`. An absolute epsilon of 2.2e-16 is meaningless beside 1e150 (it
        // would accept anything) and impossibly strict beside 1e-153 (it would accept
        // anything too, for the opposite reason). Both sides are written as the same
        // decimal literal, so the only honest mirror test is exact equality.
        for (name, key, value) in [
            (
                "REGRESSOR_PRIOR_SCALE_MIN",
                "regressor_prior_scale_min",
                REGRESSOR_PRIOR_SCALE_MIN,
            ),
            (
                "REGRESSOR_PRIOR_SCALE_MAX",
                "regressor_prior_scale_max",
                REGRESSOR_PRIOR_SCALE_MAX,
            ),
        ] {
            let from_contract = constant_f64("forecast-tool-boundary-v1", key);
            assert_eq!(
                value.to_bits(),
                from_contract.to_bits(),
                "types::{name} ({value:e}) must equal constants.{key} ({from_contract:e}) \
                 in forecast-tool-boundary-v1, bit for bit"
            );
        }
        // The DERIVATION, re-checked rather than quoted: these bounds exist because the
        // objective squares the prior scale into a denominator. If either fact below
        // stopped holding, the bounds would be protecting against nothing.
        assert_eq!(
            f64::MIN_POSITIVE * f64::MIN_POSITIVE,
            0.0,
            "the whole reason for a floor: MIN_POSITIVE squares to exactly zero"
        );
        assert!(
            REGRESSOR_PRIOR_SCALE_MIN * REGRESSOR_PRIOR_SCALE_MIN >= f64::MIN_POSITIVE,
            "the floor's square must still be NORMAL, or the floor does not do its job"
        );
        assert!(
            (REGRESSOR_PRIOR_SCALE_MAX * REGRESSOR_PRIOR_SCALE_MAX).is_finite(),
            "the ceiling's square must stay finite"
        );

        for (name, key, value) in [
            ("MAX_SPAN_DAYS", "fit_max_span_days", MAX_SPAN_DAYS as u64),
            (
                "MAX_HOLIDAY_WINDOW",
                "fit_max_holiday_window",
                MAX_HOLIDAY_WINDOW as u64,
            ),
            (
                "MAX_HOLIDAY_COLUMNS",
                "fit_max_holiday_columns",
                MAX_HOLIDAY_COLUMNS as u64,
            ),
            (
                "MAX_HOLIDAY_DATES",
                "fit_max_holiday_dates",
                MAX_HOLIDAY_DATES as u64,
            ),
            (
                "MAX_HOLIDAY_DESIGN_COST",
                "fit_max_holiday_design_cost",
                MAX_HOLIDAY_DESIGN_COST as u64,
            ),
            (
                "MAX_HOLIDAY_DATES_TOTAL",
                "fit_max_holiday_dates_total",
                MAX_HOLIDAY_DATES_TOTAL as u64,
            ),
            (
                "MAX_HOLIDAY_NAME_LEN",
                "fit_max_holiday_name_len",
                MAX_HOLIDAY_NAME_LEN as u64,
            ),
            (
                "MAX_NP_TRAIN_COST",
                "fit_max_np_train_cost",
                MAX_NP_TRAIN_COST,
            ),
            (
                "MAX_REGRESSORS",
                "fit_max_regressors",
                MAX_REGRESSORS as u64,
            ),
            (
                "MAX_REGRESSOR_DESIGN_COST",
                "fit_max_regressor_design_cost",
                MAX_REGRESSOR_DESIGN_COST as u64,
            ),
        ] {
            assert_eq!(
                value,
                constant_u64("forecast-tool-boundary-v1", key),
                "types::{name} must equal constants.{key} in forecast-tool-boundary-v1"
            );
        }
    }

    /// The router-pool DEFAULT against the contract — deliberately its own test (IN-03).
    ///
    /// A router-pool size is a DEPLOYMENT default: it says how many independent pmcp routers
    /// the streamable-HTTP server stands up, which is a throughput decision. It is not a
    /// per-request cost ceiling, and riding along in [`cost_bounds_match_contract`]'s table
    /// made that table's own doc comment wrong about what it contained. It lives in this
    /// crate only because this crate owns the memoized contract reader.
    #[test]
    fn pool_default_matches_contract() {
        assert_eq!(
            DEFAULT_POOL as u64,
            constant_u64("forecast-tool-boundary-v1", "pool_default"),
            "types::DEFAULT_POOL must equal constants.pool_default in forecast-tool-boundary-v1"
        );
    }

    /// The Chronos door's three bounds against the SAME contract.
    ///
    /// Unconditional: plan 06-05 is a declared dependency of 06-06, so `crate::chronos`
    /// exists. `chronos_native_horizon` (64) is a property of the WEIGHTS, not of this
    /// crate, and is asserted by 06-07 against the shipped config fixture.
    #[test]
    fn chronos_bounds_match_contract() {
        assert_eq!(
            crate::chronos::CHRONOS_MIN_POINTS as u64,
            constant_u64("forecast-tool-boundary-v1", "chronos_min_points"),
            "chronos::CHRONOS_MIN_POINTS must equal constants.chronos_min_points in forecast-tool-boundary-v1"
        );
        assert_eq!(
            crate::chronos::CHRONOS_MAX_POINTS as u64,
            constant_u64("forecast-tool-boundary-v1", "chronos_max_points"),
            "chronos::CHRONOS_MAX_POINTS must equal constants.chronos_max_points in forecast-tool-boundary-v1"
        );
        assert_eq!(
            crate::chronos::CHRONOS_MAX_HORIZON as u64,
            constant_u64("forecast-tool-boundary-v1", "chronos_max_horizon"),
            "chronos::CHRONOS_MAX_HORIZON must equal constants.chronos_max_horizon in forecast-tool-boundary-v1"
        );
    }

    // ------------------------------------------------- the door-surface class invariant ---
    //
    // Three rounds of gap closure each fixed exactly the ONE measured probe and the next
    // adversarial pass found the next instance of the same class. `door_surface:` in
    // forecast-tool-boundary-v1.yaml is the structural answer, and these FOUR tests are what
    // make it a CHECKED claim rather than a document that goes stale. An enumeration nobody
    // verifies is the aspirational invariant this round exists to replace.
    //
    // WHAT IS ACTUALLY CHECKED, stated narrowly on purpose (T-06-32, 06-SECURITY.md). The
    // security audit found this comment claiming more than the tests did, so:
    //
    //   1. KNOBS, both directions, DERIVED — set equality against `schemars::schema_for!` for
    //      all THREE argument structs, across BOTH doors (`ForecastArgs` + `HolidayArg` for
    //      the forecast tool, `ChronosArgs` for the Chronos tool). Until 2026-09-07 the second
    //      door was absent entirely while this file claimed to cover the whole surface.
    //   2. CEILINGS -> AXES, DERIVED — every `fit_max_*` / `chronos_max_*` key in `constants:`
    //      must be named by a cost axis or carry a checked `ceilings_subsumed` exemption.
    //      The ceiling list comes from the mapping itself, so a new ceiling cannot be added
    //      without something reasoning about the work it bounds.
    //   3. AXES -> CEILINGS — every listed axis names a real constant (or a measured
    //      disposition carrying its measurement).
    //
    // THE RESIDUAL, which is real and is NOT closed by the above: an expensive path added
    // inside `prophet.rs`/`np.rs` that introduces NO constant and NO axis entry is invisible
    // to all four tests. That is precisely C-06's shape — and note that C-06 would NOT have
    // been caught by the knobs half either, since `growth`, `freq`, `horizon` and `ds` all had
    // knobs entries the whole time it was open. The detector for that residual is not a test
    // here but `just forecast-sc1-sweep` (wired into `make tier3`), which measures the wall
    // rather than the enumeration. Do not read these four tests as covering it.

    /// The `field` + `owner` pairs of every `door_surface.knobs` entry.
    fn enumerated_knobs() -> std::collections::BTreeSet<(String, String)> {
        let doc = crate::test_support::contract_value("forecast-tool-boundary-v1");
        doc.get("door_surface")
            .and_then(|d| d.get("knobs"))
            .and_then(serde_yaml::Value::as_sequence)
            .expect("forecast-tool-boundary-v1 must carry door_surface.knobs")
            .iter()
            .map(|k| {
                let get = |key: &str| {
                    k.get(key)
                        .and_then(serde_yaml::Value::as_str)
                        .unwrap_or_else(|| panic!("every door_surface.knobs entry needs {key}"))
                        .to_string()
                };
                (get("owner"), get("field"))
            })
            .collect()
    }

    /// The `(owner, field)` pairs the ADVERTISED schema actually exposes.
    fn schema_knobs() -> std::collections::BTreeSet<(String, String)> {
        let mut out = std::collections::BTreeSet::new();
        for (owner, schema) in [
            (
                "ForecastArgs",
                serde_json::to_value(schemars::schema_for!(ForecastArgs)),
            ),
            (
                "HolidayArg",
                serde_json::to_value(schemars::schema_for!(super::HolidayArg)),
            ),
            // T-06-32 (06-SECURITY.md): the enumeration covered ONE of the two doors.
            // `ChronosArgs` is a second live, unauthenticated `tools/call` surface
            // (`aprender-mcp-chronos`), and its five caller-settable fields had no knobs
            // entry, no `enforced_by` and no cost axis — the exact condition that let C-07
            // through — while this contract's header called itself "THE DOOR'S WHOLE SURFACE".
            (
                "ChronosArgs",
                serde_json::to_value(schemars::schema_for!(crate::chronos::ChronosArgs)),
            ),
            // D-20: without this owner the five `RegressorArg` fields are invisible to the
            // MISSING direction below — they would be caller-settable, unenumerated, and
            // the set-equality test would still be green. An owner that is not listed here
            // is not "not yet covered", it is a hole shaped exactly like the one T-06-32
            // found in `ChronosArgs`.
            (
                "RegressorArg",
                serde_json::to_value(schemars::schema_for!(super::RegressorArg)),
            ),
        ] {
            let schema = schema.expect("schema serializes");
            let props = schema["properties"]
                .as_object()
                .unwrap_or_else(|| panic!("{owner} schema must publish properties"));
            for field in props.keys() {
                out.insert((owner.to_string(), field.clone()));
            }
        }
        out
    }

    /// Every caller-settable field reachable through `forecast` is enumerated, and every
    /// enumerated field still exists — BOTH directions.
    ///
    /// The field list is derived from `schemars::schema_for!`, the same generator that
    /// produces the advertised MCP tool schema, so it CANNOT fall behind the struct the way a
    /// hand-kept copy would. Set equality gives both directions structurally:
    ///
    /// - MISSING: a field added to `ForecastArgs` or `HolidayArg` with no `knobs` entry is a
    ///   knob nothing has reasoned about — which is how CR-01's cost axis got in.
    /// - PHANTOM: an entry for a field that no longer exists on either struct is how an
    ///   enumeration silently stops describing the code while still looking complete.
    ///
    /// Both directions were OBSERVED RED by mutation before this test was trusted; see the
    /// 06-14 SUMMARY for the recorded output.
    #[test]
    fn every_request_knob_is_enumerated() {
        let enumerated = enumerated_knobs();
        let actual = schema_knobs();
        let missing: Vec<_> = actual.difference(&enumerated).collect();
        let phantom: Vec<_> = enumerated.difference(&actual).collect();
        assert!(
            missing.is_empty() && phantom.is_empty(),
            "door_surface.knobs must describe exactly the caller-settable surface.\n  \
             MISSING from the contract (a field exists with no entry — nothing has reasoned \
             about its cost or its enforcement): {missing:?}\n  \
             PHANTOM in the contract (an entry names a field that exists on neither struct — \
             the enumeration has stopped describing the code): {phantom:?}"
        );
    }

    /// The `cost_axes` sequence, as `(axis, bound, measured_seconds)`.
    fn enumerated_axes() -> Vec<(String, String, Option<f64>)> {
        let doc = crate::test_support::contract_value("forecast-tool-boundary-v1");
        doc.get("door_surface")
            .and_then(|d| d.get("cost_axes"))
            .and_then(serde_yaml::Value::as_sequence)
            .expect("forecast-tool-boundary-v1 must carry door_surface.cost_axes")
            .iter()
            .map(|a| {
                let get = |key: &str| {
                    a.get(key)
                        .and_then(serde_yaml::Value::as_str)
                        .unwrap_or_else(|| panic!("every door_surface.cost_axes entry needs {key}"))
                        .to_string()
                };
                (
                    get("axis"),
                    get("bound"),
                    a.get("measured_seconds")
                        .and_then(serde_yaml::Value::as_f64),
                )
            })
            .collect()
    }

    /// The pending marker prefix. An axis carrying it is a KNOWN-OPEN axis, held open by
    /// [`no_cost_axis_is_pending`] rather than by a note nobody runs.
    const PENDING_MARKER_PREFIX: &str = "unbounded_pending_";

    /// The INVERSE direction of [`every_cost_axis_names_a_real_bound`], and the half T-06-32
    /// found missing: every cost CEILING declared in `constants:` must be named by at least one
    /// cost axis, or be explicitly recorded as subsumed by one.
    ///
    /// Why this direction is the one that matters. `every_cost_axis_names_a_real_bound` iterates
    /// `door_surface.cost_axes` and can therefore only ever check axes that are LISTED. An axis
    /// nobody wrote down is invisible to it — which is exactly how C-06, the axis behind the
    /// CRITICAL T-06-31, got in: `growth`, `freq`, `horizon` and `ds` all carried knobs entries
    /// the whole time, so the knobs half was green throughout. The contract's own claim that
    /// "MISSING (a field exists with no entry) is how CR-01's cost axis got in" describes a
    /// mechanism that did not fire.
    ///
    /// This test is derivable rather than hand-kept: the ceiling list comes from the
    /// `constants:` mapping itself (`fit_max_*` / `chronos_max_*`), so a new ceiling added
    /// there with no axis naming it turns this red without anyone remembering to update a list.
    ///
    /// `ceilings_subsumed` is the escape hatch, and it is checked in BOTH directions so it
    /// cannot rot into a licence: an entry naming a constant that does not exist is a phantom,
    /// and an entry whose `subsumed_by` is not itself a real axis bound is a dangling claim.
    #[test]
    fn every_cost_ceiling_constant_is_named_by_an_axis() {
        let doc = crate::test_support::contract_value("forecast-tool-boundary-v1");
        let constants = doc
            .get("constants")
            .and_then(serde_yaml::Value::as_mapping)
            .expect("forecast-tool-boundary-v1 must carry constants");

        let ceilings: std::collections::BTreeSet<String> = constants
            .keys()
            .filter_map(serde_yaml::Value::as_str)
            .filter(|k| k.starts_with("fit_max_") || k.starts_with("chronos_max_"))
            .map(str::to_string)
            .collect();
        // MEASURED 2026-09-21, not quoted: the contract carried 13 ceilings before plan
        // 06.1-03 (`chronos_max_horizon`, `chronos_max_points` and eleven `fit_max_*`) and
        // this plan adds exactly two — `fit_max_regressors` and
        // `fit_max_regressor_design_cost`. The floor moves with the file, because a floor
        // left at 10 would have passed a plan that added nothing.
        assert!(
            ceilings.len() >= 15,
            "vacuity guard: only {} ceiling constants found — the filter has stopped matching \
             and this test would pass by checking nothing",
            ceilings.len()
        );

        let axis_bounds: std::collections::BTreeSet<String> =
            enumerated_axes().into_iter().map(|(_, b, _)| b).collect();

        let subsumed: Vec<(String, String)> = doc
            .get("door_surface")
            .and_then(|d| d.get("ceilings_subsumed"))
            .and_then(serde_yaml::Value::as_sequence)
            .map(|seq| {
                seq.iter()
                    .map(|e| {
                        let get = |k: &str| {
                            e.get(k)
                                .and_then(serde_yaml::Value::as_str)
                                .unwrap_or_else(|| {
                                    panic!("every ceilings_subsumed entry needs {k}")
                                })
                                .to_string()
                        };
                        (get("constant"), get("subsumed_by"))
                    })
                    .collect()
            })
            .unwrap_or_default();

        // PHANTOM, direction 1: an exemption for a constant that no longer exists.
        let phantom: Vec<&String> = subsumed
            .iter()
            .map(|(c, _)| c)
            .filter(|c| !ceilings.contains(*c))
            .collect();
        assert!(
            phantom.is_empty(),
            "ceilings_subsumed names constants that are not ceilings in this file — the \
             exemption has stopped describing the contract: {phantom:?}"
        );
        // PHANTOM, direction 2: an exemption pointing at an axis bound nobody asserts.
        let dangling: Vec<&(String, String)> = subsumed
            .iter()
            .filter(|(_, by)| !axis_bounds.contains(by))
            .collect();
        assert!(
            dangling.is_empty(),
            "ceilings_subsumed claims a constant is covered by a bound that no cost axis \
             names — the exemption is a dangling claim: {dangling:?}"
        );

        let exempt: std::collections::BTreeSet<&String> = subsumed.iter().map(|(c, _)| c).collect();
        let unreferenced: Vec<&String> = ceilings
            .iter()
            .filter(|c| !axis_bounds.contains(*c) && !exempt.contains(c))
            .collect();
        assert!(
            unreferenced.is_empty(),
            "cost CEILINGS declared in constants: that NO cost axis names and that carry no \
             ceilings_subsumed entry — nothing has reasoned about what work they bound, which \
             is the T-06-32 condition: {unreferenced:?}"
        );
    }

    /// The SIBLING of [`every_cost_ceiling_constant_is_named_by_an_axis`], for the other
    /// kind of number in the pricing path: every cost COEFFICIENT declared in `constants:`
    /// must be READ by some cost axis's `formula:`.
    ///
    /// # Why this is a second test and not a widening of the ceiling filter
    ///
    /// The ceiling filter (`fit_max_*` / `chronos_max_*`) is CORRECT as it stands, and
    /// widening it to catch `fit_np_event_cost_per_column` would be a defect rather than a
    /// fix. That test's contract is "every cost CEILING is named by at least one cost axis",
    /// and a ceiling is a value a request is REFUSED for exceeding.
    /// [`FIT_NP_EVENT_COST_PER_COLUMN`] is a COEFFICIENT: it is multiplied into the priced
    /// quantity, which is THEN compared against [`MAX_NP_TRAIN_COST`]. Nothing is ever
    /// refused for exceeding it. Capturing it under the ceiling filter would make that test
    /// assert that a coefficient is a bound, and would require inventing an axis whose
    /// `bound:` is a number no door check compares against — a dangling claim in exactly the
    /// direction [`every_cost_axis_names_a_real_bound`] exists to refuse.
    ///
    /// # The hole is real, and it is a DIFFERENT hole
    ///
    /// The enumeration's PURPOSE is that a new number in the pricing path cannot be added
    /// without something reasoning about it. A coefficient added to `constants:` with nothing
    /// naming it is exactly that failure, one category over. So the obligation asserted here
    /// is the coefficient's correct one: to be READ by a priced formula, rather than to bound
    /// anything.
    ///
    /// # What stays OPEN, recorded rather than claimed closed
    ///
    /// The complete closure is the general form: every key in `constants:` falls into exactly
    /// one named category (ceiling, coefficient, deployment default, warning threshold) and
    /// each category carries its own obligation. That would immediately require categorising
    /// `pool_default`, `poisson_normal_branch_lambda`, `chronos_native_horizon`,
    /// `fit_min_points`, `default_seed`, `default_interval_width_x100` and the two regressor
    /// warning thresholds. It is NOT built here, and this narrow enumeration must not be read
    /// as covering it. The same follow-up is recorded in the contract beside the constant.
    #[test]
    fn every_cost_coefficient_constant_is_named_by_an_axis_formula() {
        let doc = crate::test_support::contract_value("forecast-tool-boundary-v1");
        let constants = doc
            .get("constants")
            .and_then(serde_yaml::Value::as_mapping)
            .expect("forecast-tool-boundary-v1 must carry constants");
        let coefficients: std::collections::BTreeSet<String> = constants
            .keys()
            .filter_map(serde_yaml::Value::as_str)
            .filter(|k| k.contains("_cost_per_"))
            .map(str::to_string)
            .collect();
        // VACUITY GUARD. A filter that has silently stopped matching passes this test by
        // checking nothing, which is the failure mode the whole door-surface class of tests
        // exists to refuse. MEASURED 2026-09-21: exactly one `_cost_per_` constant exists,
        // `fit_np_event_cost_per_column`, added by plan 06.1-05.
        assert!(
            !coefficients.is_empty(),
            "vacuity guard: no `_cost_per_` coefficient constants found — the filter has \
             stopped matching and this test would pass by checking nothing"
        );

        let formulas: Vec<String> = doc
            .get("door_surface")
            .and_then(|d| d.get("cost_axes"))
            .and_then(serde_yaml::Value::as_sequence)
            .expect("forecast-tool-boundary-v1 must carry door_surface.cost_axes")
            .iter()
            .filter_map(|a| a.get("formula").and_then(serde_yaml::Value::as_str))
            .map(str::to_string)
            .collect();
        assert!(
            !formulas.is_empty(),
            "vacuity guard: no cost axis carries a formula, so every coefficient below \
             would be reported unreferenced for the wrong reason"
        );

        let unread: Vec<&String> = coefficients
            .iter()
            .filter(|k| !formulas.iter().any(|f| f.contains(k.as_str())))
            .collect();
        assert!(
            unread.is_empty(),
            "cost COEFFICIENTS declared in constants: that NO cost axis formula reads — a \
             number was added to the pricing path with nothing reasoning about it, which is \
             the T-06-32 condition one category over from a ceiling: {unread:?}"
        );
    }

    /// Every NON-pending cost axis names a bound that actually exists.
    ///
    /// An axis whose `bound:` names a `constants:` key that is not there is worse than an
    /// unbounded axis: it LOOKS closed. The alternative disposition,
    /// `measured_at_structural_maximum`, is accepted only with a `measured_seconds` strictly
    /// under SC1's 2 s bar — a claim that an axis is safe at its maximum has to carry the
    /// measurement that says so.
    ///
    /// Pending entries are OUT OF SCOPE here on purpose: this test is the one that catches a
    /// phantom bound key, and it must stay GREEN so a real regression in that direction is
    /// visible. The known-open axes are a separate, single-purpose test.
    #[test]
    fn every_cost_axis_names_a_real_bound() {
        let doc = crate::test_support::contract_value("forecast-tool-boundary-v1");
        let constants = doc
            .get("constants")
            .and_then(serde_yaml::Value::as_mapping)
            .expect("forecast-tool-boundary-v1 must carry constants");
        for (axis, bound, measured) in enumerated_axes() {
            if bound.starts_with(PENDING_MARKER_PREFIX) {
                continue;
            }
            if bound == "measured_at_structural_maximum" {
                let seconds = measured.unwrap_or_else(|| {
                    panic!(
                        "cost axis {axis} claims measured_at_structural_maximum but carries no \
                         numeric measured_seconds — an unmeasured measurement is not a bound"
                    )
                });
                assert!(
                    seconds < 2.0,
                    "cost axis {axis} is measured at {seconds} s at its structural maximum, \
                     which is at or over SC1's 2 s bar — that is not a closed axis"
                );
                continue;
            }
            assert!(
                constants.contains_key(serde_yaml::Value::String(bound.clone())),
                "cost axis {axis} names bound {bound:?}, which is not a key in constants: — \
                 the axis LOOKS closed and is not. Valid dispositions are a real constants \
                 key, measured_at_structural_maximum with measured_seconds under 2.0, or the \
                 {PENDING_MARKER_PREFIX}* marker"
            );
        }
    }

    /// Every enumerated cost axis carries a real disposition — no axis is still pending.
    ///
    /// # This test SHIPPED RED, on purpose, and plan 06-15 is what turned it green
    ///
    /// Plan 06-14 landed this assertion FAILING, naming the two cost axes its enumeration
    /// found and did not close: **C-07** (`holidays[].name` byte amplification) and **C-08**
    /// (unbudgeted NeuralProphet training work). It was not `#[ignore]`d, not
    /// `#[should_panic]`, not commented out and not deleted, because 06-14's whole thesis is
    /// that a guard which cannot fail is theater — and a plan arguing that while reducing its
    /// OWN two open items to a prose note would be making exactly the mistake it exists to
    /// end. The verbatim failing output was:
    ///
    /// ```text
    /// 2 cost axis/axes are still UNBOUNDED and are held open by this assertion: C-07
    /// (marker unbounded_pending_06_15, owed by plan 06-15); C-08 (marker
    /// unbounded_pending_06_15, owed by plan 06-15).
    /// test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 108 filtered out
    /// ```
    ///
    /// Plan 06-15 closed it the only legitimate way: by REPLACING both markers with real
    /// bounds — [`MAX_HOLIDAY_NAME_LEN`] for C-07 (a bound, never a measurement: that axis
    /// has no structural maximum, since `holidays[].name` is an unbounded `String` with no
    /// body-size or framing cap on either transport, so any number written there would
    /// measure an arbitrarily CHOSEN length) and [`MAX_NP_TRAIN_COST`] for C-08 (a bound
    /// because the MEASUREMENT forced one: the structural maximum walled at 47.924 s on
    /// release against SC1's 2 s bar).
    ///
    /// # It stays a live guard, and the forbidden repair is still forbidden
    ///
    /// The assertion is unchanged; only this comment moved from "why it is red" to "what
    /// closed it". Deleting, weakening, `#[ignore]`-ing or CI-filtering it restores precisely
    /// the guard-that-cannot-fail contradiction it exists to remove. A NEW pending axis is
    /// supposed to turn this red again — that is the mechanism, not a defect.
    #[test]
    fn no_cost_axis_is_pending() {
        let pending: Vec<String> = enumerated_axes()
            .into_iter()
            .filter(|(_, bound, _)| bound.starts_with(PENDING_MARKER_PREFIX))
            .map(|(axis, bound, _)| {
                let owed = bound
                    .strip_prefix(PENDING_MARKER_PREFIX)
                    .unwrap_or(&bound)
                    .replace('_', "-");
                format!("{axis} (marker {bound}, owed by plan {owed})")
            })
            .collect();
        assert!(
            pending.is_empty(),
            "{} cost axis/axes are still UNBOUNDED and are held open by this assertion: {}. \
             This test is RED ON PURPOSE from the close of wave 12 (plan 06-14) until wave 13 \
             (plan 06-15) replaces both pending markers with real bounds. Do NOT repair it by \
             deleting, weakening, ignoring or CI-filtering it — that reintroduces the \
             guard-that-cannot-fail class this round exists to end. Land 06-15.",
            pending.len(),
            pending.join("; ")
        );
    }

    /// D-20: adding `regressors` introduced NO required field.
    ///
    /// The partner of `an_unknown_argument_key_is_refused_not_ignored` below: that one pins
    /// that `deny_unknown_fields` still refuses what it always refused, this one pins that
    /// the body an existing caller already sends still deserialises and still means the same
    /// thing. A new REQUIRED field would break every caller at the transport layer, which is
    /// exactly the one-way break the wire shape was frozen to avoid.
    #[test]
    fn a_payload_with_no_new_keys_still_deserialises() {
        let args = serde_json::from_value::<ForecastArgs>(serde_json::json!({
            "ds": ["2020-01-01"], "y": [1.0], "horizon": 1
        }))
        .expect("a pre-existing minimal body must still deserialise");
        assert!(
            args.regressors.is_none(),
            "absent `regressors` must mean None, not an empty vec: the door distinguishes \
             'no regressors' from 'an empty regressor list' only by this"
        );
    }

    #[test]
    fn an_unknown_argument_key_is_refused_not_ignored() {
        let err = serde_json::from_value::<ForecastArgs>(serde_json::json!({
            "ds": ["2020-01-01"], "y": [1.0], "horizon": 1, "temperature": 0.7
        }))
        .expect_err("deny_unknown_fields must refuse `temperature`");
        assert!(err.to_string().contains("temperature"), "{err}");
    }

    #[test]
    fn the_advertised_schema_is_strict_and_names_the_three_required_fields() {
        let schema =
            serde_json::to_value(schemars::schema_for!(ForecastArgs)).expect("schema serializes");
        assert_eq!(schema["additionalProperties"], serde_json::json!(false));
        let required = schema["required"].as_array().expect("required array");
        for field in ["ds", "y", "horizon"] {
            assert!(
                required.contains(&serde_json::json!(field)),
                "{field} must be required: {schema}"
            );
        }
        assert!(
            !required.contains(&serde_json::json!("freq")),
            "freq is optional on every surface"
        );
    }

    #[test]
    fn the_two_error_variants_display_their_message() {
        assert_eq!(ForecastError::Validation("nope".into()).to_string(), "nope");
        assert_eq!(ForecastError::Internal("boom".into()).to_string(), "boom");
    }
}
