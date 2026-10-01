//! NeuralProphet-lite on aprender's f32 autograd: trend (11 segments, NP's
//! parametrisation), Fourier seasonality (days since 1900-01-01, sin block then
//! cos block per NP), optional AR-Net on stationarised lags, weighted Huber
//! loss, AdamW + three-phase one-cycle cosine schedule.
//!
//! Ported verbatim from `sources/004-forecast-mcp-thin-server/src/np.rs` (D-08); the only
//! edits are the ones clippy/rustfmt forced and the `days_from_civil` import, which lives in
//! [`crate::dates`] here rather than in the spike's `prophet` module. Every training rule in
//! this file is load-bearing (D-10) and is pinned by an invariant test in `mod parity`:
//!
//! * [`weighted_huber`] is built from graph-connected ops with a constant 0/1 mask. Core's
//!   own Huber loss in `aprender::nn::loss` (the smooth-L1 type; deliberately NOT named here,
//!   so the D-10 ban stays a file-wide grep) extracts raw data and builds a fresh `Tensor`, so
//!   the parameter gradient after `backward()` is `None` — it cannot train anything. Never
//!   import it into this module; 06-09 files the core ticket to fix it.
//! * [`clear_graph`] runs after EVERY optimiser step; the tape is thread-local and unbounded.
//! * Mini-batches, never full batch: full-batch training collapses to test MAE 2.6.
//! * The learning rate is selected by TRAIN loss, never by test error.
//! * Fourier features are on days since 1900-01-01, which is NeuralProphet's own epoch.

use crate::dates::days_from_civil;
use crate::events::{EventBlock, EventDesign};
use crate::regressors::{NpRegressorRows, NpRegressors, RegressorBlock, Standardized};
use aprender::autograd::{clear_graph, graph_tape_len, no_grad, Tensor};
use aprender::nn::optim::{AdamW, Optimizer};
use aprender::nn::{Linear, Module};
use std::time::Instant;

// ------------------------------------------------------------------ rng ----
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.max(1))
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    pub fn shuffle(&mut self, v: &mut [usize]) {
        for i in (1..v.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
    }
}

// ------------------------------------------------------------- features ----
#[derive(Clone, Debug)]
pub struct NpSeason {
    pub name: String,
    pub period: f64,
    pub order: usize,
}

/// NeuralProphet's auto seasonality: same disable rules as Prophet, resolutions 6 / 3 / 6.
pub fn np_auto_seasonalities(ds_days: &[i64]) -> Vec<NpSeason> {
    let span = (ds_days[ds_days.len() - 1] - ds_days[0]) as f64;
    let min_dt = ds_days
        .windows(2)
        .map(|w| (w[1] - w[0]) as f64)
        .filter(|d| *d > 0.0)
        .fold(f64::INFINITY, f64::min);
    let mut v = Vec::new();
    if span >= 730.0 {
        v.push(NpSeason {
            name: "yearly".into(),
            period: 365.25,
            order: 6,
        });
    }
    if span >= 14.0 && min_dt < 7.0 {
        v.push(NpSeason {
            name: "weekly".into(),
            period: 7.0,
            order: 3,
        });
    }
    if span >= 2.0 && min_dt < 1.0 {
        v.push(NpSeason {
            name: "daily".into(),
            period: 1.0,
            order: 6,
        });
    }
    v
}

pub fn season_dim(s: &[NpSeason]) -> usize {
    s.iter().map(|x| 2 * x.order).sum()
}

/// NP: t in days since 1900-01-01; per seasonality all sin(k*2*pi*t/P) then all cos.
pub fn fourier_feats(day: i64, seasons: &[NpSeason], out: &mut Vec<f32>) {
    let t = (day - days_from_civil(1900, 1, 1)) as f64;
    for s in seasons {
        let f = 2.0 * std::f64::consts::PI / s.period;
        for k in 1..=s.order {
            out.push((f * k as f64 * t).sin() as f32);
        }
        for k in 1..=s.order {
            out.push((f * k as f64 * t).cos() as f32);
        }
    }
}

/// NP trend as a linear function of `[k0, delta_0..delta_S]`: slope in segment i is
/// `k0 + delta_i`, continuity offsets `-cp_j*(delta_j - delta_{j-1})` for every passed
/// changepoint `j >= 1`.
pub fn trend_feats(t: f64, cps: &[f64], out: &mut Vec<f32>) {
    let s = cps.len();
    let seg = cps[1..].iter().filter(|&&c| t >= c).count();
    let mut phi = vec![0.0f64; s];
    phi[seg] += t;
    for j in 1..s {
        if t >= cps[j] {
            phi[j] -= cps[j];
            phi[j - 1] += cps[j];
        }
    }
    out.push(t as f32);
    out.extend(phi.iter().map(|v| *v as f32));
}

/// NeuralProphet's own `end_w`, and the default [`TrainConfig::newer_w`] carries.
///
/// Used by the PREDICTION paths, whose `Rows::w` no loss ever reads; training reads the
/// caller's `cfg.newer_w` instead of this.
pub const DEFAULT_END_W: f64 = 2.0;

/// Newer-samples weight (NP `_get_time_based_sample_weight`, `end_w = 2`, `start_t = 0`).
pub fn sample_weight(t: f64, end_w: f64) -> f32 {
    let time = t.clamp(0.0, 1.0);
    let c = 0.5 * (std::f64::consts::PI * (time - 1.0)).cos() + 0.5;
    ((1.0 + c * (end_w - 1.0)) / end_w) as f32
}

// ----------------------------------------------------------------- data ----
pub struct NpData {
    /// Daily grid covering `[first, last]` of the FULL series (train + test); y linearly imputed.
    pub grid_days: Vec<i64>,
    pub grid_y: Vec<f64>,
    pub grid_observed: Vec<bool>,
    pub n_train_grid: usize,
    pub shift: f64,
    pub scale: f64,
    pub t0: i64,
    pub t_span: f64,
    pub cps: Vec<f64>,
    pub seasons: Vec<NpSeason>,
}

impl NpData {
    pub fn new(
        ds_days: &[i64],
        y: &[f64],
        n_train_rows: usize,
        n_changepoints: usize,
        changepoints_range: f64,
    ) -> Self {
        let first = ds_days[0];
        let last = ds_days[ds_days.len() - 1];
        let n = (last - first + 1) as usize;
        let mut grid_y = vec![f64::NAN; n];
        let mut grid_observed = vec![false; n];
        for (d, v) in ds_days.iter().zip(y) {
            let i = (d - first) as usize;
            grid_y[i] = *v;
            grid_observed[i] = true;
        }
        // linear imputation
        let mut i = 0;
        while i < n {
            if grid_y[i].is_nan() {
                let a = i - 1;
                let mut b = i;
                while grid_y[b].is_nan() {
                    b += 1;
                }
                for k in i..b {
                    let f = (k - a) as f64 / (b - a) as f64;
                    grid_y[k] = grid_y[a] + f * (grid_y[b] - grid_y[a]);
                }
                i = b;
            }
            i += 1;
        }
        let train_last = ds_days[n_train_rows - 1];
        let n_train_grid = (train_last - first + 1) as usize;
        // soft normalisation from OBSERVED train values: shift = min, scale = q95 - min
        let mut obs: Vec<f64> = y[..n_train_rows].to_vec();
        obs.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        let lowest = obs[0];
        let q = 0.95 * (obs.len() - 1) as f64;
        let lo = q.floor() as usize;
        let frac = q - lo as f64;
        let q95 = obs[lo] + frac * (obs[(lo + 1).min(obs.len() - 1)] - obs[lo]);
        let width = if (q95 - lowest).abs() < 1e-12 {
            obs[obs.len() - 1] - lowest
        } else {
            q95 - lowest
        };
        let cps: Vec<f64> = (0..=n_changepoints)
            .map(|i| changepoints_range * i as f64 / (n_changepoints + 1) as f64)
            .collect();
        // The auto rules can select NOTHING — 20 points seven days apart give span 133
        // (no yearly) and min_dt exactly 7.0 (no weekly, the test is `< 7.0`) — and
        // `season_dim() == 0` then builds a `Linear::without_bias(0, 1)` whose transpose
        // trips trueno's `Contract transpose: input is empty`. Through the server that is
        // an `Internal` for a caller-fixable input; with debug assertions off it is worse,
        // a silently zero seasonality term. Prophet's arm already fills the same hole with
        // a harmless weekly column (`forecast.rs`); do the same here, on the daily grid
        // this struct imputes, so the design always has K >= 1.
        let mut seasons = np_auto_seasonalities(&ds_days[..n_train_rows]);
        if seasons.is_empty() {
            seasons.push(NpSeason {
                name: "weekly".into(),
                period: 7.0,
                order: 3,
            });
        }
        NpData {
            grid_days: (first..=last).collect(),
            grid_y,
            grid_observed,
            n_train_grid,
            shift: lowest,
            scale: width,
            t0: first,
            t_span: (train_last - first) as f64,
            cps,
            seasons,
        }
    }
    pub fn t_of(&self, day: i64) -> f64 {
        (day - self.t0) as f64 / self.t_span
    }
    pub fn norm(&self, v: f64) -> f32 {
        ((v - self.shift) / self.scale) as f32
    }
    pub fn denorm(&self, v: f32) -> f64 {
        v as f64 * self.scale + self.shift
    }
    pub fn trend_dim(&self) -> usize {
        self.cps.len() + 1
    }
    pub fn season_dim(&self) -> usize {
        season_dim(&self.seasons)
    }
    pub fn row_feats(&self, day: i64, tr: &mut Vec<f32>, se: &mut Vec<f32>) {
        trend_feats(self.t_of(day), &self.cps, tr);
        fourier_feats(day, &self.seasons, se);
    }
}

// ---------------------------------------------------------------- model ----
pub struct NpModel {
    pub trend: Linear,
    pub season: Linear,
    pub bias: Tensor,
    pub ar: Vec<Linear>,
    pub n_lags: usize,
}

fn normal_tensor(rng: &mut Rng, shape: &[usize], std: f64) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_vec((0..n).map(|_| (rng.normal() * std) as f32).collect(), shape).requires_grad()
}

impl NpModel {
    /// NP init: `k0 ~ xavier` on `[1,1]` (std 1), `delta` std `sqrt(2/(2S))`, season std
    /// `sqrt(1/(2R))`, bias std 1, AR kaiming `fan_in`.
    pub fn new(d: &NpData, n_lags: usize, ar_layers: &[usize], rng: &mut Rng) -> Self {
        let td = d.trend_dim();
        let mut trend = Linear::without_bias(td, 1);
        let mut w: Vec<f32> = vec![rng.normal() as f32];
        let s = d.cps.len();
        w.extend((0..s).map(|_| (rng.normal() * (2.0 / (2.0 * s as f64)).sqrt()) as f32));
        trend.set_weight(Tensor::from_vec(w, &[1, td]).requires_grad());
        let sd = d.season_dim();
        let mut season = Linear::without_bias(sd, 1);
        let mut w = Vec::with_capacity(sd);
        for se in &d.seasons {
            let std = (1.0 / (2.0 * se.order as f64)).sqrt();
            w.extend((0..2 * se.order).map(|_| (rng.normal() * std) as f32));
        }
        season.set_weight(Tensor::from_vec(w, &[1, sd]).requires_grad());
        let bias = normal_tensor(rng, &[1], 1.0);
        let mut ar = Vec::new();
        if n_lags > 0 {
            let mut d_in = n_lags;
            for &h in ar_layers {
                let mut l = Linear::new(d_in, h);
                l.set_weight(normal_tensor(rng, &[h, d_in], (2.0 / d_in as f64).sqrt()));
                l.set_bias(Tensor::zeros(&[h]).requires_grad());
                ar.push(l);
                d_in = h;
            }
            let mut l = Linear::without_bias(d_in, 1);
            l.set_weight(normal_tensor(rng, &[1, d_in], (2.0 / d_in as f64).sqrt()));
            ar.push(l);
        }
        NpModel {
            trend,
            season,
            bias,
            ar,
            n_lags,
        }
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut v = self.trend.parameters_mut();
        v.extend(self.season.parameters_mut());
        v.push(&mut self.bias);
        for l in &mut self.ar {
            v.extend(l.parameters_mut());
        }
        v
    }

    pub fn n_params(&self) -> usize {
        self.trend.num_parameters()
            + self.season.num_parameters()
            + 1
            + self.ar.iter().map(Module::num_parameters).sum::<usize>()
    }

    /// `lags`: (raw normalised lags `[b,L]`, season feats at lag times `[b*L,F]`, trend at lag
    /// times `[b,L]` (detached)).
    pub fn forward(
        &self,
        x_trend: &Tensor,
        x_season: &Tensor,
        lags: Option<(&Tensor, &Tensor, &Tensor)>,
    ) -> Tensor {
        let b = x_trend.shape()[0];
        let out = self
            .trend
            .forward(x_trend)
            .add(&self.season.forward(x_season))
            .broadcast_add(&self.bias);
        match lags {
            None => out,
            Some((raw, se_lag, tr_lag)) => {
                let s_lag = self.season.forward(se_lag).view(&[b, self.n_lags]);
                let mut h = raw.sub(tr_lag).sub(&s_lag);
                let n = self.ar.len();
                for (i, l) in self.ar.iter().enumerate() {
                    h = l.forward(&h);
                    if i + 1 < n {
                        h = h.relu();
                    }
                }
                out.add(&h)
            }
        }
    }
}

/// Exact SmoothL1 (Huber, beta) x sample weight, mean — built from graph-connected ops with a
/// constant 0/1 mask (the mask is piecewise constant, so this IS the exact gradient).
///
/// This function exists because core's own smooth-L1 (Huber) loss in `aprender::nn::loss` is
/// DETACHED from the graph (D-10): its `forward` maps over `diff.data()` and wraps the result
/// in a fresh `Tensor`, so every parameter's gradient after `backward()` is `None`. The type is
/// deliberately not named here — the D-10 ban is enforced as a file-wide grep. Pinned by
/// `parity::weighted_huber_is_graph_connected`.
pub fn weighted_huber(pred: &Tensor, target: &Tensor, w: &Tensor, beta: f32) -> Tensor {
    let d = pred.sub(target);
    let a = d.abs();
    let mask: Vec<f32> = a
        .data()
        .iter()
        .map(|v| if *v < beta { 1.0 } else { 0.0 })
        .collect();
    let inv: Vec<f32> = mask.iter().map(|m| 1.0 - m).collect();
    let m = Tensor::from_vec(mask, a.shape());
    let im = Tensor::from_vec(inv, a.shape());
    let half_beta = Tensor::from_vec(vec![0.5 * beta; a.numel()], a.shape());
    let quad = d.pow(2.0).mul_scalar(0.5 / beta).mul(&m);
    let lin = a.sub(&half_beta).mul(&im);
    quad.add(&lin).mul(w).mean()
}

/// NP's `OneCycleLR` (`three_phase`, cos, div 10, `final_div` 10) over progress `p` in `[0,1]`.
pub fn one_cycle_lr(p: f64, max_lr: f64) -> f64 {
    let (init, fin) = (max_lr / 10.0, max_lr / 100.0);
    let cosine =
        |s: f64, e: f64, f: f64| e + (s - e) / 2.0 * (1.0 + (std::f64::consts::PI * f).cos());
    if p < 0.3 {
        cosine(init, max_lr, p / 0.3)
    } else if p < 0.6 {
        cosine(max_lr, init, (p - 0.3) / 0.3)
    } else {
        cosine(init, fin, (p - 0.6) / 0.4)
    }
}

/// NeuralProphet's auto batch size. ALWAYS a mini-batch for `n >= 64` (D-10): full-batch
/// training collapses the fit (measured test MAE 2.6 against 0.45).
pub fn auto_batch(n: usize) -> usize {
    (2usize.pow(1 + (1.5 * (n as f64).log10()) as u32))
        .clamp(8, 2048)
        .min(n)
}

/// NeuralProphet's auto epoch count. The formula ASSUMES mini-batches — see [`auto_batch`].
pub fn auto_epochs(n: usize) -> usize {
    (10.0 * (100.0 / n as f64 * 2f64.powf(2.25 * (10.0 + n as f64).log10())).ceil())
        .clamp(20.0, 500.0) as usize
}

/// The number of grid rows [`train`] will actually step over for this `n_lags`.
///
/// The DOOR needs this BEFORE it decides whether to train at all (cost axis C-08), and
/// [`train`] derives its own `n` from the same rule — a `debug_assert` inside `train` pins
/// the two together. A second inline copy of this rule in `forecast.rs` is exactly the drift
/// hazard `prophet::changepoint_count` was extracted to avoid in plan 06-14: a bound the
/// door computes differently from the work that is spent is a bound that can be evaded
/// wherever the two disagree.
#[must_use]
pub fn n_training_samples(d: &NpData, n_lags: usize) -> usize {
    // DERIVED from the sample list rather than re-deriving the branch, so the count the
    // door PRICES with and the rows the trainer SPENDS on cannot disagree. They were two
    // copies of one rule pinned only by a `debug_assert!` in `train` — compiled out of
    // `--release`, the profile the door ships in, which is the same objection this module
    // raises against `debug_assert!` elsewhere. Called once per request, not per step.
    training_samples(d, n_lags).0.len()
}

/// The training sample list AND, parallel to it, the CALLER-row index of each sample.
///
/// ONE function, because the two lists must agree element for element and a second copy of
/// the sample rule is the drift hazard [`n_training_samples`] was extracted to avoid. It is
/// the single source both [`train`] and the D-27 tests read, so "the lag-free path indexes
/// regressors by the caller's row" is a property of shipped code rather than of a test
/// fixture.
///
/// # The lag-free branch: the caller's row, never the grid row (D-27)
///
/// `NpData::new` imputes a DAILY grid over `[first, last]`, so on a gappy series the grid is
/// denser than the caller's rows. The lag-free sample list is already filtered to
/// `grid_observed`, and the observed grid rows correspond ONE FOR ONE, in ascending order,
/// to the caller's rows — so the caller-row index is simply the running count of observed
/// rows. An imputed day is never selected here, so it has no regressor slot and there is no
/// fill rule to choose.
///
/// # The lagged branch: total only because the door refuses the other case
///
/// With lags the sample list is every grid row from `n_lags` onward, imputed rows included,
/// so a caller-row index does not exist for every sample in general. D-26 REFUSES a
/// regressor on a gappy series at non-zero lags, and on a gap-free series the grid rows and
/// the caller rows coincide one for one, so `i` is the caller row. That coincidence is
/// asserted in RELEASE by [`crate::regressors::NpRegressors::new`], not by a `debug_assert!`
/// that `--release` would compile out.
#[must_use]
pub fn training_samples(d: &NpData, n_lags: usize) -> (Vec<usize>, Vec<usize>) {
    if n_lags == 0 {
        let mut samples = Vec::new();
        let mut caller_rows = Vec::new();
        let mut observed_so_far = 0usize;
        for i in 0..d.n_train_grid {
            if d.grid_observed[i] {
                samples.push(i);
                caller_rows.push(observed_so_far);
                observed_so_far += 1;
            }
        }
        (samples, caller_rows)
    } else {
        let samples: Vec<usize> = (n_lags..d.n_train_grid).collect();
        let caller_rows = samples.clone();
        (samples, caller_rows)
    }
}

/// TEST-ONLY: apply a selection to a GRID-INDEXED value vector.
///
/// The shipped path never builds a grid-indexed regressor vector at all — that is the whole
/// of D-27 — so this exists for ONE purpose: to let the four-rule falsification probe feed
/// four genuinely DIFFERENT inputs into the one place the selection happens, and observe
/// that the selection makes them identical again. Without it, after D-27 removed the
/// grid-shaped array, four notionally different fills would collapse into four IDENTICAL
/// calls and the probe would compare a function against itself and pass on any
/// implementation, including a broken one.
///
/// It is `#[cfg(test)]` so it cannot become a shipped path by accident, and the
/// no-interpolation source guard excludes `#[cfg(test)]` code by path for the same reason:
/// the four fill rules are TEST INPUTS.
///
/// Total lookup (IN-02): a row outside the vector yields `NaN` rather than panicking, so a
/// mis-built probe fails an assertion instead of aborting the test process.
#[cfg(test)]
#[must_use]
pub(crate) fn select_grid_values(grid: &[f64], rows: &[usize]) -> Vec<f64> {
    rows.iter()
        .map(|&i| grid.get(i).copied().unwrap_or(f64::NAN))
        .collect()
}

/// A door-computable PROXY for the multiply-accumulates ONE [`train`] call spends:
///
/// ```text
/// epochs * n_samples * ( (n_lags + 1) + fit_np_event_cost_per_column * E )
/// ```
///
/// # The event term sits INSIDE the per-sample width (SC4, D-34)
///
/// `(n_lags + 1)` is the PER-SAMPLE FEATURE WIDTH the optimiser sweeps, and the event block
/// adds work to that width. The measured growth is a PER-STEP slope
/// (`us/step ~= a + b*E`), so the added work is paid once per sample per epoch, exactly like
/// the width beside it.
///
/// The alternative reading — `product + c * E` — scales with NEITHER epochs NOR samples, so
/// it can be tuned to pass an observation at one geometry while under-pricing every larger
/// series. That is the shape this function must not have, and
/// `tests::the_event_cost_shape_scales_with_geometry` is what discriminates the two: the
/// width form's `priced(E)/priced(0)` is identical at two geometries differing 10x in
/// samples, while the additive-outside form's collapses toward 1 as the product grows.
///
/// The event contribution to the width is rounded UP before the multiplications — a door
/// must never under-price by a fractional column — and at `n_event_cols == 0` the `ceil` of
/// zero is zero, so the priced cost is BIT-IDENTICAL to what it was before the term existed
/// (`tests::the_priced_cost_at_zero_event_columns_is_unchanged`).
///
/// # The original contract of this function, unchanged
///
/// **It is a proxy, not a wall-clock prediction.** It counts the per-sample feature width
/// the optimiser sweeps (`n_lags` AR inputs plus the trend/seasonality block, collapsed to
/// `+ 1`) times the number of samples times the number of epochs — deliberately ignoring
/// the AR head width, the batch size and the autograd tape, none of which the door can
/// price and none of which change the SHAPE of the growth. Its only job is to be the SAME
/// number the door checks and the trainer spends, so a request cannot buy work the door
/// did not price.
///
/// The door multiplies this by the learning-rate SWEEP width (2 with lags, 3 without),
/// because the sweep is run by `forecast::forecast` and not by `train`.
///
/// Saturating throughout: at the structural maximum the product is ~3.6e8 per training,
/// nine orders of magnitude below `u64::MAX`, so saturation is unreachable — it is here so
/// that a future bound change cannot turn an overflow into a silently SMALL cost that
/// passes the door.
#[must_use]
pub fn train_cost(
    n_samples: usize,
    epochs: usize,
    n_lags: usize,
    n_event_cols: usize,
    n_regressor_cols: usize,
) -> u64 {
    let width = (n_lags as u64 + 1)
        .saturating_add(per_sample_event_width(n_event_cols))
        .saturating_add(per_sample_regressor_width(n_regressor_cols));
    (epochs as u64)
        .saturating_mul(n_samples as u64)
        .saturating_mul(width)
}

/// The event block's contribution to the per-sample feature width, in whole proxy units.
///
/// Rounded UP: a door must never under-price by a fractional column, and at
/// `n_event_cols == 0` the `ceil` of zero is zero, which is what makes the event-free price
/// bit-identical to the pre-event arithmetic.
#[must_use]
fn per_sample_event_width(n_event_cols: usize) -> u64 {
    per_sample_exog_width(crate::types::FIT_NP_EVENT_COST_PER_COLUMN, n_event_cols)
}

/// The rounding and saturation POLICY both exogenous width terms share.
///
/// The two COEFFICIENTS stay separate — separately measured, separately mirrored in the
/// contract, separately re-priceable by Phase 7 — and passing one as an argument keeps them
/// that way. What must not be duplicated is this: `ceil` then a saturating cast, so a
/// non-finite or negative coefficient cannot wrap into a silently SMALL width that passes
/// the door. Two copies of that rule means hardening one and leaving the other on the
/// door's own under-pricing path.
fn per_sample_exog_width(coefficient: f64, n_cols: usize) -> u64 {
    if n_cols == 0 {
        return 0;
    }
    let raw = coefficient * n_cols as f64;
    if raw.is_finite() && raw >= 0.0 {
        // Saturating by construction: Rust's float->int casts saturate, so an
        // enormous `raw` lands on `u64::MAX` rather than wrapping.
        raw.ceil() as u64
    } else {
        // FAIL HIGH, not to zero. `n_cols > 0` here, so the family IS present and must
        // cost something; the only inputs that reach this arm are a non-finite or
        // NEGATIVE coefficient, i.e. a broken calibration. Returning 0 — the smallest
        // possible width — was precisely the "silently SMALL width that passes the door"
        // this function's own doc says the ceil-then-saturate policy prevents: it would
        // price E event columns and R regressor columns at the bare `(n_lags + 1)`
        // arithmetic and reinstate the under-pricing the C-08 terms were added to close.
        // A cost ceiling's only safe failure direction is high, so a bad coefficient
        // refuses the request instead of discounting it.
        u64::MAX
    }
}

/// The numeric-regressor block's contribution to the per-sample feature width, in whole
/// proxy units.
///
/// A SEPARATE function from [`per_sample_event_width`] reading a SEPARATE constant, because
/// the two coefficients are separately measured — sharing one function would make a future
/// re-price of either move both, which is the drift the contract-mirror pattern exists to
/// prevent. Rounded UP for the same reason, and at `n_regressor_cols == 0` the `ceil` of
/// zero is zero, which is what keeps the regressor-free price bit-identical to plan
/// 06.1-05's arithmetic.
#[must_use]
fn per_sample_regressor_width(n_regressor_cols: usize) -> u64 {
    per_sample_exog_width(
        crate::types::FIT_NP_REGRESSOR_COST_PER_COLUMN,
        n_regressor_cols,
    )
}

/// The learning-rate sweep the DOOR runs for this `n_lags`.
///
/// spike-002's stand-in for NeuralProphet's range test (D-10: selected by TRAIN loss, never
/// by test error). It lives here, not inlined in `forecast.rs`, so the door's COST estimate
/// multiplies by the same width the door actually runs — a second inline copy is precisely
/// the drift hazard `prophet::changepoint_count` was extracted to avoid in plan 06-14.
#[must_use]
pub fn door_lr_sweep(n_lags: usize) -> &'static [f64] {
    if n_lags > 0 {
        &[0.03, 0.1]
    } else {
        &[0.01, 0.03, 0.1]
    }
}

/// The epoch count the DOOR configures for one [`train`] call.
///
/// With lags on, spike-002 gives the linear AR case 4x the auto epochs of the POINT count,
/// capped at 320; without lags the door passes `epochs: None` and `train` falls back to
/// `auto_epochs(n_samples)` — which is exactly what this returns, so the door can price the
/// lag-free arm without changing its behaviour.
#[must_use]
pub fn door_epochs(n_points: usize, n_samples: usize, n_lags: usize) -> usize {
    if n_lags > 0 {
        auto_epochs(n_points).min(320)
    } else {
        auto_epochs(n_samples)
    }
}

/// The TOTAL priced work of ONE `forecast` request on the neuralprophet arm: [`train_cost`]
/// for a single training, times the width of the learning-rate sweep the door runs.
///
/// This is the number cost axis **C-08** is bounded on. It is computable at the door — every
/// factor is known once [`NpData::new`] has run and the `n_lags` range checks have fired —
/// and it is built out of the same three functions the door then uses to CONFIGURE the
/// sweep, so a request cannot buy work the door did not price.
#[must_use]
pub fn request_train_cost(
    d: &NpData,
    n_points: usize,
    n_lags: usize,
    n_event_cols: usize,
    n_regressor_cols: usize,
) -> u64 {
    let n_samples = n_training_samples(d, n_lags);
    train_cost(
        n_samples,
        door_epochs(n_points, n_samples, n_lags),
        n_lags,
        n_event_cols,
        n_regressor_cols,
    )
    .saturating_mul(door_lr_sweep(n_lags).len() as u64)
}

// -------------------------------------------------------------- training ----
pub struct TrainConfig {
    pub n_lags: usize,
    pub ar_layers: Vec<usize>,
    pub max_lr: f64,
    pub epochs: Option<usize>,
    pub batch: Option<usize>,
    pub weight_decay: f32,
    pub huber_beta: f32,
    pub newer_w: f64,
    pub seed: u64,
    /// The expanded event design, or `None` for an event-free fit (D-30, SC3).
    ///
    /// `Some(design)` attaches an additive [`EventBlock`] BESIDE [`NpModel`] — one `.add()`
    /// on the forward, both parameter sets handed to ONE `AdamW`. `None` and an empty design
    /// are the same fit, bit for bit: no block is built, nothing is added, and the recorded
    /// [`TrainLog::train_cost`] is unchanged.
    ///
    /// Declared LAST so the exhaustive struct literals that build it stay in declaration
    /// order (`clippy::inconsistent_struct_constructor` is a workspace `warn`).
    pub event_design: Option<EventDesign>,
    /// The numeric-regressor channel, or `None` for a regressor-free fit (D-22, D-27).
    ///
    /// `Some(channel)` attaches an additive [`RegressorBlock`] BESIDE [`NpModel`] and the
    /// event block — a SECOND `.add()` on the forward, its parameters extended into the
    /// same single `AdamW` vector. `None` and a zero-width channel are the same fit bit for
    /// bit: no block is built, nothing is added, and both recorded neuralprophet invariance
    /// signatures still reproduce.
    ///
    /// The channel holds the CALLER's arrays at caller length. There is no grid-shaped
    /// regressor array anywhere in this file, which is the whole of D-27 — see the
    /// [`crate::regressors`] module section for why the spike prototype's `reg_grid` is
    /// deliberately not ported.
    ///
    /// Declared after `event_design` for the same declaration-order reason.
    pub regressors: Option<NpRegressors>,
}

pub struct TrainLog {
    pub epochs: usize,
    pub batch: usize,
    pub n_samples: usize,
    /// Scalar parameter count, INCLUDING the event block's when one was trained.
    pub n_params: usize,
    pub epoch_loss: Vec<f64>,
    pub seconds: f64,
    pub tape_len_per_step: usize,
    pub steps: usize,
    /// [`train_cost`] evaluated on the `n_samples` / `epochs` / `n_lags` this call actually
    /// used — the same number the door priced the request at before calling in.
    pub train_cost: u64,
    /// The number of event indicator columns this fit trained on; `0` when events were off.
    pub n_event_cols: usize,
    /// The number of parameter TENSORS handed to `AdamW`, recorded at the moment the
    /// optimiser was constructed.
    ///
    /// This is what makes "the block is actually being optimised" checkable. A block whose
    /// parameters are built and forwarded but never extended into that vector still
    /// contributes to the forward pass — its weights simply never move — and the failure
    /// then surfaces as "the planted effect was not recovered", which names the symptom
    /// rather than the cause.
    pub n_opt_tensors: usize,
    /// The trained event block, or `None` when events were off.
    ///
    /// It rides in the LOG rather than being returned as a third value on purpose: `train`'s
    /// return arity is unchanged, so every existing call site stays as it was. The predict
    /// helpers need the block (and the design) to add an event contribution for the rows
    /// they predict, and the caller already holds the log.
    pub events: Option<EventBlock>,
    /// The trained numeric-regressor block, or `None` when regressors were off.
    ///
    /// Rides in the LOG beside [`TrainLog::events`], for the same reason and by the same
    /// rule: `train`'s return arity is unchanged and the caller already holds the log.
    pub regressors: Option<RegressorBlock>,
    /// The standardisation constants the fit ACTUALLY used, in request order.
    ///
    /// They travel WITH the weights because a future row's contribution needs the SAME
    /// `mu`/`std` the training rows used. Keeping them here means the predict paths and the
    /// tests read them from ONE place rather than re-deriving them from the caller's array,
    /// which is how two copies of one constant drift apart.
    pub regressor_specs: Vec<Standardized>,
    /// The number of numeric-regressor columns this fit trained on; `0` when they were off.
    pub n_regressor_cols: usize,
}

impl TrainLog {
    /// The learned per-column event weights, in [`crate::events::event_columns`] order.
    ///
    /// Empty when events were off. Read from the block rather than copied into a second
    /// field, so there is exactly one source for the number a recovery or determinism test
    /// asserts on.
    #[must_use]
    pub fn event_weights(&self) -> Vec<f32> {
        self.events
            .as_ref()
            .map(EventBlock::weights)
            .unwrap_or_default()
    }
}

/// Precomputed per-grid-row features.
pub struct Rows {
    pub tr: Vec<f32>,
    pub se: Vec<f32>,
    pub y: Vec<f32>,
    pub w: Vec<f32>,
    pub td: usize,
    pub sd: usize,
}

/// `end_w` is [`TrainConfig::newer_w`]: the sample weight at the END of the series, with
/// 1.0 at the start. It is a PARAMETER rather than the literal `2.0` it used to be because
/// `newer_w` is a public knob on a public config — hardcoding it here meant a caller could
/// set `newer_w: 4.0`, get a fit trained at 2.0, and be told nothing. Prediction paths pass
/// the same default; `Rows::w` is only read by the training loss.
pub fn rows_for(d: &NpData, days: &[i64], y_norm: &[f32], end_w: f64) -> Rows {
    let (td, sd) = (d.trend_dim(), d.season_dim());
    let (mut tr, mut se, mut w) = (
        Vec::with_capacity(days.len() * td),
        Vec::with_capacity(days.len() * sd),
        Vec::with_capacity(days.len()),
    );
    for &day in days {
        d.row_feats(day, &mut tr, &mut se);
        w.push(sample_weight(d.t_of(day), end_w));
    }
    Rows {
        tr,
        se,
        y: y_norm.to_vec(),
        w,
        td,
        sd,
    }
}

fn gather(rows: &[f32], width: usize, idx: &[usize]) -> Vec<f32> {
    let mut v = Vec::with_capacity(idx.len() * width);
    for &i in idx {
        v.extend_from_slice(&rows[i * width..(i + 1) * width]);
    }
    v
}

/// Train the NeuralProphet-lite model. `clear_graph()` runs after EVERY optimiser step (D-10):
/// the autograd tape is thread-local and unbounded, so a fit that does not clear it leaks the
/// whole training history and leaves the tape dirty for the next caller on that thread.
pub fn train(d: &NpData, cfg: &TrainConfig, verbose: bool) -> (NpModel, TrainLog) {
    let mut rng = Rng::new(cfg.seed);
    let mut model = NpModel::new(d, cfg.n_lags, &cfg.ar_layers, &mut rng);
    // The event block is drawn from the SAME rng, AFTER the model, so an events-OFF run at
    // the same seed initialises the model identically — which is what makes an ON/OFF
    // comparison a comparison of the block (D-30).
    let design = cfg.event_design.as_ref().filter(|g| g.dim() > 0);
    let ed = design.map_or(0, EventDesign::dim);
    let mut block = design.map(|_| EventBlock::new(ed, &mut rng));
    // The regressor block is drawn from the SAME rng, AFTER the event block, for the same
    // reason the event block is drawn after the model: a regressor-free run at the same seed
    // then initialises the model and the event block identically, which is what keeps both
    // recorded invariance signatures reproducing and makes an ON/OFF comparison a comparison
    // of the block.
    let reg_channel = cfg.regressors.as_ref().filter(|r| r.dim() > 0);
    let rd = reg_channel.map_or(0, NpRegressors::dim);
    let mut reg_block = reg_channel.map(|_| RegressorBlock::new(rd, &mut rng));
    let n_grid = d.n_train_grid;
    let y_norm: Vec<f32> = d.grid_y[..n_grid].iter().map(|v| d.norm(*v)).collect();
    let rows = rows_for(d, &d.grid_days[..n_grid], &y_norm, cfg.newer_w);
    // Built ONCE over the training grid, with `rows_for`'s hoisting discipline: the
    // membership sets live on the design and the row loop only looks days up in them.
    let ev_rows: Vec<f32> = design.map_or_else(Vec::new, |g| g.rows(&d.grid_days[..n_grid]));
    let l = cfg.n_lags;
    // NP trains the lag-free model on the OBSERVED rows only (no imputation needed);
    // with lags it trains on the imputed daily grid so every window is complete.
    //
    // `caller_rows` is the parallel list D-27 turns on: the CALLER's own row index for each
    // sample. It is built by the same function and in the same pass as `samples`, so the
    // two cannot drift; see [`training_samples`].
    let (samples, caller_rows) = training_samples(d, l);
    let n = samples.len();
    // The standardised regressor rows, `[n, rd]` — one row per SAMPLE, indexed by the
    // CALLER's row. Not `[n_grid, rd]`: building a grid-shaped array is what would create
    // an imputed-day slot with an undefined value in it, and D-27 removes the slot rather
    // than choosing a fill rule for it (spike 014 measured two defensible rules 10.48 apart
    // on a series of scale 35.32, with a garbage probe flipping the sign of both
    // coefficients).
    let reg_rows: Vec<f32> = reg_channel.map_or_else(Vec::new, |r| r.history_rows(&caller_rows));
    let batch = cfg.batch.unwrap_or_else(|| auto_batch(n));
    let epochs = cfg.epochs.unwrap_or_else(|| auto_epochs(n));
    let n_batches = n.div_ceil(batch);
    let total_steps = epochs * n_batches;
    // The door priced this request with `train_cost(n_training_samples(d, l), epochs, l, E, R)`
    // BEFORE calling in (cost axis C-08). If the two ever derived a different `n` the bound
    // would be evadable wherever they disagreed, so the sample rule is pinned here rather
    // than argued. `debug_assert` because this is the hot path and the rule is one branch.
    debug_assert_eq!(
        n,
        n_training_samples(d, l),
        "np::train and np::n_training_samples must agree about how many rows a request buys"
    );
    let (mut opt, n_opt_tensors) = {
        // ONE optimiser over BOTH parameter sets. Two optimisers would give the event
        // weights their own moment estimates and their own schedule, which is a different
        // model rather than a wiring detail.
        let mut params = model.parameters_mut();
        if let Some(bl) = block.as_mut() {
            params.extend(bl.parameters_mut());
        }
        // The regressor block's parameters go into the SAME vector, beside the model's and
        // the event block's. A block built and forwarded but never extended into this
        // vector still contributes to the forward pass — its weights simply never move —
        // and the failure then surfaces as "the driver had no effect", which names the
        // symptom rather than the cause. `the_regressor_weights_move_under_the_optimiser`
        // is what makes the registration observable.
        if let Some(bl) = reg_block.as_mut() {
            params.extend(bl.parameters_mut());
        }
        let handed_over = params.len();
        (
            AdamW::new(params, cfg.max_lr as f32).weight_decay(cfg.weight_decay),
            handed_over,
        )
    };
    let mut log = TrainLog {
        epochs,
        batch,
        n_samples: n,
        n_params: model.n_params()
            + block.as_ref().map_or(0, EventBlock::n_params)
            + reg_block.as_ref().map_or(0, RegressorBlock::n_params),
        epoch_loss: Vec::new(),
        seconds: 0.0,
        tape_len_per_step: 0,
        steps: 0,
        train_cost: train_cost(n, epochs, l, ed, rd),
        n_event_cols: ed,
        n_opt_tensors,
        events: None,
        regressors: None,
        regressor_specs: reg_channel.map_or_else(Vec::new, |r| r.specs().to_vec()),
        n_regressor_cols: rd,
    };
    let t0 = Instant::now();
    // Shuffled over SAMPLE POSITIONS rather than over grid indices, so one chunk element
    // addresses BOTH the grid-length feature rows (through `samples[pos]`) and the
    // SAMPLE-length regressor rows (through `pos` itself) with no second lookup table and
    // no grid-shaped regressor array.
    //
    // The fit is unchanged BIT FOR BIT. `rng.shuffle` draws a permutation of `n` whatever
    // the vector holds, so the grid-index sequence is the same either way: it was
    // `order[k] = samples[perm[k]]` and it is now `samples[order[k]] = samples[perm[k]]`.
    // Both recorded neuralprophet invariance signatures reproduce, which is the evidence
    // rather than this paragraph.
    let mut order: Vec<usize> = (0..n).collect();
    // Refilled per batch rather than reallocated per batch. This buffer is on the step path
    // — `epochs * n_batches * n_lrs` times per request — and it is paid by every NP fit,
    // including the regressor-free ones that make up all existing traffic.
    let mut grid_buf: Vec<usize> = Vec::with_capacity(batch);
    let mut step = 0usize;
    for epoch in 0..epochs {
        rng.shuffle(&mut order);
        let mut acc = 0.0;
        for chunk in order.chunks(batch) {
            let p = step as f64 / total_steps as f64;
            opt.set_lr(one_cycle_lr(p, cfg.max_lr) as f32);
            let b = chunk.len();
            // The GRID rows this chunk names. Everything keyed on the daily grid — the
            // trend and seasonality features, the target, the sample weight, the lags and
            // the event indicators — reads through here; only the regressor rows read
            // through `chunk` directly, because they are the CALLER's rows (D-27).
            grid_buf.clear();
            grid_buf.extend(chunk.iter().map(|&pos| samples[pos]));
            let grid: &[usize] = &grid_buf;
            let xt = Tensor::from_vec(gather(&rows.tr, rows.td, grid), &[b, rows.td]);
            let xs = Tensor::from_vec(gather(&rows.se, rows.sd, grid), &[b, rows.sd]);
            let yt = Tensor::from_vec(grid.iter().map(|&i| rows.y[i]).collect(), &[b, 1]);
            let wt = Tensor::from_vec(grid.iter().map(|&i| rows.w[i]).collect(), &[b, 1]);
            let mut pred = if l == 0 {
                model.forward(&xt, &xs, None)
            } else {
                let lag_idx: Vec<usize> = grid.iter().flat_map(|&i| i - l..i).collect();
                let raw = Tensor::from_vec(lag_idx.iter().map(|&j| rows.y[j]).collect(), &[b, l]);
                let se_lag =
                    Tensor::from_vec(gather(&rows.se, rows.sd, &lag_idx), &[b * l, rows.sd]);
                let tr_lag_feats =
                    Tensor::from_vec(gather(&rows.tr, rows.td, &lag_idx), &[b * l, rows.td]);
                let tr_lag = no_grad(|| model.trend.forward(&tr_lag_feats).detach()).view(&[b, l]);
                let tr_lag = Tensor::from_vec(tr_lag.data().to_vec(), &[b, l]);
                model.forward(&xt, &xs, Some((&raw, &se_lag, &tr_lag)))
            };
            // The event term: ONE `.add()` on the model's own output (D-30). Note the AR
            // stationarisation above does NOT subtract the event effect at the lag times,
            // and the predict paths do not either — the two agree, which is the property
            // that matters. Subtracting it in one place and not the other would make the
            // forecast disagree with the fit.
            if let Some(bl) = block.as_ref() {
                let xe = Tensor::from_vec(gather(&ev_rows, ed, grid), &[b, ed]);
                pred = pred.add(&bl.forward(&xe));
            }
            // The regressor term: a SECOND, SEPARATE `.add()`. Not folded into the event
            // block's columns — the two have different standardisation (0/1 indicators
            // against ddof-1 standardised continuous columns) and different refusal
            // surfaces, and one shared `Linear` would make the event recovery bar and the
            // regressor behaviour inseparable.
            //
            // Gathered by `chunk` — the SAMPLE position — not by `grid`. That is the whole
            // of D-27 at the one line where it could be got wrong.
            if let Some(bl) = reg_block.as_ref() {
                let xr = Tensor::from_vec(gather(&reg_rows, rd, chunk), &[b, rd]);
                pred = pred.add(&bl.forward(&xr));
            }
            let loss = weighted_huber(&pred, &yt, &wt, cfg.huber_beta);
            acc += f64::from(loss.item()) * b as f64;
            loss.backward();
            if step == 0 {
                log.tape_len_per_step = graph_tape_len();
            }
            let mut params = model.parameters_mut();
            if let Some(bl) = block.as_mut() {
                params.extend(bl.parameters_mut());
            }
            if let Some(bl) = reg_block.as_mut() {
                params.extend(bl.parameters_mut());
            }
            opt.step_with_params(&mut params);
            opt.zero_grad();
            clear_graph();
            step += 1;
        }
        log.epoch_loss.push(acc / n as f64);
        if verbose && (epoch % 10 == 0 || epoch + 1 == epochs) {
            eprintln!(
                "  epoch {epoch:3} loss {:.5} lr {:.2e}",
                acc / n as f64,
                one_cycle_lr(step as f64 / total_steps as f64, cfg.max_lr)
            );
        }
    }
    log.seconds = t0.elapsed().as_secs_f64();
    log.steps = step;
    log.events = block;
    log.regressors = reg_block;
    (model, log)
}

/// The event channel the predict helpers take: the design that says WHICH days are events
/// and the trained block that says what an active column is WORTH.
///
/// `None` is an event-free prediction, and it is what every door path passes today —
/// `holidays` is still refused on the `"neuralprophet"` arm (plan 06.1-06 opens it).
/// Both halves are needed: weights alone cannot build an indicator row for a future day.
pub type EventChannel<'a> = Option<(&'a EventDesign, &'a EventBlock)>;

/// Add the event contribution for `days` to a `[n, 1]` prediction, inside `no_grad`.
///
/// A zero-width design is treated as absent rather than forwarded: `Tensor` shaped `[n, 0]`
/// trips trueno's empty-transpose contract deep inside `Linear::forward`, which is a
/// confusing place to learn that a caller passed an empty design.
fn add_events(p: Tensor, ev: EventChannel<'_>, days: &[i64]) -> Tensor {
    match ev {
        Some((g, b)) if g.dim() > 0 => {
            let xe = Tensor::from_vec(g.rows(days), &[days.len(), g.dim()]);
            p.add(&b.forward(&xe))
        }
        _ => p,
    }
}

/// The numeric-regressor channel the predict helpers take: the standardised rows KEYED BY
/// DAY and the trained block.
///
/// Keyed by day rather than by index because the three helpers address rows three different
/// ways — arbitrary days, grid indices, and a recursive roll-forward — and a day is the one
/// identifier all three actually hold. The rows carry the caller's `ds.len() + horizon`
/// values standardised with the constants the FIT used, so a future row's contribution uses
/// the same `mu`/`std` the training rows did.
pub type NpRegChannel<'a> = Option<(&'a NpRegressorRows, &'a RegressorBlock)>;

/// Add the numeric-regressor contribution for `days` to a `[n, 1]` prediction.
///
/// A zero-width channel is treated as absent for the reason [`add_events`] treats an empty
/// design as absent: a `[n, 0]` `Tensor` trips trueno's empty-transpose contract deep inside
/// `Linear::forward`, which is a confusing place to learn that a caller passed nothing.
fn add_regressors(p: Tensor, rc: NpRegChannel<'_>, days: &[i64]) -> Tensor {
    match rc {
        Some((rows, b)) if rows.dim() > 0 => {
            let xr = Tensor::from_vec(rows.block_rows(days), &[days.len(), rows.dim()]);
            p.add(&b.forward(&xr))
        }
        _ => p,
    }
}

/// Trend + seasonality prediction (original scale) for arbitrary days, plus the additive
/// event contribution when an [`EventChannel`] is supplied.
pub fn predict_ts(
    d: &NpData,
    m: &NpModel,
    days: &[i64],
    ev: EventChannel<'_>,
    rc: NpRegChannel<'_>,
) -> Vec<f64> {
    let rows = rows_for(d, days, &vec![0.0; days.len()], DEFAULT_END_W);
    let xt = Tensor::from_vec(rows.tr.clone(), &[days.len(), rows.td]);
    let xs = Tensor::from_vec(rows.se.clone(), &[days.len(), rows.sd]);
    let out = no_grad(|| add_regressors(add_events(m.forward(&xt, &xs, None), ev, days), rc, days));
    clear_graph();
    out.data().iter().map(|v| d.denorm(*v)).collect()
}

/// One-step-ahead AR prediction (original scale) for grid indices `idx` using true lags,
/// plus the additive event contribution for the predicted rows.
pub fn predict_ar_1step(
    d: &NpData,
    m: &NpModel,
    idx: &[usize],
    ev: EventChannel<'_>,
    rc: NpRegChannel<'_>,
) -> Vec<f64> {
    let l = m.n_lags;
    // The days `idx` names, for the event indicator rows. Total lookup, because `idx` is
    // caller-built and `grid_days` is a different field (IN-02).
    let days: Vec<i64> = idx
        .iter()
        .filter_map(|&i| d.grid_days.get(i).copied())
        .collect();
    let y_norm: Vec<f32> = d.grid_y.iter().map(|v| d.norm(*v)).collect();
    let rows = rows_for(d, &d.grid_days, &y_norm, DEFAULT_END_W);
    let b = idx.len();
    let xt = Tensor::from_vec(gather(&rows.tr, rows.td, idx), &[b, rows.td]);
    let xs = Tensor::from_vec(gather(&rows.se, rows.sd, idx), &[b, rows.sd]);
    let lag_idx: Vec<usize> = idx.iter().flat_map(|&i| i - l..i).collect();
    let raw = Tensor::from_vec(lag_idx.iter().map(|&j| rows.y[j]).collect(), &[b, l]);
    let se_lag = Tensor::from_vec(gather(&rows.se, rows.sd, &lag_idx), &[b * l, rows.sd]);
    let tr_feats = Tensor::from_vec(gather(&rows.tr, rows.td, &lag_idx), &[b * l, rows.td]);
    let out = no_grad(|| {
        let tr_lag = Tensor::from_vec(m.trend.forward(&tr_feats).data().to_vec(), &[b, l]);
        add_regressors(
            add_events(
                m.forward(&xt, &xs, Some((&raw, &se_lag, &tr_lag))),
                ev,
                &days,
            ),
            rc,
            &days,
        )
    });
    clear_graph();
    out.data().iter().map(|v| d.denorm(*v)).collect()
}

/// Trend component only (original scale) for arbitrary days.
pub fn predict_trend(d: &NpData, m: &NpModel, days: &[i64]) -> Vec<f64> {
    let rows = rows_for(d, days, &vec![0.0; days.len()], DEFAULT_END_W);
    let xt = Tensor::from_vec(rows.tr.clone(), &[days.len(), rows.td]);
    let out = no_grad(|| m.trend.forward(&xt).broadcast_add(&m.bias));
    clear_graph();
    out.data().iter().map(|v| d.denorm(*v)).collect()
}

/// Multi-step AR forecast by feeding predictions back as lags (days must continue the daily
/// grid), plus the additive event contribution for every day it rolls through.
///
/// The event term is added BEFORE the value is pushed into the recursive history, so an
/// event on a future day moves not only that day's forecast but every lagged day after it.
/// That is D-30's substance: spike 013 measured only the lag-free path.
pub fn predict_ar_recursive(
    d: &NpData,
    m: &NpModel,
    days: &[i64],
    ev: EventChannel<'_>,
    rc: NpRegChannel<'_>,
) -> Vec<f64> {
    let l = m.n_lags;
    let mut hist: Vec<f32> = d.grid_y.iter().map(|v| d.norm(*v)).collect();
    let mut hist_days: Vec<i64> = d.grid_days.clone();
    let mut out = Vec::with_capacity(days.len());
    for &day in days {
        // fill any gap between the last known day and this one recursively
        while hist_days[hist_days.len() - 1] < day {
            let next = hist_days[hist_days.len() - 1] + 1;
            let n = hist.len();
            let (mut tr, mut se) = (Vec::new(), Vec::new());
            d.row_feats(next, &mut tr, &mut se);
            let lag_days: Vec<i64> = (next - l as i64..next).collect();
            let (mut se_lag, mut tr_lag) = (Vec::new(), Vec::new());
            for &ld in &lag_days {
                d.row_feats(ld, &mut tr_lag, &mut se_lag);
            }
            let xt = Tensor::from_vec(tr, &[1, d.trend_dim()]);
            let xs = Tensor::from_vec(se, &[1, d.season_dim()]);
            let raw = Tensor::from_vec(hist[n - l..].to_vec(), &[1, l]);
            let se_l = Tensor::from_vec(se_lag, &[l, d.season_dim()]);
            let tr_feats = Tensor::from_vec(tr_lag, &[l, d.trend_dim()]);
            let v = no_grad(|| {
                let tl = Tensor::from_vec(m.trend.forward(&tr_feats).data().to_vec(), &[1, l]);
                // The regressor term is added BEFORE the value is pushed into the recursive
                // history, exactly as the event term is, so a driver on a future day moves
                // not only that day's forecast but every lagged day after it.
                add_regressors(
                    add_events(
                        m.forward(&xt, &xs, Some((&raw, &se_l, &tl))),
                        ev,
                        std::slice::from_ref(&next),
                    ),
                    rc,
                    std::slice::from_ref(&next),
                )
                .data()[0]
            });
            clear_graph();
            hist.push(v);
            hist_days.push(next);
        }
        let idx = (day - hist_days[0]) as usize;
        out.push(d.denorm(hist[idx]));
    }
    out
}

// ============================================================== C-08 cost ====
// The SC4 acceptance: the event-column term's SHAPE, and the observation that the 7.6x
// under-pricing is closed. Both numbers on the measured side are READ from the C-08
// `calibration:` mapping in `contracts/forecast-tool-boundary-v1.yaml`, never written as
// literals here — a test carrying a measured number drifts from the calibration the moment
// either is re-measured, and Phase 7 re-points that mapping at a tier without touching these
// tests.
#[cfg(test)]
mod tests {
    use super::{
        auto_epochs, door_epochs, door_lr_sweep, n_training_samples, request_train_cost,
        train_cost, NpData,
    };
    use crate::dates::days_from_civil;

    /// Read one real-valued key off the C-08 `calibration:` mapping.
    ///
    /// Panics NAMING THE KEY if it is absent: a bar or a measurement that silently defaulted
    /// to something plausible is the vacuous-guard class this crate's tests refuse.
    fn c08_calibration_f64(key: &str) -> f64 {
        let doc = crate::test_support::contract_value("forecast-tool-boundary-v1");
        let axes = doc
            .get("door_surface")
            .and_then(|d| d.get("cost_axes"))
            .and_then(serde_yaml::Value::as_sequence)
            .expect("forecast-tool-boundary-v1 must carry door_surface.cost_axes");
        let c08 = axes
            .iter()
            .find(|a| a.get("axis").and_then(serde_yaml::Value::as_str) == Some("C-08"))
            .expect("forecast-tool-boundary-v1 must carry cost axis C-08");
        let cal = c08
            .get("calibration")
            .expect("cost axis C-08 must carry a calibration: MAPPING (D-34)");
        let v = cal
            .get(key)
            .unwrap_or_else(|| panic!("C-08 calibration: must define {key}"));
        v.as_f64()
            .or_else(|| {
                #[allow(clippy::cast_precision_loss)]
                v.as_u64().map(|n| n as f64)
            })
            .unwrap_or_else(|| panic!("C-08 calibration.{key} must be a number"))
    }

    /// A contiguous daily series of `points` points, the calibration geometry.
    fn daily(points: usize) -> NpData {
        let t0 = days_from_civil(2018, 1, 1);
        let days: Vec<i64> = (0..points).map(|i| t0 + i as i64).collect();
        let y: Vec<f64> = (0..points)
            .map(|i| {
                let t = i as f64;
                10.0 + 0.01 * t + (2.0 * std::f64::consts::PI * t / 7.0).sin()
            })
            .collect();
        NpData::new(&days, &y, points, 10, 0.8)
    }

    /// At ZERO event columns the priced cost is BIT-IDENTICAL to the pre-event arithmetic.
    ///
    /// The pre-event formula was `epochs * n_samples * (n_lags + 1)`, written out on the
    /// right-hand side below, plus four values computed from it BEFORE this task for the
    /// geometries the ladder and the bound derivation actually use. Both halves matter: the
    /// identity catches a structural change, the literals catch an identity that was
    /// rewritten to agree with a changed implementation.
    #[test]
    fn the_priced_cost_at_zero_event_columns_is_unchanged() {
        for n_samples in [10usize, 1_200, 2_540, 19_993] {
            for epochs in [20usize, 50, 80, 320] {
                for n_lags in [0usize, 7, 30, 365] {
                    assert_eq!(
                        train_cost(n_samples, epochs, n_lags, 0, 0),
                        (epochs as u64) * (n_samples as u64) * (n_lags as u64 + 1),
                        "train_cost at E = 0 must equal the pre-event arithmetic exactly \
                         (n_samples={n_samples} epochs={epochs} n_lags={n_lags})"
                    );
                }
            }
        }
        // Values computed from the PRE-EVENT formula, so a rewritten identity above cannot
        // agree with a changed implementation: `np::parity`'s Peyton rungs and the three
        // at-the-bound compositions `MAX_NP_TRAIN_COST` was derived from.
        for (n_samples, epochs, n_lags, expected) in [
            (2_905usize, 80usize, 0usize, 232_400u64),
            (2_934, 80, 30, 7_276_320),
            (19_994, 50, 6, 6_997_900),
            (9_989, 80, 11, 9_589_440),
        ] {
            assert_eq!(
                train_cost(n_samples, epochs, n_lags, 0, 0),
                expected,
                "the event-free price of a geometry this crate already derived bounds from \
                 must not have moved"
            );
        }
    }

    /// The event term rises, is rounded UP, and never under-prices a fractional column.
    #[test]
    fn the_event_width_is_rounded_up_and_monotone() {
        let c = crate::types::FIT_NP_EVENT_COST_PER_COLUMN;
        let (n, e, l) = (1_000usize, 10usize, 0usize);
        let base = train_cost(n, e, l, 0, 0);
        let mut prev = base;
        for cols in [0usize, 1, 10, 100, 500, 1_000] {
            let got = train_cost(n, e, l, cols, 0);
            assert!(
                got >= prev,
                "the priced cost must never FALL as event columns are added: {cols} columns \
                 priced {got} against {prev} at the previous point"
            );
            // The width the price implies, recovered by division, must be at or above the
            // exact real-valued width. Below it is an under-price by a fractional column.
            let implied_width = got as f64 / (e as f64 * n as f64);
            let exact_width = (l as f64 + 1.0) + c * cols as f64;
            assert!(
                implied_width >= exact_width - 1e-9,
                "at {cols} event columns the priced width {implied_width} is BELOW the exact \
                 {exact_width}: the door is under-pricing by a fractional column"
            );
            prev = got;
        }
        // ONE column must already cost something, or the `ceil` is not doing its job and a
        // caller could add columns for free up to 1/c of them.
        assert!(
            train_cost(n, e, l, 1, 0) > base,
            "a single event column must raise the price: the width contribution is rounded \
             UP, so even {c} of a unit becomes a whole one"
        );
    }

    /// SC4: the 7.6x under-pricing is OBSERVED closed, as a RATIO.
    ///
    /// # Why a ratio, and not a comparison of the two numbers
    ///
    /// The priced cost is a unit-less integer proxy; the measured work is microseconds per
    /// step. There is no defensible conversion between them, and inventing one would make
    /// this test a statement about the conversion. What IS well defined — and what closure
    /// actually means — is that the priced cost must grow AT LEAST AS FAST as the measured
    /// cost as event columns are added:
    ///
    /// ```text
    /// measured(E) / measured(0)  <=  priced(E) / priced(0)
    /// ```
    ///
    /// Both ratios are dimensionless, both are taken at the same geometry, and the batch
    /// size, the step count and the sample count all cancel out of both sides.
    ///
    /// ONE-SIDED on purpose: over-pricing is safe, under-pricing is the defect.
    ///
    /// Both measured numbers are READ from the C-08 `calibration:` mapping. A test that
    /// recomputed the priced formula on both sides would prove the formula equals itself; a
    /// test carrying the measured number as a literal would drift from the calibration the
    /// moment either was re-measured.
    /// Asserted on EVERY architecture the calibration records, not only on the one the
    /// constant was derived from. The deriving host (aarch64) carries the unsuffixed keys;
    /// each confirming host carries `<key>_<arch>`. A confirmation that no test reads is a
    /// sentence rather than a confirmation.
    #[test]
    fn the_event_column_underpricing_is_observed_closed() {
        let e_max = c08_calibration_f64("measured_at_e_max") as usize;
        let n_lags_cal = c08_calibration_f64("n_lags_cal") as usize;
        // The calibration geometry, so both sides describe the same request shape. The
        // priced side does not vary by architecture — it is the door's arithmetic — so it is
        // computed once and compared against every measured side.
        let points = 1_200usize;
        let d = daily(points);
        let priced_at_zero = request_train_cost(&d, points, n_lags_cal, 0, 0);
        let priced_at_max = request_train_cost(&d, points, n_lags_cal, e_max, 0);
        let priced_ratio = priced_at_max as f64 / priced_at_zero as f64;

        let hosts: [(&str, &str, &str); 2] = [
            (
                "aarch64 (deriving)",
                "measured_us_per_step_at_e0",
                "measured_us_per_step_at_e_max",
            ),
            (
                "x86_64 (confirming)",
                "measured_us_per_step_at_e0_x86_64",
                "measured_us_per_step_at_e_max_x86_64",
            ),
        ];
        // VACUITY GUARD: a list silently shrunk to one host is this test quietly going back
        // to being single-architecture.
        assert_eq!(
            hosts.len(),
            2,
            "the closure is asserted on the deriving host AND on every confirming host"
        );

        for (what, e0_key, emax_key) in hosts {
            let us_at_zero = c08_calibration_f64(e0_key);
            let us_at_max = c08_calibration_f64(emax_key);
            assert!(
                us_at_zero > 0.0 && us_at_max > 0.0 && e_max > 0,
                "{what}: the calibration must carry positive measurements at both ends, got \
                 {us_at_zero} / {us_at_max} at E = {e_max}"
            );
            let measured_ratio = us_at_max / us_at_zero;
            let headroom = priced_ratio - measured_ratio;
            println!(
                "C-08 closure on {what} at E = {e_max} (n_lags_cal = {n_lags_cal}): measured \
                 ratio {measured_ratio:.4} ({us_at_max} / {us_at_zero} us per step) vs priced \
                 ratio {priced_ratio:.4} ({priced_at_max} / {priced_at_zero}); headroom \
                 {headroom:.4}"
            );
            assert!(
                measured_ratio > 1.0,
                "{what}: the measured cost must RISE with the event-column count, or there \
                 was never an under-pricing to close and this test proves nothing: measured \
                 ratio {measured_ratio}"
            );
            assert!(
                measured_ratio <= priced_ratio,
                "{what}: UNDER-PRICED at {e_max} event columns — the measured work grows \
                 {measured_ratio:.4}x while the price grows only {priced_ratio:.4}x. A \
                 request at the holiday-column ceiling would buy more work than the door \
                 priced it at: the SC4 defect, still open on this architecture."
            );
        }
    }

    /// The SHAPE discriminator: the priced ratio is the SAME at two geometries differing
    /// roughly 10x in samples.
    ///
    /// This is the test that tells the correct shape from the one a looser wording permits.
    /// With the event term INSIDE the per-sample width,
    /// `priced(E)/priced(0) = ((n_lags + 1) + ceil(c*E)) / (n_lags + 1)` — free of epochs and
    /// samples entirely. With `product + c*E` the ratio is `1 + c*E/product`, which collapses
    /// toward 1 as the product grows, so a coefficient tuned to pass an observation at one
    /// geometry under-prices every larger series.
    ///
    /// A SINGLE-geometry observation cannot tell the two apart, which is precisely how a
    /// passing test could have left the 7.6x open.
    #[test]
    fn the_event_cost_shape_scales_with_geometry() {
        /// EXACTLY two, and the count is carried in the failure message: a sweep silently
        /// shrunk to one geometry is this test quietly becoming unable to discriminate.
        const GEOMETRIES: usize = 2;
        let points = [1_200usize, 12_000];
        assert_eq!(
            points.len(),
            GEOMETRIES,
            "this test discriminates the two shapes ONLY by comparing across geometries; \
             with {GEOMETRIES} it is a shape test, with one it is a tautology"
        );
        let e_max = c08_calibration_f64("measured_at_e_max") as usize;

        // The two geometries really do differ ~10x in samples AND in the epoch budget, so a
        // ratio that survives both is not surviving a pair of near-identical requests.
        let samples: Vec<usize> = points
            .iter()
            .map(|p| n_training_samples(&daily(*p), 0))
            .collect();
        assert!(
            samples[1] >= samples[0] * 8,
            "the two geometries must differ by roughly 10x in samples: {samples:?}"
        );
        assert_ne!(
            auto_epochs(samples[0]),
            auto_epochs(samples[1]),
            "the two geometries must also differ in the epoch budget, or a shape that \
             depends on epochs would survive this test"
        );

        for n_lags in [0usize, 7] {
            let mut ratios: Vec<f64> = Vec::new();
            for &p in &points {
                let d = daily(p);
                let at_zero = request_train_cost(&d, p, n_lags, 0, 0);
                let at_max = request_train_cost(&d, p, n_lags, e_max, 0);
                let ratio = at_max as f64 / at_zero as f64;
                println!(
                    "shape: points={p} n_lags={n_lags} samples={} epochs={} sweep={} \
                     priced(0)={at_zero} priced({e_max})={at_max} ratio={ratio:.6}",
                    n_training_samples(&d, n_lags),
                    door_epochs(p, n_training_samples(&d, n_lags), n_lags),
                    door_lr_sweep(n_lags).len()
                );
                ratios.push(ratio);
            }
            assert_eq!(ratios.len(), GEOMETRIES);
            assert!(
                (ratios[0] - ratios[1]).abs() < 1e-12,
                "at n_lags={n_lags} the priced ratio is {} at {} points and {} at {} points. \
                 The event term is therefore NOT inside the per-sample width: a ratio that \
                 moves with the sample count is the `product + c*E` shape, which scales with \
                 neither epochs nor samples and under-prices every larger series.",
                ratios[0],
                points[0],
                ratios[1],
                points[1]
            );
            // ...and it is the CLOSED FORM the width shape predicts, not merely stable.
            let width_at_zero = n_lags as f64 + 1.0;
            let expected = (width_at_zero
                + (crate::types::FIT_NP_EVENT_COST_PER_COLUMN * e_max as f64).ceil())
                / width_at_zero;
            assert!(
                (ratios[0] - expected).abs() < 1e-12,
                "at n_lags={n_lags} the priced ratio {} is not the width form's closed form \
                 {expected}",
                ratios[0]
            );
        }

        // Separately: a differing `n_lags` MUST move the ratio, because the event term is
        // additive to `(n_lags + 1)` and therefore a smaller share of a wider sweep. A ratio
        // that ignored `n_lags` would mean the term had escaped the width in the other
        // direction.
        let d = daily(points[0]);
        let r0 = request_train_cost(&d, points[0], 0, e_max, 0) as f64
            / request_train_cost(&d, points[0], 0, 0, 0) as f64;
        let r7 = request_train_cost(&d, points[0], 7, e_max, 0) as f64
            / request_train_cost(&d, points[0], 7, 0, 0) as f64;
        println!("shape: ratio at n_lags=0 is {r0:.6}, at n_lags=7 is {r7:.6}");
        assert!(
            r7 < r0,
            "the priced ratio must SHRINK as n_lags grows ({r7} at 7 lags against {r0} at 0): \
             the event term is additive to the per-sample width, so it is a smaller share of \
             a wider one"
        );
    }

    /// SC4 for NUMERIC REGRESSORS: the per-epoch work a regressor column buys is OBSERVED
    /// priced, as a RATIO, on the same shape plan 06.1-05 used for event columns.
    ///
    /// ```text
    /// measured(R) / measured(0)  <=  priced(R) / priced(0)
    /// ```
    ///
    /// Both measured numbers are READ from the C-08 `calibration:` mapping. A test that
    /// recomputed the priced formula on both sides would prove the formula equals itself; a
    /// test carrying the measured number as a literal would drift from the calibration the
    /// moment either was re-measured.
    ///
    /// Asserted on every architecture the regressor calibration records. TODAY that is ONE
    /// — aarch64 — and that is stated rather than hidden: the event axis carries an x86_64
    /// confirmation and this one does not yet. The host list below is asserted non-empty so
    /// a silently emptied list cannot pass by checking nothing, and it is the place a second
    /// host is added.
    #[test]
    fn the_regressor_column_underpricing_is_observed_closed() {
        let r_max = c08_calibration_f64("measured_at_r_max") as usize;
        let n_lags_cal = c08_calibration_f64("regressor_n_lags_cal") as usize;
        let points = 1_200usize;
        let d = daily(points);
        let priced_at_zero = request_train_cost(&d, points, n_lags_cal, 0, 0);
        let priced_at_max = request_train_cost(&d, points, n_lags_cal, 0, r_max);
        let priced_ratio = priced_at_max as f64 / priced_at_zero as f64;

        let hosts: [(&str, &str, &str); 1] = [(
            "aarch64 (deriving)",
            "measured_us_per_step_at_r0",
            "measured_us_per_step_at_r_max",
        )];
        assert!(
            !hosts.is_empty(),
            "vacuity guard: the closure must be asserted on at least the deriving host"
        );

        for (what, r0_key, rmax_key) in hosts {
            let us_at_zero = c08_calibration_f64(r0_key);
            let us_at_max = c08_calibration_f64(rmax_key);
            assert!(
                us_at_zero > 0.0 && us_at_max > 0.0 && r_max > 0,
                "{what}: the calibration must carry positive measurements at both ends, got \
                 {us_at_zero} / {us_at_max} at R = {r_max}"
            );
            let measured_ratio = us_at_max / us_at_zero;
            let headroom = priced_ratio - measured_ratio;
            println!(
                "C-08 regressor closure on {what} at R = {r_max} (n_lags_cal = \
                 {n_lags_cal}): measured ratio {measured_ratio:.4} ({us_at_max} / \
                 {us_at_zero} us per step) vs priced ratio {priced_ratio:.4} \
                 ({priced_at_max} / {priced_at_zero}); headroom {headroom:.4}"
            );
            assert!(
                measured_ratio > 1.0,
                "{what}: the measured cost must RISE with the regressor-column count, or \
                 there was never an under-pricing to close and this test proves nothing: \
                 measured ratio {measured_ratio}"
            );
            assert!(
                measured_ratio <= priced_ratio,
                "{what}: UNDER-PRICED at {r_max} regressor columns — the measured work grows \
                 {measured_ratio:.4}x while the price grows only {priced_ratio:.4}x. A \
                 request at the regressor count ceiling would buy more work than the door \
                 priced it at: the SC4 defect, one column family over."
            );
        }

        // ---- THE COMBINED CASE: a caller can send BOTH, and the two terms must ADD ----
        //
        // Not mask each other. Without this a price that took the MAXIMUM of the two terms,
        // or that dropped one whenever the other was present, would pass every single-axis
        // assertion above and under-price every request carrying both.
        let e_max = c08_calibration_f64("measured_at_e_max") as usize;
        let e_only = request_train_cost(&d, points, n_lags_cal, e_max, 0);
        let r_only = request_train_cost(&d, points, n_lags_cal, 0, r_max);
        let both = request_train_cost(&d, points, n_lags_cal, e_max, r_max);
        println!(
            "C-08 combined: priced(0,0)={priced_at_zero} priced(E={e_max},0)={e_only} \
             priced(0,R={r_max})={r_only} priced(E,R)={both}"
        );
        assert!(
            both > e_only && both > r_only,
            "a request carrying BOTH must price above either alone: {both} against \
             {e_only} / {r_only}"
        );
        // Exactly additive in the WIDTH, which is the shape claim: the combined width is
        // the base plus both rounded-up contributions.
        let base_width = n_lags_cal as u64 + 1;
        let expected_width = base_width
            + (crate::types::FIT_NP_EVENT_COST_PER_COLUMN * e_max as f64).ceil() as u64
            + (crate::types::FIT_NP_REGRESSOR_COST_PER_COLUMN * r_max as f64).ceil() as u64;
        assert_eq!(
            both,
            priced_at_zero / base_width * expected_width,
            "the two terms must ADD inside the per-sample width; expected width \
             {expected_width} units against the base {base_width}"
        );
    }

    /// The SHAPE discriminator for the REGRESSOR term, separate from the event one.
    ///
    /// A shape argument that is only ever checked on one of two terms is an argument about
    /// the other by analogy, which is not evidence. With the term INSIDE the per-sample
    /// width the ratio `priced(R)/priced(0)` is free of epochs and samples entirely; with
    /// `product + c*R` it collapses toward 1 as the product grows, so a coefficient tuned to
    /// pass an observation at one geometry under-prices every larger series.
    #[test]
    fn the_regressor_cost_shape_scales_with_geometry() {
        const GEOMETRIES: usize = 2;
        let points = [1_200usize, 12_000];
        let r_max = c08_calibration_f64("measured_at_r_max") as usize;
        let samples: Vec<usize> = points
            .iter()
            .map(|&p| n_training_samples(&daily(p), 0))
            .collect();
        assert_eq!(samples.len(), GEOMETRIES);
        assert!(
            samples[1] >= samples[0] * 5,
            "the two geometries must differ by at least 5x in samples, or this test cannot \
             discriminate the two shapes: {samples:?}"
        );
        assert_ne!(
            auto_epochs(samples[0]),
            auto_epochs(samples[1]),
            "the two geometries must also differ in the epoch budget"
        );

        for n_lags in [0usize, 7] {
            let mut ratios: Vec<f64> = Vec::new();
            for &p in &points {
                let d = daily(p);
                let at_zero = request_train_cost(&d, p, n_lags, 0, 0);
                let at_max = request_train_cost(&d, p, n_lags, 0, r_max);
                ratios.push(at_max as f64 / at_zero as f64);
            }
            assert_eq!(ratios.len(), GEOMETRIES);
            assert!(
                (ratios[0] - ratios[1]).abs() < 1e-12,
                "at n_lags={n_lags} the priced regressor ratio is {} at {} points and {} at \
                 {} points. The regressor term is therefore NOT inside the per-sample width.",
                ratios[0],
                points[0],
                ratios[1],
                points[1]
            );
            let width_at_zero = n_lags as f64 + 1.0;
            let expected = (width_at_zero
                + (crate::types::FIT_NP_REGRESSOR_COST_PER_COLUMN * r_max as f64).ceil())
                / width_at_zero;
            assert!(
                (ratios[0] - expected).abs() < 1e-12,
                "at n_lags={n_lags} the priced regressor ratio {} is not the width form's \
                 closed form {expected}",
                ratios[0]
            );
        }
    }

    /// The REGRESSOR width rises, is rounded UP, and at `R = 0` the price is BIT-IDENTICAL
    /// to plan 06.1-05's arithmetic.
    ///
    /// The bit-identity half is what keeps every bound this crate derived from the
    /// regressor-free formula — `MAX_NP_TRAIN_COST`'s three at-the-bound compositions and
    /// `np::parity`'s own Peyton geometries — meaning what they meant before this term
    /// existed.
    #[test]
    fn the_regressor_width_is_rounded_up_and_the_zero_price_is_unchanged() {
        let c = crate::types::FIT_NP_REGRESSOR_COST_PER_COLUMN;
        let (n, e, l) = (1_000usize, 10usize, 0usize);
        let base = train_cost(n, e, l, 0, 0);
        let mut prev = base;
        for cols in [0usize, 1, 10, 100, crate::types::MAX_REGRESSORS] {
            let got = train_cost(n, e, l, 0, cols);
            assert!(
                got >= prev,
                "the priced cost must never FALL as regressor columns are added: {cols} \
                 columns priced {got} against {prev}"
            );
            let implied_width = got as f64 / (e as f64 * n as f64);
            let exact_width = (l as f64 + 1.0) + c * cols as f64;
            assert!(
                implied_width >= exact_width - 1e-9,
                "at {cols} regressor columns the priced width {implied_width} is BELOW the \
                 exact {exact_width}: the door is under-pricing by a fractional column"
            );
            prev = got;
        }
        assert!(
            train_cost(n, e, l, 0, 1) > base,
            "a single regressor column must raise the price: the width contribution is \
             rounded UP, so even {c} of a unit becomes a whole one"
        );

        // BIT-IDENTICAL at R = 0 AND E = 0, on the very geometries this crate derived bounds
        // from. Values computed from the PRE-REGRESSOR (and pre-event) formula.
        for (n_samples, epochs, n_lags, expected) in [
            (2_905usize, 80usize, 0usize, 232_400u64),
            (2_934, 80, 30, 7_276_320),
            (19_994, 50, 6, 6_997_900),
            (9_989, 80, 11, 9_589_440),
        ] {
            assert_eq!(
                train_cost(n_samples, epochs, n_lags, 0, 0),
                expected,
                "the regressor-free AND event-free price of a geometry this crate already \
                 derived bounds from must not have moved"
            );
        }
    }
}

// ================== D-27: the four-rule falsification probe (SC5) ==================
//
// Spike 014 measured a gappy daily series — 790 observed rows on an 899-day grid, 109
// imputed (12.1 %), scale 35.32 — and filled the imputed days' regressor values four ways:
// linear interpolation, zero, carry-forward, and a deliberate garbage magnitude. At
// `n_lags = 0` all four were BIT-IDENTICAL; at `n_lags = 7` two DEFENSIBLE rules were 10.48
// apart (~30 % of scale) and the garbage rule moved the forecast by 192 and flipped the sign
// of BOTH coefficients.
//
// # The probe's STATUS has changed, and this is where that is stated
//
// Before D-27 this table WAS the guarantee. After D-27 it CONFIRMS a structural fact: there
// is no grid-shaped regressor array, so there is no slot for a fill rule to write into. D-27
// rejected keeping the probe as the sole evidence precisely because a later refactor that
// started reading the slot would go green until someone re-derived it — and it keeps the
// probe anyway, because a structural claim with nothing able to DETECT its violation is the
// shape this project has been withdrawing.
//
// # Why the naive port would be a tautology
//
// With no grid-shaped array to fill, four notionally different fills become four IDENTICAL
// calls: the test would compare a function against itself and pass on any implementation.
// So the four inputs are made genuinely different AT THE BOUNDARY the test probes — four
// grid-length vectors — and pushed through [`super::select_grid_values`], the one place the
// caller-row selection happens. The structural claim is then an EQUALITY BETWEEN THE
// SELECTION'S OUTPUT AND THE CALLER'S INPUT, asserted before any training runs.
#[cfg(test)]
mod d27_probe {
    use super::{
        n_training_samples, predict_ts, select_grid_values, train, training_samples, NpData,
        TrainConfig,
    };
    use crate::dates::days_from_civil;
    use crate::prophet::Mode;
    use crate::regressors::{standardize_one, NpRegressors, RegressorBlock, RegressorSpec};

    /// EXACTLY four rules, and the count is asserted: a probe that silently lost a rule
    /// would report the same green as a complete one.
    const RULES: usize = 4;
    const RULE_NAMES: [&str; RULES] = ["linear", "zero", "carry-forward", "garbage 1e3"];
    /// The deliberate garbage magnitude, far outside the driver's own scale (which is
    /// `2 ± 3`). It is the reason a green column cannot be mistaken for a broken harness.
    const GARBAGE: f64 = 1e3;
    const HORIZON: usize = 7;

    /// A gappy daily series in spike 014's regime: 180 calendar days with 31 removed
    /// (17.2 % of the grid). Returns `(days, y, the caller's driver array of len == points)`.
    fn probe_series() -> (Vec<i64>, Vec<f64>, Vec<f64>) {
        let t0 = days_from_civil(2021, 1, 1);
        let (mut days, mut y, mut driver) = (Vec::new(), Vec::new(), Vec::new());
        for k in 0..180i64 {
            if k % 8 == 3 || k % 23 == 7 {
                continue;
            }
            let t = k as f64;
            days.push(t0 + k);
            y.push(20.0 + 0.05 * t + (2.0 * std::f64::consts::PI * t / 7.0).sin());
            driver.push(2.0 + (t * 0.37).sin() * 3.0);
        }
        (days, y, driver)
    }

    /// Build a GRID-LENGTH value vector: the caller's real values at the observed positions,
    /// and `rule`'s fill at the imputed ones.
    ///
    /// These are TEST INPUTS. Nothing here is a shipped code path, which is why the
    /// no-interpolation source guard excludes `#[cfg(test)]` code by path — see
    /// `no_shipped_regressor_path_interpolates_a_value`.
    fn grid_with_rule(d: &NpData, caller: &[f64], rule: usize) -> Vec<f64> {
        let n = d.n_train_grid;
        let mut g = vec![f64::NAN; n];
        let mut c = 0usize;
        for i in 0..n {
            if d.grid_observed[i] {
                g[i] = caller[c];
                c += 1;
            }
        }
        for i in 0..n {
            if !g[i].is_nan() {
                continue;
            }
            g[i] = match rule {
                // rule 0: linear interpolation between the neighbouring OBSERVED values.
                // A binary driver would come out at a "half promo" day under this rule,
                // which is exactly why no shipped path may do it.
                0 => {
                    let a = (0..i).rev().find(|&j| d.grid_observed[j]);
                    let b = (i + 1..n).find(|&j| d.grid_observed[j]);
                    match (a, b) {
                        (Some(a), Some(b)) => {
                            let f = (i - a) as f64 / (b - a) as f64;
                            g[a] + f * (g[b] - g[a])
                        }
                        (Some(a), None) => g[a],
                        (None, Some(b)) => g[b],
                        (None, None) => 0.0,
                    }
                }
                1 => 0.0,
                2 => (0..i)
                    .rev()
                    .find(|&j| d.grid_observed[j])
                    .map_or(0.0, |j| g[j]),
                3 => GARBAGE,
                _ => unreachable!("RULES is {RULES}"),
            };
        }
        g
    }

    /// Fit on `hist` (a CALLER-length driver array) and return `(yhat bits, weight bits)`.
    fn fit(d: &NpData, days: &[i64], hist: &[f64]) -> (Vec<u64>, Vec<u32>) {
        let last = days[days.len() - 1];
        let fut: Vec<i64> = (1..=HORIZON as i64).map(|k| last + k).collect();
        let futv: Vec<f64> = (0..HORIZON)
            .map(|i| 2.0 + ((days.len() + i) as f64 * 0.37).sin() * 3.0)
            .collect();
        let spec = RegressorSpec {
            name: "price".into(),
            mode: Mode::Additive,
            prior_scale: 10.0,
            standardize: None,
        };
        let regs = NpRegressors::new(
            d,
            0,
            vec![standardize_one(&spec, hist)],
            vec![hist.to_vec()],
            vec![futv],
        )
        .expect("a lag-free channel is always aligned");
        let cfg = TrainConfig {
            n_lags: 0,
            ar_layers: vec![],
            max_lr: 0.03,
            epochs: Some(12),
            batch: Some(64),
            weight_decay: 1e-3,
            huber_beta: 0.3,
            newer_w: 2.0,
            seed: 42,
            event_design: None,
            regressors: Some(regs.clone()),
        };
        let (m, log) = train(d, &cfg, false);
        let rows = regs.rows_for(days, &fut);
        let block = log
            .regressors
            .as_ref()
            .expect("a regressor block must have been trained");
        let yhat = predict_ts(d, &m, &fut, None, Some((&rows, block)));
        (
            yhat.iter().map(|v| v.to_bits()).collect(),
            RegressorBlock::weights(block)
                .iter()
                .map(|w| w.to_bits())
                .collect(),
        )
    }

    /// SC5, the `n_lags = 0` half: the imputed day's regressor value is UNREAD.
    ///
    /// Two claims, in order. The STRUCTURAL one — the four selections all return the
    /// CALLER'S OWN ARRAY, byte for byte — is the statement D-27 actually makes, and it is
    /// asserted before any model is fitted. The BEHAVIOURAL one — bit-identical predictions
    /// and weights — follows from it and is asserted anyway, because that is what a
    /// downstream reader can check without reading this module.
    #[test]
    fn the_four_fill_rules_are_bit_identical_at_zero_lags() {
        let (days, y, caller) = probe_series();
        let d = NpData::new(&days, &y, days.len(), 10, 0.8);
        // The PREMISE, measured rather than assumed.
        let imputed: Vec<usize> = (0..d.n_train_grid)
            .filter(|&i| !d.grid_observed[i])
            .collect();
        println!(
            "D-27 probe: {} caller rows, {} grid rows, {} imputed ({:.1} %)",
            days.len(),
            d.n_train_grid,
            imputed.len(),
            100.0 * imputed.len() as f64 / d.n_train_grid as f64
        );
        assert!(
            !imputed.is_empty(),
            "the probe needs a GAPPY series or it probes nothing"
        );

        let (samples, caller_rows) = training_samples(&d, 0);
        assert_eq!(samples.len(), n_training_samples(&d, 0));
        assert_eq!(
            caller_rows,
            (0..days.len()).collect::<Vec<usize>>(),
            "the lag-free selection must name the caller's rows"
        );

        let grids: Vec<Vec<f64>> = (0..RULES).map(|r| grid_with_rule(&d, &caller, r)).collect();
        assert_eq!(
            grids.len(),
            RULES,
            "the probe exercises EXACTLY {RULES} rules ({RULE_NAMES:?}); a probe that \
             silently lost one would report the same green as a complete one"
        );
        // The four INPUTS really do differ, or the probe is four copies of one thing.
        for r in 1..RULES {
            assert!(
                imputed
                    .iter()
                    .any(|&i| grids[r][i].to_bits() != grids[0][i].to_bits()),
                "rule {} ({}) must differ from rule 0 somewhere on the imputed days",
                r,
                RULE_NAMES[r]
            );
        }

        // ---- THE STRUCTURAL CLAIM ----
        let selected: Vec<Vec<f64>> = grids
            .iter()
            .map(|g| select_grid_values(g, &samples))
            .collect();
        for (r, s) in selected.iter().enumerate() {
            assert_eq!(
                s.len(),
                caller.len(),
                "rule {}: the selection returns one value per CALLER row",
                RULE_NAMES[r]
            );
            for (i, v) in s.iter().enumerate() {
                assert_eq!(
                    v.to_bits(),
                    caller[i].to_bits(),
                    "rule {} row {i}: the selection must return the CALLER'S OWN value. This \
                     is D-27 stated as an equality between the selection's output and the \
                     caller's input — the fill rule is unreachable, not merely ignored.",
                    RULE_NAMES[r]
                );
            }
        }

        // ---- THE BEHAVIOURAL HALF ----
        let fits: Vec<(Vec<u64>, Vec<u32>)> = selected.iter().map(|h| fit(&d, &days, h)).collect();
        for r in 1..RULES {
            assert_eq!(
                fits[r].0, fits[0].0,
                "rule {} changed the PREDICTIONS against rule 0 at n_lags = 0",
                RULE_NAMES[r]
            );
            assert_eq!(
                fits[r].1, fits[0].1,
                "rule {} changed the learned WEIGHTS against rule 0 at n_lags = 0",
                RULE_NAMES[r]
            );
        }
    }

    /// THE FALSIFICATION CONTROL: the same four rules through a deliberately WRONG,
    /// GRID-INDEXED selection produce DIFFERING results.
    ///
    /// Without it a probe that cannot distinguish a right selection from a wrong one reports
    /// the same green as a correct one. The wrong selection here is the plausible mistake
    /// the shipped code could regress into: treating the grid POSITION as the sample
    /// position (`0..n` rather than the observed rows), which picks up imputed days and
    /// therefore the fill rule.
    ///
    /// The observed difference is PRINTED, so the SUMMARY records a measurement rather than
    /// the fact that an assertion passed.
    #[test]
    fn the_four_fill_rules_differ_under_a_wrong_grid_indexed_selection() {
        let (days, y, caller) = probe_series();
        let d = NpData::new(&days, &y, days.len(), 10, 0.8);
        let n = caller.len();
        let wrong_rows: Vec<usize> = (0..n).collect();

        let grids: Vec<Vec<f64>> = (0..RULES).map(|r| grid_with_rule(&d, &caller, r)).collect();
        let selected: Vec<Vec<f64>> = grids
            .iter()
            .map(|g| select_grid_values(g, &wrong_rows))
            .collect();

        // The wrong selection must actually pick up imputed days, or the control is vacuous.
        let mut differing_inputs = 0usize;
        for r in 1..RULES {
            if selected[r]
                .iter()
                .zip(&selected[0])
                .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                differing_inputs += 1;
            }
        }
        assert_eq!(
            differing_inputs,
            RULES - 1,
            "CONTROL FAILED: the grid-indexed selection did not pick up the imputed days, so \
             this control proves nothing about the correct one"
        );

        let fits: Vec<(Vec<u64>, Vec<u32>)> = selected.iter().map(|h| fit(&d, &days, h)).collect();
        let base: Vec<f64> = fits[0].0.iter().map(|b| f64::from_bits(*b)).collect();
        let mut any_differ = false;
        for r in 1..RULES {
            let got: Vec<f64> = fits[r].0.iter().map(|b| f64::from_bits(*b)).collect();
            let max_abs = got
                .iter()
                .zip(&base)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            let w_before: Vec<f32> = fits[0].1.iter().map(|b| f32::from_bits(*b)).collect();
            let w_after: Vec<f32> = fits[r].1.iter().map(|b| f32::from_bits(*b)).collect();
            println!(
                "D-27 falsification control: rule {} under the WRONG grid-indexed selection \
                 moves the forecast by max |delta| = {max_abs:.6} (weights {w_before:?} -> \
                 {w_after:?})",
                RULE_NAMES[r]
            );
            if max_abs > 0.0 || fits[r].1 != fits[0].1 {
                any_differ = true;
            }
        }
        assert!(
            any_differ,
            "the probe cannot tell a WRONG selection from the right one: four genuinely \
             different fills produced identical fits through a grid-indexed selection, so a \
             green column in the sibling test would mean nothing"
        );
    }

    /// SC5's OTHER half: at `n_lags > 0` the chosen rule — here the REFUSAL — actually
    /// fires, on the SAME gappy series the four-rule probe proves is safe at zero lags.
    ///
    /// SC5 asks for both: a test at `n_lags = 0` proving the imputed-day value is unread,
    /// and a test at `n_lags > 0` proving the chosen rule fires. The probe alone satisfies
    /// only the first, and a phase that shipped only the first would have decided the lagged
    /// case in prose. Sharing the fixture is what makes "the same series" literal rather
    /// than approximate.
    #[test]
    fn the_gap_refusal_actually_fires_at_lags() {
        let (days, y, caller) = probe_series();
        let ds: Vec<String> = days.iter().map(|d| crate::dates::format_ymd(*d)).collect();
        let futv: Vec<f64> = (0..HORIZON)
            .map(|i| 2.0 + ((days.len() + i) as f64 * 0.37).sin() * 3.0)
            .collect();
        let mut values = caller.clone();
        values.extend(futv);
        let args = crate::types::ForecastArgs {
            ds,
            y,
            horizon: HORIZON,
            model: Some("neuralprophet".into()),
            freq: Some("D".into()),
            n_lags: Some(7),
            seed: Some(42),
            regressors: Some(vec![crate::types::RegressorArg {
                name: "price".into(),
                values,
                mode: None,
                prior_scale: None,
                standardize: None,
            }]),
            ..crate::types::ForecastArgs::default()
        };
        match crate::forecast::forecast(&args) {
            Err(crate::types::ForecastError::Validation(m)) => {
                assert!(
                    m.contains("Supply a gap-free daily series, or set n_lags to 0"),
                    "the at-lags refusal must name the fix, got {m:?}"
                );
                println!("D-27/D-26 at lags: {m}");
            }
            other => panic!(
                "the SAME series the probe proves safe at n_lags = 0 must be REFUSED at \
                 n_lags = 7 — the fill rule is load-bearing there and there is no defensible \
                 value to invent. Got {:?}",
                other.map(|r| r.model)
            ),
        }

        // THE CONTROL that keeps the refusal from being about the lag count alone: the same
        // series and the same lag count with NO regressor is accepted.
        let mut no_reg = args;
        no_reg.regressors = None;
        crate::forecast::forecast(&no_reg)
            .expect("the same gappy series at the same lag count with NO regressor is fine");
    }
}

// ============================================================== parity ====
// The SC3 correctness ladder and the D-10 training-rule invariants. Every numeric bar is
// READ from `contracts/neuralprophet-parity-v1.yaml` at test time (D-15) — a bar that also
// exists as a literal here could be loosened without the contract ever noticing.
#[cfg(test)]
mod parity {
    use super::{
        auto_batch, auto_epochs, fourier_feats, predict_ar_1step, predict_ts, train,
        weighted_huber, NpData, NpModel, NpSeason, Rng, TrainConfig,
    };
    use crate::dates::{days_from_civil, parse_ymd};
    use crate::test_support::{
        constant_u64, contract_path, equation_tolerance, load_json, read_csv,
    };
    use aprender::autograd::{clear_graph, get_grad, graph_tape_len, Tensor};

    /// The oracle holds out the last 365 daily rows (spike-002 `main.rs`: `n_train = n - H`).
    const HOLDOUT: usize = 365;

    /// Read a comma-separated `constants.<key>` learning-rate sweep from the contract.
    ///
    /// `test_support` carries `constant_u64` only, and a sweep is an ordered list rather than a
    /// scalar. Kept local to this module so the plan's `files_modified` set stays honest.
    fn constant_lr_sweep(key: &str) -> Vec<f64> {
        let path = contract_path("neuralprophet-parity-v1");
        let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "contract neuralprophet-parity-v1 must exist at {}: {e}",
                path.display()
            )
        });
        let doc: serde_yaml::Value = serde_yaml::from_str(&raw)
            .unwrap_or_else(|e| panic!("contract neuralprophet-parity-v1 is not valid YAML: {e}"));
        let s = doc
            .get("constants")
            .and_then(|c| c.get(key))
            .and_then(serde_yaml::Value::as_str)
            .unwrap_or_else(|| {
                panic!("contract neuralprophet-parity-v1 must define constants.{key} as a string")
            });
        s.split(',')
            .map(|p| {
                p.trim()
                    .parse::<f64>()
                    .unwrap_or_else(|e| panic!("constants.{key}: {p:?} is not a float: {e}"))
            })
            .collect()
    }

    /// The oracle split: the whole Peyton Manning CSV, and the index where the 365-row holdout
    /// begins. Read from the same de-duplicated CSV the oracle was generated from.
    fn peyton_split() -> (Vec<i64>, Vec<f64>, usize) {
        let (ds_s, y) = read_csv("peyton_manning.csv");
        let ds: Vec<i64> = ds_s.iter().map(|s| parse_ymd(s)).collect();
        let n_train = y.len() - HOLDOUT;
        (ds, y, n_train)
    }

    fn mae(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>() / a.len() as f64
    }

    /// The oracle stores `data_params` as pandas reprs, e.g. `"5.26269018890489"` and
    /// `"2596 days 00:00:00"`. Take the leading numeric token.
    fn leading_f64(s: &str) -> f64 {
        let head: String = s
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == '+' || *c == 'e')
            .collect();
        head.parse().unwrap_or_else(|e| {
            panic!("oracle data_params {s:?} does not start with a number: {e}")
        })
    }

    fn oracle_str<'a>(v: &'a serde_json::Value, path: &[&str]) -> &'a str {
        let mut cur = v;
        for k in path {
            cur = &cur[*k];
        }
        cur.as_str()
            .unwrap_or_else(|| panic!("oracle key {} must be a string", path.join(".")))
    }

    fn oracle_f64(v: &serde_json::Value, path: &[&str]) -> f64 {
        let mut cur = v;
        for k in path {
            cur = &cur[*k];
        }
        cur.as_f64()
            .unwrap_or_else(|| panic!("oracle key {} must be a number", path.join(".")))
    }

    // ---------------------------------------------------------- rung 1 ----

    #[test]
    fn data_prep_matches_np_oracle() {
        let (ds, y, n_train) = peyton_split();
        let d = NpData::new(&ds, &y, n_train, 10, 0.8);
        let ora = load_json("np_oracle_peyton.json");
        let ts = &ora["trend_seasonality"];
        let tol = equation_tolerance("neuralprophet-parity-v1", "data_prep_vs_oracle_abs");

        let np_shift = leading_f64(oracle_str(ts, &["data_params", "y", "shift"]));
        let np_scale = leading_f64(oracle_str(ts, &["data_params", "y", "scale"]));
        assert!(
            (d.shift - np_shift).abs() <= tol,
            "soft-normalisation shift: rust {} vs NeuralProphet {np_shift} (bar {tol:e})",
            d.shift
        );
        assert!(
            (d.scale - np_scale).abs() <= tol,
            "soft-normalisation scale (q95 - min): rust {} vs NeuralProphet {np_scale} (bar {tol:e})",
            d.scale
        );

        let np_t0_str = oracle_str(ts, &["data_params", "ds", "shift"]);
        let np_t0 = parse_ymd(&np_t0_str[..10]);
        assert_eq!(
            d.t0, np_t0,
            "time origin: rust day {} vs NeuralProphet {np_t0_str}",
            d.t0
        );
        let np_span = leading_f64(oracle_str(ts, &["data_params", "ds", "scale"]));
        assert!(
            (d.t_span - np_span).abs() <= tol,
            "train span in days: rust {} vs NeuralProphet {np_span} (bar {tol:e})",
            d.t_span
        );

        let np_cps = ts["changepoints_t"]
            .as_array()
            .expect("oracle trend_seasonality.changepoints_t");
        assert_eq!(
            d.cps.len(),
            np_cps.len(),
            "changepoint count: rust {} vs NeuralProphet {}",
            d.cps.len(),
            np_cps.len()
        );
        let worst = d
            .cps
            .iter()
            .zip(np_cps)
            .map(|(a, b)| (a - b.as_f64().expect("changepoint")).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            worst <= tol,
            "changepoints_t max abs diff {worst:e} exceeds the contract bar {tol:e}"
        );

        // The two auto formulas are asserted as EXACT integers against NeuralProphet's own
        // reported values — these are not tolerance comparisons.
        let np_batch = constant_u64("neuralprophet-parity-v1", "np_peyton_batch") as usize;
        let np_epochs = constant_u64("neuralprophet-parity-v1", "np_peyton_epochs") as usize;
        assert_eq!(
            auto_batch(n_train),
            np_batch,
            "auto_batch({n_train}) must reproduce NeuralProphet's own batch size"
        );
        assert_eq!(
            auto_epochs(n_train),
            np_epochs,
            "auto_epochs({n_train}) must reproduce NeuralProphet's own epoch count"
        );
        assert_eq!(
            auto_batch(n_train),
            ts["batch"].as_u64().expect("oracle batch") as usize,
            "the contract's np_peyton_batch and the oracle must agree"
        );
        assert_eq!(
            auto_epochs(n_train),
            ts["epochs"].as_u64().expect("oracle epochs") as usize,
            "the contract's np_peyton_epochs and the oracle must agree"
        );
    }

    // ---------------------------------------------------------- rung 2 ----

    #[test]
    fn lag_free_365_day_mae_within_contract() {
        let (ds, y, n_train) = peyton_split();
        let d = NpData::new(&ds, &y, n_train, 10, 0.8);
        let test_days = &ds[n_train..];
        let test_y = &y[n_train..];
        let sweep = constant_lr_sweep("lr_sweep_lag_free");

        // Selected by lowest final TRAIN loss, NEVER by test error (D-10).
        let mut best: Option<(f64, f64, Vec<f64>)> = None;
        for &lr in &sweep {
            let cfg = TrainConfig {
                n_lags: 0,
                ar_layers: vec![],
                max_lr: lr,
                epochs: None,
                batch: None,
                weight_decay: 1e-3,
                huber_beta: 0.3,
                newer_w: 2.0,
                seed: 42,
                event_design: None,
                regressors: None,
            };
            let (m, log) = train(&d, &cfg, false);
            let fl = *log.epoch_loss.last().unwrap_or(&f64::INFINITY);
            if !fl.is_finite() {
                continue;
            }
            if best.as_ref().is_none_or(|b| fl < b.0) {
                best = Some((fl, lr, predict_ts(&d, &m, test_days, None, None)));
            }
        }
        let (train_loss, selected_lr, yhat) =
            best.expect("at least one learning rate in the sweep must produce a finite loss");
        assert!(
            sweep.iter().any(|c| (c - selected_lr).abs() < 1e-12),
            "the selected lr {selected_lr} must come from the contract sweep {sweep:?}"
        );

        let e = mae(&yhat, test_y);
        let bar = equation_tolerance("neuralprophet-parity-v1", "lag_free_holdout_mae");
        let ora = load_json("np_oracle_peyton.json");
        let np_mae = oracle_f64(&ora, &["trend_seasonality", "mae_365ahead_on_test_rows"]);
        let np_rows = ora["trend_seasonality"]["n_test_rows"]
            .as_u64()
            .expect("oracle n_test_rows");
        assert!(
            e <= bar,
            "{HOLDOUT}-day-ahead holdout MAE {e:.4} exceeds the contract bar {bar} \
             (lr {selected_lr} selected by train loss {train_loss:.5}; \
             Python NeuralProphet 0.9.0 scored {np_mae:.4} over {np_rows} of these rows)"
        );
        eprintln!(
            "lag-free: lr {selected_lr} (train loss {train_loss:.5}) -> test MAE {e:.4} \
             over {HOLDOUT} rows; NeuralProphet {np_mae:.4} over {np_rows}; bar {bar}"
        );
    }

    // ---------------------------------------------------------- rung 3 ----

    #[test]
    fn ar_net_30_lags_beats_naive_one_step() {
        let (ds, y, n_train) = peyton_split();
        let d = NpData::new(&ds, &y, n_train, 10, 0.8);
        let test_days = &ds[n_train..];
        let test_y = &y[n_train..];
        let test_idx: Vec<usize> = test_days.iter().map(|&day| (day - d.t0) as usize).collect();
        let sweep = constant_lr_sweep("lr_sweep_with_lags");
        let cap = constant_u64("neuralprophet-parity-v1", "auto_epochs_cap_with_lags") as usize;

        let mut best: Option<(f64, f64, Vec<f64>)> = None;
        for &lr in &sweep {
            let cfg = TrainConfig {
                n_lags: 30,
                ar_layers: vec![32],
                max_lr: lr,
                epochs: Some(auto_epochs(n_train).min(cap)),
                batch: None,
                weight_decay: 1e-3,
                huber_beta: 0.3,
                newer_w: 2.0,
                seed: 42,
                event_design: None,
                regressors: None,
            };
            let (m, log) = train(&d, &cfg, false);
            let fl = *log.epoch_loss.last().unwrap_or(&f64::INFINITY);
            if !fl.is_finite() {
                continue;
            }
            if best.as_ref().is_none_or(|b| fl < b.0) {
                best = Some((fl, lr, predict_ar_1step(&d, &m, &test_idx, None, None)));
            }
        }
        let (train_loss, selected_lr, yhat) =
            best.expect("at least one learning rate in the sweep must produce a finite loss");

        let e = mae(&yhat, test_y);
        let margin = equation_tolerance("neuralprophet-parity-v1", "ar_net_vs_naive_margin");
        let ora = load_json("np_oracle_peyton.json");
        let naive = oracle_f64(&ora, &["naive_1step_mae_test"]);
        let np_ar = oracle_f64(&ora, &["ar30_hidden32", "mae_1step_test"]);
        assert!(
            e < naive - margin,
            "AR-Net(30 lags, hidden [32]) one-step MAE {e:.4} must beat the oracle's naive \
             baseline {naive:.4} by more than {margin} \
             (lr {selected_lr} selected by train loss {train_loss:.5}; \
             Python NeuralProphet's own AR-Net scored {np_ar:.4})"
        );
        eprintln!(
            "AR-Net: lr {selected_lr} (train loss {train_loss:.5}) -> 1-step test MAE {e:.4}; \
             naive {naive:.4}; Python NP AR-Net {np_ar:.4}"
        );
    }

    // ------------------------------------------------- D-10 invariants ----

    #[test]
    fn weighted_huber_is_graph_connected() {
        clear_graph();
        // A tiny NpData is enough: the property is about the LOSS reaching the parameters,
        // not about the fit. Core's own smooth-L1 loss fails this exact check.
        let t0 = days_from_civil(2020, 1, 1);
        let ds: Vec<i64> = (0..40).map(|i| t0 + i).collect();
        let y: Vec<f64> = (0..40).map(|i| 1.0 + f64::from(i) * 0.1).collect();
        let d = NpData::new(&ds, &y, ds.len(), 3, 0.8);
        let mut rng = Rng::new(42);
        let mut model = NpModel::new(&d, 0, &[], &mut rng);

        let b = 8usize;
        let (mut tr, mut se) = (Vec::new(), Vec::new());
        for &day in &ds[..b] {
            d.row_feats(day, &mut tr, &mut se);
        }
        let xt = Tensor::from_vec(tr, &[b, d.trend_dim()]);
        let xs = Tensor::from_vec(se, &[b, d.season_dim()]);
        let yt = Tensor::from_vec(y[..b].iter().map(|v| d.norm(*v)).collect(), &[b, 1]);
        let wt = Tensor::from_vec(vec![1.0; b], &[b, 1]);

        let pred = model.forward(&xt, &xs, None);
        let loss = weighted_huber(&pred, &yt, &wt, 0.3);
        assert!(loss.item().is_finite(), "the loss itself must be finite");
        loss.backward();

        let floor = equation_tolerance("neuralprophet-parity-v1", "huber_grad_min_abs");
        // Gradients live on the tape and are read through `get_grad(id)` — that is exactly
        // what `Optimizer::step_with_params` consults, so it is the ground truth for
        // "can this loss train anything".
        let ids: Vec<_> = model.parameters_mut().iter().map(|p| p.id()).collect();
        assert!(!ids.is_empty(), "the model must have parameters");
        for (i, id) in ids.iter().enumerate() {
            let g = get_grad(*id).unwrap_or_else(|| {
                panic!(
                    "parameter {i} has NO gradient after backward(): the loss is DETACHED from \
                     the graph (D-10). This is precisely how core's own smooth-L1 loss fails."
                )
            });
            assert!(
                g.data().iter().any(|v| v.abs() > floor as f32),
                "parameter {i}: every gradient entry is <= {floor}; the loss reaches the \
                 parameter but carries no signal"
            );
        }
        clear_graph();
    }

    #[test]
    fn auto_batch_is_mini_batch_over_grid() {
        let lo = constant_u64("neuralprophet-parity-v1", "min_rows_for_mini_batch") as usize;
        let bmin = constant_u64("neuralprophet-parity-v1", "auto_batch_min") as usize;
        let bmax = constant_u64("neuralprophet-parity-v1", "auto_batch_max") as usize;
        let mut n = lo;
        let mut checked = 0usize;
        while n <= crate::types::MAX_POINTS {
            let b = auto_batch(n);
            assert!(
                b < n,
                "auto_batch({n}) = {b} is a FULL batch; full-batch training collapses this \
                 model (measured test MAE 2.6 against 0.45)"
            );
            assert!(
                (bmin..=bmax).contains(&b),
                "auto_batch({n}) = {b} is outside the contract clamp [{bmin}, {bmax}]"
            );
            checked += 1;
            n += 97;
        }
        assert!(
            checked > 100,
            "the grid must actually cover the range; only {checked} points were checked"
        );
    }

    #[test]
    fn train_clears_the_tape() {
        clear_graph();
        let t0 = days_from_civil(2020, 1, 1);
        let ds: Vec<i64> = (0..120).map(|i| t0 + i).collect();
        let y: Vec<f64> = (0..120)
            .map(|i| 10.0 + f64::from(i) * 0.01 + (f64::from(i) / 7.0).sin())
            .collect();
        let d = NpData::new(&ds, &y, ds.len(), 10, 0.8);
        let cfg = TrainConfig {
            n_lags: 0,
            ar_layers: vec![],
            max_lr: 0.1,
            epochs: Some(3),
            batch: None,
            weight_decay: 1e-3,
            huber_beta: 0.3,
            newer_w: 2.0,
            seed: 42,
            event_design: None,
            regressors: None,
        };
        let (m, log) = train(&d, &cfg, false);
        assert!(log.steps > 0, "the fit must have taken at least one step");
        assert_eq!(
            graph_tape_len(),
            0,
            "train() must leave the thread-local tape empty (clear_graph() after EVERY step)"
        );
        // The prediction helpers must clear it too: they build a no_grad forward that still
        // allocates tape entries.
        let _ = predict_ts(&d, &m, &ds[..5], None, None);
        assert_eq!(
            graph_tape_len(),
            0,
            "predict_ts() must leave the thread-local tape empty"
        );
    }

    #[test]
    fn full_batch_is_not_what_auto_batch_returns() {
        let (_, y, n_train) = peyton_split();
        assert!(n_train > 0 && !y.is_empty());
        let np_batch = constant_u64("neuralprophet-parity-v1", "np_peyton_batch") as usize;
        assert_eq!(
            auto_batch(n_train),
            np_batch,
            "auto_batch on the Peyton training split must be NeuralProphet's own value, not \
             the full {n_train}-row batch"
        );
        assert!(
            auto_batch(n_train) < n_train,
            "a full batch here means 80 optimiser steps instead of 3200"
        );
    }

    #[test]
    fn fourier_features_are_phased_on_1900() {
        let epoch_year = constant_u64("neuralprophet-parity-v1", "fourier_epoch_year") as i64;
        let seasons = vec![NpSeason {
            name: "weekly".into(),
            period: 7.0,
            order: 3,
        }];
        let day = parse_ymd("2020-01-01");
        let mut out = Vec::new();
        fourier_feats(day, &seasons, &mut out);
        assert_eq!(out.len(), 6, "order 3 gives 3 sin terms then 3 cos terms");

        let t = (day - days_from_civil(epoch_year, 1, 1)) as f64;
        let f = 2.0 * std::f64::consts::PI / 7.0;
        for k in 1..=3usize {
            let want = (f * k as f64 * t).sin() as f32;
            assert!(
                (out[k - 1] - want).abs() < 1e-5,
                "sin block position {}: got {} want {want} (t = day - {epoch_year}-01-01)",
                k - 1,
                out[k - 1]
            );
            let want_cos = (f * k as f64 * t).cos() as f32;
            assert!(
                (out[2 + k] - want_cos).abs() < 1e-5,
                "cos block position {}: got {} want {want_cos}",
                2 + k,
                out[2 + k]
            );
        }
        // The epoch is load-bearing: phasing on the Unix epoch instead would move the features.
        let t_unix = day as f64;
        let unix_first_sin = (f * t_unix).sin() as f32;
        assert!(
            (out[0] - unix_first_sin).abs() > 1e-3,
            "the {epoch_year} epoch must be distinguishable from the Unix epoch, otherwise \
             this test cannot detect the epoch moving"
        );
    }

    #[test]
    fn lr_selection_is_by_train_loss() {
        // The shipped door and an independent argmin over the contract's sweep must agree.
        let t0 = days_from_civil(2020, 1, 1);
        let mut rng = crate::prophet::Rng::new(7);
        let n = 120usize;
        let ds: Vec<i64> = (0..n as i64).map(|i| t0 + i).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64;
                10.0 + 0.01 * t + (2.0 * std::f64::consts::PI * t / 7.0).sin() + 0.05 * rng.normal()
            })
            .collect();

        let args = crate::types::ForecastArgs {
            ds: ds.iter().map(|d| crate::dates::format_ymd(*d)).collect(),
            y: y.clone(),
            horizon: 7,
            freq: None,
            model: Some("neuralprophet".into()),
            growth: None,
            cap: None,
            seasonality_mode: None,
            interval_width: None,
            holidays: None,
            n_lags: None,
            seed: None,
            regressors: None,
        };
        let r = crate::forecast::forecast(&args).expect("the neuralprophet arm must dispatch");
        let reported_lr = r.diagnostics["selected_lr"]
            .as_f64()
            .expect("diagnostics.selected_lr");
        let reported_loss = r.diagnostics["final_train_loss"]
            .as_f64()
            .expect("diagnostics.final_train_loss");

        let d = NpData::new(&ds, &y, n, 10, 0.8);
        let sweep = constant_lr_sweep("lr_sweep_lag_free");
        let mut argmin: Option<(f64, f64)> = None;
        for &lr in &sweep {
            let cfg = TrainConfig {
                n_lags: 0,
                ar_layers: vec![],
                max_lr: lr,
                epochs: None,
                batch: None,
                weight_decay: 1e-3,
                huber_beta: 0.3,
                newer_w: 2.0,
                seed: 42,
                event_design: None,
                regressors: None,
            };
            let (_, log) = train(&d, &cfg, false);
            let fl = *log.epoch_loss.last().unwrap_or(&f64::INFINITY);
            if fl.is_finite() && argmin.as_ref().is_none_or(|a| fl < a.0) {
                argmin = Some((fl, lr));
            }
        }
        let (best_loss, best_lr) = argmin.expect("the sweep must produce a finite loss");
        assert!(
            (reported_lr - best_lr).abs() < 1e-12,
            "the door reported lr {reported_lr} but the lowest FINAL TRAIN LOSS over the \
             contract sweep {sweep:?} is at lr {best_lr} (loss {best_loss:.6}). Selection must \
             be by train loss, never by test error (D-10)."
        );
        assert!(
            (reported_loss - best_loss).abs() < 1e-9,
            "the door reported final_train_loss {reported_loss:.6} but the independent argmin \
             is {best_loss:.6}"
        );
    }
}

// ------------------------------------------- the C-08 wall harness (ignored) ----

/// Wall-clock harness for cost axis **C-08**, the NeuralProphet training path — the one
/// axis in this door with no budget of ANY kind on it. `fit::FIT_BUDGET_SECS` is read at
/// exactly one place, inside `fit::fit_prophet`, and the `"neuralprophet"` arm never enters
/// it; the door then drives 2 or 3 full [`train`] calls per request through its
/// learning-rate sweep.
///
/// `#[ignore]`d, because it is a MEASUREMENT and not an assertion about correctness. Run it
/// deliberately, on a RELEASE build, one mode per invocation:
///
/// ```text
/// NP_WALL_MODE=structural_max cargo test --release -p aprender-forecast --lib \
///     np_train_wall -- --ignored --nocapture
/// ```
///
/// It drives the DOOR (`crate::forecast::forecast`), not [`train`] directly, so its
/// `outcome=` field can report `refused` once a bound exists. That field is load-bearing:
/// after a bound is added the worst legal request is REFUSED rather than slow, and the
/// post-condition gate has to be able to tell those two apart from the same line.
#[cfg(test)]
mod wall {
    use crate::types::{ForecastArgs, MAX_HORIZON, MAX_POINTS};

    /// One machine-parsable line per run. `profile=` is derived from `cfg!(debug_assertions)`
    /// rather than from intent (CLAUDE.md rule 2): a debug wall labelled `release` is exactly
    /// the class of error that turns a measurement into a confident wrong answer.
    fn emit(mode: &str, args: &ForecastArgs, span_days: i64, cost: u64, outcome: &str, secs: f64) {
        println!(
            "NP TRAIN WALL: mode={mode} points={} span_days={span_days} n_lags={} horizon={} \
             train_cost={cost} outcome={outcome} total_s={secs:.3} profile={}",
            args.ds.len(),
            args.n_lags.unwrap_or(0),
            args.horizon,
            if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            }
        );
    }

    /// A daily series with a trend and a weekly term, one point per day from 2020-01-01, so
    /// `span_days == points` and `n_train_grid` is at its ceiling when `points == MAX_POINTS`.
    fn contiguous_daily(n: usize) -> (Vec<String>, Vec<f64>) {
        let t0 = crate::dates::days_from_civil(2020, 1, 1);
        let mut ds = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            ds.push(crate::dates::format_ymd(t0 + i as i64));
            let t = i as f64;
            y.push(10.0 + 0.001 * t + (2.0 * std::f64::consts::PI * t / 7.0).sin());
        }
        (ds, y)
    }

    #[test]
    #[ignore = "wall-clock measurement; run with NP_WALL_MODE=... --release -- --ignored"]
    fn np_train_wall() {
        let mode = std::env::var("NP_WALL_MODE").unwrap_or_else(|_| "mid_range".into());
        // freq "D" is the ONLY frequency this arm accepts, so every mode uses it.
        let (points, n_lags, horizon) = match mode.as_str() {
            // The worst LEGAL request: MAX_POINTS contiguous daily points (so span_days is
            // also at MAX_SPAN_DAYS and n_train_grid is at its ceiling), n_lags at its own
            // ceiling of 365, horizon at MAX_HORIZON.
            "structural_max" => (MAX_POINTS, 365usize, MAX_HORIZON),
            // A point in the middle of the legal range, so a single number is not the whole
            // basis (CLAUDE.md rule 6 — one input is an anecdote).
            "mid_range" => (2_000usize, 30usize, 365usize),
            // The OTHER arm: n_lags == 0 takes a THREE-point learning-rate sweep instead of
            // two, trains on the observed rows only, and lets `train` pick the epochs.
            "lag_free" => (MAX_POINTS, 0usize, MAX_HORIZON),
            // THREE compositions sitting just under `MAX_NP_TRAIN_COST`, used to DERIVE that
            // value rather than to assert it. They differ in every factor the proxy
            // multiplies — history length, n_lags and horizon — so a value that clears the
            // 2 s bar on all three is not clearing it on one shape (CLAUDE.md rule 6).
            // A first pass at a candidate of 20 000 000 measured 2.089 s on the long-history
            // composition — OVER the bar — which is what moved the value down to 15 000 000.
            // The REJECTED candidate, kept as a named mode so its rejection is reproducible
            // rather than only quoted: at a bound of 20 000 000 this composition prices at
            // 19 991 000 and measured 2.089 s — OVER SC1's 2 s bar. At the SHIPPED bound of
            // 15 000 000 it is refused, so reproducing the 2.089 s means raising
            // `MAX_NP_TRAIN_COST` and `constants.fit_max_np_train_cost` to 20 000 000 first.
            "rejected_candidate_20m" => (MAX_POINTS, 9usize, MAX_HORIZON),
            "at_bound_long_history" => (MAX_POINTS, 6usize, MAX_HORIZON),
            "at_bound_mid_history" => (10_000usize, 11usize, MAX_HORIZON),
            "at_bound_short_history" => (2_000usize, 41usize, 365usize),
            other => panic!(
                "NP_WALL_MODE={other:?}: want structural_max, mid_range, lag_free, \
                 at_bound_long_history, at_bound_mid_history, at_bound_short_history or \
                 rejected_candidate_20m"
            ),
        };
        let (ds, y) = contiguous_daily(points);
        let span_days = points as i64;
        let mut args = ForecastArgs {
            ds,
            y,
            horizon,
            ..ForecastArgs::default()
        };
        args.model = Some("neuralprophet".into());
        args.freq = Some("D".into());
        if n_lags > 0 {
            args.n_lags = Some(n_lags);
        }

        // The door's own pricing of this request, computed the way the door computes it, so
        // the line reports the number the bound is (or would be) compared against.
        let d = super::NpData::new(
            &args
                .ds
                .iter()
                .map(|s| crate::dates::parse_date(s).expect("harness dates are well formed"))
                .collect::<Vec<_>>(),
            &args.y,
            points,
            10,
            0.8,
        );
        // The door's OWN pricing function, so this line reports the number the bound is
        // compared against rather than a second arithmetic that could drift from it.
        let cost = super::request_train_cost(&d, points, n_lags, 0, 0);
        drop(d);

        let t0 = std::time::Instant::now();
        let result = crate::forecast::forecast(&args);
        let secs = t0.elapsed().as_secs_f64();
        let outcome = match &result {
            Ok(_) => "accepted",
            Err(crate::types::ForecastError::Validation(_)) => "refused",
            Err(e) => panic!("the harness must produce an accept or a refusal, got {e:?}"),
        };
        emit(&mode, &args, span_days, cost, outcome, secs);
        if let Err(crate::types::ForecastError::Validation(m)) = &result {
            println!("NP TRAIN WALL REFUSAL: {m}");
        }
    }
}
