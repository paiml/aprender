//! Prophet, second cut: linear / logistic / flat growth, additive AND multiplicative
//! terms, holiday indicator columns with windows, analytic gradients for all of it,
//! component decomposition, and the vectorised uncertainty simulation of
//! `forecaster.py` (`_make_trend_shift_matrix` → `_sample_uncertainty` →
//! `sample_model_vectorized` → percentiles).
//!
//! Ported VERBATIM from `sources/004-forecast-mcp-thin-server/src/prophet.rs` (D-08).
//! The only change is that the spike's private civil-date copies (its lines 8-36) are
//! replaced by the re-export below, so `dates.rs` is the single implementation (D-17)
//! and downstream code reaching `prophet::days_from_civil` still compiles.

pub use crate::dates::{civil_from_days, days_from_civil, format_ymd, parse_ymd};

// ----------------------------------------------------------------- spec ----
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Growth {
    Linear,
    Logistic,
    Flat,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode {
    Additive,
    Multiplicative,
}

#[derive(Clone, Debug)]
pub struct Seasonality {
    pub name: String,
    pub period: f64,
    pub order: usize,
    pub prior_scale: f64,
    pub mode: Mode,
}
#[derive(Clone, Debug)]
pub struct Holiday {
    pub name: String,
    pub days: Vec<i64>,
    pub lower_window: i64,
    pub upper_window: i64,
    pub prior_scale: f64,
}

#[derive(Clone, Debug)]
pub struct Spec {
    pub growth: Growth,
    /// Constant capacity (logistic only), original scale.
    pub cap: Option<f64>,
    pub seasonalities: Vec<Seasonality>,
    pub holidays: Vec<Holiday>,
    pub holidays_mode: Mode,
    pub n_changepoints: usize,
    pub changepoint_range: f64,
    pub changepoint_prior_scale: f64,
    pub interval_width: f64,
    pub uncertainty_samples: usize,
}

impl Spec {
    pub fn default_linear(seasonalities: Vec<Seasonality>) -> Self {
        Spec {
            growth: Growth::Linear,
            cap: None,
            seasonalities,
            holidays: vec![],
            holidays_mode: Mode::Additive,
            n_changepoints: 25,
            changepoint_range: 0.8,
            changepoint_prior_scale: 0.05,
            interval_width: 0.8,
            uncertainty_samples: 1000,
        }
    }
}

pub fn auto_seasonalities(ds_days: &[i64], prior_scale: f64, mode: Mode) -> Vec<Seasonality> {
    let span = (ds_days[ds_days.len() - 1] - ds_days[0]) as f64;
    let min_dt = ds_days
        .windows(2)
        .map(|w| (w[1] - w[0]) as f64)
        .filter(|d| *d > 0.0)
        .fold(f64::INFINITY, f64::min);
    let mut out = Vec::new();
    if span >= 730.0 {
        out.push(Seasonality {
            name: "yearly".into(),
            period: 365.25,
            order: 10,
            prior_scale,
            mode,
        });
    }
    if span >= 14.0 && min_dt < 7.0 {
        out.push(Seasonality {
            name: "weekly".into(),
            period: 7.0,
            order: 3,
            prior_scale,
            mode,
        });
    }
    if span >= 2.0 && min_dt < 1.0 {
        out.push(Seasonality {
            name: "daily".into(),
            period: 1.0,
            order: 4,
            prior_scale,
            mode,
        });
    }
    out
}

// -------------------------------------------------------------- columns ----
/// One regressor column of X: which component it belongs to and its mode.
#[derive(Clone, Debug)]
pub struct Column {
    pub name: String,
    pub component: String,
    pub mode: Mode,
    pub prior_scale: f64,
    pub holiday: Option<(usize, i64)>,
}

/// Prophet's column order: seasonalities in insertion order (sin, cos interleaved),
/// then holiday columns sorted by their `{holiday}_delim_{±offset}` name.
pub fn columns(spec: &Spec) -> Vec<Column> {
    let mut cols = Vec::new();
    for s in &spec.seasonalities {
        for i in 0..s.order {
            cols.push(Column {
                name: format!("{}_delim_{}", s.name, 2 * i + 1),
                component: s.name.clone(),
                mode: s.mode,
                prior_scale: s.prior_scale,
                holiday: None,
            });
            cols.push(Column {
                name: format!("{}_delim_{}", s.name, 2 * i + 2),
                component: s.name.clone(),
                mode: s.mode,
                prior_scale: s.prior_scale,
                holiday: None,
            });
        }
    }
    let mut hcols = Vec::new();
    for (hi, h) in spec.holidays.iter().enumerate() {
        for off in h.lower_window..=h.upper_window {
            hcols.push(Column {
                name: format!(
                    "{}_delim_{}{}",
                    h.name,
                    if off >= 0 { '+' } else { '-' },
                    off.abs()
                ),
                component: h.name.clone(),
                mode: spec.holidays_mode,
                prior_scale: h.prior_scale,
                holiday: Some((hi, off)),
            });
        }
    }
    hcols.sort_by(|a, b| a.name.cmp(&b.name));
    cols.extend(hcols);
    cols
}

/// One membership set per holiday, in `spec.holidays` order, built ONCE per design or
/// prediction rather than rescanned per row per column.
///
/// `feature_row` used to answer "is `day` in this holiday's window offset `off`?" with a
/// linear `.any()` over `days`, i.e. `rows x holiday_columns x dates` comparisons per
/// design build. The set answers the IDENTICAL predicate in O(1): `d + off == day` iff
/// `d == day - off`.
#[must_use]
pub fn holiday_day_sets(spec: &Spec) -> Vec<std::collections::HashSet<i64>> {
    spec.holidays
        .iter()
        .map(|h| h.days.iter().copied().collect())
        .collect()
}

/// `hol_sets` must be the [`holiday_day_sets`] `HashSet` slice for the same `spec` — one
/// set per holiday, in order. Passing it in rather than rebuilding it is the whole point:
/// the caller hoists the construction out of the row loop.
pub fn feature_row(
    day: i64,
    spec: &Spec,
    cols: &[Column],
    hol_sets: &[std::collections::HashSet<i64>],
    out: &mut Vec<f64>,
) {
    let x_t = std::f64::consts::PI * 2.0 * day as f64;
    for s in &spec.seasonalities {
        for i in 0..s.order {
            let c = (i + 1) as f64 / s.period * x_t;
            out.push(c.sin());
            out.push(c.cos());
        }
    }
    for c in cols.iter().filter(|c| c.holiday.is_some()) {
        let (hi, off) = c.holiday.expect("holiday col");
        // Same predicate as the former `days.iter().any(|&d| d + off == day)`, rearranged.
        //
        // TOTAL LOOKUP (IN-02): `hi` is derived from `cols`, and it subscripts `hol_sets` —
        // a DIFFERENT argument that no type ties to it. Every `Design` field is `pub`, so
        // `d.spec.holidays.clear()` followed by `predict(&d, ..)` reaches here with a slice
        // that does not line up, and `hol_sets[hi]` made that an out-of-bounds panic raised
        // inside a library.
        //
        // WHAT THIS BUYS AND WHAT IT DOES NOT: both in-crate callers (`make_design` and
        // `predict`) build `hol_sets` from the same `spec` immediately before the row loop,
        // so this CANNOT change any shipped result — `prophet::parity` is the measurement,
        // 32 passed / 0 failed before and after. A mismatch was a panic and is now a zero
        // column, and NEITHER is correct output for a caller who built the slice wrong. What
        // it buys is that a library does not abort the caller's process over it.
        let hit = hol_sets.get(hi).is_some_and(|s| s.contains(&(day - off)));
        out.push(if hit { 1.0 } else { 0.0 });
    }
}

// --------------------------------------------------------------- design ----
#[derive(Clone, Debug)]
pub struct Design {
    pub spec: Spec,
    pub cols: Vec<Column>,
    pub t: Vec<f64>,
    pub y_scaled: Vec<f64>,
    pub cap_scaled: Option<Vec<f64>>,
    pub y_scale: f64,
    pub start_days: i64,
    pub t_scale_days: f64,
    pub changepoints_t: Vec<f64>,
    /// T×K row-major.
    pub x: Vec<f64>,
    pub k: usize,
    pub s_a: Vec<f64>,
    pub s_m: Vec<f64>,
    pub prior_scales: Vec<f64>,
}

fn linspace_round(stop: f64, num: usize) -> Vec<usize> {
    let step = stop / (num as f64 - 1.0);
    (0..num)
        .map(|i| {
            let v = if i + 1 == num { stop } else { step * i as f64 };
            v.round_ties_even() as usize
        })
        .collect()
}

/// The ONE arithmetic site for the effective changepoint geometry: `(hist_size, n_cp)`.
///
/// Private on purpose. [`changepoint_count`] is the public reading of it and
/// [`make_design`] is the only other caller, so the rule "how many changepoints does a
/// series of `n` points get?" is written exactly once. A door that recomputed
/// `min(spec.n_changepoints, floor(n * changepoint_range) - 1)` inline could drift from
/// the design it is trying to bound, and a bound the sampler disagrees with is evadable.
fn changepoint_geometry(n: usize, spec: &Spec) -> (usize, usize) {
    let hist_size = (n as f64 * spec.changepoint_range).floor() as usize;
    let mut n_cp = spec.n_changepoints;
    if n_cp + 1 > hist_size {
        n_cp = hist_size.saturating_sub(1);
    }
    (hist_size, n_cp)
}

/// How many entries [`make_design`] will put in `Design::changepoints_t` for `n` points.
///
/// Returns the LENGTH, not the raw count: `make_design`'s `n_cp == 0` branch is
/// `vec![0.0]`, i.e. length **one**, not zero — so the effective count is `max(n_cp, 1)`.
/// That distinction is load-bearing, because this length is the `s_cnt` factor of the
/// logistic uncertainty simulation's Poisson mean
/// `lambda = changepoints_t.len() * (t_max - 1)`, which
/// [`crate::types::MAX_LOGISTIC_CHANGEPOINT_LAMBDA`] bounds at the door.
///
/// `forecast::forecast` calls THIS function to compute the lambda it refuses on, so the
/// door's lambda is `predict`'s lambda by construction rather than by agreement.
#[must_use]
pub fn changepoint_count(n: usize, spec: &Spec) -> usize {
    let (_, n_cp) = changepoint_geometry(n, spec);
    n_cp.max(1)
}

pub fn make_design(ds_days: &[i64], y: &[f64], spec: &Spec) -> Design {
    let n = y.len();
    assert!(n >= 2 && ds_days.windows(2).all(|w| w[0] < w[1]));
    let start_days = ds_days[0];
    let t_scale_days = (ds_days[n - 1] - start_days) as f64;
    let t: Vec<f64> = ds_days
        .iter()
        .map(|&d| (d - start_days) as f64 / t_scale_days)
        .collect();
    let mut y_scale = y.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    if y_scale == 0.0 {
        y_scale = 1.0;
    }
    let y_scaled: Vec<f64> = y.iter().map(|v| v / y_scale).collect();
    let cap_scaled = match (spec.growth, spec.cap) {
        (Growth::Logistic, Some(c)) => Some(vec![c / y_scale; n]),
        (Growth::Logistic, None) => panic!("logistic growth needs cap"),
        _ => None,
    };
    let (hist_size, n_cp) = changepoint_geometry(n, spec);
    let changepoints_t: Vec<f64> = if n_cp > 0 {
        let idx = linspace_round((hist_size - 1) as f64, n_cp + 1);
        idx[1..].iter().map(|&i| t[i]).collect()
    } else {
        vec![0.0]
    };
    // The tie the door's bound rests on, asserted where the design is actually built.
    debug_assert_eq!(
        changepoints_t.len(),
        changepoint_count(n, spec),
        "changepoint_count must report the length make_design produces"
    );
    let cols = columns(spec);
    let k = cols.len();
    let mut x = Vec::with_capacity(n * k);
    let hol_sets = holiday_day_sets(spec);
    for &d in ds_days {
        feature_row(d, spec, &cols, &hol_sets, &mut x);
    }
    let s_a: Vec<f64> = cols
        .iter()
        .map(|c| if c.mode == Mode::Additive { 1.0 } else { 0.0 })
        .collect();
    let s_m: Vec<f64> = cols
        .iter()
        .map(|c| {
            if c.mode == Mode::Multiplicative {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let prior_scales = cols.iter().map(|c| c.prior_scale).collect();
    Design {
        spec: spec.clone(),
        cols,
        t,
        y_scaled,
        cap_scaled,
        y_scale,
        start_days,
        t_scale_days,
        changepoints_t,
        x,
        k,
        s_a,
        s_m,
        prior_scales,
    }
}

// ---------------------------------------------------------------- model ----
#[derive(Clone, Debug)]
pub struct Params {
    pub k: f64,
    pub m: f64,
    pub delta: Vec<f64>,
    pub beta: Vec<f64>,
    pub sigma_obs: f64,
}

pub struct Model<'a> {
    pub d: &'a Design,
    pub scale: f64,
    pub guard: bool,
}

/// Piecewise-linear trend for sorted `t`.
pub fn piecewise_linear(t: &[f64], cps: &[f64], delta: &[f64], k: f64, m: f64) -> Vec<f64> {
    let (mut out, mut j, mut k_t, mut m_t) = (Vec::with_capacity(t.len()), 0, k, m);
    for &ti in t {
        while j < cps.len() && cps[j] <= ti {
            k_t += delta[j];
            m_t -= cps[j] * delta[j];
            j += 1;
        }
        out.push(k_t * ti + m_t);
    }
    out
}

/// Stan's `logistic_gamma`: continuity offsets for the logistic trend.
pub fn logistic_gammas(k: f64, m: f64, delta: &[f64], cps: &[f64]) -> Vec<f64> {
    let s = cps.len();
    let mut k_s = Vec::with_capacity(s + 1);
    k_s.push(k);
    for j in 0..s {
        k_s.push(k_s[j] + delta[j]);
    }
    let mut gamma = vec![0.0; s];
    let mut m_pr = m;
    for i in 0..s {
        gamma[i] = (cps[i] - m_pr) * (1.0 - k_s[i] / k_s[i + 1]);
        m_pr += gamma[i];
    }
    gamma
}

pub fn piecewise_logistic(
    t: &[f64],
    cap: &[f64],
    cps: &[f64],
    delta: &[f64],
    k: f64,
    m: f64,
) -> Vec<f64> {
    let gamma = logistic_gammas(k, m, delta, cps);
    let (mut out, mut j, mut k_t, mut m_t) = (Vec::with_capacity(t.len()), 0, k, m);
    for (i, &ti) in t.iter().enumerate() {
        while j < cps.len() && cps[j] <= ti {
            k_t += delta[j];
            m_t += gamma[j];
            j += 1;
        }
        out.push(cap[i] / (1.0 + (-k_t * (ti - m_t)).exp()));
    }
    out
}

impl<'a> Model<'a> {
    pub fn new(d: &'a Design) -> Self {
        Model {
            d,
            scale: 1.0 / d.t.len() as f64,
            guard: true,
        }
    }
    pub fn n_params(&self) -> usize {
        2 + self.d.changepoints_t.len() + self.d.k + 1
    }
    pub fn pack(&self, p: &Params) -> Vec<f64> {
        let mut v = vec![p.k, p.m];
        v.extend_from_slice(&p.delta);
        v.extend_from_slice(&p.beta);
        v.push(p.sigma_obs.ln());
        v
    }
    pub fn unpack(&self, th: &[f64]) -> Params {
        let (s, k) = (self.d.changepoints_t.len(), self.d.k);
        Params {
            k: th[0],
            m: th[1],
            delta: th[2..2 + s].to_vec(),
            beta: th[2 + s..2 + s + k].to_vec(),
            sigma_obs: th[2 + s + k].exp(),
        }
    }

    /// Prophet's `*_growth_init`.
    pub fn init(&self) -> Params {
        let d = self.d;
        let n = d.t.len();
        let (k, m) = match d.spec.growth {
            Growth::Linear => {
                let tt = d.t[n - 1] - d.t[0];
                let k = (d.y_scaled[n - 1] - d.y_scaled[0]) / tt;
                (k, d.y_scaled[0] - k * d.t[0])
            }
            Growth::Flat => (0.0, d.y_scaled.iter().sum::<f64>() / n as f64),
            Growth::Logistic => {
                let cap = d.cap_scaled.as_ref().expect("cap");
                let tt = d.t[n - 1] - d.t[0];
                let (c0, c1) = (cap[0], cap[n - 1]);
                let y0 = (0.01 * c0).max((0.99 * c0).min(d.y_scaled[0]));
                let y1 = (0.01 * c1).max((0.99 * c1).min(d.y_scaled[n - 1]));
                let (mut r0, r1) = (c0 / y0, c1 / y1);
                if (r0 - r1).abs() <= 0.01 {
                    r0 *= 1.05;
                }
                let (l0, l1) = ((r0 - 1.0).ln(), (r1 - 1.0).ln());
                ((l0 - l1) / tt, l0 * tt / (l0 - l1))
            }
        };
        Params {
            k,
            m,
            delta: vec![0.0; d.changepoints_t.len()],
            beta: vec![0.0; d.k],
            sigma_obs: 1.0,
        }
    }

    pub fn trend(&self, p: &Params, t: &[f64], cap: Option<&[f64]>) -> Vec<f64> {
        let d = self.d;
        match d.spec.growth {
            Growth::Linear => piecewise_linear(t, &d.changepoints_t, &p.delta, p.k, p.m),
            Growth::Flat => vec![p.m; t.len()],
            Growth::Logistic => {
                piecewise_logistic(t, cap.expect("cap"), &d.changepoints_t, &p.delta, p.k, p.m)
            }
        }
    }

    /// μ = trend·(1 + X·(β∘s_m)) + X·(β∘s_a); returns (trend, xm = X·(β∘s_m), residuals).
    fn parts(&self, p: &Params) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let d = self.d;
        let n = d.t.len();
        let trend = self.trend(p, &d.t, d.cap_scaled.as_deref());
        let (mut xm, mut r) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for i in 0..n {
            let row = &d.x[i * d.k..(i + 1) * d.k];
            let (mut a, mut m) = (0.0, 0.0);
            for c in 0..d.k {
                let v = row[c] * p.beta[c];
                a += v * d.s_a[c];
                m += v * d.s_m[c];
            }
            xm.push(m);
            r.push(d.y_scaled[i] - (trend[i] * (1.0 + m) + a));
        }
        (trend, xm, r)
    }

    pub fn objective(&self, th: &[f64]) -> f64 {
        let p = self.unpack(th);
        let (_, _, r) = self.parts(&p);
        self.objective_from(&p, &r)
    }

    fn objective_from(&self, p: &Params, r: &[f64]) -> f64 {
        let d = self.d;
        let (n, s) = (d.t.len() as f64, p.sigma_obs);
        let mut f = p.k * p.k / 50.0 + p.m * p.m / 50.0;
        f += p.delta.iter().map(|v| v.abs()).sum::<f64>() / d.spec.changepoint_prior_scale;
        f += 2.0 * s * s;
        f += p
            .beta
            .iter()
            .zip(&d.prior_scales)
            .map(|(b, sc)| b * b / (2.0 * sc * sc))
            .sum::<f64>();
        f += n * s.ln() + r.iter().map(|v| v * v).sum::<f64>() / (2.0 * s * s);
        let f = f * self.scale;
        if self.guard && !f.is_finite() {
            return 1e300;
        }
        f
    }

    pub fn gradient(&self, th: &[f64]) -> Vec<f64> {
        let p = self.unpack(th);
        let (trend, xm, r) = self.parts(&p);
        self.gradient_from(th.len(), &p, &trend, &xm, &r)
    }

    /// Objective and gradient from ONE pass through the model.
    pub fn value_and_grad(&self, th: &[f64]) -> (f64, Vec<f64>) {
        let p = self.unpack(th);
        let (trend, xm, r) = self.parts(&p);
        (
            self.objective_from(&p, &r),
            self.gradient_from(th.len(), &p, &trend, &xm, &r),
        )
    }

    fn gradient_from(
        &self,
        n_theta: usize,
        p: &Params,
        trend: &[f64],
        xm: &[f64],
        r: &[f64],
    ) -> Vec<f64> {
        let d = self.d;
        let n = d.t.len();
        let s = p.sigma_obs;
        let inv_s2 = 1.0 / (s * s);
        let ncp = d.changepoints_t.len();
        let mut g = vec![0.0; n_theta];
        // dL/dtrend_i = -r_i (1 + xm_i) / σ²   (L = negative log posterior)
        let tbar: Vec<f64> = (0..n).map(|i| -r[i] * (1.0 + xm[i]) * inv_s2).collect();
        // trend parameters
        match d.spec.growth {
            Growth::Linear => {
                g[0] += tbar.iter().zip(&d.t).map(|(a, b)| a * b).sum::<f64>();
                g[1] += tbar.iter().sum::<f64>();
                let (mut st, mut s0, mut i) = (0.0, 0.0, n);
                for j in (0..ncp).rev() {
                    while i > 0 && d.t[i - 1] >= d.changepoints_t[j] {
                        i -= 1;
                        st += tbar[i] * d.t[i];
                        s0 += tbar[i];
                    }
                    g[2 + j] += st - d.changepoints_t[j] * s0;
                }
            }
            Growth::Flat => {
                g[1] += tbar.iter().sum::<f64>();
            }
            Growth::Logistic => {
                let cap = d.cap_scaled.as_ref().expect("cap");
                let cps = &d.changepoints_t;
                let gamma = logistic_gammas(p.k, p.m, &p.delta, cps);
                let mut k_s = vec![p.k];
                for j in 0..ncp {
                    k_s.push(k_s[j] + p.delta[j]);
                }
                // forward per-point k_t, m_t and adjoints w.r.t. k_t (kt_bar) and m_t (mt_bar)
                let (mut j, mut k_t, mut m_t) = (0usize, p.k, p.m);
                let (mut kbar_seg, mut mbar_seg) = (vec![0.0; ncp + 1], vec![0.0; ncp + 1]); // accumulated per segment index
                for i in 0..n {
                    while j < ncp && cps[j] <= d.t[i] {
                        k_t += p.delta[j];
                        m_t += gamma[j];
                        j += 1;
                    }
                    let z = k_t * (d.t[i] - m_t);
                    let sig = 1.0 / (1.0 + (-z).exp());
                    let dsig = cap[i] * sig * (1.0 - sig);
                    kbar_seg[j] += tbar[i] * dsig * (d.t[i] - m_t);
                    mbar_seg[j] += tbar[i] * dsig * (-k_t);
                }
                // k_t in segment j = k + Σ_{l<j} δ_l ; m_t in segment j = m + Σ_{l<j} γ_l
                let mut suffix_k = 0.0;
                let mut suffix_m = 0.0;
                let mut gbar = vec![0.0; ncp]; // adjoint of γ_l from m_t
                let mut dbar = vec![0.0; ncp];
                for seg in (0..=ncp).rev() {
                    suffix_k += kbar_seg[seg];
                    suffix_m += mbar_seg[seg];
                    if seg > 0 {
                        dbar[seg - 1] += suffix_k;
                        gbar[seg - 1] += suffix_m;
                    }
                }
                g[0] += suffix_k; // direct k
                g[1] += suffix_m; // direct m
                                  // reverse through γ_j = a_j b_j, a_j = cp_j − m − Σ_{i<j} γ_i, b_j = 1 − k_s[j]/k_s[j+1]
                let mut ksbar = vec![0.0; ncp + 1];
                let mut acc = 0.0; // Σ_{l>j} γ̄_l b_l
                let mut gsum = 0.0; // Σ_{i<j} γ_i, computed forward; store prefix sums
                let mut prefix = vec![0.0; ncp];
                for j in 0..ncp {
                    prefix[j] = gsum;
                    gsum += gamma[j];
                }
                for j in (0..ncp).rev() {
                    let b = 1.0 - k_s[j] / k_s[j + 1];
                    let a = cps[j] - p.m - prefix[j];
                    let gb = gbar[j] - acc; // total adjoint of γ_j (later a_l depend on γ_j with −1)
                    g[1] += -gb * b; // ∂a_j/∂m = −1
                    ksbar[j] += gb * a * (-1.0 / k_s[j + 1]);
                    ksbar[j + 1] += gb * a * (k_s[j] / (k_s[j + 1] * k_s[j + 1]));
                    acc += gb * b;
                }
                // k_s[j] = k + Σ_{l<j} δ_l
                let mut suf = 0.0;
                for j in (0..=ncp).rev() {
                    suf += ksbar[j];
                    if j > 0 {
                        dbar[j - 1] += suf;
                    }
                }
                g[0] += suf;
                for j in 0..ncp {
                    g[2 + j] += dbar[j];
                }
            }
        }
        // priors on k, m, δ
        g[0] += p.k / 25.0;
        g[1] += p.m / 25.0;
        for j in 0..ncp {
            g[2 + j] += p.delta[j].signum() * if p.delta[j] == 0.0 { 0.0 } else { 1.0 }
                / d.spec.changepoint_prior_scale;
        }
        // β: dμ_i/dβ_c = X_ic (trend_i s_m,c + s_a,c)
        let off = 2 + ncp;
        // one row-major pass over X: acc_c = Σ_i −r_i X_ic (trend_i s_m,c + s_a,c)
        let mut acc = vec![0.0; d.k];
        for i in 0..n {
            let row = &d.x[i * d.k..(i + 1) * d.k];
            let (ri, ti) = (-r[i], trend[i]);
            for c in 0..d.k {
                acc[c] += ri * row[c] * (ti * d.s_m[c] + d.s_a[c]);
            }
        }
        for c in 0..d.k {
            g[off + c] = p.beta[c] / (d.prior_scales[c] * d.prior_scales[c]) + acc[c] * inv_s2;
        }
        let ss: f64 = r.iter().map(|v| v * v).sum();
        g[off + d.k] = 4.0 * s * s + n as f64 - ss * inv_s2;
        for v in g.iter_mut() {
            *v *= self.scale;
        }
        if self.guard && g.iter().any(|v| !v.is_finite()) {
            return vec![0.0; n_theta];
        }
        g
    }
}

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
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    pub fn laplace(&mut self, b: f64) -> f64 {
        let u = self.uniform() - 0.5;
        -b * u.signum() * (1.0 - 2.0 * u.abs()).ln()
    }
}

/// Above this mean the draw switches from Knuth's product method to a normal approximation.
///
/// **CONTRACT-OWNED (WR-01).** The value lives in
/// `contracts/forecast-tool-boundary-v1.yaml` as `constants.poisson_normal_branch_lambda`,
/// and `types::tests::cost_bounds_match_contract` asserts this constant EQUAL to it through
/// `test_support::constant_f64`. D-15: the contract is the source and this is the mirror.
/// It was the one new behavioural constant with no contract mirror — it decides which
/// ALGORITHM a caller's request gets, so moving it in Rust alone changed shipped behaviour
/// with nothing to notice. Changing the YAML value alone now turns that test red naming the
/// key (observed, then restored).
///
/// The value the `06-VERIFICATION.md` gap-3 report named, and it is far above the regime the
/// parity ladder reaches: `wp_log_R_logistic` is the ONLY logistic fixture and its lambda is
/// **3.1239** (25 changepoints x (t_max 1.124957 - 1), measured by
/// `sampler::wp_log_r_logistic_fixture_lambda_is_far_below_the_branch_threshold`), so every parity
/// rung takes the unchanged Knuth path and no rung can move because of this branch.
pub const POISSON_NORMAL_BRANCH_LAMBDA: f64 = 30.0;

/// Draw a Poisson count with mean `lambda`.
///
/// Extracted from `predict`'s `Growth::Logistic` uncertainty arm so its numerical domain can be
/// asserted directly — the only path to it before was a full logistic forecast.
///
/// # Why two branches
///
/// Knuth's product method compares a running product of uniforms against `l = (-lambda).exp()`,
/// and that is **exactly 0.0 for lambda > 745.13**. Past that point the loop can only end when the
/// product itself underflows through f64's subnormals, which takes ~745 steps on average
/// (E[-ln U] = 1, and the smallest positive subnormal is near e^-744) — so the count SATURATES
/// near 745 whatever lambda is. That regime is reachable through the door: 100 daily points with
/// `horizon: 3650, growth: "logistic"` gives lambda ~= 922, and the 10-point `MIN_POINTS` floor
/// with the same horizon gives ~2839. Measured pre-fix means: 745.13 at lambda 900 and 745.45 at
/// lambda 2839.
///
/// # Why a normal approximation and not a PTRS port
///
/// A transformed-rejection port of numpy's `np.random.poisson` would buy a draw-for-draw match
/// this sampler never had: the stream here is this file's own seeded xorshift `Rng`, not MT19937.
/// The bar this draw feeds is `band_width_rel` — a RELATIVE bar on mean band width, stated in
/// `contracts/prophet-parity-v1.yaml` as a Monte-Carlo estimate — and a band-width estimate
/// depends on the count's MEAN and VARIANCE, both of which are exactly lambda for
/// `lambda + sqrt(lambda) * N(0, 1)`. The approximation is standard above lambda ~= 30, where the
/// Poisson's skew (1/sqrt(lambda) <= 0.18) is already small.
pub fn poisson(rng: &mut Rng, lambda: f64) -> usize {
    if lambda > POISSON_NORMAL_BRANCH_LAMBDA {
        // Mean and variance are both exactly lambda; clamped at zero because a normal draw is
        // unbounded below while a count is not (at the threshold that is a 5.5-sigma event).
        return (lambda + lambda.sqrt() * rng.normal()).round().max(0.0) as usize;
    }
    // Poisson via Knuth
    let mut n_changes = 0usize;
    let mut pp = 1.0;
    let l = (-lambda).exp();
    loop {
        pp *= rng.uniform();
        if pp <= l {
            break;
        }
        n_changes += 1;
    }
    n_changes
}

/// numpy's default (linear) percentile.
pub fn percentile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    let pos = q / 100.0 * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    sorted[lo] + (pos - lo as f64) * (sorted[hi] - sorted[lo])
}

// ----------------------------------------------------------- prediction ----
pub struct Forecast {
    pub ds_days: Vec<i64>,
    pub trend: Vec<f64>,
    pub yhat: Vec<f64>,
    pub yhat_lower: Vec<f64>,
    pub yhat_upper: Vec<f64>,
    pub trend_lower: Vec<f64>,
    pub trend_upper: Vec<f64>,
    /// Named components on the original scale (additive) or as fractions (multiplicative),
    /// including `additive_terms`, `multiplicative_terms`, `holidays`.
    pub components: Vec<(String, Vec<f64>)>,
}

/// Point estimates, components and the uncertainty band over `ds_days`.
///
/// `regs` is the per-row external-regressor value channel; pass
/// [`RegressorChannel::NONE`] when there are none. A regressor value is not a function of
/// the day, so it cannot come out of [`feature_row`] (which is keyed on `day`) and has to
/// travel beside the design as caller data.
///
/// # Errors
///
/// Returns [`ForecastError::Validation`] when the channel is malformed: a `values` array
/// whose length is not exactly `ds_days.len()`, a spec/values count mismatch, or a channel
/// claiming more columns than the design has.
///
/// This is an ERROR and not a panic on purpose (IN-02, the same rule as `feature_row`'s
/// total lookup): every `Design` field is `pub` and these slices are caller-built, and a
/// library must not abort the caller's process over them. The check is load-bearing rather
/// than defensive — `forecast.rs` calls this with `fut`, whose length is `horizon`, NOT
/// `ds.len() + horizon`, so the door must hand over only the `values[ds.len()..]` tail. An
/// off-by-one there would silently offset every future feature instead of failing.
pub fn predict(
    d: &Design,
    p: &Params,
    ds_days: &[i64],
    seed: u64,
    regs: &crate::regressors::RegressorChannel<'_>,
) -> Result<Forecast, crate::types::ForecastError> {
    let spec = &d.spec;
    let n = ds_days.len();

    // ---- the channel is well-formed, asserted AT THE BOUNDARY ----
    if regs.values.len() != regs.specs.len() {
        return Err(crate::types::ForecastError::Validation(format!(
            "the regressor channel carries {} value arrays for {} regressors; they must \
             correspond one-to-one",
            regs.values.len(),
            regs.specs.len()
        )));
    }
    let Some(base_k) = d.k.checked_sub(regs.len()) else {
        return Err(crate::types::ForecastError::Validation(format!(
            "the regressor channel claims {} columns but the design has only {}; the \
             channel does not belong to this design",
            regs.len(),
            d.k
        )));
    };
    for (j, v) in regs.values.iter().enumerate() {
        if v.len() != n {
            return Err(crate::types::ForecastError::Validation(format!(
                "regressor {j} carries {} values for {n} predicted rows; it must carry \
                 exactly one value per row",
                v.len()
            )));
        }
    }

    let t: Vec<f64> = ds_days
        .iter()
        .map(|&x| (x - d.start_days) as f64 / d.t_scale_days)
        .collect();
    let cap: Option<Vec<f64>> = d.cap_scaled.as_ref().map(|c| vec![c[0]; n]);
    let model = Model::new(d);
    let trend_s = model.trend(p, &t, cap.as_deref());
    // The base (seasonality + holiday) cells still come from the UNCHANGED `feature_row`;
    // only the trailing regressor cells are written from the caller's channel. With an
    // empty channel `base_k == d.k` and this is arithmetically the original loop, which is
    // what the D-19 baseline re-checks.
    let base_cols = &d.cols[..base_k];
    let mut x = Vec::with_capacity(n * d.k);
    let hol_sets = holiday_day_sets(spec);
    for (i, &day) in ds_days.iter().enumerate() {
        feature_row(day, spec, base_cols, &hol_sets, &mut x);
        for (j, reg) in regs.specs.iter().enumerate() {
            x.push((regs.values[j][i] - reg.mu) / reg.std);
        }
    }
    // components
    //
    // Each distinct `component` value is collected ONCE, together with the SPAN of column
    // indices that carry it — the first and last index seen, inclusive. `d.cols` is walked
    // in ascending index order, so the first occurrence fixes the low end and every later
    // one raises the high end.
    //
    // The span is what stops the per-component roll-up being QUADRATIC in the regressor
    // count. `comp_of` is called once per distinct component and each call used to scan all
    // `d.k` columns, so the per-component work was `n * distinct_components * d.k` — and a
    // regressor contributes one to BOTH factors, making it `O(n * R^2)` for work that is
    // inherently `O(n * R)`: a regressor component is EXACTLY ONE column.
    //
    // MEASURED before and after, and the ordering of those two facts matters: under
    // `MAX_REGRESSOR_DESIGN_COST` the quadratic term was already bounded to ~5.6e6 float
    // ops (`(len(ds) + horizon) * R <= 25 000`, so `n * R * (34 + R) <= 25 000 * 226`), and
    // the stage-1 sweep measured `predict_s` at 0.000-0.016 s at every at-the-bound
    // composition — under 1 % of SC1's 2 s bar. So this was NOT urgent, and the record
    // should not pretend otherwise. It is fixed because Phase 7 re-prices these ceilings
    // per TIER: the quadratic is unreachable at the shipped bound and reachable the moment
    // the bound moves, which is the worst time to discover a cost shape.
    //
    // THE SPAN IS A SUPERSET, NEVER A SUBSTITUTE FOR THE SELECTOR, and that is what makes
    // the change BITWISE-IDENTICAL rather than merely equivalent. Every column outside
    // `first..=last` has a different `component` by construction, so the selector was
    // already false there and it contributed nothing; the set of accumulated terms, their
    // ORDER, and the accumulator's initial `0.0` are all unchanged. Dropping the accumulator
    // for the single-column case would NOT be safe — `0.0 + (-0.0)` is `+0.0` while the bare
    // product is `-0.0` — so the accumulator stays.
    // `predict::the_span_restricted_roll_up_matches_a_full_scan` is the falsification probe.
    let mut names: Vec<String> = Vec::new();
    let mut spans: Vec<(usize, usize)> = Vec::new();
    for (ci, c) in d.cols.iter().enumerate() {
        if let Some(pos) = names.iter().position(|n| n == &c.component) {
            spans[pos].1 = ci;
        } else {
            names.push(c.component.clone());
            spans.push((ci, ci));
        }
    }
    let comp_of = |sel: &dyn Fn(&Column) -> bool,
                   additive: bool,
                   range: std::ops::Range<usize>|
     -> Vec<f64> {
        (0..n)
            .map(|i| {
                let mut v = 0.0;
                for c in range.clone() {
                    if sel(&d.cols[c]) {
                        v += x[i * d.k + c] * p.beta[c];
                    }
                }
                if additive {
                    v * d.y_scale
                } else {
                    v
                }
            })
            .collect()
    };
    let mut components: Vec<(String, Vec<f64>)> = Vec::new();
    for (nm, &(first, last)) in names.iter().zip(spans.iter()) {
        // `first` IS the index of this component's first column — that is exactly what
        // `spans.push((ci, ci))` recorded at first appearance above — so this is the same
        // column the linear `find` used to walk `d.cols` to locate, at O(1) instead of
        // O(columns) per component. Every column in `first..=last` carries the same mode
        // by construction. No float term moves, so the response is bitwise unchanged;
        // `the_span_restricted_roll_up_matches_a_full_scan` is the probe.
        let mode = d.cols[first].mode;
        let v = comp_of(
            &|c: &Column| &c.component == nm,
            mode == Mode::Additive,
            first..last + 1,
        );
        components.push((nm.clone(), v));
    }
    // The AGGREGATES still scan every column, and must: `additive_terms` is a roll-up ACROSS
    // components, so it has no span to restrict to. Both are `O(n * K)` — linear — which is
    // why they were never part of the quadratic.
    let add_terms = comp_of(&|c: &Column| c.mode == Mode::Additive, true, 0..d.k);
    let mul_terms = comp_of(&|c: &Column| c.mode == Mode::Multiplicative, false, 0..d.k);
    if !spec.holidays.is_empty() {
        let hol = comp_of(
            &|c: &Column| c.holiday.is_some(),
            spec.holidays_mode == Mode::Additive,
            0..d.k,
        );
        components.push(("holidays".into(), hol));
    }
    // The two roll-ups Prophet publishes by name. Emitted ONLY when the channel is
    // non-empty, so the no-regressor component map is unchanged and the D-19 baseline holds.
    //
    // Columns are selected by the STRUCTURAL trailing index range `base_k..d.k`, never by
    // name membership. Name membership is what the spike prototype did, and it is silently
    // wrong under a collision: a regressor literally named `yearly_delim_1` would be summed
    // in here.
    if !regs.is_empty() {
        let roll_up = |want: Mode| -> Vec<f64> {
            (0..n)
                .map(|i| {
                    let mut v = 0.0;
                    for c in base_k..d.k {
                        if d.cols[c].mode == want {
                            v += x[i * d.k + c] * p.beta[c];
                        }
                    }
                    if want == Mode::Additive {
                        v * d.y_scale
                    } else {
                        v
                    }
                })
                .collect()
        };
        components.push(("extra_regressors_additive".into(), roll_up(Mode::Additive)));
        components.push((
            "extra_regressors_multiplicative".into(),
            roll_up(Mode::Multiplicative),
        ));
    }
    components.push(("additive_terms".into(), add_terms.clone()));
    components.push(("multiplicative_terms".into(), mul_terms.clone()));
    let trend: Vec<f64> = trend_s.iter().map(|v| v * d.y_scale).collect();
    let yhat: Vec<f64> = (0..n)
        .map(|i| trend[i] * (1.0 + mul_terms[i]) + add_terms[i])
        .collect();

    // ---- uncertainty: Prophet 1.4 vectorised path ----
    let ns = spec.uncertainty_samples;
    let mut rng = Rng::new(seed);
    let n_future = t.iter().filter(|&&v| v > 1.0).count();
    let n_past = n - n_future;
    // uncertainties[s][i] on the scaled trend
    let mut unc = vec![vec![0.0; n]; ns];
    if n_future > 0 {
        let future_t: Vec<f64> = t.iter().cloned().filter(|&v| v > 1.0).collect();
        let single_diff = if n_future > 1 {
            (future_t[n_future - 1] - future_t[0]) / (n_future - 1) as f64
        } else {
            (d.t[d.t.len() - 1] - d.t[0]) / (d.t.len() - 1) as f64
        };
        let likelihood = d.changepoints_t.len() as f64 * single_diff;
        let mean_delta = p.delta.iter().map(|v| v.abs()).sum::<f64>() / p.delta.len() as f64 + 1e-8;
        match spec.growth {
            Growth::Linear => {
                for row in unc.iter_mut() {
                    // _make_trend_shift_matrix: laplace shifts where U < likelihood, averaged with the previous column
                    let mut mat: Vec<f64> = (0..n_future)
                        .map(|_| {
                            let hit = rng.uniform() < likelihood;
                            let v = rng.laplace(mean_delta);
                            if hit {
                                v
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    for i in (0..n_future).rev() {
                        let prev = if i == 0 { 0.0 } else { mat[i - 1] };
                        mat[i] = (prev + mat[i]) / 2.0;
                    }
                    let mut c1 = 0.0;
                    let mut c2 = 0.0;
                    for i in 0..n_future {
                        c1 += mat[i];
                        c2 += c1;
                        row[n_past + i] = c2 * single_diff;
                    }
                }
            }
            Growth::Flat => {}
            Growth::Logistic => {
                // Prophet's pre-vectorised algorithm (sample_predictive_trend): Poisson-many new
                // changepoints on (1, T], Laplace deltas, full piecewise_logistic re-evaluation.
                let cap_s = cap.as_ref().expect("cap");
                let t_max = t[n - 1];
                let s_cnt = d.changepoints_t.len() as f64;
                let mean_trend = &trend_s;
                for row in unc.iter_mut() {
                    let lambda = s_cnt * (t_max - 1.0);
                    let n_changes = poisson(&mut rng, lambda);
                    let mut cps: Vec<f64> = d.changepoints_t.clone();
                    let mut deltas: Vec<f64> = p.delta.clone();
                    let mut new: Vec<(f64, f64)> = (0..n_changes)
                        .map(|_| (1.0 + rng.uniform() * (t_max - 1.0), rng.laplace(mean_delta)))
                        .collect();
                    new.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("f"));
                    for (c, dl) in new {
                        cps.push(c);
                        deltas.push(dl);
                    }
                    let tr = piecewise_logistic(&t, cap_s, &cps, &deltas, p.k, p.m);
                    for i in n_past..n {
                        row[i] = tr[i] - mean_trend[i];
                    }
                }
            }
        }
    }
    // sample_model_vectorized: yhat = (trend + unc)·y_scale·(1 + Xb_m) + Xb_a + N(0, σ)·y_scale
    let (lo_p, hi_p) = (
        100.0 * (1.0 - spec.interval_width) / 2.0,
        100.0 * (1.0 + spec.interval_width) / 2.0,
    );
    let (mut yl, mut yu, mut tl, mut tu) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    let mut col_y = vec![0.0; ns];
    let mut col_t = vec![0.0; ns];
    for i in 0..n {
        for s in 0..ns {
            let tr = (trend_s[i] + unc[s][i]) * d.y_scale;
            col_t[s] = tr;
            col_y[s] =
                tr * (1.0 + mul_terms[i]) + add_terms[i] + rng.normal() * p.sigma_obs * d.y_scale;
        }
        col_y.sort_by(|a, b| a.partial_cmp(b).expect("f"));
        col_t.sort_by(|a, b| a.partial_cmp(b).expect("f"));
        yl.push(percentile(&col_y, lo_p));
        yu.push(percentile(&col_y, hi_p));
        tl.push(percentile(&col_t, lo_p));
        tu.push(percentile(&col_t, hi_p));
    }
    Ok(Forecast {
        ds_days: ds_days.to_vec(),
        trend,
        yhat,
        yhat_lower: yl,
        yhat_upper: yu,
        trend_lower: tl,
        trend_upper: tu,
        components,
    })
}

#[cfg(test)]
mod sampler {
    //! FALSIFICATION of the changepoint-count sampler's numerical DOMAIN (VERIFICATION gap 3).
    //!
    //! Deliberately NOT in [`super::parity`]: `prophet::parity` means "the Prophet 1.4.0 ladder"
    //! and its count (32) is asserted by this plan's own verification, so a sampler test living
    //! there would change what that number means.
    //!
    //! What is barred here is the sampler's MEAN, its VARIANCE and its ZERO MASS against its own
    //! lambda, at seven lambdas that straddle both the branch threshold and Knuth's 745.13
    //! underflow point. One failing input is an anecdote (CLAUDE.md Verification Discipline rule
    //! 6); seven points make the shape of the failure visible — the pre-fix implementation tracks
    //! lambda up to ~745 and saturates there for anything larger, which is a different claim from
    //! "it is wrong at 900".
    //!
    //! # Why THREE statistics and not one (WR-02, and WR-01's second half)
    //!
    //! Until plan 06-17 this module barred the MEAN alone, and one statistic is not a domain
    //! check. Two implementations that are plainly wrong passed it:
    //!
    //! - `fn poisson(_rng, lambda) -> lambda.round() as usize` — a ZERO-VARIANCE stub, which would
    //!   collapse every logistic band to the deterministic trend while the caller is still told
    //!   the interval has the `interval_width` it asked for — reported `rel = 0.000000` at every
    //!   sweep point and passed outright. The VARIANCE bar is what refuses it.
    //! - `POISSON_NORMAL_BRANCH_LAMBDA` lowered to 3.2 — replacing the exact sampler with a normal
    //!   approximation for every real logistic request, in the regime the contract itself says the
    //!   approximation is not standard for — left every mean inside the 0.01 bar, because the mean
    //!   is exactly the statistic the normal approximation gets right. The ZERO-MASS bar is what
    //!   refuses it: the clamped normal's low tail draws 3.28x too many zeros at lambda = 5.
    //!
    //! Both were OBSERVED red before these bars were accepted as green (06-17 SUMMARY).
    use super::{poisson, Rng, POISSON_NORMAL_BRANCH_LAMBDA};
    use crate::dates::parse_ymd;
    use crate::test_support::{equation_float, equation_tolerance, load_json};

    /// Draws per lambda.
    ///
    /// **60 000, raised from 20 000 by plan 06-17, because the sample size is what makes all
    /// three bars feasible at every point.** The two new statistics cannot BOTH clear 4 sigma at
    /// N = 20 000: one shared `zero_mass_tolerance` must be at least 4 sigma above the noise at
    /// lambda = 5 (>= 0.3434 at N = 20 000) and at most half the smaller defect gap so it still
    /// discriminates at lambda = 3 (<= 0.2478), and those two do not overlap. At N = 60 000 the
    /// window is [0.1983, 0.2478] and 0.22 sits inside it. Raising N also strengthens the
    /// EXISTING mean bar, which the lambda = 3 point this plan adds needed: 2.45 sigma there at
    /// N = 20 000, 4.24 sigma at N = 60 000.
    ///
    /// The Knuth branch runs ~lambda iterations and the normal branch is O(1), so seven lambdas x
    /// 60 000 draws is a few million uniforms — measured well under a second.
    const N: usize = 60_000;
    const N_F: f64 = 60_000.0;

    /// Fixed so the sweep is reproducible. The bars are on the DISTRIBUTION, never on a
    /// draw-for-draw match with numpy — this file's `Rng` is a xorshift, not MT19937.
    const SEED: u64 = 20_260_907;

    /// Seven lambdas: three below the branch threshold (Knuth must stay untouched), two above it
    /// but below Knuth's underflow point, and the two the verifier measured as reachable IN
    /// BOUNDS (100 daily points + horizon 3650 gives ~922; the 10-point floor gives ~2839).
    ///
    /// `3.0` was added by 06-17: the only logistic parity fixture's own measured lambda is
    /// 3.1239, so 3.0 is the regime the ladder actually runs in, and it is the point at which a
    /// lowered branch threshold does its damage.
    const LAMBDAS: [f64; 7] = [3.0, 5.0, 29.0, 31.0, 100.0, 900.0, 2839.0];

    /// A zero-mass bar is only applied where the expected zero COUNT is large enough to measure.
    ///
    /// At N = 60 000 this selects lambda = 3.0 (2 987 expected zeros) and lambda = 5.0 (404), and
    /// skips everything above — `exp(-29)` is ~2.5e-13, so a zero at lambda = 29 is not an
    /// observation, it is a rounding artefact. The skipped points are PRINTED with their expected
    /// count and this reason, because a silently-skipped assertion is exactly the vacuous case
    /// this module exists to refuse.
    const MIN_EXPECTED_ZEROS: f64 = 30.0;

    /// One lambda's measured moments. Every field is measured in the first pass and asserted in
    /// the second, so no bar can abort the sweep before the rest have been measured.
    struct Point {
        lambda: f64,
        mean: f64,
        rel: f64,
        var: f64,
        var_rel: f64,
        expected_zeros: f64,
        p_zero: f64,
        exact_p_zero: f64,
        zero_rel: f64,
        zero_checked: bool,
    }

    #[test]
    fn poisson_mean_and_variance_track_lambda_across_its_whole_domain() {
        // D-15: every bar lives in the contract, never in a literal here. A test that hardcodes a
        // tolerance can be loosened without the contract ever noticing. `float_tolerance` is the
        // mean bar 06-13 shipped and is READ THROUGH THE SAME READER it always was; the two new
        // bars need a reader keyed on the KEY as well, because one equation now carries three.
        let mean_bar = equation_tolerance("prophet-parity-v1", "poisson_sampler_domain");
        let var_bar = equation_float(
            "prophet-parity-v1",
            "poisson_sampler_domain",
            "variance_tolerance",
        );
        let zero_bar = equation_float(
            "prophet-parity-v1",
            "poisson_sampler_domain",
            "zero_mass_tolerance",
        );
        // Measure and PRINT all seven first, then assert: a print-and-assert loop would abort at
        // the first out-of-domain lambda and hide the shape of the failure at the larger ones.
        let mut observed: Vec<Point> = Vec::with_capacity(LAMBDAS.len());
        for (i, &lambda) in LAMBDAS.iter().enumerate() {
            let mut rng = Rng::new(SEED + u64::try_from(i).expect("index fits u64"));
            // The draws are KEPT rather than only summed: a variance and a zero count cannot be
            // read off a running sum, and the whole point of this plan is that one statistic was
            // not enough.
            let mut draws: Vec<f64> = Vec::with_capacity(N);
            let mut zeros = 0_usize;
            for _ in 0..N {
                let k = poisson(&mut rng, lambda);
                if k == 0 {
                    zeros += 1;
                }
                draws.push(f64::from(
                    u32::try_from(k).expect("a Poisson count fits u32"),
                ));
            }
            let mean = draws.iter().sum::<f64>() / N_F;
            let rel = (mean - lambda).abs() / lambda;
            let var = draws.iter().map(|&k| (k - mean).powi(2)).sum::<f64>() / (N_F - 1.0);
            let var_rel = (var - lambda).abs() / lambda;
            let exact_p_zero = (-lambda).exp();
            let expected_zeros = N_F * exact_p_zero;
            let p_zero = f64::from(u32::try_from(zeros).expect("a zero count fits u32")) / N_F;
            let zero_rel = (p_zero - exact_p_zero).abs() / exact_p_zero;
            let zero_checked = expected_zeros >= MIN_EXPECTED_ZEROS;
            println!("POISSON MEAN: lambda={lambda:.1} n={N} mean={mean:.4} rel={rel:.6} (bar {mean_bar:e})");
            println!(
                "POISSON VAR:  lambda={lambda:.1} n={N} var={var:.4} rel={var_rel:.6} (bar {var_bar:e})"
            );
            if zero_checked {
                println!(
                    "POISSON ZERO: lambda={lambda:.1} CHECKED expected_zeros={expected_zeros:.1} \
                     (>= {MIN_EXPECTED_ZEROS:.0}) observed={zeros} p_zero={p_zero:.6} \
                     exact={exact_p_zero:.6} rel={zero_rel:.6} (bar {zero_bar:e})"
                );
            } else {
                println!(
                    "POISSON ZERO: lambda={lambda:.1} SKIPPED expected_zeros={expected_zeros:.3e} \
                     (< {MIN_EXPECTED_ZEROS:.0}, too few to measure) observed={zeros} \
                     exact={exact_p_zero:.3e}"
                );
            }
            observed.push(Point {
                lambda,
                mean,
                rel,
                var,
                var_rel,
                expected_zeros,
                p_zero,
                exact_p_zero,
                zero_rel,
                zero_checked,
            });
        }
        // Three assertion passes, not one interleaved pass, for the same reason the measurement
        // is separated from the assertion: a variance failure at lambda = 3 must not hide the
        // mean's verdict at lambda = 2839.
        for p in &observed {
            assert!(
                p.rel <= mean_bar,
                "poisson sampler MEAN outside its domain at lambda={:.1}: \
                 mean={:.4} rel={:.6} > bar={:e}",
                p.lambda,
                p.mean,
                p.rel,
                mean_bar
            );
        }
        for p in &observed {
            assert!(
                p.var_rel <= var_bar,
                "poisson sampler VARIANCE outside its domain at lambda={:.1}: \
                 var={:.4} rel={:.6} > bar={:e}. The codomain is a count whose mean AND VARIANCE \
                 are both lambda; a sampler with the right centre and the wrong spread returns \
                 yhat_lower / yhat_upper narrower than the interval_width it advertises",
                p.lambda,
                p.var,
                p.var_rel,
                var_bar
            );
        }
        let mut checked_zero_mass = 0_usize;
        for p in &observed {
            if !p.zero_checked {
                continue;
            }
            checked_zero_mass += 1;
            assert!(
                p.zero_rel <= zero_bar,
                "poisson sampler ZERO MASS outside its domain at lambda={:.1}: \
                 p_zero={:.6} against an exact exp(-lambda)={:.6} (rel={:.6} > bar={:e}, \
                 expected_zeros={:.1}). The low tail is where the clamped normal approximation \
                 is visibly wrong and the exact sampler is not, so this is the bar that notices \
                 POISSON_NORMAL_BRANCH_LAMBDA being lowered",
                p.lambda,
                p.p_zero,
                p.exact_p_zero,
                p.zero_rel,
                zero_bar,
                p.expected_zeros
            );
        }
        // NON-VACUITY: a zero-mass bar that checked nothing would be a green that means nothing,
        // which is the whole class this round is closing. Two points clear MIN_EXPECTED_ZEROS at
        // N = 60 000 (lambda 3.0 and 5.0) and the assertion says so by number.
        assert_eq!(
            checked_zero_mass, 2,
            "the zero-mass bar must actually run at lambda 3.0 and 5.0; it ran at \
             {checked_zero_mass} point(s), so either N, LAMBDAS or MIN_EXPECTED_ZEROS moved and \
             the bar that detects a lowered branch threshold is no longer being applied"
        );
    }

    /// The parity argument, as a MEASUREMENT rather than a claim.
    ///
    /// `wp_log_R_logistic` is the only logistic fixture, so it is the only parity rung that can
    /// reach this sampler at all. Its lambda is `changepoints_t.len() * (t_max - 1)`, and both
    /// factors are published by the fixture — the `..._data_prep_exact` rung already asserts that
    /// Rust's `make_design` reproduces `changepoints_t` exactly, so reading them here measures the
    /// same quantity the ladder runs on.
    #[test]
    fn wp_log_r_logistic_fixture_lambda_is_far_below_the_branch_threshold() {
        let fx = load_json("wp_log_R_logistic_prophet140.json");
        let n_cp = fx["changepoints_t"]
            .as_array()
            .expect("fixture publishes changepoints_t")
            .len();
        let start = parse_ymd(fx["start"].as_str().expect("fixture publishes start"));
        let t_scale = fx["t_scale_days"]
            .as_f64()
            .expect("fixture publishes t_scale_days");
        let last_ds = fx["forecast"]["ds"]
            .as_array()
            .expect("fixture publishes forecast.ds")
            .last()
            .and_then(serde_json::Value::as_str)
            .expect("forecast.ds is non-empty");
        let t_max = f64::from(
            i32::try_from(parse_ymd(last_ds) - start).expect("a forecast span fits i32 days"),
        ) / t_scale;
        let lambda =
            f64::from(u32::try_from(n_cp).expect("a changepoint count fits u32")) * (t_max - 1.0);
        println!(
            "FIXTURE LAMBDA: fixture=wp_log_R_logistic n_changepoints={n_cp} \
             t_max={t_max:.6} lambda={lambda:.4}"
        );
        // If this ever goes red the parity ladder has started exercising the
        // normal-approximation branch, and the claim that no rung moved because of it would no
        // longer be a measurement.
        assert!(
            lambda < POISSON_NORMAL_BRANCH_LAMBDA,
            "the only logistic parity fixture now reaches lambda={lambda:.4}, \
             at or above the sampler's branch threshold {POISSON_NORMAL_BRANCH_LAMBDA}"
        );
    }

    // `logistic_band_wall` USED TO LIVE HERE and was DELETED by plan 06-16 (WR-04).
    //
    // It was `#[ignore]`d, had no `just` recipe and asserted no bar, so the one harness
    // watching the axis CR-01 actually lived on could only ever be reached by someone who
    // already knew to pass `--ignored` — and all four of its recorded lines were
    // `freq: "D"`, the single value that does not exhibit the defect. Its whole geometry
    // (33 points at the tightest legal daily span, `growth: "logistic"`, the widest horizon
    // each frequency can legally ask for) is now one row of `crate::sc1_wall`'s cross
    // product, which runs with NO flag, prints `band_width=` on every line exactly as this
    // harness did, and asserts the 2 s SC1 bar on a release profile.
    //
    // Deleted rather than folded: unlike `design_cost::holiday_design_wall`, nothing drove
    // it. It had no recipe and no caller, so there was no entry point to preserve — only a
    // second geometry to stop maintaining. `just forecast-sc1-sweep` is its replacement.
}

#[cfg(test)]
mod parity {
    //! The Prophet 1.4.0 parity ladder (SC2, D-04).
    //!
    //! ONE TEST PER RUNG PER FIXTURE, so a regression names the rung and the fixture rather
    //! than "Prophet broke". The rungs, in the order they must be believed:
    //!
    //! 1. `<f>_data_prep_exact` — the Stan data block, rebuilt from raw `(ds, y)`.
    //! 2. `<f>_objective_at_python_map` — the objective at PYTHON's MAP vs Python's own `-lp`.
    //! 3. `<f>_predict_path_via_python_params` — Python's parameters through Rust `predict`,
    //!    with the named components and the `trend*(1+mul)+add == yhat` identity.
    //! 4. `<f>_fit_objective_and_forecast` — what the Rust MAP fit itself reaches, plus the
    //!    D-09 diagnostics.
    //! 5. `<f>_band_widths_within_contract` — the 80 % interval widths.
    //!
    //! NO TOLERANCE LITERAL LIVES HERE (D-15). Every bar is read at test time from
    //! `contracts/prophet-parity-v1.yaml` through [`equation_tolerance`], and `max_rounds`
    //! through [`constant_u64`]. A bar that lives in a test can be loosened without the
    //! contract noticing; a bar that lives in the contract moves only as a `pv diff`-visible
    //! edit.
    //!
    //! WHAT IS DELIBERATELY NOT BARRED. `air_passengers` and `retail_sales` have no committed
    //! future-`yhat` control band: the yearly Fourier block is near-unidentified on monthly
    //! data, so Prophet's own two optimisers disagree there by far more than any epsilon worth
    //! writing down. Those numbers are RECORDED (printed, and carried in the assertion message)
    //! and asserted against nothing — the contract says why.

    use super::{
        make_design, predict, Design, Forecast, Growth, Holiday, Mode, Model, Params, Seasonality,
        Spec,
    };
    use crate::dates::{civil_from_days, days_from_civil, future_days, parse_ymd};
    use crate::fit::{fit_prophet, FitInfo};
    use crate::test_support::{constant_u64, equation_tolerance, load_json};
    use serde_json::Value;
    use std::sync::{Arc, OnceLock};

    /// The uncertainty seed every band rung draws with. The fixtures' Python bands come from
    /// Prophet's own RNG, so this only has to be FIXED, never matched — the bar is relative
    /// and wide enough to cover the ~1.2 % seed-to-seed variance the spike measured.
    const SEED: u64 = 42;

    /// The seven committed Prophet 1.4.0 oracles: `(test-name stem, fixture file)`.
    ///
    /// The first three are spike 001 (they publish `log_posterior_at_map_unnormalized` and
    /// carry no `uncertainty` block); the last four are spike 003 (they publish `columns`,
    /// `s_a`, `s_m` and a precomputed `uncertainty` block, and no `-lp`).
    const FIXTURES: [(&str, &str); 7] = [
        ("peyton_manning", "peyton_manning_prophet140.json"),
        ("air_passengers", "air_passengers_prophet140.json"),
        ("retail_sales", "retail_sales_prophet140.json"),
        ("peyton_default", "peyton_default_prophet140.json"),
        ("peyton_holidays", "peyton_holidays_prophet140.json"),
        ("wp_log_r_logistic", "wp_log_R_logistic_prophet140.json"),
        ("air_multiplicative", "air_multiplicative_prophet140.json"),
    ];

    /// The two fixtures that carry a committed Newton-vs-L-BFGS control band on future `yhat`.
    const PEYTON_BANDED: [&str; 2] = ["peyton_manning", "peyton_default"];

    fn fixture_index(stem: &str) -> usize {
        FIXTURES
            .iter()
            .position(|(s, _)| *s == stem)
            .unwrap_or_else(|| panic!("{stem} is not one of the seven committed fixtures"))
    }

    fn f64s(v: &Value, what: &str) -> Vec<f64> {
        v.as_array()
            .unwrap_or_else(|| panic!("{what} must be a JSON array"))
            .iter()
            .map(|x| {
                x.as_f64()
                    .unwrap_or_else(|| panic!("{what} must hold only numbers"))
            })
            .collect()
    }

    fn strings(v: &Value, what: &str) -> Vec<String> {
        v.as_array()
            .unwrap_or_else(|| panic!("{what} must be a JSON array"))
            .iter()
            .map(|x| {
                x.as_str()
                    .unwrap_or_else(|| panic!("{what} must hold only strings"))
                    .to_string()
            })
            .collect()
    }

    fn days(v: &Value, what: &str) -> Vec<i64> {
        strings(v, what).iter().map(|s| parse_ymd(s)).collect()
    }

    fn max_abs_diff(a: &[f64], b: &[f64], what: &str) -> f64 {
        assert_eq!(a.len(), b.len(), "{what}: length mismatch");
        assert!(!a.is_empty(), "{what}: refusing to compare empty vectors");
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    }

    fn mean(v: &[f64]) -> f64 {
        assert!(!v.is_empty(), "mean of an empty slice");
        v.iter().sum::<f64>() / v.len() as f64
    }

    /// Mean width of an interval, the quantity Prophet's fixtures publish.
    fn mean_width(lo: &[f64], hi: &[f64]) -> f64 {
        assert_eq!(lo.len(), hi.len(), "band bounds must be parallel");
        mean(&lo.iter().zip(hi).map(|(a, b)| b - a).collect::<Vec<_>>())
    }

    /// Assert a RELATIVE agreement and report the measured deviation either way.
    fn within_rel(rust: f64, python: f64, rel_bar: f64, what: &str) {
        assert!(
            python.abs() > 0.0,
            "{what}: the Python reference is zero, so a relative bar is meaningless"
        );
        let dev = (rust - python).abs() / python.abs();
        println!(
            "    {what}: rust {rust:.6} vs python {python:.6} -> {:.4} % ",
            dev * 100.0
        );
        assert!(
            dev <= rel_bar,
            "{what}: rust {rust} vs python {python} is {dev:e} relative, over the contract bar {rel_bar:e}"
        );
    }

    /// One fixture, loaded once and rebuilt into the design the port would build itself.
    struct Ladder {
        fx: Value,
        design: Design,
        ds_days: Vec<i64>,
        forecast_days: Vec<i64>,
        n_hist: usize,
    }

    impl Ladder {
        /// Python Prophet 1.4.0's MAP, verbatim from the fixture.
        fn python_params(&self) -> Params {
            let arr = |key: &str| f64s(&self.fx["params"][key], &format!("params.{key}"));
            Params {
                k: self.fx["params"]["k"].as_f64().expect("params.k"),
                m: self.fx["params"]["m"].as_f64().expect("params.m"),
                delta: arr("delta"),
                beta: arr("beta"),
                sigma_obs: self.fx["params"]["sigma_obs"]
                    .as_f64()
                    .expect("params.sigma_obs"),
            }
        }

        /// Spike-003 fixtures carry the precomputed `uncertainty` block; spike-001 ones do not.
        fn is_spike_003(&self) -> bool {
            self.fx.get("uncertainty").is_some_and(|u| u.is_object())
        }

        /// The Python objective at Python's MAP, UNSCALED, sign convention `f = -log posterior`.
        ///
        /// Where the fixture publishes `log_posterior_at_map_unnormalized` this is Python's own
        /// number. Where it does not (the four spike-003 files), it is the Rust objective at
        /// Python's MAP — a legitimate stand-in precisely because rung 2 proves, on every
        /// fixture that DOES publish `-lp`, that this quantity is Python's objective.
        fn f_python(&self) -> f64 {
            if let Some(lp) = self
                .fx
                .get("log_posterior_at_map_unnormalized")
                .and_then(Value::as_f64)
            {
                -lp
            } else {
                let model = Model::new(&self.design);
                model.objective(&model.pack(&self.python_params())) / model.scale
            }
        }
    }

    /// Rebuild the fixture's `Spec` from the fixture alone — never from a hand-copied default.
    fn spec_of(fx: &Value) -> Spec {
        let mode_of = |v: &Value| {
            if v.as_str() == Some("multiplicative") {
                Mode::Multiplicative
            } else {
                Mode::Additive
            }
        };
        let seasonalities: Vec<Seasonality> = fx["seasonalities"]
            .as_array()
            .expect("seasonalities")
            .iter()
            .map(|s| Seasonality {
                name: s["name"].as_str().expect("seasonality.name").to_string(),
                period: s["period"].as_f64().expect("seasonality.period"),
                order: usize::try_from(s["fourier_order"].as_u64().expect("fourier_order"))
                    .expect("fourier order fits usize"),
                prior_scale: s["prior_scale"].as_f64().expect("seasonality.prior_scale"),
                mode: mode_of(&s["mode"]),
            })
            .collect();

        // Holidays arrive as one row per (name, date); group them, preserving first-seen order.
        let mut holidays: Vec<Holiday> = Vec::new();
        let holiday_prior = fx["holidays_prior_scale"].as_f64().unwrap_or(10.0);
        let no_holidays: Vec<Value> = Vec::new();
        for h in fx["holidays"].as_array().unwrap_or(&no_holidays) {
            let name = h["holiday"].as_str().expect("holiday.holiday").to_string();
            let day = parse_ymd(h["ds"].as_str().expect("holiday.ds"));
            if let Some(existing) = holidays.iter_mut().find(|x| x.name == name) {
                existing.days.push(day);
            } else {
                holidays.push(Holiday {
                    name,
                    days: vec![day],
                    lower_window: h["lower_window"].as_i64().expect("lower_window"),
                    upper_window: h["upper_window"].as_i64().expect("upper_window"),
                    prior_scale: holiday_prior,
                });
            }
        }

        let mut spec = Spec::default_linear(seasonalities);
        spec.growth = match fx["growth"].as_str() {
            Some("logistic") => Growth::Logistic,
            Some("flat") => Growth::Flat,
            _ => Growth::Linear,
        };
        spec.cap = fx["cap"].as_f64();
        spec.holidays = holidays;
        spec.holidays_mode = mode_of(&fx["seasonality_mode"]);
        spec.changepoint_prior_scale = fx["changepoint_prior_scale"]
            .as_f64()
            .expect("changepoint_prior_scale");
        if let Some(u) = fx.get("uncertainty") {
            spec.interval_width = u["interval_width"].as_f64().expect("interval_width");
            spec.uncertainty_samples = usize::try_from(
                u["uncertainty_samples"]
                    .as_u64()
                    .expect("uncertainty_samples"),
            )
            .expect("uncertainty_samples fits usize");
        }
        spec
    }

    fn ladder(stem: &'static str) -> Ladder {
        let file = FIXTURES[fixture_index(stem)].1;
        let fx = load_json(file);
        assert_eq!(
            fx["prophet_version"], "1.4.0",
            "{stem}: the bar is Python Prophet 1.4.0, not whatever regenerated the fixture"
        );
        let ds_days = days(&fx["history"]["ds"], "history.ds");
        let y = f64s(&fx["history"]["y"], "history.y");
        let forecast_days = days(&fx["forecast"]["ds"], "forecast.ds");
        let design = make_design(&ds_days, &y, &spec_of(&fx));
        let n_hist = ds_days.len();
        Ladder {
            fx,
            design,
            ds_days,
            forecast_days,
            n_hist,
        }
    }

    /// The D-09 MAP fit for one fixture, computed at most ONCE per test binary run.
    ///
    /// The fit rung and the spike-003 band rung both need it, and a Peyton-sized fit is the
    /// most expensive thing in this module (RESEARCH Pitfall 9). `OnceLock::get_or_init`
    /// blocks the second caller rather than duplicating the work.
    fn fitted(stem: &'static str) -> Arc<(Params, FitInfo)> {
        static CELLS: [OnceLock<Arc<(Params, FitInfo)>>; FIXTURES.len()] =
            [const { OnceLock::new() }; FIXTURES.len()];
        CELLS[fixture_index(stem)]
            .get_or_init(|| {
                let l = ladder(stem);
                let rounds = usize::try_from(constant_u64("prophet-parity-v1", "max_rounds"))
                    .expect("max_rounds fits usize");
                Arc::new(fit_prophet(&l.design, rounds))
            })
            .clone()
    }

    /// PYTHON's parameters through Rust `predict`, over the fixture's own forecast grid.
    ///
    /// Fit-independent by construction, which is exactly why the band rung uses it on the
    /// three spike-001 fixtures: their optimiser disagreement cannot leak into a band verdict.
    fn python_forecast(stem: &'static str) -> Arc<Forecast> {
        static CELLS: [OnceLock<Arc<Forecast>>; FIXTURES.len()] =
            [const { OnceLock::new() }; FIXTURES.len()];
        CELLS[fixture_index(stem)]
            .get_or_init(|| {
                let l = ladder(stem);
                Arc::new(
                    predict(
                        &l.design,
                        &l.python_params(),
                        &l.forecast_days,
                        SEED,
                        &crate::regressors::RegressorChannel::NONE,
                    )
                    .expect("the parity ladder passes an inert channel"),
                )
            })
            .clone()
    }

    fn component<'a>(f: &'a Forecast, name: &str) -> &'a [f64] {
        &f.components
            .iter()
            .find(|(n, _)| n.as_str() == name)
            .unwrap_or_else(|| panic!("predict must return a `{name}` component"))
            .1
    }

    // ---------------------------------------------------------------- rung 1 ----

    /// Rung 1: the Stan data block, rebuilt from the raw `(ds, y)` alone.
    ///
    /// The bar is EXACT (`data_prep_exact` is 0.0 in the contract): the spike measured
    /// `0.00e0` on all seven fixtures, so any non-zero difference is a defect, not noise.
    fn data_prep_exact(stem: &'static str) {
        let l = ladder(stem);
        let t = equation_tolerance("prophet-parity-v1", "data_prep_exact");
        let (d, fx) = (&l.design, &l.fx);

        let ours: Vec<(String, f64, usize)> = d
            .spec
            .seasonalities
            .iter()
            .map(|s| (s.name.clone(), s.period, s.order))
            .collect();
        let theirs: Vec<(String, f64, usize)> = fx["seasonalities"]
            .as_array()
            .expect("seasonalities")
            .iter()
            .map(|s| {
                (
                    s["name"].as_str().expect("name").to_string(),
                    s["period"].as_f64().expect("period"),
                    usize::try_from(s["fourier_order"].as_u64().expect("fourier_order"))
                        .expect("fits usize"),
                )
            })
            .collect();
        assert_eq!(
            ours, theirs,
            "{stem}: seasonality (name, period, fourier_order) list must match Prophet's"
        );

        assert!(
            (d.y_scale - fx["y_scale"].as_f64().expect("y_scale")).abs() <= t,
            "{stem}: y_scale {} vs {}",
            d.y_scale,
            fx["y_scale"]
        );
        assert!(
            (d.t_scale_days - fx["t_scale_days"].as_f64().expect("t_scale_days")).abs() <= t,
            "{stem}: t_scale_days {} vs {}",
            d.t_scale_days,
            fx["t_scale_days"]
        );

        let d_prior = max_abs_diff(
            &d.prior_scales,
            &f64s(&fx["prior_scales"], "prior_scales"),
            "prior_scales",
        );
        let d_t = max_abs_diff(&d.t, &f64s(&fx["history"]["t"], "history.t"), "t");
        let d_y = max_abs_diff(
            &d.y_scaled,
            &f64s(&fx["history"]["y_scaled"], "history.y_scaled"),
            "y_scaled",
        );
        let d_cp = max_abs_diff(
            &d.changepoints_t,
            &f64s(&fx["changepoints_t"], "changepoints_t"),
            "changepoints_t",
        );

        let flat = |v: &Value, what: &str| -> Vec<f64> {
            v.as_array()
                .unwrap_or_else(|| panic!("{what} must be an array of rows"))
                .iter()
                .flat_map(|r| f64s(r, what))
                .collect()
        };
        let d_x = max_abs_diff(
            &d.x[..3 * d.k],
            &flat(&fx["X_first3"], "X_first3"),
            "X first 3 rows",
        )
        .max(max_abs_diff(
            &d.x[(l.n_hist - 3) * d.k..],
            &flat(&fx["X_last3"], "X_last3"),
            "X last 3 rows",
        ));

        println!(
            "  {stem} rung 1: t {d_t:.2e}, y_scaled {d_y:.2e}, changepoints_t {d_cp:.2e} ({} cps), X {d_x:.2e} (K={})",
            d.changepoints_t.len(),
            d.k
        );
        assert!(
            d_t <= t && d_y <= t && d_cp <= t && d_x <= t && d_prior <= t,
            "{stem}: data-prep parity is EXACT in the contract; measured t {d_t:e}, y_scaled {d_y:e}, changepoints_t {d_cp:e}, X {d_x:e}, prior_scales {d_prior:e} against bar {t:e}"
        );

        if l.is_spike_003() {
            let ours_cols: Vec<String> = d.cols.iter().map(|c| c.name.clone()).collect();
            assert_eq!(
                ours_cols,
                strings(&fx["columns"], "columns"),
                "{stem}: design column names AND order must match Prophet's"
            );
            let d_sa = max_abs_diff(&d.s_a, &f64s(&fx["s_a"], "s_a"), "s_a");
            let d_sm = max_abs_diff(&d.s_m, &f64s(&fx["s_m"], "s_m"), "s_m");
            assert!(
                d_sa <= t && d_sm <= t,
                "{stem}: additive/multiplicative selectors differ: s_a {d_sa:e}, s_m {d_sm:e}"
            );
            if let Some(cap_scaled) = d.cap_scaled.as_ref() {
                let d_cap = max_abs_diff(
                    cap_scaled,
                    &f64s(&fx["history"]["cap_scaled"], "history.cap_scaled"),
                    "cap_scaled",
                );
                assert!(
                    d_cap <= t,
                    "{stem}: logistic cap_scaled differs by {d_cap:e}"
                );
            }
        } else {
            assert_eq!(
                d.k,
                fx["seasonality_columns"]
                    .as_array()
                    .expect("seasonality_columns")
                    .len(),
                "{stem}: design column count"
            );
        }
    }

    #[test]
    fn peyton_manning_data_prep_exact() {
        data_prep_exact("peyton_manning");
    }
    #[test]
    fn air_passengers_data_prep_exact() {
        data_prep_exact("air_passengers");
    }
    #[test]
    fn retail_sales_data_prep_exact() {
        data_prep_exact("retail_sales");
    }
    #[test]
    fn peyton_default_data_prep_exact() {
        data_prep_exact("peyton_default");
    }
    #[test]
    fn peyton_holidays_data_prep_exact() {
        data_prep_exact("peyton_holidays");
    }
    #[test]
    fn wp_log_r_logistic_data_prep_exact() {
        data_prep_exact("wp_log_r_logistic");
    }
    #[test]
    fn air_multiplicative_data_prep_exact() {
        data_prep_exact("air_multiplicative");
    }

    // ---------------------------------------------------------------- rung 2 ----

    /// Rung 2 (D-04): the Rust objective at PYTHON's MAP against Python's own `-lp`.
    ///
    /// Everything about the port — the priors, the exact L1 on `delta`, the `2σ²` term, the
    /// `n·ln σ` term, the Fourier column order and the changepoint grid — has to be right
    /// simultaneously for this number to land.
    ///
    /// ONLY THE THREE SPIKE-001 FIXTURES CAN CARRY THIS RUNG. The four spike-003 files publish
    /// no `log_posterior_at_map_unnormalized`, so there is no oracle to compare against there;
    /// writing a seventh "objective" test that compared Rust to Rust would be theatre. The
    /// contract states the asymmetry, and the spike-003 fixtures are held instead by
    /// `fitted_objective_slack` plus the whole Python-parameters-through-Rust chain.
    fn objective_at_python_map(stem: &'static str) {
        let l = ladder(stem);
        let t = equation_tolerance("prophet-parity-v1", "objective_at_python_map_abs");
        let lp = l.fx["log_posterior_at_map_unnormalized"]
            .as_f64()
            .expect("this rung binds only fixtures that publish Python's -lp");

        let py = l.python_params();
        assert_eq!(py.beta.len(), l.design.k, "{stem}: beta length vs design K");
        assert_eq!(
            py.delta.len(),
            l.design.changepoints_t.len(),
            "{stem}: delta length vs changepoints"
        );

        // `Model::new` scales by 1/T (the D-09 recipe); the fixture number is unscaled, so the
        // scale is undone before comparing. The fixture stores a POSITIVE log posterior while
        // the model computes a NEGATIVE one, so the residual is the SUM.
        let model = Model::new(&l.design);
        let f_rust = model.objective(&model.pack(&py)) / model.scale;
        let residual = (f_rust + lp).abs();
        println!(
            "  {stem} rung 2: f_rust {f_rust:.12}, python -lp {:.12}, residual {residual:e}",
            -lp
        );
        assert!(
            residual <= t,
            "{stem} rung 2: Rust f(theta_py) = {f_rust:.12}, Python -lp = {:.12}, abs diff {residual:e} over the contract bar {t:e}",
            -lp
        );
    }

    #[test]
    fn peyton_manning_objective_at_python_map() {
        objective_at_python_map("peyton_manning");
    }
    #[test]
    fn air_passengers_objective_at_python_map() {
        objective_at_python_map("air_passengers");
    }
    #[test]
    fn retail_sales_objective_at_python_map() {
        objective_at_python_map("retail_sales");
    }

    // ---------------------------------------------------------------- rung 3 ----

    /// Rung 3: PYTHON's parameters through Rust `predict`, plus components and the
    /// reconstruction identity. Fit-independent, so no optimiser difference can excuse it.
    fn predict_path_via_python_params(stem: &'static str) {
        let l = ladder(stem);
        let f = python_forecast(stem);

        // `make_future_dataframe(periods=365)` parity: the fixture's forecast grid is the
        // history followed by 365 DAILY rows, which is exactly what `dates::future_days` builds.
        assert_eq!(
            &l.forecast_days[..l.n_hist],
            &l.ds_days[..],
            "{stem}: the forecast grid must start with the history"
        );
        let last = *l.ds_days.last().expect("non-empty history");
        assert_eq!(
            &l.forecast_days[l.n_hist..],
            future_days(last, l.forecast_days.len() - l.n_hist, "D")
                .expect("D is supported")
                .as_slice(),
            "{stem}: the future grid must be daily from the last history day"
        );

        let py_yhat = f64s(&l.fx["forecast"]["yhat"], "forecast.yhat");
        let py_trend = f64s(&l.fx["forecast"]["trend"], "forecast.trend");
        let d_yhat = max_abs_diff(&f.yhat, &py_yhat, "yhat");
        let d_trend = max_abs_diff(&f.trend, &py_trend, "trend");

        let rel_bar =
            equation_tolerance("prophet-parity-v1", "predict_path_rel_yscale") * l.design.y_scale;
        println!(
            "  {stem} rung 3: yhat {d_yhat:.2e}, trend {d_trend:.2e} (y_scale {}, relative bar {rel_bar:.2e})",
            l.design.y_scale
        );
        assert!(
            d_yhat <= rel_bar && d_trend <= rel_bar,
            "{stem}: Python params through Rust predict differ by yhat {d_yhat:e} / trend {d_trend:e}, over the y_scale-relative bar {rel_bar:e}"
        );

        if PEYTON_BANDED.contains(&stem) {
            let abs_bar = equation_tolerance("prophet-parity-v1", "predict_path_abs_peyton");
            assert!(
                d_yhat <= abs_bar && d_trend <= abs_bar,
                "{stem}: D-04's ABSOLUTE predict-path bar {abs_bar:e} missed — yhat {d_yhat:e}, trend {d_trend:e}"
            );
        }

        // Named components: compare every component the fixture publishes AND predict returns.
        // The fixture also carries `<name>_lower` / `<name>_upper` (sampled) and, for logistic
        // growth, `cap` — none of which has a point-estimate twin, so they never match by name.
        let comp_bar = equation_tolerance("prophet-parity-v1", "components_via_python_params_abs");
        let py_comps = l.fx["forecast"]["components"]
            .as_object()
            .expect("forecast.components");
        let mut checked: Vec<String> = Vec::new();
        for (name, values) in &f.components {
            if let Some(py) = py_comps.get(name) {
                let d = max_abs_diff(values, &f64s(py, name), name);
                assert!(
                    d <= comp_bar,
                    "{stem}: component `{name}` differs by {d:e}, over the contract bar {comp_bar:e}"
                );
                checked.push(format!("{name} {d:.1e}"));
            }
        }
        assert!(
            !checked.is_empty(),
            "{stem}: no named component was compared — the fixture's component names drifted"
        );

        // The decomposition must reconstruct the number the tool returns.
        let add = component(&f, "additive_terms");
        let mul = component(&f, "multiplicative_terms");
        let recon: Vec<f64> = (0..f.yhat.len())
            .map(|i| f.trend[i] * (1.0 + mul[i]) + add[i])
            .collect();
        let d_recon = max_abs_diff(&recon, &f.yhat, "reconstruction");
        println!(
            "  {stem} rung 3 components: {} | rebuild {d_recon:.1e}",
            checked.join(", ")
        );
        assert!(
            d_recon <= equation_tolerance("prophet-parity-v1", "components_rebuild_yhat_abs"),
            "{stem}: trend*(1+multiplicative_terms)+additive_terms misses yhat by {d_recon:e}"
        );
    }

    #[test]
    fn peyton_manning_predict_path_via_python_params() {
        predict_path_via_python_params("peyton_manning");
    }
    #[test]
    fn air_passengers_predict_path_via_python_params() {
        predict_path_via_python_params("air_passengers");
    }
    #[test]
    fn retail_sales_predict_path_via_python_params() {
        predict_path_via_python_params("retail_sales");
    }
    #[test]
    fn peyton_default_predict_path_via_python_params() {
        predict_path_via_python_params("peyton_default");
    }
    #[test]
    fn peyton_holidays_predict_path_via_python_params() {
        predict_path_via_python_params("peyton_holidays");
    }
    #[test]
    fn wp_log_r_logistic_predict_path_via_python_params() {
        predict_path_via_python_params("wp_log_r_logistic");
    }
    #[test]
    fn air_multiplicative_predict_path_via_python_params() {
        predict_path_via_python_params("air_multiplicative");
    }

    // ---------------------------------------------------------------- rung 4 ----

    /// Rung 4: what the Rust MAP fit itself reaches, and the D-09 diagnostics it must report.
    ///
    /// The bar is ONE-SIDED objective slack, never parameter equality and never
    /// daily-resolution `yhat` on monthly data (`prophet-fit-and-predict.md`): the yearly
    /// Fourier block is near-unidentified on air/retail, so many parameter vectors sit within a
    /// fraction of an objective unit of Python's MAP and draw visibly different curves.
    fn fit_objective_and_forecast(stem: &'static str) {
        let l = ladder(stem);
        let max_rounds = usize::try_from(constant_u64("prophet-parity-v1", "max_rounds"))
            .expect("max_rounds fits usize");
        let handle = fitted(stem);
        let (p, info) = (&handle.0, &handle.1);

        let f_python = l.f_python();
        let slack = equation_tolerance("prophet-parity-v1", "fitted_objective_slack");
        println!(
            "  {stem} rung 4: f_rust {:.4} vs f_python {f_python:.4} (delta {:+.4}); {} rounds, {} iters, {} evals, status {}",
            info.objective,
            info.objective - f_python,
            info.rounds,
            info.iterations,
            info.evals,
            info.status
        );
        assert!(
            info.objective <= f_python + slack,
            "{stem}: the Rust fit landed at {} against Python's {f_python}, i.e. {:+} — over the contract slack {slack}",
            info.objective,
            info.objective - f_python
        );

        // D-09 diagnostics, observable on every fixture.
        assert!(
            info.rounds <= max_rounds,
            "{stem}: {} restart rounds exceeds the contract's max_rounds {max_rounds}",
            info.rounds
        );
        assert!(
            !info.budget_hit,
            "{stem}: the cooperative fit budget was crossed at a round boundary — a fixture-sized fit must never reach it"
        );
        assert!(
            info.status == "Stalled" || info.status == "Converged",
            "{stem}: L-BFGS terminal status {:?} is not one the D-09 recipe accepts",
            info.status
        );

        let f = predict(
            &l.design,
            p,
            &l.forecast_days,
            SEED,
            &crate::regressors::RegressorChannel::NONE,
        )
        .expect("the parity ladder passes an inert channel");
        let py_yhat = f64s(&l.fx["forecast"]["yhat"], "forecast.yhat");
        let future = max_abs_diff(&f.yhat[l.n_hist..], &py_yhat[l.n_hist..], "future yhat");
        let history = max_abs_diff(&f.yhat[..l.n_hist], &py_yhat[..l.n_hist], "history yhat");
        println!("  {stem} rung 4 forecast: history max|d yhat| {history:.4}, future {future:.4}");

        if PEYTON_BANDED.contains(&stem) {
            let band = equation_tolerance("prophet-parity-v1", "future_yhat_band_peyton_abs");
            assert!(
                future <= band,
                "{stem}: future max|d yhat| {future} is outside Prophet's OWN Newton-vs-L-BFGS band {band}"
            );
        } else {
            // RECORDED, NOT BARRED. The contract says why: no committed control band exists for
            // these fixtures, so an epsilon here would bar optimiser luck rather than parity.
            assert!(
                future.is_finite(),
                "{stem}: future max|d yhat| must at least be finite (measured {future})"
            );
        }
    }

    #[test]
    fn peyton_manning_fit_objective_and_forecast() {
        fit_objective_and_forecast("peyton_manning");
    }
    #[test]
    fn air_passengers_fit_objective_and_forecast() {
        fit_objective_and_forecast("air_passengers");
    }
    #[test]
    fn retail_sales_fit_objective_and_forecast() {
        fit_objective_and_forecast("retail_sales");
    }
    #[test]
    fn peyton_default_fit_objective_and_forecast() {
        fit_objective_and_forecast("peyton_default");
    }
    #[test]
    fn peyton_holidays_fit_objective_and_forecast() {
        fit_objective_and_forecast("peyton_holidays");
    }
    #[test]
    fn wp_log_r_logistic_fit_objective_and_forecast() {
        fit_objective_and_forecast("wp_log_r_logistic");
    }
    #[test]
    fn air_multiplicative_fit_objective_and_forecast() {
        fit_objective_and_forecast("air_multiplicative");
    }

    // ---------------------------------------------------------------- rung 5 ----

    /// Rung 5: the 80 % interval widths, relative to Python's.
    ///
    /// TWO REFERENCE SHAPES, because the two fixture families publish different things.
    /// The spike-003 files carry precomputed `uncertainty.*_mean_width` values, and the Rust
    /// side there is the Rust FIT (the spike-003 probe). The spike-001 files carry no
    /// `uncertainty` block at all, so their reference is DERIVED from the fixture's own
    /// `forecast.yhat_upper - forecast.yhat_lower` — and the Rust side is PYTHON's parameters
    /// through Rust `predict`, the same fit-independent call rung 3 makes, so air/retail's
    /// unbarred optimiser disagreement cannot leak into a band verdict.
    fn band_widths_within_contract(stem: &'static str) {
        let l = ladder(stem);
        let rel = equation_tolerance("prophet-parity-v1", "band_width_rel");
        let n = l.n_hist;

        if l.is_spike_003() {
            let handle = fitted(stem);
            let f = predict(
                &l.design,
                &handle.0,
                &l.forecast_days,
                SEED,
                &crate::regressors::RegressorChannel::NONE,
            )
            .expect("the parity ladder passes an inert channel");
            let u = &l.fx["uncertainty"];
            let py = |key: &str| {
                u[key]
                    .as_f64()
                    .unwrap_or_else(|| panic!("uncertainty.{key}"))
            };
            println!("  {stem} rung 5 (spike-003 reference, Rust fit params):");
            within_rel(
                mean_width(&f.yhat_lower[..n], &f.yhat_upper[..n]),
                py("hist_band_mean_width"),
                rel,
                &format!("{stem} history band width"),
            );
            within_rel(
                mean_width(&f.yhat_lower[n..], &f.yhat_upper[n..]),
                py("future_band_mean_width"),
                rel,
                &format!("{stem} future band width"),
            );
            let last30 = f.yhat.len() - 30;
            within_rel(
                mean_width(&f.yhat_lower[last30..], &f.yhat_upper[last30..]),
                py("future_band_last30_mean_width"),
                equation_tolerance("prophet-parity-v1", "band_width_last30_rel"),
                &format!("{stem} last-30 band width"),
            );
            within_rel(
                mean_width(&f.trend_lower[n..], &f.trend_upper[n..]),
                py("future_trend_band_mean_width"),
                equation_tolerance("prophet-parity-v1", "trend_band_width_rel"),
                &format!("{stem} future trend band width"),
            );
        } else {
            let f = python_forecast(stem);
            let py_lo = f64s(&l.fx["forecast"]["yhat_lower"], "forecast.yhat_lower");
            let py_hi = f64s(&l.fx["forecast"]["yhat_upper"], "forecast.yhat_upper");
            println!("  {stem} rung 5 (reference DERIVED from the fixture's own band, Python params through Rust predict):");
            within_rel(
                mean_width(&f.yhat_lower[..n], &f.yhat_upper[..n]),
                mean_width(&py_lo[..n], &py_hi[..n]),
                rel,
                &format!("{stem} history band width"),
            );
            within_rel(
                mean_width(&f.yhat_lower[n..], &f.yhat_upper[n..]),
                mean_width(&py_lo[n..], &py_hi[n..]),
                rel,
                &format!("{stem} future band width"),
            );
        }
    }

    #[test]
    fn peyton_manning_band_widths_within_contract() {
        band_widths_within_contract("peyton_manning");
    }
    #[test]
    fn air_passengers_band_widths_within_contract() {
        band_widths_within_contract("air_passengers");
    }
    #[test]
    fn retail_sales_band_widths_within_contract() {
        band_widths_within_contract("retail_sales");
    }
    #[test]
    fn peyton_default_band_widths_within_contract() {
        band_widths_within_contract("peyton_default");
    }
    #[test]
    fn peyton_holidays_band_widths_within_contract() {
        band_widths_within_contract("peyton_holidays");
    }
    #[test]
    fn wp_log_r_logistic_band_widths_within_contract() {
        band_widths_within_contract("wp_log_r_logistic");
    }
    #[test]
    fn air_multiplicative_band_widths_within_contract() {
        band_widths_within_contract("air_multiplicative");
    }

    // ------------------------------------------------- the bounded date grid ----

    /// The runnable evidence behind KANI-PROPHET-001 (declared, not executed).
    ///
    /// A grid over the three supported frequencies, 20 start days chosen to sit on leap days,
    /// month ends and year ends, and horizons up to the contract's bound: the future grid is
    /// strictly increasing and starts strictly after the last history day, and every `MS` date
    /// is the first of a month.
    #[test]
    fn future_days_strictly_increasing_bounded() {
        let starts = [
            (1968, 2, 28),
            (1968, 2, 29),
            (1968, 3, 1),
            (1970, 1, 1),
            (1999, 12, 31),
            (2000, 1, 1),
            (2000, 2, 29),
            (2001, 2, 28),
            (2004, 2, 29),
            (2008, 1, 31),
            (2012, 12, 1),
            (2015, 1, 1),
            (2016, 2, 29),
            (2019, 4, 30),
            (2020, 2, 29),
            (2021, 3, 31),
            (2023, 5, 31),
            (2024, 2, 29),
            (2024, 12, 31),
            (2100, 2, 28),
        ];
        assert_eq!(starts.len(), 20, "the grid is 20 start days wide");
        let mut cases = 0usize;
        for (y, m, d) in starts {
            let last = days_from_civil(y, m, d);
            for freq in ["D", "W", "MS"] {
                for horizon in [1_usize, 28, 365, 3650] {
                    let grid = future_days(last, horizon, freq)
                        .unwrap_or_else(|e| panic!("{freq} is supported: {e:?}"));
                    assert_eq!(grid.len(), horizon, "{freq}/{horizon}: length");
                    assert!(
                        grid[0] > last,
                        "{freq}/{horizon} from {y}-{m}-{d}: the grid must start after the history"
                    );
                    assert!(
                        grid.windows(2).all(|w| w[0] < w[1]),
                        "{freq}/{horizon} from {y}-{m}-{d}: the grid must be strictly increasing"
                    );
                    if freq == "MS" {
                        for &day in &grid {
                            let (_, _, dom) = civil_from_days(day);
                            assert_eq!(
                                dom, 1,
                                "MS/{horizon} from {y}-{m}-{d}: every date is the 1st of a month"
                            );
                        }
                    }
                    cases += 1;
                }
            }
        }
        assert_eq!(cases, 20 * 3 * 4, "the whole bounded grid was enumerated");
    }
}

#[cfg(test)]
mod predict_spans {
    //! The falsification probe for the span-restricted per-component roll-up.
    //!
    //! The restriction is only safe because the span is a SUPERSET of the columns the
    //! selector would have matched, so the accumulated terms and their order are unchanged.
    //! That is an argument; this module is the measurement. Without it, a span computed as
    //! `first..first + 1` (the "a component is one column" shortcut that is true for a
    //! regressor and FALSE for a seasonality) would produce a green parity run on any
    //! fixture whose seasonalities happen to be order 1.
    use super::{auto_seasonalities, make_design, predict, Column, Holiday, Mode, Params, Spec};
    use crate::dates::days_from_civil;
    use crate::regressors::{splice, RegressorChannel, Standardized};

    /// A design carrying all three column families: seasonalities (multi-column
    /// components), holidays (multi-column components, name-sorted) and regressors
    /// (single-column components).
    fn rich_design() -> (
        super::Design,
        Params,
        Vec<i64>,
        Vec<Vec<f64>>,
        Vec<Standardized>,
    ) {
        let t0 = days_from_civil(2018, 1, 1);
        let n = 800;
        let ds: Vec<i64> = (0..n as i64).map(|i| t0 + i).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64;
                50.0 + 0.03 * t + (2.0 * std::f64::consts::PI * t / 7.0).sin() * 3.0
            })
            .collect();
        let mut spec = Spec::default_linear(auto_seasonalities(&ds, 10.0, Mode::Additive));
        spec.holidays = vec![
            Holiday {
                name: "alpha".into(),
                days: vec![t0 + 30, t0 + 200],
                lower_window: -2,
                upper_window: 2,
                prior_scale: 10.0,
            },
            // Deliberately a name that is a PREFIX-neighbour of the first, because the
            // holiday columns are sorted by their generated name and a span is only a
            // superset if the sort cannot interleave two holidays' columns.
            Holiday {
                name: "alphabet".into(),
                days: vec![t0 + 90],
                lower_window: -1,
                upper_window: 1,
                prior_scale: 10.0,
            },
        ];
        spec.holidays_mode = Mode::Additive;
        let mut d = make_design(&ds, &y, &spec);
        let regs = vec![
            Standardized {
                name: "promo".into(),
                mu: 0.0,
                std: 1.0,
                mode: Mode::Additive,
                prior_scale: 10.0,
            },
            Standardized {
                name: "price".into(),
                mu: 2.0,
                std: 0.5,
                mode: Mode::Multiplicative,
                prior_scale: 10.0,
            },
        ];
        let hist: Vec<Vec<f64>> = vec![
            (0..n).map(|i| f64::from((i % 2) as u32)).collect(),
            (0..n).map(|i| 2.0 + (i as f64 * 0.01).sin()).collect(),
        ];
        splice(&mut d, &regs, &hist);
        // Deterministic, NON-ZERO and sign-varying betas: a zero beta makes every roll-up
        // zero and the comparison vacuous, and a uniformly positive one hides a -0.0.
        let beta: Vec<f64> = (0..d.k)
            .map(|c| if c % 3 == 0 { -0.017 } else { 0.023 } * (c + 1) as f64)
            .collect();
        let p = Params {
            k: 0.4,
            m: 0.1,
            delta: vec![0.0; d.changepoints_t.len()],
            beta,
            sigma_obs: 0.1,
        };
        let fut: Vec<i64> = (0..40).map(|i| ds[n - 1] + 1 + i).collect();
        let fut_vals: Vec<Vec<f64>> = vec![
            (0..40).map(|i| f64::from((i % 2) as u32)).collect(),
            (0..40).map(|i| 2.0 + (i as f64 * 0.01).cos()).collect(),
        ];
        (d, p, fut, fut_vals, regs)
    }

    /// Every per-component series `predict` emits is BIT-IDENTICAL to a full-scan
    /// reference computed here from the same `x` and `beta`.
    #[test]
    fn the_span_restricted_roll_up_matches_a_full_scan() {
        let (d, p, fut, fut_vals, regs) = rich_design();
        let channel = RegressorChannel {
            specs: &regs,
            values: &fut_vals,
        };
        let fc = predict(&d, &p, &fut, 42, &channel).expect("the channel is well formed");

        // Rebuild `x` over the predicted rows exactly as `predict` does, then roll each
        // component up with an UNRESTRICTED scan — the code this change replaced.
        let n = fut.len();
        let base_k = d.k - regs.len();
        let base_cols = &d.cols[..base_k];
        let hol_sets = super::holiday_day_sets(&d.spec);
        let mut x = Vec::with_capacity(n * d.k);
        for (i, &day) in fut.iter().enumerate() {
            super::feature_row(day, &d.spec, base_cols, &hol_sets, &mut x);
            for (j, reg) in regs.iter().enumerate() {
                x.push((fut_vals[j][i] - reg.mu) / reg.std);
            }
        }
        let full_scan = |nm: &str, additive: bool| -> Vec<f64> {
            (0..n)
                .map(|i| {
                    let mut v = 0.0;
                    for c in 0..d.k {
                        if d.cols[c].component == nm {
                            v += x[i * d.k + c] * p.beta[c];
                        }
                    }
                    if additive {
                        v * d.y_scale
                    } else {
                        v
                    }
                })
                .collect()
        };

        let mut checked = 0usize;
        let mut names: Vec<&str> = Vec::new();
        for c in &d.cols {
            if !names.contains(&c.component.as_str()) {
                names.push(&c.component);
            }
        }
        assert!(
            names.len() >= 5,
            "the probe must carry several distinct components (seasonalities, two holidays              and two regressors), found {names:?}"
        );
        for nm in &names {
            let mode = d
                .cols
                .iter()
                .find(|c: &&Column| c.component == *nm)
                .expect("component exists")
                .mode;
            let want = full_scan(nm, mode == Mode::Additive);
            let got = &fc
                .components
                .iter()
                .find(|(k, _)| k == nm)
                .unwrap_or_else(|| panic!("component {nm} must be emitted"))
                .1;
            assert_eq!(want.len(), got.len());
            for (i, (a, b)) in want.iter().zip(got.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "component {nm} row {i}: the span-restricted roll-up must be BIT-identical                      to a full scan, got {b} want {a}"
                );
            }
            // Non-vacuity: a component that is identically zero proves nothing.
            if want.iter().any(|v| *v != 0.0) {
                checked += 1;
            }
        }
        assert!(
            checked >= 4,
            "at least four components must carry a NON-ZERO series, or the bit comparison              above is comparing zeros; only {checked} did"
        );
    }

    /// The span really is a SUPERSET: for every distinct component, no column carrying it
    /// lies outside `first..=last`.
    ///
    /// This is the property the bitwise argument rests on, asserted directly rather than
    /// inferred from the holiday sort's behaviour.
    #[test]
    fn every_component_span_contains_every_column_of_that_component() {
        let (d, ..) = rich_design();
        let mut names: Vec<String> = Vec::new();
        let mut spans: Vec<(usize, usize)> = Vec::new();
        for (ci, c) in d.cols.iter().enumerate() {
            if let Some(pos) = names.iter().position(|n| n == &c.component) {
                spans[pos].1 = ci;
            } else {
                names.push(c.component.clone());
                spans.push((ci, ci));
            }
        }
        for (nm, &(first, last)) in names.iter().zip(spans.iter()) {
            for (ci, c) in d.cols.iter().enumerate() {
                if &c.component == nm {
                    assert!(
                        ci >= first && ci <= last,
                        "column {ci} carries component {nm} but lies outside its span                          {first}..={last} — the restriction would DROP it"
                    );
                }
            }
        }
    }
}

// ------------------------------------------------------- design-cost bench ----
/// The release-profile wall-clock harness behind `just forecast-holiday-bench`.
///
/// It ATTRIBUTES a holiday-carrying request's wall rather than asserting a bar: the body
/// emits exactly one machine-parsable measurement line (the token is written in exactly
/// one place below, so the recipe's parse is unambiguous) and asserts only that the call
/// succeeded and returned one row per horizon step. REVIEW-06-04 removed a
/// wall-clock ratio assertion from `pool_equality` for the reason that binds here too — a
/// wall inside libtest moves with CPU throttling independently of what is being measured,
/// so the BAR lives in the host-gated `just` recipe and the TEST only measures.
///
/// SINCE PLAN 06-16 IT BUILDS NOTHING OF ITS OWN (WR-04). The series, the holiday splitter,
/// the accept-and-time step and the `profile=` token all come from [`crate::sc1_wall`], so
/// this is a SINGLE-COMPOSITION ENTRY POINT onto the sweep's builder rather than a third
/// harness with a third geometry. It is kept rather than deleted because
/// `just forecast-holiday-bench` is cited by `06-EVIDENCE.md` and by
/// `contracts/forecast-tool-boundary-v1.yaml`, and its caller-facing behaviour is unchanged.
/// The COVERAGE claim now belongs to `just forecast-sc1-sweep`, which sweeps the surface
/// this one composition cannot.
#[cfg(test)]
mod design_cost {
    use crate::dates::days_from_civil;
    // ONE splitter, ONE env reader, ONE series builder, ONE accept-and-time step (WR-04):
    // all of these used to be defined here, where only this bench could reach them. They
    // now live in `crate::sc1_wall` beside the composition builder the sweep and every
    // single-composition entry point share.
    use crate::sc1_wall::{
        env_usize, holidays_for, profile_token, tight_daily_series, time_accepted,
    };
    use crate::types::ForecastArgs;

    #[test]
    #[ignore = "release-profile wall-clock measurement; run via just forecast-holiday-bench"]
    fn holiday_design_wall() {
        let points = env_usize("HOLIDAY_BENCH_POINTS", 3000);
        let columns = env_usize("HOLIDAY_BENCH_COLUMNS", 181);
        let dates = env_usize("HOLIDAY_BENCH_DATES", 84);
        let horizon = env_usize("HOLIDAY_BENCH_HORIZON", 365);

        let (ds, y, t0) = tight_daily_series(points);
        debug_assert_eq!(
            t0,
            days_from_civil(2015, 1, 1),
            "the shared series builder still starts where this bench's recorded numbers were \
             measured; a moved epoch changes the holiday dates and therefore the design"
        );
        let holidays = holidays_for(columns, dates, t0, points);
        let n_holidays = holidays.len();
        let dates_total: usize = holidays.iter().map(|h| h.dates.len()).sum();
        let args = ForecastArgs {
            ds,
            y,
            horizon,
            holidays: Some(holidays),
            ..ForecastArgs::default()
        };

        let (r, total) = time_accepted(
            &args,
            &format!("holiday_design_wall points={points} columns={columns} horizon={horizon}"),
        );
        println!(
            "HOLIDAY DESIGN WALL: points={points} columns={columns} dates={dates_total} \
             holidays={n_holidays} horizon={horizon} cells={} triple={} total_s={total:.3} \
             fit_s={:.3} predict_s={:.3} other_s={:.3} arch={} profile={}",
            (points + horizon) * columns,
            points * columns * dates_total,
            r.fit_seconds,
            r.predict_seconds,
            total - r.fit_seconds - r.predict_seconds,
            std::env::consts::ARCH,
            profile_token()
        );
    }
}

#[cfg(test)]
mod changepoints {
    //! `changepoint_count` is the SINGLE implementation of the effective changepoint count,
    //! and this module is what makes that a measurement rather than a comment.
    //!
    //! The door refuses a logistic request whose Poisson mean
    //! `lambda = changepoints_t.len() * (t_max - 1)` exceeds
    //! [`crate::types::MAX_LOGISTIC_CHANGEPOINT_LAMBDA`], and it computes that length with
    //! [`super::changepoint_count`]. If the door and [`super::make_design`] could disagree
    //! about the count, the bound would be evadable by picking a point count where they
    //! differ — so the tie is swept, not assumed.
    use super::{auto_seasonalities, changepoint_count, make_design, Mode, Spec};
    use crate::dates::days_from_civil;

    /// A strictly-ascending daily series of `n` points with a non-degenerate `y`.
    fn series(n: usize) -> (Vec<i64>, Vec<f64>) {
        let t0 = days_from_civil(2020, 1, 1);
        let ds: Vec<i64> = (0..n as i64).map(|i| t0 + i).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64;
                10.0 + 0.01 * t + (2.0 * std::f64::consts::PI * t / 7.0).sin()
            })
            .collect();
        (ds, y)
    }

    /// The count the door reads IS the length the design produces, across the whole
    /// reachable range of point counts and across both branches of the `n_cp` decision.
    ///
    /// `n = 2` and `n = 3` are in the sweep because they are the ONLY way to reach
    /// `make_design`'s `vec![0.0]` branch with the default 25 changepoints, and that branch
    /// is exactly why `changepoint_count` returns the LENGTH (1) rather than the raw count
    /// (0). A sweep that started at 10 would never exercise the distinction it exists for.
    /// The `n_changepoints = 0` spec reaches the same branch from the other direction, at
    /// every point count.
    #[test]
    fn changepoint_count_equals_the_design_it_describes() {
        for n in [2usize, 3, 10, 12, 20, 32, 33, 34, 100, 3000] {
            let (ds, y) = series(n);
            for n_changepoints in [0usize, 1, 25] {
                let mut spec = Spec::default_linear(auto_seasonalities(&ds, 10.0, Mode::Additive));
                spec.n_changepoints = n_changepoints;
                if spec.seasonalities.is_empty() {
                    // `make_design` needs K >= 1, the same way the door does.
                    spec.seasonalities.push(super::Seasonality {
                        name: "weekly".into(),
                        period: 7.0,
                        order: 1,
                        prior_scale: 1e-3,
                        mode: Mode::Additive,
                    });
                }
                let design = make_design(&ds, &y, &spec);
                assert_eq!(
                    design.changepoints_t.len(),
                    changepoint_count(n, &spec),
                    "n={n} n_changepoints={n_changepoints}: the door's count and the design \
                     it describes must be the same number — a disagreement here makes \
                     MAX_LOGISTIC_CHANGEPOINT_LAMBDA evadable"
                );
            }
        }
    }

    /// The count is never zero, because the design it describes is never empty.
    ///
    /// The lambda the door bounds is `count * (t_max - 1)`; a count of 0 would report
    /// lambda 0 for a request whose design still carries one changepoint and still runs the
    /// simulation, which is the exact shape of a bound that reads a different number from
    /// the one that is spent.
    #[test]
    fn the_effective_changepoint_count_is_never_zero() {
        let (ds, _) = series(2);
        let mut spec = Spec::default_linear(auto_seasonalities(&ds, 10.0, Mode::Additive));
        spec.n_changepoints = 0;
        for n in [2usize, 3, 10, 33, 3000] {
            assert!(
                changepoint_count(n, &spec) >= 1,
                "n={n}: the effective count must be at least 1"
            );
        }
    }
}

#[cfg(test)]
mod feature_row_totality {
    //! IN-02, NARROWED: `feature_row`'s holiday lookup must be TOTAL.
    //!
    //! The index `hi` comes from `Column.holiday`, i.e. from the `cols` argument, and it is
    //! used to subscript `hol_sets`, a DIFFERENT argument. Nothing in the type system ties
    //! the two together, and every field of [`super::Design`] is `pub`, so
    //! `d.spec.holidays.clear()` followed by `predict(&d, ...)` reaches the lookup with a
    //! slice that does not line up — an out-of-bounds panic raised inside library code.
    //!
    //! The fix is to make the LOOKUP total, not to make the signature old. The `hol_sets`
    //! parameter is plan 06-11's measured design-build improvement (the caller hoists the
    //! membership-set construction out of the row loop); reverting it would re-open that
    //! cost, and `06-REVIEW.md` records the same indexing hazard as PRE-EXISTING in the
    //! `spec.holidays[hi]` form the parameter replaced. So this is not a regression being
    //! backed out — it is a long-standing sharp edge being filed down.
    //!
    //! What the change buys and what it does NOT: a mismatched slice was a panic and is now
    //! a zero column. NEITHER is correct output for a caller who built the slice wrong. What
    //! it buys is that a library does not abort the caller's process over it.
    use super::{columns, feature_row, holiday_day_sets, Growth, Holiday, Mode, Seasonality, Spec};
    use crate::dates::days_from_civil;

    /// A spec with one seasonality (so `columns` has a non-holiday prefix to get right too)
    /// and two holidays, each contributing one column at offset 0.
    fn spec_with_holidays() -> Spec {
        Spec {
            growth: Growth::Linear,
            cap: None,
            seasonalities: vec![Seasonality {
                name: "weekly".into(),
                period: 7.0,
                order: 1,
                prior_scale: 1e-3,
                mode: Mode::Additive,
            }],
            holidays: vec![
                Holiday {
                    name: "alpha".into(),
                    days: vec![days_from_civil(2020, 1, 5)],
                    lower_window: 0,
                    upper_window: 0,
                    prior_scale: 10.0,
                },
                Holiday {
                    name: "beta".into(),
                    days: vec![days_from_civil(2020, 1, 9)],
                    lower_window: 0,
                    upper_window: 0,
                    prior_scale: 10.0,
                },
            ],
            holidays_mode: Mode::Additive,
            n_changepoints: 25,
            changepoint_range: 0.8,
            changepoint_prior_scale: 0.05,
            interval_width: 0.8,
            uncertainty_samples: 1000,
        }
    }

    /// A `hol_sets` that does not line up with `cols` RETURNS, with the holiday entries
    /// zero, instead of panicking inside a library.
    ///
    /// The empty slice is the extreme of the reachable case (`d.spec.holidays.clear()`), and
    /// the one-element slice is the off-by-one that a partially-rebuilt design produces —
    /// the second is the more likely accident and the one a `get` must also survive.
    #[test]
    fn a_short_hol_sets_is_a_miss_and_never_an_out_of_bounds_panic() {
        let spec = spec_with_holidays();
        let cols = columns(&spec);
        let n_holiday_cols = cols.iter().filter(|c| c.holiday.is_some()).count();
        assert_eq!(
            n_holiday_cols, 2,
            "the fixture must actually produce holiday columns, or this test proves nothing"
        );
        let day = days_from_civil(2020, 1, 5);
        for short in [0usize, 1] {
            let full = holiday_day_sets(&spec);
            let truncated = &full[..short];
            let mut out = Vec::new();
            feature_row(day, &spec, &cols, truncated, &mut out);
            assert_eq!(
                out.len(),
                2 + n_holiday_cols,
                "short={short}: the row must still have one entry per column"
            );
            // Every column whose set was truncated away must read as a MISS.
            for (i, v) in out.iter().skip(2).enumerate().skip(short) {
                assert!(
                    (*v - 0.0).abs() < f64::EPSILON,
                    "short={short}: holiday column {i} has no set, so it must be 0.0, got {v}"
                );
            }
        }
    }

    /// The shipped path is UNCHANGED: with a correctly-built `hol_sets`, the row is exactly
    /// what the indexing form produced.
    ///
    /// The parity ladder already covers this at the fixture level; this pins it directly at
    /// the row level so a `get`-returned miss where the index returned a HIT would be caught
    /// here rather than only as a moved parity rung.
    #[test]
    fn a_correctly_built_hol_sets_still_hits_exactly_the_same_days() {
        let spec = spec_with_holidays();
        let cols = columns(&spec);
        let hol_sets = holiday_day_sets(&spec);
        // alpha is 2020-01-05, beta is 2020-01-09; both windows are [0, 0].
        for (day, want) in [
            (days_from_civil(2020, 1, 5), [1.0, 0.0]),
            (days_from_civil(2020, 1, 9), [0.0, 1.0]),
            (days_from_civil(2020, 1, 7), [0.0, 0.0]),
        ] {
            let mut out = Vec::new();
            feature_row(day, &spec, &cols, &hol_sets, &mut out);
            let got: Vec<f64> = out[2..].to_vec();
            assert_eq!(got.len(), 2, "two holiday columns");
            // `columns` sorts holiday columns by name, and "alpha_delim_+0" precedes
            // "beta_delim_+0", so index 0 is alpha and index 1 is beta.
            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                assert!(
                    (g - w).abs() < f64::EPSILON,
                    "day={day} column {i}: want {w}, got {g}"
                );
            }
        }
    }
}
