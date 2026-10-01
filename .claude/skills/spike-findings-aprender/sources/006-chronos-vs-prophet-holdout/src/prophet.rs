//! Prophet, second cut: linear / logistic / flat growth, additive AND multiplicative
//! terms, holiday indicator columns with windows, analytic gradients for all of it,
//! component decomposition, and the vectorised uncertainty simulation of
//! `forecaster.py` (`_make_trend_shift_matrix` → `_sample_uncertainty` →
//! `sample_model_vectorized` → percentiles).

// ---------------------------------------------------------------- dates ----
pub fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m as i64 + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}
pub fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}
pub fn parse_ymd(s: &str) -> i64 {
    let mut it = s.split('-');
    let y: i64 = it.next().expect("year").parse().expect("year int");
    let m: u32 = it.next().expect("month").parse().expect("month int");
    let d: u32 = it.next().expect("day").parse().expect("day int");
    days_from_civil(y, m, d)
}
pub fn format_ymd(days: i64) -> String { let (y, m, d) = civil_from_days(days); format!("{y:04}-{m:02}-{d:02}") }

// ----------------------------------------------------------------- spec ----
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Growth { Linear, Logistic, Flat }
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode { Additive, Multiplicative }

#[derive(Clone, Debug)]
pub struct Seasonality { pub name: String, pub period: f64, pub order: usize, pub prior_scale: f64, pub mode: Mode }
#[derive(Clone, Debug)]
pub struct Holiday { pub name: String, pub days: Vec<i64>, pub lower_window: i64, pub upper_window: i64, pub prior_scale: f64 }

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
        Spec { growth: Growth::Linear, cap: None, seasonalities, holidays: vec![], holidays_mode: Mode::Additive, n_changepoints: 25, changepoint_range: 0.8, changepoint_prior_scale: 0.05, interval_width: 0.8, uncertainty_samples: 1000 }
    }
}

pub fn auto_seasonalities(ds_days: &[i64], prior_scale: f64, mode: Mode) -> Vec<Seasonality> {
    let span = (ds_days[ds_days.len() - 1] - ds_days[0]) as f64;
    let min_dt = ds_days.windows(2).map(|w| (w[1] - w[0]) as f64).filter(|d| *d > 0.0).fold(f64::INFINITY, f64::min);
    let mut out = Vec::new();
    if span >= 730.0 { out.push(Seasonality { name: "yearly".into(), period: 365.25, order: 10, prior_scale, mode }); }
    if span >= 14.0 && min_dt < 7.0 { out.push(Seasonality { name: "weekly".into(), period: 7.0, order: 3, prior_scale, mode }); }
    if span >= 2.0 && min_dt < 1.0 { out.push(Seasonality { name: "daily".into(), period: 1.0, order: 4, prior_scale, mode }); }
    out
}

// -------------------------------------------------------------- columns ----
/// One regressor column of X: which component it belongs to and its mode.
#[derive(Clone, Debug)]
pub struct Column { pub name: String, pub component: String, pub mode: Mode, pub prior_scale: f64, pub holiday: Option<(usize, i64)> }

/// Prophet's column order: seasonalities in insertion order (sin, cos interleaved),
/// then holiday columns sorted by their `{holiday}_delim_{±offset}` name.
pub fn columns(spec: &Spec) -> Vec<Column> {
    let mut cols = Vec::new();
    for s in &spec.seasonalities {
        for i in 0..s.order {
            cols.push(Column { name: format!("{}_delim_{}", s.name, 2 * i + 1), component: s.name.clone(), mode: s.mode, prior_scale: s.prior_scale, holiday: None });
            cols.push(Column { name: format!("{}_delim_{}", s.name, 2 * i + 2), component: s.name.clone(), mode: s.mode, prior_scale: s.prior_scale, holiday: None });
        }
    }
    let mut hcols = Vec::new();
    for (hi, h) in spec.holidays.iter().enumerate() {
        for off in h.lower_window..=h.upper_window {
            hcols.push(Column { name: format!("{}_delim_{}{}", h.name, if off >= 0 { '+' } else { '-' }, off.abs()), component: h.name.clone(), mode: spec.holidays_mode, prior_scale: h.prior_scale, holiday: Some((hi, off)) });
        }
    }
    hcols.sort_by(|a, b| a.name.cmp(&b.name));
    cols.extend(hcols);
    cols
}

pub fn feature_row(day: i64, spec: &Spec, cols: &[Column], out: &mut Vec<f64>) {
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
        let hit = spec.holidays[hi].days.iter().any(|&d| d + off == day);
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
    (0..num).map(|i| { let v = if i + 1 == num { stop } else { step * i as f64 }; v.round_ties_even() as usize }).collect()
}

pub fn make_design(ds_days: &[i64], y: &[f64], spec: &Spec) -> Design {
    let n = y.len();
    assert!(n >= 2 && ds_days.windows(2).all(|w| w[0] < w[1]));
    let start_days = ds_days[0];
    let t_scale_days = (ds_days[n - 1] - start_days) as f64;
    let t: Vec<f64> = ds_days.iter().map(|&d| (d - start_days) as f64 / t_scale_days).collect();
    let mut y_scale = y.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    if y_scale == 0.0 { y_scale = 1.0; }
    let y_scaled: Vec<f64> = y.iter().map(|v| v / y_scale).collect();
    let cap_scaled = match (spec.growth, spec.cap) {
        (Growth::Logistic, Some(c)) => Some(vec![c / y_scale; n]),
        (Growth::Logistic, None) => panic!("logistic growth needs cap"),
        _ => None,
    };
    let hist_size = (n as f64 * spec.changepoint_range).floor() as usize;
    let mut n_cp = spec.n_changepoints;
    if n_cp + 1 > hist_size { n_cp = hist_size.saturating_sub(1); }
    let changepoints_t: Vec<f64> = if n_cp > 0 { let idx = linspace_round((hist_size - 1) as f64, n_cp + 1); idx[1..].iter().map(|&i| t[i]).collect() } else { vec![0.0] };
    let cols = columns(spec);
    let k = cols.len();
    let mut x = Vec::with_capacity(n * k);
    for &d in ds_days { feature_row(d, spec, &cols, &mut x); }
    let s_a: Vec<f64> = cols.iter().map(|c| if c.mode == Mode::Additive { 1.0 } else { 0.0 }).collect();
    let s_m: Vec<f64> = cols.iter().map(|c| if c.mode == Mode::Multiplicative { 1.0 } else { 0.0 }).collect();
    let prior_scales = cols.iter().map(|c| c.prior_scale).collect();
    Design { spec: spec.clone(), cols, t, y_scaled, cap_scaled, y_scale, start_days, t_scale_days, changepoints_t, x, k, s_a, s_m, prior_scales }
}

// ---------------------------------------------------------------- model ----
#[derive(Clone, Debug)]
pub struct Params { pub k: f64, pub m: f64, pub delta: Vec<f64>, pub beta: Vec<f64>, pub sigma_obs: f64 }

pub struct Model<'a> { pub d: &'a Design, pub scale: f64, pub guard: bool }

/// Piecewise-linear trend for sorted `t`.
pub fn piecewise_linear(t: &[f64], cps: &[f64], delta: &[f64], k: f64, m: f64) -> Vec<f64> {
    let (mut out, mut j, mut k_t, mut m_t) = (Vec::with_capacity(t.len()), 0, k, m);
    for &ti in t {
        while j < cps.len() && cps[j] <= ti { k_t += delta[j]; m_t -= cps[j] * delta[j]; j += 1; }
        out.push(k_t * ti + m_t);
    }
    out
}

/// Stan's `logistic_gamma`: continuity offsets for the logistic trend.
pub fn logistic_gammas(k: f64, m: f64, delta: &[f64], cps: &[f64]) -> Vec<f64> {
    let s = cps.len();
    let mut k_s = Vec::with_capacity(s + 1);
    k_s.push(k);
    for j in 0..s { k_s.push(k_s[j] + delta[j]); }
    let mut gamma = vec![0.0; s];
    let mut m_pr = m;
    for i in 0..s { gamma[i] = (cps[i] - m_pr) * (1.0 - k_s[i] / k_s[i + 1]); m_pr += gamma[i]; }
    gamma
}

pub fn piecewise_logistic(t: &[f64], cap: &[f64], cps: &[f64], delta: &[f64], k: f64, m: f64) -> Vec<f64> {
    let gamma = logistic_gammas(k, m, delta, cps);
    let (mut out, mut j, mut k_t, mut m_t) = (Vec::with_capacity(t.len()), 0, k, m);
    for (i, &ti) in t.iter().enumerate() {
        while j < cps.len() && cps[j] <= ti { k_t += delta[j]; m_t += gamma[j]; j += 1; }
        out.push(cap[i] / (1.0 + (-k_t * (ti - m_t)).exp()));
    }
    out
}

impl<'a> Model<'a> {
    pub fn new(d: &'a Design) -> Self { Model { d, scale: 1.0 / d.t.len() as f64, guard: true } }
    pub fn n_params(&self) -> usize { 2 + self.d.changepoints_t.len() + self.d.k + 1 }
    pub fn pack(&self, p: &Params) -> Vec<f64> { let mut v = vec![p.k, p.m]; v.extend_from_slice(&p.delta); v.extend_from_slice(&p.beta); v.push(p.sigma_obs.ln()); v }
    pub fn unpack(&self, th: &[f64]) -> Params {
        let (s, k) = (self.d.changepoints_t.len(), self.d.k);
        Params { k: th[0], m: th[1], delta: th[2..2 + s].to_vec(), beta: th[2 + s..2 + s + k].to_vec(), sigma_obs: th[2 + s + k].exp() }
    }

    /// Prophet's `*_growth_init`.
    pub fn init(&self) -> Params {
        let d = self.d;
        let n = d.t.len();
        let (k, m) = match d.spec.growth {
            Growth::Linear => { let tt = d.t[n - 1] - d.t[0]; let k = (d.y_scaled[n - 1] - d.y_scaled[0]) / tt; (k, d.y_scaled[0] - k * d.t[0]) }
            Growth::Flat => (0.0, d.y_scaled.iter().sum::<f64>() / n as f64),
            Growth::Logistic => {
                let cap = d.cap_scaled.as_ref().expect("cap");
                let tt = d.t[n - 1] - d.t[0];
                let (c0, c1) = (cap[0], cap[n - 1]);
                let y0 = (0.01 * c0).max((0.99 * c0).min(d.y_scaled[0]));
                let y1 = (0.01 * c1).max((0.99 * c1).min(d.y_scaled[n - 1]));
                let (mut r0, r1) = (c0 / y0, c1 / y1);
                if (r0 - r1).abs() <= 0.01 { r0 *= 1.05; }
                let (l0, l1) = ((r0 - 1.0).ln(), (r1 - 1.0).ln());
                ((l0 - l1) / tt, l0 * tt / (l0 - l1))
            }
        };
        Params { k, m, delta: vec![0.0; d.changepoints_t.len()], beta: vec![0.0; d.k], sigma_obs: 1.0 }
    }

    pub fn trend(&self, p: &Params, t: &[f64], cap: Option<&[f64]>) -> Vec<f64> {
        let d = self.d;
        match d.spec.growth {
            Growth::Linear => piecewise_linear(t, &d.changepoints_t, &p.delta, p.k, p.m),
            Growth::Flat => vec![p.m; t.len()],
            Growth::Logistic => piecewise_logistic(t, cap.expect("cap"), &d.changepoints_t, &p.delta, p.k, p.m),
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
            for c in 0..d.k { let v = row[c] * p.beta[c]; a += v * d.s_a[c]; m += v * d.s_m[c]; }
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
        f += p.beta.iter().zip(&d.prior_scales).map(|(b, sc)| b * b / (2.0 * sc * sc)).sum::<f64>();
        f += n * s.ln() + r.iter().map(|v| v * v).sum::<f64>() / (2.0 * s * s);
        let f = f * self.scale;
        if self.guard && !f.is_finite() { return 1e300; }
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
        (self.objective_from(&p, &r), self.gradient_from(th.len(), &p, &trend, &xm, &r))
    }

    fn gradient_from(&self, n_theta: usize, p: &Params, trend: &[f64], xm: &[f64], r: &[f64]) -> Vec<f64> {
        let d = self.d;
        let p = p.clone();
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
                    while i > 0 && d.t[i - 1] >= d.changepoints_t[j] { i -= 1; st += tbar[i] * d.t[i]; s0 += tbar[i]; }
                    g[2 + j] += st - d.changepoints_t[j] * s0;
                }
            }
            Growth::Flat => { g[1] += tbar.iter().sum::<f64>(); }
            Growth::Logistic => {
                let cap = d.cap_scaled.as_ref().expect("cap");
                let cps = &d.changepoints_t;
                let gamma = logistic_gammas(p.k, p.m, &p.delta, cps);
                let mut k_s = vec![p.k];
                for j in 0..ncp { k_s.push(k_s[j] + p.delta[j]); }
                // forward per-point k_t, m_t and adjoints w.r.t. k_t (kt_bar) and m_t (mt_bar)
                let (mut j, mut k_t, mut m_t) = (0usize, p.k, p.m);
                let (mut kbar_seg, mut mbar_seg) = (vec![0.0; ncp + 1], vec![0.0; ncp + 1]); // accumulated per segment index
                for i in 0..n {
                    while j < ncp && cps[j] <= d.t[i] { k_t += p.delta[j]; m_t += gamma[j]; j += 1; }
                    let z = k_t * (d.t[i] - m_t);
                    let sig = 1.0 / (1.0 + (-z).exp());
                    let dsig = cap[i] * sig * (1.0 - sig);
                    kbar_seg[j] += tbar[i] * dsig * (d.t[i] - m_t);
                    mbar_seg[j] += tbar[i] * dsig * (-k_t);
                }
                // k_t in segment j = k + Σ_{l<j} δ_l ; m_t in segment j = m + Σ_{l<j} γ_l
                let mut suffix_k = 0.0; let mut suffix_m = 0.0;
                let mut gbar = vec![0.0; ncp]; // adjoint of γ_l from m_t
                let mut dbar = vec![0.0; ncp];
                for seg in (0..=ncp).rev() {
                    suffix_k += kbar_seg[seg];
                    suffix_m += mbar_seg[seg];
                    if seg > 0 { dbar[seg - 1] += suffix_k; gbar[seg - 1] += suffix_m; }
                }
                g[0] += suffix_k; // direct k
                g[1] += suffix_m; // direct m
                // reverse through γ_j = a_j b_j, a_j = cp_j − m − Σ_{i<j} γ_i, b_j = 1 − k_s[j]/k_s[j+1]
                let mut ksbar = vec![0.0; ncp + 1];
                let mut acc = 0.0; // Σ_{l>j} γ̄_l b_l
                let mut gsum = 0.0; // Σ_{i<j} γ_i, computed forward; store prefix sums
                let mut prefix = vec![0.0; ncp];
                for j in 0..ncp { prefix[j] = gsum; gsum += gamma[j]; }
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
                for j in (0..=ncp).rev() { suf += ksbar[j]; if j > 0 { dbar[j - 1] += suf; } }
                g[0] += suf;
                for j in 0..ncp { g[2 + j] += dbar[j]; }
            }
        }
        // priors on k, m, δ
        g[0] += p.k / 25.0;
        g[1] += p.m / 25.0;
        for j in 0..ncp { g[2 + j] += p.delta[j].signum() * if p.delta[j] == 0.0 { 0.0 } else { 1.0 } / d.spec.changepoint_prior_scale; }
        // β: dμ_i/dβ_c = X_ic (trend_i s_m,c + s_a,c)
        let off = 2 + ncp;
        // one row-major pass over X: acc_c = Σ_i −r_i X_ic (trend_i s_m,c + s_a,c)
        let mut acc = vec![0.0; d.k];
        for i in 0..n {
            let row = &d.x[i * d.k..(i + 1) * d.k];
            let (ri, ti) = (-r[i], trend[i]);
            for c in 0..d.k { acc[c] += ri * row[c] * (ti * d.s_m[c] + d.s_a[c]); }
        }
        for c in 0..d.k { g[off + c] = p.beta[c] / (d.prior_scales[c] * d.prior_scales[c]) + acc[c] * inv_s2; }
        let ss: f64 = r.iter().map(|v| v * v).sum();
        g[off + d.k] = 4.0 * s * s + n as f64 - ss * inv_s2;
        for v in g.iter_mut() { *v *= self.scale; }
        if self.guard && g.iter().any(|v| !v.is_finite()) { return vec![0.0; n_theta]; }
        g
    }
}

// ------------------------------------------------------------------ rng ----
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self { Rng(seed.max(1)) }
    pub fn next_u64(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x }
    pub fn uniform(&mut self) -> f64 { (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 }
    pub fn normal(&mut self) -> f64 { let u1 = self.uniform().max(1e-300); let u2 = self.uniform(); (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos() }
    pub fn laplace(&mut self, b: f64) -> f64 { let u = self.uniform() - 0.5; -b * u.signum() * (1.0 - 2.0 * u.abs()).ln() }
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

pub fn predict(d: &Design, p: &Params, ds_days: &[i64], seed: u64) -> Forecast {
    let spec = &d.spec;
    let n = ds_days.len();
    let t: Vec<f64> = ds_days.iter().map(|&x| (x - d.start_days) as f64 / d.t_scale_days).collect();
    let cap: Option<Vec<f64>> = d.cap_scaled.as_ref().map(|c| vec![c[0]; n]);
    let model = Model::new(d);
    let trend_s = model.trend(p, &t, cap.as_deref());
    let mut x = Vec::with_capacity(n * d.k);
    for &day in ds_days { feature_row(day, spec, &d.cols, &mut x); }
    // components
    let mut names: Vec<String> = Vec::new();
    for c in &d.cols { if !names.contains(&c.component) { names.push(c.component.clone()); } }
    let mut comp_of = |sel: &dyn Fn(&Column) -> bool, additive: bool| -> Vec<f64> {
        (0..n).map(|i| { let mut v = 0.0; for c in 0..d.k { if sel(&d.cols[c]) { v += x[i * d.k + c] * p.beta[c]; } } if additive { v * d.y_scale } else { v } }).collect()
    };
    let mut components: Vec<(String, Vec<f64>)> = Vec::new();
    for nm in &names {
        let mode = d.cols.iter().find(|c| &c.component == nm).expect("col").mode;
        let v = comp_of(&|c: &Column| &c.component == nm, mode == Mode::Additive);
        components.push((nm.clone(), v));
    }
    let add_terms = comp_of(&|c: &Column| c.mode == Mode::Additive, true);
    let mul_terms = comp_of(&|c: &Column| c.mode == Mode::Multiplicative, false);
    if !spec.holidays.is_empty() { let hol = comp_of(&|c: &Column| c.holiday.is_some(), spec.holidays_mode == Mode::Additive); components.push(("holidays".into(), hol)); }
    components.push(("additive_terms".into(), add_terms.clone()));
    components.push(("multiplicative_terms".into(), mul_terms.clone()));
    let trend: Vec<f64> = trend_s.iter().map(|v| v * d.y_scale).collect();
    let yhat: Vec<f64> = (0..n).map(|i| trend[i] * (1.0 + mul_terms[i]) + add_terms[i]).collect();

    // ---- uncertainty: Prophet 1.4 vectorised path ----
    let ns = spec.uncertainty_samples;
    let mut rng = Rng::new(seed);
    let n_future = t.iter().filter(|&&v| v > 1.0).count();
    let n_past = n - n_future;
    // uncertainties[s][i] on the scaled trend
    let mut unc = vec![vec![0.0; n]; ns];
    if n_future > 0 {
        let future_t: Vec<f64> = t.iter().cloned().filter(|&v| v > 1.0).collect();
        let single_diff = if n_future > 1 { (future_t[n_future - 1] - future_t[0]) / (n_future - 1) as f64 } else { (d.t[d.t.len() - 1] - d.t[0]) / (d.t.len() - 1) as f64 };
        let likelihood = d.changepoints_t.len() as f64 * single_diff;
        let mean_delta = p.delta.iter().map(|v| v.abs()).sum::<f64>() / p.delta.len() as f64 + 1e-8;
        match spec.growth {
            Growth::Linear => {
                for row in unc.iter_mut() {
                    // _make_trend_shift_matrix: laplace shifts where U < likelihood, averaged with the previous column
                    let mut mat: Vec<f64> = (0..n_future).map(|_| { let hit = rng.uniform() < likelihood; let v = rng.laplace(mean_delta); if hit { v } else { 0.0 } }).collect();
                    for i in (0..n_future).rev() { let prev = if i == 0 { 0.0 } else { mat[i - 1] }; mat[i] = (prev + mat[i]) / 2.0; }
                    let mut c1 = 0.0; let mut c2 = 0.0;
                    for i in 0..n_future { c1 += mat[i]; c2 += c1; row[n_past + i] = c2 * single_diff; }
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
                    // Poisson via Knuth
                    let mut n_changes = 0usize; let mut pp = 1.0; let l = (-lambda).exp();
                    loop { pp *= rng.uniform(); if pp <= l { break; } n_changes += 1; }
                    let mut cps: Vec<f64> = d.changepoints_t.clone();
                    let mut deltas: Vec<f64> = p.delta.clone();
                    let mut new: Vec<(f64, f64)> = (0..n_changes).map(|_| (1.0 + rng.uniform() * (t_max - 1.0), rng.laplace(mean_delta))).collect();
                    new.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("f"));
                    for (c, dl) in new { cps.push(c); deltas.push(dl); }
                    let tr = piecewise_logistic(&t, cap_s, &cps, &deltas, p.k, p.m);
                    for i in n_past..n { row[i] = tr[i] - mean_trend[i]; }
                }
            }
        }
    }
    // sample_model_vectorized: yhat = (trend + unc)·y_scale·(1 + Xb_m) + Xb_a + N(0, σ)·y_scale
    let (lo_p, hi_p) = (100.0 * (1.0 - spec.interval_width) / 2.0, 100.0 * (1.0 + spec.interval_width) / 2.0);
    let (mut yl, mut yu, mut tl, mut tu) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    let mut col_y = vec![0.0; ns];
    let mut col_t = vec![0.0; ns];
    for i in 0..n {
        for s in 0..ns {
            let tr = (trend_s[i] + unc[s][i]) * d.y_scale;
            col_t[s] = tr;
            col_y[s] = tr * (1.0 + mul_terms[i]) + add_terms[i] + rng.normal() * p.sigma_obs * d.y_scale;
        }
        col_y.sort_by(|a, b| a.partial_cmp(b).expect("f"));
        col_t.sort_by(|a, b| a.partial_cmp(b).expect("f"));
        yl.push(percentile(&col_y, lo_p)); yu.push(percentile(&col_y, hi_p));
        tl.push(percentile(&col_t, lo_p)); tu.push(percentile(&col_t, hi_p));
    }
    Forecast { ds_days: ds_days.to_vec(), trend, yhat, yhat_lower: yl, yhat_upper: yu, trend_lower: tl, trend_upper: tu, components }
}
