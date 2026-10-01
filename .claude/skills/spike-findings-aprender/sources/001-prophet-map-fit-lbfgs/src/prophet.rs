//! A from-scratch port of Prophet's data preparation, Stan objective and
//! prediction (linear growth, additive seasonality) — just enough to test
//! whether aprender's `LbfgsF64` reproduces Stan's MAP fit.

// ---------------------------------------------------------------- dates ----

/// Days since 1970-01-01 for a proleptic Gregorian civil date (H. Hinnant).
pub fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m as i64 + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Inverse of [`days_from_civil`].
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

pub fn format_ymd(days: i64) -> String {
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}")
}

// --------------------------------------------------------------- design ----

#[derive(Clone, Debug)]
pub struct Seasonality {
    pub name: String,
    pub period: f64,
    pub order: usize,
    pub prior_scale: f64,
}

/// Everything Prophet's `preprocess` hands to Stan, built from raw (ds, y).
#[derive(Clone, Debug)]
pub struct Design {
    pub t: Vec<f64>,
    pub y_scaled: Vec<f64>,
    pub y_scale: f64,
    pub start_days: i64,
    pub t_scale_days: f64,
    pub changepoints_t: Vec<f64>,
    /// T×K row-major seasonality features.
    pub x: Vec<f64>,
    pub k: usize,
    pub prior_scales: Vec<f64>,
    pub tau: f64,
    pub seasonalities: Vec<Seasonality>,
}

/// `numpy.linspace(0, n-1, num).round()` — numpy rounds half to even.
fn linspace_round(stop: f64, num: usize) -> Vec<usize> {
    let step = stop / (num as f64 - 1.0);
    (0..num)
        .map(|i| {
            let v = if i + 1 == num { stop } else { step * i as f64 };
            v.round_ties_even() as usize
        })
        .collect()
}

pub fn fourier_row(days: f64, seasonalities: &[Seasonality], out: &mut Vec<f64>) {
    let x_t = std::f64::consts::PI * 2.0 * days;
    for s in seasonalities {
        for i in 0..s.order {
            let c = (i + 1) as f64 / s.period * x_t;
            out.push(c.sin());
            out.push(c.cos());
        }
    }
}

pub fn make_design(
    ds_days: &[i64],
    y: &[f64],
    n_changepoints: usize,
    changepoint_range: f64,
    tau: f64,
    seasonalities: &[Seasonality],
) -> Design {
    assert_eq!(ds_days.len(), y.len());
    let n = y.len();
    assert!(n >= 2, "need at least 2 rows");
    assert!(ds_days.windows(2).all(|w| w[0] < w[1]), "ds must be sorted, unique");
    let start_days = ds_days[0];
    let t_scale_days = (ds_days[n - 1] - start_days) as f64;
    let t: Vec<f64> = ds_days.iter().map(|&d| (d - start_days) as f64 / t_scale_days).collect();
    let mut y_scale = y.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    if y_scale == 0.0 {
        y_scale = 1.0;
    }
    let y_scaled: Vec<f64> = y.iter().map(|v| v / y_scale).collect();

    let hist_size = (n as f64 * changepoint_range).floor() as usize;
    let mut n_cp = n_changepoints;
    if n_cp + 1 > hist_size {
        n_cp = hist_size.saturating_sub(1);
    }
    let changepoints_t: Vec<f64> = if n_cp > 0 {
        let idx = linspace_round((hist_size - 1) as f64, n_cp + 1);
        idx[1..].iter().map(|&i| t[i]).collect()
    } else {
        vec![0.0] // Prophet's dummy changepoint
    };

    let k: usize = seasonalities.iter().map(|s| 2 * s.order).sum();
    let mut x = Vec::with_capacity(n * k);
    for &d in ds_days {
        fourier_row(d as f64, seasonalities, &mut x);
    }
    let mut prior_scales = Vec::with_capacity(k);
    for s in seasonalities {
        prior_scales.extend(std::iter::repeat(s.prior_scale).take(2 * s.order));
    }
    Design { t, y_scaled, y_scale, start_days, t_scale_days, changepoints_t, x, k, prior_scales, tau, seasonalities: seasonalities.to_vec() }
}

// ------------------------------------------------------------ objective ----

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum L1Mode {
    /// `|δ|` with the sign subgradient (what Stan's L-BFGS sees).
    Exact,
    /// `sqrt(δ² + ε²)`: smooth, bounded gradient.
    Smooth(f64),
}

/// θ = [k, m, δ₁..δ_S, β₁..β_K, log σ]
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
    pub l1: L1Mode,
    /// Multiplier on the objective (1 = Stan's raw sum; 1/T = per-observation mean).
    pub scale: f64,
    /// Return a huge finite value / zero gradient instead of inf/NaN so a
    /// line search can backtrack out of an overshoot.
    pub guard: bool,
}

impl<'a> Model<'a> {
    pub fn new(d: &'a Design, l1: L1Mode) -> Self {
        Self { d, l1, scale: 1.0, guard: false }
    }
    pub fn scaled(mut self) -> Self { self.scale = 1.0 / self.d.t.len() as f64; self }
    pub fn guarded(mut self) -> Self { self.guard = true; self }

    pub fn n_params(&self) -> usize {
        2 + self.d.changepoints_t.len() + self.d.k + 1
    }

    pub fn pack(&self, p: &Params) -> Vec<f64> {
        let mut v = Vec::with_capacity(self.n_params());
        v.push(p.k);
        v.push(p.m);
        v.extend_from_slice(&p.delta);
        v.extend_from_slice(&p.beta);
        v.push(p.sigma_obs.ln());
        v
    }

    pub fn unpack(&self, theta: &[f64]) -> Params {
        let s = self.d.changepoints_t.len();
        let k = self.d.k;
        Params {
            k: theta[0],
            m: theta[1],
            delta: theta[2..2 + s].to_vec(),
            beta: theta[2 + s..2 + s + k].to_vec(),
            sigma_obs: theta[2 + s + k].exp(),
        }
    }

    /// Prophet's `linear_growth_init`: line through first and last point.
    pub fn init(&self) -> Params {
        let d = self.d;
        let n = d.t.len();
        let tt = d.t[n - 1] - d.t[0];
        let k = (d.y_scaled[n - 1] - d.y_scaled[0]) / tt;
        let m = d.y_scaled[0] - k * d.t[0];
        Params { k, m, delta: vec![0.0; d.changepoints_t.len()], beta: vec![0.0; d.k], sigma_obs: 1.0 }
    }

    /// Piecewise-linear trend on scaled t (sorted), for any t and changepoints.
    pub fn piecewise_linear(t: &[f64], cps: &[f64], delta: &[f64], k: f64, m: f64) -> Vec<f64> {
        let mut out = Vec::with_capacity(t.len());
        let mut j = 0;
        let mut k_t = k;
        let mut m_t = m;
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

    fn residuals(&self, p: &Params) -> (Vec<f64>, Vec<f64>) {
        let d = self.d;
        let trend = Self::piecewise_linear(&d.t, &d.changepoints_t, &p.delta, p.k, p.m);
        let n = d.t.len();
        let mut r = Vec::with_capacity(n);
        for i in 0..n {
            let row = &d.x[i * d.k..(i + 1) * d.k];
            let xb: f64 = row.iter().zip(&p.beta).map(|(a, b)| a * b).sum();
            r.push(d.y_scaled[i] - (trend[i] + xb));
        }
        (trend, r)
    }

    fn l1(&self, v: f64) -> f64 {
        match self.l1 {
            L1Mode::Exact => v.abs(),
            L1Mode::Smooth(eps) => (v * v + eps * eps).sqrt(),
        }
    }

    fn l1_grad(&self, v: f64) -> f64 {
        match self.l1 {
            L1Mode::Exact => {
                if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 }
            }
            L1Mode::Smooth(eps) => v / (v * v + eps * eps).sqrt(),
        }
    }

    /// Negative unnormalised log posterior of prophet.stan (linear, additive).
    pub fn objective(&self, theta: &[f64]) -> f64 {
        let d = self.d;
        let p = self.unpack(theta);
        let (_, r) = self.residuals(&p);
        let n = d.t.len() as f64;
        let s = p.sigma_obs;
        let mut f = p.k * p.k / 50.0 + p.m * p.m / 50.0;
        f += p.delta.iter().map(|&v| self.l1(v)).sum::<f64>() / d.tau;
        f += 2.0 * s * s; // sigma_obs ~ normal(0, 0.5)
        f += p.beta.iter().zip(&d.prior_scales).map(|(b, sc)| b * b / (2.0 * sc * sc)).sum::<f64>();
        f += n * s.ln();
        f += r.iter().map(|v| v * v).sum::<f64>() / (2.0 * s * s);
        let f = f * self.scale;
        if self.guard && !f.is_finite() { return 1e300; }
        f
    }

    pub fn gradient(&self, theta: &[f64]) -> Vec<f64> {
        let d = self.d;
        let p = self.unpack(theta);
        let (_, r) = self.residuals(&p);
        let n = d.t.len();
        let s = p.sigma_obs;
        let inv_s2 = 1.0 / (s * s);
        let mut g = vec![0.0; theta.len()];
        // k, m
        let sum_rt: f64 = r.iter().zip(&d.t).map(|(a, b)| a * b).sum();
        let sum_r: f64 = r.iter().sum();
        g[0] = p.k / 25.0 - sum_rt * inv_s2;
        g[1] = p.m / 25.0 - sum_r * inv_s2;
        // delta_j: -(1/σ²) Σ_{i: t_i ≥ cp_j} r_i (t_i - cp_j); suffix sums over sorted t.
        let cps = &d.changepoints_t;
        let mut suffix_rt = 0.0;
        let mut suffix_r = 0.0;
        let mut i = n;
        for j in (0..cps.len()).rev() {
            while i > 0 && d.t[i - 1] >= cps[j] {
                i -= 1;
                suffix_rt += r[i] * d.t[i];
                suffix_r += r[i];
            }
            g[2 + j] = self.l1_grad(p.delta[j]) / d.tau - (suffix_rt - cps[j] * suffix_r) * inv_s2;
        }
        // beta
        let off = 2 + cps.len();
        for c in 0..d.k {
            let mut acc = 0.0;
            for (i, ri) in r.iter().enumerate() {
                acc += ri * d.x[i * d.k + c];
            }
            g[off + c] = p.beta[c] / (d.prior_scales[c] * d.prior_scales[c]) - acc * inv_s2;
        }
        // u = log σ (no Jacobian — cmdstan optimize default)
        let ss: f64 = r.iter().map(|v| v * v).sum();
        g[off + d.k] = 4.0 * s * s + n as f64 - ss * inv_s2;
        for v in g.iter_mut() { *v *= self.scale; }
        if self.guard && g.iter().any(|v| !v.is_finite()) { return vec![0.0; theta.len()]; }
        g
    }
}

// ----------------------------------------------------------- prediction ----

pub struct Prediction {
    pub ds_days: Vec<i64>,
    pub trend: Vec<f64>,
    pub components: Vec<(String, Vec<f64>)>,
    pub yhat: Vec<f64>,
}

pub fn predict(d: &Design, p: &Params, ds_days: &[i64]) -> Prediction {
    let t: Vec<f64> = ds_days.iter().map(|&x| (x - d.start_days) as f64 / d.t_scale_days).collect();
    let trend_s = Model::piecewise_linear(&t, &d.changepoints_t, &p.delta, p.k, p.m);
    let mut comps: Vec<(String, Vec<f64>)> =
        d.seasonalities.iter().map(|s| (s.name.clone(), Vec::with_capacity(t.len()))).collect();
    let mut row = Vec::with_capacity(d.k);
    let mut yhat = Vec::with_capacity(t.len());
    let mut trend = Vec::with_capacity(t.len());
    for (i, &day) in ds_days.iter().enumerate() {
        row.clear();
        fourier_row(day as f64, &d.seasonalities, &mut row);
        let mut off = 0;
        let mut total = 0.0;
        for (si, s) in d.seasonalities.iter().enumerate() {
            let w = 2 * s.order;
            let v: f64 = row[off..off + w].iter().zip(&p.beta[off..off + w]).map(|(a, b)| a * b).sum::<f64>() * d.y_scale;
            comps[si].1.push(v);
            total += v;
            off += w;
        }
        let tr = trend_s[i] * d.y_scale;
        trend.push(tr);
        yhat.push(tr + total);
    }
    Prediction { ds_days: ds_days.to_vec(), trend, components: comps, yhat }
}

// ------------------------------------------------------ auto seasonality ----

/// Prophet's `set_auto_seasonalities` for the defaults ('auto' everywhere):
/// yearly if the span is ≥ 730 days; weekly if span ≥ 2 weeks AND the smallest
/// gap between rows is < 7 days; daily if span ≥ 2 days AND smallest gap < 1 day.
pub fn auto_seasonalities(ds_days: &[i64], prior_scale: f64) -> Vec<Seasonality> {
    let span = (ds_days[ds_days.len() - 1] - ds_days[0]) as f64;
    let min_dt = ds_days.windows(2).map(|w| (w[1] - w[0]) as f64).filter(|d| *d > 0.0).fold(f64::INFINITY, f64::min);
    let mut out = Vec::new();
    if span >= 730.0 {
        out.push(Seasonality { name: "yearly".into(), period: 365.25, order: 10, prior_scale });
    }
    if span >= 14.0 && min_dt < 7.0 {
        out.push(Seasonality { name: "weekly".into(), period: 7.0, order: 3, prior_scale });
    }
    if span >= 2.0 && min_dt < 1.0 {
        out.push(Seasonality { name: "daily".into(), period: 1.0, order: 4, prior_scale });
    }
    out
}
