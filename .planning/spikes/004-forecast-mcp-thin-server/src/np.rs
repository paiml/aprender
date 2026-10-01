//! NeuralProphet-lite on aprender's f32 autograd: trend (11 segments, NP's
//! parametrisation), Fourier seasonality (days since 1900-01-01, sin block then
//! cos block per NP), optional AR-Net on stationarised lags, weighted Huber
//! loss, AdamW + three-phase one-cycle cosine schedule.
use crate::prophet::days_from_civil;
use aprender::autograd::{clear_graph, graph_tape_len, no_grad, Tensor};
use aprender::nn::optim::{AdamW, Optimizer};
use aprender::nn::{Linear, Module};
use std::time::Instant;

// ------------------------------------------------------------------ rng ----
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self { Rng(seed.max(1)) }
    pub fn next_u64(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x }
    pub fn uniform(&mut self) -> f64 { (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 }
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    pub fn shuffle(&mut self, v: &mut [usize]) {
        for i in (1..v.len()).rev() { let j = (self.next_u64() % (i as u64 + 1)) as usize; v.swap(i, j); }
    }
}

// ------------------------------------------------------------- features ----
#[derive(Clone, Debug)]
pub struct NpSeason { pub name: String, pub period: f64, pub order: usize }

/// NeuralProphet's auto seasonality: same disable rules as Prophet, resolutions 6 / 3 / 6.
pub fn np_auto_seasonalities(ds_days: &[i64]) -> Vec<NpSeason> {
    let span = (ds_days[ds_days.len() - 1] - ds_days[0]) as f64;
    let min_dt = ds_days.windows(2).map(|w| (w[1] - w[0]) as f64).filter(|d| *d > 0.0).fold(f64::INFINITY, f64::min);
    let mut v = Vec::new();
    if span >= 730.0 { v.push(NpSeason { name: "yearly".into(), period: 365.25, order: 6 }); }
    if span >= 14.0 && min_dt < 7.0 { v.push(NpSeason { name: "weekly".into(), period: 7.0, order: 3 }); }
    if span >= 2.0 && min_dt < 1.0 { v.push(NpSeason { name: "daily".into(), period: 1.0, order: 6 }); }
    v
}

pub fn season_dim(s: &[NpSeason]) -> usize { s.iter().map(|x| 2 * x.order).sum() }

/// NP: t in days since 1900-01-01; per seasonality all sin(k·2πt/P) then all cos.
pub fn fourier_feats(day: i64, seasons: &[NpSeason], out: &mut Vec<f32>) {
    let t = (day - days_from_civil(1900, 1, 1)) as f64;
    for s in seasons {
        let f = 2.0 * std::f64::consts::PI / s.period;
        for k in 1..=s.order { out.push((f * k as f64 * t).sin() as f32); }
        for k in 1..=s.order { out.push((f * k as f64 * t).cos() as f32); }
    }
}

/// NP trend as a linear function of [k0, δ_0..δ_S]: slope in segment i is k0 + δ_i,
/// continuity offsets −cp_j·(δ_j − δ_{j−1}) for every passed changepoint j ≥ 1.
pub fn trend_feats(t: f64, cps: &[f64], out: &mut Vec<f32>) {
    let s = cps.len();
    let seg = cps[1..].iter().filter(|&&c| t >= c).count();
    let mut phi = vec![0.0f64; s];
    phi[seg] += t;
    for j in 1..s {
        if t >= cps[j] { phi[j] -= cps[j]; phi[j - 1] += cps[j]; }
    }
    out.push(t as f32);
    out.extend(phi.iter().map(|v| *v as f32));
}

/// Newer-samples weight (NP `_get_time_based_sample_weight`, end_w = 2, start_t = 0).
pub fn sample_weight(t: f64, end_w: f64) -> f32 {
    let time = t.clamp(0.0, 1.0);
    let c = 0.5 * (std::f64::consts::PI * (time - 1.0)).cos() + 0.5;
    ((1.0 + c * (end_w - 1.0)) / end_w) as f32
}

// ----------------------------------------------------------------- data ----
pub struct NpData {
    /// Daily grid covering [first, last] of the FULL series (train + test); y linearly imputed.
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
    pub fn new(ds_days: &[i64], y: &[f64], n_train_rows: usize, n_changepoints: usize, changepoints_range: f64) -> Self {
        let first = ds_days[0];
        let last = ds_days[ds_days.len() - 1];
        let n = (last - first + 1) as usize;
        let mut grid_y = vec![f64::NAN; n];
        let mut grid_observed = vec![false; n];
        for (d, v) in ds_days.iter().zip(y) { let i = (d - first) as usize; grid_y[i] = *v; grid_observed[i] = true; }
        // linear imputation
        let mut i = 0;
        while i < n {
            if grid_y[i].is_nan() {
                let a = i - 1; let mut b = i; while grid_y[b].is_nan() { b += 1; }
                for k in i..b { let f = (k - a) as f64 / (b - a) as f64; grid_y[k] = grid_y[a] + f * (grid_y[b] - grid_y[a]); }
                i = b;
            }
            i += 1;
        }
        let train_last = ds_days[n_train_rows - 1];
        let n_train_grid = (train_last - first + 1) as usize;
        // soft normalisation from OBSERVED train values: shift = min, scale = q95 − min
        let mut obs: Vec<f64> = y[..n_train_rows].to_vec();
        obs.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        let lowest = obs[0];
        let q = 0.95 * (obs.len() - 1) as f64; let lo = q.floor() as usize; let frac = q - lo as f64;
        let q95 = obs[lo] + frac * (obs[(lo + 1).min(obs.len() - 1)] - obs[lo]);
        let width = if (q95 - lowest).abs() < 1e-12 { obs[obs.len() - 1] - lowest } else { q95 - lowest };
        let cps: Vec<f64> = (0..=n_changepoints).map(|i| changepoints_range * i as f64 / (n_changepoints + 1) as f64).collect();
        let seasons = np_auto_seasonalities(&ds_days[..n_train_rows]);
        NpData { grid_days: (first..=last).collect(), grid_y, grid_observed, n_train_grid, shift: lowest, scale: width, t0: first, t_span: (train_last - first) as f64, cps, seasons }
    }
    pub fn t_of(&self, day: i64) -> f64 { (day - self.t0) as f64 / self.t_span }
    pub fn norm(&self, v: f64) -> f32 { ((v - self.shift) / self.scale) as f32 }
    pub fn denorm(&self, v: f32) -> f64 { v as f64 * self.scale + self.shift }
    pub fn trend_dim(&self) -> usize { self.cps.len() + 1 }
    pub fn season_dim(&self) -> usize { season_dim(&self.seasons) }
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
    /// NP init: k0 ~ xavier on [1,1] (std 1), δ std √(2/(2S)), season std √(1/(2R)), bias std 1, AR kaiming fan_in.
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
        for se in &d.seasons { let std = (1.0 / (2.0 * se.order as f64)).sqrt(); w.extend((0..2 * se.order).map(|_| (rng.normal() * std) as f32)); }
        season.set_weight(Tensor::from_vec(w, &[1, sd]).requires_grad());
        let bias = normal_tensor(rng, &[1], 1.0);
        let mut ar = Vec::new();
        if n_lags > 0 {
            let mut d_in = n_lags;
            for &h in ar_layers {
                let mut l = Linear::new(d_in, h);
                l.set_weight(normal_tensor(rng, &[h, d_in], (2.0 / d_in as f64).sqrt()));
                l.set_bias(Tensor::zeros(&[h]).requires_grad());
                ar.push(l); d_in = h;
            }
            let mut l = Linear::without_bias(d_in, 1);
            l.set_weight(normal_tensor(rng, &[1, d_in], (2.0 / d_in as f64).sqrt()));
            ar.push(l);
        }
        NpModel { trend, season, bias, ar, n_lags }
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut v = self.trend.parameters_mut();
        v.extend(self.season.parameters_mut());
        v.push(&mut self.bias);
        for l in &mut self.ar { v.extend(l.parameters_mut()); }
        v
    }

    pub fn n_params(&self) -> usize {
        self.trend.num_parameters() + self.season.num_parameters() + 1 + self.ar.iter().map(|l| l.num_parameters()).sum::<usize>()
    }

    /// `lags`: (raw normalised lags [b,L], season feats at lag times [b·L,F], trend at lag times [b,L] (detached)).
    pub fn forward(&self, x_trend: &Tensor, x_season: &Tensor, lags: Option<(&Tensor, &Tensor, &Tensor)>) -> Tensor {
        let b = x_trend.shape()[0];
        let out = self.trend.forward(x_trend).add(&self.season.forward(x_season)).broadcast_add(&self.bias);
        match lags {
            None => out,
            Some((raw, se_lag, tr_lag)) => {
                let s_lag = self.season.forward(se_lag).view(&[b, self.n_lags]);
                let mut h = raw.sub(tr_lag).sub(&s_lag);
                let n = self.ar.len();
                for (i, l) in self.ar.iter().enumerate() { h = l.forward(&h); if i + 1 < n { h = h.relu(); } }
                out.add(&h)
            }
        }
    }
}

/// Exact SmoothL1 (Huber, β) × sample weight, mean — built from graph-connected ops with a
/// constant 0/1 mask (the mask is piecewise constant, so this IS the exact gradient).
pub fn weighted_huber(pred: &Tensor, target: &Tensor, w: &Tensor, beta: f32) -> Tensor {
    let d = pred.sub(target);
    let a = d.abs();
    let mask: Vec<f32> = a.data().iter().map(|v| if *v < beta { 1.0 } else { 0.0 }).collect();
    let inv: Vec<f32> = mask.iter().map(|m| 1.0 - m).collect();
    let m = Tensor::from_vec(mask, a.shape());
    let im = Tensor::from_vec(inv, a.shape());
    let half_beta = Tensor::from_vec(vec![0.5 * beta; a.numel()], a.shape());
    let quad = d.pow(2.0).mul_scalar(0.5 / beta).mul(&m);
    let lin = a.sub(&half_beta).mul(&im);
    quad.add(&lin).mul(w).mean()
}

/// NP's OneCycleLR (three_phase, cos, div 10, final_div 10) over progress p ∈ [0,1].
pub fn one_cycle_lr(p: f64, max_lr: f64) -> f64 {
    let (init, fin) = (max_lr / 10.0, max_lr / 100.0);
    let cosine = |s: f64, e: f64, f: f64| e + (s - e) / 2.0 * (1.0 + (std::f64::consts::PI * f).cos());
    if p < 0.3 { cosine(init, max_lr, p / 0.3) } else if p < 0.6 { cosine(max_lr, init, (p - 0.3) / 0.3) } else { cosine(init, fin, (p - 0.6) / 0.4) }
}

pub fn auto_batch(n: usize) -> usize { (2usize.pow(1 + (1.5 * (n as f64).log10()) as u32)).clamp(8, 2048).min(n) }
pub fn auto_epochs(n: usize) -> usize { (10.0 * (100.0 / n as f64 * 2f64.powf(2.25 * (10.0 + n as f64).log10())).ceil()).clamp(20.0, 500.0) as usize }

// -------------------------------------------------------------- training ----
pub struct TrainConfig { pub n_lags: usize, pub ar_layers: Vec<usize>, pub max_lr: f64, pub epochs: Option<usize>, pub batch: Option<usize>, pub weight_decay: f32, pub huber_beta: f32, pub newer_w: f64, pub seed: u64 }

pub struct TrainLog { pub epochs: usize, pub batch: usize, pub n_samples: usize, pub n_params: usize, pub epoch_loss: Vec<f64>, pub seconds: f64, pub tape_len_per_step: usize, pub steps: usize }

/// Precomputed per-grid-row features.
pub struct Rows { pub tr: Vec<f32>, pub se: Vec<f32>, pub y: Vec<f32>, pub w: Vec<f32>, pub td: usize, pub sd: usize }

pub fn rows_for(d: &NpData, days: &[i64], y_norm: &[f32]) -> Rows {
    let (td, sd) = (d.trend_dim(), d.season_dim());
    let (mut tr, mut se, mut w) = (Vec::with_capacity(days.len() * td), Vec::with_capacity(days.len() * sd), Vec::with_capacity(days.len()));
    for &day in days { d.row_feats(day, &mut tr, &mut se); w.push(sample_weight(d.t_of(day), 2.0)); }
    Rows { tr, se, y: y_norm.to_vec(), w, td, sd }
}

fn gather(rows: &[f32], width: usize, idx: &[usize]) -> Vec<f32> {
    let mut v = Vec::with_capacity(idx.len() * width);
    for &i in idx { v.extend_from_slice(&rows[i * width..(i + 1) * width]); }
    v
}

pub fn train(d: &NpData, cfg: &TrainConfig, verbose: bool) -> (NpModel, TrainLog) {
    let mut rng = Rng::new(cfg.seed);
    let mut model = NpModel::new(d, cfg.n_lags, &cfg.ar_layers, &mut rng);
    let n_grid = d.n_train_grid;
    let y_norm: Vec<f32> = d.grid_y[..n_grid].iter().map(|v| d.norm(*v)).collect();
    let rows = rows_for(d, &d.grid_days[..n_grid], &y_norm);
    let l = cfg.n_lags;
    // NP trains the lag-free model on the OBSERVED rows only (no imputation needed);
    // with lags it trains on the imputed daily grid so every window is complete.
    let samples: Vec<usize> = if l == 0 { (0..n_grid).filter(|&i| d.grid_observed[i]).collect() } else { (l..n_grid).collect() };
    let n = samples.len();
    let batch = cfg.batch.unwrap_or_else(|| auto_batch(n));
    let epochs = cfg.epochs.unwrap_or_else(|| auto_epochs(n));
    let n_batches = n.div_ceil(batch);
    let total_steps = epochs * n_batches;
    let mut opt = {
        let params = model.parameters_mut();
        AdamW::new(params, cfg.max_lr as f32).weight_decay(cfg.weight_decay)
    };
    let mut log = TrainLog { epochs, batch, n_samples: n, n_params: model.n_params(), epoch_loss: Vec::new(), seconds: 0.0, tape_len_per_step: 0, steps: 0 };
    let t0 = Instant::now();
    let mut order = samples.clone();
    let mut step = 0usize;
    for epoch in 0..epochs {
        rng.shuffle(&mut order);
        let mut acc = 0.0;
        for chunk in order.chunks(batch) {
            let p = step as f64 / total_steps as f64;
            opt.set_lr(one_cycle_lr(p, cfg.max_lr) as f32);
            let b = chunk.len();
            let xt = Tensor::from_vec(gather(&rows.tr, rows.td, chunk), &[b, rows.td]);
            let xs = Tensor::from_vec(gather(&rows.se, rows.sd, chunk), &[b, rows.sd]);
            let yt = Tensor::from_vec(chunk.iter().map(|&i| rows.y[i]).collect(), &[b, 1]);
            let wt = Tensor::from_vec(chunk.iter().map(|&i| rows.w[i]).collect(), &[b, 1]);
            let pred = if l == 0 {
                model.forward(&xt, &xs, None)
            } else {
                let lag_idx: Vec<usize> = chunk.iter().flat_map(|&i| (i - l..i)).collect();
                let raw = Tensor::from_vec(lag_idx.iter().map(|&j| rows.y[j]).collect(), &[b, l]);
                let se_lag = Tensor::from_vec(gather(&rows.se, rows.sd, &lag_idx), &[b * l, rows.sd]);
                let tr_lag_feats = Tensor::from_vec(gather(&rows.tr, rows.td, &lag_idx), &[b * l, rows.td]);
                let tr_lag = no_grad(|| model.trend.forward(&tr_lag_feats).detach()).view(&[b, l]);
                let tr_lag = Tensor::from_vec(tr_lag.data().to_vec(), &[b, l]);
                model.forward(&xt, &xs, Some((&raw, &se_lag, &tr_lag)))
            };
            let loss = weighted_huber(&pred, &yt, &wt, cfg.huber_beta);
            acc += loss.item() as f64 * b as f64;
            loss.backward();
            if step == 0 { log.tape_len_per_step = graph_tape_len(); }
            let mut params = model.parameters_mut();
            opt.step_with_params(&mut params);
            opt.zero_grad();
            clear_graph();
            step += 1;
        }
        log.epoch_loss.push(acc / n as f64);
        if verbose && (epoch % 10 == 0 || epoch + 1 == epochs) { eprintln!("  epoch {epoch:3} loss {:.5} lr {:.2e}", acc / n as f64, one_cycle_lr(step as f64 / total_steps as f64, cfg.max_lr)); }
    }
    log.seconds = t0.elapsed().as_secs_f64();
    log.steps = step;
    (model, log)
}

/// Trend + seasonality prediction (original scale) for arbitrary days.
pub fn predict_ts(d: &NpData, m: &NpModel, days: &[i64]) -> Vec<f64> {
    let rows = rows_for(d, days, &vec![0.0; days.len()]);
    let xt = Tensor::from_vec(rows.tr.clone(), &[days.len(), rows.td]);
    let xs = Tensor::from_vec(rows.se.clone(), &[days.len(), rows.sd]);
    let out = no_grad(|| m.forward(&xt, &xs, None));
    clear_graph();
    out.data().iter().map(|v| d.denorm(*v)).collect()
}

/// One-step-ahead AR prediction (original scale) for grid indices `idx` using true lags.
pub fn predict_ar_1step(d: &NpData, m: &NpModel, idx: &[usize]) -> Vec<f64> {
    let l = m.n_lags;
    let y_norm: Vec<f32> = d.grid_y.iter().map(|v| d.norm(*v)).collect();
    let rows = rows_for(d, &d.grid_days, &y_norm);
    let b = idx.len();
    let xt = Tensor::from_vec(gather(&rows.tr, rows.td, idx), &[b, rows.td]);
    let xs = Tensor::from_vec(gather(&rows.se, rows.sd, idx), &[b, rows.sd]);
    let lag_idx: Vec<usize> = idx.iter().flat_map(|&i| (i - l..i)).collect();
    let raw = Tensor::from_vec(lag_idx.iter().map(|&j| rows.y[j]).collect(), &[b, l]);
    let se_lag = Tensor::from_vec(gather(&rows.se, rows.sd, &lag_idx), &[b * l, rows.sd]);
    let tr_feats = Tensor::from_vec(gather(&rows.tr, rows.td, &lag_idx), &[b * l, rows.td]);
    let out = no_grad(|| {
        let tr_lag = Tensor::from_vec(m.trend.forward(&tr_feats).data().to_vec(), &[b, l]);
        m.forward(&xt, &xs, Some((&raw, &se_lag, &tr_lag)))
    });
    clear_graph();
    out.data().iter().map(|v| d.denorm(*v)).collect()
}

/// Trend component only (original scale) for arbitrary days.
pub fn predict_trend(d: &NpData, m: &NpModel, days: &[i64]) -> Vec<f64> {
    let rows = rows_for(d, days, &vec![0.0; days.len()]);
    let xt = Tensor::from_vec(rows.tr.clone(), &[days.len(), rows.td]);
    let out = no_grad(|| m.trend.forward(&xt).broadcast_add(&m.bias));
    clear_graph();
    out.data().iter().map(|v| d.denorm(*v)).collect()
}

/// Multi-step AR forecast by feeding predictions back as lags (days must continue the daily grid).
pub fn predict_ar_recursive(d: &NpData, m: &NpModel, days: &[i64]) -> Vec<f64> {
    let l = m.n_lags;
    let mut hist: Vec<f32> = d.grid_y.iter().map(|v| d.norm(*v)).collect();
    let mut hist_days: Vec<i64> = d.grid_days.clone();
    let mut out = Vec::with_capacity(days.len());
    for &day in days {
        // fill any gap between the last known day and this one recursively
        while hist_days[hist_days.len() - 1] + 1 <= day {
            let next = hist_days[hist_days.len() - 1] + 1;
            let n = hist.len();
            let (mut tr, mut se) = (Vec::new(), Vec::new());
            d.row_feats(next, &mut tr, &mut se);
            let lag_days: Vec<i64> = (next - l as i64..next).collect();
            let (mut se_lag, mut tr_lag) = (Vec::new(), Vec::new());
            for &ld in &lag_days { d.row_feats(ld, &mut tr_lag, &mut se_lag); }
            let xt = Tensor::from_vec(tr, &[1, d.trend_dim()]);
            let xs = Tensor::from_vec(se, &[1, d.season_dim()]);
            let raw = Tensor::from_vec(hist[n - l..].to_vec(), &[1, l]);
            let se_l = Tensor::from_vec(se_lag, &[l, d.season_dim()]);
            let tr_feats = Tensor::from_vec(tr_lag, &[l, d.trend_dim()]);
            let v = no_grad(|| { let tl = Tensor::from_vec(m.trend.forward(&tr_feats).data().to_vec(), &[1, l]); m.forward(&xt, &xs, Some((&raw, &se_l, &tl))).data()[0] });
            clear_graph();
            hist.push(v);
            hist_days.push(next);
        }
        let idx = (day - hist_days[0]) as usize;
        out.push(d.denorm(hist[idx]));
    }
    out
}
