//! Spike 014 — what value should a future regressor take on a day NeuralProphet imputed?
//!
//! RE-SCOPED. The original question was how weekly and month-start series map onto NP's
//! daily grid; spike 012 found the door refuses `freq != "D"` for neuralprophet
//! (`forecast.rs:421`, test `neuralprophet_refuses_non_daily_freq`), so that premise does
//! not hold. What survives is the DAILY case: `NpData::new` builds a grid over
//! `[first, last]` and linearly imputes `y` on missing days. A regressor is supplied on the
//! caller's rows, so the imputed days have no value of their own.
//!
//! The question splits on `n_lags`, and the split is the whole finding:
//!   * lag-free trains on OBSERVED rows only   (`np.rs:597`)
//!   * lagged trains on `(l..n_train_grid)`    — every grid row, imputed ones included

use aprender::autograd::{clear_graph, no_grad, Tensor};
use aprender::nn::optim::{AdamW, Optimizer};
use aprender::nn::{Linear, Module};
use aprender_forecast::dates::days_from_civil;
use aprender_forecast::np::{
    auto_batch, auto_epochs, one_cycle_lr, rows_for, weighted_huber, NpData, NpModel, Rng,
};

const END_W: f64 = 2.0;
const HUBER_BETA: f32 = 0.3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rule {
    Linear,
    Zero,
    CarryForward,
    Garbage,
}
impl Rule {
    fn name(self) -> &'static str {
        match self {
            Rule::Linear => "linear interp",
            Rule::Zero => "zero fill",
            Rule::CarryForward => "carry forward",
            Rule::Garbage => "garbage (1e3)",
        }
    }
}

fn gather(rows: &[f32], width: usize, idx: &[usize]) -> Vec<f32> {
    let mut v = Vec::with_capacity(idx.len() * width);
    for &i in idx {
        v.extend_from_slice(&rows[i * width..(i + 1) * width]);
    }
    v
}

/// A daily series with deliberate BLOCK gaps, plus two drivers defined on observed days.
fn series() -> (Vec<i64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let start = days_from_civil(2019, 1, 1);
    let mut rng = Rng::new(11);
    let (mut ds, mut y, mut promo, mut price) = (vec![], vec![], vec![], vec![]);
    for i in 0..900i64 {
        // three 15-day blackouts plus a scattered 8% dropout
        let blackout = (200..215).contains(&i) || (450..465).contains(&i) || (700..715).contains(&i);
        if blackout || rng.uniform() < 0.08 {
            continue;
        }
        let day = start + i;
        let t = i as f64;
        let pm = f64::from(u8::from(i % 11 == 0));
        let pr = 50.0 + 4.0 * (2.0 * std::f64::consts::PI * t / 90.0).sin();
        let v = 100.0 + 0.03 * t
            + 6.0 * (2.0 * std::f64::consts::PI * t / 365.25).sin()
            + 3.0 * (2.0 * std::f64::consts::PI * t / 7.0).sin()
            + 9.0 * pm
            - 0.6 * (pr - 50.0)
            + 0.4 * rng.normal();
        ds.push(day);
        y.push(v);
        promo.push(pm);
        price.push(pr);
    }
    (ds, y, promo, price)
}

/// Build the regressor matrix over the WHOLE daily grid from values defined on observed
/// days, filling the imputed days by `rule`. Standardised on observed train rows (ddof=1,
/// the Prophet rule measured in spike 011); binary {0,1} is left alone.
fn reg_grid(d: &NpData, obs_days: &[i64], vals: &[Vec<f64>], rule: Rule) -> (Vec<f32>, usize) {
    let r = vals.len();
    let ng = d.grid_days.len();
    let mut out = vec![0.0f64; ng * r];
    for (j, col) in vals.iter().enumerate() {
        // place observed values
        let mut on_grid = vec![f64::NAN; ng];
        for (k, &day) in obs_days.iter().enumerate() {
            on_grid[(day - d.t0) as usize] = col[k];
        }
        // fill the holes
        let mut i = 0;
        while i < ng {
            if on_grid[i].is_nan() {
                let a = i.saturating_sub(1);
                let mut b = i;
                while b < ng && on_grid[b].is_nan() {
                    b += 1;
                }
                for k in i..b {
                    on_grid[k] = match rule {
                        Rule::Zero => 0.0,
                        Rule::Garbage => 1.0e3,
                        Rule::CarryForward => on_grid[a],
                        Rule::Linear => {
                            if b >= ng {
                                on_grid[a]
                            } else {
                                let f = (k - a) as f64 / (b - a) as f64;
                                on_grid[a] + f * (on_grid[b] - on_grid[a])
                            }
                        }
                    };
                }
                i = b;
            }
            i += 1;
        }
        // standardise on observed TRAIN values
        let obs: Vec<f64> = obs_days
            .iter()
            .enumerate()
            .filter(|(_, &day)| ((day - d.t0) as usize) < d.n_train_grid)
            .map(|(k, _)| col[k])
            .collect();
        let binary = obs.iter().all(|v| *v == 0.0 || *v == 1.0);
        let (mu, sd) = if binary {
            (0.0, 1.0)
        } else {
            let n = obs.len() as f64;
            let mu = obs.iter().sum::<f64>() / n;
            let var = obs.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / (n - 1.0);
            (mu, var.sqrt())
        };
        for i in 0..ng {
            out[i * r + j] = (on_grid[i] - mu) / sd;
        }
    }
    (out.iter().map(|v| *v as f32).collect(), r)
}

struct Out {
    pred: Vec<f64>,
    weights: Vec<f32>,
}

fn train(d: &NpData, reg: &[f32], r: usize, n_lags: usize, seed: u64) -> Out {
    let mut rng = Rng::new(seed);
    let mut model = NpModel::new(d, n_lags, &[], &mut rng);
    let mut block = Linear::without_bias(r, 1);
    let std = (1.0 / r as f64).sqrt();
    block.set_weight(
        Tensor::from_vec((0..r).map(|_| (rng.normal() * std) as f32).collect(), &[1, r]).requires_grad(),
    );

    let n_grid = d.n_train_grid;
    let y_norm: Vec<f32> = d.grid_y[..n_grid].iter().map(|v| d.norm(*v)).collect();
    let rows = rows_for(d, &d.grid_days[..n_grid], &y_norm, END_W);
    // EXACTLY np::train's rule (np.rs:597): observed rows when lag-free, the whole grid
    // from `l` onward when lagged.
    let samples: Vec<usize> = if n_lags == 0 {
        (0..n_grid).filter(|&i| d.grid_observed[i]).collect()
    } else {
        (n_lags..n_grid).collect()
    };
    let n = samples.len();
    let batch = auto_batch(n);
    let epochs = auto_epochs(n).min(120);
    let total = epochs * n.div_ceil(batch);

    let mut opt = {
        let mut p = model.parameters_mut();
        p.extend(block.parameters_mut());
        AdamW::new(p, 0.03).weight_decay(0.0)
    };
    let mut order = samples.clone();
    let mut step = 0usize;
    for _ in 0..epochs {
        rng.shuffle(&mut order);
        for chunk in order.chunks(batch) {
            opt.set_lr(one_cycle_lr(step as f64 / total as f64, 0.03) as f32);
            let b = chunk.len();
            let xt = Tensor::from_vec(gather(&rows.tr, rows.td, chunk), &[b, rows.td]);
            let xs = Tensor::from_vec(gather(&rows.se, rows.sd, chunk), &[b, rows.sd]);
            let xr = Tensor::from_vec(gather(reg, r, chunk), &[b, r]);
            let yt = Tensor::from_vec(chunk.iter().map(|&i| rows.y[i]).collect(), &[b, 1]);
            let wt = Tensor::from_vec(chunk.iter().map(|&i| rows.w[i]).collect(), &[b, 1]);
            let pred = if n_lags == 0 {
                model.forward(&xt, &xs, None)
            } else {
                let li: Vec<usize> = chunk.iter().flat_map(|&i| i - n_lags..i).collect();
                let raw = Tensor::from_vec(li.iter().map(|&j| rows.y[j]).collect(), &[b, n_lags]);
                let sl = Tensor::from_vec(gather(&rows.se, rows.sd, &li), &[b * n_lags, rows.sd]);
                let tf = Tensor::from_vec(gather(&rows.tr, rows.td, &li), &[b * n_lags, rows.td]);
                let tl = no_grad(|| model.trend.forward(&tf).detach());
                let tl = Tensor::from_vec(tl.data().to_vec(), &[b, n_lags]);
                model.forward(&xt, &xs, Some((&raw, &sl, &tl)))
            }
            .add(&block.forward(&xr));
            let loss = weighted_huber(&pred, &yt, &wt, HUBER_BETA);
            loss.backward();
            {
                let mut p = model.parameters_mut();
                p.extend(block.parameters_mut());
                opt.step_with_params(&mut p);
            }
            opt.zero_grad();
            clear_graph();
            step += 1;
        }
    }

    // trend+season+regressors over the observed train rows (comparable across rules)
    let idx: Vec<usize> = (0..n_grid).filter(|&i| d.grid_observed[i]).collect();
    let all = rows_for(d, &d.grid_days[..n_grid], &y_norm, END_W);
    let b = idx.len();
    let xt = Tensor::from_vec(gather(&all.tr, all.td, &idx), &[b, all.td]);
    let xs = Tensor::from_vec(gather(&all.se, all.sd, &idx), &[b, all.sd]);
    let xr = Tensor::from_vec(gather(reg, r, &idx), &[b, r]);
    let o = no_grad(|| model.forward(&xt, &xs, None).add(&block.forward(&xr)));
    let pred: Vec<f64> = o.data().iter().map(|v| d.denorm(*v)).collect();
    clear_graph();
    Out { pred, weights: block.weight().data().to_vec() }
}

fn main() {
    let (ds, y, promo, price) = series();
    let n = ds.len();
    let d = NpData::new(&ds, &y, n, 10, 0.9);
    let ng = d.grid_days.len();
    let imputed = ng - n;
    println!("# Spike 014 — regressor values on NeuralProphet's imputed grid days\n");
    println!("Re-scoped: the door refuses `freq != \"D\"` for neuralprophet (`forecast.rs:421`),");
    println!("so there is no weekly/month-start mapping. The daily-gap case is what remains.\n");
    println!("Daily series: **{n} observed** rows over a **{ng}-day** grid -> **{imputed} imputed days**");
    println!("({:.1}% of the grid), including three 15-day blackouts. Drivers: `promo` (binary,",
             100.0 * imputed as f64 / ng as f64);
    println!("true effect +9.0) and `price` (continuous, true effect -0.6 per unit).\n");

    let vals = vec![promo, price];
    println!("## Does the imputed-day value get read at all?\n");
    println!("Trains with the imputed days filled by each rule and compares against `linear interp`.");
    println!("`garbage (1e3)` is the falsification probe: if it changes nothing, the value is unread.\n");
    println!("| n_lags | rule | pred bit-identical to linear | max abs d pred | promo w | price w |");
    println!("|---|---|---|---|---|---|");
    for n_lags in [0usize, 7] {
        let (base_reg, r) = reg_grid(&d, &ds, &vals, Rule::Linear);
        let base = train(&d, &base_reg, r, n_lags, 42);
        for rule in [Rule::Linear, Rule::Zero, Rule::CarryForward, Rule::Garbage] {
            let (reg, r) = reg_grid(&d, &ds, &vals, rule);
            let o = train(&d, &reg, r, n_lags, 42);
            let same = o.pred.iter().zip(&base.pred).all(|(a, b)| a.to_bits() == b.to_bits());
            let dmax = o.pred.iter().zip(&base.pred).fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
            println!("| {n_lags} | {} | {} | {dmax:.3e} | {:+.4} | {:+.4} |",
                     rule.name(), if same { "**yes**" } else { "no" }, o.weights[0], o.weights[1]);
        }
    }
    println!("\nScale {:.2}; a `promo` weight w denormalises to w x scale.", d.scale);
}
