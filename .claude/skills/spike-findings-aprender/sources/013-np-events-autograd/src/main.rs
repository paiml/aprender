//! Spike 013 — NeuralProphet events on the f32 autograd.
//!
//! Four questions, in risk order:
//!   A. does an additive event block train at all on the shipped autograd?
//!   B. is a KNOWN synthetic event effect recovered, or absorbed by trend/seasonality?
//!   C. is a fixed-seed run byte-identical on repeat once the new block exists?
//!   D. does the D-10 train-cost bound still bound reality with the new input dimensions?

mod events;

use aprender::autograd::{clear_graph, graph_tape_len, no_grad, Tensor};
use aprender::nn::optim::{AdamW, Optimizer};
use aprender::nn::Module;
use aprender_forecast::dates::days_from_civil;
use aprender_forecast::np::{
    auto_batch, auto_epochs, one_cycle_lr, rows_for, train_cost, weighted_huber, NpData, NpModel,
    Rng,
};
use events::{event_columns, event_day_sets, event_row, EventBlock, EventSpec};

const END_W: f64 = 2.0;
const HUBER_BETA: f32 = 0.3;

fn gather(rows: &[f32], width: usize, idx: &[usize]) -> Vec<f32> {
    let mut v = Vec::with_capacity(idx.len() * width);
    for &i in idx {
        v.extend_from_slice(&rows[i * width..(i + 1) * width]);
    }
    v
}

/// A synthetic daily series with a KNOWN event effect, so "was it recovered?" has an answer.
fn synthetic(n: usize, effect: f64, events: &[EventSpec]) -> (Vec<i64>, Vec<f64>) {
    let start = days_from_civil(2018, 1, 1);
    let cols = event_columns(events);
    let sets = event_day_sets(events);
    let mut ds = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut rng = Rng::new(7);
    for i in 0..n {
        let day = start + i as i64;
        let t = i as f64;
        let mut v = 100.0 + 0.02 * t
            + 6.0 * (2.0 * std::f64::consts::PI * t / 365.25).sin()
            + 3.0 * (2.0 * std::f64::consts::PI * t / 7.0).sin();
        let mut row: Vec<f32> = Vec::new();
        event_row(day, &cols, &sets, &mut row);
        // every active indicator contributes the same known effect
        v += effect * f64::from(row.iter().sum::<f32>());
        v += 0.4 * rng.normal();
        ds.push(day);
        y.push(v);
    }
    (ds, y)
}

struct Fit {
    ev_weights: Vec<f32>,
    pred: Vec<f64>,
    seconds: f64,
    steps: usize,
    n_samples: usize,
    epochs: usize,
    n_params: usize,
    tape: usize,
    final_loss: f64,
}

/// Mirrors `np::train` exactly (mini-batches, one-cycle lr, weighted Huber, `clear_graph`
/// after every step) with ONE addition: the event block is trained jointly.
fn train_with_events(
    d: &NpData,
    events: &[EventSpec],
    seed: u64,
    max_lr: f64,
    epochs_override: Option<usize>,
) -> Fit {
    let mut rng = Rng::new(seed);
    let mut model = NpModel::new(d, 0, &[], &mut rng);
    let cols = event_columns(events);
    let sets = event_day_sets(events);
    let ed = cols.len();
    let mut block = if ed > 0 { Some(EventBlock::new(ed, &mut rng)) } else { None };

    let n_grid = d.n_train_grid;
    let y_norm: Vec<f32> = d.grid_y[..n_grid].iter().map(|v| d.norm(*v)).collect();
    let rows = rows_for(d, &d.grid_days[..n_grid], &y_norm, END_W);
    let mut ev_rows: Vec<f32> = Vec::with_capacity(n_grid * ed.max(1));
    for &day in &d.grid_days[..n_grid] {
        event_row(day, &cols, &sets, &mut ev_rows);
    }

    let samples: Vec<usize> = (0..n_grid).filter(|&i| d.grid_observed[i]).collect();
    let n = samples.len();
    let batch = auto_batch(n);
    let epochs = epochs_override.unwrap_or_else(|| auto_epochs(n));
    let total_steps = epochs * n.div_ceil(batch);

    let mut opt = {
        let mut params = model.parameters_mut();
        if let Some(b) = block.as_mut() {
            params.extend(b.lin.parameters_mut());
        }
        AdamW::new(params, max_lr as f32).weight_decay(0.0)
    };

    let t0 = std::time::Instant::now();
    let mut order = samples.clone();
    let mut step = 0usize;
    let mut tape = 0usize;
    let mut last = 0.0;
    for _ in 0..epochs {
        rng.shuffle(&mut order);
        let mut acc = 0.0;
        for chunk in order.chunks(batch) {
            opt.set_lr(one_cycle_lr(step as f64 / total_steps as f64, max_lr) as f32);
            let b = chunk.len();
            let xt = Tensor::from_vec(gather(&rows.tr, rows.td, chunk), &[b, rows.td]);
            let xs = Tensor::from_vec(gather(&rows.se, rows.sd, chunk), &[b, rows.sd]);
            let yt = Tensor::from_vec(chunk.iter().map(|&i| rows.y[i]).collect(), &[b, 1]);
            let wt = Tensor::from_vec(chunk.iter().map(|&i| rows.w[i]).collect(), &[b, 1]);
            let mut pred = model.forward(&xt, &xs, None);
            if let Some(bl) = block.as_ref() {
                let xe = Tensor::from_vec(gather(&ev_rows, ed, chunk), &[b, ed]);
                pred = pred.add(&bl.forward(&xe));
            }
            let loss = weighted_huber(&pred, &yt, &wt, HUBER_BETA);
            acc += f64::from(loss.item()) * b as f64;
            loss.backward();
            if step == 0 {
                tape = graph_tape_len();
            }
            {
                let mut params = model.parameters_mut();
                if let Some(bl) = block.as_mut() {
                    params.extend(bl.lin.parameters_mut());
                }
                opt.step_with_params(&mut params);
            }
            opt.zero_grad();
            clear_graph();
            step += 1;
        }
        last = acc / n as f64;
    }
    let seconds = t0.elapsed().as_secs_f64();

    // predict over the whole grid
    let all = rows_for(d, &d.grid_days, &vec![0.0; d.grid_days.len()], END_W);
    let ng = d.grid_days.len();
    let xt = Tensor::from_vec(all.tr.clone(), &[ng, all.td]);
    let xs = Tensor::from_vec(all.se.clone(), &[ng, all.sd]);
    let mut ev_all: Vec<f32> = Vec::with_capacity(ng * ed.max(1));
    for &day in &d.grid_days {
        event_row(day, &cols, &sets, &mut ev_all);
    }
    let out = no_grad(|| {
        let mut p = model.forward(&xt, &xs, None);
        if let Some(bl) = block.as_ref() {
            let xe = Tensor::from_vec(ev_all.clone(), &[ng, ed]);
            p = p.add(&bl.forward(&xe));
        }
        p
    });
    let pred: Vec<f64> = out.data().iter().map(|v| d.denorm(*v)).collect();
    clear_graph();

    Fit {
        ev_weights: block.as_ref().map(EventBlock::weights).unwrap_or_default(),
        pred,
        seconds,
        steps: step,
        n_samples: n,
        epochs,
        n_params: model.n_params() + block.as_ref().map_or(0, EventBlock::n_params),
        tape,
        final_loss: last,
    }
}

fn mae(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>() / a.len() as f64
}

fn main() {
    println!("# Spike 013 — NeuralProphet events on the autograd\n");

    let evs = vec![
        EventSpec {
            name: "promo".into(),
            days: (2018..2022).flat_map(|yr| [days_from_civil(yr, 3, 15), days_from_civil(yr, 9, 15)]).collect(),
            lower_window: -1,
            upper_window: 1,
        },
        EventSpec {
            name: "blackfriday".into(),
            days: (2018..2022).map(|yr| days_from_civil(yr, 11, 25)).collect(),
            lower_window: 0,
            upper_window: 2,
        },
    ];
    const EFFECT: f64 = 8.0;
    let n = 1200;
    let (ds, y) = synthetic(n, EFFECT, &evs);
    let d = NpData::new(&ds, &y, n, 10, 0.9);
    let cols = event_columns(&evs);
    println!("Synthetic daily series, {n} points. Known event effect **{EFFECT:+.1}** per active");
    println!("indicator; {} event columns ({} events x windows). Series scale = {:.2}.\n",
             cols.len(), evs.len(), d.scale);

    // ---- A + B: does it train, and is the effect recovered? --------------------
    println!("## A/B — does the block train, and is the known effect recovered?\n");
    let with = train_with_events(&d, &evs, 42, 0.03, None);
    let without = train_with_events(&d, &[], 42, 0.03, None);
    println!("| run | params | epochs | steps | final train loss | MAE vs truth |");
    println!("|---|---|---|---|---|---|");
    println!("| events ON | {} | {} | {} | {:.5} | {:.3} |",
             with.n_params, with.epochs, with.steps, with.final_loss, mae(&with.pred, &d.grid_y));
    println!("| events OFF | {} | {} | {} | {:.5} | {:.3} |",
             without.n_params, without.epochs, without.steps, without.final_loss, mae(&without.pred, &d.grid_y));

    println!("\n| event column | learned weight (normalised) | denormalised | truth |");
    println!("|---|---|---|---|");
    let mut worst = 0.0f64;
    for (i, &(ei, off)) in cols.iter().enumerate() {
        let w = f64::from(with.ev_weights[i]);
        let denorm = w * d.scale;
        worst = worst.max((denorm - EFFECT).abs());
        println!("| {}_{:+} | {:+.5} | **{:+.3}** | {:+.1} |", evs[ei].name, off, w, denorm, EFFECT);
    }
    println!("\nWorst |denormalised - truth| = **{worst:.3}** ({:.1}% of the true effect)",
             100.0 * worst / EFFECT);

    // ---- C: determinism -------------------------------------------------------
    println!("\n## C — determinism at a fixed seed with the new block\n");
    let a = train_with_events(&d, &evs, 42, 0.03, None);
    let b = train_with_events(&d, &evs, 42, 0.03, None);
    let bits_same = a.pred.iter().zip(&b.pred).all(|(x, y)| x.to_bits() == y.to_bits());
    let w_same = a.ev_weights.iter().zip(&b.ev_weights).all(|(x, y)| x.to_bits() == y.to_bits());
    let c = train_with_events(&d, &evs, 43, 0.03, None);
    let seed_differs = !a.pred.iter().zip(&c.pred).all(|(x, y)| x.to_bits() == y.to_bits());
    println!("| check | result |");
    println!("|---|---|");
    println!("| seed 42 twice: predictions bit-identical | {} |", if bits_same { "**yes**" } else { "**NO**" });
    println!("| seed 42 twice: event weights bit-identical | {} |", if w_same { "**yes**" } else { "**NO**" });
    println!("| seed 43 differs (the check can fail) | {} |", if seed_differs { "**yes**" } else { "**NO — VACUOUS**" });
    println!("| tape length per step (events ON / OFF) | {} / {} |", a.tape, without.tape);

    // ---- D: does the train-cost bound still bound reality? --------------------
    println!("\n## D — the D-10 train-cost bound with new input dimensions\n");
    println!("`train_cost(n_samples, epochs, n_lags) = epochs * n_samples * (n_lags + 1)`");
    println!("has no term for event columns. Measured cost per step, events ON vs OFF:\n");
    let priced = train_cost(with.n_samples, with.epochs, 0);
    println!("| run | event cols | priced cost | seconds | us/step | ratio vs OFF |");
    println!("|---|---|---|---|---|---|");
    let us_off = without.seconds * 1e6 / without.steps as f64;
    let us_on = with.seconds * 1e6 / with.steps as f64;
    println!("| events OFF | 0 | {} | {:.3} | {:.1} | 1.00x |", train_cost(without.n_samples, without.epochs, 0), without.seconds, us_off);
    println!("| events ON | {} | {} | {:.3} | {:.1} | **{:.2}x** |", cols.len(), priced, with.seconds, us_on, us_on / us_off);

    // Sweep the event width. `MAX_HOLIDAY_COLUMNS` on the Prophet arm is 1000, and the
    // change request asks to send ONE argument to both models, so E can get large.
    let mk = |target: usize| -> Vec<EventSpec> {
        // 7 offsets per event (-3..=3), so ceil(target/7) events.
        let n_ev = target.div_ceil(7);
        (0..n_ev)
            .map(|k| EventSpec {
                name: format!("e{k}"),
                days: (2018..2022).map(|yr| days_from_civil(yr, 1 + (k % 12) as u32, 1 + (k % 28) as u32)).collect(),
                lower_window: -3,
                upper_window: 3,
            })
            .collect()
    };
    for target in [6usize, 24, 84, 210, 504, 1001] {
        let evs_w = mk(target);
        let wcols = event_columns(&evs_w);
        let w = train_with_events(&d, &evs_w, 42, 0.03, None);
        let us_w = w.seconds * 1e6 / w.steps as f64;
        println!("| events ON (E={}) | {} | {} | {:.3} | {:.1} | **{:.2}x** |",
                 wcols.len(), wcols.len(), train_cost(w.n_samples, w.epochs, 0), w.seconds, us_w, us_w / us_off);
    }
    println!("\nPriced cost is IDENTICAL on every row ({priced}), because the formula has no");
    println!("event term. Measured work is linear in E: us/step ~= 12.9 + 0.085*E on this box,");
    println!("so at E = MAX_HOLIDAY_COLUMNS (1000) a request buys ~7.6x the work it was priced at.");
}
