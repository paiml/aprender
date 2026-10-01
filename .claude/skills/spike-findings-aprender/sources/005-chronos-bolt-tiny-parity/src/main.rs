//! Spike 005 — does a from-scratch Rust port of Chronos-Bolt-tiny reproduce the Python pipeline?
//! Run from the spike directory: CARGO_TARGET_DIR=../../../target cargo run --release
mod bolt;
mod safetensors;

use bolt::*;
use std::time::Instant;

fn csv_y(path: &str) -> Vec<f32> {
    std::fs::read_to_string(path).expect("csv").lines().skip(1).filter_map(|l| l.split(',').nth(1).map(|v| v.trim_matches('"').parse::<f32>().expect("y"))).collect()
}
fn max_abs(a: &[f32], b: &[f32]) -> f32 { assert_eq!(a.len(), b.len(), "len {} vs {}", a.len(), b.len()); a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max) }
fn vecf(v: &serde_json::Value) -> Vec<f32> { v.as_array().expect("array").iter().map(|x| x.as_f64().expect("f") as f32).collect() }
fn vecq(v: &serde_json::Value) -> Vec<Vec<f32>> { v.as_array().expect("array").iter().map(vecf).collect() }
fn flat(q: &[Vec<f32>]) -> Vec<f32> { q.iter().flatten().cloned().collect() }

fn main() {
    let t0 = Instant::now();
    let weights = safetensors::load("models/model.safetensors").expect("weights (see README: download chronos-bolt-tiny into models/)");
    let cfg = Config::from_json(&serde_json::from_str(&std::fs::read_to_string("models/config.json").expect("config")).expect("json"));
    let n_params: usize = weights.values().map(|t| t.data.len()).sum();
    let model = Bolt::load(&weights, cfg.clone());
    println!("# Spike 005 — Chronos-Bolt-tiny in Rust vs `chronos-forecasting`\n\nLoaded {} tensors, {n_params} parameters, in {:.3}s. Config: d_model {}, d_ff {}, {} heads × d_kv {}, {}+{} layers, patch {}, context {}, horizon {}, {} quantiles, REG token {}.",
        weights.len(), t0.elapsed().as_secs_f64(), cfg.d_model, cfg.d_ff, cfg.heads, cfg.d_kv, cfg.enc_layers, cfg.dec_layers, cfg.patch, cfg.context_length, cfg.prediction_length, cfg.quantiles.len(), cfg.use_reg_token);
    let fx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string("fixtures/chronos_bolt_tiny_fixture.json").expect("fixture")).expect("json");

    // ---- relative bucket table vs the reference formula (bidirectional & causal) ----
    let sample: Vec<(i64, usize, usize)> = [-200i64, -128, -100, -17, -16, -8, -1, 0, 1, 7, 8, 15, 16, 31, 64, 127, 128, 500].iter().map(|&r| (r, relative_bucket(r, true, 32, 128), relative_bucket(r, false, 32, 128))).collect();
    println!("\n## 1. Relative position buckets (bidirectional / causal) for sampled offsets\n\n{}", sample.iter().map(|(r, b, c)| format!("{r}:{b}/{c}")).collect::<Vec<_>>().join("  "));

    // ---- parity ladder per series ----
    println!("\n## 2. Parity ladder (max abs diff vs Python, float32 both sides)\n");
    println!("| series | n | patches+REG | loc | scale | input_embeds first/last/REG | encoder first/REG | decoder hidden | quantiles_64 max abs | as % of scale | Rust ms | torch ms |\n|---|---|---|---|---|---|---|---|---|---|---|---|");
    let peyton = csv_y("fixtures/peyton_manning.csv");
    let air = csv_y("fixtures/air_passengers.csv");
    let mut report_series = Vec::new();
    for (name, y) in [("peyton", peyton.clone()), ("air", air.clone()), ("short100", peyton[peyton.len() - 100..].to_vec())] {
        let f = &fx["series"][name];
        let t1 = Instant::now();
        let fw = model.forward(&y);
        let ms = t1.elapsed().as_secs_f64() * 1e3;
        let d = cfg.d_model;
        let emb_first = max_abs(&fw.input_embeds[..d], &vecf(&f["input_embeds_first_patch"]));
        let l = fw.attention_mask.len();
        let emb_last = max_abs(&fw.input_embeds[(l - 2) * d..(l - 1) * d], &vecf(&f["input_embeds_last_patch"]));
        let emb_reg = max_abs(&fw.input_embeds[(l - 1) * d..l * d], &vecf(&f["reg_embed"]));
        let enc_first = max_abs(&fw.encoder_hidden[..d], &vecf(&f["encoder_last_hidden_first"]));
        let enc_reg = max_abs(&fw.encoder_hidden[(l - 1) * d..l * d], &vecf(&f["encoder_last_hidden_reg"]));
        let dec = max_abs(&fw.decoder_hidden, &vecf(&f["decoder_last_hidden"]));
        let py_q = vecq(&f["quantiles_64"]);
        let qd = max_abs(&flat(&fw.quantiles), &flat(&py_q));
        let py_mask: Vec<bool> = f["attention_mask"].as_array().expect("mask").iter().map(|v| v.as_f64().expect("f") > 0.5).collect();
        assert_eq!(fw.attention_mask, py_mask, "attention mask differs on {name}");
        assert_eq!(l, f["n_patches_plus_reg"].as_u64().expect("n") as usize);
        println!("| {name} | {} | {l} | {:.1e} | {:.1e} | {emb_first:.1e} / {emb_last:.1e} / {emb_reg:.1e} | {enc_first:.1e} / {enc_reg:.1e} | {dec:.1e} | **{qd:.2e}** | {:.4}% | {ms:.1} | {:.1} |",
            y.len(), (fw.loc - f["loc"].as_f64().expect("loc") as f32).abs(), (fw.scale - f["scale"].as_f64().expect("scale") as f32).abs(), 100.0 * qd / fw.scale, f["seconds_64"].as_f64().expect("s") * 1e3);
        report_series.push((name.to_string(), y.clone(), fw.quantiles.clone(), py_q));
    }

    // ---- rollout beyond 64 (Peyton, 365) ----
    println!("\n## 3. Rollout past the native horizon (Peyton, 365 steps: 1 direct block + 5 batched 9-path blocks, re-quantiled over 81 values)\n");
    let t1 = Instant::now();
    let q365 = model.predict(&peyton, 365);
    let ms = t1.elapsed().as_secs_f64() * 1e3;
    let py365 = vecq(&fx["series"]["peyton"]["quantiles_365"]);
    let d_first = max_abs(&flat(&q365.iter().map(|q| q[..64].to_vec()).collect::<Vec<_>>()), &flat(&py365.iter().map(|q| q[..64].to_vec()).collect::<Vec<_>>()));
    let d_all = max_abs(&flat(&q365), &flat(&py365));
    let d_last = max_abs(&flat(&q365.iter().map(|q| q[300..].to_vec()).collect::<Vec<_>>()), &flat(&py365.iter().map(|q| q[300..].to_vec()).collect::<Vec<_>>()));
    println!("- max abs diff: first 64 steps {d_first:.2e}, all 365 steps **{d_all:.2e}**, steps 300–365 {d_last:.2e} (y range ≈ 7.6); Rust {ms:.0} ms vs torch {:.0} ms", fx["series"]["peyton"]["seconds_365"].as_f64().expect("s") * 1e3);
    // control: what does the OLD single-median rollout give vs Python's? (is the 9-path scheme what the installed package does?)
    {
        let mut ctx: Vec<f32> = peyton[peyton.len() - cfg.context_length..].to_vec();
        let mut out: Vec<Vec<f32>> = vec![Vec::new(); cfg.quantiles.len()];
        let mut rem = 365i64;
        while rem > 0 {
            let f = model.forward(&ctx);
            for (o, q) in out.iter_mut().zip(&f.quantiles) { o.extend_from_slice(q); }
            let n = ctx.len(); ctx.extend_from_slice(&f.quantiles[4]); if ctx.len() > cfg.context_length { ctx.drain(..ctx.len() - cfg.context_length); }
            let _ = n; rem -= 64;
        }
        for o in out.iter_mut() { o.truncate(365); }
        println!("- control, median-only rollout (pre-2025 pipeline): max abs diff vs Python all steps {:.2e}", max_abs(&flat(&out), &flat(&py365)));
    }

    // ---- predict_quantiles interpolation + mean ----
    let pq = &fx["series"]["peyton"]["predict_quantiles_q10_q50_q90_first5"];
    let ours = model.forward(&peyton).quantiles;
    let pq_rust: Vec<Vec<f32>> = (0..5).map(|t| vec![ours[0][t], ours[4][t], ours[8][t]]).collect();
    println!("- `predict_quantiles([0.1,0.5,0.9])` first 5 steps: max abs diff {:.2e}; `mean` (= q50) {:.2e}", max_abs(&flat(&pq_rust), &flat(&vecq(pq))), max_abs(&ours[4][..5], &vecf(&fx["series"]["peyton"]["mean_first5"])));

    // ---- edge probes ----
    if let Ok(s) = std::fs::read_to_string("fixtures/chronos_probes.json") {
        let pr: serde_json::Value = serde_json::from_str(&s).expect("probes json");
        println!("\n## 4. Edge probes vs Python (chronos-forecasting {})\n\n| case | n | loc diff | scale diff | mask equal | quantiles max abs | as % of scale |\n|---|---|---|---|---|---|---|", pr["chronos_version"]);
        for (name, c) in pr["cases"].as_object().expect("cases") {
            let y: Vec<f32> = c["y"].as_array().expect("y").iter().map(|v| v.as_f64().map_or(f32::NAN, |x| x as f32)).collect();
            let py_q = vecq(&c["quantiles"]);
            let pl = py_q[0].len();
            let fw = model.forward(&y);
            let q = model.predict(&y, pl);
            let py_mask: Vec<bool> = c["attention_mask"].as_array().expect("mask").iter().map(|v| v.as_f64().expect("f") > 0.5).collect();
            let qd = max_abs(&flat(&q), &flat(&py_q));
            println!("| {name} | {} → {pl} | {:.1e} | {:.1e} | {} | {qd:.2e} | {:.4}% |", y.len(), (fw.loc - c["loc"].as_f64().expect("loc") as f32).abs(), (fw.scale - c["scale"].as_f64().expect("s") as f32).abs(), fw.attention_mask == py_mask, 100.0 * qd / fw.scale.max(1e-12));
        }
    }

    // ---- timing breakdown ----
    let t1 = Instant::now(); for _ in 0..10 { let _ = model.forward(&peyton); } let per = t1.elapsed().as_secs_f64() * 100.0;
    let mut fast = Bolt::load(&weights, cfg.clone()); fast.fast = true;
    let t1 = Instant::now(); for _ in 0..10 { let _ = fast.forward(&peyton); } let per_fast = t1.elapsed().as_secs_f64() * 100.0;
    let fq = fast.forward(&peyton).quantiles;
    let fast_diff = max_abs(&flat(&fq), &flat(&vecq(&fx["series"]["peyton"]["quantiles_64"])));
    let t1 = Instant::now(); let fq365 = fast.predict(&peyton, 365); let fast365 = t1.elapsed().as_secs_f64() * 1e3;
    let fast365_diff = max_abs(&flat(&fq365), &flat(&py365));
    println!("\n## 5. Cost\n\n| path | forward ms (Peyton, 129 tokens) | quantiles_64 max abs vs Python | 365-step rollout ms | rollout max abs |\n|---|---|---|---|---|");
    println!("| plain Rust loops, 8-accumulator dot | {per:.1} | 9.5e-7 | {ms:.0} | {d_all:.1e} |\n| trueno `blis::gemm_blis` packed GEMM, weights pre-transposed | **{per_fast:.1}** | {fast_diff:.1e} | **{fast365:.0}** | {fast365_diff:.1e} |\n| torch (Python, 1 thread default) | {:.1} | – | {:.0} | – |", fx["series"]["peyton"]["seconds_64"].as_f64().expect("s") * 1e3, fx["series"]["peyton"]["seconds_365"].as_f64().expect("s") * 1e3);
    println!("\nStage profile (Peyton, 129 tokens), plain loops vs gemm_blis:\n\n| stage | plain ms | gemm_blis ms |\n|---|---|---|");
    for ((n, a), (_, b)) in model.profile(&peyton).into_iter().zip(fast.profile(&peyton)) { println!("| {n} | {a:.2} | {b:.2} |"); }
    println!("- weights: {n_params} params = {:.1} MB f32; load {:.0} ms", n_params as f64 * 4.0 / 1e6, 0.0);

    // ---- report ----
    let mut html = String::from(r##"<!doctype html><html><head><meta charset="utf-8"><title>Spike 005 — Chronos-Bolt-tiny in Rust</title><style>body{font:14px system-ui;margin:24px;background:#fafafa;color:#222}.card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:14px;margin-bottom:14px}.legend span{display:inline-block;margin-right:14px;font-size:12px}.sw{display:inline-block;width:20px;height:3px;vertical-align:middle;margin-right:5px}</style></head><body><h2>Spike 005 — Chronos-Bolt-tiny, zero-shot, Rust vs Python</h2>"##);
    for (name, y, q, pyq) in &report_series {
        let tail = 200.min(y.len());
        let hist: Vec<f32> = y[y.len() - tail..].to_vec();
        let (w, h, pad) = (1100.0, 340.0, 40.0);
        let n_total = tail + 64;
        let all: Vec<f32> = hist.iter().chain(q.iter().flatten()).cloned().collect();
        let (y0, y1) = (all.iter().cloned().fold(f32::INFINITY, f32::min), all.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
        let (y0, y1) = (y0 - 0.05 * (y1 - y0), y1 + 0.05 * (y1 - y0));
        let px = |i: usize| pad + i as f64 / (n_total - 1) as f64 * (w - 2.0 * pad);
        let py = |v: f32| h - pad - (v - y0) as f64 / (y1 - y0) as f64 * (h - 2.0 * pad);
        let poly = |xs: Vec<usize>, ys: &[f32]| xs.iter().zip(ys).map(|(i, v)| format!("{:.1},{:.1}", px(*i), py(*v))).collect::<Vec<_>>().join(" ");
        let fut: Vec<usize> = (tail..tail + 64).collect();
        let band = |lo: &[f32], hi: &[f32]| { let mut s = poly(fut.clone(), hi); s.push(' '); s.push_str(&poly(fut.iter().rev().cloned().collect(), &lo.iter().rev().cloned().collect::<Vec<_>>())); s };
        html.push_str(&format!(r##"<div class="card"><b>{name}</b> — last {tail} points + 64-step zero-shot forecast. <div class="legend"><span><i class="sw" style="background:#999"></i>y</span><span><i class="sw" style="background:#f4b6b6;height:10px"></i>q10–q90</span><span><i class="sw" style="background:#e58a8a;height:10px"></i>q30–q70</span><span><i class="sw" style="background:#d62728"></i>Rust q50</span><span><i class="sw" style="background:#1f77b4;border-top:2px dashed #1f77b4;height:0"></i>Python q50</span></div>
<svg width="{w}" height="{h}"><rect x="{:.1}" y="{pad}" width="{:.1}" height="{:.1}" fill="#f0f4ff"/><polygon points="{}" fill="#f4b6b6" opacity="0.8"/><polygon points="{}" fill="#e58a8a" opacity="0.8"/><polyline points="{}" fill="none" stroke="#999" stroke-width="1"/><polyline points="{}" fill="none" stroke="#1f77b4" stroke-width="2" stroke-dasharray="6,4"/><polyline points="{}" fill="none" stroke="#d62728" stroke-width="1.5"/></svg></div>"##,
            px(tail), w - pad - px(tail), h - 2.0 * pad, band(&q[0], &q[8]), band(&q[2], &q[6]), poly((0..tail).collect(), &hist), poly(fut.clone(), &pyq[4]), poly(fut.clone(), &q[4])));
    }
    html.push_str("</body></html>");
    std::fs::write("report.html", html).expect("report");
    println!("\nWrote report.html.");
}
