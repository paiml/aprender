//! Spike 009 driver: Chronos-2 parity ladder vs the Python oracle, edge probes, cost.
//! Run from the spike directory: CARGO_TARGET_DIR=../../../target cargo run --release [models/chronos-2]
mod chronos2;
mod safetensors;
use chronos2::{Chronos2, Config};
use std::time::Instant;

fn maxabs(a: &[f32], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x, y)| (*x as f64 - y).abs()).fold(0.0, f64::max) }
fn f64s(v: &serde_json::Value) -> Vec<f64> { v.as_array().expect("array").iter().map(|x| x.as_f64().expect("f64")).collect() }
fn load_csv(path: &str) -> Vec<f32> {
    let s = std::fs::read_to_string(path).expect("csv");
    let mut rows: Vec<(String, f32)> = s.lines().skip(1).filter_map(|l| { let mut it = l.split(','); let d = it.next()?.trim_matches('"').get(..10)?.to_string(); let v = it.next()?.trim_matches('"').parse().ok()?; Some((d, v)) }).collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0)); rows.dedup_by(|l, e| { if l.0 == e.0 { *e = l.clone(); true } else { false } });
    rows.into_iter().map(|(_, v)| v).collect()
}
fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).expect("finite")); v[v.len() / 2] }

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "models/chronos-2".into());
    let fx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string("fixtures/chronos2_fixture.json").expect("fixture")).expect("json");
    let t0 = Instant::now();
    let cfg_json: serde_json::Value = serde_json::from_slice(&std::fs::read(format!("{dir}/config.json")).expect("config")).expect("json");
    let (w, dtype) = safetensors::load(&format!("{dir}/model.safetensors")).expect("weights");
    let n_params: usize = w.values().map(|t| t.data.len()).sum();
    let m = Chronos2::load(&w, Config::from_json(&cfg_json));
    drop(w);
    let load_s = t0.elapsed().as_secs_f64();
    println!("# Spike 009 — Chronos-2 parity (Rust port vs chronos-forecasting 2.3.1)\n");
    println!("Model: {dir} — {n_params} params, weights {dtype}, load {load_s:.2} s (decode + transposes); torch threads in oracle: {}\n", fx["torch_threads"]);

    let peyton = load_csv("fixtures/peyton_manning.csv");
    let air = load_csv("fixtures/air_passengers.csv");
    let mut gaps = peyton[peyton.len() - 300..].to_vec(); for i in (0..gaps.len()).step_by(7) { gaps[i] = f32::NAN; }
    let mut lead = vec![f32::NAN; 16]; lead.extend_from_slice(&peyton[peyton.len() - 100..]);
    let series: Vec<(&str, Vec<f32>)> = vec![
        ("peyton", peyton.clone()), ("air", air.clone()), ("short100", peyton[peyton.len() - 100..].to_vec()), ("nan_gaps", gaps), ("leading_nan_patch", lead),
        ("constant", vec![3.0; 100]), ("huge_scale", peyton[peyton.len() - 256..].iter().map(|v| v * 1e6).collect()), ("negative", peyton[peyton.len() - 256..].iter().map(|v| -v).collect()), ("short5", peyton[peyton.len() - 5..].to_vec()),
    ];
    println!("## 1. Parity ladder (max |Δ| vs oracle)\n");
    println!("| series | h | n | patches | tokens | loc | scale | patch feats | embed first/last | hidden first / REG / last | quantiles (21×h) | pipeline (trunc.) | Rust s | torch s |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    let mut results = serde_json::Map::new();
    let mut worst_q = 0.0f64;
    for (name, y) in &series {
        let entry = &fx["series"][*name];
        let mut hs: Vec<usize> = entry.as_object().expect("obj").keys().map(|k| k.parse().expect("h")).collect(); hs.sort();
        for h in hs {
            let o = &entry[&h.to_string()];
            let nop = o["num_output_patches"].as_u64().expect("nop") as usize;
            let t = Instant::now(); let f = m.forward(y, nop); let dt = t.elapsed().as_secs_f64();
            let d_loc = (f.loc as f64 - o["loc"].as_f64().unwrap()).abs(); let d_scale = (f.scale as f64 - o["scale"].as_f64().unwrap()).abs();
            let d_feat = maxabs(&f.patch_feat_first, &f64s(&o["patch_feat_first"])).max(maxabs(&f.patch_feat_last, &f64s(&o["patch_feat_last"])));
            let d_emb = maxabs(&f.embed_first, &f64s(&o["embed_first"])).max(maxabs(&f.embed_last, &f64s(&o["embed_last"])));
            let d_hid = maxabs(&f.hidden_first, &f64s(&o["hidden_first"])).max(maxabs(&f.hidden_reg, &f64s(&o["hidden_reg"]))).max(maxabs(&f.hidden_last, &f64s(&o["hidden_last"])));
            let oq = o["quantiles"].as_array().unwrap();
            let d_q = f.quantiles.iter().zip(oq).map(|(a, b)| maxabs(a, &f64s(b))).fold(0.0, f64::max);
            let pq = o["pipeline_quantiles"].as_array().unwrap();
            let d_pq = f.quantiles.iter().zip(pq).map(|(a, b)| maxabs(&a[..h.min(a.len())], &f64s(b))).fold(0.0, f64::max);
            let mask_ok = f.attention_mask.iter().map(|b| *b as i64).collect::<Vec<_>>()[..f.n_patches] == o["attention_mask"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect::<Vec<_>>();
            let scale = o["scale"].as_f64().unwrap().abs().max(1e-12);
            worst_q = worst_q.max(d_q / scale);
            println!("| {name} | {h} | {} | {}{} | {} | {d_loc:.1e} | {d_scale:.1e} | {d_feat:.1e} | {d_emb:.1e} | {d_hid:.1e} | {d_q:.1e} ({:.1e} of scale) | {d_pq:.1e} | {dt:.2} | {:.2} |", y.len(), f.n_patches, if mask_ok { "" } else { " MASK≠" }, f.tokens, d_q / scale, o["model_seconds"].as_f64().unwrap());
            results.insert(format!("{name}/{h}"), serde_json::json!({"n": y.len(), "patches": f.n_patches, "tokens": f.tokens, "d_loc": d_loc, "d_scale": d_scale, "d_feat": d_feat, "d_embed": d_emb, "d_hidden": d_hid, "d_quantiles": d_q, "d_quantiles_rel": d_q / scale, "d_pipeline": d_pq, "rust_s": dt, "torch_s": o["model_seconds"], "pipeline_s": o["pipeline_seconds"]}));
        }
    }
    println!("\nWorst quantile |Δ| relative to the series scale: {worst_q:.2e}");

    println!("\n## 2. Cost (Apple M4 Pro, single thread, packed GEMM with the spike-008 NEON kernel)\n");
    println!("| context | h | patches | tokens | forward ms (median of 3) | GFLOP | GFLOP/s |\n|---|---|---|---|---|---|---|");
    let mut long = peyton.clone(); while long.len() < 8192 { let c = long.clone(); long.extend_from_slice(&c); } long.truncate(8192);
    for (ctx_len, h) in [(100usize, 64usize), (512, 64), (2048, 64), (2905, 64), (2048, 365), (2048, 1024), (8192, 64)] {
        let c = if ctx_len <= peyton.len() { &peyton[peyton.len() - ctx_len..] } else { &long[..ctx_len] };
        let nop = (h + 15) / 16;
        let mut tokens = 0;
        let ms = median((0..3).map(|_| { let t = Instant::now(); let f = m.forward(c, nop); tokens = f.tokens; t.elapsed().as_secs_f64() * 1e3 }).collect());
        let (d, dff, l) = (m.cfg.d_model as f64, m.cfg.d_ff as f64, tokens as f64);
        let gflop = m.cfg.layers as f64 * (6.0 * l * d * d * 2.0 + 2.0 * l * d * dff * 2.0 + 2.0 * l * l * d * 2.0) / 1e9;
        println!("| {ctx_len} | {h} | {nop} | {tokens} | {ms:.0} | {gflop:.1} | {:.1} |", gflop / ms * 1e3);
        results.insert(format!("cost/{ctx_len}/{h}"), serde_json::json!({"tokens": tokens, "ms": ms, "gflop": gflop}));
    }
    println!("\n## 3. Threads: rayon-parallel `blis::gemm` (crate feature `parallel`, {} cores) vs single-threaded `gemm_blis`\n", std::thread::available_parallelism().map_or(0, |n| n.get()));
    println!("| context | h | tokens | serial ms | parallel ms | speed-up | max abs Δ |\n|---|---|---|---|---|---|---|");
    for (ctx_len, h) in [(512usize, 64usize), (2048, 64), (2905, 64), (8192, 64)] {
        let c = if ctx_len <= peyton.len() { &peyton[peyton.len() - ctx_len..] } else { &long[..ctx_len] };
        let nop = (h + 15) / 16;
        chronos2::PARALLEL_GEMM.store(false, std::sync::atomic::Ordering::Relaxed);
        let f0 = m.forward(c, nop);
        let ms0 = median((0..3).map(|_| { let t = Instant::now(); let _ = m.forward(c, nop); t.elapsed().as_secs_f64() * 1e3 }).collect());
        chronos2::PARALLEL_GEMM.store(true, std::sync::atomic::Ordering::Relaxed);
        let f1 = m.forward(c, nop);
        let ms1 = median((0..3).map(|_| { let t = Instant::now(); let _ = m.forward(c, nop); t.elapsed().as_secs_f64() * 1e3 }).collect());
        chronos2::PARALLEL_GEMM.store(false, std::sync::atomic::Ordering::Relaxed);
        let d = f0.quantiles.iter().zip(&f1.quantiles).flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs())).fold(0.0f32, f32::max);
        println!("| {ctx_len} | {h} | {} | {ms0:.0} | {ms1:.0} | {:.1}× | {d:.1e} |", f0.tokens, ms0 / ms1);
        results.insert(format!("threads/{ctx_len}/{h}"), serde_json::json!({"serial_ms": ms0, "parallel_ms": ms1}));
    }
    let h = 1100; println!("\npredict(h={h}): {}", match m.predict(&peyton, h) { Ok(_) => "ok".to_string(), Err(e) => format!("refused — {e}") });
    std::fs::write("results.json", serde_json::to_string_pretty(&serde_json::Value::Object(results)).expect("json")).expect("write");
}
