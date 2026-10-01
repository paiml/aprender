//! Spike 025 parity driver: tokenizer rung -> ladder (row 0: embeddings, 28 layers, final norm, 2 head layers) ->
//! marker states -> raw logits -> served probabilities -> argmax, against Laya's torch fp32 CPU path.
//!   laya-parity <checkpoint-snapshot-dir> fixtures/laya-en_fixture.json fixtures/laya-en_ladder.bin
use serde::Deserialize;
use std::time::Instant;

#[derive(Deserialize)]
struct Fixture { max_len: usize, head_max_len: usize, cls: u32, sep: u32, mask: u32, mask_token: String, ladder: Ladder, records: Vec<Rec> }
#[derive(Deserialize)]
struct Ladder { n_tokens: usize, d: usize, blocks: Vec<String> }
#[derive(Deserialize)]
struct Rec { state: String, rows: Vec<Row> }
#[derive(Deserialize)]
struct Row { qid: String, t: String, ins: String, options: Vec<String>, ids: Vec<u32>, markers: Vec<usize>, qtype: usize,
             temperature: f32, logits: Vec<f32>, probs: Vec<f64>, m_opts: Vec<Vec<f32>>, torch_cpu_ms: f64 }

fn max_abs(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max) }
fn rms(a: &[f32]) -> f32 { (a.iter().map(|x| x * x).sum::<f32>() / a.len() as f32).sqrt() }
fn argmax<T: PartialOrd + Copy>(a: &[T]) -> usize { (0..a.len()).fold(0, |m, i| if a[i] > a[m] { i } else { m }) }

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let dir = std::path::Path::new(&a[1]);
    let fx: Fixture = serde_json::from_str(&std::fs::read_to_string(&a[2]).expect("fixture")).expect("parse fixture");
    let ladder: Vec<f32> = std::fs::read(&a[3]).expect("ladder").chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let t0 = Instant::now();
    let enc_cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("encoder/config.json")).expect("cfg")).expect("cfg json");
    let agent_cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("rl_agent_config.json")).expect("agent cfg")).expect("agent json");
    let bytes = std::fs::read(dir.join("model.safetensors")).expect("weights");
    let t_read = t0.elapsed().as_secs_f64();
    let mut model = laya::Laya::from_tensors(laya::load_tensors(&bytes), &enc_cfg, agent_cfg["head_layers"].as_u64().unwrap_or(2) as usize);
    drop(bytes);
    if let Ok(w) = std::env::var("WINDOW") { model.window = w.parse().expect("WINDOW"); }
    let builder = laya::Builder { tok: tokenizers::Tokenizer::from_file(dir.join("tokenizer/tokenizer.json")).expect("tokenizer"),
        cls: fx.cls, sep: fx.sep, mask: fx.mask, mask_token: fx.mask_token.clone(), max_len: fx.max_len, head_max_len: fx.head_max_len };
    println!("load: read {:.2} s, total {:.2} s (F16 -> f32), threads {}, local window |i-j| <= {}\n", t_read, t0.elapsed().as_secs_f64(), rayon::current_num_threads(), model.window);

    // ladder: row 0 of record 0
    let (d, l) = (fx.ladder.d, fx.ladder.n_tokens);
    let r0 = &fx.records[0].rows[0];
    let mut worst_ladder = Vec::new();
    model.forward(&r0.ids, &r0.markers, r0.qtype, |name, x| {
        if let Some(bi) = fx.ladder.blocks.iter().position(|b| b == name) {
            let refb = &ladder[bi * l * d..(bi + 1) * l * d];
            worst_ladder.push((name.to_string(), max_abs(x, refb), rms(refb)));
        }
    });
    println!("| ladder block ({l} tokens) | max abs diff | ref rms |\n|---|---|---|");
    for (i, (n, e, r)) in worst_ladder.iter().enumerate() {
        if i < 5 || i + 5 >= worst_ladder.len() || i % 6 == 0 { println!("| {n} | {e:.3e} | {r:.3} |"); }
    }

    println!("\n| rec | qid | type | tokens | ids == python | rust ms | torch ms | max abs dm | max abs dz | max abs dp | argmax |\n|---|---|---|---|---|---|---|---|---|---|---|");
    let (mut worst_p, mut worst_z, mut agree, mut n, mut id_ok) = (0f64, 0f32, 0, 0, 0);
    for (ri, rec) in fx.records.iter().enumerate() {
        for row in &rec.rows {
            let (ids, markers) = builder.build(&rec.state, &row.t, &row.ins, &row.options);
            let same = ids == row.ids && markers == row.markers;
            id_ok += usize::from(same);
            let t = laya::temperature(&agent_cfg, row.qtype, row.markers.len());
            assert!((t - row.temperature).abs() < 1e-6, "temperature bucket {t} vs python {}", row.temperature);
            let s = Instant::now();
            let mut dm = 0f32;
            let z = model.forward(&row.ids, &row.markers, row.qtype, |name, x| {
                if name == "m_opts" { dm = row.m_opts.iter().enumerate().map(|(j, r)| max_abs(&x[j * d..(j + 1) * d], r)).fold(0.0, f32::max); }
            });
            let ms = s.elapsed().as_secs_f64() * 1e3;
            let p = laya::softmax_t(&z, t);
            let dz = max_abs(&z, &row.logits);
            let dp = p.iter().zip(&row.probs).map(|(a, b)| (f64::from(*a) - b).abs()).fold(0.0, f64::max);
            let ok = argmax(&p) == argmax(&row.probs);
            worst_p = worst_p.max(dp); worst_z = worst_z.max(dz); agree += usize::from(ok); n += 1;
            println!("| {ri} | {} | {} | {} | {} | {ms:.0} | {:.0} | {dm:.2e} | {dz:.2e} | {dp:.2e} | {} |", row.qid, row.t, row.ids.len(),
                     if same { "yes" } else { "NO" }, row.torch_cpu_ms, if ok { "same" } else { "FLIP" });
        }
    }
    println!("\nids identical {id_ok}/{n}; worst |dz| {worst_z:.3e}; worst |dp| {worst_p:.3e}; argmax agree {agree}/{n}");
}
