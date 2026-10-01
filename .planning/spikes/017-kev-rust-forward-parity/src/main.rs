//! Spike 017: Kev-0.8B decision forward in Rust on upstream's Qwen3.5 CPU path, against the torch fp32 oracle.
//!
//! Ladder: final-normed hidden at every position of row 0 -> hidden at every readout -> raw pointer logits ->
//! served probabilities -> argmax. Token ids come from the fixture (the tokenizer is a separate rung).
use realizar::gguf::forward_qwen35::Qwen35Model;
use realizar::gguf::MappedGGUFModel;
use serde::Deserialize;
use std::time::Instant;

#[derive(Deserialize)]
struct Fixture { temperature: f32, records: Vec<Record> }
#[derive(Deserialize)]
struct Record { rows: Vec<Row>, n_state: usize, torch_cpu_ms: f64 }
#[derive(Deserialize)]
struct Row {
    ids: Vec<u32>, decide: usize, opts: Vec<usize>, h_decide: Vec<f32>, h_opts: Vec<Vec<f32>>,
    logits_raw: Vec<f32>, probs: Vec<f32>, ladder: Option<Ladder>,
}
#[derive(Deserialize)]
struct Ladder { final_all: Vec<Vec<f32>> }

struct Head { qw: Vec<f32>, qb: Vec<f32>, kw: Vec<f32>, kb: Vec<f32>, dp: usize, d: usize }

impl Head {
    fn load(path: &str) -> Head {
        let bytes = std::fs::read(path).expect("head.safetensors");
        let st = safetensors::SafeTensors::deserialize(&bytes).expect("parse head");
        let get = |n: &str| -> (Vec<f32>, Vec<usize>) {
            let t = st.tensor(n).expect(n);
            (t.data().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(), t.shape().to_vec())
        };
        let (qw, s) = get("q.weight"); let (qb, _) = get("q.bias"); let (kw, _) = get("k.weight"); let (kb, _) = get("k.bias");
        Head { qw, qb, kw, kb, dp: s[0], d: s[1] }
    }
    fn proj(&self, w: &[f32], b: &[f32], x: &[f32]) -> Vec<f32> {
        (0..self.dp).map(|o| b[o] + w[o * self.d..(o + 1) * self.d].iter().zip(x).map(|(a, c)| a * c).sum::<f32>()).collect()
    }
    /// logits_j = k(h_opt_j) . q(h_decide) / sqrt(dp) -- kev.model.PointerHead at T = 1
    fn logits(&self, hd: &[f32], ho: &[Vec<f32>]) -> Vec<f32> {
        let q = self.proj(&self.qw, &self.qb, hd);
        let scale = 1.0 / (self.dp as f32).sqrt();
        ho.iter().map(|h| self.proj(&self.kw, &self.kb, h).iter().zip(&q).map(|(a, b)| a * b).sum::<f32>() * scale).collect()
    }
}

fn softmax(z: &[f32], t: f32) -> Vec<f32> {
    let m = z.iter().fold(f32::MIN, |a, &b| a.max(b / t));
    let e: Vec<f32> = z.iter().map(|&v| (v / t - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}
fn max_abs(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max) }
fn rms(a: &[f32]) -> f32 { (a.iter().map(|x| x * x).sum::<f32>() / a.len() as f32).sqrt() }
fn argmax(a: &[f32]) -> usize { a.iter().enumerate().fold((0, f32::MIN), |m, (i, &v)| if v > m.1 { (i, v) } else { m }).0 }

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (gguf, head_path, fix_path) = (&args[1], &args[2], &args[3]);
    let fixture: Fixture = serde_json::from_str(&std::fs::read_to_string(fix_path).expect("fixture")).expect("parse fixture");
    let t0 = Instant::now();
    let mapped = MappedGGUFModel::from_path(gguf).expect("gguf");
    let base = Qwen35Model::create_base_model(&mapped.model, mapped.data()).expect("base");
    let qwen = Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data()).expect("layers");
    let head = Head::load(head_path);
    let load_ms = t0.elapsed().as_secs_f64() * 1e3;
    println!("load: {load_ms:.0} ms (mmap GGUF + owned layers + head)\n");
    println!("| rec | row | tokens | rust ms | torch-cpu ms | max|dh| decide (rel rms) | max|dh| opts | max|dlogit| | max|dp| | argmax |");
    println!("|---|---|---|---|---|---|---|---|---|---|");
    let (mut worst_p, mut worst_z, mut agree, mut n) = (0f32, 0f32, 0, 0);
    for (ri, rec) in fixture.records.iter().enumerate() {
        for (qi, row) in rec.rows.iter().enumerate() {
            let max_tokens: usize = std::env::var("MAX_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
            if row.ids.len() > max_tokens { continue; }
            let s = Instant::now();
            let mut state = qwen.new_state(row.ids.len() + 1);
            let mut hs: Vec<Vec<f32>> = Vec::with_capacity(row.ids.len());
            for (pos, &tok) in row.ids.iter().enumerate() {
                hs.push(qwen.forward_single_qwen35_hidden(tok, &mut state, pos).expect("forward"));
            }
            let ms = s.elapsed().as_secs_f64() * 1e3;
            if let Some(l) = &row.ladder {
                let (mut worst, mut at) = (0f32, 0);
                for (p, (a, b)) in hs.iter().zip(&l.final_all).enumerate() { let d = max_abs(a, b); if d > worst { worst = d; at = p; } }
                println!("ladder row0: final hidden at all {} positions, max|dh| {worst:.3e} at pos {at} (rms of h {:.3})", hs.len(), rms(&l.final_all[at]));
            }
            let hd = &hs[row.decide];
            let ho: Vec<Vec<f32>> = row.opts.iter().map(|&o| hs[o].clone()).collect();
            let dh_d = max_abs(hd, &row.h_decide);
            let dh_o = ho.iter().zip(&row.h_opts).map(|(a, b)| max_abs(a, b)).fold(0.0, f32::max);
            let z = head.logits(hd, &ho);
            let p = softmax(&z, fixture.temperature);
            let (dz, dp) = (max_abs(&z, &row.logits_raw), max_abs(&p, &row.probs));
            let ok = argmax(&p) == argmax(&row.probs);
            worst_p = worst_p.max(dp); worst_z = worst_z.max(dz); agree += ok as usize; n += 1;
            println!("| {ri} | {qi} | {} (state {}) | {ms:.0} | {:.0} | {dh_d:.2e} ({:.1e}) | {dh_o:.2e} | {dz:.2e} | {dp:.2e} | {} |",
                row.ids.len(), rec.n_state, rec.torch_cpu_ms / rec.rows.len() as f64, dh_d / rms(&row.h_decide), if ok { "same" } else { "FLIP" });
        }
    }
    println!("\nworst |dlogit| {worst_z:.3e}, worst |dp| {worst_p:.3e}, argmax agree {agree}/{n}");
}
