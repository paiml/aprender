//! Spike 020: batched Qwen3.5 prefill (`Qwen35Model::prefill_hidden`) vs the token-at-a-time path and the torch
//! fp32 oracle, on spike 017's Kev-0.8B fixture. Args: <gguf> <head.safetensors> <fixture.json> [--no-control]
use realizar::gguf::forward_qwen35::Qwen35Model;
use realizar::gguf::forward_qwen35_prefill::take_prefill_phases;
use realizar::gguf::MappedGGUFModel;
use serde::Deserialize;
use std::time::Instant;

#[derive(Deserialize)]
struct Fixture { temperature: f32, records: Vec<Record> }
#[derive(Deserialize)]
struct Record { rows: Vec<Row> }
#[derive(Deserialize)]
struct Row { ids: Vec<u32>, decide: usize, opts: Vec<usize>, h_decide: Vec<f32>, logits_raw: Vec<f32>, probs: Vec<f32> }

struct Head { qw: Vec<f32>, qb: Vec<f32>, kw: Vec<f32>, kb: Vec<f32>, dp: usize, d: usize }
impl Head {
    fn load(path: &str) -> Head {
        let bytes = std::fs::read(path).expect("head");
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
    fn logits(&self, hd: &[f32], ho: &[&[f32]]) -> Vec<f32> {
        let q = self.proj(&self.qw, &self.qb, hd);
        let s = 1.0 / (self.dp as f32).sqrt();
        ho.iter().map(|h| self.proj(&self.kw, &self.kb, h).iter().zip(&q).map(|(a, b)| a * b).sum::<f32>() * s).collect()
    }
}
fn softmax(z: &[f32], t: f32) -> Vec<f32> {
    let m = z.iter().fold(f32::MIN, |a, &b| a.max(b / t));
    let e: Vec<f32> = z.iter().map(|&v| (v / t - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}
fn max_abs(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max) }
fn argmax(a: &[f32]) -> usize { a.iter().enumerate().fold((0, f32::MIN), |m, (i, &v)| if v > m.1 { (i, v) } else { m }).0 }

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let control = !args.iter().any(|a| a == "--no-control");
    let fixture: Fixture = serde_json::from_str(&std::fs::read_to_string(&args[3]).expect("fixture")).expect("parse");
    let t0 = Instant::now();
    let mapped = MappedGGUFModel::from_path(&args[1]).expect("gguf");
    let base = Qwen35Model::create_base_model(&mapped.model, mapped.data()).expect("base");
    let qwen = Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data()).expect("layers");
    let head = Head::load(&args[2]);
    let d = base.config().hidden_dim;
    println!("load {:.0} ms, rayon threads {}\n", t0.elapsed().as_secs_f64() * 1e3, rayon::current_num_threads());
    println!("| row | tokens | prefill ms | token-at-a-time ms | speedup | max|dh| prefill vs step | max|dh| decide vs torch | max|dp| vs torch | argmax |");
    println!("|---|---|---|---|---|---|---|---|---|");
    let (mut worst_p, mut agree, mut n, mut tot_pre, mut tot_step) = (0f32, 0, 0, 0f64, 0f64);
    // warm-up: one short row so rayon pools and page faults are not billed to row 0
    let mut st = qwen.new_state(8); qwen.prefill_hidden(&[1, 2, 3, 4], &mut st).expect("warm"); let _ = take_prefill_phases();
    for rec in &fixture.records {
        for row in &rec.rows {
            let mut st = qwen.new_state(row.ids.len() + 1);
            let s = Instant::now();
            let h = qwen.prefill_hidden(&row.ids, &mut st).expect("prefill");
            let pre_ms = s.elapsed().as_secs_f64() * 1e3;
            if std::env::var("PHASES").is_ok() {
                let ph = take_prefill_phases();
                let other = pre_ms / 1e3 - ph.iter().sum::<f64>();
                eprintln!("{} tok: gemm {:.0} ms, transpose {:.0} ms, deltanet-loop {:.0} ms, attention {:.0} ms, other {:.0} ms",
                    row.ids.len(), ph[0] * 1e3, ph[1] * 1e3, ph[2] * 1e3, ph[3] * 1e3, other * 1e3);
            }
            let (step_ms, dstep) = if control {
                let mut st = qwen.new_state(row.ids.len() + 1);
                let s = Instant::now();
                let mut worst = 0f32;
                for (p, &tok) in row.ids.iter().enumerate() {
                    let hs = qwen.forward_single_qwen35_hidden(tok, &mut st, p).expect("step");
                    worst = worst.max(max_abs(&hs, &h[p * d..(p + 1) * d]));
                }
                (s.elapsed().as_secs_f64() * 1e3, worst)
            } else { (f64::NAN, f32::NAN) };
            let hd = &h[row.decide * d..(row.decide + 1) * d];
            let ho: Vec<&[f32]> = row.opts.iter().map(|&o| &h[o * d..(o + 1) * d]).collect();
            let z = head.logits(hd, &ho);
            let p = softmax(&z, fixture.temperature);
            let dp = max_abs(&p, &row.probs);
            let ok = argmax(&p) == argmax(&row.probs);
            let _ = max_abs(&z, &row.logits_raw);
            worst_p = worst_p.max(dp); agree += ok as usize; n += 1; tot_pre += pre_ms; tot_step += step_ms;
            println!("| {n} | {} | {pre_ms:.0} | {step_ms:.0} | {:.1}x | {dstep:.2e} | {:.2e} | {dp:.2e} | {} |",
                row.ids.len(), step_ms / pre_ms, max_abs(hd, &row.h_decide), if ok { "same" } else { "FLIP" });
        }
    }
    println!("\nworst |dp| vs torch {worst_p:.3e}, argmax {agree}/{n}, total prefill {tot_pre:.0} ms vs token-at-a-time {tot_step:.0} ms");
}
