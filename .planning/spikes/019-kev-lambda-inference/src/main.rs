//! Spike 019: Kev-0.8B as a Lambda would run it, measured locally (6 rayon threads = a 10,240 MB Lambda's vCPUs).
//! A fresh process prints a cold-start timeline with RSS at each step, whether releasing the mmap after building
//! the owned model gives memory back, first-decision latency, then steady-state latency per request shape.
//! Args: <gguf> <head.safetensors> <fixture.json> [--keep-mmap]
use realizar::gguf::forward_qwen35::Qwen35Model;
use realizar::gguf::MappedGGUFModel;
use serde::Deserialize;
use std::time::Instant;

#[derive(Deserialize)]
struct Fixture { temperature: f32, records: Vec<Record> }
#[derive(Deserialize)]
struct Record { rows: Vec<Row> }
#[derive(Deserialize)]
struct Row { ids: Vec<u32>, decide: usize, opts: Vec<usize>, probs: Vec<f32> }

fn rss_mb() -> f64 {
    let out = std::process::Command::new("ps").args(["-o", "rss=", "-p", &std::process::id().to_string()]).output().expect("ps");
    String::from_utf8_lossy(&out.stdout).trim().parse::<f64>().unwrap_or(0.0) / 1024.0
}

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
    fn probs(&self, hd: &[f32], ho: &[&[f32]], t: f32) -> Vec<f32> {
        let q = self.proj(&self.qw, &self.qb, hd);
        let s = 1.0 / (self.dp as f32).sqrt();
        let z: Vec<f32> = ho.iter().map(|h| self.proj(&self.kw, &self.kb, h).iter().zip(&q).map(|(a, b)| a * b).sum::<f32>() * s / t).collect();
        let m = z.iter().fold(f32::MIN, |a, &b| a.max(b));
        let e: Vec<f32> = z.iter().map(|v| (v - m).exp()).collect();
        let sum: f32 = e.iter().sum();
        e.iter().map(|v| v / sum).collect()
    }
}

fn decide(qwen: &Qwen35Model, head: &Head, d: usize, t: f32, row: &Row) -> Vec<f32> {
    let mut st = qwen.new_state(row.ids.len() + 1);
    let h = qwen.prefill_hidden(&row.ids, &mut st).expect("prefill");
    let ho: Vec<&[f32]> = row.opts.iter().map(|&o| &h[o * d..(o + 1) * d]).collect();
    head.probs(&h[row.decide * d..(row.decide + 1) * d], &ho, t)
}

fn pct(v: &mut [f64], p: f64) -> f64 { v.sort_by(|a, b| a.total_cmp(b)); v[((v.len() - 1) as f64 * p).round() as usize] }

fn main() {
    let t0 = Instant::now();
    let args: Vec<String> = std::env::args().collect();
    let keep_mmap = args.iter().any(|a| a == "--keep-mmap");
    let fixture: Fixture = serde_json::from_str(&std::fs::read_to_string(&args[3]).expect("fixture")).expect("parse");
    let ms = |t: Instant| t.elapsed().as_secs_f64() * 1e3;
    println!("## cold start (fresh process, rayon threads {}, keep_mmap {keep_mmap})\n", rayon::current_num_threads());
    println!("| step | t since start ms | RSS MB |\n|---|---|---|");
    println!("| process start | {:.0} | {:.0} |", ms(t0), rss_mb());
    let mapped = MappedGGUFModel::from_path(&args[1]).expect("gguf");
    println!("| mmap GGUF | {:.0} | {:.0} |", ms(t0), rss_mb());
    let base = Qwen35Model::create_base_model(&mapped.model, mapped.data()).expect("base");
    println!("| base (embeddings, norms, lm_head) | {:.0} | {:.0} |", ms(t0), rss_mb());
    let qwen = Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data()).expect("layers");
    println!("| owned layers | {:.0} | {:.0} |", ms(t0), rss_mb());
    if !keep_mmap { drop(mapped); }
    println!("| after {} mmap | {:.0} | {:.0} |", if keep_mmap { "keeping" } else { "dropping" }, ms(t0), rss_mb());
    let head = Head::load(&args[2]);
    let d = base.config().hidden_dim;
    let t = fixture.temperature;
    let first = &fixture.records[0].rows[0];
    let s = Instant::now();
    let p = decide(&qwen, &head, d, t, first);
    println!("| **first decision ({} tokens)** | {:.0} (decision {:.0}) | {:.0} |", first.ids.len(), ms(t0), ms(s), rss_mb());
    let dp0 = p.iter().zip(&first.probs).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);

    // steady state: every record as one request (all its questions), 5 rounds
    let mut per_req: Vec<(usize, usize, Vec<f64>)> = fixture.records.iter().map(|r| (r.rows.len(), r.rows.iter().map(|x| x.ids.len()).sum(), vec![])).collect();
    let mut worst = dp0;
    for _ in 0..5 {
        for (i, rec) in fixture.records.iter().enumerate() {
            let s = Instant::now();
            for row in &rec.rows {
                let p = decide(&qwen, &head, d, t, row);
                worst = worst.max(p.iter().zip(&row.probs).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max));
            }
            per_req[i].2.push(ms(s));
        }
    }
    println!("\n## steady state (5 rounds per request)\n\n| request | questions | tokens (sum of rows) | p50 ms | p95 ms |\n|---|---|---|---|---|");
    for (i, (q, tok, v)) in per_req.iter_mut().enumerate() {
        println!("| {i} | {q} | {tok} | {:.0} | {:.0} |", pct(v, 0.5), pct(v, 0.95));
    }
    println!("\npeak-so-far RSS {:.0} MB; worst |dp| vs torch fp32 {worst:.2e}", rss_mb());
}
