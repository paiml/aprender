//! #4313 positive control for the probability-space F2 metric.
//!
//! The metric change retires the whole-vocab cosine because tail rows false-reject
//! two correct backends. A metric that stops false-rejecting must still CATCH a real
//! defect, so this plants one: the super-block scale `d` of ONE attention layer's
//! `attn_output.weight` is multiplied by `S` in a private copy of the GGUF bytes,
//! and the planted model's per-position logits are judged against the clean model's
//! by `f2_multi_position_report`, exactly as F2 judges GPU against CPU.
//!
//! It needs a real model, so it is `#[ignore]`:
//!
//! ```text
//! APR_PC_MODEL=~/models/Qwen3.5-4B-Q4_K_M.gguf [APR_PC_LAYER=15] [APR_PC_TEXT=a.txt:b.txt] \
//!   cargo test --release -p aprender-serve --lib -- f2_positive_control --ignored --nocapture
//! ```

use super::Qwen35Model;
use crate::gguf::{
    MappedGGUFModel, GGUF_TYPE_Q4_K, GGUF_TYPE_Q5_K, GGUF_TYPE_Q6_K, GGUF_TYPE_Q8_0,
};

/// F2's probe length: the last 64 prompt tokens.
const PROBE: usize = 64;

/// (elements per block, block bytes, byte offset of the f16 block scale `d`).
fn d_layout(qtype: u32) -> Option<(usize, usize, usize)> {
    match qtype {
        GGUF_TYPE_Q4_K => Some((256, 144, 0)),
        GGUF_TYPE_Q5_K => Some((256, 176, 0)),
        GGUF_TYPE_Q6_K => Some((256, 210, 208)),
        GGUF_TYPE_Q8_0 => Some((32, 34, 0)),
        _ => None,
    }
}

/// Multiply every block's `d` in `buf[start..start+len]` by `s`.
fn scale_d(buf: &mut [u8], start: usize, len: usize, block: usize, d_off: usize, s: f32) {
    for b in (start..start + len).step_by(block) {
        let at = b + d_off;
        let d = half::f16::from_le_bytes([buf[at], buf[at + 1]]).to_f32();
        buf[at..at + 2].copy_from_slice(&half::f16::from_f32(d * s).to_le_bytes());
    }
}

fn per_position_logits(mapped: &MappedGGUFModel, data: &[u8], probe: &[u32]) -> Vec<Vec<f32>> {
    let base = Qwen35Model::create_base_model(&mapped.model, data).expect("base");
    let m = Qwen35Model::from_model_and_layers(&base, &mapped.model, data).expect("qwen35");
    let mut state = m.new_state(probe.len() + 2);
    probe
        .iter()
        .enumerate()
        .map(|(pos, &t)| {
            m.forward_single_qwen35(t, &mut state, pos)
                .expect("forward")
        })
        .collect()
}

/// The retired PMAT-919 rule, for the side-by-side column: any real position
/// below cosine 0.95.
fn old_cosine_min(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    a.iter()
        .zip(b)
        .skip(1)
        .map(|(x, y)| {
            let (mut d, mut nx, mut ny) = (0f64, 0f64, 0f64);
            for (&p, &q) in x.iter().zip(y) {
                d += f64::from(p) * f64::from(q);
                nx += f64::from(p) * f64::from(p);
                ny += f64::from(q) * f64::from(q);
            }
            (d / (nx.sqrt() * ny.sqrt())) as f32
        })
        .fold(1.0, f32::min)
}

const DEFAULT_PROMPTS: [&str; 4] = [
    "The river bends twice before it reaches the town. In spring the water rises over the \
     low fields and the farmers move their animals to the hill. By summer the fields are dry \
     again and children walk along the banks looking for smooth stones to throw.",
    "impl Session {\n    pub fn new(cfg: &Config) -> Self {\n        Self { cfg: cfg.clone(), \
     pos: 0, cache: Vec::new(), done: false }\n    }\n    pub fn step(&mut self, tok: u32) -> \
     bool {\n        self.pos += 1;\n        self.cache.push(tok);\n        self.done = self.pos \
     >= self.cfg.max_len;\n        self.done\n    }\n}\n",
    "Question: A train leaves at 9:40 and the trip takes 2 hours and 35 minutes. At what time \
     does it arrive? Explain each step of the calculation before giving the final answer.",
    "Mitochondria produce most of the cell's ATP through oxidative phosphorylation. The \
     electron transport chain pumps protons across the inner membrane, and ATP synthase uses \
     the resulting gradient to phosphorylate ADP.",
];

#[test]
#[ignore = "needs a real Qwen3.5 GGUF: APR_PC_MODEL"]
fn f2_positive_control_planted_attn_out_scale_is_red() {
    let Ok(path) = std::env::var("APR_PC_MODEL") else {
        panic!("set APR_PC_MODEL to a Qwen3.5 GGUF");
    };
    let mapped = MappedGGUFModel::from_path(&path).expect("map the GGUF");
    let attn: Vec<_> = mapped
        .model
        .tensors
        .iter()
        .filter(|t| t.name.starts_with("blk.") && t.name.ends_with(".attn_output.weight"))
        .collect();
    assert!(
        !attn.is_empty(),
        "no attn_output.weight: not a Qwen3.5 GGUF"
    );
    let tensor = match std::env::var("APR_PC_LAYER") {
        Ok(l) => {
            let name = format!("blk.{l}.attn_output.weight");
            *attn
                .iter()
                .find(|t| t.name == name)
                .expect("APR_PC_LAYER has no attn_output")
        },
        Err(_) => attn[attn.len() / 2],
    };
    let (per_block, block, d_off) =
        d_layout(tensor.qtype).expect("unsupported qtype for the plant");
    let n: u64 = tensor.dims.iter().product();
    let len = usize::try_from(n).expect("elements") / per_block * block;
    let start = mapped.model.tensor_data_start + usize::try_from(tensor.offset).expect("offset");
    println!(
        "[4313-pc] plant: {} qtype {} bytes {len}",
        tensor.name, tensor.qtype
    );

    let texts: Vec<String> = match std::env::var("APR_PC_TEXT") {
        Ok(list) => list
            .split(':')
            .map(|p| std::fs::read_to_string(p).expect("read text"))
            .collect(),
        Err(_) => DEFAULT_PROMPTS.iter().map(|s| (*s).to_owned()).collect(),
    };
    let mut buf = mapped.data().to_vec();
    let scales = [1.0f32, 1.02, 1.05, 1.1, 1.25, 1.5, 2.0, 0.5, 0.0];
    let mut missed = Vec::new();
    for (i, text) in texts.iter().enumerate() {
        let ids = mapped.model.encode(text).expect("encode");
        let probe = &ids[ids.len().saturating_sub(PROBE)..];
        let clean = per_position_logits(&mapped, &buf, probe);
        for &s in &scales {
            scale_d(&mut buf, start, len, block, d_off, s);
            let planted = per_position_logits(&mapped, &buf, probe);
            // Restore from the mapping rather than dividing: S = 0 is not invertible.
            buf[start..start + len].copy_from_slice(&mapped.data()[start..start + len]);
            let r = crate::infer::f2_multi_position_report(&clean, &planted);
            let cos = old_cosine_min(&clean, &planted);
            println!(
                "[4313-pc] prompt {i} ({} tok) S={s:<4} new={} ({}, pos {}, max KL {:.4}) old-cos-min {cos:.4} old={}",
                probe.len(),
                if r.accepted { "ACCEPT" } else { "REJECT" },
                r.first_bad_reason.as_str(),
                r.first_bad_pos,
                r.max_kl_real,
                if cos < 0.95 { "REJECT" } else { "ACCEPT" },
            );
            // Negative control: an unplanted model is bit-identical, so it must pass.
            if (s - 1.0).abs() < f32::EPSILON {
                assert!(
                    r.accepted && r.max_kl_real < 1e-6,
                    "clean vs clean rejected: {r:?}"
                );
            } else if (s - 1.0).abs() >= 0.25 && r.accepted {
                // A 25%+ mis-scaled (or deleted) attention output is a real defect.
                missed.push((i, s, r.max_kl_real, cos));
            }
        }
    }
    assert!(
        missed.is_empty(),
        "the new F2 metric ACCEPTED a planted defect: {missed:?}"
    );
}
