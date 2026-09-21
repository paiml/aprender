//! #3751 done_when 1: how far the default CPU forward (Q8_K activations on its
//! Q4_K matvecs) sits from the same forward with FP32 activations, per model.
//!
//! Evidence harness, not a gate: with `APR_3751_MODELS` unset it does nothing.
//! Set it to a `:`-separated list of GGUF paths and run in release:
//!
//! ```text
//! APR_3751_MODELS=/m/a.gguf:/m/b.gguf \
//!   cargo test --release -p aprender-serve --lib q8k_reference_drift -- --nocapture
//! ```
//!
//! Each model runs its OWN CPU forward (the one `apr run --no-gpu` and every F2
//! reference use): dense `forward_single_with_cache`, the Qwen3.5 hybrid's
//! `forward_single_qwen35`, or Qwen3-MoE's `forward_single_qwen3_moe_with_cache`.
//! It runs over a fixed prompt set that includes a chat-template start, once as
//! shipped and once under `quantize::with_fp32_activations`. Printed per
//! (model, prompt): the minimum per-position logits cosine, argmax mismatches,
//! and the first generated position where greedy decoding diverges (done_when
//! 3: a divergence there is a CPU ANSWER that changed, a defect, not a note).
//! One `[q8k-drift]` line per row, for the receipt.

use crate::gguf::{MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel};

/// The fixed prompt set: a chat turn (special tokens at position 0), aprender-37's
/// Qwen3.5-9B repro turn, and plain text.
const PROMPTS: [(&str, &str); 3] = [
    (
        "chat-france",
        "<|im_start|>user\nWhat is the capital of France? Answer briefly.<|im_end|>\n<|im_start|>assistant\n",
    ),
    (
        "chat-zorblat",
        "<|im_start|>user\nMy name is Zorblat. What is the capital of Peru? Answer in one word.<|im_end|>\n<|im_start|>assistant\n",
    ),
    (
        "plain",
        "The history of the printing press begins in the fifteenth century, when Johannes Gutenberg combined movable metal type, oil-based ink and a wooden screw press.",
    ),
];

/// Greedy tokens generated after the prompt, for the answer comparison.
const GENERATE: usize = 16;

/// A CPU forward over one sequence: logits for each fed token.
type Forward<'m> = Box<dyn FnMut(u32, usize) -> Option<Vec<f32>> + 'm>;

/// Prompt logits at every position, then `GENERATE` greedy tokens.
fn run(mut fwd: Forward<'_>, prompt: &[u32]) -> Option<(Vec<Vec<f32>>, Vec<u32>)> {
    let mut per_pos = Vec::with_capacity(prompt.len());
    for (pos, &t) in prompt.iter().enumerate() {
        per_pos.push(fwd(t, pos)?);
    }
    let mut out = Vec::with_capacity(GENERATE);
    let mut next = crate::infer::argmax_u32(per_pos.last()?);
    for i in 0..GENERATE {
        out.push(next);
        next = crate::infer::argmax_u32(&fwd(next, prompt.len() + i)?);
    }
    Some((per_pos, out))
}

/// The model's own CPU forward over a fresh cache, as a closure.
enum Cpu<'a> {
    Dense(&'a OwnedQuantizedModel),
    Moe(
        &'a OwnedQuantizedModel,
        &'a [crate::gguf::qwen3_moe_load::Qwen3MoeQuantizedLayer],
        (usize, usize, usize),
        &'a [u8],
    ),
    Hybrid(&'a crate::gguf::forward_qwen35::Qwen35Model<'a>),
}

fn forward<'a>(cpu: &'a Cpu<'a>, max_seq: usize) -> Forward<'a> {
    match cpu {
        Cpu::Dense(m) => {
            let c = m.config();
            let mut cache = OwnedQuantizedKVCache::new(c.num_layers, c.kv_dim(), max_seq);
            Box::new(move |t, pos| m.forward_single_with_cache(t, &mut cache, pos).ok())
        },
        Cpu::Moe(m, layers, (e, k, d), data) => {
            let mut cache = OwnedQuantizedKVCache::from_config(m.config(), max_seq);
            Box::new(move |t, pos| {
                m.forward_single_qwen3_moe_with_cache(t, &mut cache, pos, layers, *e, *k, *d, data)
                    .ok()
            })
        },
        Cpu::Hybrid(q) => {
            let mut state = q.new_state(max_seq);
            Box::new(move |t, pos| q.forward_single_qwen35(t, &mut state, pos).ok())
        },
    }
}

fn measure(path: &str, cpu: &Cpu<'_>, mapped: &MappedGGUFModel) {
    for (name, text) in PROMPTS {
        let Some(tokens) = mapped.model.encode(text) else {
            eprintln!("[q8k-drift] {path} {name}: tokenizer refused the prompt");
            continue;
        };
        let max_seq = tokens.len() + GENERATE + 2;
        let q8k = run(forward(cpu, max_seq), &tokens);
        let fp32 = crate::quantize::with_fp32_activations(|| run(forward(cpu, max_seq), &tokens));
        let (Some((a, ga)), Some((b, gb))) = (q8k, fp32) else {
            eprintln!("[q8k-drift] {path} {name}: a CPU forward failed");
            continue;
        };
        let mut min_cos = 1.0f32;
        let mut min_pos = 0;
        let mut mismatches = 0;
        for (pos, (x, y)) in a.iter().zip(&b).enumerate() {
            let c = crate::infer::logits_cosine_similarity(x, y);
            if c < min_cos {
                min_cos = c;
                min_pos = pos;
            }
            if crate::infer::argmax_u32(x) != crate::infer::argmax_u32(y) {
                mismatches += 1;
            }
        }
        let diverge = ga.iter().zip(&gb).position(|(x, y)| x != y);
        eprintln!(
            "[q8k-drift] {} | {name} | positions {} | min cosine {min_cos:.6} at pos {min_pos} | \
             argmax mismatches {mismatches} | greedy {} | {}",
            std::path::Path::new(path)
                .file_name()
                .map_or(path.into(), |f| f.to_string_lossy()),
            a.len(),
            match diverge {
                None => format!("identical for {GENERATE} tokens"),
                Some(i) => format!("DIVERGES at generated token {i}"),
            },
            if min_cos < 0.999 { "BELOW 0.999" } else { "ok" },
        );
    }
}

#[test]
fn q8k_reference_drift() {
    let Ok(list) = std::env::var("APR_3751_MODELS") else {
        eprintln!("SKIP: APR_3751_MODELS is not set");
        return;
    };
    for path in list.split(':').filter(|p| !p.is_empty()) {
        let Ok(mapped) = MappedGGUFModel::from_path(path) else {
            eprintln!("[q8k-drift] {path}: will not map");
            continue;
        };
        let arch = mapped.model.architecture().unwrap_or_default().to_string();
        if crate::gguf::hybrid_forward_handles(&arch) {
            use crate::gguf::forward_qwen35::Qwen35Model;
            let Ok(base) = Qwen35Model::create_base_model(&mapped.model, mapped.data()) else {
                eprintln!("[q8k-drift] {path}: qwen35 base will not load");
                continue;
            };
            let Ok(q) = Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data())
            else {
                eprintln!("[q8k-drift] {path}: qwen35 layers will not load");
                continue;
            };
            measure(path, &Cpu::Hybrid(&q), &mapped);
        } else if crate::tensor_names::normalize_architecture(&arch) == "qwen3_moe" {
            let Ok(m) = OwnedQuantizedModel::from_mapped(&mapped) else {
                continue;
            };
            let (Some(e), Some(k), Some(d)) = (
                mapped.model.expert_count(),
                mapped.model.expert_used_count(),
                mapped.model.expert_feed_forward_length(),
            ) else {
                continue;
            };
            let layers: Option<Vec<_>> = (0..m.config().num_layers)
                .map(|il| {
                    crate::gguf::qwen3_moe_load::load_qwen3_moe_layer(
                        &mapped.model,
                        mapped.data(),
                        il,
                    )
                    .ok()
                })
                .collect();
            let Some(layers) = layers else { continue };
            measure(
                path,
                &Cpu::Moe(&m, &layers, (e, k, d), mapped.data()),
                &mapped,
            );
        } else {
            let Ok(m) = OwnedQuantizedModel::from_mapped(&mapped) else {
                eprintln!("[q8k-drift] {path}: dense model will not load");
                continue;
            };
            measure(path, &Cpu::Dense(&m), &mapped);
        }
    }
}
