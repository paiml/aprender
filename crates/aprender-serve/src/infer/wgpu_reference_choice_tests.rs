//! #3757 / #3751: is the wgpu parity gate's rejection (cosine ~0.955 at step 2
//! of 3 on qwen2.5-coder-1.5b, on four hosts) the Q8_K CPU reference, or wgpu?
//!
//! Runs the runtime's own `try_wgpu_generate` twice per (model, prompt): as
//! shipped, and with the whole call inside `quantize::with_fp32_activations`.
//! The scope reaches only the CPU Q4_K matvec, so the only thing that changes
//! is the CPU reference the gate compares against (the wgpu side's CPU work,
//! the embedding lookup and the f32 LM-head dot, takes no Q8_K path). The gate
//! reads only the prompt's FIRST token and then follows the CPU's argmax.
//!
//! Evidence harness: set `APR_3757_MODELS` to `:`-separated GGUF paths (and
//! `APR_WGPU_PARITY_STEPS` for a longer probe); unset, it does nothing.

use crate::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};

const PROMPTS: [(&str, &str); 2] = [
    (
        "chat",
        "<|im_start|>user\nWhat is 2+2? Answer with just the number.<|im_end|>\n<|im_start|>assistant\n",
    ),
    ("plain", "What is 2+2? Answer with just the number."),
];

fn outcome(r: &crate::error::Result<(Vec<u32>, bool)>) -> String {
    match r {
        Ok((tokens, _)) => format!("PASS (generated {} tokens)", tokens.len()),
        Err(e) => format!("REJECT: {e}"),
    }
}

#[test]
fn wgpu_reference_choice() {
    let Ok(list) = std::env::var("APR_3757_MODELS") else {
        eprintln!("SKIP: APR_3757_MODELS is not set");
        return;
    };
    for path in list.split(':').filter(|p| !p.is_empty()) {
        let name = std::path::Path::new(path)
            .file_name()
            .map_or(path.to_string(), |f| f.to_string_lossy().into_owned());
        let Ok(mapped) = MappedGGUFModel::from_path(path) else {
            eprintln!("[wgpu-ref] {name}: will not map");
            continue;
        };
        let Ok(model) = OwnedQuantizedModel::from_mapped(&mapped) else {
            eprintln!("[wgpu-ref] {name}: not a dense model");
            continue;
        };
        let gen = QuantizedGenerateConfig::deterministic(4);
        for (pname, text) in PROMPTS {
            let Some(tokens) = mapped.model.encode(text) else {
                continue;
            };
            let q8k = super::try_wgpu_generate(&model, &tokens, &gen, false);
            let fp32 = crate::quantize::with_fp32_activations(|| {
                super::try_wgpu_generate(&model, &tokens, &gen, false)
            });
            eprintln!(
                "[wgpu-ref] {name} | {pname} | first token {} | vs Q8_K CPU: {} | vs FP32 CPU: {}",
                tokens.first().copied().unwrap_or(0),
                outcome(&q8k),
                outcome(&fp32),
            );
        }
    }
}
