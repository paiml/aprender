//! #3751 done_when 2, for the DENSE F2 (`validate_gpu_first_token`): which
//! CPU reference does the dense GPU path actually track?
//!
//! The qwen3moe and qwen35 GPU forwards run float GEMVs, so the FP32-activation
//! CPU forward is their exact reference, and switching to it was a correction.
//! The dense CUDA path is not exact: Q6_K runs HwDp4a (int8 activations) and
//! prefill may run FP8. So the choice has to be measured. For each model and
//! prompt this runs the dense F2's OWN probe path (`f2_probe_to_judge`,
//! `f2_select_probe_path`, `f2_gpu_logits_via`) once, and judges the same GPU
//! logits against both CPU references with the runtime rule
//! (`f2_multi_position_report`).
//!
//! Evidence harness: set `APR_3751_DENSE_MODELS` to `:`-separated GGUF paths
//! and run in release under the GPU lock; unset, it does nothing.

use crate::gguf::{MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda};

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

fn verdict(r: &super::F2PositionReport) -> String {
    if r.accepted {
        format!("ACCEPT (min cos {:.6})", r.min_cosine_real)
    } else {
        format!(
            "REJECT at pos {} (cos {:.6}, argmax {} vs {})",
            r.first_bad_pos, r.first_bad_cosine, r.first_bad_gpu_argmax, r.first_bad_cpu_argmax
        )
    }
}

#[test]
fn dense_f2_reference_choice() {
    let Ok(list) = std::env::var("APR_3751_DENSE_MODELS") else {
        eprintln!("SKIP: APR_3751_DENSE_MODELS is not set");
        return;
    };
    for path in list.split(':').filter(|p| !p.is_empty()) {
        let name = std::path::Path::new(path)
            .file_name()
            .map_or(path.to_string(), |f| f.to_string_lossy().into_owned());
        let Ok(mapped) = MappedGGUFModel::from_path(path) else {
            eprintln!("[dense-f2] {name}: will not map");
            continue;
        };
        let Ok(model) = OwnedQuantizedModel::from_mapped(&mapped) else {
            eprintln!("[dense-f2] {name}: not a dense model");
            continue;
        };
        let mut cuda = match OwnedQuantizedModelCuda::with_max_seq_len(model, 0, 2048) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("[dense-f2] {name}: CUDA model would not build: {e}");
                continue;
            },
        };
        for (pname, text) in PROMPTS {
            let Some(tokens) = mapped.model.encode(text) else {
                continue;
            };
            let Some((kv_dim, num_layers, probe)) = super::f2_probe_to_judge(&cuda, &tokens) else {
                eprintln!("[dense-f2] {name} | {pname}: nothing to judge");
                continue;
            };
            let Some(q8k) =
                super::f2_cpu_reference_logits(cuda.model(), &probe, kv_dim, num_layers)
            else {
                continue;
            };
            let fp32 = crate::quantize::with_fp32_activations(|| {
                super::f2_cpu_reference_logits(cuda.model(), &probe, kv_dim, num_layers)
            });
            let Some(fp32) = fp32 else { continue };
            let decode_token = q8k
                .get(probe.len().saturating_sub(1))
                .map_or(0, |l| super::argmax_u32(l));
            let via = super::f2_select_probe_path(
                cuda.executor.gpu_profile.prefill_path().path
                    == crate::cuda::gpu_profile::PrefillPath::Batched,
                probe.len(),
            );
            cuda.executor.reset_kv_cache_gpu();
            let gpu =
                super::f2_gpu_logits_via(&mut cuda, via, &probe, decode_token, kv_dim, num_layers);
            cuda.executor.reset_kv_cache_gpu();
            let Ok(gpu) = gpu else {
                eprintln!("[dense-f2] {name} | {pname}: GPU probe failed");
                continue;
            };
            let a = super::f2_multi_position_report(&q8k, &gpu);
            let b = super::f2_multi_position_report(&fp32, &gpu);
            eprintln!(
                "[dense-f2] {name} | {pname} | via {} | {} positions | vs Q8_K: {} | vs FP32: {}",
                via.as_str(),
                gpu.len(),
                verdict(&a),
                verdict(&b)
            );
        }
    }
}
