//! M32c.2.2.2.1.2 — `run_qwen3_moe_generate`: full inference loop for Qwen3-MoE.
//!
//! Composes M32c.2.2.2.1.1's `OwnedQuantizedModel::forward_qwen3_moe` into
//! an autoregressive token-by-token generation loop. This is the
//! sibling of `run_gguf_generate` for `qwen3_moe` arch.
//!
//! ## Design
//! Per `qwen3-moe-forward-v1` v1.2.0, this function:
//!   1. Reads MoE config (num_experts, k, intermediate) from GGUF metadata.
//!   2. Builds the per-layer `Qwen3MoeQuantizedLayer` descriptors via
//!      `load_qwen3_moe_layer` once at start.
//!   3. Runs a generation loop: for each step, calls
//!      `model.forward_qwen3_moe(...)` with the full token sequence,
//!      greedy-samples (argmax) the next token, appends, repeats.
//!
//! ## Performance note
//! This is a full-prefill-per-token loop (no KV cache). For
//! Qwen3-Coder-30B-A3B that's catastrophically slow (~minutes per
//! token on the cached 17.3 GB GGUF) but CORRECT — produces tokens
//! end-to-end. KV-cache integration is M32d follow-up.
//! M32c.2.2.2.1.4 (live falsifier `apr run -n 8`) accepts that latency
//! since it asserts ANY tokens emit, not throughput.
//!
//! ## What's NOT in scope
//! - KV cache (lazy mmap-borrow path needs separate design)
//! - Top-p / top-k / temperature sampling (greedy-only for the
//!   first-tokens proof point)
//! - Stop tokens (caller can post-process; will add when needed)
//! - Tracing / profiling

use crate::error::{RealizarError, Result};
use crate::gguf::qwen3_moe_load::load_qwen3_moe_layer;
use crate::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};

/// Run autoregressive token generation for a Qwen3-MoE GGUF model.
///
/// # Arguments
/// * `mapped` — the mmapped GGUF (caller holds it for the lifetime of
///   this call; the per-layer expert tensors borrow from it during
///   `forward_qwen3_moe`).
/// * `model` — the standard `OwnedQuantizedModel` constructed via
///   `OwnedQuantizedModel::from_mapped` (post-M32c.2.1, this dispatches
///   to `from_gguf_for_moe` for qwen3_moe arch automatically).
/// * `input_tokens` — the prompt token IDs.
/// * `gen_config` — generation config (max_tokens, sampling params).
///
/// # Returns
/// Full token sequence including prompt: `[prompt..., generated...]`.
///
/// # Errors
/// - Architecture isn't qwen3_moe (caller should dispatch correctly).
/// - MoE config metadata missing (`expert_count`, `expert_used_count`,
///   `expert_feed_forward_length`).
/// - Per-layer MoE descriptor load failure (M32c.1).
/// - Forward pass error (M32c.2.2.2.1.1).
pub fn run_qwen3_moe_generate(
    mapped: &MappedGGUFModel,
    model: &OwnedQuantizedModel,
    input_tokens: &[u32],
    gen_config: &QuantizedGenerateConfig,
) -> Result<Vec<u32>> {
    if input_tokens.is_empty() {
        return Err(RealizarError::InvalidShape {
            reason: "run_qwen3_moe_generate: prompt cannot be empty".to_string(),
        });
    }

    let canonical_arch = crate::tensor_names::normalize_architecture(&model.config().architecture);
    if canonical_arch != "qwen3_moe" {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "run_qwen3_moe_generate: arch '{}' (canonical '{}') is not qwen3_moe — \
                 caller should dispatch to run_gguf_generate instead",
                model.config().architecture,
                canonical_arch
            ),
        });
    }

    // Read MoE config from GGUF metadata
    let num_experts = mapped
        .model
        .expert_count()
        .ok_or_else(|| RealizarError::InvalidShape {
            reason: format!(
                "run_qwen3_moe_generate: missing '{}.expert_count' in GGUF metadata",
                model.config().architecture
            ),
        })?;
    let num_experts_per_tok =
        mapped
            .model
            .expert_used_count()
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: format!(
                    "run_qwen3_moe_generate: missing '{}.expert_used_count' in GGUF metadata",
                    model.config().architecture
                ),
            })?;
    let moe_intermediate =
        mapped
            .model
            .expert_feed_forward_length()
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: format!(
                "run_qwen3_moe_generate: missing '{}.expert_feed_forward_length' in GGUF metadata",
                model.config().architecture
            ),
            })?;

    // Load per-layer MoE descriptors once
    let data = mapped.data();
    let num_layers = model.config().num_layers;
    let mut moe_layers = Vec::with_capacity(num_layers);
    for layer_idx in 0..num_layers {
        moe_layers.push(load_qwen3_moe_layer(&mapped.model, data, layer_idx)?);
    }

    // Generation loop: full-prefill per token (no KV cache; M32d)
    let mut tokens = input_tokens.to_vec();
    for _step in 0..gen_config.max_tokens {
        let logits = model.forward_qwen3_moe(
            &tokens,
            &moe_layers,
            num_experts,
            num_experts_per_tok,
            moe_intermediate,
            data,
        )?;

        // Greedy argmax sampling. Top-k/top-p/temperature are M32 follow-up.
        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: "run_qwen3_moe_generate: empty logits vector".to_string(),
            })?;

        tokens.push(next_token);

        // GH-373-style stop token check (matches dense path semantics)
        if gen_config.stop_tokens.contains(&next_token) {
            break;
        }
    }

    Ok(tokens)
}
