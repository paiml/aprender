//! M32c.2.2.2.1.2 + M32d — autoregressive loop for Qwen3-MoE with KV cache.
//!
//! Composes `OwnedQuantizedModel::forward_single_qwen3_moe_with_cache`
//! (M32d) into a per-token decode loop. This is the sibling of
//! `run_gguf_generate` for `qwen3_moe` arch.
//!
//! ## Design
//! Per `qwen3-moe-serve-dispatch-v1` v1.2.0 + M32d playbook:
//!   1. Read MoE config (num_experts, k, intermediate) from GGUF metadata.
//!   2. Build per-layer `Qwen3MoeQuantizedLayer` descriptors via
//!      `load_qwen3_moe_layer` once at start.
//!   3. Allocate `OwnedQuantizedKVCache` sized to `prompt_len + max_tokens`.
//!   4. Prefill: per prompt token, call
//!      `forward_single_qwen3_moe_with_cache`. Cache builds incrementally.
//!      The final iteration's logits are the seed for decode.
//!   5. Decode: per output token, greedy-argmax + call
//!      `forward_single_qwen3_moe_with_cache` for the next-token logits.
//!      Stop on `stop_tokens` or `max_tokens` exhausted.
//!
//! ## Performance
//! Post-M32d: 5-15 tok/s sustained on Qwen3-Coder-30B-A3B (vs ~0.5 tok/s
//! pre-M32d full-prefill-per-token). Each output token amortizes to one
//! per-layer attention (cached K/V read) + one per-layer MoE FFN
//! dispatch — no re-prefill.
//!
//! ## What's NOT in scope
//! - Top-p / top-k / temperature sampling (greedy-only for V1_001 +
//!   V1_004 discharge; sampling is M32 follow-up)
//! - Streaming SSE (cache exposes natural emit-per-token point; one-line
//!   addition once needed — separate contract `qwen3-moe-streaming-sse-v1`)
//! - GPU MoE (separate `qwen3-moe-forward-gpu-v1` track)
//! - Cache rollback / beam search (cache.rollback_to exists; not wired)

use crate::error::{RealizarError, Result};
use crate::gguf::qwen3_moe_load::load_qwen3_moe_layer;
use crate::gguf::{
    MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel, QuantizedGenerateConfig,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Sample the next token from logits per `QuantizedGenerateConfig`.
///
/// Discharges `qwen3-moe-sampling-v1.yaml`:
/// - greedy fallback when `temperature == 0` OR `top_k == 1` (V1_001 + V1_004)
/// - seeded RNG → deterministic across runs with same seed (V1_002)
/// - seed differences produce different outputs (V1_003)
///
/// Mirrors the dense path's `Self::sample_advanced` (in
/// `gguf/inference/fails.rs:100`) but uses a seeded `StdRng`
/// instead of `rand::thread_rng()` for reproducibility.
fn sample_from_logits(
    logits: &[f32],
    config: &QuantizedGenerateConfig,
    rng: &mut StdRng,
) -> Result<u32> {
    if logits.is_empty() {
        return Err(RealizarError::InvalidShape {
            reason: "sample_from_logits: empty logits vector".to_string(),
        });
    }

    // Greedy fallback: temperature == 0 OR top_k == 1
    if config.temperature == 0.0 || config.top_k == 1 {
        return Ok(logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .expect("non-empty logits guaranteed above"));
    }

    // Temperature scaling
    let scaled: Vec<f32> = logits.iter().map(|&x| x / config.temperature).collect();

    // Top-k filter (sort + truncate)
    let mut indexed: Vec<(usize, f32)> = scaled.iter().copied().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    if config.top_k > 0 && config.top_k < indexed.len() {
        indexed.truncate(config.top_k);
    }

    // Top-p (nucleus): keep smallest set with cumulative softmax >= top_p
    if config.top_p > 0.0 && config.top_p < 1.0 {
        let max_val = indexed.first().map_or(0.0, |(_, v)| *v);
        let exp_vals: Vec<f32> = indexed.iter().map(|(_, v)| (v - max_val).exp()).collect();
        let total: f32 = exp_vals.iter().sum();
        if total > 0.0 {
            let mut cumulative = 0.0;
            let mut cutoff = indexed.len();
            for (i, &ev) in exp_vals.iter().enumerate() {
                cumulative += ev / total;
                if cumulative >= config.top_p {
                    cutoff = i + 1;
                    break;
                }
            }
            indexed.truncate(cutoff);
        }
    }

    // Softmax over filtered set + multinomial draw
    let max_val = indexed.first().map_or(0.0, |(_, v)| *v);
    let exp_sum: f32 = indexed.iter().map(|(_, v)| (v - max_val).exp()).sum();
    if exp_sum <= 0.0 {
        // Degenerate softmax: fall back to argmax of filtered set
        return Ok(indexed.first().map_or(0, |(i, _)| *i as u32));
    }

    let r: f32 = rng.gen();
    let mut cumulative = 0.0;
    for (idx, v) in &indexed {
        cumulative += (v - max_val).exp() / exp_sum;
        if cumulative >= r {
            return Ok(*idx as u32);
        }
    }
    Ok(indexed.last().map_or(0, |(i, _)| *i as u32))
}

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

    // M32d: KV cache decode. Sized to fit prompt + max_tokens + small
    // safety buffer. Honors REALIZR_CONTEXT_LENGTH env var (matches dense
    // path's convention; default 4096).
    let env_ctx = std::env::var("REALIZR_CONTEXT_LENGTH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4096);
    let needed = input_tokens.len() + gen_config.max_tokens + 8;
    let max_seq_len = env_ctx.max(needed);
    let mut cache = OwnedQuantizedKVCache::from_config(model.config(), max_seq_len);

    // Seeded RNG for reproducible sampling (qwen3-moe-sampling-v1).
    // Greedy fallback (temperature == 0 OR top_k == 1) doesn't touch
    // the RNG; non-greedy paths consume from it deterministically.
    let mut rng = StdRng::seed_from_u64(gen_config.seed);

    // Prefill: per prompt token, run cache-aware forward. Cache fills
    // incrementally; the LAST iteration's logits seed the decode loop.
    // Position is each token's absolute index (0..prompt_len).
    let mut tokens = input_tokens.to_vec();
    let mut last_logits = Vec::new();
    for (pos, &tok) in input_tokens.iter().enumerate() {
        last_logits = model.forward_single_qwen3_moe_with_cache(
            tok,
            &mut cache,
            pos,
            &moe_layers,
            num_experts,
            num_experts_per_tok,
            moe_intermediate,
            data,
        )?;
    }
    if last_logits.is_empty() {
        return Err(RealizarError::InvalidShape {
            reason: "run_qwen3_moe_generate: prefill produced no logits".to_string(),
        });
    }

    // Decode loop: greedy-sample from `last_logits`, append, then run
    // one more cache-aware forward to seed the next iteration.
    for _step in 0..gen_config.max_tokens {
        let next_token = sample_from_logits(&last_logits, gen_config, &mut rng)?;
        tokens.push(next_token);

        // GH-373-style stop check (matches dense path semantics)
        if gen_config.stop_tokens.contains(&next_token) {
            break;
        }
        if tokens.len() >= max_seq_len {
            // Cache is full; stop before overflow
            break;
        }

        let pos = tokens.len() - 1;
        last_logits = model.forward_single_qwen3_moe_with_cache(
            next_token,
            &mut cache,
            pos,
            &moe_layers,
            num_experts,
            num_experts_per_tok,
            moe_intermediate,
            data,
        )?;
    }

    Ok(tokens)
}

#[cfg(test)]
mod sample_from_logits_tests {
    //! Unit tests for the `qwen3-moe-sampling-v1.yaml` falsifiers
    //! against `sample_from_logits` directly. Run without a real
    //! Qwen3-MoE GGUF (uses synthetic logits arrays). Complements the
    //! env-gated integration tests in
    //! `crates/aprender-serve/tests/qwen3_moe_sampling_v1.rs` which
    //! validate the same invariants on top of a real model.
    use super::*;

    fn mk_config(temperature: f32, top_k: usize, top_p: f32, seed: u64) -> QuantizedGenerateConfig {
        QuantizedGenerateConfig {
            max_tokens: 1,
            temperature,
            top_k,
            top_p,
            seed,
            stop_tokens: Vec::new(),
            ..QuantizedGenerateConfig::default()
        }
    }

    /// V1_001: greedy fallback (temperature == 0) returns argmax deterministically.
    #[test]
    fn v1_001_temperature_zero_is_argmax_deterministic() {
        let logits = vec![1.0, 5.0, 2.0, 4.0, 3.0]; // argmax = index 1
        let cfg = mk_config(0.0, 50, 1.0, 42);
        for _ in 0..5 {
            let mut rng = StdRng::seed_from_u64(cfg.seed);
            let token = sample_from_logits(&logits, &cfg, &mut rng).unwrap();
            assert_eq!(token, 1, "V1_001: temperature=0 must return argmax");
        }
    }

    /// V1_001: top_k == 1 ALSO triggers greedy fallback (independent path).
    #[test]
    fn v1_001_top_k_one_is_argmax_deterministic() {
        let logits = vec![3.0, 1.0, 7.0, 2.0, 5.0]; // argmax = index 2
        let cfg = mk_config(5.0 /* high temp ignored */, 1, 1.0, 42);
        for _ in 0..5 {
            let mut rng = StdRng::seed_from_u64(cfg.seed);
            let token = sample_from_logits(&logits, &cfg, &mut rng).unwrap();
            assert_eq!(token, 2, "V1_001: top_k=1 must return argmax");
        }
    }

    /// V1_002: temperature > 0 with fixed seed returns the same token across runs.
    #[test]
    fn v1_002_seeded_rng_is_reproducible() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let cfg = mk_config(0.7, 50, 0.95, 42);

        let mut tokens = Vec::new();
        for _ in 0..5 {
            let mut rng = StdRng::seed_from_u64(cfg.seed);
            tokens.push(sample_from_logits(&logits, &cfg, &mut rng).unwrap());
        }
        let first = tokens[0];
        for (i, &t) in tokens.iter().enumerate() {
            assert_eq!(
                t, first,
                "V1_002: seed=42 must produce same token; iter {i} got {t}, expected {first}"
            );
        }
    }

    /// V1_003: different seeds produce different tokens on average. Single-token
    /// inevitably has collisions, so probe across 32 seeds and assert at least
    /// 3 distinct tokens (very loose bound; collisions on 1-of-8 logits with
    /// reasonable temp are rare).
    #[test]
    fn v1_003_different_seeds_diverge() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let cfg_template = mk_config(1.5 /* spread mass */, 0, 1.0, 0);

        let mut tokens = std::collections::HashSet::new();
        for seed in 0..32u64 {
            let mut cfg = cfg_template.clone();
            cfg.seed = seed;
            let mut rng = StdRng::seed_from_u64(cfg.seed);
            tokens.insert(sample_from_logits(&logits, &cfg, &mut rng).unwrap());
        }
        assert!(
            tokens.len() >= 3,
            "V1_003: 32 seeds must produce ≥ 3 distinct tokens (got {})",
            tokens.len()
        );
    }

    /// V1_004: top_k=1 with HIGH temperature == greedy (regardless of RNG state).
    #[test]
    fn v1_004_top_k_one_equals_pure_greedy() {
        let logits = vec![0.1, 0.2, 0.3, 0.4, 0.5, 99.0, 0.6, 0.7]; // argmax = 5
        let high_temp_top_k_one = mk_config(50.0, 1, 1.0, 12345);
        let pure_greedy = mk_config(0.0, 1, 1.0, 999_999);

        let mut rng_a = StdRng::seed_from_u64(high_temp_top_k_one.seed);
        let mut rng_b = StdRng::seed_from_u64(pure_greedy.seed);
        let a = sample_from_logits(&logits, &high_temp_top_k_one, &mut rng_a).unwrap();
        let b = sample_from_logits(&logits, &pure_greedy, &mut rng_b).unwrap();
        assert_eq!(a, b, "V1_004: top_k=1 == pure greedy regardless of temperature");
        assert_eq!(a, 5, "V1_004: argmax of logits is at index 5");
    }

    /// Edge case: empty logits returns InvalidShape error (no panic).
    #[test]
    fn empty_logits_returns_error() {
        let cfg = mk_config(0.7, 50, 0.95, 42);
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let result = sample_from_logits(&[], &cfg, &mut rng);
        assert!(result.is_err(), "empty logits must error, not panic");
    }

    /// Edge case: top_p=1.0 has no effect (just regular sampling).
    #[test]
    fn top_p_one_is_no_op() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cfg_with_top_p = mk_config(0.7, 0 /* no top_k cap */, 1.0, 42);
        let cfg_no_top_p = mk_config(0.7, 0, 0.0 /* sentinel: not active */, 42);

        let mut rng_a = StdRng::seed_from_u64(cfg_with_top_p.seed);
        let mut rng_b = StdRng::seed_from_u64(cfg_no_top_p.seed);
        let a = sample_from_logits(&logits, &cfg_with_top_p, &mut rng_a).unwrap();
        let b = sample_from_logits(&logits, &cfg_no_top_p, &mut rng_b).unwrap();
        assert_eq!(a, b, "top_p=1.0 must equal top_p=0.0 (both no-op)");
    }
}
