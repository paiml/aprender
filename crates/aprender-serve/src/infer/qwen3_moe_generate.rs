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
use crate::gguf::moe_session::{Qwen3MoeForward, Qwen3MoeSession};
use crate::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Sample the next token from logits per `QuantizedGenerateConfig`.
///
/// Discharges `qwen3-moe-sampling-v1.yaml`:
/// - greedy fallback when `temperature == 0` OR `top_k == 1` (V1_001 + V1_004)
/// - seeded RNG → deterministic across runs with same seed (V1_002)
/// - seed differences produce different outputs (V1_003)
///
/// Mirrors the deleted dense `sample_advanced` (#4266) but uses a seeded
/// `StdRng` instead of `rand::rng()` for reproducibility.
pub(crate) fn sample_from_logits(
    logits: &[f32],
    config: &QuantizedGenerateConfig,
    rng: &mut StdRng,
    recent_tokens: &[u32],
) -> Result<u32> {
    if logits.is_empty() {
        return Err(RealizarError::InvalidShape {
            reason: "sample_from_logits: empty logits vector".to_string(),
        });
    }

    // Step 1: Repetition penalty (qwen3-moe-repetition-penalty-v1).
    // Apply BEFORE temperature scaling. Mirrors Candle's
    // apply_repeat_penalty semantics (PMAT-383/384; the
    // dense `sample_advanced`, deleted with fails.rs in #4266).
    // No-op when repeat_penalty == 1.0 OR repeat_last_n == 0.
    let penalized: Vec<f32> =
        if config.repeat_penalty != 1.0 && config.repeat_last_n > 0 && !recent_tokens.is_empty() {
            let mut p: Vec<f32> = logits.to_vec();
            let start = recent_tokens.len().saturating_sub(config.repeat_last_n);
            for &token in &recent_tokens[start..] {
                let idx = token as usize;
                if idx < p.len() {
                    if p[idx] > 0.0 {
                        p[idx] /= config.repeat_penalty;
                    } else {
                        p[idx] *= config.repeat_penalty;
                    }
                }
            }
            p
        } else {
            logits.to_vec()
        };

    // Greedy fallback: temperature == 0 OR top_k == 1 (after repetition penalty)
    if config.temperature == 0.0 || config.top_k == 1 {
        return Ok(crate::sampling::argmax(&penalized));
    }

    // Temperature scaling
    let scaled: Vec<f32> = penalized.iter().map(|&x| x / config.temperature).collect();

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

    let r: f32 = rng.random();
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
/// PMAT-4269 (M1): a thin entry into the one engine — a
/// [`Qwen3MoeSession`](crate::gguf::moe_session::Qwen3MoeSession) owns the
/// prefill, token choice, stop tokens and context budget; this function only
/// opens it and runs one turn.
///
/// # Arguments
/// * `mapped` — the mmapped GGUF (the per-layer expert tensors borrow from it
///   for the length of the call).
/// * `model` — the standard `OwnedQuantizedModel` constructed via
///   `OwnedQuantizedModel::from_mapped`.
/// * `input_tokens` — the prompt token IDs.
/// * `gen_config` — generation config (max_tokens, sampling, stop tokens).
///
/// # Returns
/// Full token sequence including prompt: `[prompt..., generated...]`.
///
/// # Errors
/// - An empty prompt, or one the declared context cannot hold.
/// - Architecture isn't qwen3_moe (caller should dispatch correctly).
/// - MoE config metadata missing (`expert_count`, `expert_used_count`,
///   `expert_feed_forward_length`).
/// - Per-layer MoE descriptor load failure, or a forward pass error.
pub fn run_qwen3_moe_generate(
    mapped: &MappedGGUFModel,
    model: &OwnedQuantizedModel,
    input_tokens: &[u32],
    gen_config: &QuantizedGenerateConfig,
) -> Result<Vec<u32>> {
    run_qwen3_moe_generate_observed(mapped, model, input_tokens, gen_config, &mut || {})
}

/// [`run_qwen3_moe_generate`] with `on_token()` fired as each token is chosen,
/// so a caller can time the prefill boundary (SRV-TIM-001).
///
/// # Errors
/// As [`run_qwen3_moe_generate`].
pub fn run_qwen3_moe_generate_observed(
    mapped: &MappedGGUFModel,
    model: &OwnedQuantizedModel,
    input_tokens: &[u32],
    gen_config: &QuantizedGenerateConfig,
    on_token: &mut dyn FnMut(),
) -> Result<Vec<u32>> {
    let mut session = Qwen3MoeSession::new(Qwen3MoeForward::cpu(mapped, model)?);
    Ok(session
        .generate(input_tokens, gen_config, &mut |_| {
            on_token();
            true
        })?
        .tokens)
}

/// Streaming variant of `run_qwen3_moe_generate` — discharges
/// `qwen3-moe-streaming-sse-v1.yaml` per-token emit requirement.
///
/// `on_token(next_token)` fires for every chosen token, including the one that
/// triggered a stop, before the stop is honoured; returning `false` ends the
/// turn (e.g. client disconnect). The same session turn as the non-streaming
/// variant, so the two emit the same tokens.
pub fn run_qwen3_moe_generate_streaming(
    mapped: &MappedGGUFModel,
    model: &OwnedQuantizedModel,
    input_tokens: &[u32],
    gen_config: &QuantizedGenerateConfig,
    mut on_token: impl FnMut(u32) -> bool,
) -> Result<()> {
    let mut session = Qwen3MoeSession::new(Qwen3MoeForward::cpu(mapped, model)?);
    session.generate(input_tokens, gen_config, &mut on_token)?;
    Ok(())
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
            let token = sample_from_logits(&logits, &cfg, &mut rng, &[]).unwrap();
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
            let token = sample_from_logits(&logits, &cfg, &mut rng, &[]).unwrap();
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
            tokens.push(sample_from_logits(&logits, &cfg, &mut rng, &[]).unwrap());
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
            tokens.insert(sample_from_logits(&logits, &cfg, &mut rng, &[]).unwrap());
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
        let a = sample_from_logits(&logits, &high_temp_top_k_one, &mut rng_a, &[]).unwrap();
        let b = sample_from_logits(&logits, &pure_greedy, &mut rng_b, &[]).unwrap();
        assert_eq!(
            a, b,
            "V1_004: top_k=1 == pure greedy regardless of temperature"
        );
        assert_eq!(a, 5, "V1_004: argmax of logits is at index 5");
    }

    /// Edge case: empty logits returns InvalidShape error (no panic).
    #[test]
    fn empty_logits_returns_error() {
        let cfg = mk_config(0.7, 50, 0.95, 42);
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let result = sample_from_logits(&[], &cfg, &mut rng, &[]);
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
        let a = sample_from_logits(&logits, &cfg_with_top_p, &mut rng_a, &[]).unwrap();
        let b = sample_from_logits(&logits, &cfg_no_top_p, &mut rng_b, &[]).unwrap();
        assert_eq!(a, b, "top_p=1.0 must equal top_p=0.0 (both no-op)");
    }

    // ========================================================================
    // qwen3-moe-repetition-penalty-v1 falsifier tests
    // ========================================================================

    fn mk_config_with_penalty(
        temperature: f32,
        top_k: usize,
        repeat_penalty: f32,
        repeat_last_n: usize,
        seed: u64,
    ) -> QuantizedGenerateConfig {
        QuantizedGenerateConfig {
            max_tokens: 1,
            temperature,
            top_k,
            top_p: 1.0,
            repeat_penalty,
            repeat_last_n,
            seed,
            stop_tokens: Vec::new(),
            ..QuantizedGenerateConfig::default()
        }
    }

    /// V1_001 (repetition penalty): repeat_penalty == 1.0 is a no-op even with
    /// non-empty recent_tokens.
    #[test]
    fn rep_penalty_v1_001_no_op_at_one() {
        let logits = vec![3.0, 5.0, 2.0, 4.0]; // argmax = 1
        let recent = vec![1, 1, 1]; // many repetitions of token 1
        let cfg = mk_config_with_penalty(0.0, 1, 1.0 /* no-op */, 100, 42);

        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let token = sample_from_logits(&logits, &cfg, &mut rng, &recent).unwrap();
        // Without penalty, argmax stays at index 1 even though token 1 is in recent.
        assert_eq!(
            token, 1,
            "V1_001: repeat_penalty=1.0 must be a no-op (argmax stays at 1)"
        );
    }

    /// V1_001 (repetition penalty): repeat_last_n == 0 is a no-op.
    #[test]
    fn rep_penalty_v1_001_no_op_when_repeat_last_n_zero() {
        let logits = vec![3.0, 5.0, 2.0, 4.0];
        let recent = vec![1, 1, 1];
        let cfg = mk_config_with_penalty(
            0.0, 1, 2.0, /* would penalize */
            0,   /* no-op */
            42,
        );

        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let token = sample_from_logits(&logits, &cfg, &mut rng, &recent).unwrap();
        assert_eq!(
            token, 1,
            "V1_001: repeat_last_n=0 must be a no-op (argmax stays at 1)"
        );
    }

    /// V1_002: repeat_penalty > 1.0 down-weights repeated tokens (positive logit branch).
    #[test]
    fn rep_penalty_v1_002_down_weights_repeated() {
        // All positive logits → penalty divides them.
        // logits[1] = 5.0 (would be argmax). recent_tokens = [1, 1] → penalty
        // applied to logit[1] twice: 5.0 / 2.0 / 2.0 = 1.25. New argmax = 3 (4.0).
        let logits = vec![3.0, 5.0, 2.0, 4.0];
        let recent = vec![1, 1]; // token 1 repeated twice
        let cfg = mk_config_with_penalty(0.0, 1, 2.0, 100, 42);

        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let token = sample_from_logits(&logits, &cfg, &mut rng, &recent).unwrap();
        // After penalty: [3.0, 1.25, 2.0, 4.0] → argmax = 3
        assert_eq!(
            token, 3,
            "V1_002: repeat_penalty must shift argmax away from repeated token 1"
        );
    }

    /// V1_002: negative logits get MULTIPLIED by penalty (Candle's convention).
    #[test]
    fn rep_penalty_v1_002_negative_logit_branch() {
        // Mix: logit[2] is negative. Penalty multiplies it (more negative).
        let logits = vec![3.0, 1.0, -2.0, 4.0]; // argmax = 3
        let recent = vec![2]; // token 2 has negative logit
        let cfg = mk_config_with_penalty(0.0, 1, 2.0, 100, 42);

        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let token = sample_from_logits(&logits, &cfg, &mut rng, &recent).unwrap();
        // After penalty: [3.0, 1.0, -4.0, 4.0] → argmax still = 3, but logit[2]
        // is now more strongly suppressed. Confirms the branch ran without
        // accidentally amplifying.
        assert_eq!(token, 3, "V1_002 negative branch: argmax stays at 3");
    }

    /// V1_003: repeat_last_n bounds the penalty window correctly.
    #[test]
    fn rep_penalty_v1_003_window_bounds() {
        // recent_tokens = [1, 1, 1, 1, 1, 1, 1, 1] (token 1 eight times).
        // With repeat_last_n=2, only last 2 are penalized (2 applications).
        // With repeat_last_n=8, all 8 are penalized (8 applications).
        // Use repeat_penalty=1.5; logit[1]=10.0.
        // After 2 penalties: 10.0 / 1.5 / 1.5 = 4.44
        // After 8 penalties: 10.0 / 1.5^8 ≈ 0.39
        let logits = vec![1.0, 10.0, 5.0, 3.0]; // argmax = 1 initially
        let recent = vec![1, 1, 1, 1, 1, 1, 1, 1];

        let cfg_n2 = mk_config_with_penalty(0.0, 1, 1.5, 2, 42);
        let mut rng = StdRng::seed_from_u64(42);
        let token_n2 = sample_from_logits(&logits, &cfg_n2, &mut rng, &recent).unwrap();
        // After 2 penalties: logit[1] = 10.0/1.5/1.5 ≈ 4.44. Still > 5.0? No: < 5.0.
        // So argmax = 2 (logit 5.0).
        assert_eq!(token_n2, 2, "V1_003 n=2: penalty insufficient, argmax = 2");

        let cfg_n8 = mk_config_with_penalty(0.0, 1, 1.5, 8, 42);
        let mut rng = StdRng::seed_from_u64(42);
        let token_n8 = sample_from_logits(&logits, &cfg_n8, &mut rng, &recent).unwrap();
        // After 8 penalties: logit[1] ≈ 0.39. argmax = 2 (logit 5.0).
        assert_eq!(
            token_n8, 2,
            "V1_003 n=8: penalty stronger, still argmax = 2"
        );

        // The two are equivalent at this argmax level, but the underlying logit
        // values differ. Pick a config where they diverge: with smaller initial
        // gap, the deeper penalty matters more.
        let logits_close = vec![4.5, 10.0, 5.0, 3.0];
        let cfg_n2 = mk_config_with_penalty(0.0, 1, 1.5, 2, 42);
        let mut rng = StdRng::seed_from_u64(42);
        let token_close_n2 = sample_from_logits(&logits_close, &cfg_n2, &mut rng, &recent).unwrap();
        // 2 penalties: 10/1.5/1.5 = 4.44. argmax = 2 (5.0).
        assert_eq!(token_close_n2, 2);

        // n=0 means "no penalty" (per backwards-compat invariant).
        let cfg_n0 = mk_config_with_penalty(0.0, 1, 1.5, 0, 42);
        let mut rng = StdRng::seed_from_u64(42);
        let token_n0 = sample_from_logits(&logits_close, &cfg_n0, &mut rng, &recent).unwrap();
        // No penalty: argmax = 1 (10.0).
        assert_eq!(token_n0, 1, "V1_003 n=0: no-op, argmax = 1");
    }
}
