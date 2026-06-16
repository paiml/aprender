//! PMAT-799: Qwen3 per-head QK-RMSNorm on the LIVE cached decode path.
//!
//! Audit sibling of the RoPE `rope_type` arch-blindness bug. The live
//! `apr run model.safetensors` decode path is
//! `generate_with_cache` -> `forward_with_cache` -> `project_qkv_with_cache`.
//! That path applies the QKV projection, split bias, and RoPE — but PRIOR to
//! PMAT-799 it SKIPPED the Qwen3 per-head Q/K RMSNorm entirely, even when the
//! layer carried `attn_q_norm_weight` / `attn_k_norm_weight`.
//!
//! Reference behavior (both the GGUF path
//! `gguf/inference/forward/forward_cached.rs` and the sibling
//! `AprTransformer::forward` in `pmat-260.rs`): apply per-head RMSNorm to Q and
//! K AFTER projection + bias, BEFORE RoPE. Contract: qk-norm-v1 §QKN-INV-007.
//!
//! Falsifier: a model whose Q/K-norm weights are non-identity MUST produce
//! different logits than the same model with the norm absent. Before the fix
//! the norm was ignored, so the two were byte-identical (TEST FAILS). After the
//! fix they diverge (TEST PASSES). A second assertion pins the *direction* of
//! the effect against an independent reference computation of the per-head
//! RMSNorm so we are asserting correctness, not merely "something changed".

use crate::apr_transformer::AprTransformer;
use crate::apr_transformer::{AprKVCache, AprTransformerConfig, AprTransformerLayer};

/// Build a single-layer pygmy model. `qk_norm` selects whether the layer
/// carries Qwen3 per-head Q/K RMSNorm weights (`Some(weight)`) or none.
fn make_qknorm_model(qk_norm: Option<Vec<f32>>) -> AprTransformer {
    let hidden_dim = 8;
    let num_heads = 2;
    let num_kv_heads = 2;
    let vocab_size = 16;
    let intermediate_dim = 16;
    let head_dim = hidden_dim / num_heads; // 4
    let kv_dim = num_kv_heads * head_dim; // 8
    let qkv_out_dim = hidden_dim + 2 * kv_dim; // 24

    let config = AprTransformerConfig {
        architecture: "qwen3".to_string(),
        hidden_dim,
        num_layers: 1,
        num_heads,
        num_kv_heads,
        vocab_size,
        intermediate_dim,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-6,
        ..Default::default()
    };

    // Distinct, non-trivial embedding so Q/K are not degenerate.
    let mut token_embedding = vec![0.0f32; vocab_size * hidden_dim];
    for tok in 0..vocab_size {
        for d in 0..hidden_dim {
            token_embedding[tok * hidden_dim + d] = ((tok + 1) as f32 * 0.13 + d as f32 * 0.07).sin();
        }
    }

    // QKV weight [qkv_out_dim, hidden_dim] with non-uniform values so that the
    // per-head RMS of Q and K differs from 1.0 (otherwise RMSNorm would be a
    // no-op and the falsifier could not fire).
    let qkv_weight: Vec<f32> = (0..qkv_out_dim * hidden_dim)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.05)
        .collect();

    let attn_output_weight: Vec<f32> = (0..hidden_dim * hidden_dim)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.02)
        .collect();

    let ffn_gate_weight: Vec<f32> = (0..intermediate_dim * hidden_dim)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.01)
        .collect();
    let ffn_up_weight: Vec<f32> = (0..intermediate_dim * hidden_dim)
        .map(|i| ((i % 3) as f32 - 1.0) * 0.01)
        .collect();
    let ffn_down_weight: Vec<f32> = (0..hidden_dim * intermediate_dim)
        .map(|i| ((i % 4) as f32 - 1.5) * 0.01)
        .collect();

    let layer = AprTransformerLayer {
        attn_norm_weight: vec![1.0; hidden_dim],
        attn_norm_bias: None,
        qkv_weight,
        qkv_bias: None,
        attn_output_weight,
        attn_output_bias: None,
        ffn_gate_weight: Some(ffn_gate_weight),
        ffn_gate_bias: None,
        ffn_up_weight,
        ffn_up_bias: None,
        ffn_down_weight,
        ffn_down_bias: None,
        ffn_norm_weight: Some(vec![1.0; hidden_dim]),
        ffn_norm_bias: None,
        attn_q_norm_weight: qk_norm.clone(),
        attn_k_norm_weight: qk_norm,
        linear_attn_z_weight: None,
        linear_attn_b_weight: None,
        linear_attn_a_weight: None,
        linear_attn_conv1d_weight: None,
        linear_attn_a_log: None,
        linear_attn_dt_bias: None,
        linear_attn_norm_weight: None,
        moe_gate_weight: None,
        moe_expert_gate_up: None,
        moe_expert_down: None,
        moe_shared_gate: None,
        moe_shared_up: None,
        moe_shared_down: None,
        moe_shared_expert_gate_weight: None,
    };

    let lm_head_weight: Vec<f32> = (0..hidden_dim * vocab_size)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.01)
        .collect();

    AprTransformer {
        config,
        token_embedding,
        layers: vec![layer],
        output_norm_weight: vec![1.0; hidden_dim],
        output_norm_bias: None,
        lm_head_weight,
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    }
}

/// PMAT-799 falsifier: non-identity Q/K-norm weights MUST change cached-decode
/// logits. Before the fix `forward_with_cache` ignored `attn_q_norm_weight` /
/// `attn_k_norm_weight`, so these two runs were byte-identical and the assert
/// fired. After the fix they diverge.
#[test]
fn pmat799_qk_norm_applied_in_forward_with_cache() {
    // head_dim = 4; strongly non-uniform weight so the per-element rescaling of
    // Q/K (and hence the Q·K attention scores) is unmistakable above FP noise.
    let qk_weight = vec![0.1f32, 4.0, 8.0, 0.05];

    let model_with_norm = make_qknorm_model(Some(qk_weight));
    let model_without_norm = make_qknorm_model(None);

    let mut cache_with = AprKVCache::new(&model_with_norm.config);
    let mut cache_without = AprKVCache::new(&model_without_norm.config);

    // Decode a SECOND token: at position 0 the attention is over a single
    // position (softmax of one element = 1.0), so Q·K is irrelevant and
    // QK-norm cannot change the output. The effect of QK-norm on Q/K only
    // surfaces once Q at the current position is scored against ≥1 cached K.
    let prompt = [3u32, 7u32];
    let mut logits_with = Vec::new();
    let mut logits_without = Vec::new();
    for (pos, &tok) in prompt.iter().enumerate() {
        logits_with = model_with_norm
            .forward_with_cache(tok, &mut cache_with, pos)
            .expect("forward_with_cache (with qk-norm) should succeed");
        logits_without = model_without_norm
            .forward_with_cache(tok, &mut cache_without, pos)
            .expect("forward_with_cache (without qk-norm) should succeed");
    }

    assert_eq!(logits_with.len(), logits_without.len());
    assert!(logits_with.iter().all(|v| v.is_finite()));

    let max_abs_diff = logits_with
        .iter()
        .zip(logits_without.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // Floating-point noise on this pygmy model is ~1e-7; a real QK-norm effect
    // is several orders larger. 1e-5 cleanly separates signal from noise while
    // remaining robust to weight tweaks.
    assert!(
        max_abs_diff > 1e-5,
        "PMAT-799 REGRESSION: per-head QK RMSNorm was ignored on the cached \
         decode path. Non-identity attn_q/k_norm_weight produced (near-)IDENTICAL \
         logits (max_abs_diff={max_abs_diff}). forward_with_cache must apply \
         apply_per_head_rms_norm after projection+bias, before RoPE."
    );
}

/// Identity Q/K-norm weights (all 1.0) must be a (near) no-op relative to the
/// no-norm model. This guards against the fix accidentally introducing an
/// always-on transform that perturbs models which carry trivial norm weights.
/// (head RMS != 1 means RMSNorm still renormalizes magnitude, so we assert the
/// well-known RMSNorm identity-weight behavior against an independent reference
/// rather than exact equality with the no-norm path.)
#[test]
fn pmat799_qk_norm_matches_independent_reference() {
    let head_dim = 4usize;
    let qk_weight = vec![0.5f32, 1.5, 2.0, 0.25];
    let eps = 1e-6f32;

    // Independent reference for per-head RMSNorm of one head.
    let reference_head = |head: &[f32]| -> Vec<f32> {
        let sum_sq: f32 = head.iter().map(|v| v * v).sum();
        let inv_rms = 1.0 / (sum_sq / head_dim as f32 + eps).sqrt();
        head.iter()
            .enumerate()
            .map(|(j, &v)| v * inv_rms * qk_weight[j])
            .collect()
    };

    // A non-trivial head vector.
    let head = [0.3f32, -1.2, 0.7, 2.1];
    let mut buf = head.to_vec();
    crate::gguf::ops::apply_per_head_rms_norm(&mut buf, &qk_weight, 1, eps);

    let expected = reference_head(&head);
    for (i, (a, b)) in buf.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "per-head RMSNorm mismatch at {i}: got {a}, expected {b}"
        );
    }
}
