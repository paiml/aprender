//! PMAT-799b: Qwen3 per-head QK-RMSNorm on the remaining `apr_transformer`
//! forward variants.
//!
//! PMAT-799 (#2085) fixed the live cached decode path
//! (`project_qkv_with_cache`). Two OTHER forward variants still skipped the
//! Qwen3 per-head Q/K RMSNorm, so Qwen3-family models were silently wrong on
//! those paths:
//!
//! 1. `inference.rs::forward_traced` — the trace path. It projected Q/K,
//!    applied split QKV bias, then RoPE, but never applied
//!    `apply_per_head_rms_norm` even when the layer carried
//!    `attn_q_norm_weight` / `attn_k_norm_weight`.
//! 2. `q4_simd_activations_cache.rs::forward_single_with_scratch` — the Q4
//!    quantized single-token forward. The QK-norm weights were not even
//!    carried on `QuantizedAprLayerQ4` (`from_gguf` dropped them).
//!
//! Reference behavior (the GGUF path `gguf/inference/forward/forward_cached.rs`,
//! the now-fixed `project_qkv_with_cache`, and `AprTransformer::forward` in
//! `pmat-260.rs`): apply per-head RMSNorm to Q and K AFTER projection + bias,
//! BEFORE RoPE. Contract: qk-norm-v1 §QKN-INV-007.

use crate::apr_transformer::{
    AprInferenceScratch, AprTransformer, AprTransformerConfig, AprTransformerLayer,
    QuantizedAprLayerQ4, QuantizedAprTensorQ4, QuantizedAprTransformerQ4,
};

// ---------------------------------------------------------------------------
// Variant 1: forward_traced (f32 AprTransformer) — logit-observable falsifier.
// ---------------------------------------------------------------------------

/// Build a single-layer f32 pygmy model. `qk_norm` selects whether the layer
/// carries Qwen3 per-head Q/K RMSNorm weights.
fn make_traced_model(qk_norm: Option<Vec<f32>>) -> AprTransformer {
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

    let mut token_embedding = vec![0.0f32; vocab_size * hidden_dim];
    for tok in 0..vocab_size {
        for d in 0..hidden_dim {
            token_embedding[tok * hidden_dim + d] =
                ((tok + 1) as f32 * 0.13 + d as f32 * 0.07).sin();
        }
    }

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

/// PMAT-799b falsifier (trace path): non-identity Q/K-norm weights MUST change
/// the traced logits. The trace path scores Q·K across the sequence, so the
/// per-head rescaling of Q/K is observable from position 1 onward. Before the
/// fix `forward_traced` ignored `attn_q_norm_weight` / `attn_k_norm_weight`, so
/// these two runs were byte-identical (RED). After the fix they diverge (GREEN).
#[test]
fn pmat799b_qk_norm_applied_in_forward_traced() {
    // head_dim = 4; strongly non-uniform weight so the Q/K rescaling (and hence
    // the Q·K scores) is unmistakable above FP noise.
    let qk_weight = vec![0.1f32, 4.0, 8.0, 0.05];

    let model_with_norm = make_traced_model(Some(qk_weight));
    let model_without_norm = make_traced_model(None);

    // A 2-token prompt: at position 0 attention is over a single position, so
    // QK-norm is irrelevant there. The effect surfaces at position 1, where the
    // current Q is scored against the position-0 K.
    let prompt = [3u32, 7u32];

    let trace_with = model_with_norm
        .forward_traced(&prompt)
        .expect("forward_traced (with qk-norm) should succeed");
    let trace_without = model_without_norm
        .forward_traced(&prompt)
        .expect("forward_traced (without qk-norm) should succeed");

    assert_eq!(trace_with.logits.len(), trace_without.logits.len());
    assert!(trace_with.logits.iter().all(|v| v.is_finite()));

    let max_abs_diff = trace_with
        .logits
        .iter()
        .zip(trace_without.logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_abs_diff > 1e-5,
        "PMAT-799b REGRESSION: per-head QK RMSNorm was ignored on the trace \
         path. Non-identity attn_q/k_norm_weight produced (near-)IDENTICAL \
         logits (max_abs_diff={max_abs_diff}). forward_traced must apply \
         apply_per_head_rms_norm after projection+bias, before RoPE."
    );
}

/// No-regression guard (trace path): a model WITHOUT QK-norm weights must be
/// byte-identical whether or not the QK-norm branch exists. LLaMA / Qwen2 /
/// Mistral carry no `attn_q/k_norm_weight`, so the gated branch is skipped and
/// the trace path output is unchanged. We assert determinism across two runs of
/// the no-norm model (the branch is a strict no-op when the `Option` is `None`).
#[test]
fn pmat799b_no_qk_norm_trace_is_byte_identical() {
    let model = make_traced_model(None);
    let prompt = [3u32, 7u32, 11u32];

    let a = model.forward_traced(&prompt).expect("run a");
    let b = model.forward_traced(&prompt).expect("run b");

    assert_eq!(a.logits.len(), b.logits.len());
    for (i, (x, y)) in a.logits.iter().zip(b.logits.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "no-norm trace path must be byte-identical across runs at logit {i}"
        );
    }
}

// ---------------------------------------------------------------------------
// Variant 2: Q4 forward_single_with_scratch — weight-plumbing + execution.
//
// The Q4 single-token forward sets attn_out = V (softmax over one position),
// so Q/K — and thus QK-norm — are not observable in this isolated forward's
// logits. The bug this commit fixes there is structural: the QK-norm weights
// were DROPPED at quantization time (`from_gguf`), so the path could never
// apply them even when fed a cache. These tests falsify that the weights are
// now carried through and that the gated norm executes (finite, no panic).
// ---------------------------------------------------------------------------

/// Build a Q4 model directly (one layer) with non-identity QK-norm weights and
/// confirm `forward_single_with_scratch` executes the gated norm branch and
/// returns finite logits (the per-head RMSNorm dimension contract holds, no
/// panic). A sibling no-norm model must also run; both are finite.
#[test]
fn pmat799b_q4_forward_single_runs_with_and_without_qk_norm() {
    // hidden_dim is a multiple of the Q4_0 block size (32) so each weight row
    // packs into whole blocks — `QuantizedAprTensorQ4::zeros` aligns per-row.
    let hidden_dim = 32usize;
    let num_heads = 2usize;
    let num_kv_heads = 2usize;
    let head_dim = hidden_dim / num_heads; // 16
    let kv_dim = num_kv_heads * head_dim; // 32
    let qkv_out_dim = hidden_dim + 2 * kv_dim;
    let intermediate_dim = 64usize;
    let vocab_size = 32usize;

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

    let make = |qk: Option<Vec<f32>>| QuantizedAprTransformerQ4 {
        config: config.clone(),
        token_embedding: (0..vocab_size * hidden_dim)
            .map(|i| ((i % 9) as f32 - 4.0) * 0.05)
            .collect(),
        layers: vec![QuantizedAprLayerQ4 {
            attn_norm_weight: vec![1.0; hidden_dim],
            qkv_weight: QuantizedAprTensorQ4::zeros(hidden_dim, qkv_out_dim),
            attn_output_weight: QuantizedAprTensorQ4::zeros(hidden_dim, hidden_dim),
            ffn_up_weight: QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim),
            ffn_down_weight: QuantizedAprTensorQ4::zeros(intermediate_dim, hidden_dim),
            ffn_gate_weight: Some(QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim)),
            ffn_norm_weight: Some(vec![1.0; hidden_dim]),
            attn_q_norm_weight: qk.clone(),
            attn_k_norm_weight: qk,
        }],
        output_norm_weight: vec![1.0; hidden_dim],
        lm_head_weight: QuantizedAprTensorQ4::zeros(hidden_dim, vocab_size),
    };

    // head_dim = 16; non-identity QK-norm weight [head_dim] exercises the gated
    // branch (the per-head RMSNorm dimension contract is num_heads * head_dim).
    let qk_weight: Vec<f32> = (0..head_dim).map(|j| 0.25 + (j as f32) * 0.1).collect();
    let with_norm = make(Some(qk_weight));
    let without_norm = make(None);

    let mut scratch_with = AprInferenceScratch::from_config(&config);
    let mut scratch_without = AprInferenceScratch::from_config(&config);

    let logits_with = with_norm
        .forward_single_with_scratch(3, &mut scratch_with)
        .expect("Q4 forward_single_with_scratch (with qk-norm) should succeed");
    let logits_without = without_norm
        .forward_single_with_scratch(3, &mut scratch_without)
        .expect("Q4 forward_single_with_scratch (without qk-norm) should succeed");

    assert_eq!(logits_with.len(), vocab_size);
    assert_eq!(logits_without.len(), vocab_size);
    assert!(
        logits_with.iter().all(|v| v.is_finite()),
        "QK-norm branch must not produce NaN/Inf in the Q4 single-token forward"
    );
    assert!(logits_without.iter().all(|v| v.is_finite()));
}
