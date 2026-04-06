//! T-COV-95 Phase 51: Deep coverage for apr_transformer/mod.rs uncovered lines
//!
//! Covers:
//! - AprTransformer::new() constructor (lines ~1141-1162)
//! - AprTransformer::config() accessor
//! - AprTransformer::embed() with in-vocab and out-of-vocab tokens (lines ~1230-1262)
//! - AprTransformer::num_parameters() and memory_size() (lines ~1209-1226)
//! - AprTransformer::forward() on pygmy model (lines ~1311-1686)
//! - AprTransformer::forward() error on empty input
//! - AprTransformer::forward_traced() on pygmy model (lines ~1704-1940)
//! - AprTransformer::forward_traced() error on empty input
//! - AprTransformer::predict_next() (lines ~1955-1968)
//! - AprTransformer::generate() with EOS stopping (lines ~1180-1205)
//! - TracedForward trait impl
//! - AprTransformer serialization roundtrip (Serialize/Deserialize)
//! - from_apr_bytes() with valid APR v2 binary (lines ~418-1138)
//! - from_apr_bytes() error cases (too small, bad magic, metadata beyond file)
//! - forward_with_cache() on pygmy model (lines ~1983-2509)
//! - generate_with_cache() delegation

use crate::apr_transformer::AprTransformer;
use crate::apr_transformer::{
    ActivationStats, AprKVCache, AprTransformerConfig, AprTransformerLayer, ForwardTrace,
    GenerateConfig, LayerActivation, Q4KLayerWeights, TracedForward,
};

// ============================================================================
// Helper: Create a pygmy (tiny) model for testing forward pass coverage
// ============================================================================

/// Create a minimal AprTransformer with non-zero weights for testing.
/// Uses hidden_dim=8, num_layers=1, num_heads=2, vocab_size=16, intermediate_dim=16
fn make_pygmy_model() -> AprTransformer {
    let hidden_dim = 8;
    let num_heads = 2;
    let num_kv_heads = 2;
    let vocab_size = 16;
    let intermediate_dim = 16;
    let head_dim = hidden_dim / num_heads; // 4
    let kv_dim = num_kv_heads * head_dim; // 8
    let qkv_out_dim = hidden_dim + 2 * kv_dim; // 8 + 16 = 24

    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim,
        num_layers: 1,
        num_heads,
        num_kv_heads,
        vocab_size,
        intermediate_dim,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };

    // Token embedding: identity-like (each token maps to distinct vector)
    let mut token_embedding = vec![0.0f32; vocab_size * hidden_dim];
    for tok in 0..vocab_size {
        for d in 0..hidden_dim {
            token_embedding[tok * hidden_dim + d] = ((tok + d) as f32) * 0.01;
        }
    }

    // QKV weight: small non-zero values [qkv_out_dim, hidden_dim]
    let qkv_weight: Vec<f32> = (0..qkv_out_dim * hidden_dim)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.01)
        .collect();

    // Attention output weight: [hidden_dim, hidden_dim]
    let attn_output_weight: Vec<f32> = (0..hidden_dim * hidden_dim)
        .map(|i| if i % (hidden_dim + 1) == 0 { 0.1 } else { 0.01 })
        .collect();

    // FFN weights: use SwiGLU path (gate + up + down)
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
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
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

    // LM head weight: [hidden_dim, vocab_size] -> we need hidden_dim * vocab_size
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

/// Create a pygmy model WITHOUT SwiGLU (no gate weight) for standard GELU path coverage
fn make_pygmy_model_gelu() -> AprTransformer {
    let hidden_dim = 8;
    let num_heads = 2;
    let num_kv_heads = 2;
    let vocab_size = 16;
    let intermediate_dim = 16;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let qkv_out_dim = hidden_dim + 2 * kv_dim;

    let config = AprTransformerConfig {
        architecture: "test-gelu".to_string(),
        hidden_dim,
        num_layers: 1,
        num_heads,
        num_kv_heads,
        vocab_size,
        intermediate_dim,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };

    let mut token_embedding = vec![0.0f32; vocab_size * hidden_dim];
    for tok in 0..vocab_size {
        for d in 0..hidden_dim {
            token_embedding[tok * hidden_dim + d] = ((tok + d) as f32) * 0.01;
        }
    }

    let qkv_weight: Vec<f32> = (0..qkv_out_dim * hidden_dim)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.01)
        .collect();

    let attn_output_weight: Vec<f32> = (0..hidden_dim * hidden_dim)
        .map(|i| if i % (hidden_dim + 1) == 0 { 0.1 } else { 0.01 })
        .collect();

    // No ffn_gate_weight -> standard GELU MLP path
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
        qkv_bias: Some(vec![0.01; qkv_out_dim]),
        attn_output_weight,
        attn_output_bias: Some(vec![0.001; hidden_dim]),
        ffn_gate_weight: None, // No gate -> GELU path
        ffn_gate_bias: None,
        ffn_up_weight,
        ffn_up_bias: Some(vec![0.001; intermediate_dim]),
        ffn_down_weight,
        ffn_down_bias: Some(vec![0.001; hidden_dim]),
        ffn_norm_weight: None, // No FFN norm -> hidden passed directly
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
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
        output_norm_bias: Some(vec![0.0; hidden_dim]),
        lm_head_weight,
        lm_head_bias: Some(vec![0.0; vocab_size]),
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    }
}

// ============================================================================
// AprTransformer::new()
// ============================================================================

#[test]
fn test_apr_transformer_new_basic() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 32,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        intermediate_dim: 64,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };
    let model = AprTransformer::new(config.clone());

    assert_eq!(model.config.hidden_dim, 32);
    assert_eq!(model.config.num_layers, 2);
    assert_eq!(model.layers.len(), 2);
    assert_eq!(model.token_embedding.len(), 100 * 32);
    assert_eq!(model.output_norm_weight.len(), 32);
    assert_eq!(model.lm_head_weight.len(), 32 * 100);
    assert!(model.output_norm_bias.is_none());
    assert!(model.lm_head_bias.is_none());
    assert!(model.q4k_layers.is_none());
    assert!(model.lm_head_weight_q6k.is_none());
    assert!(model.lm_head_weight_q4k.is_none());
}

#[test]
fn test_apr_transformer_new_output_norm_is_ones() {
    let config = AprTransformerConfig {
        hidden_dim: 16,
        eos_token_id: None,
        ..Default::default()
    };
    let model = AprTransformer::new(config);
    // Output norm weight should be initialized to 1.0
    assert!(model
        .output_norm_weight
        .iter()
        .all(|&w| (w - 1.0).abs() < 1e-6));
}

// ============================================================================
// AprTransformer::config()
// ============================================================================

#[test]
fn test_apr_transformer_config_accessor() {
    let config = AprTransformerConfig {
        architecture: "phi2".to_string(),
        hidden_dim: 256,
        eos_token_id: None,
        ..Default::default()
    };
    let model = AprTransformer::new(config);
    let cfg = model.config();
    assert_eq!(cfg.architecture, "phi2");
    assert_eq!(cfg.hidden_dim, 256);
}

// ============================================================================
// AprTransformer::embed()
// ============================================================================

#[test]
fn test_embed_single_token() {
    let model = make_pygmy_model();
    let embeddings = model.embed(&[0]);
    assert_eq!(embeddings.len(), 8); // hidden_dim = 8
    // Token 0: values should be (0+d)*0.01 for d in 0..8
    for d in 0..8 {
        let expected = (d as f32) * 0.01;
        assert!(
            (embeddings[d] - expected).abs() < 1e-6,
            "embed[{d}]: expected {expected}, got {}",
            embeddings[d]
        );
    }
}

#[test]
fn test_embed_multiple_tokens() {
    let model = make_pygmy_model();
    let embeddings = model.embed(&[0, 1, 2]);
    assert_eq!(embeddings.len(), 3 * 8); // 3 tokens * hidden_dim=8
}

#[test]
fn test_embed_out_of_vocab_returns_zeros() {
    let model = make_pygmy_model();
    // vocab_size=16, so token 999 is out of vocab
    let embeddings = model.embed(&[999]);
    assert_eq!(embeddings.len(), 8);
    assert!(embeddings.iter().all(|&v| v == 0.0));
}

#[test]
fn test_embed_empty_input() {
    let model = make_pygmy_model();
    let embeddings = model.embed(&[]);
    assert!(embeddings.is_empty());
}

// ============================================================================
// AprTransformer::num_parameters() and memory_size()
// ============================================================================

#[test]
fn test_num_parameters_pygmy() {
    let model = make_pygmy_model();
    let params = model.num_parameters();
    // token_embedding: 16 * 8 = 128
    // layer.num_parameters(): attn_norm(8) + qkv(24*8=192) + attn_out(8*8=64)
    //   + ffn_gate(16*8=128) + ffn_up(16*8=128) + ffn_down(8*16=128) + ffn_norm(8)
    //   = 8+192+64+128+128+128+8 = 656
    // output_norm: 8
    // lm_head: 8*16 = 128
    // Total: 128 + 656 + 8 + 128 = 920
    assert!(params > 0, "num_parameters should be positive");
    // Verify exact count
    let expected = 128 + 656 + 8 + 128;
    assert_eq!(
        params, expected,
        "Expected {expected} parameters, got {params}"
    );
}

#[test]
fn test_memory_size_is_4x_params() {
    let model = make_pygmy_model();
    assert_eq!(model.memory_size(), model.num_parameters() * 4);
}

#[test]
fn test_num_parameters_with_bias() {
    let mut model = make_pygmy_model();
    let base_params = model.num_parameters();
    model.output_norm_bias = Some(vec![0.0; 8]);
    model.lm_head_bias = Some(vec![0.0; 16]);
    assert_eq!(model.num_parameters(), base_params + 8 + 16);
}

// ============================================================================
// AprTransformer::forward() - SwiGLU path
// ============================================================================

#[test]
fn test_forward_swiglu_produces_logits() {
    let model = make_pygmy_model();
    let logits = model.forward(&[1]).expect("forward should succeed");
    assert_eq!(logits.len(), 16); // vocab_size
    // Logits should be finite
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "All logits should be finite"
    );
}

#[test]
fn test_forward_swiglu_multi_token() {
    let model = make_pygmy_model();
    let logits = model.forward(&[1, 2, 3]).expect("forward should succeed");
    assert_eq!(logits.len(), 16);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn test_forward_empty_tokens_error() {
    let model = make_pygmy_model();
    let result = model.forward(&[]);
    assert!(result.is_err(), "forward with empty tokens should error");
}

// ============================================================================
// AprTransformer::forward() - GELU path (no gate weight)
// ============================================================================

#[test]
fn test_forward_gelu_path_produces_logits() {
    let model = make_pygmy_model_gelu();
    let logits = model.forward(&[1]).expect("forward should succeed");
    assert_eq!(logits.len(), 16);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn test_forward_gelu_path_multi_token() {
    let model = make_pygmy_model_gelu();
    let logits = model.forward(&[0, 1, 2]).expect("forward should succeed");
    assert_eq!(logits.len(), 16);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn test_forward_gelu_with_biases() {
    // This model has qkv_bias, attn_output_bias, ffn_up_bias, ffn_down_bias,
    // output_norm_bias, and lm_head_bias - covers all bias application paths
    let model = make_pygmy_model_gelu();
    let logits = model.forward(&[5]).expect("forward should succeed");
    assert_eq!(logits.len(), 16);
    assert!(logits.iter().all(|v| v.is_finite()));
}

// ============================================================================
// AprTransformer::forward_traced() - SwiGLU path
// ============================================================================

#[test]
fn test_forward_traced_swiglu_returns_trace() {
    let model = make_pygmy_model();
    let trace = model
        .forward_traced(&[1])
        .expect("forward_traced should succeed");
    assert_eq!(trace.input_tokens, vec![1]);
    assert_eq!(trace.logits.len(), 16);
    assert_eq!(trace.layer_activations.len(), 1);

    // Check embed stats
    assert_eq!(trace.embed_stats.count, 8); // hidden_dim=8, seq_len=1

    // Check layer activation stats
    let layer = &trace.layer_activations[0];
    assert_eq!(layer.layer_idx, 0);
    assert!(layer.attn_norm_stats.count > 0);
    assert!(layer.qkv_stats.count > 0);
    assert!(layer.attn_out_stats.count > 0);
    assert!(layer.ffn_norm_stats.count > 0);
    assert!(layer.ffn_out_stats.count > 0);
    assert!(layer.output_stats.count > 0);

    // Final norm and logits stats
    assert!(trace.final_norm_stats.count > 0);
    assert!(trace.logits_stats.count > 0);
}

include!("forward_traced.rs");
include!("build_minimal.rs");
