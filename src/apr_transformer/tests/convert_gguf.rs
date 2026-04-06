//! Tests for GGUFTransformer → AprTransformer conversion (apr_transformer/convert.rs)
//!
//! Covers the `From<&GGUFTransformer> for AprTransformer` impl that clones all
//! GGUF weights to F32 for APR format.

use crate::apr_transformer::AprTransformer;
use crate::gguf::{GGUFConfig, GGUFTransformer, GGUFTransformerLayer};

fn make_gguf_config(hidden: usize, layers: usize, heads: usize, kv_heads: usize) -> GGUFConfig {
    GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: hidden,
        num_layers: layers,
        num_heads: heads,
        num_kv_heads: kv_heads,
        vocab_size: 16,
        intermediate_dim: hidden * 4,
        eps: 1e-5,
        rope_theta: 10000.0,
        rope_type: 2,
        context_length: 2048,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    }
}

fn make_gguf_layer_swiglu(
    hidden: usize,
    intermediate: usize,
    kv_heads: usize,
) -> GGUFTransformerLayer {
    let head_dim = hidden / 4; // assume 4 heads
    let kv_dim = kv_heads * head_dim;
    let qkv_dim = hidden + 2 * kv_dim;
    GGUFTransformerLayer {
        attn_norm_weight: vec![1.0; hidden],
        attn_norm_bias: None,
        qkv_weight: vec![0.01; hidden * qkv_dim],
        qkv_bias: None,
        attn_output_weight: vec![0.01; hidden * hidden],
        attn_output_bias: None,
        ffn_gate_weight: Some(vec![0.01; hidden * intermediate]),
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.01; hidden * intermediate],
        ffn_up_bias: None,
        ffn_down_weight: vec![0.01; intermediate * hidden],
        ffn_down_bias: None,
        ffn_norm_weight: Some(vec![1.0; hidden]),
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    }
}

fn make_gguf_layer_gelu(
    hidden: usize,
    intermediate: usize,
    kv_heads: usize,
) -> GGUFTransformerLayer {
    let head_dim = hidden / 4;
    let kv_dim = kv_heads * head_dim;
    let qkv_dim = hidden + 2 * kv_dim;
    GGUFTransformerLayer {
        attn_norm_weight: vec![1.0; hidden],
        attn_norm_bias: Some(vec![0.0; hidden]),
        qkv_weight: vec![0.01; hidden * qkv_dim],
        qkv_bias: Some(vec![0.0; qkv_dim]),
        attn_output_weight: vec![0.01; hidden * hidden],
        attn_output_bias: Some(vec![0.0; hidden]),
        ffn_gate_weight: None,
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.01; hidden * intermediate],
        ffn_up_bias: Some(vec![0.0; intermediate]),
        ffn_down_weight: vec![0.01; intermediate * hidden],
        ffn_down_bias: Some(vec![0.0; hidden]),
        ffn_norm_weight: Some(vec![1.0; hidden]),
        ffn_norm_bias: Some(vec![0.0; hidden]),
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    }
}

// ============================================================================
// From<&GGUFTransformer> for AprTransformer
// ============================================================================

#[test]
fn test_gguf_to_apr_swiglu_preserves_config() {
    let hidden = 16;
    let intermediate = 64;
    let config = make_gguf_config(hidden, 1, 4, 4);
    let layer = make_gguf_layer_swiglu(hidden, intermediate, 4);

    let gguf = GGUFTransformer {
        config,
        token_embedding: vec![0.1; 16 * hidden],
        position_embedding: None,
        layers: vec![layer],
        output_norm_weight: vec![1.0; hidden],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; 16 * hidden],
        lm_head_bias: None,
    };

    let apr = AprTransformer::from(&gguf);

    assert_eq!(apr.config.architecture, "llama");
    assert_eq!(apr.config.hidden_dim, hidden);
    assert_eq!(apr.config.num_layers, 1);
    assert_eq!(apr.config.num_heads, 4);
    assert_eq!(apr.config.num_kv_heads, 4);
    assert_eq!(apr.config.vocab_size, 16);
    assert_eq!(apr.config.intermediate_dim, intermediate);
    assert_eq!(apr.config.rope_theta, 10000.0);
    assert!((apr.config.eps - 1e-5).abs() < 1e-9);
}

#[test]
fn test_gguf_to_apr_swiglu_preserves_weights() {
    let hidden = 8;
    let intermediate = 32;
    let config = make_gguf_config(hidden, 1, 4, 2);
    let layer = make_gguf_layer_swiglu(hidden, intermediate, 2);

    let gguf = GGUFTransformer {
        config,
        token_embedding: vec![0.5; 16 * hidden],
        position_embedding: None,
        layers: vec![layer],
        output_norm_weight: vec![2.0; hidden],
        output_norm_bias: None,
        lm_head_weight: vec![0.02; 16 * hidden],
        lm_head_bias: None,
    };

    let apr = AprTransformer::from(&gguf);

    // Token embedding preserved
    assert_eq!(apr.token_embedding.len(), 16 * hidden);
    assert!((apr.token_embedding[0] - 0.5).abs() < 1e-6);

    // Output norm preserved
    assert_eq!(apr.output_norm_weight.len(), hidden);
    assert!((apr.output_norm_weight[0] - 2.0).abs() < 1e-6);
    assert!(apr.output_norm_bias.is_none());

    // LM head preserved
    assert_eq!(apr.lm_head_weight.len(), 16 * hidden);

    // Layer weights preserved
    assert_eq!(apr.layers.len(), 1);
    let al = &apr.layers[0];
    assert_eq!(al.attn_norm_weight.len(), hidden);
    assert!(al.attn_norm_bias.is_none());
    assert!(al.ffn_gate_weight.is_some());
    assert!(al.ffn_norm_weight.is_some());

    // Q4K layers should be None (conversion creates pure F32)
    assert!(apr.q4k_layers.is_none());
}

#[test]
fn test_gguf_to_apr_gelu_preserves_biases() {
    let hidden = 8;
    let intermediate = 32;
    let config = make_gguf_config(hidden, 1, 4, 4);
    let layer = make_gguf_layer_gelu(hidden, intermediate, 4);

    let gguf = GGUFTransformer {
        config,
        token_embedding: vec![0.1; 16 * hidden],
        position_embedding: None,
        layers: vec![layer],
        output_norm_weight: vec![1.0; hidden],
        output_norm_bias: Some(vec![0.0; hidden]),
        lm_head_weight: vec![0.01; 16 * hidden],
        lm_head_bias: Some(vec![0.0; 16]),
    };

    let apr = AprTransformer::from(&gguf);

    // Biases preserved
    assert!(apr.output_norm_bias.is_some());
    assert!(apr.lm_head_bias.is_some());

    let al = &apr.layers[0];
    assert!(al.attn_norm_bias.is_some());
    assert!(al.qkv_bias.is_some());
    assert!(al.attn_output_bias.is_some());
    assert!(al.ffn_up_bias.is_some());
    assert!(al.ffn_down_bias.is_some());
    assert!(al.ffn_norm_bias.is_some());
    assert!(al.ffn_gate_weight.is_none()); // GELU has no gate
}

#[test]
fn test_gguf_to_apr_multi_layer() {
    let hidden = 8;
    let intermediate = 32;
    let config = make_gguf_config(hidden, 3, 4, 2);
    let layers: Vec<GGUFTransformerLayer> = (0..3)
        .map(|_| make_gguf_layer_swiglu(hidden, intermediate, 2))
        .collect();

    let gguf = GGUFTransformer {
        config,
        token_embedding: vec![0.1; 16 * hidden],
        position_embedding: None,
        layers,
        output_norm_weight: vec![1.0; hidden],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; 16 * hidden],
        lm_head_bias: None,
    };

    let apr = AprTransformer::from(&gguf);
    assert_eq!(apr.layers.len(), 3);
    assert_eq!(apr.config.num_layers, 3);
}

#[test]
fn test_gguf_to_apr_empty_layers() {
    let hidden = 8;
    let config = make_gguf_config(hidden, 0, 4, 4);

    let gguf = GGUFTransformer {
        config,
        token_embedding: vec![0.1; 16 * hidden],
        position_embedding: None,
        layers: vec![],
        output_norm_weight: vec![1.0; hidden],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; 16 * hidden],
        lm_head_bias: None,
    };

    let apr = AprTransformer::from(&gguf);
    assert_eq!(apr.layers.len(), 0);
    assert_eq!(apr.config.num_layers, 0);
}

#[test]
fn test_gguf_to_apr_context_length_preserved() {
    let hidden = 8;
    let mut config = make_gguf_config(hidden, 0, 4, 4);
    config.context_length = 131072;

    let gguf = GGUFTransformer {
        config,
        token_embedding: vec![0.1; 16 * hidden],
        position_embedding: None,
        layers: vec![],
        output_norm_weight: vec![1.0; hidden],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; 16 * hidden],
        lm_head_bias: None,
    };

    let apr = AprTransformer::from(&gguf);
    assert_eq!(apr.config.context_length, 131072);
}
