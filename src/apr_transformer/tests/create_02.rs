//! T-COV-95 Coverage Bridge: apr_transformer/mod.rs
//!
//! Targets: generate, num_parameters, memory_size, embed, layer_norm, matmul
//! error paths and edge cases.

use crate::apr_transformer::{AprTransformer, AprTransformerConfig, AprTransformerLayer};

// ============================================================================
// Helper: Create minimal test transformer
// ============================================================================

fn create_test_transformer(
    hidden_dim: usize,
    vocab_size: usize,
    num_layers: usize,
) -> AprTransformer {
    let num_heads = 4;
    let num_kv_heads = 4;
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim,
        num_layers,
        num_heads,
        num_kv_heads,
        vocab_size,
        intermediate_dim: hidden_dim * 4,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };

    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    // QKV = Q (hidden_dim) + K (kv_dim) + V (kv_dim)
    let qkv_out_dim = hidden_dim + kv_dim + kv_dim;
    let intermediate = hidden_dim * 4;

    AprTransformer {
        config,
        token_embedding: vec![0.1; vocab_size * hidden_dim],
        layers: (0..num_layers)
            .map(|_| AprTransformerLayer {
                attn_norm_weight: vec![1.0; hidden_dim],
                attn_norm_bias: None,
                qkv_weight: vec![0.01; qkv_out_dim * hidden_dim],
                qkv_bias: None,
                attn_output_weight: vec![0.01; hidden_dim * hidden_dim],
                attn_output_bias: None,
                ffn_gate_weight: Some(vec![0.01; intermediate * hidden_dim]),
                ffn_gate_bias: None,
                ffn_up_weight: vec![0.01; intermediate * hidden_dim],
                ffn_up_bias: None,
                ffn_down_weight: vec![0.01; hidden_dim * intermediate],
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
            })
            .collect(),
        output_norm_weight: vec![1.0; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; vocab_size * hidden_dim],
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    }
}

// ============================================================================
// num_parameters tests
// ============================================================================

#[test]
fn test_num_parameters_empty_model() {
    let transformer = create_test_transformer(64, 100, 0);
    let params = transformer.num_parameters();
    // token_embedding: 100 * 64 = 6400
    // output_norm: 64
    // lm_head: 100 * 64 = 6400
    assert_eq!(params, 6400 + 64 + 6400);
}

#[test]
fn test_num_parameters_with_layers() {
    let transformer = create_test_transformer(64, 100, 2);
    let params = transformer.num_parameters();
    assert!(params > 6400 + 64 + 6400); // Should be more than empty model
}

#[test]
fn test_num_parameters_with_biases() {
    let mut transformer = create_test_transformer(64, 100, 1);
    transformer.output_norm_bias = Some(vec![0.0; 64]);
    transformer.lm_head_bias = Some(vec![0.0; 100]);

    let params = transformer.num_parameters();
    // Should include bias parameters
    assert!(params > 6400 + 64 + 6400 + 64 + 100);
}

// ============================================================================
// memory_size tests
// ============================================================================

#[test]
fn test_memory_size_calculation() {
    let transformer = create_test_transformer(64, 100, 0);
    let memory = transformer.memory_size();
    let params = transformer.num_parameters();
    assert_eq!(memory, params * 4); // F32 = 4 bytes
}

#[test]
fn test_memory_size_with_layers() {
    let transformer = create_test_transformer(64, 100, 2);
    let memory = transformer.memory_size();
    assert!(memory > 0);
    assert_eq!(memory % 4, 0); // Should be multiple of 4
}

// ============================================================================
// embed tests
// ============================================================================

#[test]
fn test_embed_single_token() {
    let transformer = create_test_transformer(64, 100, 1);
    let embeddings = transformer.embed(&[0]);
    assert_eq!(embeddings.len(), 64);
}

#[test]
fn test_embed_multiple_tokens() {
    let transformer = create_test_transformer(64, 100, 1);
    let embeddings = transformer.embed(&[0, 1, 2]);
    assert_eq!(embeddings.len(), 64 * 3);
}

#[test]
fn test_embed_out_of_vocab() {
    let transformer = create_test_transformer(64, 100, 1);
    let embeddings = transformer.embed(&[999]); // Out of vocab
    assert_eq!(embeddings.len(), 64);
    // Out of vocab returns zeros
    assert!(embeddings.iter().all(|&x| x == 0.0));
}

#[test]
fn test_embed_mixed_valid_invalid() {
    let transformer = create_test_transformer(64, 100, 1);
    let embeddings = transformer.embed(&[0, 999, 1]); // Valid, invalid, valid
    assert_eq!(embeddings.len(), 64 * 3);
    // First 64 should be non-zero (valid token)
    assert!(embeddings[..64].iter().any(|&x| x != 0.0));
    // Middle 64 should be zeros (invalid token)
    assert!(embeddings[64..128].iter().all(|&x| x == 0.0));
    // Last 64 should be non-zero (valid token)
    assert!(embeddings[128..].iter().any(|&x| x != 0.0));
}

#[test]
fn test_embed_empty_tokens() {
    let transformer = create_test_transformer(64, 100, 1);
    let embeddings = transformer.embed(&[]);
    assert!(embeddings.is_empty());
}

#[test]
fn test_embed_last_valid_token() {
    let transformer = create_test_transformer(64, 100, 1);
    let embeddings = transformer.embed(&[99]); // Last valid token
    assert_eq!(embeddings.len(), 64);
    assert!(embeddings.iter().any(|&x| x != 0.0));
}

// ============================================================================
// generate tests (basic - no forward pass)
// ============================================================================

#[test]
fn test_generate_zero_tokens_returns_prompt() {
    // Use AprTransformer::new which creates valid empty layers
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        num_layers: 0, // No layers = faster test
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        intermediate_dim: 256,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    let result = transformer.generate(&[1], 0);
    assert!(result.is_ok());
    let tokens = result.expect("test value should be present");
    assert_eq!(tokens.len(), 1); // Just the prompt
    assert_eq!(tokens[0], 1);
}

// ============================================================================
// AprTransformerLayer num_parameters tests
// ============================================================================

#[test]
fn test_layer_num_parameters_minimal() {
    let layer = AprTransformerLayer {
        attn_norm_weight: vec![1.0; 64],
        attn_norm_bias: None,
        qkv_weight: vec![0.0; 64 * 192],
        qkv_bias: None,
        attn_output_weight: vec![0.0; 64 * 64],
        attn_output_bias: None,
        ffn_gate_weight: None,
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.0; 256 * 64],
        ffn_up_bias: None,
        ffn_down_weight: vec![0.0; 64 * 256],
        ffn_down_bias: None,
        ffn_norm_weight: None,
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
    let params = layer.num_parameters();
    assert!(params > 0);
}

#[test]
fn test_layer_num_parameters_with_all_biases() {
    let layer = AprTransformerLayer {
        attn_norm_weight: vec![1.0; 64],
        attn_norm_bias: Some(vec![0.0; 64]),
        qkv_weight: vec![0.0; 64 * 192],
        qkv_bias: Some(vec![0.0; 192]),
        attn_output_weight: vec![0.0; 64 * 64],
        attn_output_bias: Some(vec![0.0; 64]),
        ffn_gate_weight: Some(vec![0.0; 256 * 64]),
        ffn_gate_bias: Some(vec![0.0; 256]),
        ffn_up_weight: vec![0.0; 256 * 64],
        ffn_up_bias: Some(vec![0.0; 256]),
        ffn_down_weight: vec![0.0; 64 * 256],
        ffn_down_bias: Some(vec![0.0; 64]),
        ffn_norm_weight: Some(vec![1.0; 64]),
        ffn_norm_bias: Some(vec![0.0; 64]),
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
    let params = layer.num_parameters();
    // Should include all weights and biases
    let expected_min = 64
        + 64
        + 64 * 192
        + 192
        + 64 * 64
        + 64
        + 256 * 64
        + 256
        + 256 * 64
        + 256
        + 64 * 256
        + 64
        + 64
        + 64;
    assert!(params >= expected_min);
}

// ============================================================================
// Config accessors tests
// ============================================================================

#[test]
fn test_config_accessor() {
    let transformer = create_test_transformer(128, 200, 3);
    let config = transformer.config();
    assert_eq!(config.hidden_dim, 128);
    assert_eq!(config.vocab_size, 200);
    assert_eq!(config.num_layers, 3);
}

// ============================================================================
// AprTransformer::new tests
// ============================================================================

#[test]
fn test_new_creates_empty_layers() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        intermediate_dim: 256,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    assert_eq!(transformer.layers.len(), 2);
    assert_eq!(transformer.token_embedding.len(), 100 * 64);
    assert_eq!(transformer.output_norm_weight.len(), 64);
    assert!(transformer.output_norm_bias.is_none());
    assert_eq!(transformer.lm_head_weight.len(), 64 * 100);
    assert!(transformer.lm_head_bias.is_none());
}

#[test]
fn test_new_zero_layers() {
    let config = AprTransformerConfig {
        architecture: "empty".to_string(),
        hidden_dim: 32,
        num_layers: 0,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 50,
        intermediate_dim: 64,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    assert!(transformer.layers.is_empty());
}

// ============================================================================
// AprTransformerLayer::empty tests
// ============================================================================

#[test]
fn test_layer_empty_creates_zero_weights() {
    let layer = AprTransformerLayer::empty(64, 256);
    assert_eq!(layer.attn_norm_weight.len(), 64);
    assert!(layer.attn_norm_weight.iter().all(|&x| x == 1.0)); // Norm weight = 1
    assert!(layer.attn_norm_bias.is_none());
    assert_eq!(layer.ffn_up_weight.len(), 256 * 64);
    assert_eq!(layer.ffn_down_weight.len(), 64 * 256);
}

// ============================================================================
// Clone and Debug trait tests
// ============================================================================

#[test]
fn test_transformer_clone() {
    let transformer = create_test_transformer(32, 50, 1);
    let cloned = transformer.clone();
    assert_eq!(cloned.config.hidden_dim, 32);
    assert_eq!(cloned.config.vocab_size, 50);
    assert_eq!(cloned.layers.len(), 1);
}

#[test]
fn test_transformer_debug() {
    let transformer = create_test_transformer(32, 50, 1);
    let debug_str = format!("{:?}", transformer);
    assert!(debug_str.contains("AprTransformer"));
}

#[test]
fn test_layer_clone() {
    let layer = AprTransformerLayer::empty(64, 256);
    let cloned = layer.clone();
    assert_eq!(cloned.attn_norm_weight.len(), 64);
}

#[test]
fn test_layer_debug() {
    let layer = AprTransformerLayer::empty(64, 256);
    let debug_str = format!("{:?}", layer);
    assert!(debug_str.contains("AprTransformerLayer"));
}

// ============================================================================
// Edge case tests
// ============================================================================

#[test]
fn test_small_hidden_dim() {
    let transformer = create_test_transformer(4, 10, 1);
    assert_eq!(transformer.config.hidden_dim, 4);
    assert!(transformer.num_parameters() > 0);
}

#[test]
fn test_large_vocab_size() {
    let transformer = create_test_transformer(64, 50000, 1);
    assert_eq!(transformer.config.vocab_size, 50000);
    assert_eq!(transformer.token_embedding.len(), 50000 * 64);
}
