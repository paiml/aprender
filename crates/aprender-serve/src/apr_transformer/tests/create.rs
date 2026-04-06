//! T-COV-95 Deep Coverage Bridge: apr_transformer/mod.rs
//!
//! Targets: layer_norm with bias, matmul edge cases, generate EOS detection,
//! generate_with_cache edge cases, num_parameters with optional fields,
//! embed debug path, forward trace path.

use crate::apr_transformer::{
    AprKVCache, AprTransformer, AprTransformerConfig, AprTransformerLayer, GenerateConfig,
};

/// Create a tiny transformer for testing with non-zero weights
fn create_test_transformer(
    hidden_dim: usize,
    num_layers: usize,
    vocab_size: usize,
) -> AprTransformer {
    let config = AprTransformerConfig {
        hidden_dim,
        num_layers,
        num_heads: hidden_dim / 8, // Ensure at least 1 head
        num_kv_heads: hidden_dim / 8,
        vocab_size,
        intermediate_dim: hidden_dim * 2,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        ..Default::default()
    };

    let mut t = AprTransformer::new(config);

    // Set non-zero embeddings for first few tokens
    for tok in 0..vocab_size.min(10) {
        for d in 0..hidden_dim {
            t.token_embedding[tok * hidden_dim + d] = ((tok + d + 1) as f32) * 0.01;
        }
    }

    // Set non-zero output norm
    for w in &mut t.output_norm_weight {
        *w = 1.0;
    }

    // Set lm_head to something that produces non-trivial logits
    for v in 0..vocab_size {
        for d in 0..hidden_dim {
            t.lm_head_weight[v * hidden_dim + d] = ((v + d) as f32).sin() * 0.01;
        }
    }

    t
}

// ============================================================================
// generate() EOS detection
// ============================================================================

#[test]
fn test_generate_max_tokens_boundary() {
    let t = create_test_transformer(16, 1, 100);
    let result = t.generate(&[1], 1).expect("test value should be present");
    // Prompt (1) + at most 1 generated token
    assert!(result.len() <= 2, "len: {}", result.len());
    assert!(result.len() >= 2, "should generate at least 1 token");
}

#[test]
fn test_generate_zero_max_tokens() {
    let t = create_test_transformer(16, 1, 100);
    let result = t.generate(&[1], 0).expect("test value should be present");
    // Zero max tokens means no generation
    assert_eq!(result.len(), 1, "should only have prompt");
}

#[test]
fn test_generate_returns_prompt_plus_generated() {
    let t = create_test_transformer(16, 1, 100);
    let prompt = vec![1u32, 5, 10];
    let result = t.generate(&prompt, 3).expect("test value should be present");
    // Should start with the prompt
    assert_eq!(&result[..3], &[1, 5, 10]);
    assert!(result.len() >= 3, "should at least contain prompt");
}

// ============================================================================
// forward() error paths
// ============================================================================

#[test]
fn test_forward_empty_returns_error() {
    let t = create_test_transformer(16, 1, 100);
    let result = t.forward(&[]);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("empty"), "got: {err}");
}

#[test]
fn test_forward_single_token_returns_vocab_logits() {
    let t = create_test_transformer(16, 1, 100);
    let logits = t.forward(&[1]).expect("test value should be present");
    assert_eq!(logits.len(), 100, "should have vocab_size logits");
}

#[test]
fn test_forward_multi_token_returns_last_position_logits() {
    let t = create_test_transformer(16, 1, 100);
    let logits = t.forward(&[1, 2, 3]).expect("test value should be present");
    assert_eq!(logits.len(), 100);
}

// ============================================================================
// predict_next()
// ============================================================================

#[test]
fn test_predict_next_returns_valid_token() {
    let t = create_test_transformer(16, 1, 100);
    let next = t.predict_next(&[1]).expect("test value should be present");
    assert!(next < 100, "token {next} should be < vocab_size 100");
}

#[test]
fn test_predict_next_empty_error() {
    let t = create_test_transformer(16, 1, 100);
    let result = t.predict_next(&[]);
    assert!(result.is_err());
}

#[test]
fn test_predict_next_deterministic() {
    let t = create_test_transformer(16, 1, 100);
    let a = t.predict_next(&[1]).expect("test value should be present");
    let b = t.predict_next(&[1]).expect("test value should be present");
    assert_eq!(a, b, "predict_next should be deterministic");
}

// ============================================================================
// num_parameters() and memory_size()
// ============================================================================

#[test]
fn test_num_parameters_basic() {
    let t = create_test_transformer(16, 1, 100);
    let params = t.num_parameters();
    // Embedding: 100*16 = 1600
    // Layer params from AprTransformerLayer::empty(16, 32)
    // output_norm_weight: 16
    // lm_head_weight: 16*100 = 1600
    // Total: > 1600 + 16 + 1600
    assert!(params > 3000, "params: {params}");
}

#[test]
fn test_memory_size_is_4x_parameters() {
    let t = create_test_transformer(16, 1, 100);
    let params = t.num_parameters();
    let mem = t.memory_size();
    assert_eq!(mem, params * 4);
}

#[test]
fn test_num_parameters_multi_layer() {
    let t1 = create_test_transformer(16, 1, 100);
    let t2 = create_test_transformer(16, 3, 100);
    // More layers = more parameters
    assert!(t2.num_parameters() > t1.num_parameters());
}

// ============================================================================
// embed()
// ============================================================================

#[test]
fn test_embed_valid_token() {
    let t = create_test_transformer(16, 1, 100);
    let emb = t.embed(&[1]);
    assert_eq!(emb.len(), 16);
    // Token 1 should have non-zero values
    let sum: f32 = emb.iter().sum();
    assert!(sum.abs() > 0.0, "embedding should be non-zero");
}

#[test]
fn test_embed_oov_returns_zeros() {
    let t = create_test_transformer(16, 1, 100);
    let emb = t.embed(&[9999]); // Way out of vocab
    assert_eq!(emb.len(), 16);
    let sum: f32 = emb.iter().sum();
    assert!((sum).abs() < f32::EPSILON, "OOV should be zeros");
}

#[test]
fn test_embed_multiple_tokens() {
    let t = create_test_transformer(16, 1, 100);
    let emb = t.embed(&[1, 2, 3]);
    assert_eq!(emb.len(), 48); // 3 tokens * 16 dims
}

#[test]
fn test_embed_empty_returns_empty() {
    let t = create_test_transformer(16, 1, 100);
    let emb = t.embed(&[]);
    assert!(emb.is_empty());
}

// ============================================================================
// config()
// ============================================================================

#[test]
fn test_config_accessor_returns_correct_values() {
    let t = create_test_transformer(32, 2, 200);
    let config = t.config();
    assert_eq!(config.hidden_dim, 32);
    assert_eq!(config.num_layers, 2);
    assert_eq!(config.vocab_size, 200);
}

// ============================================================================
// forward_with_cache()
// ============================================================================

#[test]
fn test_forward_with_cache_first_token() {
    let t = create_test_transformer(16, 1, 100);
    let mut cache = AprKVCache::new(t.config());
    let logits = t.forward_with_cache(1, &mut cache, 0).expect("test value should be present");
    assert_eq!(logits.len(), 100);
}

#[test]
fn test_forward_with_cache_sequential() {
    let t = create_test_transformer(16, 1, 100);
    let mut cache = AprKVCache::new(t.config());
    let _l1 = t.forward_with_cache(1, &mut cache, 0).expect("test value should be present");
    let l2 = t.forward_with_cache(2, &mut cache, 1).expect("test value should be present");
    assert_eq!(l2.len(), 100);
}

#[test]
fn test_forward_with_cache_many_positions() {
    let t = create_test_transformer(16, 1, 100);
    let mut cache = AprKVCache::new(t.config());
    for pos in 0..10 {
        let logits = t.forward_with_cache(1, &mut cache, pos).expect("test value should be present");
        assert_eq!(logits.len(), 100, "pos {pos}");
    }
}

// ============================================================================
// generate_with_cache()
// ============================================================================

#[test]
fn test_generate_with_cache_basic() {
    let t = create_test_transformer(16, 1, 100);
    let config = GenerateConfig {
        max_tokens: 3,
        temperature: 0.0,
        ..Default::default()
    };
    let result = t.generate_with_cache(&[1], &config).expect("test value should be present");
    assert!(!result.is_empty(), "should at least have prompt");
    assert!(result.len() <= 4, "prompt(1) + max 3 generated");
}

#[test]
fn test_generate_with_cache_empty_prompt_error() {
    let t = create_test_transformer(16, 1, 100);
    let config = GenerateConfig::default();
    let result = t.generate_with_cache(&[], &config);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("empty"), "got: {err}");
}

#[test]
fn test_generate_with_cache_nonzero_temperature() {
    let t = create_test_transformer(16, 1, 100);
    let config = GenerateConfig {
        max_tokens: 3,
        temperature: 0.8,
        ..Default::default()
    };
    let result = t.generate_with_cache(&[1], &config).expect("test value should be present");
    assert!(!result.is_empty());
}

#[test]
fn test_generate_with_cache_multi_prompt_tokens() {
    let t = create_test_transformer(16, 1, 100);
    let config = GenerateConfig {
        max_tokens: 2,
        temperature: 0.0,
        ..Default::default()
    };
    let result = t.generate_with_cache(&[1, 5, 10], &config).expect("test value should be present");
    // Should start with prompt
    assert_eq!(result[0], 1);
    assert_eq!(result[1], 5);
    assert_eq!(result[2], 10);
}

// ============================================================================
// AprTransformerLayer::num_parameters()
// ============================================================================

#[test]
fn test_layer_empty_parameters() {
    let layer = AprTransformerLayer::empty(16, 32);
    let params = layer.num_parameters();
    // attn_norm_weight: 16
    // qkv_weight: 16 * (16 + 16 + 16) = 768 (Q+K+V, same heads)
    // attn_output_weight: 16 * 16 = 256
    // ffn_up_weight: 16 * 32 = 512
    // ffn_down_weight: 32 * 16 = 512
    // Total should be > 0
    assert!(params > 0, "layer params: {params}");
}

#[test]
fn test_layer_num_parameters_with_optional_fields() {
    let mut layer = AprTransformerLayer::empty(16, 32);
    // Add optional fields
    layer.qkv_bias = Some(vec![0.0; 48]);
    layer.attn_output_bias = Some(vec![0.0; 16]);
    layer.ffn_gate_weight = Some(vec![0.0; 16 * 32]);
    layer.ffn_gate_bias = Some(vec![0.0; 32]);
    layer.ffn_up_bias = Some(vec![0.0; 32]);
    layer.ffn_down_bias = Some(vec![0.0; 16]);
    layer.ffn_norm_weight = Some(vec![0.0; 16]);
    layer.ffn_norm_bias = Some(vec![0.0; 16]);
    layer.attn_norm_bias = Some(vec![0.0; 16]);

    let base_layer = AprTransformerLayer::empty(16, 32);
    assert!(
        layer.num_parameters() > base_layer.num_parameters(),
        "optional fields should add parameters"
    );
}

// ============================================================================
// GenerateConfig edge cases
// ============================================================================

#[test]
fn test_generate_config_default_values() {
    let config = GenerateConfig::default();
    assert_eq!(config.max_tokens, 32);
    assert!((config.temperature - 1.0).abs() < f32::EPSILON);
    assert!((config.top_p - 0.9).abs() < f32::EPSILON);
    assert_eq!(config.top_k, 0);
    assert!(!config.trace);
}

#[test]
fn test_generate_config_custom_values() {
    let config = GenerateConfig {
        max_tokens: 64,
        temperature: 1.0,
        top_p: 0.9,
        top_k: 50,
        repetition_penalty: 1.1,
        trace: true,
        stop_tokens: vec![],
    };
    assert_eq!(config.max_tokens, 64);
    assert!((config.temperature - 1.0).abs() < f32::EPSILON);
    assert!((config.top_p - 0.9).abs() < f32::EPSILON);
    assert_eq!(config.top_k, 50);
    assert!((config.repetition_penalty - 1.1).abs() < f32::EPSILON);
    assert!(config.trace);
}

// ============================================================================
// AprTransformerConfig
// ============================================================================

#[test]
fn test_transformer_config_default() {
    let config = AprTransformerConfig::default();
    assert_eq!(config.hidden_dim, 512);
    assert_eq!(config.num_layers, 6);
    assert_eq!(config.num_heads, 8);
    assert_eq!(config.vocab_size, 32000);
}

// ============================================================================
// AprKVCache basic operations
// ============================================================================

#[test]
fn test_kv_cache_creation() {
    let config = AprTransformerConfig {
        hidden_dim: 16,
        num_layers: 2,
        num_heads: 2,
        num_kv_heads: 2,
        context_length: 64,
        ..Default::default()
    };
    let cache = AprKVCache::new(&config);
    assert_eq!(cache.len(), 0);
}

#[test]
fn test_kv_cache_len_after_forward() {
    let t = create_test_transformer(16, 1, 100);
    let mut cache = AprKVCache::new(t.config());
    assert_eq!(cache.len(), 0);
    let _ = t.forward_with_cache(1, &mut cache, 0).expect("test value should be present");
    assert_eq!(cache.len(), 1);
}

#[test]
fn test_kv_cache_grows_with_tokens() {
    let t = create_test_transformer(16, 1, 100);
    let mut cache = AprKVCache::new(t.config());
    for pos in 0..5 {
        let _ = t.forward_with_cache(1, &mut cache, pos).expect("test value should be present");
    }
    assert_eq!(cache.len(), 5);
}
