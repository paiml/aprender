//! Tests for flash_attention_tiled (PARITY-026)
//!
//! Coverage target: src/gguf/inference/attention.rs flash_attention_tiled (73 uncov, 0%)

use crate::gguf::test_helpers::create_test_model_with_config;
use crate::gguf::GGUFConfig;

fn small_config() -> GGUFConfig {
    GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 1,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    }
}

#[test]
fn test_flash_attention_tiled_no_cache() {
    let model = create_test_model_with_config(&small_config());
    let hidden_dim = model.config.hidden_dim;

    // Query: single position
    let q = vec![1.0f32; hidden_dim];
    let k_cache: Vec<f32> = vec![]; // No cache
    let v_cache: Vec<f32> = vec![];
    let current_k = vec![0.5f32; hidden_dim];
    let current_v = vec![0.3f32; hidden_dim];

    let output = model.flash_attention_tiled(&q, &k_cache, &v_cache, &current_k, &current_v, 64);

    assert_eq!(output.len(), hidden_dim);
    // With no cache, each head attends only to current — output should be current_v per head
    assert!(output.iter().all(|x| x.is_finite()));
    // Since there's only one KV pair (current), attention output should match current_v
    for (o, &expected) in output.iter().zip(current_v.iter()) {
        let diff = (o - expected).abs();
        assert!(diff < 1e-3, "output={o} expected={expected} diff={diff}");
    }
}

#[test]
fn test_flash_attention_tiled_with_cache() {
    let model = create_test_model_with_config(&small_config());
    let hidden_dim = model.config.hidden_dim;

    // Query
    let q = vec![1.0f32; hidden_dim];

    // Cache with 3 previous positions
    let cache_len = 3;
    let k_cache = vec![0.1f32; cache_len * hidden_dim];
    let v_cache = vec![0.2f32; cache_len * hidden_dim];

    let current_k = vec![1.0f32; hidden_dim];
    let current_v = vec![0.5f32; hidden_dim];

    let output = model.flash_attention_tiled(&q, &k_cache, &v_cache, &current_k, &current_v, 64);

    assert_eq!(output.len(), hidden_dim);
    assert!(output.iter().all(|x| x.is_finite()));
    // Output should be a softmax-weighted average of v_cache entries + current_v
    // With uniform-ish queries, should be between 0.2 and 0.5
    for val in &output {
        assert!(*val > 0.0 && *val < 1.0, "val={val} out of expected range");
    }
}

#[test]
fn test_flash_attention_tiled_multi_tile() {
    let model = create_test_model_with_config(&small_config());
    let hidden_dim = model.config.hidden_dim;

    // Use small block_size=2 to force multiple tiles
    let cache_len = 5;
    let k_cache = vec![0.1f32; cache_len * hidden_dim];
    let v_cache = vec![0.2f32; cache_len * hidden_dim];
    let q = vec![1.0f32; hidden_dim];
    let current_k = vec![0.5f32; hidden_dim];
    let current_v = vec![0.3f32; hidden_dim];

    // block_size=2, total_len=6, so 3 tiles
    let output = model.flash_attention_tiled(&q, &k_cache, &v_cache, &current_k, &current_v, 2);

    assert_eq!(output.len(), hidden_dim);
    assert!(output.iter().all(|x| x.is_finite()));
}

#[test]
fn test_flash_attention_tiled_parity_with_standard() {
    let model = create_test_model_with_config(&small_config());
    let hidden_dim = model.config.hidden_dim;

    let q = vec![1.0f32; hidden_dim];
    let cache_len = 2;
    let k_cache = vec![0.3f32; cache_len * hidden_dim];
    let v_cache = vec![0.4f32; cache_len * hidden_dim];
    let current_k = vec![0.7f32; hidden_dim];
    let current_v = vec![0.6f32; hidden_dim];

    // Flash attention result
    let flash_output =
        model.flash_attention_tiled(&q, &k_cache, &v_cache, &current_k, &current_v, 64);

    // Standard attention result
    let standard_output =
        model.attention_with_cache(&q, &k_cache, &v_cache, &current_k, &current_v);

    assert_eq!(flash_output.len(), standard_output.len());
    for (f, s) in flash_output.iter().zip(standard_output.iter()) {
        let diff = (f - s).abs();
        assert!(
            diff < 1e-3,
            "flash={f} standard={s} diff={diff} — parity failure"
        );
    }
}

#[test]
fn test_flash_attention_tiled_large_cache() {
    let model = create_test_model_with_config(&small_config());
    let hidden_dim = model.config.hidden_dim;

    // Large cache with 100 positions, block_size=32
    let cache_len = 100;
    let k_cache: Vec<f32> = (0..cache_len * hidden_dim)
        .map(|i| ((i % 7) as f32) * 0.1)
        .collect();
    let v_cache: Vec<f32> = (0..cache_len * hidden_dim)
        .map(|i| ((i % 11) as f32) * 0.05)
        .collect();

    let q = vec![0.5f32; hidden_dim];
    let current_k = vec![0.3f32; hidden_dim];
    let current_v = vec![0.2f32; hidden_dim];

    let output = model.flash_attention_tiled(&q, &k_cache, &v_cache, &current_k, &current_v, 32);

    assert_eq!(output.len(), hidden_dim);
    assert!(output.iter().all(|x| x.is_finite()));
}

#[test]
fn test_flash_attention_tiled_gqa() {
    let gqa_config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_heads: 8,
        num_kv_heads: 2,
        num_layers: 1,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let model = create_test_model_with_config(&gqa_config);
    let hidden_dim = model.config.hidden_dim;

    let q = vec![1.0f32; hidden_dim];
    let k_cache = vec![0.2f32; 3 * hidden_dim];
    let v_cache = vec![0.3f32; 3 * hidden_dim];
    let current_k = vec![0.5f32; hidden_dim];
    let current_v = vec![0.4f32; hidden_dim];

    let output = model.flash_attention_tiled(&q, &k_cache, &v_cache, &current_k, &current_v, 64);

    assert_eq!(output.len(), hidden_dim);
    assert!(output.iter().all(|x| x.is_finite()));
}
