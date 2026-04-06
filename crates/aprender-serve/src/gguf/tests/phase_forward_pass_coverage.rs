//! Phase 33: Forward pass coverage tests
//!
//! These lib tests illuminate forward/core.rs:
//! - forward() - Full forward pass
//! - forward_cached() - Cached forward pass
//!
//! Located in lib tests to be included in `cargo test --lib` coverage

use crate::gguf::test_helpers::create_test_model_with_config;
use crate::gguf::{GGUFConfig, OwnedQuantizedKVCache};

// =============================================================================
// Forward Pass Tests (forward/core.rs:forward)
// =============================================================================

#[test]
fn test_phase33_forward_basic() {
    // Illuminates: forward/core.rs:forward()
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);
    let result = model.forward(&[42]);

    assert!(result.is_ok(), "forward() should succeed");
    let logits = result.expect("test value should be present");
    assert_eq!(logits.len(), config.vocab_size);
    assert!(
        logits.iter().all(|x| x.is_finite()),
        "Logits should be finite"
    );
}

#[test]
fn test_phase33_forward_multi_token() {
    // Illuminates: forward() with sequence
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);
    let result = model.forward(&[1, 2, 3, 4, 5]);

    assert!(result.is_ok());
    assert_eq!(
        result.expect("test value should be present").len(),
        config.vocab_size
    );
}

#[test]
fn test_phase33_forward_multi_layer() {
    // Illuminates: forward() with multiple layers
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);
    let result = model.forward(&[42]);

    assert!(result.is_ok());
}

// =============================================================================
// Cached Forward Tests (forward/core.rs:forward_cached)
// =============================================================================

#[test]
fn test_phase33_forward_cached_single() {
    // Illuminates: forward_cached()
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);
    let mut cache = OwnedQuantizedKVCache::new(config.num_layers, config.hidden_dim, 128);

    let result = model.forward_cached(42, &mut cache, 0);
    assert!(result.is_ok());
    assert_eq!(
        result.expect("test value should be present").len(),
        config.vocab_size
    );
}

#[test]
fn test_phase33_forward_cached_sequence() {
    // Illuminates: forward_cached() with accumulating cache
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);
    let mut cache = OwnedQuantizedKVCache::new(config.num_layers, config.hidden_dim, 128);

    // Simulate autoregressive generation
    for i in 0..10 {
        let result = model.forward_cached((i % config.vocab_size) as u32, &mut cache, i);
        assert!(
            result.is_ok(),
            "forward_cached at position {} should succeed",
            i
        );
    }
}

#[test]
fn test_phase33_forward_cached_multi_layer() {
    // Illuminates: forward_cached() with multiple layers
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);
    let mut cache = OwnedQuantizedKVCache::new(config.num_layers, config.hidden_dim, 128);

    for i in 0..5 {
        let result = model.forward_cached((i % 50) as u32, &mut cache, i);
        assert!(result.is_ok());
    }
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_phase33_forward_single_layer_single_head() {
    // Minimum viable model
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 32,
        intermediate_dim: 64,
        num_layers: 1,
        num_heads: 1,
        num_kv_heads: 1,
        vocab_size: 50,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);
    let result = model.forward(&[1]);
    assert!(result.is_ok());
}
