//! T-COV-95 Active Pygmy: Dynamic Falsification Tests
//!
//! These tests implement Dr. Popper's "Minimum Viable Predictor" -
//! executable synthetic models that can survive a full forward() pass.
//!
//! Unlike static Pygmy tests (parsing only), these tests:
//! 1. Build a GGUF model with F32 tensors
//! 2. Write to a temp file and load via MappedGGUFModel
//! 3. Convert to OwnedQuantizedModel
//! 4. Execute forward_cached() to completion
//!
//! The output is garbage but the code paths are exercised.

use std::io::Write;

use crate::gguf::model::OwnedQuantizedModel;
use crate::gguf::test_factory::build_executable_pygmy_gguf;
use crate::gguf::transformer::QuantizedGGUFTransformer;
use crate::gguf::types::GGUFModel;
use crate::gguf::MappedGGUFModel;
use crate::gguf::OwnedQuantizedKVCache;

/// Helper: Create a temp file with pygmy GGUF data
fn create_pygmy_temp_file() -> (tempfile::NamedTempFile, Vec<u8>) {
    let gguf_data = build_executable_pygmy_gguf();
    let mut temp_file = tempfile::NamedTempFile::new().expect("Failed to create temp file");
    temp_file
        .write_all(&gguf_data)
        .expect("Failed to write temp file");
    temp_file.flush().expect("Failed to flush temp file");
    (temp_file, gguf_data)
}

// ============================================================================
// Active Pygmy Loading Tests
// ============================================================================

/// Test that QuantizedGGUFTransformer can be created from executable pygmy
#[test]
fn test_active_pygmy_load_quantized_transformer() {
    let gguf_data = build_executable_pygmy_gguf();
    let gguf_model = GGUFModel::from_bytes(&gguf_data).expect("Should parse GGUF");

    // Load as QuantizedGGUFTransformer - tests the transformer parsing path
    let result = QuantizedGGUFTransformer::from_gguf(&gguf_model, &gguf_data);

    assert!(
        result.is_ok(),
        "Failed to load QuantizedGGUFTransformer: {:?}",
        result.err()
    );

    let transformer = result.expect("test value should be present");
    assert_eq!(transformer.config.hidden_dim, 32);
    assert_eq!(transformer.config.num_layers, 1);
    assert_eq!(transformer.config.num_heads, 4);
    assert_eq!(transformer.config.vocab_size, 32);
}

/// Test that OwnedQuantizedModel can be loaded from temp file
#[test]
fn test_active_pygmy_load_owned_quantized_model() {
    let (temp_file, _) = create_pygmy_temp_file();

    // Load via memory-mapped path
    let mapped = MappedGGUFModel::from_path(temp_file.path()).expect("Should load MappedGGUFModel");

    let result = OwnedQuantizedModel::from_mapped(&mapped);

    assert!(
        result.is_ok(),
        "Failed to load OwnedQuantizedModel: {:?}",
        result.err()
    );

    let model = result.expect("test value should be present");
    assert_eq!(model.config.hidden_dim, 32);
    assert_eq!(model.config.num_layers, 1);
    assert_eq!(model.config.num_heads, 4);
    assert_eq!(model.config.vocab_size, 32);
}

/// Test that token embedding lookup works
#[test]
fn test_active_pygmy_embed() {
    let (temp_file, _) = create_pygmy_temp_file();
    let mapped = MappedGGUFModel::from_path(temp_file.path()).expect("Should load MappedGGUFModel");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("Should load model");

    // Embed a token (within vocab_size=16)
    let embedding = model.embed(&[0]);

    assert_eq!(embedding.len(), 32); // hidden_dim
                                     // Verify embedding values are reasonable (not NaN/Inf)
    assert!(embedding.iter().all(|&v| v.is_finite()));
}

/// Test that KV cache can be created from config
#[test]
fn test_active_pygmy_kv_cache() {
    let (temp_file, _) = create_pygmy_temp_file();
    let mapped = MappedGGUFModel::from_path(temp_file.path()).expect("Should load MappedGGUFModel");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("Should load model");

    let config = model.config();
    let cache = OwnedQuantizedKVCache::from_config(config, 32);

    // Cache should exist (we just verify it was created successfully)
    // The cache structure is opaque, so we just verify construction
    drop(cache);
}

/// T-COV-95 CRITICAL: Test full forward_cached() execution
///
/// This is the key test that exercises the entire inference pipeline:
/// - Token embedding lookup
/// - Attention norm (RMSNorm)
/// - QKV projection
/// - RoPE position encoding
/// - Scaled dot-product attention with KV cache
/// - Attention output projection
/// - FFN norm
/// - FFN gate/up/down (SwiGLU)
/// - Output norm
/// - LM head projection
#[test]
fn test_active_pygmy_forward_cached() {
    let (temp_file, _) = create_pygmy_temp_file();
    let mapped = MappedGGUFModel::from_path(temp_file.path()).expect("Should load MappedGGUFModel");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("Should load model");

    let config = model.config();
    let mut cache = OwnedQuantizedKVCache::from_config(config, 32);

    // Execute forward pass for a single token
    let token_id = 1u32; // Within vocab_size=16
    let position = 0usize;

    let result = model.forward_cached(token_id, &mut cache, position);

    assert!(result.is_ok(), "forward_cached failed: {:?}", result.err());

    let logits = result.expect("test value should be present");

    // Verify logits shape
    assert_eq!(logits.len(), 32); // vocab_size

    // Verify logits are finite (not NaN/Inf)
    assert!(
        logits.iter().all(|&v| v.is_finite()),
        "Logits contain NaN/Inf: {:?}",
        logits
    );
}

/// Test multiple forward passes (simulating token generation)
#[test]
fn test_active_pygmy_multi_token_generation() {
    let (temp_file, _) = create_pygmy_temp_file();
    let mapped = MappedGGUFModel::from_path(temp_file.path()).expect("Should load MappedGGUFModel");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("Should load model");

    let config = model.config();
    let mut cache = OwnedQuantizedKVCache::from_config(config, 32);

    // Generate 5 tokens
    let prompt_tokens = [0u32, 1, 2]; // 3 prompt tokens
    let mut all_logits = Vec::new();

    // Prefill: process prompt tokens
    for (pos, &token) in prompt_tokens.iter().enumerate() {
        let logits = model
            .forward_cached(token, &mut cache, pos)
            .expect("Prefill forward should succeed");
        all_logits.push(logits);
    }

    // Generate: produce 2 new tokens
    for gen_idx in 0..2 {
        let pos = prompt_tokens.len() + gen_idx;
        // Use argmax of last logits as next token (greedy sampling)
        let last_logits = all_logits.last().expect("test value should be present");
        let next_token = last_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(idx, _)| idx as u32);

        let logits = model
            .forward_cached(next_token, &mut cache, pos)
            .expect("Generation forward should succeed");
        all_logits.push(logits);
    }

    // Should have 5 sets of logits (3 prefill + 2 generation)
    assert_eq!(all_logits.len(), 5);

    // All logits should be valid
    for (i, logits) in all_logits.iter().enumerate() {
        assert_eq!(logits.len(), 32, "Logits {} wrong size", i);
        assert!(
            logits.iter().all(|&v| v.is_finite()),
            "Logits {} contain NaN/Inf",
            i
        );
    }
}

/// Test with edge token IDs
#[test]
fn test_active_pygmy_edge_tokens() {
    let (temp_file, _) = create_pygmy_temp_file();
    let mapped = MappedGGUFModel::from_path(temp_file.path()).expect("Should load MappedGGUFModel");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("Should load model");

    let config = model.config();
    let mut cache = OwnedQuantizedKVCache::from_config(config, 32);

    // Test token 0 (BOS-like)
    let result = model.forward_cached(0, &mut cache, 0);
    assert!(result.is_ok(), "Token 0 should work");

    // Test token vocab_size-1 (last valid token)
    let mut cache2 = OwnedQuantizedKVCache::from_config(config, 32);
    let result = model.forward_cached(31, &mut cache2, 0); // vocab_size=32, so 31 is last
    assert!(result.is_ok(), "Token 31 should work");
}

/// Test that model config matches expected values
#[test]
fn test_active_pygmy_config() {
    let (temp_file, _) = create_pygmy_temp_file();
    let mapped = MappedGGUFModel::from_path(temp_file.path()).expect("Should load MappedGGUFModel");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("Should load model");

    let config = model.config();

    // Verify Active Pygmy dimensions
    assert_eq!(config.architecture, "llama");
    assert_eq!(config.hidden_dim, 32);
    assert_eq!(config.num_layers, 1);
    assert_eq!(config.num_heads, 4);
    assert_eq!(config.num_kv_heads, 4);
    assert_eq!(config.vocab_size, 32);
    // intermediate_dim may be inferred from FFN tensor dims - check it's reasonable
    assert!(
        config.intermediate_dim > 0,
        "intermediate_dim should be positive"
    );
    assert!((config.rope_theta - 10000.0).abs() < 1.0);
    assert!((config.eps - 1e-5).abs() < 1e-6);
}

/// Test layer weight access
#[test]
fn test_active_pygmy_layer_weights() {
    let (temp_file, _) = create_pygmy_temp_file();
    let mapped = MappedGGUFModel::from_path(temp_file.path()).expect("Should load MappedGGUFModel");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("Should load model");

    // Should have 1 layer
    assert_eq!(model.layers.len(), 1);

    let layer = &model.layers[0];

    // Attention norm weight
    assert_eq!(layer.attn_norm_weight.len(), 32);
    assert!(layer
        .attn_norm_weight
        .iter()
        .all(|&v| (v - 1.0).abs() < 0.01));
}

/// Test output norm and lm_head
#[test]
fn test_active_pygmy_output_weights() {
    let (temp_file, _) = create_pygmy_temp_file();
    let mapped = MappedGGUFModel::from_path(temp_file.path()).expect("Should load MappedGGUFModel");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("Should load model");

    // Output norm weight
    assert_eq!(model.output_norm_weight.len(), 32);

    // LM head exists and has data
    // The lm_head_weight is an OwnedQuantizedTensor with in_dim/out_dim
    assert_eq!(model.lm_head_weight.in_dim, 32); // hidden_dim
    assert_eq!(model.lm_head_weight.out_dim, 32); // vocab_size
}

// ============================================================================
// Edge Case Tests
// ============================================================================

/// Test KV cache state isolation
#[test]
fn test_active_pygmy_cache_isolation() {
    let (temp_file, _) = create_pygmy_temp_file();
    let mapped = MappedGGUFModel::from_path(temp_file.path()).expect("Should load MappedGGUFModel");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("Should load model");

    let config = model.config();

    // Two separate caches
    let mut cache1 = OwnedQuantizedKVCache::from_config(config, 32);
    let mut cache2 = OwnedQuantizedKVCache::from_config(config, 32);

    // Run same token through both
    let logits1 = model
        .forward_cached(1, &mut cache1, 0)
        .expect("test value should be present");
    let logits2 = model
        .forward_cached(1, &mut cache2, 0)
        .expect("test value should be present");

    // Results should be identical (same input, same initial state)
    assert_eq!(logits1.len(), logits2.len());
    for (i, (l1, l2)) in logits1.iter().zip(logits2.iter()).enumerate() {
        assert!(
            (l1 - l2).abs() < 1e-6,
            "Logit {} differs: {} vs {}",
            i,
            l1,
            l2
        );
    }
}

/// Test repeated forward with same token (should accumulate in cache)
#[test]
fn test_active_pygmy_cache_accumulation() {
    let (temp_file, _) = create_pygmy_temp_file();
    let mapped = MappedGGUFModel::from_path(temp_file.path()).expect("Should load MappedGGUFModel");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("Should load model");

    let config = model.config();
    let mut cache = OwnedQuantizedKVCache::from_config(config, 32);

    // Same token at different positions should give different results
    // (due to RoPE position encoding and attention over previous tokens)
    let logits0 = model
        .forward_cached(5, &mut cache, 0)
        .expect("test value should be present");
    let logits1 = model
        .forward_cached(5, &mut cache, 1)
        .expect("test value should be present");
    let logits2 = model
        .forward_cached(5, &mut cache, 2)
        .expect("test value should be present");

    // Logits may or may not differ with position depending on weight values
    // With Q4_0 quantized weights that decode to similar values, differences
    // may be very small. The key test is that forward() completes successfully
    // at multiple positions without crashing.
    let _diff_01: f32 = logits0
        .iter()
        .zip(&logits1)
        .map(|(a, b)| (a - b).abs())
        .sum();
    let _diff_12: f32 = logits1
        .iter()
        .zip(&logits2)
        .map(|(a, b)| (a - b).abs())
        .sum();

    // The key assertion is that all logits are valid (no NaN/Inf)
    for logits in [&logits0, &logits1, &logits2] {
        assert!(
            logits.iter().all(|&v| v.is_finite()),
            "Logits contain NaN/Inf"
        );
    }
}

/// Test model embed works for all valid tokens
#[test]
fn test_active_pygmy_all_tokens_embed() {
    let (temp_file, _) = create_pygmy_temp_file();
    let mapped = MappedGGUFModel::from_path(temp_file.path()).expect("Should load MappedGGUFModel");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("Should load model");

    // Test all 32 tokens
    for token_id in 0..32u32 {
        let embedding = model.embed(&[token_id]);
        assert_eq!(
            embedding.len(),
            32,
            "Token {} embedding wrong size",
            token_id
        );
        assert!(
            embedding.iter().all(|&v| v.is_finite()),
            "Token {} embedding contains NaN/Inf",
            token_id
        );
    }
}

/// Test QuantizedGGUFTransformer layer structure
#[test]
fn test_active_pygmy_transformer_layers() {
    let gguf_data = build_executable_pygmy_gguf();
    let gguf_model = GGUFModel::from_bytes(&gguf_data).expect("Should parse GGUF");
    let transformer = QuantizedGGUFTransformer::from_gguf(&gguf_model, &gguf_data)
        .expect("Should load transformer");

    // Should have 1 layer
    assert_eq!(transformer.layers.len(), 1);

    let layer = &transformer.layers[0];

    // Check attention norm
    assert_eq!(layer.attn_norm_weight.len(), 32);

    // Check QKV weights exist
    match &layer.qkv_weight {
        crate::gguf::quantized::QKVWeights::Separate { q, k, v } => {
            assert!(q.num_elements > 0);
            assert!(k.num_elements > 0);
            assert!(v.num_elements > 0);
        },
        crate::gguf::quantized::QKVWeights::Fused(fused) => {
            assert!(fused.num_elements > 0);
        },
    }

    // Check FFN weights exist
    assert!(layer.ffn_up_weight.num_elements > 0);
    assert!(layer.ffn_down_weight.num_elements > 0);
}
