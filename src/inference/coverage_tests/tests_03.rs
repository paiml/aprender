//! Part 03: KV Cache, Normalization, RoPE, and Q4KWeight tests
//!
//! Additional coverage for:
//! - KVCache edge cases and boundary conditions
//! - OptimizedKVCache transposed storage
//! - Attention with cache integration
//! - Layer normalization and RMS normalization edge cases
//! - RoPE position encoding edge cases
//! - Q4KWeight quantized inference

use crate::inference::{
    apply_rope, attention_with_cache, attention_with_transposed_v, simd_layer_norm, simd_rms_norm,
    KVCache, OptimizedKVCache, Q4KWeight,
};

// ============================================================================
// KVCache Edge Cases
// ============================================================================

#[test]
fn test_kv_cache_single_layer() {
    let mut cache = KVCache::new(1, 2, 5);

    cache.store(0, &[1.0, 2.0], &[3.0, 4.0]);
    cache.advance();

    assert_eq!(cache.len(), 1);
    assert!(!cache.is_empty());
    assert_eq!(cache.get_k(0).len(), 2);
    assert_eq!(cache.get_v(0).len(), 2);
}

#[test]
fn test_kv_cache_fill_to_max() {
    let max_seq = 5;
    let mut cache = KVCache::new(1, 2, max_seq);

    // Fill to capacity
    for i in 0..max_seq {
        cache.store(0, &[i as f32; 2], &[(i + 10) as f32; 2]);
        cache.advance();
    }

    assert_eq!(cache.len(), max_seq);
    assert_eq!(cache.get_k(0).len(), max_seq * 2);
}

#[test]
fn test_kv_cache_at_capacity_len() {
    let max_seq = 3;
    let mut cache = KVCache::new(1, 2, max_seq);

    // Fill to capacity
    for i in 0..max_seq {
        cache.store(0, &[i as f32; 2], &[i as f32; 2]);
        cache.advance();
    }

    // Verify cache is at capacity
    assert_eq!(cache.len(), max_seq);

    // Verify the stored values are correct
    let k = cache.get_k(0);
    assert_eq!(k.len(), max_seq * 2);
    assert!((k[0] - 0.0).abs() < 1e-5);
    assert!((k[2] - 1.0).abs() < 1e-5);
    assert!((k[4] - 2.0).abs() < 1e-5);
}

#[test]
fn test_kv_cache_multiple_resets() {
    let mut cache = KVCache::new(2, 4, 10);

    // First sequence
    cache.store(0, &[1.0; 4], &[2.0; 4]);
    cache.advance();
    assert_eq!(cache.len(), 1);

    // Reset and verify
    cache.reset();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);

    // Second sequence
    cache.store(0, &[3.0; 4], &[4.0; 4]);
    cache.advance();
    assert_eq!(cache.len(), 1);

    // Reset again
    cache.reset();
    assert!(cache.is_empty());
}

#[test]
fn test_kv_cache_layer_isolation() {
    let mut cache = KVCache::new(3, 2, 10);

    // Store different values in each layer
    cache.store(0, &[1.0, 1.0], &[1.0, 1.0]);
    cache.store(1, &[2.0, 2.0], &[2.0, 2.0]);
    cache.store(2, &[3.0, 3.0], &[3.0, 3.0]);
    cache.advance();

    // Verify each layer has its own data
    assert_eq!(cache.get_k(0), &[1.0, 1.0]);
    assert_eq!(cache.get_k(1), &[2.0, 2.0]);
    assert_eq!(cache.get_k(2), &[3.0, 3.0]);

    assert_eq!(cache.get_v(0), &[1.0, 1.0]);
    assert_eq!(cache.get_v(1), &[2.0, 2.0]);
    assert_eq!(cache.get_v(2), &[3.0, 3.0]);
}

#[test]
fn test_kv_cache_advance_without_store() {
    let mut cache = KVCache::new(1, 4, 10);

    // Advance without storing - seq_len increases but data is zeros
    cache.advance();
    cache.advance();
    cache.advance();

    assert_eq!(cache.len(), 3);
    // Data should be zeros
    let k = cache.get_k(0);
    for &v in k {
        assert!((v).abs() < 1e-5);
    }
}

// ============================================================================
// OptimizedKVCache Edge Cases
// ============================================================================

#[test]
fn test_optimized_cache_single_layer() {
    let mut cache = OptimizedKVCache::new(1, 4, 10);

    cache.store(0, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]);
    cache.advance();

    assert_eq!(cache.len(), 1);
    assert!(!cache.is_empty());
    assert_eq!(cache.max_len(), 10);
}

#[test]
fn test_optimized_cache_transposed_v_multiple_positions() {
    let mut cache = OptimizedKVCache::new(1, 2, 5);

    // Store 3 positions
    cache.store(0, &[1.0, 2.0], &[10.0, 20.0]);
    cache.advance();
    cache.store(0, &[3.0, 4.0], &[30.0, 40.0]);
    cache.advance();
    cache.store(0, &[5.0, 6.0], &[50.0, 60.0]);
    cache.advance();

    // Verify transposed layout
    let v_t = cache.get_v_transposed(0);
    // v[0] at positions 0,1,2: indices 0,1,2
    assert!((v_t[0] - 10.0).abs() < 1e-5);
    assert!((v_t[1] - 30.0).abs() < 1e-5);
    assert!((v_t[2] - 50.0).abs() < 1e-5);

    // v[1] at positions 0,1,2: indices 5,6,7 (max_seq_len stride)
    assert!((v_t[5] - 20.0).abs() < 1e-5);
    assert!((v_t[6] - 40.0).abs() < 1e-5);
    assert!((v_t[7] - 60.0).abs() < 1e-5);
}

#[test]
fn test_optimized_cache_fill_and_overflow() {
    let mut cache = OptimizedKVCache::new(1, 2, 3);

    // Fill to capacity
    for i in 0..3 {
        cache.store(0, &[i as f32; 2], &[i as f32; 2]);
        cache.advance();
    }
    assert_eq!(cache.len(), 3);

    // Try overflow - should be capped
    cache.store(0, &[99.0; 2], &[99.0; 2]);
    cache.advance();
    assert_eq!(cache.len(), 3); // Should not increase
}

#[test]
fn test_optimized_cache_multiple_layers() {
    let mut cache = OptimizedKVCache::new(4, 8, 16);

    for layer in 0..4 {
        let k = vec![(layer * 10) as f32; 8];
        let v = vec![(layer * 100) as f32; 8];
        cache.store(layer, &k, &v);
    }
    cache.advance();

    // Verify each layer
    for layer in 0..4 {
        let k = cache.get_k(layer);
        assert!((k[0] - (layer * 10) as f32).abs() < 1e-5);
    }
}

#[test]
fn test_optimized_cache_reset_clears_position() {
    let mut cache = OptimizedKVCache::new(1, 4, 10);

    cache.store(0, &[1.0; 4], &[2.0; 4]);
    cache.advance();
    cache.store(0, &[3.0; 4], &[4.0; 4]);
    cache.advance();

    assert_eq!(cache.len(), 2);

    cache.reset();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);

    // Should be able to store again from position 0
    cache.store(0, &[5.0; 4], &[6.0; 4]);
    cache.advance();
    assert_eq!(cache.len(), 1);

    let k = cache.get_k(0);
    assert!((k[0] - 5.0).abs() < 1e-5);
}

// ============================================================================
// Attention with Cache Edge Cases
// ============================================================================

#[test]
fn test_attention_with_cache_zero_dimension() {
    // Edge case: zero hidden dimension
    let q: Vec<f32> = vec![];
    let k_cache: Vec<f32> = vec![];
    let v_cache: Vec<f32> = vec![];
    let current_k: Vec<f32> = vec![];
    let current_v: Vec<f32> = vec![];

    let output = attention_with_cache(&q, &k_cache, &v_cache, &current_k, &current_v, 1);
    assert!(output.is_empty());
}

#[test]
fn test_attention_with_cache_large_cache() {
    let hidden_dim = 8;
    let num_heads = 2;
    let cache_len = 100;

    let q = vec![0.1; hidden_dim];
    let k_cache = vec![0.1; hidden_dim * cache_len];
    let v_cache = vec![0.5; hidden_dim * cache_len];
    let current_k = vec![0.1; hidden_dim];
    let current_v = vec![0.5; hidden_dim];

    let output = attention_with_cache(&q, &k_cache, &v_cache, &current_k, &current_v, num_heads);

    assert_eq!(output.len(), hidden_dim);
    // With uniform inputs, output should be close to v values
    for &v in &output {
        assert!((v - 0.5).abs() < 0.01);
    }
}

#[test]
fn test_attention_with_cache_varying_scores() {
    let _hidden_dim = 4; // Used for documentation
    let num_heads = 1;

    // Q that prefers certain K positions
    let q = vec![1.0, 0.0, 0.0, 0.0]; // Only cares about first dimension

    // Two cached positions with different first dimensions
    let k_cache = vec![
        0.0, 0.0, 0.0, 0.0, // pos 0: low score
        1.0, 0.0, 0.0, 0.0, // pos 1: high score
    ];
    let v_cache = vec![
        1.0, 1.0, 1.0, 1.0, // pos 0 value
        2.0, 2.0, 2.0, 2.0, // pos 1 value
    ];

    let current_k = vec![0.5, 0.0, 0.0, 0.0]; // medium score
    let current_v = vec![3.0, 3.0, 3.0, 3.0];

    let output = attention_with_cache(&q, &k_cache, &v_cache, &current_k, &current_v, num_heads);

    // Output should weight pos 1 more than pos 0
    // Exact values depend on softmax, but should be between min and max V
    for &v in &output {
        assert!(v >= 1.0);
        assert!(v <= 3.0);
    }
}

// ============================================================================
// Attention with Transposed V Edge Cases
// ============================================================================

#[test]
fn test_transposed_attention_empty_cache() {
    let hidden_dim = 4;
    let num_heads = 2;
    let max_seq_len = 10;

    let q = vec![1.0; hidden_dim];
    let k_cache: Vec<f32> = vec![];
    let v_cache_transposed = vec![0.0; hidden_dim * max_seq_len];
    let current_k = vec![1.0; hidden_dim];
    let current_v = vec![5.0; hidden_dim];

    let output = attention_with_transposed_v(
        &q,
        &k_cache,
        &v_cache_transposed,
        &current_k,
        &current_v,
        num_heads,
        max_seq_len,
    );

    assert_eq!(output.len(), hidden_dim);
    // Only current_v should contribute
    for &v in &output {
        assert!((v - 5.0).abs() < 1e-5);
    }
}

#[test]
fn test_transposed_attention_multi_position() {
    let hidden_dim = 4;
    let num_heads = 2;
    let max_seq_len = 5;

    let q = vec![1.0; hidden_dim];

    // 2 cached positions
    let k_cache = vec![
        1.0, 1.0, 1.0, 1.0, // pos 0
        1.0, 1.0, 1.0, 1.0, // pos 1
    ];

    // Transposed V: v[i] at pos j is at index i * max_seq_len + j
    let mut v_cache_transposed = vec![0.0; hidden_dim * max_seq_len];
    for i in 0..hidden_dim {
        v_cache_transposed[i * max_seq_len] = 1.0; // v[i] at pos 0
        v_cache_transposed[i * max_seq_len + 1] = 2.0; // v[i] at pos 1
    }

    let current_k = vec![1.0; hidden_dim];
    let current_v = vec![3.0; hidden_dim];

    let output = attention_with_transposed_v(
        &q,
        &k_cache,
        &v_cache_transposed,
        &current_k,
        &current_v,
        num_heads,
        max_seq_len,
    );

    assert_eq!(output.len(), hidden_dim);
    // Uniform attention: (1 + 2 + 3) / 3 = 2
    for &v in &output {
        assert!((v - 2.0).abs() < 1e-5);
    }
}

// ============================================================================
// Layer Normalization Edge Cases
// ============================================================================

#[test]
fn test_layer_norm_very_small_variance() {
    let input = vec![1.0, 1.0 + 1e-10, 1.0 - 1e-10, 1.0];
    let weight = vec![1.0; 4];
    let output = simd_layer_norm(&input, &weight, None, 1e-5);

    // Should not produce NaN/Inf due to near-zero variance
    for &v in &output {
        assert!(v.is_finite());
    }
}

#[test]
fn test_layer_norm_large_range() {
    let input = vec![-1000.0, -100.0, 100.0, 1000.0];
    let weight = vec![1.0; 4];
    let output = simd_layer_norm(&input, &weight, None, 1e-5);

    // Mean should be ~0
    let mean: f32 = output.iter().sum::<f32>() / 4.0;
    assert!(mean.abs() < 1e-4);

    // Std should be ~1
    let var: f32 = output.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / 4.0;
    assert!((var.sqrt() - 1.0).abs() < 0.01);
}

#[test]
fn test_layer_norm_asymmetric_scale() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0, 2.0, 0.5, 0.0];
    let output = simd_layer_norm(&input, &weight, None, 1e-5);

    // Last element should be 0 (scaled by 0)
    assert!((output[3]).abs() < 1e-5);

    // Other elements should have non-zero values
    assert!(output[0].abs() > 0.1);
    assert!(output[1].abs() > 0.1);
    assert!(output[2].abs() > 0.1);
}

#[test]
fn test_layer_norm_scale_and_shift() {
    let input = vec![0.0, 0.0, 0.0, 0.0];
    let weight = vec![1.0; 4];
    let bias = vec![10.0, 20.0, 30.0, 40.0];
    let output = simd_layer_norm(&input, &weight, Some(&bias), 1e-5);

    // With uniform input and bias, output = bias
    for (i, &v) in output.iter().enumerate() {
        let expected = (i + 1) as f32 * 10.0;
        assert!((v - expected).abs() < 1e-3);
    }
}

// ============================================================================
// RMS Normalization Edge Cases
// ============================================================================

#[test]
fn test_rms_norm_very_small_input() {
    let input = vec![1e-10, 1e-10, 1e-10];
    let weight = vec![1.0; 3];
    let output = simd_rms_norm(&input, &weight, 1e-5);

    // Should not produce NaN/Inf
    for &v in &output {
        assert!(v.is_finite());
    }
}

include!("rms_norm.rs");
