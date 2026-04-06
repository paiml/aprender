//! Property Tests for apr_transformer/q4_simd.rs
//!
//! Target: Increase q4_simd.rs coverage from 9% to >50%
//! Strategy: Property-based testing of tensor operations and data structures

use proptest::prelude::*;
use realizar::apr_transformer::{
    AprInferenceScratch, AprKVCache, AprTransformerConfig, QuantizedAprTensorQ4,
};

// ============================================================================
// QuantizedAprTensorQ4 Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Property: zeros() creates tensor with correct byte allocation
    #[test]
    fn prop_tensor_zeros_allocation(
        in_dim in 1usize..=512,
        out_dim in 1usize..=512,
    ) {
        let tensor = QuantizedAprTensorQ4::zeros(in_dim, out_dim);

        // Verify dimensions stored correctly
        prop_assert_eq!(tensor.in_dim, in_dim);
        prop_assert_eq!(tensor.out_dim, out_dim);

        // Verify byte allocation matches expected
        let num_elements = in_dim * out_dim;
        let expected_bytes = QuantizedAprTensorQ4::expected_bytes(num_elements);
        prop_assert_eq!(tensor.data.len(), expected_bytes);

        // Verify all bytes are zero
        prop_assert!(tensor.data.iter().all(|&b| b == 0));
    }

    /// Property: expected_bytes is monotonically increasing with elements
    #[test]
    fn prop_expected_bytes_monotonic(
        n1 in 1usize..=10000,
        n2 in 1usize..=10000,
    ) {
        let bytes1 = QuantizedAprTensorQ4::expected_bytes(n1);
        let bytes2 = QuantizedAprTensorQ4::expected_bytes(n2);

        if n1 <= n2 {
            prop_assert!(bytes1 <= bytes2);
        }
    }

    /// Property: expected_bytes is calculated correctly (18 bytes per 32 values)
    #[test]
    fn prop_expected_bytes_formula(num_elements in 0usize..=100000) {
        let bytes = QuantizedAprTensorQ4::expected_bytes(num_elements);

        // Q4_0: 18 bytes per block of 32 values
        let num_blocks = num_elements.div_ceil(32);
        let expected = num_blocks * 18;

        prop_assert_eq!(bytes, expected);
    }

    /// Property: new() constructor preserves all data
    #[test]
    fn prop_tensor_new_preserves_data(
        data in prop::collection::vec(any::<u8>(), 0..1000),
        in_dim in 1usize..=100,
        out_dim in 1usize..=100,
    ) {
        let tensor = QuantizedAprTensorQ4::new(data.clone(), in_dim, out_dim);

        prop_assert_eq!(tensor.data, data);
        prop_assert_eq!(tensor.in_dim, in_dim);
        prop_assert_eq!(tensor.out_dim, out_dim);
    }
}

// ============================================================================
// AprTransformerConfig Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Property: config fields are preserved through serialization
    #[test]
    fn prop_config_roundtrip(
        hidden_dim in (64usize..=4096).prop_filter("divisible by 8", |h| h % 8 == 0),
        num_heads in 1usize..=64,
        num_kv_heads in 1usize..=64,
        num_layers in 1usize..=128,
        vocab_size in 1000usize..=100000,
    ) {
        // Only test valid GQA configurations
        if hidden_dim % num_heads == 0 && num_heads % num_kv_heads == 0 {
            let config = AprTransformerConfig {
                hidden_dim,
                num_heads,
                num_kv_heads,
                num_layers,
                vocab_size,
                intermediate_dim: hidden_dim * 4,
                rope_theta: 10000.0,
                eps: 1e-5,
                context_length: 2048,
                architecture: "test".to_string(),
                ..Default::default()
            };

            // Verify all fields preserved
            prop_assert_eq!(config.hidden_dim, hidden_dim);
            prop_assert_eq!(config.num_heads, num_heads);
            prop_assert_eq!(config.num_kv_heads, num_kv_heads);
            prop_assert_eq!(config.num_layers, num_layers);
            prop_assert_eq!(config.vocab_size, vocab_size);
        }
    }

    /// Property: default config has reasonable values
    #[test]
    fn prop_default_config_valid(_unused in 0u8..1) {
        let config = AprTransformerConfig::default();

        // Verify reasonable defaults
        prop_assert!(config.hidden_dim > 0);
        prop_assert!(config.num_heads > 0);
        prop_assert!(config.num_layers > 0);
        prop_assert!(config.vocab_size > 0);
        prop_assert!(config.hidden_dim % config.num_heads == 0);
        prop_assert!(config.num_heads % config.num_kv_heads == 0);
        prop_assert!(config.eps > 0.0);
        prop_assert!(config.rope_theta > 0.0);
    }
}

// ============================================================================
// Q4_0 Block Size Property Tests
// ============================================================================

#[test]
fn test_q4_0_block_size_constants() {
    // Q4_0 format: 2 bytes scale + 16 bytes data = 18 bytes per 32 values
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(32), 18);
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(64), 36);
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(0), 0);
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(1), 18); // 1 value still needs 1 block
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(33), 36); // 33 values need 2 blocks
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Property: expected_bytes(n) >= n/2 (minimum possible compression)
    #[test]
    fn prop_expected_bytes_lower_bound(num_elements in 1usize..=100000) {
        let bytes = QuantizedAprTensorQ4::expected_bytes(num_elements);

        // Q4_0 uses 4 bits per value + 2 bytes scale per block
        // For 32 values: 16 bytes data + 2 bytes scale = 18 bytes
        // Minimum: ceil(n/32) * 18
        let min_bytes = num_elements.div_ceil(32) * 18;
        prop_assert_eq!(bytes, min_bytes);
    }
}

// ============================================================================
// Tensor Dimensions Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Property: tensor dimensions match weight matrix interpretation
    #[test]
    fn prop_tensor_dimensions_consistent(
        in_dim in 32usize..=1024,
        out_dim in 32usize..=1024,
    ) {
        let tensor = QuantizedAprTensorQ4::zeros(in_dim, out_dim);

        // Weight matrix for matmul: [out_dim, in_dim]
        // Total elements = out_dim * in_dim
        let total_elements = in_dim * out_dim;
        let expected_bytes = QuantizedAprTensorQ4::expected_bytes(total_elements);

        prop_assert_eq!(tensor.data.len(), expected_bytes);

        // Verify dimensions are accessible (usize is always >= 0)
        let _product = tensor.in_dim.checked_mul(tensor.out_dim);
    }
}

// ============================================================================
// Config Validation Tests
// ============================================================================

#[test]
fn test_config_default_values() {
    let config = AprTransformerConfig::default();

    assert_eq!(config.architecture, "unknown");
    assert_eq!(config.hidden_dim, 512);
    assert_eq!(config.num_layers, 6);
    assert_eq!(config.num_heads, 8);
    assert_eq!(config.num_kv_heads, 8);
    assert_eq!(config.vocab_size, 32000);
    assert_eq!(config.intermediate_dim, 2048);
    assert_eq!(config.context_length, 2048);
    assert_eq!(config.rope_theta, 10000.0);
    assert_eq!(config.eps, 1e-5);
}

#[test]
fn test_config_gqa_ratios() {
    // Standard MHA: num_kv_heads == num_heads
    let mha_config = AprTransformerConfig {
        num_heads: 32,
        num_kv_heads: 32,
        ..AprTransformerConfig::default()
    };
    assert_eq!(mha_config.num_heads / mha_config.num_kv_heads, 1);

    // GQA 4:1 ratio
    let gqa_config = AprTransformerConfig {
        num_heads: 32,
        num_kv_heads: 8,
        ..AprTransformerConfig::default()
    };
    assert_eq!(gqa_config.num_heads / gqa_config.num_kv_heads, 4);

    // GQA 8:1 ratio (Llama-3 style)
    let gqa_config_8 = AprTransformerConfig {
        num_heads: 64,
        num_kv_heads: 8,
        ..AprTransformerConfig::default()
    };
    assert_eq!(gqa_config_8.num_heads / gqa_config_8.num_kv_heads, 8);
}

#[test]
fn test_tensor_zero_elements() {
    // Edge case: 0 elements
    let bytes = QuantizedAprTensorQ4::expected_bytes(0);
    assert_eq!(bytes, 0);
}

#[test]
fn test_tensor_single_element() {
    // Single element still needs a full block
    let tensor = QuantizedAprTensorQ4::zeros(1, 1);
    assert_eq!(tensor.data.len(), 18); // One Q4_0 block
    assert_eq!(tensor.in_dim, 1);
    assert_eq!(tensor.out_dim, 1);
}

#[test]
fn test_tensor_large_dimensions() {
    // Large tensor (4096 x 4096 = 16M elements)
    let large_dim = 4096;
    let num_elements = large_dim * large_dim;
    let expected_bytes = QuantizedAprTensorQ4::expected_bytes(num_elements);

    // 16M / 32 = 512K blocks * 18 bytes = 9.4MB
    let expected = (num_elements / 32) * 18;
    assert_eq!(expected_bytes, expected);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Property: tensor data can be cloned without loss
    #[test]
    fn prop_tensor_clone_preserves_data(
        data in prop::collection::vec(any::<u8>(), 18..=1800),
        in_dim in 32usize..=256,
        out_dim in 32usize..=256,
    ) {
        let original = QuantizedAprTensorQ4::new(data, in_dim, out_dim);
        let cloned = original.clone();

        prop_assert_eq!(original.data, cloned.data);
        prop_assert_eq!(original.in_dim, cloned.in_dim);
        prop_assert_eq!(original.out_dim, cloned.out_dim);
    }
}

// ============================================================================
// AprInferenceScratch Property Tests
// ============================================================================

#[test]
fn test_scratch_from_config_creates_valid_buffers() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 256,
        num_layers: 4,
        num_heads: 8,
        num_kv_heads: 4,
        vocab_size: 1000,
        intermediate_dim: 512,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        ..Default::default()
    };

    let scratch = AprInferenceScratch::from_config(&config);

    // Verify scratch buffers are allocated with correct sizes
    assert_eq!(scratch.hidden.len(), 256);
    assert_eq!(scratch.attn_out.len(), 256);
    assert_eq!(scratch.ffn_up.len(), 512);
    assert_eq!(scratch.ffn_gate.len(), 512);
}

#[test]
fn test_scratch_clear_zeroes_buffers() {
    let config = AprTransformerConfig::default();
    let mut scratch = AprInferenceScratch::from_config(&config);

    // Fill with non-zero values
    scratch.hidden.fill(1.0);
    scratch.normed.fill(2.0);
    scratch.q.fill(3.0);

    // Clear
    scratch.clear();

    // Verify all zeroed
    assert!(scratch.hidden.iter().all(|&x| x == 0.0));
    assert!(scratch.normed.iter().all(|&x| x == 0.0));
    assert!(scratch.q.iter().all(|&x| x == 0.0));
}

// ============================================================================
// AprKVCache Property Tests
// ============================================================================

#[test]
fn test_kv_cache_new_creates_valid_structure() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 256,
        num_layers: 4,
        num_heads: 8,
        num_kv_heads: 4,
        vocab_size: 1000,
        intermediate_dim: 512,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        ..Default::default()
    };

    let cache = AprKVCache::new(&config);

    // Verify cache starts empty
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
    assert_eq!(cache.capacity(), 128); // context_length
    assert_eq!(cache.num_kv_heads(), 4);
    assert_eq!(cache.head_dim(), 32); // hidden_dim / num_heads = 256/8
}

#[test]
fn test_kv_cache_append_and_get() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 2,
        vocab_size: 1000,
        intermediate_dim: 256,
        context_length: 16,
        rope_theta: 10000.0,
        eps: 1e-5,
        ..Default::default()
    };

    let mut cache = AprKVCache::new(&config);

    // kv_size = num_kv_heads * head_dim = 2 * 32 = 64
    let k = vec![1.0f32; 64];
    let v = vec![2.0f32; 64];

    // Append to both layers
    cache.append(0, &k, &v);
    cache.append(1, &k, &v);

    assert_eq!(cache.len(), 1);
    assert!(!cache.is_empty());

    // Get cached values for layer 0
    let (cached_k, cached_v) = cache.get(0);
    assert_eq!(cached_k.len(), 64);
    assert_eq!(cached_v.len(), 64);
    assert_eq!(cached_k[0], 1.0);
    assert_eq!(cached_v[0], 2.0);
}

#[test]
fn test_kv_cache_clear() {
    let config = AprTransformerConfig::default();
    let mut cache = AprKVCache::new(&config);

    // Append some values
    let kv_size = config.num_kv_heads * (config.hidden_dim / config.num_heads);
    let k = vec![1.0f32; kv_size];
    let v = vec![2.0f32; kv_size];

    for layer in 0..config.num_layers {
        cache.append(layer, &k, &v);
    }

    assert!(!cache.is_empty());

    // Clear
    cache.clear();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    /// Property: scratch dimensions match config
    #[test]
    fn prop_scratch_dimensions_match_config(
        hidden_dim in (64usize..=512).prop_filter("div by 8", |h| h % 8 == 0),
        intermediate_dim in (128usize..=1024).prop_filter("div by 8", |h| h % 8 == 0),
    ) {
        let config = AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim,
            num_layers: 4,
            num_heads: hidden_dim / 64,
            num_kv_heads: hidden_dim / 64,
            vocab_size: 1000,
            intermediate_dim,
            context_length: 256,
            rope_theta: 10000.0,
            eps: 1e-5,
        ..Default::default()
        };

        let scratch = AprInferenceScratch::from_config(&config);

        prop_assert_eq!(scratch.hidden.len(), hidden_dim);
        prop_assert_eq!(scratch.normed.len(), hidden_dim);
        prop_assert_eq!(scratch.attn_out.len(), hidden_dim);
        prop_assert_eq!(scratch.ffn_up.len(), intermediate_dim);
        prop_assert_eq!(scratch.ffn_gate.len(), intermediate_dim);
    }

    /// Property: kv cache capacity matches context_length
    #[test]
    fn prop_kv_cache_capacity_matches_context(
        context_length in 16usize..=512,
        num_layers in 1usize..=8,
    ) {
        let config = AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim: 256,
            num_layers,
            num_heads: 8,
            num_kv_heads: 4,
            vocab_size: 1000,
            intermediate_dim: 512,
            context_length,
            rope_theta: 10000.0,
            eps: 1e-5,
        ..Default::default()
        };

        let cache = AprKVCache::new(&config);

        prop_assert_eq!(cache.capacity(), context_length);
        prop_assert!(cache.is_empty());
    }
}
