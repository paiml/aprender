//! GGUF Part 07: PARITY-005 (Contiguous KV Cache) + PARITY-006 (Batch Processing) +
//!               PARITY-007 (E2E Benchmark Verification) + PARITY-008 (Popper Score)
//!
//! Extracted from gguf_monolith.rs (PMAT-802)

use crate::gguf::test_helpers::create_test_model_with_config;
use crate::gguf::{
    ContiguousKVCache, GGUFConfig, OwnedQuantizedKVCache, OwnedQuantizedModel,
    OwnedQuantizedModelCached, QuantizedGenerateConfig,
};

// ========================================================================
// PARITY-005: Contiguous KV Cache Tests
// ========================================================================

/// PARITY-005a: ContiguousKVCache should use single contiguous allocation
#[test]
fn test_parity005a_contiguous_kv_cache_layout() {
    let num_layers = 4;
    let hidden_dim = 64;
    let max_seq_len = 32;

    let cache = ContiguousKVCache::new(num_layers, hidden_dim, max_seq_len);

    // Verify contiguous layout
    assert!(
        cache.is_contiguous(),
        "PARITY-005a: Cache should report contiguous layout"
    );

    // Verify cache line alignment
    assert!(
        cache.is_cache_aligned(),
        "PARITY-005a: Layer stride should be cache-line aligned"
    );

    // Verify stride is multiple of 16 (64 bytes / 4 bytes per float)
    assert_eq!(
        cache.layer_stride() % 16,
        0,
        "PARITY-005a: Layer stride {} should be multiple of 16 floats",
        cache.layer_stride()
    );
}

/// PARITY-005b: Cache line alignment calculation
#[test]
fn test_parity005b_cache_line_alignment() {
    // Test various sizes for proper alignment
    let test_cases = vec![
        (4, 64, 32, 2048),   // 32 * 64 = 2048, already aligned
        (4, 64, 33, 2112),   // 33 * 64 = 2112, needs alignment to 2128
        (2, 80, 16, 1280),   // 16 * 80 = 1280, aligned
        (8, 256, 64, 16384), // 64 * 256 = 16384, aligned
    ];

    for (num_layers, hidden_dim, max_seq_len, _expected_raw) in test_cases {
        let cache = ContiguousKVCache::new(num_layers, hidden_dim, max_seq_len);

        // Layer stride should be cache-line aligned
        assert!(
            cache.is_cache_aligned(),
            "PARITY-005b: Cache should be aligned for num_layers={}, hidden_dim={}, max_seq_len={}",
            num_layers,
            hidden_dim,
            max_seq_len
        );

        // Verify stride is at least raw size
        let raw_size = max_seq_len * hidden_dim;
        assert!(
            cache.layer_stride() >= raw_size,
            "PARITY-005b: Layer stride {} should be >= raw size {}",
            cache.layer_stride(),
            raw_size
        );
    }
}

/// PARITY-005c: Append and retrieve K/V data correctly
#[test]
fn test_parity005c_contiguous_kv_operations() {
    let num_layers = 2;
    let hidden_dim = 16;
    let max_seq_len = 8;

    let mut cache = ContiguousKVCache::new(num_layers, hidden_dim, max_seq_len);

    // Create test data
    let k0: Vec<f32> = (0..hidden_dim).map(|i| i as f32 * 0.1).collect();
    let v0: Vec<f32> = (0..hidden_dim).map(|i| i as f32 * 0.2).collect();
    let k1: Vec<f32> = (0..hidden_dim).map(|i| (i + 16) as f32 * 0.1).collect();
    let v1: Vec<f32> = (0..hidden_dim).map(|i| (i + 16) as f32 * 0.2).collect();

    // Append to layer 0
    cache.append(0, &k0, &v0);
    cache.advance();

    // Append to layer 0 again (position 1)
    cache.append(0, &k1, &v1);
    cache.advance();

    // Verify length
    assert_eq!(cache.len(), 2, "PARITY-005c: Cache should have 2 positions");

    // Verify K data for layer 0
    let cached_k = cache.get_k(0);
    assert_eq!(
        cached_k.len(),
        2 * hidden_dim,
        "PARITY-005c: K cache should have 2 * hidden_dim elements"
    );

    // First position's K values should match k0
    for (i, &val) in cached_k[..hidden_dim].iter().enumerate() {
        assert!(
            (val - k0[i]).abs() < 1e-6,
            "PARITY-005c: K position 0 mismatch at {}: expected {}, got {}",
            i,
            k0[i],
            val
        );
    }

    // Second position's K values should match k1
    for (i, &val) in cached_k[hidden_dim..].iter().enumerate() {
        assert!(
            (val - k1[i]).abs() < 1e-6,
            "PARITY-005c: K position 1 mismatch at {}: expected {}, got {}",
            i,
            k1[i],
            val
        );
    }
}

/// PARITY-005d: Reset cache to empty state
#[test]
fn test_parity005d_contiguous_kv_reset() {
    let mut cache = ContiguousKVCache::new(2, 16, 8);

    // Add some data
    let k: Vec<f32> = vec![1.0; 16];
    let v: Vec<f32> = vec![2.0; 16];
    cache.append(0, &k, &v);
    cache.advance();
    cache.append(0, &k, &v);
    cache.advance();

    assert_eq!(cache.len(), 2, "PARITY-005d: Before reset, len should be 2");

    // Reset
    cache.reset();

    assert_eq!(cache.len(), 0, "PARITY-005d: After reset, len should be 0");

    // Should be able to append again
    cache.append(0, &k, &v);
    cache.advance();
    assert_eq!(
        cache.len(),
        1,
        "PARITY-005d: After reset+append, len should be 1"
    );
}

/// PARITY-005e: Verify sequential memory layout
#[test]
fn test_parity005e_sequential_memory_layout() {
    let num_layers = 2;
    let hidden_dim = 8;
    let max_seq_len = 4;

    let mut cache = ContiguousKVCache::new(num_layers, hidden_dim, max_seq_len);

    // Add data for layer 0 at position 0
    let k0: Vec<f32> = (0..hidden_dim).map(|i| i as f32).collect();
    let v0: Vec<f32> = (0..hidden_dim).map(|i| (i + 10) as f32).collect();
    cache.append(0, &k0, &v0);

    // Add data for layer 1 at position 0 (same position, different layer)
    let k1: Vec<f32> = (0..hidden_dim).map(|i| (i + 20) as f32).collect();
    let v1: Vec<f32> = (0..hidden_dim).map(|i| (i + 30) as f32).collect();
    cache.append(1, &k1, &v1);

    // Advance after appending to all layers for this position
    cache.advance();

    // Verify layer 0's K data
    let layer0_k = cache.get_k(0);
    assert!(
        layer0_k[0] == 0.0,
        "PARITY-005e: Layer 0 K should start with 0.0"
    );

    // Verify layer 1's K data
    let layer1_k = cache.get_k(1);
    assert!(
        layer1_k[0] == 20.0,
        "PARITY-005e: Layer 1 K should start with 20.0"
    );
}

/// PARITY-005f: Verify memory usage matches expected
#[test]
fn test_parity005f_memory_usage() {
    let num_layers = 4;
    let hidden_dim = 64;
    let max_seq_len = 128;

    let cache = ContiguousKVCache::new(num_layers, hidden_dim, max_seq_len);

    // Expected: 2 (K+V) * num_layers * max_seq_len * hidden_dim * sizeof(f32)
    let expected_bytes = 2 * num_layers * max_seq_len * hidden_dim * std::mem::size_of::<f32>();

    // Actual should be close (may have padding for alignment)
    let actual_bytes = cache.memory_bytes();

    assert!(
        actual_bytes >= expected_bytes,
        "PARITY-005f: Memory usage {} should be >= minimum {}",
        actual_bytes,
        expected_bytes
    );

    // Should not be too much larger (< 10% overhead for alignment)
    let max_overhead = expected_bytes as f64 * 1.1;
    assert!(
        actual_bytes as f64 <= max_overhead,
        "PARITY-005f: Memory usage {} should be < {} (10% overhead)",
        actual_bytes,
        max_overhead as usize
    );
}

/// PARITY-005i: Verify cache performance compared to non-contiguous
#[test]
fn test_parity005i_cache_performance_comparison() {
    use std::time::Instant;

    let num_layers = 4;
    let hidden_dim = 64;
    let max_seq_len = 64;
    let num_iterations = 100;

    // Benchmark non-contiguous (OwnedQuantizedKVCache)
    let start = Instant::now();
    for _ in 0..num_iterations {
        let mut cache = OwnedQuantizedKVCache::new(num_layers, hidden_dim, max_seq_len);
        let k: Vec<f32> = vec![1.0; hidden_dim];
        let v: Vec<f32> = vec![2.0; hidden_dim];
        for _ in 0..max_seq_len {
            for layer in 0..num_layers {
                cache.append_kv(layer, &k, &v);
            }
        }
    }
    let owned_time = start.elapsed();

    // Benchmark contiguous
    let start = Instant::now();
    for _ in 0..num_iterations {
        let mut cache = ContiguousKVCache::new(num_layers, hidden_dim, max_seq_len);
        let k: Vec<f32> = vec![1.0; hidden_dim];
        let v: Vec<f32> = vec![2.0; hidden_dim];
        for _ in 0..max_seq_len {
            for layer in 0..num_layers {
                cache.append(layer, &k, &v);
            }
            cache.advance();
        }
    }
    let contiguous_time = start.elapsed();

    println!("PARITY-005i: Cache Performance Comparison");
    println!("  OwnedQuantizedKVCache: {:?}", owned_time);
    println!("  ContiguousKVCache: {:?}", contiguous_time);

    // Contiguous should not be significantly slower (within 10x to account for coverage instrumentation)
    let ratio = contiguous_time.as_nanos() as f64 / owned_time.as_nanos() as f64;
    println!("  Ratio (contiguous/owned): {:.2}x", ratio);

    assert!(
        ratio < 10.0,
        "PARITY-005i: Contiguous cache should not be 10x slower than owned"
    );
}

// ========================================================================
// PARITY-006: Batch Processing Tests
// ========================================================================

/// PARITY-006a: batch_generate method should exist and produce valid output
#[test]
fn test_parity006a_batch_generate_exists() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };
    let model = create_test_model_with_config(&config);
    let cached = OwnedQuantizedModelCached::new(model);

    // Single prompt batch
    let prompts = vec![vec![1u32, 2, 3]];
    let gen_config = QuantizedGenerateConfig {
        max_tokens: 5,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
        ..Default::default()
    };

    let results = cached
        .model()
        .batch_generate(
            &prompts
                .iter()
                .map(std::vec::Vec::as_slice)
                .collect::<Vec<_>>(),
            &gen_config,
        )
        .expect("PARITY-006a: batch_generate should exist and succeed");

    assert_eq!(
        results.len(),
        1,
        "PARITY-006a: Should return results for 1 prompt"
    );
    assert!(
        results[0].len() > 3,
        "PARITY-006a: Generated output should include prompt + new tokens"
    );
}

/// PARITY-006b: Single-prompt batch should match non-batch generation
#[test]
fn test_parity006b_single_prompt_optimization() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };
    let model = create_test_model_with_config(&config);

    let gen_config = QuantizedGenerateConfig {
        max_tokens: 5,
        temperature: 0.0, // Deterministic
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
        ..Default::default()
    };

    // Non-batch generation
    let prompt = vec![1u32, 2, 3];
    let single_result = model
        .generate_with_cache(&prompt, &gen_config)
        .expect("Single generate should work");

    // Batch generation with single prompt
    let cached = OwnedQuantizedModelCached::new(model);
    let prompts = vec![prompt.clone()];
    let batch_results = cached
        .model()
        .batch_generate(
            &prompts
                .iter()
                .map(std::vec::Vec::as_slice)
                .collect::<Vec<_>>(),
            &gen_config,
        )
        .expect("Batch generate should work");

    // Results should match (deterministic sampling)
    assert_eq!(
        single_result, batch_results[0],
        "PARITY-006b: Single and batch generation should produce same result"
    );
}

include!("parity006c_batch.rs");
include!("parity008c_popper_calculate.rs");
include!("parity009d_outlier_median.rs");
include!("parity010b_model_download.rs");
