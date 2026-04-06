//! Coverage tests for GGUF KV Cache implementations (PMAT-802)
//!
//! Tests for `OwnedQuantizedKVCache` and `ContiguousKVCache` structs
//! in src/gguf.rs. Uses EXTREME TDD methodology to achieve comprehensive
//! coverage of all code paths.
//!
//! Coverage targets:
//! - OwnedQuantizedKVCache: new, from_config, append, advance, append_kv,
//!   advance_by, rollback_to, snapshot_len, get_k, get_v, len, is_empty, reset, max_len
//! - ContiguousKVCache: new, from_config, is_contiguous, is_cache_aligned,
//!   layer_stride, append, advance, get_k, get_v, get_k_mut, get_v_mut,
//!   len, is_empty, reset, reset_and_zero, max_len, memory_bytes, prefetch_k, prefetch_v

use realizar::gguf::{ArchConstraints, ContiguousKVCache, GGUFConfig, OwnedQuantizedKVCache};

/// Helper function to create a test GGUFConfig with specified layers and hidden_dim
fn make_test_config(num_layers: usize, hidden_dim: usize) -> GGUFConfig {
    GGUFConfig {
        architecture: "test".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim,
        num_layers,
        num_heads: 8,
        num_kv_heads: 8,
        vocab_size: 1000,
        intermediate_dim: hidden_dim * 4,
        context_length: 2048,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    }
}

// ============================================================================
// OwnedQuantizedKVCache Tests
// ============================================================================

mod owned_quantized_kv_cache {
    use super::*;

    // ------------------------------------------------------------------------
    // Construction Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_new_creates_empty_cache() {
        let cache = OwnedQuantizedKVCache::new(4, 256, 128);

        assert_eq!(cache.len(), 0, "New cache should have length 0");
        assert!(cache.is_empty(), "New cache should be empty");
        assert_eq!(
            cache.max_len(),
            128,
            "Max length should match constructor arg"
        );
    }

    #[test]
    fn test_new_with_zero_layers() {
        let cache = OwnedQuantizedKVCache::new(0, 256, 128);

        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.max_len(), 128);
    }

    #[test]
    fn test_new_with_zero_hidden_dim() {
        let cache = OwnedQuantizedKVCache::new(4, 0, 128);

        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_new_with_zero_max_seq_len() {
        let cache = OwnedQuantizedKVCache::new(4, 256, 0);

        assert_eq!(cache.len(), 0);
        assert_eq!(cache.max_len(), 0);
    }

    #[test]
    fn test_from_config() {
        let config = make_test_config(8, 512);
        let cache = OwnedQuantizedKVCache::from_config(&config, 256);

        assert_eq!(cache.len(), 0);
        assert_eq!(cache.max_len(), 256);
    }

    #[test]
    fn test_default_creates_zero_capacity_cache() {
        let cache = OwnedQuantizedKVCache::default();

        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.max_len(), 0);
    }

    // ------------------------------------------------------------------------
    // Append Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_append_single_layer() {
        let mut cache = OwnedQuantizedKVCache::new(2, 64, 16);

        let k = vec![1.0f32; 64];
        let v = vec![2.0f32; 64];

        cache.append(0, &k, &v);

        // Verify K cache contains the appended data
        let k_cache = cache.get_k(0);
        assert_eq!(
            k_cache.len(),
            64,
            "K cache should have 64 elements after append"
        );
        assert!(
            k_cache.iter().all(|&x| (x - 1.0).abs() < 1e-6),
            "K values should be 1.0"
        );

        // Verify V cache contains the appended data
        let v_cache = cache.get_v(0);
        assert_eq!(
            v_cache.len(),
            64,
            "V cache should have 64 elements after append"
        );
        assert!(
            v_cache.iter().all(|&x| (x - 2.0).abs() < 1e-6),
            "V values should be 2.0"
        );
    }

    #[test]
    fn test_append_multiple_layers() {
        let mut cache = OwnedQuantizedKVCache::new(3, 32, 8);

        // Append to each layer with different values
        for layer in 0..3 {
            let k = vec![layer as f32 + 1.0; 32];
            let v = vec![layer as f32 + 10.0; 32];
            cache.append(layer, &k, &v);
        }

        // Verify each layer has distinct values
        for layer in 0..3 {
            let k_cache = cache.get_k(layer);
            let v_cache = cache.get_v(layer);

            assert_eq!(k_cache.len(), 32);
            assert_eq!(v_cache.len(), 32);

            let expected_k = layer as f32 + 1.0;
            let expected_v = layer as f32 + 10.0;
            assert!(
                k_cache.iter().all(|&x| (x - expected_k).abs() < 1e-6),
                "Layer {} K values should be {}",
                layer,
                expected_k
            );
            assert!(
                v_cache.iter().all(|&x| (x - expected_v).abs() < 1e-6),
                "Layer {} V values should be {}",
                layer,
                expected_v
            );
        }
    }

    #[test]
    fn test_append_multiple_positions() {
        let mut cache = OwnedQuantizedKVCache::new(1, 16, 8);

        // Append 4 positions
        for pos in 0..4 {
            let k = vec![pos as f32; 16];
            let v = vec![pos as f32 + 100.0; 16];
            cache.append(0, &k, &v);
            cache.advance();
        }

        let k_cache = cache.get_k(0);
        let v_cache = cache.get_v(0);

        assert_eq!(k_cache.len(), 64, "Should have 4 positions * 16 hidden dim");
        assert_eq!(v_cache.len(), 64);
        assert_eq!(cache.len(), 4, "Should have processed 4 positions");
    }

    #[test]
    fn test_append_to_invalid_layer_is_no_op() {
        let mut cache = OwnedQuantizedKVCache::new(2, 32, 8);

        let k = vec![1.0f32; 32];
        let v = vec![2.0f32; 32];

        // Attempt to append to layer 5 (out of bounds)
        cache.append(5, &k, &v);

        // Cache should remain empty for all valid layers
        assert!(cache.get_k(0).is_empty());
        assert!(cache.get_k(1).is_empty());
    }

    #[test]
    fn test_append_beyond_max_seq_len_is_no_op() {
        let mut cache = OwnedQuantizedKVCache::new(1, 16, 2);

        let k = vec![1.0f32; 16];
        let v = vec![2.0f32; 16];

        // Fill to capacity
        cache.append(0, &k, &v);
        cache.advance();
        cache.append(0, &k, &v);
        cache.advance();

        // At max capacity now
        assert_eq!(cache.len(), 2);

        // Try to append beyond capacity
        let k2 = vec![99.0f32; 16];
        let v2 = vec![99.0f32; 16];
        cache.append(0, &k2, &v2);

        // Should not have added the new values
        let k_cache = cache.get_k(0);
        assert_eq!(
            k_cache.len(),
            32,
            "Should still only have 2 positions worth of data"
        );
    }

    // ------------------------------------------------------------------------
    // Advance Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_advance_increments_seq_len() {
        let mut cache = OwnedQuantizedKVCache::new(2, 64, 10);

        assert_eq!(cache.len(), 0);

        cache.advance();
        assert_eq!(cache.len(), 1);

        cache.advance();
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_advance_stops_at_max_seq_len() {
        let mut cache = OwnedQuantizedKVCache::new(1, 16, 3);

        // Advance beyond max
        for _ in 0..10 {
            cache.advance();
        }

        assert_eq!(cache.len(), 3, "Should cap at max_seq_len");
    }

    #[test]
    fn test_advance_by_increments_correctly() {
        let mut cache = OwnedQuantizedKVCache::new(1, 16, 100);

        cache.advance_by(5);
        assert_eq!(cache.len(), 5);

        cache.advance_by(10);
        assert_eq!(cache.len(), 15);
    }

    #[test]
    fn test_advance_by_caps_at_max_seq_len() {
        let mut cache = OwnedQuantizedKVCache::new(1, 16, 10);

        cache.advance_by(100);
        assert_eq!(cache.len(), 10, "Should cap at max_seq_len");
    }

    // ------------------------------------------------------------------------
    // append_kv Tests (PAR-097)
    // ------------------------------------------------------------------------

    #[test]
    fn test_append_kv_batch() {
        let mut cache = OwnedQuantizedKVCache::new(2, 32, 16);

        // Append batch of 3 K/V entries at once
        let k_batch = vec![1.0f32; 32 * 3]; // 3 positions
        let v_batch = vec![2.0f32; 32 * 3];

        cache.append_kv(0, &k_batch, &v_batch);

        let k_cache = cache.get_k(0);
        let v_cache = cache.get_v(0);

        assert_eq!(k_cache.len(), 96, "Should have 3 * 32 = 96 elements");
        assert_eq!(v_cache.len(), 96);
    }

    #[test]
    fn test_append_kv_invalid_layer_is_no_op() {
        let mut cache = OwnedQuantizedKVCache::new(2, 32, 16);

        let k_batch = vec![1.0f32; 64];
        let v_batch = vec![2.0f32; 64];

        // Append to invalid layer
        cache.append_kv(10, &k_batch, &v_batch);

        // Valid layers should remain empty
        assert!(cache.get_k(0).is_empty());
        assert!(cache.get_k(1).is_empty());
    }

    // ------------------------------------------------------------------------
    // Rollback Tests (PAR-098)
    // ------------------------------------------------------------------------

    #[test]
    fn test_rollback_to_earlier_position() {
        let mut cache = OwnedQuantizedKVCache::new(2, 16, 10);
        let kv_dim = 16;

        // Fill with 5 positions
        for i in 0..5 {
            let k = vec![i as f32; kv_dim];
            let v = vec![i as f32 + 100.0; kv_dim];
            cache.append(0, &k, &v);
            cache.append(1, &k, &v);
            cache.advance();
        }

        assert_eq!(cache.len(), 5);
        assert_eq!(cache.get_k(0).len(), 80); // 5 * 16

        // Rollback to position 2
        cache.rollback_to(2, kv_dim);

        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get_k(0).len(), 32); // 2 * 16
        assert_eq!(cache.get_v(0).len(), 32);
        assert_eq!(cache.get_k(1).len(), 32);
        assert_eq!(cache.get_v(1).len(), 32);
    }

    #[test]
    fn test_rollback_to_zero() {
        let mut cache = OwnedQuantizedKVCache::new(1, 8, 10);
        let kv_dim = 8;

        // Fill some positions
        for i in 0..3 {
            let k = vec![i as f32; kv_dim];
            let v = vec![i as f32; kv_dim];
            cache.append(0, &k, &v);
            cache.advance();
        }

        assert_eq!(cache.len(), 3);

        // Rollback to 0
        cache.rollback_to(0, kv_dim);

        assert_eq!(cache.len(), 0);
        assert!(cache.get_k(0).is_empty());
        assert!(cache.get_v(0).is_empty());
    }

    #[test]
    fn test_rollback_to_same_position_is_no_op() {
        let mut cache = OwnedQuantizedKVCache::new(1, 8, 10);
        let kv_dim = 8;

        // Fill 3 positions
        for i in 0..3 {
            let k = vec![i as f32; kv_dim];
            let v = vec![i as f32; kv_dim];
            cache.append(0, &k, &v);
            cache.advance();
        }

        let original_len = cache.len();
        let original_k_len = cache.get_k(0).len();

        // Rollback to current position (no-op)
        cache.rollback_to(3, kv_dim);

        assert_eq!(cache.len(), original_len);
        assert_eq!(cache.get_k(0).len(), original_k_len);
    }

    #[test]
    fn test_rollback_to_future_position_is_no_op() {
        let mut cache = OwnedQuantizedKVCache::new(1, 8, 10);
        let kv_dim = 8;

        // Fill 2 positions
        for i in 0..2 {
            let k = vec![i as f32; kv_dim];
            let v = vec![i as f32; kv_dim];
            cache.append(0, &k, &v);
            cache.advance();
        }

        let original_len = cache.len();

        // Try to rollback to future position
        cache.rollback_to(10, kv_dim);

        assert_eq!(
            cache.len(),
            original_len,
            "Should not change when rolling back to future"
        );
    }

    // ------------------------------------------------------------------------
    // Snapshot Tests (PAR-098)
    // ------------------------------------------------------------------------

    #[test]
    fn test_snapshot_len_returns_current_seq_len() {
        let mut cache = OwnedQuantizedKVCache::new(1, 8, 10);

        assert_eq!(cache.snapshot_len(), 0);

        cache.advance();
        assert_eq!(cache.snapshot_len(), 1);

        cache.advance_by(5);
        assert_eq!(cache.snapshot_len(), 6);
    }

    // ------------------------------------------------------------------------
    // Get K/V Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_get_k_empty_cache() {
        let cache = OwnedQuantizedKVCache::new(2, 32, 8);

        assert!(cache.get_k(0).is_empty());
        assert!(cache.get_k(1).is_empty());
    }

    #[test]
    fn test_get_v_empty_cache() {
        let cache = OwnedQuantizedKVCache::new(2, 32, 8);

        assert!(cache.get_v(0).is_empty());
        assert!(cache.get_v(1).is_empty());
    }

    #[test]
    fn test_get_k_invalid_layer_returns_empty() {
        let cache = OwnedQuantizedKVCache::new(2, 32, 8);

        assert!(
            cache.get_k(10).is_empty(),
            "Invalid layer should return empty slice"
        );
    }

    #[test]
    fn test_get_v_invalid_layer_returns_empty() {
        let cache = OwnedQuantizedKVCache::new(2, 32, 8);

        assert!(
            cache.get_v(10).is_empty(),
            "Invalid layer should return empty slice"
        );
    }

    // ------------------------------------------------------------------------
    // Reset Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_reset_clears_all_data() {
        let mut cache = OwnedQuantizedKVCache::new(2, 16, 8);

        // Fill with data
        for _ in 0..4 {
            let k = vec![1.0f32; 16];
            let v = vec![2.0f32; 16];
            cache.append(0, &k, &v);
            cache.append(1, &k, &v);
            cache.advance();
        }

        assert_eq!(cache.len(), 4);
        assert!(!cache.get_k(0).is_empty());

        // Reset
        cache.reset();

        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert!(cache.get_k(0).is_empty());
        assert!(cache.get_v(0).is_empty());
        assert!(cache.get_k(1).is_empty());
        assert!(cache.get_v(1).is_empty());
    }

    #[test]
    fn test_reset_allows_reuse() {
        let mut cache = OwnedQuantizedKVCache::new(1, 8, 4);

        // Fill
        for i in 0..3 {
            let k = vec![i as f32; 8];
            let v = vec![i as f32; 8];
            cache.append(0, &k, &v);
            cache.advance();
        }

        // Reset
        cache.reset();

        // Reuse
        let k = vec![99.0f32; 8];
        let v = vec![99.0f32; 8];
        cache.append(0, &k, &v);
        cache.advance();

        assert_eq!(cache.len(), 1);
        let k_cache = cache.get_k(0);
        assert_eq!(k_cache.len(), 8);
        assert!(k_cache.iter().all(|&x| (x - 99.0).abs() < 1e-6));
    }

    // ------------------------------------------------------------------------
    // Clone Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_clone_preserves_state() {
        let mut cache = OwnedQuantizedKVCache::new(2, 16, 8);

        // Fill with data
        let k = vec![1.0f32; 16];
        let v = vec![2.0f32; 16];
        cache.append(0, &k, &v);
        cache.advance();

        // Clone
        let cloned = cache.clone();

        assert_eq!(cloned.len(), cache.len());
        assert_eq!(cloned.max_len(), cache.max_len());
        assert_eq!(cloned.get_k(0), cache.get_k(0));
        assert_eq!(cloned.get_v(0), cache.get_v(0));
    }

    #[test]
    fn test_clone_is_independent() {
        let mut cache = OwnedQuantizedKVCache::new(1, 8, 8);

        let k = vec![1.0f32; 8];
        let v = vec![2.0f32; 8];
        cache.append(0, &k, &v);
        cache.advance();

        let mut cloned = cache.clone();

        // Modify original
        let k2 = vec![99.0f32; 8];
        let v2 = vec![99.0f32; 8];
        cache.append(0, &k2, &v2);
        cache.advance();

        // Clone should be unchanged
        assert_eq!(cloned.len(), 1);
        assert_eq!(cloned.get_k(0).len(), 8);

        // Modify clone
        cloned.reset();
        assert!(cloned.is_empty());

        // Original should be unchanged
        assert_eq!(cache.len(), 2);
    }
}

// ============================================================================
// ContiguousKVCache Tests
// ============================================================================

mod contiguous_kv_cache {
    use super::*;

    // ------------------------------------------------------------------------
    // Construction Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_new_creates_empty_cache() {
        let cache = ContiguousKVCache::new(4, 256, 128);

        assert_eq!(cache.len(), 0, "New cache should have length 0");
        assert!(cache.is_empty(), "New cache should be empty");
        assert_eq!(
            cache.max_len(),
            128,
            "Max length should match constructor arg"
        );
    }

    #[test]
    fn test_new_with_zero_layers() {
        let cache = ContiguousKVCache::new(0, 256, 128);

        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.max_len(), 128);
    }

    #[test]
    fn test_new_with_zero_hidden_dim() {
        let cache = ContiguousKVCache::new(4, 0, 128);

        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_new_with_zero_max_seq_len() {
        let cache = ContiguousKVCache::new(4, 256, 0);

        assert_eq!(cache.len(), 0);
        assert_eq!(cache.max_len(), 0);
    }

    #[test]
    fn test_from_config() {
        let config = make_test_config(8, 512);
        let cache = ContiguousKVCache::from_config(&config, 256);

        assert_eq!(cache.len(), 0);
        assert_eq!(cache.max_len(), 256);
    }

    // ------------------------------------------------------------------------
    // Alignment Tests (PARITY-005)
    // ------------------------------------------------------------------------

    #[test]
    fn test_is_contiguous_always_true() {
        let cache = ContiguousKVCache::new(4, 256, 128);
        assert!(
            cache.is_contiguous(),
            "ContiguousKVCache should always report contiguous"
        );
    }

    #[test]
    fn test_is_cache_aligned() {
        let cache = ContiguousKVCache::new(4, 256, 128);
        assert!(
            cache.is_cache_aligned(),
            "Cache should be aligned to 64-byte cache lines"
        );
    }

    #[test]
    fn test_layer_stride_is_cache_line_aligned() {
        let cache = ContiguousKVCache::new(4, 256, 128);
        let stride = cache.layer_stride();

        // Layer stride should be multiple of 16 (floats per cache line)
        assert_eq!(
            stride % 16,
            0,
            "Layer stride {} should be multiple of 16 floats",
            stride
        );
    }

    #[test]
    fn test_layer_stride_various_dimensions() {
        // Test that alignment works for various dimensions
        let cases = [(1, 17, 10), (2, 33, 5), (4, 100, 20), (8, 256, 128)];

        for (num_layers, hidden_dim, max_seq) in cases {
            let cache = ContiguousKVCache::new(num_layers, hidden_dim, max_seq);
            let stride = cache.layer_stride();

            assert!(
                stride >= hidden_dim * max_seq,
                "Stride {} should be >= raw size {} for ({}, {}, {})",
                stride,
                hidden_dim * max_seq,
                num_layers,
                hidden_dim,
                max_seq
            );
            assert_eq!(
                stride % 16,
                0,
                "Stride should be cache-line aligned for ({}, {}, {})",
                num_layers,
                hidden_dim,
                max_seq
            );
        }
    }

    // ------------------------------------------------------------------------
    // Append Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_append_single_layer() {
        let mut cache = ContiguousKVCache::new(2, 64, 16);

        let k = vec![1.0f32; 64];
        let v = vec![2.0f32; 64];

        cache.append(0, &k, &v);
        cache.advance();

        let k_cache = cache.get_k(0);
        let v_cache = cache.get_v(0);

        assert_eq!(k_cache.len(), 64);
        assert_eq!(v_cache.len(), 64);
        assert!(k_cache.iter().all(|&x| (x - 1.0).abs() < 1e-6));
        assert!(v_cache.iter().all(|&x| (x - 2.0).abs() < 1e-6));
    }

    #[test]
    fn test_append_multiple_positions() {
        let mut cache = ContiguousKVCache::new(1, 16, 8);

        for pos in 0..4 {
            let k = vec![pos as f32; 16];
            let v = vec![pos as f32 + 100.0; 16];
            cache.append(0, &k, &v);
            cache.advance();
        }

        assert_eq!(cache.len(), 4);

        let k_cache = cache.get_k(0);
        let v_cache = cache.get_v(0);

        assert_eq!(k_cache.len(), 64, "Should have 4 * 16 = 64 elements");
        assert_eq!(v_cache.len(), 64);

        // Verify each position has correct values
        for pos in 0..4 {
            let start = pos * 16;
            let end = start + 16;
            let expected_k = pos as f32;
            let expected_v = pos as f32 + 100.0;

            for i in start..end {
                assert!(
                    (k_cache[i] - expected_k).abs() < 1e-6,
                    "K[{}] should be {}",
                    i,
                    expected_k
                );
                assert!(
                    (v_cache[i] - expected_v).abs() < 1e-6,
                    "V[{}] should be {}",
                    i,
                    expected_v
                );
            }
        }
    }

    #[test]
    fn test_append_to_invalid_layer_is_no_op() {
        let mut cache = ContiguousKVCache::new(2, 32, 8);

        let k = vec![1.0f32; 32];
        let v = vec![2.0f32; 32];

        // Append to invalid layer
        cache.append(10, &k, &v);
        cache.advance();

        // Should not crash, cache remains at length 1 but no data in valid layers
        // Since advance happened, len is 1 but get_k/get_v returns empty before any valid append
        // Actually, the seq_len advances regardless. Let's verify valid layers are unaffected.
        let k0 = cache.get_k(0);
        // After advance, get_k returns slice of length seq_len * hidden_dim
        // Since we only advanced but didn't append to layer 0, it will have zeros
        assert_eq!(
            k0.len(),
            32,
            "Layer 0 should have seq_len * hidden_dim zeros"
        );
    }

    #[test]
    fn test_append_beyond_max_seq_len_is_no_op() {
        let mut cache = ContiguousKVCache::new(1, 16, 2);

        let k = vec![1.0f32; 16];
        let v = vec![2.0f32; 16];

        // Fill to capacity
        cache.append(0, &k, &v);
        cache.advance();
        cache.append(0, &k, &v);
        cache.advance();

        assert_eq!(cache.len(), 2);

        // Try to append beyond capacity
        let k2 = vec![99.0f32; 16];
        let v2 = vec![99.0f32; 16];
        cache.append(0, &k2, &v2);

        // Length should not increase
        assert_eq!(cache.len(), 2);
    }

    // ------------------------------------------------------------------------
    // Advance Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_advance_increments_seq_len() {
        let mut cache = ContiguousKVCache::new(2, 64, 10);

        assert_eq!(cache.len(), 0);

        cache.advance();
        assert_eq!(cache.len(), 1);

        cache.advance();
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_advance_stops_at_max_seq_len() {
        let mut cache = ContiguousKVCache::new(1, 16, 3);

        for _ in 0..10 {
            cache.advance();
        }

        assert_eq!(cache.len(), 3);
    }

    // ------------------------------------------------------------------------
    // Get K/V Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_get_k_empty_cache() {
        let cache = ContiguousKVCache::new(2, 32, 8);

        assert!(cache.get_k(0).is_empty());
        assert!(cache.get_k(1).is_empty());
    }

    #[test]
    fn test_get_v_empty_cache() {
        let cache = ContiguousKVCache::new(2, 32, 8);

        assert!(cache.get_v(0).is_empty());
        assert!(cache.get_v(1).is_empty());
    }

    #[test]
    fn test_get_k_invalid_layer_returns_empty() {
        let cache = ContiguousKVCache::new(2, 32, 8);

        assert!(cache.get_k(10).is_empty());
    }

    #[test]
    fn test_get_v_invalid_layer_returns_empty() {
        let cache = ContiguousKVCache::new(2, 32, 8);

        assert!(cache.get_v(10).is_empty());
    }

    #[test]
    fn test_get_k_mut_empty_cache() {
        let mut cache = ContiguousKVCache::new(2, 32, 8);

        assert!(cache.get_k_mut(0).is_empty());
        assert!(cache.get_k_mut(1).is_empty());
    }

    #[test]
    fn test_get_v_mut_empty_cache() {
        let mut cache = ContiguousKVCache::new(2, 32, 8);

        assert!(cache.get_v_mut(0).is_empty());
        assert!(cache.get_v_mut(1).is_empty());
    }

    #[test]
    fn test_get_k_mut_invalid_layer_returns_empty() {
        let mut cache = ContiguousKVCache::new(2, 32, 8);

        assert!(cache.get_k_mut(10).is_empty());
    }

    #[test]
    fn test_get_v_mut_invalid_layer_returns_empty() {
        let mut cache = ContiguousKVCache::new(2, 32, 8);

        assert!(cache.get_v_mut(10).is_empty());
    }

    #[test]
    fn test_get_k_mut_allows_modification() {
        let mut cache = ContiguousKVCache::new(1, 8, 4);

        let k = vec![1.0f32; 8];
        let v = vec![2.0f32; 8];
        cache.append(0, &k, &v);
        cache.advance();

        // Modify via mut reference
        let k_mut = cache.get_k_mut(0);
        assert_eq!(k_mut.len(), 8);
        for x in k_mut.iter_mut() {
            *x = 99.0;
        }

        // Verify modification persisted
        let k_cache = cache.get_k(0);
        assert!(k_cache.iter().all(|&x| (x - 99.0).abs() < 1e-6));
    }

    #[test]
    fn test_get_v_mut_allows_modification() {
        let mut cache = ContiguousKVCache::new(1, 8, 4);

        let k = vec![1.0f32; 8];
        let v = vec![2.0f32; 8];
        cache.append(0, &k, &v);
        cache.advance();

        // Modify via mut reference
        let v_mut = cache.get_v_mut(0);
        assert_eq!(v_mut.len(), 8);
        for x in v_mut.iter_mut() {
            *x = 88.0;
        }

        // Verify modification persisted
        let v_cache = cache.get_v(0);
        assert!(v_cache.iter().all(|&x| (x - 88.0).abs() < 1e-6));
    }

    // ------------------------------------------------------------------------
    // Reset Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_reset_clears_seq_len() {
        let mut cache = ContiguousKVCache::new(2, 16, 8);

        let k = vec![1.0f32; 16];
        let v = vec![2.0f32; 16];

        for _ in 0..4 {
            cache.append(0, &k, &v);
            cache.append(1, &k, &v);
            cache.advance();
        }

        assert_eq!(cache.len(), 4);

        cache.reset();

        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert!(cache.get_k(0).is_empty());
        assert!(cache.get_v(0).is_empty());
    }

    #[test]
    fn test_reset_and_zero_clears_all_data() {
        let mut cache = ContiguousKVCache::new(1, 8, 4);

        let k = vec![1.0f32; 8];
        let v = vec![2.0f32; 8];
        cache.append(0, &k, &v);
        cache.advance();

        cache.reset_and_zero();

        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());

        // After reset_and_zero, if we advance and check, data should be zeros
        cache.advance();
        let k_cache = cache.get_k(0);
        assert!(
            k_cache.iter().all(|&x| x.abs() < 1e-6),
            "Data should be zeroed"
        );
    }

    #[test]
    fn test_reset_allows_reuse() {
        let mut cache = ContiguousKVCache::new(1, 8, 4);

        // Fill
        for i in 0..3 {
            let k = vec![i as f32; 8];
            let v = vec![i as f32; 8];
            cache.append(0, &k, &v);
            cache.advance();
        }

        // Reset
        cache.reset();

        // Reuse
        let k = vec![99.0f32; 8];
        let v = vec![99.0f32; 8];
        cache.append(0, &k, &v);
        cache.advance();

        assert_eq!(cache.len(), 1);
        let k_cache = cache.get_k(0);
        assert_eq!(k_cache.len(), 8);
        assert!(k_cache.iter().all(|&x| (x - 99.0).abs() < 1e-6));
    }

    // ------------------------------------------------------------------------
    // Memory Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_memory_bytes() {
        let cache = ContiguousKVCache::new(4, 256, 128);
        let bytes = cache.memory_bytes();

        // Should be > 0 and reasonable
        assert!(bytes > 0, "Memory usage should be positive");

        // At minimum: 2 * num_layers * max_seq * hidden_dim * 4 bytes
        // But aligned, so will be somewhat larger
        let min_bytes = 2 * 4 * 256 * 128 * 4;
        assert!(
            bytes >= min_bytes,
            "Memory {} should be >= minimum {}",
            bytes,
            min_bytes
        );
    }

    #[test]
    fn test_memory_bytes_zero_dimensions() {
        let cache = ContiguousKVCache::new(0, 0, 0);
        let bytes = cache.memory_bytes();

        assert_eq!(bytes, 0, "Zero-sized cache should use 0 bytes");
    }

    // ------------------------------------------------------------------------
    // Prefetch Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_prefetch_k_valid_layer() {
        let cache = ContiguousKVCache::new(4, 256, 128);

        // Should not panic
        cache.prefetch_k(0);
        cache.prefetch_k(1);
        cache.prefetch_k(2);
        cache.prefetch_k(3);
    }

    #[test]
    fn test_prefetch_k_invalid_layer_is_no_op() {
        let cache = ContiguousKVCache::new(4, 256, 128);

        // Should not panic for invalid layer
        cache.prefetch_k(10);
        cache.prefetch_k(100);
    }

    #[test]
    fn test_prefetch_v_valid_layer() {
        let cache = ContiguousKVCache::new(4, 256, 128);

        // Should not panic
        cache.prefetch_v(0);
        cache.prefetch_v(1);
        cache.prefetch_v(2);
        cache.prefetch_v(3);
    }

    #[test]
    fn test_prefetch_v_invalid_layer_is_no_op() {
        let cache = ContiguousKVCache::new(4, 256, 128);

        // Should not panic for invalid layer
        cache.prefetch_v(10);
        cache.prefetch_v(100);
    }

    // ------------------------------------------------------------------------
    // Multi-Layer Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_multiple_layers_independent() {
        let mut cache = ContiguousKVCache::new(3, 16, 8);

        // Append different values to each layer
        for layer in 0..3 {
            let k = vec![layer as f32 + 1.0; 16];
            let v = vec![layer as f32 + 10.0; 16];
            cache.append(layer, &k, &v);
        }
        cache.advance();

        // Verify each layer has correct values
        for layer in 0..3 {
            let k_cache = cache.get_k(layer);
            let v_cache = cache.get_v(layer);

            let expected_k = layer as f32 + 1.0;
            let expected_v = layer as f32 + 10.0;

            assert!(
                k_cache.iter().all(|&x| (x - expected_k).abs() < 1e-6),
                "Layer {} K should be {}",
                layer,
                expected_k
            );
            assert!(
                v_cache.iter().all(|&x| (x - expected_v).abs() < 1e-6),
                "Layer {} V should be {}",
                layer,
                expected_v
            );
        }
    }
}

// ============================================================================
// GGUFConfig Tests (Supporting Structure)
// ============================================================================

mod gguf_config_for_cache {
    use super::*;

    #[test]
    fn test_basic_config_for_cache_creation() {
        let config = make_test_config(4, 256);

        // Should be able to create caches from config
        let owned_cache = OwnedQuantizedKVCache::from_config(&config, 64);
        let contiguous_cache = ContiguousKVCache::from_config(&config, 64);

        assert_eq!(owned_cache.max_len(), 64);
        assert_eq!(contiguous_cache.max_len(), 64);
    }

    #[test]
    fn test_config_with_various_dimensions() {
        let configs = [
            make_test_config(1, 64),
            make_test_config(4, 256),
            make_test_config(32, 4096),
        ];

        for config in &configs {
            let owned = OwnedQuantizedKVCache::from_config(config, 128);
            let contiguous = ContiguousKVCache::from_config(config, 128);

            assert_eq!(owned.max_len(), 128);
            assert_eq!(contiguous.max_len(), 128);
            assert!(owned.is_empty());
            assert!(contiguous.is_empty());
        }
    }
}

// ============================================================================
// Edge Case and Boundary Tests
// ============================================================================

mod boundary_tests {
    use super::*;

    #[test]
    fn test_owned_single_layer_single_position() {
        let mut cache = OwnedQuantizedKVCache::new(1, 1, 1);

        let k = vec![42.0f32];
        let v = vec![84.0f32];

        cache.append(0, &k, &v);
        cache.advance();

        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());

        let k_cache = cache.get_k(0);
        let v_cache = cache.get_v(0);

        assert_eq!(k_cache.len(), 1);
        assert_eq!(v_cache.len(), 1);
        assert!((k_cache[0] - 42.0).abs() < 1e-6);
        assert!((v_cache[0] - 84.0).abs() < 1e-6);
    }

    #[test]
    fn test_contiguous_single_layer_single_position() {
        let mut cache = ContiguousKVCache::new(1, 1, 1);

        let k = vec![42.0f32];
        let v = vec![84.0f32];

        cache.append(0, &k, &v);
        cache.advance();

        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());

        let k_cache = cache.get_k(0);
        let v_cache = cache.get_v(0);

        assert_eq!(k_cache.len(), 1);
        assert_eq!(v_cache.len(), 1);
        assert!((k_cache[0] - 42.0).abs() < 1e-6);
        assert!((v_cache[0] - 84.0).abs() < 1e-6);
    }

    #[test]
    fn test_owned_large_dimensions() {
        // Test with realistic model dimensions
        let cache = OwnedQuantizedKVCache::new(32, 4096, 2048);

        assert_eq!(cache.len(), 0);
        assert_eq!(cache.max_len(), 2048);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_contiguous_large_dimensions() {
        // Test with realistic model dimensions
        let cache = ContiguousKVCache::new(32, 4096, 2048);

        assert_eq!(cache.len(), 0);
        assert_eq!(cache.max_len(), 2048);
        assert!(cache.is_empty());
        assert!(cache.is_cache_aligned());
    }

    #[test]
    fn test_owned_exact_capacity_fill() {
        let mut cache = OwnedQuantizedKVCache::new(1, 8, 4);

        // Fill exactly to capacity
        for i in 0..4 {
            let k = vec![i as f32; 8];
            let v = vec![i as f32; 8];
            cache.append(0, &k, &v);
            cache.advance();
        }

        assert_eq!(cache.len(), 4);
        assert_eq!(cache.len(), cache.max_len());
    }

    #[test]
    fn test_contiguous_exact_capacity_fill() {
        let mut cache = ContiguousKVCache::new(1, 8, 4);

        // Fill exactly to capacity
        for i in 0..4 {
            let k = vec![i as f32; 8];
            let v = vec![i as f32; 8];
            cache.append(0, &k, &v);
            cache.advance();
        }

        assert_eq!(cache.len(), 4);
        assert_eq!(cache.len(), cache.max_len());
    }
}

// ============================================================================
// Interoperability Tests
// ============================================================================

mod interoperability_tests {
    use super::*;

    #[test]
    fn test_owned_and_contiguous_produce_same_results() {
        let num_layers = 2;
        let hidden_dim = 16;
        let max_seq = 8;

        let mut owned = OwnedQuantizedKVCache::new(num_layers, hidden_dim, max_seq);
        let mut contiguous = ContiguousKVCache::new(num_layers, hidden_dim, max_seq);

        // Fill with same data
        for pos in 0..4 {
            for layer in 0..num_layers {
                let k: Vec<f32> = (0..hidden_dim)
                    .map(|i| (pos * 100 + layer * 10 + i) as f32)
                    .collect();
                let v: Vec<f32> = (0..hidden_dim)
                    .map(|i| (pos * 100 + layer * 10 + i + 1000) as f32)
                    .collect();

                owned.append(layer, &k, &v);
                contiguous.append(layer, &k, &v);
            }
            owned.advance();
            contiguous.advance();
        }

        // Verify same lengths
        assert_eq!(owned.len(), contiguous.len());
        assert_eq!(owned.max_len(), contiguous.max_len());

        // Verify same data for each layer
        for layer in 0..num_layers {
            let owned_k = owned.get_k(layer);
            let contiguous_k = contiguous.get_k(layer);

            let owned_v = owned.get_v(layer);
            let contiguous_v = contiguous.get_v(layer);

            assert_eq!(owned_k.len(), contiguous_k.len());
            assert_eq!(owned_v.len(), contiguous_v.len());

            for (i, (&ok, &ck)) in owned_k.iter().zip(contiguous_k.iter()).enumerate() {
                assert!(
                    (ok - ck).abs() < 1e-6,
                    "K mismatch at layer {} index {}: owned={}, contiguous={}",
                    layer,
                    i,
                    ok,
                    ck
                );
            }

            for (i, (&ov, &cv)) in owned_v.iter().zip(contiguous_v.iter()).enumerate() {
                assert!(
                    (ov - cv).abs() < 1e-6,
                    "V mismatch at layer {} index {}: owned={}, contiguous={}",
                    layer,
                    i,
                    ov,
                    cv
                );
            }
        }
    }
}
