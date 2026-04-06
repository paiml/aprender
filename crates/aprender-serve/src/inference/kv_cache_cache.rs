
// ============================================================================
// EXTREME TDD: Comprehensive Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------------
    // KVCache Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_kv_cache_new() {
        let cache = KVCache::new(12, 768, 2048);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_kv_cache_store_and_retrieve() {
        let mut cache = KVCache::new(2, 4, 10);

        let k = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![5.0, 6.0, 7.0, 8.0];

        cache.store(0, &k, &v);
        cache.advance();

        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get_k(0), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(cache.get_v(0), &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_kv_cache_multiple_positions() {
        let mut cache = KVCache::new(1, 2, 10);

        // Store position 0
        cache.store(0, &[1.0, 2.0], &[3.0, 4.0]);
        cache.advance();

        // Store position 1
        cache.store(0, &[5.0, 6.0], &[7.0, 8.0]);
        cache.advance();

        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get_k(0), &[1.0, 2.0, 5.0, 6.0]);
        assert_eq!(cache.get_v(0), &[3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn test_kv_cache_multiple_layers() {
        let mut cache = KVCache::new(2, 2, 10);

        cache.store(0, &[1.0, 2.0], &[3.0, 4.0]);
        cache.store(1, &[5.0, 6.0], &[7.0, 8.0]);
        cache.advance();

        assert_eq!(cache.get_k(0), &[1.0, 2.0]);
        assert_eq!(cache.get_k(1), &[5.0, 6.0]);
        assert_eq!(cache.get_v(0), &[3.0, 4.0]);
        assert_eq!(cache.get_v(1), &[7.0, 8.0]);
    }

    #[test]
    fn test_kv_cache_reset() {
        let mut cache = KVCache::new(1, 4, 10);

        cache.store(0, &[1.0; 4], &[2.0; 4]);
        cache.advance();
        assert_eq!(cache.len(), 1);

        cache.reset();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_kv_cache_is_empty() {
        let mut cache = KVCache::new(1, 4, 10);
        assert!(cache.is_empty());

        cache.store(0, &[1.0; 4], &[1.0; 4]);
        cache.advance();
        assert!(!cache.is_empty());
    }

    #[test]
    fn test_kv_cache_clone() {
        let mut cache = KVCache::new(1, 2, 10);
        cache.store(0, &[1.0, 2.0], &[3.0, 4.0]);
        cache.advance();

        let cloned = cache.clone();
        assert_eq!(cloned.len(), 1);
        assert_eq!(cloned.get_k(0), &[1.0, 2.0]);
    }

    // ------------------------------------------------------------------------
    // OptimizedKVCache Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_optimized_cache_new() {
        let cache = OptimizedKVCache::new(12, 768, 2048);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.max_len(), 2048);
    }

    #[test]
    fn test_optimized_cache_store_and_retrieve() {
        let mut cache = OptimizedKVCache::new(1, 4, 10);

        let k = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![5.0, 6.0, 7.0, 8.0];

        cache.store(0, &k, &v);
        cache.advance();

        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get_k(0), &[1.0, 2.0, 3.0, 4.0]);

        // V is transposed: v[i] at position 0 is at index i * max_seq_len
        let v_transposed = cache.get_v_transposed(0);
        assert_eq!(v_transposed[0], 5.0); // v[0] at pos 0
        assert_eq!(v_transposed[10], 6.0); // v[1] at pos 0 (stride = max_seq_len = 10)
        assert_eq!(v_transposed[20], 7.0); // v[2] at pos 0
        assert_eq!(v_transposed[30], 8.0); // v[3] at pos 0
    }

    #[test]
    fn test_optimized_cache_transposed_v_layout() {
        let mut cache = OptimizedKVCache::new(1, 2, 5);

        // Store position 0
        cache.store(0, &[1.0, 2.0], &[10.0, 20.0]);
        cache.advance();

        // Store position 1
        cache.store(0, &[3.0, 4.0], &[30.0, 40.0]);
        cache.advance();

        let v_transposed = cache.get_v_transposed(0);
        // v[0] positions: indices 0, 1, 2, ... (stride 1)
        // v[1] positions: indices 5, 6, 7, ... (stride 1, offset = hidden_dim * max_seq_len / hidden_dim = max_seq_len)
        assert_eq!(v_transposed[0], 10.0); // v[0] at pos 0
        assert_eq!(v_transposed[1], 30.0); // v[0] at pos 1
        assert_eq!(v_transposed[5], 20.0); // v[1] at pos 0
        assert_eq!(v_transposed[6], 40.0); // v[1] at pos 1
    }

    #[test]
    fn test_optimized_cache_max_capacity() {
        let mut cache = OptimizedKVCache::new(1, 2, 3);

        // Fill to capacity
        for i in 0..3 {
            cache.store(0, &[i as f32; 2], &[i as f32; 2]);
            cache.advance();
        }
        assert_eq!(cache.len(), 3);

        // Should not advance beyond max
        cache.store(0, &[99.0; 2], &[99.0; 2]);
        cache.advance();
        assert_eq!(cache.len(), 3); // Still at max
    }

    #[test]
    fn test_optimized_cache_reset() {
        let mut cache = OptimizedKVCache::new(1, 4, 10);

        cache.store(0, &[1.0; 4], &[2.0; 4]);
        cache.advance();
        cache.reset();

        assert!(cache.is_empty());
    }

    // ------------------------------------------------------------------------
    // attention_with_cache Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_attention_with_cache_no_history() {
        let hidden_dim = 4;
        let num_heads = 2;

        let q = vec![1.0; hidden_dim];
        let k_cache: Vec<f32> = vec![];
        let v_cache: Vec<f32> = vec![];
        let current_k = vec![1.0; hidden_dim];
        let current_v = vec![2.0; hidden_dim];

        let output =
            attention_with_cache(&q, &k_cache, &v_cache, &current_k, &current_v, num_heads);

        assert_eq!(output.len(), hidden_dim);
        // With no history and uniform attention, output should equal current_v
        for &v in &output {
            assert!((v - 2.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_attention_with_cache_one_cached() {
        let hidden_dim = 4;
        let num_heads = 2;

        let q = vec![1.0; hidden_dim];
        // One cached position
        let k_cache = vec![1.0; hidden_dim];
        let v_cache = vec![1.0; hidden_dim];
        let current_k = vec![1.0; hidden_dim];
        let current_v = vec![3.0; hidden_dim];

        let output =
            attention_with_cache(&q, &k_cache, &v_cache, &current_k, &current_v, num_heads);

        assert_eq!(output.len(), hidden_dim);
        // With uniform K, attention is 0.5 to each position
        // output = 0.5 * 1.0 + 0.5 * 3.0 = 2.0
        for &v in &output {
            assert!((v - 2.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_attention_with_cache_multi_head() {
        let hidden_dim = 8;
        let num_heads = 4;
        let head_dim = hidden_dim / num_heads;

        // Each head has different Q
        let mut q = vec![0.0; hidden_dim];
        for h in 0..num_heads {
            for i in 0..head_dim {
                q[h * head_dim + i] = (h + 1) as f32;
            }
        }

        let k_cache: Vec<f32> = vec![];
        let v_cache: Vec<f32> = vec![];
        let current_k = vec![1.0; hidden_dim];
        let current_v = vec![1.0; hidden_dim];

        let output =
            attention_with_cache(&q, &k_cache, &v_cache, &current_k, &current_v, num_heads);

        assert_eq!(output.len(), hidden_dim);
        // All outputs should be 1.0 (current_v with softmax weight 1.0)
        for &v in &output {
            assert!((v - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_attention_preserves_dimension() {
        for hidden_dim in [64, 128, 256, 512] {
            for num_heads in [1, 2, 4, 8] {
                if hidden_dim % num_heads != 0 {
                    continue;
                }

                let q = vec![0.1; hidden_dim];
                let k_cache = vec![0.1; hidden_dim * 2];
                let v_cache = vec![0.2; hidden_dim * 2];
                let current_k = vec![0.1; hidden_dim];
                let current_v = vec![0.3; hidden_dim];

                let output =
                    attention_with_cache(&q, &k_cache, &v_cache, &current_k, &current_v, num_heads);

                assert_eq!(output.len(), hidden_dim);
            }
        }
    }

    // ------------------------------------------------------------------------
    // attention_with_transposed_v Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_transposed_attention_no_history() {
        let hidden_dim = 4;
        let num_heads = 2;
        let max_seq_len = 10;

        let q = vec![1.0; hidden_dim];
        let k_cache: Vec<f32> = vec![];
        let v_cache_transposed = vec![0.0; hidden_dim * max_seq_len];
        let current_k = vec![1.0; hidden_dim];
        let current_v = vec![2.0; hidden_dim];

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
        // Output should equal current_v when no history
        for &v in &output {
            assert!((v - 2.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_transposed_attention_one_cached() {
        let hidden_dim = 4;
        let num_heads = 2;
        let max_seq_len = 10;

        let q = vec![1.0; hidden_dim];
        let k_cache = vec![1.0; hidden_dim]; // 1 cached position

        // Transposed V: v[i] at pos j is at index i * max_seq_len + j
        let mut v_cache_transposed = vec![0.0; hidden_dim * max_seq_len];
        for i in 0..hidden_dim {
            v_cache_transposed[i * max_seq_len] = 1.0; // v[i] = 1.0 at pos 0
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

        // Uniform attention: 0.5 * 1.0 + 0.5 * 3.0 = 2.0
        for &v in &output {
            assert!((v - 2.0).abs() < 1e-5);
        }
    }

    // ------------------------------------------------------------------------
    // Integration Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_cache_and_attention_integration() {
        let num_layers = 2;
        let hidden_dim = 4;
        let max_seq_len = 10;
        let num_heads = 2;

        let mut cache = KVCache::new(num_layers, hidden_dim, max_seq_len);

        // Simulate 3 tokens
        for pos in 0..3 {
            let k = vec![pos as f32; hidden_dim];
            let v = vec![(pos * 2) as f32; hidden_dim];

            for layer in 0..num_layers {
                cache.store(layer, &k, &v);
            }
            cache.advance();
        }

        // Now compute attention for token 4
        let q = vec![1.0; hidden_dim];
        let current_k = vec![3.0; hidden_dim];
        let current_v = vec![6.0; hidden_dim];

        let k_cached = cache.get_k(0);
        let v_cached = cache.get_v(0);

        let output =
            attention_with_cache(&q, k_cached, v_cached, &current_k, &current_v, num_heads);

        assert_eq!(output.len(), hidden_dim);
        // Output should be some weighted combination
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_optimized_cache_and_attention_integration() {
        let num_layers = 1;
        let hidden_dim = 4;
        let max_seq_len = 10;
        let num_heads = 2;

        let mut cache = OptimizedKVCache::new(num_layers, hidden_dim, max_seq_len);

        // Store 2 positions
        cache.store(0, &[1.0; 4], &[2.0; 4]);
        cache.advance();
        cache.store(0, &[1.0; 4], &[4.0; 4]);
        cache.advance();

        let q = vec![1.0; hidden_dim];
        let current_k = vec![1.0; hidden_dim];
        let current_v = vec![6.0; hidden_dim];

        let output = attention_with_transposed_v(
            &q,
            cache.get_k(0),
            cache.get_v_transposed(0),
            &current_k,
            &current_v,
            num_heads,
            max_seq_len,
        );

        assert_eq!(output.len(), hidden_dim);
        // Uniform attention: (2 + 4 + 6) / 3 = 4.0
        for &v in &output {
            assert!((v - 4.0).abs() < 1e-5);
        }
    }

    // ------------------------------------------------------------------------
    // Edge Cases
    // ------------------------------------------------------------------------

    #[test]
    fn test_single_head_attention() {
        let hidden_dim = 4;
        let num_heads = 1;

        let q = vec![1.0; hidden_dim];
        let k_cache: Vec<f32> = vec![];
        let v_cache: Vec<f32> = vec![];
        let current_k = vec![1.0; hidden_dim];
        let current_v = vec![5.0; hidden_dim];

        let output =
            attention_with_cache(&q, &k_cache, &v_cache, &current_k, &current_v, num_heads);

        for &v in &output {
            assert!((v - 5.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_many_heads_attention() {
        let hidden_dim = 64;
        let num_heads = 16;

        let q = vec![0.1; hidden_dim];
        let k_cache: Vec<f32> = vec![];
        let v_cache: Vec<f32> = vec![];
        let current_k = vec![0.1; hidden_dim];
        let current_v = vec![1.0; hidden_dim];

        let output =
            attention_with_cache(&q, &k_cache, &v_cache, &current_k, &current_v, num_heads);

        assert_eq!(output.len(), hidden_dim);
        for &v in &output {
            assert!((v - 1.0).abs() < 1e-5);
        }
    }
}
