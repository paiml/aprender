
// =========================================================================
// Extended Coverage Tests for FFN functions
// =========================================================================

#[test]
fn test_sequential_ffn_identity_cov() {
    // Test with identity-like weights
    let hidden_dim = 4;
    let inter_dim = 4;
    let hidden = vec![1.0f32; hidden_dim];
    let w1 = vec![0.25f32; hidden_dim * inter_dim]; // Uniform weights
    let w2 = vec![0.25f32; inter_dim * hidden_dim];
    let result = sequential_ffn(&hidden, &w1, &w2, hidden_dim, inter_dim);
    assert_eq!(result.len(), hidden_dim);
}

#[test]
fn test_parallel_ffn_identity_cov() {
    let hidden_dim = 4;
    let inter_dim = 4;
    let hidden = vec![1.0f32; hidden_dim];
    let w1 = vec![0.25f32; hidden_dim * inter_dim];
    let w2 = vec![0.25f32; inter_dim * hidden_dim];
    let result = parallel_ffn(&hidden, &w1, &w2, hidden_dim, inter_dim);
    assert_eq!(result.len(), hidden_dim);
}

// =========================================================================
// Extended Coverage Tests for batch_embed
// =========================================================================

#[test]
fn test_batch_embed_single_token_cov() {
    let embedding_table = vec![1.0f32; 10 * 4]; // 10 tokens, dim 4
    let tokens = vec![0usize];
    let result = batch_embed(&embedding_table, &tokens, 4);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_batch_embed_last_token_cov() {
    let embedding_table = vec![1.0f32; 10 * 4]; // 10 tokens, dim 4
    let tokens = vec![9usize]; // Last token
    let result = batch_embed(&embedding_table, &tokens, 4);
    assert_eq!(result.len(), 4);
}

// =========================================================================
// Extended Coverage Tests for prefetch
// =========================================================================

#[test]
fn test_prefetch_read_boundary_cov() {
    let data = vec![1.0f32; 10];
    prefetch_read(&data, 0, 10); // Full range
    prefetch_read(&data, 9, 1); // Last element
    prefetch_read(&data, 5, 0); // Zero count
}

// =========================================================================
// Extended Coverage Tests for Quantized Operations
// =========================================================================

#[test]
fn test_quantized_dot_q4_nonzero_cov() {
    let mut block_a = vec![0u8; 18];
    let mut block_b = vec![0u8; 18];
    // Set non-zero scale (f16 at bytes 0-1)
    block_a[0] = 0x00;
    block_a[1] = 0x3C; // f16 ≈ 1.0
    block_b[0] = 0x00;
    block_b[1] = 0x3C;
    // Set some non-zero quants
    block_a[2] = 0xFF;
    block_b[2] = 0xFF;
    let result = quantized_dot_q4(&block_a, &block_b);
    assert!(result.is_finite());
}

#[test]
fn test_quantized_dot_q8_nonzero_cov() {
    let mut block_a = vec![0u8; 34];
    let mut block_b = vec![0u8; 34];
    // Set non-zero scale
    block_a[0] = 0x00;
    block_a[1] = 0x3C;
    block_b[0] = 0x00;
    block_b[1] = 0x3C;
    // Set some non-zero quants
    block_a[2] = 127;
    block_b[2] = 127;
    let result = quantized_dot_q8(&block_a, &block_b);
    assert!(result.is_finite());
}

// =========================================================================
// Deep Coverage Tests for gpu.rs (Phase 802)
// =========================================================================

// --- exceeds_gpu_buffer_limit tests ---
#[test]
fn test_exceeds_gpu_buffer_limit_small_cov() {
    // Small buffer should not exceed limit
    assert!(!exceeds_gpu_buffer_limit(1000));
    assert!(!exceeds_gpu_buffer_limit(0));
}

#[test]
fn test_exceeds_gpu_buffer_limit_large_cov() {
    // Large buffer should exceed limit (256MB / 4 bytes = 67108864 f32s)
    let limit_elements = 256 * 1024 * 1024 / 4;
    assert!(exceeds_gpu_buffer_limit(limit_elements + 1));
    assert!(exceeds_gpu_buffer_limit(100_000_000));
}

#[test]
fn test_exceeds_gpu_buffer_limit_boundary_cov() {
    // At boundary
    let limit_elements = 256 * 1024 * 1024 / 4;
    // At limit should not exceed (<=)
    assert!(!exceeds_gpu_buffer_limit(limit_elements));
}

// --- scalar_softmax deep tests ---
#[test]
fn test_scalar_softmax_empty_deep2() {
    let result = scalar_softmax(&[]);
    assert!(result.is_empty());
}

#[test]
fn test_scalar_softmax_single_deep2() {
    let result = scalar_softmax(&[0.0]);
    assert_eq!(result.len(), 1);
    assert!((result[0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_scalar_softmax_uniform_deep2() {
    let input = vec![1.0, 1.0, 1.0, 1.0];
    let result = scalar_softmax(&input);
    assert_eq!(result.len(), 4);
    for val in &result {
        assert!((val - 0.25).abs() < 1e-5);
    }
}

#[test]
fn test_scalar_softmax_numerical_stability_deep2() {
    // Large values should not overflow
    let input = vec![1000.0, 1001.0, 1002.0];
    let result = scalar_softmax(&input);
    assert_eq!(result.len(), 3);
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

// --- simd_softmax deep tests ---
#[test]
fn test_simd_softmax_empty_deep2() {
    let result = simd_softmax(&[]);
    assert!(result.is_empty());
}

#[test]
fn test_simd_softmax_single_deep2() {
    let result = simd_softmax(&[0.0]);
    assert_eq!(result.len(), 1);
    assert!((result[0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_simd_softmax_matches_scalar_deep2() {
    let input = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let scalar_result = scalar_softmax(&input);
    let simd_result = simd_softmax(&input);
    assert_eq!(scalar_result.len(), simd_result.len());
    for i in 0..scalar_result.len() {
        assert!((scalar_result[i] - simd_result[i]).abs() < 1e-5);
    }
}

// --- scalar_rope deep tests ---
#[test]
fn test_scalar_rope_empty_deep2() {
    let result = scalar_rope(&[], 0, 0, 10000.0);
    assert!(result.is_empty());
}

#[test]
fn test_scalar_rope_zero_seq_len_deep2() {
    let result = scalar_rope(&[1.0, 2.0, 3.0, 4.0], 0, 4, 10000.0);
    assert!(result.is_empty());
}

#[test]
fn test_scalar_rope_zero_head_dim_deep2() {
    let result = scalar_rope(&[1.0, 2.0, 3.0, 4.0], 1, 0, 10000.0);
    assert!(result.is_empty());
}

#[test]
fn test_scalar_rope_basic_deep2() {
    // Single position, single head, head_dim=4
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let result = scalar_rope(&input, 1, 4, 10000.0);
    assert_eq!(result.len(), 4);
    // Position 0 should apply rotation with angle=0 (cos=1, sin=0)
    // So output should be close to input at position 0
    assert!((result[0] - 1.0).abs() < 1e-4);
}

#[test]
fn test_scalar_rope_multiple_positions_deep2() {
    // 2 positions, 1 head, head_dim=4
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let result = scalar_rope(&input, 2, 4, 10000.0);
    assert_eq!(result.len(), 8);
    // Results should be finite
    for val in &result {
        assert!(val.is_finite());
    }
}

// --- simd_rope deep tests ---
#[test]
fn test_simd_rope_empty_deep2() {
    let result = simd_rope(&[], 0, 0, 10000.0);
    assert!(result.is_empty());
}

#[test]
fn test_simd_rope_matches_scalar_deep2() {
    let input: Vec<f32> = (0..64).map(|x| x as f32 * 0.1).collect();
    let scalar_result = scalar_rope(&input, 2, 16, 10000.0);
    let simd_result = simd_rope(&input, 2, 16, 10000.0);
    assert_eq!(scalar_result.len(), simd_result.len());
    for i in 0..scalar_result.len() {
        assert!(
            (scalar_result[i] - simd_result[i]).abs() < 1e-4,
            "Mismatch at {}: scalar={}, simd={}",
            i,
            scalar_result[i],
            simd_result[i]
        );
    }
}

// --- CacheAlignedBuffer tests ---
#[test]
fn test_cache_aligned_buffer_new_cov() {
    let buf = CacheAlignedBuffer::new(100);
    assert_eq!(buf.len(), 100);
    assert!(!buf.is_empty());
}

#[test]
fn test_cache_aligned_buffer_empty_cov() {
    let buf = CacheAlignedBuffer::new(0);
    assert_eq!(buf.len(), 0);
    assert!(buf.is_empty());
}

#[test]
fn test_cache_aligned_buffer_alignment_cov() {
    let buf = CacheAlignedBuffer::new(256);
    // Should be aligned to 64 bytes
    assert!(buf.is_aligned(64));
}

#[test]
fn test_cache_aligned_buffer_slice_cov() {
    let mut buf = CacheAlignedBuffer::new(10);
    let slice = buf.as_mut_slice();
    slice[0] = 42.0;
    slice[9] = 99.0;
    assert_eq!(buf.as_slice()[0], 42.0);
    assert_eq!(buf.as_slice()[9], 99.0);
}

// --- prefetch_read tests ---
#[test]
fn test_prefetch_read_in_bounds_cov() {
    let data = vec![1.0f32; 100];
    // Should not panic
    prefetch_read(&data, 0, 50);
    prefetch_read(&data, 50, 49);
}

#[test]
fn test_prefetch_read_out_of_bounds_cov() {
    let data = vec![1.0f32; 10];
    // Should be a no-op when out of bounds
    prefetch_read(&data, 5, 100); // position + distance > len
    prefetch_read(&data, 10, 1); // position at end
}

// --- sequential_sum tests ---
#[test]
fn test_sequential_sum_empty_cov() {
    let result = sequential_sum(&[]);
    assert_eq!(result, 0.0);
}

#[test]
fn test_sequential_sum_basic_cov() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = sequential_sum(&data);
    assert!((result - 15.0).abs() < 1e-5);
}

// --- sum_with_prefetch tests ---
#[test]
fn test_sum_with_prefetch_empty_cov() {
    let result = sum_with_prefetch(&[], 8);
    assert_eq!(result, 0.0);
}

#[test]
fn test_sum_with_prefetch_basic_cov() {
    let data: Vec<f32> = (1..=100).map(|x| x as f32).collect();
    let result = sum_with_prefetch(&data, 16);
    let expected = 5050.0; // Sum of 1..100
    assert!((result - expected).abs() < 1e-3);
}

#[test]
fn test_sum_with_prefetch_matches_sequential_cov() {
    let data: Vec<f32> = (0..1000).map(|x| x as f32 * 0.001).collect();
    let seq = sequential_sum(&data);
    let prefetch = sum_with_prefetch(&data, 32);
    assert!((seq - prefetch).abs() < 1e-3);
}

// --- naive_matmul tests ---
#[test]
fn test_naive_matmul_identity_cov() {
    // 2x2 @ 2x2 identity-like
    let a = vec![1.0, 0.0, 0.0, 1.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let result = naive_matmul(&a, &b, 2, 2, 2);
    assert_eq!(result.len(), 4);
    assert!((result[0] - 5.0).abs() < 1e-5);
    assert!((result[3] - 8.0).abs() < 1e-5);
}

#[test]
fn test_naive_matmul_non_square_cov() {
    // 2x3 @ 3x1
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![1.0, 1.0, 1.0];
    let result = naive_matmul(&a, &b, 2, 3, 1);
    assert_eq!(result.len(), 2);
    assert!((result[0] - 6.0).abs() < 1e-5); // 1+2+3
    assert!((result[1] - 15.0).abs() < 1e-5); // 4+5+6
}

// --- blocked_matmul tests ---
#[test]
fn test_blocked_matmul_matches_naive_cov() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    let naive = naive_matmul(&a, &b, 3, 3, 3);
    let blocked = blocked_matmul(&a, &b, 3, 3, 3, 2);
    assert_eq!(naive.len(), blocked.len());
    for i in 0..naive.len() {
        assert!((naive[i] - blocked[i]).abs() < 1e-4);
    }
}

#[test]
fn test_blocked_matmul_large_block_cov() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    // Block size larger than matrix dimensions
    let result = blocked_matmul(&a, &b, 2, 2, 2, 100);
    assert_eq!(result.len(), 4);
    assert!((result[0] - 19.0).abs() < 1e-5);
}

// --- TensorPool tests ---
#[test]
fn test_tensor_pool_new_cov() {
    let pool = TensorPool::new(10);
    assert_eq!(pool.available(), 0);
}

#[test]
fn test_tensor_pool_acquire_release_cov() {
    let mut pool = TensorPool::new(5);
    let buf = pool.acquire(100);
    assert_eq!(buf.len(), 100);
    pool.release(buf);
    assert_eq!(pool.available(), 1);
}

#[test]
fn test_tensor_pool_reuse_cov() {
    let mut pool = TensorPool::new(5);
    let buf1 = pool.acquire(100);
    pool.release(buf1);
    let buf2 = pool.acquire(100);
    assert_eq!(buf2.len(), 100);
    // Should have reused the buffer
    assert_eq!(pool.available(), 0);
}

#[test]
fn test_tensor_pool_size_mismatch_cov() {
    let mut pool = TensorPool::new(5);
    let buf = pool.acquire(100);
    pool.release(buf);
    // Request different size - should allocate new
    let buf2 = pool.acquire(200);
    assert_eq!(buf2.len(), 200);
}

#[test]
fn test_tensor_pool_capacity_limit_cov() {
    let mut pool = TensorPool::new(2);
    let buf1 = pool.acquire(10);
    let buf2 = pool.acquire(10);
    let buf3 = pool.acquire(10);
    pool.release(buf1);
    pool.release(buf2);
    pool.release(buf3); // Should not exceed capacity
    assert!(pool.available() <= 2);
}

// --- DoubleBuffer tests ---
#[test]
fn test_double_buffer_new_cov() {
    let buf: DoubleBuffer<f32> = DoubleBuffer::new(100);
    assert_eq!(buf.capacity(), 100);
}

#[test]
fn test_double_buffer_front_back_cov() {
    let mut buf: DoubleBuffer<f32> = DoubleBuffer::new(10);
    buf.back_mut()[0] = 42.0;
    assert_eq!(buf.front()[0], 0.0);
    buf.swap();
    assert_eq!(buf.front()[0], 42.0);
}
