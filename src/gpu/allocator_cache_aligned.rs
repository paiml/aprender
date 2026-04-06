
#[cfg(test)]
mod tests {
    use super::*;

    // ==================== CacheAlignedBuffer Tests ====================

    #[test]
    fn test_cache_aligned_buffer_new() {
        let buf = CacheAlignedBuffer::new(100);
        assert_eq!(buf.len(), 100);
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_cache_aligned_buffer_empty() {
        let buf = CacheAlignedBuffer::new(0);
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_cache_aligned_buffer_is_aligned() {
        let buf = CacheAlignedBuffer::new(256);
        // Should be aligned to 64 bytes (cache line)
        assert!(buf.is_aligned(CACHE_LINE_SIZE));
    }

    #[test]
    fn test_cache_aligned_buffer_as_slice() {
        let buf = CacheAlignedBuffer::new(10);
        let slice = buf.as_slice();
        assert_eq!(slice.len(), 10);
        assert!(slice.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_cache_aligned_buffer_as_mut_slice() {
        let mut buf = CacheAlignedBuffer::new(5);
        {
            let slice = buf.as_mut_slice();
            slice[0] = 1.0;
            slice[4] = 5.0;
        }
        assert!((buf.as_slice()[0] - 1.0).abs() < 1e-6);
        assert!((buf.as_slice()[4] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_cache_aligned_buffer_large() {
        let buf = CacheAlignedBuffer::new(10000);
        assert_eq!(buf.len(), 10000);
        assert!(buf.is_aligned(CACHE_LINE_SIZE));
    }

    // ==================== Prefetch Tests ====================

    #[test]
    fn test_prefetch_read_in_bounds() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // Should not panic
        prefetch_read(&data, 0, 2);
    }

    #[test]
    fn test_prefetch_read_out_of_bounds() {
        let data = vec![1.0, 2.0, 3.0];
        // Should not panic when prefetch would be out of bounds
        prefetch_read(&data, 2, 10);
    }

    #[test]
    fn test_prefetch_read_empty() {
        let data: Vec<f32> = vec![];
        prefetch_read(&data, 0, 1);
    }

    // ==================== Sequential Sum Tests ====================

    #[test]
    fn test_sequential_sum_empty() {
        let data: Vec<f32> = vec![];
        assert!((sequential_sum(&data) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_sequential_sum_single() {
        let data = vec![5.0];
        assert!((sequential_sum(&data) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_sequential_sum_multiple() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((sequential_sum(&data) - 15.0).abs() < 1e-6);
    }

    // ==================== Sum with Prefetch Tests ====================

    #[test]
    fn test_sum_with_prefetch_empty() {
        let data: Vec<f32> = vec![];
        assert!((sum_with_prefetch(&data, 8) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_sum_with_prefetch_matches_sequential() {
        let data: Vec<f32> = (1..=100).map(|x| x as f32).collect();
        let seq_sum = sequential_sum(&data);
        let pf_sum = sum_with_prefetch(&data, 16);
        assert!((seq_sum - pf_sum).abs() < 1e-6);
    }

    #[test]
    fn test_sum_with_prefetch_various_distances() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((sum_with_prefetch(&data, 1) - 15.0).abs() < 1e-6);
        assert!((sum_with_prefetch(&data, 100) - 15.0).abs() < 1e-6);
    }

    // ==================== Naive Matmul Tests ====================

    #[test]
    fn test_naive_matmul_identity() {
        // 2x2 identity @ 2x2 matrix = same matrix
        let identity = vec![1.0, 0.0, 0.0, 1.0];
        let mat = vec![1.0, 2.0, 3.0, 4.0];
        let result = naive_matmul(&identity, &mat, 2, 2, 2);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 2.0).abs() < 1e-6);
        assert!((result[2] - 3.0).abs() < 1e-6);
        assert!((result[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_naive_matmul_simple() {
        // [1, 2] @ [[1], [3]] = [7]
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 3.0];
        let result = naive_matmul(&a, &b, 1, 2, 1);
        assert!((result[0] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_naive_matmul_2x3_3x2() {
        // (2,3) @ (3,2) = (2,2)
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let result = naive_matmul(&a, &b, 2, 3, 2);
        // result[0,0] = 1*7 + 2*9 + 3*11 = 7+18+33 = 58
        assert!((result[0] - 58.0).abs() < 1e-6);
    }

    // ==================== Blocked Matmul Tests ====================

    #[test]
    fn test_blocked_matmul_matches_naive() {
        let a: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let b: Vec<f32> = (13..=24).map(|x| x as f32).collect();

        let naive = naive_matmul(&a, &b, 3, 4, 3);
        let blocked = blocked_matmul(&a, &b, 3, 4, 3, 2);

        for (n, b) in naive.iter().zip(blocked.iter()) {
            assert!((n - b).abs() < 1e-4);
        }
    }

    #[test]
    fn test_blocked_matmul_large() {
        let size = 64;
        let a: Vec<f32> = (0..size * size).map(|x| (x as f32) * 0.01).collect();
        let b: Vec<f32> = (0..size * size).map(|x| (x as f32) * 0.01).collect();

        let naive = naive_matmul(&a, &b, size, size, size);
        let blocked = blocked_matmul(&a, &b, size, size, size, 16);

        for (n, b) in naive.iter().zip(blocked.iter()) {
            assert!((n - b).abs() < 1e-2);
        }
    }

    #[test]
    fn test_blocked_matmul_block_larger_than_matrix() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let result = blocked_matmul(&a, &b, 2, 2, 2, 100);
        let expected = naive_matmul(&a, &b, 2, 2, 2);
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-6);
        }
    }

    // ==================== TensorPool Tests ====================

    #[test]
    fn test_tensor_pool_new() {
        let pool = TensorPool::new(10);
        assert_eq!(pool.capacity(), 10);
        assert_eq!(pool.available(), 0);
    }

    #[test]
    fn test_tensor_pool_acquire_new() {
        let mut pool = TensorPool::new(5);
        let buf = pool.acquire(100);
        assert_eq!(buf.len(), 100);
        assert!(buf.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_tensor_pool_release_reuse() {
        let mut pool = TensorPool::new(5);
        let buf = pool.acquire(50);
        pool.release(buf);
        assert_eq!(pool.available(), 1);

        let buf2 = pool.acquire(40);
        assert!(buf2.capacity() >= 50); // Reused buffer
        assert_eq!(pool.available(), 0);
    }

    #[test]
    fn test_tensor_pool_capacity_limit() {
        let mut pool = TensorPool::new(2);
        pool.release(vec![0.0; 10]);
        pool.release(vec![0.0; 20]);
        pool.release(vec![0.0; 30]); // Should be dropped
        assert_eq!(pool.available(), 2);
    }

    #[test]
    fn test_tensor_pool_clear() {
        let mut pool = TensorPool::new(5);
        pool.release(vec![0.0; 10]);
        pool.release(vec![0.0; 20]);
        assert_eq!(pool.available(), 2);
        pool.clear();
        assert_eq!(pool.available(), 0);
    }

    #[test]
    fn test_tensor_pool_acquire_larger() {
        let mut pool = TensorPool::new(5);
        pool.release(vec![0.0; 10]);
        // Acquire larger than any pooled buffer
        let buf = pool.acquire(100);
        assert_eq!(buf.len(), 100);
        assert_eq!(pool.available(), 1); // Small buffer still in pool
    }

    // ==================== ForwardArena Tests ====================

    #[test]
    fn test_forward_arena_new() {
        let arena = ForwardArena::new(1000);
        assert_eq!(arena.capacity(), 1000);
        assert_eq!(arena.used(), 0);
    }

    #[test]
    fn test_forward_arena_alloc() {
        let mut arena = ForwardArena::new(100);
        let slice = arena.alloc(20);
        assert_eq!(slice.len(), 20);
        assert_eq!(arena.used(), 20);
    }

    #[test]
    fn test_forward_arena_alloc_multiple() {
        let mut arena = ForwardArena::new(100);
        let _s1 = arena.alloc(30);
        let _s2 = arena.alloc(40);
        assert_eq!(arena.used(), 70);
    }

    #[test]
    fn test_forward_arena_reset() {
        let mut arena = ForwardArena::new(100);
        arena.alloc(50);
        assert_eq!(arena.used(), 50);
        arena.reset();
        assert_eq!(arena.used(), 0);
    }

    #[test]
    fn test_forward_arena_reuse_after_reset() {
        let mut arena = ForwardArena::new(100);
        {
            let slice = arena.alloc(50);
            slice[0] = 42.0;
        }
        arena.reset();
        let slice2 = arena.alloc(50);
        // Data may still contain old values, but that's ok
        assert_eq!(slice2.len(), 50);
    }

    #[test]
    #[should_panic(expected = "insufficient capacity")]
    fn test_forward_arena_overflow() {
        let mut arena = ForwardArena::new(10);
        arena.alloc(20);
    }

    // ==================== ScratchBuffer Tests ====================

    #[test]
    fn test_scratch_buffer_new() {
        let scratch = ScratchBuffer::new(4, 100);
        assert_eq!(scratch.num_layers(), 4);
        assert_eq!(scratch.layer_size(), 100);
        assert_eq!(scratch.total_size(), 400);
    }

    #[test]
    fn test_scratch_buffer_get_layer() {
        let scratch = ScratchBuffer::new(3, 50);
        let layer0 = scratch.get_layer(0);
        let layer2 = scratch.get_layer(2);
        assert_eq!(layer0.len(), 50);
        assert_eq!(layer2.len(), 50);
    }

    #[test]
    fn test_scratch_buffer_get_layer_mut() {
        let mut scratch = ScratchBuffer::new(2, 10);
        {
            let layer = scratch.get_layer_mut(0);
            layer[0] = 1.0;
            layer[9] = 9.0;
        }
        {
            let layer = scratch.get_layer_mut(1);
            layer[0] = 100.0;
        }
        assert!((scratch.get_layer(0)[0] - 1.0).abs() < 1e-6);
        assert!((scratch.get_layer(0)[9] - 9.0).abs() < 1e-6);
        assert!((scratch.get_layer(1)[0] - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_scratch_buffer_reset() {
        let mut scratch = ScratchBuffer::new(2, 5);
        scratch.get_layer_mut(0)[0] = 42.0;
        scratch.get_layer_mut(1)[4] = 99.0;
        scratch.reset();
        assert!((scratch.get_layer(0)[0] - 0.0).abs() < 1e-6);
        assert!((scratch.get_layer(1)[4] - 0.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "layer index")]
    fn test_scratch_buffer_out_of_bounds() {
        let scratch = ScratchBuffer::new(3, 10);
        let _ = scratch.get_layer(5);
    }

    #[test]
    #[should_panic(expected = "layer index")]
    fn test_scratch_buffer_mut_out_of_bounds() {
        let mut scratch = ScratchBuffer::new(3, 10);
        scratch.get_layer_mut(3);
    }

    #[test]
    fn test_scratch_buffer_zero_layers() {
        let scratch = ScratchBuffer::new(0, 100);
        assert_eq!(scratch.num_layers(), 0);
        assert_eq!(scratch.total_size(), 0);
    }

    #[test]
    fn test_scratch_buffer_zero_size() {
        let scratch = ScratchBuffer::new(5, 0);
        assert_eq!(scratch.layer_size(), 0);
        let layer = scratch.get_layer(0);
        assert_eq!(layer.len(), 0);
    }
}
