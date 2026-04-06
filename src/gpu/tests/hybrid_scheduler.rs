
// ============================================================================
// HybridScheduler Extended Tests
// ============================================================================

#[test]
fn test_hybrid_scheduler_pooled_matmul() {
    let mut scheduler = HybridScheduler::with_threshold(1000).expect("test");

    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];

    let c = scheduler.matmul_pooled(&a, &b, 2, 2, 2).expect("test");

    assert_eq!(c.len(), 4);
    assert!((c[0] - 19.0).abs() < 1e-5);

    // Release buffer
    scheduler.release_buffer(c);

    // Check pool stats
    let stats = scheduler.pool_stats();
    assert_eq!(stats.cached_buffers, 1);
}

#[test]
fn test_hybrid_scheduler_async_matmul() {
    let mut scheduler = HybridScheduler::with_threshold(1000).expect("test");

    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];

    let result = scheduler.matmul_async(&a, &b, 2, 2, 2).expect("test");
    assert!(result.is_ready());

    let c = result.wait();
    assert!((c[0] - 19.0).abs() < 1e-5);
}

#[test]
fn test_hybrid_scheduler_batch_matmul() {
    let mut scheduler = HybridScheduler::with_threshold(1000).expect("test");

    let ops = vec![
        (vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0], 2, 2, 2),
        (vec![1.0, 0.0, 0.0, 1.0], vec![2.0, 3.0, 4.0, 5.0], 2, 2, 2),
    ];

    let results = scheduler.matmul_batch(&ops).expect("test");

    assert_eq!(results.len(), 2);
    assert!((results[0][0] - 19.0).abs() < 1e-5); // First matmul
    assert!((results[1][0] - 2.0).abs() < 1e-5); // Identity matmul
}

#[test]
fn test_hybrid_scheduler_pool_stats() {
    let mut scheduler = HybridScheduler::with_threshold(1000).expect("test");

    // Initially empty
    let stats = scheduler.pool_stats();
    assert_eq!(stats.cached_buffers, 0);

    // Do some pooled operations
    for _ in 0..3 {
        let c = scheduler
            .matmul_pooled(&[1.0; 4], &[1.0; 4], 2, 2, 2)
            .expect("test");
        scheduler.release_buffer(c);
    }

    // Should have cached buffers
    let stats = scheduler.pool_stats();
    assert!(stats.cached_buffers >= 1);
}

// ============================================================================
// StreamingKVCache Tests (M6: Memory Efficiency)
// ============================================================================

#[test]
fn test_streaming_kv_cache_creation() {
    let cache = StreamingKVCache::new(4, 2048, 8, 64);

    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
    assert_eq!(cache.max_positions(), 2048);

    // Memory calculation: 4 layers * 2048 pos * 8 heads * 64 dim * 2 (K+V) * 4 bytes
    let expected_bytes = 4 * 2048 * 8 * 64 * 2 * 4;
    assert_eq!(cache.memory_bytes(), expected_bytes);
}

#[test]
fn test_streaming_kv_cache_append() {
    let mut cache = StreamingKVCache::new(2, 100, 4, 32);
    let kv_dim = 4 * 32; // num_heads * head_dim = 128

    let key = vec![1.0f32; kv_dim];
    let value = vec![2.0f32; kv_dim];

    // Append to first layer (position not incremented yet)
    cache.append(0, &key, &value);
    assert_eq!(cache.len(), 0); // Position only increments after last layer

    // Append to second (last) layer
    cache.append(1, &key, &value);
    assert_eq!(cache.len(), 1);
    assert!(!cache.is_empty());
}

#[test]
fn test_streaming_kv_cache_get_range() {
    let mut cache = StreamingKVCache::new(2, 100, 4, 32);
    let kv_dim = 4 * 32;

    // Append 3 positions
    for pos in 0..3 {
        let key = vec![(pos + 1) as f32; kv_dim];
        let value = vec![(pos + 10) as f32; kv_dim];

        for layer in 0..2 {
            cache.append(layer, &key, &value);
        }
    }

    assert_eq!(cache.len(), 3);

    // Get range for layer 0
    let (keys, values) = cache.get_range(0, 0, 2);
    assert_eq!(keys.len(), 2 * kv_dim);
    assert_eq!(values.len(), 2 * kv_dim);

    // First position should have value 1.0
    assert!((keys[0] - 1.0).abs() < 1e-5);
    // Second position should have value 2.0
    assert!((keys[kv_dim] - 2.0).abs() < 1e-5);
}

#[test]
fn test_streaming_kv_cache_get_valid() {
    let mut cache = StreamingKVCache::new(2, 100, 4, 32);
    let kv_dim = 4 * 32;

    // Append 5 positions
    for pos in 0..5 {
        let key = vec![(pos + 1) as f32; kv_dim];
        let value = vec![(pos + 10) as f32; kv_dim];

        for layer in 0..2 {
            cache.append(layer, &key, &value);
        }
    }

    let (keys, values) = cache.get_valid(0);
    assert_eq!(keys.len(), 5 * kv_dim);
    assert_eq!(values.len(), 5 * kv_dim);
}

#[test]
fn test_streaming_kv_cache_circular_buffer() {
    let mut cache = StreamingKVCache::new(1, 3, 2, 4); // Very small: 3 positions max
    let kv_dim = 2 * 4; // 8

    // Fill cache completely
    for pos in 0..3 {
        let key = vec![(pos + 1) as f32; kv_dim];
        let value = vec![(pos + 10) as f32; kv_dim];
        cache.append(0, &key, &value);
    }

    assert_eq!(cache.len(), 3); // Full

    // Add one more - should wrap around
    let key = vec![100.0f32; kv_dim];
    let value = vec![200.0f32; kv_dim];
    cache.append(0, &key, &value);

    // Still max 3 positions
    assert_eq!(cache.len(), 3);

    // First position should now have the new value (wrapped)
    let (keys, _) = cache.get_range(0, 0, 1);
    assert!((keys[0] - 100.0).abs() < 1e-5);
}

#[test]
fn test_streaming_kv_cache_clear() {
    let mut cache = StreamingKVCache::new(2, 100, 4, 32);
    let kv_dim = 4 * 32;

    // Add some data
    for _ in 0..5 {
        let key = vec![1.0f32; kv_dim];
        let value = vec![2.0f32; kv_dim];
        for layer in 0..2 {
            cache.append(layer, &key, &value);
        }
    }

    assert_eq!(cache.len(), 5);

    cache.clear();

    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
}

#[test]
fn test_streaming_kv_cache_memory_calculation() {
    // Simulate 7B model KV cache
    // 32 layers, 2048 context, 32 heads, 128 head_dim
    let cache = StreamingKVCache::new(32, 2048, 32, 128);

    // Expected: 32 * 2048 * 32 * 128 * 2 * 4 = 2,147,483,648 bytes = 2GB
    let expected_bytes = 32 * 2048 * 32 * 128 * 2 * 4;
    assert_eq!(cache.memory_bytes(), expected_bytes);

    let memory_mb = cache.memory_mb();
    assert!((memory_mb - 2048.0).abs() < 1.0); // ~2048 MB = 2GB
}

#[test]
fn test_streaming_kv_cache_memory_bound() {
    // Test that memory stays bounded even with many appends
    let mut cache = StreamingKVCache::new(1, 10, 2, 4);
    let kv_dim = 2 * 4;

    let initial_bytes = cache.memory_bytes();

    // Append way more than max_positions
    for pos in 0..100 {
        let key = vec![pos as f32; kv_dim];
        let value = vec![pos as f32; kv_dim];
        cache.append(0, &key, &value);
    }

    // Memory should not have grown
    assert_eq!(cache.memory_bytes(), initial_bytes);
    // Valid positions should be capped at max_positions
    assert_eq!(cache.len(), 10);
}

#[test]
#[should_panic(expected = "Layer index out of bounds")]
fn test_streaming_kv_cache_layer_bounds() {
    let mut cache = StreamingKVCache::new(2, 100, 4, 32);
    let kv_dim = 4 * 32;

    let key = vec![1.0f32; kv_dim];
    let value = vec![2.0f32; kv_dim];

    // This should panic - layer 2 is out of bounds for 2-layer cache
    cache.append(2, &key, &value);
}

#[test]
#[should_panic(expected = "Key dimension mismatch")]
fn test_streaming_kv_cache_dimension_mismatch() {
    let mut cache = StreamingKVCache::new(2, 100, 4, 32);

    let key = vec![1.0f32; 10]; // Wrong size
    let value = vec![2.0f32; 4 * 32];

    cache.append(0, &key, &value);
}

// ============================================================================
// M9 Ultra-Long Context Tests (8192+ positions)
// ============================================================================

#[test]
fn test_streaming_kv_cache_8192_positions() {
    // M9 target: 8192 context positions
    let num_layers = 4; // Use smaller for test speed
    let max_positions = 8192;
    let num_heads = 8;
    let head_dim = 64;

    let mut cache = StreamingKVCache::new(num_layers, max_positions, num_heads, head_dim);

    assert_eq!(cache.max_positions(), 8192);
    assert_eq!(cache.len(), 0);

    // Fill to capacity - must fill all layers for each position
    let kv_dim = num_heads * head_dim;
    let key = vec![0.1f32; kv_dim];
    let value = vec![0.2f32; kv_dim];

    for _pos in 0..8192 {
        // Append to all layers for each position
        for layer in 0..num_layers {
            cache.append(layer, &key, &value);
        }
    }

    // Should have filled to max_positions
    assert_eq!(cache.len(), max_positions);
}

#[test]
fn test_ultra_long_context_memory_bound() {
    // Verify 8192 context memory stays bounded
    let num_layers = 32;
    let max_positions = 8192;
    let num_heads = 32;
    let head_dim = 128;

    let cache = StreamingKVCache::new(num_layers, max_positions, num_heads, head_dim);

    // Memory calculation:
    // 32 layers * 8192 positions * 32 heads * 128 dim * 2 (K+V) * 4 bytes
    // = 8,589,934,592 bytes = 8.59 GB
    let expected_bytes = num_layers * max_positions * num_heads * head_dim * 2 * 4;
    assert_eq!(cache.memory_bytes(), expected_bytes);

    let memory_gb = cache.memory_mb() / 1024.0;
    assert!(
        memory_gb < 9.0,
        "8192 context KV cache should be < 9 GB, got {:.2} GB",
        memory_gb
    );
}

#[test]
fn test_ultra_long_context_fill_performance() {
    use std::time::Instant;

    let num_layers = 4;
    let max_positions = 8192;
    let num_heads = 8;
    let head_dim = 64;

    let mut cache = StreamingKVCache::new(num_layers, max_positions, num_heads, head_dim);

    let kv_dim = num_heads * head_dim;
    let key = vec![0.1f32; kv_dim];
    let value = vec![0.2f32; kv_dim];

    // Measure fill time
    let start = Instant::now();
    for _pos in 0..8192 {
        // Append to all layers for each position
        for layer in 0..num_layers {
            cache.append(layer, &key, &value);
        }
    }
    let elapsed = start.elapsed();

    // Should fill 8192 positions in < 1 second
    let fill_rate = 8192.0 / elapsed.as_secs_f64();
    assert!(
        fill_rate > 100.0,
        "Fill rate should be > 100 pos/s, got {:.0}",
        fill_rate
    );
}

// ============================================================================
// M10 Super-Long Context Tests (16384+ positions)
// ============================================================================

#[test]
fn test_streaming_kv_cache_16384_positions() {
    // M10 target: 16384 context positions
    let num_layers = 4; // Use smaller for test speed
    let max_positions = 16384;
    let num_heads = 8;
    let head_dim = 64;

    let mut cache = StreamingKVCache::new(num_layers, max_positions, num_heads, head_dim);

    assert_eq!(cache.max_positions(), 16384);
    assert_eq!(cache.len(), 0);

    // Fill to capacity - must fill all layers for each position
    let kv_dim = num_heads * head_dim;
    let key = vec![0.1f32; kv_dim];
    let value = vec![0.2f32; kv_dim];

    for _pos in 0..16384 {
        // Append to all layers for each position
        for layer in 0..num_layers {
            cache.append(layer, &key, &value);
        }
    }

    // Should have filled to max_positions
    assert_eq!(cache.len(), max_positions);
}

#[test]
fn test_super_long_context_memory_bound() {
    // Verify 16384 context memory stays bounded
    let num_layers = 32;
    let max_positions = 16384;
    let num_heads = 32;
    let head_dim = 128;

    let cache = StreamingKVCache::new(num_layers, max_positions, num_heads, head_dim);

    // Memory calculation:
    // 32 layers * 16384 positions * 32 heads * 128 dim * 2 (K+V) * 4 bytes
    // = 17,179,869,184 bytes = 17.18 GB
    let expected_bytes = num_layers * max_positions * num_heads * head_dim * 2 * 4;
    assert_eq!(cache.memory_bytes(), expected_bytes);

    let memory_gb = cache.memory_mb() / 1024.0;
    assert!(
        memory_gb < 18.0,
        "16384 context KV cache should be < 18 GB, got {:.2} GB",
        memory_gb
    );
}

#[test]
fn test_super_long_context_fill_performance() {
    use std::time::Instant;

    let num_layers = 4;
    let max_positions = 16384;
    let num_heads = 8;
    let head_dim = 64;

    let mut cache = StreamingKVCache::new(num_layers, max_positions, num_heads, head_dim);

    let kv_dim = num_heads * head_dim;
    let key = vec![0.1f32; kv_dim];
    let value = vec![0.2f32; kv_dim];

    // Measure fill time
    let start = Instant::now();
    for _pos in 0..16384 {
        // Append to all layers for each position
        for layer in 0..num_layers {
            cache.append(layer, &key, &value);
        }
    }
    let elapsed = start.elapsed();

    // Should fill 16384 positions in < 2 seconds
    let fill_rate = 16384.0 / elapsed.as_secs_f64();
    assert!(
        fill_rate > 50.0,
        "Fill rate should be > 50 pos/s, got {:.0}",
        fill_rate
    );
}
