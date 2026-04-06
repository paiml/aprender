
#[test]
fn test_kv_cache_append_multiple_positions() {
    let config = create_test_kv_config();
    let mut cache = AprKVCache::new(&config);

    let kv_size = 4 * 16; // num_kv_heads * head_dim

    // Append 5 positions
    for pos in 0..5 {
        let k = vec![pos as f32; kv_size];
        let v = vec![(pos * 10) as f32; kv_size];
        cache.append(0, &k, &v);
        cache.append(1, &k, &v);
    }

    assert_eq!(cache.len(), 5);
}

#[test]
fn test_kv_cache_get_returns_all_cached_positions() {
    let config = create_test_kv_config();
    let mut cache = AprKVCache::new(&config);

    let kv_size = 4 * 16; // 64

    // Append 3 positions with distinct values
    for pos in 0..3 {
        let k = vec![(pos + 1) as f32; kv_size];
        let v = vec![(pos + 1) as f32 * 10.0; kv_size];
        cache.append(0, &k, &v);
        cache.append(1, &k, &v);
    }

    // Get layer 0 cache
    let (k_slice, v_slice) = cache.get(0);

    // Should have 3 positions * 64 values = 192 total
    assert_eq!(k_slice.len(), 3 * kv_size);
    assert_eq!(v_slice.len(), 3 * kv_size);

    // Verify first position values
    assert!((k_slice[0] - 1.0).abs() < f32::EPSILON);
    assert!((v_slice[0] - 10.0).abs() < f32::EPSILON);

    // Verify second position values (offset by kv_size)
    assert!((k_slice[kv_size] - 2.0).abs() < f32::EPSILON);
    assert!((v_slice[kv_size] - 20.0).abs() < f32::EPSILON);
}

#[test]
fn test_kv_cache_clear_resets_state() {
    let config = create_test_kv_config();
    let mut cache = AprKVCache::new(&config);

    let kv_size = 4 * 16;
    let k = vec![1.0f32; kv_size];
    let v = vec![2.0f32; kv_size];

    // Fill cache with 10 positions
    for _ in 0..10 {
        cache.append(0, &k, &v);
        cache.append(1, &k, &v);
    }
    assert_eq!(cache.len(), 10);

    // Clear and verify
    cache.clear();
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());

    // Verify get returns empty slices
    let (k_slice, v_slice) = cache.get(0);
    assert!(k_slice.is_empty());
    assert!(v_slice.is_empty());
}

#[test]
fn test_kv_cache_clear_allows_reuse() {
    let config = create_test_kv_config();
    let mut cache = AprKVCache::new(&config);

    let kv_size = 4 * 16;
    let k1 = vec![1.0f32; kv_size];
    let v1 = vec![2.0f32; kv_size];

    // First generation: fill with k1, v1
    for _ in 0..5 {
        cache.append(0, &k1, &v1);
        cache.append(1, &k1, &v1);
    }
    assert_eq!(cache.len(), 5);

    // Clear for new generation
    cache.clear();

    // Second generation: fill with different values
    let k2 = vec![100.0f32; kv_size];
    let v2 = vec![200.0f32; kv_size];
    for _ in 0..3 {
        cache.append(0, &k2, &v2);
        cache.append(1, &k2, &v2);
    }
    assert_eq!(cache.len(), 3);

    // Verify second generation values (not contaminated by first)
    let (k_slice, _) = cache.get(0);
    assert!((k_slice[0] - 100.0).abs() < f32::EPSILON);
}

#[test]
fn test_kv_cache_consecutive_generations_isolated() {
    // This is the key Popper falsification test:
    // Consecutive generations should produce identical results
    // if the cache is properly cleared between them.

    let config = create_test_kv_config();
    let mut cache = AprKVCache::new(&config);

    let kv_size = 4 * 16;

    // Simulate "Generation 1": populate cache
    let seq1_k: Vec<f32> = (0..3).flat_map(|i| vec![(i + 1) as f32; kv_size]).collect();
    for pos in 0..3 {
        let start = pos * kv_size;
        let end = start + kv_size;
        cache.append(0, &seq1_k[start..end], &seq1_k[start..end]);
        cache.append(1, &seq1_k[start..end], &seq1_k[start..end]);
    }
    let gen1_len = cache.len();
    let (gen1_k, gen1_v) = cache.get(0);
    let gen1_k_copy: Vec<f32> = gen1_k.to_vec();
    let gen1_v_copy: Vec<f32> = gen1_v.to_vec();

    // Clear for "Generation 2"
    cache.clear();

    // Simulate "Generation 2": same sequence should produce same cache
    for pos in 0..3 {
        let start = pos * kv_size;
        let end = start + kv_size;
        cache.append(0, &seq1_k[start..end], &seq1_k[start..end]);
        cache.append(1, &seq1_k[start..end], &seq1_k[start..end]);
    }
    let gen2_len = cache.len();
    let (gen2_k, gen2_v) = cache.get(0);

    // Verify both generations are identical
    assert_eq!(
        gen1_len, gen2_len,
        "Cache lengths differ between generations"
    );
    assert_eq!(gen1_k_copy.len(), gen2_k.len(), "K cache sizes differ");
    assert_eq!(gen1_v_copy.len(), gen2_v.len(), "V cache sizes differ");

    for i in 0..gen1_k_copy.len() {
        assert!(
            (gen1_k_copy[i] - gen2_k[i]).abs() < f32::EPSILON,
            "K cache differs at position {}: {} vs {}",
            i,
            gen1_k_copy[i],
            gen2_k[i]
        );
    }
    for i in 0..gen1_v_copy.len() {
        assert!(
            (gen1_v_copy[i] - gen2_v[i]).abs() < f32::EPSILON,
            "V cache differs at position {}: {} vs {}",
            i,
            gen1_v_copy[i],
            gen2_v[i]
        );
    }
}

#[test]
fn test_kv_cache_layer_storage_separate() {
    // Verify that layers have separate storage vectors
    // F-REGR-231: Auto-advance happens on LAST layer, so layer-0-only tests need advance()
    let config = create_test_kv_config();
    let mut cache = AprKVCache::new(&config);

    let kv_size = 4 * 16;

    // Only use layer 0 to test basic operations correctly
    let k = vec![42.0f32; kv_size];
    let v = vec![84.0f32; kv_size];

    cache.append(0, &k, &v);
    cache.advance(); // F-REGR-231: explicit advance when not using all layers
    assert_eq!(cache.len(), 1);

    let (k_out, v_out) = cache.get(0);
    assert_eq!(k_out.len(), kv_size);
    assert_eq!(v_out.len(), kv_size);
    assert!((k_out[0] - 42.0).abs() < f32::EPSILON);
    assert!((v_out[0] - 84.0).abs() < f32::EPSILON);
}

#[test]
fn test_kv_cache_capacity_tracking() {
    let config = create_test_kv_config();
    let cache = AprKVCache::new(&config);

    // Capacity should match context_length from config
    assert_eq!(cache.capacity(), 32);

    // Capacity should remain constant regardless of usage
    let mut cache2 = AprKVCache::new(&config);
    let kv_size = 4 * 16;
    let k = vec![1.0f32; kv_size];
    let v = vec![2.0f32; kv_size];

    for _ in 0..10 {
        cache2.append(0, &k, &v);
        cache2.append(1, &k, &v);
    }
    assert_eq!(cache2.capacity(), 32); // Still 32, not reduced
}

#[test]
fn test_kv_cache_clone_creates_independent_copy() {
    let config = create_test_kv_config();
    let mut cache = AprKVCache::new(&config);

    let kv_size = 4 * 16;
    let k = vec![42.0f32; kv_size];
    let v = vec![84.0f32; kv_size];

    cache.append(0, &k, &v);
    cache.append(1, &k, &v);

    // Clone the cache
    let cache_clone = cache.clone();

    // Verify clone has same state
    assert_eq!(cache_clone.len(), cache.len());
    let (orig_k, _) = cache.get(0);
    let (clone_k, _) = cache_clone.get(0);
    assert_eq!(orig_k.len(), clone_k.len());
    assert!((orig_k[0] - clone_k[0]).abs() < f32::EPSILON);

    // Verify clone is independent (modify original shouldn't affect clone)
    let k2 = vec![999.0f32; kv_size];
    let v2 = vec![888.0f32; kv_size];
    cache.append(0, &k2, &v2);
    cache.append(1, &k2, &v2);

    // Original should have 2 positions, clone should still have 1
    assert_eq!(cache.len(), 2);
    assert_eq!(cache_clone.len(), 1);
}
