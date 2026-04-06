
// ============================================================================
// Tests (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // StreamingKVCache Tests
    // =========================================================================

    #[test]
    fn test_streaming_kv_cache_new() {
        let cache = StreamingKVCache::new(4, 128, 8, 64);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.max_positions(), 128);
    }

    #[test]
    fn test_streaming_kv_cache_append() {
        let mut cache = StreamingKVCache::new(2, 16, 4, 32);
        let kv_dim = 4 * 32;
        let key = vec![1.0f32; kv_dim];
        let value = vec![2.0f32; kv_dim];

        cache.append(0, &key, &value);
        cache.append(1, &key, &value);

        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());
    }

    #[test]
    fn test_streaming_kv_cache_append_multiple() {
        let mut cache = StreamingKVCache::new(2, 16, 2, 4);
        let kv_dim = 2 * 4;
        let key1 = vec![1.0f32; kv_dim];
        let value1 = vec![2.0f32; kv_dim];
        let key2 = vec![3.0f32; kv_dim];
        let value2 = vec![4.0f32; kv_dim];

        // First position
        cache.append(0, &key1, &value1);
        cache.append(1, &key1, &value1);
        assert_eq!(cache.len(), 1);

        // Second position
        cache.append(0, &key2, &value2);
        cache.append(1, &key2, &value2);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_streaming_kv_cache_get_range() {
        let mut cache = StreamingKVCache::new(1, 16, 2, 4);
        let kv_dim = 2 * 4;
        let key = vec![1.5f32; kv_dim];
        let value = vec![2.5f32; kv_dim];

        cache.append(0, &key, &value);

        let (keys, values) = cache.get_range(0, 0, 1);
        assert_eq!(keys.len(), kv_dim);
        assert_eq!(values.len(), kv_dim);
        assert!((keys[0] - 1.5).abs() < 0.01);
        assert!((values[0] - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_streaming_kv_cache_get_valid() {
        let mut cache = StreamingKVCache::new(1, 16, 2, 4);
        let kv_dim = 2 * 4;
        let key = vec![1.0f32; kv_dim];
        let value = vec![2.0f32; kv_dim];

        cache.append(0, &key, &value);
        cache.append(0, &key, &value);

        let (keys, values) = cache.get_valid(0);
        assert_eq!(keys.len(), 2 * kv_dim);
        assert_eq!(values.len(), 2 * kv_dim);
    }

    #[test]
    fn test_streaming_kv_cache_circular_buffer() {
        // Test that cache wraps around when full
        let mut cache = StreamingKVCache::new(1, 4, 1, 2);
        let kv_dim = 2;

        // Fill cache
        for i in 0..4 {
            let key = vec![i as f32; kv_dim];
            let value = vec![(i * 10) as f32; kv_dim];
            cache.append(0, &key, &value);
        }
        assert_eq!(cache.len(), 4);

        // Overflow: should wrap around
        let key = vec![100.0f32; kv_dim];
        let value = vec![200.0f32; kv_dim];
        cache.append(0, &key, &value);
        assert_eq!(cache.len(), 4); // Still 4, max_positions

        // First position should now have the overwritten value
        let (keys, _) = cache.get_range(0, 0, 1);
        assert!((keys[0] - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_streaming_kv_cache_clear() {
        let mut cache = StreamingKVCache::new(1, 16, 2, 4);
        let kv_dim = 2 * 4;
        let key = vec![1.0f32; kv_dim];
        let value = vec![2.0f32; kv_dim];

        cache.append(0, &key, &value);
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_streaming_kv_cache_memory_bytes() {
        // 2 layers, 128 positions, 4 heads, 32 dim
        // kv_size = 128 * 4 * 32 = 16384
        // memory = 2 * 16384 * 2 * 4 = 262144 bytes = 256 KB
        let cache = StreamingKVCache::new(2, 128, 4, 32);
        assert_eq!(cache.memory_bytes(), 262_144);
    }

    #[test]
    fn test_streaming_kv_cache_memory_mb() {
        let cache = StreamingKVCache::new(2, 128, 4, 32);
        let mb = cache.memory_mb();
        assert!((mb - 0.25).abs() < 0.01); // 256 KB = 0.25 MB
    }

    #[test]
    #[should_panic(expected = "Layer index out of bounds")]
    fn test_streaming_kv_cache_invalid_layer() {
        let mut cache = StreamingKVCache::new(2, 16, 2, 4);
        let kv_dim = 2 * 4;
        let key = vec![1.0f32; kv_dim];
        let value = vec![2.0f32; kv_dim];
        cache.append(2, &key, &value); // layer 2 is out of bounds (0, 1)
    }

    #[test]
    #[should_panic(expected = "Key dimension mismatch")]
    fn test_streaming_kv_cache_key_dimension_mismatch() {
        let mut cache = StreamingKVCache::new(1, 16, 2, 4);
        let key = vec![1.0f32; 4]; // Wrong dimension (should be 8)
        let value = vec![2.0f32; 8];
        cache.append(0, &key, &value);
    }

    // =========================================================================
    // StreamingKVCacheFp16 Tests
    // =========================================================================

    #[test]
    fn test_streaming_kv_cache_fp16_new() {
        let cache = StreamingKVCacheFp16::new(4, 128, 8, 64);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.max_positions(), 128);
    }

    #[test]
    fn test_streaming_kv_cache_fp16_conversion() {
        // Test f32 -> f16 -> f32 round trip
        let val = 1.5f32;
        let fp16 = StreamingKVCacheFp16::f32_to_f16(val);
        let back = StreamingKVCacheFp16::f16_to_f32(fp16);
        assert!((back - val).abs() < 0.001);
    }

    #[test]
    fn test_streaming_kv_cache_fp16_append() {
        let mut cache = StreamingKVCacheFp16::new(2, 16, 4, 32);
        let kv_dim = 4 * 32;
        let key = vec![1.0f32; kv_dim];
        let value = vec![2.0f32; kv_dim];

        cache.append(0, &key, &value);
        cache.append(1, &key, &value);

        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());
    }

    #[test]
    fn test_streaming_kv_cache_fp16_get_range_f32() {
        let mut cache = StreamingKVCacheFp16::new(1, 16, 2, 4);
        let kv_dim = 2 * 4;
        let key = vec![1.5f32; kv_dim];
        let value = vec![2.5f32; kv_dim];

        cache.append(0, &key, &value);

        let (keys, values) = cache.get_range_f32(0, 0, 1);
        assert_eq!(keys.len(), kv_dim);
        assert_eq!(values.len(), kv_dim);
        // FP16 has some loss of precision
        assert!((keys[0] - 1.5).abs() < 0.01);
        assert!((values[0] - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_streaming_kv_cache_fp16_get_range_raw() {
        let mut cache = StreamingKVCacheFp16::new(1, 16, 2, 4);
        let kv_dim = 2 * 4;
        let key = vec![1.0f32; kv_dim];
        let value = vec![2.0f32; kv_dim];

        cache.append(0, &key, &value);

        let (keys_raw, values_raw) = cache.get_range_raw(0, 0, 1);
        assert_eq!(keys_raw.len(), kv_dim);
        assert_eq!(values_raw.len(), kv_dim);
    }

    #[test]
    fn test_streaming_kv_cache_fp16_get_valid_f32() {
        let mut cache = StreamingKVCacheFp16::new(1, 16, 2, 4);
        let kv_dim = 2 * 4;
        let key = vec![1.0f32; kv_dim];
        let value = vec![2.0f32; kv_dim];

        cache.append(0, &key, &value);
        cache.append(0, &key, &value);

        let (keys, values) = cache.get_valid_f32(0);
        assert_eq!(keys.len(), 2 * kv_dim);
        assert_eq!(values.len(), 2 * kv_dim);
    }

    #[test]
    fn test_streaming_kv_cache_fp16_clear() {
        let mut cache = StreamingKVCacheFp16::new(1, 16, 2, 4);
        let kv_dim = 2 * 4;
        let key = vec![1.0f32; kv_dim];
        let value = vec![2.0f32; kv_dim];

        cache.append(0, &key, &value);
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_streaming_kv_cache_fp16_memory_bytes() {
        // 2 layers, 128 positions, 4 heads, 32 dim
        // kv_size = 128 * 4 * 32 = 16384
        // memory = 2 * 16384 * 2 * 2 = 131072 bytes = 128 KB
        // (half of FP32 version)
        let cache = StreamingKVCacheFp16::new(2, 128, 4, 32);
        assert_eq!(cache.memory_bytes(), 131_072);
    }

    #[test]
    fn test_streaming_kv_cache_fp16_memory_mb() {
        let cache = StreamingKVCacheFp16::new(2, 128, 4, 32);
        let mb = cache.memory_mb();
        assert!((mb - 0.125).abs() < 0.01); // 128 KB = 0.125 MB
    }

    #[test]
    fn test_streaming_kv_cache_fp16_half_memory_of_fp32() {
        let fp32_cache = StreamingKVCache::new(4, 256, 8, 64);
        let fp16_cache = StreamingKVCacheFp16::new(4, 256, 8, 64);

        // FP16 should use exactly half the memory of FP32
        assert_eq!(fp16_cache.memory_bytes(), fp32_cache.memory_bytes() / 2);
    }

    #[test]
    fn test_streaming_kv_cache_fp16_circular_buffer() {
        let mut cache = StreamingKVCacheFp16::new(1, 4, 1, 2);
        let kv_dim = 2;

        // Fill cache
        for i in 0..4 {
            let key = vec![i as f32; kv_dim];
            let value = vec![(i * 10) as f32; kv_dim];
            cache.append(0, &key, &value);
        }
        assert_eq!(cache.len(), 4);

        // Overflow: should wrap around
        let key = vec![100.0f32; kv_dim];
        let value = vec![200.0f32; kv_dim];
        cache.append(0, &key, &value);
        assert_eq!(cache.len(), 4);

        // First position should now have the overwritten value
        let (keys, _) = cache.get_range_f32(0, 0, 1);
        assert!((keys[0] - 100.0).abs() < 0.1); // FP16 precision
    }

    #[test]
    #[should_panic(expected = "Layer index out of bounds")]
    fn test_streaming_kv_cache_fp16_invalid_layer() {
        let mut cache = StreamingKVCacheFp16::new(2, 16, 2, 4);
        let kv_dim = 2 * 4;
        let key = vec![1.0f32; kv_dim];
        let value = vec![2.0f32; kv_dim];
        cache.append(2, &key, &value);
    }

    #[test]
    #[should_panic(expected = "Value dimension mismatch")]
    fn test_streaming_kv_cache_fp16_value_dimension_mismatch() {
        let mut cache = StreamingKVCacheFp16::new(1, 16, 2, 4);
        let key = vec![1.0f32; 8];
        let value = vec![2.0f32; 4]; // Wrong dimension (should be 8)
        cache.append(0, &key, &value);
    }
}
