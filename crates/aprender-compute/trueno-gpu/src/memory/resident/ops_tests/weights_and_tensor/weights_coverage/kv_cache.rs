//! KV Cache Extended Tests and Field Access Tests

use super::*;

// ============================================================================
// KV Cache Extended Tests
// ============================================================================

#[test]
fn test_gpu_kv_cache_key_and_value_access() {
    use crate::memory::resident::clear_kernel_cache;
    use crate::memory::resident::GpuKvCache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let max_seq_len = 32usize;
    let d_model = 16usize;

    let cache = GpuKvCache::new(&ctx, max_seq_len, d_model).unwrap();

    // Verify key tensor is correctly allocated
    assert_eq!(cache.key.len(), max_seq_len * d_model);
    assert!(cache.key.is_device_resident());

    // Verify value tensor is correctly allocated
    assert_eq!(cache.value.len(), max_seq_len * d_model);
    assert!(cache.value.is_device_resident());

    // Verify fields are accessible
    assert_eq!(cache.max_seq_len, max_seq_len);
    assert_eq!(cache.d_model, d_model);
}

#[test]
fn test_gpu_kv_cache_len_changes() {
    use crate::memory::resident::clear_kernel_cache;
    use crate::memory::resident::GpuKvCache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let mut cache = GpuKvCache::new(&ctx, 64, 32).unwrap();

    // Start empty
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);

    // Simulate adding tokens
    cache.seq_len = 5;
    assert!(!cache.is_empty());
    assert_eq!(cache.len(), 5);

    // Add more tokens
    cache.seq_len = 20;
    assert!(!cache.is_empty());
    assert_eq!(cache.len(), 20);

    // Reset
    cache.reset();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

// ============================================================================
// KV Cache Field Access Test
// ============================================================================

#[test]
fn test_gpu_kv_cache_field_access() {
    use crate::memory::resident::clear_kernel_cache;
    use crate::memory::resident::GpuKvCache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let max_seq_len = 64usize;
    let d_model = 32usize;
    let total_size = max_seq_len * d_model;

    let mut cache = GpuKvCache::new(&ctx, max_seq_len, d_model).unwrap();

    // Verify key and value tensors can be accessed
    assert_eq!(cache.key.len(), total_size);
    assert_eq!(cache.value.len(), total_size);

    // Verify peek_host works on both caches
    let key_data = cache.key.peek_host().unwrap();
    let value_data = cache.value.peek_host().unwrap();
    assert_eq!(key_data.len(), total_size);
    assert_eq!(value_data.len(), total_size);

    // Verify all zeros initially
    assert!(key_data.iter().all(|&v| v == 0.0));
    assert!(value_data.iter().all(|&v| v == 0.0));

    // Test sequence length management
    cache.seq_len = 10;
    assert_eq!(cache.len(), 10);
    assert!(!cache.is_empty());

    cache.reset();
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
}
