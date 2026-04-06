
#[test]
#[serial]
fn test_cov025_get_staging_buffer() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Get a staging buffer
    let buf = executor.get_staging_buffer(1024);
    assert_eq!(buf.len(), 1024, "Staging buffer should have requested size");
    // Note: is_pinned() may return false depending on pool implementation

    // Return it
    executor.return_staging_buffer(buf);
}

#[test]
#[serial]
fn test_cov025_staging_pool_stats() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Get initial stats
    let stats1 = executor.staging_pool_stats();
    assert_eq!(stats1.pool_hits, 0, "Initial hits should be 0");
    assert_eq!(stats1.pool_misses, 0, "Initial misses should be 0");

    // Allocate a buffer - should be a miss
    let buf = executor.get_staging_buffer(1024);
    let stats2 = executor.staging_pool_stats();
    assert_eq!(stats2.pool_misses, 1, "Should have 1 miss");

    // Return and get again - should be a hit
    executor.return_staging_buffer(buf);
    let _buf2 = executor.get_staging_buffer(1024);
    let stats3 = executor.staging_pool_stats();
    assert!(stats3.pool_hits >= 1, "Should have at least 1 hit");
}

#[test]
#[serial]
fn test_cov025_cached_weight_count() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Initially zero
    assert_eq!(
        executor.cached_weight_count(),
        0,
        "Initial count should be 0"
    );

    // Load a weight
    let weights = vec![1.0f32; 256];
    executor
        .load_weights("test_weight", &weights)
        .expect("load");

    assert_eq!(
        executor.cached_weight_count(),
        1,
        "Count should be 1 after loading"
    );
}

#[test]
#[serial]
fn test_cov025_cached_weight_bytes() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Initially zero
    assert_eq!(
        executor.cached_weight_bytes(),
        0,
        "Initial bytes should be 0"
    );

    // Load a weight (256 f32 = 1024 bytes)
    let weights = vec![1.0f32; 256];
    executor
        .load_weights("test_weight", &weights)
        .expect("load");

    let bytes = executor.cached_weight_bytes();
    assert!(bytes >= 1024, "Should have at least 1024 bytes cached");
}

#[test]
#[serial]
fn test_cov025_cached_quantized_weight_count() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Initially zero
    assert_eq!(
        executor.cached_quantized_weight_count(),
        0,
        "Initial count should be 0"
    );

    // Load a quantized weight (Q4_K: 144 bytes per super-block)
    let weights = vec![0u8; 144];
    executor
        .load_quantized_weights("test_q4k", &weights)
        .expect("load");

    assert_eq!(
        executor.cached_quantized_weight_count(),
        1,
        "Count should be 1 after loading"
    );
}

#[test]
#[serial]
fn test_cov025_clear_weights() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Load some weights
    let weights = vec![1.0f32; 256];
    executor.load_weights("test1", &weights).expect("load1");
    executor.load_weights("test2", &weights).expect("load2");
    assert_eq!(executor.cached_weight_count(), 2, "Should have 2 weights");

    // Clear
    executor.clear_weights();
    assert_eq!(
        executor.cached_weight_count(),
        0,
        "Should have 0 after clear"
    );
}

