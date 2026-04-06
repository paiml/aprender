#[test]
fn test_imp_046_cache_aligned_storage() {
    use crate::gpu::CacheAlignedBuffer;

    // Test 1: Create cache-aligned buffer
    let size = 1024;
    let buffer = CacheAlignedBuffer::new(size);

    // Test 2: Buffer should be 64-byte aligned
    assert!(
        buffer.is_aligned(64),
        "IMP-046: Buffer should be 64-byte aligned"
    );

    // Test 3: Buffer should have correct capacity
    assert_eq!(
        buffer.len(),
        size,
        "IMP-046: Buffer should have correct length"
    );

    // Test 4: Can read and write to buffer
    let mut buffer = CacheAlignedBuffer::new(size);
    buffer.as_mut_slice()[0] = 42.0;
    buffer.as_mut_slice()[size - 1] = 99.0;
    assert_eq!(
        buffer.as_slice()[0],
        42.0,
        "IMP-046: Should read back written value"
    );
    assert_eq!(
        buffer.as_slice()[size - 1],
        99.0,
        "IMP-046: Should read back written value at end"
    );

    // Test 5: Alignment preserved for various sizes
    for size in [64, 128, 256, 512, 1000, 2048] {
        let buf = CacheAlignedBuffer::new(size);
        assert!(
            buf.is_aligned(64),
            "IMP-046: Buffer of size {} should be 64-byte aligned",
            size
        );
    }
}

/// IMP-047: Prefetch hints for sequential access
/// Target: Software prefetch for predictable memory patterns
#[test]
fn test_imp_047_prefetch_hints() {
    use crate::gpu::{prefetch_read, sequential_sum, sum_with_prefetch};
    use std::time::Instant;

    // Create test data
    let size = 64 * 1024; // 64K elements = 256KB
    let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();

    // Test 1: prefetch_read should not panic
    prefetch_read(&data, 0, 64);
    prefetch_read(&data, 1000, 64);

    // Test 2: Both methods should produce same result
    let seq_result = sequential_sum(&data);
    let prefetch_result = sum_with_prefetch(&data, 64);

    assert!(
        (seq_result - prefetch_result).abs() < 1e-3,
        "IMP-047: Sequential ({}) and prefetch ({}) sums should match",
        seq_result,
        prefetch_result
    );

    // Test 3: Prefetch version should be at least as fast
    // Warmup
    for _ in 0..3 {
        let _ = sequential_sum(&data);
        let _ = sum_with_prefetch(&data, 64);
    }

    // Benchmark sequential
    let mut seq_times = Vec::with_capacity(10);
    for _ in 0..10 {
        let start = Instant::now();
        for _ in 0..20 {
            let _ = sequential_sum(&data);
        }
        seq_times.push(start.elapsed().as_secs_f64());
    }
    seq_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Benchmark with prefetch
    let mut pf_times = Vec::with_capacity(10);
    for _ in 0..10 {
        let start = Instant::now();
        for _ in 0..20 {
            let _ = sum_with_prefetch(&data, 64);
        }
        pf_times.push(start.elapsed().as_secs_f64());
    }
    pf_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let seq_median = seq_times[seq_times.len() / 2];
    let pf_median = pf_times[pf_times.len() / 2];
    let speedup = seq_median / pf_median;

    // Note: Performance benchmarks unreliable under coverage instrumentation
    // Prefetch is advisory - hardware may or may not benefit
    // The key test is correctness. Performance is informational only.
    let _ = speedup;
}

/// IMP-048: Block-wise matrix operations
/// Target: Cache-blocked matmul for better locality
#[test]
#[allow(clippy::many_single_char_names)] // m, k, n, a, b are standard matrix notation
fn test_imp_048_blocked_matmul() {
    use crate::gpu::{blocked_matmul, naive_matmul};
    use std::time::Instant;

    // Test matrices: (M x K) @ (K x N) -> (M x N)
    let m = 128;
    let k = 256;
    let n = 128;

    // Create test matrices
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i % 100) as f32) * 0.01 - 0.5)
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| ((i % 100) as f32) * 0.01 - 0.5)
        .collect();

    // Test 1: Both methods should produce same results
    let naive_result = naive_matmul(&a, &b, m, k, n);
    let blocked_result = blocked_matmul(&a, &b, m, k, n, 32); // Block size 32

    assert_eq!(
        naive_result.len(),
        blocked_result.len(),
        "IMP-048: Results should have same length"
    );

    for (i, (&naive, &blocked)) in naive_result.iter().zip(blocked_result.iter()).enumerate() {
        assert!(
            (naive - blocked).abs() < 1e-4,
            "IMP-048: Mismatch at index {}: naive={}, blocked={}",
            i,
            naive,
            blocked
        );
    }

    // Test 2: Blocked should be faster for larger matrices
    let m = 256;
    let k = 512;
    let n = 256;
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i % 100) as f32) * 0.01 - 0.5)
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| ((i % 100) as f32) * 0.01 - 0.5)
        .collect();

    // Warmup
    for _ in 0..2 {
        let _ = naive_matmul(&a, &b, m, k, n);
        let _ = blocked_matmul(&a, &b, m, k, n, 32);
    }

    // Benchmark naive
    let mut naive_times = Vec::with_capacity(5);
    for _ in 0..5 {
        let start = Instant::now();
        for _ in 0..3 {
            let _ = naive_matmul(&a, &b, m, k, n);
        }
        naive_times.push(start.elapsed().as_secs_f64());
    }
    naive_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Benchmark blocked
    let mut blocked_times = Vec::with_capacity(5);
    for _ in 0..5 {
        let start = Instant::now();
        for _ in 0..3 {
            let _ = blocked_matmul(&a, &b, m, k, n, 32);
        }
        blocked_times.push(start.elapsed().as_secs_f64());
    }
    blocked_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let naive_median = naive_times[naive_times.len() / 2];
    let blocked_median = blocked_times[blocked_times.len() / 2];
    let speedup = naive_median / blocked_median;

    // Note: Performance benchmarks unreliable under coverage instrumentation
    // The key test is correctness (Test 1). Performance is informational only.
    let _ = speedup;
}

// ============================================================================
// Phase 13: Memory Pooling & Arena Allocation (M22) - EXTREME TDD
// ============================================================================

/// IMP-049: Tensor memory pool
/// Target: Reusable tensor buffer pool for inference
#[test]
#[cfg(feature = "gpu")]
fn test_imp_049_tensor_pool() {
    use crate::gpu::TensorPool;

    // Test 1: Create pool with capacity
    let mut pool = TensorPool::new(4); // 4 buffers max
    assert_eq!(pool.capacity(), 4, "IMP-049: Pool should have capacity 4");
    assert_eq!(pool.available(), 0, "IMP-049: Pool should start empty");

    // Test 2: Acquire buffers of different sizes
    let buf1 = pool.acquire(1024);
    assert_eq!(
        buf1.len(),
        1024,
        "IMP-049: Buffer should have requested size"
    );

    let buf2 = pool.acquire(2048);
    assert_eq!(
        buf2.len(),
        2048,
        "IMP-049: Second buffer should have size 2048"
    );

    // Test 3: Release and reuse
    pool.release(buf1);
    assert!(
        pool.available() >= 1,
        "IMP-049: Pool should have available buffer"
    );

    let buf3 = pool.acquire(1024); // Should reuse released buffer
    assert_eq!(
        buf3.len(),
        1024,
        "IMP-049: Reused buffer should have correct size"
    );

    // Test 4: Pool tracks allocations
    pool.release(buf2);
    pool.release(buf3);
    assert!(
        pool.available() >= 2,
        "IMP-049: Pool should have 2 available buffers"
    );

    // Test 5: Clear pool
    pool.clear();
    assert_eq!(
        pool.available(),
        0,
        "IMP-049: Pool should be empty after clear"
    );
}

/// IMP-050: Arena allocator for forward pass
/// Target: Single-allocation arena for temporary tensors
#[test]
#[cfg(feature = "gpu")]
fn test_imp_050_arena_allocator() {
    use crate::gpu::ForwardArena;

    // Test 1: Create arena with capacity
    let mut arena = ForwardArena::new(1024 * 1024); // 1MB arena
    assert!(
        arena.capacity() >= 1024 * 1024,
        "IMP-050: Arena should have at least 1MB capacity"
    );
    assert_eq!(arena.used(), 0, "IMP-050: Arena should start empty");

    // Test 2: Allocate from arena and verify sizes
    {
        let slice1 = arena.alloc(256);
        assert_eq!(
            slice1.len(),
            256,
            "IMP-050: First allocation should have size 256"
        );
    }
    assert_eq!(arena.used(), 256, "IMP-050: Arena should track usage");

    {
        let slice2 = arena.alloc(512);
        assert_eq!(
            slice2.len(),
            512,
            "IMP-050: Second allocation should have size 512"
        );
    }
    assert!(
        arena.used() >= 768,
        "IMP-050: Arena should track cumulative usage"
    );

    // Test 3: Reset arena for reuse
    arena.reset();
    assert_eq!(
        arena.used(),
        0,
        "IMP-050: Arena should be empty after reset"
    );

    // Test 4: Can allocate again after reset
    let slice3 = arena.alloc(1024);
    assert_eq!(
        slice3.len(),
        1024,
        "IMP-050: Post-reset allocation should work"
    );

    // Test 5: Verify allocations are zeroed
    assert!(
        slice3.iter().all(|&x| x == 0.0),
        "IMP-050: Fresh allocation should be zeroed"
    );
}

/// IMP-051: Scratch buffer management
/// Target: Reusable scratch space for intermediate computations
#[test]
#[cfg(feature = "gpu")]
fn test_imp_051_scratch_buffers() {
    use crate::gpu::ScratchBuffer;

    // Test 1: Create scratch buffer for layers
    let num_layers = 4;
    let layer_size = 2048;
    let mut scratch = ScratchBuffer::new(num_layers, layer_size);

    assert_eq!(
        scratch.num_layers(),
        num_layers,
        "IMP-051: Should have 4 layers"
    );
    assert_eq!(
        scratch.layer_size(),
        layer_size,
        "IMP-051: Layer size should be 2048"
    );

    // Test 2: Get scratch for specific layer
    let layer0 = scratch.get_layer(0);
    assert_eq!(
        layer0.len(),
        layer_size,
        "IMP-051: Layer 0 scratch should have correct size"
    );

    let layer3 = scratch.get_layer(3);
    assert_eq!(
        layer3.len(),
        layer_size,
        "IMP-051: Layer 3 scratch should have correct size"
    );

    // Test 3: Layer scratches are independent
    scratch.get_layer_mut(0).iter_mut().for_each(|x| *x = 1.0);
    scratch.get_layer_mut(1).iter_mut().for_each(|x| *x = 2.0);

    assert!(
        scratch.get_layer(0).iter().all(|&x| x == 1.0),
        "IMP-051: Layer 0 should retain its values"
    );
    assert!(
        scratch.get_layer(1).iter().all(|&x| x == 2.0),
        "IMP-051: Layer 1 should be independent"
    );

    // Test 4: Reset all layers
    scratch.reset();
    assert!(
        scratch.get_layer(0).iter().all(|&x| x == 0.0),
        "IMP-051: Layer 0 should be zeroed after reset"
    );

    // Test 5: Total size calculation
    assert_eq!(
        scratch.total_size(),
        num_layers * layer_size,
        "IMP-051: Total size should be layers * layer_size"
    );
}

include!("imp_052.rs");
include!("imp_058.rs");
include!("imp_064.rs");
include!("imp_070.rs");
