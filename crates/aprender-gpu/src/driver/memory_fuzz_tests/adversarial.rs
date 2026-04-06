//! Adversarial tests (Dr. Popper's Falsification Protocol): Tests 1-9
//! These tests try to BREAK the driver, not validate happy paths.

use super::*;

/// Falsification Test 1: Oversize Allocation
/// Attempt to allocate 100GB - must return OOM, not panic or hang
#[test]
fn test_alloc_oversize_100gb() {
    let ctx = CudaContext::new(0).expect("Context");

    // 100GB of f32 = 25 billion elements
    let oversize = 25_000_000_000usize;

    let result = GpuBuffer::<f32>::new(&ctx, oversize);

    match result {
        Err(GpuError::OutOfMemory { .. }) => {
            // Expected - driver correctly reported OOM
        }
        Err(GpuError::MemoryAllocation(_)) => {
            // Also acceptable - allocation failed
        }
        Err(e) => {
            // Any other error is acceptable as long as it doesn't panic
            println!("Oversize alloc returned: {:?}", e);
        }
        Ok(_) => {
            panic!("CRITICAL: 100GB allocation succeeded - this should be impossible on RTX 4090!");
        }
    }
}

/// Falsification Test 2: Copy from host with size mismatch (too small host)
#[test]
fn test_copy_from_host_too_small() {
    let ctx = CudaContext::new(0).expect("Context");
    let mut buf = GpuBuffer::<f32>::new(&ctx, 1000).expect("Alloc");

    // Try to copy from a smaller host buffer
    let small_data = vec![1.0f32; 500];
    let result = buf.copy_from_host(&small_data);

    assert!(
        result.is_err(),
        "copy_from_host should fail when host buffer is smaller"
    );
    if let Err(e) = result {
        assert!(
            format!("{:?}", e).contains("mismatch") || format!("{:?}", e).contains("Transfer"),
            "Error should mention size mismatch: {:?}",
            e
        );
    }
}

/// Falsification Test 3: Copy to host with size mismatch (too large host)
#[test]
fn test_copy_to_host_too_large() {
    let ctx = CudaContext::new(0).expect("Context");
    let buf = GpuBuffer::<f32>::new(&ctx, 100).expect("Alloc");

    // Try to copy to a larger host buffer
    let mut large_data = vec![0.0f32; 500];
    let result = buf.copy_to_host(&mut large_data);

    assert!(
        result.is_err(),
        "copy_to_host should fail when host buffer size doesn't match"
    );
}

/// Falsification Test 4: Partial copy out of bounds (offset too large)
#[test]
fn test_copy_from_host_at_out_of_bounds() {
    let ctx = CudaContext::new(0).expect("Context");
    let mut buf = GpuBuffer::<f32>::new(&ctx, 100).expect("Alloc");

    let data = vec![1.0f32; 50];

    // Offset 60 + len 50 = 110 > 100 buffer size
    let result = buf.copy_from_host_at(&data, 60);
    assert!(
        result.is_err(),
        "copy_from_host_at should fail when offset+len > buffer size"
    );
}

/// Falsification Test 5: Partial copy to host out of bounds
#[test]
fn test_copy_to_host_at_out_of_bounds() {
    let ctx = CudaContext::new(0).expect("Context");
    let data = vec![1.0f32; 100];
    let buf = GpuBuffer::from_host(&ctx, &data).expect("Alloc");

    let mut result = vec![0.0f32; 50];

    // Offset 60 + len 50 = 110 > 100 buffer size
    let copy_result = buf.copy_to_host_at(&mut result, 60);
    assert!(
        copy_result.is_err(),
        "copy_to_host_at should fail when offset+len > buffer size"
    );
}

/// Falsification Test 6: D2D copy size mismatch
#[test]
fn test_d2d_copy_size_mismatch() {
    let ctx = CudaContext::new(0).expect("Context");

    let src = GpuBuffer::<f32>::new(&ctx, 100).expect("Alloc src");
    let mut dst = GpuBuffer::<f32>::new(&ctx, 200).expect("Alloc dst");

    let result = dst.copy_from_buffer(&src);
    assert!(
        result.is_err(),
        "D2D copy should fail when buffer sizes don't match"
    );
}

/// Falsification Test 7: D2D partial copy out of bounds (dst)
#[test]
fn test_d2d_copy_at_dst_out_of_bounds() {
    let ctx = CudaContext::new(0).expect("Context");

    let src = GpuBuffer::<f32>::new(&ctx, 50).expect("Alloc src");
    let mut dst = GpuBuffer::<f32>::new(&ctx, 100).expect("Alloc dst");

    // dst_offset 60 + count 50 = 110 > dst.len 100
    let result = dst.copy_from_buffer_at(&src, 60, 0, 50);
    assert!(
        result.is_err(),
        "D2D copy_at should fail when dst_offset+count > dst.len"
    );
}

/// Falsification Test 8: D2D partial copy out of bounds (src)
#[test]
fn test_d2d_copy_at_src_out_of_bounds() {
    let ctx = CudaContext::new(0).expect("Context");

    let src = GpuBuffer::<f32>::new(&ctx, 50).expect("Alloc src");
    let mut dst = GpuBuffer::<f32>::new(&ctx, 100).expect("Alloc dst");

    // src_offset 30 + count 50 = 80 > src.len 50
    let result = dst.copy_from_buffer_at(&src, 0, 30, 50);
    assert!(
        result.is_err(),
        "D2D copy_at should fail when src_offset+count > src.len"
    );
}

/// Falsification Test 9: RAII cleanup verification
/// Allocate, drop, verify memory returns to the pool
#[test]
fn test_raii_cleanup_single_buffer() {
    let ctx = CudaContext::new(0).expect("Context");

    let (free_before, _) = ctx.memory_info().expect("Memory info");

    // Allocate 100MB
    let size = 25_000_000; // 100MB of f32
    {
        let _buf = GpuBuffer::<f32>::new(&ctx, size).expect("Alloc");
        let (free_during, _) = ctx.memory_info().expect("Memory info");

        // Memory should be allocated (less free memory)
        assert!(
            free_during < free_before,
            "Memory should decrease after allocation: before={}, during={}",
            free_before,
            free_during
        );
    }
    // Buffer dropped here

    let (free_after, _) = ctx.memory_info().expect("Memory info");

    // Memory should be returned (within 10MB tolerance for driver overhead)
    let tolerance = 10 * 1024 * 1024;
    assert!(
        free_after >= free_before - tolerance,
        "Memory leak detected! before={}, after={}, diff={}",
        free_before,
        free_after,
        free_before.saturating_sub(free_after)
    );
}
