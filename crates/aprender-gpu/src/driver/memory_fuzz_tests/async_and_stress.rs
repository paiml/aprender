//! Async copy tests and stress tests: Falsification Tests 10-18

use super::*;

/// Falsification Test 10: Async copy size mismatch
#[test]
fn test_async_d2d_copy_size_mismatch() {
    use super::super::stream::CudaStream;

    let ctx = CudaContext::new(0).expect("Context");
    let stream = CudaStream::new(&ctx).expect("Stream");

    let src = GpuBuffer::<f32>::new(&ctx, 100).expect("Alloc src");
    let mut dst = GpuBuffer::<f32>::new(&ctx, 200).expect("Alloc dst");

    let result = unsafe { dst.copy_from_buffer_async(&src, &stream) };
    assert!(
        result.is_err(),
        "Async D2D copy should fail when buffer sizes don't match"
    );
}

/// Falsification Test 11: Async partial copy out of bounds
#[test]
fn test_async_d2d_copy_at_out_of_bounds() {
    use super::super::stream::CudaStream;

    let ctx = CudaContext::new(0).expect("Context");
    let stream = CudaStream::new(&ctx).expect("Stream");

    let src = GpuBuffer::<f32>::new(&ctx, 50).expect("Alloc src");
    let mut dst = GpuBuffer::<f32>::new(&ctx, 100).expect("Alloc dst");

    // dst out of bounds
    let result = unsafe { dst.copy_from_buffer_at_async(&src, 60, 0, 50, &stream) };
    assert!(
        result.is_err(),
        "Async D2D copy_at should fail when dst out of bounds"
    );

    // src out of bounds
    let result = unsafe { dst.copy_from_buffer_at_async(&src, 0, 30, 50, &stream) };
    assert!(
        result.is_err(),
        "Async D2D copy_at should fail when src out of bounds"
    );
}

/// Falsification Test 12: Async H2D copy size mismatch
#[test]
fn test_async_h2d_copy_size_mismatch() {
    use super::super::stream::CudaStream;

    let ctx = CudaContext::new(0).expect("Context");
    let stream = CudaStream::new(&ctx).expect("Stream");

    let mut buf = GpuBuffer::<f32>::new(&ctx, 100).expect("Alloc");
    let small_data = vec![1.0f32; 50];

    let result = unsafe { buf.copy_from_host_async(&small_data, &stream) };
    assert!(
        result.is_err(),
        "Async H2D copy should fail when host buffer size doesn't match"
    );
}

/// Falsification Test 13: Async D2H copy size mismatch
#[test]
fn test_async_d2h_copy_size_mismatch() {
    use super::super::stream::CudaStream;

    let ctx = CudaContext::new(0).expect("Context");
    let stream = CudaStream::new(&ctx).expect("Stream");

    let buf = GpuBuffer::<f32>::new(&ctx, 100).expect("Alloc");
    let mut large_data = vec![0.0f32; 200];

    let result = unsafe { buf.copy_to_host_async(&mut large_data, &stream) };
    assert!(
        result.is_err(),
        "Async D2H copy should fail when host buffer size doesn't match"
    );
}

/// Falsification Test 14: Empty buffer operations
#[test]
fn test_empty_buffer_operations() {
    let ctx = CudaContext::new(0).expect("Context");

    let mut empty_buf = GpuBuffer::<f32>::new(&ctx, 0).expect("Alloc empty");
    assert!(empty_buf.is_empty());
    assert_eq!(empty_buf.len(), 0);
    assert_eq!(empty_buf.size_bytes(), 0);

    // All these should succeed as no-ops
    let empty_data: Vec<f32> = vec![];
    empty_buf
        .copy_from_host(&empty_data)
        .expect("Empty H2D should succeed");

    let mut empty_out: Vec<f32> = vec![];
    empty_buf
        .copy_to_host(&mut empty_out)
        .expect("Empty D2H should succeed");

    // D2D with empty buffers
    let mut empty_dst = GpuBuffer::<f32>::new(&ctx, 0).expect("Alloc empty dst");
    empty_dst
        .copy_from_buffer(&empty_buf)
        .expect("Empty D2D should succeed");
}

/// Falsification Test 15: Partial copy with zero count
#[test]
fn test_partial_copy_zero_count() {
    let ctx = CudaContext::new(0).expect("Context");

    let src = GpuBuffer::<f32>::new(&ctx, 100).expect("Alloc src");
    let mut dst = GpuBuffer::<f32>::new(&ctx, 100).expect("Alloc dst");

    // Zero count should be a no-op, regardless of offsets
    dst.copy_from_buffer_at(&src, 0, 0, 0)
        .expect("Zero count D2D should succeed");
    dst.copy_from_buffer_at(&src, 50, 50, 0)
        .expect("Zero count D2D with offsets should succeed");
}

/// Falsification Test 16: Async raw copy bounds check
#[test]
fn test_async_raw_copy_bounds_check() {
    use super::super::stream::CudaStream;

    let ctx = CudaContext::new(0).expect("Context");
    let stream = CudaStream::new(&ctx).expect("Stream");

    let src = GpuBuffer::<f32>::new(&ctx, 50).expect("Alloc src");
    let mut dst = GpuBuffer::<f32>::new(&ctx, 100).expect("Alloc dst");

    // Use raw stream handle
    let stream_handle = stream.raw();

    // dst out of bounds
    let result = unsafe { dst.copy_from_buffer_at_async_raw(&src, 60, 0, 50, stream_handle) };
    assert!(
        result.is_err(),
        "Async raw D2D should fail when dst out of bounds"
    );

    // src out of bounds
    let result = unsafe { dst.copy_from_buffer_at_async_raw(&src, 0, 30, 50, stream_handle) };
    assert!(
        result.is_err(),
        "Async raw D2D should fail when src out of bounds"
    );

    // Zero count should succeed
    unsafe {
        dst.copy_from_buffer_at_async_raw(&src, 0, 0, 0, stream_handle)
            .expect("Zero count should succeed");
    }
}

/// Falsification Test 17: Buffer view properties
#[test]
fn test_buffer_view_properties() {
    let ctx = CudaContext::new(0).expect("Context");
    let buf = GpuBuffer::<f32>::new(&ctx, 256).expect("Alloc");

    let view = buf.clone_metadata();

    assert_eq!(view.as_ptr(), buf.as_ptr());
    assert_eq!(view.len(), buf.len());
    assert_eq!(view.is_empty(), buf.is_empty());
    assert_eq!(view.size_bytes(), buf.size_bytes());

    // View should NOT free memory when dropped (non-owning)
    drop(view);

    // Original buffer should still be valid
    assert_eq!(buf.len(), 256);
}

/// Falsification Test 18: Stress multiple allocations and drops
///
/// GPU-ORD-4: measured against this thread's own accounting rather than
/// device-global free memory, which neighbouring tests move underneath it.
/// Exact, so a single leaked byte fails — the old device-global reading needed
/// a 50MB tolerance and still failed spuriously ("leaked=146800640" when
/// nothing had leaked).
#[test]
fn test_stress_alloc_dealloc_cycle() {
    let ctx = CudaContext::new(0).expect("Context");
    let start = device_bytes_outstanding();

    // Allocate and drop 100 buffers
    for i in 0..100 {
        let size = (i + 1) * 1000; // 1K to 100K elements
        let _buf = GpuBuffer::<f32>::new(&ctx, size).expect("Alloc");
        // Buffer dropped at end of iteration
    }

    assert_eq!(
        device_bytes_outstanding(),
        start,
        "Memory leak after 100 alloc/dealloc cycles"
    );
}
