//! Stress tests and advanced GPU buffer operations (async, raw, etc.)

use super::*;

// ============================================================================
// Stress Tests - Force All Code Paths
// ============================================================================

#[test]
fn test_cuda_stress_100_contexts() {
    // Create and destroy 100 contexts rapidly
    // GH-194: Assert memory_info succeeds instead of silently discarding errors
    for i in 0..100 {
        let ctx = CudaContext::new(0).unwrap_or_else(|_| panic!("Context {} MUST succeed", i));
        ctx.memory_info()
            .unwrap_or_else(|e| panic!("memory_info failed on context {}: {}", i, e));
    }
}

#[test]
fn test_cuda_stress_concurrent_streams() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");

    // Create 32 streams concurrently
    let mut streams = Vec::new();
    for _ in 0..32 {
        streams.push(CudaStream::new(&ctx).expect("Stream MUST succeed"));
    }

    // Sync all
    for stream in &streams {
        stream.synchronize().expect("Sync MUST succeed");
    }
}

/// GPU-ORD-4: "the card must hand me at least 2GB" is a claim on shared device
/// capacity. It failed with "Memory exhausted after 2 chunks (512MB)" on a 24GB
/// card, because `test_oom_resilience` was concurrently holding nearly all of
/// it. Both take `device_memory_exclusive()`.
#[test]
fn test_cuda_stress_memory_pressure() {
    let _exclusive = crate::driver::device_memory_exclusive();
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");

    // Allocate 4GB total in 256MB chunks
    let chunk_size = 64 * 1024 * 1024; // 64M floats = 256MB
    let mut buffers: Vec<GpuBuffer<f32>> = Vec::new();

    for i in 0..16 {
        match GpuBuffer::<f32>::new(&ctx, chunk_size) {
            Ok(buf) => buffers.push(buf),
            Err(_) => {
                eprintln!("Memory exhausted after {} chunks ({}MB)", i, i * 256);
                break;
            }
        }
    }

    // We should have allocated at least 8 chunks (2GB) on RTX 4090
    assert!(
        buffers.len() >= 8,
        "RTX 4090 should handle at least 2GB allocation"
    );

    // Drop all buffers - verify cleanup
    drop(buffers);

    // Should be able to allocate again
    let _new_buf: GpuBuffer<f32> =
        GpuBuffer::new(&ctx, chunk_size).expect("Post-cleanup allocation MUST succeed");
}

// ============================================================================
// GPU Buffer Advanced Tests - Force 95% Coverage
// ============================================================================

#[test]
fn test_gpu_buffer_copy_from_buffer_at_async_raw() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let stream = CudaStream::new(&ctx).expect("Stream creation MUST succeed");
    let stream_handle = stream.raw();

    let src_data: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let src = GpuBuffer::from_host(&ctx, &src_data).expect("src buffer MUST succeed");

    let mut dst: GpuBuffer<f32> = GpuBuffer::new(&ctx, 64).expect("dst buffer MUST succeed");
    let zeros = vec![0.0f32; 64];
    dst.copy_from_host(&zeros)
        .expect("copy_from_host MUST succeed");

    // Use raw stream handle API (line 636-669)
    unsafe {
        dst.copy_from_buffer_at_async_raw(&src, 16, 0, 32, stream_handle)
            .expect("copy_from_buffer_at_async_raw MUST succeed");
    }
    stream.synchronize().expect("Sync MUST succeed");

    let mut result = vec![0.0f32; 64];
    dst.copy_to_host(&mut result)
        .expect("copy_to_host MUST succeed");

    // Verify copy was correct
    assert_eq!(result[15], 0.0, "Before copy region should be 0");
    assert_eq!(result[16], 0.0, "First copied element should be 0.0");
    assert_eq!(result[47], 31.0, "Last copied element should be 31.0");
    assert_eq!(result[48], 0.0, "After copy region should be 0");
}

#[test]
fn test_gpu_buffer_copy_from_buffer_at_async_raw_bounds_check() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let stream = CudaStream::new(&ctx).expect("Stream creation MUST succeed");
    let stream_handle = stream.raw();

    let src: GpuBuffer<f32> = GpuBuffer::new(&ctx, 10).expect("src buffer MUST succeed");
    let mut dst: GpuBuffer<f32> = GpuBuffer::new(&ctx, 20).expect("dst buffer MUST succeed");

    // Test dst out of bounds
    let result = unsafe { dst.copy_from_buffer_at_async_raw(&src, 15, 0, 10, stream_handle) };
    assert!(result.is_err(), "dst out of bounds MUST fail");

    // Test src out of bounds
    let result = unsafe { dst.copy_from_buffer_at_async_raw(&src, 0, 5, 10, stream_handle) };
    assert!(result.is_err(), "src out of bounds MUST fail");

    // Test zero count (should succeed)
    let result = unsafe { dst.copy_from_buffer_at_async_raw(&src, 0, 0, 0, stream_handle) };
    assert!(result.is_ok(), "Zero count copy MUST succeed");
}

#[test]
fn test_gpu_buffer_async_host_to_device() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let stream = CudaStream::new(&ctx).expect("Stream creation MUST succeed");

    let mut buffer: GpuBuffer<f32> = GpuBuffer::new(&ctx, 256).expect("Buffer MUST succeed");
    let data: Vec<f32> = (0..256).map(|i| i as f32).collect();

    // Async host-to-device copy
    unsafe {
        buffer
            .copy_from_host_async(&data, &stream)
            .expect("copy_from_host_async MUST succeed");
    }
    stream.synchronize().expect("Sync MUST succeed");

    // Verify data
    let mut result = vec![0.0f32; 256];
    buffer
        .copy_to_host(&mut result)
        .expect("copy_to_host MUST succeed");
    assert_eq!(result, data);
}

#[test]
fn test_gpu_buffer_async_device_to_host() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let stream = CudaStream::new(&ctx).expect("Stream creation MUST succeed");

    let data: Vec<f32> = (0..128).map(|i| i as f32).collect();
    let buffer = GpuBuffer::from_host(&ctx, &data).expect("Buffer MUST succeed");

    let mut result = vec![0.0f32; 128];
    unsafe {
        buffer
            .copy_to_host_async(&mut result, &stream)
            .expect("copy_to_host_async MUST succeed");
    }
    stream.synchronize().expect("Sync MUST succeed");

    assert_eq!(result, data);
}

#[test]
fn test_gpu_buffer_async_copy_size_mismatch_h2d() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let stream = CudaStream::new(&ctx).expect("Stream creation MUST succeed");

    let mut buffer: GpuBuffer<f32> = GpuBuffer::new(&ctx, 100).expect("Buffer MUST succeed");
    let data: Vec<f32> = vec![1.0f32; 200]; // Wrong size

    let result = unsafe { buffer.copy_from_host_async(&data, &stream) };
    assert!(result.is_err(), "Size mismatch MUST fail");
}

#[test]
fn test_gpu_buffer_async_copy_size_mismatch_d2h() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let stream = CudaStream::new(&ctx).expect("Stream creation MUST succeed");

    let buffer: GpuBuffer<f32> = GpuBuffer::new(&ctx, 100).expect("Buffer MUST succeed");
    let mut result: Vec<f32> = vec![0.0f32; 50]; // Wrong size

    let copy_result = unsafe { buffer.copy_to_host_async(&mut result, &stream) };
    assert!(copy_result.is_err(), "Size mismatch MUST fail");
}

#[test]
fn test_gpu_buffer_async_copy_empty_h2d() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let stream = CudaStream::new(&ctx).expect("Stream creation MUST succeed");

    let mut buffer: GpuBuffer<f32> = GpuBuffer::new(&ctx, 0).expect("Buffer MUST succeed");
    let data: Vec<f32> = vec![];

    let result = unsafe { buffer.copy_from_host_async(&data, &stream) };
    assert!(result.is_ok(), "Empty copy MUST succeed");
}

#[test]
fn test_gpu_buffer_async_copy_empty_d2h() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let stream = CudaStream::new(&ctx).expect("Stream creation MUST succeed");

    let buffer: GpuBuffer<f32> = GpuBuffer::new(&ctx, 0).expect("Buffer MUST succeed");
    let mut result: Vec<f32> = vec![];

    let copy_result = unsafe { buffer.copy_to_host_async(&mut result, &stream) };
    assert!(copy_result.is_ok(), "Empty copy MUST succeed");
}
