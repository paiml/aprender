//! GPU buffer allocation, copy, and round-trip tests

use super::*;

#[test]
fn test_gpu_buffer_new() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let buffer: GpuBuffer<f32> = GpuBuffer::new(&ctx, 1024).expect("Buffer new MUST succeed");
    assert_eq!(buffer.len(), 1024);
    assert!(buffer.as_ptr() != 0);
}

#[test]
fn test_gpu_buffer_from_host() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let buffer = GpuBuffer::from_host(&ctx, &data).expect("Buffer from_host MUST succeed");
    assert_eq!(buffer.len(), 8);
}

#[test]
fn test_gpu_buffer_round_trip() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let buffer = GpuBuffer::from_host(&ctx, &data).expect("Buffer creation MUST succeed");

    let mut result = vec![0.0f32; 4];
    buffer.copy_to_host(&mut result).expect("copy_to_host MUST succeed");
    assert_eq!(result, data, "Round-trip data MUST match");
}

#[test]
fn test_gpu_buffer_large_allocation() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    // Allocate 1GB buffer (RTX 4090 can handle this)
    let size = 256 * 1024 * 1024; // 256M floats = 1GB
    let buffer: GpuBuffer<f32> = GpuBuffer::new(&ctx, size).expect("Large buffer new MUST succeed");
    assert_eq!(buffer.len(), size);
}

#[test]
fn test_gpu_buffer_copy_from_host() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let mut buffer: GpuBuffer<f32> = GpuBuffer::new(&ctx, 1024).expect("Buffer new MUST succeed");

    let data = vec![42.0f32; 1024];
    buffer.copy_from_host(&data).expect("copy_from_host MUST succeed");

    let mut result = vec![0.0f32; 1024];
    buffer.copy_to_host(&mut result).expect("copy_to_host MUST succeed");
    assert_eq!(result[0], 42.0);
    assert_eq!(result[1023], 42.0);
}

#[test]
fn test_gpu_buffer_size_bytes() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let buffer: GpuBuffer<f32> = GpuBuffer::new(&ctx, 256).expect("Buffer new MUST succeed");
    assert_eq!(buffer.size_bytes(), 256 * std::mem::size_of::<f32>());
}

#[test]
fn test_gpu_buffer_is_empty() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let buffer: GpuBuffer<f32> = GpuBuffer::new(&ctx, 1).expect("Buffer new MUST succeed");
    assert!(!buffer.is_empty());
}

#[test]
fn test_gpu_buffer_clone() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let original = GpuBuffer::from_host(&ctx, &data).expect("Buffer creation MUST succeed");

    let cloned = original.clone(&ctx).expect("Buffer clone MUST succeed");
    assert_eq!(cloned.len(), original.len());

    let mut result = vec![0.0f32; 4];
    cloned.copy_to_host(&mut result).expect("copy_to_host MUST succeed");
    assert_eq!(result, data);
}

#[test]
fn test_gpu_buffer_copy_from_host_at_bounds_check() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let mut buffer: GpuBuffer<f32> = GpuBuffer::new(&ctx, 100).expect("Buffer MUST succeed");

    // Out of bounds
    let data = vec![1.0f32; 50];
    let result = buffer.copy_from_host_at(&data, 60); // 60 + 50 > 100
    assert!(result.is_err(), "Out of bounds MUST fail");

    // Empty data
    let empty: Vec<f32> = vec![];
    let result = buffer.copy_from_host_at(&empty, 50);
    assert!(result.is_ok(), "Empty copy MUST succeed");
}

#[test]
fn test_gpu_buffer_copy_to_host_at_bounds_check() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let buffer: GpuBuffer<f32> = GpuBuffer::new(&ctx, 100).expect("Buffer MUST succeed");

    // Out of bounds
    let mut data = vec![0.0f32; 50];
    let result = buffer.copy_to_host_at(&mut data, 60); // 60 + 50 > 100
    assert!(result.is_err(), "Out of bounds MUST fail");

    // Empty data
    let mut empty: Vec<f32> = vec![];
    let result = buffer.copy_to_host_at(&mut empty, 50);
    assert!(result.is_ok(), "Empty copy MUST succeed");
}

#[test]
fn test_gpu_buffer_copy_from_buffer_size_mismatch() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let src: GpuBuffer<f32> = GpuBuffer::new(&ctx, 100).expect("src MUST succeed");
    let mut dst: GpuBuffer<f32> = GpuBuffer::new(&ctx, 50).expect("dst MUST succeed");

    let result = dst.copy_from_buffer(&src);
    assert!(result.is_err(), "Size mismatch MUST fail");
}

#[test]
fn test_gpu_buffer_copy_from_buffer_empty() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let src: GpuBuffer<f32> = GpuBuffer::new(&ctx, 0).expect("src MUST succeed");
    let mut dst: GpuBuffer<f32> = GpuBuffer::new(&ctx, 0).expect("dst MUST succeed");

    let result = dst.copy_from_buffer(&src);
    assert!(result.is_ok(), "Empty copy MUST succeed");
}

#[test]
fn test_gpu_buffer_copy_from_buffer_at_bounds_check_dst() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let src: GpuBuffer<f32> = GpuBuffer::new(&ctx, 20).expect("src MUST succeed");
    let mut dst: GpuBuffer<f32> = GpuBuffer::new(&ctx, 50).expect("dst MUST succeed");

    // dst out of bounds: 40 + 20 > 50
    let result = dst.copy_from_buffer_at(&src, 40, 0, 20);
    assert!(result.is_err(), "dst out of bounds MUST fail");
}

#[test]
fn test_gpu_buffer_copy_from_buffer_at_bounds_check_src() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let src: GpuBuffer<f32> = GpuBuffer::new(&ctx, 20).expect("src MUST succeed");
    let mut dst: GpuBuffer<f32> = GpuBuffer::new(&ctx, 50).expect("dst MUST succeed");

    // src out of bounds: 15 + 20 > 20
    let result = dst.copy_from_buffer_at(&src, 0, 15, 20);
    assert!(result.is_err(), "src out of bounds MUST fail");
}

#[test]
fn test_gpu_buffer_copy_from_buffer_at_zero_count() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let src: GpuBuffer<f32> = GpuBuffer::new(&ctx, 20).expect("src MUST succeed");
    let mut dst: GpuBuffer<f32> = GpuBuffer::new(&ctx, 50).expect("dst MUST succeed");

    let result = dst.copy_from_buffer_at(&src, 0, 0, 0);
    assert!(result.is_ok(), "Zero count copy MUST succeed");
}

#[test]
fn test_gpu_buffer_view_operations() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let buffer: GpuBuffer<f32> = GpuBuffer::new(&ctx, 128).expect("Buffer MUST succeed");

    let view = buffer.clone_metadata();

    // Test all view methods
    assert_eq!(view.as_ptr(), buffer.as_ptr());
    assert_eq!(view.len(), 128);
    assert!(!view.is_empty());
    assert_eq!(view.size_bytes(), 128 * 4);
}

#[test]
fn test_gpu_buffer_empty_view() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let buffer: GpuBuffer<f32> = GpuBuffer::new(&ctx, 0).expect("Buffer MUST succeed");

    let view = buffer.clone_metadata();
    assert!(view.is_empty());
    assert_eq!(view.len(), 0);
    assert_eq!(view.size_bytes(), 0);
}

#[test]
fn test_gpu_buffer_kernel_arg() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let buffer: GpuBuffer<f32> = GpuBuffer::new(&ctx, 64).expect("Buffer MUST succeed");

    let arg = buffer.as_kernel_arg();
    assert!(!arg.is_null());

    // The pointer should point to a valid device address
    let ptr_to_ptr = arg as *const u64;
    let device_ptr = unsafe { *ptr_to_ptr };
    assert!(device_ptr != 0);
}
