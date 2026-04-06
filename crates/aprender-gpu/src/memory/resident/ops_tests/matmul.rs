//! Matmul GPU-Resident Tensor tests: naive, tiled, WMMA, stream, force FP32

use crate::driver::{CudaContext, CudaStream};
use crate::memory::resident::{reset_transfer_counters, GpuResidentTensor};

/// Helper to create CUDA context, skipping test if unavailable
macro_rules! cuda_ctx {
    () => {
        match CudaContext::new(0) {
            Ok(ctx) => ctx,
            Err(e) => {
                eprintln!("Skipping CUDA test: {:?}", e);
                return;
            }
        }
    };
}

// ============================================================================
// Matmul Tests
// ============================================================================

#[test]
fn test_ops_matmul_naive_small() {
    let ctx = cuda_ctx!();
    reset_transfer_counters();

    // 4x4 @ 4x4 = 4x4 (naive path: k < 64)
    let m = 4u32;
    let n = 4u32;
    let k = 4u32;

    // A: 4x4 identity-like matrix
    let a_data: Vec<f32> = (0..16).map(|i| if i % 5 == 0 { 1.0 } else { 0.0 }).collect();
    // B: 4x4 matrix with values 1-16
    let b_data: Vec<f32> = (1..=16).map(|i| i as f32).collect();

    let a = GpuResidentTensor::from_host(&ctx, &a_data).unwrap();
    let b = GpuResidentTensor::from_host(&ctx, &b_data).unwrap();

    let c = a.matmul(&ctx, &b, m, n, k).unwrap();
    assert_eq!(c.len(), (m * n) as usize);

    // Verify result is on GPU (device resident)
    assert!(c.is_device_resident());
}

#[test]
fn test_ops_matmul_tiled() {
    let ctx = cuda_ctx!();

    // 64x64 @ 64x64 = 64x64 (tiled path: k >= 64)
    let m = 64u32;
    let n = 64u32;
    let k = 64u32;

    let size = (m * k) as usize;
    let a_data: Vec<f32> = (0..size).map(|i| (i % 10) as f32 * 0.1).collect();
    let b_data: Vec<f32> = (0..size).map(|i| (i % 7) as f32 * 0.1).collect();

    let a = GpuResidentTensor::from_host(&ctx, &a_data).unwrap();
    let b = GpuResidentTensor::from_host(&ctx, &b_data).unwrap();

    let c = a.matmul(&ctx, &b, m, n, k).unwrap();
    assert_eq!(c.len(), (m * n) as usize);
}

#[test]
fn test_ops_matmul_wmma() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    // 128x128 @ 128x128 (WMMA path: m,n,k all >= 64)
    let m = 128u32;
    let n = 128u32;
    let k = 128u32;

    let size_a = (m * k) as usize;
    let size_b = (k * n) as usize;
    let a_data: Vec<f32> = (0..size_a).map(|i| (i % 5) as f32 * 0.01).collect();
    let b_data: Vec<f32> = (0..size_b).map(|i| (i % 3) as f32 * 0.01).collect();

    let a = GpuResidentTensor::from_host(&ctx, &a_data).unwrap();
    let b = GpuResidentTensor::from_host(&ctx, &b_data).unwrap();

    let c = a.matmul(&ctx, &b, m, n, k).unwrap();
    assert_eq!(c.len(), (m * n) as usize);
}

#[test]
fn test_ops_matmul_dimension_error() {
    let ctx = cuda_ctx!();

    // Create tensors with wrong sizes
    let a_data = vec![1.0f32; 16]; // 4x4
    let b_data = vec![1.0f32; 9]; // 3x3

    let a = GpuResidentTensor::from_host(&ctx, &a_data).unwrap();
    let b = GpuResidentTensor::from_host(&ctx, &b_data).unwrap();

    // Try to multiply 4x4 @ 4x4, but B has only 9 elements
    let result = a.matmul(&ctx, &b, 4, 4, 4);
    assert!(result.is_err());
}

#[test]
fn test_ops_matmul_with_stream() {
    let ctx = cuda_ctx!();

    let m = 32u32;
    let n = 32u32;
    let k = 32u32;

    let a_data: Vec<f32> = vec![1.0; (m * k) as usize];
    let b_data: Vec<f32> = vec![1.0; (k * n) as usize];

    let a = GpuResidentTensor::from_host(&ctx, &a_data).unwrap();
    let b = GpuResidentTensor::from_host(&ctx, &b_data).unwrap();

    let stream = CudaStream::new(&ctx).unwrap();
    let c = a.matmul_with_stream(&ctx, &b, m, n, k, &stream).unwrap();
    stream.synchronize().unwrap();

    assert_eq!(c.len(), (m * n) as usize);
}

// ============================================================================
// WAPR-PERF-014: Force FP32 GEMM Path Test
// ============================================================================

#[test]
fn test_ops_matmul_force_fp32() {
    let ctx = cuda_ctx!();

    // Set env var to force FP32 path (skip WMMA)
    std::env::set_var("TRUENO_FORCE_FP32_GEMM", "1");

    let m = 128u32;
    let n = 128u32;
    let k = 128u32;

    let a_data = vec![1.0f32; (m * k) as usize];
    let b_data = vec![1.0f32; (k * n) as usize];

    let a = GpuResidentTensor::from_host(&ctx, &a_data).unwrap();
    let b = GpuResidentTensor::from_host(&ctx, &b_data).unwrap();

    let c = a.matmul(&ctx, &b, m, n, k).unwrap();
    assert_eq!(c.len(), (m * n) as usize);

    // Clean up env var
    std::env::remove_var("TRUENO_FORCE_FP32_GEMM");
}

// ============================================================================
// PMAT-018: Additional matmul coverage tests for 95%+ target
// ============================================================================

#[test]
fn test_ops_matmul_with_stream_dimension_error_a() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    // Create tensor A with wrong size
    let a_data = vec![1.0f32; 10]; // Wrong: should be 32
    let b_data = vec![1.0f32; 32];

    let a = GpuResidentTensor::from_host(&ctx, &a_data).unwrap();
    let b = GpuResidentTensor::from_host(&ctx, &b_data).unwrap();

    let stream = CudaStream::new(&ctx).unwrap();
    // Try 4x8 @ 8x4, but A has only 10 elements (should be 32)
    let result = a.matmul_with_stream(&ctx, &b, 4, 4, 8, &stream);
    assert!(result.is_err());
}

#[test]
fn test_ops_matmul_with_stream_dimension_error_b() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let a_data = vec![1.0f32; 32]; // 4x8 = 32
    let b_data = vec![1.0f32; 20]; // Wrong: should be 32 for 8x4

    let a = GpuResidentTensor::from_host(&ctx, &a_data).unwrap();
    let b = GpuResidentTensor::from_host(&ctx, &b_data).unwrap();

    let stream = CudaStream::new(&ctx).unwrap();
    let result = a.matmul_with_stream(&ctx, &b, 4, 4, 8, &stream);
    assert!(result.is_err());
}

#[test]
fn test_ops_matmul_with_stream_wmma_path() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    // Large matrices to trigger WMMA path (m, n, k all >= 64)
    let m = 128u32;
    let n = 128u32;
    let k = 128u32;

    let a_data = vec![0.01f32; (m * k) as usize];
    let b_data = vec![0.01f32; (k * n) as usize];

    let a = GpuResidentTensor::from_host(&ctx, &a_data).unwrap();
    let b = GpuResidentTensor::from_host(&ctx, &b_data).unwrap();

    let stream = CudaStream::new(&ctx).unwrap();
    let c = a.matmul_with_stream(&ctx, &b, m, n, k, &stream).unwrap();
    stream.synchronize().unwrap();

    assert_eq!(c.len(), (m * n) as usize);
}

#[test]
fn test_ops_matmul_with_stream_tiled_path() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    // k >= 64 but m or n < 64 triggers tiled path
    let m = 32u32;
    let n = 32u32;
    let k = 64u32;

    let a_data = vec![0.1f32; (m * k) as usize];
    let b_data = vec![0.1f32; (k * n) as usize];

    let a = GpuResidentTensor::from_host(&ctx, &a_data).unwrap();
    let b = GpuResidentTensor::from_host(&ctx, &b_data).unwrap();

    let stream = CudaStream::new(&ctx).unwrap();
    let c = a.matmul_with_stream(&ctx, &b, m, n, k, &stream).unwrap();
    stream.synchronize().unwrap();

    assert_eq!(c.len(), (m * n) as usize);
}

#[test]
fn test_ops_matmul_with_stream_force_fp32() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    // Set env var to force FP32 path
    std::env::set_var("TRUENO_FORCE_FP32_GEMM", "1");

    let m = 128u32;
    let n = 128u32;
    let k = 128u32;

    let a_data = vec![0.01f32; (m * k) as usize];
    let b_data = vec![0.01f32; (k * n) as usize];

    let a = GpuResidentTensor::from_host(&ctx, &a_data).unwrap();
    let b = GpuResidentTensor::from_host(&ctx, &b_data).unwrap();

    let stream = CudaStream::new(&ctx).unwrap();
    let c = a.matmul_with_stream(&ctx, &b, m, n, k, &stream).unwrap();
    stream.synchronize().unwrap();

    assert_eq!(c.len(), (m * n) as usize);

    std::env::remove_var("TRUENO_FORCE_FP32_GEMM");
}

#[test]
fn test_ops_matmul_naive_path() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    // Small matrix (k < 64) uses naive path
    let m = 8u32;
    let n = 8u32;
    let k = 8u32;

    let a_data = vec![0.5f32; (m * k) as usize];
    let b_data = vec![0.5f32; (k * n) as usize];

    let a = GpuResidentTensor::from_host(&ctx, &a_data).unwrap();
    let b = GpuResidentTensor::from_host(&ctx, &b_data).unwrap();

    let c = a.matmul(&ctx, &b, m, n, k).unwrap();
    assert_eq!(c.len(), (m * n) as usize);
}

#[test]
fn test_ops_matmul_with_stream_naive_path() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    // Small matrix (k < 64) uses naive path
    let m = 16u32;
    let n = 16u32;
    let k = 16u32;

    let a_data = vec![0.25f32; (m * k) as usize];
    let b_data = vec![0.25f32; (k * n) as usize];

    let a = GpuResidentTensor::from_host(&ctx, &a_data).unwrap();
    let b = GpuResidentTensor::from_host(&ctx, &b_data).unwrap();

    let stream = CudaStream::new(&ctx).unwrap();
    let c = a.matmul_with_stream(&ctx, &b, m, n, k, &stream).unwrap();
    stream.synchronize().unwrap();

    assert_eq!(c.len(), (m * n) as usize);
}
