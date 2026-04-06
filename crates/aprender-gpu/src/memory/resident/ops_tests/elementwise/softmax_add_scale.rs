//! Softmax, Add, and Scale tests

use super::*;

// ============================================================================
// Softmax Tests
// ============================================================================

#[test]
fn test_ops_softmax_warp() {
    let ctx = cuda_ctx!();

    // Small row size (<=32) uses warp shuffle softmax
    let seq_len = 8u32;
    let row_size = 16u32;
    let data: Vec<f32> = (0..(seq_len * row_size))
        .map(|i| (i % row_size) as f32 * 0.1)
        .collect();

    let tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();
    let mut result = tensor.softmax(&ctx, seq_len).unwrap();

    assert_eq!(result.len(), (seq_len * row_size) as usize);

    // Verify softmax output - values should be positive and <= 1
    let host_result = result.to_host().unwrap();
    for val in &host_result {
        assert!(*val >= 0.0 && *val <= 1.0 + 1e-5);
    }
}

#[test]
fn test_ops_softmax_long_row() {
    let ctx = cuda_ctx!();

    // Large row size (>32) uses long row softmax
    let seq_len = 4u32;
    let row_size = 128u32;
    let data: Vec<f32> = (0..(seq_len * row_size))
        .map(|i| (i % row_size) as f32 * 0.01)
        .collect();

    let tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();
    let result = tensor.softmax(&ctx, seq_len).unwrap();

    assert_eq!(result.len(), (seq_len * row_size) as usize);
}

#[test]
fn test_ops_softmax_dimension_error() {
    let ctx = cuda_ctx!();

    // Tensor size not divisible by seq_len
    let data: Vec<f32> = vec![1.0; 10];
    let tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();

    let result = tensor.softmax(&ctx, 3); // 10 not divisible by 3
    assert!(result.is_err());
}

#[test]
fn test_ops_softmax_with_stream() {
    let ctx = cuda_ctx!();

    let seq_len = 4u32;
    let row_size = 64u32;
    let data: Vec<f32> = vec![1.0; (seq_len * row_size) as usize];

    let tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();
    let stream = CudaStream::new(&ctx).unwrap();

    let result = tensor.softmax_with_stream(&ctx, seq_len, &stream).unwrap();
    stream.synchronize().unwrap();

    assert_eq!(result.len(), (seq_len * row_size) as usize);
}

// ============================================================================
// Add Tests
// ============================================================================

#[test]
fn test_ops_add() {
    let ctx = cuda_ctx!();

    let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b_data = vec![0.5f32, 0.5, 0.5, 0.5];

    let a = GpuResidentTensor::from_host(&ctx, &a_data).unwrap();
    let b = GpuResidentTensor::from_host(&ctx, &b_data).unwrap();

    let mut c = a.add(&ctx, &b).unwrap();
    let result = c.to_host().unwrap();

    assert_eq!(result, vec![1.5, 2.5, 3.5, 4.5]);
}

#[test]
fn test_ops_add_size_mismatch() {
    let ctx = cuda_ctx!();

    let a_data = vec![1.0f32; 10];
    let b_data = vec![1.0f32; 5];

    let a = GpuResidentTensor::from_host(&ctx, &a_data).unwrap();
    let b = GpuResidentTensor::from_host(&ctx, &b_data).unwrap();

    let result = a.add(&ctx, &b);
    assert!(result.is_err());
}

#[test]
fn test_ops_add_with_stream() {
    let ctx = cuda_ctx!();

    let a_data = vec![1.0f32; 256];
    let b_data = vec![2.0f32; 256];

    let a = GpuResidentTensor::from_host(&ctx, &a_data).unwrap();
    let b = GpuResidentTensor::from_host(&ctx, &b_data).unwrap();

    let stream = CudaStream::new(&ctx).unwrap();
    let mut c = a.add_with_stream(&ctx, &b, &stream).unwrap();
    stream.synchronize().unwrap();

    let result = c.to_host().unwrap();
    assert!(result.iter().all(|&v| (v - 3.0).abs() < 1e-5));
}

// ============================================================================
// Scale Tests
// ============================================================================

#[test]
fn test_ops_scale() {
    let ctx = cuda_ctx!();

    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();

    let mut scaled = tensor.scale(&ctx, 2.0).unwrap();
    let result = scaled.to_host().unwrap();

    assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0]);
}
