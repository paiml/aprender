//! Layer Norm, GELU, and Bias Add tests

use super::*;

// ============================================================================
// Layer Norm Tests
// ============================================================================

#[test]
fn test_ops_layer_norm() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let hidden_size = 16u32;
    let batch_size = 4u32;

    let input_data: Vec<f32> = (0..(hidden_size * batch_size))
        .map(|i| i as f32 * 0.1)
        .collect();
    let gamma_data = vec![1.0f32; hidden_size as usize];
    let beta_data = vec![0.0f32; hidden_size as usize];

    let input = GpuResidentTensor::from_host(&ctx, &input_data).unwrap();
    let gamma = GpuResidentTensor::from_host(&ctx, &gamma_data).unwrap();
    let beta = GpuResidentTensor::from_host(&ctx, &beta_data).unwrap();

    let mut output = input
        .layer_norm(&ctx, &gamma, &beta, hidden_size, batch_size)
        .unwrap();
    assert_eq!(output.len(), (hidden_size * batch_size) as usize);

    let host_output = output.to_host().unwrap();
    let first_row: Vec<f32> = host_output[0..hidden_size as usize].to_vec();
    let mean: f32 = first_row.iter().sum::<f32>() / hidden_size as f32;
    assert!(
        (mean).abs() < 0.1,
        "LayerNorm output mean should be ~0, got {}",
        mean
    );
}

#[test]
fn test_ops_layer_norm_with_stream() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let hidden_size = 32u32;
    let batch_size = 2u32;

    let input_data = vec![1.0f32; (hidden_size * batch_size) as usize];
    let gamma_data = vec![1.0f32; hidden_size as usize];
    let beta_data = vec![0.0f32; hidden_size as usize];

    let input = GpuResidentTensor::from_host(&ctx, &input_data).unwrap();
    let gamma = GpuResidentTensor::from_host(&ctx, &gamma_data).unwrap();
    let beta = GpuResidentTensor::from_host(&ctx, &beta_data).unwrap();

    let stream = CudaStream::new(&ctx).unwrap();
    let output = input
        .layer_norm_with_stream(&ctx, &gamma, &beta, hidden_size, batch_size, &stream)
        .unwrap();
    stream.synchronize().unwrap();

    assert_eq!(output.len(), (hidden_size * batch_size) as usize);
}

// ============================================================================
// GELU Tests
// ============================================================================

#[test]
fn test_ops_gelu() {
    let ctx = cuda_ctx!();

    let data = vec![0.0f32, 1.0, 2.0, -1.0, -2.0];
    let tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();

    let mut output = tensor.gelu(&ctx).unwrap();
    let result = output.to_host().unwrap();

    assert!((result[0]).abs() < 1e-5, "GELU(0) should be ~0");
    assert!((result[1] - 0.841).abs() < 0.1, "GELU(1) should be ~0.841");
    assert!(
        (result[3] - (-0.159)).abs() < 0.1,
        "GELU(-1) should be ~-0.159"
    );
}

#[test]
fn test_ops_gelu_with_stream() {
    let ctx = cuda_ctx!();

    let data = vec![1.0f32; 512];
    let tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();

    let stream = CudaStream::new(&ctx).unwrap();
    let output = tensor.gelu_with_stream(&ctx, &stream).unwrap();
    stream.synchronize().unwrap();

    assert_eq!(output.len(), 512);
}

// ============================================================================
// Bias Add Tests
// ============================================================================

#[test]
fn test_ops_bias_add() {
    let ctx = cuda_ctx!();

    let input_data = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let bias_data = vec![0.1f32, 0.2, 0.3];

    let input = GpuResidentTensor::from_host(&ctx, &input_data).unwrap();
    let bias = GpuResidentTensor::from_host(&ctx, &bias_data).unwrap();

    let mut output = input.bias_add(&ctx, &bias).unwrap();
    let result = output.to_host().unwrap();

    assert!((result[0] - 1.1).abs() < 1e-5);
    assert!((result[1] - 2.2).abs() < 1e-5);
    assert!((result[2] - 3.3).abs() < 1e-5);
}

#[test]
fn test_ops_bias_add_with_stream() {
    let ctx = cuda_ctx!();

    let input_data = vec![1.0f32; 256];
    let bias_data = vec![0.5f32; 64];

    let input = GpuResidentTensor::from_host(&ctx, &input_data).unwrap();
    let bias = GpuResidentTensor::from_host(&ctx, &bias_data).unwrap();

    let stream = CudaStream::new(&ctx).unwrap();
    let mut output = input.bias_add_with_stream(&ctx, &bias, &stream).unwrap();
    stream.synchronize().unwrap();

    let result = output.to_host().unwrap();
    assert!((result[0] - 1.5).abs() < 1e-5);
}
