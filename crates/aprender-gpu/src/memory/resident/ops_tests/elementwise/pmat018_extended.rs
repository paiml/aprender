//! PMAT-018: ops.rs elementwise coverage tests for 95%+ target

use super::*;

#[test]
fn test_ops_softmax_with_stream_warp_path() {
    let ctx = fresh_ctx!();
    let (seq_len, row_size) = (4u32, 16u32);
    let tensor = upload(&ctx, &ramp_f32((seq_len * row_size) as usize, 0.1, 0.0));
    let stream = CudaStream::new(&ctx).unwrap();

    let result = tensor.softmax_with_stream(&ctx, seq_len, &stream).unwrap();
    stream.synchronize().unwrap();
    assert_eq!(result.len(), (seq_len * row_size) as usize);
}

#[test]
fn test_ops_softmax_with_stream_long_row_path() {
    let ctx = fresh_ctx!();
    let (seq_len, row_size) = (4u32, 64u32);
    let tensor = upload(&ctx, &ramp_f32((seq_len * row_size) as usize, 0.1, 0.0));
    let stream = CudaStream::new(&ctx).unwrap();

    let result = tensor.softmax_with_stream(&ctx, seq_len, &stream).unwrap();
    stream.synchronize().unwrap();
    assert_eq!(result.len(), (seq_len * row_size) as usize);
}

#[test]
fn test_ops_softmax_with_stream_dimension_error() {
    let ctx = fresh_ctx!();
    let tensor = upload(&ctx, &vec![1.0; 10]);
    let stream = CudaStream::new(&ctx).unwrap();

    assert!(tensor.softmax_with_stream(&ctx, 3, &stream).is_err());
}

#[test]
fn test_ops_add_with_stream_dimension_error() {
    let ctx = fresh_ctx!();
    let a = upload(&ctx, &vec![1.0f32; 10]);
    let b = upload(&ctx, &vec![1.0f32; 5]);
    let stream = CudaStream::new(&ctx).unwrap();

    assert!(a.add_with_stream(&ctx, &b, &stream).is_err());
}

#[test]
fn test_ops_scale_larger_tensor() {
    let ctx = fresh_ctx!();
    let tensor = upload(&ctx, &ramp_f32(1024, 0.01, 0.0));

    let mut scaled = tensor.scale(&ctx, 0.5).unwrap();
    let result = scaled.to_host().unwrap();

    assert!((result[0] - 0.0).abs() < 1e-5);
    assert!((result[100] - 0.5).abs() < 1e-5);
    assert!((result[1000] - 5.0).abs() < 1e-5);
}

#[test]
fn test_ops_gelu_larger_tensor() {
    let ctx = fresh_ctx!();
    let tensor = upload(&ctx, &ramp_f32(1024, 0.01, -512.0 * 0.01));
    assert_eq!(tensor.gelu(&ctx).unwrap().len(), 1024);
}

#[test]
fn test_ops_layer_norm_larger_batch() {
    let ctx = fresh_ctx!();
    let (hidden, batch) = (32u32, 16u32);

    let input = upload(&ctx, &ramp_f32((hidden * batch) as usize, 0.01, 0.0));
    let gamma = upload(&ctx, &vec![1.0f32; hidden as usize]);
    let beta = upload(&ctx, &vec![0.0f32; hidden as usize]);

    let output = input.layer_norm(&ctx, &gamma, &beta, hidden, batch).unwrap();
    assert_eq!(output.len(), (hidden * batch) as usize);
}

#[test]
fn test_ops_bias_add_larger_tensor() {
    let ctx = fresh_ctx!();
    let (n_rows, bias_size) = (256usize, 64usize);

    let input = upload(&ctx, &vec![1.0f32; n_rows * bias_size]);
    let bias = upload(&ctx, &ramp_f32(bias_size, 0.1, 0.0));

    let mut output = input.bias_add(&ctx, &bias).unwrap();
    let result = output.to_host().unwrap();

    assert!((result[0] - 1.0).abs() < 1e-5);
    assert!((result[1] - 1.1).abs() < 1e-5);
    assert!((result[64] - 1.0).abs() < 1e-5);
}

#[test]
fn test_ops_add_larger_tensor() {
    let ctx = fresh_ctx!();
    let size = 1024;

    let a = upload(&ctx, &ramp_f32(size, 1.0, 0.0));
    let b_data: Vec<f32> = (0..size).map(|i| (size - i) as f32).collect();
    let b = upload(&ctx, &b_data);

    let mut c = a.add(&ctx, &b).unwrap();
    let result = c.to_host().unwrap();
    assert!(result.iter().all(|&v| (v - 1024.0).abs() < 1e-3));
}

#[test]
fn test_ops_gelu_with_stream_larger() {
    let ctx = fresh_ctx!();
    let tensor = upload(&ctx, &ramp_f32(2048, 0.001, -1.024));
    let stream = CudaStream::new(&ctx).unwrap();

    let output = tensor.gelu_with_stream(&ctx, &stream).unwrap();
    stream.synchronize().unwrap();
    assert_eq!(output.len(), 2048);
}

#[test]
fn test_ops_layer_norm_with_stream_larger() {
    let ctx = fresh_ctx!();
    let (hidden, batch) = (64u32, 8u32);

    let input = upload(&ctx, &ramp_f32((hidden * batch) as usize, 0.01, 0.0));
    let gamma = upload(&ctx, &vec![1.0f32; hidden as usize]);
    let beta = upload(&ctx, &vec![0.5f32; hidden as usize]);

    let stream = CudaStream::new(&ctx).unwrap();
    let output = input.layer_norm_with_stream(&ctx, &gamma, &beta, hidden, batch, &stream).unwrap();
    stream.synchronize().unwrap();
    assert_eq!(output.len(), (hidden * batch) as usize);
}

#[test]
fn test_ops_bias_add_with_stream_larger() {
    let ctx = fresh_ctx!();
    let (n_rows, bias_size) = (128usize, 32usize);

    let input = upload(&ctx, &vec![0.5f32; n_rows * bias_size]);
    let bias = upload(&ctx, &ramp_f32(bias_size, 0.05, 0.0));

    let stream = CudaStream::new(&ctx).unwrap();
    let output = input.bias_add_with_stream(&ctx, &bias, &stream).unwrap();
    stream.synchronize().unwrap();
    assert_eq!(output.len(), n_rows * bias_size);
}

#[test]
fn test_ops_add_with_stream_larger() {
    let ctx = fresh_ctx!();
    let size = 2048;

    let a = upload(&ctx, &ramp_f32(size, 0.001, 0.0));
    let b_data: Vec<f32> = (0..size).map(|i| (size - i) as f32 * 0.001).collect();
    let b = upload(&ctx, &b_data);

    let stream = CudaStream::new(&ctx).unwrap();
    let c = a.add_with_stream(&ctx, &b, &stream).unwrap();
    stream.synchronize().unwrap();
    assert_eq!(c.len(), size);
}
