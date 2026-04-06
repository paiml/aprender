use super::*;

#[test]
#[cfg(feature = "cuda")]
fn test_layer_norm_forward_basic() {
    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).expect("operation should succeed");
    let ctx = std::sync::Arc::new(ctx);
    init_forward_kernel_cache(ctx.clone()).expect("operation should succeed");

    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    // 1 batch of 4 elements
    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let gamma: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
    let beta: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0];
    let batch_size = 1u32;
    let hidden_size = 4u32;
    let output_data = vec![0.0f32; (batch_size * hidden_size) as usize];

    let input = GpuBuffer::from_host(&ctx, &input_data).expect("operation should succeed");
    let gamma_buf = GpuBuffer::from_host(&ctx, &gamma).expect("operation should succeed");
    let beta_buf = GpuBuffer::from_host(&ctx, &beta).expect("operation should succeed");
    let mut output = GpuBuffer::from_host(&ctx, &output_data).expect("operation should succeed");

    layer_norm_forward(
        &input,
        &gamma_buf,
        &beta_buf,
        &mut output,
        batch_size,
        hidden_size,
        &stream,
    )
    .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; (batch_size * hidden_size) as usize];
    output.copy_to_host(&mut result).expect("operation should succeed");
    // Normalized mean should be ~0
    let mean: f32 = result.iter().sum::<f32>() / hidden_size as f32;
    assert!(mean.abs() < 1e-3, "Mean should be ~0, got {mean}");
    // Check values are normalized (not NaN/Inf)
    assert!(!result.iter().any(|x| x.is_nan()), "Output should not contain NaN");
}

#[test]
#[cfg(feature = "cuda")]
fn test_rms_norm_forward_basic() {
    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).expect("operation should succeed");
    let ctx = std::sync::Arc::new(ctx);
    init_forward_kernel_cache(ctx.clone()).expect("operation should succeed");

    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let weight: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
    let batch_size = 1u32;
    let hidden_size = 4u32;
    let output_data = vec![0.0f32; (batch_size * hidden_size) as usize];

    let input = GpuBuffer::from_host(&ctx, &input_data).expect("operation should succeed");
    let weight_buf = GpuBuffer::from_host(&ctx, &weight).expect("operation should succeed");
    let mut output = GpuBuffer::from_host(&ctx, &output_data).expect("operation should succeed");

    rms_norm_forward(&input, &weight_buf, &mut output, batch_size, hidden_size, &stream)
        .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; (batch_size * hidden_size) as usize];
    output.copy_to_host(&mut result).expect("operation should succeed");
    // RMS normalized values should be reasonable
    assert!(!result.iter().any(|x| x.is_nan()), "Output should not contain NaN");
    assert!(!result.iter().any(|x| x.is_infinite()), "Output should not contain Inf");
}

#[test]
#[cfg(feature = "cuda")]
fn test_layer_norm_forward_mutation_killing() {
    // Verify output is normalized (mean ≈ beta, std ≈ gamma)
    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).expect("operation should succeed");
    let ctx = std::sync::Arc::new(ctx);
    init_forward_kernel_cache(ctx.clone()).expect("operation should succeed");

    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    let batch_size = 1u32;
    let hidden_size = 4u32;
    // Input with different values
    let input_data: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0];
    let gamma_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
    let beta_data: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0];

    let input = GpuBuffer::from_host(&ctx, &input_data).expect("operation should succeed");
    let gamma = GpuBuffer::from_host(&ctx, &gamma_data).expect("operation should succeed");
    let beta = GpuBuffer::from_host(&ctx, &beta_data).expect("operation should succeed");
    let output_data = vec![0.0f32; (batch_size * hidden_size) as usize];
    let mut output = GpuBuffer::from_host(&ctx, &output_data).expect("operation should succeed");

    layer_norm_forward(&input, &gamma, &beta, &mut output, batch_size, hidden_size, &stream)
        .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; (batch_size * hidden_size) as usize];
    output.copy_to_host(&mut result).expect("operation should succeed");

    // After layer norm with gamma=1, beta=0, mean should be ~0
    let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
    assert!(mean.abs() < 0.1, "LayerNorm output mean should be ~0, got {mean}");
}

#[test]
#[cfg(feature = "cuda")]
fn test_rms_norm_forward_scaling() {
    // Verify RMS normalization scales output correctly
    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).expect("operation should succeed");
    let ctx = std::sync::Arc::new(ctx);
    init_forward_kernel_cache(ctx.clone()).expect("operation should succeed");

    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    let batch_size = 1u32;
    let hidden_size = 4u32;
    // Constant input
    let input_data: Vec<f32> = vec![2.0, 2.0, 2.0, 2.0];
    let gamma_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];

    let input = GpuBuffer::from_host(&ctx, &input_data).expect("operation should succeed");
    let gamma = GpuBuffer::from_host(&ctx, &gamma_data).expect("operation should succeed");
    let output_data = vec![0.0f32; (batch_size * hidden_size) as usize];
    let mut output = GpuBuffer::from_host(&ctx, &output_data).expect("operation should succeed");

    rms_norm_forward(&input, &gamma, &mut output, batch_size, hidden_size, &stream)
        .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; (batch_size * hidden_size) as usize];
    output.copy_to_host(&mut result).expect("operation should succeed");

    // For constant input x, RMS = x, so output = gamma * x / RMS = gamma = 1
    for (i, &r) in result.iter().enumerate() {
        assert!((r - 1.0).abs() < 0.1, "RMSNorm of constant input should give ~1, got {r} at {i}");
    }
}
