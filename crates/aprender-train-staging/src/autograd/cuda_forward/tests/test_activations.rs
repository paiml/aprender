use super::*;

#[test]
#[cfg(feature = "cuda")]
fn test_relu_forward_basic() {
    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).expect("operation should succeed");
    let ctx = std::sync::Arc::new(ctx);
    init_forward_kernel_cache(ctx.clone()).expect("operation should succeed");

    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    // Input: [-2, -1, 0, 1, 2]
    let input_data: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let n = input_data.len() as u32;

    let input = GpuBuffer::from_host(&ctx, &input_data).expect("operation should succeed");
    let output_data = vec![0.0f32; n as usize];
    let mut output = GpuBuffer::from_host(&ctx, &output_data).expect("operation should succeed");

    relu_forward(&input, &mut output, n, &stream).expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; n as usize];
    output.copy_to_host(&mut result).expect("operation should succeed");
    // ReLU: max(0, x)
    assert_eq!(result, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
}

#[test]
#[cfg(feature = "cuda")]
fn test_softmax_forward_basic() {
    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).expect("operation should succeed");
    let ctx = std::sync::Arc::new(ctx);
    init_forward_kernel_cache(ctx.clone()).expect("operation should succeed");

    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    // Simple 4-element softmax (needs to be power of 2 for warp operations)
    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let n = input_data.len() as u32;

    let input = GpuBuffer::from_host(&ctx, &input_data).expect("operation should succeed");
    let output_data = vec![0.0f32; n as usize];
    let mut output = GpuBuffer::from_host(&ctx, &output_data).expect("operation should succeed");

    softmax_forward(&input, &mut output, n, &stream).expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; n as usize];
    output.copy_to_host(&mut result).expect("operation should succeed");
    // Softmax sums to 1
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4, "Softmax should sum to 1, got {sum}");
    // Last element should be largest (exp(4) > exp(3) > ...)
    assert!(result[3] > result[2] && result[2] > result[1] && result[1] > result[0]);
}

#[test]
#[cfg(feature = "cuda")]
fn test_gelu_forward_basic() {
    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).expect("operation should succeed");
    let ctx = std::sync::Arc::new(ctx);
    init_forward_kernel_cache(ctx.clone()).expect("operation should succeed");

    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    let input_data: Vec<f32> = vec![-1.0, 0.0, 1.0, 2.0];
    let n = input_data.len() as u32;

    let input = GpuBuffer::from_host(&ctx, &input_data).expect("operation should succeed");
    let output_data = vec![0.0f32; n as usize];
    let mut output = GpuBuffer::from_host(&ctx, &output_data).expect("operation should succeed");

    gelu_forward(&input, &mut output, n, &stream).expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; n as usize];
    output.copy_to_host(&mut result).expect("operation should succeed");
    // GELU(0) = 0
    assert!((result[1]).abs() < 1e-4, "GELU(0) should be 0, got {}", result[1]);
    // GELU(-1) ≈ -0.159
    assert!(result[0] < 0.0 && result[0] > -0.2, "GELU(-1) should be ~-0.159, got {}", result[0]);
    // GELU(1) ≈ 0.841
    assert!(result[2] > 0.8 && result[2] < 0.9, "GELU(1) should be ~0.841, got {}", result[2]);
}

#[test]
#[cfg(feature = "cuda")]
fn test_silu_forward_basic() {
    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).expect("operation should succeed");
    let ctx = std::sync::Arc::new(ctx);
    init_forward_kernel_cache(ctx.clone()).expect("operation should succeed");

    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    let input_data: Vec<f32> = vec![-1.0, 0.0, 1.0, 2.0];
    let n = input_data.len() as u32;

    let input = GpuBuffer::from_host(&ctx, &input_data).expect("operation should succeed");
    let output_data = vec![0.0f32; n as usize];
    let mut output = GpuBuffer::from_host(&ctx, &output_data).expect("operation should succeed");

    silu_forward(&input, &mut output, n, &stream).expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; n as usize];
    output.copy_to_host(&mut result).expect("operation should succeed");
    // SiLU(0) = 0
    assert!((result[1]).abs() < 1e-4, "SiLU(0) should be 0, got {}", result[1]);
    // SiLU(x) = x * sigmoid(x)
    // SiLU(-1) ≈ -0.269
    assert!(result[0] < 0.0 && result[0] > -0.3, "SiLU(-1) should be ~-0.269, got {}", result[0]);
    // SiLU(1) ≈ 0.731
    assert!(result[2] > 0.7 && result[2] < 0.8, "SiLU(1) should be ~0.731, got {}", result[2]);
}

#[test]
#[cfg(feature = "cuda")]
fn test_relu_forward_mutation_killing() {
    // Mutation-killing: verify ReLU doesn't just return zeros or input
    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).expect("operation should succeed");
    let ctx = std::sync::Arc::new(ctx);
    init_forward_kernel_cache(ctx.clone()).expect("operation should succeed");

    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0];
    let n = input_data.len() as u32;

    let input = GpuBuffer::from_host(&ctx, &input_data).expect("operation should succeed");
    let output_data = vec![0.0f32; n as usize];
    let mut output = GpuBuffer::from_host(&ctx, &output_data).expect("operation should succeed");

    relu_forward(&input, &mut output, n, &stream).expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; n as usize];
    output.copy_to_host(&mut result).expect("operation should succeed");

    // Result should equal input for positive values
    assert_eq!(result, input_data, "ReLU(positive) should equal input");
}

#[test]
#[cfg(feature = "cuda")]
fn test_gelu_forward_mutation_killing() {
    // Mutation-killing: verify GELU is not identity or ReLU
    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).expect("operation should succeed");
    let ctx = std::sync::Arc::new(ctx);
    init_forward_kernel_cache(ctx.clone()).expect("operation should succeed");

    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    let input_data: Vec<f32> = vec![1.0, 2.0];
    let n = input_data.len() as u32;

    let input = GpuBuffer::from_host(&ctx, &input_data).expect("operation should succeed");
    let output_data = vec![0.0f32; n as usize];
    let mut output = GpuBuffer::from_host(&ctx, &output_data).expect("operation should succeed");

    gelu_forward(&input, &mut output, n, &stream).expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; n as usize];
    output.copy_to_host(&mut result).expect("operation should succeed");

    // GELU(x) ≈ x for large positive x, but not exactly
    // GELU(1) ≈ 0.841, not 1.0
    assert!((result[0] - 0.841).abs() < 0.01, "GELU(1) should be ~0.841, got {}", result[0]);
}

#[test]
#[cfg(feature = "cuda")]
fn test_silu_forward_mutation_killing() {
    // Mutation-killing: verify SiLU is not identity
    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).expect("operation should succeed");
    let ctx = std::sync::Arc::new(ctx);
    init_forward_kernel_cache(ctx.clone()).expect("operation should succeed");

    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    let input_data: Vec<f32> = vec![1.0, 2.0];
    let n = input_data.len() as u32;

    let input = GpuBuffer::from_host(&ctx, &input_data).expect("operation should succeed");
    let output_data = vec![0.0f32; n as usize];
    let mut output = GpuBuffer::from_host(&ctx, &output_data).expect("operation should succeed");

    silu_forward(&input, &mut output, n, &stream).expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; n as usize];
    output.copy_to_host(&mut result).expect("operation should succeed");

    // SiLU(x) = x * sigmoid(x)
    // SiLU(1) = 1 * sigmoid(1) ≈ 1 * 0.731 ≈ 0.731
    assert!((result[0] - 0.731).abs() < 0.01, "SiLU(1) should be ~0.731, got {}", result[0]);
}
