use super::super::*;
use super::{gelu_backward_cpu, relu_backward_cpu, silu_backward_cpu};

#[test]
#[cfg(feature = "cuda")]
fn test_relu_backward_basic() {
    use trueno_gpu::driver::{CudaStream, GpuBuffer};

    let ctx = match super::get_test_gpu_context() {
        Some(c) => c,
        None => return,
    };
    init_kernel_cache(ctx.clone()).expect("operation should succeed");
    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    // Test data: mix of positive and negative values
    let input_data: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let grad_output_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let grad_input_data: Vec<f32> = vec![0.0; 6];

    let input = GpuBuffer::from_host(&ctx, &input_data).expect("operation should succeed");
    let grad_output =
        GpuBuffer::from_host(&ctx, &grad_output_data).expect("operation should succeed");
    let mut grad_input =
        GpuBuffer::from_host(&ctx, &grad_input_data).expect("operation should succeed");

    relu_backward(&input, &grad_output, &mut grad_input, &stream)
        .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; 6];
    grad_input.copy_to_host(&mut result).expect("operation should succeed");

    // CPU reference
    let expected = relu_backward_cpu(&input_data, &grad_output_data);

    for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "ReLU backward mismatch at {i}: got {got}, expected {exp}"
        );
    }
}

#[test]
#[cfg(feature = "cuda")]
fn test_relu_backward_not_hardcoded() {
    use trueno_gpu::driver::{CudaStream, GpuBuffer};

    // Mutation-killing test: verify result is NOT all zeros
    let ctx = match super::get_test_gpu_context() {
        Some(c) => c,
        None => return,
    };
    init_kernel_cache(ctx.clone()).expect("operation should succeed");
    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0]; // All positive
    let grad_output_data: Vec<f32> = vec![1.0, 2.0, 3.0];
    let grad_input_data: Vec<f32> = vec![0.0; 3];

    let input = GpuBuffer::from_host(&ctx, &input_data).expect("operation should succeed");
    let grad_output =
        GpuBuffer::from_host(&ctx, &grad_output_data).expect("operation should succeed");
    let mut grad_input =
        GpuBuffer::from_host(&ctx, &grad_input_data).expect("operation should succeed");

    relu_backward(&input, &grad_output, &mut grad_input, &stream)
        .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; 3];
    grad_input.copy_to_host(&mut result).expect("operation should succeed");

    // Kill mutant: result should NOT be all zeros for positive inputs
    assert_ne!(result, vec![0.0, 0.0, 0.0], "mutant: ReLU backward returned all zeros");
    // Verify correct values
    assert!((result[0] - 1.0).abs() < 1e-5);
    assert!((result[1] - 2.0).abs() < 1e-5);
    assert!((result[2] - 3.0).abs() < 1e-5);
}

#[test]
#[cfg(feature = "cuda")]
fn test_gelu_backward_basic() {
    use trueno_gpu::driver::{CudaStream, GpuBuffer};

    let ctx = match super::get_test_gpu_context() {
        Some(c) => c,
        None => return,
    };
    init_kernel_cache(ctx.clone()).expect("operation should succeed");
    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    let input_data: Vec<f32> = vec![-1.0, 0.0, 1.0, 2.0];
    let grad_output_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
    let grad_input_data: Vec<f32> = vec![0.0; 4];

    let input = GpuBuffer::from_host(&ctx, &input_data).expect("operation should succeed");
    let grad_output =
        GpuBuffer::from_host(&ctx, &grad_output_data).expect("operation should succeed");
    let mut grad_input =
        GpuBuffer::from_host(&ctx, &grad_input_data).expect("operation should succeed");

    gelu_backward(&input, &grad_output, &mut grad_input, &stream)
        .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; 4];
    grad_input.copy_to_host(&mut result).expect("operation should succeed");

    // CPU reference
    let expected = gelu_backward_cpu(&input_data, &grad_output_data);

    for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-3,
            "GELU backward mismatch at {i}: got {got}, expected {exp}"
        );
    }
}

#[test]
#[cfg(feature = "cuda")]
fn test_gelu_backward_not_hardcoded() {
    use trueno_gpu::driver::{CudaStream, GpuBuffer};

    // Mutation-killing test
    let ctx = match super::get_test_gpu_context() {
        Some(c) => c,
        None => return,
    };
    init_kernel_cache(ctx.clone()).expect("operation should succeed");
    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    let input_data: Vec<f32> = vec![0.5, 1.0, 1.5];
    let grad_output_data: Vec<f32> = vec![1.0, 1.0, 1.0];
    let grad_input_data: Vec<f32> = vec![0.0; 3];

    let input = GpuBuffer::from_host(&ctx, &input_data).expect("operation should succeed");
    let grad_output =
        GpuBuffer::from_host(&ctx, &grad_output_data).expect("operation should succeed");
    let mut grad_input =
        GpuBuffer::from_host(&ctx, &grad_input_data).expect("operation should succeed");

    gelu_backward(&input, &grad_output, &mut grad_input, &stream)
        .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; 3];
    grad_input.copy_to_host(&mut result).expect("operation should succeed");

    // Kill mutant: verify results are different for different inputs
    assert!(
        (result[0] - result[1]).abs() > 1e-5 || (result[1] - result[2]).abs() > 1e-5,
        "mutant: GELU backward returned identical values"
    );
    // All values should be positive for positive inputs
    assert!(result.iter().all(|&x| x > 0.0), "GELU gradient should be positive for x > 0");
}

#[test]
#[cfg(feature = "cuda")]
fn test_silu_backward_basic() {
    use trueno_gpu::driver::{CudaStream, GpuBuffer};

    let ctx = match super::get_test_gpu_context() {
        Some(c) => c,
        None => return,
    };
    init_kernel_cache(ctx.clone()).expect("operation should succeed");
    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    let input_data: Vec<f32> = vec![-1.0, 0.0, 1.0, 2.0];
    let grad_output_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
    let grad_input_data: Vec<f32> = vec![0.0; 4];

    let input = GpuBuffer::from_host(&ctx, &input_data).expect("operation should succeed");
    let grad_output =
        GpuBuffer::from_host(&ctx, &grad_output_data).expect("operation should succeed");
    let mut grad_input =
        GpuBuffer::from_host(&ctx, &grad_input_data).expect("operation should succeed");

    silu_backward(&input, &grad_output, &mut grad_input, &stream)
        .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; 4];
    grad_input.copy_to_host(&mut result).expect("operation should succeed");

    // CPU reference
    let expected = silu_backward_cpu(&input_data, &grad_output_data);

    for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-3,
            "SiLU backward mismatch at {i}: got {got}, expected {exp}"
        );
    }
}

#[test]
#[cfg(feature = "cuda")]
fn test_silu_backward_not_hardcoded() {
    use trueno_gpu::driver::{CudaStream, GpuBuffer};

    // Mutation-killing test
    let ctx = match super::get_test_gpu_context() {
        Some(c) => c,
        None => return,
    };
    init_kernel_cache(ctx.clone()).expect("operation should succeed");
    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    let input_data: Vec<f32> = vec![0.5, 1.0, 2.0];
    let grad_output_data: Vec<f32> = vec![1.0, 1.0, 1.0];
    let grad_input_data: Vec<f32> = vec![0.0; 3];

    let input = GpuBuffer::from_host(&ctx, &input_data).expect("operation should succeed");
    let grad_output =
        GpuBuffer::from_host(&ctx, &grad_output_data).expect("operation should succeed");
    let mut grad_input =
        GpuBuffer::from_host(&ctx, &grad_input_data).expect("operation should succeed");

    silu_backward(&input, &grad_output, &mut grad_input, &stream)
        .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; 3];
    grad_input.copy_to_host(&mut result).expect("operation should succeed");

    // Kill mutant: verify gradient at x=0 is ~0.5 (sigma(0) = 0.5)
    let grad_at_zero = silu_backward_cpu(&[0.0], &[1.0])[0];
    assert!((grad_at_zero - 0.5).abs() < 1e-3, "SiLU gradient at 0 should be ~0.5");

    // All gradients should be positive for positive inputs
    assert!(result.iter().all(|&x| x > 0.0), "SiLU gradient should be positive for x > 0");
}
