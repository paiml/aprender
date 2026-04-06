use super::*;

mod elementwise;
mod gemm;
mod structured;

#[test]
fn test_cuda_backward_module_compiles() {
    // This test verifies the module compiles correctly
    // Actual CUDA tests require GPU hardware
    assert!(true);
}

#[test]
#[cfg(feature = "cuda")]
fn test_kernel_cache_initialization() {
    use std::sync::Arc;
    use trueno_gpu::driver::{cuda_available, CudaContext};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).expect("operation should succeed");
    let ctx = Arc::new(ctx);
    let result = init_kernel_cache(ctx);
    assert!(result.is_ok());
}

/// Create a fresh GPU context for a test
/// Note: Using fresh contexts per-test avoids CUDA driver state issues
/// when running multiple tests sequentially
#[cfg(feature = "cuda")]
pub(super) fn get_test_gpu_context() -> Option<std::sync::Arc<trueno_gpu::driver::CudaContext>> {
    use trueno_gpu::driver::cuda_available;

    if cuda_available() {
        trueno_gpu::driver::CudaContext::new(0).ok().map(std::sync::Arc::new)
    } else {
        None
    }
}

/// CPU reference implementation for ReLU backward
pub(super) fn relu_backward_cpu(input: &[f32], grad_output: &[f32]) -> Vec<f32> {
    input.iter().zip(grad_output.iter()).map(|(&x, &dy)| if x > 0.0 { dy } else { 0.0 }).collect()
}

/// CPU reference implementation for GELU backward (tanh approximation)
pub(super) fn gelu_backward_cpu(input: &[f32], grad_output: &[f32]) -> Vec<f32> {
    let sqrt_2_pi = (2.0 / std::f32::consts::PI).sqrt();
    input
        .iter()
        .zip(grad_output.iter())
        .map(|(&x, &dy)| {
            let inner = sqrt_2_pi * (x + 0.044715 * x.powi(3));
            let tanh_inner = inner.tanh();
            let sech2 = 1.0 - tanh_inner.powi(2);
            let gelu_deriv = 0.5 * (1.0 + tanh_inner)
                + 0.5 * x * sech2 * sqrt_2_pi * (1.0 + 3.0 * 0.044715 * x.powi(2));
            dy * gelu_deriv
        })
        .collect()
}

/// CPU reference implementation for SiLU backward
pub(super) fn silu_backward_cpu(input: &[f32], grad_output: &[f32]) -> Vec<f32> {
    input
        .iter()
        .zip(grad_output.iter())
        .map(|(&x, &dy)| {
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            let silu_deriv = sigmoid + x * sigmoid * (1.0 - sigmoid);
            dy * silu_deriv
        })
        .collect()
}

/// CPU reference implementation for softmax backward
pub(super) fn softmax_backward_cpu(softmax_output: &[f32], grad_output: &[f32]) -> Vec<f32> {
    // grad_input = softmax_output * (grad_output - sum(grad_output * softmax_output))
    let dot: f32 = softmax_output.iter().zip(grad_output.iter()).map(|(s, g)| s * g).sum();
    softmax_output.iter().zip(grad_output.iter()).map(|(&s, &g)| s * (g - dot)).collect()
}
