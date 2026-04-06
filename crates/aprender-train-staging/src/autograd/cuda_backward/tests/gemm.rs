use super::super::*;

#[test]
#[cfg(feature = "cuda")]
fn test_gemm_backward_a_basic() {
    use trueno_gpu::driver::{CudaStream, GpuBuffer};

    let ctx = match super::get_test_gpu_context() {
        Some(c) => c,
        None => return,
    };
    init_kernel_cache(ctx.clone()).expect("operation should succeed");
    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    // C = A @ B where A is 2x2, B is 2x2
    // grad_A = grad_C @ B^T
    let m = 2u32;
    let k = 2u32;
    let n = 2u32;

    // grad_C = [[1, 0], [0, 1]] (identity-like)
    let grad_output_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
    // B = [[1, 2], [3, 4]]
    // B^T = [[1, 3], [2, 4]]
    let b_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let grad_a_data: Vec<f32> = vec![0.0; (m * k) as usize];

    let grad_output =
        GpuBuffer::from_host(&ctx, &grad_output_data).expect("operation should succeed");
    let b = GpuBuffer::from_host(&ctx, &b_data).expect("operation should succeed");
    let mut grad_a = GpuBuffer::from_host(&ctx, &grad_a_data).expect("operation should succeed");

    gemm_backward_a(&grad_output, &b, &mut grad_a, m, k, n, &stream)
        .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; (m * k) as usize];
    grad_a.copy_to_host(&mut result).expect("operation should succeed");

    // Verify gradients are computed and not NaN
    assert!(!result.iter().any(|x| x.is_nan()), "GEMM backward A should not produce NaN");
    // grad_A = [[1, 0], [0, 1]] @ [[1, 3], [2, 4]] = [[1, 3], [2, 4]]
    // Expected: [1, 3, 2, 4]
    let expected = vec![1.0, 3.0, 2.0, 4.0];
    for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-3,
            "GEMM backward A mismatch at {i}: got {got}, expected {exp}"
        );
    }
}

#[test]
#[cfg(feature = "cuda")]
fn test_gemm_backward_b_basic() {
    use trueno_gpu::driver::{CudaStream, GpuBuffer};

    let ctx = match super::get_test_gpu_context() {
        Some(c) => c,
        None => return,
    };
    init_kernel_cache(ctx.clone()).expect("operation should succeed");
    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    // C = A @ B where A is 2x2, B is 2x2
    // grad_B = A^T @ grad_C
    let m = 2u32;
    let k = 2u32;
    let n = 2u32;

    // A = [[1, 2], [3, 4]]
    // A^T = [[1, 3], [2, 4]]
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    // grad_C = [[1, 0], [0, 1]] (identity-like)
    let grad_output_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
    let grad_b_data: Vec<f32> = vec![0.0; (k * n) as usize];

    let a = GpuBuffer::from_host(&ctx, &a_data).expect("operation should succeed");
    let grad_output =
        GpuBuffer::from_host(&ctx, &grad_output_data).expect("operation should succeed");
    let mut grad_b = GpuBuffer::from_host(&ctx, &grad_b_data).expect("operation should succeed");

    gemm_backward_b(&a, &grad_output, &mut grad_b, m, k, n, &stream)
        .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; (k * n) as usize];
    grad_b.copy_to_host(&mut result).expect("operation should succeed");

    // Verify gradients are computed and not NaN
    assert!(!result.iter().any(|x| x.is_nan()), "GEMM backward B should not produce NaN");
    // grad_B = [[1, 3], [2, 4]] @ [[1, 0], [0, 1]] = [[1, 3], [2, 4]]
    // Expected: [1, 3, 2, 4]
    let expected = vec![1.0, 3.0, 2.0, 4.0];
    for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-3,
            "GEMM backward B mismatch at {i}: got {got}, expected {exp}"
        );
    }
}

#[test]
#[cfg(feature = "cuda")]
fn test_gemm_backward_larger_matrices() {
    use trueno_gpu::driver::{CudaStream, GpuBuffer};

    let ctx = match super::get_test_gpu_context() {
        Some(c) => c,
        None => return,
    };
    init_kernel_cache(ctx.clone()).expect("operation should succeed");
    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    // Test with larger matrices to exercise block tiling
    let m = 32u32;
    let k = 32u32;
    let n = 32u32;

    let grad_output_data: Vec<f32> = (0..(m * n)).map(|i| (i as f32) * 0.01).collect();
    let b_data: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.01).collect();
    let a_data: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.01).collect();
    let grad_a_data: Vec<f32> = vec![0.0; (m * k) as usize];
    let grad_b_data: Vec<f32> = vec![0.0; (k * n) as usize];

    let grad_output =
        GpuBuffer::from_host(&ctx, &grad_output_data).expect("operation should succeed");
    let b = GpuBuffer::from_host(&ctx, &b_data).expect("operation should succeed");
    let a = GpuBuffer::from_host(&ctx, &a_data).expect("operation should succeed");
    let mut grad_a = GpuBuffer::from_host(&ctx, &grad_a_data).expect("operation should succeed");
    let mut grad_b = GpuBuffer::from_host(&ctx, &grad_b_data).expect("operation should succeed");

    gemm_backward_a(&grad_output, &b, &mut grad_a, m, k, n, &stream)
        .expect("operation should succeed");
    gemm_backward_b(&a, &grad_output, &mut grad_b, m, k, n, &stream)
        .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result_a = vec![0.0f32; (m * k) as usize];
    let mut result_b = vec![0.0f32; (k * n) as usize];
    grad_a.copy_to_host(&mut result_a).expect("operation should succeed");
    grad_b.copy_to_host(&mut result_b).expect("operation should succeed");

    // Verify no NaN or Inf
    assert!(
        !result_a.iter().any(|x| x.is_nan() || x.is_infinite()),
        "GEMM backward A should not produce NaN/Inf"
    );
    assert!(
        !result_b.iter().any(|x| x.is_nan() || x.is_infinite()),
        "GEMM backward B should not produce NaN/Inf"
    );
    // Verify non-zero gradients
    assert!(
        result_a.iter().any(|&x| x.abs() > 1e-5),
        "GEMM backward A should produce non-zero gradients"
    );
    assert!(
        result_b.iter().any(|&x| x.abs() > 1e-5),
        "GEMM backward B should produce non-zero gradients"
    );
}
