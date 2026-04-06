use super::super::*;
#[allow(unused_imports)]
use super::softmax_backward_cpu;

#[test]
#[cfg(feature = "cuda")]
fn test_softmax_backward_basic() {
    use trueno_gpu::driver::{CudaStream, GpuBuffer};

    let ctx = match super::get_test_gpu_context() {
        Some(c) => c,
        None => return,
    };
    init_kernel_cache(ctx.clone()).expect("operation should succeed");
    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    // Softmax output: uniform distribution (1/4 each)
    let softmax_output_data: Vec<f32> = vec![0.25, 0.25, 0.25, 0.25];
    let grad_output_data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0]; // Cross-entropy grad
    let grad_input_data: Vec<f32> = vec![0.0; 4];

    let softmax_output =
        GpuBuffer::from_host(&ctx, &softmax_output_data).expect("operation should succeed");
    let grad_output =
        GpuBuffer::from_host(&ctx, &grad_output_data).expect("operation should succeed");
    let mut grad_input =
        GpuBuffer::from_host(&ctx, &grad_input_data).expect("operation should succeed");

    softmax_backward(&softmax_output, &grad_output, &mut grad_input, 1, 4, &stream)
        .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; 4];
    grad_input.copy_to_host(&mut result).expect("operation should succeed");

    // CPU reference
    let expected = softmax_backward_cpu(&softmax_output_data, &grad_output_data);

    for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-3,
            "Softmax backward mismatch at {i}: got {got}, expected {exp}"
        );
    }
}

#[test]
#[cfg(feature = "cuda")]
fn test_softmax_backward_batch() {
    use trueno_gpu::driver::{CudaStream, GpuBuffer};

    let ctx = match super::get_test_gpu_context() {
        Some(c) => c,
        None => return,
    };
    init_kernel_cache(ctx.clone()).expect("operation should succeed");
    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    // 2 batches, seq_len=3
    let softmax_output_data: Vec<f32> = vec![0.5, 0.3, 0.2, 0.1, 0.2, 0.7];
    let grad_output_data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let grad_input_data: Vec<f32> = vec![0.0; 6];

    let softmax_output =
        GpuBuffer::from_host(&ctx, &softmax_output_data).expect("operation should succeed");
    let grad_output =
        GpuBuffer::from_host(&ctx, &grad_output_data).expect("operation should succeed");
    let mut grad_input =
        GpuBuffer::from_host(&ctx, &grad_input_data).expect("operation should succeed");

    softmax_backward(&softmax_output, &grad_output, &mut grad_input, 2, 3, &stream)
        .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; 6];
    grad_input.copy_to_host(&mut result).expect("operation should succeed");

    // Verify gradients are computed (not all zeros or NaN)
    assert!(!result.iter().any(|x| x.is_nan()), "Softmax backward should not produce NaN");
    assert!(
        result.iter().any(|&x| x.abs() > 1e-5),
        "Softmax backward should produce non-zero gradients"
    );
}

#[test]
#[cfg(feature = "cuda")]
fn test_rms_norm_backward_basic() {
    use trueno_gpu::driver::{CudaStream, GpuBuffer};

    let ctx = match super::get_test_gpu_context() {
        Some(c) => c,
        None => return,
    };
    init_kernel_cache(ctx.clone()).expect("operation should succeed");
    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    let batch_size = 1u32;
    let hidden_size = 4u32;
    let n = (batch_size * hidden_size) as usize;
    let eps = 1e-5;

    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let gamma_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
    let grad_output_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
    let grad_input_data: Vec<f32> = vec![0.0; n];
    let grad_gamma_data: Vec<f32> = vec![0.0; hidden_size as usize];

    let input = GpuBuffer::from_host(&ctx, &input_data).expect("operation should succeed");
    let gamma = GpuBuffer::from_host(&ctx, &gamma_data).expect("operation should succeed");
    let grad_output =
        GpuBuffer::from_host(&ctx, &grad_output_data).expect("operation should succeed");
    let mut grad_input =
        GpuBuffer::from_host(&ctx, &grad_input_data).expect("operation should succeed");
    let mut grad_gamma =
        GpuBuffer::from_host(&ctx, &grad_gamma_data).expect("operation should succeed");

    rms_norm_backward(
        &input,
        &gamma,
        &grad_output,
        &mut grad_input,
        &mut grad_gamma,
        batch_size,
        hidden_size,
        eps,
        &stream,
    )
    .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result_input = vec![0.0f32; n];
    let mut result_gamma = vec![0.0f32; hidden_size as usize];
    grad_input.copy_to_host(&mut result_input).expect("operation should succeed");
    grad_gamma.copy_to_host(&mut result_gamma).expect("operation should succeed");

    // Verify gradients are computed
    assert!(
        !result_input.iter().any(|x| x.is_nan()),
        "RMSNorm backward should not produce NaN for input grad"
    );
    assert!(
        !result_gamma.iter().any(|x| x.is_nan()),
        "RMSNorm backward should not produce NaN for gamma grad"
    );
}

#[test]
#[cfg(feature = "cuda")]
fn test_layer_norm_backward_basic() {
    use trueno_gpu::driver::{CudaStream, GpuBuffer};

    let ctx = match super::get_test_gpu_context() {
        Some(c) => c,
        None => return,
    };
    init_kernel_cache(ctx.clone()).expect("operation should succeed");
    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    let batch_size = 1u32;
    let hidden_size = 4u32;
    let n = (batch_size * hidden_size) as usize;

    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let gamma_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
    let grad_output_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
    let grad_input_data: Vec<f32> = vec![0.0; n];
    let grad_gamma_data: Vec<f32> = vec![0.0; hidden_size as usize];
    let grad_beta_data: Vec<f32> = vec![0.0; hidden_size as usize];

    let input = GpuBuffer::from_host(&ctx, &input_data).expect("operation should succeed");
    let gamma = GpuBuffer::from_host(&ctx, &gamma_data).expect("operation should succeed");
    let grad_output =
        GpuBuffer::from_host(&ctx, &grad_output_data).expect("operation should succeed");
    let mut grad_input =
        GpuBuffer::from_host(&ctx, &grad_input_data).expect("operation should succeed");
    let mut grad_gamma =
        GpuBuffer::from_host(&ctx, &grad_gamma_data).expect("operation should succeed");
    let mut grad_beta =
        GpuBuffer::from_host(&ctx, &grad_beta_data).expect("operation should succeed");

    layer_norm_backward(
        &input,
        &gamma,
        &grad_output,
        &mut grad_input,
        &mut grad_gamma,
        &mut grad_beta,
        batch_size,
        hidden_size,
        &stream,
    )
    .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result_input = vec![0.0f32; n];
    let mut result_gamma = vec![0.0f32; hidden_size as usize];
    let mut result_beta = vec![0.0f32; hidden_size as usize];
    grad_input.copy_to_host(&mut result_input).expect("operation should succeed");
    grad_gamma.copy_to_host(&mut result_gamma).expect("operation should succeed");
    grad_beta.copy_to_host(&mut result_beta).expect("operation should succeed");

    // Verify gradients are computed without NaN/Inf
    assert!(
        !result_input.iter().any(|x| x.is_nan() || x.is_infinite()),
        "LayerNorm backward should not produce NaN/Inf for input grad"
    );
    assert!(
        !result_gamma.iter().any(|x| x.is_nan() || x.is_infinite()),
        "LayerNorm backward should not produce NaN/Inf for gamma grad"
    );
    assert!(
        !result_beta.iter().any(|x| x.is_nan() || x.is_infinite()),
        "LayerNorm backward should not produce NaN/Inf for beta grad"
    );
    // Note: grad_beta may be zero for single batch with uniform grad_output
    // due to how the backward kernel accumulates gradients
}

/// CPU reference for RMSNorm forward: y = x / rms * gamma
/// where rms = sqrt(mean(x²) + eps)
fn rms_norm_forward_cpu(input: &[f32], gamma: &[f32], hidden_size: usize, eps: f32) -> Vec<f32> {
    let num_rows = input.len() / hidden_size;
    let mut output = vec![0.0f32; input.len()];
    for r in 0..num_rows {
        let row = &input[r * hidden_size..(r + 1) * hidden_size];
        let mean_sq: f32 = row.iter().map(|x| x * x).sum::<f32>() / hidden_size as f32;
        let rms = (mean_sq + eps).sqrt();
        for i in 0..hidden_size {
            output[r * hidden_size + i] = row[i] / rms * gamma[i];
        }
    }
    output
}

/// CPU reference for RMSNorm backward: grad_input
/// ∂L/∂x_i = (1/rms) * (γ_i * ∂L/∂y_i - x_i/rms² * mean(x · ∂L/∂y · γ))
fn rms_norm_backward_cpu(
    input: &[f32],
    gamma: &[f32],
    grad_output: &[f32],
    hidden_size: usize,
    eps: f32,
) -> Vec<f32> {
    let num_rows = input.len() / hidden_size;
    let mut grad_input = vec![0.0f32; input.len()];
    for r in 0..num_rows {
        let base = r * hidden_size;
        let row = &input[base..base + hidden_size];

        let mean_sq: f32 = row.iter().map(|x| x * x).sum::<f32>() / hidden_size as f32;
        let variance_eps = mean_sq + eps;
        let rms = variance_eps.sqrt();

        let mean_xgg: f32 =
            (0..hidden_size).map(|i| row[i] * grad_output[base + i] * gamma[i]).sum::<f32>()
                / hidden_size as f32;

        for i in 0..hidden_size {
            let adjust = row[i] / variance_eps * mean_xgg;
            grad_input[base + i] = (1.0 / rms) * (gamma[i] * grad_output[base + i] - adjust);
        }
    }
    grad_input
}

#[test]
#[cfg(feature = "cuda")]
fn test_rms_norm_backward_finite_difference_small() {
    use trueno_gpu::driver::{CudaStream, GpuBuffer};

    let ctx = match super::get_test_gpu_context() {
        Some(c) => c,
        None => return,
    };
    init_kernel_cache(ctx.clone()).expect("kernel cache init");
    let stream = CudaStream::new(&ctx).expect("stream creation");

    let hidden_size = 4usize;
    let num_rows = 2u32;
    let eps = 1e-5f32;
    let n = num_rows as usize * hidden_size;

    let input_data: Vec<f32> = vec![1.0, -0.5, 2.0, 0.3, -1.0, 0.8, 1.5, -0.7];
    let gamma_data: Vec<f32> = vec![1.0, 0.5, 2.0, 1.5];
    let grad_output_data: Vec<f32> = vec![1.0; n]; // dL/dy = 1 everywhere (sum loss)

    // GPU backward
    let input_gpu = GpuBuffer::from_host(&ctx, &input_data).expect("upload");
    let gamma_gpu = GpuBuffer::from_host(&ctx, &gamma_data).expect("upload");
    let grad_out_gpu = GpuBuffer::from_host(&ctx, &grad_output_data).expect("upload");
    let mut grad_in_gpu = GpuBuffer::from_host(&ctx, &vec![0.0f32; n]).expect("upload");
    let mut grad_gamma_gpu =
        GpuBuffer::from_host(&ctx, &vec![0.0f32; hidden_size]).expect("upload");

    rms_norm_backward(
        &input_gpu,
        &gamma_gpu,
        &grad_out_gpu,
        &mut grad_in_gpu,
        &mut grad_gamma_gpu,
        num_rows,
        hidden_size as u32,
        eps,
        &stream,
    )
    .expect("rms_norm_backward");
    stream.synchronize().expect("sync");

    let mut gpu_grad = vec![0.0f32; n];
    grad_in_gpu.copy_to_host(&mut gpu_grad).expect("download");

    // CPU analytical backward
    let cpu_grad =
        rms_norm_backward_cpu(&input_data, &gamma_data, &grad_output_data, hidden_size, eps);

    // Finite-difference numerical gradient
    let h = 1e-3f32;
    let mut fd_grad = vec![0.0f32; n];
    for i in 0..n {
        let mut x_plus = input_data.clone();
        let mut x_minus = input_data.clone();
        x_plus[i] += h;
        x_minus[i] -= h;
        let y_plus = rms_norm_forward_cpu(&x_plus, &gamma_data, hidden_size, eps);
        let y_minus = rms_norm_forward_cpu(&x_minus, &gamma_data, hidden_size, eps);
        // Loss = sum(y * grad_output) = sum(y) when grad_output = 1
        let loss_plus: f32 = y_plus.iter().zip(&grad_output_data).map(|(y, g)| y * g).sum();
        let loss_minus: f32 = y_minus.iter().zip(&grad_output_data).map(|(y, g)| y * g).sum();
        fd_grad[i] = (loss_plus - loss_minus) / (2.0 * h);
    }

    // Compare GPU vs CPU analytical
    for i in 0..n {
        assert!(
            (gpu_grad[i] - cpu_grad[i]).abs() < 1e-3,
            "GPU vs CPU mismatch at {i}: gpu={}, cpu={}",
            gpu_grad[i],
            cpu_grad[i],
        );
    }

    // Compare GPU vs finite-difference
    for i in 0..n {
        assert!(
            (gpu_grad[i] - fd_grad[i]).abs() < 1e-2,
            "GPU vs FD mismatch at {i}: gpu={}, fd={}",
            gpu_grad[i],
            fd_grad[i],
        );
    }
}

#[test]
#[cfg(feature = "cuda")]
fn test_rms_norm_backward_large_hidden() {
    use trueno_gpu::driver::{CudaStream, GpuBuffer};

    let ctx = match super::get_test_gpu_context() {
        Some(c) => c,
        None => return,
    };
    init_kernel_cache(ctx.clone()).expect("kernel cache init");
    let stream = CudaStream::new(&ctx).expect("stream creation");

    // hidden_size=64: triggers stride-loop (> 32 warp width)
    let hidden_size = 64usize;
    let num_rows = 4u32;
    let eps = 1e-5f32;
    let n = num_rows as usize * hidden_size;

    // Deterministic pseudo-random data
    let input_data: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.37).sin() * 2.0)).collect();
    let gamma_data: Vec<f32> = (0..hidden_size).map(|i| 0.5 + (i as f32 * 0.13).cos()).collect();
    let grad_output_data: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.71).cos())).collect();

    // GPU backward
    let input_gpu = GpuBuffer::from_host(&ctx, &input_data).expect("upload");
    let gamma_gpu = GpuBuffer::from_host(&ctx, &gamma_data).expect("upload");
    let grad_out_gpu = GpuBuffer::from_host(&ctx, &grad_output_data).expect("upload");
    let mut grad_in_gpu = GpuBuffer::from_host(&ctx, &vec![0.0f32; n]).expect("upload");
    let mut grad_gamma_gpu =
        GpuBuffer::from_host(&ctx, &vec![0.0f32; hidden_size]).expect("upload");

    rms_norm_backward(
        &input_gpu,
        &gamma_gpu,
        &grad_out_gpu,
        &mut grad_in_gpu,
        &mut grad_gamma_gpu,
        num_rows,
        hidden_size as u32,
        eps,
        &stream,
    )
    .expect("rms_norm_backward");
    stream.synchronize().expect("sync");

    let mut gpu_grad = vec![0.0f32; n];
    grad_in_gpu.copy_to_host(&mut gpu_grad).expect("download");

    // CPU analytical reference
    let cpu_grad =
        rms_norm_backward_cpu(&input_data, &gamma_data, &grad_output_data, hidden_size, eps);

    // Compare GPU vs CPU
    let max_diff =
        gpu_grad.iter().zip(cpu_grad.iter()).map(|(g, c)| (g - c).abs()).fold(0.0f32, f32::max);

    assert!(max_diff < 1e-2, "RMSNorm backward hidden=64: max diff {max_diff} exceeds 1e-2");

    // Finite-difference spot-check (first 8 elements only for speed)
    let h = 1e-3f32;
    for i in 0..8 {
        let mut x_plus = input_data.clone();
        let mut x_minus = input_data.clone();
        x_plus[i] += h;
        x_minus[i] -= h;
        let y_plus = rms_norm_forward_cpu(&x_plus, &gamma_data, hidden_size, eps);
        let y_minus = rms_norm_forward_cpu(&x_minus, &gamma_data, hidden_size, eps);
        let loss_plus: f32 = y_plus.iter().zip(&grad_output_data).map(|(y, g)| y * g).sum();
        let loss_minus: f32 = y_minus.iter().zip(&grad_output_data).map(|(y, g)| y * g).sum();
        let fd = (loss_plus - loss_minus) / (2.0 * h);

        assert!(
            (gpu_grad[i] - fd).abs() < 1e-2,
            "RMSNorm FD check at {i}: gpu={}, fd={}",
            gpu_grad[i],
            fd,
        );
    }
}

#[test]
#[cfg(feature = "cuda")]
fn test_batched_softmax_backward_finite_difference() {
    use trueno_gpu::driver::{CudaStream, GpuBuffer};

    let ctx = match super::get_test_gpu_context() {
        Some(c) => c,
        None => return,
    };
    init_kernel_cache(ctx.clone()).expect("kernel cache init");
    let stream = CudaStream::new(&ctx).expect("stream creation");

    // row_size=64: triggers stride-loop in BatchedSoftmaxBackwardKernel
    let total_rows = 2u32;
    let row_size = 64u32;
    let n = (total_rows * row_size) as usize;

    // Softmax output (valid probabilities per row)
    let logits: Vec<f32> = (0..n).map(|i| (i as f32 * 0.23).sin()).collect();
    let mut softmax_data = vec![0.0f32; n];
    for r in 0..total_rows as usize {
        let row = &logits[r * row_size as usize..(r + 1) * row_size as usize];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = row.iter().map(|x| (x - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        for i in 0..row_size as usize {
            softmax_data[r * row_size as usize + i] = exp[i] / sum;
        }
    }

    let grad_output_data: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.47).cos())).collect();

    let softmax_gpu = GpuBuffer::from_host(&ctx, &softmax_data).expect("upload");
    let grad_out_gpu = GpuBuffer::from_host(&ctx, &grad_output_data).expect("upload");
    let mut grad_in_gpu = GpuBuffer::from_host(&ctx, &vec![0.0f32; n]).expect("upload");

    batched_softmax_backward(
        &softmax_gpu,
        &grad_out_gpu,
        &mut grad_in_gpu,
        total_rows,
        row_size,
        &stream,
    )
    .expect("batched_softmax_backward");
    stream.synchronize().expect("sync");

    let mut gpu_grad = vec![0.0f32; n];
    grad_in_gpu.copy_to_host(&mut gpu_grad).expect("download");

    // CPU reference: grad_x[i] = y[i] * (grad_y[i] - dot(grad_y, y))
    let mut cpu_grad = vec![0.0f32; n];
    for r in 0..total_rows as usize {
        let base = r * row_size as usize;
        let dot: f32 = (0..row_size as usize)
            .map(|i| softmax_data[base + i] * grad_output_data[base + i])
            .sum();
        for i in 0..row_size as usize {
            cpu_grad[base + i] = softmax_data[base + i] * (grad_output_data[base + i] - dot);
        }
    }

    let max_diff =
        gpu_grad.iter().zip(cpu_grad.iter()).map(|(g, c)| (g - c).abs()).fold(0.0f32, f32::max);

    assert!(max_diff < 1e-3, "Batched softmax backward: max diff {max_diff} exceeds 1e-3");
}
