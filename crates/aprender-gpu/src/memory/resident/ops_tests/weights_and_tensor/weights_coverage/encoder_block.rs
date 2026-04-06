//! Forward Encoder Block tests: output shape, n_heads, varied input, field sizes

use super::*;

use crate::memory::resident::{
    clear_kernel_cache, forward_encoder_block_gpu, GpuEncoderBlockWeights, GpuEncoderConfig,
};

/// Build uniform `GpuEncoderBlockWeights` where every weight matrix element
/// is `weight_val` and every bias element is `bias_val`.
fn build_uniform_weights(
    ctx: &crate::driver::CudaContext,
    d_model: usize,
    ffn_dim: usize,
    weight_val: f32,
    bias_val: f32,
) -> GpuEncoderBlockWeights {
    let t = |val: f32, len: usize| GpuResidentTensor::from_host(ctx, &vec![val; len]).unwrap();
    GpuEncoderBlockWeights {
        ln1_gamma: t(1.0, d_model),
        ln1_beta: t(0.0, d_model),
        w_q: t(weight_val, d_model * d_model),
        b_q: t(bias_val, d_model),
        w_k: t(weight_val, d_model * d_model),
        b_k: t(bias_val, d_model),
        w_v: t(weight_val, d_model * d_model),
        b_v: t(bias_val, d_model),
        w_o: t(weight_val, d_model * d_model),
        b_o: t(bias_val, d_model),
        ln2_gamma: t(1.0, d_model),
        ln2_beta: t(0.0, d_model),
        ffn_up_w: t(weight_val, d_model * ffn_dim),
        ffn_up_b: t(bias_val, ffn_dim),
        ffn_down_w: t(weight_val, ffn_dim * d_model),
        ffn_down_b: t(bias_val, d_model),
    }
}

/// Build `GpuEncoderBlockWeights` with sequential weight values (for varied-input tests).
fn build_sequential_weights(
    ctx: &crate::driver::CudaContext,
    d_model: usize,
    ffn_dim: usize,
    scale: f32,
    bias_val: f32,
) -> GpuEncoderBlockWeights {
    let seq = |len: usize| -> Vec<f32> { (0..len).map(|i| (i as f32) * scale).collect() };
    let t_const =
        |val: f32, len: usize| GpuResidentTensor::from_host(ctx, &vec![val; len]).unwrap();
    let t_seq = |len: usize| GpuResidentTensor::from_host(ctx, &seq(len)).unwrap();
    GpuEncoderBlockWeights {
        ln1_gamma: t_const(1.0, d_model),
        ln1_beta: t_const(0.0, d_model),
        w_q: t_seq(d_model * d_model),
        b_q: t_const(bias_val, d_model),
        w_k: t_seq(d_model * d_model),
        b_k: t_const(bias_val, d_model),
        w_v: t_seq(d_model * d_model),
        b_v: t_const(bias_val, d_model),
        w_o: t_seq(d_model * d_model),
        b_o: t_const(bias_val, d_model),
        ln2_gamma: t_const(1.0, d_model),
        ln2_beta: t_const(0.0, d_model),
        ffn_up_w: t_seq(d_model * ffn_dim),
        ffn_up_b: t_const(bias_val, ffn_dim),
        ffn_down_w: t_seq(ffn_dim * d_model),
        ffn_down_b: t_const(bias_val, d_model),
    }
}

/// Build sequential input data of given size.
fn build_input(
    ctx: &crate::driver::CudaContext,
    seq_len: usize,
    d_model: usize,
    gen: impl Fn(usize) -> f32,
) -> GpuResidentTensor<f32> {
    let data: Vec<f32> = (0..(seq_len * d_model)).map(|i| gen(i)).collect();
    GpuResidentTensor::from_host(ctx, &data).unwrap()
}

// ============================================================================
// Forward Encoder Block Output Shape Tests
// ============================================================================

#[test]
fn test_forward_encoder_block_gpu_verifies_output_shape() {
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let d_model = 32usize;
    let ffn_dim = 64usize;
    let seq_len = 8usize;

    let config = GpuEncoderConfig {
        d_model: d_model as u32,
        n_heads: 2,
        ffn_dim: ffn_dim as u32,
    };
    let weights = build_uniform_weights(&ctx, d_model, ffn_dim, 0.01, 0.0);
    let input = build_input(&ctx, seq_len, d_model, |i| (i as f32) * 0.01);

    let output = forward_encoder_block_gpu(&ctx, &input, &weights, &config).unwrap();

    assert_eq!(output.len(), seq_len * d_model);
    assert!(output.is_device_resident());

    let mut output_copy = output;
    let host_output = output_copy.to_host().unwrap();
    assert!(host_output.iter().all(|v| v.is_finite()));
}

#[test]
fn test_forward_encoder_block_gpu_with_different_n_heads() {
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let d_model = 32usize;
    let ffn_dim = 128usize;
    let seq_len = 4usize;

    let config = GpuEncoderConfig {
        d_model: d_model as u32,
        n_heads: 4,
        ffn_dim: ffn_dim as u32,
    };
    let weights = build_uniform_weights(&ctx, d_model, ffn_dim, 0.02, 0.0);
    let input = build_input(&ctx, seq_len, d_model, |i| (i as f32) * 0.01);

    let output = forward_encoder_block_gpu(&ctx, &input, &weights, &config).unwrap();
    assert_eq!(output.len(), seq_len * d_model);
}

// ============================================================================
// Forward Encoder Block with Varied Input
// ============================================================================

#[test]
fn test_forward_encoder_block_with_varied_input_values() {
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let d_model = 16usize;
    let ffn_dim = 32usize;
    let seq_len = 4usize;

    let config = GpuEncoderConfig {
        d_model: d_model as u32,
        n_heads: 2,
        ffn_dim: ffn_dim as u32,
    };
    let weights = build_sequential_weights(&ctx, d_model, ffn_dim, 0.001, 0.1);
    let input = build_input(&ctx, seq_len, d_model, |i| ((i % 10) as f32) * 0.1 - 0.5);

    let output = forward_encoder_block_gpu(&ctx, &input, &weights, &config).unwrap();
    assert_eq!(output.len(), seq_len * d_model);
}

// ============================================================================
// Encoder Block Weights Field Size Verification
// ============================================================================

#[test]
fn test_gpu_encoder_block_weights_field_sizes() {
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let d_model = 32usize;
    let ffn_dim = 64usize;

    let weights = build_uniform_weights(&ctx, d_model, ffn_dim, 0.01, 0.0);

    assert_eq!(weights.ln1_gamma.len(), d_model);
    assert_eq!(weights.ln1_beta.len(), d_model);
    assert_eq!(weights.w_q.len(), d_model * d_model);
    assert_eq!(weights.b_q.len(), d_model);
    assert_eq!(weights.w_k.len(), d_model * d_model);
    assert_eq!(weights.b_k.len(), d_model);
    assert_eq!(weights.w_v.len(), d_model * d_model);
    assert_eq!(weights.b_v.len(), d_model);
    assert_eq!(weights.w_o.len(), d_model * d_model);
    assert_eq!(weights.b_o.len(), d_model);
    assert_eq!(weights.ln2_gamma.len(), d_model);
    assert_eq!(weights.ln2_beta.len(), d_model);
    assert_eq!(weights.ffn_up_w.len(), d_model * ffn_dim);
    assert_eq!(weights.ffn_up_b.len(), ffn_dim);
    assert_eq!(weights.ffn_down_w.len(), ffn_dim * d_model);
    assert_eq!(weights.ffn_down_b.len(), d_model);
}
