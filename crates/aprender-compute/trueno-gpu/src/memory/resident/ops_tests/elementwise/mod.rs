//! Elementwise GPU-Resident Tensor tests: softmax, add, scale, gelu, bias_add, layer_norm

use crate::driver::{CudaContext, CudaStream};
use crate::memory::resident::{clear_kernel_cache, reset_transfer_counters, GpuResidentTensor};

/// Generate ramp data: `(0..size).map(|i| i as f32 * scale + offset)`.
fn ramp_f32(size: usize, scale: f32, offset: f32) -> Vec<f32> {
    (0..size).map(|i| i as f32 * scale + offset).collect()
}

/// Upload f32 data to GPU, panicking on failure.
fn upload(ctx: &CudaContext, data: &[f32]) -> GpuResidentTensor<f32> {
    GpuResidentTensor::from_host(ctx, data).expect("GPU upload failed in test")
}

/// Standard test preamble: clear kernel cache and create CUDA context.
macro_rules! fresh_ctx {
    () => {{
        clear_kernel_cache();
        cuda_ctx!()
    }};
}

/// Helper to create CUDA context, skipping test if unavailable
macro_rules! cuda_ctx {
    () => {
        match CudaContext::new(0) {
            Ok(ctx) => ctx,
            Err(e) => {
                eprintln!("Skipping CUDA test: {:?}", e);
                return;
            }
        }
    };
}

mod layernorm_gelu_bias;
mod pmat018_extended;
mod softmax_add_scale;
