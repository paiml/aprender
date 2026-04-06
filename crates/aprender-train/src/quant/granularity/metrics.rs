//! Quantization error metrics and comparison functions

use super::{
    calibrate_per_channel, calibrate_per_tensor, dequantize_with_params, quantize_with_params,
    QuantMode,
};

/// Compute quantization error (MSE)
pub fn quantization_mse(original: &[f32], dequantized: &[f32]) -> f32 {
    if original.len() != dequantized.len() || original.is_empty() {
        return f32::MAX;
    }

    let sum_sq: f32 = original.iter().zip(dequantized.iter()).map(|(a, b)| (a - b).powi(2)).sum();

    sum_sq / original.len().max(1) as f32
}

/// Compare per-channel vs per-tensor quantization error
///
/// # Arguments
/// * `values` - Input tensor values (row-major)
/// * `num_channels` - Number of channels
/// * `bits` - Bit width
///
/// # Returns
/// (per_tensor_mse, per_channel_mse)
pub fn compare_granularities(values: &[f32], num_channels: usize, bits: u8) -> (f32, f32) {
    // Per-tensor
    let pt_params = calibrate_per_tensor(values, bits, QuantMode::Symmetric);
    let pt_quantized = quantize_with_params(values, &pt_params);
    let pt_dequantized = dequantize_with_params(&pt_quantized, &pt_params);
    let pt_mse = quantization_mse(values, &pt_dequantized);

    // Per-channel
    let pc_params = calibrate_per_channel(values, num_channels, bits, QuantMode::Symmetric);
    let pc_quantized = quantize_with_params(values, &pc_params);
    let pc_dequantized = dequantize_with_params(&pc_quantized, &pc_params);
    let pc_mse = quantization_mse(values, &pc_dequantized);

    (pt_mse, pc_mse)
}
