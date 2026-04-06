//! Quantization and dequantization functions

use super::{
    calibrate_per_channel, calibrate_per_group, calibrate_per_tensor, QuantGranularity, QuantMode,
    QuantParams, QuantizedTensor,
};

/// Quantize values using given parameters
///
/// # Arguments
/// * `values` - Input f32 values
/// * `params` - Quantization parameters
pub fn quantize_with_params(values: &[f32], params: &QuantParams) -> Vec<i8> {
    let qmax_signed = ((1i32 << (params.bits - 1)) - 1) as f32;
    let qmin_signed = -qmax_signed - 1.0;
    let qmax_unsigned = ((1i32 << params.bits) - 1) as f32;

    let group_size = match params.granularity {
        QuantGranularity::PerTensor => values.len(),
        QuantGranularity::PerChannel => values.len() / params.scales.len().max(1),
        QuantGranularity::PerGroup(size) => size,
    };

    let mut result = Vec::with_capacity(values.len());

    for (i, &val) in values.iter().enumerate() {
        let group_idx = i / group_size.max(1);
        let scale = params.scales.get(group_idx).copied().unwrap_or(1.0);

        let q_val = match params.mode {
            QuantMode::Symmetric => (val / scale).round().clamp(qmin_signed, qmax_signed) as i8,
            QuantMode::Asymmetric => {
                let zp = params.zero_points.get(group_idx).copied().unwrap_or(0);
                let q = (val / scale + zp as f32).round().clamp(0.0, qmax_unsigned);
                // Store as signed for uniform representation
                (q as i32 - 128) as i8
            }
        };

        result.push(q_val);
    }

    result
}

/// Dequantize values using given parameters
///
/// # Arguments
/// * `quantized` - Quantized i8 values
/// * `params` - Quantization parameters
pub fn dequantize_with_params(quantized: &[i8], params: &QuantParams) -> Vec<f32> {
    let group_size = match params.granularity {
        QuantGranularity::PerTensor => quantized.len(),
        QuantGranularity::PerChannel => quantized.len() / params.scales.len().max(1),
        QuantGranularity::PerGroup(size) => size,
    };

    let mut result = Vec::with_capacity(quantized.len());

    for (i, &q_val) in quantized.iter().enumerate() {
        let group_idx = i / group_size.max(1);
        let scale = params.scales.get(group_idx).copied().unwrap_or(1.0);

        let val = match params.mode {
            QuantMode::Symmetric => f32::from(q_val) * scale,
            QuantMode::Asymmetric => {
                let zp = params.zero_points.get(group_idx).copied().unwrap_or(0);
                // Convert back from signed storage
                let q_unsigned = (i32::from(q_val) + 128) as f32;
                (q_unsigned - zp as f32) * scale
            }
        };

        result.push(val);
    }

    result
}

/// Quantize tensor with specified granularity
///
/// # Arguments
/// * `values` - Input tensor values
/// * `shape` - Tensor shape
/// * `granularity` - Quantization granularity
/// * `mode` - Quantization mode
/// * `bits` - Bit width (4 or 8)
pub fn quantize_tensor(
    values: &[f32],
    shape: &[usize],
    granularity: QuantGranularity,
    mode: QuantMode,
    bits: u8,
) -> QuantizedTensor {
    let params = match granularity {
        QuantGranularity::PerTensor => calibrate_per_tensor(values, bits, mode),
        QuantGranularity::PerChannel => {
            let num_channels = shape.first().copied().unwrap_or(1);
            calibrate_per_channel(values, num_channels, bits, mode)
        }
        QuantGranularity::PerGroup(group_size) => {
            calibrate_per_group(values, group_size, bits, mode)
        }
    };

    let data = quantize_with_params(values, &params);

    QuantizedTensor { data, params, shape: shape.to_vec() }
}

/// Dequantize tensor
pub fn dequantize_tensor(quantized: &QuantizedTensor) -> Vec<f32> {
    dequantize_with_params(&quantized.data, &quantized.params)
}
