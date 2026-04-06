//! Calibration functions for different quantization granularities

use super::{QuantGranularity, QuantMode, QuantParams};

/// Calibrate quantization parameters for per-tensor quantization
///
/// # Arguments
/// * `values` - Input tensor values
/// * `bits` - Bit width (4 or 8)
/// * `mode` - Symmetric or asymmetric quantization
pub fn calibrate_per_tensor(values: &[f32], bits: u8, mode: QuantMode) -> QuantParams {
    let (scale, zero_point) = match mode {
        QuantMode::Symmetric => {
            let max_abs = values
                .iter()
                .map(|v| v.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(1e-8)
                .max(1e-8);

            let qmax = ((1i32 << (bits - 1)) - 1) as f32;
            let scale = max_abs / qmax;
            (scale, 0)
        }
        QuantMode::Asymmetric => {
            let (min_val, max_val) =
                values.iter().fold((f32::MAX, f32::MIN), |(min, max), &v| (min.min(v), max.max(v)));

            let range = (max_val - min_val).max(1e-8);
            let qmax = ((1i32 << bits) - 1) as f32;
            let scale = range / qmax;
            let zero_point = ((-min_val / scale).round() as i32).clamp(0, qmax as i32);
            (scale, zero_point)
        }
    };

    QuantParams {
        scales: vec![scale],
        zero_points: if mode == QuantMode::Asymmetric { vec![zero_point] } else { vec![] },
        granularity: QuantGranularity::PerTensor,
        mode,
        bits,
    }
}

/// Calibrate quantization parameters for per-channel quantization
///
/// # Arguments
/// * `values` - Input tensor values (row-major: [channels, features])
/// * `num_channels` - Number of channels (first dimension)
/// * `bits` - Bit width (4 or 8)
/// * `mode` - Symmetric or asymmetric quantization
pub fn calibrate_per_channel(
    values: &[f32],
    num_channels: usize,
    bits: u8,
    mode: QuantMode,
) -> QuantParams {
    if num_channels == 0 || values.is_empty() {
        return QuantParams {
            scales: vec![1.0],
            zero_points: if mode == QuantMode::Asymmetric { vec![0] } else { vec![] },
            granularity: QuantGranularity::PerChannel,
            mode,
            bits,
        };
    }

    let features_per_channel = values.len() / num_channels;
    let qmax_signed = ((1i32 << (bits - 1)) - 1) as f32;
    let qmax_unsigned = ((1i32 << bits) - 1) as f32;

    let mut scales = Vec::with_capacity(num_channels);
    let mut zero_points = Vec::with_capacity(num_channels);

    for ch in 0..num_channels {
        let start = ch * features_per_channel;
        let end = start + features_per_channel;
        let channel_values = &values[start..end];

        match mode {
            QuantMode::Symmetric => {
                let max_abs = channel_values
                    .iter()
                    .map(|v| v.abs())
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(1e-8)
                    .max(1e-8);

                scales.push(max_abs / qmax_signed);
            }
            QuantMode::Asymmetric => {
                let (min_val, max_val) = channel_values
                    .iter()
                    .fold((f32::MAX, f32::MIN), |(min, max), &v| (min.min(v), max.max(v)));

                let range = (max_val - min_val).max(1e-8);
                let scale = range / qmax_unsigned;
                let zp = ((-min_val / scale).round() as i32).clamp(0, qmax_unsigned as i32);

                scales.push(scale);
                zero_points.push(zp);
            }
        }
    }

    QuantParams { scales, zero_points, granularity: QuantGranularity::PerChannel, mode, bits }
}

/// Calibrate quantization parameters for per-group quantization
///
/// # Arguments
/// * `values` - Input tensor values
/// * `group_size` - Number of elements per group
/// * `bits` - Bit width (4 or 8)
/// * `mode` - Symmetric or asymmetric quantization
pub fn calibrate_per_group(
    values: &[f32],
    group_size: usize,
    bits: u8,
    mode: QuantMode,
) -> QuantParams {
    let group_size = group_size.max(1);
    let num_groups = values.len().div_ceil(group_size);
    let qmax_signed = ((1i32 << (bits - 1)) - 1) as f32;
    let qmax_unsigned = ((1i32 << bits) - 1) as f32;

    let mut scales = Vec::with_capacity(num_groups);
    let mut zero_points = Vec::with_capacity(num_groups);

    for group_idx in 0..num_groups {
        let start = group_idx * group_size;
        let end = (start + group_size).min(values.len());
        let group_values = &values[start..end];

        match mode {
            QuantMode::Symmetric => {
                let max_abs = group_values
                    .iter()
                    .map(|v| v.abs())
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(1e-8)
                    .max(1e-8);

                scales.push(max_abs / qmax_signed);
            }
            QuantMode::Asymmetric => {
                let (min_val, max_val) = group_values
                    .iter()
                    .fold((f32::MAX, f32::MIN), |(min, max), &v| (min.min(v), max.max(v)));

                let range = (max_val - min_val).max(1e-8);
                let scale = range / qmax_unsigned;
                let zp = ((-min_val / scale).round() as i32).clamp(0, qmax_unsigned as i32);

                scales.push(scale);
                zero_points.push(zp);
            }
        }
    }

    QuantParams {
        scales,
        zero_points,
        granularity: QuantGranularity::PerGroup(group_size),
        mode,
        bits,
    }
}
