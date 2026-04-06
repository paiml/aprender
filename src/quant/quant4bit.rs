//! 4-bit quantization for QLoRA
//!
//! Implements block-wise symmetric 4-bit quantization to reduce memory usage
//! of frozen base weights by ~75% (4 bits vs 32 bits per value).
//!
//! Uses block-wise quantization with 64-element blocks, where each block has:
//! - 1 scale factor (f32)
//! - 64 quantized values (4 bits each = 32 bytes total)
//!
//! Quantization: q = round(clamp(x / scale, -7, 7))
//! Dequantization: x ≈ q * scale

use serde::{Deserialize, Serialize};

/// Block size for quantization (64 elements per block)
pub const BLOCK_SIZE: usize = 64;

/// 4-bit quantized representation with block-wise scale factors
///
/// Memory layout:
/// - scales: `Vec<f32>` with length = ceil(n / BLOCK_SIZE)
/// - data: `Vec<u8>` where each byte stores 2 quantized values (4 bits each)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Quantized4Bit {
    /// Scale factors (one per block)
    pub scales: Vec<f32>,
    /// Quantized data: 2 values per byte (4 bits each)
    pub data: Vec<u8>,
    /// Original number of elements
    pub len: usize,
}

impl Quantized4Bit {
    /// Get memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.scales.len() * 4 + self.data.len()
    }

    /// Get compression ratio vs f32
    #[provable_contracts_macros::contract("quantization-v1", equation = "compression_ratio")]
    pub fn compression_ratio(&self) -> f32 {
        let original_bytes = self.len * 4; // f32
        let compressed_bytes = self.memory_bytes();
        original_bytes as f32 / compressed_bytes as f32
    }
}

/// Quantize f32 values to 4-bit with block-wise scaling
///
/// # Arguments
/// * `values` - Input f32 values
///
/// # Returns
/// Quantized4Bit structure with scales and packed 4-bit data
pub fn quantize_4bit(values: &[f32]) -> Quantized4Bit {
    let len = values.len();
    let num_blocks = len.div_ceil(BLOCK_SIZE);

    let mut scales = Vec::with_capacity(num_blocks);
    let mut data = Vec::with_capacity(len.div_ceil(2)); // 2 values per byte

    for block_idx in 0..num_blocks {
        let start = block_idx * BLOCK_SIZE;
        let end = (start + BLOCK_SIZE).min(len);
        let block = &values[start..end];

        // Compute scale factor for this block: max absolute value / 7
        // (7 is max representable in 4-bit signed: -7 to 7)
        let max_abs = block.iter().map(|v| v.abs()).max_by(f32::total_cmp).unwrap_or(1e-8);

        let scale = max_abs / 7.0;
        scales.push(scale);

        // Quantize each value in the block
        for (i, &val) in block.iter().enumerate() {
            let quantized = quantize_value(val, scale);

            // Pack 2 values per byte (mask to 4 bits: 0x0F)
            if i.is_multiple_of(2) {
                // First value: store in upper 4 bits
                data.push(((quantized as u8) & 0x0F) << 4);
            } else {
                // Second value: store in lower 4 bits
                let last_idx = data.len() - 1;
                data[last_idx] |= (quantized as u8) & 0x0F;
            }
        }

        // If block has odd number of elements, the last byte is already pushed
        // with upper 4 bits filled and lower 4 bits as 0
    }

    Quantized4Bit { scales, data, len }
}

/// Dequantize 4-bit values back to f32
///
/// # Arguments
/// * `quantized` - Quantized4Bit data
///
/// # Returns
/// Dequantized f32 values
pub fn dequantize_4bit(quantized: &Quantized4Bit) -> Vec<f32> {
    let mut result = Vec::with_capacity(quantized.len);

    let num_blocks = quantized.scales.len();

    for block_idx in 0..num_blocks {
        let scale = quantized.scales[block_idx];
        let start = block_idx * BLOCK_SIZE;
        let end = (start + BLOCK_SIZE).min(quantized.len);
        let block_len = end - start;

        for i in 0..block_len {
            let byte_idx = usize::midpoint(start, i);
            let byte = quantized.data[byte_idx];

            // Extract 4-bit value and sign-extend
            let q_val = if (start + i).is_multiple_of(2) {
                // Upper 4 bits - shift right to get value, then sign extend
                let nibble = (byte >> 4) & 0x0F;
                // Sign extend: if bit 3 is set, extend with 1s
                if nibble & 0x08 != 0 {
                    (nibble | 0xF0) as i8
                } else {
                    nibble as i8
                }
            } else {
                // Lower 4 bits
                let nibble = byte & 0x0F;
                // Sign extend: if bit 3 is set, extend with 1s
                if nibble & 0x08 != 0 {
                    (nibble | 0xF0) as i8
                } else {
                    nibble as i8
                }
            };

            // Dequantize
            let deq_val = f32::from(q_val) * scale;
            result.push(deq_val);
        }
    }

    result
}

/// Quantize a single value to 4-bit
///
/// Maps f32 to integer in range [-7, 7]
fn quantize_value(val: f32, scale: f32) -> i8 {
    let normalized = val / scale;
    let clamped = normalized.clamp(-7.0, 7.0);
    clamped.round() as i8
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_quantize_dequantize_round_trip() {
        let values = vec![1.0, -2.0, 3.5, -4.2, 0.5, -0.8, 2.1, -1.5];
        let quantized = quantize_4bit(&values);
        let dequantized = dequantize_4bit(&quantized);

        assert_eq!(dequantized.len(), values.len());

        // Check approximate equality (quantization introduces some error)
        // 4-bit quantization has limited precision (15 values from -7 to 7)
        for (original, deq) in values.iter().zip(dequantized.iter()) {
            let error = (original - deq).abs();
            let relative_error = error / original.abs().max(1e-6);
            assert!(
                relative_error < 0.3,
                "Relative error too large: {original} vs {deq} (error: {error}, rel_error: {relative_error})"
            );
        }
    }

    #[test]
    fn test_quantize_zeros() {
        let values = vec![0.0; 64];
        let quantized = quantize_4bit(&values);
        let dequantized = dequantize_4bit(&quantized);

        for val in dequantized {
            assert_abs_diff_eq!(val, 0.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_quantize_uniform() {
        let values = vec![1.0; 64];
        let quantized = quantize_4bit(&values);
        let dequantized = dequantize_4bit(&quantized);

        for val in dequantized {
            assert_abs_diff_eq!(val, 1.0, epsilon = 0.2);
        }
    }

    #[test]
    fn test_quantize_range() {
        // Test values spanning the quantization range
        let values: Vec<f32> = (-7..=7).map(|x| x as f32).collect();
        let quantized = quantize_4bit(&values);
        let dequantized = dequantize_4bit(&quantized);

        for (original, deq) in values.iter().zip(dequantized.iter()) {
            assert_abs_diff_eq!(original, deq, epsilon = 0.5);
        }
    }

    #[test]
    fn test_quantize_multiple_blocks() {
        // Test with more than one block (>64 elements)
        let values: Vec<f32> = (0..200).map(|i| (i as f32 * 0.1).sin()).collect();
        let quantized = quantize_4bit(&values);
        let dequantized = dequantize_4bit(&quantized);

        assert_eq!(dequantized.len(), values.len());

        // Verify multiple blocks were created
        let expected_blocks = 200_usize.div_ceil(BLOCK_SIZE);
        assert_eq!(quantized.scales.len(), expected_blocks);
    }

    #[test]
    fn test_memory_savings() {
        let values = vec![1.0; 1024];
        let quantized = quantize_4bit(&values);

        let original_bytes = values.len() * 4; // f32 = 4 bytes
        let compressed_bytes = quantized.memory_bytes();

        // Should achieve significant compression (close to 8x for large arrays)
        let compression = original_bytes as f32 / compressed_bytes as f32;
        assert!(compression > 6.0, "Compression ratio {compression} should be > 6.0");
    }

    #[test]
    fn test_compression_ratio() {
        let values = vec![1.5; 1024];
        let quantized = quantize_4bit(&values);

        let ratio = quantized.compression_ratio();
        assert!(ratio > 6.0, "Compression ratio {ratio} should be > 6.0");
    }

    #[test]
    fn test_quantize_small_values() {
        let values = vec![0.001, 0.002, 0.003, 0.004];
        let quantized = quantize_4bit(&values);
        let dequantized = dequantize_4bit(&quantized);

        // Small values should be preserved relatively well
        for (original, deq) in values.iter().zip(dequantized.iter()) {
            let error = (original - deq).abs();
            assert!(error < 0.001, "Error {error} too large for small value");
        }
    }

    #[test]
    fn test_quantize_mixed_magnitudes() {
        // Use values with similar magnitudes to avoid quantization precision issues
        // When values span 1000x range in one block, small values may quantize to zero
        let values = vec![10.0, 1.0, -5.0, 0.5, 7.5, -2.0];
        let quantized = quantize_4bit(&values);
        let dequantized = dequantize_4bit(&quantized);

        assert_eq!(dequantized.len(), values.len());

        // Each value should be within reasonable error
        // 4-bit quantization has only 15 discrete values, so error can be substantial
        // For values near zero, use absolute error; for larger values, use relative error
        for (original, deq) in values.iter().zip(dequantized.iter()) {
            let error = (original - deq).abs();
            if original.abs() < 1.0 {
                // Small values: use absolute error tolerance
                assert!(
                    error < 1.5,
                    "Absolute error {error} too large for small value {original} vs {deq}"
                );
            } else {
                // Larger values: use relative error
                let relative_error = error / original.abs();
                assert!(
                    relative_error < 0.5,
                    "Relative error {relative_error} too large for {original} vs {deq} (error: {error})"
                );
            }
        }
    }

    #[test]
    fn test_quantize_odd_length() {
        // Test with odd number of elements (not divisible by 2 or BLOCK_SIZE)
        let values: Vec<f32> = (0..77).map(|i| i as f32 * 0.5).collect();
        let quantized = quantize_4bit(&values);
        let dequantized = dequantize_4bit(&quantized);

        assert_eq!(dequantized.len(), 77);
    }

    #[test]
    fn test_quantized_data_size() {
        let values = vec![1.0; 128];
        let quantized = quantize_4bit(&values);

        // 128 values = 64 bytes (2 values per byte)
        assert_eq!(quantized.data.len(), 64);

        // 128 values = 2 blocks of 64
        assert_eq!(quantized.scales.len(), 2);
    }
}
