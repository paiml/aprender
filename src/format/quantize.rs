//! Quantization support for .apr format (spec §6.2)
//!
//! Provides GGUF-compatible quantization types with block-wise quantization:
//! - Q8_0: 8-bit, 32-element blocks, f16 scale per block
//! - Q4_0: 4-bit, 32-element blocks, f16 scale per block
//! - Q4_1: 4-bit with min, 32-element blocks, f16 scale + f16 min per block
//!
//! # Design Principles
//!
//! 1. **Explicit opt-in only**: Quantization is NEVER automatic
//! 2. **GGUF compatible**: Block sizes and layouts match llama.cpp
//! 3. **WASM compatible**: Pure Rust, no FFI dependencies
//! 4. **Plugin architecture**: Custom quantizers via `Quantizer` trait
//!
//! # References
//!
//! - GGML quantization: <https://github.com/ggerganov/ggml>
//! - llama.cpp Q8_0/Q4_0: <https://github.com/ggerganov/llama.cpp>

use crate::error::{AprenderError, Result};
use half::f16;
use serde::{Deserialize, Serialize};

/// GGUF-compatible block size (32 elements per block)
pub const BLOCK_SIZE: usize = 32;

/// Q8_0 block size in bytes: 2 (f16 scale) + 32 (i8 × 32)
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// Q4_0 block size in bytes: 2 (f16 scale) + 16 (packed nibbles)
pub const Q4_0_BLOCK_BYTES: usize = 18;

/// Q4_1 block size in bytes: 2 (f16 scale) + 2 (f16 min) + 16 (packed nibbles)
pub const Q4_1_BLOCK_BYTES: usize = 20;

/// Quantization type identifier (spec §6.2.2)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum QuantType {
    /// 8-bit, block-wise, GGUF compatible (8.5 bits/weight)
    Q8_0 = 0x01,
    /// 4-bit, block-wise, GGUF compatible (4.5 bits/weight)
    Q4_0 = 0x02,
    /// 4-bit with min, block-wise, GGUF compatible (5.0 bits/weight)
    Q4_1 = 0x03,
    /// 8-bit, per-tensor, SafeTensors style (8 bits/weight)
    Q8Tensor = 0x10,
    /// Plugin-defined custom quantization
    Custom = 0xFF,
}

impl QuantType {
    /// Convert from u8 value
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0x01 => Some(Self::Q8_0),
            0x02 => Some(Self::Q4_0),
            0x03 => Some(Self::Q4_1),
            0x10 => Some(Self::Q8Tensor),
            0xFF => Some(Self::Custom),
            _ => None,
        }
    }

    /// Get bits per weight for this quantization type
    #[must_use]
    pub fn bits_per_weight(&self) -> f32 {
        match self {
            Self::Q8_0 => 8.5, // 32 i8 + 1 f16 scale = 272 bits / 32 = 8.5
            Self::Q4_0 => 4.5, // 16 nibble pairs + 1 f16 scale = 144 bits / 32 = 4.5
            Self::Q4_1 => 5.0, // 16 nibble pairs + f16 scale + f16 min = 160 bits / 32 = 5.0
            Self::Q8Tensor => 8.0,
            Self::Custom => 0.0, // Unknown
        }
    }
}

/// Block-wise quantized tensor (GGUF-compatible, spec §6.2.2)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedBlock {
    /// Quantization type
    pub quant_type: QuantType,
    /// Original tensor shape
    pub shape: Vec<usize>,
    /// Raw block data (format depends on quant_type)
    pub blocks: Vec<u8>,
    /// Block size (32 for GGUF compatibility)
    pub block_size: usize,
}

impl QuantizedBlock {
    /// Get the number of blocks
    #[must_use]
    pub fn num_blocks(&self) -> usize {
        let total_elements: usize = self.shape.iter().product();
        (total_elements + self.block_size - 1) / self.block_size
    }

    /// Get the total number of elements in the original tensor
    #[must_use]
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Estimate compressed size in bytes
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.blocks.len()
    }

    /// Estimate original size in bytes (f32)
    #[must_use]
    pub fn original_size_bytes(&self) -> usize {
        self.num_elements() * 4
    }

    /// Compression ratio (original / quantized)
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        if self.blocks.is_empty() {
            return 1.0;
        }
        self.original_size_bytes() as f32 / self.size_bytes() as f32
    }
}

/// Per-tensor quantized tensor (SafeTensors-compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedTensor {
    /// Quantization type
    pub quant_type: QuantType,
    /// Original tensor shape
    pub shape: Vec<usize>,
    /// Quantized values
    pub data: Vec<i8>,
    /// Scale factor: r = S(q - Z)
    pub scale: f32,
    /// Zero point
    pub zero_point: i8,
}

/// Quantization metadata for model header
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationInfo {
    /// Quantization type
    pub quant_type: QuantType,
    /// Calibration method used ("minmax", "percentile", "mse")
    pub calibration_method: String,
    /// Number of calibration samples (0 if no calibration data)
    pub calibration_samples: u32,
    /// Original data type ("f32", "f16", "bf16")
    pub original_dtype: String,
    /// Quantization error (MSE if available)
    pub quantization_error: Option<f32>,
}

impl Default for QuantizationInfo {
    fn default() -> Self {
        Self {
            quant_type: QuantType::Q8_0,
            calibration_method: "minmax".to_string(),
            calibration_samples: 0,
            original_dtype: "f32".to_string(),
            quantization_error: None,
        }
    }
}

/// Quantizer trait for implementing custom quantization schemes (spec §6.2.3)
pub trait Quantizer: Send + Sync {
    /// Unique identifier for this quantizer
    fn name(&self) -> &'static str;

    /// Quantize f32 tensor to blocks
    fn quantize(&self, data: &[f32], shape: &[usize]) -> Result<QuantizedBlock>;

    /// Dequantize blocks back to f32
    fn dequantize(&self, block: &QuantizedBlock) -> Result<Vec<f32>>;

    /// Bits per weight (for size estimation)
    fn bits_per_weight(&self) -> f32;
}

/// Q8_0 quantizer (GGUF-compatible, 8.5 bits/weight)
///
/// Block layout (34 bytes per 32 elements):
/// - scale (f16): 2 bytes
/// - quants (i8 × 32): 32 bytes
#[derive(Debug, Clone, Copy, Default)]
pub struct Q8_0Quantizer;

impl Quantizer for Q8_0Quantizer {
    fn name(&self) -> &'static str {
        "Q8_0"
    }

    fn quantize(&self, data: &[f32], shape: &[usize]) -> Result<QuantizedBlock> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(AprenderError::DimensionMismatch {
                expected: expected_len.to_string(),
                actual: data.len().to_string(),
            });
        }

        let num_blocks = (data.len() + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let mut blocks = Vec::with_capacity(num_blocks * Q8_0_BLOCK_BYTES);

        for block_idx in 0..num_blocks {
            let start = block_idx * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(data.len());
            let block_data = &data[start..end];

            // Find max absolute value for scale
            let max_abs = block_data.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);

            // Calculate scale (avoid division by zero)
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

            // Write scale as f16
            let scale_f16 = f16::from_f32(scale);
            blocks.extend_from_slice(&scale_f16.to_le_bytes());

            // Quantize values to i8
            for &val in block_data {
                let q = (val * inv_scale).round().clamp(-127.0, 127.0) as i8;
                blocks.push(q as u8);
            }

            // Pad remaining elements in last block with zeros
            let padding_count = BLOCK_SIZE - block_data.len();
            if padding_count > 0 {
                blocks.resize(blocks.len() + padding_count, 0);
            }
        }

        Ok(QuantizedBlock {
            quant_type: QuantType::Q8_0,
            shape: shape.to_vec(),
            blocks,
            block_size: BLOCK_SIZE,
        })
    }

    fn dequantize(&self, block: &QuantizedBlock) -> Result<Vec<f32>> {
        if block.quant_type != QuantType::Q8_0 {
            return Err(AprenderError::FormatError {
                message: format!("Expected Q8_0 block, got {:?}", block.quant_type),
            });
        }

        let total_elements: usize = block.shape.iter().product();
        let num_blocks = block.num_blocks();

        if block.blocks.len() != num_blocks * Q8_0_BLOCK_BYTES {
            return Err(AprenderError::FormatError {
                message: format!(
                    "Invalid Q8_0 block data size: expected {}, got {}",
                    num_blocks * Q8_0_BLOCK_BYTES,
                    block.blocks.len()
                ),
            });
        }

        let mut result = Vec::with_capacity(total_elements);

        for block_idx in 0..num_blocks {
            let block_start = block_idx * Q8_0_BLOCK_BYTES;

            // Read scale (f16)
            let scale_bytes = [block.blocks[block_start], block.blocks[block_start + 1]];
            let scale = f16::from_le_bytes(scale_bytes).to_f32();

            // Read and dequantize values
            let quants_start = block_start + 2;
            let elements_in_block = if block_idx == num_blocks - 1 {
                let remaining = total_elements % BLOCK_SIZE;
                if remaining == 0 {
                    BLOCK_SIZE
                } else {
                    remaining
                }
            } else {
                BLOCK_SIZE
            };

            for i in 0..elements_in_block {
                let q = block.blocks[quants_start + i] as i8;
                let val = f32::from(q) * scale;
                result.push(val);
            }
        }

        Ok(result)
    }

    fn bits_per_weight(&self) -> f32 {
        8.5
    }
}

/// Q4_0 quantizer (GGUF-compatible, 4.5 bits/weight)
///
/// Block layout (18 bytes per 32 elements):
/// - scale (f16): 2 bytes
/// - quants (nibbles × 32 packed in 16 bytes): 16 bytes
#[derive(Debug, Clone, Copy, Default)]
pub struct Q4_0Quantizer;

impl Quantizer for Q4_0Quantizer {
    fn name(&self) -> &'static str {
        "Q4_0"
    }

    fn quantize(&self, data: &[f32], shape: &[usize]) -> Result<QuantizedBlock> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(AprenderError::DimensionMismatch {
                expected: expected_len.to_string(),
                actual: data.len().to_string(),
            });
        }

        let num_blocks = (data.len() + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let mut blocks = Vec::with_capacity(num_blocks * Q4_0_BLOCK_BYTES);

        for block_idx in 0..num_blocks {
            let start = block_idx * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(data.len());
            let block_data = &data[start..end];

            // Find max absolute value for scale
            let max_abs = block_data.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);

            // Calculate scale (avoid division by zero)
            // Q4_0 uses signed 4-bit: -8 to 7
            let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
            let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

            // Write scale as f16
            let scale_f16 = f16::from_f32(scale);
            blocks.extend_from_slice(&scale_f16.to_le_bytes());

            // Quantize values to 4-bit and pack into nibbles
            // Each byte contains two 4-bit values: low nibble first, then high nibble
            let mut padded_data = block_data.to_vec();
            padded_data.resize(BLOCK_SIZE, 0.0);

            for i in (0..BLOCK_SIZE).step_by(2) {
                let q0 = ((padded_data[i] * inv_scale).round().clamp(-8.0, 7.0) as i8 + 8) as u8;
                let q1 =
                    ((padded_data[i + 1] * inv_scale).round().clamp(-8.0, 7.0) as i8 + 8) as u8;
                // Pack: low nibble = q0 & 0xF, high nibble = q1 << 4
                let packed = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
                blocks.push(packed);
            }
        }

        Ok(QuantizedBlock {
            quant_type: QuantType::Q4_0,
            shape: shape.to_vec(),
            blocks,
            block_size: BLOCK_SIZE,
        })
    }

    fn dequantize(&self, block: &QuantizedBlock) -> Result<Vec<f32>> {
        if block.quant_type != QuantType::Q4_0 {
            return Err(AprenderError::FormatError {
                message: format!("Expected Q4_0 block, got {:?}", block.quant_type),
            });
        }

        let total_elements: usize = block.shape.iter().product();
        let num_blocks = block.num_blocks();

        if block.blocks.len() != num_blocks * Q4_0_BLOCK_BYTES {
            return Err(AprenderError::FormatError {
                message: format!(
                    "Invalid Q4_0 block data size: expected {}, got {}",
                    num_blocks * Q4_0_BLOCK_BYTES,
                    block.blocks.len()
                ),
            });
        }

        let mut result = Vec::with_capacity(total_elements);

        for block_idx in 0..num_blocks {
            let block_start = block_idx * Q4_0_BLOCK_BYTES;

            // Read scale (f16)
            let scale_bytes = [block.blocks[block_start], block.blocks[block_start + 1]];
            let scale = f16::from_le_bytes(scale_bytes).to_f32();

            // Read and dequantize packed nibbles
            let quants_start = block_start + 2;
            let elements_in_block = if block_idx == num_blocks - 1 {
                let remaining = total_elements % BLOCK_SIZE;
                if remaining == 0 {
                    BLOCK_SIZE
                } else {
                    remaining
                }
            } else {
                BLOCK_SIZE
            };

            for i in 0..(elements_in_block + 1) / 2 {
                let packed = block.blocks[quants_start + i];
                let q0 = (packed & 0x0F) as i8 - 8;
                let q1 = ((packed >> 4) & 0x0F) as i8 - 8;

                result.push(f32::from(q0) * scale);
                if result.len() < total_elements && (i * 2 + 1) < elements_in_block {
                    result.push(f32::from(q1) * scale);
                }
            }
        }

        // Ensure we have exactly the right number of elements
        result.truncate(total_elements);

        Ok(result)
    }

    fn bits_per_weight(&self) -> f32 {
        4.5
    }
}

/// Quantize f32 data to the specified type
pub fn quantize(data: &[f32], shape: &[usize], quant_type: QuantType) -> Result<QuantizedBlock> {
    match quant_type {
        QuantType::Q8_0 => Q8_0Quantizer.quantize(data, shape),
        QuantType::Q4_0 => Q4_0Quantizer.quantize(data, shape),
        QuantType::Q4_1 => Err(AprenderError::FormatError {
            message: "Q4_1 quantization not yet implemented".to_string(),
        }),
        QuantType::Q8Tensor => Err(AprenderError::FormatError {
            message: "Q8Tensor quantization not yet implemented".to_string(),
        }),
        QuantType::Custom => Err(AprenderError::FormatError {
            message: "Custom quantization requires a custom Quantizer implementation".to_string(),
        }),
    }
}

/// Dequantize block back to f32
pub fn dequantize(block: &QuantizedBlock) -> Result<Vec<f32>> {
    match block.quant_type {
        QuantType::Q8_0 => Q8_0Quantizer.dequantize(block),
        QuantType::Q4_0 => Q4_0Quantizer.dequantize(block),
        QuantType::Q4_1 => Err(AprenderError::FormatError {
            message: "Q4_1 dequantization not yet implemented".to_string(),
        }),
        QuantType::Q8Tensor => Err(AprenderError::FormatError {
            message: "Q8Tensor dequantization not yet implemented".to_string(),
        }),
        QuantType::Custom => Err(AprenderError::FormatError {
            message: "Custom dequantization requires a custom Quantizer implementation".to_string(),
        }),
    }
}

/// Calculate mean squared error between original and dequantized values
pub fn quantization_mse(original: &[f32], dequantized: &[f32]) -> f32 {
    if original.len() != dequantized.len() || original.is_empty() {
        return f32::NAN;
    }

    let sum_sq_error: f32 = original
        .iter()
        .zip(dequantized.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum();

    sum_sq_error / original.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_type_from_u8() {
        assert_eq!(QuantType::from_u8(0x01), Some(QuantType::Q8_0));
        assert_eq!(QuantType::from_u8(0x02), Some(QuantType::Q4_0));
        assert_eq!(QuantType::from_u8(0x03), Some(QuantType::Q4_1));
        assert_eq!(QuantType::from_u8(0x10), Some(QuantType::Q8Tensor));
        assert_eq!(QuantType::from_u8(0xFF), Some(QuantType::Custom));
        assert_eq!(QuantType::from_u8(0x99), None);
    }

    #[test]
    fn test_quant_type_bits_per_weight() {
        assert!((QuantType::Q8_0.bits_per_weight() - 8.5).abs() < 0.01);
        assert!((QuantType::Q4_0.bits_per_weight() - 4.5).abs() < 0.01);
        assert!((QuantType::Q4_1.bits_per_weight() - 5.0).abs() < 0.01);
        assert!((QuantType::Q8Tensor.bits_per_weight() - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_q8_0_roundtrip_simple() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let shape = vec![8];

        let quantizer = Q8_0Quantizer;
        let quantized = quantizer.quantize(&data, &shape).expect("quantize");
        let dequantized = quantizer.dequantize(&quantized).expect("dequantize");

        assert_eq!(dequantized.len(), data.len());

        // Check values are close (quantization introduces error)
        for (orig, deq) in data.iter().zip(dequantized.iter()) {
            assert!(
                (orig - deq).abs() < 0.1,
                "Values differ too much: {} vs {}",
                orig,
                deq
            );
        }
    }

    #[test]
    fn test_q8_0_roundtrip_large() {
        // Test with more than one block (32 elements)
        let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.1 - 5.0).collect();
        let shape = vec![100];

        let quantized = quantize(&data, &shape, QuantType::Q8_0).expect("quantize");
        let dequantized = dequantize(&quantized).expect("dequantize");

        assert_eq!(dequantized.len(), data.len());

        let mse = quantization_mse(&data, &dequantized);
        assert!(mse < 0.01, "MSE too high: {}", mse);
    }

    #[test]
    fn test_q8_0_block_size() {
        let data: Vec<f32> = vec![1.0; 64]; // 2 blocks
        let shape = vec![64];

        let quantized = quantize(&data, &shape, QuantType::Q8_0).expect("quantize");

        assert_eq!(quantized.num_blocks(), 2);
        assert_eq!(quantized.blocks.len(), 2 * Q8_0_BLOCK_BYTES);
    }

    #[test]
    fn test_q8_0_compression_ratio() {
        let data: Vec<f32> = vec![1.0; 128]; // 4 blocks
        let shape = vec![128];

        let quantized = quantize(&data, &shape, QuantType::Q8_0).expect("quantize");

        // Original: 128 * 4 = 512 bytes
        // Quantized: 4 blocks * 34 bytes = 136 bytes
        // Ratio: 512 / 136 ≈ 3.76
        let ratio = quantized.compression_ratio();
        assert!(ratio > 3.5, "Compression ratio too low: {}", ratio);
    }

    #[test]
    fn test_q4_0_roundtrip_simple() {
        let data: Vec<f32> = vec![1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 0.0, 0.25];
        let shape = vec![8];

        let quantizer = Q4_0Quantizer;
        let quantized = quantizer.quantize(&data, &shape).expect("quantize");
        let dequantized = quantizer.dequantize(&quantized).expect("dequantize");

        assert_eq!(dequantized.len(), data.len());

        // Q4_0 has lower precision, allow more error
        for (orig, deq) in data.iter().zip(dequantized.iter()) {
            assert!(
                (orig - deq).abs() < 0.5,
                "Values differ too much: {} vs {}",
                orig,
                deq
            );
        }
    }

    #[test]
    fn test_q4_0_roundtrip_large() {
        let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.1 - 5.0).collect();
        let shape = vec![100];

        let quantized = quantize(&data, &shape, QuantType::Q4_0).expect("quantize");
        let dequantized = dequantize(&quantized).expect("dequantize");

        assert_eq!(dequantized.len(), data.len());

        // Q4_0 has lower precision, MSE will be higher
        let mse = quantization_mse(&data, &dequantized);
        assert!(mse < 0.5, "MSE too high for Q4_0: {}", mse);
    }

    #[test]
    fn test_q4_0_block_size() {
        let data: Vec<f32> = vec![1.0; 64]; // 2 blocks
        let shape = vec![64];

        let quantized = quantize(&data, &shape, QuantType::Q4_0).expect("quantize");

        assert_eq!(quantized.num_blocks(), 2);
        assert_eq!(quantized.blocks.len(), 2 * Q4_0_BLOCK_BYTES);
    }

    #[test]
    fn test_q4_0_compression_ratio() {
        let data: Vec<f32> = vec![1.0; 128]; // 4 blocks
        let shape = vec![128];

        let quantized = quantize(&data, &shape, QuantType::Q4_0).expect("quantize");

        // Original: 128 * 4 = 512 bytes
        // Quantized: 4 blocks * 18 bytes = 72 bytes
        // Ratio: 512 / 72 ≈ 7.1
        let ratio = quantized.compression_ratio();
        assert!(ratio > 6.0, "Compression ratio too low: {}", ratio);
    }

    #[test]
    fn test_quantize_zeros() {
        let data: Vec<f32> = vec![0.0; 32];
        let shape = vec![32];

        let quantized = quantize(&data, &shape, QuantType::Q8_0).expect("quantize");
        let dequantized = dequantize(&quantized).expect("dequantize");

        for val in &dequantized {
            assert!((val.abs()) < 0.001, "Expected zero, got {}", val);
        }
    }

    #[test]
    fn test_quantize_shape_mismatch() {
        let data: Vec<f32> = vec![1.0; 10];
        let shape = vec![20]; // Wrong shape

        let result = quantize(&data, &shape, QuantType::Q8_0);
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_wrong_type() {
        let data: Vec<f32> = vec![1.0; 32];
        let shape = vec![32];

        let mut quantized = quantize(&data, &shape, QuantType::Q8_0).expect("quantize");
        quantized.quant_type = QuantType::Q4_0; // Wrong type

        let result = Q8_0Quantizer.dequantize(&quantized);
        assert!(result.is_err());
    }

    #[test]
    fn test_quantization_mse() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.1, 2.1, 3.1, 4.1];

        let mse = quantization_mse(&a, &b);
        assert!((mse - 0.01).abs() < 0.001);
    }

    #[test]
    fn test_quantization_mse_empty() {
        let mse = quantization_mse(&[], &[]);
        assert!(mse.is_nan());
    }

    #[test]
    fn test_quantization_mse_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];

        let mse = quantization_mse(&a, &b);
        assert!(mse.is_nan());
    }

    #[test]
    fn test_quantization_info_default() {
        let info = QuantizationInfo::default();
        assert_eq!(info.quant_type, QuantType::Q8_0);
        assert_eq!(info.calibration_method, "minmax");
        assert_eq!(info.original_dtype, "f32");
    }

    #[test]
    fn test_q8_0_negative_values() {
        let data: Vec<f32> = vec![-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0, 7.0];
        let shape = vec![8];

        let quantized = quantize(&data, &shape, QuantType::Q8_0).expect("quantize");
        let dequantized = dequantize(&quantized).expect("dequantize");

        for (orig, deq) in data.iter().zip(dequantized.iter()) {
            assert!(
                (orig - deq).abs() < 0.1,
                "Values differ: {} vs {}",
                orig,
                deq
            );
        }
    }

    #[test]
    fn test_q4_0_exact_block_boundary() {
        // Test exactly 32 elements (one block)
        let data: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let shape = vec![32];

        let quantized = quantize(&data, &shape, QuantType::Q4_0).expect("quantize");
        assert_eq!(quantized.num_blocks(), 1);

        let dequantized = dequantize(&quantized).expect("dequantize");
        assert_eq!(dequantized.len(), 32);
    }

    #[test]
    fn test_q8_0_multidimensional_shape() {
        let data: Vec<f32> = (0..96).map(|i| i as f32 * 0.01).collect();
        let shape = vec![4, 24]; // 4x24 = 96 elements

        let quantized = quantize(&data, &shape, QuantType::Q8_0).expect("quantize");
        assert_eq!(quantized.shape, vec![4, 24]);
        assert_eq!(quantized.num_elements(), 96);

        let dequantized = dequantize(&quantized).expect("dequantize");
        assert_eq!(dequantized.len(), 96);
    }
}

// ============================================================================
// Property-Based Tests (EXTREME TDD)
// ============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    // ================================================================
    // Arbitrary Strategies
    // ================================================================

    /// Generate arbitrary QuantType (only implemented ones)
    fn arb_quant_type() -> impl Strategy<Value = QuantType> {
        prop_oneof![Just(QuantType::Q8_0), Just(QuantType::Q4_0),]
    }

    /// Generate arbitrary valid shape (1-3 dimensions, reasonable sizes)
    fn arb_shape() -> impl Strategy<Value = Vec<usize>> {
        prop_oneof![
            // 1D shapes
            (1usize..200).prop_map(|n| vec![n]),
            // 2D shapes
            (1usize..50, 1usize..50).prop_map(|(a, b)| vec![a, b]),
            // 3D shapes
            (1usize..20, 1usize..20, 1usize..20).prop_map(|(a, b, c)| vec![a, b, c]),
        ]
    }

    // ================================================================
    // QuantType Property Tests
    // ================================================================

    proptest! {
        /// Property: QuantType roundtrip via u8
        #[test]
        fn prop_quant_type_roundtrip(qt in arb_quant_type()) {
            let value = qt as u8;
            let parsed = QuantType::from_u8(value);
            prop_assert_eq!(parsed, Some(qt));
        }

        /// Property: Invalid QuantType values return None
        #[test]
        fn prop_invalid_quant_type_none(value in 4u8..0xFE) {
            // Skip defined values: 0x01, 0x02, 0x03, 0x10, 0xFF
            if value == 0x10 {
                return Ok(());
            }
            let parsed = QuantType::from_u8(value);
            prop_assert!(parsed.is_none());
        }

        /// Property: bits_per_weight is always positive for valid types
        #[test]
        fn prop_bits_per_weight_positive(qt in arb_quant_type()) {
            prop_assert!(qt.bits_per_weight() > 0.0);
        }

        // ================================================================
        // Q8_0 Quantization Property Tests
        // ================================================================

        /// Property: Q8_0 quantization preserves element count
        #[test]
        fn prop_q8_0_preserves_count(shape in arb_shape()) {
            let len: usize = shape.iter().product();
            let data: Vec<f32> = (0..len).map(|i| i as f32 * 0.1).collect();

            let quantized = Q8_0Quantizer.quantize(&data, &shape).expect("quantize");
            let dequantized = Q8_0Quantizer.dequantize(&quantized).expect("dequantize");

            prop_assert_eq!(dequantized.len(), data.len());
        }

        /// Property: Q8_0 block count is ceiling division
        #[test]
        fn prop_q8_0_block_count(len in 1usize..500) {
            let data: Vec<f32> = vec![1.0; len];
            let shape = vec![len];

            let quantized = Q8_0Quantizer.quantize(&data, &shape).expect("quantize");
            let expected_blocks = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;

            prop_assert_eq!(quantized.num_blocks(), expected_blocks);
        }

        /// Property: Q8_0 quantized size matches block count
        #[test]
        fn prop_q8_0_size_matches_blocks(len in 1usize..500) {
            let data: Vec<f32> = vec![1.0; len];
            let shape = vec![len];

            let quantized = Q8_0Quantizer.quantize(&data, &shape).expect("quantize");

            prop_assert_eq!(quantized.blocks.len(), quantized.num_blocks() * Q8_0_BLOCK_BYTES);
        }

        /// Property: Q8_0 roundtrip error is bounded (MSE < 0.1 for normalized data)
        #[test]
        fn prop_q8_0_error_bounded(
            len in 32usize..200,
            scale in 0.01f32..10.0
        ) {
            let data: Vec<f32> = (0..len).map(|i| (i as f32 / len as f32 - 0.5) * scale).collect();
            let shape = vec![len];

            let quantized = Q8_0Quantizer.quantize(&data, &shape).expect("quantize");
            let dequantized = Q8_0Quantizer.dequantize(&quantized).expect("dequantize");

            let mse = quantization_mse(&data, &dequantized);
            // Q8_0 should have very low error for normalized data
            prop_assert!(mse < scale * scale * 0.01, "MSE {} too high for scale {}", mse, scale);
        }

        /// Property: Q8_0 zeros stay approximately zero
        #[test]
        fn prop_q8_0_zeros(len in 1usize..100) {
            let data: Vec<f32> = vec![0.0; len];
            let shape = vec![len];

            let quantized = Q8_0Quantizer.quantize(&data, &shape).expect("quantize");
            let dequantized = Q8_0Quantizer.dequantize(&quantized).expect("dequantize");

            for val in &dequantized {
                prop_assert!(val.abs() < 0.001, "Expected ~0, got {}", val);
            }
        }

        /// Property: Q8_0 compression ratio is approximately 3.76x (full blocks only)
        #[test]
        fn prop_q8_0_compression_ratio(blocks in 2usize..16) {
            // Use multiples of BLOCK_SIZE to avoid padding effects
            let len = blocks * BLOCK_SIZE;
            let data: Vec<f32> = vec![1.0; len];
            let shape = vec![len];

            let quantized = Q8_0Quantizer.quantize(&data, &shape).expect("quantize");
            let ratio = quantized.compression_ratio();

            // f32 (4 bytes) -> Q8_0 (8.5 bits/weight = 1.0625 bytes/weight)
            // Expected ratio: 4 / 1.0625 ≈ 3.76
            prop_assert!(ratio > 3.5 && ratio < 4.0, "Ratio {} out of expected range", ratio);
        }

        // ================================================================
        // Q4_0 Quantization Property Tests
        // ================================================================

        /// Property: Q4_0 quantization preserves element count
        #[test]
        fn prop_q4_0_preserves_count(shape in arb_shape()) {
            let len: usize = shape.iter().product();
            let data: Vec<f32> = (0..len).map(|i| i as f32 * 0.1).collect();

            let quantized = Q4_0Quantizer.quantize(&data, &shape).expect("quantize");
            let dequantized = Q4_0Quantizer.dequantize(&quantized).expect("dequantize");

            prop_assert_eq!(dequantized.len(), data.len());
        }

        /// Property: Q4_0 block count is ceiling division
        #[test]
        fn prop_q4_0_block_count(len in 1usize..500) {
            let data: Vec<f32> = vec![1.0; len];
            let shape = vec![len];

            let quantized = Q4_0Quantizer.quantize(&data, &shape).expect("quantize");
            let expected_blocks = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;

            prop_assert_eq!(quantized.num_blocks(), expected_blocks);
        }

        /// Property: Q4_0 quantized size matches block count
        #[test]
        fn prop_q4_0_size_matches_blocks(len in 1usize..500) {
            let data: Vec<f32> = vec![1.0; len];
            let shape = vec![len];

            let quantized = Q4_0Quantizer.quantize(&data, &shape).expect("quantize");

            prop_assert_eq!(quantized.blocks.len(), quantized.num_blocks() * Q4_0_BLOCK_BYTES);
        }

        /// Property: Q4_0 compression ratio is approximately 7.1x (full blocks only)
        #[test]
        fn prop_q4_0_compression_ratio(blocks in 2usize..16) {
            // Use multiples of BLOCK_SIZE to avoid padding effects
            let len = blocks * BLOCK_SIZE;
            let data: Vec<f32> = vec![1.0; len];
            let shape = vec![len];

            let quantized = Q4_0Quantizer.quantize(&data, &shape).expect("quantize");
            let ratio = quantized.compression_ratio();

            // f32 (4 bytes) -> Q4_0 (4.5 bits/weight = 0.5625 bytes/weight)
            // Expected ratio: 4 / 0.5625 ≈ 7.1
            prop_assert!(ratio > 6.5 && ratio < 7.5, "Ratio {} out of expected range", ratio);
        }

        /// Property: Q4_0 zeros stay approximately zero
        #[test]
        fn prop_q4_0_zeros(len in 1usize..100) {
            let data: Vec<f32> = vec![0.0; len];
            let shape = vec![len];

            let quantized = Q4_0Quantizer.quantize(&data, &shape).expect("quantize");
            let dequantized = Q4_0Quantizer.dequantize(&quantized).expect("dequantize");

            for val in &dequantized {
                prop_assert!(val.abs() < 0.01, "Expected ~0, got {}", val);
            }
        }

        // ================================================================
        // Cross-Quantizer Property Tests
        // ================================================================

        /// Property: Shape is preserved through quantization
        #[test]
        fn prop_shape_preserved(qt in arb_quant_type(), shape in arb_shape()) {
            let len: usize = shape.iter().product();
            let data: Vec<f32> = vec![1.0; len];

            let quantized = quantize(&data, &shape, qt).expect("quantize");
            prop_assert_eq!(&quantized.shape, &shape);
        }

        /// Property: num_elements matches shape product
        #[test]
        fn prop_num_elements(qt in arb_quant_type(), shape in arb_shape()) {
            let len: usize = shape.iter().product();
            let data: Vec<f32> = vec![1.0; len];

            let quantized = quantize(&data, &shape, qt).expect("quantize");
            prop_assert_eq!(quantized.num_elements(), len);
        }

        /// Property: original_size_bytes is 4x num_elements
        #[test]
        fn prop_original_size_bytes(qt in arb_quant_type(), len in 1usize..200) {
            let data: Vec<f32> = vec![1.0; len];
            let shape = vec![len];

            let quantized = quantize(&data, &shape, qt).expect("quantize");
            prop_assert_eq!(quantized.original_size_bytes(), len * 4);
        }

        // ================================================================
        // MSE Helper Property Tests
        // ================================================================

        /// Property: MSE of identical vectors is 0
        #[test]
        fn prop_mse_identical(data in proptest::collection::vec(-10.0f32..10.0, 1..100)) {
            let mse = quantization_mse(&data, &data);
            prop_assert!(mse.abs() < 1e-10, "Expected 0, got {}", mse);
        }

        /// Property: MSE is symmetric
        #[test]
        fn prop_mse_symmetric(
            a in proptest::collection::vec(-10.0f32..10.0, 1..50),
            offset in -1.0f32..1.0
        ) {
            let b: Vec<f32> = a.iter().map(|x| x + offset).collect();

            let mse_ab = quantization_mse(&a, &b);
            let mse_ba = quantization_mse(&b, &a);

            prop_assert!((mse_ab - mse_ba).abs() < 1e-6, "MSE not symmetric: {} vs {}", mse_ab, mse_ba);
        }

        /// Property: MSE is non-negative
        #[test]
        fn prop_mse_nonnegative(
            a in proptest::collection::vec(-10.0f32..10.0, 1..50),
            b in proptest::collection::vec(-10.0f32..10.0, 1..50)
        ) {
            if a.len() != b.len() {
                return Ok(());
            }
            let mse = quantization_mse(&a, &b);
            prop_assert!(mse >= 0.0 || mse.is_nan(), "MSE is negative: {}", mse);
        }
    }
}

/// Falsification tests per spec v3.0.0 Section BB (Quantization)
#[cfg(test)]
mod tests_falsification_bb {
    use super::*;

    /// BB1: Q4_0 round-trip reconstruction error must be <5%
    /// Falsification: If error >5%, quantization is lossy beyond acceptable threshold
    #[test]
    fn test_bb1_q4_0_roundtrip_error_under_5_percent() {
        // Generate realistic weight distribution (normal-ish around 0)
        let data: Vec<f32> = (0..1024)
            .map(|i| {
                let x = (i as f32 - 512.0) / 512.0; // Range [-1, 1]
                x * 0.1 // Small weights typical in neural nets
            })
            .collect();
        let shape = vec![1024];

        let quantized = quantize(&data, &shape, QuantType::Q4_0).expect("quantize failed");
        let dequantized = dequantize(&quantized).expect("dequantize failed");

        // Calculate relative error
        let mut total_sq_error = 0.0_f64;
        let mut total_sq_orig = 0.0_f64;
        for (orig, deq) in data.iter().zip(dequantized.iter()) {
            total_sq_error += ((*orig - *deq) as f64).powi(2);
            total_sq_orig += (*orig as f64).powi(2);
        }

        let relative_error = if total_sq_orig > 0.0 {
            (total_sq_error / total_sq_orig).sqrt()
        } else {
            0.0
        };

        assert!(
            relative_error < 0.05,
            "BB1 FALSIFIED: Q4_0 relative error {:.2}% exceeds 5% threshold",
            relative_error * 100.0
        );
    }

    /// BB3: Quantization must be deterministic
    /// Falsification: Same input produces different output
    #[test]
    fn test_bb3_quantization_deterministic() {
        let data: Vec<f32> = (0..128)
            .map(|i| (i as f32 - 64.0) * 0.01)
            .collect();
        let shape = vec![128];

        // Run quantization 10 times
        let mut results: Vec<Vec<u8>> = Vec::new();
        for _ in 0..10 {
            let quantized = quantize(&data, &shape, QuantType::Q8_0).expect("quantize");
            results.push(quantized.blocks.clone());
        }

        // All results must be identical
        let first = &results[0];
        for (i, result) in results.iter().enumerate().skip(1) {
            assert_eq!(
                first, result,
                "BB3 FALSIFIED: Quantization run {} differs from run 0",
                i
            );
        }
    }

    /// BB3b: Q4_0 quantization must also be deterministic
    #[test]
    fn test_bb3_q4_0_deterministic() {
        let data: Vec<f32> = (0..128)
            .map(|i| (i as f32 - 64.0) * 0.01)
            .collect();
        let shape = vec![128];

        let q1 = quantize(&data, &shape, QuantType::Q4_0).expect("quantize 1");
        let q2 = quantize(&data, &shape, QuantType::Q4_0).expect("quantize 2");

        assert_eq!(
            q1.blocks, q2.blocks,
            "BB3 FALSIFIED: Q4_0 quantization is non-deterministic"
        );
    }

    /// BB4: Block size must be 32 elements (GGUF compatibility)
    /// Falsification: Non-32 block size is accepted
    #[test]
    fn test_bb4_block_size_is_32() {
        assert_eq!(
            BLOCK_SIZE, 32,
            "BB4 FALSIFIED: Block size is {} instead of 32",
            BLOCK_SIZE
        );
    }

    /// BB4b: Verify quantized blocks use correct size
    #[test]
    fn test_bb4_quantized_block_size_correct() {
        let data: Vec<f32> = vec![1.0; 64]; // 2 blocks
        let shape = vec![64];

        let quantized = quantize(&data, &shape, QuantType::Q8_0).expect("quantize");

        assert_eq!(
            quantized.block_size, 32,
            "BB4 FALSIFIED: QuantizedBlock has wrong block_size: {}",
            quantized.block_size
        );
    }

    /// BB5: Scale factors must be stored and applied correctly
    /// Falsification: dequant(quant(x)) != x / scale (approximately)
    #[test]
    fn test_bb5_scale_factors_correct() {
        // Use known values to verify scale calculation
        let data: Vec<f32> = vec![127.0; 32]; // Max value for Q8_0
        let shape = vec![32];

        let quantized = quantize(&data, &shape, QuantType::Q8_0).expect("quantize");

        // Extract scale from block (first 2 bytes as f16)
        let scale_bytes = [quantized.blocks[0], quantized.blocks[1]];
        let scale = half::f16::from_le_bytes(scale_bytes).to_f32();

        // Scale should be max_abs / 127 = 127 / 127 = 1.0
        assert!(
            (scale - 1.0).abs() < 0.01,
            "BB5 FALSIFIED: Scale {:.4} != expected 1.0",
            scale
        );

        // Dequantized values should match original
        let dequantized = dequantize(&quantized).expect("dequantize");
        for (i, (orig, deq)) in data.iter().zip(dequantized.iter()).enumerate() {
            assert!(
                (orig - deq).abs() < 0.5,
                "BB5 FALSIFIED: Element {} differs: {} vs {}",
                i,
                orig,
                deq
            );
        }
    }

    /// BB6: Verify mixed quantization doesn't corrupt data
    /// (Test that Q8 and Q4 can coexist in same workflow)
    #[test]
    fn test_bb6_mixed_quantization_no_corruption() {
        let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let shape = vec![64];

        // Quantize same data with both methods
        let q8 = quantize(&data, &shape, QuantType::Q8_0).expect("Q8_0");
        let q4 = quantize(&data, &shape, QuantType::Q4_0).expect("Q4_0");

        // Both should dequantize without error
        let d8 = dequantize(&q8).expect("dequantize Q8_0");
        let d4 = dequantize(&q4).expect("dequantize Q4_0");

        assert_eq!(d8.len(), data.len(), "BB6 FALSIFIED: Q8_0 length mismatch");
        assert_eq!(d4.len(), data.len(), "BB6 FALSIFIED: Q4_0 length mismatch");

        // Q8_0 should be more accurate than Q4_0
        let mse8 = quantization_mse(&data, &d8);
        let mse4 = quantization_mse(&data, &d4);

        assert!(
            mse8 < mse4,
            "BB6 FALSIFIED: Q8_0 MSE ({}) should be less than Q4_0 MSE ({})",
            mse8,
            mse4
        );
    }
}
