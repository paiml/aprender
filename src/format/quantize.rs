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

include!("quantize_part_02.rs");
include!("quantize_part_03.rs");
include!("quantize_part_04.rs");
