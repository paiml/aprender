//! Dequantization functions for GGUF quantization formats (PMAT-802)
//!
//! Extracted from quantize/mod.rs - Basic dequantization for Q4_0, Q8_0, F16,
//! Q4_1, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K, Q2_K formats.
//!
//! ## Contents
//! - `dequantize_q4_0` - 4-bit quantization (block size 32)
//! - `dequantize_q8_0` - 8-bit quantization (block size 32)
//! - `dequantize_f16` - Half-precision to f32
//! - `dequantize_q4_1` - 4-bit with scale and min
//! - `dequantize_q5_0` - 5-bit quantization
//! - `dequantize_q5_1` - 5-bit with scale and min
//! - `dequantize_q4_k` - K-quantization 4-bit (super-block 256)
//! - `dequantize_q5_k` - K-quantization 5-bit (super-block 256)
//! - `dequantize_q6_k` - K-quantization 6-bit (super-block 256)
//! - `dequantize_q2_k` - K-quantization 2-bit (super-block 256)

use crate::error::{RealizarError, Result};
use crate::quantize::{BLOCK_SIZE, QK_K};

// Re-use helpers from simd module
use super::simd::extract_scale_min;

/// Dequantize `Q4_0` format weights
///
/// # Arguments
///
/// * `data` - Raw `Q4_0` quantized data (blocks of scale + 16 bytes)
///
/// # Returns
///
/// Dequantized float32 values
///
/// # Errors
///
/// Returns error if data length is not a multiple of block size
///
/// # Examples
///
/// ```rust,ignore
/// let quantized = load_q4_0_weights();
/// let weights = dequantize_q4_0(&quantized)?;
/// ```
pub fn dequantize_q4_0(data: &[u8]) -> Result<Vec<f32>> {
    // Q4_0 block: 2 bytes (f16 scale) + 16 bytes (quants) = 18 bytes
    // GGML spec: typedef struct { ggml_half d; uint8_t qs[QK4_0/2]; } block_q4_0;
    const BLOCK_BYTES: usize = 2 + 16;

    if !data.len().is_multiple_of(BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q4_0 data length {} is not a multiple of block size {}",
                data.len(),
                BLOCK_BYTES
            ),
        });
    }

    let num_blocks = data.len() / BLOCK_BYTES;
    let mut result = vec![0.0f32; num_blocks * BLOCK_SIZE];

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_BYTES;
        let out_start = block_idx * BLOCK_SIZE;

        // Read scale (f16, per GGML spec)
        let scale_bytes = &data[block_start..block_start + 2];
        let scale = half::f16::from_le_bytes([scale_bytes[0], scale_bytes[1]]).to_f32();

        // Read quantized values (16 bytes)
        let quants_start = block_start + 2;
        let quants = &data[quants_start..quants_start + 16];

        // Dequantize following candle's layout:
        // - Positions 0-15: low nibbles of bytes 0-15
        // - Positions 16-31: high nibbles of bytes 0-15
        for (j, &byte) in quants.iter().enumerate() {
            // Low 4 bits go to position j
            #[allow(clippy::cast_possible_wrap)]
            let low = (byte & 0x0F) as i16 - 8;
            result[out_start + j] = scale * (low as f32);

            // High 4 bits go to position j + 16
            #[allow(clippy::cast_possible_wrap)]
            let high = (byte >> 4) as i16 - 8;
            result[out_start + j + 16] = scale * (high as f32);
        }
    }

    Ok(result)
}

/// Dequantize `Q8_0` format weights
///
/// # Arguments
///
/// * `data` - Raw `Q8_0` quantized data (blocks of scale + 32 int8 values)
///
/// # Returns
///
/// Dequantized float32 values
///
/// # Errors
///
/// Returns error if data length is not a multiple of block size
pub fn dequantize_q8_0(data: &[u8]) -> Result<Vec<f32>> {
    // Q8_0 block: 2 bytes (f16 scale) + 32 bytes (int8 quants) = 34 bytes
    // Note: GGML spec uses f16 for scale, not f32!
    const BLOCK_BYTES: usize = 2 + 32;

    if !data.len().is_multiple_of(BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q8_0 data length {} is not a multiple of block size {}",
                data.len(),
                BLOCK_BYTES
            ),
        });
    }

    let num_blocks = data.len() / BLOCK_BYTES;
    let mut result = Vec::with_capacity(num_blocks * BLOCK_SIZE);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_BYTES;

        // Read scale (f16 -> f32)
        let scale_bits = u16::from_le_bytes([data[block_start], data[block_start + 1]]);
        let scale = f16_to_f32(scale_bits);

        // Read quantized values (32 int8 values)
        let quants_start = block_start + 2;
        let quants = &data[quants_start..quants_start + 32];

        // Dequantize
        for &byte in quants {
            let value = i8::from_le_bytes([byte]);
            result.push(scale * f32::from(value));
        }
    }

    Ok(result)
}

/// Convert IEEE 754 half-precision (f16) to single-precision (f32)
///
/// ONE PATH: Delegates to `trueno::f16_to_f32` (UCBD §4).
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    trueno::f16_to_f32(h)
}

/// Dequantize `F16` format weights to `F32`
///
/// # Arguments
///
/// * `data` - Raw F16 data (2 bytes per value)
///
/// # Returns
///
/// Dequantized float32 values
///
/// # Errors
///
/// Returns error if data length is not a multiple of 2 bytes
pub fn dequantize_f16(data: &[u8]) -> Result<Vec<f32>> {
    if !data.len().is_multiple_of(2) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "F16 data length {} is not a multiple of 2 bytes",
                data.len()
            ),
        });
    }

    let num_values = data.len() / 2;
    let mut result = Vec::with_capacity(num_values);

    for chunk in data.chunks_exact(2) {
        let h = u16::from_le_bytes([chunk[0], chunk[1]]);
        result.push(f16_to_f32(h));
    }

    Ok(result)
}

/// Dequantize `Q4_1` format weights
///
/// Q4_1 format: 2 bytes (f16 scale) + 2 bytes (f16 min) + 16 bytes (quants) = 20 bytes
/// GGUF/candle layout:
/// - Positions 0-15: low nibbles of bytes 0-15
/// - Positions 16-31: high nibbles of bytes 0-15
pub fn dequantize_q4_1(data: &[u8]) -> Result<Vec<f32>> {
    const BLOCK_BYTES: usize = 20;

    if !data.len().is_multiple_of(BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q4_1 data length {} is not a multiple of block size {}",
                data.len(),
                BLOCK_BYTES
            ),
        });
    }

    let num_blocks = data.len() / BLOCK_BYTES;
    // Pre-allocate with correct size for candle layout
    let mut result = vec![0.0f32; num_blocks * BLOCK_SIZE];

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_BYTES;
        let out_start = block_idx * BLOCK_SIZE;

        let d_bytes = &data[block_start..block_start + 2];
        let d = f16_to_f32(u16::from_le_bytes([d_bytes[0], d_bytes[1]]));

        let min_bytes = &data[block_start + 2..block_start + 4];
        let min = f16_to_f32(u16::from_le_bytes([min_bytes[0], min_bytes[1]]));

        let quants = &data[block_start + 4..block_start + 20];

        // Use candle layout (same as Q4_0):
        // - Low nibbles (byte & 0xF) at positions 0-15
        // - High nibbles (byte >> 4) at positions 16-31
        for (j, &byte) in quants.iter().enumerate() {
            // Low 4 bits go to position j (0-15)
            let low = byte & 0x0F;
            result[out_start + j] = d * f32::from(low) + min;

            // High 4 bits go to position j + 16 (16-31)
            let high = (byte >> 4) & 0x0F;
            result[out_start + j + 16] = d * f32::from(high) + min;
        }
    }

    Ok(result)
}

/// Dequantize `Q5_0` format weights
///
/// Q5_0 format: 2 bytes (f16 scale) + 4 bytes (high bits) + 16 bytes (quants) = 22 bytes
/// GGUF/candle layout:
/// - Positions 0-15: low nibbles + high bits from qh
/// - Positions 16-31: high nibbles + high bits from qh
pub fn dequantize_q5_0(data: &[u8]) -> Result<Vec<f32>> {
    const BLOCK_BYTES: usize = 22;

    if !data.len().is_multiple_of(BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q5_0 data length {} is not a multiple of block size {}",
                data.len(),
                BLOCK_BYTES
            ),
        });
    }

    let num_blocks = data.len() / BLOCK_BYTES;
    // Pre-allocate with correct size for candle layout
    let mut result = vec![0.0f32; num_blocks * BLOCK_SIZE];

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_BYTES;
        let out_start = block_idx * BLOCK_SIZE;

        let d_bytes = &data[block_start..block_start + 2];
        let d = f16_to_f32(u16::from_le_bytes([d_bytes[0], d_bytes[1]]));

        let qh = u32::from_le_bytes([
            data[block_start + 2],
            data[block_start + 3],
            data[block_start + 4],
            data[block_start + 5],
        ]);

        let qs = &data[block_start + 6..block_start + 22];

        // Use candle layout:
        // - Low nibbles (byte & 0xF) at positions 0-15
        // - High nibbles (byte >> 4) at positions 16-31
        for (i, &byte) in qs.iter().enumerate() {
            // Low 4 bits + 5th bit go to position i (0-15)
            let low_q = byte & 0x0F;
            let high_bit_low = ((qh >> i) & 1) as u8;
            let q_low = low_q | (high_bit_low << 4);
            #[allow(clippy::cast_possible_wrap)]
            let value_low = q_low as i8 - 16;
            result[out_start + i] = d * f32::from(value_low);

            // High 4 bits + 5th bit go to position i + 16 (16-31)
            let high_q = (byte >> 4) & 0x0F;
            let high_bit_high = ((qh >> (i + 16)) & 1) as u8;
            let q_high = high_q | (high_bit_high << 4);
            #[allow(clippy::cast_possible_wrap)]
            let value_high = q_high as i8 - 16;
            result[out_start + i + 16] = d * f32::from(value_high);
        }
    }

    Ok(result)
}

/// Dequantize `Q5_1` format weights
///
/// Q5_1 format: 2 bytes (f16 scale) + 2 bytes (f16 min) + 4 bytes (high bits) + 16 bytes (quants) = 24 bytes
/// GGUF/candle layout:
/// - Positions 0-15: low nibbles + high bits from qh
/// - Positions 16-31: high nibbles + high bits from qh
pub fn dequantize_q5_1(data: &[u8]) -> Result<Vec<f32>> {
    const BLOCK_BYTES: usize = 24;

    if !data.len().is_multiple_of(BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q5_1 data length {} is not a multiple of block size {}",
                data.len(),
                BLOCK_BYTES
            ),
        });
    }

    let num_blocks = data.len() / BLOCK_BYTES;
    // Pre-allocate with correct size for candle layout
    let mut result = vec![0.0f32; num_blocks * BLOCK_SIZE];

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_BYTES;
        let out_start = block_idx * BLOCK_SIZE;

        let d_bytes = &data[block_start..block_start + 2];
        let d = f16_to_f32(u16::from_le_bytes([d_bytes[0], d_bytes[1]]));

        let min_bytes = &data[block_start + 2..block_start + 4];
        let min = f16_to_f32(u16::from_le_bytes([min_bytes[0], min_bytes[1]]));

        let qh = u32::from_le_bytes([
            data[block_start + 4],
            data[block_start + 5],
            data[block_start + 6],
            data[block_start + 7],
        ]);

        let qs = &data[block_start + 8..block_start + 24];

        // Use candle layout:
        // - Low nibbles (byte & 0xF) at positions 0-15
        // - High nibbles (byte >> 4) at positions 16-31
        for (i, &byte) in qs.iter().enumerate() {
            // Low 4 bits + 5th bit go to position i (0-15)
            let low_q = byte & 0x0F;
            let high_bit_low = ((qh >> i) & 1) as u8;
            let q_low = low_q | (high_bit_low << 4);
            result[out_start + i] = d * f32::from(q_low) + min;

            // High 4 bits + 5th bit go to position i + 16 (16-31)
            let high_q = (byte >> 4) & 0x0F;
            let high_bit_high = ((qh >> (i + 16)) & 1) as u8;
            let q_high = high_q | (high_bit_high << 4);
            result[out_start + i + 16] = d * f32::from(q_high) + min;
        }
    }

    Ok(result)
}

include!("dequant_q4k.rs");
include!("dequant_f16.rs");
