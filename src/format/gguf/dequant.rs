//! GGUF dequantization kernels
//!
//! K-quant formats (Q4_K, Q5_K, Q6_K) delegate to `trueno_quant` — the single
//! source of truth in the Sovereign AI Stack.  Legacy GGML formats (Q4_0, Q8_0,
//! Q5_0, Q5_1, Q4_1, Q2_K, Q3_K, IQ*) have no trueno equivalent and keep their
//! inline implementations.

use crate::error::{AprenderError, Result};

/// Convert F16 (IEEE 754 half-precision) to F32.
///
/// Delegates to `trueno_quant::f16_to_f32` which uses the `half` crate —
/// the industry-standard implementation.
///
/// Contract: f16-conversion-v1, equation "f16_to_f32_bias"
#[provable_contracts_macros::contract("f16-conversion-v1", equation = "f16_to_f32_bias")]
pub(crate) fn f16_to_f32(bits: u16) -> f32 {
    trueno_quant::f16_to_f32(bits)
}

/// Convert F16 to F32 with NaN/Inf/subnormal clamping for use as scale factors.
///
/// GH-186 FIX: GGUF files may contain f16 scale values that are NaN, Inf, or
/// subnormal (e.g., from corrupted files or edge-case quantization). When these
/// are used as multipliers in dequantization, they propagate NaN throughout the
/// entire tensor. This function clamps such values to 0.0, matching the safe
/// behavior in `converter/mod.rs::dequantize_q4_k_to_f32`.
#[inline]
fn safe_f16_scale(bits: u16) -> f32 {
    // PMAT-238: Only clamp NaN/Inf, NOT subnormals. Subnormal f16 values are
    // valid scale factors for quantized blocks with very small weights.
    let val = f16_to_f32(bits);
    if val.is_nan() || val.is_infinite() {
        0.0
    } else {
        val
    }
}

/// Dequantize `Q4_0` format
/// `Q4_0`: blocks of 32 elements, each block has 2-byte f16 scale + 16 bytes of 4-bit quants
///
/// PMAT-231 FIX: Element order matches llama.cpp/GGML layout:
/// - Low nibbles first (elements 0-15): byte[0]&0xF, byte[1]&0xF, ..., byte[15]&0xF
/// - High nibbles second (elements 16-31): byte[0]>>4, byte[1]>>4, ..., byte[15]>>4
///
/// This was previously wrong (interleaved: low0, high0, low1, high1, ...) which
/// caused APR inference to produce garbage output for Q4_0 quantized models.
pub fn dequantize_q4_0(data: &[u8], start: usize, num_elements: usize) -> Result<Vec<f32>> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 2 + 16; // f16 scale + 16 bytes of 4-bit values

    let num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let total_bytes = num_blocks * BLOCK_BYTES;

    if start + total_bytes > data.len() {
        return Err(AprenderError::FormatError {
            message: "Q4_0 data exceeds file size".to_string(),
        });
    }

    let mut result = Vec::with_capacity(num_elements);
    let mut offset = start;

    for _ in 0..num_blocks {
        // Read scale (f16)
        // GH-186 FIX: Use safe_f16_scale to clamp NaN/Inf/subnormal
        let scale_bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
        let scale = safe_f16_scale(scale_bits);
        offset += 2;

        // PMAT-231: Low nibbles first (elements 0-15), matching GGML/llama.cpp layout
        for i in 0..16 {
            let byte = data[offset + i];
            let v0 = f32::from((byte & 0x0F) as i8 - 8);
            result.push(v0 * scale);
        }

        // PMAT-231: High nibbles second (elements 16-31)
        for i in 0..16 {
            let byte = data[offset + i];
            let v1 = f32::from((byte >> 4) as i8 - 8);
            result.push(v1 * scale);
        }

        offset += 16;
    }

    // Truncate to exact element count
    result.truncate(num_elements);
    Ok(result)
}

/// Dequantize `Q8_0` format
/// `Q8_0`: blocks of 32 elements, each block has 2-byte f16 scale + 32 bytes of int8 quants
/// Dequantize Q8_0 data to f32
///
/// Q8_0: blocks of 32 elements, each block has 2-byte f16 scale + 32 int8 values
/// Total: 34 bytes per block
pub fn dequantize_q8_0(data: &[u8], start: usize, num_elements: usize) -> Result<Vec<f32>> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 2 + 32; // f16 scale + 32 bytes of int8 values

    let num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let total_bytes = num_blocks * BLOCK_BYTES;

    if start + total_bytes > data.len() {
        return Err(AprenderError::FormatError {
            message: "Q8_0 data exceeds file size".to_string(),
        });
    }

    let mut result = Vec::with_capacity(num_elements);
    let mut offset = start;

    for _ in 0..num_blocks {
        // Read scale (f16)
        // GH-186 FIX: Use safe_f16_scale to clamp NaN/Inf/subnormal
        let scale_bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
        let scale = safe_f16_scale(scale_bits);
        offset += 2;

        // Read 32 int8 values
        for i in 0..32 {
            let v = f32::from(data[offset + i] as i8);
            result.push(v * scale);
        }
        offset += 32;
    }

    // Truncate to exact element count
    result.truncate(num_elements);
    Ok(result)
}

/// Dequantize `Q5_0` format to f32
///
/// Q5_0: blocks of 32 elements, each block has:
/// - 2-byte f16 scale
/// - 4 bytes of high bits (32 bits = 1 per element)
/// - 16 bytes of low 4-bit values (32 values packed 2 per byte)
///
/// Total: 22 bytes per block
pub fn dequantize_q5_0(data: &[u8], start: usize, num_elements: usize) -> Result<Vec<f32>> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 2 + 4 + 16; // f16 scale + 4 high bits + 16 low nibbles = 22

    let num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let total_bytes = num_blocks * BLOCK_BYTES;

    if start + total_bytes > data.len() {
        return Err(AprenderError::FormatError {
            message: "Q5_0 data exceeds file size".to_string(),
        });
    }

    let mut result = Vec::with_capacity(num_elements);
    let mut offset = start;

    for _ in 0..num_blocks {
        // Read scale (f16)
        // GH-186 FIX: Use safe_f16_scale to clamp NaN/Inf/subnormal
        let scale_bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
        let scale = safe_f16_scale(scale_bits);
        offset += 2;

        // Read 4 bytes of high bits (32 bits, 1 per element)
        let high_bits = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        offset += 4;

        // Read 16 bytes = 32 4-bit low values
        for i in 0..16 {
            let byte = data[offset + i];
            // Extract low 4-bit values
            let low0 = byte & 0x0F;
            let low1 = byte >> 4;

            // Get high bits for these two elements
            let high0 = ((high_bits >> (i * 2)) & 1) as u8;
            let high1 = ((high_bits >> (i * 2 + 1)) & 1) as u8;

            // Combine: 5-bit value = high_bit << 4 | low_4_bits, centered at 16
            let v0 = f32::from(((high0 << 4) | low0) as i8 - 16);
            let v1 = f32::from(((high1 << 4) | low1) as i8 - 16);

            result.push(v0 * scale);
            result.push(v1 * scale);
        }
        offset += 16;
    }

    result.truncate(num_elements);
    Ok(result)
}

/// Dequantize `Q5_1` format
/// `Q5_1`: blocks of 32 elements, each block has:
/// - 2-byte f16 scale
/// - 2-byte f16 min
/// - 4 bytes of high bits (32 bits = 1 per element)
/// - 16 bytes of low 4-bit values (32 values packed 2 per byte)
///
/// Total: 24 bytes per block
pub(crate) fn dequantize_q5_1(data: &[u8], start: usize, num_elements: usize) -> Result<Vec<f32>> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 2 + 2 + 4 + 16; // f16 scale + f16 min + 4 high bits + 16 low nibbles = 24

    let num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let total_bytes = num_blocks * BLOCK_BYTES;

    if start + total_bytes > data.len() {
        return Err(AprenderError::FormatError {
            message: "Q5_1 data exceeds file size".to_string(),
        });
    }

    let mut result = Vec::with_capacity(num_elements);
    let mut offset = start;

    for _ in 0..num_blocks {
        // Read scale (f16) and min (f16)
        // GH-186 FIX: Use safe_f16_scale to clamp NaN/Inf/subnormal
        let scale_bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
        let scale = safe_f16_scale(scale_bits);
        let min_bits = u16::from_le_bytes([data[offset + 2], data[offset + 3]]);
        let min_val = safe_f16_scale(min_bits);
        offset += 4;

        // Read 4 bytes of high bits (32 bits, 1 per element)
        let high_bits = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        offset += 4;

        // Read 16 bytes = 32 4-bit low values
        for i in 0..16 {
            let byte = data[offset + i];
            // Extract low 4-bit values
            let low0 = byte & 0x0F;
            let low1 = byte >> 4;

            // Get high bits for these two elements
            let high0 = ((high_bits >> (i * 2)) & 1) as u8;
            let high1 = ((high_bits >> (i * 2 + 1)) & 1) as u8;

            // Combine: 5-bit value = high_bit << 4 | low_4_bits
            // Q5_1 uses scale * q + min (no centering needed)
            let v0 = f32::from((high0 << 4) | low0);
            let v1 = f32::from((high1 << 4) | low1);

            result.push(v0 * scale + min_val);
            result.push(v1 * scale + min_val);
        }
        offset += 16;
    }

    result.truncate(num_elements);
    Ok(result)
}

/// Dequantize `Q4_K` format (K-quants)
/// `Q4_K`: super blocks of 256 elements
/// Each super block: d (f16) + dmin (f16) + scales (12 bytes) + qs (128 bytes) = 144 bytes
///
/// Delegates to `trueno_quant::dequantize_q4_k_to_f32` — the single source of truth.
pub(crate) fn dequantize_q4_k(data: &[u8], start: usize, num_elements: usize) -> Result<Vec<f32>> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 144;

    let num_blocks = (num_elements + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let total_bytes = num_blocks * SUPER_BLOCK_BYTES;

    if start + total_bytes > data.len() {
        return Err(AprenderError::FormatError {
            message: "Q4_K data exceeds file size".to_string(),
        });
    }

    Ok(trueno_quant::dequantize_q4_k_to_f32(
        &data[start..],
        num_elements,
    ))
}

/// Dequantize `Q5_K` format (K-quants)
/// `Q5_K`: super blocks of 256 elements
/// Each super block: d (f16) + dmin (f16) + scales (12 bytes) + qh (32 bytes) + qs (128 bytes) = 176 bytes
///
/// Delegates to `trueno_quant::dequantize_q5_k_to_f32` — the single source of truth.
pub(crate) fn dequantize_q5_k(data: &[u8], start: usize, num_elements: usize) -> Result<Vec<f32>> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 176;

    let num_blocks = (num_elements + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let total_bytes = num_blocks * SUPER_BLOCK_BYTES;

    if start + total_bytes > data.len() {
        return Err(AprenderError::FormatError {
            message: "Q5_K data exceeds file size".to_string(),
        });
    }

    Ok(trueno_quant::dequantize_q5_k_to_f32(
        &data[start..],
        num_elements,
    ))
}

include!("dequant_part_02.rs");
include!("dequant_part_03.rs");
