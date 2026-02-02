//! GGUF dequantization kernels

use crate::error::{AprenderError, Result};

/// Convert F16 (IEEE 754 half-precision) to F32
pub(crate) fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from((bits >> 15) & 1);
    let exp = u32::from((bits >> 10) & 0x1F);
    let mant = u32::from(bits & 0x3FF);

    if exp == 0 {
        if mant == 0 {
            // Zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal - convert to normalized f32
            let mut m = mant;
            let mut e = 0i32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            let f32_exp = (127 - 15 + 1 + e) as u32;
            f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13))
        }
    } else if exp == 31 {
        // Inf or NaN
        if mant == 0 {
            f32::from_bits((sign << 31) | (0xFF << 23))
        } else {
            f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13))
        }
    } else {
        // Normal number
        let f32_exp = (exp as i32 - 15 + 127) as u32;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (mant << 13))
    }
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
    const F16_MIN_NORMAL: f32 = 6.1e-5;
    let val = f16_to_f32(bits);
    if val.is_nan() || val.is_infinite() || val.abs() < F16_MIN_NORMAL {
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
pub(crate) fn dequantize_q4_k(data: &[u8], start: usize, num_elements: usize) -> Result<Vec<f32>> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 2 + 2 + 12 + 128; // 144 bytes

    let num_blocks = (num_elements + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let total_bytes = num_blocks * SUPER_BLOCK_BYTES;

    if start + total_bytes > data.len() {
        return Err(AprenderError::FormatError {
            message: "Q4_K data exceeds file size".to_string(),
        });
    }

    let mut result = Vec::with_capacity(num_elements);
    let mut offset = start;

    for _ in 0..num_blocks {
        // Read d (f16 scale) and dmin (f16 min)
        // GH-186 FIX: Use safe_f16_scale to clamp NaN/Inf/subnormal
        let d = safe_f16_scale(u16::from_le_bytes([data[offset], data[offset + 1]]));
        let dmin = safe_f16_scale(u16::from_le_bytes([data[offset + 2], data[offset + 3]]));
        offset += 4;

        // Read scales (12 bytes = 8 6-bit values packed)
        // scales[0..7] are the 8 sub-block scales
        let scales_bytes = &data[offset..offset + 12];
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];

        // Unpack 6-bit scales and mins from 12 bytes (llama.cpp Q4_K format)
        // The packing is:
        //   bytes 0-3: lower 6 bits of scales[0-3]
        //   bytes 4-7: lower 6 bits of mins[0-3]
        //   bytes 8-11: combined upper 2 bits for scales[4-7] and mins[4-7]
        //
        // For j < 4:  scale = q[j] & 63,     min = q[j+4] & 63
        // For j >= 4: scale = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
        //             min   = (q[j+4] >> 4) | ((q[j] >> 6) << 4)
        //
        // F-REGR-231 FIX: Previous implementation was wrong for scales[4-7] and mins[4-7]
        for i in 0..4 {
            // j < 4: direct 6-bit extraction
            scales[i] = scales_bytes[i] & 0x3F;
            mins[i] = scales_bytes[i + 4] & 0x3F;
        }
        for i in 0..4 {
            // j >= 4: combine lower 4 bits from bytes 8-11 with upper 2 bits from bytes 0-3/4-7
            scales[i + 4] = (scales_bytes[i + 8] & 0x0F) | ((scales_bytes[i] >> 6) << 4);
            mins[i + 4] = (scales_bytes[i + 8] >> 4) | ((scales_bytes[i + 4] >> 6) << 4);
        }
        offset += 12;

        // Read 128 bytes = 256 4-bit quantized values
        let qs = &data[offset..offset + 128];
        offset += 128;

        // Dequantize: each sub-block has 32 elements
        for j in 0..8 {
            let scale = d * f32::from(scales[j]);
            let min_val = dmin * f32::from(mins[j]);

            for l in 0..16 {
                let q_byte = qs[j * 16 + l];
                let q0 = f32::from(q_byte & 0x0F);
                let q1 = f32::from(q_byte >> 4);
                result.push(q0 * scale - min_val);
                result.push(q1 * scale - min_val);
            }
        }
    }

    result.truncate(num_elements);
    Ok(result)
}

/// Dequantize `Q5_K` format (K-quants)
/// `Q5_K`: super blocks of 256 elements
/// Each super block: d (f16) + dmin (f16) + scales (12 bytes) + qh (32 bytes) + qs (128 bytes) = 176 bytes
pub(crate) fn dequantize_q5_k(data: &[u8], start: usize, num_elements: usize) -> Result<Vec<f32>> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 2 + 2 + 12 + 32 + 128; // 176 bytes

    let num_blocks = (num_elements + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let total_bytes = num_blocks * SUPER_BLOCK_BYTES;

    if start + total_bytes > data.len() {
        return Err(AprenderError::FormatError {
            message: "Q5_K data exceeds file size".to_string(),
        });
    }

    let mut result = Vec::with_capacity(num_elements);
    let mut offset = start;

    for _ in 0..num_blocks {
        // Read d and dmin
        // GH-186 FIX: Use safe_f16_scale to clamp NaN/Inf/subnormal
        let d = safe_f16_scale(u16::from_le_bytes([data[offset], data[offset + 1]]));
        let dmin = safe_f16_scale(u16::from_le_bytes([data[offset + 2], data[offset + 3]]));
        offset += 4;

        // Read and unpack scales (same as Q4_K)
        let scales_bytes = &data[offset..offset + 12];
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];

        // Unpack 6-bit scales and mins from 12 bytes (llama.cpp Q5_K format - same as Q4_K)
        // F-REGR-231 FIX: Use correct llama.cpp unpacking
        for i in 0..4 {
            // j < 4: direct 6-bit extraction
            scales[i] = scales_bytes[i] & 0x3F;
            mins[i] = scales_bytes[i + 4] & 0x3F;
        }
        for i in 0..4 {
            // j >= 4: combine lower 4 bits from bytes 8-11 with upper 2 bits from bytes 0-3/4-7
            scales[i + 4] = (scales_bytes[i + 8] & 0x0F) | ((scales_bytes[i] >> 6) << 4);
            mins[i + 4] = (scales_bytes[i + 8] >> 4) | ((scales_bytes[i + 4] >> 6) << 4);
        }
        offset += 12;

        // Read qh (32 bytes = 256 high bits)
        let qh = &data[offset..offset + 32];
        offset += 32;

        // Read qs (128 bytes = 256 low 4-bit values)
        let qs = &data[offset..offset + 128];
        offset += 128;

        // Dequantize
        for j in 0..8 {
            let scale = d * f32::from(scales[j]);
            let min_val = dmin * f32::from(mins[j]);

            for l in 0..16 {
                let idx = j * 16 + l;
                let q_byte = qs[idx];
                let qh_byte = qh[idx / 8];
                let bit_pos = (idx % 8) as u8;

                let q0 = f32::from((q_byte & 0x0F) | (((qh_byte >> bit_pos) & 1) << 4));
                let q1 = f32::from((q_byte >> 4) | ((((qh_byte >> bit_pos) >> 1) & 1) << 4));

                result.push(q0 * scale - min_val);
                result.push(q1 * scale - min_val);
            }
        }
    }

    result.truncate(num_elements);
    Ok(result)
}

/// Dequantize `Q6_K` format (K-quants)
/// `Q6_K`: super blocks of 256 elements
/// Each super block: ql (128 bytes) + qh (64 bytes) + scales (16 bytes) + d (f16) = 210 bytes
pub(crate) fn dequantize_q6_k(data: &[u8], start: usize, num_elements: usize) -> Result<Vec<f32>> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 128 + 64 + 16 + 2; // 210 bytes

    let num_blocks = (num_elements + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let total_bytes = num_blocks * SUPER_BLOCK_BYTES;

    if start + total_bytes > data.len() {
        return Err(AprenderError::FormatError {
            message: "Q6_K data exceeds file size".to_string(),
        });
    }

    let mut result = Vec::with_capacity(num_elements);
    let mut offset = start;

    for _ in 0..num_blocks {
        // Read ql (128 bytes = low 4 bits of 256 6-bit values)
        let ql = &data[offset..offset + 128];
        offset += 128;

        // Read qh (64 bytes = high 2 bits of 256 6-bit values)
        let qh = &data[offset..offset + 64];
        offset += 64;

        // Read scales (16 bytes = 16 int8 scales)
        let scales = &data[offset..offset + 16];
        offset += 16;

        // Read d (f16)
        // GH-186 FIX: Use safe_f16_scale to clamp NaN/Inf/subnormal
        let d = safe_f16_scale(u16::from_le_bytes([data[offset], data[offset + 1]]));
        offset += 2;

        // Dequantize 16 sub-blocks of 16 elements each
        for j in 0..16 {
            let scale = d * f32::from(scales[j] as i8);

            for l in 0..8 {
                let idx = j * 8 + l;
                let ql_byte = ql[idx];
                let qh_byte = qh[idx / 2];

                // Extract two 6-bit values
                let qh_shift = (l % 2) * 4;
                let q0 = ((ql_byte & 0x0F) | ((qh_byte >> qh_shift) & 0x03) << 4) as i8 - 32;
                let q1 = ((ql_byte >> 4) | (((qh_byte >> qh_shift) >> 2) & 0x03) << 4) as i8 - 32;

                result.push(f32::from(q0) * scale);
                result.push(f32::from(q1) * scale);
            }
        }
    }

    result.truncate(num_elements);
    Ok(result)
}

/// Dequantize `Q4_1` format
/// `Q4_1`: blocks of 32 elements, each block has f16 scale + f16 min + 16 bytes of 4-bit quants
///
/// PMAT-231 FIX: Element order matches llama.cpp/GGML layout:
/// - Low nibbles first (elements 0-15)
/// - High nibbles second (elements 16-31)
pub fn dequantize_q4_1(data: &[u8], start: usize, num_elements: usize) -> Result<Vec<f32>> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 2 + 2 + 16; // f16 scale + f16 min + 16 bytes

    let num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let total_bytes = num_blocks * BLOCK_BYTES;

    if start + total_bytes > data.len() {
        return Err(AprenderError::FormatError {
            message: "Q4_1 data exceeds file size".to_string(),
        });
    }

    let mut result = Vec::with_capacity(num_elements);
    let mut offset = start;

    for _ in 0..num_blocks {
        // GH-186 FIX: Use safe_f16_scale to clamp NaN/Inf/subnormal
        let scale = safe_f16_scale(u16::from_le_bytes([data[offset], data[offset + 1]]));
        let min = safe_f16_scale(u16::from_le_bytes([data[offset + 2], data[offset + 3]]));
        offset += 4;

        // PMAT-231: Low nibbles first (elements 0-15)
        for i in 0..16 {
            let byte = data[offset + i];
            let v0 = f32::from(byte & 0x0F) * scale + min;
            result.push(v0);
        }

        // PMAT-231: High nibbles second (elements 16-31)
        for i in 0..16 {
            let byte = data[offset + i];
            let v1 = f32::from(byte >> 4) * scale + min;
            result.push(v1);
        }

        offset += 16;
    }

    result.truncate(num_elements);
    Ok(result)
}

/// Dequantize `Q2_K` format (K-quants)
/// `Q2_K`: super blocks of 256 elements
pub(crate) fn dequantize_q2_k(data: &[u8], start: usize, num_elements: usize) -> Result<Vec<f32>> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 2 + 2 + 16 + 64; // d, dmin, scales, qs = 84 bytes

    let num_blocks = (num_elements + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let total_bytes = num_blocks * SUPER_BLOCK_BYTES;

    if start + total_bytes > data.len() {
        return Err(AprenderError::FormatError {
            message: "Q2_K data exceeds file size".to_string(),
        });
    }

    let mut result = Vec::with_capacity(num_elements);
    let mut offset = start;

    for _ in 0..num_blocks {
        // Read scales (16 bytes = 16 4-bit scale/min pairs)
        let scales_bytes = &data[offset..offset + 16];
        offset += 16;

        // Read qs (64 bytes = 256 2-bit values)
        let qs = &data[offset..offset + 64];
        offset += 64;

        // Read d and dmin
        // GH-186 FIX: Use safe_f16_scale to clamp NaN/Inf/subnormal
        let d = safe_f16_scale(u16::from_le_bytes([data[offset], data[offset + 1]]));
        let dmin = safe_f16_scale(u16::from_le_bytes([data[offset + 2], data[offset + 3]]));
        offset += 4;

        // Dequantize 16 sub-blocks of 16 elements
        for j in 0..16 {
            let sc_byte = scales_bytes[j];
            let scale = d * f32::from(sc_byte & 0x0F);
            let min_val = dmin * f32::from(sc_byte >> 4);

            for l in 0..4 {
                let q_byte = qs[j * 4 + l];
                for k in 0..4 {
                    let q = (q_byte >> (k * 2)) & 0x03;
                    result.push(f32::from(q) * scale - min_val);
                }
            }
        }
    }

    result.truncate(num_elements);
    Ok(result)
}

/// Dequantize `Q3_K` format (K-quants)
/// `Q3_K`: super blocks of 256 elements
pub(crate) fn dequantize_q3_k(data: &[u8], start: usize, num_elements: usize) -> Result<Vec<f32>> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 32 + 64 + 12 + 2; // hmask, qs, scales, d = 110 bytes

    let num_blocks = (num_elements + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let total_bytes = num_blocks * SUPER_BLOCK_BYTES;

    if start + total_bytes > data.len() {
        return Err(AprenderError::FormatError {
            message: "Q3_K data exceeds file size".to_string(),
        });
    }

    let mut result = Vec::with_capacity(num_elements);
    let mut offset = start;

    for _ in 0..num_blocks {
        // Read hmask (32 bytes = 256 high bits)
        let hmask = &data[offset..offset + 32];
        offset += 32;

        // Read qs (64 bytes = 256 low 2-bit values)
        let qs = &data[offset..offset + 64];
        offset += 64;

        // Read scales (12 bytes = packed 6-bit scales)
        let scales_bytes = &data[offset..offset + 12];
        offset += 12;

        // Read d
        // GH-186 FIX: Use safe_f16_scale to clamp NaN/Inf/subnormal
        let d = safe_f16_scale(u16::from_le_bytes([data[offset], data[offset + 1]]));
        offset += 2;

        // Unpack scales
        let mut scales = [0i8; 16];
        for i in 0..8 {
            scales[i] = (scales_bytes[i] & 0x0F) as i8 - 8;
            scales[i + 8] = (scales_bytes[i] >> 4) as i8 - 8;
        }

        // Dequantize
        for j in 0..256 {
            let sub_block = j / 16;
            let q_idx = j / 4;
            let q_shift = (j % 4) * 2;
            let h_idx = j / 8;
            let h_shift = j % 8;

            let q_low = (qs[q_idx] >> q_shift) & 0x03;
            let q_high = ((hmask[h_idx] >> h_shift) & 1) << 2;
            let q = (q_low | q_high) as i8 - 4;

            result.push(d * f32::from(scales[sub_block]) * f32::from(q));
        }
    }

    result.truncate(num_elements);
    Ok(result)
}

/// Approximate dequantization for I-quants (IQ2, IQ3, IQ4)
/// These use importance-weighted quantization with lookup tables.
/// For import purposes, we approximate with a simple linear mapping.
pub(crate) fn dequantize_iq_approximate(
    data: &[u8],
    start: usize,
    num_elements: usize,
    dtype: u32,
) -> Vec<f32> {
    // I-quants have variable block sizes and complex lookup tables
    // Approximate by treating as low-bit quantization with estimated scale

    let (bits_per_element, block_size) = match dtype {
        13..=15 => (2, 256), // IQ2_XXS, IQ2_XS, IQ2_S
        16 | 17 => (3, 256), // IQ3_XXS, IQ3_S
        18 => (1, 256),      // IQ1_S
        _ => (4, 256),       // IQ4_NL, IQ4_XS, and default
    };

    let bytes_per_block = (block_size * bits_per_element + 7) / 8 + 4; // data + scale overhead
    let num_blocks = (num_elements + block_size - 1) / block_size;

    // For approximation, create small random-ish values based on byte patterns
    // This is NOT correct dequantization but allows import to proceed
    let mut result = Vec::with_capacity(num_elements);
    let scale = 0.01; // Small scale for approximate values

    for block_idx in 0..num_blocks {
        let block_start = start + block_idx * bytes_per_block;

        for i in 0..block_size {
            if result.len() >= num_elements {
                break;
            }

            // Use byte pattern to generate approximate value
            let byte_idx = block_start + (i * bits_per_element) / 8;
            if byte_idx < data.len() {
                let byte_val = data[byte_idx];
                // Map to roughly centered distribution
                let approx = (f32::from(byte_val) - 128.0) * scale;
                result.push(approx);
            } else {
                result.push(0.0);
            }
        }
    }

    result.truncate(num_elements);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // f16_to_f32 edge cases
    // =========================================================================

    #[test]
    fn test_f16_positive_zero() {
        let result = f16_to_f32(0x0000);
        assert_eq!(result, 0.0);
        assert!(result.is_sign_positive());
    }

    #[test]
    fn test_f16_negative_zero() {
        let result = f16_to_f32(0x8000);
        assert_eq!(result, 0.0);
        assert!(result.is_sign_negative());
    }

    #[test]
    fn test_f16_positive_infinity() {
        let result = f16_to_f32(0x7C00);
        assert!(result.is_infinite());
        assert!(result.is_sign_positive());
    }

    #[test]
    fn test_f16_negative_infinity() {
        let result = f16_to_f32(0xFC00);
        assert!(result.is_infinite());
        assert!(result.is_sign_negative());
    }

    #[test]
    fn test_f16_nan() {
        // NaN has exp=31 and non-zero mantissa
        let result = f16_to_f32(0x7C01);
        assert!(result.is_nan());
    }

    #[test]
    fn test_f16_subnormal() {
        // Smallest subnormal: 0x0001 = 2^-24 ~= 5.96e-8
        let result = f16_to_f32(0x0001);
        assert!(result > 0.0);
        assert!(result < 1e-4);
    }

    #[test]
    fn test_f16_normal_one() {
        // f16 representation of 1.0 = 0x3C00
        let result = f16_to_f32(0x3C00);
        assert!((result - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_f16_normal_negative() {
        // f16 representation of -1.0 = 0xBC00
        let result = f16_to_f32(0xBC00);
        assert!((result - (-1.0)).abs() < 1e-6);
    }

    // =========================================================================
    // Dequantize Q4_0
    // =========================================================================

    #[test]
    fn test_dequantize_q4_0_basic() {
        // Build a minimal Q4_0 block: 2 bytes scale + 16 bytes data = 18 bytes per block of 32
        let mut data = vec![0u8; 18];
        // Scale = 1.0 in f16 = 0x3C00
        data[0] = 0x00;
        data[1] = 0x3C;
        // Fill 16 quant bytes with 0x88 (both nibbles = 8, so value = 8-8 = 0)
        for i in 2..18 {
            data[i] = 0x88;
        }

        let result = dequantize_q4_0(&data, 0, 32).expect("should succeed");
        assert_eq!(result.len(), 32);
        // All values should be 0 (quant value 8 - 8 = 0, scaled by 1.0)
        for v in &result {
            assert!(v.abs() < 1e-6, "Expected ~0.0 but got {v}");
        }
    }

    #[test]
    fn test_dequantize_q4_0_exceeds_file_size() {
        let data = vec![0u8; 10]; // Too small for 1 block (needs 18 bytes)
        let result = dequantize_q4_0(&data, 0, 32);
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q4_0_partial_block() {
        // Request fewer elements than a full block
        let mut data = vec![0u8; 18];
        data[0] = 0x00;
        data[1] = 0x3C;
        let result = dequantize_q4_0(&data, 0, 16).expect("should succeed");
        assert_eq!(result.len(), 16);
    }

    // =========================================================================
    // Dequantize Q8_0
    // =========================================================================

    #[test]
    fn test_dequantize_q8_0_basic() {
        // Build Q8_0 block: 2 bytes scale + 32 bytes data = 34 bytes
        let mut data = vec![0u8; 34];
        // Scale = 1.0 in f16
        data[0] = 0x00;
        data[1] = 0x3C;
        // All quant bytes = 0 (int8 value 0)
        let result = dequantize_q8_0(&data, 0, 32).expect("should succeed");
        assert_eq!(result.len(), 32);
        for v in &result {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn test_dequantize_q8_0_exceeds_file_size() {
        let data = vec![0u8; 10];
        let result = dequantize_q8_0(&data, 0, 32);
        assert!(result.is_err());
    }

    // =========================================================================
    // Dequantize Q5_0
    // =========================================================================

    #[test]
    fn test_dequantize_q5_0_basic() {
        // Q5_0 block: 2 + 4 + 16 = 22 bytes
        let data = vec![0u8; 22];
        let result = dequantize_q5_0(&data, 0, 32).expect("should succeed");
        assert_eq!(result.len(), 32);
    }

    #[test]
    fn test_dequantize_q5_0_exceeds_file_size() {
        let data = vec![0u8; 10];
        let result = dequantize_q5_0(&data, 0, 32);
        assert!(result.is_err());
    }

    // =========================================================================
    // Dequantize Q5_1
    // =========================================================================

    #[test]
    fn test_dequantize_q5_1_basic() {
        // Q5_1 block: 2 + 2 + 4 + 16 = 24 bytes
        let data = vec![0u8; 24];
        let result = dequantize_q5_1(&data, 0, 32).expect("should succeed");
        assert_eq!(result.len(), 32);
    }

    #[test]
    fn test_dequantize_q5_1_exceeds_file_size() {
        let data = vec![0u8; 10];
        let result = dequantize_q5_1(&data, 0, 32);
        assert!(result.is_err());
    }

    // =========================================================================
    // Dequantize Q4_K
    // =========================================================================

    #[test]
    fn test_dequantize_q4_k_basic() {
        // Q4_K block: 2 + 2 + 12 + 128 = 144 bytes
        let data = vec![0u8; 144];
        let result = dequantize_q4_k(&data, 0, 256).expect("should succeed");
        assert_eq!(result.len(), 256);
    }

    #[test]
    fn test_dequantize_q4_k_exceeds_file_size() {
        let data = vec![0u8; 100];
        let result = dequantize_q4_k(&data, 0, 256);
        assert!(result.is_err());
    }

    // =========================================================================
    // Dequantize Q5_K
    // =========================================================================

    #[test]
    fn test_dequantize_q5_k_basic() {
        // Q5_K block: 2 + 2 + 12 + 32 + 128 = 176 bytes
        let data = vec![0u8; 176];
        let result = dequantize_q5_k(&data, 0, 256).expect("should succeed");
        assert_eq!(result.len(), 256);
    }

    #[test]
    fn test_dequantize_q5_k_exceeds_file_size() {
        let data = vec![0u8; 100];
        let result = dequantize_q5_k(&data, 0, 256);
        assert!(result.is_err());
    }

    // =========================================================================
    // Dequantize Q6_K
    // =========================================================================

    #[test]
    fn test_dequantize_q6_k_basic() {
        // Q6_K block: 128 + 64 + 16 + 2 = 210 bytes
        let data = vec![0u8; 210];
        let result = dequantize_q6_k(&data, 0, 256).expect("should succeed");
        assert_eq!(result.len(), 256);
    }

    #[test]
    fn test_dequantize_q6_k_exceeds_file_size() {
        let data = vec![0u8; 100];
        let result = dequantize_q6_k(&data, 0, 256);
        assert!(result.is_err());
    }

    // =========================================================================
    // Dequantize Q4_1
    // =========================================================================

    #[test]
    fn test_dequantize_q4_1_basic() {
        // Q4_1 block: 2 + 2 + 16 = 20 bytes
        let data = vec![0u8; 20];
        let result = dequantize_q4_1(&data, 0, 32).expect("should succeed");
        assert_eq!(result.len(), 32);
    }

    #[test]
    fn test_dequantize_q4_1_exceeds_file_size() {
        let data = vec![0u8; 10];
        let result = dequantize_q4_1(&data, 0, 32);
        assert!(result.is_err());
    }

    // =========================================================================
    // Dequantize Q2_K
    // =========================================================================

    #[test]
    fn test_dequantize_q2_k_basic() {
        // Q2_K block: 2 + 2 + 16 + 64 = 84 bytes
        let data = vec![0u8; 84];
        let result = dequantize_q2_k(&data, 0, 256).expect("should succeed");
        assert_eq!(result.len(), 256);
    }

    #[test]
    fn test_dequantize_q2_k_exceeds_file_size() {
        let data = vec![0u8; 50];
        let result = dequantize_q2_k(&data, 0, 256);
        assert!(result.is_err());
    }

    // =========================================================================
    // Dequantize Q3_K
    // =========================================================================

    #[test]
    fn test_dequantize_q3_k_basic() {
        // Q3_K block: 32 + 64 + 12 + 2 = 110 bytes
        let data = vec![0u8; 110];
        let result = dequantize_q3_k(&data, 0, 256).expect("should succeed");
        assert_eq!(result.len(), 256);
    }

    #[test]
    fn test_dequantize_q3_k_exceeds_file_size() {
        let data = vec![0u8; 50];
        let result = dequantize_q3_k(&data, 0, 256);
        assert!(result.is_err());
    }

    // =========================================================================
    // Dequantize IQ approximate
    // =========================================================================

    #[test]
    fn test_dequantize_iq_approximate_iq2() {
        let data = vec![128u8; 1024];
        let result = dequantize_iq_approximate(&data, 0, 64, 13); // IQ2_XXS
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_dequantize_iq_approximate_iq3() {
        let data = vec![128u8; 1024];
        let result = dequantize_iq_approximate(&data, 0, 64, 16); // IQ3_XXS
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_dequantize_iq_approximate_iq1() {
        let data = vec![128u8; 1024];
        let result = dequantize_iq_approximate(&data, 0, 64, 18); // IQ1_S
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_dequantize_iq_approximate_default_dtype() {
        let data = vec![128u8; 1024];
        let result = dequantize_iq_approximate(&data, 0, 64, 99); // Unknown dtype
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_dequantize_iq_approximate_byte_out_of_range() {
        // Small data so byte_idx exceeds data.len()
        let data = vec![128u8; 4];
        let result = dequantize_iq_approximate(&data, 0, 256, 13);
        // Should still produce 256 elements (some will be 0.0 for out-of-range bytes)
        assert_eq!(result.len(), 256);
        // Verify some elements are 0.0 (from the byte_idx >= data.len() path)
        assert!(result.iter().any(|&v| v == 0.0));
    }

    #[test]
    fn test_dequantize_q4_0_with_nonzero_start() {
        let mut data = vec![0u8; 36]; // 18 bytes padding + 18 bytes block
                                      // Put scale at offset 18
        data[18] = 0x00;
        data[19] = 0x3C;
        let result = dequantize_q4_0(&data, 18, 32).expect("should succeed");
        assert_eq!(result.len(), 32);
    }

    // =========================================================================
    // GH-186: safe_f16_scale NaN/Inf/subnormal clamping
    // =========================================================================

    #[test]
    fn test_safe_f16_scale_normal() {
        // 1.0 in f16 = 0x3C00
        assert!((safe_f16_scale(0x3C00) - 1.0).abs() < 1e-3);
        // 2.0 in f16 = 0x4000
        assert!((safe_f16_scale(0x4000) - 2.0).abs() < 1e-3);
        // -1.0 in f16 = 0xBC00
        assert!((safe_f16_scale(0xBC00) - (-1.0)).abs() < 1e-3);
    }

    #[test]
    fn test_safe_f16_scale_nan_clamped() {
        // NaN in f16: exp=31, mantissa!=0 → 0x7E00
        assert_eq!(safe_f16_scale(0x7E00), 0.0);
        // Another NaN pattern
        assert_eq!(safe_f16_scale(0x7C01), 0.0);
    }

    #[test]
    fn test_safe_f16_scale_inf_clamped() {
        // +Inf in f16: 0x7C00
        assert_eq!(safe_f16_scale(0x7C00), 0.0);
        // -Inf in f16: 0xFC00
        assert_eq!(safe_f16_scale(0xFC00), 0.0);
    }

    #[test]
    fn test_safe_f16_scale_subnormal_clamped() {
        // Smallest subnormal: 0x0001 → ~5.96e-8 (well below F16_MIN_NORMAL = 6.1e-5)
        assert_eq!(safe_f16_scale(0x0001), 0.0);
        // Largest subnormal: 0x03FF → still below F16_MIN_NORMAL
        assert_eq!(safe_f16_scale(0x03FF), 0.0);
    }

    #[test]
    fn test_safe_f16_scale_zero_preserved() {
        // Positive zero
        assert_eq!(safe_f16_scale(0x0000), 0.0);
        // Negative zero - abs(0.0) < F16_MIN_NORMAL so clamped to 0.0
        assert_eq!(safe_f16_scale(0x8000), 0.0);
    }

    #[test]
    fn test_gh186_nan_does_not_propagate_q4_0() {
        // Build a Q4_0 block with NaN scale (0x7E00)
        let mut data = vec![0u8; 18]; // 2-byte scale + 16-byte quants
        data[0] = 0x00; // NaN f16 = 0x7E00 (little-endian: 0x00, 0x7E)
        data[1] = 0x7E;
        // Fill quants with non-zero data
        for i in 2..18 {
            data[i] = 0x55; // non-zero nibbles
        }
        let result = dequantize_q4_0(&data, 0, 32).expect("should succeed");
        // With NaN clamping, all values should be finite (0.0 * anything = 0.0)
        assert!(
            result.iter().all(|v| v.is_finite()),
            "GH-186: NaN scale should not propagate to output"
        );
    }

    #[test]
    fn test_gh186_nan_does_not_propagate_q4_k() {
        // Build a Q4_K block with NaN scale
        // Q4_K block: 4 bytes (d+dmin) + 12 bytes (scales) + 128 bytes (quants) = 144 bytes
        let mut data = vec![0u8; 144];
        data[0] = 0x00; // d = NaN f16 = 0x7E00 (LE)
        data[1] = 0x7E;
        data[2] = 0x00; // dmin = NaN
        data[3] = 0x7E;
        // Fill scales and quants with non-zero
        for i in 4..144 {
            data[i] = 0x33;
        }
        let result = dequantize_q4_k(&data, 0, 256).expect("should succeed");
        assert!(
            result.iter().all(|v| v.is_finite()),
            "GH-186: NaN scale should not propagate to Q4_K output"
        );
    }
}
