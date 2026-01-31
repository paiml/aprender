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
        let scale_bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
        let scale = f16_to_f32(scale_bits);
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
        let scale_bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
        let scale = f16_to_f32(scale_bits);
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
        let scale_bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
        let scale = f16_to_f32(scale_bits);
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
        let scale_bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
        let scale = f16_to_f32(scale_bits);
        let min_bits = u16::from_le_bytes([data[offset + 2], data[offset + 3]]);
        let min_val = f16_to_f32(min_bits);
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
        let d = f16_to_f32(u16::from_le_bytes([data[offset], data[offset + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([data[offset + 2], data[offset + 3]]));
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
        let d = f16_to_f32(u16::from_le_bytes([data[offset], data[offset + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([data[offset + 2], data[offset + 3]]));
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
        let d = f16_to_f32(u16::from_le_bytes([data[offset], data[offset + 1]]));
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
        let scale = f16_to_f32(u16::from_le_bytes([data[offset], data[offset + 1]]));
        let min = f16_to_f32(u16::from_le_bytes([data[offset + 2], data[offset + 3]]));
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
        let d = f16_to_f32(u16::from_le_bytes([data[offset], data[offset + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([data[offset + 2], data[offset + 3]]));
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
        let d = f16_to_f32(u16::from_le_bytes([data[offset], data[offset + 1]]));
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
