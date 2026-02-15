
/// Dequantize `Q6_K` format (K-quants)
/// `Q6_K`: super blocks of 256 elements
/// Each super block: ql (128 bytes) + qh (64 bytes) + scales (16 bytes) + d (f16) = 210 bytes
///
/// Layout matches llama.cpp/ggml `dequantize_row_q6_K`:
/// - Two half-blocks of 128 elements each
/// - For each half-block, 32 iterations produce 4 values each at positions [l, l+32, l+64, l+96]
/// - ql[l] and ql[l+32] provide low 4 bits; qh[l] provides high 2 bits (shifts 0,2,4,6)
/// - 16 scales: [0..7] for first half, [8..15] for second half
///
/// PMAT-238 FIX: Previous implementation had wrong index mapping and qh bit extraction,
/// causing 99.7% of dequantized values to be zero (false positive in contract validator).
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

        // Two half-blocks of 128 elements each (matching llama.cpp layout)
        let mut y = [0.0f32; 256];

        for half in 0..2usize {
            let ql_off = half * 64;
            let qh_off = half * 32;
            let sc_off = half * 8;

            for l in 0..32usize {
                let is = l / 16;
                let ql_lo = ql[ql_off + l];
                let ql_hi = ql[ql_off + l + 32];
                let qh_byte = qh[qh_off + l];

                // Each qh byte provides high 2 bits for 4 values at different bit positions
                let q1 = (i32::from(ql_lo & 0x0F) | (i32::from((qh_byte >> 0) & 3) << 4)) - 32;
                let q2 = (i32::from(ql_hi & 0x0F) | (i32::from((qh_byte >> 2) & 3) << 4)) - 32;
                let q3 = (i32::from(ql_lo >> 4) | (i32::from((qh_byte >> 4) & 3) << 4)) - 32;
                let q4 = (i32::from(ql_hi >> 4) | (i32::from((qh_byte >> 6) & 3) << 4)) - 32;

                let base = half * 128;
                y[base + l] = d * f32::from(scales[sc_off + is] as i8) * q1 as f32;
                y[base + l + 32] = d * f32::from(scales[sc_off + is + 2] as i8) * q2 as f32;
                y[base + l + 64] = d * f32::from(scales[sc_off + is + 4] as i8) * q3 as f32;
                y[base + l + 96] = d * f32::from(scales[sc_off + is + 6] as i8) * q4 as f32;
            }
        }

        result.extend_from_slice(&y);
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
