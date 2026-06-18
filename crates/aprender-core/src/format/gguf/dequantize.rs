
/// Dequantize `Q6_K` format (K-quants)
/// `Q6_K`: super blocks of 256 elements
/// Each super block: ql (128 bytes) + qh (64 bytes) + scales (16 bytes) + d (f16) = 210 bytes
///
/// Delegates to `trueno_quant::dequantize_q6_k_to_f32` — the single source of truth.
#[ensures(ret.as_ref().map_or(true, |v| v.len() == num_elements))]
pub(crate) fn dequantize_q6_k(data: &[u8], start: usize, num_elements: usize) -> Result<Vec<f32>> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 210;

    let num_blocks = num_elements.div_ceil(SUPER_BLOCK_SIZE);
    let total_bytes = num_blocks * SUPER_BLOCK_BYTES;

    if start + total_bytes > data.len() {
        return Err(AprenderError::FormatError {
            message: "Q6_K data exceeds file size".to_string(),
        });
    }

    Ok(trueno_quant::dequantize_q6_k_to_f32(
        &data[start..],
        num_elements,
    ))
}

/// Dequantize `Q4_1` format
/// `Q4_1`: blocks of 32 elements, each block has f16 scale + f16 min + 16 bytes of 4-bit quants
///
/// PMAT-231 FIX: Element order matches llama.cpp/GGML layout:
/// - Low nibbles first (elements 0-15)
/// - High nibbles second (elements 16-31)
#[ensures(ret.as_ref().map_or(true, |v| v.len() == num_elements))]
pub fn dequantize_q4_1(data: &[u8], start: usize, num_elements: usize) -> Result<Vec<f32>> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 2 + 2 + 16; // f16 scale + f16 min + 16 bytes

    let num_blocks = num_elements.div_ceil(BLOCK_SIZE);
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
#[ensures(ret.as_ref().map_or(true, |v| v.len() == num_elements))]
pub(crate) fn dequantize_q2_k(data: &[u8], start: usize, num_elements: usize) -> Result<Vec<f32>> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 2 + 2 + 16 + 64; // d, dmin, scales, qs = 84 bytes

    let num_blocks = num_elements.div_ceil(SUPER_BLOCK_SIZE);
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

        // ggml `dequantize_row_q2_K` ordering (mirrors candle BlockQ2K::to_float
        // and llama.cpp ggml-quants.c): two groups of 128 elements, each over a
        // 32-byte qs window. Within a group, 4 sub-iterations at shift 0/2/4/6,
        // each consuming TWO scale bytes — one for the window's low 16 bytes and
        // one for its high 16 bytes. The previous "16 sub-blocks reading
        // qs[j*4+l]" scheme applied the wrong scale to the wrong 2-bit lanes and
        // produced corrupt output (185/256 elements wrong vs ggml).
        let mut is = 0usize;
        for group in 0..2 {
            let chunk = &qs[group * 32..group * 32 + 32];
            let mut shift = 0u8;
            for _ in 0..4 {
                let sc = scales_bytes[is];
                is += 1;
                let dl = d * f32::from(sc & 0x0F);
                let ml = dmin * f32::from(sc >> 4);
                for &q in &chunk[0..16] {
                    result.push(dl * f32::from((q >> shift) & 0x03) - ml);
                }
                let sc = scales_bytes[is];
                is += 1;
                let dl = d * f32::from(sc & 0x0F);
                let ml = dmin * f32::from(sc >> 4);
                for &q in &chunk[16..32] {
                    result.push(dl * f32::from((q >> shift) & 0x03) - ml);
                }
                shift += 2;
            }
        }
    }

    result.truncate(num_elements);
    Ok(result)
}

/// Dequantize `Q3_K` format (K-quants)
/// `Q3_K`: super blocks of 256 elements
#[ensures(ret.as_ref().map_or(true, |v| v.len() == num_elements))]
pub(crate) fn dequantize_q3_k(data: &[u8], start: usize, num_elements: usize) -> Result<Vec<f32>> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 32 + 64 + 12 + 2; // hmask, qs, scales, d = 110 bytes

    let num_blocks = num_elements.div_ceil(SUPER_BLOCK_SIZE);
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

        // Reconstruct the 16 SIX-bit signed scales from the 12 packed bytes via the GGML
        // aux[] shuffle (kmask1/kmask2). The previous code read only the low/high nibbles of
        // the first 8 bytes — 16 FOUR-bit values with a -8 offset — clipping every scale from
        // [-32,31] to [-8,7] and dropping the upper 2 bits, so ~252/256 elements were wrong.
        const KMASK1: u32 = 0x0303_0303;
        const KMASK2: u32 = 0x0f0f_0f0f;
        let mut aux = [
            u32::from_le_bytes([scales_bytes[0], scales_bytes[1], scales_bytes[2], scales_bytes[3]]),
            u32::from_le_bytes([scales_bytes[4], scales_bytes[5], scales_bytes[6], scales_bytes[7]]),
            u32::from_le_bytes([
                scales_bytes[8],
                scales_bytes[9],
                scales_bytes[10],
                scales_bytes[11],
            ]),
            0u32,
        ];
        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
        aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
        let mut scales = [0i8; 16];
        for (w, word) in aux.iter().enumerate() {
            for (k, &b) in word.to_le_bytes().iter().enumerate() {
                scales[w * 4 + k] = b as i8; // 6-bit value; the -32 offset is applied below
            }
        }

        // Dequantize: two 128-element halves (qs advances 32 bytes per half); within a half,
        // four 32-element shift-blocks (shift = 0,2,4,6) each carrying two 16-lane scale
        // groups; the high-mask bit `m` advances per shift-block. value = d*(scale-32)*(low-high)
        // where high = 4 when the hmask bit is CLEAR, else 0.
        let mut block_out = [0.0f32; 256];
        let mut m: u32 = 1;
        let mut is = 0usize;
        for half in 0..2 {
            let qs_half = &qs[half * 32..half * 32 + 32];
            let out_half = half * 128;
            let mut shift: u32 = 0;
            for blk in 0..4 {
                let out_blk = out_half + blk * 32;
                for scale_index in 0..2 {
                    let dl = d * (f32::from(scales[is]) - 32.0);
                    let out_grp = out_blk + scale_index * 16;
                    for i in 0..16 {
                        let idx = i + 16 * scale_index;
                        let low = ((qs_half[idx] >> shift) & 3) as i8;
                        let high = if u32::from(hmask[idx]) & m == 0 { 4i8 } else { 0i8 };
                        block_out[out_grp + i] = dl * f32::from(low - high);
                    }
                    is += 1;
                }
                shift += 2;
                m <<= 1;
            }
        }
        result.extend_from_slice(&block_out);
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

    let (bits_per_element, block_size): (usize, usize) = match dtype {
        13..=15 => (2, 256), // IQ2_XXS, IQ2_XS, IQ2_S
        16 | 17 => (3, 256), // IQ3_XXS, IQ3_S
        18 => (1, 256),      // IQ1_S
        _ => (4, 256),       // IQ4_NL, IQ4_XS, and default
    };

    let bytes_per_block = (block_size * bits_per_element).div_ceil(8) + 4; // data + scale overhead
    let num_blocks = num_elements.div_ceil(block_size);

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
