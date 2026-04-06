//! GGUF K-quant Dequantization Helpers (PMAT-802)
//!
//! Dequantization routines for APR Q4_K/Q6_K support.

// ============================================================================
// GGUF K-quant Dequantization Helpers (for APR Q4_K/Q6_K support)
// =============================================================================

/// Convert IEEE 754 half-precision (f16) bits to f32
///
/// ONE PATH: Delegates to `trueno::f16_to_f32` (UCBD §4).
pub(crate) fn f16_to_f32(bits: u16) -> f32 {
    trueno::f16_to_f32(bits)
}

/// Extract scale and min from Q4_K 12-byte packed scales
///
/// PAR-001 FIX: Matches llama.cpp's get_scale_min_k4 packing scheme:
/// - Blocks 0-3: scale = q[j] & 63, min = q[j+4] & 63
/// - Blocks 4-7: scale = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
///   min = (q[j+4] >> 4) | ((q[j] >> 6) << 4)
#[inline]
pub(crate) fn extract_scale_min_apr(scales: &[u8], block_idx: usize) -> (f32, f32) {
    let j = block_idx;
    let (scale_bits, min_bits) = if j < 4 {
        // First 4 blocks: simple layout
        let d = scales[j] & 63;
        let m = scales[j + 4] & 63;
        (d, m)
    } else {
        // Last 4 blocks: packed layout using high bits from first 4 bytes
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    };

    (f32::from(scale_bits), f32::from(min_bits))
}

/// Dequantize Q4_K format (K-quants) for APR tensors
/// Q4_K: super blocks of 256 elements
/// Each super block: d (f16) + dmin (f16) + scales (12 bytes) + qs (128 bytes) = 144 bytes
///
/// PMAT-086 FIX: Correct implementation matching llama.cpp/candle layout:
/// - For each 64-value chunk, output 32 low nibbles THEN 32 high nibbles
/// - Use sc1/dm1 for low nibbles, sc2/dm2 for high nibbles (different scales per half)
pub(crate) fn dequantize_q4_k_apr(data: &[u8], num_elements: usize) -> Vec<f32> {
    const QK_K: usize = 256; // Super-block size
    const SUPER_BLOCK_BYTES: usize = 2 + 2 + 12 + 128; // 144 bytes

    let num_blocks = num_elements.div_ceil(QK_K);
    let total_bytes = num_blocks * SUPER_BLOCK_BYTES;

    if total_bytes > data.len() {
        // Return zeros if data is insufficient
        return vec![0.0; num_elements];
    }

    let mut result = vec![0.0f32; num_blocks * QK_K];

    for sb_idx in 0..num_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let out_start = sb_idx * QK_K;

        // Read d (f16 scale) and dmin (f16 min)
        let d = f16_to_f32(u16::from_le_bytes([data[sb_start], data[sb_start + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([data[sb_start + 2], data[sb_start + 3]]));

        // Read scales (12 bytes)
        let scales = &data[sb_start + 4..sb_start + 16];

        // Read qs (128 bytes)
        let qs = &data[sb_start + 16..sb_start + 144];

        // Dequantize following candle's layout:
        // For each 64-value chunk, output 32 low nibbles then 32 high nibbles
        let mut ys_index = out_start;

        for j in (0..QK_K).step_by(64) {
            let q = &qs[j / 2..j / 2 + 32];

            // Get scales for the two 32-value halves
            let is = j / 32;
            let (sc1, m1) = extract_scale_min_apr(scales, is);
            let d1 = d * sc1;
            let dm1 = dmin * m1;

            let (sc2, m2) = extract_scale_min_apr(scales, is + 1);
            let d2 = d * sc2;
            let dm2 = dmin * m2;

            // First pass: 32 low nibbles
            for &byte in q {
                result[ys_index] = d1 * (byte & 0xF) as f32 - dm1;
                ys_index += 1;
            }

            // Second pass: 32 high nibbles
            for &byte in q {
                result[ys_index] = d2 * (byte >> 4) as f32 - dm2;
                ys_index += 1;
            }
        }
    }

    result.truncate(num_elements);
    result
}

/// Dequantize Q6_K format (K-quants) for APR tensors
/// Q6_K super-block layout (per llama.cpp block_q6_K and candle):
/// - ql: 128 bytes (low 4 bits, 256 values, 2 per byte)
/// - qh: 64 bytes (high 2 bits, 256 values, 4 per byte)
/// - scales: 16 bytes (i8 signed scales for 16 blocks)
/// - d: 2 bytes (f16)
///
/// Total: 128 + 64 + 16 + 2 = 210 bytes
pub(crate) fn dequantize_q6_k_apr(data: &[u8], num_elements: usize) -> Vec<f32> {
    const QK_K: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 210;

    let num_blocks = num_elements.div_ceil(QK_K);
    let total_bytes = num_blocks * SUPER_BLOCK_BYTES;

    if total_bytes > data.len() {
        return vec![0.0; num_elements];
    }

    let mut result = vec![0.0f32; num_blocks * QK_K];

    for sb_idx in 0..num_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let out_start = sb_idx * QK_K;

        // Read ql - low 4 bits (128 bytes) at offset 0
        let ql = &data[sb_start..sb_start + 128];

        // Read qh - high 2 bits (64 bytes) at offset 128
        let qh = &data[sb_start + 128..sb_start + 192];

        // Read scales (16 bytes, i8) at offset 192
        let mut scales = [0i8; 16];
        #[allow(clippy::cast_possible_wrap)]
        for (i, scale) in scales.iter_mut().enumerate() {
            *scale = data[sb_start + 192 + i] as i8;
        }

        // Read d (f16 -> f32) at offset 208 (last 2 bytes)
        let d = f16_to_f32(u16::from_le_bytes([
            data[sb_start + 208],
            data[sb_start + 209],
        ]));

        // Dequantize 256 values following candle's exact layout
        // Process 128 values at a time (n=0, n=128)
        for n in (0..QK_K).step_by(128) {
            let idx = n / 128;
            let sc = &scales[8 * idx..];
            let ql_slice = &ql[64 * idx..];
            let qh_slice = &qh[32 * idx..];

            for l in 0..32 {
                let is = l / 16; // Scale index selector (0 or 1 within this 128-block)

                // Extract 4 values per iteration (at positions l, l+32, l+64, l+96)
                // q1: low 4 bits of ql[l] + bits 0-1 of qh[l]
                let q1 = ((ql_slice[l] & 0xF) | ((qh_slice[l] & 3) << 4)) as i32 - 32;
                // q2: low 4 bits of ql[l+32] + bits 2-3 of qh[l]
                let q2 = ((ql_slice[l + 32] & 0xF) | (((qh_slice[l] >> 2) & 3) << 4)) as i32 - 32;
                // q3: high 4 bits of ql[l] + bits 4-5 of qh[l]
                let q3 = ((ql_slice[l] >> 4) | (((qh_slice[l] >> 4) & 3) << 4)) as i32 - 32;
                // q4: high 4 bits of ql[l+32] + bits 6-7 of qh[l]
                let q4 = ((ql_slice[l + 32] >> 4) | (((qh_slice[l] >> 6) & 3) << 4)) as i32 - 32;

                // Write to output with correct scale indexing
                result[out_start + n + l] = d * (sc[is] as f32) * (q1 as f32);
                result[out_start + n + l + 32] = d * (sc[is + 2] as f32) * (q2 as f32);
                result[out_start + n + l + 64] = d * (sc[is + 4] as f32) * (q3 as f32);
                result[out_start + n + l + 96] = d * (sc[is + 6] as f32) * (q4 as f32);
            }
        }
    }

    result.truncate(num_elements);
    result
}

/// Dequantize Q8_0 format for APR tensors (GH-191)
///
/// Q8_0 block layout (per GGML):
/// - scale: 2 bytes (f16)
/// - quants: 32 bytes (i8 values)
///   Total: 34 bytes per block, 32 elements per block
pub(crate) fn dequantize_q8_0_apr(data: &[u8], num_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 34; // 2 (f16 scale) + 32 (i8 quants)
    const ELEMENTS_PER_BLOCK: usize = 32;

    let num_blocks = num_elements.div_ceil(ELEMENTS_PER_BLOCK);
    let total_bytes = num_blocks * BLOCK_SIZE;

    if total_bytes > data.len() {
        return vec![0.0; num_elements];
    }

    let mut result = vec![0.0f32; num_blocks * ELEMENTS_PER_BLOCK];

    for blk in 0..num_blocks {
        let blk_start = blk * BLOCK_SIZE;
        let out_start = blk * ELEMENTS_PER_BLOCK;

        // Read scale (f16 → f32)
        let scale = f16_to_f32(u16::from_le_bytes([data[blk_start], data[blk_start + 1]]));

        // Dequantize 32 i8 values
        #[allow(clippy::cast_possible_wrap)]
        for i in 0..ELEMENTS_PER_BLOCK {
            let q = data[blk_start + 2 + i] as i8;
            result[out_start + i] = scale * (q as f32);
        }
    }

    result.truncate(num_elements);
    result
}

/// Dequantize APR-native Q8 format (dtype=9) — GH-239
///
/// APR Q8 layout (per `AprV2Writer::add_q8_tensor`):
/// - scale: 4 bytes (f32, little-endian) — `max_abs / 127.0`
/// - quants: N bytes (i8 values, one per element)
///   Total: 4 + N bytes
///
/// Dequantization: `value = scale * (byte as i8) as f32`
///
/// This is distinct from GGML Q8_0 which uses f16 scale + 32-element blocks.
pub(crate) fn dequantize_apr_q8_native(data: &[u8], num_elements: usize) -> Vec<f32> {
    if data.len() < 4 {
        return vec![0.0; num_elements];
    }
    let scale = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    let mut result = Vec::with_capacity(num_elements);
    #[allow(clippy::cast_possible_wrap)]
    for i in 0..num_elements {
        if 4 + i < data.len() {
            result.push(scale * (data[4 + i] as i8) as f32);
        } else {
            result.push(0.0);
        }
    }
    result
}

/// Dequantize APR-native Q4 format (dtype=8) — GH-239
///
/// APR Q4 layout (per `AprV2Writer::add_q4_tensor`):
/// - Blocks of 32 elements: [f16 scale (2 bytes)] + [16 packed nibble bytes] = 18 bytes/block
/// - Scale computed as `max_abs / 7.0`
/// - Nibble packing: even index → low nibble (byte & 0x0F), odd index → high nibble (byte >> 4)
/// - Dequant: `value = scale * ((nibble as i8) - 8)` (unsigned 0-15 → signed -8..+7)
///
/// This is distinct from GGML Q8_0 (dtype=8 in old APR files) which uses 34 bytes/block.
/// Disambiguation: check `data.len()` against expected byte counts for each format.
pub(crate) fn dequantize_apr_q4_native(data: &[u8], num_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 32;

    let mut result = Vec::with_capacity(num_elements);
    let mut pos = 0;
    let mut remaining = num_elements;

    while remaining > 0 && pos + 2 <= data.len() {
        let scale = f16_to_f32(u16::from_le_bytes([data[pos], data[pos + 1]]));
        pos += 2;
        let count = remaining.min(BLOCK_SIZE);
        #[allow(clippy::cast_possible_wrap)]
        for i in 0..count {
            let byte_idx = pos + i / 2;
            if byte_idx >= data.len() {
                result.push(0.0);
                continue;
            }
            let byte = data[byte_idx];
            let nibble = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            let q = (nibble as i8) - 8;
            result.push(scale * (q as f32));
        }
        pos += 16; // Always 16 nibble bytes per block
        remaining -= count;
    }
    // Pad if data ran out
    result.resize(num_elements, 0.0);
    result
}

include!("dequant_falsify_gh44.rs");
