//! K-Quant dequantization functions (`Q4_K`, `Q5_K`, `Q6_K`)

use crate::f16_to_f32;

/// Dequantize `Q4_K` bytes to F32
#[must_use]
pub fn dequantize_q4_k_to_f32(data: &[u8], num_elements: usize) -> Vec<f32> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 144;

    let num_blocks = num_elements.div_ceil(SUPER_BLOCK_SIZE);
    let mut result = vec![0.0f32; num_blocks * SUPER_BLOCK_SIZE];

    for sb_idx in 0..num_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let out_start = sb_idx * SUPER_BLOCK_SIZE;

        if sb_start + SUPER_BLOCK_BYTES > data.len() {
            break;
        }

        let d = sanitize_f16_scale(data[sb_start], data[sb_start + 1]);
        let dmin = sanitize_f16_scale(data[sb_start + 2], data[sb_start + 3]);

        let (scales, mins) = unpack_q4k_scales(&data[sb_start + 4..sb_start + 16]);
        let qs = &data[sb_start + 16..sb_start + 144];

        dequantize_q4k_block(d, dmin, &scales, &mins, qs, &mut result[out_start..]);
    }

    result.truncate(num_elements);
    result
}

/// Sanitize an f16-encoded scale value: return 0.0 for NaN or infinity.
///
/// BUG-IMPORT-002 FIX: Subnormal F16 values (< 6.1e-5) are VALID in GGML quantization.
/// `Q6_K`/`Q4_K` super-block scales can be 1e-6 to 1e-4 for small-weight tensors.
/// Clamping subnormals to 0.0 destroys >99% of dequantized data, producing
/// all-zero tensors and Q8 density violations downstream.
#[inline]
fn sanitize_f16_scale(lo: u8, hi: u8) -> f32 {
    let raw = f16_to_f32(u16::from_le_bytes([lo, hi]));
    if raw.is_nan() || raw.is_infinite() {
        0.0
    } else {
        raw
    }
}

/// Unpack `Q4_K` 12-byte packed scales into 8 scale + 8 min values.
fn unpack_q4k_scales(scales_bytes: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];
    for i in 0..4 {
        scales[i] = scales_bytes[i] & 0x3F;
        mins[i] = scales_bytes[i + 4] & 0x3F;
        scales[i + 4] = (scales_bytes[i + 8] & 0x0F) | ((scales_bytes[i] >> 6) << 4);
        mins[i + 4] = (scales_bytes[i + 8] >> 4) | ((scales_bytes[i + 4] >> 6) << 4);
    }
    (scales, mins)
}

/// Dequantize one `Q4_K` block (256 values) from packed nibbles.
fn dequantize_q4k_block(
    d: f32,
    dmin: f32,
    scales: &[u8; 8],
    mins: &[u8; 8],
    qs: &[u8],
    output: &mut [f32],
) {
    let mut ys_index = 0;
    for chunk in 0..4 {
        let is = chunk * 2;
        let scale_lo = d * f32::from(scales[is]);
        let min_lo = dmin * f32::from(mins[is]);
        let scale_hi = d * f32::from(scales[is + 1]);
        let min_hi = dmin * f32::from(mins[is + 1]);

        for l in 0..32 {
            let byte = qs[chunk * 32 + l];
            output[ys_index] = scale_lo * f32::from(byte & 0x0F) - min_lo;
            ys_index += 1;
        }
        for l in 0..32 {
            let byte = qs[chunk * 32 + l];
            output[ys_index] = scale_hi * f32::from((byte >> 4) & 0x0F) - min_hi;
            ys_index += 1;
        }
    }
}

/// Dequantize `Q5_K` bytes to F32
#[must_use]
pub fn dequantize_q5_k_to_f32(data: &[u8], num_elements: usize) -> Vec<f32> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 176;

    let num_blocks = num_elements.div_ceil(SUPER_BLOCK_SIZE);
    let mut result = vec![0.0f32; num_blocks * SUPER_BLOCK_SIZE];

    for sb_idx in 0..num_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let out_start = sb_idx * SUPER_BLOCK_SIZE;

        if sb_start + SUPER_BLOCK_BYTES > data.len() {
            break;
        }

        let d = sanitize_f16_scale(data[sb_start], data[sb_start + 1]);
        let dmin = sanitize_f16_scale(data[sb_start + 2], data[sb_start + 3]);

        let scales_bytes = &data[sb_start + 4..sb_start + 16];
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];

        for i in 0..4 {
            scales[i] = scales_bytes[i] & 0x3F;
            mins[i] = scales_bytes[i + 4] & 0x3F;
            scales[i + 4] = (scales_bytes[i + 8] & 0x0F) | ((scales_bytes[i] >> 6) << 4);
            mins[i + 4] = (scales_bytes[i + 8] >> 4) | ((scales_bytes[i + 4] >> 6) << 4);
        }

        let qh = &data[sb_start + 16..sb_start + 48];
        let qs = &data[sb_start + 48..sb_start + 176];

        for j in 0..8 {
            let scale = d * f32::from(scales[j]);
            let min_val = dmin * f32::from(mins[j]);
            for k in 0..32 {
                let idx = j * 32 + k;
                let qs_idx = j * 16 + (k % 16);
                let q_lo = if k < 16 { qs[qs_idx] & 0x0F } else { (qs[qs_idx] >> 4) & 0x0F };
                let q_hi = (qh[k] >> j) & 1;
                let q = q_lo | (q_hi << 4);
                result[out_start + idx] = scale * f32::from(q) - min_val;
            }
        }
    }

    result.truncate(num_elements);
    result
}

/// Dequantize `Q6_K` bytes to F32
#[must_use]
pub fn dequantize_q6_k_to_f32(data: &[u8], num_elements: usize) -> Vec<f32> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 210;

    let num_blocks = num_elements.div_ceil(SUPER_BLOCK_SIZE);
    let mut result = vec![0.0f32; num_blocks * SUPER_BLOCK_SIZE];

    for sb_idx in 0..num_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let out_start = sb_idx * SUPER_BLOCK_SIZE;

        if sb_start + SUPER_BLOCK_BYTES > data.len() {
            break;
        }

        let ql = &data[sb_start..sb_start + 128];
        let qh = &data[sb_start + 128..sb_start + 192];
        let scales = &data[sb_start + 192..sb_start + 208];
        let d = sanitize_f16_scale(data[sb_start + 208], data[sb_start + 209]);

        for half in 0..2 {
            let ql_base = half * 64;
            let qh_base = half * 32;
            let out_base = out_start + half * 128;

            for l in 0..32 {
                let q1_lo = ql[ql_base + l] & 0x0F;
                let q2_lo = ql[ql_base + l + 32] & 0x0F;
                let q3_lo = (ql[ql_base + l] >> 4) & 0x0F;
                let q4_lo = (ql[ql_base + l + 32] >> 4) & 0x0F;

                let qh_byte = qh[qh_base + l];
                let q1_hi = (qh_byte & 0x03) << 4;
                let q2_hi = ((qh_byte >> 2) & 0x03) << 4;
                let q3_hi = ((qh_byte >> 4) & 0x03) << 4;
                let q4_hi = ((qh_byte >> 6) & 0x03) << 4;

                let q1 = (q1_lo | q1_hi) as i8 - 32;
                let q2 = (q2_lo | q2_hi) as i8 - 32;
                let q3 = (q3_lo | q3_hi) as i8 - 32;
                let q4 = (q4_lo | q4_hi) as i8 - 32;

                let scale_idx_1 = (half * 8) + (l / 16);
                let scale_idx_2 = (half * 8) + (l / 16) + 2;
                let scale_idx_3 = (half * 8) + (l / 16) + 4;
                let scale_idx_4 = (half * 8) + (l / 16) + 6;

                let s1 = scales[scale_idx_1] as i8;
                let s2 = scales[scale_idx_2] as i8;
                let s3 = scales[scale_idx_3] as i8;
                let s4 = scales[scale_idx_4] as i8;

                result[out_base + l] = d * f32::from(s1) * f32::from(q1);
                result[out_base + l + 32] = d * f32::from(s2) * f32::from(q2);
                result[out_base + l + 64] = d * f32::from(s3) * f32::from(q3);
                result[out_base + l + 96] = d * f32::from(s4) * f32::from(q4);
            }
        }
    }

    result.truncate(num_elements);
    result
}
