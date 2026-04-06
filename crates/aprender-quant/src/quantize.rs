//! K-Quant quantization functions (`Q4_K`, `Q5_K`, `Q6_K`)
//!
//! Shared helpers and all quantize functions extracted from lib.rs.

use crate::{f32_to_f16, F16_MIN_NORMAL};

// ============================================================================
// Shared K-Quant Helpers (extracted for cognitive complexity reduction)
// ============================================================================

/// Compute per-sub-block scale and min values from padded data.
///
/// Returns (`sub_scales`, `sub_mins`) for 8 sub-blocks of 32 elements each.
/// `quant_max` is the maximum quantized value (15 for `Q4_K`, 31 for `Q5_K`).
pub(crate) fn compute_sub_block_stats(padded: &[f32; 256], quant_max: f32) -> ([f32; 8], [f32; 8]) {
    const SUB_BLOCK_SIZE: usize = 32;
    let mut sub_scales = [0.0f32; 8];
    let mut sub_mins = [0.0f32; 8];

    for (j, sub_block) in padded.chunks(SUB_BLOCK_SIZE).enumerate().take(8) {
        let min = sub_block.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = sub_block.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let range = max - min;

        sub_scales[j] = if range > F16_MIN_NORMAL {
            range / quant_max
        } else {
            F16_MIN_NORMAL
        };
        sub_mins[j] = (-min).max(0.0);
    }

    (sub_scales, sub_mins)
}

/// Compute global d and dmin from sub-block statistics, plus quantized 6-bit scales/mins.
pub(crate) fn compute_global_scales(
    sub_scales: &[f32; 8],
    sub_mins: &[f32; 8],
) -> (f32, f32, [u8; 8], [u8; 8]) {
    let max_scale = sub_scales.iter().fold(0.0f32, |a, &b| a.max(b));
    let max_min = sub_mins.iter().fold(0.0f32, |a, &b| a.max(b));

    let d = if max_scale > F16_MIN_NORMAL {
        max_scale / 63.0
    } else {
        F16_MIN_NORMAL
    };
    let dmin = if max_min > F16_MIN_NORMAL {
        max_min / 63.0
    } else {
        F16_MIN_NORMAL
    };

    let mut scales_6bit = [0u8; 8];
    let mut mins_6bit = [0u8; 8];
    for j in 0..8 {
        scales_6bit[j] = ((sub_scales[j] / d).round() as u8).min(63);
        mins_6bit[j] = ((sub_mins[j] / dmin).round() as u8).min(63);
    }

    (d, dmin, scales_6bit, mins_6bit)
}

/// Write the K-quant header: d (f16) + dmin (f16) + packed 12-byte scales.
pub(crate) fn write_kquant_header(
    result: &mut Vec<u8>,
    d: f32,
    dmin: f32,
    scales_6bit: &[u8; 8],
    mins_6bit: &[u8; 8],
) {
    result.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    result.extend_from_slice(&f32_to_f16(dmin).to_le_bytes());

    let mut scales_packed = [0u8; 12];
    for i in 0..4 {
        scales_packed[i] = (scales_6bit[i] & 0x3F) | ((scales_6bit[i + 4] & 0x30) << 2);
        scales_packed[i + 4] = (mins_6bit[i] & 0x3F) | ((mins_6bit[i + 4] & 0x30) << 2);
    }
    for i in 0..4 {
        scales_packed[i + 8] = (scales_6bit[i + 4] & 0x0F) | ((mins_6bit[i + 4] & 0x0F) << 4);
    }
    result.extend_from_slice(&scales_packed);
}

/// Quantize a single value: (value + `min_val`) / scale, clamped to [0, `max_q`].
#[inline]
pub(crate) fn quantize_one(value: f32, min_val: f32, scale: f32, max_q: f32) -> u8 {
    if scale > 1e-10 {
        ((value + min_val) / scale).round().clamp(0.0, max_q) as u8
    } else {
        0
    }
}

// ============================================================================
// Q4_K Quantization
// ============================================================================

/// Quantize F32 data to `Q4_K` format (llama.cpp/candle compatible)
///
/// `Q4_K` format: 256 elements per super-block, 144 bytes per block
/// Layout: d (2B) + dmin (2B) + scales (12B) + qs (128B)
///
/// Value packing (candle/llama.cpp layout):
/// - For each 64-value chunk: 32 bytes store low nibbles first, then high nibbles
/// - Low nibbles use scale[is], high nibbles use scale[is+1]
#[must_use]
pub fn quantize_q4_k(data: &[f32]) -> Vec<u8> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 144;

    if data.is_empty() {
        return vec![];
    }

    let num_blocks = data.len().div_ceil(SUPER_BLOCK_SIZE);
    let mut result = Vec::with_capacity(num_blocks * SUPER_BLOCK_BYTES);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * SUPER_BLOCK_SIZE;
        let block_end = (block_start + SUPER_BLOCK_SIZE).min(data.len());
        let block_data = &data[block_start..block_end];

        let mut padded = [0.0f32; SUPER_BLOCK_SIZE];
        padded[..block_data.len()].copy_from_slice(block_data);

        let (sub_scales, sub_mins) = compute_sub_block_stats(&padded, 15.0);
        let (d, dmin, scales_6bit, mins_6bit) = compute_global_scales(&sub_scales, &sub_mins);
        write_kquant_header(&mut result, d, dmin, &scales_6bit, &mins_6bit);

        // Quantize values into 4-bit packed nibbles
        let mut qs = [0u8; 128];
        for chunk in 0..4 {
            let chunk_start = chunk * 64;
            let is = chunk * 2;
            let scale_lo = d * f32::from(scales_6bit[is]);
            let min_lo = dmin * f32::from(mins_6bit[is]);
            let scale_hi = d * f32::from(scales_6bit[is + 1]);
            let min_hi = dmin * f32::from(mins_6bit[is + 1]);

            for l in 0..32 {
                let q_lo = quantize_one(padded[chunk_start + l], min_lo, scale_lo, 15.0);
                let q_hi = quantize_one(padded[chunk_start + l + 32], min_hi, scale_hi, 15.0);
                qs[chunk * 32 + l] = (q_lo & 0x0F) | ((q_hi & 0x0F) << 4);
            }
        }
        result.extend_from_slice(&qs);
    }

    result
}

/// Quantize F32 matrix to `Q4_K` format with proper row layout
///
/// Processes each row independently to maintain row-major layout.
#[must_use]
pub fn quantize_q4_k_matrix(data: &[f32], shape: &[usize]) -> Vec<u8> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 144;

    if shape.len() != 2 {
        return quantize_q4_k(data);
    }

    let rows = shape[0];
    let cols = shape[1];

    let super_blocks_per_row = cols.div_ceil(SUPER_BLOCK_SIZE);
    let padded_cols = super_blocks_per_row * SUPER_BLOCK_SIZE;

    let mut result = Vec::with_capacity(rows * super_blocks_per_row * SUPER_BLOCK_BYTES);

    for row_idx in 0..rows {
        let mut padded_row = vec![0.0f32; padded_cols];
        let row_start = row_idx * cols;
        let row_end = row_start + cols;
        if row_end <= data.len() {
            padded_row[..cols].copy_from_slice(&data[row_start..row_end]);
        }

        let row_q4k = quantize_q4_k(&padded_row);
        result.extend_from_slice(&row_q4k);
    }

    result
}

// ============================================================================
// Q5_K Quantization
// ============================================================================

/// Quantize F32 data to `Q5_K` format
///
/// `Q5_K`: 256 elements per super-block, 176 bytes per block
/// Layout: d (2B) + dmin (2B) + scales (12B) + qh (32B) + qs (128B)
#[must_use]
pub fn quantize_q5_k(data: &[f32]) -> Vec<u8> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 176;

    if data.is_empty() {
        return vec![];
    }

    let num_blocks = data.len().div_ceil(SUPER_BLOCK_SIZE);
    let mut result = Vec::with_capacity(num_blocks * SUPER_BLOCK_BYTES);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * SUPER_BLOCK_SIZE;
        let block_end = (block_start + SUPER_BLOCK_SIZE).min(data.len());
        let block_data = &data[block_start..block_end];

        let mut padded = [0.0f32; SUPER_BLOCK_SIZE];
        padded[..block_data.len()].copy_from_slice(block_data);

        let (sub_scales, sub_mins) = compute_sub_block_stats(&padded, 31.0);
        let (d, dmin, scales_6bit, mins_6bit) = compute_global_scales(&sub_scales, &sub_mins);
        write_kquant_header(&mut result, d, dmin, &scales_6bit, &mins_6bit);

        // Quantize all 256 values to 5-bit
        let mut q5_vals = [0u8; 256];
        for j in 0..8 {
            let scale = d * f32::from(scales_6bit[j]);
            let min_val = dmin * f32::from(mins_6bit[j]);
            for k in 0..32 {
                q5_vals[j * 32 + k] = quantize_one(padded[j * 32 + k], min_val, scale, 31.0);
            }
        }

        // Pack high bits (qh)
        result.extend_from_slice(&pack_q5k_high_bits(&q5_vals));

        // Pack low 4 bits (qs)
        result.extend_from_slice(&pack_q5k_low_nibbles(&q5_vals));
    }

    result
}

/// Pack `Q5_K` high bits: extract bit 4 from each value into 32 bytes.
fn pack_q5k_high_bits(q5_vals: &[u8; 256]) -> [u8; 32] {
    let mut qh = [0u8; 32];
    for i in 0..32 {
        let mut h = 0u8;
        for j in 0..8 {
            h |= ((q5_vals[j * 32 + i] >> 4) & 1) << j;
        }
        qh[i] = h;
    }
    qh
}

/// Pack `Q5_K` low nibbles: combine pairs of 4-bit values into 128 bytes.
fn pack_q5k_low_nibbles(q5_vals: &[u8; 256]) -> [u8; 128] {
    let mut qs = [0u8; 128];
    for j in 0..8 {
        for k in 0..16 {
            let idx1 = j * 32 + k;
            let idx2 = j * 32 + k + 16;
            qs[j * 16 + k] = (q5_vals[idx1] & 0x0F) | ((q5_vals[idx2] & 0x0F) << 4);
        }
    }
    qs
}

/// Quantize F32 matrix to `Q5_K` format with proper row layout
#[must_use]
pub fn quantize_q5_k_matrix(data: &[f32], shape: &[usize]) -> Vec<u8> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 176;

    if shape.len() != 2 {
        return quantize_q5_k(data);
    }

    let rows = shape[0];
    let cols = shape[1];
    let super_blocks_per_row = cols.div_ceil(SUPER_BLOCK_SIZE);
    let padded_cols = super_blocks_per_row * SUPER_BLOCK_SIZE;

    let mut result = Vec::with_capacity(rows * super_blocks_per_row * SUPER_BLOCK_BYTES);

    for row_idx in 0..rows {
        let mut padded_row = vec![0.0f32; padded_cols];
        let row_start = row_idx * cols;
        let row_end = row_start + cols;
        if row_end <= data.len() {
            padded_row[..cols].copy_from_slice(&data[row_start..row_end]);
        }

        let row_q5k = quantize_q5_k(&padded_row);
        result.extend_from_slice(&row_q5k);
    }

    result
}

// ============================================================================
// Q6_K Quantization
// ============================================================================

/// Quantize F32 data to `Q6_K` format (candle/GGUF compatible)
///
/// `Q6_K` format: 256-element super-blocks
/// Each super block: ql (128 bytes) + qh (64 bytes) + scales (16 bytes) + d (f16) = 210 bytes
/// - 6-bit values stored split: low 4 bits in ql, high 2 bits in qh
/// - 16 sub-blocks of 16 elements each, with int8 scale per sub-block
#[must_use]
pub fn quantize_q6_k(data: &[f32]) -> Vec<u8> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 210;

    if data.is_empty() {
        return vec![];
    }

    let num_blocks = data.len().div_ceil(SUPER_BLOCK_SIZE);
    let mut result = Vec::with_capacity(num_blocks * SUPER_BLOCK_BYTES);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * SUPER_BLOCK_SIZE;
        let block_end = (block_start + SUPER_BLOCK_SIZE).min(data.len());
        let block_data = &data[block_start..block_end];

        let mut padded = [0.0f32; SUPER_BLOCK_SIZE];
        padded[..block_data.len()].copy_from_slice(block_data);

        let (d, scales_i8) = compute_q6k_scales(&padded);
        let q6_vals = quantize_q6k_values(&padded, d, &scales_i8);
        let (ql, qh) = pack_q6k_bits(&q6_vals);

        // Write in candle order: ql, qh, scales, d
        result.extend_from_slice(&ql);
        result.extend_from_slice(&qh);
        for s in &scales_i8 {
            result.push(*s as u8);
        }
        result.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    }

    result
}

/// Compute `Q6_K` global scale and per-sub-block int8 scales.
fn compute_q6k_scales(padded: &[f32; 256]) -> (f32, [i8; 16]) {
    let mut sub_scales = [0.0f32; 16];
    for (j, sub_block) in padded.chunks(16).enumerate().take(16) {
        let max_abs = sub_block.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        sub_scales[j] = if max_abs > F16_MIN_NORMAL {
            max_abs / 31.0
        } else {
            F16_MIN_NORMAL
        };
    }

    let max_scale = sub_scales.iter().fold(0.0f32, |a, &b| a.max(b));
    let d = if max_scale > F16_MIN_NORMAL {
        max_scale / 127.0
    } else {
        F16_MIN_NORMAL
    };

    let mut scales_i8 = [0i8; 16];
    for j in 0..16 {
        scales_i8[j] = (sub_scales[j] / d).round().clamp(-127.0, 127.0) as i8;
    }

    (d, scales_i8)
}

/// Quantize 256 padded values to 6-bit `Q6_K` format.
fn quantize_q6k_values(padded: &[f32; 256], d: f32, scales_i8: &[i8; 16]) -> [u8; 256] {
    let mut q6_vals = [0u8; 256];
    for j in 0..16 {
        let scale = d * f32::from(scales_i8[j]);
        let inv_scale = if scale.abs() > 1e-10 {
            1.0 / scale
        } else {
            0.0
        };
        for k in 0..16 {
            let idx = j * 16 + k;
            let q = (padded[idx] * inv_scale).round().clamp(-32.0, 31.0) as i8;
            q6_vals[idx] = (q + 32) as u8;
        }
    }
    q6_vals
}

/// Pack 256 `Q6_K` values into ql (128 bytes) and qh (64 bytes) candle/GGUF layout.
fn pack_q6k_bits(q6_vals: &[u8; 256]) -> ([u8; 128], [u8; 64]) {
    let mut ql = [0u8; 128];
    let mut qh = [0u8; 64];

    for half in 0..2 {
        let n = half * 128;
        let ql_base = half * 64;
        let qh_base = half * 32;

        for l in 0..32 {
            let q1 = q6_vals[n + l];
            let q2 = q6_vals[n + l + 32];
            let q3 = q6_vals[n + l + 64];
            let q4 = q6_vals[n + l + 96];

            ql[ql_base + l] = (q1 & 0x0F) | ((q3 & 0x0F) << 4);
            ql[ql_base + l + 32] = (q2 & 0x0F) | ((q4 & 0x0F) << 4);

            qh[qh_base + l] = ((q1 >> 4) & 0x03)
                | (((q2 >> 4) & 0x03) << 2)
                | (((q3 >> 4) & 0x03) << 4)
                | (((q4 >> 4) & 0x03) << 6);
        }
    }

    (ql, qh)
}

/// Quantize F32 matrix to `Q6_K` format with proper row layout
#[must_use]
pub fn quantize_q6_k_matrix(data: &[f32], shape: &[usize]) -> Vec<u8> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 210;

    if shape.len() != 2 {
        return quantize_q6_k(data);
    }

    let rows = shape[0];
    let cols = shape[1];
    let super_blocks_per_row = cols.div_ceil(SUPER_BLOCK_SIZE);
    let padded_cols = super_blocks_per_row * SUPER_BLOCK_SIZE;

    let mut result = Vec::with_capacity(rows * super_blocks_per_row * SUPER_BLOCK_BYTES);

    for row_idx in 0..rows {
        let mut padded_row = vec![0.0f32; padded_cols];
        let row_start = row_idx * cols;
        let row_end = row_start + cols;
        if row_end <= data.len() {
            padded_row[..cols].copy_from_slice(&data[row_start..row_end]);
        }

        let row_q6k = quantize_q6_k(&padded_row);
        result.extend_from_slice(&row_q6k);
    }

    result
}
