#![allow(missing_docs)]
//! Scalar Q4_K GEMV implementations.
//!
//! Contains the baseline scalar dot product, 4-way unrolled fused GEMV,
//! and the scalar chunk processor for parallel dispatch.

use super::super::{parse_q4k_header, SUPER_BLOCK_BYTES, SUPER_BLOCK_SIZE};

/// Scalar dot product for one Q4K super-block row.
#[inline(always)]
fn process_q4k_superblock_scalar(
    sb_data: &[u8],
    input: &[f32],
    input_offset: usize,
    in_dim: usize,
) -> f32 {
    let (d, dmin, scales, mins) = parse_q4k_header(sb_data);
    let qs = sb_data.get(16..144).expect("Q4_K: need ≥144 bytes for qs");
    let mut sum = 0.0f32;

    for chunk in 0..4 {
        let chunk_start = chunk * 64;
        let q_start = chunk * 32;

        let d1 = d * f32::from(scales[chunk * 2]);
        let dm1 = dmin * f32::from(mins[chunk * 2]);
        let d2 = d * f32::from(scales[chunk * 2 + 1]);
        let dm2 = dmin * f32::from(mins[chunk * 2 + 1]);

        for i in 0..32 {
            let input_idx = input_offset + chunk_start + i;
            if input_idx < in_dim {
                let q_val = (qs[q_start + i] & 0x0F) as f32;
                sum += (d1 * q_val - dm1) * input[input_idx];
            }
        }
        for i in 0..32 {
            let input_idx = input_offset + chunk_start + 32 + i;
            if input_idx < in_dim {
                let q_val = (qs[q_start + i] >> 4) as f32;
                sum += (d2 * q_val - dm2) * input[input_idx];
            }
        }
    }
    sum
}

pub fn matmul_q4k_f32_scalar(
    q4k_data: &[u8],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    assert_eq!(input.len(), in_dim, "Input length mismatch");
    assert!(
        in_dim % SUPER_BLOCK_SIZE == 0 || in_dim < SUPER_BLOCK_SIZE,
        "in_dim must be multiple of 256 (or smaller for padding)"
    );

    let num_blocks_per_row = (in_dim + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let row_bytes = num_blocks_per_row * SUPER_BLOCK_BYTES;
    let expected_size = out_dim * row_bytes;

    assert!(
        q4k_data.len() >= expected_size,
        "Q4K data too small: {} < {}",
        q4k_data.len(),
        expected_size
    );

    // Uninit: output[out_idx] = sum (SET) for every out_idx.
    let mut output: Vec<f32> = Vec::with_capacity(out_dim);
    // SAFETY: Each output[out_idx] is SET to the accumulated sum. No reads before writes.
    unsafe {
        output.set_len(out_dim);
    }

    for out_idx in 0..out_dim {
        let row_start = out_idx * row_bytes;
        let mut sum = 0.0f32;

        for sb_idx in 0..num_blocks_per_row {
            let sb_start = row_start + sb_idx * SUPER_BLOCK_BYTES;
            let sb_data = &q4k_data[sb_start..sb_start + SUPER_BLOCK_BYTES];
            let input_offset = sb_idx * SUPER_BLOCK_SIZE;
            sum += process_q4k_superblock_scalar(sb_data, input, input_offset, in_dim);
        }

        output[out_idx] = sum;
    }

    output
}

/// Process one Q4K nibble half (low or high) with 4-way unrolled accumulation.
#[inline(always)]
fn process_q4k_nibble_half(
    qs: &[u8],
    q_start: usize,
    input: &[f32],
    input_base: usize,
    in_dim: usize,
    d_val: f32,
    dm_val: f32,
    shift: u8,
    acc: &mut [f32; 4],
) {
    let mut i = 0;
    while i + 3 < 32 {
        let idx = input_base + i;
        if idx + 3 < in_dim {
            let q0 = ((qs[q_start + i] >> shift) & 0x0F) as f32;
            let q1 = ((qs[q_start + i + 1] >> shift) & 0x0F) as f32;
            let q2 = ((qs[q_start + i + 2] >> shift) & 0x0F) as f32;
            let q3 = ((qs[q_start + i + 3] >> shift) & 0x0F) as f32;

            acc[0] = (d_val * q0 - dm_val).mul_add(input[idx], acc[0]);
            acc[1] = (d_val * q1 - dm_val).mul_add(input[idx + 1], acc[1]);
            acc[2] = (d_val * q2 - dm_val).mul_add(input[idx + 2], acc[2]);
            acc[3] = (d_val * q3 - dm_val).mul_add(input[idx + 3], acc[3]);
        }
        i += 4;
    }
    while i < 32 {
        let idx = input_base + i;
        if idx < in_dim {
            let q_val = ((qs[q_start + i] >> shift) & 0x0F) as f32;
            acc[0] = (d_val * q_val - dm_val).mul_add(input[idx], acc[0]);
        }
        i += 1;
    }
}

/// Fused Q4_K matrix-vector multiply (optimized with 4-way unrolling)
///
/// This version uses 4 independent accumulators to improve instruction-level
/// parallelism while maintaining scalar correctness.
///
/// # Arguments
/// Same as `matmul_q4k_f32_scalar`
pub fn matmul_q4k_f32(q4k_data: &[u8], input: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    assert_eq!(input.len(), in_dim, "Input length mismatch");

    let num_blocks_per_row = (in_dim + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let row_bytes = num_blocks_per_row * SUPER_BLOCK_BYTES;

    // Uninit: output[out_idx] = (acc[0]+acc[1])+(acc[2]+acc[3]) (SET) for every out_idx.
    let mut output: Vec<f32> = Vec::with_capacity(out_dim);
    // SAFETY: Each output[out_idx] is SET from local accumulator. No reads before writes.
    unsafe {
        output.set_len(out_dim);
    }

    for out_idx in 0..out_dim {
        let row_start = out_idx * row_bytes;
        let mut acc = [0.0f32; 4];

        for sb_idx in 0..num_blocks_per_row {
            let sb_start = row_start + sb_idx * SUPER_BLOCK_BYTES;
            let sb_data = &q4k_data[sb_start..sb_start + SUPER_BLOCK_BYTES];

            let (d, dmin, scales, mins) = parse_q4k_header(sb_data);
            let qs = sb_data.get(16..144).expect("Q4_K: need ≥144 bytes for qs");
            let input_offset = sb_idx * SUPER_BLOCK_SIZE;

            for chunk in 0..4 {
                let chunk_start = chunk * 64;
                let q_start = chunk * 32;

                let d1 = d * f32::from(scales[chunk * 2]);
                let dm1 = dmin * f32::from(mins[chunk * 2]);
                let d2 = d * f32::from(scales[chunk * 2 + 1]);
                let dm2 = dmin * f32::from(mins[chunk * 2 + 1]);

                let base_low = input_offset + chunk_start;
                process_q4k_nibble_half(qs, q_start, input, base_low, in_dim, d1, dm1, 0, &mut acc);

                let base_high = input_offset + chunk_start + 32;
                process_q4k_nibble_half(
                    qs, q_start, input, base_high, in_dim, d2, dm2, 4, &mut acc,
                );
            }
        }

        output[out_idx] = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    }

    output
}

/// Scalar chunk processor for parallel dispatch.
#[allow(dead_code)]
pub(crate) fn compute_chunk_q4k_scalar(
    q4k_data: &[u8],
    input: &[f32],
    chunk: &mut [f32],
    start_row: usize,
    out_dim: usize,
    in_dim: usize,
    num_blocks_per_row: usize,
    row_bytes: usize,
) {
    for (local_idx, out_val) in chunk.iter_mut().enumerate() {
        let out_idx = start_row + local_idx;
        if out_idx >= out_dim {
            break;
        }

        let row_start = out_idx * row_bytes;
        let mut sum = 0.0f32;

        for sb_idx in 0..num_blocks_per_row {
            let sb_start = row_start + sb_idx * SUPER_BLOCK_BYTES;
            if sb_start + SUPER_BLOCK_BYTES > q4k_data.len() {
                break;
            }
            let sb_data = &q4k_data[sb_start..sb_start + SUPER_BLOCK_BYTES];
            let input_offset = sb_idx * SUPER_BLOCK_SIZE;
            sum += process_q4k_superblock_scalar(sb_data, input, input_offset, in_dim);
        }

        *out_val = sum;
    }
}
