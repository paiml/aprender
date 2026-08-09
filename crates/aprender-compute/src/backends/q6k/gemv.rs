#![allow(missing_docs)]
//! Row-major Q6_K matrix-vector multiplication.
//!
//! This module implements row-major GEMV for Q6_K format.
//! Includes scalar, AVX2-optimized, and parallel dispatch implementations.

use super::{f16_to_f32, SUPER_BLOCK_BYTES, SUPER_BLOCK_SIZE};

/// Fused Q6_K matrix-vector multiply (scalar reference)
/// Extract a single Q6K quantized value from packed ql/qh arrays.
#[inline(always)]
fn extract_q6k_scalar(ql: &[u8], qh: &[u8], idx: usize) -> i8 {
    let ql_byte = ql[idx / 2];
    let low4 = if idx % 2 == 0 { ql_byte & 0x0F } else { ql_byte >> 4 };
    let qh_byte = qh[idx / 4];
    let high2 = (qh_byte >> ((idx % 4) * 2)) & 0x03;
    (low4 | (high2 << 4)) as i8 - 32
}

/// Scalar dot product for one Q6K super-block row.
#[inline(always)]
fn process_q6k_superblock_scalar(
    sb_data: &[u8],
    input: &[f32],
    input_offset: usize,
    in_dim: usize,
) -> f32 {
    let ql = sb_data.get(0..128).expect("Q6_K: need ≥128 bytes for ql");
    let qh = sb_data.get(128..192).expect("Q6_K: need ≥192 bytes for qh");
    let scales = sb_data.get(192..208).expect("Q6_K: need ≥208 bytes for scales");
    let d = f16_to_f32(u16::from_le_bytes([sb_data[208], sb_data[209]]));
    let mut sum = 0.0f32;

    for group in 0..16 {
        let scale = (scales[group] as i8) as f32;
        let group_offset = group * 16;

        for j in 0..16 {
            let idx = group_offset + j;
            let input_idx = input_offset + idx;
            if input_idx >= in_dim {
                continue;
            }
            let q6 = extract_q6k_scalar(ql, qh, idx);
            sum += d * scale * q6 as f32 * input[input_idx];
        }
    }
    sum
}

pub fn matmul_q6k_f32_scalar(
    q6k_data: &[u8],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    assert_eq!(input.len(), in_dim, "Input length mismatch");

    let num_blocks_per_row = (in_dim + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let row_bytes = num_blocks_per_row * SUPER_BLOCK_BYTES;

    // Uninit: output[out_idx] = sum (SET) for every out_idx.
    let mut output: Vec<f32> = Vec::with_capacity(out_dim);
    // SAFETY: Each output[out_idx] is SET from local accumulator sum.
    unsafe {
        output.set_len(out_dim);
    }

    for out_idx in 0..out_dim {
        let row_start = out_idx * row_bytes;
        let mut sum = 0.0f32;

        for sb_idx in 0..num_blocks_per_row {
            let sb_start = row_start + sb_idx * SUPER_BLOCK_BYTES;
            if sb_start + SUPER_BLOCK_BYTES > q6k_data.len() {
                break;
            }
            let sb_data = &q6k_data[sb_start..sb_start + SUPER_BLOCK_BYTES];
            let input_offset = sb_idx * SUPER_BLOCK_SIZE;
            sum += process_q6k_superblock_scalar(sb_data, input, input_offset, in_dim);
        }

        output[out_idx] = sum;
    }

    output
}

/// Extract 8 Q6K quantized values from packed ql/qh arrays.
#[inline(always)]
#[allow(dead_code)]
fn extract_q6k_values(ql: &[u8], qh: &[u8], idx_base: usize) -> [i32; 8] {
    let mut q6_vals = [0i32; 8];
    for i in 0..8 {
        let idx = idx_base + i;
        let ql_byte = ql[idx / 2];
        let low4 = if idx % 2 == 0 { ql_byte & 0x0F } else { ql_byte >> 4 };
        let qh_byte = qh[idx / 4];
        let qh_shift = (idx % 4) * 2;
        let high2 = (qh_byte >> qh_shift) & 0x03;
        q6_vals[i] = ((low4 | (high2 << 4)) as i32) - 32;
    }
    q6_vals
}

/// AVX2 horizontal sum of 8 f32 lanes to a single f32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// SAFETY: caller verifies AVX2 support, input slices meet alignment/length requirements
unsafe fn hsum_q6k_avx2(acc: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi128 = _mm256_extractf128_ps(acc, 1);
    let lo128 = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo128, hi128);
    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);
    let hi32 = _mm_shuffle_ps(sum64, sum64, 1);
    let sum32 = _mm_add_ss(sum64, hi32);
    _mm_cvtss_f32(sum32)
}

/// Process one Q6K super-block with AVX2, accumulating into `acc`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
// SAFETY: Caller ensures AVX2+FMA are available and sb_data is a valid Q6_K super-block
unsafe fn process_q6k_superblock_avx2(
    sb_data: &[u8],
    input: &[f32],
    input_offset: usize,
    in_dim: usize,
    acc: &mut std::arch::x86_64::__m256,
) {
    unsafe {
        use std::arch::x86_64::*;

        let ql = sb_data.get(0..128).expect("Q6_K: need ≥128 bytes for ql");
        let qh = sb_data.get(128..192).expect("Q6_K: need ≥192 bytes for qh");
        let scales = sb_data.get(192..208).expect("Q6_K: need ≥208 bytes for scales");
        let d = f16_to_f32(u16::from_le_bytes([sb_data[208], sb_data[209]]));
        let d_vec = _mm256_set1_ps(d);

        for group in 0..16 {
            let scale = (scales[group] as i8) as f32;
            let ds_vec = _mm256_mul_ps(d_vec, _mm256_set1_ps(scale));
            let group_offset = group * 16;
            let input_group = input_offset + group_offset;

            for half in 0..2 {
                let half_offset = half * 8;
                let input_base = input_group + half_offset;
                if input_base + 8 > in_dim {
                    continue;
                }

                let q6_vals = extract_q6k_values(ql, qh, group_offset + half_offset);
                let q6_i32 = _mm256_loadu_si256(q6_vals.as_ptr() as *const __m256i);
                let q6_f32 = _mm256_cvtepi32_ps(q6_i32);
                let x = _mm256_loadu_ps(input.as_ptr().add(input_base));
                let dequant = _mm256_mul_ps(ds_vec, q6_f32);
                *acc = _mm256_fmadd_ps(dequant, x, *acc);
            }
        }
    }
}

/// Fused Q6_K matrix-vector multiply with AVX2 SIMD
///
/// Optimized to process groups of 8 values at a time, computing
/// dequant and dot product in one pass without intermediate buffer.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
// SAFETY: Caller ensures AVX2+FMA are available and q6k_data is valid Q6_K layout
unsafe fn matmul_q6k_f32_avx2(
    q6k_data: &[u8],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    unsafe {
        use std::arch::x86_64::*;

        let num_blocks_per_row = (in_dim + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
        let row_bytes = num_blocks_per_row * SUPER_BLOCK_BYTES;

        // Uninit: output[out_idx] = hsum_q6k_avx2(acc) (SET) for every out_idx.
        let mut output: Vec<f32> = Vec::with_capacity(out_dim);
        // SAFETY: Each output[out_idx] is SET from local SIMD accumulator.
        output.set_len(out_dim);

        for out_idx in 0..out_dim {
            let row_start = out_idx * row_bytes;
            let mut acc = _mm256_setzero_ps();

            for sb_idx in 0..num_blocks_per_row {
                let sb_start = row_start + sb_idx * SUPER_BLOCK_BYTES;
                if sb_start + SUPER_BLOCK_BYTES > q6k_data.len() {
                    break;
                }
                let sb_data = &q6k_data[sb_start..sb_start + SUPER_BLOCK_BYTES];
                let input_offset = sb_idx * SUPER_BLOCK_SIZE;
                process_q6k_superblock_avx2(sb_data, input, input_offset, in_dim, &mut acc);
            }

            output[out_idx] = hsum_q6k_avx2(acc);
        }

        output
    }
}

/// Runtime dispatch for Q6K matmul - uses AVX2 if available
///
/// # Contract (GH-279)
///
/// Preconditions validated via `debug_assert!` (zero-cost in release):
/// - `q6k_data.len() >= contracts::Q6_K.expected_bytes(out_dim, in_dim)`
/// - `input.len() == in_dim`
///
/// These guarantee that inner-loop `expect()` calls on super-block sub-slices
/// are unreachable: each super-block is sliced to exactly `SUPER_BLOCK_BYTES`
/// (210), and all sub-accesses (`get(0..128)`, `get(128..192)`, `get(192..208)`)
/// fit within that.
#[inline]
pub fn matmul_q6k_f32_dispatch(
    q6k_data: &[u8],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    // GH-279: Contract validation at dispatch boundary.
    // Inner expect() calls are defense-in-depth — provably unreachable when
    // this precondition holds, because every sb_data slice is SUPER_BLOCK_BYTES.
    debug_assert_eq!(input.len(), in_dim, "Q6K dispatch: input length mismatch");
    debug_assert!(
        q6k_data.len() >= crate::contracts::Q6_K.expected_bytes(out_dim, in_dim),
        "Q6K dispatch: buffer too small: {} bytes for [{}, {}] (need {})",
        q6k_data.len(),
        out_dim,
        in_dim,
        crate::contracts::Q6_K.expected_bytes(out_dim, in_dim),
    );

    // For large matmuls (total work >= ~8M ops), use parallel execution
    // This catches FFN layers (8960x1536) and lm_head (151936x1536)
    // Also catches ffn_down (1536x8960) where out_dim is small but in_dim is large
    let total_work = out_dim * in_dim;
    if total_work >= 8_000_000 {
        return matmul_q6k_f32_parallel(q6k_data, input, out_dim, in_dim);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: preconditions verified by caller
            return unsafe { matmul_q6k_f32_avx2(q6k_data, input, out_dim, in_dim) };
        }
    }
    matmul_q6k_f32_scalar(q6k_data, input, out_dim, in_dim)
}

/// Parallel Q6K matmul using multiple threads with AVX2
#[cfg(target_arch = "x86_64")]
fn matmul_q6k_f32_parallel(
    q6k_data: &[u8],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    use std::thread;

    // Use fewer threads with larger chunks for better cache efficiency
    let num_threads = thread::available_parallelism().map(|p| p.get()).unwrap_or(4).min(12); // Use 12 threads max for better cache behavior

    let chunk_size = (out_dim + num_threads - 1) / num_threads;
    let num_blocks_per_row = (in_dim + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let row_bytes = num_blocks_per_row * SUPER_BLOCK_BYTES;

    // Uninit: compute_chunk writes *out_val = sum/hsum(acc) (SET) for every element.
    let mut output: Vec<f32> = Vec::with_capacity(out_dim);
    // SAFETY: Each thread's compute_chunk writes every element in its chunk (SET).
    unsafe {
        output.set_len(out_dim);
    }
    let has_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");

    thread::scope(|s| {
        let input_ref = input;
        let q6k_ref = q6k_data;
        // CGP-DBUF: iterate directly instead of collecting into Vec.
        for (chunk_idx, chunk) in output.chunks_mut(chunk_size).enumerate() {
            let start_row = chunk_idx * chunk_size;

            s.spawn(move || {
                if has_avx2 {
                    // SAFETY: AVX2+FMA availability verified via is_x86_feature_detected!()
                    // before thread::scope entry; has_avx2 captures that result.
                    unsafe {
                        compute_chunk_avx2(
                            q6k_ref,
                            input_ref,
                            chunk,
                            start_row,
                            out_dim,
                            in_dim,
                            num_blocks_per_row,
                            row_bytes,
                        );
                    }
                } else {
                    compute_chunk_scalar(
                        q6k_ref,
                        input_ref,
                        chunk,
                        start_row,
                        out_dim,
                        in_dim,
                        num_blocks_per_row,
                        row_bytes,
                    );
                }
            });
        }
    });

    output
}

/// Fallback for non-x86_64
#[cfg(not(target_arch = "x86_64"))]
fn matmul_q6k_f32_parallel(
    q6k_data: &[u8],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    matmul_q6k_f32_scalar(q6k_data, input, out_dim, in_dim)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
// SAFETY: Caller ensures AVX2+FMA are available and chunk bounds are valid
unsafe fn compute_chunk_avx2(
    q6k_data: &[u8],
    input: &[f32],
    chunk: &mut [f32],
    start_row: usize,
    out_dim: usize,
    in_dim: usize,
    num_blocks_per_row: usize,
    row_bytes: usize,
) {
    unsafe {
        use std::arch::x86_64::*;

        for (local_idx, out_val) in chunk.iter_mut().enumerate() {
            let out_idx = start_row + local_idx;
            if out_idx >= out_dim {
                break;
            }

            let row_start = out_idx * row_bytes;
            let mut acc = _mm256_setzero_ps();

            for sb_idx in 0..num_blocks_per_row {
                let sb_start = row_start + sb_idx * SUPER_BLOCK_BYTES;
                if sb_start + SUPER_BLOCK_BYTES > q6k_data.len() {
                    break;
                }
                let sb_data = &q6k_data[sb_start..sb_start + SUPER_BLOCK_BYTES];
                let input_offset = sb_idx * SUPER_BLOCK_SIZE;
                process_q6k_superblock_avx2(sb_data, input, input_offset, in_dim, &mut acc);
            }

            *out_val = hsum_q6k_avx2(acc);
        }
    }
}

#[allow(dead_code)]
pub(crate) fn compute_chunk_scalar(
    q6k_data: &[u8],
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
            if sb_start + SUPER_BLOCK_BYTES > q6k_data.len() {
                break;
            }
            let sb_data = &q6k_data[sb_start..sb_start + SUPER_BLOCK_BYTES];
            let input_offset = sb_idx * SUPER_BLOCK_SIZE;
            sum += process_q6k_superblock_scalar(sb_data, input, input_offset, in_dim);
        }

        *out_val = sum;
    }
}

/// Public alias for the optimized Q6K matmul
pub fn matmul_q6k_f32(q6k_data: &[u8], input: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    matmul_q6k_f32_dispatch(q6k_data, input, out_dim, in_dim)
}
