//! AVX2 SIMD Q4_K GEMV implementations.
//!
//! Contains 8-wide AVX2+FMA optimized GEMV, super-block processor,
//! horizontal sum helper, and the AVX2 chunk processor for parallel dispatch.

use super::super::{parse_q4k_header, SUPER_BLOCK_BYTES, SUPER_BLOCK_SIZE};

/// Fused Q4_K matrix-vector multiply with AVX2 SIMD (8-wide)
///
/// Processes 8 elements at a time using AVX2 intrinsics.
/// Delegates per-super-block work to `process_q4k_superblock_avx2`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
// SAFETY: Caller ensures AVX2+FMA are available and q4k_data is valid Q4_K layout
pub(crate) unsafe fn matmul_q4k_f32_avx2(
    q4k_data: &[u8],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    unsafe {
        use std::arch::x86_64::*;

        let num_blocks_per_row = (in_dim + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
        let row_bytes = num_blocks_per_row * SUPER_BLOCK_BYTES;
        let low_mask = _mm256_set1_epi32(0x0F);

        // Uninit: output[out_idx] = hsum_avx2(acc) (SET) for every out_idx.
        let mut output: Vec<f32> = Vec::with_capacity(out_dim);
        // SAFETY: Each output[out_idx] is SET from local SIMD accumulator.
        output.set_len(out_dim);

        for out_idx in 0..out_dim {
            let row_start = out_idx * row_bytes;
            let mut acc = _mm256_setzero_ps();

            for sb_idx in 0..num_blocks_per_row {
                let sb_start = row_start + sb_idx * SUPER_BLOCK_BYTES;
                let sb_data = &q4k_data[sb_start..sb_start + SUPER_BLOCK_BYTES];
                let input_offset = sb_idx * SUPER_BLOCK_SIZE;
                process_q4k_superblock_avx2(
                    sb_data,
                    input,
                    input_offset,
                    in_dim,
                    low_mask,
                    &mut acc,
                );
            }

            output[out_idx] = hsum_avx2(acc);
        }

        output
    }
}

/// Process one Q4K super-block row with AVX2 and accumulate into `acc`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
// SAFETY: Caller ensures AVX2+FMA are available, sb_data is a valid super-block
pub(crate) unsafe fn process_q4k_superblock_avx2(
    sb_data: &[u8],
    input: &[f32],
    input_offset: usize,
    in_dim: usize,
    low_mask: std::arch::x86_64::__m256i,
    acc: &mut std::arch::x86_64::__m256,
) {
    unsafe {
        use std::arch::x86_64::*;

        let (d, dmin, scales, mins) = parse_q4k_header(sb_data);
        let qs = sb_data.get(16..144).expect("Q4_K: need ≥144 bytes for qs");

        for chunk_i in 0..4 {
            let chunk_start = chunk_i * 64;
            let q_start = chunk_i * 32;

            let d1 = d * f32::from(scales[chunk_i * 2]);
            let dm1 = dmin * f32::from(mins[chunk_i * 2]);
            let d2 = d * f32::from(scales[chunk_i * 2 + 1]);
            let dm2 = dmin * f32::from(mins[chunk_i * 2 + 1]);

            let d1_vec = _mm256_set1_ps(d1);
            let dm1_vec = _mm256_set1_ps(dm1);
            let d2_vec = _mm256_set1_ps(d2);
            let dm2_vec = _mm256_set1_ps(dm2);

            // Process low nibbles (32 values) in groups of 8
            let mut i = 0;
            while i + 8 <= 32 {
                let input_base = input_offset + chunk_start + i;
                if input_base + 8 <= in_dim {
                    let q_bytes = _mm_loadl_epi64(qs.as_ptr().add(q_start + i) as *const __m128i);
                    let q_i32 = _mm256_cvtepu8_epi32(q_bytes);
                    let q_low = _mm256_and_si256(q_i32, low_mask);
                    let q_f32 = _mm256_cvtepi32_ps(q_low);
                    let x = _mm256_loadu_ps(input.as_ptr().add(input_base));
                    let dequant = _mm256_fmsub_ps(d1_vec, q_f32, dm1_vec);
                    *acc = _mm256_fmadd_ps(dequant, x, *acc);
                }
                i += 8;
            }

            // Process high nibbles (32 values) in groups of 8
            let mut i = 0;
            while i + 8 <= 32 {
                let input_base = input_offset + chunk_start + 32 + i;
                if input_base + 8 <= in_dim {
                    let q_bytes = _mm_loadl_epi64(qs.as_ptr().add(q_start + i) as *const __m128i);
                    let q_i32 = _mm256_cvtepu8_epi32(q_bytes);
                    let q_high = _mm256_srli_epi32(q_i32, 4);
                    let q_f32 = _mm256_cvtepi32_ps(q_high);
                    let x = _mm256_loadu_ps(input.as_ptr().add(input_base));
                    let dequant = _mm256_fmsub_ps(d2_vec, q_f32, dm2_vec);
                    *acc = _mm256_fmadd_ps(dequant, x, *acc);
                }
                i += 8;
            }
        }
    }
}

/// AVX2 horizontal sum of 8 f32 lanes to a single f32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// SAFETY: caller verifies AVX2 support, input slices meet alignment/length requirements
pub(crate) unsafe fn hsum_avx2(acc: std::arch::x86_64::__m256) -> f32 {
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
// SAFETY: Caller ensures AVX2+FMA are available and chunk bounds are valid
pub(crate) unsafe fn compute_chunk_q4k_avx2(
    q4k_data: &[u8],
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

        let low_mask = _mm256_set1_epi32(0x0F);

        for (local_idx, out_val) in chunk.iter_mut().enumerate() {
            let out_idx = start_row + local_idx;
            if out_idx >= out_dim {
                break;
            }

            let row_start = out_idx * row_bytes;
            let mut acc = _mm256_setzero_ps();

            for sb_idx in 0..num_blocks_per_row {
                let sb_start = row_start + sb_idx * SUPER_BLOCK_BYTES;
                if sb_start + SUPER_BLOCK_BYTES > q4k_data.len() {
                    break;
                }
                let sb_data = &q4k_data[sb_start..sb_start + SUPER_BLOCK_BYTES];
                let input_offset = sb_idx * SUPER_BLOCK_SIZE;
                process_q4k_superblock_avx2(
                    sb_data,
                    input,
                    input_offset,
                    in_dim,
                    low_mask,
                    &mut acc,
                );
            }

            *out_val = hsum_avx2(acc);
        }
    }
}
