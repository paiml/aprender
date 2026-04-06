//! Parallel dequantization functions (PMAT-802)
//!
//! Implements parallel dequantization using rayon for multi-core acceleration:
//! - `dequantize_q4_k_parallel` - Parallel Q4_K dequantization
//! - `dequantize_q4_k_simd` - SIMD-accelerated Q4_K dequantization
//! - `dequantize_q8_0_parallel` - Parallel Q8_0 dequantization
//! - `dequantize_q8_0_simd` - SIMD-accelerated Q8_0 dequantization

use super::dequant::{f16_to_f32, read_f16};
use super::simd::extract_scale_min;
use super::types::QK_K;
use crate::error::{RealizarError, Result};

// ============================================================================

/// Parallel Q4_K dequantization using rayon
///
/// Processes super-blocks in parallel for faster model loading.
/// Each super-block (256 values) is dequantized independently.
///
/// # Arguments
///
/// * `data` - Raw Q4_K quantized data (144 bytes per super-block)
///
/// # Returns
///
/// Dequantized f32 values
///
/// # Errors
///
/// Returns error if data length is not a multiple of super-block size
///
/// # Performance
///
/// On a 16-core system, achieves ~10x speedup over serial dequantization
/// for large models (1B+ parameters).
pub fn dequantize_q4_k_parallel(data: &[u8]) -> Result<Vec<f32>> {
    use rayon::prelude::*;

    const SUPER_BLOCK_BYTES: usize = 144;

    if !data.len().is_multiple_of(SUPER_BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q4_K data length {} is not a multiple of super-block size {}",
                data.len(),
                SUPER_BLOCK_BYTES
            ),
        });
    }

    let num_super_blocks = data.len() / SUPER_BLOCK_BYTES;

    // Process super-blocks in parallel
    let result: Vec<f32> = (0..num_super_blocks)
        .into_par_iter()
        .flat_map(|sb_idx| {
            let sb_start = sb_idx * SUPER_BLOCK_BYTES;
            let sb_data = &data[sb_start..sb_start + SUPER_BLOCK_BYTES];
            dequantize_q4_k_superblock(sb_data)
        })
        .collect();

    Ok(result)
}

/// Dequantize a single Q4_K super-block (256 values)
///
/// Internal helper for parallel processing.
///
/// Exposed as `pub(crate)` for direct testing.
#[inline]
pub(crate) fn dequantize_q4_k_superblock(sb_data: &[u8]) -> Vec<f32> {
    // CB-506: Real bounds check (not debug-only) prevents OOB panic on corrupted data
    if sb_data.len() < 144 {
        return vec![0.0f32; QK_K];
    }
    let mut result = vec![0.0f32; QK_K];

    // Read d (f16 -> f32) — bounds guaranteed by check above
    let d = read_f16(&sb_data[0..2]);

    // Read dmin (f16 -> f32)
    let dmin = read_f16(&sb_data[2..4]);

    // Read scales (12 bytes)
    let mut scales = [0u8; 12];
    scales.copy_from_slice(&sb_data[4..16]);

    // Read qs (128 bytes)
    let qs = &sb_data[16..144];

    // Dequantize following candle's layout:
    // For each 64-value chunk, output 32 low nibbles then 32 high nibbles
    let mut ys_index = 0;

    for j in (0..QK_K).step_by(64) {
        let q = &qs[j / 2..j / 2 + 32];

        // Get scales for the two 32-value halves
        let is = j / 32;
        let (sc1, m1) = extract_scale_min(&scales, is);
        let d1 = d * sc1;
        let dm1 = dmin * m1;

        let (sc2, m2) = extract_scale_min(&scales, is + 1);
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

    result
}

/// SIMD-accelerated Q4_K dequantization with runtime feature detection
///
/// Uses AVX2 when available, falls back to parallel scalar otherwise.
///
/// # Arguments
///
/// * `data` - Raw Q4_K quantized data (144 bytes per super-block)
///
/// # Returns
///
/// Dequantized f32 values
///
/// # Errors
///
/// Returns error if data length is not a multiple of super-block size
///
/// # Performance
///
/// AVX2: ~4x speedup over scalar per super-block
/// Combined with rayon: ~40x speedup on 16-core system
pub fn dequantize_q4_k_simd(data: &[u8]) -> Result<Vec<f32>> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 verified at runtime
            return unsafe { dequantize_q4_k_avx2_parallel(data) };
        }
    }

    // Fallback to parallel scalar
    dequantize_q4_k_parallel(data)
}

/// AVX2-accelerated parallel Q4_K dequantization
///
/// # Safety
///
/// Caller must ensure AVX2 is available (use runtime feature detection)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// SAFETY: Caller must satisfy the documented preconditions
unsafe fn dequantize_q4_k_avx2_parallel(data: &[u8]) -> Result<Vec<f32>> {
    use rayon::prelude::*;

    const SUPER_BLOCK_BYTES: usize = 144;
    // Process 64 super-blocks per parallel task to reduce scheduling overhead
    const CHUNK_SIZE: usize = 64;
    const CHUNK_BYTES: usize = SUPER_BLOCK_BYTES * CHUNK_SIZE;

    if !data.len().is_multiple_of(SUPER_BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q4_K data length {} is not a multiple of super-block size {}",
                data.len(),
                SUPER_BLOCK_BYTES
            ),
        });
    }

    let num_super_blocks = data.len() / SUPER_BLOCK_BYTES;

    // For small data, skip parallelism overhead
    if num_super_blocks < CHUNK_SIZE * 2 {
        let mut result = Vec::with_capacity(num_super_blocks * QK_K);
        for sb_idx in 0..num_super_blocks {
            let sb_start = sb_idx * SUPER_BLOCK_BYTES;
            let sb_data = &data[sb_start..sb_start + SUPER_BLOCK_BYTES];
            // SAFETY: AVX2 availability verified by caller
            result.extend(unsafe { dequantize_q4_k_superblock_avx2(sb_data) });
        }
        return Ok(result);
    }

    // Process chunks of super-blocks in parallel
    let result: Vec<f32> = data
        .par_chunks(CHUNK_BYTES)
        .flat_map(|chunk| {
            let mut chunk_result = Vec::with_capacity(chunk.len() / SUPER_BLOCK_BYTES * QK_K);
            for sb_data in chunk.chunks_exact(SUPER_BLOCK_BYTES) {
                // SAFETY: AVX2 availability verified by caller
                chunk_result.extend(unsafe { dequantize_q4_k_superblock_avx2(sb_data) });
            }
            chunk_result
        })
        .collect();

    Ok(result)
}

/// AVX2 SIMD dequantization for a single Q4_K super-block
///
/// Uses 256-bit SIMD to process 8 values simultaneously.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
// SAFETY: Caller must satisfy the documented preconditions
unsafe fn dequantize_q4_k_superblock_avx2(sb_data: &[u8]) -> Vec<f32> {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    // CB-506: Real bounds check (not debug-only) prevents OOB panic on corrupted data
    if sb_data.len() < 144 {
        return vec![0.0f32; QK_K];
    }
    let mut result = vec![0.0f32; QK_K];

    // Read d and dmin — bounds guaranteed by check above
    let d = read_f16(&sb_data[0..2]);
    let dmin = read_f16(&sb_data[2..4]);

    // SAFETY: AVX2 availability verified by caller's target_feature
    unsafe {
        // Read scales
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&sb_data[4..16]);

        let qs = &sb_data[16..144];

        // Dequantize following candle's layout:
        // For each 64-value chunk, output 32 low nibbles then 32 high nibbles
        let mut ys_index = 0;

        for j in (0..QK_K).step_by(64) {
            let q = &qs[j / 2..j / 2 + 32];

            // Get scales for the two 32-value halves
            let is = j / 32;
            let (sc1, m1) = extract_scale_min(&scales, is);
            let d1 = d * sc1;
            let dm1 = dmin * m1;
            let d1_vec = _mm256_set1_ps(d1);
            let dm1_vec = _mm256_set1_ps(dm1);

            let (sc2, m2) = extract_scale_min(&scales, is + 1);
            let d2 = d * sc2;
            let dm2 = dmin * m2;
            let d2_vec = _mm256_set1_ps(d2);
            let dm2_vec = _mm256_set1_ps(dm2);

            // First pass: 32 low nibbles in 4 iterations of 8
            for chunk in 0..4 {
                let byte_start = chunk * 8;

                // Extract 8 low nibbles from 8 bytes
                let q0 = (q[byte_start] & 0x0F) as i32;
                let q1 = (q[byte_start + 1] & 0x0F) as i32;
                let q2 = (q[byte_start + 2] & 0x0F) as i32;
                let q3 = (q[byte_start + 3] & 0x0F) as i32;
                let q4 = (q[byte_start + 4] & 0x0F) as i32;
                let q5 = (q[byte_start + 5] & 0x0F) as i32;
                let q6 = (q[byte_start + 6] & 0x0F) as i32;
                let q7 = (q[byte_start + 7] & 0x0F) as i32;

                let q_vec = _mm256_setr_epi32(q0, q1, q2, q3, q4, q5, q6, q7);
                let q_f32 = _mm256_cvtepi32_ps(q_vec);
                let dequant = _mm256_fmsub_ps(d1_vec, q_f32, dm1_vec);

                _mm256_storeu_ps(result.as_mut_ptr().add(ys_index), dequant);
                ys_index += 8;
            }

            // Second pass: 32 high nibbles in 4 iterations of 8
            for chunk in 0..4 {
                let byte_start = chunk * 8;

                // Extract 8 high nibbles from 8 bytes
                let q0 = (q[byte_start] >> 4) as i32;
                let q1 = (q[byte_start + 1] >> 4) as i32;
                let q2 = (q[byte_start + 2] >> 4) as i32;
                let q3 = (q[byte_start + 3] >> 4) as i32;
                let q4 = (q[byte_start + 4] >> 4) as i32;
                let q5 = (q[byte_start + 5] >> 4) as i32;
                let q6 = (q[byte_start + 6] >> 4) as i32;
                let q7 = (q[byte_start + 7] >> 4) as i32;

                let q_vec = _mm256_setr_epi32(q0, q1, q2, q3, q4, q5, q6, q7);
                let q_f32 = _mm256_cvtepi32_ps(q_vec);
                let dequant = _mm256_fmsub_ps(d2_vec, q_f32, dm2_vec);

                _mm256_storeu_ps(result.as_mut_ptr().add(ys_index), dequant);
                ys_index += 8;
            }
        }
    }

    result
}

/// Parallel Q8_0 dequantization using rayon
///
/// Q8_0 is simpler than Q4_K (no scale packing), making SIMD even more effective.
///
/// # Arguments
///
/// * `data` - Raw Q8_0 quantized data (36 bytes per block: 4 scale + 32 quants)
///
/// # Returns
///
/// Dequantized f32 values
///
/// # Errors
///
/// Returns error if data length is not a multiple of block size
pub fn dequantize_q8_0_parallel(data: &[u8]) -> Result<Vec<f32>> {
    use rayon::prelude::*;

    const BLOCK_BYTES: usize = 34; // 2 (f16 scale) + 32 (i8 quants)

    if !data.len().is_multiple_of(BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q8_0 data length {} is not a multiple of block size {}",
                data.len(),
                BLOCK_BYTES
            ),
        });
    }

    let num_blocks = data.len() / BLOCK_BYTES;

    // Process blocks in parallel
    let result: Vec<f32> = (0..num_blocks)
        .into_par_iter()
        .flat_map(|block_idx| {
            let block_start = block_idx * BLOCK_BYTES;
            let block_data = &data[block_start..block_start + BLOCK_BYTES];
            dequantize_q8_0_block(block_data)
        })
        .collect();

    Ok(result)
}

/// Dequantize a single Q8_0 block (32 values)
///
/// Exposed as `pub(crate)` for direct testing.
#[inline]
pub(crate) fn dequantize_q8_0_block(block_data: &[u8]) -> Vec<f32> {
    // CB-506: Real bounds check (not debug-only) prevents OOB panic on corrupted data
    if block_data.len() < 34 {
        return vec![0.0f32; 32];
    }
    let mut result = Vec::with_capacity(32);

    // Read scale (f16 -> f32) — bounds guaranteed by check above
    let scale_bits = u16::from_le_bytes([block_data[0], block_data[1]]);
    let scale = f16_to_f32(scale_bits);

    // Dequantize 32 int8 values
    for &byte in &block_data[2..34] {
        let value = i8::from_le_bytes([byte]);
        result.push(scale * f32::from(value));
    }

    result
}

/// SIMD-accelerated Q8_0 dequantization
///
/// Uses AVX2 when available for 8x parallel i8→f32 conversion.
///
/// # Errors
///
/// Returns error if data length is not a multiple of block size (36 bytes)
pub fn dequantize_q8_0_simd(data: &[u8]) -> Result<Vec<f32>> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 verified at runtime
            return unsafe { dequantize_q8_0_avx2_parallel(data) };
        }
    }

    // Fallback to parallel scalar
    dequantize_q8_0_parallel(data)
}

/// AVX2-accelerated parallel Q8_0 dequantization
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// SAFETY: Caller must satisfy the documented preconditions
unsafe fn dequantize_q8_0_avx2_parallel(data: &[u8]) -> Result<Vec<f32>> {
    use rayon::prelude::*;

    const BLOCK_BYTES: usize = 34; // 2 (f16 scale) + 32 (i8 quants)

    if !data.len().is_multiple_of(BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q8_0 data length {} is not a multiple of block size {}",
                data.len(),
                BLOCK_BYTES
            ),
        });
    }

    let num_blocks = data.len() / BLOCK_BYTES;

    let result: Vec<f32> = (0..num_blocks)
        .into_par_iter()
        .flat_map(|block_idx| {
            let block_start = block_idx * BLOCK_BYTES;
            let block_data = &data[block_start..block_start + BLOCK_BYTES];
            // SAFETY: AVX2 availability verified by caller
            unsafe { dequantize_q8_0_block_avx2(block_data) }
        })
        .collect();

    Ok(result)
}

include!("rope.rs");
include!("parallel_dequant_dequantize.rs");
