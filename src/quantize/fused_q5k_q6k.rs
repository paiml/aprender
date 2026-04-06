//! Fused Q5K and Q6K dot product operations (PMAT-802)
//!
//! Implements fused dequant+dot operations for Q5_K and Q6_K formats:
//! - `fused_q6k_dot`, `fused_q6k_dot_simd` - Q6_K dot products
//! - `fused_q5k_dot`, `fused_q5k_dot_simd` - Q5_K dot products

use super::dequant::read_f16;
use super::simd::extract_scale_min;
use super::types::{Q8_0Block, QK_K};
use crate::error::{RealizarError, Result};

/// Fused Q6_K dequantize + dot product
///
/// Computes the dot product of Q6_K quantized weights with f32 activations.
pub fn fused_q6k_dot(q6k_data: &[u8], activations: &[f32]) -> Result<f32> {
    const SUPER_BLOCK_BYTES: usize = 210;

    // Validate Q6_K data length
    if !q6k_data.len().is_multiple_of(SUPER_BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q6_K data length {} is not a multiple of super-block size {}",
                q6k_data.len(),
                SUPER_BLOCK_BYTES
            ),
        });
    }

    let num_super_blocks = q6k_data.len() / SUPER_BLOCK_BYTES;
    let expected_values = num_super_blocks * QK_K;

    // Validate activation length matches
    if activations.len() != expected_values {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Activation length {} doesn't match Q6_K values count {}",
                activations.len(),
                expected_values
            ),
        });
    }

    // Accumulator for dot product result
    let mut acc = 0.0f32;

    for sb_idx in 0..num_super_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let act_start = sb_idx * QK_K;

        // Q6_K layout: ql (128) + qh (64) + scales (16) + d (2)
        let ql = &q6k_data[sb_start..sb_start + 128];
        let qh = &q6k_data[sb_start + 128..sb_start + 192];

        // Read scales (16 bytes, i8)
        let mut scales = [0i8; 16];
        for (i, scale) in scales.iter_mut().enumerate() {
            #[allow(clippy::cast_possible_wrap)]
            {
                *scale = q6k_data[sb_start + 192 + i] as i8;
            }
        }

        // Read d (f16 -> f32) at offset 208
        let d = read_f16(&q6k_data[sb_start + 208..sb_start + 210]);

        // Fused dequant+dot following candle's exact layout
        // Process 128 values at a time (n=0, n=128)
        for n in (0..QK_K).step_by(128) {
            let idx = n / 128;
            let sc = &scales[8 * idx..];
            let ql_slice = &ql[64 * idx..];
            let qh_slice = &qh[32 * idx..];

            for l in 0..32 {
                let is = l / 16; // Scale index selector

                // Extract 4 values per iteration (at positions l, l+32, l+64, l+96)
                let q1 = ((ql_slice[l] & 0xF) | ((qh_slice[l] & 3) << 4)) as i32 - 32;
                let q2 = ((ql_slice[l + 32] & 0xF) | (((qh_slice[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql_slice[l] >> 4) | (((qh_slice[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql_slice[l + 32] >> 4) | (((qh_slice[l] >> 6) & 3) << 4)) as i32 - 32;

                // Dequantize and accumulate dot product
                let v1 = d * (sc[is] as f32) * (q1 as f32);
                let v2 = d * (sc[is + 2] as f32) * (q2 as f32);
                let v3 = d * (sc[is + 4] as f32) * (q3 as f32);
                let v4 = d * (sc[is + 6] as f32) * (q4 as f32);

                acc += v1 * activations[act_start + n + l];
                acc += v2 * activations[act_start + n + l + 32];
                acc += v3 * activations[act_start + n + l + 64];
                acc += v4 * activations[act_start + n + l + 96];
            }
        }
    }

    Ok(acc)
}

/// SIMD-accelerated fused Q6_K dequant+dot (with scalar fallback)
///
/// Per Williams et al. (2009) roofline model, memory bandwidth is the bottleneck.
/// This function provides a unified interface with runtime feature detection.
/// Currently uses scalar implementation; SIMD Q6_K optimization can be added later.
///
/// # Arguments
///
/// * `q6k_data` - Raw Q6_K quantized data (210 bytes per super-block)
/// * `activations` - Input activations (256 values per super-block)
///
/// # Returns
///
/// Dot product result as f32
///
/// # Errors
///
/// Returns error if data sizes don't match or are malformed
pub fn fused_q6k_dot_simd(q6k_data: &[u8], activations: &[f32]) -> Result<f32> {
    // PAR-126: AVX2 SIMD implementation for Q6_K
    // Critical optimization: Q6_K scalar was 9x slower than Q4_K SIMD
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: We've verified AVX2 and FMA are available at runtime
            return unsafe { fused_q6k_dot_avx2(q6k_data, activations) };
        }
    }
    // pmat-ignore: hardware-path (scalar fallback tested directly via fused_q6k_dot)
    // Fallback to scalar implementation
    fused_q6k_dot(q6k_data, activations)
}

/// PAR-126: AVX2 SIMD implementation for Q6_K dot product
///
/// Uses AVX2 + FMA to achieve ~8x speedup over scalar.
/// Q6_K layout: ql (128) + qh (64) + scales (16) + d (2) = 210 bytes
///
/// # Safety
/// Requires AVX2 and FMA instruction sets
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(unsafe_op_in_unsafe_fn)]
// SAFETY: Caller must satisfy the documented preconditions
unsafe fn fused_q6k_dot_avx2(q6k_data: &[u8], activations: &[f32]) -> Result<f32> {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    const SUPER_BLOCK_BYTES: usize = 210;

    if !q6k_data.len().is_multiple_of(SUPER_BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q6_K data length {} is not a multiple of super-block size {}",
                q6k_data.len(),
                SUPER_BLOCK_BYTES
            ),
        });
    }

    let num_super_blocks = q6k_data.len() / SUPER_BLOCK_BYTES;
    let expected_values = num_super_blocks * QK_K;

    if activations.len() != expected_values {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Activation length {} doesn't match Q6_K values count {}",
                activations.len(),
                expected_values
            ),
        });
    }

    // 4 independent accumulators to hide FMA latency
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    // Masks for 6-bit extraction (reserved for optimized path)
    let _mask_0f = _mm256_set1_epi8(0x0F_i8);
    let _mask_03 = _mm256_set1_epi8(0x03_i8);
    let offset_32 = _mm256_set1_epi32(32);

    for sb_idx in 0..num_super_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let act_start = sb_idx * QK_K;

        // Prefetch next super-block (dual: Q6_K weights + activations)
        if sb_idx + 1 < num_super_blocks {
            let next_sb = (sb_idx + 1) * SUPER_BLOCK_BYTES;
            _mm_prefetch(q6k_data.as_ptr().add(next_sb).cast::<i8>(), _MM_HINT_T0);
            _mm_prefetch(
                activations.as_ptr().add((sb_idx + 1) * QK_K).cast::<i8>(),
                _MM_HINT_T0,
            );
        }

        // Q6_K layout: ql (128) + qh (64) + scales (16) + d (2)
        let ql_ptr = q6k_data.as_ptr().add(sb_start);
        let qh_ptr = q6k_data.as_ptr().add(sb_start + 128);
        let scales_ptr = q6k_data.as_ptr().add(sb_start + 192);

        // Read d (f16 -> f32)
        let d = read_f16(&q6k_data[sb_start + 208..sb_start + 210]);
        let d_vec = _mm256_set1_ps(d);

        // Read all 16 scales (as i8, will use in inner loop)
        let mut scales = [0i8; 16];
        std::ptr::copy_nonoverlapping(scales_ptr, scales.as_mut_ptr().cast::<u8>(), 16);

        // Process 128 values at a time (n=0, n=128)
        for n in (0..QK_K).step_by(128) {
            let idx = n / 128;
            let sc = &scales[8 * idx..];
            let ql_slice = ql_ptr.add(64 * idx);
            let qh_slice = qh_ptr.add(32 * idx);
            let act_base = activations.as_ptr().add(act_start + n);

            // Process 32 values at a time using AVX2
            // Each iteration handles l=0..8, extracting 4 values each (32 total)
            for l_base in (0..32).step_by(8) {
                // Load 8 bytes of ql[l], ql[l+32], qh[l]
                let ql_lo_64 = std::ptr::read_unaligned(ql_slice.add(l_base).cast::<u64>());
                let ql_hi_64 = std::ptr::read_unaligned(ql_slice.add(l_base + 32).cast::<u64>());
                let qh_64 = std::ptr::read_unaligned(qh_slice.add(l_base).cast::<u64>());

                // Convert to SIMD vectors (expand u8 to i32 for arithmetic)
                let ql_lo = _mm256_cvtepu8_epi32(_mm_set_epi64x(0, ql_lo_64 as i64));
                let ql_hi = _mm256_cvtepu8_epi32(_mm_set_epi64x(0, ql_hi_64 as i64));
                let qh = _mm256_cvtepu8_epi32(_mm_set_epi64x(0, qh_64 as i64));

                // Extract 6-bit values (4 values per input byte)
                // q1 = (ql[l] & 0xF) | ((qh[l] & 3) << 4) - 32
                let q1_lo = _mm256_and_si256(ql_lo, _mm256_set1_epi32(0x0F));
                let q1_hi = _mm256_slli_epi32(_mm256_and_si256(qh, _mm256_set1_epi32(0x03)), 4);
                let q1 = _mm256_sub_epi32(_mm256_or_si256(q1_lo, q1_hi), offset_32);

                // q2 = (ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4) - 32
                let q2_lo = _mm256_and_si256(ql_hi, _mm256_set1_epi32(0x0F));
                let q2_hi = _mm256_slli_epi32(
                    _mm256_and_si256(_mm256_srli_epi32(qh, 2), _mm256_set1_epi32(0x03)),
                    4,
                );
                let q2 = _mm256_sub_epi32(_mm256_or_si256(q2_lo, q2_hi), offset_32);

                // q3 = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4) - 32
                let q3_lo = _mm256_srli_epi32(ql_lo, 4);
                let q3_hi = _mm256_slli_epi32(
                    _mm256_and_si256(_mm256_srli_epi32(qh, 4), _mm256_set1_epi32(0x03)),
                    4,
                );
                let q3 = _mm256_sub_epi32(_mm256_or_si256(q3_lo, q3_hi), offset_32);

                // q4 = (ql[l+32] >> 4) | (((qh[l] >> 6) & 3) << 4) - 32
                let q4_lo = _mm256_srli_epi32(ql_hi, 4);
                let q4_hi = _mm256_slli_epi32(_mm256_srli_epi32(qh, 6), 4);
                let q4 = _mm256_sub_epi32(_mm256_or_si256(q4_lo, q4_hi), offset_32);

                // Determine scale index: is = l / 16 (0 for l<16, 1 for l>=16)
                let is = l_base / 16;

                // Get scales for each of the 4 output values
                let sc1 = sc[is] as f32;
                let sc2 = sc[is + 2] as f32;
                let sc3 = sc[is + 4] as f32;
                let sc4 = sc[is + 6] as f32;

                // Convert quantized values to f32 and multiply by d*scale
                let q1_f32 = _mm256_cvtepi32_ps(q1);
                let q2_f32 = _mm256_cvtepi32_ps(q2);
                let q3_f32 = _mm256_cvtepi32_ps(q3);
                let q4_f32 = _mm256_cvtepi32_ps(q4);

                let dequant1 = _mm256_mul_ps(_mm256_mul_ps(d_vec, _mm256_set1_ps(sc1)), q1_f32);
                let dequant2 = _mm256_mul_ps(_mm256_mul_ps(d_vec, _mm256_set1_ps(sc2)), q2_f32);
                let dequant3 = _mm256_mul_ps(_mm256_mul_ps(d_vec, _mm256_set1_ps(sc3)), q3_f32);
                let dequant4 = _mm256_mul_ps(_mm256_mul_ps(d_vec, _mm256_set1_ps(sc4)), q4_f32);

                // Load activations
                let act1 = _mm256_loadu_ps(act_base.add(l_base));
                let act2 = _mm256_loadu_ps(act_base.add(l_base + 32));
                let act3 = _mm256_loadu_ps(act_base.add(l_base + 64));
                let act4 = _mm256_loadu_ps(act_base.add(l_base + 96));

                // FMA: acc += dequant * act
                acc0 = _mm256_fmadd_ps(dequant1, act1, acc0);
                acc1 = _mm256_fmadd_ps(dequant2, act2, acc1);
                acc2 = _mm256_fmadd_ps(dequant3, act3, acc2);
                acc3 = _mm256_fmadd_ps(dequant4, act4, acc3);
            }
        }
    }

    // Combine 4 accumulators
    let acc_01 = _mm256_add_ps(acc0, acc1);
    let acc_23 = _mm256_add_ps(acc2, acc3);
    let acc = _mm256_add_ps(acc_01, acc_23);

    // Horizontal sum
    let sum_halves = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    let temp = _mm_add_ps(sum_halves, _mm_movehl_ps(sum_halves, sum_halves));
    let temp = _mm_add_ss(temp, _mm_shuffle_ps(temp, temp, 1));
    let result = _mm_cvtss_f32(temp);

    Ok(result)
}

/// Fused Q5_K dequantize + dot product
///
/// Computes the dot product of Q5_K quantized weights with f32 activations
/// WITHOUT allocating an intermediate f32 buffer. Dequantization happens
/// inline, accumulating directly into a register.
///
/// # Arguments
///
/// * `q5k_data` - Raw Q5_K quantized data (super-blocks of 176 bytes)
/// * `activations` - f32 activation values (must match dequantized length)
///
/// # Returns
///
/// The dot product as f32
///
/// # Errors
///
/// Returns error if:
/// - `q5k_data` length is not a multiple of 176 bytes (super-block size)
/// - `activations` length doesn't match the number of quantized values
///
/// # Examples
///
/// ```rust,ignore
/// let weights_q5k = load_q5k_weights();
/// let activations = get_layer_activations();
/// let result = fused_q5k_dot(&weights_q5k, &activations)?;
/// ```
#[allow(clippy::similar_names)]
pub fn fused_q5k_dot(q5k_data: &[u8], activations: &[f32]) -> Result<f32> {
    const SUPER_BLOCK_BYTES: usize = 176;

    // Validate Q5_K data length
    if !q5k_data.len().is_multiple_of(SUPER_BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q5_K data length {} is not a multiple of super-block size {}",
                q5k_data.len(),
                SUPER_BLOCK_BYTES
            ),
        });
    }

    let num_super_blocks = q5k_data.len() / SUPER_BLOCK_BYTES;
    let expected_values = num_super_blocks * QK_K;

    // Validate activation length matches
    if activations.len() != expected_values {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Activation length {} doesn't match Q5_K values count {}",
                activations.len(),
                expected_values
            ),
        });
    }

    // Accumulator for dot product result
    let mut acc = 0.0f32;
    let mut activation_idx = 0;

    for sb_idx in 0..num_super_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;

        // Read d (f16 -> f32)
        let d = read_f16(&q5k_data[sb_start..sb_start + 2]);

        // Read dmin (f16 -> f32)
        let dmin = read_f16(&q5k_data[sb_start + 2..sb_start + 4]);

        // Read scales (12 bytes)
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&q5k_data[sb_start + 4..sb_start + 16]);

        // Read qh - high bits (32 bytes)
        let qh_start = sb_start + 16;
        let qh = &q5k_data[qh_start..qh_start + 32];

        // Read qs - low 4 bits (128 bytes)
        let qs_start = sb_start + 48;
        let qs = &q5k_data[qs_start..qs_start + 128];

        // Fused dequant+dot for 8 blocks of 32 values each
        for block_idx in 0..8 {
            // Extract 6-bit scale and min for this block
            let (scale, min) = extract_scale_min(&scales, block_idx);

            // Process 32 values
            let block_start = block_idx * 16;
            let qh_block_start = block_idx * 4;

            for byte_idx in 0..16 {
                let qs_byte = qs[block_start + byte_idx];
                let high_bits_byte = qh[qh_block_start + byte_idx / 4];
                let bit_offset = (byte_idx % 4) * 2;

                // Low value: dequantize and accumulate
                let q_low_4bit = qs_byte & 0x0F;
                let q_low_high_bit = (high_bits_byte >> bit_offset) & 0x01;
                #[allow(clippy::cast_possible_wrap)]
                let q_low = ((q_low_high_bit << 4) | q_low_4bit) as i8;
                let value_low = d * scale * f32::from(q_low) - dmin * min;
                acc += value_low * activations[activation_idx];
                activation_idx += 1;

                // High value: dequantize and accumulate
                let q_high_4bit = (qs_byte >> 4) & 0x0F;
                let q_high_high_bit = (high_bits_byte >> (bit_offset + 1)) & 0x01;
                #[allow(clippy::cast_possible_wrap)]
                let q_high = ((q_high_high_bit << 4) | q_high_4bit) as i8;
                let value_high = d * scale * f32::from(q_high) - dmin * min;
                acc += value_high * activations[activation_idx];
                activation_idx += 1;
            }
        }
    }

    Ok(acc)
}

/// SIMD-accelerated fused Q5_K dequant+dot (with scalar fallback)
///
/// Provides unified interface with runtime feature detection.
/// Currently uses scalar implementation; SIMD Q5_K can be added later.
///
/// # Errors
///
/// Returns error if data sizes don't match or are malformed.
/// See [`fused_q5k_dot`] for details.
pub fn fused_q5k_dot_simd(q5k_data: &[u8], activations: &[f32]) -> Result<f32> {
    // Q5_K SIMD optimization deferred to Phase 2
    fused_q5k_dot(q5k_data, activations)
}

include!("fused_q4k_q8_dot.rs");
include!("create.rs");
