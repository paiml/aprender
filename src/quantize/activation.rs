//! Fused activation functions for quantized inference (PMAT-802)
//!
//! Implements fused operations combining normalization with quantization:
//! - `quantize_rmsnorm_q8_0` - RMSNorm with Q8_0 quantization
//! - `quantize_rmsnorm_q8_0_into` - Zero-allocation variant
//! - `fused_rmsnorm_q4_0_matmul` - RMSNorm + matmul fusion
//! - `fused_rmsnorm_ffn_up_gate` - RMSNorm + FFN up/gate fusion
//! - `fused_swiglu_simd` - SIMD-accelerated SwiGLU activation
//! - `softmax_simd` - SIMD-accelerated softmax

use super::fused_q4_0_q8_0_dot_simd;
use crate::error::{RealizarError, Result};

// ============================================================================
// Key insight: llama.cpp quantizes activations to Q8_0 and uses integer
// multiply-accumulate (maddubs_epi16), which is 4-5x faster than f32 FMA.

/// Fused RMSNorm + Q8_0 quantization
///
/// Computes RMSNorm and quantizes in a single pass:
/// normalized[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]
/// Then quantizes to Q8_0 format.
///
/// This avoids allocating an intermediate normalized vector.
#[inline]
pub fn quantize_rmsnorm_q8_0(input: &[f32], norm_weight: &[f32], eps: f32) -> (Vec<f32>, Vec<i8>) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: Memory safety ensured by bounds checking and alignment
            return unsafe { quantize_rmsnorm_q8_0_avx2(input, norm_weight, eps) };
        }
    }
    quantize_rmsnorm_q8_0_scalar(input, norm_weight, eps)
}

/// Scalar implementation of fused RMSNorm + Q8_0 quantization
///
/// This is exposed as `pub(crate)` for direct testing. The production code
/// uses the public `quantize_rmsnorm_q8_0` wrapper which dispatches to AVX2
/// when available.
pub(crate) fn quantize_rmsnorm_q8_0_scalar(
    input: &[f32],
    norm_weight: &[f32],
    eps: f32,
) -> (Vec<f32>, Vec<i8>) {
    let hidden_dim = input.len();
    debug_assert_eq!(hidden_dim, norm_weight.len());

    // Compute sum of squares for RMSNorm
    let sum_sq: f32 = input.iter().map(|x| x * x).sum();
    let mean_sq = sum_sq / hidden_dim as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();

    // Now quantize the normalized values directly
    let num_blocks = hidden_dim.div_ceil(32);
    let mut scales = Vec::with_capacity(num_blocks);
    let mut quants = Vec::with_capacity(num_blocks * 32);

    for block_idx in 0..num_blocks {
        let start = block_idx * 32;
        let end = (start + 32).min(hidden_dim);

        // Find max absolute value of normalized values for this block
        let mut max_abs = 0.0f32;
        for i in start..end {
            // Fused: x[i] * inv_rms * weight[i]
            let normalized = input[i] * inv_rms * norm_weight[i];
            let abs = normalized.abs();
            if abs > max_abs {
                max_abs = abs;
            }
        }

        // Compute scale
        let scale = if max_abs > 1e-10 {
            max_abs / 127.0
        } else {
            1.0 / 127.0
        };
        let inv_scale = 1.0 / scale;
        scales.push(scale);

        // Quantize normalized values
        for i in start..end {
            let normalized = input[i] * inv_rms * norm_weight[i];
            let q = (normalized * inv_scale).round();
            quants.push(q.clamp(-128.0, 127.0) as i8);
        }
        // Pad to 32 if partial block
        for _ in end..(start + 32) {
            quants.push(0i8);
        }
    }

    (scales, quants)
}

/// AVX2-accelerated fused RMSNorm + Q8_0 quantization
///
/// Processes 8 floats at a time using SIMD for:
/// - Sum of squares computation
/// - Max abs finding per block
/// - Normalization and quantization
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
// SAFETY: Caller must satisfy the documented preconditions
unsafe fn quantize_rmsnorm_q8_0_avx2(
    input: &[f32],
    norm_weight: &[f32],
    eps: f32,
) -> (Vec<f32>, Vec<i8>) {
    // SAFETY: Memory safety ensured by bounds checking and alignment
    unsafe {
        use std::arch::x86_64::{
            _mm256_add_ps, _mm256_and_ps, _mm256_andnot_ps, _mm256_castps256_ps128,
            _mm256_castsi256_ps, _mm256_castsi256_si128, _mm256_cvtps_epi32, _mm256_extractf128_ps,
            _mm256_extracti128_si256, _mm256_floor_ps, _mm256_fmadd_ps, _mm256_loadu_ps,
            _mm256_max_ps, _mm256_min_ps, _mm256_mul_ps, _mm256_or_ps, _mm256_set1_epi32,
            _mm256_set1_ps, _mm256_setzero_ps, _mm_add_ps, _mm_cvtss_f32, _mm_hadd_ps, _mm_max_ps,
            _mm_movehl_ps, _mm_packs_epi16, _mm_packs_epi32, _mm_shuffle_ps, _mm_storel_epi64,
        };

        let hidden_dim = input.len();
        debug_assert_eq!(hidden_dim, norm_weight.len());

        // SIMD sum of squares
        let mut sum_sq_vec = _mm256_setzero_ps();
        let mut i = 0;

        // Process 8 floats at a time
        while i + 8 <= hidden_dim {
            let v = _mm256_loadu_ps(input.as_ptr().add(i));
            sum_sq_vec = _mm256_fmadd_ps(v, v, sum_sq_vec);
            i += 8;
        }

        // Horizontal sum
        let hi = _mm256_extractf128_ps(sum_sq_vec, 1);
        let lo = _mm256_castps256_ps128(sum_sq_vec);
        let sum128 = _mm_add_ps(lo, hi);
        let sum64 = _mm_hadd_ps(sum128, sum128);
        let sum32 = _mm_hadd_ps(sum64, sum64);
        let mut sum_sq = _mm_cvtss_f32(sum32);

        // Handle remaining elements
        while i < hidden_dim {
            sum_sq += input[i] * input[i];
            i += 1;
        }

        let mean_sq = sum_sq / hidden_dim as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        let inv_rms_vec = _mm256_set1_ps(inv_rms);

        // Quantize with SIMD
        let num_blocks = hidden_dim.div_ceil(32);
        let mut scales = Vec::with_capacity(num_blocks);
        let mut quants = vec![0i8; num_blocks * 32];

        let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF_u32 as i32));
        let round_const = _mm256_set1_ps(0.5);
        let clamp_min = _mm256_set1_ps(-128.0);
        let clamp_max = _mm256_set1_ps(127.0);

        for block_idx in 0..num_blocks {
            let start = block_idx * 32;
            let block_end = (start + 32).min(hidden_dim);
            let valid_len = block_end - start;

            // Find max abs in this block using SIMD
            let mut max_vec = _mm256_setzero_ps();
            let mut j = 0;
            while j + 8 <= valid_len {
                let idx = start + j;
                let inp = _mm256_loadu_ps(input.as_ptr().add(idx));
                let wgt = _mm256_loadu_ps(norm_weight.as_ptr().add(idx));
                let normalized = _mm256_mul_ps(_mm256_mul_ps(inp, inv_rms_vec), wgt);
                let abs_val = _mm256_and_ps(normalized, abs_mask);
                max_vec = _mm256_max_ps(max_vec, abs_val);
                j += 8;
            }

            // Horizontal max
            let max_hi = _mm256_extractf128_ps(max_vec, 1);
            let max_lo = _mm256_castps256_ps128(max_vec);
            let max_128 = _mm_max_ps(max_lo, max_hi);
            let max_64 = _mm_max_ps(max_128, _mm_movehl_ps(max_128, max_128));
            let max_32 = _mm_max_ps(max_64, _mm_shuffle_ps(max_64, max_64, 1));
            let mut max_abs = _mm_cvtss_f32(max_32);

            // Handle remaining elements in block
            while j < valid_len {
                let normalized = input[start + j] * inv_rms * norm_weight[start + j];
                let abs = normalized.abs();
                if abs > max_abs {
                    max_abs = abs;
                }
                j += 1;
            }

            // Compute scale
            let scale = if max_abs > 1e-10 {
                max_abs / 127.0
            } else {
                1.0 / 127.0
            };
            let inv_scale = 1.0 / scale;
            let inv_scale_vec = _mm256_set1_ps(inv_scale);
            scales.push(scale);

            // Quantize with SIMD
            let quant_ptr = quants.as_mut_ptr().add(block_idx * 32);
            let mut k = 0;
            while k + 8 <= valid_len {
                let idx = start + k;
                let inp = _mm256_loadu_ps(input.as_ptr().add(idx));
                let wgt = _mm256_loadu_ps(norm_weight.as_ptr().add(idx));
                let normalized = _mm256_mul_ps(_mm256_mul_ps(inp, inv_rms_vec), wgt);
                let scaled = _mm256_mul_ps(normalized, inv_scale_vec);

                // Round to nearest (add 0.5 and truncate, handle sign)
                let sign = _mm256_and_ps(
                    scaled,
                    _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000_u32 as i32)),
                );
                let abs_scaled = _mm256_andnot_ps(sign, scaled);
                let rounded = _mm256_or_ps(
                    _mm256_floor_ps(_mm256_add_ps(abs_scaled, round_const)),
                    sign,
                );

                // Clamp to [-128, 127]
                let clamped = _mm256_max_ps(clamp_min, _mm256_min_ps(clamp_max, rounded));

                // Convert to int32 and extract to i8
                let int32 = _mm256_cvtps_epi32(clamped);

                // Pack i32 -> i16 -> i8 (only need lower 8 values)
                let lo128 = _mm256_castsi256_si128(int32);
                let hi128 = _mm256_extracti128_si256(int32, 1);
                let packed16 = _mm_packs_epi32(lo128, hi128);
                let packed8 = _mm_packs_epi16(packed16, packed16);

                // Store 8 i8 values
                _mm_storel_epi64(quant_ptr.add(k).cast(), packed8);
                k += 8;
            }

            // Handle remaining elements
            while k < valid_len {
                let normalized = input[start + k] * inv_rms * norm_weight[start + k];
                let q = (normalized * inv_scale).round();
                *quant_ptr.add(k) = q.clamp(-128.0, 127.0) as i8;
                k += 1;
            }
        }

        (scales, quants)
    }
}

/// Fused RMSNorm + Q4_0 matmul
///
/// Combines RMSNorm normalization with quantized matmul in one operation:
/// 1. Computes inv_rms = 1 / sqrt(mean(x^2) + eps)
/// 2. Quantizes (x * inv_rms * norm_weight) to Q8_0
/// 3. Performs Q4_0 × Q8_0 integer matmul
///
/// This eliminates the intermediate normalized vector allocation.
#[allow(clippy::similar_names)]
pub fn fused_rmsnorm_q4_0_matmul(
    input: &[f32],
    norm_weight: &[f32],
    eps: f32,
    weight_data: &[u8],
    in_dim: usize,
    out_dim: usize,
) -> Result<Vec<f32>> {
    use rayon::prelude::*;

    const Q4_0_BLOCK_BYTES: usize = 18;
    const Q4_0_BLOCK_SIZE: usize = 32;

    if input.len() != in_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Input length {} doesn't match in_dim {}",
                input.len(),
                in_dim
            ),
        });
    }

    let blocks_per_row = in_dim.div_ceil(Q4_0_BLOCK_SIZE);
    let bytes_per_row = blocks_per_row * Q4_0_BLOCK_BYTES;

    let expected_weight_bytes = out_dim * bytes_per_row;
    if weight_data.len() < expected_weight_bytes {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q4_0 weight data too small: need {} bytes for {}x{}, have {}",
                expected_weight_bytes,
                out_dim,
                in_dim,
                weight_data.len()
            ),
        });
    }

    // Fused RMSNorm + Q8_0 quantization (single pass, no intermediate allocation)
    let (q8_scales, q8_quants) = quantize_rmsnorm_q8_0(input, norm_weight, eps);

    // Parallel matmul with chunking
    const CHUNK_SIZE: usize = 64;
    let output: Vec<f32> = (0..out_dim)
        .into_par_iter()
        .with_min_len(CHUNK_SIZE)
        .map(|o| {
            let row_start = o * bytes_per_row;
            let row_end = row_start + bytes_per_row;
            let row_data = &weight_data[row_start..row_end];
            fused_q4_0_q8_0_dot_simd(row_data, &q8_scales, &q8_quants, in_dim)
        })
        .collect();

    Ok(output)
}

/// Fused RMSNorm + parallel FFN up/gate projections
///
/// For SwiGLU models, FFN has two parallel matmuls (up and gate) that share
/// the same normalized input. This function:
/// 1. Computes inv_rms once
/// 2. Quantizes normalized input to Q8_0 once
/// 3. Runs both up and gate matmuls in parallel
///
/// Eliminates: 1 RMSNorm pass, 1 intermediate allocation, 1 Q8_0 quantization
#[allow(clippy::similar_names)]
#[allow(clippy::too_many_arguments)]
pub fn fused_rmsnorm_ffn_up_gate(
    input: &[f32],
    norm_weight: &[f32],
    eps: f32,
    up_weight_data: &[u8],
    gate_weight_data: &[u8],
    in_dim: usize,
    out_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    use rayon::prelude::*;

    const Q4_0_BLOCK_BYTES: usize = 18;
    const Q4_0_BLOCK_SIZE: usize = 32;

    if input.len() != in_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Input length {} doesn't match in_dim {}",
                input.len(),
                in_dim
            ),
        });
    }

    let blocks_per_row = in_dim.div_ceil(Q4_0_BLOCK_SIZE);
    let bytes_per_row = blocks_per_row * Q4_0_BLOCK_BYTES;
    let expected_weight_bytes = out_dim * bytes_per_row;

    if up_weight_data.len() < expected_weight_bytes {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "FFN up weight data too small: need {} bytes, have {}",
                expected_weight_bytes,
                up_weight_data.len()
            ),
        });
    }
    if gate_weight_data.len() < expected_weight_bytes {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "FFN gate weight data too small: need {} bytes, have {}",
                expected_weight_bytes,
                gate_weight_data.len()
            ),
        });
    }

    // Fused RMSNorm + Q8_0 quantization - computed ONCE for both matmuls
    let (q8_scales, q8_quants) = quantize_rmsnorm_q8_0(input, norm_weight, eps);

    // Run both matmuls in parallel using rayon::join
    // Each matmul uses parallel iteration with chunking to reduce overhead
    let (up_output, gate_output) = rayon::join(
        || {
            const CHUNK_SIZE: usize = 64;
            (0..out_dim)
                .into_par_iter()
                .with_min_len(CHUNK_SIZE)
                .map(|o| {
                    let row_start = o * bytes_per_row;
                    let row_end = row_start + bytes_per_row;
                    let row_data = &up_weight_data[row_start..row_end];
                    fused_q4_0_q8_0_dot_simd(row_data, &q8_scales, &q8_quants, in_dim)
                })
                .collect::<Vec<f32>>()
        },
        || {
            const CHUNK_SIZE: usize = 64;
            (0..out_dim)
                .into_par_iter()
                .with_min_len(CHUNK_SIZE)
                .map(|o| {
                    let row_start = o * bytes_per_row;
                    let row_end = row_start + bytes_per_row;
                    let row_data = &gate_weight_data[row_start..row_end];
                    fused_q4_0_q8_0_dot_simd(row_data, &q8_scales, &q8_quants, in_dim)
                })
                .collect::<Vec<f32>>()
        },
    );

    Ok((up_output, gate_output))
}

include!("quantize_rmsnorm_into.rs");
include!("activation_quantize_activations.rs");
