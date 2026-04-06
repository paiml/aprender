
// ============================================================================
// FUSED Q8_0 × Q8_0 MATMUL (For Q8_0 quantized weights like Qwen2.5 LM head)
// ============================================================================
//
// Q8_0 format: 34 bytes per block (2 byte f16 scale + 32 i8 quants)
// This avoids the massive dequantization allocation that was causing
// Qwen2.5's 152K vocab LM head to allocate 544MB per forward pass.
// ============================================================================

/// AVX2 accelerated Q8_0 × Q8_0 dot product using integer SIMD
///
/// Uses AVX2 maddubs with sign trick for i8×i8 multiplication.
/// This is simpler than Q4_0×Q8_0 since no nibble unpacking is needed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
// SAFETY: Caller must satisfy the documented preconditions
unsafe fn fused_q8_0_q8_0_dot_avx2(
    q8_weight_data: &[u8],
    q8_act_scales: &[f32],
    q8_act_quants: &[i8],
    in_dim: usize,
) -> f32 {
    // SAFETY: Memory safety ensured by bounds checking and alignment
    unsafe {
        use std::arch::x86_64::{
            _mm256_cvtepi32_ps, _mm256_fmadd_ps, _mm256_loadu_si256, _mm256_madd_epi16,
            _mm256_maddubs_epi16, _mm256_set1_epi16, _mm256_set1_ps, _mm256_setzero_ps,
            _mm256_sign_epi8, _mm_cvtss_f32, _mm_hadd_ps, _mm_prefetch, _MM_HINT_T0,
        };

        const Q8_0_BLOCK_BYTES: usize = 34; // 2 byte scale + 32 byte quants
        const Q8_0_BLOCK_SIZE: usize = 32;

        let num_blocks = in_dim.div_ceil(Q8_0_BLOCK_SIZE);

        // Float accumulator for final sum
        let mut acc = _mm256_setzero_ps();
        let ones = _mm256_set1_epi16(1);

        let mut block_idx = 0;

        // Process 2 blocks at a time for better ILP
        while block_idx + 2 <= num_blocks {
            // Prefetch next iteration's blocks
            if block_idx + 4 <= num_blocks {
                let prefetch_w = q8_weight_data
                    .as_ptr()
                    .add((block_idx + 2) * Q8_0_BLOCK_BYTES);
                let prefetch_a = q8_act_quants
                    .as_ptr()
                    .add((block_idx + 2) * Q8_0_BLOCK_SIZE);
                _mm_prefetch(prefetch_w.cast(), _MM_HINT_T0);
                _mm_prefetch(prefetch_a.cast(), _MM_HINT_T0);
            }

            // === Block 0 ===
            let w_ptr_0 = q8_weight_data.as_ptr().add(block_idx * Q8_0_BLOCK_BYTES);
            let a_ptr_0 = q8_act_quants.as_ptr().add(block_idx * Q8_0_BLOCK_SIZE);

            // Read Q8_0 weight scale (f16 -> f32)
            let w_scale_bits_0 = u16::from_le_bytes([*w_ptr_0, *w_ptr_0.add(1)]);
            let w_scale_0 = f16_to_f32_lut(w_scale_bits_0);
            let a_scale_0 = q8_act_scales[block_idx];
            let combined_scale_0 = _mm256_set1_ps(w_scale_0 * a_scale_0);

            // Load Q8_0 weight quants (32 bytes at offset 2)
            let w_vec_0 = _mm256_loadu_si256(w_ptr_0.add(2).cast());
            // Load Q8_0 activation quants (32 bytes)
            let a_vec_0 = _mm256_loadu_si256(a_ptr_0.cast());

            // Integer multiply-accumulate using signed multiply trick:
            // maddubs requires unsigned × signed, so we use sign trick
            // |w| * sign(a, w) = w * a
            let w_abs_0 = _mm256_sign_epi8(w_vec_0, w_vec_0);
            let a_signed_0 = _mm256_sign_epi8(a_vec_0, w_vec_0);

            // maddubs: multiply pairs and add horizontally to i16
            let prod_i16_0 = _mm256_maddubs_epi16(w_abs_0, a_signed_0);
            // madd: pairwise add i16 to i32
            let prod_i32_0 = _mm256_madd_epi16(prod_i16_0, ones);
            // Convert to float
            let prod_f32_0 = _mm256_cvtepi32_ps(prod_i32_0);

            // Scale and accumulate
            acc = _mm256_fmadd_ps(combined_scale_0, prod_f32_0, acc);

            // === Block 1 ===
            let w_ptr_1 = q8_weight_data
                .as_ptr()
                .add((block_idx + 1) * Q8_0_BLOCK_BYTES);
            let a_ptr_1 = q8_act_quants
                .as_ptr()
                .add((block_idx + 1) * Q8_0_BLOCK_SIZE);

            let w_scale_bits_1 = u16::from_le_bytes([*w_ptr_1, *w_ptr_1.add(1)]);
            let w_scale_1 = f16_to_f32_lut(w_scale_bits_1);
            let a_scale_1 = q8_act_scales[block_idx + 1];
            let combined_scale_1 = _mm256_set1_ps(w_scale_1 * a_scale_1);

            let w_vec_1 = _mm256_loadu_si256(w_ptr_1.add(2).cast());
            let a_vec_1 = _mm256_loadu_si256(a_ptr_1.cast());

            let w_abs_1 = _mm256_sign_epi8(w_vec_1, w_vec_1);
            let a_signed_1 = _mm256_sign_epi8(a_vec_1, w_vec_1);

            let prod_i16_1 = _mm256_maddubs_epi16(w_abs_1, a_signed_1);
            let prod_i32_1 = _mm256_madd_epi16(prod_i16_1, ones);
            let prod_f32_1 = _mm256_cvtepi32_ps(prod_i32_1);

            acc = _mm256_fmadd_ps(combined_scale_1, prod_f32_1, acc);

            block_idx += 2;
        }

        // Handle remaining single block
        while block_idx < num_blocks {
            let w_ptr = q8_weight_data.as_ptr().add(block_idx * Q8_0_BLOCK_BYTES);
            let a_ptr = q8_act_quants.as_ptr().add(block_idx * Q8_0_BLOCK_SIZE);

            let w_scale_bits = u16::from_le_bytes([*w_ptr, *w_ptr.add(1)]);
            let w_scale = f16_to_f32_lut(w_scale_bits);
            let a_scale = q8_act_scales[block_idx];
            let combined_scale = _mm256_set1_ps(w_scale * a_scale);

            let w_vec = _mm256_loadu_si256(w_ptr.add(2).cast());
            let a_vec = _mm256_loadu_si256(a_ptr.cast());

            let w_abs = _mm256_sign_epi8(w_vec, w_vec);
            let a_signed = _mm256_sign_epi8(a_vec, w_vec);

            let prod_i16 = _mm256_maddubs_epi16(w_abs, a_signed);
            let prod_i32 = _mm256_madd_epi16(prod_i16, ones);
            let prod_f32 = _mm256_cvtepi32_ps(prod_i32);

            acc = _mm256_fmadd_ps(combined_scale, prod_f32, acc);

            block_idx += 1;
        }

        // Horizontal sum of 8 floats
        let hi = std::arch::x86_64::_mm256_extractf128_ps(acc, 1);
        let lo = std::arch::x86_64::_mm256_castps256_ps128(acc);
        let sum128 = std::arch::x86_64::_mm_add_ps(lo, hi);
        let sum64 = _mm_hadd_ps(sum128, sum128);
        let sum32 = _mm_hadd_ps(sum64, sum64);
        _mm_cvtss_f32(sum32)
    }
}

/// Scalar fallback for Q8_0 × Q8_0 dot product
///
/// Exposed as `pub(crate)` for direct testing on AVX2 machines.
#[inline]
pub(crate) fn fused_q8_0_q8_0_dot_scalar(
    q8_weight_data: &[u8],
    q8_act_scales: &[f32],
    q8_act_quants: &[i8],
    in_dim: usize,
) -> f32 {
    const Q8_0_BLOCK_BYTES: usize = 34;
    const Q8_0_BLOCK_SIZE: usize = 32;

    let num_blocks = in_dim.div_ceil(Q8_0_BLOCK_SIZE);
    let mut total_sum = 0.0f32;

    for block_idx in 0..num_blocks {
        let block_start = block_idx * Q8_0_BLOCK_BYTES;
        if block_start + Q8_0_BLOCK_BYTES > q8_weight_data.len() {
            break;
        }
        let block = &q8_weight_data[block_start..block_start + Q8_0_BLOCK_BYTES];

        // Read weight scale (f16)
        let w_scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let a_scale = q8_act_scales[block_idx];
        let combined_scale = w_scale * a_scale;

        let act_start = block_idx * Q8_0_BLOCK_SIZE;

        // Sum of weight_quant[i] * act_quant[i] in i32
        let mut block_sum = 0i32;
        for j in 0..32 {
            if act_start + j >= in_dim {
                break;
            }
            #[allow(clippy::cast_possible_wrap)]
            let w_quant = block[2 + j] as i8;
            let a_quant = q8_act_quants[act_start + j];
            block_sum += (w_quant as i32) * (a_quant as i32);
        }

        total_sum += combined_scale * (block_sum as f32);
    }

    total_sum
}

/// SIMD dispatcher for Q8_0 × Q8_0 dot product
#[inline]
fn fused_q8_0_q8_0_dot_simd(
    q8_weight_data: &[u8],
    q8_act_scales: &[f32],
    q8_act_quants: &[i8],
    in_dim: usize,
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2 and FMA features checked above
            unsafe {
                return fused_q8_0_q8_0_dot_avx2(
                    q8_weight_data,
                    q8_act_scales,
                    q8_act_quants,
                    in_dim,
                );
            }
        }
    }
    fused_q8_0_q8_0_dot_scalar(q8_weight_data, q8_act_scales, q8_act_quants, in_dim)
}

/// Parallel Q8_0 × Q8_0 matrix-vector multiply
///
/// This avoids the massive dequantization allocation that was causing
/// Qwen2.5's 152K vocab LM head (Q8_0) to allocate 544MB per forward pass.
///
/// For Q8_0 weights (e.g., Qwen2.5 LM head), this is ~100x faster than
/// dequantize + matmul because:
/// 1. No 544MB allocation per forward pass
/// 2. Integer SIMD is faster than FP32
/// 3. Better cache locality (34 bytes vs 128 bytes per block)
#[allow(clippy::similar_names)]
pub fn fused_q8_0_q8_0_parallel_matvec(
    weight_data: &[u8],
    activations: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> Result<Vec<f32>> {
    use rayon::prelude::*;

    const Q8_0_BLOCK_BYTES: usize = 34;
    const Q8_0_BLOCK_SIZE: usize = 32;

    let blocks_per_row = in_dim.div_ceil(Q8_0_BLOCK_SIZE);
    let bytes_per_row = blocks_per_row * Q8_0_BLOCK_BYTES;

    let expected_weight_bytes = out_dim * bytes_per_row;
    if weight_data.len() < expected_weight_bytes {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q8_0 weight data too small: need {} bytes for {}x{}, have {}",
                expected_weight_bytes,
                out_dim,
                in_dim,
                weight_data.len()
            ),
        });
    }

    if activations.len() != in_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Activation length {} doesn't match in_dim {}",
                activations.len(),
                in_dim
            ),
        });
    }

    // Quantize activations to Q8_0 ONCE (amortized over all rows)
    let (q8_scales, q8_quants) = quantize_activations_q8_0(activations);

    // Parallel over output rows with chunking
    const CHUNK_SIZE: usize = 64;
    let output: Vec<f32> = (0..out_dim)
        .into_par_iter()
        .with_min_len(CHUNK_SIZE)
        .map(|o| {
            let row_start = o * bytes_per_row;
            let row_end = row_start + bytes_per_row;
            let row_data = &weight_data[row_start..row_end];
            fused_q8_0_q8_0_dot_simd(row_data, &q8_scales, &q8_quants, in_dim)
        })
        .collect();

    Ok(output)
}

/// Fused Q8_0 × Q8_0 parallel matvec - writes to pre-allocated buffer
///
/// IMP-131: Zero-allocation variant for hot-path inference.
#[allow(clippy::similar_names)]
pub fn fused_q8_0_q8_0_parallel_matvec_into(
    weight_data: &[u8],
    activations: &[f32],
    in_dim: usize,
    out_dim: usize,
    output: &mut [f32],
) -> Result<()> {
    use rayon::prelude::*;

    const Q8_0_BLOCK_BYTES: usize = 34;
    const Q8_0_BLOCK_SIZE: usize = 32;

    let blocks_per_row = in_dim.div_ceil(Q8_0_BLOCK_SIZE);
    let bytes_per_row = blocks_per_row * Q8_0_BLOCK_BYTES;

    let expected_weight_bytes = out_dim * bytes_per_row;
    if weight_data.len() < expected_weight_bytes {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q8_0 weight data too small: need {} bytes for {}x{}, have {}",
                expected_weight_bytes,
                out_dim,
                in_dim,
                weight_data.len()
            ),
        });
    }

    if activations.len() != in_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Activation length {} doesn't match in_dim {}",
                activations.len(),
                in_dim
            ),
        });
    }

    if output.len() < out_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Output buffer too small: need {}, have {}",
                out_dim,
                output.len()
            ),
        });
    }

    // Quantize activations to Q8_0 ONCE (amortized over all rows)
    let (q8_scales, q8_quants) = quantize_activations_q8_0(activations);

    // Parallel over output rows with chunking
    const CHUNK_SIZE: usize = 64;
    output[..out_dim]
        .par_iter_mut()
        .enumerate()
        .with_min_len(CHUNK_SIZE)
        .for_each(|(o, out)| {
            let row_start = o * bytes_per_row;
            let row_end = row_start + bytes_per_row;
            let row_data = &weight_data[row_start..row_end];
            *out = fused_q8_0_q8_0_dot_simd(row_data, &q8_scales, &q8_quants, in_dim);
        });

    Ok(())
}

#[cfg(test)]
mod tests;

// T-COV-95 Coverage tests for quantize/mod.rs pure functions
#[cfg(test)]
#[path = "tests_coverage.rs"]
mod tests_coverage;
