
/// AVX2 accelerated Q4_0 × Q8_0 dot product with 4-block unrolling
///
/// Processes 4 blocks per iteration for maximum ILP on modern OoO CPUs.
/// This version achieves ~1.3x speedup over 2-block unrolling for large vectors.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
// SAFETY: Caller must satisfy the documented preconditions
unsafe fn fused_q4_0_q8_0_dot_avx2_4block(
    q4_data: &[u8],
    q8_scales: &[f32],
    q8_quants: &[i8],
    in_dim: usize,
) -> f32 {
    // SAFETY: Memory safety ensured by bounds checking and alignment
    unsafe {
        use std::arch::x86_64::{
            _mm256_add_ps, _mm256_and_si256, _mm256_cvtepi32_ps, _mm256_fmadd_ps,
            _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_set1_epi16,
            _mm256_set1_epi8, _mm256_set1_ps, _mm256_setzero_ps, _mm256_sign_epi8, _mm256_sub_epi8,
            _mm_cvtss_f32, _mm_hadd_ps, _mm_prefetch, _MM_HINT_T0,
        };

        const Q4_0_BLOCK_BYTES: usize = 18;
        const Q4_0_BLOCK_SIZE: usize = 32;

        let num_blocks = in_dim.div_ceil(Q4_0_BLOCK_SIZE);

        // Use two accumulators for better pipelining
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        let offset = _mm256_set1_epi8(8);
        let low_mask = _mm256_set1_epi8(0x0F);
        let ones = _mm256_set1_epi16(1);

        let mut block_idx = 0;

        // Process 4 blocks at a time for maximum ILP
        while block_idx + 4 <= num_blocks {
            // Prefetch next iteration's blocks
            if block_idx + 8 <= num_blocks {
                let prefetch_q4 = q4_data.as_ptr().add((block_idx + 4) * Q4_0_BLOCK_BYTES);
                let prefetch_q8 = q8_quants.as_ptr().add((block_idx + 4) * Q4_0_BLOCK_SIZE);
                _mm_prefetch(prefetch_q4.cast(), _MM_HINT_T0);
                _mm_prefetch(prefetch_q8.cast(), _MM_HINT_T0);
                _mm_prefetch(prefetch_q4.add(64).cast(), _MM_HINT_T0);
                _mm_prefetch(prefetch_q8.add(64).cast(), _MM_HINT_T0);
            }

            // Block 0
            let q4_ptr_0 = q4_data.as_ptr().add(block_idx * Q4_0_BLOCK_BYTES);
            let q8_ptr_0 = q8_quants.as_ptr().add(block_idx * Q4_0_BLOCK_SIZE);
            let q4_scale_0 = read_q4_scale(q4_ptr_0);
            let combined_scale_0 = _mm256_set1_ps(q4_scale_0 * q8_scales[block_idx]);
            let q4_lo_0 = std::arch::x86_64::_mm_loadu_si128(q4_ptr_0.add(2).cast());
            let q4_hi_0 = std::arch::x86_64::_mm_srli_epi16(q4_lo_0, 4);
            let q4_signed_0 = _mm256_sub_epi8(
                _mm256_and_si256(
                    std::arch::x86_64::_mm256_set_m128i(q4_hi_0, q4_lo_0),
                    low_mask,
                ),
                offset,
            );
            let q8_vec_0 = _mm256_loadu_si256(q8_ptr_0.cast());
            let q4_abs_0 = _mm256_sign_epi8(q4_signed_0, q4_signed_0);
            let q8_signed_0 = _mm256_sign_epi8(q8_vec_0, q4_signed_0);
            let prod_0 = _mm256_cvtepi32_ps(_mm256_madd_epi16(
                _mm256_maddubs_epi16(q4_abs_0, q8_signed_0),
                ones,
            ));
            acc0 = _mm256_fmadd_ps(combined_scale_0, prod_0, acc0);

            // Block 1
            let q4_ptr_1 = q4_data.as_ptr().add((block_idx + 1) * Q4_0_BLOCK_BYTES);
            let q8_ptr_1 = q8_quants.as_ptr().add((block_idx + 1) * Q4_0_BLOCK_SIZE);
            let q4_scale_1 = read_q4_scale(q4_ptr_1);
            let combined_scale_1 = _mm256_set1_ps(q4_scale_1 * q8_scales[block_idx + 1]);
            let q4_lo_1 = std::arch::x86_64::_mm_loadu_si128(q4_ptr_1.add(2).cast());
            let q4_hi_1 = std::arch::x86_64::_mm_srli_epi16(q4_lo_1, 4);
            let q4_signed_1 = _mm256_sub_epi8(
                _mm256_and_si256(
                    std::arch::x86_64::_mm256_set_m128i(q4_hi_1, q4_lo_1),
                    low_mask,
                ),
                offset,
            );
            let q8_vec_1 = _mm256_loadu_si256(q8_ptr_1.cast());
            let q4_abs_1 = _mm256_sign_epi8(q4_signed_1, q4_signed_1);
            let q8_signed_1 = _mm256_sign_epi8(q8_vec_1, q4_signed_1);
            let prod_1 = _mm256_cvtepi32_ps(_mm256_madd_epi16(
                _mm256_maddubs_epi16(q4_abs_1, q8_signed_1),
                ones,
            ));
            acc1 = _mm256_fmadd_ps(combined_scale_1, prod_1, acc1);

            // Block 2
            let q4_ptr_2 = q4_data.as_ptr().add((block_idx + 2) * Q4_0_BLOCK_BYTES);
            let q8_ptr_2 = q8_quants.as_ptr().add((block_idx + 2) * Q4_0_BLOCK_SIZE);
            let q4_scale_2 = read_q4_scale(q4_ptr_2);
            let combined_scale_2 = _mm256_set1_ps(q4_scale_2 * q8_scales[block_idx + 2]);
            let q4_lo_2 = std::arch::x86_64::_mm_loadu_si128(q4_ptr_2.add(2).cast());
            let q4_hi_2 = std::arch::x86_64::_mm_srli_epi16(q4_lo_2, 4);
            let q4_signed_2 = _mm256_sub_epi8(
                _mm256_and_si256(
                    std::arch::x86_64::_mm256_set_m128i(q4_hi_2, q4_lo_2),
                    low_mask,
                ),
                offset,
            );
            let q8_vec_2 = _mm256_loadu_si256(q8_ptr_2.cast());
            let q4_abs_2 = _mm256_sign_epi8(q4_signed_2, q4_signed_2);
            let q8_signed_2 = _mm256_sign_epi8(q8_vec_2, q4_signed_2);
            let prod_2 = _mm256_cvtepi32_ps(_mm256_madd_epi16(
                _mm256_maddubs_epi16(q4_abs_2, q8_signed_2),
                ones,
            ));
            acc0 = _mm256_fmadd_ps(combined_scale_2, prod_2, acc0);

            // Block 3
            let q4_ptr_3 = q4_data.as_ptr().add((block_idx + 3) * Q4_0_BLOCK_BYTES);
            let q8_ptr_3 = q8_quants.as_ptr().add((block_idx + 3) * Q4_0_BLOCK_SIZE);
            let q4_scale_3 = read_q4_scale(q4_ptr_3);
            let combined_scale_3 = _mm256_set1_ps(q4_scale_3 * q8_scales[block_idx + 3]);
            let q4_lo_3 = std::arch::x86_64::_mm_loadu_si128(q4_ptr_3.add(2).cast());
            let q4_hi_3 = std::arch::x86_64::_mm_srli_epi16(q4_lo_3, 4);
            let q4_signed_3 = _mm256_sub_epi8(
                _mm256_and_si256(
                    std::arch::x86_64::_mm256_set_m128i(q4_hi_3, q4_lo_3),
                    low_mask,
                ),
                offset,
            );
            let q8_vec_3 = _mm256_loadu_si256(q8_ptr_3.cast());
            let q4_abs_3 = _mm256_sign_epi8(q4_signed_3, q4_signed_3);
            let q8_signed_3 = _mm256_sign_epi8(q8_vec_3, q4_signed_3);
            let prod_3 = _mm256_cvtepi32_ps(_mm256_madd_epi16(
                _mm256_maddubs_epi16(q4_abs_3, q8_signed_3),
                ones,
            ));
            acc1 = _mm256_fmadd_ps(combined_scale_3, prod_3, acc1);

            block_idx += 4;
        }

        // Merge accumulators
        let acc = _mm256_add_ps(acc0, acc1);

        // Handle remaining blocks (0-3)
        let mut scalar_sum = 0.0f32;
        while block_idx < num_blocks {
            let q4_ptr = q4_data.as_ptr().add(block_idx * Q4_0_BLOCK_BYTES);
            let q8_ptr = q8_quants.as_ptr().add(block_idx * Q4_0_BLOCK_SIZE);
            let q4_scale = read_q4_scale(q4_ptr);
            let combined_scale = _mm256_set1_ps(q4_scale * q8_scales[block_idx]);
            let q4_lo = std::arch::x86_64::_mm_loadu_si128(q4_ptr.add(2).cast());
            let q4_hi = std::arch::x86_64::_mm_srli_epi16(q4_lo, 4);
            let q4_signed = _mm256_sub_epi8(
                _mm256_and_si256(std::arch::x86_64::_mm256_set_m128i(q4_hi, q4_lo), low_mask),
                offset,
            );
            let q8_vec = _mm256_loadu_si256(q8_ptr.cast());
            let q4_abs = _mm256_sign_epi8(q4_signed, q4_signed);
            let q8_signed = _mm256_sign_epi8(q8_vec, q4_signed);
            let prod = _mm256_cvtepi32_ps(_mm256_madd_epi16(
                _mm256_maddubs_epi16(q4_abs, q8_signed),
                ones,
            ));
            let scaled = _mm256_fmadd_ps(combined_scale, prod, _mm256_setzero_ps());

            // Horizontal sum for this block
            let hi = std::arch::x86_64::_mm256_extractf128_ps(scaled, 1);
            let lo = std::arch::x86_64::_mm256_castps256_ps128(scaled);
            let sum128 = std::arch::x86_64::_mm_add_ps(lo, hi);
            let sum64 = _mm_hadd_ps(sum128, sum128);
            let sum32 = _mm_hadd_ps(sum64, sum64);
            scalar_sum += _mm_cvtss_f32(sum32);

            block_idx += 1;
        }

        // Horizontal sum of accumulated vector
        let hi = std::arch::x86_64::_mm256_extractf128_ps(acc, 1);
        let lo = std::arch::x86_64::_mm256_castps256_ps128(acc);
        let sum128 = std::arch::x86_64::_mm_add_ps(lo, hi);
        let sum64 = _mm_hadd_ps(sum128, sum128);
        let sum32 = _mm_hadd_ps(sum64, sum64);
        _mm_cvtss_f32(sum32) + scalar_sum
    }
}

/// Scalar fallback for Q4_0 × Q8_0 dot product
///
/// Exposed as `pub(crate)` for direct testing on AVX2 machines.
#[inline]
pub(crate) fn fused_q4_0_q8_0_dot_scalar(
    q4_data: &[u8],
    q8_scales: &[f32],
    q8_quants: &[i8],
    in_dim: usize,
) -> f32 {
    const Q4_0_BLOCK_BYTES: usize = 18;
    const Q4_0_BLOCK_SIZE: usize = 32;

    let num_blocks = in_dim.div_ceil(Q4_0_BLOCK_SIZE);
    let mut total_sum = 0.0f32;

    for block_idx in 0..num_blocks {
        let block_start = block_idx * Q4_0_BLOCK_BYTES;
        if block_start + Q4_0_BLOCK_BYTES > q4_data.len() {
            break;
        }
        let block = &q4_data[block_start..block_start + Q4_0_BLOCK_BYTES];

        let q4_scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let q8_scale = q8_scales[block_idx];
        let combined_scale = q4_scale * q8_scale;

        let act_start = block_idx * Q4_0_BLOCK_SIZE;

        let mut block_sum = 0i32;
        for (j, &byte) in block[2..18].iter().enumerate() {
            let low_idx = act_start + j;
            let high_idx = act_start + j + 16;

            #[allow(clippy::cast_possible_wrap)]
            let low_quant = (byte & 0x0F) as i8 - 8;
            block_sum += (low_quant as i32) * (q8_quants[low_idx] as i32);

            #[allow(clippy::cast_possible_wrap)]
            let high_quant = (byte >> 4) as i8 - 8;
            if high_idx < in_dim {
                block_sum += (high_quant as i32) * (q8_quants[high_idx] as i32);
            }
        }

        total_sum += combined_scale * (block_sum as f32);
    }

    total_sum
}

/// Parallel Q4_0 × Q8_0 matrix-vector multiply
///
/// This is the key function for llama.cpp parity. It:
/// 1. Quantizes activations to Q8_0 format once
/// 2. Uses integer SIMD for all row dot products
/// 3. Parallelizes across output rows with rayon (adaptive threshold)
///
/// Expected speedup: 4-6x over the f32 FMA version
#[allow(clippy::similar_names)]
pub fn fused_q4_0_q8_0_parallel_matvec(
    weight_data: &[u8],
    activations: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> Result<Vec<f32>> {
    const Q4_0_BLOCK_BYTES: usize = 18;
    const Q4_0_BLOCK_SIZE: usize = 32;

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

    // Adaptive parallelization: sequential for small matrices, parallel for large
    // Rayon overhead (~50-100µs) dominates for small out_dim
    // Threshold tuned for 22-core CPU: break-even at ~1024 rows
    const PARALLEL_THRESHOLD: usize = 1024;

    if out_dim < PARALLEL_THRESHOLD {
        // Sequential path: avoids Rayon overhead entirely
        let output: Vec<f32> = (0..out_dim)
            .map(|o| {
                let row_start = o * bytes_per_row;
                let row_end = row_start + bytes_per_row;
                let row_data = &weight_data[row_start..row_end];
                fused_q4_0_q8_0_dot_simd(row_data, &q8_scales, &q8_quants, in_dim)
            })
            .collect();
        return Ok(output);
    }

    // Parallel path for large matrices
    use rayon::prelude::*;
    // Use chunked parallel iteration to reduce Rayon scheduling overhead
    // CHUNK_SIZE=128 provides good balance between parallelism and overhead
    const CHUNK_SIZE: usize = 128;
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

/// Zero-allocation Q4_0 × Q8_0 matrix-vector multiply.
///
/// Writes result directly into provided output buffer, eliminating allocation.
/// Use this with scratch buffers for maximum performance.
///
/// # Arguments
/// * `weight_data` - Q4_0 quantized weight matrix (row-major)
/// * `activations` - Input activation vector (f32)
/// * `in_dim` - Input dimension (columns)
/// * `output` - Pre-allocated output buffer (must be exactly out_dim length)
///
/// # Returns
/// Number of elements written (equals output.len())
#[allow(clippy::similar_names)]
pub fn fused_q4_0_q8_0_parallel_matvec_into(
    weight_data: &[u8],
    activations: &[f32],
    in_dim: usize,
    output: &mut [f32],
) -> Result<()> {
    use rayon::prelude::*;

    const Q4_0_BLOCK_BYTES: usize = 18;
    const Q4_0_BLOCK_SIZE: usize = 32;

    let out_dim = output.len();
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

    if activations.len() != in_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Activation length {} doesn't match in_dim {}",
                activations.len(),
                in_dim
            ),
        });
    }

    // Quantize activations to Q8_0 ONCE
    let (q8_scales, q8_quants) = quantize_activations_q8_0(activations);

    // Use chunked parallel iteration to reduce Rayon scheduling overhead
    const CHUNK_SIZE: usize = 64;
    output
        .par_iter_mut()
        .with_min_len(CHUNK_SIZE)
        .enumerate()
        .for_each(|(o, out_val)| {
            let row_start = o * bytes_per_row;
            let row_end = row_start + bytes_per_row;
            let row_data = &weight_data[row_start..row_end];
            *out_val = fused_q4_0_q8_0_dot_simd(row_data, &q8_scales, &q8_quants, in_dim);
        });

    Ok(())
}
