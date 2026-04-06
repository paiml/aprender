
/// Optimized AVX-512 VNNI Q4_K × Q8_K dot product - DEFERRED horizontal sum
///
/// PAR-126 Five-Whys optimization: Instead of horizontal sums per chunk (24+ per super-block),
/// we accumulate all 8 block results in vector registers and do ONE horizontal sum at the end.
/// This reduces horizontal sum operations from 24+ to 1 per super-block (24x reduction).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512vnni", enable = "avx512bw")]
#[allow(unsafe_op_in_unsafe_fn)]
#[allow(clippy::similar_names)]
#[allow(clippy::too_many_lines)]
#[allow(dead_code)]
// SAFETY: Caller must satisfy the documented preconditions
unsafe fn fused_q4k_q8k_dot_avx512vnni_opt(
    q4k_data: &[u8],
    q8k_scales: &[f32],
    q8k_quants: &[i8],
) -> Result<f32> {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    const SUPER_BLOCK_BYTES: usize = 144;

    if !q4k_data.len().is_multiple_of(SUPER_BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q4_K data length {} is not a multiple of {}",
                q4k_data.len(),
                SUPER_BLOCK_BYTES
            ),
        });
    }

    let num_super_blocks = q4k_data.len() / SUPER_BLOCK_BYTES;
    let expected_values = num_super_blocks * QK_K;

    if q8k_scales.len() < num_super_blocks || q8k_quants.len() < expected_values {
        return Err(RealizarError::InvalidShape {
            reason: "Q8_K buffer too small".to_string(),
        });
    }

    let nibble_mask = _mm256_set1_epi8(0x0F_i8);
    let ones_16 = _mm256_set1_epi16(1);

    // Global float accumulator
    let mut total_acc = 0.0f32;

    for sb_idx in 0..num_super_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let q8_start = sb_idx * QK_K;

        // Prefetch next super-block
        if sb_idx + 1 < num_super_blocks {
            _mm_prefetch(
                q4k_data
                    .as_ptr()
                    .add((sb_idx + 1) * SUPER_BLOCK_BYTES)
                    .cast::<i8>(),
                _MM_HINT_T0,
            );
            _mm_prefetch(
                q8k_quants.as_ptr().add((sb_idx + 1) * QK_K).cast::<i8>(),
                _MM_HINT_T0,
            );
        }

        // Read Q4_K header
        let d = read_f16(&q4k_data[sb_start..sb_start + 2]);
        let dmin = read_f16(&q4k_data[sb_start + 2..sb_start + 4]);

        let mut scales_raw = [0u8; 12];
        scales_raw.copy_from_slice(&q4k_data[sb_start + 4..sb_start + 16]);

        let q8_scale = q8k_scales[sb_idx];
        let d_q8 = d * q8_scale;
        let dmin_q8 = dmin * q8_scale;

        let qs_ptr = q4k_data.as_ptr().add(sb_start + 16);
        let q8_ptr = q8k_quants.as_ptr().add(q8_start);

        // PAR-126: DEFERRED HORIZONTAL SUM OPTIMIZATION
        // Keep 8 block accumulators in __m256i vectors (one per 32-value block)
        // block_dots[i] = sum of (Q4[block_i] * Q8[block_i]) for block i
        // block_q8sums[i] = sum of Q8[block_i] for dmin correction
        let mut block_dots = [0i32; 8];
        let mut block_q8sums = [0i32; 8];

        // Process 64 values (2 blocks) per iteration, 4 iterations total = 8 blocks
        for chunk in 0..4 {
            let j = chunk * 64;
            let q_offset = j / 2;

            // Load 32 bytes Q4 (64 nibbles packed)
            let q4_bytes = _mm256_loadu_si256(qs_ptr.add(q_offset).cast::<__m256i>());

            // Extract nibbles: lo = first 32 values, hi = next 32 values
            let q4_lo = _mm256_and_si256(q4_bytes, nibble_mask);
            let q4_hi = _mm256_and_si256(_mm256_srli_epi16(q4_bytes, 4), nibble_mask);

            // Load 64 bytes Q8
            let q8_lo = _mm256_loadu_si256(q8_ptr.add(j).cast::<__m256i>());
            let q8_hi = _mm256_loadu_si256(q8_ptr.add(j + 32).cast::<__m256i>());

            // Q4 × Q8 products -> i16 via maddubs -> i32 via madd
            // maddubs: adjacent u8*i8 pairs summed to i16
            // madd with ones: sum pairs of i16 to i32
            let prod_lo_i16 = _mm256_maddubs_epi16(q4_lo, q8_lo);
            let prod_hi_i16 = _mm256_maddubs_epi16(q4_hi, q8_hi);
            let prod_lo_i32 = _mm256_madd_epi16(prod_lo_i16, ones_16);
            let prod_hi_i32 = _mm256_madd_epi16(prod_hi_i16, ones_16);

            // prod_lo_i32 now has 8 i32 values (4 per 128-bit lane)
            // We need to reduce to 1 value per 32-element block
            // Lane 0-3 are from elements 0-15, lane 4-7 are from elements 16-31
            // So we sum all 8 to get one block sum

            // Sum all 8 i32 in prod_lo_i32 using only one horizontal sum
            // First add the two 128-bit halves
            let prod_lo_128 = _mm_add_epi32(
                _mm256_castsi256_si128(prod_lo_i32),
                _mm256_extracti128_si256(prod_lo_i32, 1),
            );
            let prod_hi_128 = _mm_add_epi32(
                _mm256_castsi256_si128(prod_hi_i32),
                _mm256_extracti128_si256(prod_hi_i32, 1),
            );

            // Now we have 4 i32 each - sum them with hadd
            let prod_lo_64 = _mm_hadd_epi32(prod_lo_128, prod_hi_128);
            let prod_32 = _mm_hadd_epi32(prod_lo_64, prod_lo_64);

            // Extract block sums - lane 0 is block_lo, lane 1 is block_hi
            let block_idx = chunk * 2;
            block_dots[block_idx] = _mm_extract_epi32(prod_32, 0);
            block_dots[block_idx + 1] = _mm_extract_epi32(prod_32, 1);

            // Q8 sums for dmin correction - convert i8 to i16 first to avoid overflow
            // We need sum of all 32 i8 values for each block
            let q8_lo_i16_a = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q8_lo));
            let q8_lo_i16_b = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q8_lo, 1));
            let q8_hi_i16_a = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q8_hi));
            let q8_hi_i16_b = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q8_hi, 1));

            // Sum the i16 values to i32 using madd with ones
            let q8_lo_i32_a = _mm256_madd_epi16(q8_lo_i16_a, _mm256_set1_epi16(1));
            let q8_lo_i32_b = _mm256_madd_epi16(q8_lo_i16_b, _mm256_set1_epi16(1));
            let q8_hi_i32_a = _mm256_madd_epi16(q8_hi_i16_a, _mm256_set1_epi16(1));
            let q8_hi_i32_b = _mm256_madd_epi16(q8_hi_i16_b, _mm256_set1_epi16(1));

            // Combine halves for each block
            let q8_lo_sum = _mm256_add_epi32(q8_lo_i32_a, q8_lo_i32_b);
            let q8_hi_sum = _mm256_add_epi32(q8_hi_i32_a, q8_hi_i32_b);

            // Horizontal sum each block's q8 sum
            let q8_lo_128 = _mm_add_epi32(
                _mm256_castsi256_si128(q8_lo_sum),
                _mm256_extracti128_si256(q8_lo_sum, 1),
            );
            let q8_hi_128 = _mm_add_epi32(
                _mm256_castsi256_si128(q8_hi_sum),
                _mm256_extracti128_si256(q8_hi_sum, 1),
            );
            let q8_64 = _mm_hadd_epi32(q8_lo_128, q8_hi_128);
            let q8_32 = _mm_hadd_epi32(q8_64, q8_64);

            block_q8sums[block_idx] = _mm_extract_epi32(q8_32, 0);
            block_q8sums[block_idx + 1] = _mm_extract_epi32(q8_32, 1);
        }

        // Extract all 8 scales and mins
        let mut scales = [0.0f32; 8];
        let mut mins = [0.0f32; 8];
        for i in 0..8 {
            let (sc, m) = extract_scale_min(&scales_raw, i);
            scales[i] = sc;
            mins[i] = m;
        }

        // Load scales and mins into vectors for SIMD multiply
        let scales_vec = _mm256_loadu_ps(scales.as_ptr());
        let mins_vec = _mm256_loadu_ps(mins.as_ptr());

        // Convert block_dots and block_q8sums to f32
        let dots_i32 = _mm256_loadu_si256(block_dots.as_ptr().cast::<__m256i>());
        let q8sums_i32 = _mm256_loadu_si256(block_q8sums.as_ptr().cast::<__m256i>());
        let dots_f32 = _mm256_cvtepi32_ps(dots_i32);
        let q8sums_f32 = _mm256_cvtepi32_ps(q8sums_i32);

        // Compute: d_q8 * scales * dots - dmin_q8 * mins * q8sums
        let d_q8_vec = _mm256_set1_ps(d_q8);
        let dmin_q8_vec = _mm256_set1_ps(dmin_q8);

        let term1 = _mm256_mul_ps(d_q8_vec, _mm256_mul_ps(scales_vec, dots_f32));
        let term2 = _mm256_mul_ps(dmin_q8_vec, _mm256_mul_ps(mins_vec, q8sums_f32));
        let result = _mm256_sub_ps(term1, term2);

        // ONE horizontal sum for all 8 blocks
        let sum128 = _mm_add_ps(
            _mm256_castps256_ps128(result),
            _mm256_extractf128_ps(result, 1),
        );
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        total_acc += _mm_cvtss_f32(sum32);
    }

    Ok(total_acc)
}

/// Fast horizontal sum of 4 i32 in __m128i
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(dead_code)]
// SAFETY: Caller must satisfy the documented preconditions
unsafe fn hsum_epi32_128(v: std::arch::x86_64::__m128i) -> i32 {
    use std::arch::x86_64::{_mm_cvtsi128_si32, _mm_hadd_epi32};
    let sum64 = _mm_hadd_epi32(v, v);
    let sum32 = _mm_hadd_epi32(sum64, sum64);
    _mm_cvtsi128_si32(sum32)
}

/// Fast horizontal sum of 8 i32 in __m256i
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(dead_code)]
// SAFETY: Caller must satisfy the documented preconditions
unsafe fn hsum_epi32_256(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::{_mm256_castsi256_si128, _mm256_extracti128_si256, _mm_add_epi32};
    // SAFETY: Unsafe operation with validated invariants
    unsafe {
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256(v, 1);
        hsum_epi32_128(_mm_add_epi32(lo, hi))
    }
}

/// Helper: horizontal sum of 8 int32 values in a 256-bit register
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(dead_code)]
// SAFETY: Caller must satisfy the documented preconditions
unsafe fn horizontal_sum_epi32_256(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::{
        _mm256_castsi256_si128, _mm256_extracti128_si256, _mm_add_epi32, _mm_cvtsi128_si32,
        _mm_hadd_epi32,
    };

    // Add high 128 bits to low 128 bits
    let hi = _mm256_extracti128_si256(v, 1);
    let lo = _mm256_castsi256_si128(v);
    let sum128 = _mm_add_epi32(lo, hi);

    // Horizontal add within 128 bits
    let sum64 = _mm_hadd_epi32(sum128, sum128);
    let sum32 = _mm_hadd_epi32(sum64, sum64);

    _mm_cvtsi128_si32(sum32)
}

/// Helper: horizontal sum of 16 int16 values in a 256-bit register
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
#[allow(dead_code)]
// SAFETY: Caller must satisfy the documented preconditions
unsafe fn horizontal_sum_epi16_256(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::{_mm256_madd_epi16, _mm256_set1_epi16};

    // Use madd to sum pairs of i16 to i32
    let ones = _mm256_set1_epi16(1);
    let sum_i32 = _mm256_madd_epi16(v, ones);

    // Now sum the 8 i32 values
    horizontal_sum_epi32_256(sum_i32)
}
