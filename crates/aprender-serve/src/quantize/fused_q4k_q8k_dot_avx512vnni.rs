
/// PAR-126 V2: AVX-512 VNNI kernel with deferred horizontal sums
///
/// Key optimization: Instead of extracting block sums to scalars in the inner loop,
/// keeps all 8 block accumulators in __m256i vectors and only reduces at the end.
/// This eliminates 24 horizontal sums per row (from 8 to 1).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512vnni", enable = "avx512bw")]
#[allow(unsafe_op_in_unsafe_fn)]
#[allow(clippy::similar_names)]
#[allow(clippy::too_many_lines)]
#[allow(dead_code)]
// SAFETY: Caller must satisfy the documented preconditions
unsafe fn fused_q4k_q8k_dot_avx512vnni_v2(
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
    let mut total_acc = _mm256_setzero_ps();

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

        // PAR-126 V2: Keep all 8 block results in vectors
        // block_dots_vec[i] = sum of (Q4[block_i] * Q8[block_i])
        // block_q8sums_vec[i] = sum of Q8[block_i]
        let mut block_dots_vec = _mm256_setzero_si256();
        let mut block_q8sums_vec = _mm256_setzero_si256();

        // Process 64 values (2 blocks) per iteration, 4 iterations = 8 blocks
        // Each iteration produces 2 block sums that go into specific lanes
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
            let prod_lo_i16 = _mm256_maddubs_epi16(q4_lo, q8_lo);
            let prod_hi_i16 = _mm256_maddubs_epi16(q4_hi, q8_hi);
            let prod_lo_i32 = _mm256_madd_epi16(prod_lo_i16, ones_16);
            let prod_hi_i32 = _mm256_madd_epi16(prod_hi_i16, ones_16);

            // Reduce each 256-bit register to ONE sum using hadd (within-lane reduction)
            // prod_lo_i32 has 8 i32 values, we need their sum for block chunk*2
            // prod_hi_i32 has 8 i32 values, we need their sum for block chunk*2+1

            // Use hadd twice to get 8 -> 4 -> 2 values per register
            let prod_lo_h1 = _mm256_hadd_epi32(prod_lo_i32, prod_hi_i32); // interleaves lo and hi
            let prod_h2 = _mm256_hadd_epi32(prod_lo_h1, prod_lo_h1); // further reduce

            // Now prod_h2 has: [sum_lo_lane0, sum_hi_lane0, sum_lo_lane0, sum_hi_lane0,
            //                   sum_lo_lane1, sum_hi_lane1, sum_lo_lane1, sum_hi_lane1]
            // We need to extract the two unique sums and place them in block_dots_vec

            // Extract two sums: add lane0 and lane1 of each block
            let lane0 = _mm256_castsi256_si128(prod_h2);
            let lane1 = _mm256_extracti128_si256(prod_h2, 1);
            let sums_128 = _mm_add_epi32(lane0, lane1); // [dot_lo, dot_hi, dot_lo, dot_hi]

            // Insert into correct position based on chunk
            // block_dots_vec lanes: [0,1,2,3,4,5,6,7] for blocks [0,1,2,3,4,5,6,7]
            match chunk {
                0 => {
                    // Put in lanes 0,1
                    block_dots_vec =
                        _mm256_insert_epi32(block_dots_vec, _mm_extract_epi32(sums_128, 0), 0);
                    block_dots_vec =
                        _mm256_insert_epi32(block_dots_vec, _mm_extract_epi32(sums_128, 1), 1);
                },
                1 => {
                    block_dots_vec =
                        _mm256_insert_epi32(block_dots_vec, _mm_extract_epi32(sums_128, 0), 2);
                    block_dots_vec =
                        _mm256_insert_epi32(block_dots_vec, _mm_extract_epi32(sums_128, 1), 3);
                },
                2 => {
                    block_dots_vec =
                        _mm256_insert_epi32(block_dots_vec, _mm_extract_epi32(sums_128, 0), 4);
                    block_dots_vec =
                        _mm256_insert_epi32(block_dots_vec, _mm_extract_epi32(sums_128, 1), 5);
                },
                3 => {
                    block_dots_vec =
                        _mm256_insert_epi32(block_dots_vec, _mm_extract_epi32(sums_128, 0), 6);
                    block_dots_vec =
                        _mm256_insert_epi32(block_dots_vec, _mm_extract_epi32(sums_128, 1), 7);
                },
                _ => unreachable!(),
            }

            // Q8 sums for dmin correction using sad_epu8 (sum of absolute differences)
            // Convert signed i8 to unsigned by adding 128, compute SAD, then adjust
            let bias = _mm256_set1_epi8(-128_i8);
            let _q8_lo_u = _mm256_sub_epi8(q8_lo, bias); // signed [-128,127] -> unsigned [0,255]
            let _q8_hi_u = _mm256_sub_epi8(q8_hi, bias);

            // Use mpsadbw or manual approach for sum
            // Simpler: sign-extend and sum using madd
            let q8_lo_i16_a = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q8_lo));
            let q8_lo_i16_b = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q8_lo, 1));
            let q8_hi_i16_a = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q8_hi));
            let q8_hi_i16_b = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q8_hi, 1));

            let q8_lo_i32_a = _mm256_madd_epi16(q8_lo_i16_a, ones_16);
            let q8_lo_i32_b = _mm256_madd_epi16(q8_lo_i16_b, ones_16);
            let q8_hi_i32_a = _mm256_madd_epi16(q8_hi_i16_a, ones_16);
            let q8_hi_i32_b = _mm256_madd_epi16(q8_hi_i16_b, ones_16);

            let q8_lo_sum = _mm256_add_epi32(q8_lo_i32_a, q8_lo_i32_b);
            let q8_hi_sum = _mm256_add_epi32(q8_hi_i32_a, q8_hi_i32_b);

            // Reduce to two sums using hadd
            let q8_h1 = _mm256_hadd_epi32(q8_lo_sum, q8_hi_sum);
            let q8_h2 = _mm256_hadd_epi32(q8_h1, q8_h1);

            let q8_lane0 = _mm256_castsi256_si128(q8_h2);
            let q8_lane1 = _mm256_extracti128_si256(q8_h2, 1);
            let q8_sums_128 = _mm_add_epi32(q8_lane0, q8_lane1);

            match chunk {
                0 => {
                    block_q8sums_vec =
                        _mm256_insert_epi32(block_q8sums_vec, _mm_extract_epi32(q8_sums_128, 0), 0);
                    block_q8sums_vec =
                        _mm256_insert_epi32(block_q8sums_vec, _mm_extract_epi32(q8_sums_128, 1), 1);
                },
                1 => {
                    block_q8sums_vec =
                        _mm256_insert_epi32(block_q8sums_vec, _mm_extract_epi32(q8_sums_128, 0), 2);
                    block_q8sums_vec =
                        _mm256_insert_epi32(block_q8sums_vec, _mm_extract_epi32(q8_sums_128, 1), 3);
                },
                2 => {
                    block_q8sums_vec =
                        _mm256_insert_epi32(block_q8sums_vec, _mm_extract_epi32(q8_sums_128, 0), 4);
                    block_q8sums_vec =
                        _mm256_insert_epi32(block_q8sums_vec, _mm_extract_epi32(q8_sums_128, 1), 5);
                },
                3 => {
                    block_q8sums_vec =
                        _mm256_insert_epi32(block_q8sums_vec, _mm_extract_epi32(q8_sums_128, 0), 6);
                    block_q8sums_vec =
                        _mm256_insert_epi32(block_q8sums_vec, _mm_extract_epi32(q8_sums_128, 1), 7);
                },
                _ => unreachable!(),
            }
        }

        // Extract all 8 scales and mins into vectors
        let mut scales = [0.0f32; 8];
        let mut mins = [0.0f32; 8];
        for i in 0..8 {
            let (sc, m) = extract_scale_min(&scales_raw, i);
            scales[i] = sc;
            mins[i] = m;
        }

        let scales_vec = _mm256_loadu_ps(scales.as_ptr());
        let mins_vec = _mm256_loadu_ps(mins.as_ptr());

        // Convert block results to f32
        let dots_f32 = _mm256_cvtepi32_ps(block_dots_vec);
        let q8sums_f32 = _mm256_cvtepi32_ps(block_q8sums_vec);

        // Compute: d_q8 * scales * dots - dmin_q8 * mins * q8sums
        let d_q8_vec = _mm256_set1_ps(d_q8);
        let dmin_q8_vec = _mm256_set1_ps(dmin_q8);

        let term1 = _mm256_mul_ps(d_q8_vec, _mm256_mul_ps(scales_vec, dots_f32));
        let term2 = _mm256_mul_ps(dmin_q8_vec, _mm256_mul_ps(mins_vec, q8sums_f32));
        let result = _mm256_sub_ps(term1, term2);

        // Accumulate into total (defer final horizontal sum)
        total_acc = _mm256_add_ps(total_acc, result);
    }

    // ONE horizontal sum at the very end
    let sum128 = _mm_add_ps(
        _mm256_castps256_ps128(total_acc),
        _mm256_extractf128_ps(total_acc, 1),
    );
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));

    Ok(_mm_cvtss_f32(sum32))
}

/// AVX-512 VNNI optimized Q4_K × Q8_K dot product (llama.cpp style)
///
/// Uses `vpdpbusd` instruction for 16-way uint8×int8→int32 multiply-accumulate.
/// This is the fastest CPU path for quantized matmul.
///
/// # Safety
/// Requires AVX-512F and AVX-512 VNNI CPU features.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512vnni", enable = "avx512bw")]
#[allow(unsafe_op_in_unsafe_fn)]
#[allow(dead_code)]
// SAFETY: Caller must satisfy the documented preconditions
unsafe fn fused_q4k_q8k_dot_avx512vnni(
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

    let _nibble_mask = _mm512_set1_epi8(0x0F_i8); // Reserved for future optimization
    let mut total_acc = 0.0f32;

    for sb_idx in 0..num_super_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let q8_start = sb_idx * QK_K;

        // Prefetch next super-block (dual: Q4_K + Q8_K — match V2 pattern)
        if sb_idx + 1 < num_super_blocks {
            _mm_prefetch(
                q4k_data
                    .as_ptr()
                    .add((sb_idx + 1) * SUPER_BLOCK_BYTES)
                    .cast::<i8>(),
                _MM_HINT_T0,
            );
            _mm_prefetch(
                q8k_quants
                    .as_ptr()
                    .add((sb_idx + 1) * QK_K)
                    .cast::<i8>(),
                _MM_HINT_T0,
            );
        }

        // Read Q4_K header
        let d = read_f16(&q4k_data[sb_start..sb_start + 2]);
        let dmin = read_f16(&q4k_data[sb_start + 2..sb_start + 4]);

        let mut scales = [0u8; 12];
        scales.copy_from_slice(&q4k_data[sb_start + 4..sb_start + 16]);

        let q8_scale = q8k_scales[sb_idx];
        let d_q8 = d * q8_scale;
        let dmin_q8 = dmin * q8_scale;

        let qs_ptr = q4k_data.as_ptr().add(sb_start + 16);
        let q8_ptr = q8k_quants.as_ptr().add(q8_start);

        // Process 4 chunks of 64 values (matching dequantize_q4_k layout)
        for j in (0..QK_K).step_by(64) {
            let q_offset = j / 2; // 32 bytes per 64-value chunk

            let is = j / 32;
            let (sc1, m1) = extract_scale_min(&scales, is);
            let (sc2, m2) = extract_scale_min(&scales, is + 1);

            // Load 32 bytes of Q4_K (as lower half of 512-bit register)
            // We need to process 64 Q4 values (32 bytes) with 64 Q8 values (64 bytes)
            let q4_256 = _mm256_loadu_si256(qs_ptr.add(q_offset).cast::<__m256i>());

            // Zero-extend to 512-bit (q4 data in lower 256 bits) - reserved for full-width optimization
            let _q4_512 = _mm512_castsi256_si512(q4_256);

            // Extract low nibbles (first 32 values) and high nibbles (second 32 values)
            let q4_lo_256 = _mm256_and_si256(q4_256, _mm256_set1_epi8(0x0F_i8));
            let q4_hi_256 =
                _mm256_and_si256(_mm256_srli_epi16(q4_256, 4), _mm256_set1_epi8(0x0F_i8));

            // Load Q8 values: first 32 for low nibbles, next 32 for high nibbles
            // (matching dequantize_q4_k output order: 32 low, then 32 high)
            let q8_lo_256 = _mm256_loadu_si256(q8_ptr.add(j).cast::<__m256i>());
            let q8_hi_256 = _mm256_loadu_si256(q8_ptr.add(j + 32).cast::<__m256i>());

            // For VNNI vpdpbusd: accumulator += (unsigned a) × (signed b)
            // Q4 values are 0-15 (unsigned), Q8 values are -128..127 (signed)
            // We need to cast Q4 to unsigned interpretation

            // Convert to 512-bit for VNNI
            let q4_lo_512 = _mm512_castsi256_si512(q4_lo_256);
            let q4_hi_512 = _mm512_castsi256_si512(q4_hi_256);
            let q8_lo_512 = _mm512_castsi256_si512(q8_lo_256);
            let q8_hi_512 = _mm512_castsi256_si512(q8_hi_256);

            // VNNI multiply-accumulate: result[i] += q4[4i..4i+4] · q8[4i..4i+4]
            // This computes 4 int8×int8 products and adds them to int32 accumulator
            let acc_lo = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q4_lo_512, q8_lo_512);
            let acc_hi = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q4_hi_512, q8_hi_512);

            // Horizontal sum the accumulators (each has 16 int32 values)
            // Extract lower 256 bits which contain our results
            let acc_lo_256 = _mm512_castsi512_si256(acc_lo);
            let acc_hi_256 = _mm512_castsi512_si256(acc_hi);

            // Sum all 8 int32 values in each 256-bit register
            let sum_lo = horizontal_sum_epi32_256(acc_lo_256);
            let sum_hi = horizontal_sum_epi32_256(acc_hi_256);

            // Sum Q8 values for min contribution
            // For signed i8, we need to convert to i16 first to avoid overflow
            let q8_lo_256_i16_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q8_lo_256));
            let q8_lo_256_i16_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q8_lo_256, 1));
            let q8_sum_lo = horizontal_sum_epi16_256(q8_lo_256_i16_lo)
                + horizontal_sum_epi16_256(q8_lo_256_i16_hi);

            let q8_hi_256_i16_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q8_hi_256));
            let q8_hi_256_i16_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q8_hi_256, 1));
            let q8_sum_hi = horizontal_sum_epi16_256(q8_hi_256_i16_lo)
                + horizontal_sum_epi16_256(q8_hi_256_i16_hi);

            // Apply scales
            total_acc += d_q8 * sc1 * (sum_lo as f32) - dmin_q8 * m1 * (q8_sum_lo as f32);
            total_acc += d_q8 * sc2 * (sum_hi as f32) - dmin_q8 * m2 * (q8_sum_hi as f32);
        }
    }

    Ok(total_acc)
}
