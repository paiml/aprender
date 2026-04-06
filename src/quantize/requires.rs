
/// AVX2-optimized Q4_K × Q8_K dot product
///
/// # Safety
/// Requires AVX2 CPU feature.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
// SAFETY: Caller must satisfy the documented preconditions
unsafe fn fused_q4k_q8k_dot_avx2(
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

        let mut scales = [0u8; 12];
        scales.copy_from_slice(&q4k_data[sb_start + 4..sb_start + 16]);

        let q8_scale = q8k_scales[sb_idx];
        let d_q8 = d * q8_scale;
        let dmin_q8 = dmin * q8_scale;

        let qs_ptr = q4k_data.as_ptr().add(sb_start + 16);
        let q8_ptr = q8k_quants.as_ptr().add(q8_start);

        // (accumulator variables for future fixed-point optimization)
        let _acc_sum = _mm256_setzero_si256();
        let _acc_min = _mm256_setzero_si256();

        // Process 4 iterations of 64 values each (256 total)
        for j in (0..QK_K).step_by(64) {
            let q_offset = j / 2; // 32 bytes per 64 values

            // Get scales for two 32-value blocks
            let is = j / 32;
            let (sc1, m1) = extract_scale_min(&scales, is);
            let (sc2, m2) = extract_scale_min(&scales, is + 1);

            // Fixed-point scales (8.8 format) - used for integer path
            let sc1_i16 = (sc1 * 256.0).round() as i16;
            let sc2_i16 = (sc2 * 256.0).round() as i16;
            let _m1_i16 = (m1 * 256.0).round() as i16;
            let _m2_i16 = (m2 * 256.0).round() as i16;

            // Load 32 bytes of Q4_K (64 nibbles)
            let q4_bytes = _mm256_loadu_si256(qs_ptr.add(q_offset).cast::<__m256i>());

            // Extract nibbles
            let q4_lo = _mm256_and_si256(q4_bytes, nibble_mask);
            let q4_hi = _mm256_and_si256(_mm256_srli_epi16(q4_bytes, 4), nibble_mask);

            // Load 64 bytes of Q8_K (sequential values)
            // CORRECT LAYOUT: dequantize_q4_k outputs 32 low nibbles, then 32 high nibbles
            // So Q8[j..j+32] corresponds to low nibbles, Q8[j+32..j+64] to high nibbles
            let q8_lo = _mm256_loadu_si256(q8_ptr.add(j).cast::<__m256i>());
            let q8_hi = _mm256_loadu_si256(q8_ptr.add(j + 32).cast::<__m256i>());

            // Q4_lo × Q8_lo: low nibbles times first 32 Q8 values (unsigned × signed → i16)
            let prod_lo = _mm256_maddubs_epi16(q4_lo, q8_lo);
            // Q4_hi × Q8_hi: high nibbles times second 32 Q8 values
            let prod_hi = _mm256_maddubs_epi16(q4_hi, q8_hi);

            // Apply block scales and accumulate (for future integer-only path)
            let _scale_lo = _mm256_set1_epi16(sc1_i16);
            let _scale_hi = _mm256_set1_epi16(sc2_i16);

            // Split products by block (first 128 bits = block 1, second 128 bits = block 2)
            let prod_lo_128 = _mm256_castsi256_si128(prod_lo);
            let prod_lo_hi128 = _mm256_extracti128_si256(prod_lo, 1);
            let prod_hi_128 = _mm256_castsi256_si128(prod_hi);
            let prod_hi_hi128 = _mm256_extracti128_si256(prod_hi, 1);

            // Horizontal sum to i32
            let sum_lo_1 = _mm_madd_epi16(prod_lo_128, _mm_set1_epi16(1));
            let sum_lo_2 = _mm_madd_epi16(prod_lo_hi128, _mm_set1_epi16(1));
            let sum_hi_1 = _mm_madd_epi16(prod_hi_128, _mm_set1_epi16(1));
            let sum_hi_2 = _mm_madd_epi16(prod_hi_hi128, _mm_set1_epi16(1));

            // Add low and high nibble products
            let sum_1 = _mm_add_epi32(sum_lo_1, sum_hi_1);
            let sum_2 = _mm_add_epi32(sum_lo_2, sum_hi_2);

            // Apply scales (as f32 to avoid overflow)
            let sum_1_f = _mm_cvtepi32_ps(sum_1);
            let sum_2_f = _mm_cvtepi32_ps(sum_2);

            let scaled_1 = _mm_mul_ps(sum_1_f, _mm_set1_ps(sc1));
            let scaled_2 = _mm_mul_ps(sum_2_f, _mm_set1_ps(sc2));

            // Sum for min contribution (sum of Q8 values)
            let q8_sum_lo =
                _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(q8_lo)), ones_16);
            let q8_sum_hi = _mm256_madd_epi16(
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q8_lo, 1)),
                ones_16,
            );

            // Horizontal reduce
            let hsum_lo = _mm_add_epi32(
                _mm256_castsi256_si128(q8_sum_lo),
                _mm256_extracti128_si256(q8_sum_lo, 1),
            );
            let _hsum_hi = _mm_add_epi32(
                _mm256_castsi256_si128(q8_sum_hi),
                _mm256_extracti128_si256(q8_sum_hi, 1),
            );

            // Include both halves in block sum
            let q8_block1_sum = _mm_add_epi32(hsum_lo, _mm_shuffle_epi32(hsum_lo, 0b10_11_00_01));
            let q8_block1_sum = _mm_add_epi32(
                q8_block1_sum,
                _mm_shuffle_epi32(q8_block1_sum, 0b00_00_10_10),
            );
            let q8_block1_val = _mm_cvtsi128_si32(q8_block1_sum);

            // Similar for second block (q8_hi)
            let q8_sum_hi2 =
                _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(q8_hi)), ones_16);
            let q8_sum_hi3 = _mm256_madd_epi16(
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q8_hi, 1)),
                ones_16,
            );
            let hsum2_lo = _mm_add_epi32(
                _mm256_castsi256_si128(q8_sum_hi2),
                _mm256_extracti128_si256(q8_sum_hi2, 1),
            );
            let hsum2_hi = _mm_add_epi32(
                _mm256_castsi256_si128(q8_sum_hi3),
                _mm256_extracti128_si256(q8_sum_hi3, 1),
            );
            let q8_block2_sum = _mm_add_epi32(hsum2_lo, hsum2_hi);
            let q8_block2_sum = _mm_add_epi32(
                q8_block2_sum,
                _mm_shuffle_epi32(q8_block2_sum, 0b10_11_00_01),
            );
            let q8_block2_sum = _mm_add_epi32(
                q8_block2_sum,
                _mm_shuffle_epi32(q8_block2_sum, 0b00_00_10_10),
            );
            let q8_block2_val = _mm_cvtsi128_si32(q8_block2_sum);

            // Final accumulation with f32 precision
            let scaled_sum = _mm_add_ps(scaled_1, scaled_2);
            let hsum = _mm_hadd_ps(scaled_sum, scaled_sum);
            let hsum = _mm_hadd_ps(hsum, hsum);
            let block_prod = _mm_cvtss_f32(hsum);

            total_acc += d_q8 * block_prod;
            total_acc -= dmin_q8 * (m1 * q8_block1_val as f32 + m2 * q8_block2_val as f32);
        }
    }

    Ok(total_acc)
}

/// TCB 4-row micro-kernel: Process 4 rows simultaneously sharing Q8K loads
///
/// This implements the trueno TCB micro-tile pattern (4×1×256):
/// - Load Q8K input ONCE per superblock
/// - Process 4 weight rows using the SAME Q8K loads
/// - Return 4 output values
///
/// # Performance
///
/// Sharing Q8K loads across 4 rows reduces memory bandwidth by ~4x for the
/// input vector, which is the key optimization from TCB (Tiling Compute Blocks).
///
/// # Safety
/// Requires AVX-512F, AVX-512 VNNI, and AVX-512BW CPU features.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512vnni", enable = "avx512bw")]
#[allow(unsafe_op_in_unsafe_fn)]
#[allow(clippy::similar_names)]
#[allow(clippy::too_many_lines)]
// SAFETY: Caller must satisfy the documented preconditions
pub(crate) unsafe fn fused_q4k_q8k_dot_4rows_avx512vnni(
    row_ptrs: [*const u8; 4],
    bytes_per_row: usize,
    q8k_scales: &[f32],
    q8k_quants: &[i8],
) -> [f32; 4] {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    const SUPER_BLOCK_BYTES: usize = 144;
    let num_super_blocks = bytes_per_row / SUPER_BLOCK_BYTES;

    let nibble_mask = _mm256_set1_epi8(0x0F_i8);
    let ones_16 = _mm256_set1_epi16(1);

    // 4 accumulators for 4 output rows (8 blocks × f32 = 8 values each)
    let mut total_acc = [_mm256_setzero_ps(); 4];

    for sb_idx in 0..num_super_blocks {
        let q8_start = sb_idx * QK_K;
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;

        // PMAT-299: Deep prefetch — 2 levels to hide DRAM latency.
        // L2 prefetch 2 SBs ahead, L1 prefetch 1 SB ahead.
        // Each Q4K SB = 144 bytes = 3 cache lines (64B each).
        // Tested: 2 SBs = 29.9 tok/s, 3 SBs = 29.1 (L2 pollution). 2 is optimal.
        if sb_idx + 2 < num_super_blocks {
            let far_sb = (sb_idx + 2) * SUPER_BLOCK_BYTES;
            for row in 0..4 {
                let p = row_ptrs[row].add(far_sb).cast::<i8>();
                _mm_prefetch(p, _MM_HINT_T1);
                _mm_prefetch(p.add(64), _MM_HINT_T1);
                _mm_prefetch(p.add(128), _MM_HINT_T1);
            }
        }
        if sb_idx + 1 < num_super_blocks {
            let next_sb = (sb_idx + 1) * SUPER_BLOCK_BYTES;
            _mm_prefetch(
                q8k_quants.as_ptr().add((sb_idx + 1) * QK_K).cast::<i8>(),
                _MM_HINT_T0,
            );
            for row in 0..4 {
                let p = row_ptrs[row].add(next_sb).cast::<i8>();
                _mm_prefetch(p, _MM_HINT_T0);
                _mm_prefetch(p.add(64), _MM_HINT_T0);
                _mm_prefetch(p.add(128), _MM_HINT_T0);
            }
        }

        let q8_scale = q8k_scales[sb_idx];
        let q8_ptr = q8k_quants.as_ptr().add(q8_start);

        // ============================================================
        // CRITICAL: Pre-load Q8K data and compute Q8 sums ONCE per superblock
        // These are shared across ALL 4 rows
        // ============================================================

        // Pre-load all Q8K chunks for this superblock (4 chunks × 2 registers = 8 loads)
        let q8_chunk0_lo = _mm256_loadu_si256(q8_ptr.cast::<__m256i>());
        let q8_chunk0_hi = _mm256_loadu_si256(q8_ptr.add(32).cast::<__m256i>());
        let q8_chunk1_lo = _mm256_loadu_si256(q8_ptr.add(64).cast::<__m256i>());
        let q8_chunk1_hi = _mm256_loadu_si256(q8_ptr.add(96).cast::<__m256i>());
        let q8_chunk2_lo = _mm256_loadu_si256(q8_ptr.add(128).cast::<__m256i>());
        let q8_chunk2_hi = _mm256_loadu_si256(q8_ptr.add(160).cast::<__m256i>());
        let q8_chunk3_lo = _mm256_loadu_si256(q8_ptr.add(192).cast::<__m256i>());
        let q8_chunk3_hi = _mm256_loadu_si256(q8_ptr.add(224).cast::<__m256i>());

        // Pre-compute Q8 sums for dmin correction (same for all rows)
        let q8_sums = compute_q8_sums_8blocks(
            q8_chunk0_lo,
            q8_chunk0_hi,
            q8_chunk1_lo,
            q8_chunk1_hi,
            q8_chunk2_lo,
            q8_chunk2_hi,
            q8_chunk3_lo,
            q8_chunk3_hi,
            ones_16,
        );

        // Process 4 rows using the pre-loaded Q8K data
        for row in 0..4 {
            let row_data = row_ptrs[row].add(sb_start);

            // Read Q4_K header for this row
            let d = read_f16(std::slice::from_raw_parts(row_data, 2));
            let dmin = read_f16(std::slice::from_raw_parts(row_data.add(2), 2));

            let mut scales_raw = [0u8; 12];
            std::ptr::copy_nonoverlapping(row_data.add(4), scales_raw.as_mut_ptr(), 12);

            let d_q8 = d * q8_scale;
            let dmin_q8 = dmin * q8_scale;

            let qs_ptr = row_data.add(16);

            // PMAT-298: AVX-512 dot product FALSIFIED (-16% on Cascade Lake).
            // The 512-bit instructions cause CPU frequency downclocking from
            // 3.2GHz to ~2.5GHz, canceling the 2x throughput benefit.
            // Using AVX2 path which runs at full frequency.
            let block_dots = compute_q4_q8_dots_8blocks(
                qs_ptr,
                q8_chunk0_lo, q8_chunk0_hi, q8_chunk1_lo, q8_chunk1_hi,
                q8_chunk2_lo, q8_chunk2_hi, q8_chunk3_lo, q8_chunk3_hi,
                nibble_mask, ones_16,
            );

            // Extract 6-bit scales and mins
            let mut scales = [0.0f32; 8];
            let mut mins = [0.0f32; 8];
            for i in 0..8 {
                let (sc, m) = extract_scale_min(&scales_raw, i);
                scales[i] = sc;
                mins[i] = m;
            }

            // Final computation: d_q8 * scales * dots - dmin_q8 * mins * q8sums
            let scales_vec = _mm256_loadu_ps(scales.as_ptr());
            let mins_vec = _mm256_loadu_ps(mins.as_ptr());
            let d_q8_vec = _mm256_set1_ps(d_q8);
            let dmin_q8_vec = _mm256_set1_ps(dmin_q8);

            let dots_f32 = _mm256_cvtepi32_ps(block_dots);
            let q8sums_f32 = _mm256_cvtepi32_ps(q8_sums);

            let term1 = _mm256_mul_ps(d_q8_vec, _mm256_mul_ps(scales_vec, dots_f32));
            let term2 = _mm256_mul_ps(dmin_q8_vec, _mm256_mul_ps(mins_vec, q8sums_f32));
            let result = _mm256_sub_ps(term1, term2);

            total_acc[row] = _mm256_add_ps(total_acc[row], result);
        }
    }

    // Final horizontal sums for each row
    let mut outputs = [0.0f32; 4];
    for row in 0..4 {
        let sum128 = _mm_add_ps(
            _mm256_castps256_ps128(total_acc[row]),
            _mm256_extractf128_ps(total_acc[row], 1),
        );
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        outputs[row] = _mm_cvtss_f32(sum32);
    }

    outputs
}

/// Helper: Compute Q8 sums for 8 blocks (shared across rows)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(clippy::too_many_arguments)]
// SAFETY: Caller must satisfy the documented preconditions
unsafe fn compute_q8_sums_8blocks(
    c0_lo: std::arch::x86_64::__m256i,
    c0_hi: std::arch::x86_64::__m256i,
    c1_lo: std::arch::x86_64::__m256i,
    c1_hi: std::arch::x86_64::__m256i,
    c2_lo: std::arch::x86_64::__m256i,
    c2_hi: std::arch::x86_64::__m256i,
    c3_lo: std::arch::x86_64::__m256i,
    c3_hi: std::arch::x86_64::__m256i,
    ones_16: std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256i {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    // SAFETY: Caller must satisfy the documented preconditions
    // SAFETY: All calls are to unsafe intrinsics, within already-unsafe fn
    // Sum Q8 values for each of the 8 blocks (32 values each)
    unsafe {
        let sum0 = sum_i8_to_i32(c0_lo, ones_16);
        let sum1 = sum_i8_to_i32(c0_hi, ones_16);
        let sum2 = sum_i8_to_i32(c1_lo, ones_16);
        let sum3 = sum_i8_to_i32(c1_hi, ones_16);
        let sum4 = sum_i8_to_i32(c2_lo, ones_16);
        let sum5 = sum_i8_to_i32(c2_hi, ones_16);
        let sum6 = sum_i8_to_i32(c3_lo, ones_16);
        let sum7 = sum_i8_to_i32(c3_hi, ones_16);

        // Pack 8 sums into a single __m256i
        let mut result = _mm256_setzero_si256();
        result = _mm256_insert_epi32(result, sum0, 0);
        result = _mm256_insert_epi32(result, sum1, 1);
        result = _mm256_insert_epi32(result, sum2, 2);
        result = _mm256_insert_epi32(result, sum3, 3);
        result = _mm256_insert_epi32(result, sum4, 4);
        result = _mm256_insert_epi32(result, sum5, 5);
        result = _mm256_insert_epi32(result, sum6, 6);
        result = _mm256_insert_epi32(result, sum7, 7);
        result
    }
}

/// Helper: Sum 32 i8 values to i32
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
// SAFETY: Caller must satisfy the documented preconditions
unsafe fn sum_i8_to_i32(v: std::arch::x86_64::__m256i, ones: std::arch::x86_64::__m256i) -> i32 {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    let lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v));
    let hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1));
    let sum_lo = _mm256_madd_epi16(lo, ones);
    let sum_hi = _mm256_madd_epi16(hi, ones);
    let sum = _mm256_add_epi32(sum_lo, sum_hi);

    // Horizontal sum of 8 i32 -> 1 i32
    let sum128 = _mm_add_epi32(
        _mm256_castsi256_si128(sum),
        _mm256_extracti128_si256(sum, 1),
    );
    let sum64 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b10_11_00_01));
    let sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32(sum64, 0b00_00_10_10));
    _mm_cvtsi128_si32(sum32)
}
