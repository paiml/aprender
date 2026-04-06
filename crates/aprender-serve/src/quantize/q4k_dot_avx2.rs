
/// AVX2-accelerated fused Q4_K dequant+dot kernel (PARITY-003: llama.cpp-style SIMD)
///
/// # Safety
///
/// Caller must ensure:
/// 1. AVX2 and FMA CPU features are available (use `is_x86_feature_detected!`)
/// 2. Input slices are valid (handled by Rust's slice guarantees)
///
/// This function is marked unsafe due to SIMD intrinsics, but is logically
/// equivalent to the scalar `fused_q4k_dot` (within ULP tolerance).
///
/// # Optimizations (PARITY-003)
/// - SIMD loads: AVX2 256-bit unaligned load for 32-byte bulk loads (vs scalar byte loads)
/// - SIMD nibble extraction: AVX2 bitwise AND with 0x0F mask (vs scalar & 0x0F)
/// - 4 independent accumulators to hide FMA latency
/// - Software prefetching for next super-block
/// - Matches llama.cpp ggml_vec_dot_q4_K_q8_K pattern
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(unsafe_op_in_unsafe_fn)]
// SAFETY: Caller must satisfy the documented preconditions
unsafe fn fused_q4k_dot_avx2(q4k_data: &[u8], activations: &[f32]) -> Result<f32> {
    // Allow wildcard import for SIMD intrinsics (standard pattern for arch-specific code)
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    const SUPER_BLOCK_BYTES: usize = 144;

    // Validate inputs (same as scalar)
    if !q4k_data.len().is_multiple_of(SUPER_BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q4_K data length {} is not a multiple of super-block size {}",
                q4k_data.len(),
                SUPER_BLOCK_BYTES
            ),
        });
    }

    let num_super_blocks = q4k_data.len() / SUPER_BLOCK_BYTES;
    let expected_values = num_super_blocks * QK_K;

    if activations.len() != expected_values {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Activation length {} doesn't match Q4_K values count {}",
                activations.len(),
                expected_values
            ),
        });
    }

    // Nibble mask for extracting 4-bit values (llama.cpp pattern)
    let nibble_mask = _mm256_set1_epi8(0x0F_i8);

    // PARITY-003: 4 independent accumulators to hide FMA latency
    // FMA latency = 4 cycles, throughput = 2/cycle
    // With 4 independent chains, we saturate the FMA throughput
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut activation_idx = 0;

    for sb_idx in 0..num_super_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;

        // Prefetch next super-block (dual: Q4_K weights + activations)
        if sb_idx + 1 < num_super_blocks {
            let next_sb = (sb_idx + 1) * SUPER_BLOCK_BYTES;
            _mm_prefetch(q4k_data.as_ptr().add(next_sb).cast::<i8>(), _MM_HINT_T0);
            _mm_prefetch(
                activations.as_ptr().add((sb_idx + 1) * QK_K).cast::<i8>(),
                _MM_HINT_T0,
            );
        }

        // Read d and dmin (f16 -> f32)
        let d = read_f16(&q4k_data[sb_start..sb_start + 2]);
        let dmin = read_f16(&q4k_data[sb_start + 2..sb_start + 4]);

        // Read scales (12 bytes)
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&q4k_data[sb_start + 4..sb_start + 16]);

        // Pointer to quantized data (128 bytes = 256 nibbles = 256 values)
        let qs_ptr = q4k_data.as_ptr().add(sb_start + 16);

        // PAR-001: Match dequantize_q4_k layout (llama.cpp/candle compatible)
        // Process 4 chunks of 64 values each (j=0, 64, 128, 192)
        // Each chunk: 32 low nibbles (sc1), then 32 high nibbles (sc2)
        for j in (0..QK_K).step_by(64) {
            let q_start = j / 2; // 32 bytes per 64-value chunk

            // Get scales for the two 32-value halves
            let is = j / 32;
            let (sc1, m1) = extract_scale_min(&scales, is);
            let (sc2, m2) = extract_scale_min(&scales, is + 1);

            // Precompute d*scale and dmin*min for both halves
            let d_scale1 = d * sc1;
            let dm1 = dmin * m1;
            let d_scale2 = d * sc2;
            let dm2 = dmin * m2;

            // SIMD OPTIMIZATION: Load 32 bytes at once (64 nibbles = 64 values)
            // llama.cpp pattern: _mm256_loadu_si256 + AND/SHIFT for nibble extraction
            let q_bytes = _mm256_loadu_si256(qs_ptr.add(q_start).cast::<__m256i>());

            // Extract low nibbles (first 32 values) and high nibbles (second 32 values)
            let q_lo = _mm256_and_si256(q_bytes, nibble_mask);
            let q_hi = _mm256_and_si256(_mm256_srli_epi16(q_bytes, 4), nibble_mask);

            // Process low nibbles (32 values with scale sc1/m1)
            // Split into 4 groups of 8 for f32 conversion (AVX2 can convert 8 i32→f32)
            let d_scale1_vec = _mm256_set1_ps(d_scale1);
            let dm1_vec = _mm256_set1_ps(dm1);

            // Extract bytes 0-7 (low nibbles) → convert to f32
            let q_lo_128_0 = _mm256_castsi256_si128(q_lo);
            let q_lo_i32_0 = _mm256_cvtepu8_epi32(q_lo_128_0);
            let q_lo_f32_0 = _mm256_cvtepi32_ps(q_lo_i32_0);
            let dequant0 = _mm256_fmsub_ps(d_scale1_vec, q_lo_f32_0, dm1_vec);
            let act0 = _mm256_loadu_ps(activations.as_ptr().add(activation_idx));
            acc0 = _mm256_fmadd_ps(dequant0, act0, acc0);
            activation_idx += 8;

            // Extract bytes 8-15 (low nibbles)
            let q_lo_shifted = _mm_srli_si128(q_lo_128_0, 8);
            let q_lo_i32_1 = _mm256_cvtepu8_epi32(q_lo_shifted);
            let q_lo_f32_1 = _mm256_cvtepi32_ps(q_lo_i32_1);
            let dequant1 = _mm256_fmsub_ps(d_scale1_vec, q_lo_f32_1, dm1_vec);
            let act1 = _mm256_loadu_ps(activations.as_ptr().add(activation_idx));
            acc1 = _mm256_fmadd_ps(dequant1, act1, acc1);
            activation_idx += 8;

            // Extract bytes 16-23 (low nibbles from high 128 bits)
            let q_lo_128_1 = _mm256_extracti128_si256(q_lo, 1);
            let q_lo_i32_2 = _mm256_cvtepu8_epi32(q_lo_128_1);
            let q_lo_f32_2 = _mm256_cvtepi32_ps(q_lo_i32_2);
            let dequant2 = _mm256_fmsub_ps(d_scale1_vec, q_lo_f32_2, dm1_vec);
            let act2 = _mm256_loadu_ps(activations.as_ptr().add(activation_idx));
            acc2 = _mm256_fmadd_ps(dequant2, act2, acc2);
            activation_idx += 8;

            // Extract bytes 24-31 (low nibbles)
            let q_lo_shifted2 = _mm_srli_si128(q_lo_128_1, 8);
            let q_lo_i32_3 = _mm256_cvtepu8_epi32(q_lo_shifted2);
            let q_lo_f32_3 = _mm256_cvtepi32_ps(q_lo_i32_3);
            let dequant3 = _mm256_fmsub_ps(d_scale1_vec, q_lo_f32_3, dm1_vec);
            let act3 = _mm256_loadu_ps(activations.as_ptr().add(activation_idx));
            acc3 = _mm256_fmadd_ps(dequant3, act3, acc3);
            activation_idx += 8;

            // Process high nibbles (32 values with scale sc2/m2)
            let d_scale2_vec = _mm256_set1_ps(d_scale2);
            let dm2_vec = _mm256_set1_ps(dm2);

            // Extract bytes 0-7 (high nibbles)
            let q_hi_128_0 = _mm256_castsi256_si128(q_hi);
            let q_hi_i32_0 = _mm256_cvtepu8_epi32(q_hi_128_0);
            let q_hi_f32_0 = _mm256_cvtepi32_ps(q_hi_i32_0);
            let dequant4 = _mm256_fmsub_ps(d_scale2_vec, q_hi_f32_0, dm2_vec);
            let act4 = _mm256_loadu_ps(activations.as_ptr().add(activation_idx));
            acc0 = _mm256_fmadd_ps(dequant4, act4, acc0);
            activation_idx += 8;

            // Extract bytes 8-15 (high nibbles)
            let q_hi_shifted = _mm_srli_si128(q_hi_128_0, 8);
            let q_hi_i32_1 = _mm256_cvtepu8_epi32(q_hi_shifted);
            let q_hi_f32_1 = _mm256_cvtepi32_ps(q_hi_i32_1);
            let dequant5 = _mm256_fmsub_ps(d_scale2_vec, q_hi_f32_1, dm2_vec);
            let act5 = _mm256_loadu_ps(activations.as_ptr().add(activation_idx));
            acc1 = _mm256_fmadd_ps(dequant5, act5, acc1);
            activation_idx += 8;

            // Extract bytes 16-23 (high nibbles from high 128 bits)
            let q_hi_128_1 = _mm256_extracti128_si256(q_hi, 1);
            let q_hi_i32_2 = _mm256_cvtepu8_epi32(q_hi_128_1);
            let q_hi_f32_2 = _mm256_cvtepi32_ps(q_hi_i32_2);
            let dequant6 = _mm256_fmsub_ps(d_scale2_vec, q_hi_f32_2, dm2_vec);
            let act6 = _mm256_loadu_ps(activations.as_ptr().add(activation_idx));
            acc2 = _mm256_fmadd_ps(dequant6, act6, acc2);
            activation_idx += 8;

            // Extract bytes 24-31 (high nibbles)
            let q_hi_shifted2 = _mm_srli_si128(q_hi_128_1, 8);
            let q_hi_i32_3 = _mm256_cvtepu8_epi32(q_hi_shifted2);
            let q_hi_f32_3 = _mm256_cvtepi32_ps(q_hi_i32_3);
            let dequant7 = _mm256_fmsub_ps(d_scale2_vec, q_hi_f32_3, dm2_vec);
            let act7 = _mm256_loadu_ps(activations.as_ptr().add(activation_idx));
            acc3 = _mm256_fmadd_ps(dequant7, act7, acc3);
            activation_idx += 8;
        }
    }

    // Combine 4 accumulators → single accumulator
    let acc_01 = _mm256_add_ps(acc0, acc1);
    let acc_23 = _mm256_add_ps(acc2, acc3);
    let acc = _mm256_add_ps(acc_01, acc_23);

    // Horizontal sum: reduce 8 lanes to single value
    let sum_halves = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    let temp = _mm_add_ps(sum_halves, _mm_movehl_ps(sum_halves, sum_halves));
    let temp = _mm_add_ss(temp, _mm_shuffle_ps(temp, temp, 1));
    let result = _mm_cvtss_f32(temp);

    Ok(result)
}

// ============================================================================
// Q4_K × Q8_K KERNELS (Super-block aligned integer-only arithmetic)
// ============================================================================

/// Fused Q4_K × Q8_K dot product (super-block aligned, llama.cpp-style)
///
/// Uses Q8_K format (256 values per super-block, single scale) for maximum
/// SIMD efficiency. This matches llama.cpp's `ggml_vec_dot_q4_K_q8_K`.
///
/// # Arguments
/// * `q4k_data` - Raw Q4_K quantized data (144 bytes per super-block)
/// * `q8k_scales` - Q8_K scales (one per super-block)
/// * `q8k_quants` - Q8_K quantized int8 values (256 per super-block)
///
/// # Performance
///
/// Compared to Q4_K × f32:
/// - 8x fewer memory reads for activations
/// - Integer-only inner loop (no f32 conversion until end)
/// - Single scale multiplication per super-block (vs 8 for Q8_0)
pub fn fused_q4k_q8k_dot(q4k_data: &[u8], q8k_scales: &[f32], q8k_quants: &[i8]) -> Result<f32> {
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

    if q8k_scales.len() < num_super_blocks {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q8_K scales count {} < expected {}",
                q8k_scales.len(),
                num_super_blocks
            ),
        });
    }

    if q8k_quants.len() < expected_values {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q8_K quants count {} < expected {}",
                q8k_quants.len(),
                expected_values
            ),
        });
    }

    let mut total_acc = 0.0f32;

    for sb_idx in 0..num_super_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let q8_start = sb_idx * QK_K;

        // Read Q4_K super-block header
        let d = read_f16(&q4k_data[sb_start..sb_start + 2]);
        let dmin = read_f16(&q4k_data[sb_start + 2..sb_start + 4]);

        // Read scales (12 bytes for 8 blocks)
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&q4k_data[sb_start + 4..sb_start + 16]);

        // Q8_K scale for this super-block
        let q8_scale = q8k_scales[sb_idx];

        // Process 4 chunks of 64 values (matching dequantize_q4_k layout)
        // The dequantized output order is: 32 low nibbles, then 32 high nibbles
        // So activations[j..j+32] correspond to low nibbles, activations[j+32..j+64] to high
        for j in (0..QK_K).step_by(64) {
            let q_offset = sb_start + 16 + j / 2; // 32 bytes per 64-value chunk
            let q8_offset = q8_start + j;

            // Get scales for low and high nibbles
            let is = j / 32;
            let (sc1, m1) = extract_scale_min(&scales, is);
            let (sc2, m2) = extract_scale_min(&scales, is + 1);

            // Combined scale factors
            let d_sc1_q8 = d * sc1 * q8_scale;
            let dm1_q8 = dmin * m1 * q8_scale;
            let d_sc2_q8 = d * sc2 * q8_scale;
            let dm2_q8 = dmin * m2 * q8_scale;

            // Accumulators for low and high nibbles
            let mut sum_lo: i32 = 0; // q4_lo × q8 (for activations[j..j+32])
            let mut sum_hi: i32 = 0; // q4_hi × q8 (for activations[j+32..j+64])
            let mut q8_sum_lo: i32 = 0;
            let mut q8_sum_hi: i32 = 0;

            for b in 0..32 {
                let q4_byte = q4k_data[q_offset + b];

                // Low nibble × activation[j + b] (first 32 positions in dequant order)
                let q4_lo = (q4_byte & 0x0F) as i32;
                let q8_lo = q8k_quants[q8_offset + b] as i32;
                sum_lo += q4_lo * q8_lo;
                q8_sum_lo += q8_lo;

                // High nibble × activation[j + 32 + b] (second 32 positions in dequant order)
                let q4_hi = ((q4_byte >> 4) & 0x0F) as i32;
                let q8_hi = q8k_quants[q8_offset + 32 + b] as i32;
                sum_hi += q4_hi * q8_hi;
                q8_sum_hi += q8_hi;
            }

            // Apply formula: (d * scale * sum_q4_q8 - dmin * min * sum_q8) * q8_scale
            total_acc += d_sc1_q8 * (sum_lo as f32) - dm1_q8 * (q8_sum_lo as f32);
            total_acc += d_sc2_q8 * (sum_hi as f32) - dm2_q8 * (q8_sum_hi as f32);
        }
    }

    Ok(total_acc)
}

/// SIMD-accelerated Q4_K × Q8_K dot product
///
/// Uses AVX-512 VNNI (vpdpbusd) for maximum throughput, falls back to AVX2 or scalar.
/// Single scale per super-block eliminates per-block overhead.
pub fn fused_q4k_q8k_dot_simd(
    q4k_data: &[u8],
    q8k_scales: &[f32],
    q8k_quants: &[i8],
) -> Result<f32> {
    #[cfg(target_arch = "x86_64")]
    {
        // PAR-126: Use V2 optimized AVX-512 VNNI kernel (deferred horizontal sums)
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vnni") {
            // SAFETY: Memory safety ensured by bounds checking and alignment
            return unsafe { fused_q4k_q8k_dot_avx512vnni_v2(q4k_data, q8k_scales, q8k_quants) };
        }
        // pmat-ignore: hardware-path (AVX2 fallback never reached when AVX-512 VNNI available)
        // Fallback to AVX2 (layout issue resolved)
        if is_x86_feature_detected!("avx2") {
            // SAFETY: Memory safety ensured by bounds checking and alignment
            return unsafe { fused_q4k_q8k_dot_avx2(q4k_data, q8k_scales, q8k_quants) };
        }
    }

    // pmat-ignore: hardware-path (scalar fallback tested directly via fused_q4k_q8k_dot)
    fused_q4k_q8k_dot(q4k_data, q8k_scales, q8k_quants)
}
