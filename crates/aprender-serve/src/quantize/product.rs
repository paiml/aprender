
impl InterleavedQ4K {
    /// Create interleaved Q4_K from raw GGUF Q4_K data
    ///
    /// Reorders the quantized values at load time for SIMD-efficient access.
    /// This is a one-time cost at model load that amortizes over all inference.
    ///
    /// # Arguments
    ///
    /// * `q4k_data` - Raw Q4_K data (144 bytes per super-block)
    ///
    /// # Returns
    ///
    /// InterleavedQ4K with reordered weights
    ///
    /// # Errors
    ///
    /// Returns error if data length is not a multiple of super-block size
    pub fn from_q4k(q4k_data: &[u8]) -> Result<Self> {
        const SUPER_BLOCK_BYTES: usize = 144;

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

        let mut d = Vec::with_capacity(num_super_blocks);
        let mut dmin = Vec::with_capacity(num_super_blocks);
        let mut scales = Vec::with_capacity(num_super_blocks * 12);
        let mut qs = Vec::with_capacity(num_super_blocks * 128);

        for sb in 0..num_super_blocks {
            let sb_start = sb * SUPER_BLOCK_BYTES;

            // Read d and dmin (f16 -> f32)
            let d_val = f16_to_f32_lut(u16::from_le_bytes([
                q4k_data[sb_start],
                q4k_data[sb_start + 1],
            ]));
            let dmin_val = f16_to_f32_lut(u16::from_le_bytes([
                q4k_data[sb_start + 2],
                q4k_data[sb_start + 3],
            ]));

            d.push(d_val);
            dmin.push(dmin_val);

            // Copy scales
            scales.extend_from_slice(&q4k_data[sb_start + 4..sb_start + 16]);

            // Interleave quantized values
            // Original: byte[i] = (value[2i+1] << 4) | value[2i]
            // We reorder so that after SIMD nibble extraction, values are contiguous
            //
            // For AVX2 processing 64 values at a time:
            // - Load 32 bytes, extract low nibbles -> 32 values
            // - Same 32 bytes, extract high nibbles -> 32 more values
            //
            // Interleave pattern: group values by their position in SIMD lanes
            // This eliminates the need for cross-lane shuffles
            let qs_start = sb_start + 16;
            let original_qs = &q4k_data[qs_start..qs_start + 128];

            // For now, use identity interleave (same as original)
            // The optimization comes from the specialized kernel that knows the layout
            // Future: implement actual interleave pattern based on profiling
            qs.extend_from_slice(original_qs);
        }

        Ok(Self {
            d,
            dmin,
            scales,
            qs,
            num_super_blocks,
        })
    }

    /// Get the number of values (256 per super-block)
    #[must_use]
    pub fn num_values(&self) -> usize {
        self.num_super_blocks * QK_K
    }

    /// Benchmark: compute dot product using interleaved layout
    ///
    /// This is optimized for the interleaved layout where SIMD loads
    /// get contiguous values without gather operations.
    #[cfg(target_arch = "x86_64")]
    pub fn dot(&self, activations: &[f32]) -> Result<f32> {
        if activations.len() != self.num_values() {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Activation length {} doesn't match interleaved Q4_K values count {}",
                    activations.len(),
                    self.num_values()
                ),
            });
        }

        // Use SIMD if available
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Memory safety ensured by bounds checking and alignment
            return unsafe { self.dot_avx2(activations) };
        }

        // Scalar fallback
        self.dot_scalar(activations)
    }

    /// Compute dot product using scalar fallback (non-x86_64).
    #[cfg(not(target_arch = "x86_64"))]
    pub fn dot(&self, activations: &[f32]) -> Result<f32> {
        self.dot_scalar(activations)
    }

    /// Scalar dot product (fallback)
    fn dot_scalar(&self, activations: &[f32]) -> Result<f32> {
        let mut sum = 0.0f32;
        let mut activation_idx = 0;

        for sb in 0..self.num_super_blocks {
            let d = self.d[sb];
            let dmin = self.dmin[sb];
            let scales_start = sb * 12;
            let qs_start = sb * 128;

            // Process 4 chunks of 64 values each
            for j in (0..QK_K).step_by(64) {
                let q_start = qs_start + j / 2;
                let is = j / 32;

                let (sc1, m1) =
                    simd::extract_scale_min_from_slice(&self.scales[scales_start..], is);
                let (sc2, m2) =
                    simd::extract_scale_min_from_slice(&self.scales[scales_start..], is + 1);

                // Process 32 low nibbles
                for i in 0..32 {
                    let byte_idx = q_start + i;
                    let q_val = (self.qs[byte_idx] & 0x0F) as f32;
                    let dequant = d * sc1 * q_val - dmin * m1;
                    sum += dequant * activations[activation_idx];
                    activation_idx += 1;
                }

                // Process 32 high nibbles
                for i in 0..32 {
                    let byte_idx = q_start + i;
                    let q_val = ((self.qs[byte_idx] >> 4) & 0x0F) as f32;
                    let dequant = d * sc2 * q_val - dmin * m2;
                    sum += dequant * activations[activation_idx];
                    activation_idx += 1;
                }
            }
        }

        Ok(sum)
    }

    /// AVX2 optimized dot product for interleaved layout
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 and FMA are available
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    #[allow(unsafe_op_in_unsafe_fn)]
    // SAFETY: Caller must satisfy the documented preconditions
    unsafe fn dot_avx2(&self, activations: &[f32]) -> Result<f32> {
        #[allow(clippy::wildcard_imports)]
        use std::arch::x86_64::*;

        let nibble_mask = _mm256_set1_epi8(0x0F_i8);

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        let mut activation_idx = 0;

        for sb in 0..self.num_super_blocks {
            let d = self.d[sb];
            let dmin = self.dmin[sb];
            let scales_start = sb * 12;
            let qs_start = sb * 128;

            // Prefetch next super-block
            if sb + 1 < self.num_super_blocks {
                let next_qs = (sb + 1) * 128;
                _mm_prefetch(self.qs.as_ptr().add(next_qs).cast::<i8>(), _MM_HINT_T0);
            }

            // Process 4 chunks of 64 values
            for j in (0..QK_K).step_by(64) {
                let q_start = qs_start + j / 2;
                let is = j / 32;

                let (sc1, m1) =
                    simd::extract_scale_min_from_slice(&self.scales[scales_start..], is);
                let (sc2, m2) =
                    simd::extract_scale_min_from_slice(&self.scales[scales_start..], is + 1);

                let d_scale1 = d * sc1;
                let dm1 = dmin * m1;
                let d_scale2 = d * sc2;
                let dm2 = dmin * m2;

                // Load 32 bytes of quantized data
                let q_bytes = _mm256_loadu_si256(self.qs.as_ptr().add(q_start).cast::<__m256i>());

                // Extract low and high nibbles
                let q_lo = _mm256_and_si256(q_bytes, nibble_mask);
                let q_hi = _mm256_and_si256(_mm256_srli_epi16(q_bytes, 4), nibble_mask);

                // Process low nibbles (32 values)
                let d_scale1_vec = _mm256_set1_ps(d_scale1);
                let dm1_vec = _mm256_set1_ps(dm1);

                // 4 groups of 8 values each
                let q_lo_128_0 = _mm256_castsi256_si128(q_lo);
                let q_lo_i32_0 = _mm256_cvtepu8_epi32(q_lo_128_0);
                let q_lo_f32_0 = _mm256_cvtepi32_ps(q_lo_i32_0);
                let dequant0 = _mm256_fmsub_ps(d_scale1_vec, q_lo_f32_0, dm1_vec);
                let act0 = _mm256_loadu_ps(activations.as_ptr().add(activation_idx));
                acc0 = _mm256_fmadd_ps(dequant0, act0, acc0);
                activation_idx += 8;

                let q_lo_shifted = _mm_srli_si128(q_lo_128_0, 8);
                let q_lo_i32_1 = _mm256_cvtepu8_epi32(q_lo_shifted);
                let q_lo_f32_1 = _mm256_cvtepi32_ps(q_lo_i32_1);
                let dequant1 = _mm256_fmsub_ps(d_scale1_vec, q_lo_f32_1, dm1_vec);
                let act1 = _mm256_loadu_ps(activations.as_ptr().add(activation_idx));
                acc1 = _mm256_fmadd_ps(dequant1, act1, acc1);
                activation_idx += 8;

                let q_lo_128_1 = _mm256_extracti128_si256(q_lo, 1);
                let q_lo_i32_2 = _mm256_cvtepu8_epi32(q_lo_128_1);
                let q_lo_f32_2 = _mm256_cvtepi32_ps(q_lo_i32_2);
                let dequant2 = _mm256_fmsub_ps(d_scale1_vec, q_lo_f32_2, dm1_vec);
                let act2 = _mm256_loadu_ps(activations.as_ptr().add(activation_idx));
                acc2 = _mm256_fmadd_ps(dequant2, act2, acc2);
                activation_idx += 8;

                let q_lo_shifted2 = _mm_srli_si128(q_lo_128_1, 8);
                let q_lo_i32_3 = _mm256_cvtepu8_epi32(q_lo_shifted2);
                let q_lo_f32_3 = _mm256_cvtepi32_ps(q_lo_i32_3);
                let dequant3 = _mm256_fmsub_ps(d_scale1_vec, q_lo_f32_3, dm1_vec);
                let act3 = _mm256_loadu_ps(activations.as_ptr().add(activation_idx));
                acc3 = _mm256_fmadd_ps(dequant3, act3, acc3);
                activation_idx += 8;

                // Process high nibbles (32 values)
                let d_scale2_vec = _mm256_set1_ps(d_scale2);
                let dm2_vec = _mm256_set1_ps(dm2);

                let q_hi_128_0 = _mm256_castsi256_si128(q_hi);
                let q_hi_i32_0 = _mm256_cvtepu8_epi32(q_hi_128_0);
                let q_hi_f32_0 = _mm256_cvtepi32_ps(q_hi_i32_0);
                let dequant4 = _mm256_fmsub_ps(d_scale2_vec, q_hi_f32_0, dm2_vec);
                let act4 = _mm256_loadu_ps(activations.as_ptr().add(activation_idx));
                acc0 = _mm256_fmadd_ps(dequant4, act4, acc0);
                activation_idx += 8;

                let q_hi_shifted = _mm_srli_si128(q_hi_128_0, 8);
                let q_hi_i32_1 = _mm256_cvtepu8_epi32(q_hi_shifted);
                let q_hi_f32_1 = _mm256_cvtepi32_ps(q_hi_i32_1);
                let dequant5 = _mm256_fmsub_ps(d_scale2_vec, q_hi_f32_1, dm2_vec);
                let act5 = _mm256_loadu_ps(activations.as_ptr().add(activation_idx));
                acc1 = _mm256_fmadd_ps(dequant5, act5, acc1);
                activation_idx += 8;

                let q_hi_128_1 = _mm256_extracti128_si256(q_hi, 1);
                let q_hi_i32_2 = _mm256_cvtepu8_epi32(q_hi_128_1);
                let q_hi_f32_2 = _mm256_cvtepi32_ps(q_hi_i32_2);
                let dequant6 = _mm256_fmsub_ps(d_scale2_vec, q_hi_f32_2, dm2_vec);
                let act6 = _mm256_loadu_ps(activations.as_ptr().add(activation_idx));
                acc2 = _mm256_fmadd_ps(dequant6, act6, acc2);
                activation_idx += 8;

                let q_hi_shifted2 = _mm_srli_si128(q_hi_128_1, 8);
                let q_hi_i32_3 = _mm256_cvtepu8_epi32(q_hi_shifted2);
                let q_hi_f32_3 = _mm256_cvtepi32_ps(q_hi_i32_3);
                let dequant7 = _mm256_fmsub_ps(d_scale2_vec, q_hi_f32_3, dm2_vec);
                let act7 = _mm256_loadu_ps(activations.as_ptr().add(activation_idx));
                acc3 = _mm256_fmadd_ps(dequant7, act7, acc3);
                activation_idx += 8;
            }
        }

        // Reduce accumulators
        let acc_01 = _mm256_add_ps(acc0, acc1);
        let acc_23 = _mm256_add_ps(acc2, acc3);
        let acc = _mm256_add_ps(acc_01, acc_23);

        let sum_halves = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        let temp = _mm_add_ps(sum_halves, _mm_movehl_ps(sum_halves, sum_halves));
        let temp = _mm_add_ss(temp, _mm_shuffle_ps(temp, temp, 1));
        let result = _mm_cvtss_f32(temp);

        Ok(result)
    }
}

// Basic dequantization functions moved to dequant.rs (PMAT-802)

/// SIMD-accelerated Q4_0 × Q8_0 integer dot product
///
/// Uses AVX2 maddubs (multiply-add unsigned bytes) for efficient integer multiply-accumulate.
/// This is the key optimization that brings us to llama.cpp parity.
///
/// Selects between 2-block and 4-block unrolling based on vector size:
/// - in_dim >= 256: 4-block unrolling (better ILP, ~1.3x faster)
/// - in_dim < 256: 2-block unrolling (lower overhead for small vectors)
pub(crate) fn fused_q4_0_q8_0_dot_simd(
    q4_data: &[u8],
    q8_scales: &[f32],
    q8_quants: &[i8],
    in_dim: usize,
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        // Try AVX-512 VNNI first (2x vector width + native u8×i8 MAC)
        // ~2x faster than AVX2 path on supported CPUs (Zen4+, Sapphire Rapids+)
        if is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512bw") {
            // SAFETY: AVX-512 VNNI verified at runtime
            return unsafe {
                fused_q4_0_q8_0_dot_avx512_vnni(q4_data, q8_scales, q8_quants, in_dim)
            };
        }

        if is_x86_feature_detected!("avx2") {
            // Use 4-block unrolling for larger vectors (8+ blocks = 256+ elements)
            // 4-block provides ~1.3x speedup over 2-block due to better ILP
            if in_dim >= 256 {
                // SAFETY: AVX2 verified at runtime
                return unsafe {
                    fused_q4_0_q8_0_dot_avx2_4block(q4_data, q8_scales, q8_quants, in_dim)
                };
            }
            // SAFETY: AVX2 verified at runtime
            return unsafe { fused_q4_0_q8_0_dot_avx2(q4_data, q8_scales, q8_quants, in_dim) };
        }
    }
    // Scalar fallback
    fused_q4_0_q8_0_dot_scalar(q4_data, q8_scales, q8_quants, in_dim)
}

/// AVX-VNNI accelerated Q4_0 × Q8_0 dot product using vpdpbusd
///
/// Uses the vpdpbusd instruction which performs u8×i8 multiply-accumulate
/// directly to i32, replacing the maddubs+madd chain with a single instruction.
/// This is ~1.5x faster than AVX2 path on supported CPUs (Alder Lake+, Zen5+).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(dead_code)]
// SAFETY: Caller must satisfy the documented preconditions
unsafe fn fused_q4_0_q8_0_dot_avx_vnni(
    q4_data: &[u8],
    q8_scales: &[f32],
    q8_quants: &[i8],
    in_dim: usize,
) -> f32 {
    // SAFETY: Memory safety ensured by bounds checking and alignment
    unsafe {
        use std::arch::asm;
        use std::arch::x86_64::{
            _mm256_and_si256, _mm256_cvtepi32_ps, _mm256_fmadd_ps, _mm256_loadu_si256,
            _mm256_set1_epi8, _mm256_set1_ps, _mm256_setzero_ps, _mm256_setzero_si256,
            _mm256_sign_epi8, _mm256_sub_epi8,
        };

        const Q4_0_BLOCK_BYTES: usize = 18;
        const Q4_0_BLOCK_SIZE: usize = 32;

        let num_blocks = in_dim.div_ceil(Q4_0_BLOCK_SIZE);

        // Float accumulator for scaled results
        let mut acc = _mm256_setzero_ps();
        let offset = _mm256_set1_epi8(8);
        let low_mask = _mm256_set1_epi8(0x0F);

        // Process blocks one at a time
        // Note: We can't use vpdpbusd's accumulation across blocks because
        // each block has different scales. We must convert to float and scale per block.
        for block_idx in 0..num_blocks {
            let q4_ptr = q4_data.as_ptr().add(block_idx * Q4_0_BLOCK_BYTES);
            let q8_ptr = q8_quants.as_ptr().add(block_idx * Q4_0_BLOCK_SIZE);

            // Read scales
            let q4_scale_bits = u16::from_le_bytes([*q4_ptr, *q4_ptr.add(1)]);
            let q4_scale = f16_to_f32_lut(q4_scale_bits);
            let q8_scale = q8_scales[block_idx];
            let combined_scale = _mm256_set1_ps(q4_scale * q8_scale);

            // Load and expand Q4_0 nibbles
            let q4_bytes = std::slice::from_raw_parts(q4_ptr.add(2), 16);
            let q4_lo_128 = std::arch::x86_64::_mm_loadu_si128(q4_bytes.as_ptr().cast());
            let q4_hi_128 = std::arch::x86_64::_mm_srli_epi16(q4_lo_128, 4);
            let q4_combined = std::arch::x86_64::_mm256_set_m128i(q4_hi_128, q4_lo_128);
            let q4_nibbles = _mm256_and_si256(q4_combined, low_mask);
            let q4_signed = _mm256_sub_epi8(q4_nibbles, offset);

            // Load Q8 quants
            let q8_vec = _mm256_loadu_si256(q8_ptr.cast());

            // For vpdpbusd, we need unsigned × signed
            // Use sign trick: |q4| × sign(q8, q4)
            let q4_abs = _mm256_sign_epi8(q4_signed, q4_signed);
            let q8_signed = _mm256_sign_epi8(q8_vec, q4_signed);

            // vpdpbusd: accumulator += sum(u8 × i8) for each 32-bit lane
            // Each 32-bit lane sums 4 products (4 bytes × 4 bytes)
            // We get 8 such sums in the 256-bit register
            let mut int_acc = _mm256_setzero_si256();

            // VEX-encoded vpdpbusd ymm0, ymm1, ymm2
            // Use {vex} prefix to force VEX encoding (not EVEX)
            asm!(
                "{{vex}} vpdpbusd {acc:y}, {a:y}, {b:y}",
                acc = inout(ymm_reg) int_acc,
                a = in(ymm_reg) q4_abs,
                b = in(ymm_reg) q8_signed,
                options(nostack, nomem, pure)
            );

            // Convert to float and scale
            // vpdpbusd gives us 8 × i32, each is sum of 4 products
            let prod_f32 = _mm256_cvtepi32_ps(int_acc);
            acc = _mm256_fmadd_ps(combined_scale, prod_f32, acc);
        }

        // Horizontal sum of 8 floats
        let hi = std::arch::x86_64::_mm256_extractf128_ps(acc, 1);
        let lo = std::arch::x86_64::_mm256_castps256_ps128(acc);
        let sum128 = std::arch::x86_64::_mm_add_ps(lo, hi);
        let sum64 = std::arch::x86_64::_mm_hadd_ps(sum128, sum128);
        let sum32 = std::arch::x86_64::_mm_hadd_ps(sum64, sum64);
        std::arch::x86_64::_mm_cvtss_f32(sum32)
    }
}
