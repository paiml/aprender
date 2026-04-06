// PMAT-301: ggml-style Q4K x Q8K dot product (scale-shuffle-accumulate).
//
// Key differences from q4_q8_dot_avx2.rs:
// 1. Scale applied as i16 multiply (madd_epi16) in integer path, NOT as f32 post-hoc
// 2. Partial sums accumulated across ALL blocks in SIMD registers (no per-block hsum)
// 3. ONE horizontal sum at the very end
// 4. Pre-computed bsums for min correction (4 instructions vs ~40)
//
// This eliminates 8 hsum_epi32 calls per super-block (240 instructions per row).
// Reference: ggml ggml_vec_dot_q4_K_q8_K in arch/x86/quants.c lines 1760-1823.

// Scale shuffle table from ggml (get_scale_shuffle_k4).
// Replicates i16 scale values across 32-byte YMM register for madd_epi16.
// Index i selects the 32 bytes for shuffle iteration i.
// The pattern: index 2*j broadcasts scale[j] as i16 pairs across all positions.
#[cfg(target_arch = "x86_64")]
static K_SCALE_SHUFFLE_K4: [u8; 256] = [
     0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
     4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
     8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
    12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
];

/// PMAT-301: ggml-style single-row Q4K x Q8K dot product.
///
/// Processes ALL super-blocks for one output row. Accumulates scale-weighted
/// partial sums in SIMD registers with ONE horizontal sum at the end.
///
/// Returns the final dot product as f32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
/// PMAT-306: Raw pointer variant — zero slice overhead in the hot loop.
pub(crate) unsafe fn ggml_style_q4k_q8k_dot_avx2(
    weight_row: *const u8,
    q8k_scales: &[f32],
    q8k_quants: &[i8],
    q8k_bsums: &[i16],
    num_super_blocks: usize,
) -> f32 {
    // SAFETY: Caller guarantees all pointers are valid and aligned for the given
    // num_super_blocks count. The raw variant requires AVX2+FMA (enforced by
    // #[target_feature]).
    unsafe {
        ggml_style_q4k_q8k_dot_avx2_raw(
            weight_row,
            q8k_scales.as_ptr(),
            q8k_quants.as_ptr(),
            q8k_bsums.as_ptr(),
            num_super_blocks,
        )
    }
}

/// PMAT-306: Inner loop with raw pointers — no slice indexing, no bounds checks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub(crate) unsafe fn ggml_style_q4k_q8k_dot_avx2_raw(
    weight_row: *const u8,
    q8k_scales_ptr: *const f32,
    q8k_quants_ptr: *const i8,
    q8k_bsums_ptr: *const i16,
    num_super_blocks: usize,
) -> f32 {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    const SB_BYTES: usize = 144;
    const QK_K: usize = 256;

    // SAFETY: All SIMD intrinsics and raw pointer operations below require an
    // unsafe block. The caller guarantees:
    // - weight_row points to at least num_super_blocks * SB_BYTES valid bytes
    // - q8k_scales_ptr, q8k_quants_ptr, q8k_bsums_ptr are valid for the
    //   corresponding super-block counts
    // - AVX2 + FMA are available (enforced by #[target_feature])
    unsafe {
        let m4 = _mm256_set1_epi8(0x0F_i8);
        let kmask1: u32 = 0x3f3f_3f3f;
        let kmask2: u32 = 0x0f0f_0f0f;
        let kmask3: u32 = 0x0303_0303;

        let mut acc = _mm256_setzero_ps();
        let mut acc_m = _mm_setzero_ps();

        for sb in 0..num_super_blocks {
            let sb_ptr = weight_row.add(sb * SB_BYTES);

            let d_raw = (sb_ptr as *const u16).read_unaligned();
            let dmin_raw = (sb_ptr.add(2) as *const u16).read_unaligned();
            // Raw pointer read — no bounds check
            let q8_scale = *q8k_scales_ptr.add(sb);
            let d = q8_scale * f16_to_f32(d_raw);
            let dmin = -q8_scale * f16_to_f32(dmin_raw);

            // Decode packed 6-bit scales (ggml's utmp trick)
            let mut utmp = [0u32; 4];
            std::ptr::copy_nonoverlapping(sb_ptr.add(4), utmp.as_mut_ptr().cast::<u8>(), 12);
            utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
            let uaux = utmp[1] & kmask1;
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[2] = uaux;
            utmp[0] &= kmask1;

            // Build scales+mins vector: 16 x i16 (8 scales in low half, 8 mins in high half)
            let mins_and_scales = _mm256_cvtepu8_epi16(
                _mm_set_epi32(utmp[3] as i32, utmp[2] as i32, utmp[1] as i32, utmp[0] as i32),
            );

            // Min correction via pre-computed bsums (ggml pattern).
            let bsums_ptr = q8k_bsums_ptr.add(sb * 16);
            let q8sums = _mm256_loadu_si256(bsums_ptr.cast::<__m256i>());
            let q8s = _mm_hadd_epi16(
                _mm256_extracti128_si256(q8sums, 0),
                _mm256_extracti128_si256(q8sums, 1),
            );
            let mins_128 = _mm256_extracti128_si256(mins_and_scales, 1);
            let prod = _mm_madd_epi16(mins_128, q8s);
            acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

            // Extract scales for shuffle-broadcast
            let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
            let scales = _mm256_set_m128i(sc128, sc128);

            let mut sumi = _mm256_setzero_si256();

            let qs_ptr = sb_ptr.add(16);
            let q8_ptr = q8k_quants_ptr.add(sb * QK_K);

            // Inner loop: 4 chunks of 64 values each
            for j in 0..4usize {
                let scale_l = _mm256_shuffle_epi8(
                    scales,
                    _mm256_loadu_si256(K_SCALE_SHUFFLE_K4.as_ptr().add(32 * (2 * j)).cast::<__m256i>()),
                );
                let scale_h = _mm256_shuffle_epi8(
                    scales,
                    _mm256_loadu_si256(K_SCALE_SHUFFLE_K4.as_ptr().add(32 * (2 * j + 1)).cast::<__m256i>()),
                );

                let q4bits = _mm256_loadu_si256(qs_ptr.add(j * 32).cast::<__m256i>());
                let q4l = _mm256_and_si256(q4bits, m4);
                let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

                let q8l = _mm256_loadu_si256(q8_ptr.add(j * 64).cast::<__m256i>());
                let mut p16l = _mm256_maddubs_epi16(q4l, q8l);
                p16l = _mm256_madd_epi16(scale_l, p16l); // SCALE BAKED INTO INTEGER PATH

                let q8h = _mm256_loadu_si256(q8_ptr.add(j * 64 + 32).cast::<__m256i>());
                let mut p16h = _mm256_maddubs_epi16(q4h, q8h);
                p16h = _mm256_madd_epi16(scale_h, p16h); // SCALE BAKED INTO INTEGER PATH

                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
            }

            // ONE fmadd per super-block (no per-block hsum!)
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
        }

        // Final horizontal sum (ONCE across all super-blocks)
        let sum128 = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_movehdup_ps(sum64));

        let acc_m_sum = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
        let acc_m_final = _mm_add_ss(acc_m_sum, _mm_movehdup_ps(acc_m_sum));

        _mm_cvtss_f32(sum32) + _mm_cvtss_f32(acc_m_final)
    }
}

/// Convert f16 bits to f32 — inline SIMD, no function call.
/// Convert f16 bits to f32.
/// PMAT-312: Attempted inline F16C and software conversion — both WORSE.
/// The `half` crate already uses F16C via runtime detection and is optimal.
#[inline]
fn f16_to_f32(h: u16) -> f32 {
    half::f16::from_bits(h).to_f32()
}

/// Pre-compute Q8K block sums (bsums) matching ggml's layout.
/// Returns num_superblocks * 16 i16 values (one sum per 16-value sub-block).
/// Each SB has 256 values = 16 sub-blocks of 16 values each.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn precompute_q8k_bsums_i16(
    q8k_quants: &[i8],
    num_super_blocks: usize,
) -> Vec<i16> {
    use std::arch::x86_64::*;

    let mut bsums = vec![0i16; num_super_blocks * 16];

    // SAFETY: All SIMD intrinsics and raw pointer arithmetic below require an
    // unsafe block. The caller guarantees q8k_quants has at least
    // num_super_blocks * 256 elements, and AVX2 is available (enforced by
    // #[target_feature]).
    unsafe {
        let ones_u8 = _mm_set1_epi8(1);

        for sb in 0..num_super_blocks {
            let q8_base = q8k_quants.as_ptr().add(sb * 256);
            // 16 sub-blocks of 16 values each
            for sub in 0..16 {
                let q8 = _mm_loadu_si128(q8_base.add(sub * 16).cast::<__m128i>());
                // maddubs(1, q8) = 8 pairwise sums as i16
                let pair_sums = _mm_maddubs_epi16(ones_u8, q8);
                // hadd to get 4 i16 sums
                let quad = _mm_hadd_epi16(pair_sums, pair_sums);
                // hadd again to get 2 i16
                let duo = _mm_hadd_epi16(quad, quad);
                // hadd once more for 1 i16
                let single = _mm_hadd_epi16(duo, duo);
                bsums[sb * 16 + sub] = _mm_extract_epi16(single, 0) as i16;
            }
        }
    }

    bsums
}

// Tests in parallel_k_fused_q4k.rs
