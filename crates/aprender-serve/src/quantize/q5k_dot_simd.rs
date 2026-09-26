// #2880: SIMD Q5_K × f32 dot products (x86_64 AVX2+FMA, aarch64 NEON).
//
// Before these kernels, `fused_q5k_dot_simd` was the scalar `fused_q5k_dot` on
// every host. Q5_K superblock (176 B): d f16, dmin f16, 12 packed scale/min
// bytes, qh[32], qs[128]. Sub-block `sub` (32 values) takes the low (even sub)
// or high (odd sub) nibble of qs[(sub/2)*32 ..][l] plus bit `sub` of qh[l]
// as bit 4, the order `for_each_q5k_value` reads (FALSIFY-QDOT-007).
//
// Each sub-block contributes d·sc·Σ(q·a) − dmin·m·Σa, which equals the scalar
// Σ(d·sc·q − dmin·m)·a up to f32 summation order; the tests compare both
// against an f64 reference.

/// AVX2+FMA Q5_K × f32 dot; same contract as [`fused_q5k_dot`].
///
/// # Safety
///
/// The caller must have verified that the CPU supports `avx2` and `fma`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn fused_q5k_dot_avx2(q5k_data: &[u8], activations: &[f32]) -> Result<f32> {
    use std::arch::x86_64::*;
    const SUPER_BLOCK_BYTES: usize = 176;
    if !q5k_data.len().is_multiple_of(SUPER_BLOCK_BYTES)
        || activations.len() != q5k_data.len() / SUPER_BLOCK_BYTES * QK_K
    {
        // The scalar kernel owns the error messages.
        return fused_q5k_dot(q5k_data, activations);
    }

    let low4 = _mm256_set1_epi8(0x0F);
    let one = _mm256_set1_epi8(1);
    let mut acc = _mm256_setzero_ps();
    for (sb, act) in q5k_data
        .as_chunks::<SUPER_BLOCK_BYTES>()
        .0
        .iter()
        .zip(activations.as_chunks::<QK_K>().0.iter())
    {
        let d = read_f16(&sb[0..2]);
        let dmin = read_f16(&sb[2..4]);
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&sb[4..16]);
        // SAFETY: sb is 176 bytes; qh is bytes 16..48, qs 48..176.
        let qh = _mm256_loadu_si256(sb.as_ptr().add(16).cast());
        for sub in 0..8 {
            let q = _mm256_loadu_si256(sb.as_ptr().add(48 + (sub / 2) * 32).cast());
            let nib = if sub % 2 == 0 {
                _mm256_and_si256(q, low4)
            } else {
                _mm256_and_si256(_mm256_srli_epi16::<4>(q), low4)
            };
            // Bit `sub` of every qh byte: a 16-bit lane shift by sub < 8 keeps
            // each byte's own bit `sub` at bit 0 of that byte.
            let hb = _mm256_and_si256(_mm256_srl_epi16(qh, _mm_cvtsi32_si128(sub as i32)), one);
            let q5 = _mm256_or_si256(nib, _mm256_slli_epi16::<4>(hb));

            let ap = act.as_ptr().add(sub * 32);
            let halves = [_mm256_castsi256_si128(q5), _mm256_extracti128_si256::<1>(q5)];
            let mut qa = _mm256_setzero_ps();
            let mut asum = _mm256_setzero_ps();
            for (g, bytes) in [
                halves[0],
                _mm_srli_si128::<8>(halves[0]),
                halves[1],
                _mm_srli_si128::<8>(halves[1]),
            ]
            .into_iter()
            .enumerate()
            {
                let qf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes));
                // SAFETY: act is QK_K = 256 floats; sub*32 + g*8 + 8 <= 256.
                let a = _mm256_loadu_ps(ap.add(g * 8));
                qa = _mm256_fmadd_ps(qf, a, qa);
                asum = _mm256_add_ps(asum, a);
            }
            let (sc, m) = extract_scale_min(&scales, sub);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d * sc), qa, acc);
            acc = _mm256_fnmadd_ps(_mm256_set1_ps(dmin * m), asum, acc);
        }
    }
    let s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps::<1>(acc));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps::<1>(s, s));
    Ok(_mm_cvtss_f32(s))
}

/// NEON Q5_K × f32 dot; same contract as [`fused_q5k_dot`].
#[cfg(target_arch = "aarch64")]
pub fn fused_q5k_dot_neon(q5k_data: &[u8], activations: &[f32]) -> Result<f32> {
    const SUPER_BLOCK_BYTES: usize = 176;
    if !q5k_data.len().is_multiple_of(SUPER_BLOCK_BYTES)
        || activations.len() != q5k_data.len() / SUPER_BLOCK_BYTES * QK_K
    {
        // The scalar kernel owns the error messages.
        return fused_q5k_dot(q5k_data, activations);
    }
    // SAFETY: NEON is mandatory on aarch64; the lengths were checked above, so
    // every load below stays inside its superblock / activation chunk.
    Ok(unsafe { q5k_dot_neon_unchecked(q5k_data, activations) })
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn q5k_dot_neon_unchecked(q5k_data: &[u8], activations: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    const SUPER_BLOCK_BYTES: usize = 176;

    let low4 = vdupq_n_u8(0x0F);
    let one = vdupq_n_u8(1);
    let mut acc = vdupq_n_f32(0.0);
    for (sb, act) in q5k_data
        .as_chunks::<SUPER_BLOCK_BYTES>()
        .0
        .iter()
        .zip(activations.as_chunks::<QK_K>().0.iter())
    {
        let d = read_f16(&sb[0..2]);
        let dmin = read_f16(&sb[2..4]);
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&sb[4..16]);
        let p = sb.as_ptr();
        let qh = [vld1q_u8(p.add(16)), vld1q_u8(p.add(32))];
        for sub in 0..8 {
            let shift = vdupq_n_s8(-(sub as i8));
            let mut qa = vdupq_n_f32(0.0);
            let mut asum = vdupq_n_f32(0.0);
            for h in 0..2 {
                let q = vld1q_u8(p.add(48 + (sub / 2) * 32 + h * 16));
                let nib = if sub % 2 == 0 {
                    vandq_u8(q, low4)
                } else {
                    vshrq_n_u8::<4>(q)
                };
                let hb = vandq_u8(vshlq_u8(qh[h], shift), one);
                let q5 = vorrq_u8(nib, vshlq_n_u8::<4>(hb));
                let w = [vmovl_u8(vget_low_u8(q5)), vmovl_high_u8(q5)];
                let ap = act.as_ptr().add(sub * 32 + h * 16);
                for (k, qf) in [
                    vmovl_u16(vget_low_u16(w[0])),
                    vmovl_high_u16(w[0]),
                    vmovl_u16(vget_low_u16(w[1])),
                    vmovl_high_u16(w[1]),
                ]
                .into_iter()
                .enumerate()
                {
                    let a = vld1q_f32(ap.add(k * 4));
                    qa = vfmaq_f32(qa, vcvtq_f32_u32(qf), a);
                    asum = vaddq_f32(asum, a);
                }
            }
            let (sc, m) = extract_scale_min(&scales, sub);
            acc = vfmaq_f32(acc, qa, vdupq_n_f32(d * sc));
            acc = vfmsq_f32(acc, asum, vdupq_n_f32(dmin * m));
        }
    }
    vaddvq_f32(acc)
}
