// #2880: aarch64 NEON Q4_K × f32 dot product.
//
// Before this kernel, `fused_q4k_dot_simd` had only an AVX2 arm, so every
// aarch64 host ran the scalar `fused_q4k_dot`. Q4_K layout (PAR-001, as the
// scalar kernel reads it): each 64-value chunk j takes 32 bytes
// qs[j/2 .. j/2+32]; the low nibbles are values j..j+32 (sub-block j/32), the
// high nibbles values j+32..j+64 (sub-block j/32+1).
//
// Each sub-block contributes d·sc·Σ(q·a) − dmin·m·Σa, the scalar
// Σ(d·sc·q − dmin·m)·a up to f32 summation order; the tests compare both
// against an f64 reference.

/// NEON Q4_K × f32 dot; same contract as [`fused_q4k_dot`].
#[cfg(target_arch = "aarch64")]
pub fn fused_q4k_dot_neon(q4k_data: &[u8], activations: &[f32]) -> Result<f32> {
    const SUPER_BLOCK_BYTES: usize = 144;
    if !q4k_data.len().is_multiple_of(SUPER_BLOCK_BYTES)
        || activations.len() != q4k_data.len() / SUPER_BLOCK_BYTES * QK_K
    {
        // The scalar kernel owns the error messages.
        return fused_q4k_dot(q4k_data, activations);
    }
    // SAFETY: NEON is mandatory on aarch64; the lengths were checked above, so
    // every load below stays inside its superblock / activation chunk.
    Ok(unsafe { q4k_dot_neon_unchecked(q4k_data, activations) })
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn q4k_dot_neon_unchecked(q4k_data: &[u8], activations: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    const SUPER_BLOCK_BYTES: usize = 144;

    /// Σ(q·a) and Σa over 16 quants against 16 consecutive activations.
    #[inline]
    unsafe fn dot16(
        q: uint8x16_t,
        ap: *const f32,
        qa: float32x4_t,
        asum: float32x4_t,
    ) -> (float32x4_t, float32x4_t) {
        let w = [vmovl_u8(vget_low_u8(q)), vmovl_high_u8(q)];
        let (mut qa, mut asum) = (qa, asum);
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
        (qa, asum)
    }

    let low4 = vdupq_n_u8(0x0F);
    let zero = vdupq_n_f32(0.0);
    let mut acc = zero;
    for (sb, act) in q4k_data
        .as_chunks::<SUPER_BLOCK_BYTES>()
        .0
        .iter()
        .zip(activations.as_chunks::<QK_K>().0.iter())
    {
        let d = read_f16(&sb[0..2]);
        let dmin = read_f16(&sb[2..4]);
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&sb[4..16]);
        for j in (0..QK_K).step_by(64) {
            let qp = sb.as_ptr().add(16 + j / 2);
            let ap = act.as_ptr().add(j);
            let (mut qa_lo, mut as_lo, mut qa_hi, mut as_hi) = (zero, zero, zero, zero);
            for h in 0..2 {
                let q = vld1q_u8(qp.add(h * 16));
                (qa_lo, as_lo) = dot16(vandq_u8(q, low4), ap.add(h * 16), qa_lo, as_lo);
                (qa_hi, as_hi) = dot16(vshrq_n_u8::<4>(q), ap.add(32 + h * 16), qa_hi, as_hi);
            }
            let is = j / 32;
            let (sc1, m1) = extract_scale_min(&scales, is);
            let (sc2, m2) = extract_scale_min(&scales, is + 1);
            acc = vfmaq_f32(acc, qa_lo, vdupq_n_f32(d * sc1));
            acc = vfmsq_f32(acc, as_lo, vdupq_n_f32(dmin * m1));
            acc = vfmaq_f32(acc, qa_hi, vdupq_n_f32(d * sc2));
            acc = vfmsq_f32(acc, as_hi, vdupq_n_f32(dmin * m2));
        }
    }
    vaddvq_f32(acc)
}
