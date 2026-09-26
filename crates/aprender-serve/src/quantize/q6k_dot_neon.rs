// #2880: aarch64 NEON Q6_K × f32 dot product.
//
// `fused_q6k_dot_simd` had only an AVX2 arm, so on aarch64 every Q6_K decode
// matvec (lm_head and the Q6_K tensors of Q4_K_M models) ran the scalar
// `fused_q6k_dot`. Baseline NEON only: no runtime detection on aarch64.
//
// Each 16-lane group of a 128-value chunk shares one scale (scalar `is =
// l / 16`), so the kernel forms Σ q·a per group in f32 and applies d·scale
// once. The f32 summation order differs from the scalar kernel, so results
// agree to rounding, not bitwise.

/// NEON Q6_K dot product; same contract as [`fused_q6k_dot`].
#[cfg(target_arch = "aarch64")]
pub fn fused_q6k_dot_neon(q6k_data: &[u8], activations: &[f32]) -> Result<f32> {
    const SUPER_BLOCK_BYTES: usize = 210;
    let num_super_blocks = q6k_data.len() / SUPER_BLOCK_BYTES;
    if !q6k_data.len().is_multiple_of(SUPER_BLOCK_BYTES)
        || activations.len() != num_super_blocks * QK_K
    {
        // The scalar kernel owns the error messages.
        return fused_q6k_dot(q6k_data, activations);
    }
    // SAFETY: NEON is mandatory on aarch64; the lengths were checked above, so
    // every load below stays inside q6k_data / activations.
    Ok(unsafe { q6k_dot_neon_unchecked(q6k_data, activations, num_super_blocks) })
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn)]
#[inline(always)]
unsafe fn q6k_group_dot_neon(
    q: std::arch::aarch64::int8x16_t,
    a: *const f32,
) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    let lo = vmovl_s8(vget_low_s8(q));
    let hi = vmovl_high_s8(q);
    let mut acc = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), vld1q_f32(a));
    acc = vfmaq_f32(acc, vcvtq_f32_s32(vmovl_high_s16(lo)), vld1q_f32(a.add(4)));
    acc = vfmaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))), vld1q_f32(a.add(8)));
    vfmaq_f32(acc, vcvtq_f32_s32(vmovl_high_s16(hi)), vld1q_f32(a.add(12)))
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn, clippy::cast_possible_wrap)]
unsafe fn q6k_dot_neon_unchecked(q6k_data: &[u8], activations: &[f32], num_super_blocks: usize) -> f32 {
    use std::arch::aarch64::*;
    const SUPER_BLOCK_BYTES: usize = 210;

    let m4 = vdupq_n_u8(0x0F);
    let m2 = vdupq_n_u8(0x03);
    let bias = vdupq_n_s8(32);
    let mut total = vdupq_n_f32(0.0);

    for sb_idx in 0..num_super_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        // Q6_K layout: ql (128) + qh (64) + scales (16, i8) + d (f16)
        let base = q6k_data.as_ptr().add(sb_start);
        let scales = &q6k_data[sb_start + 192..sb_start + 208];
        let d = read_f16(&q6k_data[sb_start + 208..sb_start + 210]);
        let act = activations.as_ptr().add(sb_idx * QK_K);

        for idx in 0..2 {
            let ql = base.add(64 * idx);
            let qh = base.add(128 + 32 * idx);
            let sc = &scales[8 * idx..];
            let a = act.add(128 * idx);
            for half in 0..2 {
                let l0 = 16 * half;
                let l = vld1q_u8(ql.add(l0));
                let l32 = vld1q_u8(ql.add(l0 + 32));
                let h = vld1q_u8(qh.add(l0));
                let q = [
                    vorrq_u8(vandq_u8(l, m4), vshlq_n_u8::<4>(vandq_u8(h, m2))),
                    vorrq_u8(vandq_u8(l32, m4), vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<2>(h), m2))),
                    vorrq_u8(vshrq_n_u8::<4>(l), vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<4>(h), m2))),
                    vorrq_u8(vshrq_n_u8::<4>(l32), vshlq_n_u8::<4>(vshrq_n_u8::<6>(h))),
                ];
                for (k, qk) in q.into_iter().enumerate() {
                    let qs = vsubq_s8(vreinterpretq_s8_u8(qk), bias);
                    let dot = q6k_group_dot_neon(qs, a.add(l0 + 32 * k));
                    let s = d * f32::from(sc[half + 2 * k] as i8);
                    total = vfmaq_n_f32(total, dot, s);
                }
            }
        }
    }
    vaddvq_f32(total)
}
