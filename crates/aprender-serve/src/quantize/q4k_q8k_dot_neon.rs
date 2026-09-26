// #2880: aarch64 NEON Q4_K × Q8_K dot product.
//
// Before this kernel, `fused_q4k_q8k_dot_simd` had only x86 arms, so every
// aarch64 host (Apple M-series, GB10 Grace) ran the scalar
// `fused_q4k_q8k_dot` in the decode matvec. Baseline NEON only (no dotprod):
// NEON is mandatory on aarch64, so no runtime detection is needed.
//
// The integer sums are exact and the f32 accumulation follows the scalar
// kernel's order, so the result is bitwise identical to `fused_q4k_q8k_dot`.

/// NEON Q4_K × Q8_K dot product; same contract as [`fused_q4k_q8k_dot`].
#[cfg(target_arch = "aarch64")]
pub fn fused_q4k_q8k_dot_neon(
    q4k_data: &[u8],
    q8k_scales: &[f32],
    q8k_quants: &[i8],
) -> Result<f32> {
    const SUPER_BLOCK_BYTES: usize = 144;
    let num_super_blocks = q4k_data.len() / SUPER_BLOCK_BYTES;
    if !q4k_data.len().is_multiple_of(SUPER_BLOCK_BYTES)
        || q8k_scales.len() < num_super_blocks
        || q8k_quants.len() < num_super_blocks * QK_K
    {
        // The scalar kernel owns the error messages.
        return fused_q4k_q8k_dot(q4k_data, q8k_scales, q8k_quants);
    }
    // SAFETY: NEON is mandatory on aarch64; the lengths were checked above, so
    // every 16-byte load below stays inside q4k_data / q8k_quants.
    Ok(unsafe { q4k_q8k_dot_neon_unchecked(q4k_data, q8k_scales, q8k_quants, num_super_blocks) })
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn q4k_q8k_dot_neon_unchecked(
    q4k_data: &[u8],
    q8k_scales: &[f32],
    q8k_quants: &[i8],
    num_super_blocks: usize,
) -> f32 {
    use std::arch::aarch64::*;
    const SUPER_BLOCK_BYTES: usize = 144;

    let mask = vdupq_n_u8(0x0F);
    let mut total_acc = 0.0f32;

    for sb_idx in 0..num_super_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let q8_start = sb_idx * QK_K;
        let d = read_f16(&q4k_data[sb_start..sb_start + 2]);
        let dmin = read_f16(&q4k_data[sb_start + 2..sb_start + 4]);
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&q4k_data[sb_start + 4..sb_start + 16]);
        let q8_scale = q8k_scales[sb_idx];

        for j in (0..QK_K).step_by(64) {
            let qp = q4k_data.as_ptr().add(sb_start + 16 + j / 2);
            let ap = q8k_quants.as_ptr().add(q8_start + j);

            let mut sum_lo = vdupq_n_s32(0);
            let mut sum_hi = vdupq_n_s32(0);
            let mut q8_lo = vdupq_n_s32(0);
            let mut q8_hi = vdupq_n_s32(0);
            for h in 0..2 {
                let q = vld1q_u8(qp.add(h * 16));
                let lo = vreinterpretq_s8_u8(vandq_u8(q, mask));
                let hi = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q));
                let a_lo = vld1q_s8(ap.add(h * 16));
                let a_hi = vld1q_s8(ap.add(32 + h * 16));

                // |q4 * q8| <= 15 * 128, so each i16 product is exact.
                sum_lo = vpadalq_s16(sum_lo, vmull_s8(vget_low_s8(lo), vget_low_s8(a_lo)));
                sum_lo = vpadalq_s16(sum_lo, vmull_high_s8(lo, a_lo));
                sum_hi = vpadalq_s16(sum_hi, vmull_s8(vget_low_s8(hi), vget_low_s8(a_hi)));
                sum_hi = vpadalq_s16(sum_hi, vmull_high_s8(hi, a_hi));
                q8_lo = vpadalq_s16(q8_lo, vpaddlq_s8(a_lo));
                q8_hi = vpadalq_s16(q8_hi, vpaddlq_s8(a_hi));
            }

            let is = j / 32;
            let (sc1, m1) = extract_scale_min(&scales, is);
            let (sc2, m2) = extract_scale_min(&scales, is + 1);
            let d_sc1_q8 = d * sc1 * q8_scale;
            let dm1_q8 = dmin * m1 * q8_scale;
            let d_sc2_q8 = d * sc2 * q8_scale;
            let dm2_q8 = dmin * m2 * q8_scale;

            total_acc += d_sc1_q8 * (vaddvq_s32(sum_lo) as f32) - dm1_q8 * (vaddvq_s32(q8_lo) as f32);
            total_acc += d_sc2_q8 * (vaddvq_s32(sum_hi) as f32) - dm2_q8 * (vaddvq_s32(q8_hi) as f32);
        }
    }
    total_acc
}
