//! NEON fused dequant+dot kernels for Q4_K and Q6_K on aarch64 (0.73 R3)
//!
//! Before these kernels, every aarch64 host (gx10, mini) ran the scalar
//! `fused_q4k_dot` / `fused_q6k_dot`: the dispatchers only had an x86 arm.
//! NEON is part of the aarch64 baseline, so no runtime detection is needed.
//!
//! The scalar kernels are the oracles. Contract: `contracts/neon-q4k-q6k-v1.yaml`
//! (parity |neon - scalar| < 1e-3 * max(1, |scalar|)).

#[allow(clippy::wildcard_imports)]
use std::arch::aarch64::*;

use super::dequant::read_f16;
use super::simd::extract_scale_min;
use super::types::QK_K;
use crate::error::{RealizarError, Result};

const Q4K_SUPER_BLOCK_BYTES: usize = 144;
const Q6K_SUPER_BLOCK_BYTES: usize = 210;

/// Same shape checks and messages as the scalar kernels; returns the super-block count.
fn validate(data: &[u8], activations: &[f32], block_bytes: usize, name: &str) -> Result<usize> {
    if !data.len().is_multiple_of(block_bytes) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "{name} data length {} is not a multiple of super-block size {block_bytes}",
                data.len()
            ),
        });
    }
    let num_super_blocks = data.len() / block_bytes;
    let expected_values = num_super_blocks * QK_K;
    if activations.len() != expected_values {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Activation length {} doesn't match {name} values count {expected_values}",
                activations.len()
            ),
        });
    }
    Ok(num_super_blocks)
}

/// Returns (Σ w_i·x_i, Σ x_i) over 16 lanes, with `w` the unsigned bytes of `v`.
///
/// # Safety
/// `x` must point at 16 readable f32 values.
#[inline]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dot_u8x16(v: uint8x16_t, x: *const f32) -> (float32x4_t, float32x4_t) {
    let lo = vmovl_u8(vget_low_u8(v));
    let hi = vmovl_high_u8(v);
    let w0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo)));
    let w1 = vcvtq_f32_u32(vmovl_high_u16(lo));
    let w2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi)));
    let w3 = vcvtq_f32_u32(vmovl_high_u16(hi));
    // SAFETY: caller guarantees x[0..16] is in bounds.
    let x0 = vld1q_f32(x);
    let x1 = vld1q_f32(x.add(4));
    let x2 = vld1q_f32(x.add(8));
    let x3 = vld1q_f32(x.add(12));
    let p = vaddq_f32(
        vfmaq_f32(vmulq_f32(w0, x0), w1, x1),
        vfmaq_f32(vmulq_f32(w2, x2), w3, x3),
    );
    let s = vaddq_f32(vaddq_f32(x0, x1), vaddq_f32(x2, x3));
    (p, s)
}

/// Returns Σ w_i·x_i over 16 lanes, with `w` the signed bytes of `v`.
///
/// # Safety
/// `x` must point at 16 readable f32 values.
#[inline]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dot_i8x16(v: int8x16_t, x: *const f32) -> float32x4_t {
    let lo = vmovl_s8(vget_low_s8(v));
    let hi = vmovl_high_s8(v);
    let w0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo)));
    let w1 = vcvtq_f32_s32(vmovl_high_s16(lo));
    let w2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi)));
    let w3 = vcvtq_f32_s32(vmovl_high_s16(hi));
    // SAFETY: caller guarantees x[0..16] is in bounds.
    let x0 = vld1q_f32(x);
    let x1 = vld1q_f32(x.add(4));
    let x2 = vld1q_f32(x.add(8));
    let x3 = vld1q_f32(x.add(12));
    vaddq_f32(
        vfmaq_f32(vmulq_f32(w0, x0), w1, x1),
        vfmaq_f32(vmulq_f32(w2, x2), w3, x3),
    )
}

/// NEON fused Q4_K dequant+dot. Same layout and result as `fused_q4k_dot`.
///
/// Per 32-value sub-block, Σ(d1·q − dm1)·x = d1·Σq·x − dm1·Σx, so the kernel
/// accumulates Σq·x and Σx in vectors and applies the two scales once.
///
/// # Safety
/// aarch64 only; NEON is baseline there. Every pointer read is bounded by
/// `validate`: `activations.len() == num_super_blocks * 256`.
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn fused_q4k_dot_neon(q4k_data: &[u8], activations: &[f32]) -> Result<f32> {
    let num_super_blocks = validate(q4k_data, activations, Q4K_SUPER_BLOCK_BYTES, "Q4_K")?;
    let mask = vdupq_n_u8(0x0F);
    let mut acc = vdupq_n_f32(0.0);

    for sb_idx in 0..num_super_blocks {
        let sb = &q4k_data[sb_idx * Q4K_SUPER_BLOCK_BYTES..(sb_idx + 1) * Q4K_SUPER_BLOCK_BYTES];
        let d = read_f16(&sb[0..2]);
        let dmin = read_f16(&sb[2..4]);
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&sb[4..16]);
        let qs = &sb[16..144];
        let act = &activations[sb_idx * QK_K..(sb_idx + 1) * QK_K];

        for j in (0..QK_K).step_by(64) {
            let is = j / 32;
            let (sc1, m1) = extract_scale_min(&scales, is);
            let (sc2, m2) = extract_scale_min(&scales, is + 1);

            // SAFETY: qs has 128 bytes and j/2 + 32 <= 128; act has 256 values and
            // j + 64 <= 256, so all four 16-lane reads below are in bounds.
            let q0 = vld1q_u8(qs.as_ptr().add(j / 2));
            let q1 = vld1q_u8(qs.as_ptr().add(j / 2 + 16));
            let x = act.as_ptr().add(j);

            // Low nibbles -> activations [j, j+32), scales (sc1, m1)
            let (p0, s0) = dot_u8x16(vandq_u8(q0, mask), x);
            let (p1, s1) = dot_u8x16(vandq_u8(q1, mask), x.add(16));
            acc = vfmaq_f32(acc, vaddq_f32(p0, p1), vdupq_n_f32(d * sc1));
            acc = vfmsq_f32(acc, vaddq_f32(s0, s1), vdupq_n_f32(dmin * m1));

            // High nibbles -> activations [j+32, j+64), scales (sc2, m2)
            let (p2, s2) = dot_u8x16(vshrq_n_u8::<4>(q0), x.add(32));
            let (p3, s3) = dot_u8x16(vshrq_n_u8::<4>(q1), x.add(48));
            acc = vfmaq_f32(acc, vaddq_f32(p2, p3), vdupq_n_f32(d * sc2));
            acc = vfmsq_f32(acc, vaddq_f32(s2, s3), vdupq_n_f32(dmin * m2));
        }
    }
    Ok(vaddvq_f32(acc))
}

/// Decodes 16 Q6_K values: `(nibble | high2 << 4) - 32`, as signed bytes.
///
/// The value is 0..=63 before the bias, so a wrapping u8 subtract of 32
/// reinterpreted as i8 lands exactly in -32..=31.
#[inline]
#[target_feature(enable = "neon")]
fn q6k_decode(nibble: uint8x16_t, high2: uint8x16_t) -> int8x16_t {
    let q = vorrq_u8(nibble, vshlq_n_u8::<4>(high2));
    vreinterpretq_s8_u8(vsubq_u8(q, vdupq_n_u8(32)))
}

/// NEON fused Q6_K dequant+dot. Same layout and result as `fused_q6k_dot`.
///
/// Layout per super-block: ql[128] @0, qh[64] @128, scales i8[16] @192, d f16 @208.
///
/// # Safety
/// aarch64 only; NEON is baseline there. Every pointer read is bounded by
/// `validate`: `activations.len() == num_super_blocks * 256`.
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn fused_q6k_dot_neon(q6k_data: &[u8], activations: &[f32]) -> Result<f32> {
    let num_super_blocks = validate(q6k_data, activations, Q6K_SUPER_BLOCK_BYTES, "Q6_K")?;
    let low4 = vdupq_n_u8(0x0F);
    let low2 = vdupq_n_u8(0x03);
    let mut acc = vdupq_n_f32(0.0);

    for sb_idx in 0..num_super_blocks {
        let sb = &q6k_data[sb_idx * Q6K_SUPER_BLOCK_BYTES..(sb_idx + 1) * Q6K_SUPER_BLOCK_BYTES];
        let d = read_f16(&sb[208..210]);
        let act = &activations[sb_idx * QK_K..(sb_idx + 1) * QK_K];

        for half in 0..2 {
            let ql = &sb[64 * half..64 * half + 64];
            let qh = &sb[128 + 32 * half..128 + 32 * half + 32];
            let scales = &sb[192 + 8 * half..192 + 8 * half + 8];
            #[allow(clippy::cast_possible_wrap)]
            let dsc = |k: usize| vdupq_n_f32(d * f32::from(scales[k] as i8));
            let n = 128 * half;

            // l in [16g, 16g+16) uses scale selector is = g
            for g in 0..2 {
                let l = 16 * g;
                // SAFETY: ql has 64 bytes (l + 32 + 16 <= 64), qh has 32 (l + 16 <= 32),
                // and act has 256 values (n + l + 96 + 16 <= 256).
                let a = vld1q_u8(ql.as_ptr().add(l));
                let b = vld1q_u8(ql.as_ptr().add(l + 32));
                let h = vld1q_u8(qh.as_ptr().add(l));
                let x = act.as_ptr().add(n + l);

                let q1 = q6k_decode(vandq_u8(a, low4), vandq_u8(h, low2));
                let q2 = q6k_decode(vandq_u8(b, low4), vandq_u8(vshrq_n_u8::<2>(h), low2));
                let q3 = q6k_decode(vshrq_n_u8::<4>(a), vandq_u8(vshrq_n_u8::<4>(h), low2));
                let q4 = q6k_decode(vshrq_n_u8::<4>(b), vshrq_n_u8::<6>(h));

                acc = vfmaq_f32(acc, dot_i8x16(q1, x), dsc(g));
                acc = vfmaq_f32(acc, dot_i8x16(q2, x.add(32)), dsc(g + 2));
                acc = vfmaq_f32(acc, dot_i8x16(q3, x.add(64)), dsc(g + 4));
                acc = vfmaq_f32(acc, dot_i8x16(q4, x.add(96)), dsc(g + 6));
            }
        }
    }
    Ok(vaddvq_f32(acc))
}

/// Integer sums for one 64-value Q4_K chunk against Q8_K quants:
/// (Σ lo·q8[0..32], Σ q8[0..32], Σ hi·q8[32..64], Σ q8[32..64]).
///
/// # Safety
/// `q` must point at 32 readable bytes and `a` at 64 readable i8 values.
#[inline]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn q4k_q8k_chunk_widen(q: *const u8, a: *const i8) -> (i32, i32, i32, i32) {
    let mask = vdupq_n_u8(0x0F);
    let (q0, q1) = (vld1q_u8(q), vld1q_u8(q.add(16)));
    let (a0, a1, a2, a3) = (
        vld1q_s8(a),
        vld1q_s8(a.add(16)),
        vld1q_s8(a.add(32)),
        vld1q_s8(a.add(48)),
    );
    // Nibbles are 0..=15, so they are exact as i8; |nibble·q8| ≤ 15·128 fits i16.
    let mul = |w: int8x16_t, x: int8x16_t, acc: int32x4_t| {
        let lo = vmull_s8(vget_low_s8(w), vget_low_s8(x));
        let hi = vmull_high_s8(w, x);
        vpadalq_s16(vpadalq_s16(acc, lo), hi)
    };
    let s = |w: uint8x16_t| vreinterpretq_s8_u8(w);
    let zero = vdupq_n_s32(0);
    let lo = mul(
        s(vandq_u8(q1, mask)),
        a1,
        mul(s(vandq_u8(q0, mask)), a0, zero),
    );
    let hi = mul(
        s(vshrq_n_u8::<4>(q1)),
        a3,
        mul(s(vshrq_n_u8::<4>(q0)), a2, zero),
    );
    let qsum = |x: int8x16_t, y: int8x16_t| vaddlvq_s16(vaddq_s16(vpaddlq_s8(x), vpaddlq_s8(y)));
    (vaddvq_s32(lo), qsum(a0, a1), vaddvq_s32(hi), qsum(a2, a3))
}

/// `acc + Σ_4 a·b` per i32 lane: one SDOT instruction.
///
/// `vdotq_s32` is still unstable (`stdarch_neon_dotprod`) on the pinned toolchain,
/// so this is the instruction itself. Pure register op: no memory, no stack.
///
/// # Safety
/// The CPU must have the `dotprod` feature.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn sdot(mut acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    // SAFETY: register-only instruction; dotprod is guaranteed by the caller.
    unsafe {
        std::arch::asm!(
            "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) acc,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    acc
}

/// Same as `q4k_q8k_chunk_widen`, with the products done by SDOT.
///
/// # Safety
/// Requires the `dotprod` feature (checked by the caller) plus the bounds of
/// `q4k_q8k_chunk_widen`.
#[inline]
#[target_feature(enable = "neon,dotprod")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn q4k_q8k_chunk_sdot(q: *const u8, a: *const i8) -> (i32, i32, i32, i32) {
    let mask = vdupq_n_u8(0x0F);
    let (q0, q1) = (vld1q_u8(q), vld1q_u8(q.add(16)));
    let (a0, a1, a2, a3) = (
        vld1q_s8(a),
        vld1q_s8(a.add(16)),
        vld1q_s8(a.add(32)),
        vld1q_s8(a.add(48)),
    );
    let s = |w: uint8x16_t| vreinterpretq_s8_u8(w);
    let zero = vdupq_n_s32(0);
    let lo = sdot(
        sdot(zero, s(vandq_u8(q0, mask)), a0),
        s(vandq_u8(q1, mask)),
        a1,
    );
    let hi = sdot(
        sdot(zero, s(vshrq_n_u8::<4>(q0)), a2),
        s(vshrq_n_u8::<4>(q1)),
        a3,
    );
    let qsum = |x: int8x16_t, y: int8x16_t| vaddlvq_s16(vaddq_s16(vpaddlq_s8(x), vpaddlq_s8(y)));
    (vaddvq_s32(lo), qsum(a0, a1), vaddvq_s32(hi), qsum(a2, a3))
}

/// Stamps a Q4_K × Q8_K kernel around a chunk function. The float combine is
/// the scalar `fused_q4k_q8k_dot`'s, term for term; the integer sums are exact,
/// so the result matches the scalar oracle bit for bit.
macro_rules! q4k_q8k_neon_kernel {
    ($name:ident, $chunk:ident, $features:literal) => {
        /// NEON Q4_K × Q8_K dot. Same checks and result as `fused_q4k_q8k_dot`.
        ///
        /// # Safety
        /// The target features named in the attribute must be present. Reads are
        /// bounded by the length checks below.
        #[target_feature(enable = $features)]
        #[allow(unsafe_op_in_unsafe_fn)]
        pub(crate) unsafe fn $name(
            q4k_data: &[u8],
            q8k_scales: &[f32],
            q8k_quants: &[i8],
        ) -> Result<f32> {
            if !q4k_data.len().is_multiple_of(Q4K_SUPER_BLOCK_BYTES) {
                return Err(RealizarError::InvalidShape {
                    reason: format!(
                        "Q4_K data length {} is not a multiple of {}",
                        q4k_data.len(),
                        Q4K_SUPER_BLOCK_BYTES
                    ),
                });
            }
            let num_super_blocks = q4k_data.len() / Q4K_SUPER_BLOCK_BYTES;
            if q8k_scales.len() < num_super_blocks {
                return Err(RealizarError::InvalidShape {
                    reason: format!(
                        "Q8_K scales count {} < expected {}",
                        q8k_scales.len(),
                        num_super_blocks
                    ),
                });
            }
            if q8k_quants.len() < num_super_blocks * QK_K {
                return Err(RealizarError::InvalidShape {
                    reason: format!(
                        "Q8_K quants count {} < expected {}",
                        q8k_quants.len(),
                        num_super_blocks * QK_K
                    ),
                });
            }

            let mut total_acc = 0.0f32;
            for sb_idx in 0..num_super_blocks {
                let sb =
                    &q4k_data[sb_idx * Q4K_SUPER_BLOCK_BYTES..(sb_idx + 1) * Q4K_SUPER_BLOCK_BYTES];
                let d = read_f16(&sb[0..2]);
                let dmin = read_f16(&sb[2..4]);
                let mut scales = [0u8; 12];
                scales.copy_from_slice(&sb[4..16]);
                let q8_scale = q8k_scales[sb_idx];
                let q8 = &q8k_quants[sb_idx * QK_K..(sb_idx + 1) * QK_K];

                for j in (0..QK_K).step_by(64) {
                    let is = j / 32;
                    let (sc1, m1) = extract_scale_min(&scales, is);
                    let (sc2, m2) = extract_scale_min(&scales, is + 1);
                    // SAFETY: sb has 144 bytes and 16 + j/2 + 32 <= 144; q8 has 256
                    // values and j + 64 <= 256.
                    let (sum_lo, q8_sum_lo, sum_hi, q8_sum_hi) =
                        $chunk(sb.as_ptr().add(16 + j / 2), q8.as_ptr().add(j));
                    total_acc += d * sc1 * q8_scale * (sum_lo as f32)
                        - dmin * m1 * q8_scale * (q8_sum_lo as f32);
                    total_acc += d * sc2 * q8_scale * (sum_hi as f32)
                        - dmin * m2 * q8_scale * (q8_sum_hi as f32);
                }
            }
            Ok(total_acc)
        }
    };
}

q4k_q8k_neon_kernel!(fused_q4k_q8k_dot_neon_widen, q4k_q8k_chunk_widen, "neon");
q4k_q8k_neon_kernel!(
    fused_q4k_q8k_dot_neon_sdot,
    q4k_q8k_chunk_sdot,
    "neon,dotprod"
);
