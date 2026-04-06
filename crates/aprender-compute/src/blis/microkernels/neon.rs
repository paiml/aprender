//! ARM NEON Microkernel
//!
//! Contains the NEON SIMD microkernel for ARM64 (aarch64) targets.

/// NEON microkernel (8x8 output tile)
#[cfg(target_arch = "aarch64")]
// SAFETY: Caller ensures NEON is available (always on aarch64) and pointers/dims are valid
pub unsafe fn microkernel_8x8_neon(
    k: usize,
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    ldc: usize,
) {
    use std::arch::aarch64::*;

    // Load C into registers (8 columns, split into 2x float32x4)
    let mut c00 = vld1q_f32(c);
    let mut c01 = vld1q_f32(c.add(4));
    let mut c10 = vld1q_f32(c.add(ldc));
    let mut c11 = vld1q_f32(c.add(ldc + 4));
    let mut c20 = vld1q_f32(c.add(2 * ldc));
    let mut c21 = vld1q_f32(c.add(2 * ldc + 4));
    let mut c30 = vld1q_f32(c.add(3 * ldc));
    let mut c31 = vld1q_f32(c.add(3 * ldc + 4));
    let mut c40 = vld1q_f32(c.add(4 * ldc));
    let mut c41 = vld1q_f32(c.add(4 * ldc + 4));
    let mut c50 = vld1q_f32(c.add(5 * ldc));
    let mut c51 = vld1q_f32(c.add(5 * ldc + 4));
    let mut c60 = vld1q_f32(c.add(6 * ldc));
    let mut c61 = vld1q_f32(c.add(6 * ldc + 4));
    let mut c70 = vld1q_f32(c.add(7 * ldc));
    let mut c71 = vld1q_f32(c.add(7 * ldc + 4));

    for p in 0..k {
        let a0 = vld1q_f32(a.add(p * 8));
        let a1 = vld1q_f32(a.add(p * 8 + 4));

        let b0 = vld1q_dup_f32(b.add(p * 8));
        let b1 = vld1q_dup_f32(b.add(p * 8 + 1));
        let b2 = vld1q_dup_f32(b.add(p * 8 + 2));
        let b3 = vld1q_dup_f32(b.add(p * 8 + 3));
        let b4 = vld1q_dup_f32(b.add(p * 8 + 4));
        let b5 = vld1q_dup_f32(b.add(p * 8 + 5));
        let b6 = vld1q_dup_f32(b.add(p * 8 + 6));
        let b7 = vld1q_dup_f32(b.add(p * 8 + 7));

        c00 = vfmaq_f32(c00, a0, b0);
        c01 = vfmaq_f32(c01, a1, b0);
        c10 = vfmaq_f32(c10, a0, b1);
        c11 = vfmaq_f32(c11, a1, b1);
        c20 = vfmaq_f32(c20, a0, b2);
        c21 = vfmaq_f32(c21, a1, b2);
        c30 = vfmaq_f32(c30, a0, b3);
        c31 = vfmaq_f32(c31, a1, b3);
        c40 = vfmaq_f32(c40, a0, b4);
        c41 = vfmaq_f32(c41, a1, b4);
        c50 = vfmaq_f32(c50, a0, b5);
        c51 = vfmaq_f32(c51, a1, b5);
        c60 = vfmaq_f32(c60, a0, b6);
        c61 = vfmaq_f32(c61, a1, b6);
        c70 = vfmaq_f32(c70, a0, b7);
        c71 = vfmaq_f32(c71, a1, b7);
    }

    vst1q_f32(c, c00);
    vst1q_f32(c.add(4), c01);
    vst1q_f32(c.add(ldc), c10);
    vst1q_f32(c.add(ldc + 4), c11);
    vst1q_f32(c.add(2 * ldc), c20);
    vst1q_f32(c.add(2 * ldc + 4), c21);
    vst1q_f32(c.add(3 * ldc), c30);
    vst1q_f32(c.add(3 * ldc + 4), c31);
    vst1q_f32(c.add(4 * ldc), c40);
    vst1q_f32(c.add(4 * ldc + 4), c41);
    vst1q_f32(c.add(5 * ldc), c50);
    vst1q_f32(c.add(5 * ldc + 4), c51);
    vst1q_f32(c.add(6 * ldc), c60);
    vst1q_f32(c.add(6 * ldc + 4), c61);
    vst1q_f32(c.add(7 * ldc), c70);
    vst1q_f32(c.add(7 * ldc + 4), c71);
}
