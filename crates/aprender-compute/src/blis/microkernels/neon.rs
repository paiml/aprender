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

/// NEON microkernel for the BLIS packing contract: an 8×6 output tile (`MR = 8`, `NR = 6`).
///
/// Computes `C[8×6] += A[8×K] · B[K×6]` with exactly the panel layout of
/// [`microkernel_scalar`](super::microkernel_scalar): A packed column-major with stride
/// `MR`, B packed row-major with stride `NR`, C column-major with leading dimension `ldc`.
/// This is the aarch64 twin of the AVX2 `microkernel_8x6_*` family and is what
/// `gemm_blis` dispatches to on aarch64 (before it, every shape ran the scalar kernel).
///
/// Twelve `float32x4` accumulators (6 columns × two halves of the 8-row A vector) stay in
/// registers for the whole K loop. Each K step is two A loads, one q-load plus one d-load
/// of B, and twelve lane-indexed FMAs (`fmla.4s v, v, v[lane]`), so B is never broadcast
/// through memory.
///
/// # Safety
/// `a` must be valid for `8 * k` reads, `b` for `6 * k` reads, and `c` for an 8×6 tile at
/// column stride `ldc` (`ldc >= 8`). NEON is baseline on aarch64, so no feature check is
/// needed.
#[cfg(target_arch = "aarch64")]
pub unsafe fn microkernel_8x6_neon(
    k: usize,
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    ldc: usize,
) {
    use std::arch::aarch64::*;

    // SAFETY: bounds are the caller's contract (see above); every access below stays inside
    // `8 * k` of `a`, `6 * k` of `b`, and the 8×6 tile of `c`.
    unsafe {
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

        for p in 0..k {
            let a0 = vld1q_f32(a.add(p * 8));
            let a1 = vld1q_f32(a.add(p * 8 + 4));
            let bq = vld1q_f32(b.add(p * 6)); // b[p][0..4]
            let bd = vld1_f32(b.add(p * 6 + 4)); // b[p][4..6]

            c00 = vfmaq_laneq_f32::<0>(c00, a0, bq);
            c01 = vfmaq_laneq_f32::<0>(c01, a1, bq);
            c10 = vfmaq_laneq_f32::<1>(c10, a0, bq);
            c11 = vfmaq_laneq_f32::<1>(c11, a1, bq);
            c20 = vfmaq_laneq_f32::<2>(c20, a0, bq);
            c21 = vfmaq_laneq_f32::<2>(c21, a1, bq);
            c30 = vfmaq_laneq_f32::<3>(c30, a0, bq);
            c31 = vfmaq_laneq_f32::<3>(c31, a1, bq);
            c40 = vfmaq_lane_f32::<0>(c40, a0, bd);
            c41 = vfmaq_lane_f32::<0>(c41, a1, bd);
            c50 = vfmaq_lane_f32::<1>(c50, a0, bd);
            c51 = vfmaq_lane_f32::<1>(c51, a1, bd);
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
    }
}
