//! AVX-512 SIMD Microkernels
//!
//! Two microkernel variants:
//! - **16×8**: Original tile, 8 zmm accumulators. Used by gemm_blis_avx512_large.
//! - **32×6**: Larger tile (Phase 4, Appendix D), 12 zmm accumulators.
//!   Uses 2 rows of 16 f32 × 6 columns = 12 accumulators + 2 A loads.
//!   1.5× more FMAs per K step than 16×8, better register utilization.
//!
//! Register allocation:
//! - zmm0-zmm7: 8 columns of C (16 f32 each) = 128 outputs in registers
//! - A column loaded per iteration, B broadcast from memory via vbroadcastss
//!
//! 4-way K-unrolled main loop hides 5-cycle FMA latency across 2 FMA ports.

/// 16×8 AVX-512 microkernel — 4-way K-unrolled.
/// A: 16×K packed column-major. B: K×8 packed row-major.
/// C: 16×8 column-major with stride ldc.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn microkernel_16x8_avx512(
    k: usize,
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    ldc: usize,
) {
    unsafe {
        use std::arch::x86_64::*;

        // Load C (8 columns of 16 elements)
        let mut c0 = _mm512_loadu_ps(c);
        let mut c1 = _mm512_loadu_ps(c.add(ldc));
        let mut c2 = _mm512_loadu_ps(c.add(2 * ldc));
        let mut c3 = _mm512_loadu_ps(c.add(3 * ldc));
        let mut c4 = _mm512_loadu_ps(c.add(4 * ldc));
        let mut c5 = _mm512_loadu_ps(c.add(5 * ldc));
        let mut c6 = _mm512_loadu_ps(c.add(6 * ldc));
        let mut c7 = _mm512_loadu_ps(c.add(7 * ldc));

        let k4 = k / 4;
        let k_rem = k % 4;

        for p4 in 0..k4 {
            let base = p4 * 4;

            // K+0
            let a0 = _mm512_loadu_ps(a.add(base * 16));
            let bp0 = b.add(base * 8);
            c0 = _mm512_fmadd_ps(a0, _mm512_set1_ps(*bp0), c0);
            c1 = _mm512_fmadd_ps(a0, _mm512_set1_ps(*bp0.add(1)), c1);
            c2 = _mm512_fmadd_ps(a0, _mm512_set1_ps(*bp0.add(2)), c2);
            c3 = _mm512_fmadd_ps(a0, _mm512_set1_ps(*bp0.add(3)), c3);
            c4 = _mm512_fmadd_ps(a0, _mm512_set1_ps(*bp0.add(4)), c4);
            c5 = _mm512_fmadd_ps(a0, _mm512_set1_ps(*bp0.add(5)), c5);
            c6 = _mm512_fmadd_ps(a0, _mm512_set1_ps(*bp0.add(6)), c6);
            c7 = _mm512_fmadd_ps(a0, _mm512_set1_ps(*bp0.add(7)), c7);

            // K+1
            let a1 = _mm512_loadu_ps(a.add((base + 1) * 16));
            let bp1 = b.add((base + 1) * 8);
            c0 = _mm512_fmadd_ps(a1, _mm512_set1_ps(*bp1), c0);
            c1 = _mm512_fmadd_ps(a1, _mm512_set1_ps(*bp1.add(1)), c1);
            c2 = _mm512_fmadd_ps(a1, _mm512_set1_ps(*bp1.add(2)), c2);
            c3 = _mm512_fmadd_ps(a1, _mm512_set1_ps(*bp1.add(3)), c3);
            c4 = _mm512_fmadd_ps(a1, _mm512_set1_ps(*bp1.add(4)), c4);
            c5 = _mm512_fmadd_ps(a1, _mm512_set1_ps(*bp1.add(5)), c5);
            c6 = _mm512_fmadd_ps(a1, _mm512_set1_ps(*bp1.add(6)), c6);
            c7 = _mm512_fmadd_ps(a1, _mm512_set1_ps(*bp1.add(7)), c7);

            // K+2
            let a2 = _mm512_loadu_ps(a.add((base + 2) * 16));
            let bp2 = b.add((base + 2) * 8);
            c0 = _mm512_fmadd_ps(a2, _mm512_set1_ps(*bp2), c0);
            c1 = _mm512_fmadd_ps(a2, _mm512_set1_ps(*bp2.add(1)), c1);
            c2 = _mm512_fmadd_ps(a2, _mm512_set1_ps(*bp2.add(2)), c2);
            c3 = _mm512_fmadd_ps(a2, _mm512_set1_ps(*bp2.add(3)), c3);
            c4 = _mm512_fmadd_ps(a2, _mm512_set1_ps(*bp2.add(4)), c4);
            c5 = _mm512_fmadd_ps(a2, _mm512_set1_ps(*bp2.add(5)), c5);
            c6 = _mm512_fmadd_ps(a2, _mm512_set1_ps(*bp2.add(6)), c6);
            c7 = _mm512_fmadd_ps(a2, _mm512_set1_ps(*bp2.add(7)), c7);

            // K+3
            let a3 = _mm512_loadu_ps(a.add((base + 3) * 16));
            let bp3 = b.add((base + 3) * 8);
            c0 = _mm512_fmadd_ps(a3, _mm512_set1_ps(*bp3), c0);
            c1 = _mm512_fmadd_ps(a3, _mm512_set1_ps(*bp3.add(1)), c1);
            c2 = _mm512_fmadd_ps(a3, _mm512_set1_ps(*bp3.add(2)), c2);
            c3 = _mm512_fmadd_ps(a3, _mm512_set1_ps(*bp3.add(3)), c3);
            c4 = _mm512_fmadd_ps(a3, _mm512_set1_ps(*bp3.add(4)), c4);
            c5 = _mm512_fmadd_ps(a3, _mm512_set1_ps(*bp3.add(5)), c5);
            c6 = _mm512_fmadd_ps(a3, _mm512_set1_ps(*bp3.add(6)), c6);
            c7 = _mm512_fmadd_ps(a3, _mm512_set1_ps(*bp3.add(7)), c7);
        }

        // Remainder
        let base_rem = k4 * 4;
        for p in 0..k_rem {
            let pp = base_rem + p;
            let a_col = _mm512_loadu_ps(a.add(pp * 16));
            let bp = b.add(pp * 8);
            c0 = _mm512_fmadd_ps(a_col, _mm512_set1_ps(*bp), c0);
            c1 = _mm512_fmadd_ps(a_col, _mm512_set1_ps(*bp.add(1)), c1);
            c2 = _mm512_fmadd_ps(a_col, _mm512_set1_ps(*bp.add(2)), c2);
            c3 = _mm512_fmadd_ps(a_col, _mm512_set1_ps(*bp.add(3)), c3);
            c4 = _mm512_fmadd_ps(a_col, _mm512_set1_ps(*bp.add(4)), c4);
            c5 = _mm512_fmadd_ps(a_col, _mm512_set1_ps(*bp.add(5)), c5);
            c6 = _mm512_fmadd_ps(a_col, _mm512_set1_ps(*bp.add(6)), c6);
            c7 = _mm512_fmadd_ps(a_col, _mm512_set1_ps(*bp.add(7)), c7);
        }

        // Store C
        _mm512_storeu_ps(c, c0);
        _mm512_storeu_ps(c.add(ldc), c1);
        _mm512_storeu_ps(c.add(2 * ldc), c2);
        _mm512_storeu_ps(c.add(3 * ldc), c3);
        _mm512_storeu_ps(c.add(4 * ldc), c4);
        _mm512_storeu_ps(c.add(5 * ldc), c5);
        _mm512_storeu_ps(c.add(6 * ldc), c6);
        _mm512_storeu_ps(c.add(7 * ldc), c7);
    }
}

/// 32×6 AVX-512 microkernel — 2-way K-unrolled.
///
/// A: 32×K packed column-major (two consecutive zmm rows per K step).
/// B: K×6 packed row-major.
/// C: 32×6 column-major with stride ldc (32 = 2 zmm rows).
///
/// Register allocation (14 of 32 zmm used):
///   zmm0-zmm5:  row 0 accumulators (C[0..16, j] for j=0..6)
///   zmm6-zmm11: row 1 accumulators (C[16..32, j] for j=0..6)
///   zmm12-zmm13: A column loads (rows 0-15, 16-31)
///   B: broadcast from memory via vbroadcastss (no register needed)
///
/// FMAs per K step: 12 (2 rows × 6 cols). With 2-way unroll: 24 FMAs.
/// vs 16×8: 8 FMAs/step → 32 FMAs/4-unroll. This kernel: 1.5× more FMA/step.
///
/// Appendix D optimization #1: increase register utilization from 25% to 44%.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn microkernel_32x6_avx512(
    k: usize,
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    ldc: usize,
) {
    unsafe {
        use std::arch::x86_64::*;

        // Load C: 2 zmm rows × 6 columns = 12 accumulators
        let mut c00 = _mm512_loadu_ps(c);
        let mut c01 = _mm512_loadu_ps(c.add(ldc));
        let mut c02 = _mm512_loadu_ps(c.add(2 * ldc));
        let mut c03 = _mm512_loadu_ps(c.add(3 * ldc));
        let mut c04 = _mm512_loadu_ps(c.add(4 * ldc));
        let mut c05 = _mm512_loadu_ps(c.add(5 * ldc));
        let mut c10 = _mm512_loadu_ps(c.add(16));
        let mut c11 = _mm512_loadu_ps(c.add(ldc + 16));
        let mut c12 = _mm512_loadu_ps(c.add(2 * ldc + 16));
        let mut c13 = _mm512_loadu_ps(c.add(3 * ldc + 16));
        let mut c14 = _mm512_loadu_ps(c.add(4 * ldc + 16));
        let mut c15 = _mm512_loadu_ps(c.add(5 * ldc + 16));

        let k2 = k / 2;
        let k_rem = k % 2;

        // Main loop: 2-way K-unrolled
        for p2 in 0..k2 {
            let base = p2 * 2;

            // K+0: load A row0 and row1
            let a0_lo = _mm512_loadu_ps(a.add(base * 32));
            let a0_hi = _mm512_loadu_ps(a.add(base * 32 + 16));
            let bp0 = b.add(base * 6);

            // 6 FMAs for row 0
            let b0 = _mm512_set1_ps(*bp0);
            c00 = _mm512_fmadd_ps(a0_lo, b0, c00);
            c10 = _mm512_fmadd_ps(a0_hi, b0, c10);
            let b1 = _mm512_set1_ps(*bp0.add(1));
            c01 = _mm512_fmadd_ps(a0_lo, b1, c01);
            c11 = _mm512_fmadd_ps(a0_hi, b1, c11);
            let b2 = _mm512_set1_ps(*bp0.add(2));
            c02 = _mm512_fmadd_ps(a0_lo, b2, c02);
            c12 = _mm512_fmadd_ps(a0_hi, b2, c12);
            let b3 = _mm512_set1_ps(*bp0.add(3));
            c03 = _mm512_fmadd_ps(a0_lo, b3, c03);
            c13 = _mm512_fmadd_ps(a0_hi, b3, c13);
            let b4 = _mm512_set1_ps(*bp0.add(4));
            c04 = _mm512_fmadd_ps(a0_lo, b4, c04);
            c14 = _mm512_fmadd_ps(a0_hi, b4, c14);
            let b5 = _mm512_set1_ps(*bp0.add(5));
            c05 = _mm512_fmadd_ps(a0_lo, b5, c05);
            c15 = _mm512_fmadd_ps(a0_hi, b5, c15);

            // K+1
            let a1_lo = _mm512_loadu_ps(a.add((base + 1) * 32));
            let a1_hi = _mm512_loadu_ps(a.add((base + 1) * 32 + 16));
            let bp1 = b.add((base + 1) * 6);

            let b0 = _mm512_set1_ps(*bp1);
            c00 = _mm512_fmadd_ps(a1_lo, b0, c00);
            c10 = _mm512_fmadd_ps(a1_hi, b0, c10);
            let b1 = _mm512_set1_ps(*bp1.add(1));
            c01 = _mm512_fmadd_ps(a1_lo, b1, c01);
            c11 = _mm512_fmadd_ps(a1_hi, b1, c11);
            let b2 = _mm512_set1_ps(*bp1.add(2));
            c02 = _mm512_fmadd_ps(a1_lo, b2, c02);
            c12 = _mm512_fmadd_ps(a1_hi, b2, c12);
            let b3 = _mm512_set1_ps(*bp1.add(3));
            c03 = _mm512_fmadd_ps(a1_lo, b3, c03);
            c13 = _mm512_fmadd_ps(a1_hi, b3, c13);
            let b4 = _mm512_set1_ps(*bp1.add(4));
            c04 = _mm512_fmadd_ps(a1_lo, b4, c04);
            c14 = _mm512_fmadd_ps(a1_hi, b4, c14);
            let b5 = _mm512_set1_ps(*bp1.add(5));
            c05 = _mm512_fmadd_ps(a1_lo, b5, c05);
            c15 = _mm512_fmadd_ps(a1_hi, b5, c15);
        }

        // Remainder
        let base_rem = k2 * 2;
        for p in 0..k_rem {
            let pp = base_rem + p;
            let a_lo = _mm512_loadu_ps(a.add(pp * 32));
            let a_hi = _mm512_loadu_ps(a.add(pp * 32 + 16));
            let bp = b.add(pp * 6);
            let b0 = _mm512_set1_ps(*bp);
            c00 = _mm512_fmadd_ps(a_lo, b0, c00);
            c10 = _mm512_fmadd_ps(a_hi, b0, c10);
            let b1 = _mm512_set1_ps(*bp.add(1));
            c01 = _mm512_fmadd_ps(a_lo, b1, c01);
            c11 = _mm512_fmadd_ps(a_hi, b1, c11);
            let b2 = _mm512_set1_ps(*bp.add(2));
            c02 = _mm512_fmadd_ps(a_lo, b2, c02);
            c12 = _mm512_fmadd_ps(a_hi, b2, c12);
            let b3 = _mm512_set1_ps(*bp.add(3));
            c03 = _mm512_fmadd_ps(a_lo, b3, c03);
            c13 = _mm512_fmadd_ps(a_hi, b3, c13);
            let b4 = _mm512_set1_ps(*bp.add(4));
            c04 = _mm512_fmadd_ps(a_lo, b4, c04);
            c14 = _mm512_fmadd_ps(a_hi, b4, c14);
            let b5 = _mm512_set1_ps(*bp.add(5));
            c05 = _mm512_fmadd_ps(a_lo, b5, c05);
            c15 = _mm512_fmadd_ps(a_hi, b5, c15);
        }

        // Store C: 2 rows × 6 columns
        _mm512_storeu_ps(c, c00);
        _mm512_storeu_ps(c.add(ldc), c01);
        _mm512_storeu_ps(c.add(2 * ldc), c02);
        _mm512_storeu_ps(c.add(3 * ldc), c03);
        _mm512_storeu_ps(c.add(4 * ldc), c04);
        _mm512_storeu_ps(c.add(5 * ldc), c05);
        _mm512_storeu_ps(c.add(16), c10);
        _mm512_storeu_ps(c.add(ldc + 16), c11);
        _mm512_storeu_ps(c.add(2 * ldc + 16), c12);
        _mm512_storeu_ps(c.add(3 * ldc + 16), c13);
        _mm512_storeu_ps(c.add(4 * ldc + 16), c14);
        _mm512_storeu_ps(c.add(5 * ldc + 16), c15);
    }
}
