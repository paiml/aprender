//! Matrix transpose kernel: out-of-place B = A^T with AVX2 8×8 micro-kernel.
//!
//! Matches `transpose-kernel-v1.yaml`.
//! Three phases: `outer_blocking` -> `avx2_8x8_microkernel` -> `remainder`.
//!
//! # Algorithm
//!
//! Process the matrix in 8×8 blocks. For each block, load 8 source rows
//! into YMM registers, perform 3-phase in-register transpose (unpack →
//! shuffle → permute), then store 8 transposed rows. Contiguous 32-byte
//! stores coalesce cache misses (8 vs 64 in scalar).
//!
//! # References
//!
//! - Lam, Rothberg & Wolf (1991) Cache Performance of Blocked Algorithms
//! - Intel Intrinsics Guide: _mm256_unpacklo_ps, _mm256_shuffle_ps, _mm256_permute2f128_ps

use provable_contracts_macros::requires;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ────────────────────────────────────────────────────────────────────────────
// Scalar implementation
// ────────────────────────────────────────────────────────────────────────────

/// Scalar reference transpose: B[j * rows + i] = A[i * cols + j].
///
/// Uses 8×8 blocking for cache efficiency. Handles arbitrary dimensions
/// via remainder loops for non-8-aligned edges.
///
/// # Panics
///
/// Panics if `a.len() != rows * cols` or `b.len() != rows * cols`.
pub fn transpose_scalar(rows: usize, cols: usize, a: &[f32], b: &mut [f32]) {
    const BLOCK: usize = 8;

    assert_eq!(a.len(), rows * cols, "a length mismatch");
    assert_eq!(b.len(), rows * cols, "b length mismatch");

    let rb_end = rows / BLOCK * BLOCK;
    let cb_end = cols / BLOCK * BLOCK;

    // Full 8×8 blocks
    for r0 in (0..rb_end).step_by(BLOCK) {
        for c0 in (0..cb_end).step_by(BLOCK) {
            for r in r0..r0 + BLOCK {
                let src_base = r * cols;
                for c in c0..c0 + BLOCK {
                    b[c * rows + r] = a[src_base + c];
                }
            }
        }
    }

    // Right edge remainder (cols not divisible by 8)
    if cb_end < cols {
        for r in 0..rb_end {
            let src_base = r * cols;
            for c in cb_end..cols {
                b[c * rows + r] = a[src_base + c];
            }
        }
    }

    // Bottom edge remainder (rows not divisible by 8)
    if rb_end < rows {
        for r in rb_end..rows {
            let src_base = r * cols;
            for c in 0..cols {
                b[c * rows + r] = a[src_base + c];
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// AVX2 implementation
// ────────────────────────────────────────────────────────────────────────────

/// AVX2 8×8 in-register transpose micro-kernel.
///
/// Loads 8 rows of 8 f32 from source (stride = `src_stride` elements),
/// performs 3-phase shuffle/permute, stores 8 transposed rows to dest
/// (stride = `dst_stride` elements).
///
/// # Safety
///
/// Requires AVX2 support. Caller must ensure sufficient data at
/// `src` and `dst` pointers (8 rows × stride elements each).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn transpose_8x8_avx2(src: *const f32, src_stride: usize, dst: *mut f32, dst_stride: usize) {
    unsafe {
        // Load 8 source rows
        let r0 = _mm256_loadu_ps(src);
        let r1 = _mm256_loadu_ps(src.add(src_stride));
        let r2 = _mm256_loadu_ps(src.add(src_stride * 2));
        let r3 = _mm256_loadu_ps(src.add(src_stride * 3));
        let r4 = _mm256_loadu_ps(src.add(src_stride * 4));
        let r5 = _mm256_loadu_ps(src.add(src_stride * 5));
        let r6 = _mm256_loadu_ps(src.add(src_stride * 6));
        let r7 = _mm256_loadu_ps(src.add(src_stride * 7));

        // Phase 1: Interleave adjacent row pairs using unpacklo/unpackhi
        let t0 = _mm256_unpacklo_ps(r0, r1); // a0 b0 a1 b1 | a4 b4 a5 b5
        let t1 = _mm256_unpackhi_ps(r0, r1); // a2 b2 a3 b3 | a6 b6 a7 b7
        let t2 = _mm256_unpacklo_ps(r2, r3); // c0 d0 c1 d1 | c4 d4 c5 d5
        let t3 = _mm256_unpackhi_ps(r2, r3); // c2 d2 c3 d3 | c6 d6 c7 d7
        let t4 = _mm256_unpacklo_ps(r4, r5); // e0 f0 e1 f1 | e4 f4 e5 f5
        let t5 = _mm256_unpackhi_ps(r4, r5); // e2 f2 e3 f3 | e6 f6 e7 f7
        let t6 = _mm256_unpacklo_ps(r6, r7); // g0 h0 g1 h1 | g4 h4 g5 h5
        let t7 = _mm256_unpackhi_ps(r6, r7); // g2 h2 g3 h3 | g6 h6 g7 h7

        // Phase 2: Shuffle 64-bit pairs within 128-bit lanes
        let u0 = _mm256_shuffle_ps(t0, t2, 0x44); // a0 b0 c0 d0 | a4 b4 c4 d4
        let u1 = _mm256_shuffle_ps(t0, t2, 0xEE); // a1 b1 c1 d1 | a5 b5 c5 d5
        let u2 = _mm256_shuffle_ps(t1, t3, 0x44); // a2 b2 c2 d2 | a6 b6 c6 d6
        let u3 = _mm256_shuffle_ps(t1, t3, 0xEE); // a3 b3 c3 d3 | a7 b7 c7 d7
        let u4 = _mm256_shuffle_ps(t4, t6, 0x44); // e0 f0 g0 h0 | e4 f4 g4 h4
        let u5 = _mm256_shuffle_ps(t4, t6, 0xEE); // e1 f1 g1 h1 | e5 f5 g5 h5
        let u6 = _mm256_shuffle_ps(t5, t7, 0x44); // e2 f2 g2 h2 | e6 f6 g6 h6
        let u7 = _mm256_shuffle_ps(t5, t7, 0xEE); // e3 f3 g3 h3 | e7 f7 g7 h7

        // Phase 3: Swap 128-bit halves across YMM registers
        let v0 = _mm256_permute2f128_ps(u0, u4, 0x20); // row 0 of transpose
        let v1 = _mm256_permute2f128_ps(u1, u5, 0x20); // row 1
        let v2 = _mm256_permute2f128_ps(u2, u6, 0x20); // row 2
        let v3 = _mm256_permute2f128_ps(u3, u7, 0x20); // row 3
        let v4 = _mm256_permute2f128_ps(u0, u4, 0x31); // row 4
        let v5 = _mm256_permute2f128_ps(u1, u5, 0x31); // row 5
        let v6 = _mm256_permute2f128_ps(u2, u6, 0x31); // row 6
        let v7 = _mm256_permute2f128_ps(u3, u7, 0x31); // row 7

        // Store 8 transposed rows
        _mm256_storeu_ps(dst, v0);
        _mm256_storeu_ps(dst.add(dst_stride), v1);
        _mm256_storeu_ps(dst.add(dst_stride * 2), v2);
        _mm256_storeu_ps(dst.add(dst_stride * 3), v3);
        _mm256_storeu_ps(dst.add(dst_stride * 4), v4);
        _mm256_storeu_ps(dst.add(dst_stride * 5), v5);
        _mm256_storeu_ps(dst.add(dst_stride * 6), v6);
        _mm256_storeu_ps(dst.add(dst_stride * 7), v7);
    }
}

/// AVX2 matrix transpose using 8×8 in-register micro-kernel.
///
/// Processes full 8×8 blocks with SIMD, remainder edges with scalar.
/// Source row stride = `cols`, dest row stride = `rows` (transposed layout).
///
/// # Safety
///
/// Requires AVX2 support. Caller must verify with `is_x86_feature_detected!("avx2")`.
///
/// # Panics
///
/// Panics if `a.len() != rows * cols` or `b.len() != rows * cols`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn transpose_avx2(rows: usize, cols: usize, a: &[f32], b: &mut [f32]) {
    assert_eq!(a.len(), rows * cols, "a length mismatch");
    assert_eq!(b.len(), rows * cols, "b length mismatch");

    let rb_end = rows / 8 * 8;
    let cb_end = cols / 8 * 8;

    // SAFETY: AVX2 verified by caller + target_feature gate.
    unsafe {
        // Full 8×8 blocks: AVX2 micro-kernel
        for r0 in (0..rb_end).step_by(8) {
            for c0 in (0..cb_end).step_by(8) {
                let src = a.as_ptr().add(r0 * cols + c0);
                let dst = b.as_mut_ptr().add(c0 * rows + r0);
                transpose_8x8_avx2(src, cols, dst, rows);
            }
        }
    }

    // Right edge remainder (cols % 8 != 0): scalar
    if cb_end < cols {
        for r in 0..rb_end {
            let src_base = r * cols;
            for c in cb_end..cols {
                b[c * rows + r] = a[src_base + c];
            }
        }
    }

    // Bottom edge remainder (rows % 8 != 0): scalar
    if rb_end < rows {
        for r in rb_end..rows {
            let src_base = r * cols;
            for c in 0..cols {
                b[c * rows + r] = a[src_base + c];
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Dispatch
// ────────────────────────────────────────────────────────────────────────────

/// Transpose a matrix: B = A^T. Dispatches to AVX2 or scalar.
///
/// # Panics
///
/// Panics if `a.len() != rows * cols` or `b.len() != rows * cols`.
#[requires(a.len() == rows * cols && b.len() == rows * cols)]
pub fn transpose(rows: usize, cols: usize, a: &[f32], b: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 verified by feature detection above.
            unsafe {
                transpose_avx2(rows, cols, a, b);
            }
            return;
        }
    }
    transpose_scalar(rows, cols, a, b);
}

// ────────────────────────────────────────────────────────────────────────────
// Tests — contract falsification
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive reference transpose for validation.
    fn transpose_naive(rows: usize, cols: usize, a: &[f32], b: &mut [f32]) {
        for i in 0..rows {
            for j in 0..cols {
                b[j * rows + i] = a[i * cols + j];
            }
        }
    }

    /// FALSIFY-TP-001: Element correctness
    /// transpose(A)[j][i] == A[i][j] for random A
    #[test]
    fn falsify_tp_001_element_correctness() {
        for (rows, cols) in [(4, 5), (8, 8), (16, 32), (31, 17), (64, 64)] {
            let a: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
            let mut b = vec![0.0f32; rows * cols];
            transpose(rows, cols, &a, &mut b);

            for i in 0..rows {
                for j in 0..cols {
                    assert_eq!(
                        b[j * rows + i],
                        a[i * cols + j],
                        "Mismatch at ({i},{j}) for {rows}×{cols}"
                    );
                }
            }
        }
    }

    /// FALSIFY-TP-002: Involution
    /// transpose(transpose(A)) == A (bitwise exact)
    #[test]
    fn falsify_tp_002_involution() {
        for (rows, cols) in [(7, 13), (16, 16), (33, 17), (64, 128)] {
            let a: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.1 + 0.37).collect();
            let mut b = vec![0.0f32; rows * cols];
            let mut c = vec![0.0f32; rows * cols];

            transpose(rows, cols, &a, &mut b);
            transpose(cols, rows, &b, &mut c);

            assert_eq!(a, c, "Involution failed for {rows}×{cols}");
        }
    }

    /// FALSIFY-TP-003: Non-8-aligned dimensions
    /// Correct for 7×13, 17×3, 1×N, N×1
    #[test]
    fn falsify_tp_003_non_aligned() {
        for (rows, cols) in [(7, 13), (17, 3), (1, 32), (32, 1), (1, 1), (3, 3)] {
            let a: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
            let mut b_test = vec![0.0f32; rows * cols];
            let mut b_ref = vec![0.0f32; rows * cols];

            transpose(rows, cols, &a, &mut b_test);
            transpose_naive(rows, cols, &a, &mut b_ref);

            assert_eq!(b_test, b_ref, "Mismatch for {rows}×{cols}");
        }
    }

    /// FALSIFY-TP-004: AVX2 vs scalar parity (bitwise exact)
    #[test]
    fn falsify_tp_004_avx2_scalar_parity() {
        let rows = 2048;
        let cols = 128;
        let a: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.001).collect();
        let mut b_scalar = vec![0.0f32; rows * cols];
        let mut b_dispatch = vec![0.0f32; rows * cols];

        transpose_scalar(rows, cols, &a, &mut b_scalar);
        transpose(rows, cols, &a, &mut b_dispatch);

        assert_eq!(b_scalar, b_dispatch, "AVX2 vs scalar mismatch at 2048×128");
    }

    /// FALSIFY-TP-005: Identity matrix
    /// transpose(I) == I for square identity
    #[test]
    fn falsify_tp_005_identity() {
        for n in [4, 8, 16, 32] {
            let mut a = vec![0.0f32; n * n];
            for i in 0..n {
                a[i * n + i] = 1.0;
            }
            let mut b = vec![0.0f32; n * n];
            transpose(n, n, &a, &mut b);
            assert_eq!(a, b, "Identity matrix not preserved for {n}×{n}");
        }
    }

    /// FALSIFY-TP-006: Attention shape (2048×128)
    /// Matches naive reference
    #[test]
    fn falsify_tp_006_attention_shape() {
        let rows = 2048;
        let cols = 128;
        let a: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 17 + 31) % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        let mut b_test = vec![0.0f32; rows * cols];
        let mut b_ref = vec![0.0f32; rows * cols];

        transpose(rows, cols, &a, &mut b_test);
        transpose_naive(rows, cols, &a, &mut b_ref);

        assert_eq!(b_test, b_ref, "Attention shape 2048×128 mismatch");
    }

    /// Cover scalar remainder paths (rows/cols not divisible by 8)
    #[test]
    fn scalar_remainder_paths() {
        for (rows, cols) in [(3, 5), (10, 13), (15, 9), (7, 7)] {
            let a: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
            let mut b_scalar = vec![0.0f32; rows * cols];
            let mut b_ref = vec![0.0f32; rows * cols];

            transpose_scalar(rows, cols, &a, &mut b_scalar);
            transpose_naive(rows, cols, &a, &mut b_ref);

            assert_eq!(b_scalar, b_ref, "Scalar mismatch for {rows}×{cols}");
        }
    }
}
