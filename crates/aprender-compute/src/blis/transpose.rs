//! Matrix transpose operations.
//!
//! AVX2 8×8 in-register transpose with scalar fallback for small matrices
//! and non-8-aligned remainder edges.
//!
//! # Algorithm
//!
//! Process matrix in 8×8 blocks. For each block, load 8 rows into YMM
//! registers, perform 3-phase shuffle/permute transpose, store 8 transposed
//! rows. Contiguous 32-byte stores coalesce cache misses (8 vs 64 in scalar).
//!
//! Contract: provable-contracts/contracts/transpose-kernel-v1.yaml
//!
//! # References
//!
//! - Lam, Rothberg & Wolf (1991). Cache Performance of Blocked Algorithms. ASPLOS
//! - Intel Intrinsics Guide: _mm256_unpacklo_ps, _mm256_shuffle_ps, _mm256_permute2f128_ps
//! - GH-388: transpose 242x slower than ndarray at attention shapes

use crate::error::TruenoError;

/// Scalar transpose of a sub-region of a row-major matrix.
#[inline(always)]
fn transpose_region(
    a: &[f32],
    b: &mut [f32],
    rows: std::ops::Range<usize>,
    cols: std::ops::Range<usize>,
    src_cols: usize,
    dst_rows: usize,
) {
    for r in rows {
        let src_base = r * src_cols;
        for c in cols.clone() {
            b[c * dst_rows + r] = a[src_base + c];
        }
    }
}

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
        use std::arch::x86_64::*;

        // Load 8 source rows
        let r0 = _mm256_loadu_ps(src);
        let r1 = _mm256_loadu_ps(src.add(src_stride));
        let r2 = _mm256_loadu_ps(src.add(src_stride * 2));
        let r3 = _mm256_loadu_ps(src.add(src_stride * 3));
        let r4 = _mm256_loadu_ps(src.add(src_stride * 4));
        let r5 = _mm256_loadu_ps(src.add(src_stride * 5));
        let r6 = _mm256_loadu_ps(src.add(src_stride * 6));
        let r7 = _mm256_loadu_ps(src.add(src_stride * 7));

        // Phase 1: Interleave adjacent row pairs
        let t0 = _mm256_unpacklo_ps(r0, r1);
        let t1 = _mm256_unpackhi_ps(r0, r1);
        let t2 = _mm256_unpacklo_ps(r2, r3);
        let t3 = _mm256_unpackhi_ps(r2, r3);
        let t4 = _mm256_unpacklo_ps(r4, r5);
        let t5 = _mm256_unpackhi_ps(r4, r5);
        let t6 = _mm256_unpacklo_ps(r6, r7);
        let t7 = _mm256_unpackhi_ps(r6, r7);

        // Phase 2: Shuffle 64-bit pairs within 128-bit lanes
        let u0 = _mm256_shuffle_ps(t0, t2, 0x44);
        let u1 = _mm256_shuffle_ps(t0, t2, 0xEE);
        let u2 = _mm256_shuffle_ps(t1, t3, 0x44);
        let u3 = _mm256_shuffle_ps(t1, t3, 0xEE);
        let u4 = _mm256_shuffle_ps(t4, t6, 0x44);
        let u5 = _mm256_shuffle_ps(t4, t6, 0xEE);
        let u6 = _mm256_shuffle_ps(t5, t7, 0x44);
        let u7 = _mm256_shuffle_ps(t5, t7, 0xEE);

        // Phase 3: Swap 128-bit halves across YMM registers
        let v0 = _mm256_permute2f128_ps(u0, u4, 0x20);
        let v1 = _mm256_permute2f128_ps(u1, u5, 0x20);
        let v2 = _mm256_permute2f128_ps(u2, u6, 0x20);
        let v3 = _mm256_permute2f128_ps(u3, u7, 0x20);
        let v4 = _mm256_permute2f128_ps(u0, u4, 0x31);
        let v5 = _mm256_permute2f128_ps(u1, u5, 0x31);
        let v6 = _mm256_permute2f128_ps(u2, u6, 0x31);
        let v7 = _mm256_permute2f128_ps(u3, u7, 0x31);

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

/// Transpose a matrix: B = A^T
///
/// Uses AVX2 8×8 in-register micro-kernel for full blocks, scalar for
/// remainder edges. Runtime feature detection selects AVX2 or scalar.
///
/// Contract: transpose-kernel-v1, equations "transpose"
///
/// # Arguments
///
/// * `rows` - Number of rows in A (cols in B)
/// * `cols` - Number of cols in A (rows in B)
/// * `a` - Input matrix A (rows x cols, row-major)
/// * `b` - Output matrix B (cols x rows, row-major)
///
/// # Returns
///
/// `Ok(())` on success, `Err` if dimensions mismatch
pub fn transpose(rows: usize, cols: usize, a: &[f32], b: &mut [f32]) -> Result<(), TruenoError> {
    // Contract: transpose-kernel-v1.yaml, equation = transpose
    debug_assert!(!a.is_empty(), "Contract transpose: input is empty");
    debug_assert!(rows > 0 && cols > 0, "Contract transpose: zero dimensions");
    let expected = rows * cols;
    if a.len() != expected || b.len() != expected {
        return Err(TruenoError::InvalidInput(format!(
            "transpose size mismatch: a[{}], b[{}], expected {}",
            a.len(),
            b.len(),
            expected
        )));
    }

    if expected < 64 {
        transpose_region(a, b, 0..rows, 0..cols, cols, rows);
        return Ok(());
    }

    // Parallel transpose for large matrices (DRAM-bound → multi-channel).
    // CGP-DBUF: lowered threshold from 4M to 1M. Previous regression at 1M was
    // likely from higher thread::scope overhead; Rayon dispatch is ~3µs.
    // 1024×1024 = 1M elements, 4MB traffic, ~200µs single-thread → 3µs is 1.5%.
    #[cfg(feature = "parallel")]
    {
        const PARALLEL_THRESHOLD: usize = 1_000_000;
        if expected >= PARALLEL_THRESHOLD {
            return transpose_parallel(rows, cols, a, b);
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 verified by feature detection above.
            // Slice bounds: 8×8 blocks within rows×cols guaranteed by loop bounds.
            unsafe {
                return transpose_avx2_impl(rows, cols, a, b);
            }
        }
    }

    transpose_scalar_impl(rows, cols, a, b);
    Ok(())
}

/// Parallel transpose: chunk A by row ranges, each thread transposes its strip.
/// Each thread writes to a disjoint COLUMN range of B (non-overlapping mutations).
#[cfg(feature = "parallel")]
fn transpose_parallel(
    rows: usize,
    cols: usize,
    a: &[f32],
    b: &mut [f32],
) -> Result<(), TruenoError> {
    use rayon::prelude::*;

    let num_threads = rayon::current_num_threads().min(8);
    // Chunk A's rows; each thread handles rows_per rows of A = cols_per (in B)
    let rows_per = (rows + num_threads - 1) / num_threads;

    // SAFETY: each thread writes to disjoint slices of B via raw pointer.
    // Thread t writes B[c, rt..rt+rows_per] for all c in 0..cols — disjoint row ranges in A
    // map to disjoint COLUMN ranges in B; B is accessed via (c*rows + r) indexing so
    // threads write non-overlapping memory regions (separated by rows_per elements per row).
    let b_ptr = b.as_mut_ptr() as usize;

    (0..num_threads).into_par_iter().try_for_each(|t| {
        let r_start = t * rows_per;
        let r_end = (r_start + rows_per).min(rows);
        if r_start >= r_end {
            return Ok::<(), TruenoError>(());
        }
        let sub_rows = r_end - r_start;

        // Input strip: A[r_start..r_end, 0..cols]
        let a_strip = &a[r_start * cols..r_end * cols];

        // SAFETY: see above - disjoint column range in B
        unsafe {
            let b_ptr_mut = b_ptr as *mut f32;
            // For this strip, transpose into B using absolute B coordinates.
            // B[c, r] for r in r_start..r_end, c in 0..cols.
            // Equivalent: substrip of B has shape [cols, sub_rows] with stride rows.
            transpose_strided_avx2(sub_rows, cols, a_strip, b_ptr_mut.add(r_start), rows)?;
        }
        Ok(())
    })?;
    Ok(())
}

/// Transpose A[sub_rows×cols] → B viewed as [cols × ...] with given b_stride (= rows of full B).
/// Writes into B at offsets (c * b_stride + r) for r in 0..sub_rows, c in 0..cols.
#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
unsafe fn transpose_strided_avx2(
    sub_rows: usize,
    cols: usize,
    a: &[f32],
    b_ptr: *mut f32,
    b_stride: usize,
) -> Result<(), TruenoError> {
    const TILE: usize = 64; // L1-resident outer tile (16KB working set)
    const BLOCK: usize = 8; // AVX2 micro-kernel

    let rb_end = sub_rows / BLOCK * BLOCK;
    let cb_end = cols / BLOCK * BLOCK;

    unsafe {
        // Two-level tiling: 64×64 outer (L1) + 8×8 inner (AVX2)
        // Standard inner order (r0 outer, c0 inner): sequential A reads.
        for rt in (0..rb_end).step_by(TILE) {
            let rt_end = (rt + TILE).min(rb_end);
            for ct in (0..cb_end).step_by(TILE) {
                let ct_end = (ct + TILE).min(cb_end);
                for r0 in (rt..rt_end).step_by(BLOCK) {
                    for c0 in (ct..ct_end).step_by(BLOCK) {
                        let src = a.as_ptr().add(r0 * cols + c0);
                        let dst = b_ptr.add(c0 * b_stride + r0);
                        transpose_8x8_avx2(src, cols, dst, b_stride);
                    }
                }
            }
        }
        // Remainder: edge rows
        if rb_end < sub_rows {
            for r in rb_end..sub_rows {
                for c in 0..cols {
                    *b_ptr.add(c * b_stride + r) = *a.get_unchecked(r * cols + c);
                }
            }
        }
        // Remainder: edge cols (for row-aligned part)
        if cb_end < cols {
            for r in 0..rb_end {
                for c in cb_end..cols {
                    *b_ptr.add(c * b_stride + r) = *a.get_unchecked(r * cols + c);
                }
            }
        }
    }
    Ok(())
}

/// Scalar fallback strided transpose (for non-x86 or no-AVX2).
#[cfg(all(feature = "parallel", not(target_arch = "x86_64")))]
unsafe fn transpose_strided_avx2(
    sub_rows: usize,
    cols: usize,
    a: &[f32],
    b_ptr: *mut f32,
    b_stride: usize,
) -> Result<(), TruenoError> {
    unsafe {
        for r in 0..sub_rows {
            for c in 0..cols {
                *b_ptr.add(c * b_stride + r) = *a.get_unchecked(r * cols + c);
            }
        }
    }
    Ok(())
}

/// AVX2 transpose with two-level tiling: 64×64 outer (L1), 8×8 inner (AVX2).
///
/// Two-level tiling keeps the working set within L1 cache (64×64×4 = 16KB < 32KB).
///
/// **Shape-adaptive loop order**:
/// - Tall-skinny (rows ≥ 4×cols): inner loop over row-blocks (r0), outer column-blocks.
///   This makes destination writes sequential: B[c0..c0+8, r0], B[c0..c0+8, r0+8], ...
///   are adjacent in memory, maximizing cache line reuse on the write side.
/// - Otherwise: inner loop over column-blocks (standard order).
///
/// Software prefetch hints for the next micro-kernel's destination lines.
///
/// # Safety
///
/// Requires AVX2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn transpose_avx2_impl(
    rows: usize,
    cols: usize,
    a: &[f32],
    b: &mut [f32],
) -> Result<(), TruenoError> {
    use std::arch::x86_64::*;

    const TILE: usize = 64; // L1-resident outer tile
    const BLOCK: usize = 8; // AVX2 micro-kernel

    let rb_end = rows / BLOCK * BLOCK;
    let cb_end = cols / BLOCK * BLOCK;

    // Tall-skinny: rows >> cols → destination stride (=rows) is large.
    // Swap loop order so inner loop walks consecutive r0 values,
    // making destination writes sequential within each cache line.
    // Tested rows >= 1024 extension (2026-04-05): regressed 1024×1024
    // from 25 → 15 GB/s. Standard order is better for square matrices
    // due to sequential A reads (prefetcher friendly).
    let tall_skinny = rows >= 4 * cols;

    unsafe {
        for rt in (0..rb_end).step_by(TILE) {
            let rt_end = (rt + TILE).min(rb_end);
            for ct in (0..cb_end).step_by(TILE) {
                let ct_end = (ct + TILE).min(cb_end);

                if tall_skinny {
                    // Outer c0, inner r0: destination writes are sequential
                    for c0 in (ct..ct_end).step_by(BLOCK) {
                        for r0 in (rt..rt_end).step_by(BLOCK) {
                            // Prefetch next micro-kernel's destination
                            if r0 + BLOCK < rt_end {
                                let pf_dst = b.as_ptr().add(c0 * rows + r0 + BLOCK);
                                _mm_prefetch(pf_dst as *const i8, _MM_HINT_T0);
                                _mm_prefetch(pf_dst.add(rows) as *const i8, _MM_HINT_T0);
                            }
                            let src = a.as_ptr().add(r0 * cols + c0);
                            let dst = b.as_mut_ptr().add(c0 * rows + r0);
                            transpose_8x8_avx2(src, cols, dst, rows);
                        }
                    }
                } else {
                    // Square/wide: standard order (no prefetch — at large strides
                    // the destination is too far apart for L1 prefetch to help)
                    for r0 in (rt..rt_end).step_by(BLOCK) {
                        for c0 in (ct..ct_end).step_by(BLOCK) {
                            let src = a.as_ptr().add(r0 * cols + c0);
                            let dst = b.as_mut_ptr().add(c0 * rows + r0);
                            transpose_8x8_avx2(src, cols, dst, rows);
                        }
                    }
                }
            }
        }
    }

    // Right edge remainder (cols % 8 != 0): scalar
    if cb_end < cols {
        transpose_region(a, b, 0..rb_end, cb_end..cols, cols, rows);
    }

    // Bottom edge remainder (rows % 8 != 0): scalar
    if rb_end < rows {
        transpose_region(a, b, rb_end..rows, 0..cols, cols, rows);
    }

    Ok(())
}

/// Scalar transpose with 8×8 blocking.
fn transpose_scalar_impl(rows: usize, cols: usize, a: &[f32], b: &mut [f32]) {
    const BLOCK: usize = 8;
    let row_blocks = rows / BLOCK;
    let col_blocks = cols / BLOCK;

    for rb in 0..row_blocks {
        for cb in 0..col_blocks {
            let rs = rb * BLOCK;
            let cs = cb * BLOCK;
            transpose_region(a, b, rs..rs + BLOCK, cs..cs + BLOCK, cols, rows);
        }
    }

    let col_rem = col_blocks * BLOCK;
    if col_rem < cols {
        transpose_region(a, b, 0..row_blocks * BLOCK, col_rem..cols, cols, rows);
    }

    let row_rem = row_blocks * BLOCK;
    if row_rem < rows {
        transpose_region(a, b, row_rem..rows, 0..cols, cols, rows);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transpose_naive(rows: usize, cols: usize, a: &[f32], b: &mut [f32]) {
        for i in 0..rows {
            for j in 0..cols {
                b[j * rows + i] = a[i * cols + j];
            }
        }
    }

    /// FALSIFY-TP-001: Element correctness
    #[test]
    fn test_element_correctness() {
        for (rows, cols) in [(4, 5), (8, 8), (16, 32), (31, 17), (64, 64)] {
            let a: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
            let mut b = vec![0.0f32; rows * cols];
            transpose(rows, cols, &a, &mut b).unwrap();

            for i in 0..rows {
                for j in 0..cols {
                    assert_eq!(b[j * rows + i], a[i * cols + j], "({i},{j}) {rows}×{cols}");
                }
            }
        }
    }

    /// FALSIFY-TP-002: Involution
    #[test]
    fn test_involution() {
        for (rows, cols) in [(7, 13), (16, 16), (33, 17), (64, 128)] {
            let a: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.1 + 0.37).collect();
            let mut b = vec![0.0f32; rows * cols];
            let mut c = vec![0.0f32; rows * cols];

            transpose(rows, cols, &a, &mut b).unwrap();
            transpose(cols, rows, &b, &mut c).unwrap();

            assert_eq!(a, c, "Involution failed for {rows}×{cols}");
        }
    }

    /// FALSIFY-TP-003: Non-8-aligned dimensions
    #[test]
    fn test_non_aligned() {
        for (rows, cols) in [(7, 13), (17, 3), (1, 32), (32, 1), (1, 1), (3, 3)] {
            let a: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
            let mut b_test = vec![0.0f32; rows * cols];
            let mut b_ref = vec![0.0f32; rows * cols];

            transpose(rows, cols, &a, &mut b_test).unwrap();
            transpose_naive(rows, cols, &a, &mut b_ref);

            assert_eq!(b_test, b_ref, "Mismatch for {rows}×{cols}");
        }
    }

    /// FALSIFY-TP-004: AVX2 vs scalar parity (bitwise exact)
    #[test]
    fn test_avx2_scalar_parity() {
        let rows = 2048;
        let cols = 128;
        let a: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.001).collect();
        let mut b_scalar = vec![0.0f32; rows * cols];
        let mut b_dispatch = vec![0.0f32; rows * cols];

        transpose_scalar_impl(rows, cols, &a, &mut b_scalar);
        transpose(rows, cols, &a, &mut b_dispatch).unwrap();

        assert_eq!(b_scalar, b_dispatch, "AVX2 vs scalar mismatch at 2048×128");
    }

    /// FALSIFY-TP-005: Identity matrix
    #[test]
    fn test_identity() {
        for n in [4, 8, 16, 32] {
            let mut a = vec![0.0f32; n * n];
            for i in 0..n {
                a[i * n + i] = 1.0;
            }
            let mut b = vec![0.0f32; n * n];
            transpose(n, n, &a, &mut b).unwrap();
            assert_eq!(a, b, "Identity not preserved for {n}×{n}");
        }
    }

    /// FALSIFY-TP-006: Attention shape (2048×128)
    #[test]
    fn test_attention_shape() {
        let rows = 2048;
        let cols = 128;
        let a: Vec<f32> =
            (0..rows * cols).map(|i| ((i * 17 + 31) % 1000) as f32 / 1000.0 - 0.5).collect();
        let mut b_test = vec![0.0f32; rows * cols];
        let mut b_ref = vec![0.0f32; rows * cols];

        transpose(rows, cols, &a, &mut b_test).unwrap();
        transpose_naive(rows, cols, &a, &mut b_ref);

        assert_eq!(b_test, b_ref, "Attention shape 2048×128 mismatch");
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = vec![1.0f32; 12];
        let mut b = vec![0.0f32; 10]; // wrong size
        assert!(transpose(3, 4, &a, &mut b).is_err());
    }

    #[test]
    fn test_small_matrix() {
        // Below 64 elements threshold — uses scalar directly
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut b = vec![0.0f32; 6];
        transpose(2, 3, &a, &mut b).unwrap();
        assert_eq!(b, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
