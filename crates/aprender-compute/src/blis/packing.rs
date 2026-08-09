//! BLIS matrix packing routines.
//!
//! Packing transforms row-major matrices into micro-panel layouts optimized
//! for sequential access in the microkernel, ensuring optimal cache line
//! utilization and aligned loads for SIMD.
//!
//! # References
//!
//! - Van Zee, F. G., & Van de Geijn, R. A. (2015). BLIS: A Framework for Rapidly Instantiating
//!   BLAS Functionality. ACM TOMS, 41(3), Fig. 4.

use super::{MR, NR};
#[cfg(target_arch = "x86_64")]
use super::{MR_512, NR_512};

/// Pack A into MC x KC panel with MR-aligned micro-panels
///
/// Memory layout (Van Zee & Van de Geijn, 2015, Fig. 4):
/// Original A (row-major):     Packed A (column-major micro-panels):
/// [a00 a01 a02 ...]           [a00 a10 a20 ... a(MR-1)0 | a01 a11 ...]
/// [a10 a11 a12 ...]            \____ MR elements ____/
///
/// This layout ensures:
/// 1. Sequential access in the microkernel
/// 2. Optimal cache line utilization
/// 3. Aligned loads for SIMD
pub fn pack_a(
    a: &[f32],
    lda: usize, // Leading dimension of A (number of columns in original)
    mc: usize,  // Number of rows to pack
    kc: usize,  // Number of columns to pack
    packed: &mut [f32],
) {
    let mut pack_idx = 0;

    // Process MR rows at a time
    let full_panels = mc / MR;
    let remainder = mc % MR;

    for panel in 0..full_panels {
        let row_start = panel * MR;

        for col in 0..kc {
            for row in 0..MR {
                packed[pack_idx] = a[(row_start + row) * lda + col];
                pack_idx += 1;
            }
        }
    }

    // Handle remainder rows (pad with zeros)
    if remainder > 0 {
        let row_start = full_panels * MR;

        for col in 0..kc {
            for row in 0..MR {
                if row < remainder {
                    packed[pack_idx] = a[(row_start + row) * lda + col];
                } else {
                    packed[pack_idx] = 0.0; // Zero padding
                }
                pack_idx += 1;
            }
        }
    }
}

/// Pack B into KC x NC panel with NR-aligned micro-panels
///
/// Memory layout:
/// Original B (row-major):     Packed B (row-major micro-panels):
/// [b00 b01 b02 ...]           [b00 b01 ... b(NR-1) | b10 b11 ...]
/// [b10 b11 b12 ...]            \____ NR elements ____/
pub fn pack_b(
    b: &[f32],
    ldb: usize, // Leading dimension of B (number of columns in original)
    kc: usize,  // Number of rows to pack
    nc: usize,  // Number of columns to pack
    packed: &mut [f32],
) {
    let mut pack_idx = 0;

    let full_panels = nc / NR;
    let remainder = nc % NR;

    for panel in 0..full_panels {
        let col_start = panel * NR;

        for row in 0..kc {
            for col in 0..NR {
                packed[pack_idx] = b[row * ldb + col_start + col];
                pack_idx += 1;
            }
        }
    }

    // Handle remainder columns (pad with zeros)
    if remainder > 0 {
        let col_start = full_panels * NR;

        for row in 0..kc {
            for col in 0..NR {
                if col < remainder {
                    packed[pack_idx] = b[row * ldb + col_start + col];
                } else {
                    packed[pack_idx] = 0.0;
                }
                pack_idx += 1;
            }
        }
    }
}

/// Compute required packed A buffer size
#[inline]
pub fn packed_a_size(mc: usize, kc: usize) -> usize {
    let panels = (mc + MR - 1) / MR;
    panels * MR * kc
}

/// Compute required packed B buffer size
#[inline]
pub fn packed_b_size(kc: usize, nc: usize) -> usize {
    let panels = (nc + NR - 1) / NR;
    panels * NR * kc
}

/// Pack A block from row-major source.
/// Full MR=8 panels use AVX2 8×8 transpose when available (8× fewer scalar ops).
pub(super) fn pack_a_block(
    a: &[f32],
    lda: usize,
    row_start: usize,
    col_start: usize,
    rows: usize,
    cols: usize,
    packed: &mut [f32],
) {
    let panels = (rows + MR - 1) / MR;

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 verified above
        unsafe {
            pack_a_block_avx2(a, lda, row_start, col_start, rows, cols, panels, packed);
        }
        return;
    }

    // Scalar fallback
    let mut pack_idx = 0;
    for panel in 0..panels {
        let ir = panel * MR;
        let mr_actual = MR.min(rows - ir);

        for col in 0..cols {
            for row in 0..MR {
                if row < mr_actual {
                    packed[pack_idx] = a[(row_start + ir + row) * lda + col_start + col];
                } else {
                    packed[pack_idx] = 0.0;
                }
                pack_idx += 1;
            }
        }
    }
}

/// AVX2 SIMD-accelerated A packing: 8×8 transpose blocks.
/// Processes 8 columns at a time using AVX2 unpack/shuffle/permute
/// to transpose 8×8 blocks from row-major to column-major in-register.
/// Reduces per-element cost from 1 scalar load + 1 scalar store to
/// ~0.25 SIMD ops per element (8× improvement).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pack_a_block_avx2(
    a: &[f32],
    lda: usize,
    row_start: usize,
    col_start: usize,
    rows: usize,
    cols: usize,
    panels: usize,
    packed: &mut [f32],
) {
    use std::arch::x86_64::*;

    let mut pack_idx = 0;

    for panel in 0..panels {
        let ir = panel * MR;
        let mr_actual = MR.min(rows - ir);

        if mr_actual == MR {
            // Full panel: 8×8 SIMD transpose blocks
            let k_blocks = cols / 8;
            let k_rem_start = k_blocks * 8;

            for kb in 0..k_blocks {
                let p = kb * 8;
                let base = row_start + ir;
                let col = col_start + p;

                // SAFETY: AVX2 verified by caller. Pointers are within bounds
                // (base+7 < row_start+rows, col+7 < col_start+cols).
                unsafe {
                    let r0 = _mm256_loadu_ps(a.as_ptr().add(base * lda + col));
                    let r1 = _mm256_loadu_ps(a.as_ptr().add((base + 1) * lda + col));
                    let r2 = _mm256_loadu_ps(a.as_ptr().add((base + 2) * lda + col));
                    let r3 = _mm256_loadu_ps(a.as_ptr().add((base + 3) * lda + col));
                    let r4 = _mm256_loadu_ps(a.as_ptr().add((base + 4) * lda + col));
                    let r5 = _mm256_loadu_ps(a.as_ptr().add((base + 5) * lda + col));
                    let r6 = _mm256_loadu_ps(a.as_ptr().add((base + 6) * lda + col));
                    let r7 = _mm256_loadu_ps(a.as_ptr().add((base + 7) * lda + col));

                    // 8×8 transpose via unpack + shuffle + permute2f128
                    let t0 = _mm256_unpacklo_ps(r0, r1);
                    let t1 = _mm256_unpackhi_ps(r0, r1);
                    let t2 = _mm256_unpacklo_ps(r2, r3);
                    let t3 = _mm256_unpackhi_ps(r2, r3);
                    let t4 = _mm256_unpacklo_ps(r4, r5);
                    let t5 = _mm256_unpackhi_ps(r4, r5);
                    let t6 = _mm256_unpacklo_ps(r6, r7);
                    let t7 = _mm256_unpackhi_ps(r6, r7);

                    let u0 = _mm256_shuffle_ps(t0, t2, 0x44);
                    let u1 = _mm256_shuffle_ps(t0, t2, 0xEE);
                    let u2 = _mm256_shuffle_ps(t1, t3, 0x44);
                    let u3 = _mm256_shuffle_ps(t1, t3, 0xEE);
                    let u4 = _mm256_shuffle_ps(t4, t6, 0x44);
                    let u5 = _mm256_shuffle_ps(t4, t6, 0xEE);
                    let u6 = _mm256_shuffle_ps(t5, t7, 0x44);
                    let u7 = _mm256_shuffle_ps(t5, t7, 0xEE);

                    let dst = packed.as_mut_ptr().add(pack_idx);
                    _mm256_storeu_ps(dst, _mm256_permute2f128_ps(u0, u4, 0x20));
                    _mm256_storeu_ps(dst.add(8), _mm256_permute2f128_ps(u1, u5, 0x20));
                    _mm256_storeu_ps(dst.add(16), _mm256_permute2f128_ps(u2, u6, 0x20));
                    _mm256_storeu_ps(dst.add(24), _mm256_permute2f128_ps(u3, u7, 0x20));
                    _mm256_storeu_ps(dst.add(32), _mm256_permute2f128_ps(u0, u4, 0x31));
                    _mm256_storeu_ps(dst.add(40), _mm256_permute2f128_ps(u1, u5, 0x31));
                    _mm256_storeu_ps(dst.add(48), _mm256_permute2f128_ps(u2, u6, 0x31));
                    _mm256_storeu_ps(dst.add(56), _mm256_permute2f128_ps(u3, u7, 0x31));
                }
                pack_idx += 64;
            }

            // Remainder columns: scalar
            for col in k_rem_start..cols {
                for row in 0..MR {
                    packed[pack_idx] = a[(row_start + ir + row) * lda + col_start + col];
                    pack_idx += 1;
                }
            }
        } else {
            // Edge panel: scalar with zero-pad
            for col in 0..cols {
                for row in 0..MR {
                    if row < mr_actual {
                        packed[pack_idx] = a[(row_start + ir + row) * lda + col_start + col];
                    } else {
                        packed[pack_idx] = 0.0;
                    }
                    pack_idx += 1;
                }
            }
        }
    }
}

/// Pack B block from row-major source
pub(super) fn pack_b_block(
    b: &[f32],
    ldb: usize,
    row_start: usize,
    col_start: usize,
    rows: usize,
    cols: usize,
    packed: &mut [f32],
) {
    let mut pack_idx = 0;
    let panels = (cols + NR - 1) / NR;

    for panel in 0..panels {
        let jr = panel * NR;
        let nr_actual = NR.min(cols - jr);

        for row in 0..rows {
            for col in 0..NR {
                if col < nr_actual {
                    packed[pack_idx] = b[(row_start + row) * ldb + col_start + jr + col];
                } else {
                    packed[pack_idx] = 0.0;
                }
                pack_idx += 1;
            }
        }
    }
}

// ============================================================================
// AVX-512 packing (MR=16, NR=8)
// ============================================================================

/// Compute required packed A buffer size for AVX-512
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn packed_a_size_512(mc: usize, kc: usize) -> usize {
    let panels = (mc + MR_512 - 1) / MR_512;
    panels * MR_512 * kc
}

/// Compute required packed B buffer size for AVX-512
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn packed_b_size_512(kc: usize, nc: usize) -> usize {
    let panels = (nc + NR_512 - 1) / NR_512;
    panels * NR_512 * kc
}

/// Pack A block with MR_512=16 micro-panels for AVX-512 microkernel.
/// A is row-major; packed A is column-major micro-panels of width MR_512=16.
/// Full panels (mr_actual == 16): SIMD gather from 16 rows via scalar
/// (gather is limited by row stride), but the inner loop is tight.
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)] // Used by gemm_blis_avx512_packed (retained for AVX-512-only systems)
pub(super) fn pack_a_block_512(
    a: &[f32],
    lda: usize,
    row_start: usize,
    col_start: usize,
    rows: usize,
    cols: usize,
    packed: &mut [f32],
) {
    let mut pack_idx = 0;
    let panels = (rows + MR_512 - 1) / MR_512;

    for panel in 0..panels {
        let ir = panel * MR_512;
        let mr_actual = MR_512.min(rows - ir);

        if mr_actual == MR_512 {
            // Full panel: no zero-padding needed
            for col in 0..cols {
                for row in 0..MR_512 {
                    packed[pack_idx + row] = a[(row_start + ir + row) * lda + col_start + col];
                }
                pack_idx += MR_512;
            }
        } else {
            // Edge panel: zero-pad
            for col in 0..cols {
                for row in 0..mr_actual {
                    packed[pack_idx + row] = a[(row_start + ir + row) * lda + col_start + col];
                }
                for row in mr_actual..MR_512 {
                    packed[pack_idx + row] = 0.0;
                }
                pack_idx += MR_512;
            }
        }
    }
}

/// Pack B block with NR_512=8 micro-panels for AVX-512 microkernel.
/// B is row-major; each NR_512=8 column slice is contiguous → SIMD copy.
/// Uses AVX2 _mm256_loadu_ps / _mm256_storeu_ps for full panels (8 f32 = 32B).
#[cfg(target_arch = "x86_64")]
pub(super) fn pack_b_block_512(
    b: &[f32],
    ldb: usize,
    row_start: usize,
    col_start: usize,
    rows: usize,
    cols: usize,
    packed: &mut [f32],
) {
    let panels = (cols + NR_512 - 1) / NR_512;
    let use_simd = is_x86_feature_detected!("avx2");

    for panel in 0..panels {
        let jr = panel * NR_512;
        let nr_actual = NR_512.min(cols - jr);
        let dst_base = panel * NR_512 * rows;

        if nr_actual == NR_512 && use_simd {
            // Full panel: SIMD 8-wide copy per row
            // SAFETY: AVX2 verified above, src/dst aligned to f32.
            unsafe {
                use std::arch::x86_64::*;
                for row in 0..rows {
                    let src = b.as_ptr().add((row_start + row) * ldb + col_start + jr);
                    let dst = packed.as_mut_ptr().add(dst_base + row * NR_512);
                    _mm256_storeu_ps(dst, _mm256_loadu_ps(src));
                }
            }
        } else {
            // Edge panel or no SIMD: scalar with zero-pad
            let mut pack_idx = dst_base;
            for row in 0..rows {
                for col in 0..NR_512 {
                    if col < nr_actual {
                        packed[pack_idx] = b[(row_start + row) * ldb + col_start + jr + col];
                    } else {
                        packed[pack_idx] = 0.0;
                    }
                    pack_idx += 1;
                }
            }
        }
    }
}

// ============================================================================
// 32×6 Packing (Phase 4, Appendix D)
// ============================================================================

#[allow(unused_imports)]
use super::{MR_512V2, NR_512V2};

/// Compute required packed A buffer size for 32×6 microkernel.
#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(dead_code)] // Reserved for column-major 32×6 BLIS path
pub fn packed_a_size_v2(mc: usize, kc: usize) -> usize {
    let panels = (mc + MR_512V2 - 1) / MR_512V2;
    panels * MR_512V2 * kc
}

/// Compute required packed B buffer size for 32×6 microkernel.
#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(dead_code)] // Reserved for column-major 32×6 BLIS path
pub fn packed_b_size_v2(kc: usize, nc: usize) -> usize {
    let panels = (nc + NR_512V2 - 1) / NR_512V2;
    panels * NR_512V2 * kc
}

/// Pack A block with MR_512V2=32 micro-panels for 32×6 microkernel.
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)] // Reserved for column-major 32×6 BLIS path
pub(super) fn pack_a_block_v2(
    a: &[f32],
    lda: usize,
    row_start: usize,
    col_start: usize,
    rows: usize,
    cols: usize,
    packed: &mut [f32],
) {
    let mut pack_idx = 0;
    let panels = (rows + MR_512V2 - 1) / MR_512V2;

    for panel in 0..panels {
        let ir = panel * MR_512V2;
        let mr_actual = MR_512V2.min(rows - ir);

        for col in 0..cols {
            for row in 0..mr_actual {
                packed[pack_idx + row] = a[(row_start + ir + row) * lda + col_start + col];
            }
            for row in mr_actual..MR_512V2 {
                packed[pack_idx + row] = 0.0;
            }
            pack_idx += MR_512V2;
        }
    }
}

/// Pack B block with NR_512V2=6 micro-panels for 32×6 microkernel.
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)] // Reserved for column-major 32×6 BLIS path
pub(super) fn pack_b_block_v2(
    b: &[f32],
    ldb: usize,
    row_start: usize,
    col_start: usize,
    rows: usize,
    cols: usize,
    packed: &mut [f32],
) {
    let panels = (cols + NR_512V2 - 1) / NR_512V2;

    for panel in 0..panels {
        let jr = panel * NR_512V2;
        let nr_actual = NR_512V2.min(cols - jr);
        let dst_base = panel * NR_512V2 * rows;

        for row in 0..rows {
            let pack_idx = dst_base + row * NR_512V2;
            for col in 0..nr_actual {
                packed[pack_idx + col] = b[(row_start + row) * ldb + col_start + jr + col];
            }
            for col in nr_actual..NR_512V2 {
                packed[pack_idx + col] = 0.0;
            }
        }
    }
}
