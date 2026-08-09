// Rust 1.93+: BLIS microkernels use bare unsafe ops inside `unsafe fn`.
// Wrapping each intrinsic in `unsafe {}` would add 100+ blocks with no safety benefit.
#![allow(unsafe_op_in_unsafe_fn)]
//! Core BLIS compute routines: microkernel dispatch, macroblock execution,
//! and the cache-blocked GEMM main loop.
//!
//! Implements the 5-loop BLIS algorithm (Van Zee & Van de Geijn, 2015):
//! - Loop 5 (jc): N dimension, L3 blocking
//! - Loop 4 (pc): K dimension, L2 blocking
//! - Loop 3 (ic): M dimension, L1 blocking
//! - Loop 2 (jr): Microkernel columns
//! - Loop 1 (ir): Microkernel rows

use std::cell::RefCell;
use std::time::Instant;

use crate::error::TruenoError;

#[cfg(target_arch = "x86_64")]
use super::microkernels::microkernel_16x8_avx512;
#[cfg(target_arch = "x86_64")]
use super::microkernels::microkernel_8x6_true_asm;
use super::microkernels::microkernel_scalar;
use super::packing::{pack_a_block, pack_b_block, packed_a_size, packed_b_size};
#[cfg(target_arch = "x86_64")]
use super::packing::{pack_a_block_512, pack_b_block_512, packed_a_size_512, packed_b_size_512};
use super::prepacked::PrepackedB;
use super::profiler::{BlisProfileLevel, BlisProfiler};
use super::reference::gemm_reference;
use super::{KC, MC, MR, NC, NR};
#[cfg(target_arch = "x86_64")]
use super::{KC_512, MC_512, MR_512, NC_512, NR_512};

// Thread-local workspace buffers to eliminate allocation churn in gemm_blis.
// These grow to the high-water mark and are reused across calls, avoiding
// ~4.3 MB of allocation+deallocation per GEMM invocation.
thread_local! {
    static TL_PACKED_A: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static TL_PACKED_B: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static TL_C_MICRO: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Load a tile of C into the micro workspace for accumulation.
#[inline(always)]
fn load_c_tile(
    c: &[f32],
    c_micro: &mut [f32],
    row: usize,
    col: usize,
    mr: usize,
    nr: usize,
    n: usize,
) {
    for jj in 0..nr {
        for ii in 0..mr {
            c_micro[jj * MR + ii] = c[(row + ii) * n + (col + jj)];
        }
        for ii in mr..MR {
            c_micro[jj * MR + ii] = 0.0;
        }
    }
    for jj in nr..NR {
        for ii in 0..MR {
            c_micro[jj * MR + ii] = 0.0;
        }
    }
}

/// Store a micro tile back into C.
#[inline(always)]
fn store_c_tile(
    c: &mut [f32],
    c_micro: &[f32],
    row: usize,
    col: usize,
    mr: usize,
    nr: usize,
    n: usize,
) {
    for jj in 0..nr {
        for ii in 0..mr {
            c[(row + ii) * n + (col + jj)] = c_micro[jj * MR + ii];
        }
    }
}

/// Dispatch to the best available microkernel (AVX2 ASM or scalar fallback).
#[inline(always)]
fn dispatch_microkernel(
    kc: usize,
    a_panel: &[f32],
    b_panel: &[f32],
    c_micro: &mut [f32],
    _mr_block: usize,
    _nr_block: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && mr_block == MR
            && nr_block == NR
        {
            // SAFETY: AVX2+FMA verified by is_x86_feature_detected!() above.
            unsafe {
                microkernel_8x6_true_asm(
                    kc,
                    a_panel.as_ptr(),
                    b_panel.as_ptr(),
                    c_micro.as_mut_ptr(),
                    MR,
                );
            }
            return;
        }
    }
    microkernel_scalar(kc, a_panel, b_panel, c_micro, MR);
}

/// Execute microkernel tile iterations over one MC x NC x KC macro-block.
#[allow(clippy::too_many_arguments)]
fn compute_macroblock(
    c: &mut [f32],
    packed_a: &[f32],
    packed_b: &[f32],
    c_micro: &mut [f32],
    ic: usize,
    jc: usize,
    mc_block: usize,
    nc_block: usize,
    kc_block: usize,
    n: usize,
    profiler: &mut Option<&mut BlisProfiler>,
) {
    // KAIZEN-038: Avoid Instant::now() syscall (~20-40ns) when profiler is disabled.
    // For 1024x1024 GEMM, this eliminates thousands of syscalls per macroblock.
    let track_time = profiler.is_some();
    let midi_start = if track_time { Some(Instant::now()) } else { None };

    for ir in (0..mc_block).step_by(MR) {
        let mr_block = MR.min(mc_block - ir);
        for jr in (0..nc_block).step_by(NR) {
            let nr_block = NR.min(nc_block - jr);
            let micro_start = if track_time { Some(Instant::now()) } else { None };

            let a_panel = &packed_a[(ir / MR) * MR * kc_block..];
            let b_panel = &packed_b[(jr / NR) * NR * kc_block..];

            load_c_tile(c, c_micro, ic + ir, jc + jr, mr_block, nr_block, n);
            dispatch_microkernel(kc_block, a_panel, b_panel, c_micro, mr_block, nr_block);
            store_c_tile(c, c_micro, ic + ir, jc + jr, mr_block, nr_block, n);

            if let (Some(ref mut prof), Some(start)) = (profiler.as_deref_mut(), micro_start) {
                prof.record(
                    BlisProfileLevel::Micro,
                    start.elapsed().as_nanos() as u64,
                    (2 * mr_block * nr_block * kc_block) as u64,
                );
            }
        }
    }

    if let (Some(ref mut prof), Some(start)) = (profiler.as_deref_mut(), midi_start) {
        prof.record(
            BlisProfileLevel::Midi,
            start.elapsed().as_nanos() as u64,
            (2 * mc_block * nc_block * kc_block) as u64,
        );
    }
}

/// Zero-pack row-major GEMM for small matrices (≤128).
///
/// Key design: NO packing of A or B, NO C layout conversion.
/// - A: broadcast scalar elements directly from row-major layout
/// - B: SIMD load contiguous rows directly from row-major layout
/// - C: SIMD load/store of contiguous rows (row-major accumulation)
///
/// This eliminates ~2µs of overhead for 64×64 GEMM:
/// - No heap allocation for packed buffers
/// - No SIMD transpose for A packing
/// - No scalar C load/store (128 ops → 16 SIMD ops per tile)
///
/// Inner loop: for each K, load 1 B row (8 f32), broadcast 8 A elements,
/// execute 8 FMAs into 8 C row accumulators. 4-way K unrolled.
///
/// Reference: Goto & Van de Geijn (2008), "Anatomy of High-Performance
/// Matrix Multiplication" — panel-panel multiply with register blocking.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn gemm_direct_rowmajor(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Result<(), TruenoError> {
    use std::arch::x86_64::*;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_mut_ptr();

    unsafe {
        for ir in (0..m).step_by(8) {
            for jr in (0..n).step_by(8) {
                // Load 8 rows of C (each row is contiguous in row-major)
                let c_base = c_ptr.add(ir * n + jr);
                let mut c0 = _mm256_loadu_ps(c_base);
                let mut c1 = _mm256_loadu_ps(c_base.add(n));
                let mut c2 = _mm256_loadu_ps(c_base.add(2 * n));
                let mut c3 = _mm256_loadu_ps(c_base.add(3 * n));
                let mut c4 = _mm256_loadu_ps(c_base.add(4 * n));
                let mut c5 = _mm256_loadu_ps(c_base.add(5 * n));
                let mut c6 = _mm256_loadu_ps(c_base.add(6 * n));
                let mut c7 = _mm256_loadu_ps(c_base.add(7 * n));

                // A row base pointers (stride = k between columns)
                let a0 = a_ptr.add(ir * k);
                let a1 = a_ptr.add((ir + 1) * k);
                let a2 = a_ptr.add((ir + 2) * k);
                let a3 = a_ptr.add((ir + 3) * k);
                let a4 = a_ptr.add((ir + 4) * k);
                let a5 = a_ptr.add((ir + 5) * k);
                let a6 = a_ptr.add((ir + 6) * k);
                let a7 = a_ptr.add((ir + 7) * k);

                // B base (stride = n between K rows)
                let b_base = b_ptr.add(jr);

                // 4-way K-unrolled main loop
                let k4 = k / 4;
                let k_rem = k % 4;

                for p4 in 0..k4 {
                    let p = p4 * 4;

                    // K+0
                    let b_row = _mm256_loadu_ps(b_base.add(p * n));
                    c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a0.add(p)), b_row, c0);
                    c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a1.add(p)), b_row, c1);
                    c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a2.add(p)), b_row, c2);
                    c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a3.add(p)), b_row, c3);
                    c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a4.add(p)), b_row, c4);
                    c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a5.add(p)), b_row, c5);
                    c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a6.add(p)), b_row, c6);
                    c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a7.add(p)), b_row, c7);

                    // K+1
                    let b_row = _mm256_loadu_ps(b_base.add((p + 1) * n));
                    c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a0.add(p + 1)), b_row, c0);
                    c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a1.add(p + 1)), b_row, c1);
                    c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a2.add(p + 1)), b_row, c2);
                    c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a3.add(p + 1)), b_row, c3);
                    c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a4.add(p + 1)), b_row, c4);
                    c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a5.add(p + 1)), b_row, c5);
                    c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a6.add(p + 1)), b_row, c6);
                    c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a7.add(p + 1)), b_row, c7);

                    // K+2
                    let b_row = _mm256_loadu_ps(b_base.add((p + 2) * n));
                    c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a0.add(p + 2)), b_row, c0);
                    c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a1.add(p + 2)), b_row, c1);
                    c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a2.add(p + 2)), b_row, c2);
                    c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a3.add(p + 2)), b_row, c3);
                    c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a4.add(p + 2)), b_row, c4);
                    c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a5.add(p + 2)), b_row, c5);
                    c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a6.add(p + 2)), b_row, c6);
                    c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a7.add(p + 2)), b_row, c7);

                    // K+3
                    let b_row = _mm256_loadu_ps(b_base.add((p + 3) * n));
                    c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a0.add(p + 3)), b_row, c0);
                    c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a1.add(p + 3)), b_row, c1);
                    c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a2.add(p + 3)), b_row, c2);
                    c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a3.add(p + 3)), b_row, c3);
                    c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a4.add(p + 3)), b_row, c4);
                    c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a5.add(p + 3)), b_row, c5);
                    c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a6.add(p + 3)), b_row, c6);
                    c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a7.add(p + 3)), b_row, c7);
                }

                // Remainder
                let base_rem = k4 * 4;
                for rp in 0..k_rem {
                    let pp = base_rem + rp;
                    let b_row = _mm256_loadu_ps(b_base.add(pp * n));
                    c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a0.add(pp)), b_row, c0);
                    c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a1.add(pp)), b_row, c1);
                    c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a2.add(pp)), b_row, c2);
                    c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a3.add(pp)), b_row, c3);
                    c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a4.add(pp)), b_row, c4);
                    c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a5.add(pp)), b_row, c5);
                    c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a6.add(pp)), b_row, c6);
                    c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a7.add(pp)), b_row, c7);
                }

                // Store 8 rows of C (contiguous SIMD stores)
                _mm256_storeu_ps(c_base, c0);
                _mm256_storeu_ps(c_base.add(n), c1);
                _mm256_storeu_ps(c_base.add(2 * n), c2);
                _mm256_storeu_ps(c_base.add(3 * n), c3);
                _mm256_storeu_ps(c_base.add(4 * n), c4);
                _mm256_storeu_ps(c_base.add(5 * n), c5);
                _mm256_storeu_ps(c_base.add(6 * n), c6);
                _mm256_storeu_ps(c_base.add(7 * n), c7);
            }
        }
    }
    Ok(())
}

/// Small-matrix stride-based GEMM — no packing, no c_micro buffer.
/// For m,n,k <= 96 where packing overhead > cache benefit.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn gemm_small_strided_avx2(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Result<(), TruenoError> {
    use std::arch::x86_64::*;
    unsafe {
        for jr in (0..n).step_by(NR) {
            let nr = NR.min(n - jr);
            for ir in (0..m).step_by(MR) {
                let mr = MR.min(m - ir);
                let mut cv = [_mm256_setzero_ps(); 6];
                for j in 0..nr {
                    if mr == MR {
                        cv[j] = _mm256_set_ps(
                            *c.get_unchecked((ir + 7) * n + jr + j),
                            *c.get_unchecked((ir + 6) * n + jr + j),
                            *c.get_unchecked((ir + 5) * n + jr + j),
                            *c.get_unchecked((ir + 4) * n + jr + j),
                            *c.get_unchecked((ir + 3) * n + jr + j),
                            *c.get_unchecked((ir + 2) * n + jr + j),
                            *c.get_unchecked((ir + 1) * n + jr + j),
                            *c.get_unchecked(ir * n + jr + j),
                        );
                    } else {
                        let mut t = [0.0f32; 8];
                        for i in 0..mr {
                            t[i] = *c.get_unchecked((ir + i) * n + jr + j);
                        }
                        cv[j] = _mm256_loadu_ps(t.as_ptr());
                    }
                }
                for p in 0..k {
                    let a_col = if mr == MR {
                        _mm256_set_ps(
                            *a.get_unchecked((ir + 7) * k + p),
                            *a.get_unchecked((ir + 6) * k + p),
                            *a.get_unchecked((ir + 5) * k + p),
                            *a.get_unchecked((ir + 4) * k + p),
                            *a.get_unchecked((ir + 3) * k + p),
                            *a.get_unchecked((ir + 2) * k + p),
                            *a.get_unchecked((ir + 1) * k + p),
                            *a.get_unchecked(ir * k + p),
                        )
                    } else {
                        let mut t = [0.0f32; 8];
                        for i in 0..mr {
                            t[i] = *a.get_unchecked((ir + i) * k + p);
                        }
                        _mm256_loadu_ps(t.as_ptr())
                    };
                    let bp = b.as_ptr().add(p * n + jr);
                    // Unrolled FMA for NR=6 common case
                    if nr == NR {
                        cv[0] = _mm256_fmadd_ps(a_col, _mm256_set1_ps(*bp), cv[0]);
                        cv[1] = _mm256_fmadd_ps(a_col, _mm256_set1_ps(*bp.add(1)), cv[1]);
                        cv[2] = _mm256_fmadd_ps(a_col, _mm256_set1_ps(*bp.add(2)), cv[2]);
                        cv[3] = _mm256_fmadd_ps(a_col, _mm256_set1_ps(*bp.add(3)), cv[3]);
                        cv[4] = _mm256_fmadd_ps(a_col, _mm256_set1_ps(*bp.add(4)), cv[4]);
                        cv[5] = _mm256_fmadd_ps(a_col, _mm256_set1_ps(*bp.add(5)), cv[5]);
                    } else {
                        for j in 0..nr {
                            cv[j] = _mm256_fmadd_ps(a_col, _mm256_set1_ps(*bp.add(j)), cv[j]);
                        }
                    }
                }
                for j in 0..nr {
                    let mut t = [0.0f32; 8];
                    _mm256_storeu_ps(t.as_mut_ptr(), cv[j]);
                    for i in 0..mr {
                        *c.get_unchecked_mut((ir + i) * n + jr + j) = t[i];
                    }
                }
            }
        }
    }
    Ok(())
}

/// 8x8 GEMM: pre-packed B, SIMD transpose A packing, K-unrolled micro-kernel.
/// For m,n divisible by 8 and m,n,k ≤ 256.
///
/// Key optimization: B panels are packed ONCE before the tile loop, eliminating
/// panels_m redundant repacking passes. For 128x128 this removes 15/16 = 94%
/// of B packing work.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn gemm_small_nopack_8x8(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Result<(), TruenoError> {
    use crate::blis::microkernels::microkernel_8x8_avx2_fma;
    use std::arch::x86_64::*;

    let panels_m = m / 8;
    let panels_n = n / 8;

    // Stack buffer for one A panel (8×K col-major).
    let mut packed_a = [0.0f32; 8 * 256];
    let mut c_micro = [0.0f32; 64];

    // Pre-pack ALL B panels once (eliminates panels_m redundant repacking).
    // For 256x256: 32 panels × 256 × 8 = 256KB — acceptable heap alloc.
    let mut all_packed_b = vec![0.0f32; panels_n * k * 8];

    unsafe {
        for jr_panel in 0..panels_n {
            let jr = jr_panel * 8;
            let b_dst = all_packed_b.as_mut_ptr().add(jr_panel * k * 8);
            for p in 0..k {
                _mm256_storeu_ps(b_dst.add(p * 8), _mm256_loadu_ps(b.as_ptr().add(p * n + jr)));
            }
        }

        for ir_panel in 0..panels_m {
            let ir = ir_panel * 8;

            // Pack A panel: SIMD 8×8 transpose blocks (row-major → col-major)
            let k_blocks = k / 8;
            let k_rem = k_blocks * 8;
            for kb in 0..k_blocks {
                let p = kb * 8;
                let r0 = _mm256_loadu_ps(a.as_ptr().add(ir * k + p));
                let r1 = _mm256_loadu_ps(a.as_ptr().add((ir + 1) * k + p));
                let r2 = _mm256_loadu_ps(a.as_ptr().add((ir + 2) * k + p));
                let r3 = _mm256_loadu_ps(a.as_ptr().add((ir + 3) * k + p));
                let r4 = _mm256_loadu_ps(a.as_ptr().add((ir + 4) * k + p));
                let r5 = _mm256_loadu_ps(a.as_ptr().add((ir + 5) * k + p));
                let r6 = _mm256_loadu_ps(a.as_ptr().add((ir + 6) * k + p));
                let r7 = _mm256_loadu_ps(a.as_ptr().add((ir + 7) * k + p));

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

                let dst = packed_a.as_mut_ptr().add(p * 8);
                _mm256_storeu_ps(dst, _mm256_permute2f128_ps(u0, u4, 0x20));
                _mm256_storeu_ps(dst.add(8), _mm256_permute2f128_ps(u1, u5, 0x20));
                _mm256_storeu_ps(dst.add(16), _mm256_permute2f128_ps(u2, u6, 0x20));
                _mm256_storeu_ps(dst.add(24), _mm256_permute2f128_ps(u3, u7, 0x20));
                _mm256_storeu_ps(dst.add(32), _mm256_permute2f128_ps(u0, u4, 0x31));
                _mm256_storeu_ps(dst.add(40), _mm256_permute2f128_ps(u1, u5, 0x31));
                _mm256_storeu_ps(dst.add(48), _mm256_permute2f128_ps(u2, u6, 0x31));
                _mm256_storeu_ps(dst.add(56), _mm256_permute2f128_ps(u3, u7, 0x31));
            }
            for p in k_rem..k {
                for i in 0..8 {
                    *packed_a.get_unchecked_mut(p * 8 + i) = *a.get_unchecked((ir + i) * k + p);
                }
            }

            for jr_panel in 0..panels_n {
                let jr = jr_panel * 8;
                let packed_b_ptr = all_packed_b.as_ptr().add(jr_panel * k * 8);

                // Load C tile (column-major for micro-kernel)
                for jj in 0..8 {
                    for ii in 0..8 {
                        c_micro[jj * 8 + ii] = *c.get_unchecked((ir + ii) * n + jr + jj);
                    }
                }

                microkernel_8x8_avx2_fma(
                    k,
                    packed_a.as_ptr(),
                    packed_b_ptr,
                    c_micro.as_mut_ptr(),
                    8,
                );

                // Store C tile back to row-major
                for jj in 0..8 {
                    for ii in 0..8 {
                        *c.get_unchecked_mut((ir + ii) * n + jr + jj) = c_micro[jj * 8 + ii];
                    }
                }
            }
        }
    }
    Ok(())
}

/// Small-matrix 8x8 GEMM — stack-packed A/B, striped 8x8 AVX2 kernel.
/// Fewer tiles than 8x6 (64 outputs vs 48 per tile = 33% fewer tiles).
/// For dimensions that are multiples of 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(dead_code)] // Superseded by gemm_small_nopack_8x8 but retained for profiling comparison
unsafe fn gemm_small_8x8(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Result<(), TruenoError> {
    use crate::blis::microkernels::microkernel_8x8_avx2_fma;
    // Stack pack buffers (max 128*128 = 64KB each for 128x128)
    let mut packed_a = vec![0.0f32; m * k];
    let mut packed_b = vec![0.0f32; k * n];
    let mut c_micro = [0.0f32; 8 * 8]; // 8x8 tile

    // Pack A: row-major a[i*k+p] → column-major packed_a[p*8 + (i%8)] per 8-row panel
    let panels_m = (m + 7) / 8;
    for panel in 0..panels_m {
        let ir = panel * 8;
        let mr = 8.min(m - ir);
        for p in 0..k {
            for i in 0..8 {
                unsafe {
                    packed_a[panel * 8 * k + p * 8 + i] =
                        if i < mr { *a.get_unchecked((ir + i) * k + p) } else { 0.0 };
                }
            }
        }
    }
    // Pack B: row-major b[p*n+j] → row-major packed_b[panel*8*k + p*8 + (j%8)]
    let panels_n = (n + 7) / 8;
    for panel in 0..panels_n {
        let jr = panel * 8;
        let nr = 8.min(n - jr);
        for p in 0..k {
            for j in 0..8 {
                unsafe {
                    packed_b[panel * 8 * k + p * 8 + j] =
                        if j < nr { *b.get_unchecked(p * n + jr + j) } else { 0.0 };
                }
            }
        }
    }

    // Run 8x8 micro-tiles
    unsafe {
        for ir_panel in 0..panels_m {
            let ir = ir_panel * 8;
            let mr = 8.min(m - ir);
            for jr_panel in 0..panels_n {
                let jr = jr_panel * 8;
                let nr = 8.min(n - jr);
                // Load C tile (8x8 column-major)
                for jj in 0..8 {
                    for ii in 0..8 {
                        c_micro[jj * 8 + ii] = if ii < mr && jj < nr {
                            *c.get_unchecked((ir + ii) * n + jr + jj)
                        } else {
                            0.0
                        };
                    }
                }
                let ap = packed_a.as_ptr().add(ir_panel * 8 * k);
                let bp = packed_b.as_ptr().add(jr_panel * 8 * k);
                microkernel_8x8_avx2_fma(k, ap, bp, c_micro.as_mut_ptr(), 8);
                // Store C tile
                for jj in 0..nr {
                    for ii in 0..mr {
                        *c.get_unchecked_mut((ir + ii) * n + jr + jj) = c_micro[jj * 8 + ii];
                    }
                }
            }
        }
    }
    Ok(())
}

/// AVX-512 small GEMM: 16×8 tiles, pre-packed B, scalar-packed A.
/// For m divisible by 16, n divisible by 8, m,n,k ≤ 256.
/// 2× vector width over AVX2 gives ~2× throughput on compute-bound tiles.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn gemm_small_avx512_16x8(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Result<(), TruenoError> {
    use super::microkernels::microkernel_16x8_avx512;
    use std::arch::x86_64::*;

    let panels_m = m / 16;
    let panels_n = n / 8;

    // Pre-pack ALL B panels once (SIMD 8-wide contiguous copies).
    let mut all_packed_b = vec![0.0f32; panels_n * k * 8];
    unsafe {
        for jr_panel in 0..panels_n {
            let jr = jr_panel * 8;
            let b_dst = all_packed_b.as_mut_ptr().add(jr_panel * k * 8);
            for p in 0..k {
                _mm256_storeu_ps(b_dst.add(p * 8), _mm256_loadu_ps(b.as_ptr().add(p * n + jr)));
            }
        }
    }

    // Stack buffers — A panel: 16×K column-major, C micro tile: 16×8.
    let mut packed_a = [0.0f32; 16 * 256];
    let mut c_micro = [0.0f32; 16 * 8];

    unsafe {
        for ir_panel in 0..panels_m {
            let ir = ir_panel * 16;

            // Pack A: row-major → column-major via two 8×8 SIMD transposes.
            let k_blocks = k / 8;
            let k_rem_start = k_blocks * 8;

            for kb in 0..k_blocks {
                let p = kb * 8;

                // Upper 8 rows
                let r0 = _mm256_loadu_ps(a.as_ptr().add(ir * k + p));
                let r1 = _mm256_loadu_ps(a.as_ptr().add((ir + 1) * k + p));
                let r2 = _mm256_loadu_ps(a.as_ptr().add((ir + 2) * k + p));
                let r3 = _mm256_loadu_ps(a.as_ptr().add((ir + 3) * k + p));
                let r4 = _mm256_loadu_ps(a.as_ptr().add((ir + 4) * k + p));
                let r5 = _mm256_loadu_ps(a.as_ptr().add((ir + 5) * k + p));
                let r6 = _mm256_loadu_ps(a.as_ptr().add((ir + 6) * k + p));
                let r7 = _mm256_loadu_ps(a.as_ptr().add((ir + 7) * k + p));

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

                // Direct stores — stride 16 between K columns
                let dst = packed_a.as_mut_ptr().add(p * 16);
                _mm256_storeu_ps(dst, _mm256_permute2f128_ps(u0, u4, 0x20));
                _mm256_storeu_ps(dst.add(16), _mm256_permute2f128_ps(u1, u5, 0x20));
                _mm256_storeu_ps(dst.add(32), _mm256_permute2f128_ps(u2, u6, 0x20));
                _mm256_storeu_ps(dst.add(48), _mm256_permute2f128_ps(u3, u7, 0x20));
                _mm256_storeu_ps(dst.add(64), _mm256_permute2f128_ps(u0, u4, 0x31));
                _mm256_storeu_ps(dst.add(80), _mm256_permute2f128_ps(u1, u5, 0x31));
                _mm256_storeu_ps(dst.add(96), _mm256_permute2f128_ps(u2, u6, 0x31));
                _mm256_storeu_ps(dst.add(112), _mm256_permute2f128_ps(u3, u7, 0x31));

                // Lower 8 rows
                let r0 = _mm256_loadu_ps(a.as_ptr().add((ir + 8) * k + p));
                let r1 = _mm256_loadu_ps(a.as_ptr().add((ir + 9) * k + p));
                let r2 = _mm256_loadu_ps(a.as_ptr().add((ir + 10) * k + p));
                let r3 = _mm256_loadu_ps(a.as_ptr().add((ir + 11) * k + p));
                let r4 = _mm256_loadu_ps(a.as_ptr().add((ir + 12) * k + p));
                let r5 = _mm256_loadu_ps(a.as_ptr().add((ir + 13) * k + p));
                let r6 = _mm256_loadu_ps(a.as_ptr().add((ir + 14) * k + p));
                let r7 = _mm256_loadu_ps(a.as_ptr().add((ir + 15) * k + p));

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

                // Lower rows at +8 offset
                let dst_lo = packed_a.as_mut_ptr().add(p * 16 + 8);
                _mm256_storeu_ps(dst_lo, _mm256_permute2f128_ps(u0, u4, 0x20));
                _mm256_storeu_ps(dst_lo.add(16), _mm256_permute2f128_ps(u1, u5, 0x20));
                _mm256_storeu_ps(dst_lo.add(32), _mm256_permute2f128_ps(u2, u6, 0x20));
                _mm256_storeu_ps(dst_lo.add(48), _mm256_permute2f128_ps(u3, u7, 0x20));
                _mm256_storeu_ps(dst_lo.add(64), _mm256_permute2f128_ps(u0, u4, 0x31));
                _mm256_storeu_ps(dst_lo.add(80), _mm256_permute2f128_ps(u1, u5, 0x31));
                _mm256_storeu_ps(dst_lo.add(96), _mm256_permute2f128_ps(u2, u6, 0x31));
                _mm256_storeu_ps(dst_lo.add(112), _mm256_permute2f128_ps(u3, u7, 0x31));
            }

            // Remainder k columns: scalar pack
            for p in k_rem_start..k {
                for i in 0..16 {
                    *packed_a.get_unchecked_mut(p * 16 + i) = *a.get_unchecked((ir + i) * k + p);
                }
            }

            for jr_panel in 0..panels_n {
                let jr = jr_panel * 8;
                let packed_b_ptr = all_packed_b.as_ptr().add(jr_panel * k * 8);

                // Load C tile (column-major for micro-kernel: c_micro[j*16+i])
                for jj in 0..8 {
                    for ii in 0..16 {
                        c_micro[jj * 16 + ii] = *c.get_unchecked((ir + ii) * n + jr + jj);
                    }
                }

                microkernel_16x8_avx512(
                    k,
                    packed_a.as_ptr(),
                    packed_b_ptr,
                    c_micro.as_mut_ptr(),
                    16,
                );

                // Store C tile back to row-major
                for jj in 0..8 {
                    for ii in 0..16 {
                        *c.get_unchecked_mut((ir + ii) * n + jr + jj) = c_micro[jj * 16 + ii];
                    }
                }
            }
        }
    }
    Ok(())
}

/// Validate GEMM dimension inputs (Poka-yoke).
fn validate_gemm_dims(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &[f32],
) -> Result<(), TruenoError> {
    if a.len() != m * k {
        return Err(TruenoError::InvalidInput(format!(
            "A size mismatch: expected {}, got {}",
            m * k,
            a.len()
        )));
    }
    if b.len() != k * n {
        return Err(TruenoError::InvalidInput(format!(
            "B size mismatch: expected {}, got {}",
            k * n,
            b.len()
        )));
    }
    if c.len() != m * n {
        return Err(TruenoError::InvalidInput(format!(
            "C size mismatch: expected {}, got {}",
            m * n,
            c.len()
        )));
    }
    Ok(())
}

/// Record a profiler event if profiler is active.
#[inline(always)]
fn record_prof(
    profiler: &mut Option<&mut BlisProfiler>,
    level: BlisProfileLevel,
    start: Option<Instant>,
    flops: u64,
) {
    if let (Some(ref mut prof), Some(s)) = (profiler.as_deref_mut(), start) {
        prof.record(level, s.elapsed().as_nanos() as u64, flops);
    }
}

/// BLIS-style blocked GEMM
///
/// Implements the 5-loop BLIS algorithm (Van Zee & Van de Geijn, 2015):
/// Loop 5 (jc): N dimension, L3 blocking
/// Loop 4 (pc): K dimension, L2 blocking
/// Loop 3 (ic): M dimension, L1 blocking
/// Loop 2 (jr): Microkernel columns
/// Loop 1 (ir): Microkernel rows
pub fn gemm_blis(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    mut profiler: Option<&mut BlisProfiler>,
) -> Result<(), TruenoError> {
    contract_pre_flops_per_tile!();
    validate_gemm_dims(m, n, k, a, b, c)?;

    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }
    if m * n * k < 4096 {
        return gemm_reference(m, n, k, a, b, c);
    }

    // Small: optimized GEMM paths (skip when profiler active).
    #[cfg(target_arch = "x86_64")]
    if profiler.is_none()
        && m <= 256
        && n <= 256
        && k <= 256
        && is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("fma")
    {
        unsafe {
            // Zero-pack row-major GEMM for ≤128: no packing, no C transpose.
            if m <= 128 && n <= 128 && m % 8 == 0 && n % 8 == 0 {
                return gemm_direct_rowmajor(m, n, k, a, b, c);
            }
            // AVX-512 for 129-256: 16×8 tiles, pre-packed B.
            if is_x86_feature_detected!("avx512f") && m >= 16 && m % 16 == 0 && n % 8 == 0 {
                return gemm_small_avx512_16x8(m, n, k, a, b, c);
            }
            if m >= MR && m % 8 == 0 && n % 8 == 0 {
                return gemm_small_nopack_8x8(m, n, k, a, b, c);
            }
            return gemm_small_strided_avx2(m, n, k, a, b, c);
        }
    }

    // AVX-512 BLIS: MR=8, NR=16 using zmm registers (2× throughput vs AVX2).
    // This closes the gap with OpenBLAS which uses AVX-512 on Zen 4.
    // CRITICAL: Without this, trueno is 0.49x NumPy at 8T (shipping blocker).
    // Contract: avx512-blis-v1.yaml (C-AVX512-BLIS-001, C-AVX512-PROF-001)
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("fma") {
        return unsafe { gemm_blis_avx512_large(m, n, k, a, b, c, &mut profiler) };
    }

    // NR=8 BLIS with row-major C SIMD load/store (AVX2 fallback).
    #[cfg(target_arch = "x86_64")]
    if profiler.is_none() && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return unsafe { gemm_blis_nr8_rowmajor_c(m, n, k, a, b, c) };
    }

    // KAIZEN-038: Only call Instant::now() when profiler is active
    let track_time = profiler.is_some();
    let start = if track_time { Some(Instant::now()) } else { None };

    let mc = MC.min(m);
    let nc = NC.min(n);
    let kc = KC.min(k);

    let needed_a = packed_a_size(mc, kc);
    let needed_b = packed_b_size(kc, nc);
    let needed_c = MR * NR;

    // Borrow thread-local workspace buffers, growing if necessary.
    // This eliminates ~4.3 MB of allocation churn per gemm_blis call.
    TL_PACKED_A.with(|tl_a| {
        TL_PACKED_B.with(|tl_b| {
            TL_C_MICRO.with(|tl_c| {
                let mut packed_a = tl_a.borrow_mut();
                let mut packed_b = tl_b.borrow_mut();
                let mut c_micro = tl_c.borrow_mut();

                // Grow buffers to required size (high-water mark).
                // Zero-fill to match the semantics of the original vec![0.0; N].
                if packed_a.len() < needed_a {
                    packed_a.resize(needed_a, 0.0);
                }
                if packed_b.len() < needed_b {
                    packed_b.resize(needed_b, 0.0);
                }
                if c_micro.len() < needed_c {
                    c_micro.resize(needed_c, 0.0);
                }

                for jc in (0..n).step_by(NC) {
                    let nc_block = NC.min(n - jc);

                    for pc in (0..k).step_by(KC) {
                        let kc_block = KC.min(k - pc);

                        let pack_start = if track_time { Some(Instant::now()) } else { None };
                        pack_b_block(b, n, pc, jc, kc_block, nc_block, &mut packed_b);
                        record_prof(&mut profiler, BlisProfileLevel::Pack, pack_start, 0);

                        for ic in (0..m).step_by(MC) {
                            let mc_block = MC.min(m - ic);

                            let pack_start = if track_time { Some(Instant::now()) } else { None };
                            pack_a_block(a, k, ic, pc, mc_block, kc_block, &mut packed_a);
                            record_prof(&mut profiler, BlisProfileLevel::Pack, pack_start, 0);

                            compute_macroblock(
                                c,
                                &packed_a,
                                &packed_b,
                                &mut c_micro,
                                ic,
                                jc,
                                mc_block,
                                nc_block,
                                kc_block,
                                n,
                                &mut profiler,
                            );
                        }
                    }
                }

                if let (Some(prof), Some(s)) = (profiler, start) {
                    prof.record(
                        BlisProfileLevel::Macro,
                        s.elapsed().as_nanos() as u64,
                        (2 * m * n * k) as u64,
                    );
                }
            });
        });
    });

    contract_post_flops_per_tile!(c);
    Ok(())
}

/// AVX-512 BLIS 5-loop GEMM — MR=8, NR=16 using zmm registers.
///
/// 2× throughput vs AVX2 path: each C row = 16 f32 = 1 zmm register.
/// 8 zmm accumulators (8 rows × 16 cols = 128 elements), well within
/// AVX-512's 32-register budget. B loaded as zmm (16 f32), A broadcast.
///
/// Cache blocking: MC=64, KC=256, NC=1024.
/// A packing: MR=8 column-major panels (reuses pack_a_block).
/// B packing: NR=16 row-major panels.
/// AVX-512 BLIS 5-loop GEMM — MR=8, NR=16 with BlisProfiler support.
/// Contract: avx512-blis-v1.yaml (C-AVX512-BLIS-001, C-AVX512-PROF-001)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "fma")]
unsafe fn gemm_blis_avx512_large(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    profiler: &mut Option<&mut BlisProfiler>,
) -> Result<(), TruenoError> {
    // KAIZEN-038: Only call Instant::now() when profiler is active
    let track_time = profiler.is_some();
    let start = if track_time { Some(Instant::now()) } else { None };

    // Phase 4-6: tiered NR selection with dynamic cache blocking.
    // Contract: cgp-dynamic-cache-v1.yaml, cgp-gemm-codegen-v1.yaml
    // NR=48 (codegen): 24 FMA/K-step, KC=128 (L1-limited)
    // NR=32 (hand-written): 16 FMA/K-step, KC=256
    // NR=16 (hand-written): 8 FMA/K-step, KC=256
    // NEGATIVE RESULT (2026-04-05): NR=48 codegen with KC=128 regressed:
    //   512: 41 → 135 GFLOPS after reverting to NR=32
    //   1024: 85 → 130 GFLOPS after reverting to NR=32
    // Root cause: KC halved (128 vs 256) → 2× more K-loop packing passes.
    // The 24 FMA/K-step doesn't compensate for the packing overhead increase.
    // NR=48 path disabled pending KC optimization (prefetch, double-buffer).
    let blk = if n >= 32 {
        super::cache_topology::blocking_8x32()
    } else {
        super::cache_topology::blocking_8x16()
    };
    let mr = blk.mr;
    let nr = blk.nr;
    let mc = blk.mc.min(m);
    let nc = blk.nc.min(n);
    let kc_param = blk.kc;

    TL_PACKED_A.with(|tl_a| {
        TL_PACKED_B.with(|tl_b| {
            let mut packed_a = tl_a.borrow_mut();
            let mut packed_b = tl_b.borrow_mut();

            let needed_a = packed_a_size(mc, kc_param);
            // packed B: panels * nr * kc, where panels rounds up nc/nr
            let b_panels = (nc + nr - 1) / nr;
            let needed_b = b_panels * nr * kc_param;
            if packed_a.len() < needed_a {
                packed_a.resize(needed_a, 0.0);
            }
            if packed_b.len() < needed_b {
                packed_b.resize(needed_b, 0.0);
            }

            for jc in (0..n).step_by(nc) {
                let nc_block = nc.min(n - jc);

                for pc in (0..k).step_by(kc_param) {
                    let kc_block = kc_param.min(k - pc);

                    // Pack B with NR matching the selected microkernel
                    if nr == 48 {
                        pack_b_block_generic(b, n, pc, jc, kc_block, nc_block, 48, &mut packed_b);
                    } else if nr == 32 {
                        pack_b_block_generic(b, n, pc, jc, kc_block, nc_block, 32, &mut packed_b);
                    } else {
                        pack_b_block_nr16(b, n, pc, jc, kc_block, nc_block, &mut packed_b);
                    }

                    for ic in (0..m).step_by(mc) {
                        let mc_block = mc.min(m - ic);

                        pack_a_block(a, k, ic, pc, mc_block, kc_block, &mut packed_a);

                        let panels_m = (mc_block + mr - 1) / mr;
                        let panels_n = (nc_block + nr - 1) / nr;

                        for ir_panel in 0..panels_m {
                            let ir = ir_panel * mr;
                            let mr_block = mr.min(mc_block - ir);

                            for jr_panel in 0..panels_n {
                                let jr = jr_panel * nr;
                                let nr_block = nr.min(nc_block - jr);

                                let a_panel = &packed_a[ir_panel * mr * kc_block..];
                                let b_panel = &packed_b[jr_panel * nr * kc_block..];

                                if mr_block == 8 && nr_block == 48 && nr == 48 {
                                    // Full 8×48 codegen tile (Phase 6: 24 accumulators)
                                    // Contract: cgp-gemm-codegen-v1.yaml C-CODEGEN-002
                                    unsafe {
                                        super::microkernels::codegen::microkernel_8x48_avx512_gen(
                                            kc_block,
                                            a_panel.as_ptr(),
                                            b_panel.as_ptr(),
                                            c.as_mut_ptr().add((ic + ir) * n + (jc + jr)),
                                            n,
                                        );
                                    }
                                } else if mr_block == 8 && nr_block == 32 && nr == 32 {
                                    // Full 8×32 AVX-512 tile (Phase 4: 16 accumulators)
                                    unsafe {
                                        avx512_microkernel_8x32_rowmajor(
                                            kc_block,
                                            a_panel.as_ptr(),
                                            b_panel.as_ptr(),
                                            c.as_mut_ptr().add((ic + ir) * n + (jc + jr)),
                                            n,
                                        );
                                    }
                                } else if mr_block == 8 && nr_block == 16 && nr == 16 {
                                    // Full 8×16 AVX-512 tile (original path)
                                    unsafe {
                                        avx512_microkernel_8x16_rowmajor(
                                            kc_block,
                                            a_panel.as_ptr(),
                                            b_panel.as_ptr(),
                                            c.as_mut_ptr().add((ic + ir) * n + (jc + jr)),
                                            n,
                                        );
                                    }
                                } else {
                                    // Scalar fallback for edge tiles
                                    for ir_local in 0..mr_block {
                                        for jr_local in 0..nr_block {
                                            let mut sum =
                                                c[(ic + ir + ir_local) * n + (jc + jr + jr_local)];
                                            for p in 0..kc_block {
                                                sum += a_panel[p * mr + ir_local]
                                                    * b_panel[p * nr + jr_local];
                                            }
                                            c[(ic + ir + ir_local) * n + (jc + jr + jr_local)] =
                                                sum;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Record profiler event if active (C-AVX512-PROF-001)
            if let (Some(prof), Some(s)) = (profiler.as_mut(), start) {
                prof.record_avx512_blis(m, n, k, s.elapsed());
            }

            Ok(())
        })
    })
}

/// AVX-512 broadcast-B BLIS GEMM — MR=64, NR=6 (faer-style).
///
/// Key difference from broadcast-A path:
/// - A is loaded as zmm vectors (64 elements = 4 zmm per K step)
/// - B is broadcast as scalars (6 per K step)
/// - 24 FMA accumulators = 75% register utilization (matching faer)
/// - NR=6 → B panel is tiny (6×KC×4 bytes) → KC can stay large (256+)
///
/// This avoids the KC-halving problem that killed the 8×48 attempt.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "fma")]
unsafe fn gemm_blis_avx512_bcast_b(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Result<(), TruenoError> {
    let blk = super::cache_topology::blocking_64x6_bcast_b();
    let mr = blk.mr; // 64
    let nr = blk.nr; // 6
    let mc = blk.mc.min(m);
    let nc = blk.nc.min(n);
    let kc = blk.kc;

    // Allocate packing buffers
    let a_panels = (mc + mr - 1) / mr;
    let needed_a = a_panels * mr * kc;
    let b_panels = (nc + nr - 1) / nr;
    let needed_b = b_panels * nr * kc;

    let mut packed_a = vec![0.0f32; needed_a];
    let mut packed_b = vec![0.0f32; needed_b];

    for jc in (0..n).step_by(nc) {
        let nc_block = nc.min(n - jc);

        for pc in (0..k).step_by(kc) {
            let kc_block = kc.min(k - pc);

            // Pack B with NR=6
            pack_b_block_generic(b, n, pc, jc, kc_block, nc_block, nr, &mut packed_b);

            for ic in (0..m).step_by(mc) {
                let mc_block = mc.min(m - ic);

                // Pack A with MR=64 (generic column-major packing)
                pack_a_block_generic(a, k, ic, pc, mc_block, kc_block, mr, &mut packed_a);

                let panels_m = (mc_block + mr - 1) / mr;
                let panels_n = (nc_block + nr - 1) / nr;

                for ir_panel in 0..panels_m {
                    let ir = ir_panel * mr;
                    let mr_block = mr.min(mc_block - ir);

                    for jr_panel in 0..panels_n {
                        let jr = jr_panel * nr;
                        let nr_block = nr.min(nc_block - jr);

                        let a_panel = &packed_a[ir_panel * mr * kc_block..];
                        let b_panel = &packed_b[jr_panel * nr * kc_block..];

                        if mr_block == 64 && nr_block == 6 {
                            // Full 64×6 broadcast-B tile
                            unsafe {
                                super::microkernels::codegen::microkernel_64x6_avx512_bcast_b(
                                    kc_block,
                                    a_panel.as_ptr(),
                                    b_panel.as_ptr(),
                                    c.as_mut_ptr().add((ic + ir) * n + (jc + jr)),
                                    n,
                                );
                            }
                        } else {
                            // Scalar fallback for edge tiles
                            for ir_local in 0..mr_block {
                                for jr_local in 0..nr_block {
                                    let mut sum = 0.0f32;
                                    for p in 0..kc_block {
                                        sum +=
                                            a_panel[p * mr + ir_local] * b_panel[p * nr + jr_local];
                                    }
                                    c[(ic + ir + ir_local) * n + (jc + jr + jr_local)] += sum;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Generic A-packing: column-major panels with arbitrary MR.
/// Packs A[row_start..row_start+rows][col_start..col_start+cols] into
/// panels of MR×cols (column-major within each panel).
#[allow(dead_code)]
fn pack_a_block_generic(
    a: &[f32],
    lda: usize,
    row_start: usize,
    col_start: usize,
    rows: usize,
    cols: usize,
    mr: usize,
    packed: &mut [f32],
) {
    let panels = (rows + mr - 1) / mr;
    let mut pack_idx = 0;
    for panel in 0..panels {
        let ir = panel * mr;
        let mr_actual = mr.min(rows - ir);
        for col in 0..cols {
            for row in 0..mr {
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

/// Safe public wrapper for broadcast-B GEMM (experimental).
/// Uses MR=64, NR=6 codegen microkernel with large KC.
#[cfg(target_arch = "x86_64")]
pub fn gemm_blis_broadcast_b(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Result<(), TruenoError> {
    if a.len() != m * k || b.len() != k * n || c.len() != m * n {
        return Err(TruenoError::InvalidInput("Dimension mismatch".to_string()));
    }
    if std::arch::is_x86_feature_detected!("avx512f") {
        // SAFETY: AVX-512 detected, dimensions validated
        unsafe { gemm_blis_avx512_bcast_b(m, n, k, a, b, c) }
    } else {
        gemm_blis(m, n, k, a, b, c, None)
    }
}

/// AVX-512 8×16 microkernel for row-major C (stride = n).
/// Processes 8 rows × 16 columns using zmm registers for C rows.
/// Each C row is 16 f32 = 1 zmm register. 8 rows = 8 zmm accumulators.
/// A: broadcast scalar to zmm. B: load 16 f32 (1 zmm) per K step.
/// 8 FMA ops per K step, each processing 16 elements = 2× throughput vs AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "fma")]
pub(super) unsafe fn avx512_microkernel_8x16_rowmajor(
    k: usize,
    a: *const f32, // MR=8 packed column-major
    b: *const f32, // NR=16 packed row-major
    c: *mut f32,
    ldc: usize, // row stride = n for row-major
) {
    use std::arch::x86_64::*;

    // Load 8 C rows (each row = 16 f32 = 1 zmm)
    let mut c0 = _mm512_loadu_ps(c);
    let mut c1 = _mm512_loadu_ps(c.add(ldc));
    let mut c2 = _mm512_loadu_ps(c.add(2 * ldc));
    let mut c3 = _mm512_loadu_ps(c.add(3 * ldc));
    let mut c4 = _mm512_loadu_ps(c.add(4 * ldc));
    let mut c5 = _mm512_loadu_ps(c.add(5 * ldc));
    let mut c6 = _mm512_loadu_ps(c.add(6 * ldc));
    let mut c7 = _mm512_loadu_ps(c.add(7 * ldc));

    // Main loop: for each K, load B[16] into zmm, broadcast A[i] to zmm, FMA.
    // NOTE: Manual 4-way K-unrolling was tested (2026-04-05) but REGRESSED
    // from 567→400 GFLOPS at 12T. The compiler (LLVM) already unrolls this
    // loop optimally. Manual unrolling causes register spills from 4× live
    // B vectors + address calculations exceeding the register budget.
    for p in 0..k {
        let b_row = _mm512_loadu_ps(b.add(p * 16));
        let ap = a.add(p * 8); // MR=8

        c0 = _mm512_fmadd_ps(_mm512_set1_ps(*ap), b_row, c0);
        c1 = _mm512_fmadd_ps(_mm512_set1_ps(*ap.add(1)), b_row, c1);
        c2 = _mm512_fmadd_ps(_mm512_set1_ps(*ap.add(2)), b_row, c2);
        c3 = _mm512_fmadd_ps(_mm512_set1_ps(*ap.add(3)), b_row, c3);
        c4 = _mm512_fmadd_ps(_mm512_set1_ps(*ap.add(4)), b_row, c4);
        c5 = _mm512_fmadd_ps(_mm512_set1_ps(*ap.add(5)), b_row, c5);
        c6 = _mm512_fmadd_ps(_mm512_set1_ps(*ap.add(6)), b_row, c6);
        c7 = _mm512_fmadd_ps(_mm512_set1_ps(*ap.add(7)), b_row, c7);
    }

    // Store 8 C rows
    _mm512_storeu_ps(c, c0);
    _mm512_storeu_ps(c.add(ldc), c1);
    _mm512_storeu_ps(c.add(2 * ldc), c2);
    _mm512_storeu_ps(c.add(3 * ldc), c3);
    _mm512_storeu_ps(c.add(4 * ldc), c4);
    _mm512_storeu_ps(c.add(5 * ldc), c5);
    _mm512_storeu_ps(c.add(6 * ldc), c6);
    _mm512_storeu_ps(c.add(7 * ldc), c7);
}

/// AVX-512 8×32 microkernel for row-major C (stride = n).
/// Phase 4 (Appendix D): doubles NR from 16→32 to use 16 zmm accumulators.
/// Each C row spans 2 zmm (32 f32). 8 rows = 16 zmm accumulators.
/// B: 2 zmm loads per K step (32 columns). A: 8 scalar broadcasts.
/// FMAs per K step: 16 (2× the 8×16 kernel).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "fma")]
pub(super) unsafe fn avx512_microkernel_8x32_rowmajor(
    k: usize,
    a: *const f32, // MR=8 packed column-major
    b: *const f32, // NR=32 packed row-major
    c: *mut f32,
    ldc: usize, // row stride = n for row-major
) {
    use std::arch::x86_64::*;

    // Load 8 C rows × 2 zmm halves = 16 accumulators
    let mut c0l = _mm512_loadu_ps(c);
    let mut c0h = _mm512_loadu_ps(c.add(16));
    let mut c1l = _mm512_loadu_ps(c.add(ldc));
    let mut c1h = _mm512_loadu_ps(c.add(ldc + 16));
    let mut c2l = _mm512_loadu_ps(c.add(2 * ldc));
    let mut c2h = _mm512_loadu_ps(c.add(2 * ldc + 16));
    let mut c3l = _mm512_loadu_ps(c.add(3 * ldc));
    let mut c3h = _mm512_loadu_ps(c.add(3 * ldc + 16));
    let mut c4l = _mm512_loadu_ps(c.add(4 * ldc));
    let mut c4h = _mm512_loadu_ps(c.add(4 * ldc + 16));
    let mut c5l = _mm512_loadu_ps(c.add(5 * ldc));
    let mut c5h = _mm512_loadu_ps(c.add(5 * ldc + 16));
    let mut c6l = _mm512_loadu_ps(c.add(6 * ldc));
    let mut c6h = _mm512_loadu_ps(c.add(6 * ldc + 16));
    let mut c7l = _mm512_loadu_ps(c.add(7 * ldc));
    let mut c7h = _mm512_loadu_ps(c.add(7 * ldc + 16));

    // NOTE: Manual 2-way K-unrolling was tested (2026-04-05) but regressed
    // from 15.62ms→15.9ms at 1024 and 34.3→34.9µs at 128. The 8×32 kernel
    // uses 16 zmm accumulators + 2 B loads = 18 zmm live, leaving only 14
    // for unrolled state. LLVM's autounroll is better at managing this pressure.
    // Also tested MC=192 (from 96): regressed at 128-256 due to increased A-packing.
    for p in 0..k {
        let bl = _mm512_loadu_ps(b.add(p * 32));
        let bh = _mm512_loadu_ps(b.add(p * 32 + 16));
        let ap = a.add(p * 8);

        let a0 = _mm512_set1_ps(*ap);
        c0l = _mm512_fmadd_ps(a0, bl, c0l);
        c0h = _mm512_fmadd_ps(a0, bh, c0h);
        let a1 = _mm512_set1_ps(*ap.add(1));
        c1l = _mm512_fmadd_ps(a1, bl, c1l);
        c1h = _mm512_fmadd_ps(a1, bh, c1h);
        let a2 = _mm512_set1_ps(*ap.add(2));
        c2l = _mm512_fmadd_ps(a2, bl, c2l);
        c2h = _mm512_fmadd_ps(a2, bh, c2h);
        let a3 = _mm512_set1_ps(*ap.add(3));
        c3l = _mm512_fmadd_ps(a3, bl, c3l);
        c3h = _mm512_fmadd_ps(a3, bh, c3h);
        let a4 = _mm512_set1_ps(*ap.add(4));
        c4l = _mm512_fmadd_ps(a4, bl, c4l);
        c4h = _mm512_fmadd_ps(a4, bh, c4h);
        let a5 = _mm512_set1_ps(*ap.add(5));
        c5l = _mm512_fmadd_ps(a5, bl, c5l);
        c5h = _mm512_fmadd_ps(a5, bh, c5h);
        let a6 = _mm512_set1_ps(*ap.add(6));
        c6l = _mm512_fmadd_ps(a6, bl, c6l);
        c6h = _mm512_fmadd_ps(a6, bh, c6h);
        let a7 = _mm512_set1_ps(*ap.add(7));
        c7l = _mm512_fmadd_ps(a7, bl, c7l);
        c7h = _mm512_fmadd_ps(a7, bh, c7h);
    }

    // Store 8 C rows × 2 zmm
    _mm512_storeu_ps(c, c0l);
    _mm512_storeu_ps(c.add(16), c0h);
    _mm512_storeu_ps(c.add(ldc), c1l);
    _mm512_storeu_ps(c.add(ldc + 16), c1h);
    _mm512_storeu_ps(c.add(2 * ldc), c2l);
    _mm512_storeu_ps(c.add(2 * ldc + 16), c2h);
    _mm512_storeu_ps(c.add(3 * ldc), c3l);
    _mm512_storeu_ps(c.add(3 * ldc + 16), c3h);
    _mm512_storeu_ps(c.add(4 * ldc), c4l);
    _mm512_storeu_ps(c.add(4 * ldc + 16), c4h);
    _mm512_storeu_ps(c.add(5 * ldc), c5l);
    _mm512_storeu_ps(c.add(5 * ldc + 16), c5h);
    _mm512_storeu_ps(c.add(6 * ldc), c6l);
    _mm512_storeu_ps(c.add(6 * ldc + 16), c6h);
    _mm512_storeu_ps(c.add(7 * ldc), c7l);
    _mm512_storeu_ps(c.add(7 * ldc + 16), c7h);
}

/// Pack B block with NR=16 row-major panels for AVX-512.
/// Each panel is KC × 16, stored as kc_block × nr contiguous.
#[allow(dead_code)]
pub(super) fn pack_b_block_nr16(
    b: &[f32],
    ldb: usize,
    pc: usize,
    jc: usize,
    kc: usize,
    nc: usize,
    packed: &mut [f32],
) {
    let nr = 16;
    let panels = (nc + nr - 1) / nr;
    for panel in 0..panels {
        let j_start = panel * nr;
        let nr_local = nr.min(nc - j_start);
        for p in 0..kc {
            for j in 0..nr_local {
                packed[panel * nr * kc + p * nr + j] = b[(pc + p) * ldb + (jc + j_start + j)];
            }
            // Zero-pad if nr_local < 16
            for j in nr_local..nr {
                packed[panel * nr * kc + p * nr + j] = 0.0;
            }
        }
    }
}

/// Pack B block with generic NR for AVX-512 (NR=32 for 8×32 microkernel).
/// For full NR=32 panels with contiguous source, uses AVX-512 vectorized copy
/// (2 zmm loads + 2 zmm stores per K step vs 32 scalar copies).
pub(super) fn pack_b_block_generic(
    b: &[f32],
    ldb: usize,
    pc: usize,
    jc: usize,
    kc: usize,
    nc: usize,
    nr: usize,
    packed: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    if nr == 32 && std::arch::is_x86_feature_detected!("avx512f") {
        // SAFETY: AVX-512 detected, nr=32 = 2 zmm
        unsafe {
            pack_b_block_nr32_avx512(b, ldb, pc, jc, kc, nc, packed);
        }
        return;
    }

    let panels = (nc + nr - 1) / nr;
    for panel in 0..panels {
        let j_start = panel * nr;
        let nr_local = nr.min(nc - j_start);
        for p in 0..kc {
            let dst_base = panel * nr * kc + p * nr;
            for j in 0..nr_local {
                packed[dst_base + j] = b[(pc + p) * ldb + (jc + j_start + j)];
            }
            for j in nr_local..nr {
                packed[dst_base + j] = 0.0;
            }
        }
    }
}

/// AVX-512 optimized B packing for NR=32 (2 zmm per K step).
/// Full panels: 2× _mm512_loadu_ps + 2× _mm512_storeu_ps.
/// Edge panels: scalar fallback for partial rows.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn pack_b_block_nr32_avx512(
    b: &[f32],
    ldb: usize,
    pc: usize,
    jc: usize,
    kc: usize,
    nc: usize,
    packed: &mut [f32],
) {
    use std::arch::x86_64::*;

    let nr = 32;
    let panels = (nc + nr - 1) / nr;
    for panel in 0..panels {
        let j_start = panel * nr;
        let nr_local = nr.min(nc - j_start);

        if nr_local == 32 {
            // Full panel: SIMD copy — 2 zmm per K step, 2-way K-unrolled.
            // CGP-DBUF: unrolling amortizes loop overhead (~1 cycle/iter saved).
            let panel_base = panel * nr * kc;
            let b_col = jc + j_start;
            let kc2 = kc / 2 * 2;
            let mut p = 0;
            while p < kc2 {
                let src0 = b.as_ptr().add((pc + p) * ldb + b_col);
                let src1 = b.as_ptr().add((pc + p + 1) * ldb + b_col);
                let dst0 = packed.as_mut_ptr().add(panel_base + p * nr);
                let dst1 = packed.as_mut_ptr().add(panel_base + (p + 1) * nr);
                let v0a = _mm512_loadu_ps(src0);
                let v0b = _mm512_loadu_ps(src0.add(16));
                let v1a = _mm512_loadu_ps(src1);
                let v1b = _mm512_loadu_ps(src1.add(16));
                _mm512_storeu_ps(dst0, v0a);
                _mm512_storeu_ps(dst0.add(16), v0b);
                _mm512_storeu_ps(dst1, v1a);
                _mm512_storeu_ps(dst1.add(16), v1b);
                p += 2;
            }
            // Remainder K
            while p < kc {
                let src = b.as_ptr().add((pc + p) * ldb + b_col);
                let dst = packed.as_mut_ptr().add(panel_base + p * nr);
                let v0 = _mm512_loadu_ps(src);
                let v1 = _mm512_loadu_ps(src.add(16));
                _mm512_storeu_ps(dst, v0);
                _mm512_storeu_ps(dst.add(16), v1);
                p += 1;
            }
        } else {
            // Edge panel: scalar with zero-padding
            for p in 0..kc {
                let dst_base = panel * nr * kc + p * nr;
                for j in 0..nr_local {
                    packed[dst_base + j] = b[(pc + p) * ldb + (jc + j_start + j)];
                }
                for j in nr_local..nr {
                    packed[dst_base + j] = 0.0;
                }
            }
        }
    }
}

/// BLIS 5-loop GEMM with NR=8 and row-major C SIMD load/store (AVX2).
///
/// Key optimization vs standard BLIS: C rows are loaded/stored with
/// `_mm256_loadu_ps`/`_mm256_storeu_ps` (NR=8 = 1 ymm per row).
/// This replaces 96 scalar C ops per tile with 16 SIMD ops.
/// Matches matrixmultiply's approach (MR=8, NR=8, contiguous C rows).
///
/// Uses existing `pack_a_block` (MR=8) and `pack_b_block_512` (NR=8).
/// Inner loop: broadcast packed A elements, load packed B row, FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn gemm_blis_nr8_rowmajor_c(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Result<(), TruenoError> {
    use std::arch::x86_64::*;

    // Cache blocking: match matrixmultiply's parameters.
    // MC=64: A~ = MC×KC×4 = 64KB, fits in L2 (1MB/core on Zen 4).
    // KC=256: B panel per tile = KC×8×4 = 8KB, fits in L1 (32KB).
    // NC=1024: B~ = KC×NC×4 = 1MB, fits in L3 (~5MB/CCX on Zen 4).
    let mc = 64_usize.min(m);
    let nc = 1024_usize.min(n);
    let kc_param = KC;
    let nr = 8_usize; // NR=8 for ymm-width C rows
    let mr = MR; // MR=8

    TL_PACKED_A.with(|tl_a| {
        TL_PACKED_B.with(|tl_b| {
            let mut packed_a = tl_a.borrow_mut();
            let mut packed_b = tl_b.borrow_mut();

            let needed_a = packed_a_size(mc, kc_param);
            let needed_b = packed_b_size_512(kc_param, nc); // NR=8 packing
            if packed_a.len() < needed_a {
                packed_a.resize(needed_a, 0.0);
            }
            if packed_b.len() < needed_b {
                packed_b.resize(needed_b, 0.0);
            }

            // BLIS 5-loop with NR=8 packing and row-major C
            for jc in (0..n).step_by(nc) {
                let nc_block = nc.min(n - jc);

                for pc in (0..k).step_by(kc_param) {
                    let kc_block = kc_param.min(k - pc);

                    // Pack B with NR=8
                    pack_b_block_512(b, n, pc, jc, kc_block, nc_block, &mut packed_b);

                    for ic in (0..m).step_by(mc) {
                        let mc_block = mc.min(m - ic);

                        // Pack A with MR=8
                        pack_a_block(a, k, ic, pc, mc_block, kc_block, &mut packed_a);

                        // Microkernel loop: 8×8 tiles with row-major C
                        let panels_m = (mc_block + mr - 1) / mr;
                        let panels_n = (nc_block + nr - 1) / nr;

                        for ir_panel in 0..panels_m {
                            let ir = ir_panel * mr;
                            let mr_block = mr.min(mc_block - ir);

                            for jr_panel in 0..panels_n {
                                let jr = jr_panel * nr;
                                let nr_block = nr.min(nc_block - jr);

                                let a_panel = &packed_a[ir_panel * mr * kc_block..];
                                let b_panel = &packed_b[jr_panel * nr * kc_block..];

                                if mr_block == 8 && nr_block == 8 {
                                    // Full 8×8 tile: SIMD C load/store + FMA
                                    unsafe {
                                        let c_base = c.as_mut_ptr().add((ic + ir) * n + (jc + jr));

                                        // Load 8 C rows (each 8 f32 = 1 ymm)
                                        let mut c0 = _mm256_loadu_ps(c_base);
                                        let mut c1 = _mm256_loadu_ps(c_base.add(n));
                                        let mut c2 = _mm256_loadu_ps(c_base.add(2 * n));
                                        let mut c3 = _mm256_loadu_ps(c_base.add(3 * n));
                                        let mut c4 = _mm256_loadu_ps(c_base.add(4 * n));
                                        let mut c5 = _mm256_loadu_ps(c_base.add(5 * n));
                                        let mut c6 = _mm256_loadu_ps(c_base.add(6 * n));
                                        let mut c7 = _mm256_loadu_ps(c_base.add(7 * n));

                                        let ap = a_panel.as_ptr();
                                        let bp = b_panel.as_ptr();

                                        // 4-way K-unrolled inner loop
                                        let k4 = kc_block / 4;
                                        let k_rem = kc_block % 4;

                                        for p4 in 0..k4 {
                                            let p = p4 * 4;

                                            let b_row = _mm256_loadu_ps(bp.add(p * 8));
                                            c0 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add(p * 8)),
                                                b_row,
                                                c0,
                                            );
                                            c1 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add(p * 8 + 1)),
                                                b_row,
                                                c1,
                                            );
                                            c2 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add(p * 8 + 2)),
                                                b_row,
                                                c2,
                                            );
                                            c3 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add(p * 8 + 3)),
                                                b_row,
                                                c3,
                                            );
                                            c4 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add(p * 8 + 4)),
                                                b_row,
                                                c4,
                                            );
                                            c5 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add(p * 8 + 5)),
                                                b_row,
                                                c5,
                                            );
                                            c6 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add(p * 8 + 6)),
                                                b_row,
                                                c6,
                                            );
                                            c7 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add(p * 8 + 7)),
                                                b_row,
                                                c7,
                                            );

                                            let b_row = _mm256_loadu_ps(bp.add((p + 1) * 8));
                                            c0 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add((p + 1) * 8)),
                                                b_row,
                                                c0,
                                            );
                                            c1 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add((p + 1) * 8 + 1)),
                                                b_row,
                                                c1,
                                            );
                                            c2 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add((p + 1) * 8 + 2)),
                                                b_row,
                                                c2,
                                            );
                                            c3 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add((p + 1) * 8 + 3)),
                                                b_row,
                                                c3,
                                            );
                                            c4 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add((p + 1) * 8 + 4)),
                                                b_row,
                                                c4,
                                            );
                                            c5 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add((p + 1) * 8 + 5)),
                                                b_row,
                                                c5,
                                            );
                                            c6 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add((p + 1) * 8 + 6)),
                                                b_row,
                                                c6,
                                            );
                                            c7 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add((p + 1) * 8 + 7)),
                                                b_row,
                                                c7,
                                            );

                                            let b_row = _mm256_loadu_ps(bp.add((p + 2) * 8));
                                            c0 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add((p + 2) * 8)),
                                                b_row,
                                                c0,
                                            );
                                            c1 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add((p + 2) * 8 + 1)),
                                                b_row,
                                                c1,
                                            );
                                            c2 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add((p + 2) * 8 + 2)),
                                                b_row,
                                                c2,
                                            );
                                            c3 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add((p + 2) * 8 + 3)),
                                                b_row,
                                                c3,
                                            );
                                            c4 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add((p + 2) * 8 + 4)),
                                                b_row,
                                                c4,
                                            );
                                            c5 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add((p + 2) * 8 + 5)),
                                                b_row,
                                                c5,
                                            );
                                            c6 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add((p + 2) * 8 + 6)),
                                                b_row,
                                                c6,
                                            );
                                            c7 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add((p + 2) * 8 + 7)),
                                                b_row,
                                                c7,
                                            );

                                            let b_row = _mm256_loadu_ps(bp.add((p + 3) * 8));
                                            c0 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add((p + 3) * 8)),
                                                b_row,
                                                c0,
                                            );
                                            c1 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add((p + 3) * 8 + 1)),
                                                b_row,
                                                c1,
                                            );
                                            c2 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add((p + 3) * 8 + 2)),
                                                b_row,
                                                c2,
                                            );
                                            c3 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add((p + 3) * 8 + 3)),
                                                b_row,
                                                c3,
                                            );
                                            c4 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add((p + 3) * 8 + 4)),
                                                b_row,
                                                c4,
                                            );
                                            c5 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add((p + 3) * 8 + 5)),
                                                b_row,
                                                c5,
                                            );
                                            c6 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add((p + 3) * 8 + 6)),
                                                b_row,
                                                c6,
                                            );
                                            c7 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add((p + 3) * 8 + 7)),
                                                b_row,
                                                c7,
                                            );
                                        }

                                        let base_rem = k4 * 4;
                                        for rp in 0..k_rem {
                                            let pp = base_rem + rp;
                                            let b_row = _mm256_loadu_ps(bp.add(pp * 8));
                                            c0 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add(pp * 8)),
                                                b_row,
                                                c0,
                                            );
                                            c1 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add(pp * 8 + 1)),
                                                b_row,
                                                c1,
                                            );
                                            c2 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add(pp * 8 + 2)),
                                                b_row,
                                                c2,
                                            );
                                            c3 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add(pp * 8 + 3)),
                                                b_row,
                                                c3,
                                            );
                                            c4 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add(pp * 8 + 4)),
                                                b_row,
                                                c4,
                                            );
                                            c5 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add(pp * 8 + 5)),
                                                b_row,
                                                c5,
                                            );
                                            c6 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add(pp * 8 + 6)),
                                                b_row,
                                                c6,
                                            );
                                            c7 = _mm256_fmadd_ps(
                                                _mm256_broadcast_ss(&*ap.add(pp * 8 + 7)),
                                                b_row,
                                                c7,
                                            );
                                        }

                                        // Store 8 C rows (SIMD)
                                        _mm256_storeu_ps(c_base, c0);
                                        _mm256_storeu_ps(c_base.add(n), c1);
                                        _mm256_storeu_ps(c_base.add(2 * n), c2);
                                        _mm256_storeu_ps(c_base.add(3 * n), c3);
                                        _mm256_storeu_ps(c_base.add(4 * n), c4);
                                        _mm256_storeu_ps(c_base.add(5 * n), c5);
                                        _mm256_storeu_ps(c_base.add(6 * n), c6);
                                        _mm256_storeu_ps(c_base.add(7 * n), c7);
                                    }
                                } else {
                                    // Edge tile: scalar fallback
                                    for p in 0..kc_block {
                                        for jj in 0..nr_block {
                                            let b_val = b_panel[p * nr + jj];
                                            for ii in 0..mr_block {
                                                c[(ic + ir + ii) * n + (jc + jr + jj)] +=
                                                    a_panel[p * mr + ii] * b_val;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
    });

    Ok(())
}

/// AVX-512 BLIS 5-loop GEMM with packed 16×8 microkernel.
///
/// Uses MR_512=16, NR_512=8 packing for 2× compute density over AVX2 8×6.
/// C tiles loaded/stored with AVX-512 `_mm512_loadu_ps` (16 f32 per load).
/// Packing converts strided A/B into contiguous micro-panel layout for
/// sequential access in the microkernel.
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)] // Retained for AVX-512-only systems; superseded by nr8_rowmajor_c on AVX2
fn gemm_blis_avx512_packed(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Result<(), TruenoError> {
    let mc = MC_512.min(m);
    let nc = NC_512.min(n);
    let kc = KC_512.min(k);

    let needed_a = packed_a_size_512(mc, kc);
    let needed_b = packed_b_size_512(kc, nc);
    let needed_c = MR_512 * NR_512;

    TL_PACKED_A.with(|tl_a| {
        TL_PACKED_B.with(|tl_b| {
            TL_C_MICRO.with(|tl_c| {
                let mut packed_a = tl_a.borrow_mut();
                let mut packed_b = tl_b.borrow_mut();
                let mut c_micro = tl_c.borrow_mut();

                if packed_a.len() < needed_a {
                    packed_a.resize(needed_a, 0.0);
                }
                if packed_b.len() < needed_b {
                    packed_b.resize(needed_b, 0.0);
                }
                if c_micro.len() < needed_c {
                    c_micro.resize(needed_c, 0.0);
                }

                for jc in (0..n).step_by(NC_512) {
                    let nc_block = NC_512.min(n - jc);

                    for pc in (0..k).step_by(KC_512) {
                        let kc_block = KC_512.min(k - pc);

                        pack_b_block_512(b, n, pc, jc, kc_block, nc_block, &mut packed_b);

                        for ic in (0..m).step_by(MC_512) {
                            let mc_block = MC_512.min(m - ic);

                            pack_a_block_512(a, k, ic, pc, mc_block, kc_block, &mut packed_a);

                            // AVX-512 macroblock: 16×8 tiles
                            for ir in (0..mc_block).step_by(MR_512) {
                                let mr_block = MR_512.min(mc_block - ir);
                                for jr in (0..nc_block).step_by(NR_512) {
                                    let nr_block = NR_512.min(nc_block - jr);

                                    let a_panel = &packed_a[(ir / MR_512) * MR_512 * kc_block..];
                                    let b_panel = &packed_b[(jr / NR_512) * NR_512 * kc_block..];

                                    // Load C tile (column-major for microkernel)
                                    for jj in 0..nr_block {
                                        for ii in 0..mr_block {
                                            c_micro[jj * MR_512 + ii] =
                                                c[(ic + ir + ii) * n + (jc + jr + jj)];
                                        }
                                        for ii in mr_block..MR_512 {
                                            c_micro[jj * MR_512 + ii] = 0.0;
                                        }
                                    }
                                    for jj in nr_block..NR_512 {
                                        for ii in 0..MR_512 {
                                            c_micro[jj * MR_512 + ii] = 0.0;
                                        }
                                    }

                                    // Full tile → AVX-512, edge → scalar
                                    if mr_block == MR_512 && nr_block == NR_512 {
                                        // SAFETY: AVX-512 verified by is_x86_feature_detected
                                        // in caller. Packed layout matches microkernel.
                                        unsafe {
                                            microkernel_16x8_avx512(
                                                kc_block,
                                                a_panel.as_ptr(),
                                                b_panel.as_ptr(),
                                                c_micro.as_mut_ptr(),
                                                MR_512,
                                            );
                                        }
                                    } else {
                                        // Edge tiles: scalar
                                        for p in 0..kc_block {
                                            for jj in 0..NR_512 {
                                                let b_val = b_panel[p * NR_512 + jj];
                                                for ii in 0..MR_512 {
                                                    c_micro[jj * MR_512 + ii] +=
                                                        a_panel[p * MR_512 + ii] * b_val;
                                                }
                                            }
                                        }
                                    }

                                    // Store C tile
                                    for jj in 0..nr_block {
                                        for ii in 0..mr_block {
                                            c[(ic + ir + ii) * n + (jc + jr + jj)] =
                                                c_micro[jj * MR_512 + ii];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            });
        });
    });

    Ok(())
}

/// BLIS-style blocked GEMM with pre-packed B matrix.
///
/// Identical to [`gemm_blis`] but skips B packing entirely, reading packed
/// tiles from `prepacked_b` instead. This eliminates redundant B packing
/// when the same weight matrix is reused across calls (e.g., in parallel GEMM
/// where each thread would otherwise pack B independently).
///
/// # WAPR-KAIZEN Cycle 12
///
/// For encoder FFN: 16 threads × 2 GEMMs × 4 layers = 128 B packings eliminated.
pub fn gemm_blis_with_prepacked_b(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    prepacked_b: &PrepackedB,
    c: &mut [f32],
    mut profiler: Option<&mut BlisProfiler>,
) -> Result<(), TruenoError> {
    if a.len() != m * k {
        return Err(TruenoError::InvalidInput(format!(
            "A size mismatch: expected {}, got {}",
            m * k,
            a.len()
        )));
    }
    if c.len() != m * n {
        return Err(TruenoError::InvalidInput(format!(
            "C size mismatch: expected {}, got {}",
            m * n,
            c.len()
        )));
    }
    if prepacked_b.k != k || prepacked_b.n != n {
        return Err(TruenoError::InvalidInput(format!(
            "PrepackedB dimension mismatch: expected ({}, {}), got ({}, {})",
            k, n, prepacked_b.k, prepacked_b.n
        )));
    }

    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }

    let track_time = profiler.is_some();
    let start = if track_time { Some(Instant::now()) } else { None };

    let mc = MC.min(m);
    let kc = KC.min(k);

    let needed_a = packed_a_size(mc, kc);
    let needed_c = MR * NR;

    // Only need A and C micro buffers — B is already packed
    TL_PACKED_A.with(|tl_a| {
        TL_C_MICRO.with(|tl_c| {
            let mut packed_a = tl_a.borrow_mut();
            let mut c_micro = tl_c.borrow_mut();

            if packed_a.len() < needed_a {
                packed_a.resize(needed_a, 0.0);
            } else {
                packed_a[..needed_a].fill(0.0);
            }
            if c_micro.len() < needed_c {
                c_micro.resize(needed_c, 0.0);
            } else {
                c_micro[..needed_c].fill(0.0);
            }

            for (jc_idx, jc) in (0..n).step_by(NC).enumerate() {
                let nc_block = NC.min(n - jc);

                for (pc_idx, pc) in (0..k).step_by(KC).enumerate() {
                    let kc_block = KC.min(k - pc);

                    // Use pre-packed B tile instead of runtime packing
                    let packed_b_tile = prepacked_b.tile(jc_idx, pc_idx);

                    for ic in (0..m).step_by(MC) {
                        let mc_block = MC.min(m - ic);

                        let pack_start = if track_time { Some(Instant::now()) } else { None };
                        pack_a_block(a, k, ic, pc, mc_block, kc_block, &mut packed_a);
                        record_prof(&mut profiler, BlisProfileLevel::Pack, pack_start, 0);

                        compute_macroblock(
                            c,
                            &packed_a,
                            packed_b_tile,
                            &mut c_micro,
                            ic,
                            jc,
                            mc_block,
                            nc_block,
                            kc_block,
                            n,
                            &mut profiler,
                        );
                    }
                }
            }

            if let (Some(prof), Some(s)) = (profiler, start) {
                prof.record(
                    BlisProfileLevel::Macro,
                    s.elapsed().as_nanos() as u64,
                    (2 * m * n * k) as u64,
                );
            }
        });
    });

    Ok(())
}
