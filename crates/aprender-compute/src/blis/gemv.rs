//! SIMD-accelerated GEMV (General Matrix-Vector Multiply)
//!
//! Specialized kernel for M=1 matrix-vector product: c = a × B
//! where a is 1×K and B is K×N, both row-major.
//!
//! This bypasses the BLIS 5-loop packing overhead which dominates for M=1.
//! Instead, uses direct AVX2 VFMADD on unpacked row-major data.
//!
//! # Algorithm
//!
//! Two strategies based on N:
//!
//! - **Small N (≤ 4096)**: Axpy pattern — outer K, inner N. c[] fits in L1.
//! - **Large N (> 4096)**: N-tiled — outer N-tiles (64), inner K. c[] stays
//!   in YMM registers for all K iterations, eliminating L1 thrashing.
//!
//! # References
//!
//! - GH-380: matvec (M=1) performance gap vs ndarray

/// Threshold: when N > this, switch to tiled GEMV kernel.
/// Raised from 4096 → 8192 (2026-04-05): tiled kernel has strided B access
/// (stride=N*4 bytes between rows) which is TLB-unfriendly at large N.
/// Measured: vecmat 4096×4096: tiled 9.3 GFLOPS vs axpy predicts better.
/// 4096 path benchmarks to use axpy. c[] still fits L1 at N=8192 (32KB).
#[allow(dead_code)]
const GEMV_TILE_THRESHOLD: usize = 8192;

/// AVX2 GEMV using axpy pattern: c += a[k] * B[k,:] for each k
///
/// Outer loop over K (4-way unrolled), inner loop over N with AVX2 VFMADD.
/// This matches row-major B access: B[k,:] is contiguous → sequential reads.
///
/// Best for small N where c[] fits in L1 cache.
///
/// # Safety
///
/// Requires AVX2+FMA CPU features. Caller must ensure:
/// - `a` has length >= `k`
/// - `b` has length >= `k * n`
/// - `c` has length >= `n`
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn gemv_avx2(k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    unsafe {
        use std::arch::x86_64::*;

        let n8 = n / 8 * 8;

        // 4-way K-unrolled axpy with AVX2 VFMADD on inner N loop
        let k4 = k / 4 * 4;
        let mut ki = 0;
        while ki < k4 {
            let a0 = _mm256_set1_ps(*a.get_unchecked(ki));
            let a1 = _mm256_set1_ps(*a.get_unchecked(ki + 1));
            let a2 = _mm256_set1_ps(*a.get_unchecked(ki + 2));
            let a3 = _mm256_set1_ps(*a.get_unchecked(ki + 3));
            let b0_base = ki * n;
            let b1_base = b0_base + n;
            let b2_base = b1_base + n;
            let b3_base = b2_base + n;

            let mut j = 0;
            let b_ptr = b.as_ptr();
            let c_ptr = c.as_mut_ptr();
            while j < n8 {
                let cv = _mm256_loadu_ps(c_ptr.add(j));
                let bv0 = _mm256_loadu_ps(b_ptr.add(b0_base + j));
                let bv1 = _mm256_loadu_ps(b_ptr.add(b1_base + j));
                let bv2 = _mm256_loadu_ps(b_ptr.add(b2_base + j));
                let bv3 = _mm256_loadu_ps(b_ptr.add(b3_base + j));

                let r = _mm256_fmadd_ps(a0, bv0, cv);
                let r = _mm256_fmadd_ps(a1, bv1, r);
                let r = _mm256_fmadd_ps(a2, bv2, r);
                let r = _mm256_fmadd_ps(a3, bv3, r);

                _mm256_storeu_ps(c_ptr.add(j), r);
                j += 8;
            }

            // Scalar remainder for N % 8
            while j < n {
                *c.get_unchecked_mut(j) += *a.get_unchecked(ki) * *b.get_unchecked(b0_base + j)
                    + *a.get_unchecked(ki + 1) * *b.get_unchecked(b1_base + j)
                    + *a.get_unchecked(ki + 2) * *b.get_unchecked(b2_base + j)
                    + *a.get_unchecked(ki + 3) * *b.get_unchecked(b3_base + j);
                j += 1;
            }

            ki += 4;
        }

        // Remainder K (scalar axpy)
        while ki < k {
            let ak = *a.get_unchecked(ki);
            let bk_base = ki * n;
            let ak_v = _mm256_set1_ps(ak);

            let mut j = 0;
            let b_ptr = b.as_ptr();
            let c_ptr = c.as_mut_ptr();
            while j < n8 {
                let cv = _mm256_loadu_ps(c_ptr.add(j));
                let bv = _mm256_loadu_ps(b_ptr.add(bk_base + j));
                let r = _mm256_fmadd_ps(ak_v, bv, cv);
                _mm256_storeu_ps(c_ptr.add(j), r);
                j += 8;
            }
            while j < n {
                *c.get_unchecked_mut(j) += ak * *b.get_unchecked(bk_base + j);
                j += 1;
            }
            ki += 1;
        }
    }
}

/// AVX2 GEMV with N-dimension tiling for bandwidth-bound sizes.
///
/// Tiles the N dimension into strips of 64, keeping the c[] accumulator
/// in 8 YMM registers for ALL K iterations. This eliminates the repeated
/// L1 load/store of c[] that dominates the axpy pattern when N > L1.
///
/// For 4096×11008: original axpy does 1024 load-store sweeps of c[] (43KB).
/// Tiled: each c[j0..j0+64] is loaded 0 times (initialized in registers)
/// and stored once at the end. Saves ~88MB of c[] traffic.
///
/// # Safety
///
/// Requires AVX2+FMA CPU features.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn gemv_tiled_avx2(k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    unsafe {
        use std::arch::x86_64::*;

        // NT=64: 8 YMM accumulators × 8 f32 = 64 elements.
        // 4 registers for broadcast a, 1-2 for B loads = ~14 registers total.
        const NT: usize = 64;

        let k4 = k / 4 * 4;
        let nt_end = n / NT * NT;

        for j0 in (0..nt_end).step_by(NT) {
            // 8 YMM accumulators — stay in registers for ALL K iterations
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();
            let mut acc3 = _mm256_setzero_ps();
            let mut acc4 = _mm256_setzero_ps();
            let mut acc5 = _mm256_setzero_ps();
            let mut acc6 = _mm256_setzero_ps();
            let mut acc7 = _mm256_setzero_ps();

            // Process ALL K for this N-tile (4-way unrolled)
            let mut ki = 0;
            while ki < k4 {
                let a0 = _mm256_set1_ps(*a.get_unchecked(ki));
                let a1 = _mm256_set1_ps(*a.get_unchecked(ki + 1));
                let a2 = _mm256_set1_ps(*a.get_unchecked(ki + 2));
                let a3 = _mm256_set1_ps(*a.get_unchecked(ki + 3));

                let b0 = ki * n + j0;
                let b1 = b0 + n;
                let b2 = b1 + n;
                let b3 = b2 + n;

                // Software prefetch: B rows 8 iterations ahead
                if ki + 8 < k {
                    let pf = (ki + 8) * n + j0;
                    _mm_prefetch(b.as_ptr().add(pf) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(b.as_ptr().add(pf + 32) as *const i8, _MM_HINT_T0);
                }

                // 8 chunks × 4 K iterations = 32 FMAs
                let bv = _mm256_loadu_ps(b.get_unchecked(b0));
                acc0 = _mm256_fmadd_ps(a0, bv, acc0);
                let bv = _mm256_loadu_ps(b.get_unchecked(b1));
                acc0 = _mm256_fmadd_ps(a1, bv, acc0);
                let bv = _mm256_loadu_ps(b.get_unchecked(b2));
                acc0 = _mm256_fmadd_ps(a2, bv, acc0);
                let bv = _mm256_loadu_ps(b.get_unchecked(b3));
                acc0 = _mm256_fmadd_ps(a3, bv, acc0);

                let bv = _mm256_loadu_ps(b.get_unchecked(b0 + 8));
                acc1 = _mm256_fmadd_ps(a0, bv, acc1);
                let bv = _mm256_loadu_ps(b.get_unchecked(b1 + 8));
                acc1 = _mm256_fmadd_ps(a1, bv, acc1);
                let bv = _mm256_loadu_ps(b.get_unchecked(b2 + 8));
                acc1 = _mm256_fmadd_ps(a2, bv, acc1);
                let bv = _mm256_loadu_ps(b.get_unchecked(b3 + 8));
                acc1 = _mm256_fmadd_ps(a3, bv, acc1);

                let bv = _mm256_loadu_ps(b.get_unchecked(b0 + 16));
                acc2 = _mm256_fmadd_ps(a0, bv, acc2);
                let bv = _mm256_loadu_ps(b.get_unchecked(b1 + 16));
                acc2 = _mm256_fmadd_ps(a1, bv, acc2);
                let bv = _mm256_loadu_ps(b.get_unchecked(b2 + 16));
                acc2 = _mm256_fmadd_ps(a2, bv, acc2);
                let bv = _mm256_loadu_ps(b.get_unchecked(b3 + 16));
                acc2 = _mm256_fmadd_ps(a3, bv, acc2);

                let bv = _mm256_loadu_ps(b.get_unchecked(b0 + 24));
                acc3 = _mm256_fmadd_ps(a0, bv, acc3);
                let bv = _mm256_loadu_ps(b.get_unchecked(b1 + 24));
                acc3 = _mm256_fmadd_ps(a1, bv, acc3);
                let bv = _mm256_loadu_ps(b.get_unchecked(b2 + 24));
                acc3 = _mm256_fmadd_ps(a2, bv, acc3);
                let bv = _mm256_loadu_ps(b.get_unchecked(b3 + 24));
                acc3 = _mm256_fmadd_ps(a3, bv, acc3);

                let bv = _mm256_loadu_ps(b.get_unchecked(b0 + 32));
                acc4 = _mm256_fmadd_ps(a0, bv, acc4);
                let bv = _mm256_loadu_ps(b.get_unchecked(b1 + 32));
                acc4 = _mm256_fmadd_ps(a1, bv, acc4);
                let bv = _mm256_loadu_ps(b.get_unchecked(b2 + 32));
                acc4 = _mm256_fmadd_ps(a2, bv, acc4);
                let bv = _mm256_loadu_ps(b.get_unchecked(b3 + 32));
                acc4 = _mm256_fmadd_ps(a3, bv, acc4);

                let bv = _mm256_loadu_ps(b.get_unchecked(b0 + 40));
                acc5 = _mm256_fmadd_ps(a0, bv, acc5);
                let bv = _mm256_loadu_ps(b.get_unchecked(b1 + 40));
                acc5 = _mm256_fmadd_ps(a1, bv, acc5);
                let bv = _mm256_loadu_ps(b.get_unchecked(b2 + 40));
                acc5 = _mm256_fmadd_ps(a2, bv, acc5);
                let bv = _mm256_loadu_ps(b.get_unchecked(b3 + 40));
                acc5 = _mm256_fmadd_ps(a3, bv, acc5);

                let bv = _mm256_loadu_ps(b.get_unchecked(b0 + 48));
                acc6 = _mm256_fmadd_ps(a0, bv, acc6);
                let bv = _mm256_loadu_ps(b.get_unchecked(b1 + 48));
                acc6 = _mm256_fmadd_ps(a1, bv, acc6);
                let bv = _mm256_loadu_ps(b.get_unchecked(b2 + 48));
                acc6 = _mm256_fmadd_ps(a2, bv, acc6);
                let bv = _mm256_loadu_ps(b.get_unchecked(b3 + 48));
                acc6 = _mm256_fmadd_ps(a3, bv, acc6);

                let bv = _mm256_loadu_ps(b.get_unchecked(b0 + 56));
                acc7 = _mm256_fmadd_ps(a0, bv, acc7);
                let bv = _mm256_loadu_ps(b.get_unchecked(b1 + 56));
                acc7 = _mm256_fmadd_ps(a1, bv, acc7);
                let bv = _mm256_loadu_ps(b.get_unchecked(b2 + 56));
                acc7 = _mm256_fmadd_ps(a2, bv, acc7);
                let bv = _mm256_loadu_ps(b.get_unchecked(b3 + 56));
                acc7 = _mm256_fmadd_ps(a3, bv, acc7);

                ki += 4;
            }

            // Remainder K (1 at a time)
            while ki < k {
                let av = _mm256_set1_ps(*a.get_unchecked(ki));
                let base = ki * n + j0;

                acc0 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b.get_unchecked(base)), acc0);
                acc1 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b.get_unchecked(base + 8)), acc1);
                acc2 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b.get_unchecked(base + 16)), acc2);
                acc3 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b.get_unchecked(base + 24)), acc3);
                acc4 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b.get_unchecked(base + 32)), acc4);
                acc5 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b.get_unchecked(base + 40)), acc5);
                acc6 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b.get_unchecked(base + 48)), acc6);
                acc7 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b.get_unchecked(base + 56)), acc7);
                ki += 1;
            }

            // Store accumulators (one store per tile, not K/4 stores)
            _mm256_storeu_ps(c.get_unchecked_mut(j0), acc0);
            _mm256_storeu_ps(c.get_unchecked_mut(j0 + 8), acc1);
            _mm256_storeu_ps(c.get_unchecked_mut(j0 + 16), acc2);
            _mm256_storeu_ps(c.get_unchecked_mut(j0 + 24), acc3);
            _mm256_storeu_ps(c.get_unchecked_mut(j0 + 32), acc4);
            _mm256_storeu_ps(c.get_unchecked_mut(j0 + 40), acc5);
            _mm256_storeu_ps(c.get_unchecked_mut(j0 + 48), acc6);
            _mm256_storeu_ps(c.get_unchecked_mut(j0 + 56), acc7);
        }

        // Remainder N (< 64 elements) — axpy is fine since c fits in L1
        if nt_end < n {
            let rem_n = n - nt_end;
            let rem8 = rem_n / 8 * 8;
            let k4 = k / 4 * 4;

            let mut ki = 0;
            while ki < k4 {
                let a0 = _mm256_set1_ps(*a.get_unchecked(ki));
                let a1 = _mm256_set1_ps(*a.get_unchecked(ki + 1));
                let a2 = _mm256_set1_ps(*a.get_unchecked(ki + 2));
                let a3 = _mm256_set1_ps(*a.get_unchecked(ki + 3));
                let b0 = ki * n + nt_end;
                let b1 = b0 + n;
                let b2 = b1 + n;
                let b3 = b2 + n;

                let mut j = 0;
                while j < rem8 {
                    let cv = _mm256_loadu_ps(c.get_unchecked(nt_end + j));
                    let r = _mm256_fmadd_ps(a0, _mm256_loadu_ps(b.get_unchecked(b0 + j)), cv);
                    let r = _mm256_fmadd_ps(a1, _mm256_loadu_ps(b.get_unchecked(b1 + j)), r);
                    let r = _mm256_fmadd_ps(a2, _mm256_loadu_ps(b.get_unchecked(b2 + j)), r);
                    let r = _mm256_fmadd_ps(a3, _mm256_loadu_ps(b.get_unchecked(b3 + j)), r);
                    _mm256_storeu_ps(c.get_unchecked_mut(nt_end + j), r);
                    j += 8;
                }
                while j < rem_n {
                    let idx = nt_end + j;
                    *c.get_unchecked_mut(idx) += *a.get_unchecked(ki) * *b.get_unchecked(b0 + j)
                        + *a.get_unchecked(ki + 1) * *b.get_unchecked(b1 + j)
                        + *a.get_unchecked(ki + 2) * *b.get_unchecked(b2 + j)
                        + *a.get_unchecked(ki + 3) * *b.get_unchecked(b3 + j);
                    j += 1;
                }
                ki += 4;
            }

            while ki < k {
                let ak = *a.get_unchecked(ki);
                let bk = ki * n + nt_end;
                let ak_v = _mm256_set1_ps(ak);

                let mut j = 0;
                while j < rem8 {
                    let cv = _mm256_loadu_ps(c.get_unchecked(nt_end + j));
                    let bv = _mm256_loadu_ps(b.get_unchecked(bk + j));
                    _mm256_storeu_ps(
                        c.get_unchecked_mut(nt_end + j),
                        _mm256_fmadd_ps(ak_v, bv, cv),
                    );
                    j += 8;
                }
                while j < rem_n {
                    *c.get_unchecked_mut(nt_end + j) += ak * *b.get_unchecked(bk + j);
                    j += 1;
                }
                ki += 1;
            }
        }
    }
}

/// AVX-512 GEMV with N-dimension tiling — 2× throughput vs AVX2.
///
/// NT=128: 8 ZMM accumulators × 16 f32 = 128 elements per tile.
/// 4-way K-unrolled: 32 FMAs per tile per iteration.
/// Software prefetch: B rows 4 iterations ahead.
///
/// For attention scoring (Q @ K_cache^T): head_dim=128, seq_len varies.
/// This is the hottest path in LLM inference (44.3% of compute).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "fma")]
#[allow(dead_code)] // Retained for Intel SPR (no AVX-512 throttle). See negative result above.
unsafe fn gemv_tiled_avx512(k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    unsafe {
        use std::arch::x86_64::*;

        // NT=128: 8 ZMM accumulators × 16 f32 = 128 elements.
        // Fits in 8 of 32 ZMM registers. 4 for A broadcasts + 1 for B load = 13 total.
        const NT: usize = 128;

        let k4 = k / 4 * 4;
        let nt_end = n / NT * NT;

        for j0 in (0..nt_end).step_by(NT) {
            // 8 ZMM accumulators — stay in registers for ALL K iterations
            let mut acc0 = _mm512_setzero_ps();
            let mut acc1 = _mm512_setzero_ps();
            let mut acc2 = _mm512_setzero_ps();
            let mut acc3 = _mm512_setzero_ps();
            let mut acc4 = _mm512_setzero_ps();
            let mut acc5 = _mm512_setzero_ps();
            let mut acc6 = _mm512_setzero_ps();
            let mut acc7 = _mm512_setzero_ps();

            // Process ALL K for this N-tile (4-way unrolled)
            let mut ki = 0;
            while ki < k4 {
                let a0 = _mm512_set1_ps(*a.get_unchecked(ki));
                let a1 = _mm512_set1_ps(*a.get_unchecked(ki + 1));
                let a2 = _mm512_set1_ps(*a.get_unchecked(ki + 2));
                let a3 = _mm512_set1_ps(*a.get_unchecked(ki + 3));

                let b0 = ki * n + j0;
                let b1 = b0 + n;
                let b2 = b1 + n;
                let b3 = b2 + n;

                // Prefetch B rows 4 iterations ahead
                if ki + 4 < k {
                    let pf = (ki + 4) * n + j0;
                    _mm_prefetch(b.as_ptr().add(pf) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(b.as_ptr().add(pf + 64) as *const i8, _MM_HINT_T0);
                }

                // 8 chunks × 4 K iterations = 32 FMAs
                let bv = _mm512_loadu_ps(b.get_unchecked(b0));
                acc0 = _mm512_fmadd_ps(a0, bv, acc0);
                let bv = _mm512_loadu_ps(b.get_unchecked(b1));
                acc0 = _mm512_fmadd_ps(a1, bv, acc0);
                let bv = _mm512_loadu_ps(b.get_unchecked(b2));
                acc0 = _mm512_fmadd_ps(a2, bv, acc0);
                let bv = _mm512_loadu_ps(b.get_unchecked(b3));
                acc0 = _mm512_fmadd_ps(a3, bv, acc0);

                let bv = _mm512_loadu_ps(b.get_unchecked(b0 + 16));
                acc1 = _mm512_fmadd_ps(a0, bv, acc1);
                let bv = _mm512_loadu_ps(b.get_unchecked(b1 + 16));
                acc1 = _mm512_fmadd_ps(a1, bv, acc1);
                let bv = _mm512_loadu_ps(b.get_unchecked(b2 + 16));
                acc1 = _mm512_fmadd_ps(a2, bv, acc1);
                let bv = _mm512_loadu_ps(b.get_unchecked(b3 + 16));
                acc1 = _mm512_fmadd_ps(a3, bv, acc1);

                let bv = _mm512_loadu_ps(b.get_unchecked(b0 + 32));
                acc2 = _mm512_fmadd_ps(a0, bv, acc2);
                let bv = _mm512_loadu_ps(b.get_unchecked(b1 + 32));
                acc2 = _mm512_fmadd_ps(a1, bv, acc2);
                let bv = _mm512_loadu_ps(b.get_unchecked(b2 + 32));
                acc2 = _mm512_fmadd_ps(a2, bv, acc2);
                let bv = _mm512_loadu_ps(b.get_unchecked(b3 + 32));
                acc2 = _mm512_fmadd_ps(a3, bv, acc2);

                let bv = _mm512_loadu_ps(b.get_unchecked(b0 + 48));
                acc3 = _mm512_fmadd_ps(a0, bv, acc3);
                let bv = _mm512_loadu_ps(b.get_unchecked(b1 + 48));
                acc3 = _mm512_fmadd_ps(a1, bv, acc3);
                let bv = _mm512_loadu_ps(b.get_unchecked(b2 + 48));
                acc3 = _mm512_fmadd_ps(a2, bv, acc3);
                let bv = _mm512_loadu_ps(b.get_unchecked(b3 + 48));
                acc3 = _mm512_fmadd_ps(a3, bv, acc3);

                let bv = _mm512_loadu_ps(b.get_unchecked(b0 + 64));
                acc4 = _mm512_fmadd_ps(a0, bv, acc4);
                let bv = _mm512_loadu_ps(b.get_unchecked(b1 + 64));
                acc4 = _mm512_fmadd_ps(a1, bv, acc4);
                let bv = _mm512_loadu_ps(b.get_unchecked(b2 + 64));
                acc4 = _mm512_fmadd_ps(a2, bv, acc4);
                let bv = _mm512_loadu_ps(b.get_unchecked(b3 + 64));
                acc4 = _mm512_fmadd_ps(a3, bv, acc4);

                let bv = _mm512_loadu_ps(b.get_unchecked(b0 + 80));
                acc5 = _mm512_fmadd_ps(a0, bv, acc5);
                let bv = _mm512_loadu_ps(b.get_unchecked(b1 + 80));
                acc5 = _mm512_fmadd_ps(a1, bv, acc5);
                let bv = _mm512_loadu_ps(b.get_unchecked(b2 + 80));
                acc5 = _mm512_fmadd_ps(a2, bv, acc5);
                let bv = _mm512_loadu_ps(b.get_unchecked(b3 + 80));
                acc5 = _mm512_fmadd_ps(a3, bv, acc5);

                let bv = _mm512_loadu_ps(b.get_unchecked(b0 + 96));
                acc6 = _mm512_fmadd_ps(a0, bv, acc6);
                let bv = _mm512_loadu_ps(b.get_unchecked(b1 + 96));
                acc6 = _mm512_fmadd_ps(a1, bv, acc6);
                let bv = _mm512_loadu_ps(b.get_unchecked(b2 + 96));
                acc6 = _mm512_fmadd_ps(a2, bv, acc6);
                let bv = _mm512_loadu_ps(b.get_unchecked(b3 + 96));
                acc6 = _mm512_fmadd_ps(a3, bv, acc6);

                let bv = _mm512_loadu_ps(b.get_unchecked(b0 + 112));
                acc7 = _mm512_fmadd_ps(a0, bv, acc7);
                let bv = _mm512_loadu_ps(b.get_unchecked(b1 + 112));
                acc7 = _mm512_fmadd_ps(a1, bv, acc7);
                let bv = _mm512_loadu_ps(b.get_unchecked(b2 + 112));
                acc7 = _mm512_fmadd_ps(a2, bv, acc7);
                let bv = _mm512_loadu_ps(b.get_unchecked(b3 + 112));
                acc7 = _mm512_fmadd_ps(a3, bv, acc7);

                ki += 4;
            }

            // Remainder K (no unroll)
            while ki < k {
                let av = _mm512_set1_ps(*a.get_unchecked(ki));
                let base = ki * n + j0;
                acc0 = _mm512_fmadd_ps(av, _mm512_loadu_ps(b.get_unchecked(base)), acc0);
                acc1 = _mm512_fmadd_ps(av, _mm512_loadu_ps(b.get_unchecked(base + 16)), acc1);
                acc2 = _mm512_fmadd_ps(av, _mm512_loadu_ps(b.get_unchecked(base + 32)), acc2);
                acc3 = _mm512_fmadd_ps(av, _mm512_loadu_ps(b.get_unchecked(base + 48)), acc3);
                acc4 = _mm512_fmadd_ps(av, _mm512_loadu_ps(b.get_unchecked(base + 64)), acc4);
                acc5 = _mm512_fmadd_ps(av, _mm512_loadu_ps(b.get_unchecked(base + 80)), acc5);
                acc6 = _mm512_fmadd_ps(av, _mm512_loadu_ps(b.get_unchecked(base + 96)), acc6);
                acc7 = _mm512_fmadd_ps(av, _mm512_loadu_ps(b.get_unchecked(base + 112)), acc7);
                ki += 1;
            }

            // Store accumulators
            let cp = c.as_mut_ptr().add(j0);
            _mm512_storeu_ps(cp, acc0);
            _mm512_storeu_ps(cp.add(16), acc1);
            _mm512_storeu_ps(cp.add(32), acc2);
            _mm512_storeu_ps(cp.add(48), acc3);
            _mm512_storeu_ps(cp.add(64), acc4);
            _mm512_storeu_ps(cp.add(80), acc5);
            _mm512_storeu_ps(cp.add(96), acc6);
            _mm512_storeu_ps(cp.add(112), acc7);
        }

        // Remainder N (< 128 elements) — process with AVX-512 individual zmm loads
        // and scalar fallback for < 16 elements. No allocation needed.
        if nt_end < n {
            let rem = n - nt_end;
            let rem16 = rem / 16 * 16;

            // Process 16-wide chunks
            for j0 in (0..rem16).step_by(16) {
                let j = nt_end + j0;
                let mut acc = _mm512_setzero_ps();
                for ki in 0..k {
                    let av = _mm512_set1_ps(*a.get_unchecked(ki));
                    let bv = _mm512_loadu_ps(b.get_unchecked(ki * n + j));
                    acc = _mm512_fmadd_ps(av, bv, acc);
                }
                _mm512_storeu_ps(c.as_mut_ptr().add(j), acc);
            }

            // Scalar remainder (< 16 elements)
            for j in (nt_end + rem16)..n {
                let mut sum = 0.0f32;
                for ki in 0..k {
                    sum += *a.get_unchecked(ki) * *b.get_unchecked(ki * n + j);
                }
                *c.get_unchecked_mut(j) = sum;
            }
        }
    }
}

/// Scalar fallback GEMV for non-x86 or non-AVX2 platforms
pub fn gemv_scalar(k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    // 4-way K-unrolled axpy (auto-vectorizable)
    let k4 = k / 4 * 4;
    for ki in (0..k4).step_by(4) {
        let a0 = a[ki];
        let a1 = a[ki + 1];
        let a2 = a[ki + 2];
        let a3 = a[ki + 3];
        let b0 = ki * n;
        let b1 = b0 + n;
        let b2 = b1 + n;
        let b3 = b2 + n;
        for j in 0..n {
            c[j] += a0 * b[b0 + j] + a1 * b[b1 + j] + a2 * b[b2 + j] + a3 * b[b3 + j];
        }
    }

    // Remainder K
    for ki in k4..k {
        let a_k = a[ki];
        let b_start = ki * n;
        for j in 0..n {
            c[j] += a_k * b[b_start + j];
        }
    }
}

/// Dispatch GEMV to best available implementation
pub fn gemv(k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    contract_pre_gemv!(a, b);
    #[cfg(target_arch = "x86_64")]
    {
        // NEGATIVE RESULT (2026-04-05): AVX-512 GEMV is slower than AVX2 at ALL sizes.
        // GEMV is bandwidth-bound (not compute-bound like GEMM). Zen 4 reduces clock
        // ~10-15% during AVX-512 ops, and the wider SIMD can't compensate since the
        // bottleneck is DRAM bandwidth.
        // Measured: 128×512 AVX2=74.7 vs 512=61.6, 4096×4096 AVX2=16.3 vs 512=10.2.
        // AVX-512 GEMV disabled. AVX2 GEMV remains the optimal path.
        // gemv_tiled_avx512() retained for future Intel SPR (no AVX-512 throttle).
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA verified by feature detection above.
            // Slice bounds are checked by the caller (matmul_vector_matrix).
            unsafe {
                if n > GEMV_TILE_THRESHOLD {
                    gemv_tiled_avx2(k, n, a, b, c);
                } else {
                    gemv_avx2(k, n, a, b, c);
                }
            }
            return;
        }
    }
    gemv_scalar(k, n, a, b, c);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemv_basic() {
        // 1×3 @ 3×4 → 1×4
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut c = [0.0f32; 4];

        gemv(3, 4, &a, &b, &mut c);

        // c[j] = 1*B[0,j] + 2*B[1,j] + 3*B[2,j]
        assert!((c[0] - 38.0).abs() < 1e-5);
        assert!((c[1] - 44.0).abs() < 1e-5);
        assert!((c[2] - 50.0).abs() < 1e-5);
        assert!((c[3] - 56.0).abs() < 1e-5);
    }

    #[test]
    fn test_gemv_identity_row_select() {
        // e_1 @ B should give B[1,:]
        let a = [0.0, 1.0, 0.0];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut c = [0.0f32; 3];

        gemv(3, 3, &a, &b, &mut c);

        assert!((c[0] - 4.0).abs() < 1e-5);
        assert!((c[1] - 5.0).abs() < 1e-5);
        assert!((c[2] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_gemv_large_n() {
        // K=2, N=17 (tests AVX2 8-element chunks + scalar remainder)
        let k = 2;
        let n = 17;
        let a = [1.0f32, 2.0];
        let b: Vec<f32> = (0..k * n).map(|i| i as f32).collect();
        let mut c = vec![0.0f32; n];

        gemv(k, n, &a, &b, &mut c);

        // Verify against scalar reference
        for j in 0..n {
            let expected = a[0] * b[j] + a[1] * b[n + j];
            assert!((c[j] - expected).abs() < 1e-4, "c[{j}] = {} expected {expected}", c[j]);
        }
    }

    #[test]
    fn test_gemv_zeros() {
        let a = [0.0f32; 4];
        let b = vec![1.0f32; 4 * 8];
        let mut c = vec![0.0f32; 8];

        gemv(4, 8, &a, &b, &mut c);

        for j in 0..8 {
            assert!((c[j]).abs() < 1e-10);
        }
    }

    /// Test tiled path: N > GEMV_TILE_THRESHOLD triggers tiled kernel
    #[test]
    fn test_gemv_tiled_large_n() {
        let k = 64;
        let n = 8192; // > 4096 → tiled path

        let a: Vec<f32> = (0..k).map(|i| ((i * 7 + 3) % 100) as f32 / 100.0 - 0.5).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i * 13 + 7) % 1000) as f32 / 1000.0 - 0.5).collect();
        let mut c_tiled = vec![0.0f32; n];
        let mut c_scalar = vec![0.0f32; n];

        gemv(k, n, &a, &b, &mut c_tiled);
        gemv_scalar(k, n, &a, &b, &mut c_scalar);

        for j in 0..n {
            let diff = (c_tiled[j] - c_scalar[j]).abs();
            assert!(diff < 1e-2, "j={j}: tiled={} scalar={} diff={diff}", c_tiled[j], c_scalar[j]);
        }
    }

    /// Test tiled path with LLM-size dimensions
    #[test]
    fn test_gemv_tiled_llm_size() {
        let k = 256; // reduced from 4096 for test speed
        let n = 11008;

        let a: Vec<f32> = (0..k).map(|i| ((i * 17 + 31) % 1000) as f32 / 1000.0 - 0.5).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i * 13 + 7) % 1000) as f32 / 1000.0 - 0.5).collect();
        let mut c_tiled = vec![0.0f32; n];
        let mut c_scalar = vec![0.0f32; n];

        gemv(k, n, &a, &b, &mut c_tiled);
        gemv_scalar(k, n, &a, &b, &mut c_scalar);

        for j in 0..n {
            let diff = (c_tiled[j] - c_scalar[j]).abs();
            assert!(diff < 1e-1, "j={j}: tiled={} scalar={} diff={diff}", c_tiled[j], c_scalar[j]);
        }
    }

    /// Test tiled path with N not a multiple of 64 (exercises remainder)
    #[test]
    fn test_gemv_tiled_remainder() {
        let k = 32;
        let n = 5000; // > 4096, not multiple of 64 → remainder = 5000 - 4992 = 8

        let a: Vec<f32> = (0..k).map(|i| ((i * 7 + 3) % 100) as f32 / 100.0 - 0.5).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i * 13 + 7) % 1000) as f32 / 1000.0 - 0.5).collect();
        let mut c_tiled = vec![0.0f32; n];
        let mut c_scalar = vec![0.0f32; n];

        gemv(k, n, &a, &b, &mut c_tiled);
        gemv_scalar(k, n, &a, &b, &mut c_scalar);

        for j in 0..n {
            let diff = (c_tiled[j] - c_scalar[j]).abs();
            assert!(diff < 1e-2, "j={j}: tiled={} scalar={} diff={diff}", c_tiled[j], c_scalar[j]);
        }
    }

    /// Test tiled path with non-multiple-of-4 K (exercises K remainder)
    #[test]
    fn test_gemv_tiled_k_remainder() {
        let k = 67; // not multiple of 4
        let n = 8192;

        let a: Vec<f32> = (0..k).map(|i| ((i * 7 + 3) % 100) as f32 / 100.0 - 0.5).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i * 13 + 7) % 1000) as f32 / 1000.0 - 0.5).collect();
        let mut c_tiled = vec![0.0f32; n];
        let mut c_scalar = vec![0.0f32; n];

        gemv(k, n, &a, &b, &mut c_tiled);
        gemv_scalar(k, n, &a, &b, &mut c_scalar);

        for j in 0..n {
            let diff = (c_tiled[j] - c_scalar[j]).abs();
            assert!(diff < 1e-2, "j={j}: tiled={} scalar={} diff={diff}", c_tiled[j], c_scalar[j]);
        }
    }

    /// FALSIFY-AVX512-GEMV-001: AVX-512 GEMV matches scalar for attention-size dims.
    /// k=128 (head_dim), n=512 (seq_len) — exercises AVX-512 path (n >= 128).
    #[test]
    fn test_gemv_avx512_attention_size() {
        let k = 128;
        let n = 512;

        let a: Vec<f32> = (0..k).map(|i| ((i * 17 + 31) % 1000) as f32 / 1000.0 - 0.5).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i * 13 + 7) % 1000) as f32 / 1000.0 - 0.5).collect();
        let mut c_gemv = vec![0.0f32; n];
        let mut c_scalar = vec![0.0f32; n];

        gemv(k, n, &a, &b, &mut c_gemv);
        gemv_scalar(k, n, &a, &b, &mut c_scalar);

        let max_diff =
            c_gemv.iter().zip(c_scalar.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_diff < 1e-2, "FALSIFY-AVX512-GEMV-001: max diff {max_diff}");
    }

    /// FALSIFY-AVX512-GEMV-002: AVX-512 GEMV with N not a multiple of 128.
    /// Exercises the 16-wide remainder and scalar tail paths.
    #[test]
    fn test_gemv_avx512_remainder() {
        let k = 128;
        let n = 300; // 300 = 256 (2 tiles) + 32 (2 zmm remainder) + 12 (scalar)

        let a: Vec<f32> = (0..k).map(|i| ((i * 7 + 3) % 100) as f32 / 100.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i * 13 + 7) % 1000) as f32 / 1000.0 - 0.5).collect();
        let mut c_gemv = vec![0.0f32; n];
        let mut c_scalar = vec![0.0f32; n];

        gemv(k, n, &a, &b, &mut c_gemv);
        gemv_scalar(k, n, &a, &b, &mut c_scalar);

        let max_diff =
            c_gemv.iter().zip(c_scalar.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_diff < 1e-2, "FALSIFY-AVX512-GEMV-002: max diff {max_diff}");
    }
}
