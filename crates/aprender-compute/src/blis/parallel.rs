//! Parallel GEMM with Heijunka (load-leveling) scheduling.
//!
//! Uses Rayon for parallel execution when the `parallel` feature is enabled,
//! with balanced M-dimension partitioning via [`HeijunkaScheduler`].

use crate::error::TruenoError;

use super::compute::{gemm_blis, gemm_blis_with_prepacked_b};
use super::prepacked::PrepackedB;
#[cfg(feature = "parallel")]
use super::{MC, MR};

/// Heijunka (load-leveling) scheduler for parallel GEMM
#[derive(Debug, Clone)]
pub struct HeijunkaScheduler {
    /// Number of threads
    pub num_threads: usize,
    /// Target load variance threshold
    pub variance_threshold: f32,
}

impl Default for HeijunkaScheduler {
    fn default() -> Self {
        #[cfg(feature = "parallel")]
        let threads = rayon::current_num_threads();
        #[cfg(not(feature = "parallel"))]
        let threads = 1;

        Self {
            num_threads: threads,
            variance_threshold: 0.05, // 5% variance target
        }
    }
}

impl HeijunkaScheduler {
    /// Partition M dimension into balanced chunks
    pub fn partition_m(&self, m: usize, mc: usize) -> Vec<std::ops::Range<usize>> {
        let num_blocks = (m + mc - 1) / mc;
        let blocks_per_thread = num_blocks / self.num_threads;
        let remainder = num_blocks % self.num_threads;

        let mut partitions = Vec::with_capacity(self.num_threads);
        let mut start_block = 0;

        for t in 0..self.num_threads {
            let extra = if t < remainder { 1 } else { 0 };
            let thread_blocks = blocks_per_thread + extra;

            let start_row = start_block * mc;
            let end_row = ((start_block + thread_blocks) * mc).min(m);

            if start_row < end_row {
                partitions.push(start_row..end_row);
            }

            start_block += thread_blocks;
        }

        partitions
    }
}

/// Whether a GEMM of these dims should run serially instead of via the rayon
/// parallel path. Pure + unit-testable so the dispatch policy can't silently
/// regress. Serial when: tiny (`<8M` FLOP — rayon ~3µs dispatch dominates) OR a
/// THIN NN-scale GEMM (`8M..64M` FLOP with `n < 192`), where the parallel path's
/// per-thread B-packing + dispatch was measured 2.2x SLOWER than serial
/// (2026-06-13, the `[1024x256]@[256x128]` MLP-layer shape — NN forward/backward
/// is ~all such thin GEMMs). Square sub-64M (`n >= 192`) still parallelizes
/// (~1.24x, cgp 2026-04-05). Falsifier: `tests::nn_thin_gemm_prefers_serial`.
#[cfg(feature = "parallel")]
pub(crate) fn gemm_should_run_serial(m: usize, n: usize, k: usize) -> bool {
    let flops = m * n * k;
    flops < 8_000_000 || (flops < 64_000_000 && n < 192)
}

/// Parallel BLIS GEMM using Rayon
#[cfg(feature = "parallel")]
pub fn gemm_blis_parallel(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Result<(), TruenoError> {
    use rayon::prelude::*;
    contract_pre_amdahl_speedup!();

    // Dimension validation
    if a.len() != m * k || b.len() != k * n || c.len() != m * n {
        return Err(TruenoError::InvalidInput("Dimension mismatch".to_string()));
    }

    // Single-threaded threshold: 8M FLOPs ≈ 200³.
    // Rayon dispatch costs ~3µs. For GEMM ≤128 (~4M FLOP, ~35µs compute),
    // rayon overhead dominates. GEMM 256+ (33M FLOP, ~300µs) benefits.
    let flops = m * n * k;
    if gemm_should_run_serial(m, n, k) {
        return gemm_blis(m, n, k, a, b, c, None);
    }

    // Scale thread count to problem size and cache topology.
    // cgp profile scaling measurements (2026-04-05, Threadripper 7960X 24C/48T):
    //
    //   256x256: 1T=27.8, 2T=34.5 (peak), 4T=35.2 → cap at 2
    //   512x512: 1T=82.6, 4T=176 (peak), 8T=158 → cap at 4
    //   1024x1024: 1T=106, 8T=489 (peak), 12T=417, 16T=450, 24T=426 → cap at 8
    //
    // Root cause for small-problem regression: L3 contention and thread spawn
    // overhead (~40µs per thread::scope) dominate when compute < 1ms.
    // Root cause for 1024 12T regression: cross-CCD L3 thrashing. 8T fits
    // in a single CCD (12 cores, 32MB L3). 12+ threads span both CCDs.
    let phys_cores = num_cpus::get_physical();
    let max_threads = if flops < 64_000_000 {
        // 256³ and below: barely benefits from parallelism
        2.min(phys_cores)
    } else if flops < 512_000_000 {
        // 512³ range: 4T is peak, >4 regresses due to L3 contention
        4.min(phys_cores)
    } else if flops < 4_000_000_000 {
        // 1024³ range (~2B FLOPs): 8T is empirical peak (626 GFLOPS).
        // 12T regresses to 559 GFLOPS due to cross-CCD L3 thrashing — each thread
        // independently packs B, and 12 copies × ~1MB packed_b exceeds one CCD's
        // 32MB L3 share. Capping at 8 keeps all threads on one CCD.
        // Measured 2026-04-05 on Threadripper 7960X (2 CCDs × 12 cores).
        8.min(phys_cores)
    } else {
        // Very large (>4B FLOPs): use phys_cores/2 (one thread per CCD core).
        // Beyond phys_cores/2, SMT contention regresses AVX-512 throughput.
        (phys_cores / 2).max(8).min(phys_cores)
    };

    let mut scheduler = HeijunkaScheduler::default();
    scheduler.num_threads = scheduler.num_threads.min(max_threads);
    let ps = if m <= MC { MR.max(m / scheduler.num_threads) } else { MC };
    let partitions = scheduler.partition_m(m, ps);

    // NEGATIVE RESULT (2026-04-06): shared-B per (jc,pc) block REGRESSED 597→318 GFLOPS.
    // Root cause: Rayon barrier after each K-tile pack forces thread synchronization.
    // With K=4 tiles for 1024×1024, threads stall 4× per GEMM waiting for B pack.
    // Per-thread independent packing (below) avoids synchronization entirely.
    // The 8× redundant B packing (~8MB) fits in L3 (64MB) and eliminates barriers.
    // Future fix: producer-consumer B packing (one thread packs while others compute).
    let c_ptr = c.as_mut_ptr() as usize;

    partitions.into_par_iter().for_each(|m_range| {
        let m_local = m_range.len();
        let m_start = m_range.start;

        let a_local = &a[m_start * k..(m_start + m_local) * k];

        // SAFETY: Each thread accesses a disjoint row range of C.
        let c_local = unsafe {
            let ptr = c_ptr as *mut f32;
            std::slice::from_raw_parts_mut(ptr.add(m_start * n), m_local * n)
        };

        let _ = gemm_blis(m_local, n, k, a_local, b, c_local, None);
    });

    Ok(())
}

/// Parallel GEMM with shared packed-B: pack B once per (jc,pc) block,
/// distribute M-slices across threads. Each thread only packs its own A.
/// This eliminates O(threads) redundant B packings.
///
/// BLIS loop structure:
///   for jc (N tiles):      ← sequential
///     for pc (K tiles):    ← sequential, pack B ONCE
///       for ic (M tiles):  ← PARALLEL across threads
///         pack A_local
///         microkernel(packed_a, shared_packed_b, c_local)
#[cfg(feature = "parallel")]
pub fn gemm_blis_parallel_shared_b(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Result<(), TruenoError> {
    use rayon::prelude::*;

    if a.len() != m * k || b.len() != k * n || c.len() != m * n {
        return Err(TruenoError::InvalidInput("Dimension mismatch".to_string()));
    }

    // For small problems, use single-thread path
    let flops = m * n * k;
    if flops < 8_000_000 {
        return gemm_blis(m, n, k, a, b, c, None);
    }

    // Require AVX-512 for the 8×32 microkernel
    #[cfg(target_arch = "x86_64")]
    if !std::arch::is_x86_feature_detected!("avx512f") {
        return gemm_blis(m, n, k, a, b, c, None);
    }

    let phys_cores = num_cpus::get_physical();
    let max_threads = if flops < 64_000_000 {
        2.min(phys_cores)
    } else if flops < 512_000_000 {
        4.min(phys_cores)
    } else if flops < 4_000_000_000 {
        // Shared-B means less L3 pressure per thread, so we can potentially
        // use more threads than the per-thread-B path. Try phys_cores/2.
        (phys_cores / 2).max(8).min(phys_cores)
    } else {
        (phys_cores / 2).max(8).min(phys_cores)
    };

    let blk = super::cache_topology::blocking_8x32();
    let mr = blk.mr; // 8
    let nr = blk.nr; // 32
    let mc = blk.mc.min(m);
    let nc = blk.nc.min(n);
    let kc = blk.kc;

    // Shared packed B: one allocation for the largest B panel
    let b_panels = (nc + nr - 1) / nr;
    let packed_b_size = b_panels * nr * kc;
    let mut packed_b = vec![0.0f32; packed_b_size];

    let c_ptr = c.as_mut_ptr() as usize;
    let num_threads = max_threads.min(rayon::current_num_threads());

    for jc in (0..n).step_by(nc) {
        let nc_block = nc.min(n - jc);

        for pc in (0..k).step_by(kc) {
            let kc_block = kc.min(k - pc);

            // Pack B ONCE (sequential) — shared by all threads
            super::compute::pack_b_block_generic(
                b,
                n,
                pc,
                jc,
                kc_block,
                nc_block,
                nr,
                &mut packed_b,
            );
            let shared_b: &[f32] = &packed_b;

            // Parallel ic loop: each thread gets a slice of M
            let m_per_thread = ((m + num_threads - 1) / num_threads + mr - 1) / mr * mr;

            (0..num_threads).into_par_iter().for_each(|tid| {
                let ic_start = tid * m_per_thread;
                if ic_start >= m {
                    return;
                }
                let ic_end = (ic_start + m_per_thread).min(m);

                // Thread-local packed A — reuse across (jc, pc) iterations
                // via thread_local! to avoid heap allocation per iteration.
                thread_local! {
                    static TL_A: std::cell::RefCell<Vec<f32>> =
                        const { std::cell::RefCell::new(Vec::new()) };
                }
                TL_A.with(|tl| {
                    let a_panels = (m_per_thread + mr - 1) / mr;
                    let needed = a_panels * mr * kc_block;
                    let mut packed_a = tl.borrow_mut();
                    if packed_a.len() < needed {
                        packed_a.resize(needed, 0.0);
                    }

                    let panels_n = (nc_block + nr - 1) / nr;

                    for ic in (ic_start..ic_end).step_by(mc) {
                        let mc_block = mc.min(ic_end - ic);

                        super::packing::pack_a_block(
                            a,
                            k,
                            ic,
                            pc,
                            mc_block,
                            kc_block,
                            &mut packed_a,
                        );

                        let panels_m = (mc_block + mr - 1) / mr;

                        for ir_panel in 0..panels_m {
                            let ir = ir_panel * mr;
                            let mr_block = mr.min(mc_block - ir);

                            for jr_panel in 0..panels_n {
                                let jr = jr_panel * nr;
                                let nr_block = nr.min(nc_block - jr);

                                let a_panel = &packed_a[ir_panel * mr * kc_block..];
                                let b_panel = &shared_b[jr_panel * nr * kc_block..];

                                if mr_block == 8 && nr_block == 32 {
                                    #[cfg(target_arch = "x86_64")]
                                    unsafe {
                                        super::compute::avx512_microkernel_8x32_rowmajor(
                                            kc_block,
                                            a_panel.as_ptr(),
                                            b_panel.as_ptr(),
                                            (c_ptr as *mut f32).add((ic + ir) * n + (jc + jr)),
                                            n,
                                        );
                                    }
                                } else {
                                    // Scalar fallback for edge tiles
                                    for ir_local in 0..mr_block {
                                        for jr_local in 0..nr_block {
                                            let mut sum = 0.0f32;
                                            for p in 0..kc_block {
                                                sum += a_panel[p * mr + ir_local]
                                                    * b_panel[p * nr + jr_local];
                                            }
                                            unsafe {
                                                let c = c_ptr as *mut f32;
                                                *c.add(
                                                    (ic + ir + ir_local) * n + (jc + jr + jr_local),
                                                ) += sum;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }); // TL_A.with
            });
        }
    }

    Ok(())
}

/// Non-parallel fallback
#[cfg(not(feature = "parallel"))]
pub fn gemm_blis_parallel(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Result<(), TruenoError> {
    gemm_blis(m, n, k, a, b, c, None)
}

/// Parallel BLIS GEMM with pre-packed B matrix.
///
/// Key optimization: the pre-packed B is shared immutably across all threads.
/// Each thread only packs A (which differs per M partition). This eliminates
/// N_threads × redundant B packings per GEMM call.
///
/// # WAPR-KAIZEN Cycle 12
///
/// For 16-thread encoder FFN: eliminates 15 redundant B packings per GEMM call
/// (128 total across 2 GEMMs × 4 layers).
#[cfg(feature = "parallel")]
pub fn gemm_blis_parallel_with_prepacked_b(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    prepacked_b: &PrepackedB,
    c: &mut [f32],
) -> Result<(), TruenoError> {
    use rayon::prelude::*;

    if a.len() != m * k || c.len() != m * n {
        return Err(TruenoError::InvalidInput("Dimension mismatch".to_string()));
    }
    if prepacked_b.k != k || prepacked_b.n != n {
        return Err(TruenoError::InvalidInput(format!(
            "PrepackedB dimension mismatch: expected ({}, {}), got ({}, {})",
            k, n, prepacked_b.k, prepacked_b.n
        )));
    }

    // Small matrices: single-threaded
    if m * n * k < 1_000_000 {
        return gemm_blis_with_prepacked_b(m, n, k, a, prepacked_b, c, None);
    }

    let scheduler = HeijunkaScheduler::default();
    let partitions = scheduler.partition_m(m, MC);

    let c_ptr = c.as_mut_ptr() as usize;

    // Key: prepacked_b is shared (immutable &) across all threads — zero redundant packing
    partitions.into_par_iter().for_each(|m_range| {
        let m_local = m_range.len();
        let m_start = m_range.start;

        let a_local = &a[m_start * k..(m_start + m_local) * k];

        // SAFETY: Each thread accesses a disjoint row range of C.
        // Partitions are non-overlapping by construction in HeijunkaScheduler::partition_m.
        let c_local = unsafe {
            let ptr = c_ptr as *mut f32;
            std::slice::from_raw_parts_mut(ptr.add(m_start * n), m_local * n)
        };

        let _ = gemm_blis_with_prepacked_b(m_local, n, k, a_local, prepacked_b, c_local, None);
    });

    Ok(())
}

/// Non-parallel fallback for pre-packed B
#[cfg(not(feature = "parallel"))]
pub fn gemm_blis_parallel_with_prepacked_b(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    prepacked_b: &PrepackedB,
    c: &mut [f32],
) -> Result<(), TruenoError> {
    gemm_blis_with_prepacked_b(m, n, k, a, prepacked_b, c, None)
}
