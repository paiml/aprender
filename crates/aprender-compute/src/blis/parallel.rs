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

/// Whether this target and CPU actually have the 8x32 microkernel that
/// [`gemm_blis_parallel_shared_b`]'s hot loop requires.
///
/// It exists ONLY as an x86_64 AVX-512 path. The full-tile branch of that loop
/// (`if mr_block == 8 && nr_block == 32`) contains a single
/// `#[cfg(target_arch = "x86_64")]` statement, so on every other target it compiles to an
/// EMPTY block: the loop runs to completion having written nothing for the aligned tiles
/// and C is returned at its `vec![0.0; ...]` initial value — silently ZERO, not merely
/// inaccurate. Edge tiles take the scalar `else` arm, which is why a shape not aligned to
/// mr/nr came out partially correct rather than wholly zero.
///
/// Pure + named + `pub` for the same reason as [`gemm_should_run_serial`]: the dispatch
/// policy must be one testable statement rather than a condition re-derived at each site.
/// `examples/blis_benchmark.rs` calls it too — it previously re-derived HALF of this
/// (`target_arch` only, missing the runtime `avx512f` half) and so mislabelled its own
/// output on an x86_64 host without AVX-512.
/// Falsifier: `tests::shared_b_falls_back_without_the_8x32_microkernel`.
/// Refs `.planning/debug/gemm-shared-b-parallel.md` (FALSIFY-SHARED-B-001).
#[cfg(feature = "parallel")]
#[must_use]
pub fn shared_b_has_microkernel() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// THE M-partitioning decision of [`gemm_blis_parallel`], in one place.
///
/// Extracted (not copied) so the determinism-gate accessor below and the GEMM
/// itself cannot disagree. A hand-mirrored second implementation would drift,
/// and a drifted copy makes the gate's evidence unreliable in exactly the way
/// it exists to prevent — it would report a partitioning the kernel never used.
///
/// Pure and side-effect free apart from reading `rayon::current_num_threads()`
/// and the physical core count, which is precisely the environment dependence
/// the gate measures.
///
/// # Behaviour is unchanged
///
/// This is a lift of the existing statements, verbatim and in order: the
/// `max_threads` FLOP ladder, `HeijunkaScheduler::default()`, the
/// `num_threads.min(max_threads)` cap, the partition-size choice and the
/// `partition_m` call.
#[cfg(feature = "parallel")]
fn gemm_m_partitions(m: usize, n: usize, k: usize) -> Vec<std::ops::Range<usize>> {
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
    let flops = m * n * k;
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
    scheduler.partition_m(m, ps)
}

/// How many M-partitions `gemm_blis_parallel` would use for these dimensions.
///
/// # Internal determinism-gate observation hook — NOT public API
///
/// `#[doc(hidden)]` and **exempt from semver**: it may change or disappear in any
/// release. Its sole consumer is
/// `crates/aprender-core/tests/gemm_thread_determinism.rs`, which proves that
/// `Tensor::matmul` is byte-identical across rayon pool sizes.
///
/// It CANNOT be `pub(crate)` — the caller is a different crate, so a crate-private
/// item is unreachable and the gate would not compile. That is stated here so the
/// next reader does not "tighten" the visibility and turn a passing gate into a
/// build failure.
///
/// # Why the gate needs it
///
/// Identical hashes across pool sizes prove nothing about the hazard if every
/// pool size partitioned the work identically — the parallel path would simply
/// never have been exercised differently. This is the mechanism-engaged evidence
/// (CLAUDE.md verification discipline 2): a run that cannot show the partitioning
/// MOVED cannot claim to have falsified a partitioning hazard.
///
/// Returns `1` for a GEMM the dispatcher sends down the serial path, which is
/// what that path effectively does with M.
#[doc(hidden)]
#[cfg(feature = "parallel")]
pub fn gemm_partition_count_for(m: usize, n: usize, k: usize) -> usize {
    if gemm_should_run_serial(m, n, k) {
        return 1;
    }
    gemm_m_partitions(m, n, k).len()
}

/// Non-parallel fallback for the determinism-gate accessor.
///
/// Without `parallel` there is no rayon pool and no M-partitioning at all, so the
/// answer is a constant `1`. The gate then observes an unchanging partition count
/// and reports SKIPPED-WITH-EVIDENCE rather than claiming a falsification.
#[doc(hidden)]
#[cfg(not(feature = "parallel"))]
pub fn gemm_partition_count_for(_m: usize, _n: usize, _k: usize) -> usize {
    1
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
    if gemm_should_run_serial(m, n, k) {
        return gemm_blis(m, n, k, a, b, c, None);
    }

    // The thread cap and the M-partitioning both live in `gemm_m_partitions`, so
    // `gemm_partition_count_for` reports the partitioning this call ACTUALLY uses
    // rather than a second implementation of it.
    let partitions = gemm_m_partitions(m, n, k);

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

    let flops = m * n * k;
    if !shared_b_path_available(flops) {
        return gemm_blis(m, n, k, a, b, c, None);
    }

    let num_threads = shared_b_thread_count(flops).min(rayon::current_num_threads());
    let blk = super::cache_topology::blocking_8x32();
    let geo = SharedBGeometry {
        mr: blk.mr, // 8
        nr: blk.nr, // 32
        mc: blk.mc.min(m),
        nc: blk.nc.min(n),
        kc: blk.kc,
    };

    // Shared packed B: one allocation for the largest B panel
    let b_panels = geo.nc.div_ceil(geo.nr);
    let mut packed_b = vec![0.0f32; b_panels * geo.nr * geo.kc];

    let c_ptr = c.as_mut_ptr() as usize;

    for jc in (0..n).step_by(geo.nc) {
        let nc_block = geo.nc.min(n - jc);

        for pc in (0..k).step_by(geo.kc) {
            let kc_block = geo.kc.min(k - pc);

            // Pack B ONCE (sequential) — shared by all threads
            super::compute::pack_b_block_generic(
                b,
                n,
                pc,
                jc,
                kc_block,
                nc_block,
                geo.nr,
                &mut packed_b,
            );
            let block = SharedBBlock {
                a,
                k,
                n,
                c_ptr,
                jc,
                pc,
                nc_block,
                kc_block,
                geo: &geo,
                shared_b: &packed_b,
            };

            // Parallel ic loop: each thread gets a slice of M
            let m_per_thread = m.div_ceil(num_threads).div_ceil(geo.mr) * geo.mr;

            (0..num_threads).into_par_iter().for_each(|tid| {
                let ic_start = tid * m_per_thread;
                if ic_start < m {
                    block.run_slice(ic_start, (ic_start + m_per_thread).min(m), m_per_thread);
                }
            });
        }
    }

    Ok(())
}

/// Whether the shared-B path may run at all: big enough to pay for the
/// packing, and on a target that has the 8×32 microkernel.
///
/// FALSIFY-SHARED-B-001 on aarch64 (gx10, 2026-09-12): the AVX-512 check used
/// to exist only under `cfg(target_arch = "x86_64")`, and the microkernel call
/// in the tile loop is `cfg(x86_64)` inside an `if` with no other arm — so on
/// every other target the full 8×32 tiles were silently SKIPPED and only the
/// edge tiles were computed: max diff 39.2 against the reference at 256³ (the
/// 100×96 row passed only because it sits under the 8M-flop cut). The shared-B
/// path is AVX-512-only by construction; everything else takes the same plain
/// BLIS path a no-AVX-512 x86 box takes.
#[cfg(feature = "parallel")]
fn shared_b_path_available(flops: usize) -> bool {
    // The microkernel half is the one public statement of that policy,
    // [`shared_b_has_microkernel`] (also read by examples/blis_benchmark.rs),
    // so the two cannot drift apart.
    flops >= 8_000_000 && shared_b_has_microkernel()
}

/// Thread budget by problem size. Shared-B means less L3 pressure per thread
/// than the per-thread-B path, so the large tiers use phys_cores/2 (≥ 8).
#[cfg(feature = "parallel")]
fn shared_b_thread_count(flops: usize) -> usize {
    let phys_cores = num_cpus::get_physical();
    if flops < 64_000_000 {
        2.min(phys_cores)
    } else if flops < 512_000_000 {
        4.min(phys_cores)
    } else {
        (phys_cores / 2).max(8).min(phys_cores)
    }
}

/// The 8×32 blocking the shared-B path packs for.
#[cfg(feature = "parallel")]
struct SharedBGeometry {
    mr: usize,
    nr: usize,
    mc: usize,
    nc: usize,
    kc: usize,
}

/// One (jc, pc) block: B is packed once and read by every thread; each thread
/// packs its own A slice and writes a disjoint row range of C.
#[cfg(feature = "parallel")]
struct SharedBBlock<'a> {
    a: &'a [f32],
    k: usize,
    n: usize,
    /// `*mut f32` of C as usize so the block is `Sync`; every write lands in
    /// this thread's own row range (see `run_slice`).
    c_ptr: usize,
    jc: usize,
    pc: usize,
    nc_block: usize,
    kc_block: usize,
    geo: &'a SharedBGeometry,
    shared_b: &'a [f32],
}

#[cfg(feature = "parallel")]
impl SharedBBlock<'_> {
    /// This thread's M-slice `[ic_start, ic_end)`: pack A per mc block into a
    /// thread-local buffer (reused across (jc, pc) iterations — no allocation
    /// per iteration) and run the panel loop over it.
    fn run_slice(&self, ic_start: usize, ic_end: usize, m_per_thread: usize) {
        thread_local! {
            static TL_A: std::cell::RefCell<Vec<f32>> =
                const { std::cell::RefCell::new(Vec::new()) };
        }
        TL_A.with(|tl| {
            let geo = self.geo;
            let needed = m_per_thread.div_ceil(geo.mr) * geo.mr * self.kc_block;
            let mut packed_a = tl.borrow_mut();
            if packed_a.len() < needed {
                packed_a.resize(needed, 0.0);
            }

            for ic in (ic_start..ic_end).step_by(geo.mc) {
                let mc_block = geo.mc.min(ic_end - ic);
                super::packing::pack_a_block(
                    self.a,
                    self.k,
                    ic,
                    self.pc,
                    mc_block,
                    self.kc_block,
                    &mut packed_a,
                );
                self.run_panels(&packed_a, ic, mc_block);
            }
        });
    }

    /// Every (mr × nr) tile of one packed-A block against the shared B.
    fn run_panels(&self, packed_a: &[f32], ic: usize, mc_block: usize) {
        let geo = self.geo;
        let panels_n = self.nc_block.div_ceil(geo.nr);
        for ir_panel in 0..mc_block.div_ceil(geo.mr) {
            let ir = ir_panel * geo.mr;
            let mr_block = geo.mr.min(mc_block - ir);
            for jr_panel in 0..panels_n {
                let jr = jr_panel * geo.nr;
                let nr_block = geo.nr.min(self.nc_block - jr);
                let a_panel = &packed_a[ir_panel * geo.mr * self.kc_block..];
                let b_panel = &self.shared_b[jr_panel * geo.nr * self.kc_block..];
                self.tile(a_panel, b_panel, ic + ir, self.jc + jr, mr_block, nr_block);
            }
        }
    }

    /// One tile at C[row.., col..]: the AVX-512 microkernel for a full 8×32,
    /// the scalar loop for every edge tile (and for every full tile on a
    /// target without the microkernel — reachable only if the path guard is
    /// ever widened).
    fn tile(
        &self,
        a_panel: &[f32],
        b_panel: &[f32],
        row: usize,
        col: usize,
        mr_block: usize,
        nr_block: usize,
    ) {
        #[cfg(target_arch = "x86_64")]
        if mr_block == 8 && nr_block == 32 {
            // SAFETY: the path guard proved avx512f; a_panel/b_panel hold
            // kc_block × 8 and kc_block × 32 packed floats; (row, col) is
            // inside this thread's disjoint row range of C, which outlives
            // the block.
            unsafe {
                super::compute::avx512_microkernel_8x32_rowmajor(
                    self.kc_block,
                    a_panel.as_ptr(),
                    b_panel.as_ptr(),
                    (self.c_ptr as *mut f32).add(row * self.n + col),
                    self.n,
                );
            }
            return;
        }
        self.scalar_tile(a_panel, b_panel, row, col, mr_block, nr_block);
    }

    /// Scalar fallback for edge tiles.
    fn scalar_tile(
        &self,
        a_panel: &[f32],
        b_panel: &[f32],
        row: usize,
        col: usize,
        mr_block: usize,
        nr_block: usize,
    ) {
        let geo = self.geo;
        for ir_local in 0..mr_block {
            for jr_local in 0..nr_block {
                let mut sum = 0.0f32;
                for p in 0..self.kc_block {
                    sum += a_panel[p * geo.mr + ir_local] * b_panel[p * geo.nr + jr_local];
                }
                // SAFETY: (row + ir_local, col + jr_local) is inside this
                // thread's disjoint row range of C (see `run_slice`).
                unsafe {
                    let c = self.c_ptr as *mut f32;
                    *c.add((row + ir_local) * self.n + (col + jr_local)) += sum;
                }
            }
        }
    }
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
