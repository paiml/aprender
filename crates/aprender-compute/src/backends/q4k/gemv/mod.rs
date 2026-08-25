//! Row-major Q4_K matrix-vector multiplication.
//!
//! This module implements row-major GEMV where weights are stored row-first.
//! Includes scalar, AVX2-optimized, and parallel dispatch implementations.

mod scalar;

#[cfg(target_arch = "x86_64")]
mod avx2;

#[cfg(target_arch = "x86_64")]
mod avx512;

use super::{SUPER_BLOCK_BYTES, SUPER_BLOCK_SIZE};

// Re-export public API (preserves exact public surface)
pub use scalar::{matmul_q4k_f32, matmul_q4k_f32_scalar};

// Re-export crate-internal API (used by sibling test modules)
#[allow(unused_imports)]
pub(crate) use scalar::compute_chunk_q4k_scalar;

/// Runtime dispatch for Q4K matmul - uses AVX2 if available, otherwise scalar
///
/// # Contract (GH-279)
///
/// Preconditions validated via `debug_assert!` (zero-cost in release):
/// - `q4k_data.len() >= contracts::Q4_K.expected_bytes(out_dim, in_dim)`
/// - `input.len() == in_dim`
///
/// These guarantee that inner-loop `expect()` calls on super-block sub-slices
/// are unreachable: each super-block is sliced to exactly `SUPER_BLOCK_BYTES`
/// (144), and all sub-accesses (`get(4..16)`, `get(16..144)`) fit within that.
#[inline]
pub fn matmul_q4k_f32_dispatch(
    q4k_data: &[u8],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    // GH-279: Contract validation at dispatch boundary.
    // Inner expect() calls are defense-in-depth — provably unreachable when
    // this precondition holds, because every sb_data slice is SUPER_BLOCK_BYTES.
    debug_assert_eq!(input.len(), in_dim, "Q4K dispatch: input length mismatch");
    debug_assert!(
        q4k_data.len() >= crate::contracts::Q4_K.expected_bytes(out_dim, in_dim),
        "Q4K dispatch: buffer too small: {} bytes for [{}, {}] (need {})",
        q4k_data.len(),
        out_dim,
        in_dim,
        crate::contracts::Q4_K.expected_bytes(out_dim, in_dim),
    );

    #[cfg(target_arch = "x86_64")]
    {
        // For large Q4K matmuls (total work >= ~8M elements), use parallel execution.
        // This catches FFN layers (8960×1536 = 13.7M) and lm_head (151936×1536).
        // Threshold tested at 2M (2026-04-05) but REGRESSED: 1536×1536 went from
        // 17→14 GFLOPS because parallel overhead (~40µs) dominates at 277µs total.
        // Contract: cgp-q4k-parallel-threshold-v1.yaml documents negative result.
        let total_work = out_dim * in_dim;
        if total_work >= 8_000_000 {
            return matmul_q4k_f32_parallel(q4k_data, input, out_dim, in_dim);
        }

        // AVX-512: 16-wide dequant+FMA (2× throughput vs AVX2)
        // Contract: avx512-q4k-v1.yaml (C-AVX512-Q4K-001, C-AVX512-Q4K-002)
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("fma")
        {
            return unsafe { avx512::matmul_q4k_f32_avx512(q4k_data, input, out_dim, in_dim) };
        }

        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: We just verified AVX2 + FMA are available
            return unsafe { avx2::matmul_q4k_f32_avx2(q4k_data, input, out_dim, in_dim) };
        }
    }

    // Fallback to scalar with 4-way unroll
    scalar::matmul_q4k_f32(q4k_data, input, out_dim, in_dim)
}

/// Fused Q4_K matrix-vector multiply for GGML column-major layout
///
/// Computes: output = input @ Q4K_weight (GGML convention: y = x @ W)
/// where weight is stored in Q4_K format with GGML column-major super-block organization.
///
/// # GGML Column-Major Layout (PMAT-103)
///
/// For a weight tensor with shape [ne0, ne1] in GGML notation:
/// - ne0 is the output dimension (rows)
/// - ne1 is the input/reduction dimension (columns)
/// - Elements are stored column-major: W[i,j] at offset i + j*ne0
/// - Each column j (length ne0) contains weights from input[j] to all outputs
/// - Super-blocks are organized by columns: column j uses super-blocks [j*blocks_per_col, (j+1)*blocks_per_col)
///
/// This matches GGUF tensor storage and enables fused kernel execution without transposition.
///
/// # Arguments
/// * `q4k_data` - Raw Q4K bytes in GGML column-major layout [ne0, ne1]
/// * `input` - F32 input vector [ne1] (input/reduction dimension)
/// * `ne0` - Size of output dimension (rows in GGML, output size)
/// * `ne1` - Size of input/reduction dimension (columns in GGML, input size)
///
/// # Returns
/// F32 output vector [ne0]
///
/// # Example
/// ```rust,ignore
/// // GGUF ffn_gate: shape [intermediate_dim, hidden_dim] = [8960, 1536]
/// // Computes: intermediate = hidden @ ffn_gate
/// let output = matmul_q4k_f32_colmajor(&q4k_bytes, &hidden, 8960, 1536);
/// // output has 8960 elements
/// ```

// ============================================================================
// Parallel Execution Helpers
// ============================================================================

#[cfg(target_arch = "x86_64")]
fn matmul_q4k_f32_parallel(
    q4k_data: &[u8],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    use std::thread;

    // Use fewer threads with larger chunks for better cache efficiency
    let num_threads = thread::available_parallelism().map(|p| p.get()).unwrap_or(4).min(12);

    let chunk_size = (out_dim + num_threads - 1) / num_threads;
    let num_blocks_per_row = (in_dim + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let row_bytes = num_blocks_per_row * SUPER_BLOCK_BYTES;

    // Uninit: compute_chunk_* writes *out_val = hsum(acc) for every element.
    let mut output: Vec<f32> = Vec::with_capacity(out_dim);
    // SAFETY: Each thread's compute_chunk writes every element in its chunk (SET).
    unsafe {
        output.set_len(out_dim);
    }
    let has_avx512 = is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("fma");
    let has_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");

    thread::scope(|s| {
        let input_ref = input;
        let q4k_ref = q4k_data;
        // CGP-DBUF: iterate directly instead of collecting into Vec.
        for (chunk_idx, chunk) in output.chunks_mut(chunk_size).enumerate() {
            let start_row = chunk_idx * chunk_size;

            s.spawn(move || {
                if has_avx512 {
                    // Contract: avx512-q4k-v1.yaml (C-AVX512-Q4K-001)
                    unsafe {
                        avx512::compute_chunk_q4k_avx512(
                            q4k_ref,
                            input_ref,
                            chunk,
                            start_row,
                            out_dim,
                            in_dim,
                            num_blocks_per_row,
                            row_bytes,
                        );
                    }
                } else if has_avx2 {
                    unsafe {
                        avx2::compute_chunk_q4k_avx2(
                            q4k_ref,
                            input_ref,
                            chunk,
                            start_row,
                            out_dim,
                            in_dim,
                            num_blocks_per_row,
                            row_bytes,
                        );
                    }
                } else {
                    scalar::compute_chunk_q4k_scalar(
                        q4k_ref,
                        input_ref,
                        chunk,
                        start_row,
                        out_dim,
                        in_dim,
                        num_blocks_per_row,
                        row_bytes,
                    );
                }
            });
        }
    });

    output
}

/// Q4_K GEMV on non-x86_64: SCALAR PER ROW, BUT ACTUALLY PARALLEL (#2567).
///
/// This function was named `..._parallel` and called `scalar::matmul_q4k_f32`
/// — the serial routine — directly. On every ARM machine the hottest kernel in
/// quantized inference therefore ran single-threaded, silently, and no gate
/// could see it: the numbers are correct and only the speed is wrong.
///
/// The fix is not new SIMD. The parallel structure above has NOTHING
/// x86-specific in it — chunking rows across threads is architecture-neutral,
/// and the x86 path already falls back to `scalar::compute_chunk_q4k_scalar`
/// per chunk when neither AVX-512 nor AVX2 is present. That is exactly the
/// shape needed here, so aarch64 gets N-core parallelism over the same scalar
/// inner kernel it was already using, with no new unsafe code and no change to
/// the arithmetic.
///
/// A NEON inner kernel remains open (#2567 also covers the SIMD half). This
/// closes the half that is a plain structural omission rather than missing
/// intrinsics — and it is the half the function's own NAME already promised.
///
/// MEASURED ON gx10 (GB10, aarch64, 20 cores), 8960x1536, release, 10 warmups:
///
///   serial   median 2.17 ms   (2.166 - 2.181, 0.7% spread)
///   parallel median 1.79 ms   (1.757 - 1.811, 3% spread)
///   speedup  1.21x
///
/// 1.21x from up to 12 threads is modest, and the reason is in this file
/// already: thread::scope spawns threads on EVERY CALL, and the x86 threshold
/// comment above puts that overhead at ~40us. Twelve spawns is ~0.48 ms, about
/// 27% of the 1.79 ms parallel time. It is not DRAM bandwidth — 7.4 MiB in
/// 1.79 ms is 4.3 GB/s, far below what GB10 unified memory sustains.
///
/// So a thread POOL would recover most of the remaining headroom, and the NEON
/// kernel would cut the per-byte work the threads are dividing. Both are
/// larger changes; this one is the structural defect, fixed, with the number
/// stated rather than rounded up.
#[cfg(not(target_arch = "x86_64"))]
fn matmul_q4k_f32_parallel(
    q4k_data: &[u8],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    use std::thread;

    // Same policy as the x86 path: fewer threads with larger chunks, for cache
    // efficiency rather than raw thread count.
    let num_threads = thread::available_parallelism().map(|p| p.get()).unwrap_or(4).min(12);

    // One thread is not parallel; fall through rather than pay scope overhead.
    if num_threads <= 1 || out_dim == 0 {
        return scalar::matmul_q4k_f32(q4k_data, input, out_dim, in_dim);
    }

    let chunk_size = out_dim.div_ceil(num_threads);
    let num_blocks_per_row = in_dim.div_ceil(SUPER_BLOCK_SIZE);
    let row_bytes = num_blocks_per_row * SUPER_BLOCK_BYTES;

    let mut output: Vec<f32> = Vec::with_capacity(out_dim);
    // SAFETY: every chunk's compute_chunk_q4k_scalar writes every element in
    // its chunk before any read, exactly as the x86_64 path above relies on.
    unsafe {
        output.set_len(out_dim);
    }

    thread::scope(|s| {
        let input_ref = input;
        let q4k_ref = q4k_data;
        for (chunk_idx, chunk) in output.chunks_mut(chunk_size).enumerate() {
            let start_row = chunk_idx * chunk_size;
            s.spawn(move || {
                scalar::compute_chunk_q4k_scalar(
                    q4k_ref,
                    input_ref,
                    chunk,
                    start_row,
                    out_dim,
                    in_dim,
                    num_blocks_per_row,
                    row_bytes,
                );
            });
        }
    });

    output
}

// ── #2567: the aarch64 "parallel" path is parallel, and still exact ────────
#[cfg(test)]
mod issue_2567_aarch64_parallel_tests {
    use super::*;

    /// Build a Q4_K buffer of the right size. The CONTENTS do not need to be
    /// meaningful for this test: both paths dequantise the same bytes with the
    /// same routine, so any deterministic filling exercises the property under
    /// test — that CHUNKING ACROSS THREADS changes nothing.
    fn q4k_buffer(out_dim: usize, in_dim: usize) -> Vec<u8> {
        let blocks = in_dim.div_ceil(SUPER_BLOCK_SIZE);
        let n = out_dim * blocks * SUPER_BLOCK_BYTES;
        // Bytes are kept <= 0x3F so that any f16 scale field built from
        // them has a small exponent and decodes FINITE. Unconstrained bytes
        // produce NaN scales, which trips the AVX-512 path's own dequant
        // postcondition (`result.iter().all(|v| v.is_finite())`) — a fixture
        // defect that reads as a kernel failure.
        (0..n).map(|i| ((i * 31 + 7) % 0x40) as u8).collect()
    }

    fn input_vec(in_dim: usize) -> Vec<f32> {
        (0..in_dim).map(|i| ((i % 17) as f32 - 8.0) * 0.125).collect()
    }

    /// THE CORRECTNESS PROPERTY, tested on EVERY architecture: the CHUNK
    /// BOUNDARY is invisible.
    ///
    /// Each output row is an independent dot product, so splitting rows across
    /// threads must give bit-identical results however the split falls. This
    /// is the only thing the aarch64 change actually claims, and it is about
    /// chunking rather than about NEON — so it runs here on x86 too.
    ///
    /// TWO WRONG ORACLES CAME FIRST, and both are worth recording because each
    /// looked like a bug in the new code:
    ///
    ///   1. Comparing `matmul_q4k_f32_parallel` against `scalar::matmul_q4k_f32`
    ///      failed on x86 with `-15746.977 != -15746.998`. On x86 that function
    ///      IS the AVX path; ~1e-6 is FMA reassociation, not a chunking fault.
    ///
    ///   2. Comparing chunked `compute_chunk_q4k_scalar` against
    ///      `scalar::matmul_q4k_f32` failed EVEN AT ONE CHUNK, with
    ///      `-15746.977 != -15746.947`. The two scalar routines are not each
    ///      other's oracle: `matmul_q4k_f32` accumulates into a 4-wide array
    ///      (`acc = [0.0f32; 4]`) while `compute_chunk_q4k_scalar` uses a
    ///      single running `sum`. Different summation order, ~2e-6, and it
    ///      predates this change — the x86 parallel path has always had it in
    ///      its own scalar fallback.
    ///
    /// So the assertion is chunk-invariance of ONE kernel against ITSELF.
    #[test]
    fn the_chunk_boundary_is_invisible() {
        for (out_dim, in_dim) in [(1, 256), (7, 256), (16, 512), (33, 256), (64, 256)] {
            let data = q4k_buffer(out_dim, in_dim);
            let input = input_vec(in_dim);
            let num_blocks_per_row = in_dim.div_ceil(SUPER_BLOCK_SIZE);
            let row_bytes = num_blocks_per_row * SUPER_BLOCK_BYTES;

            let run = |threads: usize| -> Vec<f32> {
                let chunk_size = out_dim.div_ceil(threads);
                let mut out = vec![0.0f32; out_dim];
                for (idx, chunk) in out.chunks_mut(chunk_size).enumerate() {
                    scalar::compute_chunk_q4k_scalar(
                        &data,
                        &input,
                        chunk,
                        idx * chunk_size,
                        out_dim,
                        in_dim,
                        num_blocks_per_row,
                        row_bytes,
                    );
                }
                out
            };

            let one = run(1);
            for threads in [2usize, 3, 5, 12, 64] {
                let many = run(threads);
                for (i, (a, b)) in one.iter().zip(many.iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "[{out_dim}, {in_dim}] threads={threads} row {i}: \
                         {a} != {b}. A chunk boundary changed the arithmetic."
                    );
                }
            }
        }
    }

    /// The aarch64 path now uses `compute_chunk_q4k_scalar`, so its numbers
    /// shift from the old `matmul_q4k_f32` by that summation-order difference.
    /// It must be that difference and nothing larger — a real defect would not
    /// sit at 1e-5 relative.
    #[test]
    fn the_kernel_switch_is_only_summation_order() {
        for (out_dim, in_dim) in [(7, 256), (33, 256), (64, 512)] {
            let data = q4k_buffer(out_dim, in_dim);
            let input = input_vec(in_dim);
            let old = scalar::matmul_q4k_f32(&data, &input, out_dim, in_dim);

            let num_blocks_per_row = in_dim.div_ceil(SUPER_BLOCK_SIZE);
            let row_bytes = num_blocks_per_row * SUPER_BLOCK_BYTES;
            let mut new = vec![0.0f32; out_dim];
            scalar::compute_chunk_q4k_scalar(
                &data,
                &input,
                &mut new,
                0,
                out_dim,
                in_dim,
                num_blocks_per_row,
                row_bytes,
            );

            for (i, (a, b)) in old.iter().zip(new.iter()).enumerate() {
                // The synthetic buffer is arbitrary bytes, so some rows decode
                // f16 scales that are NaN or infinite. Both kernels must AGREE
                // on that — a row finite in one and not the other would be a
                // real defect — but a NaN pair carries no magnitude to compare.
                assert_eq!(
                    a.is_finite(),
                    b.is_finite(),
                    "[{out_dim}, {in_dim}] row {i}: finiteness disagrees ({a} vs {b})"
                );
                if !a.is_finite() {
                    continue;
                }
                let denom = a.abs().max(b.abs()).max(1.0);
                let rel = (a - b).abs() / denom;
                assert!(
                    rel < 1e-4,
                    "[{out_dim}, {in_dim}] row {i}: {a} vs {b} (rel {rel:.3e}) — \
                     larger than summation-order reassociation explains"
                );
            }
        }
    }

    /// Degenerate shapes must not panic or read out of bounds — the short
    /// final chunk and the single-row case are where an off-by-one in the
    /// chunk arithmetic would show.
    #[test]
    fn degenerate_shapes_are_safe() {
        for (out_dim, in_dim) in [(1, 256), (2, 256), (3, 256)] {
            let data = q4k_buffer(out_dim, in_dim);
            let input = input_vec(in_dim);
            let out = matmul_q4k_f32_parallel(&data, &input, out_dim, in_dim);
            assert_eq!(out.len(), out_dim);
            // Not asserting finiteness: the buffer is arbitrary bytes and
            // some decode to NaN f16 scales. What matters here is that the
            // chunk arithmetic does not panic or read out of bounds.
        }
    }
}

// ── #2567: the measurement, run explicitly on the host that has the defect ─
//
// `--ignored` on purpose: this is a TIMING observation, and a wall-clock
// assertion in a normally-running test is the class that has failed eleven
// times in this repo. It prints numbers for a human to read and asserts only
// that the parallel path is not SLOWER, which is a correctness-of-dispatch
// claim rather than a performance threshold.
//
//   cargo test -p aprender-compute --lib issue_2567_measure -- --ignored --nocapture
#[cfg(test)]
mod issue_2567_measure {
    use super::*;
    use std::time::Instant;

    #[test]
    #[ignore = "timing observation; run explicitly on the host under test"]
    fn parallel_vs_serial_on_an_ffn_shaped_matmul() {
        // 8960x1536 is the FFN layer the x86 threshold comment names as the
        // case the parallel dispatch exists to catch.
        let (out_dim, in_dim) = (8960usize, 1536usize);
        let blocks = in_dim.div_ceil(SUPER_BLOCK_SIZE);
        // See q4k_buffer: bytes <= 0x3F keep every f16 scale finite.
        let data: Vec<u8> = (0..out_dim * blocks * SUPER_BLOCK_BYTES)
            .map(|i| ((i * 31 + 7) % 0x40) as u8)
            .collect();
        let input: Vec<f32> = (0..in_dim).map(|i| ((i % 17) as f32 - 8.0) * 0.125).collect();

        let time = |f: &dyn Fn() -> Vec<f32>| -> Vec<f64> {
            let mut out = Vec::new();
            // TEN warmups, not two. A first measurement on gx10 discarded two
            // and produced a BIMODAL serial series — 6.59, 6.60, 6.60, 4.55,
            // 2.13, 2.14, 2.15 ms — whose median (4.55) sat in the empty space
            // between the two modes and inflated the reported speedup. The
            // machine was still settling (cache/DVFS). A median is only
            // meaningful over a settled distribution, so the warmup runs until
            // it is one.
            for i in 0..17 {
                let t = Instant::now();
                let r = f();
                std::hint::black_box(&r);
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                if i >= 10 {
                    out.push(ms);
                }
            }
            out
        };

        let serial = time(&|| scalar::matmul_q4k_f32(&data, &input, out_dim, in_dim));
        let parallel = time(&|| matmul_q4k_f32_parallel(&data, &input, out_dim, in_dim));

        let median = |v: &[f64]| {
            let mut s = v.to_vec();
            s.sort_by(|a, b| a.partial_cmp(b).expect("finite timings"));
            s[s.len() / 2]
        };
        let (ms_s, ms_p) = (median(&serial), median(&parallel));
        println!("arch            {}", std::env::consts::ARCH);
        println!("threads         {:?}", std::thread::available_parallelism());
        println!("shape           {out_dim}x{in_dim}");
        println!("serial   median {ms_s:.2} ms   samples {serial:?}");
        println!("parallel median {ms_p:.2} ms   samples {parallel:?}");
        println!("speedup         {:.2}x", ms_s / ms_p);

        // NOT a threshold. Only: dispatching to the parallel path must not be
        // slower than the serial one it replaced. On x86 both are the same
        // family so this is near 1.0; on aarch64 before this fix it was
        // exactly 1.0 by construction, because "parallel" called serial.
        assert!(
            ms_p <= ms_s * 1.5,
            "parallel ({ms_p:.2} ms) is materially slower than serial ({ms_s:.2} ms)"
        );
    }
}
