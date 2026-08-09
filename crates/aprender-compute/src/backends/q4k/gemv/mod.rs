//! Row-major Q4_K matrix-vector multiplication.
//!
//! This module implements row-major GEMV where weights are stored row-first.
//! Includes scalar, AVX2-optimized, and parallel dispatch implementations.

mod scalar;

#[cfg(target_arch = "x86_64")]
mod avx2;

#[cfg(target_arch = "x86_64")]
mod avx512;

#[allow(unused_imports)]
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

/// Fallback for non-x86_64
#[cfg(not(target_arch = "x86_64"))]
#[allow(dead_code)]
fn matmul_q4k_f32_parallel(
    q4k_data: &[u8],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    scalar::matmul_q4k_f32(q4k_data, input, out_dim, in_dim)
}
