//! APR Transformer Helper Functions (PMAT-802)
//!
//! Row-major matmul wrappers and SIMD primitives for APR inference.

use crate::error::Result;
use crate::quantize::{fused_q4k_parallel_matvec, fused_q6k_parallel_matvec};

/// Row-major Q4K matmul wrapper (LAYOUT-001)
///
/// Wraps `fused_q4k_parallel_matvec` with dimension order matching the old API.
/// OLD API: `matmul_q4k_rowmajor(bytes, input, out_dim, in_dim)` - column-major, WRONG
/// NEW API: `matmul_q4k_rowmajor(bytes, input, out_dim, in_dim)` - row-major, CORRECT
///
/// FORBIDDEN: Never use `trueno::backends::q4k::matmul_q4k_f32_colmajor*` for GGUF/APR.
///
/// # Errors
///
/// Returns error if tensor dimensions are mismatched or data is corrupted.
#[inline]
pub(crate) fn matmul_q4k_rowmajor(
    q4k_bytes: &[u8],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Result<Vec<f32>> {
    // fused_q4k_parallel_matvec expects (bytes, input, in_dim, out_dim) - swap order!
    // AUDIT-301 FIX: Propagate error instead of expect()
    fused_q4k_parallel_matvec(q4k_bytes, input, in_dim, out_dim)
}

/// Row-major Q6K matmul wrapper (LAYOUT-001)
///
/// # Errors
///
/// Returns error if tensor dimensions are mismatched or data is corrupted.
#[inline]
pub(crate) fn matmul_q6k_rowmajor(
    q6k_bytes: &[u8],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Result<Vec<f32>> {
    // AUDIT-301 FIX: Propagate error instead of expect()
    fused_q6k_parallel_matvec(q6k_bytes, input, in_dim, out_dim)
}

// ============================================================================
// PMAT-103: SIMD Attention Primitives for 5.0+ tok/s target
// ============================================================================

/// SIMD dot product with AVX2 acceleration (PMAT-103)
///
/// Computes the dot product of two f32 slices using AVX2 when available.
/// Falls back to scalar when AVX2 is not supported or slices are small.
#[inline]
pub(crate) fn simd_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "SIMD dot: length mismatch");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && a.len() >= 8 {
            // SAFETY: AVX2+FMA verified by is_x86_feature_detected!, len >= 8 checked above
            return unsafe { simd_dot_f32_avx2(a, b) };
        }
    }

    // Scalar fallback
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// AVX2 dot product implementation (PMAT-103)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
// SAFETY: Caller must satisfy the documented preconditions
unsafe fn simd_dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: Memory safety ensured by bounds checking before SIMD operations
    unsafe {
        use std::arch::x86_64::{
            _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_loadu_ps,
            _mm256_setzero_ps, _mm_add_ps, _mm_cvtss_f32, _mm_hadd_ps,
        };

        let n = a.len();
        let mut acc = _mm256_setzero_ps();

        // Process 8 elements at a time
        let chunks = n / 8;
        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
            acc = _mm256_fmadd_ps(va, vb, acc);
        }

        // Horizontal sum of 8 floats
        let hi = _mm256_extractf128_ps(acc, 1);
        let lo = _mm256_castps256_ps128(acc);
        let sum128 = _mm_add_ps(lo, hi);
        let sum128 = _mm_hadd_ps(sum128, sum128);
        let sum128 = _mm_hadd_ps(sum128, sum128);
        let mut result = _mm_cvtss_f32(sum128);

        // Handle remaining elements
        let remainder = n % 8;
        if remainder > 0 {
            let start = chunks * 8;
            for i in start..n {
                result += a[i] * b[i];
            }
        }

        result
    }
}

/// SIMD weighted accumulation: out[i] += weight * val[i] (PMAT-103)
///
/// Uses AVX2 FMA for efficient multiply-accumulate operations.
#[inline]
pub(crate) fn simd_add_weighted(out: &mut [f32], val: &[f32], weight: f32) {
    debug_assert_eq!(out.len(), val.len(), "SIMD add_weighted: length mismatch");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && out.len() >= 8 {
            // SAFETY: is_x86_feature_detected! ensures CPU supports AVX2/FMA before calling
            unsafe { simd_add_weighted_avx2(out, val, weight) };
            return;
        }
    }

    // Scalar fallback
    for (o, v) in out.iter_mut().zip(val.iter()) {
        *o += weight * v;
    }
}

/// AVX2 weighted accumulation implementation (PMAT-103)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
// SAFETY: Caller must satisfy the documented preconditions
unsafe fn simd_add_weighted_avx2(out: &mut [f32], val: &[f32], weight: f32) {
    // SAFETY: Memory safety ensured by bounds checking before SIMD operations
    unsafe {
        use std::arch::x86_64::{
            _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps,
        };

        let n = out.len();
        let w = _mm256_set1_ps(weight);

        // Process 8 elements at a time
        let chunks = n / 8;
        for i in 0..chunks {
            let offset = i * 8;
            let v_out = _mm256_loadu_ps(out.as_ptr().add(offset));
            let v_val = _mm256_loadu_ps(val.as_ptr().add(offset));
            let result = _mm256_fmadd_ps(w, v_val, v_out);
            _mm256_storeu_ps(out.as_mut_ptr().add(offset), result);
        }

        // Handle remaining elements
        let remainder = n % 8;
        if remainder > 0 {
            let start = chunks * 8;
            for i in start..n {
                out[i] += weight * val[i];
            }
        }
    }
}

// ============================================================================
// F32 Compute Helpers (PMAT-COMPLY: extracted from mod.rs)
// ============================================================================

/// Parallel threshold for F32 matmul (GH-284: match Q4K path)
const F32_PARALLEL_THRESHOLD: usize = 256;

/// Chunk size for rayon work-stealing (GH-284)
const F32_PARALLEL_CHUNK: usize = 64;

/// F32 matrix-vector multiplication: output[out_dim] = weight[out_dim, in_dim] @ input[in_dim]
///
/// PMAT-095: Weights stored in matvec-optimal [out_dim, in_dim] format.
/// PMAT-103: 4-wide unrolled dot product for cache utilization.
/// GH-284: Rayon parallelism for out_dim >= 256 (matching Q4K path).
pub(crate) fn f32_matmul(input: &[f32], weight: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let seq_len = input.len() / in_dim;
    let expected_size = in_dim * out_dim;

    if weight.len() != expected_size {
        return f32_matmul_scalar(input, weight, in_dim, out_dim);
    }

    let mut output = vec![0.0f32; seq_len * out_dim];

    for s in 0..seq_len {
        let input_start = s * in_dim;
        let input_slice = &input[input_start..input_start + in_dim];
        let out_start = s * out_dim;

        if out_dim >= F32_PARALLEL_THRESHOLD {
            f32_matvec_parallel(
                input_slice,
                weight,
                in_dim,
                out_dim,
                &mut output[out_start..out_start + out_dim],
            );
        } else {
            f32_matvec_sequential(
                input_slice,
                weight,
                in_dim,
                out_dim,
                &mut output[out_start..out_start + out_dim],
            );
        }
    }

    output
}

/// Parallel F32 matvec using rayon work-stealing (GH-284)
fn f32_matvec_parallel(
    input: &[f32],
    weight: &[f32],
    in_dim: usize,
    _out_dim: usize,
    output: &mut [f32],
) {
    use rayon::prelude::*;

    output
        .par_chunks_mut(F32_PARALLEL_CHUNK)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let o_start = chunk_idx * F32_PARALLEL_CHUNK;
            for (local_o, out_val) in out_chunk.iter_mut().enumerate() {
                let o = o_start + local_o;
                *out_val = simd_dot_f32(input, &weight[o * in_dim..(o + 1) * in_dim]);
            }
        });
}

/// Sequential F32 matvec with SIMD dot product (small out_dim)
fn f32_matvec_sequential(
    input: &[f32],
    weight: &[f32],
    in_dim: usize,
    out_dim: usize,
    output: &mut [f32],
) {
    for o in 0..out_dim {
        output[o] = simd_dot_f32(input, &weight[o * in_dim..(o + 1) * in_dim]);
    }
}

/// Scalar fallback for matmul (PMAT-095: weight is [out_dim, in_dim] row-major)
pub(crate) fn f32_matmul_scalar(
    input: &[f32],
    weight: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let seq_len = input.len() / in_dim;
    let mut output = Vec::with_capacity(seq_len * out_dim);

    for s in 0..seq_len {
        let input_start = s * in_dim;
        let input_slice = &input[input_start..input_start + in_dim];

        for o in 0..out_dim {
            let mut sum = 0.0;
            for (i, &input_val) in input_slice.iter().enumerate() {
                let weight_idx = o * in_dim + i;
                if weight_idx < weight.len() {
                    sum += input_val * weight[weight_idx];
                }
            }
            output.push(sum);
        }
    }

    output
}

/// Add bias in-place
pub(crate) fn add_bias_inplace(data: &mut [f32], bias: &[f32]) {
    let dim = bias.len();
    for (i, val) in data.iter_mut().enumerate() {
        *val += bias[i % dim];
    }
}

/// GELU activation in-place (tanh approximation)
///
/// ONE PATH: Per-element delegates to `trueno::gelu_scalar` (UCBD §4).
pub(crate) fn gelu_inplace(data: &mut [f32]) {
    for x in data.iter_mut() {
        *x = trueno::gelu_scalar(*x);
    }
}

/// Apply Rotary Position Embedding (RoPE) to Q or K vectors
///
/// RoPE encodes position information by rotating pairs of elements
/// with position-dependent angles.
pub(crate) fn apply_rope_f32(
    x: &mut [f32],
    position: usize,
    num_heads: usize,
    head_dim: usize,
    rope_theta: f32,
) {
    let half_dim = head_dim / 2;
    let pos_f32 = position as f32;
    let head_dim_f32 = head_dim as f32;

    for h in 0..num_heads {
        let head_start = h * head_dim;
        let idx2_start = head_start + half_dim;

        if idx2_start + half_dim > x.len() {
            continue;
        }

        for i in 0..half_dim {
            let freq = 1.0 / rope_theta.powf(2.0 * i as f32 / head_dim_f32);
            let angle = pos_f32 * freq;
            let (sin_val, cos_val) = angle.sin_cos();

            let x1 = x[head_start + i];
            let x2 = x[idx2_start + i];

            x[head_start + i] = x1 * cos_val - x2 * sin_val;
            x[idx2_start + i] = x1 * sin_val + x2 * cos_val;
        }
    }
}

/// RMSNorm (Root Mean Square Layer Normalization)
///
/// PMAT-094 FIX: Qwen2, LLaMA, Mistral use RMSNorm, NOT LayerNorm.
/// Formula: output = x / sqrt(mean(x^2) + eps) * weight + bias
#[allow(clippy::cast_precision_loss)]
pub(crate) fn rms_norm(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    hidden_dim: usize,
    eps: f32,
) -> Vec<f32> {
    let seq_len = input.len() / hidden_dim;
    let mut output = Vec::with_capacity(input.len());

    for s in 0..seq_len {
        let start = s * hidden_dim;
        let slice = &input[start..start + hidden_dim];

        let sum_sq: f32 = slice.iter().map(|x| x * x).sum();
        let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();

        for (i, &x) in slice.iter().enumerate() {
            let normalized = x / rms;
            let scaled = normalized * weight[i];
            let shifted = if let Some(b) = bias {
                scaled + b[i]
            } else {
                scaled
            };
            output.push(shifted);
        }
    }

    output
}

include!("helpers_simd_dot.rs");

#[cfg(test)]
mod determinism_tests {
    use super::*;

    /// FALSIFY-FFN-GGUF-005 / M-FFN-GGUF-4 step (a):
    /// `f32_matmul` is byte-deterministic across repeated calls.
    ///
    /// SHIP-007 §28 hypothesis: APR's `f32_matvec_parallel` uses rayon
    /// `par_chunks_mut` which COULD produce non-deterministic ordering of
    /// per-output-element computations across runs. F32 accumulation is
    /// non-associative; different orders → different results at the
    /// per-element level. Over 3 layers, per-element differences could
    /// compound to the layer-3 ffn_swigl 18.23× ratio observed in §27.
    ///
    /// This test FALSIFIES the §28 hypothesis at the kernel level.
    /// `par_chunks_mut` parallelizes ACROSS output elements; each output
    /// element is computed by exactly one thread; the per-element dot
    /// product (`simd_dot_f32`) is serial. So the kernel SHOULD be
    /// byte-deterministic across runs.
    ///
    /// If this test PASSES: §28 parallel-reduction hypothesis is
    /// FALSIFIED. SHIP-007 root cause is elsewhere (likely f32 reduction
    /// order DIFFERENCE between APR and GGUF — APR uses
    /// `simd_dot_f32_avx2` 4-wide unrolled FMA; GGUF
    /// `fused_q4k_q8k_parallel_matvec_into` may use different unroll
    /// or block boundaries).
    ///
    /// If this test FAILS: §28 hypothesis CONFIRMED. Fix = ensure
    /// deterministic reduction order in `f32_matvec_parallel`.
    ///
    /// Per `contracts/trace-ffn-sub-block-gguf-v1.yaml` v1.1.0 amendment
    /// (§28 hypothesis test).
    #[test]
    fn falsify_ffn_gguf_005_f32_matmul_byte_deterministic_above_parallel_threshold() {
        // out_dim above F32_PARALLEL_THRESHOLD (256) so f32_matvec_parallel fires
        let in_dim = 128;
        let out_dim = 512;
        let seq_len = 4;

        // Synthetic but reproducible inputs (no random — same byte pattern across runs)
        let input: Vec<f32> = (0..seq_len * in_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.1)
            .collect();
        let weight: Vec<f32> = (0..in_dim * out_dim)
            .map(|i| (((i * 31) % 23) as f32 - 11.0) * 0.05)
            .collect();

        // Run twice with identical inputs
        let result_a = f32_matmul(&input, &weight, in_dim, out_dim);
        let result_b = f32_matmul(&input, &weight, in_dim, out_dim);

        // Byte-identity assertion (not just "close" — the §28 hypothesis is
        // about NON-DETERMINISM, which would manifest as differing bits).
        assert_eq!(
            result_a.len(),
            result_b.len(),
            "matmul output length differs across runs (sanity check failed)"
        );
        for (i, (&a, &b)) in result_a.iter().zip(result_b.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "f32_matmul non-deterministic at element {i}: {a} ({:#x}) vs {b} ({:#x}) — \
                 §28 parallel-reduction hypothesis CONFIRMED. Fix scope = make \
                 f32_matvec_parallel deterministic.",
                a.to_bits(),
                b.to_bits()
            );
        }
    }

    /// Same test but for the `f32_matmul_scalar` fallback path (out_dim
    /// below threshold). Should also be deterministic — no rayon, fully
    /// sequential.
    #[test]
    fn falsify_ffn_gguf_005b_f32_matmul_byte_deterministic_below_parallel_threshold() {
        let in_dim = 128;
        let out_dim = 64; // Below F32_PARALLEL_THRESHOLD = 256
        let seq_len = 1;

        let input: Vec<f32> = (0..seq_len * in_dim)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
            .collect();
        let weight: Vec<f32> = (0..in_dim * out_dim)
            .map(|i| (((i * 23) % 19) as f32 - 9.0) * 0.05)
            .collect();

        let result_a = f32_matmul(&input, &weight, in_dim, out_dim);
        let result_b = f32_matmul(&input, &weight, in_dim, out_dim);

        for (i, (&a, &b)) in result_a.iter().zip(result_b.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "f32_matmul (sequential path) non-deterministic at element {i}"
            );
        }
    }

    /// FALSIFY-FFN-GGUF-006 / M-FFN-GGUF-4 step (b):
    /// APR's `simd_dot_f32_avx2` (AVX2 8-wide FMA) and the scalar
    /// fallback (`iter().zip().map(*).sum()`) produce **byte-identical**
    /// f32 results for typical synthetic inputs.
    ///
    /// SURPRISING EMPIRICAL RESULT (asserted here as a regression
    /// test): on the canonical synthetic input below, AVX2 8-wide FMA
    /// and scalar left-fold sum BOTH produce `0x44191e70 = 612.4756`.
    ///
    /// This **FALSIFIES the refined H2a' hypothesis** at the SIMD-vs-
    /// scalar level. The cumulative APR↔GGUF drift cannot be explained
    /// by APR's SIMD vs APR's scalar path differing on this class of
    /// f32 inputs.
    ///
    /// WHY THIS MATTERS FOR SHIP-007 §22 / §27 / §28:
    ///
    /// Two reduction-order hypotheses are now empirically falsified:
    /// - §28 (parallel-reduction non-determinism, M91 PR #1535):
    ///   FALSIFIED — APR's `f32_matmul` is byte-deterministic
    /// - H2a' (SIMD-vs-scalar reduction-order, this test):
    ///   FALSIFIED — AVX2 and scalar produce byte-identical output
    ///
    /// The SHIP-007 root cause must be at a different boundary:
    /// - H2b: Layer-3-specific upstream divergence (gate or up at L3)
    /// - H2c: Quantization dequant alignment differs at certain layer
    ///        configs
    /// - H2d (NEW post-falsification): APR↔GGUF differ in the
    ///        QUANTIZED matvec path (Q4K dequant + activation
    ///        quantization to Q8K + fused matvec) NOT in F32-vs-F32
    ///        kernels. APR's f32_matmul takes F32 weights (already
    ///        dequantized at load time); GGUF's
    ///        fused_q4k_q8k_parallel_matvec_into takes raw Q4K bytes
    ///        + Q8K-quantized activations and fuses dequant +
    ///        matvec. Different reduction order at the QUANTIZED-
    ///        kernel level (which neither this test nor §28 falsifier
    ///        exercises) is the remaining viable hypothesis.
    ///
    /// REGRESSION-TEST INTENT:
    ///
    /// This test asserts BYTE-IDENTITY between SIMD and scalar paths
    /// for the canonical synthetic input. If a future change makes
    /// them DIFFER (e.g., scalar path is removed and replaced with a
    /// chunked reduction), this test will fail and force re-derivation
    /// of the SHIP-007 hypothesis class.
    ///
    /// Per `contracts/trace-ffn-sub-block-gguf-v1.yaml` v1.2.0 → v1.3.0
    /// refined-hypothesis amendment.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn falsify_ffn_gguf_006_simd_vs_scalar_reduction_order_byte_identity() {
        // Skip if AVX2+FMA not available — the test requires both paths
        // to be exercised and only AVX2 hosts have both.
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!(
                "FALSIFY-FFN-GGUF-006: skipped — host lacks AVX2+FMA (required for SIMD path)"
            );
            return;
        }

        // Canonical synthetic input. Reproducible across runs; pinned
        // to the values that produced 0x44191e70 = 612.4756 on
        // 2026-05-06 via empirical verification.
        let len = 128;
        let a: Vec<f32> = (0..len)
            .map(|i| ((i as f32) - 64.0) * 0.1 + ((i % 7) as f32) * 0.013)
            .collect();
        let b: Vec<f32> = (0..len)
            .map(|i| ((i as f32) * 0.7 - 50.0) * 0.05 + ((i % 11) as f32) * 0.011)
            .collect();

        // SAFETY: AVX2+FMA verified above
        let result_simd = unsafe { simd_dot_f32_avx2(&a, &b) };

        // Scalar reduction: left-fold sum (Rust's default Iterator::sum)
        let result_scalar: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        let bits_simd = result_simd.to_bits();
        let bits_scalar = result_scalar.to_bits();

        // EMPIRICAL FINDING (2026-05-06): both paths produce
        // 0x44191e70 = 612.4756 on this canonical input. Asserted as
        // regression-test invariant.
        assert_eq!(
            bits_simd, bits_scalar,
            "AVX2 SIMD ({:#x} = {result_simd}) and scalar ({:#x} = {result_scalar}) \
             produced DIFFERENT byte patterns — H2a' refined hypothesis would be \
             CONFIRMED. The SHIP-007 root cause may then live in this reduction-\
             order boundary; expand investigation to GGUF's quantized matvec \
             reduction tree.",
            bits_simd, bits_scalar
        );

        // Document the empirical canonical value so a future engineer
        // can re-verify without re-running the test.
        eprintln!(
            "FALSIFY-FFN-GGUF-006: byte-identical at {result_simd} ({bits_simd:#x}). \
             H2a' refined hypothesis FALSIFIED at SIMD-vs-scalar level."
        );
    }

    /// FALSIFY-FFN-GGUF-008 / M-FFN-GGUF-4 step (c) candidate H2d.4:
    /// Compare APR's standalone-dequant + f32_matmul path vs GGUF's
    /// fused q4k+q8k matvec path on the same Q4K weight bytes and
    /// (after Q8K activation quant) the same activation values.
    ///
    /// THE TWO PATHS:
    ///
    /// Path A (APR-style): standalone dequant + F32 matmul
    ///   weights_f32 = dequantize_q4_k_simd(weight_bytes)
    ///   result_a    = f32_matmul(activation_f32, weights_f32, in_dim, out_dim)
    ///
    /// Path B (GGUF-style): Q8K activation quant + fused inline dequant
    ///   (q8k_scales, q8k_quants) = quantize_activations_q8k(activation_f32)
    ///   result_b = fused_q4k_q8k_parallel_matvec_into(
    ///       weight_bytes, q8k_scales, q8k_quants, in_dim, out_dim
    ///   )
    ///
    /// Both compute the same mathematical operation (W @ a) but Path B
    /// has an additional Q8K quantization step on the activation that
    /// Path A doesn't have. The Q8K step rounds to ~7-bit precision per
    /// 256-element super-block.
    ///
    /// EXPECTATION: paths produce DIFFERENT bit patterns due to Q8K
    /// activation precision loss. The test asserts the BIT-LEVEL
    /// difference (analogous to "must differ" at the activation
    /// quantization boundary). The cosine similarity is also asserted
    /// to be high (>0.99) to confirm Q8K precision loss is mathematically
    /// reasonable but not bit-exact.
    ///
    /// WHY THIS MATTERS FOR SHIP-007 §22:
    ///
    /// Three reduction-order hypotheses falsified so far (M91, M92, M93).
    /// The remaining viable hypotheses are H2d.1 (per-block dequant
    /// boundaries), H2d.3 (Q8K activation quant), and H2d.4 (fused
    /// inline dequant differs from standalone).
    ///
    /// This test directly addresses H2d.3 + H2d.4 simultaneously. If
    /// the paths produce DIFFERENT bits (as expected), then SHIP-007
    /// §22 root cause has a concrete mechanism: APR's loader uses
    /// Path A semantics (full F32 dequant + F32 matmul), while GGUF's
    /// inference uses Path B semantics (Q8K activation quant + fused
    /// inline dequant). The cumulative bit-level differences compound
    /// across layers to the §27 18.23× drift.
    ///
    /// If the paths produce BYTE-IDENTICAL bits (unexpected): all
    /// three remaining hypotheses (H2d.1, H2d.3, H2d.4) collapse to
    /// "no measurable kernel-level difference", and SHIP-007 §22
    /// must come from elsewhere entirely (RMSNorm precision,
    /// per-token tokenization, accumulator precision in residual
    /// addition, ...).
    ///
    /// Per `contracts/trace-ffn-sub-block-gguf-v1.yaml` v1.4.0 →
    /// v1.5.0 amendment.
    #[test]
    fn falsify_ffn_gguf_008_fused_vs_standalone_q4k_matvec() {
        use crate::quantize::{
            dequantize_q4_k_simd, fused_q4k_q8k_parallel_matvec_into,
            quantize_activations_q8k_into,
        };

        // Build synthetic Q4K weights: 256 columns × 1 row = 144 bytes
        // (one super-block). Both paths consume this same byte buffer.
        let mut weight_bytes = vec![0u8; 144];
        weight_bytes[0] = 0x00;
        weight_bytes[1] = 0x3C; // f16 d = 1.0
        weight_bytes[2] = 0x00;
        weight_bytes[3] = 0xB4; // f16 dmin = -0.25
        for (i, b) in weight_bytes[4..16].iter_mut().enumerate() {
            *b = ((i * 7 + 3) % 256) as u8;
        }
        for (i, b) in weight_bytes[16..144].iter_mut().enumerate() {
            *b = ((i * 13 + 17) % 256) as u8;
        }

        let in_dim = 256;
        let out_dim = 1;

        // Synthetic F32 activation (256 elements, reproducible)
        let activation: Vec<f32> = (0..in_dim)
            .map(|i| ((i as f32) - 128.0) * 0.05 + ((i % 7) as f32) * 0.01)
            .collect();

        // ---- Path A: standalone dequant + manual f32 dot product ----
        let weights_f32 =
            dequantize_q4_k_simd(&weight_bytes).expect("dequantize_q4_k_simd failed");
        assert_eq!(weights_f32.len(), 256);
        let result_a: f32 = activation.iter().zip(weights_f32.iter()).map(|(x, y)| x * y).sum();

        // ---- Path B: Q8K quant + fused matvec ----
        let mut q8k_scales = vec![0.0f32; 1]; // 1 super-block
        let mut q8k_quants = vec![0i8; in_dim];
        quantize_activations_q8k_into(&activation, &mut q8k_scales, &mut q8k_quants)
            .expect("quantize_activations_q8k_into failed");

        let mut result_b_buf = vec![0.0f32; out_dim];
        fused_q4k_q8k_parallel_matvec_into(
            &weight_bytes,
            &q8k_scales,
            &q8k_quants,
            in_dim,
            out_dim,
            &mut result_b_buf,
        )
        .expect("fused_q4k_q8k_parallel_matvec_into failed");
        let result_b = result_b_buf[0];

        eprintln!(
            "FALSIFY-FFN-GGUF-008: Path A (standalone) = {result_a} ({:#x}); \
             Path B (fused+Q8K) = {result_b} ({:#x}); diff = {}; rel_diff = {}",
            result_a.to_bits(),
            result_b.to_bits(),
            (result_a - result_b).abs(),
            (result_a - result_b).abs() / result_a.abs().max(1e-9)
        );

        // Sanity: both paths should produce mathematically reasonable
        // results (within Q8K precision tolerance ~5%).
        let rel_diff = (result_a - result_b).abs() / result_a.abs().max(1e-9);
        assert!(
            rel_diff < 0.10,
            "Mathematical sanity failed: Path A and Path B disagree by more than 10% \
             (rel_diff = {rel_diff}). Q8K precision loss should be < 5% per super-block."
        );

        // EXPECTED RESULT: paths produce DIFFERENT bit patterns due to
        // Q8K activation quantization. Asserted as the regression-test
        // invariant for the Q8K precision-loss boundary.
        let bits_a = result_a.to_bits();
        let bits_b = result_b.to_bits();
        assert_ne!(
            bits_a, bits_b,
            "FALSIFY-FFN-GGUF-008: Path A and Path B produced BYTE-IDENTICAL output \
             ({result_a} vs {result_b}, both {bits_a:#x}). H2d.3 + H2d.4 hypotheses \
             FALSIFIED at the kernel level. SHIP-007 §22 root cause must be elsewhere \
             (RMSNorm, residual accumulator precision, per-token tokenization, ...). \
             Update contract trace-ffn-sub-block-gguf-v1 v1.4.0 → v1.5.0."
        );
    }
}
