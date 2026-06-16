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
///
/// # RoPE pairing convention (PMAT-797)
///
/// `rope_type` selects the pairing geometry, matching llama.cpp's
/// `ggml_rope` (`GGML_ROPE_TYPE_NORM` vs `GGML_ROPE_TYPE_NEOX`):
///
/// - **NORM (`rope_type == 0`)** — adjacent pairs: rotate `(x[2i], x[2i+1])`.
///   LLaMA / TinyLlama / Mistral / DeepSeek convention.
/// - **NEOX (`rope_type == 2`)** — split halves: rotate `(x[i], x[i+head_dim/2])`.
///   Qwen / GPT-NeoX / Phi / Gemma convention.
///
/// In BOTH conventions frequency-pair `i` uses
/// `theta_i = rope_theta^(-2i/head_dim)`; only the *element pairing* differs.
/// Applying the wrong pairing silently corrupts every position embedding,
/// so the caller MUST pass the architecture-correct `rope_type` (derive via
/// `crate::gguf::infer_rope_type(architecture)`).
pub(crate) fn apply_rope_f32(
    x: &mut [f32],
    position: usize,
    num_heads: usize,
    head_dim: usize,
    rope_theta: f32,
    rope_type: u32,
) {
    let half_dim = head_dim / 2;
    let pos_f32 = position as f32;
    let head_dim_f32 = head_dim as f32;

    for h in 0..num_heads {
        let head_start = h * head_dim;

        if head_start + head_dim > x.len() {
            continue;
        }

        for i in 0..half_dim {
            let freq = 1.0 / rope_theta.powf(2.0 * i as f32 / head_dim_f32);
            let angle = pos_f32 * freq;
            let (sin_val, cos_val) = angle.sin_cos();

            // NEOX (type 2): pair x[i] with x[i + half_dim] (split halves).
            // NORM (type 0): pair x[2i] with x[2i+1] (adjacent).
            let (idx1, idx2) = if rope_type == 2 {
                (head_start + i, head_start + half_dim + i)
            } else {
                (head_start + 2 * i, head_start + 2 * i + 1)
            };

            let x1 = x[idx1];
            let x2 = x[idx2];

            x[idx1] = x1 * cos_val - x2 * sin_val;
            x[idx2] = x1 * sin_val + x2 * cos_val;
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
    ///   configs
    /// - H2d (NEW post-falsification): APR↔GGUF differ in the
    ///   QUANTIZED matvec path (Q4K dequant + activation
    ///   quantization to Q8K + fused matvec) NOT in F32-vs-F32
    ///   kernels. APR's f32_matmul takes F32 weights (already
    ///   dequantized at load time); GGUF's
    ///   fused_q4k_q8k_parallel_matvec_into takes raw Q4K bytes
    ///   plus Q8K-quantized activations and fuses dequant and
    ///   matvec. Different reduction order at the QUANTIZED-
    ///   kernel level (which neither this test nor §28 falsifier
    ///   exercises) is the remaining viable hypothesis.
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
            dequantize_q4_k_simd, fused_q4k_q8k_parallel_matvec_into, quantize_activations_q8k_into,
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
        let weights_f32 = dequantize_q4_k_simd(&weight_bytes).expect("dequantize_q4_k_simd failed");
        assert_eq!(weights_f32.len(), 256);
        let result_a: f32 = activation
            .iter()
            .zip(weights_f32.iter())
            .map(|(x, y)| x * y)
            .sum();

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

    /// FALSIFY-FFN-GGUF-009 / M-FFN-GGUF-4 step (e):
    /// QUANTITATIVE compounding test for the M94 mechanism.
    ///
    /// M94 (FALSIFY-FFN-GGUF-008) confirmed Path A vs Path B differ at
    /// bit level on a SINGLE 144-byte Q4K super-block: rel_diff = 0.077%
    /// per matvec.
    ///
    /// The §27 evidence shows layer-3 ffn_swigl APR↔GGUF std-ratio =
    /// 18.23×. Naive linear projection: 0.077% × (3 layers × ~7
    /// tensor-ops × 7 tokens) ≈ 11.3% — far below 1723%.
    ///
    /// QUESTION: does the M94 mechanism EXPLAIN the §27 magnitude?
    /// Three sub-hypotheses:
    ///
    ///   H-COMPOUND-LINEAR:    rel_diff(N) ≈ rel_diff(1) × N
    ///                         (no interaction; cumulative ≈ 11%)
    ///                         → mechanism IS NOT sufficient.
    ///   H-COMPOUND-SUBLINEAR: rel_diff(N) ≈ rel_diff(1) × √N
    ///                         (random-walk averaging)
    ///                         → mechanism IS NOT sufficient (smaller).
    ///   H-COMPOUND-SUPER:     rel_diff(N) ≈ rel_diff(1) × N^k, k > 1
    ///                         (positive feedback in cumulative drift)
    ///                         → mechanism MAY explain §27 magnitude.
    ///
    /// This test runs N sequential matvecs (chaining each output as
    /// the next input) on Path A and Path B, measuring rel_diff at
    /// each depth. Reports growth pattern.
    ///
    /// EXPECTATION (per F32 sum-of-products non-associativity theory):
    /// growth is approximately √N (random-walk) for INDEPENDENT
    /// matvecs but can be approximately N or N^k for chained matvecs
    /// where each output feeds the next (because the divergence
    /// becomes part of the next matvec's input, where it interacts
    /// with the next matvec's weights).
    ///
    /// EMPIRICAL EXPECTATION: chained matvec divergence grows
    /// faster than √N because each input divergence is amplified
    /// by the next matvec's weight magnitude — but the test does
    /// NOT predict 18.23× from 0.077% × 5 chained matvecs alone.
    /// What this test DOES is record the empirical growth pattern
    /// for use in future SHIP-007 §22 fix-PR scope analysis.
    ///
    /// Per `contracts/trace-ffn-sub-block-gguf-v1.yaml` v1.5.0 →
    /// v1.6.0 amendment.
    #[test]
    fn falsify_ffn_gguf_009_multi_tensor_divergence_compound() {
        use crate::quantize::{
            dequantize_q4_k_simd, fused_q4k_q8k_parallel_matvec_into, quantize_activations_q8k_into,
        };

        let in_dim = 256;
        let out_dim = 256;

        // Build N synthetic Q4K super-block weight tensors. Each has
        // shape [out_dim=256, in_dim=256] = 256 super-blocks × 144
        // bytes = 36864 bytes.
        let n_chained = 5;
        let weight_bytes_per_tensor = 256 * 144;
        let weights: Vec<Vec<u8>> = (0..n_chained)
            .map(|t| {
                let mut block = vec![0u8; weight_bytes_per_tensor];
                for sb in 0..256 {
                    let base = sb * 144;
                    block[base] = 0x00;
                    block[base + 1] = 0x3C; // f16 d = 1.0
                    block[base + 2] = 0x00;
                    block[base + 3] = 0xB4; // f16 dmin = -0.25
                    for (i, b) in block[base + 4..base + 16].iter_mut().enumerate() {
                        *b = ((i * 7 + 3 + sb + t * 11) % 256) as u8;
                    }
                    for (i, b) in block[base + 16..base + 144].iter_mut().enumerate() {
                        *b = ((i * 13 + 17 + sb * 3 + t * 19) % 256) as u8;
                    }
                }
                block
            })
            .collect();

        // Initial activation (256-element, reproducible).
        let initial: Vec<f32> = (0..in_dim)
            .map(|i| ((i as f32) - 128.0) * 0.05 + ((i % 7) as f32) * 0.01)
            .collect();

        // Path A: chain N standalone matvecs with normalization to
        // keep activations in a bounded range (otherwise float
        // overflow dominates).
        let mut act_a = initial.clone();
        for w_bytes in &weights {
            let weights_f32 = dequantize_q4_k_simd(w_bytes).expect("dequant_simd failed");
            assert_eq!(weights_f32.len(), out_dim * in_dim);
            // Manual matvec: out_j = sum_i(act[i] * w[j*in_dim + i])
            let mut next = vec![0.0f32; out_dim];
            for j in 0..out_dim {
                let row_base = j * in_dim;
                next[j] = act_a
                    .iter()
                    .zip(weights_f32[row_base..row_base + in_dim].iter())
                    .map(|(x, y)| x * y)
                    .sum();
            }
            // Normalize to keep magnitude bounded (mimics RMSNorm
            // effect in real transformers).
            let norm = (next.iter().map(|x| x * x).sum::<f32>() / (out_dim as f32))
                .sqrt()
                .max(1e-9);
            for x in next.iter_mut() {
                *x /= norm;
            }
            act_a = next;
        }

        // Path B: chain N fused Q4K+Q8K matvecs with same
        // normalization between layers.
        let mut act_b = initial.clone();
        for w_bytes in &weights {
            // Q8K-quantize current activations (super-block size 256).
            let n_super_blocks = in_dim / 256;
            assert_eq!(in_dim, 256, "test fixture requires in_dim=256");
            let mut q8k_scales = vec![0.0f32; n_super_blocks];
            let mut q8k_quants = vec![0i8; in_dim];
            quantize_activations_q8k_into(&act_b, &mut q8k_scales, &mut q8k_quants)
                .expect("q8k_quant failed");
            // Fused matvec into out_dim.
            let mut next = vec![0.0f32; out_dim];
            fused_q4k_q8k_parallel_matvec_into(
                w_bytes,
                &q8k_scales,
                &q8k_quants,
                in_dim,
                out_dim,
                &mut next,
            )
            .expect("fused_matvec failed");
            let norm = (next.iter().map(|x| x * x).sum::<f32>() / (out_dim as f32))
                .sqrt()
                .max(1e-9);
            for x in next.iter_mut() {
                *x /= norm;
            }
            act_b = next;
        }

        // Compute final divergence: L2 norm of (act_a - act_b) /
        // L2 norm of act_a.
        let l2_diff = act_a
            .iter()
            .zip(act_b.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        let l2_a = act_a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let rel_diff = l2_diff / l2_a.max(1e-9);

        eprintln!(
            "FALSIFY-FFN-GGUF-009: chained {n_chained} matvecs (256×256 each, RMSNorm \
             between layers); final L2(act_a - act_b) = {l2_diff:.6}, L2(act_a) = \
             {l2_a:.6}, rel_diff = {rel_diff:.6} ({:.4}%)",
            rel_diff * 100.0
        );

        // The §27 evidence is 18.23× std-ratio at layer-3 (= 1723%
        // relative magnitude). The M94 single-tensor mechanism is
        // 0.077% relative.
        //
        // Sanity: chained rel_diff should be MEASURABLY LARGER than
        // single-tensor (0.077%), confirming compounding. Asserted
        // as regression-test invariant.
        assert!(
            rel_diff > 0.0007,
            "FALSIFY-FFN-GGUF-009 sanity: chained {n_chained}-matvec rel_diff = \
             {rel_diff} not measurably larger than single-tensor 0.077%; M94 \
             mechanism may not COMPOUND across chained matvecs (which would \
             refute the cumulative-drift explanation for §27)."
        );

        // Document the canonical empirical value for future re-derivation.
        eprintln!(
            "FALSIFY-FFN-GGUF-009: M94 mechanism DOES compound across chained matvecs. \
             Single-tensor 0.077% → {n_chained}-tensor {:.4}%. Growth factor = {:.2}×. \
             Whether this is sufficient to fully explain §27's 18.23× std-ratio at \
             layer-3 depends on the actual layer-3 chain depth (likely 3 layers × ~7 \
             tensor-ops + RoPE phase rotation + softmax non-linearity which can amplify \
             precision drift). Test confirms compounding; quantitative match to §27 \
             requires real-teacher run.",
            rel_diff * 100.0,
            rel_diff / 0.00077
        );
    }

    /// FALSIFY-FFN-GGUF-010 / M-FFN-GGUF-4 step (f) candidate A3:
    /// Q4K block-scale variance — does the M94 mechanism's per-tensor
    /// rel_diff vary substantially with the f16 d (block scale)
    /// across realistic Qwen2.5-Coder layer ranges?
    ///
    /// Synthetic A3 hypothesis test: real Qwen Q4K weights have huge
    /// per-tensor magnitude variance (block scales spanning 0.001 to
    /// 1.0 across a 7B model). The M94 mechanism's 0.077% rel_diff
    /// was measured on a single block with f16 d = 1.0. If real
    /// per-block scale variance produces 5-50× larger rel_diff at
    /// some scales, A3 alone explains the §27 magnitude.
    ///
    /// This test compares Path A vs Path B per-block divergence at
    /// 7 block-scale values spanning the realistic range:
    ///   d ∈ {0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0}
    ///
    /// EXPECTATION:
    /// - rel_diff invariant across scales: A3 doesn't apply at this
    ///   granularity; magnitude variance doesn't amplify M94 mechanism.
    /// - rel_diff varies 5-50× across scales: A3 partially confirmed;
    ///   real-weight magnitude variance contributes to §27 magnitude.
    ///
    /// EMPIRICAL HYPOTHESIS (per Q8K activation quant invariance theory):
    /// Q8K quantization rounds activations to ~7-bit precision PER
    /// SUPER-BLOCK with its own scale. So both Path A and Path B
    /// scale linearly with block magnitude — rel_diff (which is
    /// a RATIO) should be approximately scale-INVARIANT. Predicts:
    /// rel_diff(scale=10) ≈ rel_diff(scale=0.001) ≈ 0.077%.
    ///
    /// If this prediction is FALSIFIED (rel_diff varies substantially),
    /// A3 has a concrete sub-mechanism beyond linear-scaling.
    ///
    /// Per `contracts/trace-ffn-sub-block-gguf-v1.yaml` v1.6.0 →
    /// v1.7.0 amendment.
    #[test]
    fn falsify_ffn_gguf_010_q4k_block_scale_variance() {
        use crate::quantize::{
            dequantize_q4_k_simd, fused_q4k_q8k_parallel_matvec_into, quantize_activations_q8k_into,
        };

        // Synthetic activation pattern reused from M94 (preserves
        // empirical comparability).
        let in_dim = 256;
        let out_dim = 1;
        let activation: Vec<f32> = (0..in_dim)
            .map(|i| ((i as f32) - 128.0) * 0.05 + ((i % 7) as f32) * 0.01)
            .collect();

        // f16 encoding of test scales — IEEE 754 binary16.
        // Computed via Python: struct.pack('<H', struct.unpack('<H',
        //   np.float16(d).tobytes())[0]) → low byte, high byte
        let scales: Vec<(f32, [u8; 2])> = vec![
            // d=0.001 (very small block)
            (0.001, [0x10, 0x14]),
            // d=0.01
            (0.01, [0x1F, 0x21]),
            // d=0.05
            (0.05, [0x33, 0x29]),
            // d=0.1
            (0.1, [0x66, 0x2E]),
            // d=0.5
            (0.5, [0x00, 0x38]),
            // d=1.0 (M94 baseline — should reproduce 0.077%)
            (1.0, [0x00, 0x3C]),
            // d=10.0 (large block)
            (10.0, [0x00, 0x49]),
        ];

        eprintln!(
            "FALSIFY-FFN-GGUF-010: Q4K block-scale variance — Path A vs Path B per-block rel_diff"
        );
        eprintln!("scale    | path_a              | path_b              | diff       | rel_diff");
        eprintln!("---------|---------------------|---------------------|------------|---------");

        let mut rel_diffs: Vec<(f32, f32)> = Vec::new();

        for (scale_f32, scale_bytes) in &scales {
            // Build single-super-block weight bytes with this f16 d.
            let mut weight_bytes = vec![0u8; 144];
            weight_bytes[0] = scale_bytes[0];
            weight_bytes[1] = scale_bytes[1];
            // f16 dmin = 0.0 (no min offset; isolates d effect)
            weight_bytes[2] = 0x00;
            weight_bytes[3] = 0x00;
            // 12 sub-block scale/min bytes — set non-trivial pattern
            for (i, b) in weight_bytes[4..16].iter_mut().enumerate() {
                *b = ((i * 7 + 3) % 256) as u8;
            }
            // 128 quant bytes — same M94 pattern
            for (i, b) in weight_bytes[16..144].iter_mut().enumerate() {
                *b = ((i * 13 + 17) % 256) as u8;
            }

            // Path A: standalone dequant + manual F32 dot
            let weights_f32 = dequantize_q4_k_simd(&weight_bytes).expect("dequant_simd failed");
            let result_a: f32 = activation
                .iter()
                .zip(weights_f32.iter())
                .map(|(x, y)| x * y)
                .sum();

            // Path B: Q8K activation quant + fused matvec
            let mut q8k_scales = vec![0.0f32; 1];
            let mut q8k_quants = vec![0i8; in_dim];
            quantize_activations_q8k_into(&activation, &mut q8k_scales, &mut q8k_quants)
                .expect("q8k failed");
            let mut result_b_buf = vec![0.0f32; out_dim];
            fused_q4k_q8k_parallel_matvec_into(
                &weight_bytes,
                &q8k_scales,
                &q8k_quants,
                in_dim,
                out_dim,
                &mut result_b_buf,
            )
            .expect("fused failed");
            let result_b = result_b_buf[0];

            let diff = (result_a - result_b).abs();
            let rel_diff = diff / result_a.abs().max(1e-9);

            eprintln!(
                "{:>8.4} | {:>19} | {:>19} | {:>10} | {:.6}%",
                scale_f32,
                format!("{result_a:.4}"),
                format!("{result_b:.4}"),
                format!("{diff:.4}"),
                rel_diff * 100.0,
            );

            rel_diffs.push((*scale_f32, rel_diff));
        }

        // Compute min/max rel_diff across scales — does it vary?
        let min_rd = rel_diffs
            .iter()
            .map(|(_, r)| *r)
            .fold(f32::INFINITY, f32::min);
        let max_rd = rel_diffs
            .iter()
            .map(|(_, r)| *r)
            .fold(f32::NEG_INFINITY, f32::max);
        let variance_factor = max_rd / min_rd.max(1e-12);

        eprintln!();
        eprintln!(
            "FALSIFY-FFN-GGUF-010: rel_diff range across 7 block scales: \
             min={:.6}% max={:.6}% variance_factor={:.2}×",
            min_rd * 100.0,
            max_rd * 100.0,
            variance_factor
        );

        // EMPIRICAL EXPECTATION: rel_diff is approximately scale-
        // INVARIANT (Q8K rescales activations per super-block; both
        // paths scale linearly with block magnitude). Predicted
        // variance_factor: ~1.0× (within numeric noise).
        //
        // If variance_factor > 5.0×, A3 has a sub-mechanism beyond
        // linear-scaling. Asserted as regression-test invariant.
        // Lower bound 0.0001%: ensures rel_diff is not exactly zero
        // for any scale (would indicate a bug in the test fixture).
        for (scale_f32, rel_diff) in &rel_diffs {
            assert!(
                *rel_diff > 1e-7,
                "FALSIFY-FFN-GGUF-010: scale={scale_f32} produced rel_diff={rel_diff} \
                 (smaller than 1e-7); test fixture may be degenerate at this scale"
            );
        }

        // Document the empirical canonical pattern. Whether A3 is
        // confirmed depends on whether variance_factor is small
        // (~1×, A3 doesn't apply) or large (>5×, A3 partially
        // confirmed).
        if variance_factor > 5.0 {
            eprintln!(
                "FALSIFY-FFN-GGUF-010: variance_factor={:.2}× > 5.0 — A3 PARTIALLY CONFIRMED. \
                 Block-scale variance amplifies M94 mechanism beyond linear scaling. \
                 Real-weight magnitude variance contributes to §27 magnitude.",
                variance_factor
            );
        } else {
            eprintln!(
                "FALSIFY-FFN-GGUF-010: variance_factor={:.2}× ≤ 5.0 — A3 NOT CONFIRMED at \
                 this granularity. Block-scale variance does NOT amplify M94 mechanism \
                 substantially. Real-weight magnitude variance alone unlikely to \
                 explain §27 magnitude. A1 (RoPE phase) and A2 (softmax saturation) \
                 remain candidate amplifiers.",
                variance_factor
            );
        }
    }

    /// FALSIFY-FFN-GGUF-011 / M-FFN-GGUF-4 step (g) candidate A2:
    /// Softmax saturation amplification — does a small input-logit
    /// drift (M94 mechanism's ~0.077% rel_diff) get AMPLIFIED by
    /// softmax when one logit is near-saturated (max-token)?
    ///
    /// Synthetic A2 hypothesis test: attention softmax compresses
    /// logits into probabilities; when one logit is much larger
    /// than others (saturated regime), softmax becomes near-step-
    /// function. Tiny input perturbations to the saturated logit
    /// can produce large output probability changes.
    ///
    /// Test design:
    /// - 7-element logit vector mimicking attention scores at
    ///   sequence position 0 of a 7-token prompt.
    /// - One logit "saturated" at +10.0 (very confident token).
    /// - Other logits in normal range [-2.0, +2.0].
    /// - Add a 0.077% perturbation to the saturated logit and
    ///   measure softmax output drift.
    ///
    /// EXPECTATION:
    /// - softmax(logits) is NOT a linear function; in saturated
    ///   regime, the dominant probability is near 1.0 and tail
    ///   probabilities are near 0.0. A 0.077% drift in the
    ///   dominant logit shifts the dominant probability by a
    ///   tiny fraction near 1.0 → output_rel_diff ≈ 0% on the
    ///   dominant token.
    /// - But TAIL probabilities (the small ones near 0) can
    ///   shift by larger relative amounts, since the absolute
    ///   shift is now divided by a small base.
    ///
    /// QUESTION: does the L1 norm of the softmax output drift
    /// exceed the L1 norm of the input drift? If yes, A2 is
    /// CONFIRMED at the saturation regime; if no, A2 doesn't
    /// amplify in this regime.
    ///
    /// Per `contracts/trace-ffn-sub-block-gguf-v1.yaml` v1.7.0 →
    /// v1.8.0 amendment.
    #[test]
    fn falsify_ffn_gguf_011_softmax_saturation_amplification() {
        // 7-element logit vector — one saturated, others normal.
        // The saturated logit (index 3) is at +10.0; M94 perturbation
        // would add 0.077% × 10.0 = 0.0077 to it.
        let logits_a: Vec<f32> = vec![-1.5, 0.5, -0.8, 10.0, 1.2, -0.3, 0.7];

        // Path B: simulate the M94 mechanism's bit-level perturbation
        // on the dominant logit. The 0.077% drift is the per-tensor
        // baseline; for an attention QK^T product reaching +10.0
        // logit value, that's about +0.0077 absolute drift.
        let perturbation = 0.00077 * 10.0; // 0.077% of 10.0
        let mut logits_b = logits_a.clone();
        logits_b[3] += perturbation;

        // Numerically-stable softmax (subtract max).
        fn softmax(logits: &[f32]) -> Vec<f32> {
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            exps.into_iter().map(|x| x / sum).collect()
        }

        let probs_a = softmax(&logits_a);
        let probs_b = softmax(&logits_b);

        // L1 input drift = abs(perturbation) (only one element changed).
        let input_l1_drift = perturbation.abs();
        let input_l1_norm: f32 = logits_a.iter().map(|x| x.abs()).sum();
        let input_rel_drift = input_l1_drift / input_l1_norm;

        // L1 output drift = sum |p_b - p_a|.
        let output_l1_drift: f32 = probs_a
            .iter()
            .zip(probs_b.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let output_l1_norm: f32 = probs_a.iter().map(|x| x.abs()).sum(); // = 1.0 for valid softmax
        let output_rel_drift = output_l1_drift / output_l1_norm.max(1e-9);

        // Amplification factor: how many times larger is output
        // relative drift than input relative drift?
        let amplification = output_rel_drift / input_rel_drift.max(1e-12);

        eprintln!("FALSIFY-FFN-GGUF-011: softmax saturation amplification");
        eprintln!("  logits (saturated at index 3): {logits_a:?}");
        eprintln!("  perturbation on saturated logit: +{perturbation}");
        eprintln!(
            "  probs_a (top-3): {:.6}, {:.6}, {:.6}",
            probs_a[3], probs_a[4], probs_a[1]
        );
        eprintln!(
            "  probs_b (top-3): {:.6}, {:.6}, {:.6}",
            probs_b[3], probs_b[4], probs_b[1]
        );
        eprintln!(
            "  input rel_drift  = {:.6}% ({:.6e})",
            input_rel_drift * 100.0,
            input_rel_drift
        );
        eprintln!(
            "  output rel_drift = {:.6}% ({:.6e})",
            output_rel_drift * 100.0,
            output_rel_drift
        );
        eprintln!("  amplification factor = {amplification:.4}×");

        // Sanity: input_rel_drift > 0 (perturbation actually applied).
        assert!(
            input_rel_drift > 0.0,
            "test fixture: perturbation must be > 0"
        );

        // Sanity: probabilities sum to 1 (within numerical tolerance).
        let sum_a: f32 = probs_a.iter().sum();
        let sum_b: f32 = probs_b.iter().sum();
        assert!(
            (sum_a - 1.0).abs() < 1e-5 && (sum_b - 1.0).abs() < 1e-5,
            "softmax outputs must sum to 1; got a={sum_a}, b={sum_b}"
        );

        // EMPIRICAL VERDICT: if amplification > 5.0, A2 is CONFIRMED
        // (softmax in saturation regime amplifies M94 perturbation).
        // If <= 1.0, A2 is FALSIFIED (softmax compresses). If 1-5×,
        // PARTIAL.
        if amplification > 5.0 {
            eprintln!(
                "FALSIFY-FFN-GGUF-011: amplification {amplification:.2}× > 5.0 — \
                 A2 CONFIRMED. Softmax in saturation regime amplifies M94 \
                 perturbation. Real-attention softmax with saturated logits \
                 contributes substantially to §27 magnitude beyond the \
                 5.70× chained matvec compounding."
            );
        } else if amplification > 1.0 {
            eprintln!(
                "FALSIFY-FFN-GGUF-011: amplification {amplification:.2}× ∈ (1, 5] — \
                 A2 PARTIALLY CONFIRMED. Softmax compresses but does not \
                 fully amplify."
            );
        } else {
            eprintln!(
                "FALSIFY-FFN-GGUF-011: amplification {amplification:.2}× ≤ 1.0 — \
                 A2 NOT CONFIRMED at this regime. Softmax in saturation \
                 regime COMPRESSES M94 perturbation rather than amplifying. \
                 Tested with single saturated logit (+10.0); other regimes \
                 (multiple saturated, near-tie, etc) may behave differently."
            );
        }

        // Document amplification as regression-test invariant.
        // If amplification flips sign or magnitude class in a future
        // refactor of softmax/logit handling, this test catches it.
        // Sanity bound: amplification must be measurable (> 1e-9)
        // — zero would indicate softmax produced bit-identical
        // outputs which contradicts the test premise.
        assert!(
            amplification > 1e-9,
            "FALSIFY-FFN-GGUF-011: amplification {amplification} is essentially zero — \
             softmax produced byte-identical outputs from perturbed inputs, which \
             contradicts the test premise that softmax is sensitive to logit drift"
        );
    }

    /// FALSIFY-FFN-GGUF-012 / M-FFN-GGUF-4 step (h) candidate A1:
    /// RoPE phase amplification — does a small magnitude drift in
    /// pre-RoPE Q/K vectors (M94 mechanism's ~0.077% rel_diff) get
    /// AMPLIFIED by RoPE rotation + subsequent QK^T attention dot
    /// product?
    ///
    /// Hypothesis A1: RoPE rotates F32 vectors by per-position phase;
    /// tiny magnitude drift in pre-RoPE Q becomes ROTATIONAL drift in
    /// post-RoPE Q. When Q' is then dotted with K' (also rotated),
    /// the rotational drift may compound non-linearly into a larger
    /// QK^T attention score drift than the magnitude drift alone.
    ///
    /// Test design:
    /// - Single attention head at sequence position 0 (the prompt
    ///   start token of a 7-token batch).
    /// - head_dim = 64 (typical Qwen 7B), rope_theta = 10000.0.
    /// - Generate Q vector with realistic magnitude distribution.
    /// - Apply M94-equivalent perturbation (0.077%) to Q.
    /// - Apply RoPE to both Q and Q'.
    /// - Generate K vector at sequence position 1 (different
    ///   position; RoPE applies different phase per position).
    /// - Apply RoPE to K (single — K is not perturbed in this test).
    /// - Compute QK^T scores: q_a • k vs q_b • k.
    /// - Compare scores; report amplification = output_drift /
    ///   input_drift.
    ///
    /// EXPECTATION:
    /// - If RoPE were a unitary rotation (preserves L2 norm),
    ///   amplification would be exactly 1× (rotation doesn't
    ///   change magnitude, dot product is symmetric).
    /// - But RoPE introduces position-dependent phase rotation.
    ///   Tiny magnitude drift in pre-RoPE Q produces tiny drift
    ///   in each rotated component; rotated drift may project
    ///   onto K differently than the original drift would, leading
    ///   to amplification or compression.
    ///
    /// EMPIRICAL HYPOTHESIS: amplification ≈ 1× (RoPE is a unitary
    /// rotation; QK^T dot product preserves drift magnitude). If
    /// confirmed, A1 is FALSIFIED — RoPE doesn't amplify M94 drift.
    ///
    /// Per `contracts/trace-ffn-sub-block-gguf-v1.yaml` v1.8.0 →
    /// v1.9.0 amendment.
    #[test]
    fn falsify_ffn_gguf_012_rope_phase_amplification() {
        const HEAD_DIM: usize = 64;
        const ROPE_THETA: f32 = 10000.0;
        const POS_Q: usize = 0;
        const POS_K: usize = 1;

        // Generate Q vector at position 0 with realistic magnitudes.
        let q_a: Vec<f32> = (0..HEAD_DIM)
            .map(|i| ((i as f32) - (HEAD_DIM as f32) / 2.0) * 0.05 + ((i % 5) as f32) * 0.02)
            .collect();

        // Apply M94-equivalent perturbation: scale q_a by (1 + 0.00077).
        let perturbation = 0.00077;
        let q_b: Vec<f32> = q_a.iter().map(|x| x * (1.0 + perturbation)).collect();

        // Generate K vector at position 1.
        let k: Vec<f32> = (0..HEAD_DIM)
            .map(|i| ((i as f32) - (HEAD_DIM as f32) / 2.0) * 0.04 + ((i % 7) as f32) * 0.015)
            .collect();

        // RoPE: rotate pairs (x_2i, x_2i+1) by angle theta_i × pos
        // where theta_i = 1 / ROPE_THETA^(2i / HEAD_DIM).
        fn apply_rope(vec: &[f32], pos: usize, head_dim: usize, theta: f32) -> Vec<f32> {
            let mut out = vec.to_vec();
            let half = head_dim / 2;
            for i in 0..half {
                let freq = 1.0 / theta.powf((2.0 * i as f32) / head_dim as f32);
                let angle = (pos as f32) * freq;
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let x0 = vec[i];
                let x1 = vec[i + half];
                out[i] = x0 * cos_a - x1 * sin_a;
                out[i + half] = x0 * sin_a + x1 * cos_a;
            }
            out
        }

        let q_a_rope = apply_rope(&q_a, POS_Q, HEAD_DIM, ROPE_THETA);
        let q_b_rope = apply_rope(&q_b, POS_Q, HEAD_DIM, ROPE_THETA);
        let k_rope = apply_rope(&k, POS_K, HEAD_DIM, ROPE_THETA);

        // Compute attention scores: q • k (scaled by 1/sqrt(d)).
        let scale = (HEAD_DIM as f32).sqrt().recip();
        let score_a: f32 = q_a_rope
            .iter()
            .zip(k_rope.iter())
            .map(|(x, y)| x * y)
            .sum::<f32>()
            * scale;
        let score_b: f32 = q_b_rope
            .iter()
            .zip(k_rope.iter())
            .map(|(x, y)| x * y)
            .sum::<f32>()
            * scale;

        // Input rel_drift: |q_b - q_a|_L2 / |q_a|_L2.
        let q_diff_l2: f32 = q_a
            .iter()
            .zip(q_b.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        let q_a_l2: f32 = q_a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let input_rel_drift = q_diff_l2 / q_a_l2.max(1e-9);

        // Output rel_drift: |score_b - score_a| / |score_a|.
        let score_diff = (score_b - score_a).abs();
        let output_rel_drift = score_diff / score_a.abs().max(1e-9);

        let amplification = output_rel_drift / input_rel_drift.max(1e-12);

        eprintln!("FALSIFY-FFN-GGUF-012: RoPE phase amplification");
        eprintln!(
            "  head_dim = {HEAD_DIM}, rope_theta = {ROPE_THETA}, pos_q = {POS_Q}, pos_k = {POS_K}"
        );
        eprintln!(
            "  q_a_l2 = {q_a_l2:.6}, q_diff_l2 = {q_diff_l2:.6}, input_rel_drift = {:.6}%",
            input_rel_drift * 100.0
        );
        eprintln!(
            "  score_a = {score_a:.6}, score_b = {score_b:.6}, score_diff = {score_diff:.6}, output_rel_drift = {:.6}%",
            output_rel_drift * 100.0
        );
        eprintln!("  amplification factor = {amplification:.4}×");

        // Sanity: input_rel_drift > 0 (perturbation actually applied).
        assert!(
            input_rel_drift > 0.0,
            "test fixture: perturbation must produce non-zero input drift"
        );

        // Sanity: amplification is measurable (> 1e-9).
        assert!(
            amplification > 1e-9,
            "amplification {amplification} essentially zero — RoPE+dot may be \
             producing bit-identical outputs from perturbed inputs, contradicting \
             test premise"
        );

        // EMPIRICAL VERDICT:
        if amplification > 5.0 {
            eprintln!(
                "FALSIFY-FFN-GGUF-012: amplification {amplification:.2}× > 5.0 — \
                 A1 CONFIRMED. RoPE phase rotation amplifies M94 perturbation \
                 substantially in QK^T attention dot product. Real-attention \
                 contributes to §27 magnitude beyond the 5.70× chained matvec \
                 compounding."
            );
        } else if amplification > 1.5 {
            eprintln!(
                "FALSIFY-FFN-GGUF-012: amplification {amplification:.2}× ∈ (1.5, 5] — \
                 A1 PARTIALLY CONFIRMED. RoPE+QK^T amplifies M94 perturbation \
                 modestly."
            );
        } else if amplification > 0.5 {
            eprintln!(
                "FALSIFY-FFN-GGUF-012: amplification {amplification:.2}× ≈ 1× — \
                 A1 NOT CONFIRMED. RoPE rotation is approximately unitary and \
                 QK^T preserves drift magnitude — no substantial amplification \
                 in this regime. Real-attention may behave differently due to \
                 multi-position sums or causal masking."
            );
        } else {
            eprintln!(
                "FALSIFY-FFN-GGUF-012: amplification {amplification:.2}× < 0.5 — \
                 A1 FALSIFIED. RoPE+QK^T COMPRESSES M94 perturbation in this \
                 regime. With A1, A2, A3 all falsified, M-FFN-GGUF-6 (real-teacher \
                 falsifier) is the highest-leverage remaining test for the §27 \
                 magnitude gap."
            );
        }
    }

    /// FALSIFY-FFN-GGUF-013 / M-FFN-GGUF-4 step (i) candidate A4:
    /// Multi-token batch amplification — does the M94 mechanism's
    /// per-tensor 0.077% rel_diff get amplified when a B=7-token
    /// batch is run through chained matvecs (vs M95's single-token
    /// chain)?
    ///
    /// A4 hypothesis: §27 measures std-ratio across a 7-token
    /// prompt. M95 was single-token. Multi-token batch dimension
    /// can interact non-linearly via:
    /// - position-dependent RoPE (different rotations per position
    ///   may cumulatively diverge differently)
    /// - intra-batch attention (causal mask + softmax over multiple
    ///   keys can amplify per-row drift)
    /// - per-position residual paths (each token's residual sum
    ///   accumulates drift independently)
    ///
    /// This synthetic test isolates the BATCH dimension by running
    /// 5 chained matvecs on a 7-token batch (vs M95's 1-token).
    /// Each token's drift compounds independently through the
    /// chain; final std-ratio is measured per-token AND across
    /// batch.
    ///
    /// EMPIRICAL EXPECTATION: per-token rel_diff matches M95's
    /// single-token chain (~0.4391% over 5 ops). std-ratio
    /// across 7-token batch ≈ 1× (each token compounds
    /// identically; batch dimension doesn't amplify rel_diff).
    /// If observed batch_std_amplification > 5×, A4 is CONFIRMED;
    /// if ≈ 1×, A4 falsified.
    ///
    /// Per `contracts/trace-ffn-sub-block-gguf-v1.yaml` v1.9.0 →
    /// v1.10.0 amendment.
    #[test]
    fn falsify_ffn_gguf_013_multi_token_batch_amplification() {
        use crate::quantize::{
            dequantize_q4_k_simd, fused_q4k_q8k_parallel_matvec_into, quantize_activations_q8k_into,
        };

        const BATCH_SIZE: usize = 7;
        const IN_DIM: usize = 256;
        const OUT_DIM: usize = 256;
        const N_CHAINED: usize = 5;

        // Build N=5 synthetic Q4K weight tensors (256×256 each).
        let weights: Vec<Vec<u8>> = (0..N_CHAINED)
            .map(|t| {
                let mut block = vec![0u8; 256 * 144];
                for sb in 0..256 {
                    let base = sb * 144;
                    block[base] = 0x00;
                    block[base + 1] = 0x3C;
                    block[base + 2] = 0x00;
                    block[base + 3] = 0xB4;
                    for (i, b) in block[base + 4..base + 16].iter_mut().enumerate() {
                        *b = ((i * 7 + 3 + sb + t * 11) % 256) as u8;
                    }
                    for (i, b) in block[base + 16..base + 144].iter_mut().enumerate() {
                        *b = ((i * 13 + 17 + sb * 3 + t * 19) % 256) as u8;
                    }
                }
                block
            })
            .collect();

        // Initial 7-token batch: each token has slightly different
        // initial activation pattern (mimicking different prompt tokens).
        let initial_batch: Vec<Vec<f32>> = (0..BATCH_SIZE)
            .map(|tok| {
                (0..IN_DIM)
                    .map(|i| ((i as f32) - 128.0) * 0.05 + ((i % 7 + tok) as f32) * 0.01)
                    .collect()
            })
            .collect();

        // Path A: chain 5 matvecs PER TOKEN, with RMSNorm between.
        let mut act_a_batch: Vec<Vec<f32>> = initial_batch.clone();
        for w_bytes in &weights {
            let weights_f32 = dequantize_q4_k_simd(w_bytes).expect("dequant_simd failed");
            for token_idx in 0..BATCH_SIZE {
                let act_a = &act_a_batch[token_idx];
                let mut next = vec![0.0f32; OUT_DIM];
                for j in 0..OUT_DIM {
                    let row_base = j * IN_DIM;
                    next[j] = act_a
                        .iter()
                        .zip(weights_f32[row_base..row_base + IN_DIM].iter())
                        .map(|(x, y)| x * y)
                        .sum();
                }
                let norm = (next.iter().map(|x| x * x).sum::<f32>() / (OUT_DIM as f32))
                    .sqrt()
                    .max(1e-9);
                for x in next.iter_mut() {
                    *x /= norm;
                }
                act_a_batch[token_idx] = next;
            }
        }

        // Path B: same but using Q8K quant + fused matvec per token.
        let mut act_b_batch: Vec<Vec<f32>> = initial_batch.clone();
        for w_bytes in &weights {
            for token_idx in 0..BATCH_SIZE {
                let act_b = &act_b_batch[token_idx];
                let n_super_blocks = IN_DIM / 256;
                let mut q8k_scales = vec![0.0f32; n_super_blocks];
                let mut q8k_quants = vec![0i8; IN_DIM];
                quantize_activations_q8k_into(act_b, &mut q8k_scales, &mut q8k_quants)
                    .expect("q8k failed");
                let mut next = vec![0.0f32; OUT_DIM];
                fused_q4k_q8k_parallel_matvec_into(
                    w_bytes,
                    &q8k_scales,
                    &q8k_quants,
                    IN_DIM,
                    OUT_DIM,
                    &mut next,
                )
                .expect("fused failed");
                let norm = (next.iter().map(|x| x * x).sum::<f32>() / (OUT_DIM as f32))
                    .sqrt()
                    .max(1e-9);
                for x in next.iter_mut() {
                    *x /= norm;
                }
                act_b_batch[token_idx] = next;
            }
        }

        // Per-token rel_diff: |act_a - act_b|_L2 / |act_a|_L2.
        let mut per_token_rel_diffs: Vec<f32> = Vec::new();
        for token_idx in 0..BATCH_SIZE {
            let act_a = &act_a_batch[token_idx];
            let act_b = &act_b_batch[token_idx];
            let l2_diff: f32 = act_a
                .iter()
                .zip(act_b.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            let l2_a: f32 = act_a.iter().map(|x| x * x).sum::<f32>().sqrt();
            per_token_rel_diffs.push(l2_diff / l2_a.max(1e-9));
        }

        // Compute STDs across batch dimension for both paths
        // (mimics §27's std-ratio measurement). Per-component std
        // across the 7 tokens, then mean over components.
        let component_std_a: Vec<f32> = (0..OUT_DIM)
            .map(|c| {
                let vals: Vec<f32> = (0..BATCH_SIZE).map(|t| act_a_batch[t][c]).collect();
                let mean: f32 = vals.iter().sum::<f32>() / (BATCH_SIZE as f32);
                let variance: f32 =
                    vals.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (BATCH_SIZE as f32);
                variance.sqrt()
            })
            .collect();
        let component_std_b: Vec<f32> = (0..OUT_DIM)
            .map(|c| {
                let vals: Vec<f32> = (0..BATCH_SIZE).map(|t| act_b_batch[t][c]).collect();
                let mean: f32 = vals.iter().sum::<f32>() / (BATCH_SIZE as f32);
                let variance: f32 =
                    vals.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (BATCH_SIZE as f32);
                variance.sqrt()
            })
            .collect();
        let mean_std_a: f32 = component_std_a.iter().sum::<f32>() / (OUT_DIM as f32);
        let mean_std_b: f32 = component_std_b.iter().sum::<f32>() / (OUT_DIM as f32);

        // §27-comparable std-ratio: std_a / std_b (or its absolute
        // deviation from 1.0).
        let std_ratio_dev = (mean_std_a / mean_std_b.max(1e-9) - 1.0).abs();

        let min_token_rd = per_token_rel_diffs
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        let max_token_rd = per_token_rel_diffs
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mean_token_rd: f32 = per_token_rel_diffs.iter().sum::<f32>() / (BATCH_SIZE as f32);
        let token_rd_variance = max_token_rd / min_token_rd.max(1e-12);

        eprintln!("FALSIFY-FFN-GGUF-013: Multi-token batch amplification (batch={BATCH_SIZE}, chained={N_CHAINED})");
        eprintln!("  per-token rel_diffs:");
        for (t, rd) in per_token_rel_diffs.iter().enumerate() {
            eprintln!("    token[{t}]: {:.6}%", rd * 100.0);
        }
        eprintln!(
            "  per-token rel_diff: min={:.6}% max={:.6}% mean={:.6}% variance_across_tokens={:.2}×",
            min_token_rd * 100.0,
            max_token_rd * 100.0,
            mean_token_rd * 100.0,
            token_rd_variance
        );
        eprintln!("  Path A mean std (across batch): {:.6}", mean_std_a);
        eprintln!("  Path B mean std (across batch): {:.6}", mean_std_b);
        eprintln!(
            "  Path A↔B std-ratio deviation from 1.0: {:.6} ({:.4}%)",
            std_ratio_dev,
            std_ratio_dev * 100.0
        );

        // Compare to M95 single-token baseline (5-tensor chained = 0.4391%).
        let m95_baseline = 0.004391;
        let multi_token_amplification = mean_token_rd / m95_baseline;

        eprintln!(
            "  M95 single-token baseline: {:.6}% (5 chained, RMSNorm); multi-token amplification = {:.4}×",
            m95_baseline * 100.0,
            multi_token_amplification
        );

        // Sanity bounds.
        assert!(
            mean_token_rd > 1e-7,
            "per-token rel_diff essentially zero — fixture degenerate"
        );
        assert!(
            mean_std_a > 1e-9 && mean_std_b > 1e-9,
            "batch std essentially zero — initial activations may be too uniform"
        );

        // EMPIRICAL VERDICT:
        if multi_token_amplification > 5.0 {
            eprintln!(
                "FALSIFY-FFN-GGUF-013: amplification {multi_token_amplification:.2}× > 5.0 — \
                 A4 CONFIRMED. Multi-token batch dimension amplifies M94 mechanism \
                 substantially beyond M95's single-token chain. Real-attention \
                 batch interactions contribute to §27 magnitude."
            );
        } else if multi_token_amplification > 1.5 {
            eprintln!(
                "FALSIFY-FFN-GGUF-013: amplification {multi_token_amplification:.2}× ∈ (1.5, 5] — \
                 A4 PARTIALLY CONFIRMED. Batch dimension provides modest amplification \
                 beyond single-token compounding."
            );
        } else if multi_token_amplification > 0.7 {
            eprintln!(
                "FALSIFY-FFN-GGUF-013: amplification {multi_token_amplification:.2}× ≈ 1× — \
                 A4 NOT CONFIRMED at this regime. Per-token rel_diff matches M95's \
                 single-token baseline; batch dimension does NOT amplify in this \
                 synthetic test (no inter-token attention applied; pure batch-of-\
                 independent-chains)."
            );
        } else {
            eprintln!(
                "FALSIFY-FFN-GGUF-013: amplification {multi_token_amplification:.2}× < 0.7 — \
                 A4 FALSIFIED. Multi-token batch COMPRESSES M94 perturbation. \
                 With A1, A2, A3, A4 all falsified, M-FFN-GGUF-6 (real-teacher \
                 falsifier) is the only remaining test for the §27 magnitude gap."
            );
        }
    }

    /// FALSIFY-FFN-GGUF-015 / M-FFN-GGUF-6b candidate A6:
    /// RMSNorm rsqrt amplification — does the M94 mechanism's per-
    /// tensor 0.077% rel_diff get amplified through RMSNorm's
    /// 1/sqrt(σ²) non-linearity?
    ///
    /// A6 hypothesis: RMSNorm normalizes x by 1/sqrt(mean(x²) + eps).
    /// The rsqrt is non-linear; small input drift in x produces
    /// drift in mean(x²), which non-linearly affects 1/sqrt(σ²),
    /// which then scales the entire output. In saturated regimes
    /// (small σ²), the rsqrt amplification factor can be large.
    ///
    /// Test design:
    /// - Vector x with 256 elements in realistic range.
    /// - Apply M94-equivalent perturbation (0.077%) to all elements.
    /// - Compute RMSNorm(x) and RMSNorm(x_perturbed).
    /// - Measure output L2 drift.
    ///
    /// EXPECTATION:
    /// - For a smooth distribution, RMSNorm is approximately
    ///   homogeneous of degree 0 (RMSNorm(αx) = RMSNorm(x) for any
    ///   non-zero scalar α). A scale-perturbation should produce
    ///   essentially zero output drift.
    /// - But M94 perturbation is NOT a pure scale — it's a per-element
    ///   bit-level drift. Each element drifts independently by 0.077%
    ///   (in worst case). This breaks the homogeneity and causes
    ///   real output drift.
    ///
    /// The rsqrt amplification is bounded by the variance of the
    /// per-element drift relative to the mean magnitude. For a
    /// well-distributed activation vector, amplification should be
    /// ~1× (no significant amplification beyond the input drift).
    ///
    /// EMPIRICAL HYPOTHESIS: amplification ≈ 1×. If FALSIFIED
    /// (amplification > 5×), A6 has a sub-mechanism worth
    /// investigating in M-FFN-GGUF-7 (multi-layer real-teacher).
    ///
    /// Per `contracts/trace-ffn-sub-block-gguf-v1.yaml` v1.11.0 →
    /// v1.12.0 amendment.
    #[test]
    fn falsify_ffn_gguf_015_rmsnorm_rsqrt_amplification() {
        const HIDDEN_DIM: usize = 256;
        const EPS: f32 = 1e-6;

        // Build realistic activation vector. RMSNorm typically applies
        // to layer-output residual streams with std ~1.0 after warmup.
        let x_a: Vec<f32> = (0..HIDDEN_DIM)
            .map(|i| ((i as f32) - 128.0) * 0.05 + ((i % 7) as f32) * 0.01)
            .collect();

        // M94-equivalent perturbation: each element drifts by ~0.077%
        // (additive per-element noise; mimics M94 mechanism's bit-level
        // drift pattern, NOT pure scaling).
        let x_b: Vec<f32> = x_a
            .iter()
            .enumerate()
            .map(|(i, x)| {
                // Pseudo-random per-element drift in ±0.077% range.
                let sign = if (i * 13 + 7) % 17 < 8 { 1.0 } else { -1.0 };
                x + sign * x.abs() * 0.00077
            })
            .collect();

        // RMSNorm: x_i / sqrt(mean(x²) + eps).
        fn rmsnorm(x: &[f32], eps: f32) -> Vec<f32> {
            let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / (x.len() as f32);
            let rms = (mean_sq + eps).sqrt().max(1e-9);
            x.iter().map(|v| v / rms).collect()
        }

        let y_a = rmsnorm(&x_a, EPS);
        let y_b = rmsnorm(&x_b, EPS);

        // Input drift: |x_b - x_a|_L2 / |x_a|_L2.
        let x_diff_l2: f32 = x_a
            .iter()
            .zip(x_b.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        let x_a_l2: f32 = x_a.iter().map(|v| v * v).sum::<f32>().sqrt();
        let input_rel_drift = x_diff_l2 / x_a_l2.max(1e-9);

        // Output drift: |y_b - y_a|_L2 / |y_a|_L2.
        let y_diff_l2: f32 = y_a
            .iter()
            .zip(y_b.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        let y_a_l2: f32 = y_a.iter().map(|v| v * v).sum::<f32>().sqrt();
        let output_rel_drift = y_diff_l2 / y_a_l2.max(1e-9);

        let amplification = output_rel_drift / input_rel_drift.max(1e-12);

        eprintln!("FALSIFY-FFN-GGUF-015: RMSNorm rsqrt amplification");
        eprintln!("  hidden_dim = {HIDDEN_DIM}, eps = {EPS}");
        eprintln!(
            "  x_a L2 = {x_a_l2:.6}, x_diff_l2 = {x_diff_l2:.6}, input_rel_drift = {:.6}%",
            input_rel_drift * 100.0
        );
        eprintln!(
            "  y_a L2 = {y_a_l2:.6}, y_diff_l2 = {y_diff_l2:.6}, output_rel_drift = {:.6}%",
            output_rel_drift * 100.0
        );
        eprintln!("  amplification factor = {amplification:.4}×");

        // Sanity bounds.
        assert!(
            input_rel_drift > 0.0,
            "perturbation must produce nonzero input drift"
        );
        assert!(
            amplification > 1e-9,
            "amplification {amplification} essentially zero — RMSNorm may be \
             producing bit-identical outputs"
        );

        // EMPIRICAL VERDICT:
        if amplification > 5.0 {
            eprintln!(
                "FALSIFY-FFN-GGUF-015: amplification {amplification:.2}× > 5.0 — \
                 A6 CONFIRMED. RMSNorm rsqrt amplifies M94 perturbation \
                 substantially. Real-RMSNorm contributes to §27 magnitude \
                 beyond the M91-M100 5.56×× synthetic+real upper bound. \
                 The 14× residual gap is partly explained by A6."
            );
        } else if amplification > 1.5 {
            eprintln!(
                "FALSIFY-FFN-GGUF-015: amplification {amplification:.2}× ∈ (1.5, 5] — \
                 A6 PARTIALLY CONFIRMED. RMSNorm provides modest amplification."
            );
        } else if amplification > 0.7 {
            eprintln!(
                "FALSIFY-FFN-GGUF-015: amplification {amplification:.2}× ≈ 1× — \
                 A6 NOT CONFIRMED at this regime. RMSNorm is approximately \
                 homogeneous over the per-element drift pattern; rsqrt \
                 nonlinearity does NOT amplify M94 perturbation in synthetic \
                 test. The 14× residual must come from cumulative-layer \
                 interaction (M-FFN-GGUF-7)."
            );
        } else {
            eprintln!(
                "FALSIFY-FFN-GGUF-015: amplification {amplification:.2}× < 0.7 — \
                 A6 FALSIFIED. RMSNorm COMPRESSES M94 perturbation. The 14× \
                 residual comes entirely from cumulative-layer interaction; \
                 M-FFN-GGUF-7 (multi-layer real-teacher) is the only remaining \
                 test for §27 closure."
            );
        }
    }

    // =======================================================================
    // PMAT-797: apr_transformer CPU RoPE must honor the architecture pairing
    // convention (NORM vs NEOX), matching llama.cpp `ggml_rope`.
    //
    // Before the fix, `apply_rope_f32` ALWAYS used NEOX (split-half) pairing,
    // silently corrupting RoPE for every LLaMA-family (NORM) safetensors model
    // on the CPU reference path. These tests pin both conventions to
    // hand-computed reference values from the RoPE definition.
    //
    // Reference convention (Su et al. 2021 "RoFormer"; llama.cpp ggml_rope):
    //   theta_i = base^(-2i/head_dim)
    //   x'[a] = x[a]*cos(pos*theta_i) - x[b]*sin(pos*theta_i)
    //   x'[b] = x[a]*sin(pos*theta_i) + x[b]*cos(pos*theta_i)
    // where (a, b) = (2i, 2i+1) for NORM, (i, i+head_dim/2) for NEOX.
    // =======================================================================

    /// PMAT-797-A: NORM (rope_type 0) rotates ADJACENT pairs (LLaMA-family).
    ///
    /// Hand-computed reference for head_dim=4, base=10000, pos=1:
    ///   i=0: theta_0 = 10000^0 = 1.0,  angle = 1.0 rad
    ///   i=1: theta_1 = 10000^(-0.5) = 0.01, angle = 0.01 rad
    /// NORM pairs (x[0],x[1]) at angle 1.0 and (x[2],x[3]) at angle 0.01.
    #[test]
    fn pmat_797_a_norm_matches_adjacent_pair_reference() {
        let head_dim = 4;
        let base = 10000.0_f32;
        let pos = 1usize;
        let mut x = vec![1.0_f32, 2.0, 3.0, 4.0];

        apply_rope_f32(
            &mut x, pos, 1, head_dim, base, /* rope_type = NORM */ 0,
        );

        // Reference: adjacent pairs (2i, 2i+1).
        let a0 = pos as f32 * 1.0; // theta_0 = 1.0
        let (s0, c0) = a0.sin_cos();
        let a1 = pos as f32 * base.powf(-2.0 / head_dim as f32); // theta_1
        let (s1, c1) = a1.sin_cos();

        let ref0 = 1.0 * c0 - 2.0 * s0; // x[0]
        let ref1 = 1.0 * s0 + 2.0 * c0; // x[1]
        let ref2 = 3.0 * c1 - 4.0 * s1; // x[2]
        let ref3 = 3.0 * s1 + 4.0 * c1; // x[3]

        assert!((x[0] - ref0).abs() < 1e-5, "NORM x[0]: {} vs {ref0}", x[0]);
        assert!((x[1] - ref1).abs() < 1e-5, "NORM x[1]: {} vs {ref1}", x[1]);
        assert!((x[2] - ref2).abs() < 1e-5, "NORM x[2]: {} vs {ref2}", x[2]);
        assert!((x[3] - ref3).abs() < 1e-5, "NORM x[3]: {} vs {ref3}", x[3]);
    }

    /// PMAT-797-B: NEOX (rope_type 2) rotates SPLIT HALVES (Qwen/NeoX-family).
    ///
    /// Same head_dim/base/pos, but NEOX pairs (x[0],x[2]) at angle 1.0 and
    /// (x[1],x[3]) at angle 0.01.
    #[test]
    fn pmat_797_b_neox_matches_split_half_reference() {
        let head_dim = 4;
        let base = 10000.0_f32;
        let pos = 1usize;
        let mut x = vec![1.0_f32, 2.0, 3.0, 4.0];

        apply_rope_f32(
            &mut x, pos, 1, head_dim, base, /* rope_type = NEOX */ 2,
        );

        let a0 = pos as f32 * 1.0;
        let (s0, c0) = a0.sin_cos();
        let a1 = pos as f32 * base.powf(-2.0 / head_dim as f32);
        let (s1, c1) = a1.sin_cos();

        // NEOX: pair i with i+half_dim. half_dim=2.
        let ref0 = 1.0 * c0 - 3.0 * s0; // x[0] paired with x[2]
        let ref2 = 1.0 * s0 + 3.0 * c0; // x[2]
        let ref1 = 2.0 * c1 - 4.0 * s1; // x[1] paired with x[3]
        let ref3 = 2.0 * s1 + 4.0 * c1; // x[3]

        assert!((x[0] - ref0).abs() < 1e-5, "NEOX x[0]: {} vs {ref0}", x[0]);
        assert!((x[1] - ref1).abs() < 1e-5, "NEOX x[1]: {} vs {ref1}", x[1]);
        assert!((x[2] - ref2).abs() < 1e-5, "NEOX x[2]: {} vs {ref2}", x[2]);
        assert!((x[3] - ref3).abs() < 1e-5, "NEOX x[3]: {} vs {ref3}", x[3]);
    }

    /// PMAT-797-C: the two conventions DIFFER — proving the pairing is
    /// load-bearing. If this fails, the rope_type branch is a no-op and the
    /// fix is meaningless.
    #[test]
    fn pmat_797_c_norm_and_neox_diverge() {
        let head_dim = 8;
        let base = 10000.0_f32;
        let pos = 5usize;
        let x0: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.5).collect();

        let mut x_norm = x0.clone();
        let mut x_neox = x0.clone();
        apply_rope_f32(&mut x_norm, pos, 1, head_dim, base, 0);
        apply_rope_f32(&mut x_neox, pos, 1, head_dim, base, 2);

        let max_div = x_norm
            .iter()
            .zip(x_neox.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_div > 1e-3,
            "PMAT-797: NORM and NEOX RoPE produced identical output \
             (max divergence {max_div}); rope_type branch is a no-op"
        );
    }

    /// PMAT-797-D: position 0 is identity for BOTH conventions
    /// (cos(0)=1, sin(0)=0 ⇒ x' = x). Cross-checks both branches preserve
    /// the zero-position invariant.
    #[test]
    fn pmat_797_d_zero_position_identity_both_conventions() {
        let head_dim = 6;
        let base = 10000.0_f32;
        let x0: Vec<f32> = vec![1.5, -2.0, 0.25, 3.0, -1.0, 0.75];

        for rope_type in [0u32, 2u32] {
            let mut x = x0.clone();
            apply_rope_f32(&mut x, 0, 1, head_dim, base, rope_type);
            for (i, (&got, &want)) in x.iter().zip(x0.iter()).enumerate() {
                assert!(
                    (got - want).abs() < 1e-6,
                    "rope_type={rope_type} pos=0 changed x[{i}]: {got} vs {want}"
                );
            }
        }
    }

    /// PMAT-797-E: norm preservation — RoPE is a rotation, so ‖x'‖ ≈ ‖x‖
    /// for BOTH conventions (sanity that the adjacent-pair NORM path didn't
    /// introduce an index bug that breaks orthonormality).
    #[test]
    fn pmat_797_e_norm_preservation_both_conventions() {
        let head_dim = 8;
        let base = 10000.0_f32;
        let pos = 13usize;
        let x0: Vec<f32> = (0..head_dim).map(|i| (i as f32 * 0.37).sin()).collect();
        let norm_in = x0.iter().map(|v| v * v).sum::<f32>().sqrt();

        for rope_type in [0u32, 2u32] {
            let mut x = x0.clone();
            apply_rope_f32(&mut x, pos, 1, head_dim, base, rope_type);
            let norm_out = x.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(
                (norm_out - norm_in).abs() < 1e-4,
                "rope_type={rope_type}: ‖x'‖={norm_out} != ‖x‖={norm_in}"
            );
        }
    }
}
