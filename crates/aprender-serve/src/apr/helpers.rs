//! APR inference helper functions (PMAT-802)
//!
//! Extracted from apr/mod.rs - Pure Rust inference primitives.
//!
//! ## Contents
//! - RMS normalization
//! - Matrix multiplication with SIMD
//! - Transpose operations
//! - Simple attention computation
//! - Format detection utilities

use super::MAGIC;
use std::fs;
use std::path::Path;

/// RMS normalization
///
/// When the `gpu` feature is enabled, delegates to `gpu::layer_norm_static`
/// (canonical RMSNorm). Otherwise, uses an inline implementation.
///
/// Five-Whys (whisper.apr compile failure): `crate::gpu` is `#[cfg(feature = "gpu")]`
/// but this function was called unconditionally. Crates without `gpu` feature
/// (e.g. whisper-apr) got E0433 unresolved import.
pub(crate) fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    #[cfg(feature = "gpu")]
    {
        let hidden_dim = weight.len();
        let zero_bias = vec![0.0f32; hidden_dim];
        crate::gpu::layer_norm_static(x, weight, &zero_bias, hidden_dim, eps)
    }
    #[cfg(not(feature = "gpu"))]
    {
        let hidden_dim = weight.len();
        let n_tokens = x.len() / hidden_dim;
        let mut output = vec![0.0f32; x.len()];
        for t in 0..n_tokens {
            let offset = t * hidden_dim;
            crate::gguf::ops::rms_norm_scalar_into(
                &x[offset..offset + hidden_dim],
                weight,
                eps,
                crate::gguf::ops::RmsScale::Divide,
                &mut output[offset..offset + hidden_dim],
            );
        }
        output
    }
}

/// Matrix multiplication: x @ w^T
/// [seq, in_dim] @ [out_dim, in_dim]^T -> [seq, out_dim]
///
/// When the `gpu` feature is enabled, delegates to `gpu::cpu_matmul_transpose_b`.
/// Otherwise, uses an inline triple-loop implementation.
pub(crate) fn matmul(
    x: &[f32],
    w: &[f32],
    seq_len: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    #[cfg(feature = "gpu")]
    {
        crate::gpu::cpu_matmul_transpose_b(x, w, seq_len, in_dim, out_dim)
    }
    #[cfg(not(feature = "gpu"))]
    {
        let mut output = vec![0.0f32; seq_len * out_dim];
        for m in 0..seq_len {
            for n in 0..out_dim {
                let mut sum = 0.0f32;
                for k in 0..in_dim {
                    sum += x[m * in_dim + k] * w[n * in_dim + k];
                }
                output[m * out_dim + n] = sum;
            }
        }
        output
    }
}

/// Transpose a matrix from [rows, cols] to [cols, rows] for GEMM compatibility.
///
/// PMAT-285: Delegates to `contract_gate::transpose_f32` (single source of truth).
#[cfg(feature = "cuda")]
pub(crate) fn transpose_matrix(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    crate::contract_gate::transpose_f32(m, rows, cols)
}

/// SIMD-accelerated dot product
#[inline]
pub fn simd_dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 feature is runtime-checked above, simd_dot_avx2 requires AVX2
            return unsafe { simd_dot_avx2(a, b) };
        }
    }
    // Scalar fallback
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
// SAFETY: Caller must satisfy the documented preconditions
unsafe fn simd_dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::{
        _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_loadu_ps,
        _mm256_setzero_ps, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_movehl_ps, _mm_shuffle_ps,
    };

    let n = a.len().min(b.len());
    let chunks = n / 8;

    // SAFETY: This entire fn is unsafe with target_feature(avx2, fma)
    // All intrinsics are safe to call given the target_feature guarantee
    // The unsafe block is required for Rust 2024 edition compliance
    unsafe {
        let mut sum = _mm256_setzero_ps();

        for i in 0..chunks {
            let av = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let bv = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            sum = _mm256_fmadd_ps(av, bv, sum);
        }

        // Horizontal sum
        let hi = _mm256_extractf128_ps(sum, 1);
        let lo = _mm256_castps256_ps128(sum);
        let sum128 = _mm_add_ps(lo, hi);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        let mut result = _mm_cvtss_f32(sum32);

        // Handle remainder (scalar)
        for i in (chunks * 8)..n {
            result += a.get(i).copied().unwrap_or(0.0) * b.get(i).copied().unwrap_or(0.0);
        }

        result
    }
}

/// Simplified multi-head attention (no RoPE, causal mask)
///
/// NOTE: This is the **multi-sequence** variant — Q/K/V are `[seq_len, dim]`
/// and causal masking iterates over all sequence positions.  In contrast,
/// `gpu::scheduler::ops::gqa_multihead_attention` is the **single-position**
/// (incremental decode) variant where Q is `[num_heads * head_dim]` and K/V
/// come from a KV cache of length `kv_len`.  The two have fundamentally
/// different loop structures and cannot be unified without a mode flag that
/// would hurt clarity, so this implementation is intentionally kept separate.
///
/// Each (position, head) row is one call to the shared one-row home
/// `gguf::ops::attend_row_scalar` over keys `0..=s` (PP-ARCH-001 §9.14).
pub(crate) fn simple_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    contract_pre_attention!();
    let hidden_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let heads_per_kv = num_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // The per-element reads this replaced treated an index past the end of a
    // short buffer as 0.0; zero-padding up front keeps that exactly, and costs
    // nothing when the buffers are full-length (the only case forward uses).
    fn padded(x: &[f32], len: usize) -> std::borrow::Cow<'_, [f32]> {
        if x.len() >= len {
            std::borrow::Cow::Borrowed(x)
        } else {
            let mut p = x.to_vec();
            p.resize(len, 0.0);
            std::borrow::Cow::Owned(p)
        }
    }
    let q = padded(q, seq_len * hidden_dim);
    let k = padded(k, seq_len * kv_dim);
    let v = padded(v, seq_len * kv_dim);

    let mut output = vec![0.0; seq_len * hidden_dim];
    let mut scores = Vec::with_capacity(seq_len);

    for s in 0..seq_len {
        for h in 0..num_heads {
            let kv_off = (h / heads_per_kv) * head_dim;
            let q_base = s * hidden_dim + h * head_dim;
            // Causal: query s attends to keys 0..=s.
            crate::gguf::ops::attend_row_scalar(
                &q[q_base..q_base + head_dim],
                s + 1,
                |t| &k[t * kv_dim + kv_off..t * kv_dim + kv_off + head_dim],
                |t| &v[t * kv_dim + kv_off..t * kv_dim + kv_off + head_dim],
                crate::gguf::ops::ScoreScale::Mul(scale),
                crate::gguf::ops::RowSoftmax {
                    norm: crate::gguf::ops::SoftmaxNorm::Divide,
                    guard_positive_sum: false,
                },
                &mut scores,
                &mut output[q_base..q_base + head_dim],
            );
        }
    }

    contract_post_attention!(&output);
    output
}

/// BUG-2 FIX: Apply RoPE rotation with rope_type support
///
/// PMAT-110 / CORRECTNESS-011: Qwen2.5 models require rope_type=2 (NEOX style)
///
/// # Arguments
/// * `x` - Tensor to apply RoPE to (modified in place)
/// * `num_heads` - Number of attention heads
/// * `head_dim` - Dimension per head
/// * `position` - Position in sequence
/// * `theta` - RoPE theta (frequency base)
/// * `rope_type` - 0=NORM (adjacent pairs), 2=NEOX (split halves)
pub(crate) fn apply_rope_norm(
    x: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    position: usize,
    theta: f32,
    rope_type: u32,
) {
    crate::gguf::ops::rope_into(
        x,
        num_heads,
        head_dim,
        position,
        theta,
        crate::gguf::ops::RopeStyle::from_rope_type(rope_type),
    );
}

/// Check if a file is a valid .apr v2 file
pub fn is_apr_file<P: AsRef<Path>>(path: P) -> bool {
    fs::read(path.as_ref()).is_ok_and(|data| data.len() >= 4 && data[0..4] == MAGIC)
}

/// Detect model format from file extension
fn format_from_extension(path: &Path) -> Option<&'static str> {
    let ext = path.extension()?.to_string_lossy().to_lowercase();
    match ext.as_str() {
        "apr" => Some("apr"),
        "gguf" => Some("gguf"),
        "safetensors" => Some("safetensors"),
        _ => None,
    }
}

/// Detect model format from file magic bytes
fn format_from_magic(path: &Path) -> &'static str {
    let Ok(data) = fs::read(path) else {
        return "unknown";
    };
    if data.len() < 4 {
        return "unknown";
    }
    if data[0..4] == MAGIC {
        return "apr";
    }
    if data[0..4] == [0x47, 0x47, 0x55, 0x46] {
        return "gguf";
    }
    if data[0] == b'{' {
        return "safetensors";
    }
    "unknown"
}

/// Detect model format from extension or magic bytes
pub fn detect_format<P: AsRef<Path>>(path: P) -> &'static str {
    let path = path.as_ref();
    format_from_extension(path).unwrap_or_else(|| format_from_magic(path))
}

include!("helpers_tests.rs");
