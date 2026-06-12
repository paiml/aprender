//! SIMD-Optimized Attention Operation (PMAT-017)
//!
//! This module contains the scaled dot-product attention operation
//! with SIMD optimization for CPU inference.
//!
//! # Algorithm
//!
//! Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
//!
//! # SIMD Optimizations
//!
//! - Q @ K^T: Batched dot products with AVX2/AVX-512/FMA
//! - Softmax: Row-wise numerically stable implementation
//! - Scores @ V: SIMD-friendly weighted accumulation
//!
//! # Performance Target
//!
//! Close the 1.66x gap in CPU inference (25.4 → 42 tok/s) by replacing
//! scalar triple-nested loops with SIMD operations.

use super::{Backend, ComputeOp};
use crate::error::TruenoError;

/// Scaled dot-product attention operation.
///
/// Computes: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
///
/// # SIMD Optimization (PMAT-017)
///
/// Uses trueno's SIMD backends for:
/// - Q @ K^T: Batched dot products with AVX2/AVX-512
/// - Softmax: Row-wise numerically stable softmax
/// - Scores @ V: Batched weighted sums
///
/// # Performance Target
///
/// Close the 1.66x gap in CPU inference (25.4 → 42 tok/s) by replacing
/// scalar triple-nested loops with SIMD operations.
#[derive(Debug, Clone)]
pub struct AttentionOp {
    /// Sequence length (Q rows)
    pub seq_len: usize,
    /// Key/Value sequence length (may differ for cross-attention)
    pub kv_seq_len: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Scale factor (1/sqrt(head_dim))
    pub scale: f32,
}

impl AttentionOp {
    /// Create a new attention operation.
    ///
    /// # Arguments
    ///
    /// * `seq_len` - Query sequence length
    /// * `kv_seq_len` - Key/Value sequence length
    /// * `head_dim` - Dimension per head
    #[must_use]
    pub fn new(seq_len: usize, kv_seq_len: usize, head_dim: usize) -> Self {
        Self { seq_len, kv_seq_len, head_dim, scale: 1.0 / (head_dim as f32).sqrt() }
    }

    /// Create for self-attention (seq_len == kv_seq_len).
    #[must_use]
    pub fn self_attention(seq_len: usize, head_dim: usize) -> Self {
        Self::new(seq_len, seq_len, head_dim)
    }

    /// SIMD-optimized dot product for attention scores.
    ///
    /// Computes Q[i] · K[j] using SIMD when available.
    #[inline]
    pub(crate) fn simd_dot(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        // Use architecture-specific SIMD
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: preconditions verified by caller
                return unsafe { Self::avx2_dot(a, b) };
            }
        }

        // Scalar fallback with manual unrolling for better vectorization
        let mut sum0 = 0.0f32;
        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        let mut sum3 = 0.0f32;

        let chunks = a.len() / 4;
        for i in 0..chunks {
            let base = i * 4;
            sum0 += a[base] * b[base];
            sum1 += a[base + 1] * b[base + 1];
            sum2 += a[base + 2] * b[base + 2];
            sum3 += a[base + 3] * b[base + 3];
        }

        // Handle remainder
        for i in (chunks * 4)..a.len() {
            sum0 += a[i] * b[i];
        }

        sum0 + sum1 + sum2 + sum3
    }

    /// AVX2-optimized dot product.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    // SAFETY: caller verifies AVX2 support, input slices meet alignment/length requirements
    unsafe fn avx2_dot(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            use std::arch::x86_64::*;

            let mut sum = _mm256_setzero_ps();
            let chunks = a.len() / 8;

            for i in 0..chunks {
                let base = i * 8;
                let va = _mm256_loadu_ps(a.as_ptr().add(base));
                let vb = _mm256_loadu_ps(b.as_ptr().add(base));
                sum = _mm256_fmadd_ps(va, vb, sum);
            }

            // Horizontal sum
            let high = _mm256_extractf128_ps(sum, 1);
            let low = _mm256_castps256_ps128(sum);
            let sum128 = _mm_add_ps(high, low);
            let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
            let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
            let mut result = _mm_cvtss_f32(sum32);

            // Handle remainder
            for i in (chunks * 8)..a.len() {
                result += a[i] * b[i];
            }

            result
        }
    }

    /// SIMD axpy: out[i] += alpha * x[i] for all i.
    /// Used in attention weighted sum: out += weight * v_row.
    #[inline]
    pub(crate) fn simd_axpy(alpha: f32, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), out.len());

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // SAFETY: AVX2+FMA verified, slices same length
                unsafe {
                    Self::avx2_axpy(alpha, x, out);
                }
                return;
            }
        }

        // Scalar fallback
        for (o, &xi) in out.iter_mut().zip(x.iter()) {
            *o += alpha * xi;
        }
    }

    /// AVX2-optimized axpy: out[i] += alpha * x[i].
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn avx2_axpy(alpha: f32, x: &[f32], out: &mut [f32]) {
        unsafe {
            use std::arch::x86_64::*;

            let alpha_v = _mm256_set1_ps(alpha);
            let n = x.len();
            let n8 = n / 8 * 8;

            let mut i = 0;
            while i < n8 {
                let xv = _mm256_loadu_ps(x.as_ptr().add(i));
                let ov = _mm256_loadu_ps(out.as_ptr().add(i));
                let r = _mm256_fmadd_ps(alpha_v, xv, ov);
                _mm256_storeu_ps(out.as_mut_ptr().add(i), r);
                i += 8;
            }
            // Scalar remainder
            while i < n {
                *out.get_unchecked_mut(i) += alpha * *x.get_unchecked(i);
                i += 1;
            }
        }
    }

    /// Row-wise softmax with AVX2 SIMD exp + horizontal ops.
    /// CGP-DBUF: replaced scalar exp() loop with AVX2 polynomial fast_exp
    /// from blis::softmax (6th-order Remez minimax, <1 ULP error).
    /// For seq_len=512: 64 AVX2 iterations vs 512 scalar exp() calls.
    #[inline]
    pub(crate) fn simd_softmax_row(scores: &mut [f32]) {
        if scores.is_empty() {
            return;
        }

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA verified. scores is valid mutable slice.
            unsafe {
                Self::avx2_softmax_row(scores);
            }
            return;
        }

        // Scalar fallback
        Self::scalar_softmax_row(scores);
    }

    /// Scalar softmax fallback.
    fn scalar_softmax_row(scores: &mut [f32]) {
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max).exp();
            sum += *s;
        }
        let inv_sum = 1.0 / sum.max(f32::EPSILON);
        for s in scores.iter_mut() {
            *s *= inv_sum;
        }
    }

    /// AVX2 softmax: fused exp+sum in one pass, then SIMD normalize.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn avx2_softmax_row(scores: &mut [f32]) {
        unsafe {
            use std::arch::x86_64::*;

            let n = scores.len();
            let n8 = n / 8 * 8;

            // Pass 1: find max (AVX2 horizontal max)
            let mut max_v = _mm256_set1_ps(f32::NEG_INFINITY);
            let mut i = 0;
            while i < n8 {
                let v = _mm256_loadu_ps(scores.as_ptr().add(i));
                max_v = _mm256_max_ps(max_v, v);
                i += 8;
            }
            // Horizontal reduce max_v
            let hi = _mm256_permute2f128_ps(max_v, max_v, 1);
            max_v = _mm256_max_ps(max_v, hi);
            let shuf = _mm256_shuffle_ps(max_v, max_v, 0b01_00_11_10);
            max_v = _mm256_max_ps(max_v, shuf);
            let shuf2 = _mm256_shuffle_ps(max_v, max_v, 0b10_11_00_01);
            max_v = _mm256_max_ps(max_v, shuf2);
            let mut max_val = _mm_cvtss_f32(_mm256_castps256_ps128(max_v));
            for j in n8..n {
                max_val = max_val.max(scores[j]);
            }

            // Pass 2: fused exp(x-max) + sum (using fast_exp_avx2)
            let max_broadcast = _mm256_set1_ps(max_val);
            let mut sum_v = _mm256_setzero_ps();
            i = 0;
            while i < n8 {
                let x = _mm256_sub_ps(_mm256_loadu_ps(scores.as_ptr().add(i)), max_broadcast);
                let e = crate::blis::softmax::fast_exp_avx2(x);
                _mm256_storeu_ps(scores.as_mut_ptr().add(i), e);
                sum_v = _mm256_add_ps(sum_v, e);
                i += 8;
            }
            // Horizontal reduce sum_v
            let hi = _mm256_permute2f128_ps(sum_v, sum_v, 1);
            sum_v = _mm256_add_ps(sum_v, hi);
            let shuf = _mm256_shuffle_ps(sum_v, sum_v, 0b01_00_11_10);
            sum_v = _mm256_add_ps(sum_v, shuf);
            let shuf2 = _mm256_shuffle_ps(sum_v, sum_v, 0b10_11_00_01);
            sum_v = _mm256_add_ps(sum_v, shuf2);
            let mut sum_val = _mm_cvtss_f32(_mm256_castps256_ps128(sum_v));
            // Scalar remainder for exp+sum
            for j in n8..n {
                let e = (scores[j] - max_val).exp();
                scores[j] = e;
                sum_val += e;
            }

            // Pass 3: normalize (SIMD multiply by 1/sum)
            let inv_sum = 1.0 / sum_val.max(f32::EPSILON);
            let inv_v = _mm256_set1_ps(inv_sum);
            i = 0;
            while i < n8 {
                let v = _mm256_loadu_ps(scores.as_ptr().add(i));
                _mm256_storeu_ps(scores.as_mut_ptr().add(i), _mm256_mul_ps(v, inv_v));
                i += 8;
            }
            for j in n8..n {
                scores[j] *= inv_sum;
            }
        } // unsafe
    }
}

impl ComputeOp for AttentionOp {
    /// Input: (Q, K, V) tensors as flat vectors
    /// Q: [seq_len * head_dim]
    /// K: [kv_seq_len * head_dim]
    /// V: [kv_seq_len * head_dim]
    type Input = (Vec<f32>, Vec<f32>, Vec<f32>);
    /// Output: attention output [seq_len * head_dim]
    type Output = Vec<f32>;

    fn name(&self) -> &'static str {
        "attention"
    }

    fn execute(&self, input: Self::Input, _backend: Backend) -> Result<Self::Output, TruenoError> {
        let (q, k, v) = input;

        // Validate dimensions
        let expected_q = self.seq_len * self.head_dim;
        let expected_kv = self.kv_seq_len * self.head_dim;

        if q.len() != expected_q {
            return Err(TruenoError::SizeMismatch { expected: expected_q, actual: q.len() });
        }
        if k.len() != expected_kv || v.len() != expected_kv {
            return Err(TruenoError::SizeMismatch { expected: expected_kv, actual: k.len() });
        }

        // Uninit: output is zeroed per-row via out_row.fill(0.0) before accumulation.
        // scores is SET via scores[ki] = simd_dot(...) before softmax reads.
        let mut output: Vec<f32> = Vec::with_capacity(expected_q);
        // SAFETY: Each qi loop iteration calls out_row.fill(0.0) before accumulating.
        unsafe {
            output.set_len(expected_q);
        }
        let mut scores: Vec<f32> = Vec::with_capacity(self.kv_seq_len);
        // SAFETY: scores[ki] = ... (SET) for all ki before simd_softmax_row reads.
        unsafe {
            scores.set_len(self.kv_seq_len);
        }

        // For each query position
        for qi in 0..self.seq_len {
            let q_row = &q[qi * self.head_dim..(qi + 1) * self.head_dim];

            // Compute Q[qi] · K[ki] for all ki (SIMD dot products)
            for ki in 0..self.kv_seq_len {
                let k_row = &k[ki * self.head_dim..(ki + 1) * self.head_dim];
                scores[ki] = Self::simd_dot(q_row, k_row) * self.scale;
            }

            // Softmax over scores
            Self::simd_softmax_row(&mut scores);

            // Compute weighted sum: output[qi] = sum(scores[ki] * V[ki])
            let out_row = &mut output[qi * self.head_dim..(qi + 1) * self.head_dim];
            out_row.fill(0.0);

            for ki in 0..self.kv_seq_len {
                let v_row = &v[ki * self.head_dim..(ki + 1) * self.head_dim];
                let weight = scores[ki];

                // CGP-DBUF: AVX2 broadcast-multiply-add (was scalar loop)
                Self::simd_axpy(weight, v_row, out_row);
            }
        }

        Ok(output)
    }

    fn tokens(&self, _input: &Self::Input) -> usize {
        // Output tokens = seq_len * head_dim
        self.seq_len * self.head_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Assert simd_dot of two slices equals expected within tolerance.
    fn assert_dot(a: &[f32], b: &[f32], expected: f32) {
        let dot = AttentionOp::simd_dot(a, b);
        assert!((dot - expected).abs() < 1e-3, "dot={dot}, expected={expected}");
    }

    /// Assert simd_dot of [1..=n] · [1.0; n] equals n*(n+1)/2.
    fn assert_dot_iota(n: usize) {
        let a: Vec<f32> = (1..=n).map(|x| x as f32).collect();
        let b = vec![1.0f32; n];
        let expected = (n * (n + 1)) / 2;
        assert_dot(&a, &b, expected as f32);
    }

    /// Assert softmax normalizes scores to sum=1.
    fn assert_softmax_normalized(values: &[f32]) {
        let mut scores = values.to_vec();
        AttentionOp::simd_softmax_row(&mut scores);
        let sum: f32 = scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum={sum}");
    }

    /// Execute attention and assert output length and finiteness.
    fn assert_attention_ok(
        op: &AttentionOp,
        q: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
        expected_len: usize,
    ) -> Vec<f32> {
        let output = op.execute((q, k, v), Backend::Scalar).unwrap();
        assert_eq!(output.len(), expected_len);
        for val in &output {
            assert!(val.is_finite());
        }
        output
    }

    #[test]
    fn test_attention_basic() {
        let op = AttentionOp::self_attention(2, 4); // seq=2, head_dim=4

        // Simple identity-like setup
        let q = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 2x4
        let k = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 2x4
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2x4

        let output = op.execute((q, k, v), Backend::Scalar).unwrap();

        assert_eq!(output.len(), 8);
        // Output should be weighted combination of V rows
    }

    #[test]
    fn test_attention_dimension_mismatch_q() {
        let op = AttentionOp::self_attention(2, 4);
        let q = vec![1.0; 4]; // Wrong size - should be 8
        let k = vec![1.0; 8];
        let v = vec![1.0; 8];

        let result = op.execute((q, k, v), Backend::Scalar);
        assert!(result.is_err());
    }

    #[test]
    fn test_attention_dimension_mismatch_kv() {
        let op = AttentionOp::self_attention(2, 4);
        let q = vec![1.0; 8];
        let k = vec![1.0; 4]; // Wrong size - should be 8
        let v = vec![1.0; 8];

        let result = op.execute((q, k, v), Backend::Scalar);
        assert!(result.is_err());
    }

    #[test]
    fn test_attention_cross_attention() {
        // Cross-attention: Q from decoder (seq=1), K/V from encoder (seq=4)
        let op = AttentionOp::new(1, 4, 8); // q_seq=1, kv_seq=4, head_dim=8

        let q = vec![1.0; 8]; // 1 x 8
        let k = vec![1.0; 32]; // 4 x 8
        let v = vec![1.0; 32]; // 4 x 8

        let output = op.execute((q, k, v), Backend::Scalar).unwrap();
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_attention_tokens() {
        let op = AttentionOp::self_attention(16, 64);
        let input = (vec![], vec![], vec![]);
        // tokens = seq_len * head_dim = 16 * 64 = 1024
        assert_eq!(op.tokens(&input), 1024);
    }

    #[test]
    fn test_simd_softmax_row_empty() {
        let mut scores: Vec<f32> = vec![];
        AttentionOp::simd_softmax_row(&mut scores);
        assert!(scores.is_empty());
    }

    #[test]
    fn test_simd_softmax_row_single() {
        let mut scores = vec![5.0];
        AttentionOp::simd_softmax_row(&mut scores);
        assert!((scores[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_softmax_row_uniform() {
        let mut scores = vec![1.0, 1.0, 1.0, 1.0];
        AttentionOp::simd_softmax_row(&mut scores);

        // All equal inputs → uniform distribution
        for s in &scores {
            assert!((s - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_simd_softmax_row_sum_to_one() {
        assert_softmax_normalized(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_simd_dot_basic() {
        assert_dot(&[1.0, 2.0, 3.0, 4.0], &[1.0, 1.0, 1.0, 1.0], 10.0);
    }

    #[test]
    fn test_simd_dot_unaligned() {
        assert_dot(&[1.0, 2.0, 3.0, 4.0, 5.0], &[2.0; 5], 30.0);
    }

    // =========================================================================
    // Additional Coverage Tests
    // =========================================================================

    #[test]
    fn test_attention_op_fields() {
        let op = AttentionOp::new(4, 8, 64);
        assert_eq!(op.seq_len, 4);
        assert_eq!(op.kv_seq_len, 8);
        assert_eq!(op.head_dim, 64);
        // scale = 1/sqrt(64) = 1/8 = 0.125
        assert!((op.scale - 0.125).abs() < 1e-6);
    }

    #[test]
    fn test_attention_self_attention_fields() {
        let op = AttentionOp::self_attention(16, 32);
        assert_eq!(op.seq_len, 16);
        assert_eq!(op.kv_seq_len, 16); // Self-attention: same lengths
        assert_eq!(op.head_dim, 32);
    }

    #[test]
    fn test_attention_name() {
        let op = AttentionOp::self_attention(1, 4);
        assert_eq!(op.name(), "attention");
    }

    #[test]
    fn test_attention_v_size_mismatch() {
        let op = AttentionOp::self_attention(2, 4);
        let q = vec![1.0; 8];
        let k = vec![1.0; 8];
        let v = vec![1.0; 4]; // Wrong: should be 8

        let result = op.execute((q, k, v), Backend::Scalar);
        assert!(result.is_err());
    }

    #[test]
    fn test_attention_single_position() {
        // seq=1, kv=1, head_dim=4
        let op = AttentionOp::self_attention(1, 4);
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 0.0];
        let v = vec![2.0, 3.0, 4.0, 5.0];

        let output = op.execute((q, k, v), Backend::Scalar).unwrap();
        assert_eq!(output.len(), 4);
        // With single position, softmax of single score is 1.0
        // Output = 1.0 * V = V
        assert!((output[0] - 2.0).abs() < 1e-5);
        assert!((output[1] - 3.0).abs() < 1e-5);
        assert!((output[2] - 4.0).abs() < 1e-5);
        assert!((output[3] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_attention_uniform_scores() {
        // If Q and K are identical for all positions, scores are equal
        // Output should be average of V rows
        let op = AttentionOp::new(1, 2, 2);
        let head_dim = 2;

        let q = vec![1.0, 1.0]; // 1x2
        let k = vec![1.0, 1.0, 1.0, 1.0]; // 2x2, both identical
        let v = vec![1.0, 0.0, 0.0, 1.0]; // 2x2

        let output = op.execute((q, k, v), Backend::Scalar).unwrap();
        assert_eq!(output.len(), head_dim);
        // Scores are equal => softmax gives [0.5, 0.5]
        // Output = 0.5 * [1, 0] + 0.5 * [0, 1] = [0.5, 0.5]
        assert!((output[0] - 0.5).abs() < 1e-5);
        assert!((output[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_simd_dot_exact_multiple_of_four() {
        assert_dot_iota(8); // sum(1..=8) = 36
    }

    #[test]
    fn test_simd_dot_single_element() {
        assert_dot(&[3.0], &[4.0], 12.0);
    }

    #[test]
    fn test_simd_dot_two_elements() {
        assert_dot(&[2.0, 3.0], &[4.0, 5.0], 23.0);
    }

    #[test]
    fn test_simd_dot_three_elements() {
        assert_dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 32.0);
    }

    #[test]
    fn test_simd_dot_large_avx2_aligned() {
        assert_dot_iota(16); // sum(1..=16) = 136
    }

    #[test]
    fn test_simd_dot_large_avx2_remainder() {
        assert_dot_iota(19); // sum(1..=19) = 190
    }

    #[test]
    fn test_simd_dot_zeros() {
        assert_dot(&[0.0; 16], &[1.0; 16], 0.0);
    }

    #[test]
    fn test_simd_dot_negative_values() {
        assert_dot(&[-1.0, -2.0, -3.0, -4.0], &[1.0; 4], -10.0);
    }

    #[test]
    fn test_simd_softmax_row_large_values() {
        assert_softmax_normalized(&[1000.0, 1001.0, 1002.0]);
    }

    #[test]
    fn test_simd_softmax_row_negative_values() {
        assert_softmax_normalized(&[-10.0, -20.0, -5.0]);
    }

    #[test]
    fn test_attention_clone() {
        let op = AttentionOp::new(4, 8, 64);
        let cloned = op.clone();
        assert_eq!(cloned.seq_len, 4);
        assert_eq!(cloned.kv_seq_len, 8);
        assert_eq!(cloned.head_dim, 64);
        assert!((cloned.scale - op.scale).abs() < 1e-10);
    }

    #[test]
    fn test_attention_multi_query_rows() {
        let op = AttentionOp::new(3, 2, 2);
        let q = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![10.0, 20.0, 30.0, 40.0];
        assert_attention_ok(&op, q, k, v, 6);
    }

    #[test]
    fn test_attention_tokens_cross_attention() {
        let op = AttentionOp::new(1, 100, 64);
        assert_eq!(op.tokens(&(vec![], vec![], vec![])), 64);
    }

    // simd_dot coverage — AVX2 remainder and various non-aligned sizes

    #[test]
    fn test_simd_dot_avx2_remainders() {
        // Test various sizes: 1 chunk+1, +2, +7, 3 chunks exact, sub-chunk
        for n in [9, 10, 15, 24, 5, 6, 7] {
            assert_dot_iota(n);
        }
    }

    #[test]
    fn test_simd_dot_large_64_elements() {
        assert_dot_iota(64); // sum(1..=64) = 2080
    }

    #[test]
    fn test_simd_dot_orthogonal() {
        let mut a = vec![0.0; 9];
        let mut b = vec![0.0; 9];
        a[0] = 1.0;
        b[1] = 1.0;
        assert_dot(&a, &b, 0.0);
    }

    #[test]
    fn test_attention_execute_non_aligned_head_dim() {
        let op = AttentionOp::self_attention(2, 9);
        let output = assert_attention_ok(&op, vec![1.0; 18], vec![1.0; 18], vec![1.0; 18], 18);
        // Uniform Q/K → uniform softmax → output = mean of V rows = 1.0
        for val in &output {
            assert!((val - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_attention_execute_head_dim_17() {
        let op = AttentionOp::new(1, 3, 17);
        let q: Vec<f32> = (0..17).map(|i| (i as f32) * 0.1).collect();
        let k: Vec<f32> = (0..51).map(|i| ((i % 5) as f32) * 0.2).collect();
        let v: Vec<f32> = (0..51).map(|i| (i as f32) * 0.01).collect();
        assert_attention_ok(&op, q, k, v, 17);
    }

    // =========================================================================
    // simd_dot coverage: AVX2 path with every remainder size (Refs CB-130)
    // =========================================================================

    /// Verify simd_dot with vectors of exactly `n` elements where each element
    /// is a known value, checking against a scalar reference implementation.
    fn assert_dot_scalar_ref(n: usize) {
        let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 + 1.0).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32) * 0.7 - 0.5).collect();
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let result = AttentionOp::simd_dot(&a, &b);
        assert!(
            (result - expected).abs() < 1e-2 * expected.abs().max(1.0),
            "n={n}: dot={result}, expected={expected}"
        );
    }

    #[test]
    fn test_simd_dot_avx2_remainder_0() {
        // 32 elements: exactly 4 AVX2 chunks, 0 remainder
        assert_dot_scalar_ref(32);
    }

    #[test]
    fn test_simd_dot_avx2_remainder_1() {
        // 33 elements: 4 AVX2 chunks + 1 remainder
        assert_dot_scalar_ref(33);
    }

    #[test]
    fn test_simd_dot_avx2_remainder_2() {
        // 34 elements: 4 AVX2 chunks + 2 remainder
        assert_dot_scalar_ref(34);
    }

    #[test]
    fn test_simd_dot_avx2_remainder_3() {
        // 35 elements: 4 AVX2 chunks + 3 remainder
        assert_dot_scalar_ref(35);
    }

    #[test]
    fn test_simd_dot_avx2_remainder_4() {
        // 36 elements: 4 AVX2 chunks + 4 remainder
        assert_dot_scalar_ref(36);
    }

    #[test]
    fn test_simd_dot_avx2_remainder_5() {
        // 37 elements: 4 AVX2 chunks + 5 remainder
        assert_dot_scalar_ref(37);
    }

    #[test]
    fn test_simd_dot_avx2_remainder_6() {
        // 38 elements: 4 AVX2 chunks + 6 remainder
        assert_dot_scalar_ref(38);
    }

    #[test]
    fn test_simd_dot_avx2_remainder_7() {
        // 39 elements: 4 AVX2 chunks + 7 remainder
        assert_dot_scalar_ref(39);
    }

    #[test]
    fn test_simd_dot_large_128() {
        // 128 elements: 16 AVX2 chunks, exercises sustained SIMD loop
        assert_dot_scalar_ref(128);
    }

    #[test]
    fn test_simd_dot_large_1024() {
        // 1024 elements: 128 AVX2 chunks, large vector stress test
        assert_dot_scalar_ref(1024);
    }

    #[test]
    fn test_simd_dot_large_1024_plus_5() {
        // 1029 elements: 128 AVX2 chunks + 5 remainder, large + non-aligned
        assert_dot_scalar_ref(1029);
    }

    #[test]
    fn test_simd_dot_known_identity() {
        // Unit vector dot product = 1.0
        let n = 64;
        let a: Vec<f32> = {
            let mut v = vec![0.0; n];
            v[0] = 1.0;
            v
        };
        let b = a.clone();
        let result = AttentionOp::simd_dot(&a, &b);
        assert!((result - 1.0).abs() < 1e-6, "identity dot = {result}");
    }

    #[test]
    fn test_simd_dot_alternating_signs() {
        // Alternating +1/-1 should cancel to 0 for even length
        let n = 64;
        let a: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let b = vec![1.0; n];
        let result = AttentionOp::simd_dot(&a, &b);
        assert!((result).abs() < 1e-5, "alternating dot = {result}");
    }

    #[test]
    fn test_simd_dot_large_values() {
        // Large values should still compute correctly
        let a = vec![1000.0; 16];
        let b = vec![1000.0; 16];
        let expected = 1000.0 * 1000.0 * 16.0;
        let result = AttentionOp::simd_dot(&a, &b);
        assert!((result - expected).abs() < 1.0, "large dot = {result}, expected = {expected}");
    }

    #[test]
    fn test_simd_dot_mixed_positive_negative() {
        // 10 elements: 1 AVX2 chunk (8) + 2 remainder, with mixed signs
        let a = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0];
        let b = vec![10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let result = AttentionOp::simd_dot(&a, &b);
        assert!((result - expected).abs() < 1e-3, "mixed dot = {result}, expected = {expected}");
    }

    #[test]
    fn test_simd_dot_very_small_values() {
        let a = vec![1e-10; 16];
        let b = vec![1e-10; 16];
        let expected = 1e-20 * 16.0;
        let result = AttentionOp::simd_dot(&a, &b);
        assert!((result - expected).abs() < 1e-24, "small dot = {result}, expected = {expected}");
    }

    // =========================================================================
    // Full attention execute path with head_dim sizes that stress simd_dot
    // =========================================================================

    #[test]
    fn test_attention_head_dim_64_multi_seq() {
        // head_dim=64 (8 AVX2 chunks exactly): realistic transformer config
        let op = AttentionOp::self_attention(4, 64);
        let q = vec![0.1; 4 * 64];
        let k = vec![0.1; 4 * 64];
        let v = vec![1.0; 4 * 64];
        let output = assert_attention_ok(&op, q, k, v, 4 * 64);
        // Uniform input => uniform softmax => output = mean of V rows = 1.0
        for val in &output {
            assert!((val - 1.0).abs() < 1e-4, "expected ~1.0, got {val}");
        }
    }

    #[test]
    fn test_attention_head_dim_128() {
        // head_dim=128 (16 AVX2 chunks): large head dimension
        let op = AttentionOp::new(2, 3, 128);
        let q: Vec<f32> = (0..2 * 128).map(|i| (i as f32) * 0.001).collect();
        let k: Vec<f32> = (0..3 * 128).map(|i| ((i % 7) as f32) * 0.01).collect();
        let v: Vec<f32> = (0..3 * 128).map(|i| (i as f32) * 0.005).collect();
        assert_attention_ok(&op, q, k, v, 2 * 128);
    }

    #[test]
    fn test_attention_head_dim_33() {
        // head_dim=33: 4 AVX2 chunks + 1 remainder element in simd_dot
        let op = AttentionOp::new(2, 2, 33);
        let q = vec![0.5; 2 * 33];
        let k = vec![0.5; 2 * 33];
        let v = vec![2.0; 2 * 33];
        let output = assert_attention_ok(&op, q, k, v, 2 * 33);
        for val in &output {
            assert!((val - 2.0).abs() < 1e-4, "expected ~2.0, got {val}");
        }
    }

    #[test]
    fn test_attention_head_dim_7() {
        // head_dim=7: 0 AVX2 chunks, all remainder (exercises pure remainder path)
        let op = AttentionOp::self_attention(2, 7);
        let q = vec![1.0; 2 * 7];
        let k = vec![1.0; 2 * 7];
        let v = vec![3.0; 2 * 7];
        let output = assert_attention_ok(&op, q, k, v, 2 * 7);
        for val in &output {
            assert!((val - 3.0).abs() < 1e-4, "expected ~3.0, got {val}");
        }
    }

    // =========================================================================
    // FALSIFY-ATT: attention-kernel-v1.yaml contract (trueno AttentionOp)
    //
    // Five-Whys (PMAT-354):
    //   Why 1: trueno had 50+ attention unit tests but zero FALSIFY-ATT-* tests
    //   Why 2: unit tests verify shapes/finiteness, not mathematical invariants
    //   Why 3: no mapping from attention-kernel-v1.yaml to trueno test names
    //   Why 4: trueno predates the provable-contracts YAML convention
    //   Why 5: attention was "obviously correct" (standard formula)
    //
    // References:
    //   - provable-contracts/contracts/attention-kernel-v1.yaml
    //   - Vaswani et al. (2017) "Attention Is All You Need"
    // =========================================================================

    /// FALSIFY-ATT-001: Weight normalization — each softmax row sums to 1.0
    ///
    /// Contract: Σ_j softmax(QK^T/√d_k)_{ij} = 1 for all i
    #[test]
    fn falsify_att_001_weight_normalization() {
        let test_rows: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![-5.0, 0.0, 5.0, 10.0],
            vec![1000.0, 1001.0, 1002.0],
            vec![1e-7, 1e-7, 1e-7],
            vec![0.0; 8],
            vec![-100.0, 100.0],
        ];

        for values in &test_rows {
            let mut scores = values.clone();
            AttentionOp::simd_softmax_row(&mut scores);
            let sum: f32 = scores.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "FALSIFIED ATT-001: softmax row sum = {sum}, expected 1.0 for input {values:?}"
            );
        }
    }

    /// FALSIFY-ATT-002: Output convexity — output rows are convex combinations of V rows
    ///
    /// Contract: min_j(V[j][d]) ≤ output[i][d] ≤ max_j(V[j][d]) for all i, d
    #[test]
    fn falsify_att_002_output_convexity() {
        let seq_len = 2;
        let kv_seq_len = 3;
        let head_dim = 4;
        let op = AttentionOp::new(seq_len, kv_seq_len, head_dim);

        let q = vec![1.0, 0.5, -0.3, 0.8, -1.0, 0.2, 0.7, -0.5];
        let k = vec![0.3, -0.7, 1.0, 0.2, -0.5, 0.8, 0.1, -0.3, 0.6, -0.1, 0.4, 0.9];
        let v = vec![2.0, -3.0, 5.0, 1.0, -1.0, 4.0, -2.0, 7.0, 3.0, 0.0, -4.0, 6.0];

        let output = op.execute((q, k, v.clone()), Backend::Scalar).unwrap();

        for qi in 0..seq_len {
            for d in 0..head_dim {
                let out_val = output[qi * head_dim + d];

                let v_col_min =
                    (0..kv_seq_len).map(|ki| v[ki * head_dim + d]).fold(f32::INFINITY, f32::min);
                let v_col_max = (0..kv_seq_len)
                    .map(|ki| v[ki * head_dim + d])
                    .fold(f32::NEG_INFINITY, f32::max);

                assert!(
                    out_val >= v_col_min - 1e-5 && out_val <= v_col_max + 1e-5,
                    "FALSIFIED ATT-002: output[{qi}][{d}] = {out_val} outside V column [{v_col_min}, {v_col_max}]"
                );
            }
        }
    }

    /// FALSIFY-ATT-003: Scaling factor — uses 1/√d_k not 1/d_k
    ///
    /// Contract: scale = 1/√d_k
    #[test]
    fn falsify_att_003_scaling_factor() {
        for d_k in [4, 8, 16, 32, 64, 128] {
            let op = AttentionOp::self_attention(1, d_k);
            let expected = 1.0 / (d_k as f32).sqrt();
            assert!(
                (op.scale - expected).abs() < 1e-6,
                "FALSIFIED ATT-003: scale = {}, expected 1/√{d_k} = {expected}",
                op.scale
            );
            // Verify it's NOT the wrong 1/d_k scaling
            if d_k > 1 {
                let wrong = 1.0 / d_k as f32;
                assert!(
                    (op.scale - wrong).abs() > 1e-6,
                    "FALSIFIED ATT-003: scale matches wrong 1/{d_k} = {wrong}",
                );
            }
        }
    }

    /// FALSIFY-ATT-005: Weights bounded — all attention weights in [0, 1)
    ///
    /// Contract: 0 < attn_{ij} < 1 for all i,j in exact arithmetic.
    /// In f32, exp(-200) underflows to 0.0 so we test w >= 0 and w < 1.
    /// For moderate inputs (max gap < 80), strict w > 0 holds.
    #[test]
    fn falsify_att_005_weights_bounded() {
        // Moderate-range inputs where exp() doesn't underflow to 0
        let test_rows: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![-5.0, 0.0, 5.0],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![1e-10, 1e-10],
            vec![-10.0, -10.0, -10.0],
            vec![20.0, 20.5, 21.0],
        ];

        for values in &test_rows {
            let mut scores = values.clone();
            AttentionOp::simd_softmax_row(&mut scores);
            for (j, &w) in scores.iter().enumerate() {
                assert!(
                    w > 0.0,
                    "FALSIFIED ATT-005: weight[{j}] = {w} not > 0 for input {values:?}"
                );
                assert!(
                    w < 1.0,
                    "FALSIFIED ATT-005: weight[{j}] = {w} not < 1 for input {values:?} (m >= 2)"
                );
            }
        }
    }

    /// FALSIFY-ATT-002b: Convexity with uniform V — output must equal V
    ///
    /// If all V rows are identical, output = V regardless of Q, K
    #[test]
    fn falsify_att_002b_uniform_v_identity() {
        let op = AttentionOp::new(2, 4, 8);
        let q: Vec<f32> = (0..16).map(|i| (i as f32) * 0.37).collect();
        let k: Vec<f32> = (0..32).map(|i| (i as f32) * 0.13).collect();
        // All 4 V rows are identical: [1, 2, 3, 4, 5, 6, 7, 8]
        let v_row = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let v: Vec<f32> = v_row.iter().copied().cycle().take(32).collect();

        let output = op.execute((q, k, v), Backend::Scalar).unwrap();

        for qi in 0..2 {
            for d in 0..8 {
                let diff = (output[qi * 8 + d] - v_row[d]).abs();
                assert!(
                    diff < 1e-5,
                    "FALSIFIED ATT-002: uniform V output[{qi}][{d}] = {}, expected {}",
                    output[qi * 8 + d],
                    v_row[d]
                );
            }
        }
    }
}
