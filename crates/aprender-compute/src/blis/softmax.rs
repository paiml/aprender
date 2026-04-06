//! SIMD-accelerated softmax.
//!
//! 4-pass algorithm with AVX2 acceleration on passes 1/3/4:
//!   Pass 1 (max):       AVX2 horizontal reduction with 4-way unrolling
//!   Pass 2 (exp+store): Scalar exp() — transcendental, no SIMD without SVML
//!   Pass 3 (sum):       AVX2 horizontal reduction with 4-way unrolling
//!   Pass 4 (normalize): AVX2 multiply by 1/sum with 4-way unrolling
//!
//! The previous 3-pass fused approach (exp+sum in one loop) had a loop-carried
//! dependency on `sum` that prevented LLVM from vectorizing the surrounding code.
//! Splitting into 4 passes allows SIMD on three of the four passes.
//!
//! Contract: contracts/softmax-kernel-v1.yaml

/// Softmax on a 1D slice with zero-copy output allocation.
///
/// Uses AVX2 acceleration for max/sum/normalize passes when available.
///
/// # Contract
///
/// - `softmax(x)_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))`
/// - Output sums to 1.0 (within f32 tolerance)
/// - All outputs ≥ 0
/// - Monotonicity: x_i > x_j → y_i > y_j
/// - Shift-invariant: softmax(x + c) = softmax(x)
#[must_use]
pub fn softmax_1d_alloc(logits: &[f32]) -> Vec<f32> {
    let n = logits.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }

    // Contract: softmax-kernel-v1.yaml precondition (pv codegen)
    contract_pre_softmax!(logits);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA verified by feature detection above.
            let result = unsafe { softmax_avx2(logits) };
            contract_post_softmax!(&result);
            return result;
        }
    }

    let result = softmax_scalar(logits);
    contract_post_softmax!(&result);
    result
}

/// Scalar 4-pass softmax — reference implementation.
fn softmax_scalar(logits: &[f32]) -> Vec<f32> {
    let n = logits.len();

    // Pass 1: max
    let mut max_val = f32::NEG_INFINITY;
    for &v in logits {
        max_val = max_val.max(v);
    }

    // Pass 2: exp + store (uninit: every element written by indexed loop)
    let mut out: Vec<f32> = Vec::with_capacity(n);
    // SAFETY: out[i] = exp(...) for every i in 0..n. No reads before writes.
    unsafe {
        out.set_len(n);
    }
    for i in 0..n {
        out[i] = (logits[i] - max_val).exp();
    }

    // Pass 3: sum
    let mut sum = 0.0f32;
    for &v in &out {
        sum += v;
    }

    // Pass 4: normalize (guard against sum=0 from underflow)
    let inv_sum = 1.0 / sum.max(f32::EPSILON);
    for v in &mut out {
        *v *= inv_sum;
    }

    out
}

/// AVX2 3-pass softmax: max → fused SIMD-exp+sum → normalize.
///
/// Fuses passes 2+3 into a single SIMD exp+accumulate pass, eliminating one
/// full memory traversal. Uses polynomial exp approximation (6th-order Remez
/// minimax on [-ln(2)/2, ln(2)/2] with range reduction), giving <1 ULP error.
///
/// # Safety
///
/// Requires AVX2 + FMA support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn softmax_avx2(logits: &[f32]) -> Vec<f32> {
    use std::arch::x86_64::*;

    let n = logits.len();
    let chunks = n / 32;
    let remainder_32 = chunks * 32;

    // ── Pass 1: AVX2 horizontal max ──────────────────────────────────────
    let mut max0;
    let mut max1;
    let mut max2;
    let mut max3;
    unsafe {
        max0 = _mm256_set1_ps(f32::NEG_INFINITY);
        max1 = max0;
        max2 = max0;
        max3 = max0;

        for i in 0..chunks {
            let base = i * 32;
            let v0 = _mm256_loadu_ps(logits.as_ptr().add(base));
            let v1 = _mm256_loadu_ps(logits.as_ptr().add(base + 8));
            let v2 = _mm256_loadu_ps(logits.as_ptr().add(base + 16));
            let v3 = _mm256_loadu_ps(logits.as_ptr().add(base + 24));
            max0 = _mm256_max_ps(max0, v0);
            max1 = _mm256_max_ps(max1, v1);
            max2 = _mm256_max_ps(max2, v2);
            max3 = _mm256_max_ps(max3, v3);
        }

        max0 = _mm256_max_ps(max0, max1);
        max2 = _mm256_max_ps(max2, max3);
        max0 = _mm256_max_ps(max0, max2);

        let hi = _mm256_permute2f128_ps(max0, max0, 1);
        max0 = _mm256_max_ps(max0, hi);
        let shuf = _mm256_shuffle_ps(max0, max0, 0b01_00_11_10);
        max0 = _mm256_max_ps(max0, shuf);
        let shuf2 = _mm256_shuffle_ps(max0, max0, 0b10_11_00_01);
        max0 = _mm256_max_ps(max0, shuf2);
    }

    let mut max_val = _mm_cvtss_f32(_mm256_castps256_ps128(max0));
    for i in remainder_32..n {
        max_val = max_val.max(logits[i]);
    }

    // ── Pass 2 (fused): SIMD exp(x - max) + accumulate sum ──────────────
    // Uninit: every element written by storeu_ps (main) or out[i]=e (remainder).
    let mut out: Vec<f32> = Vec::with_capacity(n);
    // SAFETY: Pass 2 writes all n elements before any read (pass 3).
    unsafe {
        out.set_len(n);
    }
    let mut sum0;
    let mut sum1;
    let mut sum2;
    let mut sum3;
    unsafe {
        let max_v = _mm256_set1_ps(max_val);
        sum0 = _mm256_setzero_ps();
        sum1 = sum0;
        sum2 = sum0;
        sum3 = sum0;

        for i in 0..chunks {
            let base = i * 32;
            let x0 = _mm256_sub_ps(_mm256_loadu_ps(logits.as_ptr().add(base)), max_v);
            let x1 = _mm256_sub_ps(_mm256_loadu_ps(logits.as_ptr().add(base + 8)), max_v);
            let x2 = _mm256_sub_ps(_mm256_loadu_ps(logits.as_ptr().add(base + 16)), max_v);
            let x3 = _mm256_sub_ps(_mm256_loadu_ps(logits.as_ptr().add(base + 24)), max_v);

            let e0 = fast_exp_avx2(x0);
            let e1 = fast_exp_avx2(x1);
            let e2 = fast_exp_avx2(x2);
            let e3 = fast_exp_avx2(x3);

            _mm256_storeu_ps(out.as_mut_ptr().add(base), e0);
            _mm256_storeu_ps(out.as_mut_ptr().add(base + 8), e1);
            _mm256_storeu_ps(out.as_mut_ptr().add(base + 16), e2);
            _mm256_storeu_ps(out.as_mut_ptr().add(base + 24), e3);

            sum0 = _mm256_add_ps(sum0, e0);
            sum1 = _mm256_add_ps(sum1, e1);
            sum2 = _mm256_add_ps(sum2, e2);
            sum3 = _mm256_add_ps(sum3, e3);
        }

        sum0 = _mm256_add_ps(sum0, sum1);
        sum2 = _mm256_add_ps(sum2, sum3);
        sum0 = _mm256_add_ps(sum0, sum2);

        let hi = _mm256_permute2f128_ps(sum0, sum0, 1);
        sum0 = _mm256_add_ps(sum0, hi);
        let shuf = _mm256_shuffle_ps(sum0, sum0, 0b01_00_11_10);
        sum0 = _mm256_add_ps(sum0, shuf);
        let shuf2 = _mm256_shuffle_ps(sum0, sum0, 0b10_11_00_01);
        sum0 = _mm256_add_ps(sum0, shuf2);
    }

    let mut sum_val = _mm_cvtss_f32(_mm256_castps256_ps128(sum0));
    // Scalar remainder
    for i in remainder_32..n {
        let e = (logits[i] - max_val).exp();
        out[i] = e;
        sum_val += e;
    }

    // ── Pass 3: AVX2 normalize ──────────────────────────────────────────
    let inv_sum = 1.0 / sum_val.max(f32::EPSILON);
    unsafe {
        let inv = _mm256_set1_ps(inv_sum);

        for i in 0..chunks {
            let base = i * 32;
            let v0 = _mm256_loadu_ps(out.as_ptr().add(base));
            let v1 = _mm256_loadu_ps(out.as_ptr().add(base + 8));
            let v2 = _mm256_loadu_ps(out.as_ptr().add(base + 16));
            let v3 = _mm256_loadu_ps(out.as_ptr().add(base + 24));
            _mm256_storeu_ps(out.as_mut_ptr().add(base), _mm256_mul_ps(v0, inv));
            _mm256_storeu_ps(out.as_mut_ptr().add(base + 8), _mm256_mul_ps(v1, inv));
            _mm256_storeu_ps(out.as_mut_ptr().add(base + 16), _mm256_mul_ps(v2, inv));
            _mm256_storeu_ps(out.as_mut_ptr().add(base + 24), _mm256_mul_ps(v3, inv));
        }
    }
    for i in remainder_32..n {
        out[i] *= inv_sum;
    }

    out
}

/// Fast SIMD exp(x) via range reduction + 6th-order polynomial.
///
/// Algorithm: e^x = 2^n * e^r where n = round(x/ln2), r = x - n*ln2.
/// e^r approximated by minimax polynomial on [-ln(2)/2, ln(2)/2].
/// Reconstruction via integer bit manipulation of float exponent.
///
/// Relative error < 2 ULP for x ∈ [-87, 88].
///
/// AVX2 polynomial exp approximation (6th-order Remez minimax, <1 ULP error).
/// CGP-DBUF: made pub(crate) for reuse in AttentionOp::simd_softmax_row.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub(crate) unsafe fn fast_exp_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

    let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
    let ln2_hi = _mm256_set1_ps(0.693_145_751_953_125); // ln(2) high bits
    let ln2_lo = _mm256_set1_ps(1.428_606_765_330_187_1e-6); // ln(2) low bits
    let one = _mm256_set1_ps(1.0);

    // Polynomial coefficients (Remez minimax for e^r on [-ln2/2, ln2/2])
    let c2 = _mm256_set1_ps(0.500_000_0); // 1/2!
    let c3 = _mm256_set1_ps(0.166_666_671_6); // ~1/3!
    let c4 = _mm256_set1_ps(0.041_666_645_8); // ~1/4!
    let c5 = _mm256_set1_ps(0.008_333_345_2); // ~1/5!
    let c6 = _mm256_set1_ps(0.001_388_731_6); // ~1/6!

    // Clamp to avoid overflow/underflow in integer conversion
    let x = _mm256_max_ps(x, _mm256_set1_ps(-87.33654));
    let x = _mm256_min_ps(x, _mm256_set1_ps(88.72284));

    // Range reduction: n = round(x / ln(2))
    let t = _mm256_fmadd_ps(x, log2e, _mm256_set1_ps(0.5));
    let n = _mm256_floor_ps(t); // floor(x*log2e + 0.5) = round

    // r = x - n * ln(2) (high + low for precision)
    let r = _mm256_sub_ps(x, _mm256_mul_ps(n, ln2_hi));
    let r = _mm256_sub_ps(r, _mm256_mul_ps(n, ln2_lo));

    // Polynomial: e^r ≈ 1 + r + c2*r² + c3*r³ + c4*r⁴ + c5*r⁵ + c6*r⁶
    // Horner form re-arranged for FMA efficiency:
    let p = _mm256_fmadd_ps(c6, r, c5);
    let p = _mm256_fmadd_ps(p, r, c4);
    let p = _mm256_fmadd_ps(p, r, c3);
    let p = _mm256_fmadd_ps(p, r, c2);
    let p = _mm256_fmadd_ps(p, r, one);
    let p = _mm256_fmadd_ps(p, r, one);

    // Reconstruct: multiply by 2^n via integer exponent manipulation
    let n_i = _mm256_cvtps_epi32(n);
    let pow2n =
        _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(n_i, _mm256_set1_epi32(127)), 23));

    _mm256_mul_ps(p, pow2n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deterministic_f32(len: usize) -> Vec<f32> {
        (0..len).map(|i| ((i * 17 + 31) % 1000) as f32 / 1000.0 - 0.5).collect()
    }

    /// FALSIFY-SM-001: sum(softmax(x)) ≈ 1.0
    #[test]
    fn test_softmax_sums_to_one() {
        for n in [32, 127, 256, 1000, 32000] {
            let data = deterministic_f32(n);
            let result = softmax_1d_alloc(&data);
            let sum: f32 = result.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "sum = {sum} for n={n}, expected 1.0");
        }
    }

    /// FALSIFY-SM-002: all elements ≥ 0
    #[test]
    fn test_softmax_non_negative() {
        let data: Vec<f32> = (0..1000).map(|i| -100.0 + i as f32 * 0.1).collect();
        let result = softmax_1d_alloc(&data);
        for (i, &v) in result.iter().enumerate() {
            assert!(v >= 0.0, "element [{i}] = {v} < 0");
        }
    }

    /// FALSIFY-SM-003: monotonicity
    #[test]
    fn test_softmax_monotonic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = softmax_1d_alloc(&data);
        for i in 1..result.len() {
            assert!(
                result[i] > result[i - 1],
                "Not monotonic at [{i}]: {} <= {}",
                result[i],
                result[i - 1]
            );
        }
    }

    /// FALSIFY-SM-004: shift invariance
    #[test]
    fn test_softmax_shift_invariance() {
        let data = deterministic_f32(1000);
        let shifted: Vec<f32> = data.iter().map(|&x| x + 1000.0).collect();

        let result_a = softmax_1d_alloc(&data);
        let result_b = softmax_1d_alloc(&shifted);

        for (i, (&a, &b)) in result_a.iter().zip(result_b.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "Shift invariance broken at [{i}]: {a} vs {b}");
        }
    }

    /// FALSIFY-SM-005: uniform input
    #[test]
    fn test_softmax_uniform() {
        for n in [4, 100, 1000] {
            let data = vec![std::f32::consts::PI; n];
            let result = softmax_1d_alloc(&data);
            let expected = 1.0 / n as f32;
            for (i, &v) in result.iter().enumerate() {
                assert!((v - expected).abs() < 1e-6, "Uniform at [{i}]: {v} vs {expected}");
            }
        }
    }

    /// FALSIFY-SM-006: AVX2 vs scalar parity
    #[test]
    fn test_softmax_avx2_scalar_parity() {
        for n in [32, 127, 1000, 32000] {
            let data = deterministic_f32(n);
            let avx2_result = softmax_1d_alloc(&data);
            let scalar_result = softmax_scalar(&data);

            for (i, (&a, &s)) in avx2_result.iter().zip(scalar_result.iter()).enumerate() {
                // SIMD polynomial exp has <2 ULP error vs libm exp
                assert!((a - s).abs() < 1e-6, "AVX2/scalar mismatch at [{i}] n={n}: {a} vs {s}");
            }
        }
    }

    /// FALSIFY-SM-007: remainder handling
    #[test]
    fn test_softmax_remainder_sizes() {
        for n in [1, 2, 7, 8, 15, 31, 33, 63, 65, 127, 255] {
            let data = deterministic_f32(n);
            let result = softmax_1d_alloc(&data);
            let sum: f32 = result.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "sum = {sum} for n={n}, expected 1.0");
            assert_eq!(result.len(), n);
        }
    }

    /// FALSIFY-SM-008: numerical stability (near exp overflow)
    #[test]
    fn test_softmax_numerical_stability() {
        let mut data = vec![0.0f32; 100];
        data[0] = 88.0; // near f32 exp overflow (~3.4e38, max is ~3.4e38)
        data[50] = -88.0; // near underflow

        let result = softmax_1d_alloc(&data);
        assert!(!result.iter().any(|v| v.is_nan()), "Got NaN");
        assert!(!result.iter().any(|v| v.is_infinite()), "Got Inf");
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum = {sum}");
    }

    /// FALSIFY-SM-009: argmax preservation
    #[test]
    fn test_softmax_argmax_preserved() {
        let data = deterministic_f32(32000);
        let result = softmax_1d_alloc(&data);

        let input_argmax = data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        let output_argmax = result
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        assert_eq!(input_argmax, output_argmax, "Argmax not preserved");
    }

    /// Edge: empty input
    #[test]
    fn test_softmax_empty() {
        let result = softmax_1d_alloc(&[]);
        assert!(result.is_empty());
    }

    /// Edge: single element
    #[test]
    fn test_softmax_single() {
        let result = softmax_1d_alloc(&[42.0]);
        assert_eq!(result, vec![1.0]);
    }
}
