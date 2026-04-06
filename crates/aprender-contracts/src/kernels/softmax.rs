//! Softmax kernel: numerically stable exponential normalization.
//!
//! Matches `softmax-kernel-v1.yaml`.
//! Four phases: `find_max` -> `exp_subtract` -> `sum_exp` -> `normalize`.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ────────────────────────────────────────────────────────────────────────────
// Scalar implementation
// ────────────────────────────────────────────────────────────────────────────

/// Scalar reference implementation of numerically stable softmax.
///
/// Computes `softmax(x)_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))`.
///
/// # Panics
///
/// Panics if `input` and `output` have different lengths or if `input` is empty.
pub fn softmax_scalar(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len(), "input/output length mismatch");
    assert!(!input.is_empty(), "softmax requires non-empty input");

    // Phase 1: find max for numerical stability
    let mut max_val = input[0];
    for &x in &input[1..] {
        if x > max_val {
            max_val = x;
        }
    }

    // Phase 2: exp(x_i - max)
    for (i, &x) in input.iter().enumerate() {
        output[i] = (x - max_val).exp();
    }

    // Phase 3: sum of exponentials
    let mut sum = 0.0_f32;
    for &e in output.iter() {
        sum += e;
    }

    // Phase 4: normalize
    let inv_sum = 1.0 / sum;
    for o in output.iter_mut() {
        *o *= inv_sum;
    }
}

// ────────────────────────────────────────────────────────────────────────────
// AVX2 implementation
// ────────────────────────────────────────────────────────────────────────────

/// AVX2 SIMD implementation of numerically stable softmax.
///
/// Uses `_mm256_max_ps` for horizontal max reduction across 8-wide lanes,
/// then scalar fallback for exp (no AVX2 exp intrinsic), and vectorized
/// final division.
///
/// # Safety
///
/// Requires AVX2 support. Caller must verify with `is_x86_feature_detected!("avx2")`.
///
/// # Panics
///
/// Panics if `input` and `output` have different lengths or if `input` is empty.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn softmax_avx2(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len(), "input/output length mismatch");
    let n = input.len();
    assert!(n > 0, "softmax requires non-empty input");

    let chunks = n / 8;
    let remainder = n % 8;

    // SAFETY: caller guarantees AVX2 is available; target_feature gate enforces it.
    unsafe {
        // Phase 1: find max using AVX2 horizontal reduction
        let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);
        for i in 0..chunks {
            let v = _mm256_loadu_ps(input.as_ptr().add(i * 8));
            max_vec = _mm256_max_ps(max_vec, v);
        }

        // Horizontal max reduction of the 8-wide vector
        let mut max_val = f32::NEG_INFINITY;
        let mut tmp = [0.0_f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), max_vec);
        for &v in &tmp {
            if v > max_val {
                max_val = v;
            }
        }
        // Check remainder elements
        for i in (chunks * 8)..n {
            if input[i] > max_val {
                max_val = input[i];
            }
        }

        // Phase 2: exp(x_i - max) — scalar fallback (no AVX2 exp intrinsic)
        for i in 0..n {
            output[i] = (input[i] - max_val).exp();
        }

        // Phase 3: sum of exponentials using AVX2 accumulation
        let mut sum_vec = _mm256_setzero_ps();
        for i in 0..chunks {
            let v = _mm256_loadu_ps(output.as_ptr().add(i * 8));
            sum_vec = _mm256_add_ps(sum_vec, v);
        }
        _mm256_storeu_ps(tmp.as_mut_ptr(), sum_vec);
        let mut sum = 0.0_f32;
        for &v in &tmp {
            sum += v;
        }
        for i in (chunks * 8)..n {
            sum += output[i];
        }

        // Phase 4: normalize using AVX2 division
        let inv_sum = 1.0 / sum;
        let inv_vec = _mm256_set1_ps(inv_sum);
        for i in 0..chunks {
            let v = _mm256_loadu_ps(output.as_ptr().add(i * 8));
            let r = _mm256_mul_ps(v, inv_vec);
            _mm256_storeu_ps(output.as_mut_ptr().add(i * 8), r);
        }
        for i in (chunks * 8)..(chunks * 8 + remainder) {
            output[i] *= inv_sum;
        }
    }
}

include!("softmax_ptx.rs");

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::ulp::assert_ulp_eq;
    use super::*;
    use proptest::prelude::*;

    // ── Scalar known-answer tests ────────────────────────────────────────

    #[test]
    fn test_softmax_uniform() {
        let input = [1.0_f32, 1.0, 1.0];
        let mut output = [0.0_f32; 3];
        softmax_scalar(&input, &mut output);
        let expected = 1.0 / 3.0;
        for &o in &output {
            assert!((o - expected).abs() < 1e-6, "expected ~{expected}, got {o}");
        }
    }

    #[test]
    fn test_softmax_two_equal() {
        let input = [0.0_f32, 0.0];
        let mut output = [0.0_f32; 2];
        softmax_scalar(&input, &mut output);
        for &o in &output {
            assert!((o - 0.5).abs() < 1e-6, "expected 0.5, got {o}");
        }
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values should not overflow thanks to max-subtraction trick
        let input = [1000.0_f32, 0.0, 0.0];
        let mut output = [0.0_f32; 3];
        softmax_scalar(&input, &mut output);
        assert!(output[0].is_finite(), "output[0] must be finite");
        assert!(output[1].is_finite(), "output[1] must be finite");
        assert!(output[2].is_finite(), "output[2] must be finite");
        // Dominant element should be close to 1.0
        assert!((output[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_single_element() {
        let input = [42.0_f32];
        let mut output = [0.0_f32; 1];
        softmax_scalar(&input, &mut output);
        assert!(
            (output[0] - 1.0).abs() < 1e-7,
            "softmax of single element must be 1.0"
        );
    }

    #[test]
    #[should_panic(expected = "input/output length mismatch")]
    fn test_softmax_length_mismatch() {
        let input = [1.0_f32, 2.0];
        let mut output = [0.0_f32; 3];
        softmax_scalar(&input, &mut output);
    }

    #[test]
    #[should_panic(expected = "softmax requires non-empty input")]
    fn test_softmax_empty_input() {
        let input: [f32; 0] = [];
        let mut output: [f32; 0] = [];
        softmax_scalar(&input, &mut output);
    }

    // ── Property-based tests ─────────────────────────────────────────────

    proptest! {
        #[test]
        fn prop_softmax_sums_to_one(
            v in proptest::collection::vec(-100.0_f32..100.0, 1..64)
        ) {
            let mut out = vec![0.0_f32; v.len()];
            softmax_scalar(&v, &mut out);
            let sum: f32 = out.iter().sum();
            prop_assert!(
                (sum - 1.0).abs() < 1e-5,
                "softmax sum = {sum}, expected ~1.0"
            );
        }

        #[test]
        fn prop_softmax_outputs_in_unit_interval(
            v in proptest::collection::vec(-100.0_f32..100.0, 1..64)
        ) {
            let mut out = vec![0.0_f32; v.len()];
            softmax_scalar(&v, &mut out);
            for (i, &o) in out.iter().enumerate() {
                prop_assert!(
                    (0.0..=1.0).contains(&o),
                    "output[{i}] = {o} not in [0,1]"
                );
            }
        }

        #[test]
        fn prop_softmax_order_preservation(
            v in proptest::collection::vec(-50.0_f32..50.0, 2..32)
        ) {
            let mut out = vec![0.0_f32; v.len()];
            softmax_scalar(&v, &mut out);
            for i in 0..v.len() {
                for j in (i + 1)..v.len() {
                    if v[i] > v[j] {
                        prop_assert!(
                            out[i] >= out[j],
                            "order violated: v[{i}]={} > v[{j}]={} but out[{i}]={} < out[{j}]={}",
                            v[i], v[j], out[i], out[j]
                        );
                    }
                }
            }
        }

        #[test]
        fn prop_softmax_translation_invariance(
            v in proptest::collection::vec(-50.0_f32..50.0, 2..32),
            c in -50.0_f32..50.0
        ) {
            let mut out1 = vec![0.0_f32; v.len()];
            softmax_scalar(&v, &mut out1);

            let shifted: Vec<f32> = v.iter().map(|&x| x + c).collect();
            let mut out2 = vec![0.0_f32; v.len()];
            softmax_scalar(&shifted, &mut out2);

            for i in 0..v.len() {
                prop_assert!(
                    (out1[i] - out2[i]).abs() < 1e-5,
                    "translation invariance violated at {i}: {} vs {}",
                    out1[i], out2[i]
                );
            }
        }
    }

    // ── AVX2 parity tests ────────────────────────────────────────────────

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_softmax_avx2_basic() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let input = [
            1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ];
        let mut scalar_out = [0.0_f32; 16];
        let mut avx2_out = [0.0_f32; 16];
        softmax_scalar(&input, &mut scalar_out);
        unsafe { softmax_avx2(&input, &mut avx2_out) };
        assert_ulp_eq(&scalar_out, &avx2_out, 8);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_softmax_avx2_non_multiple_of_8() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let mut scalar_out = [0.0_f32; 5];
        let mut avx2_out = [0.0_f32; 5];
        softmax_scalar(&input, &mut scalar_out);
        unsafe { softmax_avx2(&input, &mut avx2_out) };
        assert_ulp_eq(&scalar_out, &avx2_out, 8);
    }

    #[cfg(target_arch = "x86_64")]
    proptest! {
        #[test]
        fn prop_softmax_avx2_parity(
            v in proptest::collection::vec(-100.0_f32..100.0, 1..64)
        ) {
            if !is_x86_feature_detected!("avx2") {
                return Ok(());
            }
            let mut scalar_out = vec![0.0_f32; v.len()];
            let mut avx2_out = vec![0.0_f32; v.len()];
            softmax_scalar(&v, &mut scalar_out);
            unsafe { softmax_avx2(&v, &mut avx2_out) };
            assert_ulp_eq(&scalar_out, &avx2_out, 8);
        }
    }

    // ── PTX structural tests ─────────────────────────────────────────────

    #[test]
    fn test_softmax_ptx_version() {
        let ptx = softmax_ptx();
        assert!(ptx.contains(".version 8.5"), "missing PTX version");
    }

    #[test]
    fn test_softmax_ptx_target() {
        let ptx = softmax_ptx();
        assert!(ptx.contains(".target sm_90"), "missing PTX target");
    }

    #[test]
    fn test_softmax_ptx_entry() {
        let ptx = softmax_ptx();
        assert!(ptx.contains(".entry softmax_kernel"), "missing entry point");
    }

    #[test]
    fn test_softmax_ptx_ret() {
        let ptx = softmax_ptx();
        assert!(ptx.contains("ret;"), "missing ret instruction");
    }

    #[test]
    fn test_softmax_ptx_shared_memory() {
        let ptx = softmax_ptx();
        assert!(ptx.contains(".shared"), "missing shared memory declaration");
    }

    #[test]
    fn test_softmax_ptx_warp_shuffle() {
        let ptx = softmax_ptx();
        assert!(
            ptx.contains("shfl.sync"),
            "missing warp shuffle instructions"
        );
    }

    #[test]
    fn test_softmax_ptx_bar_sync() {
        let ptx = softmax_ptx();
        assert!(
            ptx.contains("bar.sync"),
            "missing bar.sync for block synchronization"
        );
    }

    #[test]
    fn test_softmax_ptx_balanced_braces() {
        let ptx = softmax_ptx();
        let open = ptx.matches('{').count();
        let close = ptx.matches('}').count();
        assert_eq!(
            open, close,
            "unbalanced braces: {open} open vs {close} close"
        );
    }
}
