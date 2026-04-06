use super::*;

#[test]
fn test_avx512_dot() {
    avx512_test(|| {
        let a: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        let b: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
        let result = unsafe { Avx512Backend::dot(&a, &b) };
        let expected: f32 = (1..=32).map(|i| (i * i) as f32).sum();
        assert!((result - expected).abs() < 1e-3);
    });
}

#[test]
fn test_avx512_sum() {
    avx512_test(|| assert_reduction_f32(528.0, 1e-3, Avx512Backend::sum));
}

#[test]
fn test_avx512_max() {
    avx512_test(|| assert_reduction_f32(32.0, 1e-6, Avx512Backend::max));
}

#[test]
fn test_avx512_min() {
    avx512_test(|| assert_reduction_f32(1.0, 1e-6, Avx512Backend::min));
}

#[test]
fn test_avx512_argmax() {
    avx512_test(|| assert_reduction_usize(31, Avx512Backend::argmax));
}

#[test]
fn test_avx512_argmin() {
    avx512_test(|| assert_reduction_usize(0, Avx512Backend::argmin));
}

#[test]
fn test_avx512_sum_kahan() {
    avx512_test(|| assert_reduction_f32(528.0, 1e-3, Avx512Backend::sum_kahan));
}

// ========== Norms ==========

#[test]
fn test_avx512_norm_l2() {
    avx512_test(|| {
        let a = vec![3.0, 4.0];
        // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
        let result = unsafe { Avx512Backend::norm_l2(&a) };
        assert!((result - 5.0).abs() < 1e-5);
    });
}

#[test]
fn test_avx512_norm_l1() {
    avx512_test(|| {
        let a = vec![-1.0, 2.0, -3.0, 4.0];
        // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
        let result = unsafe { Avx512Backend::norm_l1(&a) };
        assert!((result - 10.0).abs() < 1e-5);
    });
}

#[test]
fn test_avx512_norm_linf() {
    avx512_test(|| {
        let a = vec![-5.0, 2.0, -3.0, 4.0];
        // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
        let result = unsafe { Avx512Backend::norm_linf(&a) };
        assert!((result - 5.0).abs() < 1e-5);
    });
}
