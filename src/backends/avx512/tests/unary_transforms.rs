use super::*;

#[test]
fn test_avx512_scale() {
    avx512_test(|| {
        let a = vec![1.0; 32];
        let mut result = vec![0.0; 32];
        // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
        unsafe {
            Avx512Backend::scale(&a, 3.0, &mut result);
        }
        assert!(result.iter().all(|&x| (x - 3.0).abs() < 1e-6));
    });
}

#[test]
fn test_avx512_abs() {
    avx512_test(|| {
        assert_unary_transform(
            &[-1.0, 2.0, -3.0, 4.0],
            &[1.0, 2.0, 3.0, 4.0],
            1e-6,
            Avx512Backend::abs,
        );
    });
}

#[test]
fn test_avx512_clamp() {
    avx512_test(|| {
        let a = vec![0.0, 5.0, 10.0, 15.0];
        let mut result = vec![0.0; 4];
        // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
        unsafe {
            Avx512Backend::clamp(&a, 2.0, 12.0, &mut result);
        }
        assert_eq!(result, vec![2.0, 5.0, 10.0, 12.0]);
    });
}

#[test]
fn test_avx512_lerp() {
    avx512_test(|| {
        let a = vec![0.0; 32];
        let b = vec![10.0; 32];
        let mut result = vec![0.0; 32];
        // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
        unsafe {
            Avx512Backend::lerp(&a, &b, 0.5, &mut result);
        }
        assert!(result.iter().all(|&x| (x - 5.0).abs() < 1e-5));
    });
}

#[test]
fn test_avx512_fma() {
    avx512_test(|| {
        let a = vec![2.0; 32];
        let b = vec![3.0; 32];
        let c = vec![1.0; 32];
        let mut result = vec![0.0; 32];
        // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
        unsafe {
            Avx512Backend::fma(&a, &b, &c, &mut result);
        }
        assert!(result.iter().all(|&x| (x - 7.0).abs() < 1e-5));
    });
}

#[test]
fn test_avx512_relu() {
    avx512_test(|| {
        assert_unary_transform(
            &[-1.0, 0.0, 1.0, 2.0],
            &[0.0, 0.0, 1.0, 2.0],
            1e-6,
            Avx512Backend::relu,
        );
    });
}

#[test]
fn test_avx512_exp() {
    avx512_test(|| {
        assert_unary_transform(&[0.0, 1.0], &[1.0, std::f32::consts::E], 1e-3, Avx512Backend::exp);
    });
}

#[test]
fn test_avx512_sigmoid() {
    avx512_test(|| {
        assert_unary_transform(&[0.0], &[0.5], 1e-5, Avx512Backend::sigmoid);
    });
}

#[test]
fn test_avx512_gelu() {
    avx512_test(|| {
        assert_unary_transform(&[0.0, 1.0], &[0.0, 0.841_192], 1e-3, Avx512Backend::gelu);
    });
}

#[test]
fn test_avx512_swish() {
    avx512_test(|| {
        assert_unary_transform(&[0.0, 1.0], &[0.0, 0.731_059], 1e-3, Avx512Backend::swish);
    });
}

#[test]
fn test_avx512_tanh() {
    avx512_test(|| {
        assert_unary_transform(&[0.0, 1.0], &[0.0, 0.761_594_2], 1e-3, Avx512Backend::tanh);
    });
}

#[test]
fn test_avx512_sqrt() {
    avx512_test(|| {
        assert_unary_transform(&[4.0, 9.0, 16.0], &[2.0, 3.0, 4.0], 1e-5, Avx512Backend::sqrt);
    });
}

#[test]
fn test_avx512_recip() {
    avx512_test(|| {
        assert_unary_transform(&[2.0, 4.0, 5.0], &[0.5, 0.25, 0.2], 1e-5, Avx512Backend::recip);
    });
}

#[test]
fn test_avx512_transcendental() {
    avx512_test(|| {
        let a = vec![1.0, std::f32::consts::E, 10.0];
        let mut ln_result = vec![0.0; 3];
        let mut log2_result = vec![0.0; 3];
        let mut log10_result = vec![0.0; 3];
        // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
        unsafe {
            Avx512Backend::ln(&a, &mut ln_result);
            Avx512Backend::log2(&a, &mut log2_result);
            Avx512Backend::log10(&a, &mut log10_result);
        }
        assert!((ln_result[0]).abs() < 1e-5);
        assert!((ln_result[1] - 1.0).abs() < 1e-4);
        assert!((log10_result[2] - 1.0).abs() < 1e-5);
    });
}

#[test]
fn test_avx512_trig() {
    avx512_test(|| {
        let a = vec![0.0, std::f32::consts::FRAC_PI_2];
        let mut sin_result = vec![0.0; 2];
        let mut cos_result = vec![0.0; 2];
        let mut tan_result = vec![0.0; 2];
        // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
        unsafe {
            Avx512Backend::sin(&a, &mut sin_result);
            Avx512Backend::cos(&a, &mut cos_result);
            Avx512Backend::tan(&a, &mut tan_result);
        }
        assert!((sin_result[0]).abs() < 1e-5);
        assert!((sin_result[1] - 1.0).abs() < 1e-5);
        assert!((cos_result[0] - 1.0).abs() < 1e-5);
    });
}

#[test]
fn test_avx512_rounding() {
    avx512_test(|| {
        let a = vec![1.3, 1.5, 1.7, -1.3, -1.5, -1.7];
        let mut floor_result = vec![0.0; 6];
        let mut ceil_result = vec![0.0; 6];
        let mut round_result = vec![0.0; 6];
        // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
        unsafe {
            Avx512Backend::floor(&a, &mut floor_result);
            Avx512Backend::ceil(&a, &mut ceil_result);
            Avx512Backend::round(&a, &mut round_result);
        }
        assert_eq!(floor_result, vec![1.0, 1.0, 1.0, -2.0, -2.0, -2.0]);
        assert_eq!(ceil_result, vec![2.0, 2.0, 2.0, -1.0, -1.0, -1.0]);
    });
}
