use super::*;

#[test]
fn test_avx512_exp_large() {
    avx512_test(|| {
        let input: Vec<f32> = (0..48).map(|i| i as f32 * 0.1).collect();
        assert_unary_large_relative(input, 0.05, Avx512Backend::exp, f32::exp, "exp");
    });
}

#[test]
fn test_avx512_exp_non_aligned() {
    avx512_test(|| {
        for size in [17, 19, 23, 31, 33] {
            let input: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
            assert_unary_large_relative(
                input,
                0.05,
                Avx512Backend::exp,
                f32::exp,
                &format!("exp non-aligned size={size}"),
            );
        }
    });
}

#[test]
fn test_avx512_relu_large() {
    avx512_test(|| {
        let input: Vec<f32> = (-24..24).map(|i| i as f32).collect();
        assert_unary_large(input, 1e-6, Avx512Backend::relu, |x| x.max(0.0), "relu");
    });
}

#[test]
fn test_avx512_tanh_large() {
    avx512_test(|| {
        let input: Vec<f32> = (-24..24).map(|i| i as f32 * 0.2).collect();
        assert_unary_large(input, 1e-3, Avx512Backend::tanh, f32::tanh, "tanh");
    });
}

#[test]
fn test_avx512_sigmoid_large() {
    avx512_test(|| {
        let a: Vec<f32> = (-16..16).map(|i| i as f32 * 0.5).collect();
        let mut result = vec![0.0; 32];
        // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
        unsafe {
            Avx512Backend::sigmoid(&a, &mut result);
        }
        for (i, &val) in result.iter().enumerate() {
            assert!((0.0..=1.0).contains(&val), "sigmoid out of range at {i}: {val}");
            let expected = 1.0 / (1.0 + (-a[i]).exp());
            assert!(
                (val - expected).abs() < 1e-4,
                "sigmoid large mismatch at {i}: {val} vs {expected}"
            );
        }
    });
}

#[test]
fn test_avx512_gelu_large() {
    avx512_test(|| {
        let a: Vec<f32> = (-16..16).map(|i| i as f32 * 0.3).collect();
        let mut result = vec![0.0; 32];
        // SAFETY: test-only; result matches input length
        unsafe { Avx512Backend::gelu(&a, &mut result) };
        assert!((result[16]).abs() < 1e-4, "gelu(0) should be 0, got {}", result[16]);
    });
}

#[test]
fn test_avx512_swish_large() {
    avx512_test(|| {
        let a: Vec<f32> = (-16..16).map(|i| i as f32 * 0.3).collect();
        let mut result = vec![0.0; 32];
        // SAFETY: test-only; result matches input length
        unsafe { Avx512Backend::swish(&a, &mut result) };
        assert!((result[16]).abs() < 1e-4, "swish(0) should be 0, got {}", result[16]);
    });
}

#[test]
fn test_avx512_backend_equivalence() {
    avx512_test(|| {
        let a: Vec<f32> = (0..100).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..100).map(|i| (100 - i) as f32 * 0.1).collect();
        let mut avx512_add = vec![0.0; 100];
        let mut scalar_add = vec![0.0; 100];
        // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
        unsafe {
            Avx512Backend::add(&a, &b, &mut avx512_add);
            ScalarBackend::add(&a, &b, &mut scalar_add);
        }
        for i in 0..100 {
            assert!((avx512_add[i] - scalar_add[i]).abs() < 1e-5, "add mismatch at {i}",);
        }
    });
}
