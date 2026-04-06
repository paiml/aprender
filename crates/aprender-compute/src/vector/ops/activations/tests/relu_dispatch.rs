use super::*;

// ========== Comprehensive Backend Dispatch Coverage ==========

#[test]
fn test_relu_scalar_backend_comprehensive() {
    let data = [-5.0, -1.0, -0.001, 0.0, 0.001, 1.0, 5.0];
    assert_activation_elementwise(
        &data,
        Backend::Scalar,
        act_relu,
        |x| x.max(0.0),
        1e-6,
        "relu Scalar",
    );
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_relu_sse2_backend_comprehensive() {
    let data: Vec<f32> = (-20..20).map(|i| i as f32 * 0.5).collect();
    assert_activation_elementwise(
        &data,
        Backend::SSE2,
        act_relu,
        |x| x.max(0.0),
        1e-6,
        "relu SSE2",
    );
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_relu_avx_backend_comprehensive() {
    let data: Vec<f32> = (-20..20).map(|i| i as f32 * 0.5).collect();
    assert_activation_elementwise(&data, Backend::AVX, act_relu, |x| x.max(0.0), 1e-6, "relu AVX");
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_relu_avx2_backend_comprehensive() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let data: Vec<f32> = (-16..17).map(|i| i as f32 * 0.3).collect();
    assert_activation_elementwise(
        &data,
        Backend::AVX2,
        act_relu,
        |x| x.max(0.0),
        1e-6,
        "relu AVX2",
    );
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_relu_avx512_backend_comprehensive() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    let data: Vec<f32> = (-17..18).map(|i| i as f32 * 0.2).collect();
    assert_activation_elementwise(
        &data,
        Backend::AVX512,
        act_relu,
        |x| x.max(0.0),
        1e-6,
        "relu AVX512",
    );
}

#[test]
fn test_relu_neon_fallback_backend() {
    let data = [-3.0, -1.0, 0.0, 1.0, 3.0];
    assert_activation_elementwise(
        &data,
        Backend::NEON,
        act_relu,
        |x| x.max(0.0),
        1e-6,
        "relu NEON",
    );
}

#[test]
fn test_relu_wasm_fallback_backend() {
    let data = [-3.0, -1.0, 0.0, 1.0, 3.0];
    assert_activation_elementwise(
        &data,
        Backend::WasmSIMD,
        act_relu,
        |x| x.max(0.0),
        1e-6,
        "relu WasmSIMD",
    );
}

#[test]
fn test_relu_gpu_backend() {
    let data = [-3.0, -1.0, 0.0, 1.0, 3.0];
    assert_activation_elementwise(&data, Backend::GPU, act_relu, |x| x.max(0.0), 1e-6, "relu GPU");
}

#[test]
fn test_relu_auto_backend() {
    let data = [-3.0, -1.0, 0.0, 1.0, 3.0];
    assert_activation_elementwise(
        &data,
        Backend::Auto,
        act_relu,
        |x| x.max(0.0),
        1e-6,
        "relu Auto",
    );
}

// ========== Backend Equivalence Across All Backends ==========

#[test]
#[cfg(target_arch = "x86_64")]
fn test_relu_backend_equivalence_comprehensive() {
    let data: Vec<f32> = (-50..50).map(|i| i as f32 * 0.37).collect();
    let scalar_v = Vector::from_slice_with_backend(&data, Backend::Scalar);
    let expected = scalar_v.relu().unwrap();

    for &backend in &[Backend::SSE2, Backend::AVX] {
        let v = Vector::from_slice_with_backend(&data, backend);
        let result = v.relu().unwrap();
        for (i, (&got, &exp)) in
            result.as_slice().iter().zip(expected.as_slice().iter()).enumerate()
        {
            assert!(
                (got - exp).abs() < 1e-6,
                "relu Scalar vs {backend:?} at [{i}]: {got} vs {exp}",
            );
        }
    }

    if is_x86_feature_detected!("avx2") {
        for &backend in &[Backend::AVX2, Backend::AVX512] {
            let v = Vector::from_slice_with_backend(&data, backend);
            let result = v.relu().unwrap();
            for (i, (&got, &exp)) in
                result.as_slice().iter().zip(expected.as_slice().iter()).enumerate()
            {
                assert!(
                    (got - exp).abs() < 1e-6,
                    "relu Scalar vs {backend:?} at [{i}]: {got} vs {exp}",
                );
            }
        }
    }

    for &backend in &[Backend::NEON, Backend::WasmSIMD, Backend::GPU, Backend::Auto] {
        let v = Vector::from_slice_with_backend(&data, backend);
        let result = v.relu().unwrap();
        for (i, (&got, &exp)) in
            result.as_slice().iter().zip(expected.as_slice().iter()).enumerate()
        {
            assert!(
                (got - exp).abs() < 1e-6,
                "relu Scalar vs {backend:?} at [{i}]: {got} vs {exp}",
            );
        }
    }
}

// ========== Non-Aligned Size Tests Per Backend ==========

#[test]
fn test_relu_non_aligned_sizes_per_backend() {
    let sizes = [1, 2, 3, 5, 7, 9, 11, 15, 17, 31, 33, 63, 65];
    let backends = [Backend::Scalar, Backend::NEON, Backend::WasmSIMD, Backend::GPU, Backend::Auto];

    for &size in &sizes {
        let data: Vec<f32> = (0..size).map(|i| (i as f32) - (size as f32 / 2.0)).collect();
        for &backend in &backends {
            let v = Vector::from_slice_with_backend(&data, backend);
            let result = v.relu().unwrap();
            assert_eq!(result.as_slice().len(), size, "relu {backend:?} size={size}");
            for (i, &val) in result.as_slice().iter().enumerate() {
                let exp = data[i].max(0.0);
                assert!(
                    (val - exp).abs() < 1e-6,
                    "relu {backend:?} size={size} [{i}]: {val} vs {exp}",
                );
            }
        }
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_relu_non_aligned_sizes_simd_backends() {
    let sizes = [1, 3, 5, 7, 9, 11, 15, 17, 31, 33, 63, 65];
    for &size in &sizes {
        let data: Vec<f32> = (0..size).map(|i| (i as f32) - (size as f32 / 2.0)).collect();

        for &backend in &[Backend::SSE2, Backend::AVX] {
            let v = Vector::from_slice_with_backend(&data, backend);
            let result = v.relu().unwrap();
            for (i, &val) in result.as_slice().iter().enumerate() {
                let exp = data[i].max(0.0);
                assert!(
                    (val - exp).abs() < 1e-6,
                    "relu {backend:?} size={size} [{i}]: {val} vs {exp}",
                );
            }
        }

        if is_x86_feature_detected!("avx2") {
            for &backend in &[Backend::AVX2, Backend::AVX512] {
                let v = Vector::from_slice_with_backend(&data, backend);
                let result = v.relu().unwrap();
                for (i, &val) in result.as_slice().iter().enumerate() {
                    let exp = data[i].max(0.0);
                    assert!(
                        (val - exp).abs() < 1e-6,
                        "relu {backend:?} size={size} [{i}]: {val} vs {exp}",
                    );
                }
            }
        }
    }
}

// ========== Edge Cases ==========

#[test]
fn test_relu_single_element_backends() {
    for &backend in
        &[Backend::Scalar, Backend::NEON, Backend::WasmSIMD, Backend::GPU, Backend::Auto]
    {
        let v_neg = Vector::from_slice_with_backend(&[-1.0], backend);
        assert!(
            (v_neg.relu().unwrap().as_slice()[0] - 0.0).abs() < 1e-6,
            "relu {backend:?} single neg"
        );

        let v_pos = Vector::from_slice_with_backend(&[1.0], backend);
        assert!(
            (v_pos.relu().unwrap().as_slice()[0] - 1.0).abs() < 1e-6,
            "relu {backend:?} single pos"
        );

        let v_zero = Vector::from_slice_with_backend(&[0.0], backend);
        assert!(
            (v_zero.relu().unwrap().as_slice()[0] - 0.0).abs() < 1e-6,
            "relu {backend:?} single zero"
        );
    }
}

#[test]
fn test_relu_special_float_values() {
    let data = [f32::INFINITY, f32::NEG_INFINITY, f32::MIN, f32::MAX, f32::EPSILON, -f32::EPSILON];
    let v = Vector::from_slice_with_backend(&data, Backend::Scalar);
    let result = v.relu().unwrap();
    assert_eq!(result.as_slice()[0], f32::INFINITY); // relu(+inf) = +inf
    assert_eq!(result.as_slice()[1], 0.0); // relu(-inf) = 0
    assert_eq!(result.as_slice()[2], 0.0); // relu(f32::MIN) = 0 (MIN is negative)
    assert_eq!(result.as_slice()[3], f32::MAX); // relu(f32::MAX) = MAX
    assert_eq!(result.as_slice()[4], f32::EPSILON); // relu(EPSILON) = EPSILON
    assert_eq!(result.as_slice()[5], 0.0); // relu(-EPSILON) = 0
}

#[test]
fn test_relu_nan_handling() {
    let data = [f32::NAN, -1.0, 1.0, f32::NAN];
    let v = Vector::from_slice_with_backend(&data, Backend::Scalar);
    let result = v.relu().unwrap();
    assert_eq!(result.as_slice()[1], 0.0);
    assert_eq!(result.as_slice()[2], 1.0);
}

// ========== Parallel Path (Refs CB-130) ==========

#[test]
#[cfg(feature = "parallel")]
fn test_relu_parallel_large_vector() {
    let size = 500_000;
    let data: Vec<f32> = (0..size).map(|i| (i as f32) - (size as f32 / 2.0)).collect();
    let v = Vector::from_slice_with_backend(&data, Backend::Scalar);
    let result = v.relu().unwrap();
    assert_eq!(result.as_slice().len(), size);
    for (i, &val) in result.as_slice().iter().enumerate() {
        let exp = data[i].max(0.0);
        assert!((val - exp).abs() < 1e-6, "relu parallel [{i}]: got {val} expected {exp}",);
    }
}

#[test]
#[cfg(feature = "parallel")]
fn test_relu_parallel_boundary() {
    let size = 499_999;
    let data: Vec<f32> = (0..size).map(|i| (i as f32) - (size as f32 / 2.0)).collect();
    let v = Vector::from_slice_with_backend(&data, Backend::Scalar);
    let result = v.relu().unwrap();
    assert_eq!(result.as_slice().len(), size);

    let size = 500_000;
    let data: Vec<f32> = (0..size).map(|i| (i as f32) - (size as f32 / 2.0)).collect();
    let v = Vector::from_slice_with_backend(&data, Backend::Scalar);
    let result = v.relu().unwrap();
    assert_eq!(result.as_slice().len(), size);
}

#[test]
#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
fn test_relu_parallel_with_simd_backends() {
    let size = 600_000;
    let data: Vec<f32> = (0..size).map(|i| ((i % 100) as f32) - 50.0).collect();

    let v_scalar = Vector::from_slice_with_backend(&data, Backend::Scalar);
    let expected = v_scalar.relu().unwrap();

    let v_sse = Vector::from_slice_with_backend(&data, Backend::SSE2);
    let result_sse = v_sse.relu().unwrap();
    for (i, (&got, &exp)) in
        result_sse.as_slice().iter().zip(expected.as_slice().iter()).enumerate().take(100)
    {
        assert!((got - exp).abs() < 1e-6, "relu parallel Scalar vs SSE2 [{i}]: {got} vs {exp}",);
    }

    if is_x86_feature_detected!("avx2") {
        let v_avx2 = Vector::from_slice_with_backend(&data, Backend::AVX2);
        let result_avx2 = v_avx2.relu().unwrap();
        for (i, (&got, &exp)) in
            result_avx2.as_slice().iter().zip(expected.as_slice().iter()).enumerate().take(100)
        {
            assert!((got - exp).abs() < 1e-6, "relu parallel Scalar vs AVX2 [{i}]: {got} vs {exp}",);
        }
    }
}

#[test]
#[cfg(feature = "parallel")]
fn test_relu_parallel_all_negative() {
    let size = 500_000;
    let data = vec![-1.0; size];
    let v = Vector::from_slice_with_backend(&data, Backend::Scalar);
    let result = v.relu().unwrap();
    for &val in result.as_slice() {
        assert!((val - 0.0).abs() < 1e-6);
    }
}

#[test]
#[cfg(feature = "parallel")]
fn test_relu_parallel_all_positive() {
    let size = 500_000;
    let data = vec![42.0; size];
    let v = Vector::from_slice_with_backend(&data, Backend::Scalar);
    let result = v.relu().unwrap();
    for &val in result.as_slice() {
        assert!((val - 42.0).abs() < 1e-6);
    }
}
