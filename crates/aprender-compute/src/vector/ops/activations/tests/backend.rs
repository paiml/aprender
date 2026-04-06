use super::*;

#[test]
fn test_all_activations_scalar_backend() {
    for (act_fn, label, zero_out) in activation_specs() {
        assert_activation_at(&[0.0], Backend::Scalar, act_fn, 0, zero_out, 1e-5, label);
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_all_activations_sse2_backend() {
    for (act_fn, label, zero_out) in activation_specs() {
        assert_activation_at(
            &[-1.0, 0.0, 1.0, 2.0],
            Backend::SSE2,
            act_fn,
            1,
            zero_out,
            1e-5,
            label,
        );
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_all_activations_avx2_backend() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let data: Vec<f32> = (-8..8).map(|i| i as f32).collect();
    for (act_fn, label, zero_out) in activation_specs() {
        assert_activation_at(&data, Backend::AVX2, act_fn, 8, zero_out, 1e-5, label);
    }
}

#[test]
fn test_all_activations_fallback_backends() {
    for (act_fn, label, zero_out) in activation_specs() {
        for backend in [Backend::NEON, Backend::WasmSIMD, Backend::GPU, Backend::Auto] {
            assert_activation_at(&[0.0, 1.0], backend, act_fn, 0, zero_out, 1e-5, label);
        }
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_all_activations_avx512_backend() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    let data: Vec<f32> = (-8..8).map(|i| i as f32 * 0.5).collect();
    for (act_fn, label, zero_out) in activation_specs() {
        assert_activation_at(&data, Backend::AVX512, act_fn, 8, zero_out, 1e-5, label);
    }
}

#[test]
fn test_all_activations_backend_equivalence() {
    let tolerances = [1e-5, 1e-2, 1e-2, 1e-2];
    for ((act_fn, label, _), &tol) in activation_specs().iter().zip(tolerances.iter()) {
        let data: Vec<f32> = (-20..20).map(|i| i as f32 * 0.5).collect();
        #[cfg(target_arch = "x86_64")]
        assert_backend_equivalence(&data, *act_fn, tol, label);
        #[cfg(not(target_arch = "x86_64"))]
        let _ = (data, act_fn, tol, label);
    }
}

#[test]
fn test_all_activations_non_aligned_sizes() {
    let sizes = [1, 3, 5, 7, 9, 13, 15, 17, 31, 33];
    for (act_fn, label, _) in activation_specs() {
        for &size in &sizes {
            let data: Vec<f32> = (0..size).map(|i| (i as f32) - (size as f32 / 2.0)).collect();
            let v = Vector::from_slice(&data);
            let result = act_fn(&v).unwrap();
            assert_eq!(result.as_slice().len(), size, "{label} non-aligned size={size}");
        }
    }
}

#[test]
fn test_relu_elementwise_avx2() {
    #[cfg(target_arch = "x86_64")]
    {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let data: Vec<f32> = (-16..16).map(|i| i as f32).collect();
        assert_activation_elementwise(&data, Backend::AVX2, act_relu, |x| x.max(0.0), 1e-6, "relu");
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_relu_avx512_non_aligned() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    for size in [17, 19, 23, 31, 33] {
        let data: Vec<f32> = (0..size).map(|i| (i as f32) - (size as f32 / 2.0)).collect();
        assert_activation_elementwise(
            &data,
            Backend::AVX512,
            act_relu,
            |x| x.max(0.0),
            1e-6,
            "relu AVX512",
        );
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_sigmoid_avx2_range() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let data: Vec<f32> = (-8..8).map(|i| i as f32).collect();
    assert_activation_in_range(&data, Backend::AVX2, act_sigmoid, 0.0, 1.0, "sigmoid");
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_sigmoid_avx512_range() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    let data: Vec<f32> = (-16..16).map(|i| i as f32 * 0.5).collect();
    assert_activation_in_range(&data, Backend::AVX512, act_sigmoid, 0.0, 1.0, "sigmoid");
}
