use crate::vector::Vector;
use crate::{Backend, TruenoError};

// ========== Exponential Functions ==========

#[test]
fn test_exp_basic() {
    let v = Vector::from_slice(&[0.0, 1.0, 2.0]);
    let result = v.exp().unwrap();
    assert!((result.as_slice()[0] - 1.0).abs() < 1e-6); // e^0 = 1
    assert!((result.as_slice()[1] - std::f32::consts::E).abs() < 1e-5); // e^1 = e
    assert!((result.as_slice()[2] - std::f32::consts::E.powi(2)).abs() < 1e-4);
}

#[test]
fn test_exp_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    let result = v.exp().unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_exp_negative() {
    let v = Vector::from_slice(&[-1.0, -2.0]);
    let result = v.exp().unwrap();
    assert!((result.as_slice()[0] - 1.0 / std::f32::consts::E).abs() < 1e-5);
}

#[test]
fn test_ln_basic() {
    let v = Vector::from_slice(&[1.0, std::f32::consts::E, std::f32::consts::E.powi(2)]);
    let result = v.ln().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
    assert!((result.as_slice()[1] - 1.0).abs() < 1e-5);
    assert!((result.as_slice()[2] - 2.0).abs() < 1e-5);
}

#[test]
fn test_ln_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    let result = v.ln().unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_log2_basic() {
    let v = Vector::from_slice(&[1.0, 2.0, 4.0, 8.0]);
    let result = v.log2().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
    assert!((result.as_slice()[1] - 1.0).abs() < 1e-6);
    assert!((result.as_slice()[2] - 2.0).abs() < 1e-6);
    assert!((result.as_slice()[3] - 3.0).abs() < 1e-6);
}

#[test]
fn test_log2_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    let result = v.log2().unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_log10_basic() {
    let v = Vector::from_slice(&[1.0, 10.0, 100.0, 1000.0]);
    let result = v.log10().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
    assert!((result.as_slice()[1] - 1.0).abs() < 1e-5);
    assert!((result.as_slice()[2] - 2.0).abs() < 1e-5);
    assert!((result.as_slice()[3] - 3.0).abs() < 1e-4);
}

#[test]
fn test_log10_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    let result = v.log10().unwrap();
    assert!(result.is_empty());
}

// ========== Trigonometric Functions ==========

#[test]
fn test_sin_basic() {
    let v = Vector::from_slice(&[0.0, std::f32::consts::PI / 2.0, std::f32::consts::PI]);
    let result = v.sin().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
    assert!((result.as_slice()[1] - 1.0).abs() < 1e-6);
    assert!((result.as_slice()[2] - 0.0).abs() < 1e-5);
}

#[test]
fn test_sin_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    let result = v.sin().unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_cos_basic() {
    let v = Vector::from_slice(&[0.0, std::f32::consts::PI / 2.0, std::f32::consts::PI]);
    let result = v.cos().unwrap();
    assert!((result.as_slice()[0] - 1.0).abs() < 1e-6);
    assert!((result.as_slice()[1] - 0.0).abs() < 1e-6);
    assert!((result.as_slice()[2] - (-1.0)).abs() < 1e-5);
}

#[test]
fn test_cos_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    let result = v.cos().unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_tan_basic() {
    let v = Vector::from_slice(&[0.0, std::f32::consts::PI / 4.0]);
    let result = v.tan().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
    assert!((result.as_slice()[1] - 1.0).abs() < 1e-5);
}

#[test]
fn test_tan_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    let result = v.tan().unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_asin_basic() {
    let v = Vector::from_slice(&[0.0, 0.5, 1.0]);
    let result = v.asin().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
    assert!((result.as_slice()[1] - std::f32::consts::FRAC_PI_6).abs() < 1e-3);
    assert!((result.as_slice()[2] - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
}

#[test]
fn test_asin_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    let result = v.asin().unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_acos_basic() {
    let v = Vector::from_slice(&[1.0, 0.5, 0.0]);
    let result = v.acos().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
    assert!((result.as_slice()[1] - std::f32::consts::FRAC_PI_3).abs() < 1e-3);
    assert!((result.as_slice()[2] - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
}

#[test]
fn test_acos_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    let result = v.acos().unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_atan_basic() {
    let v = Vector::from_slice(&[0.0, 1.0, -1.0]);
    let result = v.atan().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
    assert!((result.as_slice()[1] - std::f32::consts::FRAC_PI_4).abs() < 1e-5);
    assert!((result.as_slice()[2] - (-std::f32::consts::FRAC_PI_4)).abs() < 1e-5);
}

#[test]
fn test_atan_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    let result = v.atan().unwrap();
    assert!(result.is_empty());
}

// ========== Hyperbolic Functions ==========

#[test]
fn test_sinh_basic() {
    let v = Vector::from_slice(&[0.0, 1.0, -1.0]);
    let result = v.sinh().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
    assert!((result.as_slice()[1] - 1.1752).abs() < 1e-3);
    assert!((result.as_slice()[2] - (-1.1752)).abs() < 1e-3);
}

#[test]
fn test_sinh_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    let result = v.sinh().unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_cosh_basic() {
    let v = Vector::from_slice(&[0.0, 1.0, -1.0]);
    let result = v.cosh().unwrap();
    assert!((result.as_slice()[0] - 1.0).abs() < 1e-6);
    assert!((result.as_slice()[1] - 1.5431).abs() < 1e-3);
    assert!((result.as_slice()[2] - 1.5431).abs() < 1e-3); // cosh is even
}

#[test]
fn test_cosh_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    let result = v.cosh().unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_tanh_basic() {
    let v = Vector::from_slice(&[0.0, 1.0, -1.0]);
    let result = v.tanh().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
    assert!((result.as_slice()[1] - 0.7616).abs() < 1e-3);
    assert!((result.as_slice()[2] - (-0.7616)).abs() < 1e-3);
}

#[test]
fn test_tanh_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    // tanh returns EmptyVector error for empty input
    assert!(matches!(v.tanh(), Err(TruenoError::EmptyVector)));
}

#[test]
fn test_asinh_basic() {
    let v = Vector::from_slice(&[0.0, 1.0, -1.0]);
    let result = v.asinh().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
    assert!((result.as_slice()[1] - 0.8814).abs() < 1e-3);
    assert!((result.as_slice()[2] - (-0.8814)).abs() < 1e-3);
}

#[test]
fn test_asinh_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    let result = v.asinh().unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_acosh_basic() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = v.acosh().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
    assert!((result.as_slice()[1] - 1.3170).abs() < 1e-3);
    assert!((result.as_slice()[2] - 1.7627).abs() < 1e-3);
}

#[test]
fn test_acosh_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    let result = v.acosh().unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_atanh_basic() {
    let v = Vector::from_slice(&[0.0, 0.5, -0.5]);
    let result = v.atanh().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
    assert!((result.as_slice()[1] - 0.5493).abs() < 1e-3);
    assert!((result.as_slice()[2] - (-0.5493)).abs() < 1e-3);
}

#[test]
fn test_atanh_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    let result = v.atanh().unwrap();
    assert!(result.is_empty());
}

// ========== Backend-specific Tests ==========

#[test]
fn test_exp_scalar_backend() {
    let v = Vector::from_slice_with_backend(&[0.0, 1.0, 2.0], Backend::Scalar);
    let result = v.exp().unwrap();
    assert!((result.as_slice()[0] - 1.0).abs() < 1e-6);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_exp_sse2_backend() {
    let v = Vector::from_slice_with_backend(&[0.0, 1.0, 2.0, 3.0], Backend::SSE2);
    let result = v.exp().unwrap();
    assert!((result.as_slice()[0] - 1.0).abs() < 1e-6);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_exp_avx2_backend() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let v = Vector::from_slice_with_backend(&[0.0; 16], Backend::AVX2);
    let result = v.exp().unwrap();
    for val in result.as_slice() {
        assert!((val - 1.0).abs() < 1e-6);
    }
}

#[test]
fn test_sin_scalar_backend() {
    let v = Vector::from_slice_with_backend(&[0.0, std::f32::consts::FRAC_PI_2], Backend::Scalar);
    let result = v.sin().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
    assert!((result.as_slice()[1] - 1.0).abs() < 1e-6);
}

#[test]
fn test_cos_scalar_backend() {
    let v = Vector::from_slice_with_backend(&[0.0, std::f32::consts::PI], Backend::Scalar);
    let result = v.cos().unwrap();
    assert!((result.as_slice()[0] - 1.0).abs() < 1e-6);
    assert!((result.as_slice()[1] - (-1.0)).abs() < 1e-5);
}

// ========== Large Array Tests ==========

#[test]
fn test_exp_large() {
    let v = Vector::from_slice(&[1.0; 1000]);
    let result = v.exp().unwrap();
    assert_eq!(result.len(), 1000);
    for val in result.as_slice() {
        assert!((val - std::f32::consts::E).abs() < 1e-5);
    }
}

#[test]
fn test_sin_large() {
    let v = Vector::from_slice(&[0.0; 1000]);
    let result = v.sin().unwrap();
    assert_eq!(result.len(), 1000);
    for val in result.as_slice() {
        assert!((val - 0.0).abs() < 1e-6);
    }
}

// ========== Inverse Relationship Tests ==========

#[test]
fn test_exp_ln_inverse() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let exp_result = v.exp().unwrap();
    let roundtrip = exp_result.ln().unwrap();
    for (orig, rt) in v.as_slice().iter().zip(roundtrip.as_slice()) {
        assert!((orig - rt).abs() < 1e-5);
    }
}

#[test]
fn test_sin_asin_inverse() {
    let v = Vector::from_slice(&[0.0, 0.3, 0.5, 0.7]);
    let sin_result = v.sin().unwrap();
    let roundtrip = sin_result.asin().unwrap();
    for (orig, rt) in v.as_slice().iter().zip(roundtrip.as_slice()) {
        assert!((orig - rt).abs() < 1e-5);
    }
}

#[test]
fn test_cos_acos_inverse() {
    let v = Vector::from_slice(&[0.0, 0.3, 0.5, 0.7]);
    let cos_result = v.cos().unwrap();
    let roundtrip = cos_result.acos().unwrap();
    for (orig, rt) in v.as_slice().iter().zip(roundtrip.as_slice()) {
        assert!((orig - rt).abs() < 1e-5);
    }
}

#[test]
fn test_sinh_asinh_inverse() {
    let v = Vector::from_slice(&[0.0, 1.0, 2.0, -1.0, -2.0]);
    let sinh_result = v.sinh().unwrap();
    let roundtrip = sinh_result.asinh().unwrap();
    for (orig, rt) in v.as_slice().iter().zip(roundtrip.as_slice()) {
        assert!((orig - rt).abs() < 1e-4);
    }
}

#[test]
fn test_tanh_atanh_inverse() {
    let v = Vector::from_slice(&[0.0, 0.3, 0.5, -0.3, -0.5]);
    let tanh_result = v.tanh().unwrap();
    let roundtrip = tanh_result.atanh().unwrap();
    for (orig, rt) in v.as_slice().iter().zip(roundtrip.as_slice()) {
        assert!((orig - rt).abs() < 1e-4);
    }
}

// ========== Backend Dispatch Coverage for tanh ==========

#[test]
fn test_tanh_scalar_backend() {
    let v = Vector::from_slice_with_backend(&[0.0, 1.0, -1.0], Backend::Scalar);
    let result = v.tanh().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
    assert!((result.as_slice()[1] - 0.7616).abs() < 1e-3);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_tanh_sse2_backend() {
    let v = Vector::from_slice_with_backend(&[0.0, 1.0, -1.0, 2.0], Backend::SSE2);
    let result = v.tanh().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
    assert!((result.as_slice()[1] - 0.7616).abs() < 1e-3);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_tanh_avx2_backend() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let data: Vec<f32> = (-16..16).map(|i| i as f32 * 0.25).collect();
    let v = Vector::from_slice_with_backend(&data, Backend::AVX2);
    let result = v.tanh().unwrap();
    for &val in result.as_slice() {
        assert!((-1.0..=1.0).contains(&val), "tanh AVX2 out of range: {}", val);
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_tanh_avx512_backend() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    // Use 32+ elements to exercise the main SIMD loop (16 elements per iteration)
    let data: Vec<f32> = (-20..20).map(|i| i as f32 * 0.25).collect();
    let v = Vector::from_slice_with_backend(&data, Backend::AVX512);
    let result = v.tanh().unwrap();
    for (i, &val) in result.as_slice().iter().enumerate() {
        assert!((-1.0..=1.0).contains(&val), "tanh AVX512 out of range at {}: {}", i, val);
        let expected = data[i].tanh();
        assert!(
            (val - expected).abs() < 1e-3,
            "tanh AVX512 mismatch at {}: {} vs {}",
            i,
            val,
            expected
        );
    }
}

#[test]
fn test_tanh_neon_backend_fallback() {
    let v = Vector::from_slice_with_backend(&[0.0, 1.0, -1.0], Backend::NEON);
    let result = v.tanh().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
}

#[test]
fn test_tanh_wasm_backend_fallback() {
    let v = Vector::from_slice_with_backend(&[0.0, 1.0], Backend::WasmSIMD);
    let result = v.tanh().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
}

#[test]
fn test_tanh_gpu_backend_fallback() {
    let v = Vector::from_slice_with_backend(&[0.0, 1.0], Backend::GPU);
    let result = v.tanh().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
}

#[test]
fn test_tanh_auto_backend_fallback() {
    let v = Vector::from_slice_with_backend(&[0.0, 1.0], Backend::Auto);
    let result = v.tanh().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
}

// ========== Exp backend dispatch coverage ==========

#[test]
#[cfg(target_arch = "x86_64")]
fn test_exp_avx512_backend() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    // 32 elements = 2 full AVX-512 iterations (16 f32s each)
    let data: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
    let v = Vector::from_slice_with_backend(&data, Backend::AVX512);
    let result = v.exp().unwrap();
    for (i, &val) in result.as_slice().iter().enumerate() {
        let expected = data[i].exp();
        // AVX-512 exp uses polynomial approximation — allow 5% relative error
        assert!(
            (val - expected).abs() / expected.max(1e-6) < 0.05,
            "exp AVX512 mismatch at {}: {} vs {}",
            i,
            val,
            expected
        );
    }
}

// ========== Non-aligned AVX-512 remainder handling ==========

#[test]
#[cfg(target_arch = "x86_64")]
fn test_tanh_avx512_non_aligned() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    // 19 elements: 16 in SIMD loop + 3 remainder
    for size in [17, 19, 23, 31, 33] {
        let data: Vec<f32> = (0..size).map(|i| (i as f32 - size as f32 / 2.0) * 0.2).collect();
        let v = Vector::from_slice_with_backend(&data, Backend::AVX512);
        let result = v.tanh().unwrap();
        for (i, &val) in result.as_slice().iter().enumerate() {
            let expected = data[i].tanh();
            assert!(
                (val - expected).abs() < 1e-3,
                "tanh AVX512 non-aligned size={} mismatch at {}: {} vs {}",
                size,
                i,
                val,
                expected
            );
        }
    }
}
