use super::*;

#[test]
fn test_abs_basic() {
    let v = Vector::from_slice(&[3.0, -4.0, 5.0, -2.0]);
    let result = v.abs().unwrap();
    assert_eq!(result.as_slice(), &[3.0, 4.0, 5.0, 2.0]);
}

#[test]
fn test_abs_empty() {
    let v: Vector<f32> = Vector::from_slice(&[]);
    let result = v.abs().unwrap();
    assert_eq!(result.len(), 0);
}

#[test]
fn test_clip_basic() {
    let v = Vector::from_slice(&[-5.0, 0.0, 5.0, 10.0, 15.0]);
    let clipped = v.clip(0.0, 10.0).unwrap();
    assert_eq!(clipped.as_slice(), &[0.0, 0.0, 5.0, 10.0, 10.0]);
}

#[test]
fn test_clip_invalid_range() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = v.clip(10.0, 5.0);
    assert!(matches!(result, Err(TruenoError::InvalidInput(_))));
}

#[test]
fn test_clamp_basic() {
    let v = Vector::from_slice(&[-5.0, 0.0, 5.0, 10.0, 15.0]);
    let result = v.clamp(0.0, 10.0).unwrap();
    assert_eq!(result.as_slice(), &[0.0, 0.0, 5.0, 10.0, 10.0]);
}

#[test]
fn test_clamp_negative_range() {
    let v = Vector::from_slice(&[-10.0, -5.0, 0.0, 5.0]);
    let result = v.clamp(-8.0, -2.0).unwrap();
    assert_eq!(result.as_slice(), &[-8.0, -5.0, -2.0, -2.0]);
}

#[test]
fn test_lerp_midpoint() {
    let a = Vector::from_slice(&[0.0, 10.0, 20.0]);
    let b = Vector::from_slice(&[100.0, 110.0, 120.0]);
    let result = a.lerp(&b, 0.5).unwrap();
    assert_eq!(result.as_slice(), &[50.0, 60.0, 70.0]);
}

#[test]
fn test_lerp_extrapolation() {
    let a = Vector::from_slice(&[0.0, 10.0]);
    let b = Vector::from_slice(&[10.0, 20.0]);
    let result = a.lerp(&b, 2.0).unwrap();
    assert_eq!(result.as_slice(), &[20.0, 30.0]);
}

#[test]
fn test_lerp_size_mismatch() {
    let a = Vector::from_slice(&[0.0, 10.0]);
    let b = Vector::from_slice(&[10.0, 20.0, 30.0]);
    let result = a.lerp(&b, 0.5);
    assert!(matches!(result, Err(TruenoError::SizeMismatch { .. })));
}

#[test]
fn test_sqrt_basic() {
    let a = Vector::from_slice(&[4.0, 9.0, 16.0, 25.0]);
    let result = a.sqrt().unwrap();
    assert_eq!(result.as_slice(), &[2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_sqrt_negative() {
    let a = Vector::from_slice(&[-1.0, 4.0]);
    let result = a.sqrt().unwrap();
    assert!(result.as_slice()[0].is_nan());
    assert_eq!(result.as_slice()[1], 2.0);
}

#[test]
fn test_recip_basic() {
    let a = Vector::from_slice(&[2.0, 4.0, 5.0, 10.0]);
    let result = a.recip().unwrap();
    assert_eq!(result.as_slice(), &[0.5, 0.25, 0.2, 0.1]);
}

#[test]
fn test_recip_zero() {
    let a = Vector::from_slice(&[0.0, 2.0]);
    let result = a.recip().unwrap();
    assert!(result.as_slice()[0].is_infinite());
    assert_eq!(result.as_slice()[1], 0.5);
}

#[test]
fn test_pow_squared() {
    let v = Vector::from_slice(&[2.0, 3.0, 4.0]);
    let squared = v.pow(2.0).unwrap();
    assert_eq!(squared.as_slice(), &[4.0, 9.0, 16.0]);
}

#[test]
fn test_pow_square_root() {
    let v = Vector::from_slice(&[4.0, 9.0, 16.0]);
    let sqrt = v.pow(0.5).unwrap();
    assert!((sqrt.as_slice()[0] - 2.0).abs() < 1e-5);
    assert!((sqrt.as_slice()[1] - 3.0).abs() < 1e-5);
    assert!((sqrt.as_slice()[2] - 4.0).abs() < 1e-5);
}

// =====================================================================
// Coverage: GPU and Auto backend error paths
// =====================================================================

#[test]
fn test_abs_gpu_backend_returns_error() {
    let v = Vector::from_slice_with_backend(&[1.0, -2.0, 3.0], Backend::GPU);
    let result = v.abs();
    assert!(matches!(result, Err(TruenoError::UnsupportedBackend(Backend::GPU))));
}

#[test]
fn test_abs_auto_backend_returns_error() {
    // Auto is resolved at construction, but we can test the error path
    // by using from_slice_with_backend which resolves Auto to best available.
    // For the GPU path we already tested above. Let's ensure Scalar path works.
    let v = Vector::from_slice_with_backend(&[3.0, -4.0, 5.0], Backend::Scalar);
    let result = v.abs().unwrap();
    assert_eq!(result.as_slice(), &[3.0, 4.0, 5.0]);
}

#[test]
fn test_clamp_gpu_backend_returns_error() {
    let v = Vector::from_slice_with_backend(&[1.0, 2.0, 3.0], Backend::GPU);
    let result = v.clamp(0.0, 2.0);
    assert!(matches!(result, Err(TruenoError::UnsupportedBackend(Backend::GPU))));
}

#[test]
fn test_clamp_scalar_backend() {
    let v = Vector::from_slice_with_backend(&[-5.0, 0.0, 5.0, 10.0], Backend::Scalar);
    let result = v.clamp(0.0, 8.0).unwrap();
    assert_eq!(result.as_slice(), &[0.0, 0.0, 5.0, 8.0]);
}

#[test]
fn test_clamp_invalid_range() {
    let v = Vector::from_slice(&[1.0, 2.0]);
    let result = v.clamp(10.0, 5.0);
    assert!(matches!(result, Err(TruenoError::InvalidInput(_))));
}

#[test]
fn test_clamp_empty() {
    let v: Vector<f32> = Vector::from_slice(&[]);
    let result = v.clamp(0.0, 1.0).unwrap();
    assert_eq!(result.len(), 0);
}

#[test]
fn test_lerp_gpu_backend_returns_error() {
    let a = Vector::from_slice_with_backend(&[1.0, 2.0], Backend::GPU);
    let b = Vector::from_slice_with_backend(&[3.0, 4.0], Backend::GPU);
    let result = a.lerp(&b, 0.5);
    assert!(matches!(result, Err(TruenoError::UnsupportedBackend(Backend::GPU))));
}

#[test]
fn test_lerp_scalar_backend() {
    let a = Vector::from_slice_with_backend(&[0.0, 10.0], Backend::Scalar);
    let b = Vector::from_slice_with_backend(&[10.0, 20.0], Backend::Scalar);
    let result = a.lerp(&b, 0.5).unwrap();
    assert_eq!(result.as_slice(), &[5.0, 15.0]);
}

#[test]
fn test_lerp_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let b: Vector<f32> = Vector::from_slice(&[]);
    let result = a.lerp(&b, 0.5).unwrap();
    assert_eq!(result.len(), 0);
}

#[test]
fn test_sqrt_empty() {
    let v: Vector<f32> = Vector::from_slice(&[]);
    let result = v.sqrt().unwrap();
    assert_eq!(result.len(), 0);
}

#[test]
fn test_recip_empty() {
    let v: Vector<f32> = Vector::from_slice(&[]);
    let result = v.recip().unwrap();
    assert_eq!(result.len(), 0);
}

#[test]
fn test_pow_zero_exponent() {
    let v = Vector::from_slice(&[2.0, 0.0, -3.0]);
    let result = v.pow(0.0).unwrap();
    // x^0 = 1.0 for all x
    assert_eq!(result.as_slice(), &[1.0, 1.0, 1.0]);
}

#[test]
fn test_pow_empty() {
    let v: Vector<f32> = Vector::from_slice(&[]);
    let result = v.pow(2.0).unwrap();
    assert_eq!(result.len(), 0);
}

// =====================================================================
// Coverage: sqrt with explicit backend dispatch paths (Refs CB-130)
// =====================================================================

/// Helper to verify sqrt results against scalar reference for a given backend.
fn assert_sqrt_backend_eq(data: &[f32], backend: Backend) {
    let v = Vector::from_slice_with_backend(data, backend);
    let result = v.sqrt().unwrap();
    assert_eq!(result.len(), data.len());
    for (i, (&input, &output)) in data.iter().zip(result.as_slice().iter()).enumerate() {
        let expected = input.sqrt();
        if expected.is_nan() {
            assert!(output.is_nan(), "index {i}: expected NaN, got {output}");
        } else if expected.is_infinite() {
            assert!(
                output.is_infinite() && output.is_sign_positive() == expected.is_sign_positive(),
                "index {i}: expected {expected}, got {output} (backend: {backend:?})"
            );
        } else {
            assert!(
                (output - expected).abs() < 1e-6,
                "index {i}: expected {expected}, got {output} (backend: {backend:?})"
            );
        }
    }
}

#[test]
fn test_sqrt_scalar_backend_perfect_squares() {
    assert_sqrt_backend_eq(&[0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0], Backend::Scalar);
}

#[test]
fn test_sqrt_scalar_backend_non_aligned() {
    // 5 elements: not a multiple of 4 or 8 — exercises scalar remainder
    assert_sqrt_backend_eq(&[2.0, 3.0, 5.0, 7.0, 11.0], Backend::Scalar);
}

#[test]
fn test_sqrt_scalar_backend_edge_cases() {
    assert_sqrt_backend_eq(
        &[0.0, f32::INFINITY, f32::MIN_POSITIVE, 1e-30, 1e30, -1.0],
        Backend::Scalar,
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_sqrt_sse2_backend_aligned() {
    // 8 elements: exactly 2 SSE2 chunks (4 elements each), no remainder
    assert_sqrt_backend_eq(&[1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0], Backend::SSE2);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_sqrt_sse2_backend_remainder() {
    // 11 elements: 2 full SSE2 chunks (8 elements) + 3 remainder
    assert_sqrt_backend_eq(
        &[1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0, 121.0],
        Backend::SSE2,
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_sqrt_sse2_backend_single_element() {
    assert_sqrt_backend_eq(&[144.0], Backend::SSE2);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_sqrt_avx2_backend_aligned() {
    // 16 elements: exactly 2 AVX2 chunks (8 elements each)
    let data: Vec<f32> = (1..=16).map(|x| (x * x) as f32).collect();
    assert_sqrt_backend_eq(&data, Backend::AVX2);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_sqrt_avx2_backend_remainder() {
    // 13 elements: 1 full AVX2 chunk (8) + 5 remainder
    let data: Vec<f32> = (1..=13).map(|x| (x * x) as f32).collect();
    assert_sqrt_backend_eq(&data, Backend::AVX2);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_sqrt_avx2_backend_sub_chunk() {
    // 7 elements: 0 AVX2 chunks, all remainder
    let data: Vec<f32> = (1..=7).map(|x| (x * x) as f32).collect();
    assert_sqrt_backend_eq(&data, Backend::AVX2);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_sqrt_avx2_backend_edge_cases() {
    // Mix of edge cases: exactly 8 elements for 1 AVX2 chunk
    assert_sqrt_backend_eq(
        &[0.0, f32::MIN_POSITIVE, 1e-38, 1.0, 100.0, 1e10, f32::INFINITY, -4.0],
        Backend::AVX2,
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_sqrt_avx_backend() {
    // AVX backend dispatches to SSE2 in the dispatch macro
    assert_sqrt_backend_eq(&[4.0, 9.0, 16.0, 25.0, 36.0], Backend::AVX);
}

#[test]
fn test_sqrt_neon_backend_fallback() {
    // NEON on non-ARM falls back to scalar
    assert_sqrt_backend_eq(&[4.0, 9.0, 16.0, 25.0], Backend::NEON);
}

#[test]
fn test_sqrt_wasm_simd_backend_fallback() {
    // WasmSIMD on non-WASM falls back to scalar
    assert_sqrt_backend_eq(&[4.0, 9.0, 16.0, 25.0], Backend::WasmSIMD);
}

#[test]
fn test_sqrt_large_vector() {
    // 256 elements: exercises multiple AVX2 chunks (32 chunks of 8)
    let data: Vec<f32> = (1..=256).map(|x| x as f32).collect();
    let v = Vector::from_slice(&data);
    let result = v.sqrt().unwrap();
    assert_eq!(result.len(), 256);
    for (i, (&input, &output)) in data.iter().zip(result.as_slice().iter()).enumerate() {
        let expected = input.sqrt();
        assert!((output - expected).abs() < 1e-5, "index {i}: expected {expected}, got {output}");
    }
}

#[cfg(feature = "parallel")]
#[test]
fn test_sqrt_parallel_path() {
    // 100_000 elements triggers the parallel path (PARALLEL_THRESHOLD = 100_000)
    let data: Vec<f32> = (1..=100_000).map(|x| x as f32).collect();
    let v = Vector::from_slice(&data);
    let result = v.sqrt().unwrap();
    assert_eq!(result.len(), 100_000);
    // Spot-check a few values
    assert!((result.as_slice()[0] - 1.0).abs() < 1e-5);
    assert!((result.as_slice()[3] - 2.0).abs() < 1e-5); // sqrt(4) = 2
    assert!((result.as_slice()[99] - 10.0).abs() < 1e-4); // sqrt(100) = 10
    let last = result.as_slice()[99_999];
    let expected_last = 100_000.0f32.sqrt();
    assert!((last - expected_last).abs() < 1e-1, "last: {last}, expected: {expected_last}");
}

#[cfg(feature = "parallel")]
#[test]
fn test_sqrt_parallel_correctness_vs_scalar() {
    // Verify parallel path produces same results as scalar
    let data: Vec<f32> = (1..=150_000).map(|x| x as f32).collect();
    let v_parallel = Vector::from_slice(&data);
    let result_parallel = v_parallel.sqrt().unwrap();

    let v_scalar = Vector::from_slice_with_backend(&data, Backend::Scalar);
    let result_scalar = v_scalar.sqrt().unwrap();

    assert_eq!(result_parallel.len(), result_scalar.len());
    for (i, (&par, &sca)) in
        result_parallel.as_slice().iter().zip(result_scalar.as_slice().iter()).enumerate()
    {
        assert!((par - sca).abs() < 1e-6, "index {i}: parallel={par}, scalar={sca}");
    }
}

#[cfg(feature = "parallel")]
#[test]
fn test_sqrt_below_parallel_threshold() {
    // 99_999 elements: just under the 100_000 threshold, takes non-parallel path
    let data: Vec<f32> = (1..=99_999).map(|x| x as f32).collect();
    let v = Vector::from_slice(&data);
    let result = v.sqrt().unwrap();
    assert_eq!(result.len(), 99_999);
    assert!((result.as_slice()[0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_sqrt_zero_vector() {
    let data = vec![0.0; 16];
    let v = Vector::from_slice(&data);
    let result = v.sqrt().unwrap();
    for &val in result.as_slice() {
        assert_eq!(val, 0.0);
    }
}

#[test]
fn test_sqrt_one_vector() {
    let data = vec![1.0; 20];
    let v = Vector::from_slice(&data);
    let result = v.sqrt().unwrap();
    for &val in result.as_slice() {
        assert!((val - 1.0).abs() < 1e-7);
    }
}

#[test]
fn test_sqrt_preserves_backend() {
    let v = Vector::from_slice_with_backend(&[4.0, 9.0], Backend::Scalar);
    let result = v.sqrt().unwrap();
    assert_eq!(result.backend(), Backend::Scalar);
}

#[test]
fn test_sqrt_subnormal_values() {
    // Subnormal (denormalized) float values
    let data = vec![f32::MIN_POSITIVE / 2.0, f32::MIN_POSITIVE, f32::MIN_POSITIVE * 4.0];
    let v = Vector::from_slice(&data);
    let result = v.sqrt().unwrap();
    for (i, (&input, &output)) in data.iter().zip(result.as_slice().iter()).enumerate() {
        let expected = input.sqrt();
        assert!((output - expected).abs() < 1e-20, "index {i}: expected {expected}, got {output}");
    }
}

#[test]
fn test_sqrt_all_negative() {
    let data = vec![-1.0, -2.0, -100.0, -f32::INFINITY];
    let v = Vector::from_slice(&data);
    let result = v.sqrt().unwrap();
    for &val in result.as_slice() {
        assert!(val.is_nan(), "sqrt of negative should be NaN, got {val}");
    }
}
