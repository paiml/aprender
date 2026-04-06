use super::super::super::super::*;
use crate::Backend;

// Basic construction tests
#[test]
fn test_from_slice() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    assert_eq!(v.as_slice(), &[1.0, 2.0, 3.0]);
    assert_eq!(v.len(), 3);
}

#[test]
fn test_from_slice_empty() {
    let v: Vector<f32> = Vector::from_slice(&[]);
    assert_eq!(v.len(), 0);
    assert!(v.is_empty());
}

#[test]
fn test_from_slice_single_element() {
    let v = Vector::from_slice(&[42.0]);
    assert_eq!(v.as_slice(), &[42.0]);
    assert_eq!(v.len(), 1);
}

#[test]
fn test_from_slice_with_backend() {
    let v = Vector::from_slice_with_backend(&[1.0, 2.0], Backend::Scalar);
    assert_eq!(v.backend(), Backend::Scalar);
}

#[test]
fn test_scalar_backend_operations() {
    // Test various operations with explicit Scalar backend to cover Backend::Scalar match arms
    let v1 = Vector::from_slice_with_backend(&[1.0, 2.0, 3.0, 4.0], Backend::Scalar);
    let v2 = Vector::from_slice_with_backend(&[4.0, 3.0, 2.0, 1.0], Backend::Scalar);

    // Test dot product (covers line 599)
    let dot = v1.dot(&v2).unwrap();
    assert_eq!(dot, 1.0 * 4.0 + 2.0 * 3.0 + 3.0 * 2.0 + 4.0 * 1.0); // = 20.0

    // Test sum (covers line 856)
    let sum = v1.sum().unwrap();
    assert_eq!(sum, 10.0);

    // Test max (covers line 661)
    let max = v1.max().unwrap();
    assert_eq!(max, 4.0);

    // Test min (covers line 709)
    let min = v1.min().unwrap();
    assert_eq!(min, 1.0);

    // Test argmax (covers line 757)
    let argmax = v1.argmax().unwrap();
    assert_eq!(argmax, 3);

    // Test argmin (covers line 805)
    let argmin = v1.argmin().unwrap();
    assert_eq!(argmin, 0);
}

#[test]
fn test_gpu_and_auto_backend_fallback() {
    // Test operations with GPU/Auto backend which fallback to scalar
    let v1 = Vector::from_slice_with_backend(&[1.0, 2.0, 3.0], Backend::GPU);
    let v2 = Vector::from_slice_with_backend(&[3.0, 2.0, 1.0], Backend::GPU);

    // These should all work (fallback to scalar)
    let dot = v1.dot(&v2).unwrap();
    assert_eq!(dot, 10.0);

    let sum = v1.sum().unwrap();
    assert_eq!(sum, 6.0);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_avx512_backend_vector_ops() {
    // Test operations with explicit AVX-512 backend to cover Backend::AVX512 match arms
    if !is_x86_feature_detected!("avx512f") {
        return;
    }

    // Use large vectors to exercise SIMD paths
    let data1: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let data2: Vec<f32> = (0..1024).map(|i| (1024 - i) as f32).collect();

    let v1 = Vector::from_slice_with_backend(&data1, Backend::AVX512);
    let v2 = Vector::from_slice_with_backend(&data2, Backend::AVX512);

    // Test dot product (covers line 605)
    let dot = v1.dot(&v2).unwrap();
    let expected_dot: f32 = data1.iter().zip(data2.iter()).map(|(a, b)| a * b).sum();
    let dot_rel_err = (dot - expected_dot).abs() / expected_dot.abs().max(1.0);
    assert!(dot_rel_err < 0.01, "dot mismatch: {} vs {}", dot, expected_dot);

    // Test sum
    let sum = v1.sum().unwrap();
    let expected_sum: f32 = data1.iter().sum();
    let sum_rel_err = (sum - expected_sum).abs() / expected_sum.abs().max(1.0);
    assert!(sum_rel_err < 0.01, "sum mismatch: {} vs {}", sum, expected_sum);

    // Test max
    let max = v1.max().unwrap();
    assert_eq!(max, 1023.0);

    // Test min
    let min = v1.min().unwrap();
    assert_eq!(min, 0.0);

    // Test argmax
    let argmax = v1.argmax().unwrap();
    assert_eq!(argmax, 1023);

    // Test argmin
    let argmin = v1.argmin().unwrap();
    assert_eq!(argmin, 0);
}

#[test]
fn test_auto_backend_resolution() {
    let v = Vector::from_slice_with_backend(&[1.0], Backend::Auto);
    // Auto should be resolved to best available backend
    let expected_backend = crate::select_best_available_backend();
    assert_eq!(v.backend(), expected_backend);

    // Verify it's not still Backend::Auto after resolution
    assert_ne!(v.backend(), Backend::Auto);

    // On x86_64, should be a SIMD backend (not Scalar)
    #[cfg(target_arch = "x86_64")]
    {
        assert_ne!(v.backend(), Backend::Scalar);
        assert!(matches!(
            v.backend(),
            Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512
        ));
    }
}

#[test]
fn test_with_alignment_valid() {
    let v = Vector::with_alignment(100, Backend::SSE2, 16).unwrap();
    assert_eq!(v.len(), 100);
    assert_eq!(v.backend(), Backend::SSE2);
}

#[test]
fn test_with_alignment_power_of_two() {
    // Test various power-of-2 alignments
    assert!(Vector::with_alignment(10, Backend::Scalar, 1).is_ok());
    assert!(Vector::with_alignment(10, Backend::Scalar, 2).is_ok());
    assert!(Vector::with_alignment(10, Backend::Scalar, 4).is_ok());
    assert!(Vector::with_alignment(10, Backend::Scalar, 8).is_ok());
    assert!(Vector::with_alignment(10, Backend::Scalar, 16).is_ok());
    assert!(Vector::with_alignment(10, Backend::Scalar, 32).is_ok());
    assert!(Vector::with_alignment(10, Backend::Scalar, 64).is_ok());
}

#[test]
fn test_with_alignment_invalid_zero() {
    let result = Vector::with_alignment(100, Backend::Scalar, 0);
    assert!(result.is_err());
    match result {
        Err(TruenoError::InvalidInput(msg)) => {
            assert!(msg.contains("power of 2"));
            assert!(msg.contains('0'));
        }
        _ => panic!("Expected InvalidInput error for zero alignment"),
    }
}

#[test]
fn test_with_alignment_invalid_not_power_of_two() {
    // Test various non-power-of-2 values
    for alignment in &[3, 5, 6, 7, 9, 10, 12, 15, 17, 20, 24, 31, 33] {
        let result = Vector::with_alignment(100, Backend::Scalar, *alignment);
        assert!(result.is_err(), "Alignment {} should be invalid", alignment);
        match result {
            Err(TruenoError::InvalidInput(msg)) => {
                assert!(msg.contains("power of 2"), "Error message should mention power of 2");
                assert!(
                    msg.contains(&alignment.to_string()),
                    "Error message should mention the invalid alignment"
                );
            }
            _ => panic!("Expected InvalidInput error for non-power-of-2 alignment {}", alignment),
        }
    }
}

#[test]
fn test_with_alignment_auto_backend_resolution() {
    let v = Vector::with_alignment(100, Backend::Auto, 16).unwrap();
    // Backend::Auto should be resolved to best available backend
    assert_ne!(v.backend(), Backend::Auto);
}
