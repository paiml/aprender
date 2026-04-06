//! AVX2 basic arithmetic and activation function tests

use crate::backends::avx2::Avx2Backend;
use crate::backends::scalar::ScalarBackend;
use crate::backends::VectorBackend;

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_add() {
    if !is_x86_feature_detected!("avx2") {
        eprintln!("Skipping AVX2 test: CPU does not support AVX2");
        return;
    }

    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    let mut result = vec![0.0; 9];

    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        Avx2Backend::add(&a, &b, &mut result);
    }

    assert_eq!(result, vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_mul() {
    if !is_x86_feature_detected!("avx2") {
        eprintln!("Skipping AVX2 test: CPU does not support AVX2");
        return;
    }

    let a = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let mut result = vec![0.0; 9];

    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        Avx2Backend::mul(&a, &b, &mut result);
    }

    assert_eq!(result, vec![2.0, 6.0, 12.0, 20.0, 30.0, 42.0, 56.0, 72.0, 90.0]);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_dot() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
        return;
    }

    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

    // SAFETY: Test code calling backend trait methods marked unsafe
    let result = unsafe { Avx2Backend::dot(&a, &b) };

    // 1*9 + 2*8 + 3*7 + 4*6 + 5*5 + 6*4 + 7*3 + 8*2 + 9*1
    // = 9 + 16 + 21 + 24 + 25 + 24 + 21 + 16 + 9 = 165
    assert!((result - 165.0).abs() < 1e-5);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_sum() {
    if !is_x86_feature_detected!("avx2") {
        eprintln!("Skipping AVX2 test: CPU does not support AVX2");
        return;
    }

    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

    // SAFETY: Test code calling backend trait methods marked unsafe
    let result = unsafe { Avx2Backend::sum(&a) };

    assert!((result - 45.0).abs() < 1e-5);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_max() {
    if !is_x86_feature_detected!("avx2") {
        eprintln!("Skipping AVX2 test: CPU does not support AVX2");
        return;
    }

    let a = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0];

    // SAFETY: Test code calling backend trait methods marked unsafe
    let result = unsafe { Avx2Backend::max(&a) };

    assert_eq!(result, 9.0);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_min() {
    if !is_x86_feature_detected!("avx2") {
        eprintln!("Skipping AVX2 test: CPU does not support AVX2");
        return;
    }

    let a = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0];

    // SAFETY: Test code calling backend trait methods marked unsafe
    let result = unsafe { Avx2Backend::min(&a) };

    assert_eq!(result, 1.0);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_matches_scalar() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
        return;
    }

    let a = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5];
    let b = vec![10.5, 9.5, 8.5, 7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5];

    // Test add
    let mut avx2_result = vec![0.0; 10];
    let mut scalar_result = vec![0.0; 10];
    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        Avx2Backend::add(&a, &b, &mut avx2_result);
        ScalarBackend::add(&a, &b, &mut scalar_result);
    }
    for (avx2, scalar) in avx2_result.iter().zip(&scalar_result) {
        assert!((avx2 - scalar).abs() < 1e-5);
    }

    // Test dot
    let (avx2_dot, scalar_dot) =
        // SAFETY: Calling backend methods with verified safety invariants
        unsafe { (Avx2Backend::dot(&a, &b), ScalarBackend::dot(&a, &b)) };
    assert!((avx2_dot - scalar_dot).abs() < 1e-3); // Relaxed tolerance for FMA

    // Test sum
    // SAFETY: CPU feature verified at runtime, slices bounds-checked
    let (avx2_sum, scalar_sum) = unsafe { (Avx2Backend::sum(&a), ScalarBackend::sum(&a)) };
    assert!((avx2_sum - scalar_sum).abs() < 1e-3);

    // Test max
    // SAFETY: Calling backend methods with verified safety invariants
    let (avx2_max, scalar_max) = unsafe { (Avx2Backend::max(&a), ScalarBackend::max(&a)) };
    assert_eq!(avx2_max, scalar_max);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_relu() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
        return;
    }

    // Test with 16 elements (2 AVX2 registers of 8 f32s)
    let a =
        [-3.0, -1.0, 0.0, 1.0, 3.0, -2.0, 2.0, -0.5, -4.0, 4.0, -5.0, 5.0, 0.0, -0.1, 0.1, 10.0];
    let mut result = [0.0; 16];
    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        Avx2Backend::relu(&a, &mut result);
    }
    let expected =
        [0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 2.0, 0.0, 0.0, 4.0, 0.0, 5.0, 0.0, 0.0, 0.1, 10.0];
    assert_eq!(result, expected);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_relu_matches_scalar() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
        return;
    }

    let a = [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0, -2.0, 2.0, -4.0, 4.0];
    let mut avx2_result = [0.0; 11];
    let mut scalar_result = [0.0; 11];

    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        Avx2Backend::relu(&a, &mut avx2_result);
        ScalarBackend::relu(&a, &mut scalar_result);
    }

    assert_eq!(avx2_result, scalar_result);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_sigmoid_matches_scalar() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
        return;
    }

    let a = [-10.0, -1.0, 0.0, 1.0, 10.0];
    let mut avx2_result = [0.0; 5];
    let mut scalar_result = [0.0; 5];

    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        Avx2Backend::sigmoid(&a, &mut avx2_result);
        ScalarBackend::sigmoid(&a, &mut scalar_result);
    }

    for (avx2, scalar) in avx2_result.iter().zip(scalar_result.iter()) {
        assert!((avx2 - scalar).abs() < 1e-6, "sigmoid mismatch: avx2={}, scalar={}", avx2, scalar);
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_exp_matches_scalar() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
        return;
    }

    // Test various ranges: negative, zero, positive, large values
    let test_values = vec![
        -10.0, -5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, -50.0, 87.0,
        -87.0, // near overflow/underflow limits
    ];
    let mut avx2_result = vec![0.0; test_values.len()];
    let mut scalar_result = vec![0.0; test_values.len()];

    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        Avx2Backend::exp(&test_values, &mut avx2_result);
        ScalarBackend::exp(&test_values, &mut scalar_result);
    }

    for (i, (avx2, scalar)) in avx2_result.iter().zip(scalar_result.iter()).enumerate() {
        let rel_error = if scalar.abs() > 1e-10 {
            (avx2 - scalar).abs() / scalar.abs()
        } else {
            (avx2 - scalar).abs()
        };
        assert!(
            rel_error < 1e-5,
            "exp({}) mismatch: avx2={}, scalar={}, rel_error={}",
            test_values[i],
            avx2,
            scalar,
            rel_error
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_gelu_matches_scalar() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
        return;
    }

    let a = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let mut avx2_result = [0.0; 5];
    let mut scalar_result = [0.0; 5];

    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        Avx2Backend::gelu(&a, &mut avx2_result);
        ScalarBackend::gelu(&a, &mut scalar_result);
    }

    for (avx2, scalar) in avx2_result.iter().zip(scalar_result.iter()) {
        assert!((avx2 - scalar).abs() < 1e-5, "gelu mismatch: avx2={}, scalar={}", avx2, scalar);
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_swish_matches_scalar() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
        return;
    }

    let a = [-10.0, -1.0, 0.0, 1.0, 10.0];
    let mut avx2_result = [0.0; 5];
    let mut scalar_result = [0.0; 5];

    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        Avx2Backend::swish(&a, &mut avx2_result);
        ScalarBackend::swish(&a, &mut scalar_result);
    }

    for (avx2, scalar) in avx2_result.iter().zip(scalar_result.iter()) {
        assert!((avx2 - scalar).abs() < 1e-5, "swish mismatch: avx2={}, scalar={}", avx2, scalar);
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_sub_matches_scalar() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
        return;
    }

    let a = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
    let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let mut avx2_result = [0.0; 9];
    let mut scalar_result = [0.0; 9];

    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        Avx2Backend::sub(&a, &b, &mut avx2_result);
        ScalarBackend::sub(&a, &b, &mut scalar_result);
    }

    assert_eq!(avx2_result, scalar_result);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_div_matches_scalar() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
        return;
    }

    let a = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
    let b = [2.0, 4.0, 5.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0];
    let mut avx2_result = [0.0; 9];
    let mut scalar_result = [0.0; 9];

    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        Avx2Backend::div(&a, &b, &mut avx2_result);
        ScalarBackend::div(&a, &b, &mut scalar_result);
    }

    assert_eq!(avx2_result, scalar_result);
}
