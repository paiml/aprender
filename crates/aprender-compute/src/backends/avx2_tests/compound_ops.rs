//! AVX2 compound operation and scalar parity tests

use crate::backends::avx2::Avx2Backend;
use crate::backends::scalar::ScalarBackend;
use crate::backends::VectorBackend;

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_scale_matches_scalar() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
        return;
    }

    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let scalar = 2.5;
    let mut avx2_result = [0.0; 9];
    let mut scalar_result = [0.0; 9];

    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        Avx2Backend::scale(&a, scalar, &mut avx2_result);
        ScalarBackend::scale(&a, scalar, &mut scalar_result);
    }

    assert_eq!(avx2_result, scalar_result);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_clamp_matches_scalar() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
        return;
    }

    let a = [1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0];
    let mut avx2_result = [0.0; 9];
    let mut scalar_result = [0.0; 9];

    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        Avx2Backend::clamp(&a, 5.0, 30.0, &mut avx2_result);
        ScalarBackend::clamp(&a, 5.0, 30.0, &mut scalar_result);
    }

    assert_eq!(avx2_result, scalar_result);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_fma_matches_scalar() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
        return;
    }

    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let b = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let c = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
    let mut avx2_result = [0.0; 9];
    let mut scalar_result = [0.0; 9];

    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        Avx2Backend::fma(&a, &b, &c, &mut avx2_result);
        ScalarBackend::fma(&a, &b, &c, &mut scalar_result);
    }

    assert_eq!(avx2_result, scalar_result);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_lerp_matches_scalar() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
        return;
    }

    let a = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
    let b = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0];
    let mut avx2_result = [0.0; 9];
    let mut scalar_result = [0.0; 9];

    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        Avx2Backend::lerp(&a, &b, 0.25, &mut avx2_result);
        ScalarBackend::lerp(&a, &b, 0.25, &mut scalar_result);
    }

    for (avx2, scalar) in avx2_result.iter().zip(scalar_result.iter()) {
        assert!((avx2 - scalar).abs() < 1e-5, "lerp mismatch: avx2={}, scalar={}", avx2, scalar);
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_argmax_matches_scalar() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
        return;
    }

    let a = [1.0, 5.0, 3.0, 10.0, 2.0, 8.0, 4.0, 9.0, 6.0];

    // SAFETY: Test code calling backend trait methods marked unsafe
    let avx2_result = unsafe { Avx2Backend::argmax(&a) };
    // SAFETY: CPU feature verified at runtime, slices bounds-checked
    let scalar_result = unsafe { ScalarBackend::argmax(&a) };

    assert_eq!(avx2_result, scalar_result);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_argmin_matches_scalar() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
        return;
    }

    let a = [5.0, 1.0, 3.0, 10.0, 2.0, 8.0, 4.0, 9.0, 6.0];

    // SAFETY: Test code calling backend trait methods marked unsafe
    let avx2_result = unsafe { Avx2Backend::argmin(&a) };
    // SAFETY: CPU feature verified at runtime, slices bounds-checked
    let scalar_result = unsafe { ScalarBackend::argmin(&a) };

    assert_eq!(avx2_result, scalar_result);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_sum_kahan_matches_scalar() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
        return;
    }

    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

    // SAFETY: Test code calling backend trait methods marked unsafe
    let avx2_result = unsafe { Avx2Backend::sum_kahan(&a) };
    // SAFETY: CPU feature verified at runtime, slices bounds-checked
    let scalar_result = unsafe { ScalarBackend::sum_kahan(&a) };

    assert!((avx2_result - scalar_result).abs() < 1e-5);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_norm_l1_matches_scalar() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
        return;
    }

    let a = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0];

    // SAFETY: Test code calling backend trait methods marked unsafe
    let avx2_result = unsafe { Avx2Backend::norm_l1(&a) };
    // SAFETY: CPU feature verified at runtime, slices bounds-checked
    let scalar_result = unsafe { ScalarBackend::norm_l1(&a) };

    assert!((avx2_result - scalar_result).abs() < 1e-5);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_norm_l2_matches_scalar() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
        return;
    }

    let a = [3.0, 4.0, 0.0, 0.0, 5.0, 12.0, 0.0, 8.0, 15.0];

    // SAFETY: Test code calling backend trait methods marked unsafe
    let avx2_result = unsafe { Avx2Backend::norm_l2(&a) };
    // SAFETY: CPU feature verified at runtime, slices bounds-checked
    let scalar_result = unsafe { ScalarBackend::norm_l2(&a) };

    assert!((avx2_result - scalar_result).abs() < 1e-5);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_dot_matches_scalar() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
        return;
    }

    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let b = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

    // SAFETY: Test code calling backend trait methods marked unsafe
    let avx2_result = unsafe { Avx2Backend::dot(&a, &b) };
    // SAFETY: CPU feature verified at runtime, slices bounds-checked
    let scalar_result = unsafe { ScalarBackend::dot(&a, &b) };

    assert!((avx2_result - scalar_result).abs() < 1e-5);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_mul_matches_scalar() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
        return;
    }

    let a = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5];
    let b = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let mut avx2_result = [0.0; 9];
    let mut scalar_result = [0.0; 9];

    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        Avx2Backend::mul(&a, &b, &mut avx2_result);
        ScalarBackend::mul(&a, &b, &mut scalar_result);
    }

    assert_eq!(avx2_result, scalar_result);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_add_matches_scalar() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
        return;
    }

    let a = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5];
    let b = [8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5];
    let mut avx2_result = [0.0; 9];
    let mut scalar_result = [0.0; 9];

    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        Avx2Backend::add(&a, &b, &mut avx2_result);
        ScalarBackend::add(&a, &b, &mut scalar_result);
    }

    assert_eq!(avx2_result, scalar_result);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_sum_matches_scalar() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
        return;
    }

    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

    // SAFETY: Test code calling backend trait methods marked unsafe
    let avx2_result = unsafe { Avx2Backend::sum(&a) };
    // SAFETY: CPU feature verified at runtime, slices bounds-checked
    let scalar_result = unsafe { ScalarBackend::sum(&a) };

    assert!((avx2_result - scalar_result).abs() < 1e-5);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_max_matches_scalar() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
        return;
    }

    let a = [1.0, 5.0, 3.0, 10.0, 2.0, 8.0, 4.0, 9.0, 6.0];

    // SAFETY: Test code calling backend trait methods marked unsafe
    let avx2_result = unsafe { Avx2Backend::max(&a) };
    // SAFETY: CPU feature verified at runtime, slices bounds-checked
    let scalar_result = unsafe { ScalarBackend::max(&a) };

    assert_eq!(avx2_result, scalar_result);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_min_matches_scalar() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
        return;
    }

    let a = [5.0, 1.0, 3.0, 10.0, 2.0, 8.0, 4.0, 9.0, 6.0];

    // SAFETY: Test code calling backend trait methods marked unsafe
    let avx2_result = unsafe { Avx2Backend::min(&a) };
    // SAFETY: CPU feature verified at runtime, slices bounds-checked
    let scalar_result = unsafe { ScalarBackend::min(&a) };

    assert_eq!(avx2_result, scalar_result);
}
