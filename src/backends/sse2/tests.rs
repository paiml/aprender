use super::*;

#[test]
fn test_sse2_add() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0];
    let b = [10.0, 20.0, 30.0, 40.0, 50.0];
    let mut result = [0.0f32; 5];
    // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
    unsafe {
        Sse2Backend::add(&a, &b, &mut result);
    }
    assert_eq!(result, [11.0, 22.0, 33.0, 44.0, 55.0]);
}

#[test]
fn test_sse2_sub() {
    let a = [10.0, 20.0, 30.0, 40.0, 50.0];
    let b = [1.0, 2.0, 3.0, 4.0, 5.0];
    let mut result = [0.0f32; 5];
    // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
    unsafe {
        Sse2Backend::sub(&a, &b, &mut result);
    }
    assert_eq!(result, [9.0, 18.0, 27.0, 36.0, 45.0]);
}

#[test]
fn test_sse2_mul() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0];
    let b = [2.0, 3.0, 4.0, 5.0, 6.0];
    let mut result = [0.0f32; 5];
    // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
    unsafe {
        Sse2Backend::mul(&a, &b, &mut result);
    }
    assert_eq!(result, [2.0, 6.0, 12.0, 20.0, 30.0]);
}

#[test]
fn test_sse2_div() {
    let a = [10.0, 20.0, 30.0, 40.0, 50.0];
    let b = [2.0, 4.0, 5.0, 8.0, 10.0];
    let mut result = [0.0f32; 5];
    // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
    unsafe {
        Sse2Backend::div(&a, &b, &mut result);
    }
    assert_eq!(result, [5.0, 5.0, 6.0, 5.0, 5.0]);
}

#[test]
fn test_sse2_dot() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0];
    let b = [2.0, 3.0, 4.0, 5.0, 6.0];
    // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
    let result = unsafe { Sse2Backend::dot(&a, &b) };
    assert!((result - 70.0).abs() < 1e-6);
}

#[test]
fn test_sse2_sum() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0];
    // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
    let result = unsafe { Sse2Backend::sum(&a) };
    assert!((result - 15.0).abs() < 1e-6);
}

#[test]
fn test_sse2_max() {
    let a = [1.0, 5.0, 3.0, 2.0, 4.0];
    // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
    let result = unsafe { Sse2Backend::max(&a) };
    assert!((result - 5.0).abs() < 1e-6);
}

#[test]
fn test_sse2_min() {
    let a = [5.0, 1.0, 3.0, 2.0, 4.0];
    // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
    let result = unsafe { Sse2Backend::min(&a) };
    assert!((result - 1.0).abs() < 1e-6);
}

#[test]
fn test_sse2_argmax() {
    let a = [1.0, 5.0, 3.0, 2.0, 4.0];
    // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
    let result = unsafe { Sse2Backend::argmax(&a) };
    assert_eq!(result, 1);
}

#[test]
fn test_sse2_argmin() {
    let a = [5.0, 1.0, 3.0, 2.0, 4.0];
    // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
    let result = unsafe { Sse2Backend::argmin(&a) };
    assert_eq!(result, 1);
}

#[test]
fn test_sse2_norm_linf() {
    let a = [-5.0, 1.0, 3.0, 2.0, -4.0];
    // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
    let result = unsafe { Sse2Backend::norm_linf(&a) };
    assert!((result - 5.0).abs() < 1e-6);
}

#[test]
fn test_sse2_scale() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0];
    let mut result = [0.0f32; 5];
    // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
    unsafe {
        Sse2Backend::scale(&a, 2.0, &mut result);
    }
    assert_eq!(result, [2.0, 4.0, 6.0, 8.0, 10.0]);
}

#[test]
fn test_sse2_abs() {
    let a = [-1.0, 2.0, -3.0, 4.0, -5.0];
    let mut result = [0.0f32; 5];
    // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
    unsafe {
        Sse2Backend::abs(&a, &mut result);
    }
    assert_eq!(result, [1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_sse2_clamp() {
    let a = [-1.0, 0.5, 1.5, 2.0, 3.0];
    let mut result = [0.0f32; 5];
    // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
    unsafe {
        Sse2Backend::clamp(&a, 0.0, 1.0, &mut result);
    }
    assert_eq!(result, [0.0, 0.5, 1.0, 1.0, 1.0]);
}

#[test]
fn test_sse2_relu() {
    let a = [-1.0, 0.0, 1.0, -2.0, 3.0];
    let mut result = [0.0f32; 5];
    // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
    unsafe {
        Sse2Backend::relu(&a, &mut result);
    }
    assert_eq!(result, [0.0, 0.0, 1.0, 0.0, 3.0]);
}

#[test]
fn test_sse2_exp() {
    let a = [0.0, 1.0, -1.0, 2.0];
    let mut result = [0.0f32; 4];
    // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
    unsafe {
        Sse2Backend::exp(&a, &mut result);
    }
    assert!((result[0] - 1.0).abs() < 0.05);
    assert!((result[1] - std::f32::consts::E).abs() < 0.1);
}

#[test]
fn test_sse2_sigmoid() {
    let a = [0.0, 1.0, -1.0, 10.0];
    let mut result = [0.0f32; 4];
    // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
    unsafe {
        Sse2Backend::sigmoid(&a, &mut result);
    }
    assert!((result[0] - 0.5).abs() < 0.01);
    assert!(result[1] > 0.5);
    assert!(result[2] < 0.5);
}

#[test]
fn test_sse2_sqrt() {
    let a = [1.0, 4.0, 9.0, 16.0, 25.0];
    let mut result = [0.0f32; 5];
    // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
    unsafe {
        Sse2Backend::sqrt(&a, &mut result);
    }
    assert_eq!(result, [1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_sse2_sum_kahan() {
    let a: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
    let result = unsafe { Sse2Backend::sum_kahan(&a) };
    assert!((result - 136.0).abs() < 1e-3);
}

#[test]
fn test_sse2_norm_l2() {
    let a = vec![3.0, 4.0];
    // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
    let result = unsafe { Sse2Backend::norm_l2(&a) };
    assert!((result - 5.0).abs() < 1e-5);
}

#[test]
fn test_sse2_norm_l1() {
    let a = vec![-1.0, 2.0, -3.0, 4.0];
    // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
    let result = unsafe { Sse2Backend::norm_l1(&a) };
    assert!((result - 10.0).abs() < 1e-5);
}

#[test]
fn test_sse2_lerp() {
    let a = vec![0.0; 16];
    let b = vec![10.0; 16];
    let mut result = vec![0.0; 16];
    // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
    unsafe {
        Sse2Backend::lerp(&a, &b, 0.5, &mut result);
    }
    assert!(result.iter().all(|&x| (x - 5.0).abs() < 1e-5));
}

#[test]
fn test_sse2_fma() {
    let a = vec![2.0; 16];
    let b = vec![3.0; 16];
    let c = vec![1.0; 16];
    let mut result = vec![0.0; 16];
    // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
    unsafe {
        Sse2Backend::fma(&a, &b, &c, &mut result);
    }
    assert!(result.iter().all(|&x| (x - 7.0).abs() < 1e-5));
}

#[test]
fn test_sse2_gelu() {
    let a = vec![0.0, 1.0];
    let mut result = vec![0.0; 2];
    // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
    unsafe {
        Sse2Backend::gelu(&a, &mut result);
    }
    assert!((result[0]).abs() < 1e-5);
    assert!((result[1] - 0.841_192).abs() < 1e-2);
}

#[test]
fn test_sse2_swish() {
    let a = vec![0.0, 1.0];
    let mut result = vec![0.0; 2];
    // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
    unsafe {
        Sse2Backend::swish(&a, &mut result);
    }
    assert!((result[0]).abs() < 1e-5);
    assert!((result[1] - 0.731_059).abs() < 1e-2);
}

#[test]
fn test_sse2_tanh() {
    let a = vec![0.0, 1.0];
    let mut result = vec![0.0; 2];
    // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
    unsafe {
        Sse2Backend::tanh(&a, &mut result);
    }
    assert!((result[0]).abs() < 1e-5);
    assert!((result[1] - 0.761_594_2).abs() < 1e-2);
}

#[test]
fn test_sse2_recip() {
    let a = vec![2.0, 4.0, 5.0];
    let mut result = vec![0.0; 3];
    // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
    unsafe {
        Sse2Backend::recip(&a, &mut result);
    }
    assert!((result[0] - 0.5).abs() < 1e-5);
    assert!((result[1] - 0.25).abs() < 1e-5);
    assert!((result[2] - 0.2).abs() < 1e-5);
}

#[test]
fn test_sse2_transcendental() {
    let a = vec![1.0, std::f32::consts::E, 10.0];
    let mut ln_result = vec![0.0; 3];
    let mut log10_result = vec![0.0; 3];
    // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
    unsafe {
        Sse2Backend::ln(&a, &mut ln_result);
        Sse2Backend::log10(&a, &mut log10_result);
    }
    assert!((ln_result[0]).abs() < 1e-5);
    assert!((ln_result[1] - 1.0).abs() < 1e-4);
    assert!((log10_result[2] - 1.0).abs() < 1e-5);
}

#[test]
fn test_sse2_trig() {
    let a = vec![0.0, std::f32::consts::FRAC_PI_2];
    let mut sin_result = vec![0.0; 2];
    let mut cos_result = vec![0.0; 2];
    // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
    unsafe {
        Sse2Backend::sin(&a, &mut sin_result);
        Sse2Backend::cos(&a, &mut cos_result);
    }
    assert!((sin_result[0]).abs() < 1e-5);
    assert!((sin_result[1] - 1.0).abs() < 1e-5);
    assert!((cos_result[0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_sse2_rounding() {
    let a = vec![1.3, 1.5, 1.7, -1.3, -1.5, -1.7];
    let mut floor_result = vec![0.0; 6];
    let mut ceil_result = vec![0.0; 6];
    // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
    unsafe {
        Sse2Backend::floor(&a, &mut floor_result);
        Sse2Backend::ceil(&a, &mut ceil_result);
    }
    assert_eq!(floor_result, vec![1.0, 1.0, 1.0, -2.0, -2.0, -2.0]);
    assert_eq!(ceil_result, vec![2.0, 2.0, 2.0, -1.0, -1.0, -1.0]);
}
