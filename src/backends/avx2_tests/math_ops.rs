//! AVX2 mathematical operation and golden parity tests

use crate::backends::avx2::Avx2Backend;
use crate::backends::scalar::ScalarBackend;
use crate::backends::VectorBackend;

// Tests for mathematical operations
#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_sqrt_matches_scalar() {
    let a = [4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0, 144.0];
    let mut avx2_result = vec![0.0; a.len()];
    let mut scalar_result = vec![0.0; a.len()];

    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        Avx2Backend::sqrt(&a, &mut avx2_result);
        ScalarBackend::sqrt(&a, &mut scalar_result);
    }

    for i in 0..a.len() {
        assert!(
            (avx2_result[i] - scalar_result[i]).abs() < 1e-5,
            "sqrt({}) mismatch: avx2={}, scalar={}",
            a[i],
            avx2_result[i],
            scalar_result[i]
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_recip_matches_scalar() {
    let a = [1.0, 2.0, 4.0, 5.0, 8.0, 10.0, 16.0, 20.0, 25.0, 32.0];
    let mut avx2_result = vec![0.0; a.len()];
    let mut scalar_result = vec![0.0; a.len()];

    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        Avx2Backend::recip(&a, &mut avx2_result);
        ScalarBackend::recip(&a, &mut scalar_result);
    }

    for i in 0..a.len() {
        assert!(
            (avx2_result[i] - scalar_result[i]).abs() < 1e-5,
            "recip({}) mismatch: avx2={}, scalar={}",
            a[i],
            avx2_result[i],
            scalar_result[i]
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_ln_matches_scalar() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0];
    let mut avx2_result = vec![0.0; a.len()];
    let mut scalar_result = vec![0.0; a.len()];

    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        Avx2Backend::ln(&a, &mut avx2_result);
        ScalarBackend::ln(&a, &mut scalar_result);
    }

    for i in 0..a.len() {
        assert!(
            (avx2_result[i] - scalar_result[i]).abs() < 1e-5,
            "ln({}) mismatch: avx2={}, scalar={}",
            a[i],
            avx2_result[i],
            scalar_result[i]
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_log2_matches_scalar() {
    let a = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0];
    let mut avx2_result = vec![0.0; a.len()];
    let mut scalar_result = vec![0.0; a.len()];

    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        Avx2Backend::log2(&a, &mut avx2_result);
        ScalarBackend::log2(&a, &mut scalar_result);
    }

    for i in 0..a.len() {
        assert!(
            (avx2_result[i] - scalar_result[i]).abs() < 1e-5,
            "log2({}) mismatch: avx2={}, scalar={}",
            a[i],
            avx2_result[i],
            scalar_result[i]
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_log10_matches_scalar() {
    let a = [1.0, 10.0, 100.0, 1000.0, 2.0, 20.0, 200.0, 5.0, 50.0, 500.0];
    let mut avx2_result = vec![0.0; a.len()];
    let mut scalar_result = vec![0.0; a.len()];

    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        Avx2Backend::log10(&a, &mut avx2_result);
        ScalarBackend::log10(&a, &mut scalar_result);
    }

    for i in 0..a.len() {
        assert!(
            (avx2_result[i] - scalar_result[i]).abs() < 1e-5,
            "log10({}) mismatch: avx2={}, scalar={}",
            a[i],
            avx2_result[i],
            scalar_result[i]
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_sin_matches_scalar() {
    use std::f32::consts::PI;
    let a =
        [0.0, PI / 6.0, PI / 4.0, PI / 3.0, PI / 2.0, PI, 1.5 * PI, 2.0 * PI, -PI / 4.0, -PI / 2.0];
    let mut avx2_result = vec![0.0; a.len()];
    let mut scalar_result = vec![0.0; a.len()];

    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        Avx2Backend::sin(&a, &mut avx2_result);
        ScalarBackend::sin(&a, &mut scalar_result);
    }

    for i in 0..a.len() {
        assert!(
            (avx2_result[i] - scalar_result[i]).abs() < 1e-5,
            "sin({}) mismatch: avx2={}, scalar={}",
            a[i],
            avx2_result[i],
            scalar_result[i]
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_cos_matches_scalar() {
    use std::f32::consts::PI;
    let a =
        [0.0, PI / 6.0, PI / 4.0, PI / 3.0, PI / 2.0, PI, 1.5 * PI, 2.0 * PI, -PI / 4.0, -PI / 2.0];
    let mut avx2_result = vec![0.0; a.len()];
    let mut scalar_result = vec![0.0; a.len()];

    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        Avx2Backend::cos(&a, &mut avx2_result);
        ScalarBackend::cos(&a, &mut scalar_result);
    }

    for i in 0..a.len() {
        assert!(
            (avx2_result[i] - scalar_result[i]).abs() < 1e-5,
            "cos({}) mismatch: avx2={}, scalar={}",
            a[i],
            avx2_result[i],
            scalar_result[i]
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_tan_matches_scalar() {
    use std::f32::consts::PI;
    let a = [0.0, PI / 6.0, PI / 4.0, PI / 3.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0];
    let mut avx2_result = vec![0.0; a.len()];
    let mut scalar_result = vec![0.0; a.len()];

    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        Avx2Backend::tan(&a, &mut avx2_result);
        ScalarBackend::tan(&a, &mut scalar_result);
    }

    for i in 0..a.len() {
        assert!(
            (avx2_result[i] - scalar_result[i]).abs() < 1e-5,
            "tan({}) mismatch: avx2={}, scalar={}",
            a[i],
            avx2_result[i],
            scalar_result[i]
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_floor_matches_scalar() {
    let a = [1.1, 2.5, 3.9, -1.1, -2.5, -3.9, 0.1, 0.9, -0.1, -0.9];
    let mut avx2_result = vec![0.0; a.len()];
    let mut scalar_result = vec![0.0; a.len()];

    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        Avx2Backend::floor(&a, &mut avx2_result);
        ScalarBackend::floor(&a, &mut scalar_result);
    }

    for i in 0..a.len() {
        assert_eq!(
            avx2_result[i], scalar_result[i],
            "floor({}) mismatch: avx2={}, scalar={}",
            a[i], avx2_result[i], scalar_result[i]
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_ceil_matches_scalar() {
    let a = [1.1, 2.5, 3.9, -1.1, -2.5, -3.9, 0.1, 0.9, -0.1, -0.9];
    let mut avx2_result = vec![0.0; a.len()];
    let mut scalar_result = vec![0.0; a.len()];

    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        Avx2Backend::ceil(&a, &mut avx2_result);
        ScalarBackend::ceil(&a, &mut scalar_result);
    }

    for i in 0..a.len() {
        assert_eq!(
            avx2_result[i], scalar_result[i],
            "ceil({}) mismatch: avx2={}, scalar={}",
            a[i], avx2_result[i], scalar_result[i]
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_round_matches_scalar() {
    let a = [1.1, 2.5, 3.9, -1.1, -2.5, -3.9, 0.1, 0.9, -0.1, -0.9];
    let mut avx2_result = vec![0.0; a.len()];
    let mut scalar_result = vec![0.0; a.len()];

    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        Avx2Backend::round(&a, &mut avx2_result);
        ScalarBackend::round(&a, &mut scalar_result);
    }

    for i in 0..a.len() {
        assert!(
            (avx2_result[i] - scalar_result[i]).abs() < 1e-5,
            "round({}) mismatch: avx2={}, scalar={}",
            a[i],
            avx2_result[i],
            scalar_result[i]
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_norm_linf_matches_scalar() {
    if !is_x86_feature_detected!("avx2") {
        eprintln!("Skipping AVX2 test: CPU does not support AVX2");
        return;
    }

    // Test various input sizes
    let test_cases = vec![
        vec![],                                                // empty
        vec![5.0],                                             // single element
        vec![-3.0, 1.0, -4.0, 1.0, 5.0],                       // small vector
        vec![-10.0, 5.0, 3.0, 7.0, -2.0, 8.0, 4.0],            // 7 elements
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],          // 8 elements (aligned)
        vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0], // 9 elements (remainder)
    ];

    for test_vec in test_cases {
        // SAFETY: Test code calling backend trait methods marked unsafe
        let scalar_result = unsafe { ScalarBackend::norm_linf(&test_vec) };
        // SAFETY: CPU feature verified at runtime, slices bounds-checked
        let avx2_result = unsafe { Avx2Backend::norm_linf(&test_vec) };

        assert!(
            (scalar_result - avx2_result).abs() < 1e-5,
            "norm_linf mismatch for {:?}: scalar={}, avx2={}",
            test_vec,
            scalar_result,
            avx2_result
        );
    }
}

// =========================================================================
// Golden Parity Test (PMAT-018): AVX2 vs Scalar Exhaustive Comparison
// =========================================================================

/// PMAT-018 Golden Parity: Run 1,000 operations on random vectors
/// and assert AVX2 and Scalar produce identical results (within FP tolerance).
///
/// This test locks in the numerical behavior of AVX2 optimizations.
/// If they differ by more than 1e-5 for f32 operations, the optimization
/// is a "Precision-Sacrificing Conjecture" that must be documented.
#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_vs_scalar_golden_parity() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("Skipping AVX2 golden parity test: CPU does not support AVX2+FMA");
        return;
    }

    // Simple deterministic pseudo-random generator (xorshift32)
    // Using deterministic seed for reproducibility
    let mut rng_state: u32 = 12345;
    let mut next_rand = || -> f32 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        // Map to range [-100.0, 100.0]
        (rng_state as f32 / u32::MAX as f32) * 200.0 - 100.0
    };

    const NUM_ITERATIONS: usize = 1000;
    const VECTOR_SIZE: usize = 127; // Non-power-of-2 to stress remainder handling
    const FP_TOLERANCE: f32 = 1e-5;

    let mut total_ops = 0;
    let mut max_diff: f32 = 0.0;

    for iteration in 0..NUM_ITERATIONS {
        // Generate random vectors
        let a: Vec<f32> = (0..VECTOR_SIZE).map(|_| next_rand()).collect();
        let b: Vec<f32> = (0..VECTOR_SIZE).map(|_| next_rand()).collect();
        let mut avx2_result = vec![0.0f32; VECTOR_SIZE];
        let mut scalar_result = vec![0.0f32; VECTOR_SIZE];

        // Test add
        // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
        unsafe {
            Avx2Backend::add(&a, &b, &mut avx2_result);
            ScalarBackend::add(&a, &b, &mut scalar_result);
        }
        for (i, (&av, &sc)) in avx2_result.iter().zip(scalar_result.iter()).enumerate() {
            let diff = (av - sc).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff < FP_TOLERANCE,
                "ADD parity fail iter={} idx={}: avx2={} scalar={} diff={}",
                iteration,
                i,
                av,
                sc,
                diff
            );
        }
        total_ops += 1;

        // Test mul
        // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
        unsafe {
            Avx2Backend::mul(&a, &b, &mut avx2_result);
            ScalarBackend::mul(&a, &b, &mut scalar_result);
        }
        for (i, (&av, &sc)) in avx2_result.iter().zip(scalar_result.iter()).enumerate() {
            let diff = (av - sc).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff < FP_TOLERANCE,
                "MUL parity fail iter={} idx={}: avx2={} scalar={} diff={}",
                iteration,
                i,
                av,
                sc,
                diff
            );
        }
        total_ops += 1;

        // Test dot product
        // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
        let avx2_dot = unsafe { Avx2Backend::dot(&a, &b) };
        // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
        let scalar_dot = unsafe { ScalarBackend::dot(&a, &b) };
        // Dot products accumulate, so tolerance scales with vector size
        let dot_tolerance = FP_TOLERANCE * VECTOR_SIZE as f32 * 100.0;
        let dot_diff = (avx2_dot - scalar_dot).abs();
        max_diff = max_diff.max(dot_diff / (VECTOR_SIZE as f32 * 100.0));
        assert!(
            dot_diff < dot_tolerance,
            "DOT parity fail iter={}: avx2={} scalar={} diff={}",
            iteration,
            avx2_dot,
            scalar_dot,
            dot_diff
        );
        total_ops += 1;

        // Test sum
        // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
        let avx2_sum = unsafe { Avx2Backend::sum(&a) };
        // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
        let scalar_sum = unsafe { ScalarBackend::sum(&a) };
        let sum_tolerance = FP_TOLERANCE * VECTOR_SIZE as f32 * 100.0;
        let sum_diff = (avx2_sum - scalar_sum).abs();
        assert!(
            sum_diff < sum_tolerance,
            "SUM parity fail iter={}: avx2={} scalar={} diff={}",
            iteration,
            avx2_sum,
            scalar_sum,
            sum_diff
        );
        total_ops += 1;
    }

    eprintln!(
        "Golden Parity PASSED: {} operations, max element diff = {:.2e}",
        total_ops, max_diff
    );
}
