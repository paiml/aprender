//! Tests for SIMD-accelerated optimizer operations

use super::*;
use approx::assert_abs_diff_eq;
use proptest::prelude::*;

// ========================================================================
// PROPERTY TESTS - Verify SIMD vs scalar equivalence
// ========================================================================

/// Scalar reference implementation for AXPY
fn scalar_axpy(a: f32, x: &[f32], y: &mut [f32]) {
    for i in 0..x.len() {
        y[i] += a * x[i];
    }
}

/// Scalar reference implementation for Adam update
fn scalar_adam_update(
    grad: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    param: &mut [f32],
    beta1: f32,
    beta2: f32,
    lr_t: f32,
    epsilon: f32,
) {
    for i in 0..grad.len() {
        m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
        param[i] -= lr_t * m[i] / (v[i].sqrt() + epsilon);
    }
}

/// Scalar reference implementation for AdamW update
fn scalar_adamw_update(
    grad: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    param: &mut [f32],
    beta1: f32,
    beta2: f32,
    lr: f32,
    lr_t: f32,
    weight_decay: f32,
    epsilon: f32,
) {
    for i in 0..grad.len() {
        m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
        param[i] = (1.0 - lr * weight_decay) * param[i] - lr_t * m[i] / (v[i].sqrt() + epsilon);
    }
}

proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(500))]

    #[test]
    fn prop_simd_axpy_matches_scalar(
        a in -10.0f32..10.0,
        x in prop::collection::vec(-100.0f32..100.0, 1..128),
    ) {
        let mut y_simd: Vec<f32> = (0..x.len()).map(|i| i as f32).collect();
        let mut y_scalar = y_simd.clone();

        simd_axpy(a, &x, &mut y_simd);
        scalar_axpy(a, &x, &mut y_scalar);

        for i in 0..x.len() {
            prop_assert!(
                (y_simd[i] - y_scalar[i]).abs() < 1e-4,
                "Mismatch at index {}: simd={} scalar={}",
                i, y_simd[i], y_scalar[i]
            );
        }
    }

    #[test]
    fn prop_simd_adam_matches_scalar(
        grad in prop::collection::vec(-10.0f32..10.0, 4..64),
        beta1 in 0.8f32..0.99,
        beta2 in 0.9f32..0.9999,
        lr_t in 0.0001f32..0.1,
    ) {
        let n = grad.len();
        let mut m_simd = vec![0.0f32; n];
        let mut v_simd = vec![0.0f32; n];
        let mut param_simd: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();

        let mut m_scalar = m_simd.clone();
        let mut v_scalar = v_simd.clone();
        let mut param_scalar = param_simd.clone();

        let epsilon = 1e-8;

        simd_adam_update(&grad, &mut m_simd, &mut v_simd, &mut param_simd, beta1, beta2, lr_t, epsilon);
        scalar_adam_update(&grad, &mut m_scalar, &mut v_scalar, &mut param_scalar, beta1, beta2, lr_t, epsilon);

        for i in 0..n {
            prop_assert!(
                (m_simd[i] - m_scalar[i]).abs() < 1e-4,
                "m mismatch at {}: simd={} scalar={}", i, m_simd[i], m_scalar[i]
            );
            prop_assert!(
                (v_simd[i] - v_scalar[i]).abs() < 1e-4,
                "v mismatch at {}: simd={} scalar={}", i, v_simd[i], v_scalar[i]
            );
            prop_assert!(
                (param_simd[i] - param_scalar[i]).abs() < 1e-3,
                "param mismatch at {}: simd={} scalar={}", i, param_simd[i], param_scalar[i]
            );
        }
    }

    #[test]
    fn prop_simd_adamw_matches_scalar(
        grad in prop::collection::vec(-10.0f32..10.0, 4..64),
        weight_decay in 0.0f32..0.1,
    ) {
        let n = grad.len();
        let mut m_simd = vec![0.0f32; n];
        let mut v_simd = vec![0.0f32; n];
        let mut param_simd: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.5).collect();

        let mut m_scalar = m_simd.clone();
        let mut v_scalar = v_simd.clone();
        let mut param_scalar = param_simd.clone();

        let beta1 = 0.9;
        let beta2 = 0.999;
        let lr = 0.001;
        let lr_t = 0.001;
        let epsilon = 1e-8;

        simd_adamw_update(&grad, &mut m_simd, &mut v_simd, &mut param_simd, beta1, beta2, lr, lr_t, weight_decay, epsilon);
        scalar_adamw_update(&grad, &mut m_scalar, &mut v_scalar, &mut param_scalar, beta1, beta2, lr, lr_t, weight_decay, epsilon);

        for i in 0..n {
            prop_assert!(
                (m_simd[i] - m_scalar[i]).abs() < 1e-4,
                "m mismatch at {}: simd={} scalar={}", i, m_simd[i], m_scalar[i]
            );
            prop_assert!(
                (v_simd[i] - v_scalar[i]).abs() < 1e-4,
                "v mismatch at {}: simd={} scalar={}", i, v_simd[i], v_scalar[i]
            );
            prop_assert!(
                (param_simd[i] - param_scalar[i]).abs() < 1e-3,
                "param mismatch at {}: simd={} scalar={}", i, param_simd[i], param_scalar[i]
            );
        }
    }

    #[test]
    fn prop_simd_axpy_various_sizes(
        size in 1usize..256
    ) {
        // Test various vector sizes to exercise SIMD boundaries
        let a = 2.5f32;
        let x: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
        let mut y: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let mut y_expected = y.clone();

        simd_axpy(a, &x, &mut y);
        scalar_axpy(a, &x, &mut y_expected);

        for i in 0..size {
            prop_assert!(
                (y[i] - y_expected[i]).abs() < 1e-4,
                "Size {} mismatch at {}", size, i
            );
        }
    }
}

// ========================================================================
// DETERMINISTIC UNIT TESTS
// ========================================================================

#[test]
fn test_simd_axpy() {
    let a = 2.0;
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let mut y = vec![10.0, 20.0, 30.0, 40.0];

    simd_axpy(a, &x, &mut y);

    // Expected: y = 2.0*x + y = [2, 4, 6, 8] + [10, 20, 30, 40] = [12, 24, 36, 48]
    assert_abs_diff_eq!(y[0], 12.0, epsilon = 1e-6);
    assert_abs_diff_eq!(y[1], 24.0, epsilon = 1e-6);
    assert_abs_diff_eq!(y[2], 36.0, epsilon = 1e-6);
    assert_abs_diff_eq!(y[3], 48.0, epsilon = 1e-6);
}

#[test]
fn test_simd_adam_update() {
    let grad = vec![1.0, -1.0, 2.0, -2.0];
    let mut m = vec![0.0, 0.0, 0.0, 0.0];
    let mut v = vec![0.0, 0.0, 0.0, 0.0];
    let mut param = vec![5.0, -3.0, 2.0, -7.0];

    let beta1 = 0.9;
    let beta2 = 0.999;
    let lr_t = 0.001;
    let epsilon = 1e-8;

    simd_adam_update(&grad, &mut m, &mut v, &mut param, beta1, beta2, lr_t, epsilon);

    // First moment should be (1 - 0.9) * grad = 0.1 * grad
    assert_abs_diff_eq!(m[0], 0.1, epsilon = 1e-6);
    assert_abs_diff_eq!(m[1], -0.1, epsilon = 1e-6);

    // Second moment should be (1 - 0.999) * grad² = 0.001 * grad²
    assert_abs_diff_eq!(v[0], 0.001, epsilon = 1e-6);
    assert_abs_diff_eq!(v[1], 0.001, epsilon = 1e-6);

    // Parameters should have moved (exact values depend on the computation)
    assert!(param[0] < 5.0, "Parameter should decrease for positive gradient");
    assert!(param[1] > -3.0, "Parameter should increase for negative gradient");
}

#[test]
fn test_simd_adamw_update() {
    let grad = vec![1.0, -1.0, 2.0, -2.0];
    let mut m = vec![0.0, 0.0, 0.0, 0.0];
    let mut v = vec![0.0, 0.0, 0.0, 0.0];
    let mut param = vec![5.0, -3.0, 2.0, -7.0];

    let beta1 = 0.9;
    let beta2 = 0.999;
    let lr = 0.001;
    let lr_t = 0.001;
    let weight_decay = 0.01;
    let epsilon = 1e-8;

    simd_adamw_update(
        &grad,
        &mut m,
        &mut v,
        &mut param,
        beta1,
        beta2,
        lr,
        lr_t,
        weight_decay,
        epsilon,
    );

    // First moment should be (1 - 0.9) * grad = 0.1 * grad
    assert_abs_diff_eq!(m[0], 0.1, epsilon = 1e-6);
    assert_abs_diff_eq!(m[1], -0.1, epsilon = 1e-6);

    // Second moment should be (1 - 0.999) * grad² = 0.001 * grad²
    assert_abs_diff_eq!(v[0], 0.001, epsilon = 1e-6);
    assert_abs_diff_eq!(v[1], 0.001, epsilon = 1e-6);

    // Weight decay should reduce parameter magnitudes
    assert!(param[0].abs() < 5.0, "Weight decay should reduce magnitude");
    assert!(param[3].abs() < 7.0, "Weight decay should reduce magnitude");
}

#[test]
fn test_simd_operations_consistent_with_scalar() {
    // Test that SIMD operations produce same results as scalar operations
    let a = 3.0;
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut y_simd = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let mut y_scalar = y_simd.clone();

    // SIMD version
    simd_axpy(a, &x, &mut y_simd);

    // Scalar version
    for i in 0..x.len() {
        y_scalar[i] += a * x[i];
    }

    // Should be identical
    for i in 0..x.len() {
        assert_abs_diff_eq!(y_simd[i], y_scalar[i], epsilon = 1e-5);
    }
}

#[test]
#[should_panic(expected = "Vector lengths must match")]
fn test_simd_axpy_length_mismatch() {
    let a = 2.0;
    let x = vec![1.0, 2.0, 3.0];
    let mut y = vec![10.0, 20.0]; // Wrong length!

    simd_axpy(a, &x, &mut y);
}

// ========================================================================
// EDGE CASE TESTS
// ========================================================================

#[test]
fn test_simd_axpy_single_element() {
    let a = 3.0;
    let x = vec![5.0];
    let mut y = vec![10.0];

    simd_axpy(a, &x, &mut y);

    assert_abs_diff_eq!(y[0], 25.0, epsilon = 1e-6); // 3*5 + 10 = 25
}

#[test]
fn test_simd_axpy_large_vector() {
    let size = 10000;
    let a = 0.5;
    let x: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let mut y: Vec<f32> = vec![1.0; size];

    simd_axpy(a, &x, &mut y);

    // Spot check some values
    assert_abs_diff_eq!(y[0], 1.0, epsilon = 1e-5); // 0.5*0 + 1 = 1
    assert_abs_diff_eq!(y[100], 51.0, epsilon = 1e-5); // 0.5*100 + 1 = 51
    assert_abs_diff_eq!(y[9999], 5000.5, epsilon = 1e-3); // 0.5*9999 + 1 = 5000.5
}

#[test]
fn test_simd_adam_multiple_steps() {
    // Test that multiple steps accumulate correctly
    let grad = vec![1.0, 1.0, 1.0, 1.0];
    let mut m = vec![0.0; 4];
    let mut v = vec![0.0; 4];
    let mut param = vec![10.0; 4];

    let beta1 = 0.9;
    let beta2 = 0.999;
    let lr_t = 0.1;
    let epsilon = 1e-8;

    // Run 10 steps
    for _ in 0..10 {
        simd_adam_update(&grad, &mut m, &mut v, &mut param, beta1, beta2, lr_t, epsilon);
    }

    // Momentum should have accumulated
    assert!(m[0] > 0.5, "Momentum should accumulate: {}", m[0]);

    // Parameters should have decreased
    assert!(param[0] < 10.0, "Parameters should decrease: {}", param[0]);

    // All values should be finite
    assert!(param.iter().all(|&p| p.is_finite()));
    assert!(m.iter().all(|&x| x.is_finite()));
    assert!(v.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_simd_adamw_weight_decay_effect() {
    // AdamW with weight decay should reduce parameter magnitudes
    let grad = vec![0.0; 4]; // Zero gradient - only weight decay acts
    let mut m = vec![0.0; 4];
    let mut v = vec![1e-6; 4]; // Small non-zero to avoid division issues
    let mut param = vec![10.0, 10.0, 10.0, 10.0];

    let beta1 = 0.9;
    let beta2 = 0.999;
    let lr = 0.1;
    let lr_t = 0.1;
    let weight_decay = 0.1;
    let epsilon = 1e-8;

    let initial_norm: f32 = param.iter().map(|x| x * x).sum();

    // Run several steps
    for _ in 0..10 {
        simd_adamw_update(
            &grad,
            &mut m,
            &mut v,
            &mut param,
            beta1,
            beta2,
            lr,
            lr_t,
            weight_decay,
            epsilon,
        );
    }

    let final_norm: f32 = param.iter().map(|x| x * x).sum();

    // Weight decay should have reduced magnitude
    assert!(
        final_norm < initial_norm,
        "Weight decay should reduce norm: {initial_norm} -> {final_norm}"
    );
}

#[test]
fn test_simd_operations_preserve_sign() {
    // Test that signs are preserved correctly
    let grad = vec![1.0, -1.0, 0.0, 2.0];
    let mut m = vec![0.0; 4];
    let mut v = vec![0.0; 4];
    let mut param = vec![0.0; 4];

    simd_adam_update(&grad, &mut m, &mut v, &mut param, 0.9, 0.999, 0.1, 1e-8);

    // Positive gradient -> negative param update
    assert!(param[0] < 0.0, "Positive grad should give negative update");
    // Negative gradient -> positive param update
    assert!(param[1] > 0.0, "Negative grad should give positive update");
}

#[test]
fn test_simd_numerical_stability_small_values() {
    // Test with very small gradients
    let grad = vec![1e-10; 8];
    let mut m = vec![0.0; 8];
    let mut v = vec![0.0; 8];
    let mut param = vec![1.0; 8];

    simd_adam_update(&grad, &mut m, &mut v, &mut param, 0.9, 0.999, 0.001, 1e-8);

    // Should not produce NaN or Inf
    assert!(param.iter().all(|&p| p.is_finite()));
    assert!(m.iter().all(|&x| x.is_finite()));
    assert!(v.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_simd_numerical_stability_large_values() {
    // Test with large gradients
    let grad = vec![1e6; 8];
    let mut m = vec![0.0; 8];
    let mut v = vec![0.0; 8];
    let mut param = vec![1.0; 8];

    simd_adam_update(&grad, &mut m, &mut v, &mut param, 0.9, 0.999, 0.001, 1e-8);

    // Should not produce NaN or Inf
    assert!(param.iter().all(|&p| p.is_finite()));
    assert!(m.iter().all(|&x| x.is_finite()));
    assert!(v.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_simd_axpy_zero_scalar() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let mut y = vec![10.0, 20.0, 30.0, 40.0];
    let y_original = y.clone();

    simd_axpy(0.0, &x, &mut y);

    // y should be unchanged when a=0
    for i in 0..y.len() {
        assert_abs_diff_eq!(y[i], y_original[i], epsilon = 1e-6);
    }
}

#[test]
fn test_simd_axpy_negative_scalar() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let mut y = vec![10.0, 20.0, 30.0, 40.0];

    simd_axpy(-2.0, &x, &mut y);

    // y = -2*x + y = [-2, -4, -6, -8] + [10, 20, 30, 40] = [8, 16, 24, 32]
    assert_abs_diff_eq!(y[0], 8.0, epsilon = 1e-6);
    assert_abs_diff_eq!(y[1], 16.0, epsilon = 1e-6);
    assert_abs_diff_eq!(y[2], 24.0, epsilon = 1e-6);
    assert_abs_diff_eq!(y[3], 32.0, epsilon = 1e-6);
}
