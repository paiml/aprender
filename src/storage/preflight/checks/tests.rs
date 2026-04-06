//! Tests for preflight checks.

use super::*;

// =========================================================================
// PreflightCheck Tests - No NaN
// =========================================================================

#[test]
fn test_no_nan_values_passes() {
    let check = PreflightCheck::no_nan_values();
    let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let result = check.run(&data, &PreflightContext::default());
    assert!(result.is_passed());
}

#[test]
fn test_no_nan_values_fails() {
    let check = PreflightCheck::no_nan_values();
    let data = vec![vec![1.0, f64::NAN], vec![3.0, 4.0]];
    let result = check.run(&data, &PreflightContext::default());
    assert!(result.is_failed());
}

#[test]
fn test_no_nan_values_empty_data() {
    let check = PreflightCheck::no_nan_values();
    let data: Vec<Vec<f64>> = vec![];
    let result = check.run(&data, &PreflightContext::default());
    assert!(result.is_passed());
}

// =========================================================================
// PreflightCheck Tests - No Inf
// =========================================================================

#[test]
fn test_no_inf_values_passes() {
    let check = PreflightCheck::no_inf_values();
    let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let result = check.run(&data, &PreflightContext::default());
    assert!(result.is_passed());
}

#[test]
fn test_no_inf_values_fails_positive() {
    let check = PreflightCheck::no_inf_values();
    let data = vec![vec![1.0, f64::INFINITY], vec![3.0, 4.0]];
    let result = check.run(&data, &PreflightContext::default());
    assert!(result.is_failed());
}

#[test]
fn test_no_inf_values_fails_negative() {
    let check = PreflightCheck::no_inf_values();
    let data = vec![vec![1.0, f64::NEG_INFINITY]];
    let result = check.run(&data, &PreflightContext::default());
    assert!(result.is_failed());
}

// =========================================================================
// PreflightCheck Tests - Min Samples
// =========================================================================

#[test]
fn test_min_samples_passes() {
    let check = PreflightCheck::min_samples(2);
    let data = vec![vec![1.0], vec![2.0], vec![3.0]];
    let result = check.run(&data, &PreflightContext::default());
    assert!(result.is_passed());
}

#[test]
fn test_min_samples_fails() {
    let check = PreflightCheck::min_samples(10);
    let data = vec![vec![1.0], vec![2.0]];
    let result = check.run(&data, &PreflightContext::default());
    assert!(result.is_failed());
}

#[test]
fn test_min_samples_uses_context() {
    let check = PreflightCheck::min_samples(2);
    let data = vec![vec![1.0], vec![2.0]];
    let ctx = PreflightContext::new().with_min_samples(5);
    let result = check.run(&data, &ctx);
    assert!(result.is_failed()); // Context overrides default
}

// =========================================================================
// PreflightCheck Tests - Min Features
// =========================================================================

#[test]
fn test_min_features_passes() {
    let check = PreflightCheck::min_features(2);
    let data = vec![vec![1.0, 2.0, 3.0]];
    let result = check.run(&data, &PreflightContext::default());
    assert!(result.is_passed());
}

#[test]
fn test_min_features_fails() {
    let check = PreflightCheck::min_features(5);
    let data = vec![vec![1.0, 2.0]];
    let result = check.run(&data, &PreflightContext::default());
    assert!(result.is_failed());
}

// =========================================================================
// PreflightCheck Tests - Consistent Dimensions
// =========================================================================

#[test]
fn test_consistent_dimensions_passes() {
    let check = PreflightCheck::consistent_dimensions();
    let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
    let result = check.run(&data, &PreflightContext::default());
    assert!(result.is_passed());
}

#[test]
fn test_consistent_dimensions_fails() {
    let check = PreflightCheck::consistent_dimensions();
    let data = vec![vec![1.0, 2.0], vec![3.0, 4.0, 5.0]];
    let result = check.run(&data, &PreflightContext::default());
    assert!(result.is_failed());
}

#[test]
fn test_consistent_dimensions_empty() {
    let check = PreflightCheck::consistent_dimensions();
    let data: Vec<Vec<f64>> = vec![];
    let result = check.run(&data, &PreflightContext::default());
    assert!(result.is_skipped());
}

// =========================================================================
// PreflightCheck Tests - No Constant Features
// =========================================================================

#[test]
fn test_no_constant_features_passes() {
    let check = PreflightCheck::no_constant_features();
    let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let result = check.run(&data, &PreflightContext::default());
    assert!(result.is_passed());
}

#[test]
fn test_no_constant_features_warns() {
    let check = PreflightCheck::no_constant_features();
    let data = vec![vec![1.0, 2.0], vec![1.0, 4.0]]; // First column constant
    let result = check.run(&data, &PreflightContext::default());
    assert!(result.is_warning());
}

// =========================================================================
// PreflightCheck Tests - Label Balance
// =========================================================================

#[test]
fn test_label_balance_passes() {
    let check = PreflightCheck::label_balance(2.0);
    // Last column is label: 2 samples of class 0, 2 of class 1
    let data = vec![vec![1.0, 0.0], vec![2.0, 0.0], vec![3.0, 1.0], vec![4.0, 1.0]];
    let result = check.run(&data, &PreflightContext::default());
    assert!(result.is_passed());
}

#[test]
fn test_label_balance_warns() {
    let check = PreflightCheck::label_balance(2.0);
    // Imbalanced: 1 sample class 0, 5 samples class 1
    let data = vec![
        vec![1.0, 0.0],
        vec![2.0, 1.0],
        vec![3.0, 1.0],
        vec![4.0, 1.0],
        vec![5.0, 1.0],
        vec![6.0, 1.0],
    ];
    let result = check.run(&data, &PreflightContext::default());
    assert!(result.is_warning());
}

// =========================================================================
// Environment Check Tests (may be skipped on some systems)
// =========================================================================

#[test]
fn test_disk_space_check() {
    let check = PreflightCheck::disk_space_mb(1);
    let result = check.run(&[], &PreflightContext::default());
    // Should pass or be a valid result
    assert!(result.is_passed() || result.is_failed() || result.is_skipped());
}

#[test]
fn test_memory_check() {
    let check = PreflightCheck::memory_mb(1);
    let result = check.run(&[], &PreflightContext::default());
    assert!(result.is_passed() || result.is_failed() || result.is_skipped());
}

#[test]
fn test_gpu_available_check() {
    let check = PreflightCheck::gpu_available();
    let result = check.run(&[], &PreflightContext::default());
    // May pass or warn depending on system
    assert!(result.is_passed() || result.is_warning());
}

// =========================================================================
// Property Tests
// =========================================================================

use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_no_nan_passes_for_valid_data(
        rows in 1usize..100,
        cols in 1usize..10
    ) {
        let data: Vec<Vec<f64>> = (0..rows)
            .map(|i| (0..cols).map(|j| (i * cols + j) as f64).collect())
            .collect();

        let check = PreflightCheck::no_nan_values();
        let result = check.run(&data, &PreflightContext::default());
        prop_assert!(result.is_passed());
    }

    #[test]
    fn prop_consistent_dimensions_passes_for_rectangular(
        rows in 1usize..50,
        cols in 1usize..10
    ) {
        let data: Vec<Vec<f64>> = (0..rows)
            .map(|_| vec![0.0; cols])
            .collect();

        let check = PreflightCheck::consistent_dimensions();
        let result = check.run(&data, &PreflightContext::default());
        prop_assert!(result.is_passed());
    }

    #[test]
    fn prop_min_samples_respects_threshold(
        actual in 0usize..100,
        required in 1usize..50
    ) {
        let data: Vec<Vec<f64>> = (0..actual).map(|_| vec![1.0]).collect();
        let check = PreflightCheck::min_samples(required);
        let result = check.run(&data, &PreflightContext::default());

        if actual >= required {
            prop_assert!(result.is_passed());
        } else {
            prop_assert!(result.is_failed());
        }
    }
}
