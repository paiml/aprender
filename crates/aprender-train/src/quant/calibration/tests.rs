//! Tests for PTQ calibration module

use super::*;
use crate::Tensor;
use approx::assert_abs_diff_eq;
use proptest::prelude::*;

// ========================================================================
// PROPERTY TESTS - Calibration correctness
// ========================================================================

proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(200))]

    /// Min-max calibration should capture the full range
    #[test]
    fn prop_min_max_captures_range(
        data in prop::collection::vec(-100.0f32..100.0, 10..100),
    ) {
        let result = calibrate_min_max(&data, 8, true);

        let actual_min = data.iter().copied().fold(f32::INFINITY, f32::min);
        let actual_max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        prop_assert!((result.observed_min - actual_min).abs() < 1e-5);
        prop_assert!((result.observed_max - actual_max).abs() < 1e-5);
    }

    /// Symmetric calibration should have zero_point = 0
    #[test]
    fn prop_symmetric_zero_point(
        data in prop::collection::vec(-10.0f32..10.0, 10..50),
        bits in 4usize..9,
    ) {
        let result = calibrate_min_max(&data, bits, true);
        prop_assert_eq!(result.zero_point, 0);
    }

    /// Scale should be positive and reasonable
    #[test]
    fn prop_scale_positive(
        data in prop::collection::vec(-10.0f32..10.0, 10..50),
        bits in 4usize..9,
    ) {
        let result = calibrate_min_max(&data, bits, true);

        prop_assert!(result.scale > 0.0);
        prop_assert!(result.scale < 1e10);
    }

    /// Percentile calibration should produce bounds within data range
    #[test]
    fn prop_percentile_within_range(
        data in prop::collection::vec(-10.0f32..10.0, 100..500),
    ) {
        let result = calibrate_percentile(&data, 8, true, 1.0, 99.0);

        let actual_min = data.iter().copied().fold(f32::INFINITY, f32::min);
        let actual_max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Percentile bounds should be within actual range
        prop_assert!(result.observed_min >= actual_min - 1e-5);
        prop_assert!(result.observed_max <= actual_max + 1e-5);
    }

    /// Multiple batch observation should accumulate correctly
    #[test]
    fn prop_multi_batch_accumulates(
        batch1 in prop::collection::vec(-5.0f32..5.0, 10..30),
        batch2 in prop::collection::vec(-10.0f32..10.0, 10..30),
    ) {
        let mut calibrator = Calibrator::min_max(8, true);
        calibrator.observe(&batch1);
        calibrator.observe(&batch2);

        let result = calibrator.compute();

        let all_data: Vec<f32> = batch1.iter().chain(batch2.iter()).copied().collect();
        let expected_min = all_data.iter().copied().fold(f32::INFINITY, f32::min);
        let expected_max = all_data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        prop_assert!((result.observed_min - expected_min).abs() < 1e-5);
        prop_assert!((result.observed_max - expected_max).abs() < 1e-5);
        prop_assert_eq!(calibrator.num_batches(), 2);
    }
}

// ========================================================================
// UNIT TESTS
// ========================================================================

#[test]
fn test_min_max_calibration() {
    let data = vec![0.0, 1.0, -2.0, 1.5, -1.5, 3.0];
    let result = calibrate_min_max(&data, 8, true);

    assert_abs_diff_eq!(result.observed_min, -2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(result.observed_max, 3.0, epsilon = 1e-6);
    assert_eq!(result.zero_point, 0);

    // Scale = max_abs / qmax = 3.0 / 127
    let expected_scale = 3.0 / 127.0;
    assert_abs_diff_eq!(result.scale, expected_scale, epsilon = 1e-6);
}

#[test]
fn test_percentile_calibration() {
    // Create data with outliers
    let mut data: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
    data.push(1000.0); // Outlier
    data.push(-1000.0); // Outlier

    let result = calibrate_percentile(&data, 8, true, 1.0, 99.0);

    // Percentile should ignore outliers
    // 1% of 102 ≈ 1, 99% ≈ 100
    // So bounds should be close to 0.1 and 9.9 (not -1000 and 1000)
    assert!(result.observed_min > -100.0, "Should ignore negative outlier");
    assert!(result.observed_max < 100.0, "Should ignore positive outlier");
}

#[test]
fn test_moving_average_calibration() {
    let mut calibrator = Calibrator::moving_average(8, true, 0.5);

    calibrator.observe(&[0.0, 1.0, -1.0]); // min=-1, max=1
    let r1 = calibrator.compute();
    assert_abs_diff_eq!(r1.observed_min, -1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(r1.observed_max, 1.0, epsilon = 1e-5);

    calibrator.observe(&[0.0, 2.0, -2.0]); // batch min=-2, max=2
    let r2 = calibrator.compute();
    // With momentum=0.5: new_min = -1*0.5 + -2*0.5 = -1.5
    assert_abs_diff_eq!(r2.observed_min, -1.5, epsilon = 1e-5);
    assert_abs_diff_eq!(r2.observed_max, 1.5, epsilon = 1e-5);
}

#[test]
fn test_asymmetric_calibration() {
    let data = vec![0.0, 1.0, 2.0, 3.0, 4.0]; // All positive
    let result = calibrate_min_max(&data, 8, false);

    assert_abs_diff_eq!(result.observed_min, 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(result.observed_max, 4.0, epsilon = 1e-6);

    // Asymmetric should have non-zero zero_point
    // scale = (4-0) / 255 ≈ 0.0157
    // zero_point = round(0 - 0/scale) = 0
    assert!(result.scale > 0.0);
}

#[test]
fn test_calibrator_reset() {
    let mut calibrator = Calibrator::min_max(8, true);
    calibrator.observe(&[1.0, 2.0, 3.0]);
    assert!(calibrator.has_data());

    calibrator.reset();
    assert!(!calibrator.has_data());
    assert_eq!(calibrator.num_batches(), 0);
}

#[test]
fn test_calibrator_observe_tensor() {
    let tensor = Tensor::from_vec(vec![0.0, 1.0, -1.0, 2.0], false);
    let mut calibrator = Calibrator::min_max(8, true);

    calibrator.observe_tensor(&tensor);

    let result = calibrator.compute();
    assert_abs_diff_eq!(result.observed_min, -1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(result.observed_max, 2.0, epsilon = 1e-6);
}

#[test]
fn test_calibration_method_default() {
    let method = CalibrationMethod::default();
    assert_eq!(method, CalibrationMethod::MinMax);
}

#[test]
fn test_calibration_with_zeros() {
    let data = vec![0.0; 100];
    let result = calibrate_min_max(&data, 8, true);

    // Should handle zero data without division by zero
    assert!(result.scale > 0.0);
    assert!(result.scale.is_finite());
}

#[test]
fn test_calibration_single_value() {
    let data = vec![5.0; 50];
    let result = calibrate_min_max(&data, 8, true);

    // Single value should work
    assert_abs_diff_eq!(result.observed_min, 5.0, epsilon = 1e-6);
    assert_abs_diff_eq!(result.observed_max, 5.0, epsilon = 1e-6);
    assert!(result.scale.is_finite());
}

#[test]
fn test_4bit_calibration() {
    let data = vec![0.0, 1.0, -1.0];
    let result = calibrate_min_max(&data, 4, true);

    // 4-bit symmetric: qmax = 7
    let expected_scale = 1.0 / 7.0;
    assert_abs_diff_eq!(result.scale, expected_scale, epsilon = 1e-6);
}

// =========================================================================
// Additional Coverage Tests
// =========================================================================

#[test]
fn test_calibrator_percentile_constructor() {
    let cal = Calibrator::percentile(8, true, 0.01, 99.99, 1000);
    assert!(cal.method() == &CalibrationMethod::Percentile { lower: 0.01, upper: 99.99 });
}

#[test]
fn test_calibrator_percentile_basic() {
    let mut cal = Calibrator::percentile(8, true, 1.0, 99.0, 10000);

    // Observe data with outliers
    let mut data: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
    data.push(-1000.0); // Outlier
    data.push(1000.0); // Outlier

    cal.observe(&data);
    let result = cal.compute();

    // Percentile should filter outliers
    assert!(result.observed_min > -100.0);
    assert!(result.observed_max < 100.0);
}

#[test]
fn test_calibrator_percentile_multiple_batches() {
    let mut cal = Calibrator::percentile(8, true, 0.01, 99.99, 1000);

    cal.observe(&[1.0, 2.0, 3.0]);
    cal.observe(&[4.0, 5.0, 6.0]);

    assert_eq!(cal.num_batches(), 2);
    assert!(cal.has_data());
}

#[test]
fn test_calibrator_moving_average_multiple_batches() {
    let mut cal = Calibrator::moving_average(8, true, 0.5);

    cal.observe(&[0.0, 1.0, -1.0]);
    let r1 = cal.compute();

    cal.observe(&[0.0, 4.0, -4.0]);
    let r2 = cal.compute();

    // Moving average should smooth the values
    assert!(r2.observed_max > r1.observed_max);
    assert!(r2.observed_max < 4.0);
}

#[test]
fn test_calibration_result_clone() {
    let result = CalibrationResult {
        scale: 0.1,
        zero_point: 0,
        observed_min: -1.0,
        observed_max: 1.0,
        method: CalibrationMethod::MinMax,
    };
    let cloned = result.clone();
    assert_abs_diff_eq!(result.scale, cloned.scale, epsilon = 1e-6);
}

#[test]
fn test_calibration_method_percentile_variant() {
    let method = CalibrationMethod::Percentile { lower: 0.1, upper: 99.9 };
    let cloned = method.clone();
    assert_eq!(method, cloned);
}

#[test]
fn test_calibration_method_moving_average_variant() {
    let method = CalibrationMethod::MovingAverage { momentum: 0.9 };
    let cloned = method.clone();
    assert_eq!(method, cloned);
}

#[test]
fn test_calibrator_asymmetric() {
    let mut cal = Calibrator::min_max(8, false); // Asymmetric
    cal.observe(&[0.0, 1.0, 2.0, 3.0, 4.0]);

    let result = cal.compute();
    // Asymmetric can have non-zero zero_point
    assert!(result.scale > 0.0);
}

#[test]
fn test_calibrator_empty_compute() {
    let cal = Calibrator::min_max(8, true);
    // Empty calibrator should still compute (with default values)
    let result = cal.compute();
    assert!(result.scale.is_finite());
}

#[test]
fn test_calibrate_min_max_helper() {
    let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let result = calibrate_min_max(&data, 8, true);

    assert_abs_diff_eq!(result.observed_min, -2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(result.observed_max, 2.0, epsilon = 1e-6);
}

#[test]
fn test_calibrate_percentile_helper() {
    let data: Vec<f32> = (0..1000).map(|i| i as f32 / 100.0).collect();
    let result = calibrate_percentile(&data, 8, true, 1.0, 99.0);

    // Should trim 1% from each end
    assert!(result.observed_min > 0.0);
    assert!(result.observed_max < 9.99);
}

#[test]
fn test_observe_tensors() {
    let t1 = Tensor::from_vec(vec![0.0, 1.0, 2.0], false);
    let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0], false);
    let mut cal = Calibrator::min_max(8, true);

    cal.observe_tensors(&[&t1, &t2]);

    let result = cal.compute();
    assert_abs_diff_eq!(result.observed_min, 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(result.observed_max, 5.0, epsilon = 1e-6);
}

#[test]
fn test_observe_empty_data() {
    let mut cal = Calibrator::min_max(8, true);
    cal.observe(&[]); // Empty data
    assert_eq!(cal.num_batches(), 0);
}

#[test]
fn test_calibrator_method_getter() {
    let cal = Calibrator::min_max(8, true);
    assert_eq!(cal.method(), &CalibrationMethod::MinMax);

    let cal2 = Calibrator::percentile(8, false, 1.0, 99.0, 1000);
    matches!(cal2.method(), CalibrationMethod::Percentile { .. });

    let cal3 = Calibrator::moving_average(8, true, 0.5);
    matches!(cal3.method(), CalibrationMethod::MovingAverage { .. });
}

#[test]
fn test_percentile_reservoir_sampling() {
    let mut cal = Calibrator::percentile(8, true, 0.01, 99.99, 10);

    // First batch fills reservoir
    cal.observe(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

    // Second batch triggers reservoir sampling
    cal.observe(&[11.0, 12.0, 13.0, 14.0, 15.0]);

    // Third batch also uses reservoir sampling
    cal.observe(&[21.0, 22.0, 23.0, 24.0, 25.0]);

    let result = cal.compute();
    assert!(result.scale > 0.0);
    assert_eq!(cal.num_batches(), 3);
}

#[test]
fn test_asymmetric_with_zero_point() {
    // Data with significant negative offset
    let data = vec![-5.0, -4.0, -3.0, -2.0, -1.0, 0.0];
    let result = calibrate_min_max(&data, 8, false);

    assert_abs_diff_eq!(result.observed_min, -5.0, epsilon = 1e-6);
    assert_abs_diff_eq!(result.observed_max, 0.0, epsilon = 1e-6);
    assert!(result.scale > 0.0);
}

#[test]
fn test_asymmetric_positive_only() {
    let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let result = calibrate_min_max(&data, 8, false);

    assert_abs_diff_eq!(result.observed_min, 10.0, epsilon = 1e-6);
    assert_abs_diff_eq!(result.observed_max, 50.0, epsilon = 1e-6);
    // scale = 40 / 255
    let expected_scale = 40.0 / 255.0;
    assert_abs_diff_eq!(result.scale, expected_scale, epsilon = 1e-6);
}

#[test]
fn test_percentile_empty_samples_fallback() {
    let cal = Calibrator::percentile(8, true, 1.0, 99.0, 0);
    // With max_samples = 0, no samples collected
    let result = cal.compute();
    assert!(result.scale.is_finite());
}

#[test]
fn test_calibrator_reset_full() {
    let mut cal = Calibrator::percentile(8, true, 1.0, 99.0, 1000);
    cal.observe(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    assert!(cal.has_data());

    cal.reset();
    assert!(!cal.has_data());
    assert_eq!(cal.num_batches(), 0);

    // Can observe again after reset
    cal.observe(&[10.0, 20.0, 30.0]);
    let result = cal.compute();
    assert_abs_diff_eq!(result.observed_min, 10.0, epsilon = 1e-6);
}

#[test]
fn test_calibration_result_method_field() {
    let result = calibrate_min_max(&[1.0, 2.0, 3.0], 8, true);
    assert_eq!(result.method, CalibrationMethod::MinMax);

    let result2 = calibrate_percentile(&[1.0, 2.0, 3.0], 8, true, 1.0, 99.0);
    matches!(result2.method, CalibrationMethod::Percentile { .. });
}

#[test]
fn test_moving_average_zero_momentum() {
    let mut cal = Calibrator::moving_average(8, true, 0.0);
    cal.observe(&[0.0, 1.0, -1.0]);
    let r1 = cal.compute();

    cal.observe(&[0.0, 10.0, -10.0]);
    let r2 = cal.compute();

    // With momentum=0, new values should be ignored
    assert_abs_diff_eq!(r1.observed_max, r2.observed_max, epsilon = 1e-5);
}

#[test]
fn test_moving_average_full_momentum() {
    let mut cal = Calibrator::moving_average(8, true, 1.0);
    cal.observe(&[0.0, 1.0, -1.0]);
    let r1 = cal.compute();
    assert_abs_diff_eq!(r1.observed_max, 1.0, epsilon = 1e-5);

    cal.observe(&[0.0, 10.0, -10.0]);
    let r2 = cal.compute();

    // With momentum=1, new values should fully replace old
    assert_abs_diff_eq!(r2.observed_max, 10.0, epsilon = 1e-5);
}
