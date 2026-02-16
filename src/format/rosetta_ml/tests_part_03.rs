use super::*;

// =========================================================================
// Coverage: TensorCanary - shape mismatch
// =========================================================================

#[test]
fn test_tensor_canary_shape_mismatch() {
    let baseline = TensorCanary {
        name: "t".into(),
        shape: vec![2, 2],
        dtype: "F32".into(),
        mean: 0.0,
        std: 0.1,
        min: -1.0,
        max: 1.0,
        checksum: 0,
    };

    let current = TensorCanary {
        name: "t".into(),
        shape: vec![3, 3], // different shape
        dtype: "F32".into(),
        mean: 0.0,
        std: 0.1,
        min: -1.0,
        max: 1.0,
        checksum: 0,
    };

    match baseline.detect_regression(&current).unwrap() {
        Regression::ShapeMismatch { .. } => {}
        other => panic!("Expected ShapeMismatch, got {:?}", other),
    }
}

// =========================================================================
// Coverage: TensorCanary - checksum mismatch
// =========================================================================

#[test]
fn test_tensor_canary_checksum_mismatch() {
    let baseline = TensorCanary {
        name: "t".into(),
        shape: vec![2],
        dtype: "F32".into(),
        mean: 0.0,
        std: 0.1,
        min: -1.0,
        max: 1.0,
        checksum: 100,
    };

    let current = TensorCanary {
        name: "t".into(),
        shape: vec![2],
        dtype: "F32".into(),
        mean: 0.0,
        std: 0.1,
        min: -1.0,
        max: 1.0,
        checksum: 999, // different checksum
    };

    match baseline.detect_regression(&current).unwrap() {
        Regression::ChecksumMismatch { .. } => {}
        other => panic!("Expected ChecksumMismatch, got {:?}", other),
    }
}

// =========================================================================
// Coverage: CanaryFile new, add_tensor, verify
// =========================================================================

#[test]
fn test_canary_file_new() {
    let canary = CanaryFile::new("test-model");
    assert_eq!(canary.model_name, "test-model");
    assert!(canary.tensors.is_empty());
    assert!(!canary.created_at.is_empty());
}

#[test]
fn test_canary_file_verify_missing_tensor() {
    let mut canary = CanaryFile::new("model");
    canary.add_tensor(TensorCanary {
        name: "missing_tensor".into(),
        shape: vec![2, 2],
        dtype: "F32".into(),
        mean: 0.0,
        std: 0.1,
        min: -1.0,
        max: 1.0,
        checksum: 0,
    });

    // Verify against empty list of current tensors
    let regressions = canary.verify(&[]);
    assert_eq!(regressions.len(), 1);
    assert_eq!(regressions[0].0, "missing_tensor");
}

#[test]
fn test_canary_file_verify_no_regressions() {
    let tensor = TensorCanary {
        name: "t".into(),
        shape: vec![2],
        dtype: "F32".into(),
        mean: 0.5,
        std: 0.1,
        min: 0.0,
        max: 1.0,
        checksum: 42,
    };

    let mut canary = CanaryFile::new("model");
    canary.add_tensor(tensor.clone());

    let regressions = canary.verify(&[tensor]);
    assert!(regressions.is_empty());
}

// =========================================================================
// Coverage: Trend and Regression Debug/Clone
// =========================================================================

#[test]
fn test_trend_variants_debug() {
    let trends = [
        Trend::Improving,
        Trend::Stable,
        Trend::Degrading,
        Trend::Oscillating,
    ];
    for t in &trends {
        let debug = format!("{:?}", t);
        assert!(!debug.is_empty());
    }
}

#[test]
fn test_regression_debug_clone() {
    let r = Regression::MeanDrift {
        expected: 1.0,
        actual: 2.0,
        error: 1.0,
    };
    let debug = format!("{:?}", r);
    assert!(debug.contains("MeanDrift"));
    let cloned = r.clone();
    let debug2 = format!("{:?}", cloned);
    assert_eq!(debug, debug2);
}

// =========================================================================
// Coverage: HanseiReport from_results with failures
// =========================================================================

#[test]
fn test_hansei_report_with_mixed_results() {
    let results = vec![
        (ConversionCategory::GgufToApr, true),
        (ConversionCategory::GgufToApr, false),
        (ConversionCategory::AprToGguf, true),
        (ConversionCategory::SafeTensorsToApr, false),
        (ConversionCategory::SafeTensorsToApr, false),
    ];
    let report = HanseiReport::from_results(&results);
    assert_eq!(report.total_attempts, 5);
    assert_eq!(report.successes, 2);
    assert!(!report.pareto_categories.is_empty());
    assert!(report.success_rate > 0.0 && report.success_rate < 1.0);
}

#[test]
fn test_hansei_report_all_success() {
    let results = vec![
        (ConversionCategory::GgufToApr, true),
        (ConversionCategory::AprToGguf, true),
    ];
    let report = HanseiReport::from_results(&results);
    assert!((report.success_rate - 1.0).abs() < 1e-6);
    assert!(report.pareto_categories.is_empty()); // No failures -> no Pareto
}
