//\! Rosetta ML Tests - Extreme TDD
//\! PMAT-085: Extracted from mod.rs for PMAT file health compliance

pub(crate) use super::*;

#[test]
fn test_tensor_features_basic() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let features = TensorFeatures::from_data(&data);

    assert!((features.mean - 3.0).abs() < 0.01);
    assert!(features.std > 0.0);
    assert!((features.min - 1.0).abs() < 0.01);
    assert!((features.max - 5.0).abs() < 0.01);
    assert!(features.nan_count == 0.0);
    assert!(features.inf_count == 0.0);
}

#[test]
fn test_tensor_features_with_nan() {
    let data = vec![1.0, f32::NAN, 3.0];
    let features = TensorFeatures::from_data(&data);

    assert!(features.nan_count == 1.0);
    assert!(features.has_jidoka_violation().is_some());
}

#[test]
fn test_tarantula_suspiciousness() {
    let mut tracker = TarantulaTracker::new();

    // Simulate: QuantQ4_K_Block32 fails often
    for _ in 0..10 {
        tracker.record_fail(&[ConversionDecision::QuantQ4_K_Block32]);
    }
    for _ in 0..2 {
        tracker.record_pass(&[ConversionDecision::QuantQ4_K_Block32]);
    }

    // Simulate: DtypeF32 almost always passes
    for _ in 0..10 {
        tracker.record_pass(&[ConversionDecision::DtypeF32]);
    }
    for _ in 0..1 {
        tracker.record_fail(&[ConversionDecision::DtypeF32]);
    }

    let q4k_sus = tracker.suspiciousness(ConversionDecision::QuantQ4_K_Block32);
    let f32_sus = tracker.suspiciousness(ConversionDecision::DtypeF32);

    // Q4_K should be more suspicious than F32
    assert!(q4k_sus > f32_sus, "Q4_K: {q4k_sus}, F32: {f32_sus}");
}

#[test]
fn test_wilson_score() {
    let score = WilsonScore::calculate(85, 100, 0.95);

    assert!((score.proportion - 0.85).abs() < 0.01);
    assert!(score.lower < 0.85);
    assert!(score.upper > 0.85);
    assert!(score.lower > 0.0);
    assert!(score.upper < 1.0);
}

#[test]
fn test_wilson_andon_levels() {
    let green = WilsonScore::calculate(95, 100, 0.95);
    let yellow = WilsonScore::calculate(70, 100, 0.95);
    let red = WilsonScore::calculate(40, 100, 0.95);

    assert_eq!(green.andon_level(0.90), AndonLevel::Green);
    assert_eq!(yellow.andon_level(0.90), AndonLevel::Yellow);
    assert_eq!(red.andon_level(0.90), AndonLevel::Red);
}

#[test]
fn test_error_pattern_matching() {
    let mut lib = ErrorPatternLibrary::bootstrap();

    let error = "Dimension mismatch: expected [512, 768], got [768, 512]";
    let pattern = lib.find_match(error);

    assert!(pattern.is_some());
    assert_eq!(pattern.expect("pattern exists").id, "COL_MAJOR_GHOST");
}

#[test]
fn test_pattern_retirement() {
    let mut pattern = ErrorPattern::new("TEST_PATTERN", vec!["test".into()], FixAction::SkipTensor);

    // 5 applications, only 1 success = 20% success rate
    for i in 0..5 {
        pattern.record_application(i == 0);
    }

    assert!(pattern.should_retire());
}

#[test]
fn test_hansei_pareto() {
    let results = vec![
        (ConversionCategory::GgufToApr, false),
        (ConversionCategory::GgufToApr, false),
        (ConversionCategory::GgufToApr, false),
        (ConversionCategory::GgufToApr, false),
        (ConversionCategory::AprToGguf, true),
        (ConversionCategory::SafeTensorsToApr, false),
        (ConversionCategory::SafeTensorsToApr, true),
    ];

    let report = HanseiReport::from_results(&results);

    // GgufToApr has 4 failures out of 5 total failures (80%)
    assert!(report
        .pareto_categories
        .contains(&ConversionCategory::GgufToApr));
}

#[test]
fn test_tensor_canary_regression() {
    let original = TensorCanary::from_data(
        "layer.0.weight",
        vec![512, 768],
        "f32",
        &vec![0.1; 512 * 768],
    );

    // Same data - no regression
    let same = TensorCanary::from_data(
        "layer.0.weight",
        vec![512, 768],
        "f32",
        &vec![0.1; 512 * 768],
    );
    assert!(original.detect_regression(&same).is_none());

    // Different shape - regression
    let diff_shape = TensorCanary::from_data(
        "layer.0.weight",
        vec![768, 512], // Swapped!
        "f32",
        &vec![0.1; 512 * 768],
    );
    let regression = original.detect_regression(&diff_shape);
    assert!(matches!(regression, Some(Regression::ShapeMismatch { .. })));
}

#[test]
fn test_matrix_inversion() {
    // Simple 2x2 matrix
    let matrix = vec![vec![4.0, 7.0], vec![2.0, 6.0]];

    let inv = invert_matrix(&matrix).expect("invertible");

    // Verify A * A^-1 ≈ I
    for i in 0..2 {
        for j in 0..2 {
            let mut sum = 0.0;
            for k in 0..2 {
                sum += matrix[i][k] * inv[k][j];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (sum - expected).abs() < 0.01,
                "({i},{j}): {sum} != {expected}"
            );
        }
    }
}

// ========================================================================
// Additional Coverage Tests
// ========================================================================

#[test]
fn test_tensor_features_empty() {
    let features = TensorFeatures::from_data(&[]);
    assert_eq!(features.mean, 0.0);
    assert_eq!(features.std, 0.0);
    assert!(features.has_jidoka_violation().is_none());
}

#[test]
fn test_tensor_features_with_inf() {
    let data = vec![1.0, f32::INFINITY, 3.0];
    let features = TensorFeatures::from_data(&data);

    assert_eq!(features.inf_count, 1.0);
    let violation = features.has_jidoka_violation();
    assert!(matches!(violation, Some(JidokaViolation::Inf { count: 1 })));
}

#[test]
fn test_tensor_features_zero_variance() {
    let data = vec![5.0, 5.0, 5.0, 5.0];
    let features = TensorFeatures::from_data(&data);

    assert_eq!(features.std, 0.0);
    assert_eq!(features.mean, 5.0);
    // Zero variance with non-zero mean triggers Jidoka
    let violation = features.has_jidoka_violation();
    assert!(matches!(
        violation,
        Some(JidokaViolation::ZeroVariance { .. })
    ));
}

#[test]
fn test_tensor_features_to_vec() {
    let data = vec![1.0, 2.0, 3.0];
    let features = TensorFeatures::from_data(&data);
    let vec = features.to_vec();

    assert_eq!(vec.len(), 12);
    assert!((vec[0] - features.mean).abs() < 1e-6);
    assert!((vec[1] - features.std).abs() < 1e-6);
}

#[test]
fn test_decision_stats_pass_rate() {
    let mut stats = DecisionStats::default();
    assert_eq!(stats.pass_rate(), 0.0);

    stats.passed = 8;
    stats.failed = 2;
    assert!((stats.pass_rate() - 0.8).abs() < 0.01);
}

#[test]
fn test_tarantula_ranked_suspiciousness() {
    let mut tracker = TarantulaTracker::new();

    tracker.record_fail(&[ConversionDecision::QuantQ4_0_Block32]);
    tracker.record_fail(&[ConversionDecision::QuantQ4_0_Block32]);
    tracker.record_pass(&[ConversionDecision::DtypeF32]);
    tracker.record_pass(&[ConversionDecision::DtypeF32]);

    let ranked = tracker.ranked_suspiciousness();
    assert!(!ranked.is_empty());
    // Q4_0 should be ranked higher (more failures)
    assert_eq!(ranked[0].0, ConversionDecision::QuantQ4_0_Block32);
}

#[test]
fn test_tarantula_priority() {
    let mut tracker = TarantulaTracker::new();

    // 25% failure rate = Critical
    for _ in 0..75 {
        tracker.record_pass(&[ConversionDecision::QuantQ4_K_Block256]);
    }
    for _ in 0..25 {
        tracker.record_fail(&[ConversionDecision::QuantQ4_K_Block256]);
    }
    assert_eq!(
        tracker.priority(ConversionDecision::QuantQ4_K_Block256),
        Priority::Critical
    );

    // 3% failure rate = Low
    for _ in 0..97 {
        tracker.record_pass(&[ConversionDecision::DtypeF16]);
    }
    for _ in 0..3 {
        tracker.record_fail(&[ConversionDecision::DtypeF16]);
    }
    assert_eq!(
        tracker.priority(ConversionDecision::DtypeF16),
        Priority::Low
    );

    // Unknown decision = Low
    assert_eq!(
        tracker.priority(ConversionDecision::VocabMerge),
        Priority::Low
    );
}

#[test]
fn test_tarantula_empty() {
    let tracker = TarantulaTracker::new();
    // No data = 0 suspiciousness
    assert_eq!(tracker.suspiciousness(ConversionDecision::DtypeF32), 0.0);
}

#[test]
fn test_wilson_score_edge_cases() {
    // Zero total
    let zero = WilsonScore::calculate(0, 0, 0.95);
    assert_eq!(zero.proportion, 0.0);

    // 100% success
    let perfect = WilsonScore::calculate(100, 100, 0.95);
    assert!((perfect.proportion - 1.0).abs() < 0.01);

    // Different confidence levels
    let high_conf = WilsonScore::calculate(50, 100, 0.99);
    let low_conf = WilsonScore::calculate(50, 100, 0.90);
    assert!(high_conf.upper - high_conf.lower > low_conf.upper - low_conf.lower);
}

#[test]
fn test_error_pattern_library_hit_rate() {
    let mut lib = ErrorPatternLibrary::bootstrap();

    // No queries yet
    assert_eq!(lib.hit_rate(), 0.0);

    // Make some queries
    let _ = lib.find_match("dimension mismatch error");
    let _ = lib.find_match("unknown error type");

    // At least one should match
    assert!(lib.hit_rate() > 0.0);
}

#[test]
fn test_error_pattern_library_retire() {
    let mut lib = ErrorPatternLibrary::new();

    let mut bad_pattern =
        ErrorPattern::new("BAD_PATTERN", vec!["bad".into()], FixAction::SkipTensor);
    // 5 applications, 0 successes = should retire
    for _ in 0..5 {
        bad_pattern.record_application(false);
    }
    lib.add_pattern(bad_pattern);

    lib.add_pattern(ErrorPattern::new(
        "GOOD_PATTERN",
        vec!["good".into()],
        FixAction::SkipTensor,
    ));

    lib.retire_failing_patterns();
    // BAD_PATTERN should be retired
    assert!(lib.find_match("bad error").is_none());
}

#[test]
fn test_error_pattern_library_record_result() {
    let mut lib = ErrorPatternLibrary::bootstrap();

    // Record some results
    lib.record_result("COL_MAJOR_GHOST", true);
    lib.record_result("COL_MAJOR_GHOST", false);
    lib.record_result("NONEXISTENT", true); // Should be ignored
}

#[test]
fn test_hansei_empty_results() {
    let report = HanseiReport::from_results(&[]);
    assert_eq!(report.total_attempts, 0);
    assert_eq!(report.successes, 0);
    assert!(report.pareto_categories.is_empty());
}

#[test]
fn test_hansei_all_success() {
    let results = vec![
        (ConversionCategory::GgufToApr, true),
        (ConversionCategory::AprToGguf, true),
    ];

    let report = HanseiReport::from_results(&results);
    assert_eq!(report.success_rate, 1.0);
    assert!(report.pareto_categories.is_empty()); // No failures
    assert_eq!(report.andon_level(0.90), AndonLevel::Green);
}

#[test]
fn test_canary_file_operations() {
    let mut canary_file = CanaryFile::new("test_model");

    let tensor = TensorCanary::from_data(
        "layer.0.weight",
        vec![128, 256],
        "f32",
        &vec![0.5; 128 * 256],
    );
    canary_file.add_tensor(tensor.clone());

    // Verify with same data - no regression
    let current = vec![TensorCanary::from_data(
        "layer.0.weight",
        vec![128, 256],
        "f32",
        &vec![0.5; 128 * 256],
    )];
    let regressions = canary_file.verify(&current);
    assert!(regressions.is_empty());

    // Verify with missing tensor - should report
    let empty: Vec<TensorCanary> = vec![];
    let regressions = canary_file.verify(&empty);
    assert_eq!(regressions.len(), 1);
}

#[test]
fn test_canary_mean_drift() {
    let original = TensorCanary::from_data("test", vec![100], "f32", &vec![1.0; 100]);

    // Mean drifted by > 1%
    let drifted = TensorCanary::from_data("test", vec![100], "f32", &vec![1.1; 100]);

    let regression = original.detect_regression(&drifted);
    assert!(matches!(regression, Some(Regression::MeanDrift { .. })));
}

#[test]
fn test_canary_std_drift() {
    let original = TensorCanary::from_data(
        "test",
        vec![10],
        "f32",
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    );

    // Std drifted by > 5% (uniform vs spread data)
    let drifted = TensorCanary::from_data(
        "test",
        vec![10],
        "f32",
        &[5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9],
    );

    let regression = original.detect_regression(&drifted);
    // Either StdDrift or MeanDrift depending on magnitude
    assert!(regression.is_some());
}

#[test]
fn test_canary_checksum_mismatch() {
    let original = TensorCanary::from_data("test", vec![100], "f32", &vec![1.0; 100]);

    // Same stats but different actual data
    let mut different = original.clone();
    different.checksum = 0xDEADBEEF;

    let regression = original.detect_regression(&different);
    assert!(matches!(
        regression,
        Some(Regression::ChecksumMismatch { .. })
    ));
}

#[path = "tests_jidoka.rs"]
mod tests_jidoka;
#[path = "tests_canary.rs"]
mod tests_canary;
