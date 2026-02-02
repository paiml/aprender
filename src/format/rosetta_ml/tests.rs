//\! Rosetta ML Tests - Extreme TDD
//\! PMAT-085: Extracted from mod.rs for PMAT file health compliance

use super::*;

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

#[test]
fn test_jidoka_violation_display() {
    let nan = JidokaViolation::NaN { count: 5 };
    assert!(format!("{nan}").contains("5"));

    let inf = JidokaViolation::Inf { count: 3 };
    assert!(format!("{inf}").contains("3"));

    let zero_var = JidokaViolation::ZeroVariance { mean: 2.5 };
    assert!(format!("{zero_var}").contains("2.5"));

    let shape = JidokaViolation::ShapeMismatch {
        expected: vec![512, 768],
        actual: vec![768, 512],
    };
    assert!(format!("{shape}").contains("512"));

    let checksum = JidokaViolation::ChecksumFailed {
        expected: 0xABCD,
        actual: 0x1234,
    };
    assert!(format!("{checksum}").contains("0xabcd"));
}

#[test]
fn test_conversion_decision_display() {
    let decision = ConversionDecision::QuantQ4_K_Block32;
    let s = format!("{decision}");
    assert!(s.contains("QuantQ4_K_Block32"));
}

#[test]
fn test_andon_level_display() {
    assert_eq!(format!("{}", AndonLevel::Green), "GREEN");
    assert_eq!(format!("{}", AndonLevel::Yellow), "YELLOW");
    assert_eq!(format!("{}", AndonLevel::Red), "RED");
}

#[test]
fn test_conversion_category_display() {
    assert!(format!("{}", ConversionCategory::GgufToApr).contains("GGUF"));
    assert!(format!("{}", ConversionCategory::SafeTensorsToGguf).contains("SafeTensors"));
}

#[test]
fn test_anomaly_detector_insufficient_data() {
    // Less than 13 samples = None
    let data: Vec<TensorFeatures> = (0..10)
        .map(|i| TensorFeatures::from_data(&[i as f32]))
        .collect();

    let detector = AnomalyDetector::fit(&data, 0.1, 10.0);
    assert!(detector.is_none());
}

#[test]
fn test_anomaly_detector_with_data() {
    // Generate 20 similar feature vectors
    let data: Vec<TensorFeatures> = (0..20)
        .map(|i| {
            let base = vec![1.0 + (i as f32 * 0.01); 100];
            TensorFeatures::from_data(&base)
        })
        .collect();

    let detector = AnomalyDetector::fit(&data, 0.1, 100.0);
    assert!(detector.is_some());

    let detector = detector.unwrap();
    assert_eq!(detector.n_samples(), 20);
    assert!((detector.shrinkage() - 0.1).abs() < 0.01);
    assert!((detector.threshold() - 100.0).abs() < 0.01);

    // Normal point should not be anomaly
    let normal = TensorFeatures::from_data(&[1.05; 100]);
    assert!(!detector.is_anomaly(&normal));

    // Extreme point should be anomaly
    let extreme = TensorFeatures::from_data(&[1000.0; 100]);
    assert!(detector.anomaly_score(&extreme) > 1.0);
}

#[test]
fn test_fix_action_variants() {
    // Test all FixAction variants can be created
    let _swap = FixAction::SwapDimensions;
    let _requant = FixAction::Requantize { block_size: 32 };
    let _checksum = FixAction::RecomputeChecksum;
    let _pad = FixAction::PadAlignment { alignment: 64 };
    let _skip = FixAction::SkipTensor;
    let _fallback = FixAction::FallbackF32;
    let _custom = FixAction::Custom {
        description: "test".into(),
    };
}

#[test]
fn test_pattern_source_variants() {
    assert_ne!(PatternSource::Bootstrap, PatternSource::Corpus);
    assert_ne!(PatternSource::Llm, PatternSource::Manual);
}

#[test]
fn test_severity_ordering() {
    assert!(Severity::Info < Severity::Warning);
    assert!(Severity::Warning < Severity::Error);
    assert!(Severity::Error < Severity::Critical);
}

#[test]
fn test_priority_ordering() {
    assert!(Priority::Low < Priority::Medium);
    assert!(Priority::Medium < Priority::High);
    assert!(Priority::High < Priority::Critical);
}

#[test]
fn test_matrix_inversion_singular() {
    // Singular matrix (all zeros)
    let singular = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
    assert!(invert_matrix(&singular).is_none());

    // Empty matrix
    let empty: Vec<Vec<f32>> = vec![];
    assert!(invert_matrix(&empty).is_none());

    // Non-square (not valid input)
    let nonsquare = vec![vec![1.0, 2.0]];
    assert!(invert_matrix(&nonsquare).is_none());
}

// =========================================================================
// Coverage: WilsonScore with low confidence (<0.90)
// =========================================================================

#[test]
fn test_wilson_score_low_confidence() {
    let score = WilsonScore::calculate(8, 10, 0.80); // Below 0.90 threshold
    assert!(score.proportion > 0.0);
    // With low confidence, the default z=1.96 should be used
    assert!(score.lower < score.proportion);
    assert!(score.upper > score.proportion);
}

#[test]
fn test_wilson_score_exact_boundaries() {
    // Test each z-score branch
    let score_99 = WilsonScore::calculate(5, 10, 0.99);
    assert!(score_99.confidence >= 0.99);

    let score_95 = WilsonScore::calculate(5, 10, 0.95);
    assert!((score_95.confidence - 0.95).abs() < 0.01);

    let score_90 = WilsonScore::calculate(5, 10, 0.90);
    assert!((score_90.confidence - 0.90).abs() < 0.01);
}

#[test]
fn test_wilson_score_zero_total() {
    let score = WilsonScore::calculate(0, 0, 0.95);
    assert_eq!(score.n, 0);
    assert!(score.proportion.abs() < 1e-6);
    assert!(score.lower.abs() < 1e-6);
    assert!(score.upper.abs() < 1e-6);
}

// =========================================================================
// Coverage: AndonLevel Display (all variants)
// =========================================================================

#[test]
fn test_andon_level_display_all_variants() {
    assert_eq!(format!("{}", AndonLevel::Green), "GREEN");
    assert_eq!(format!("{}", AndonLevel::Yellow), "YELLOW");
    assert_eq!(format!("{}", AndonLevel::Red), "RED");
}

// =========================================================================
// Coverage: WilsonScore andon_level
// =========================================================================

#[test]
fn test_wilson_score_andon_level_green() {
    let score = WilsonScore::calculate(9, 10, 0.95);
    assert_eq!(score.andon_level(0.8), AndonLevel::Green);
}

#[test]
fn test_wilson_score_andon_level_yellow() {
    let score = WilsonScore::calculate(6, 10, 0.95);
    assert_eq!(score.andon_level(0.8), AndonLevel::Yellow);
}

#[test]
fn test_wilson_score_andon_level_red() {
    let score = WilsonScore::calculate(1, 10, 0.95);
    assert_eq!(score.andon_level(0.8), AndonLevel::Red);
}

// =========================================================================
// Coverage: HanseiReport empty() and andon_level()
// =========================================================================

#[test]
fn test_hansei_report_empty() {
    let report = HanseiReport::empty();
    assert_eq!(report.total_attempts, 0);
    assert_eq!(report.successes, 0);
    assert!(report.success_rate.abs() < 1e-6);
    assert!(report.pareto_categories.is_empty());
    assert!(report.issues.is_empty());
}

#[test]
fn test_hansei_report_andon_level() {
    let results = vec![
        (ConversionCategory::GgufToApr, true),
        (ConversionCategory::GgufToApr, true),
        (ConversionCategory::GgufToApr, false),
    ];
    let report = HanseiReport::from_results(&results);
    // success_rate ~= 0.67
    let level = report.andon_level(0.9);
    assert_eq!(level, AndonLevel::Yellow);
}

#[test]
fn test_hansei_report_from_empty_results() {
    let results: Vec<(ConversionCategory, bool)> = vec![];
    let report = HanseiReport::from_results(&results);
    assert_eq!(report.total_attempts, 0);
}

// =========================================================================
// Coverage: ConversionCategory Display all variants
// =========================================================================

#[test]
fn test_conversion_category_display_all() {
    assert_eq!(
        format!("{}", ConversionCategory::GgufToApr),
        "GGUF\u{2192}APR"
    );
    assert_eq!(
        format!("{}", ConversionCategory::AprToGguf),
        "APR\u{2192}GGUF"
    );
    assert_eq!(
        format!("{}", ConversionCategory::SafeTensorsToApr),
        "SafeTensors\u{2192}APR"
    );
    assert_eq!(
        format!("{}", ConversionCategory::AprToSafeTensors),
        "APR\u{2192}SafeTensors"
    );
    assert_eq!(
        format!("{}", ConversionCategory::GgufToSafeTensors),
        "GGUF\u{2192}SafeTensors"
    );
    assert_eq!(
        format!("{}", ConversionCategory::SafeTensorsToGguf),
        "SafeTensors\u{2192}GGUF"
    );
}

// =========================================================================
// Coverage: ErrorPattern match_confidence
// =========================================================================

#[test]
fn test_error_pattern_match_confidence_zero() {
    let pattern = ErrorPattern::new(
        "test",
        vec!["alignment".into(), "padding".into()],
        FixAction::PadAlignment { alignment: 64 },
    );
    let conf = pattern.match_confidence("completely unrelated error message");
    assert!(conf.abs() < 1e-6, "No keyword matches should give 0.0");
}

#[test]
fn test_error_pattern_match_confidence_partial() {
    let pattern = ErrorPattern::new(
        "test",
        vec!["alignment".into(), "padding".into()],
        FixAction::PadAlignment { alignment: 64 },
    );
    let conf = pattern.match_confidence("alignment issue in tensor");
    assert!(
        (conf - 0.5).abs() < 1e-6,
        "One of two keywords should give 0.5"
    );
}

#[test]
fn test_error_pattern_match_confidence_full() {
    let pattern = ErrorPattern::new(
        "test",
        vec!["alignment".into(), "padding".into()],
        FixAction::PadAlignment { alignment: 64 },
    );
    let conf = pattern.match_confidence("alignment and padding issue");
    assert!((conf - 1.0).abs() < 1e-6, "All keywords should give 1.0");
}

// =========================================================================
// Coverage: ErrorPattern should_retire
// =========================================================================

#[test]
fn test_error_pattern_should_retire_not_enough_applications() {
    let pattern = ErrorPattern::new("test", vec!["error".into()], FixAction::SkipTensor);
    // < 5 applications, should not retire
    assert!(!pattern.should_retire());
}

#[test]
fn test_error_pattern_should_retire_low_success() {
    let mut pattern = ErrorPattern::new("test", vec!["error".into()], FixAction::SkipTensor);
    // 5 applications, 1 success = 20% < 30%
    for _ in 0..4 {
        pattern.record_application(false);
    }
    pattern.record_application(true);
    assert!(
        pattern.should_retire(),
        "Low success rate after 5 apps should retire"
    );
}

#[test]
fn test_error_pattern_should_not_retire_high_success() {
    let mut pattern = ErrorPattern::new("test", vec!["error".into()], FixAction::SkipTensor);
    // 5 applications, 4 successes = 80% > 30%
    for _ in 0..4 {
        pattern.record_application(true);
    }
    pattern.record_application(false);
    assert!(
        !pattern.should_retire(),
        "High success rate should not retire"
    );
}

// =========================================================================
// Coverage: TensorCanary detect_regression - RangeDrift
// =========================================================================

#[test]
fn test_tensor_canary_range_drift() {
    let baseline = TensorCanary {
        name: "test_tensor".into(),
        shape: vec![4, 4],
        dtype: "F32".into(),
        mean: 0.5,
        std: 0.1,
        min: 0.0,
        max: 1.0,
        checksum: 12345,
    };

    let current = TensorCanary {
        name: "test_tensor".into(),
        shape: vec![4, 4],
        dtype: "F32".into(),
        mean: 0.5, // same
        std: 0.1,  // same
        min: -0.5, // drifted below min - range_tolerance
        max: 1.0,  // same
        checksum: 12345,
    };

    let regression = baseline.detect_regression(&current);
    assert!(regression.is_some());
    match regression.unwrap() {
        Regression::RangeDrift { .. } => {} // Expected
        other => panic!("Expected RangeDrift, got {:?}", other),
    }
}

#[test]
fn test_tensor_canary_range_drift_max() {
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
        max: 2.0, // drifted above max + range_tolerance
        checksum: 100,
    };

    let regression = baseline.detect_regression(&current);
    assert!(regression.is_some());
    match regression.unwrap() {
        Regression::RangeDrift { .. } => {}
        other => panic!("Expected RangeDrift, got {:?}", other),
    }
}

#[test]
fn test_tensor_canary_no_regression() {
    let baseline = TensorCanary {
        name: "t".into(),
        shape: vec![2, 2],
        dtype: "F32".into(),
        mean: 0.5,
        std: 0.1,
        min: 0.0,
        max: 1.0,
        checksum: 42,
    };

    let current = baseline.clone();
    assert!(baseline.detect_regression(&current).is_none());
}

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
