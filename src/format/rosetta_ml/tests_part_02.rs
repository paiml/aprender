
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
