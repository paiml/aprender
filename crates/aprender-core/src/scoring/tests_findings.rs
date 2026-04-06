use super::*;

#[test]
fn test_score_breakdown_debug() {
    let breakdown = ScoreBreakdown {
        criterion: "test".to_string(),
        score: 5.0,
        max: 10.0,
    };
    let debug_str = format!("{:?}", breakdown);
    assert!(debug_str.contains("ScoreBreakdown"));
}

#[test]
fn test_finding_warning_clone() {
    let warning = Finding::Warning {
        message: "test".to_string(),
        recommendation: "fix".to_string(),
    };
    let cloned = warning.clone();
    if let Finding::Warning { message, .. } = cloned {
        assert_eq!(message, "test");
    } else {
        panic!("Expected Warning variant");
    }
}

// =========================================================================
// Extended coverage tests - min_score, completion_ratio, efficiency, gaps
// =========================================================================

#[test]
fn test_grade_min_score_all_variants() {
    assert!((Grade::APlus.min_score() - 97.0).abs() < f32::EPSILON);
    assert!((Grade::A.min_score() - 93.0).abs() < f32::EPSILON);
    assert!((Grade::AMinus.min_score() - 90.0).abs() < f32::EPSILON);
    assert!((Grade::BPlus.min_score() - 87.0).abs() < f32::EPSILON);
    assert!((Grade::B.min_score() - 83.0).abs() < f32::EPSILON);
    assert!((Grade::BMinus.min_score() - 80.0).abs() < f32::EPSILON);
    assert!((Grade::CPlus.min_score() - 77.0).abs() < f32::EPSILON);
    assert!((Grade::C.min_score() - 73.0).abs() < f32::EPSILON);
    assert!((Grade::CMinus.min_score() - 70.0).abs() < f32::EPSILON);
    assert!((Grade::DPlus.min_score() - 67.0).abs() < f32::EPSILON);
    assert!((Grade::D.min_score() - 63.0).abs() < f32::EPSILON);
    assert!((Grade::DMinus.min_score() - 60.0).abs() < f32::EPSILON);
    assert!((Grade::F.min_score() - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_dimension_score_completion_ratio_zero_max() {
    let dim = DimensionScore::new(0.0);
    assert!((dim.completion_ratio() - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_dimension_score_completion_ratio_normal() {
    let mut dim = DimensionScore::new(10.0);
    dim.add_score("test", 7.5, 10.0);
    assert!((dim.completion_ratio() - 0.75).abs() < 0.01);
}

#[test]
fn test_score_breakdown_percentage_zero_max() {
    let breakdown = ScoreBreakdown {
        criterion: "test".to_string(),
        score: 5.0,
        max: 0.0,
    };
    assert!((breakdown.percentage() - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_acceptable_threshold_all_model_types() {
    // Classification models (0.8)
    assert!(
        (ScoredModelType::LogisticRegression.acceptable_threshold() - 0.8).abs() < f32::EPSILON
    );
    assert!((ScoredModelType::DecisionTree.acceptable_threshold() - 0.8).abs() < f32::EPSILON);
    assert!((ScoredModelType::RandomForest.acceptable_threshold() - 0.8).abs() < f32::EPSILON);
    assert!((ScoredModelType::GradientBoosting.acceptable_threshold() - 0.8).abs() < f32::EPSILON);
    assert!((ScoredModelType::Knn.acceptable_threshold() - 0.8).abs() < f32::EPSILON);
    assert!((ScoredModelType::NaiveBayes.acceptable_threshold() - 0.8).abs() < f32::EPSILON);
    assert!((ScoredModelType::Svm.acceptable_threshold() - 0.8).abs() < f32::EPSILON);

    // Clustering (0.5)
    assert!((ScoredModelType::KMeans.acceptable_threshold() - 0.5).abs() < f32::EPSILON);

    // Neural (0.1)
    assert!((ScoredModelType::NeuralSequential.acceptable_threshold() - 0.1).abs() < f32::EPSILON);
    assert!((ScoredModelType::NeuralCustom.acceptable_threshold() - 0.1).abs() < f32::EPSILON);

    // Regression and other (0.7)
    assert!((ScoredModelType::LinearRegression.acceptable_threshold() - 0.7).abs() < f32::EPSILON);
    assert!((ScoredModelType::Other.acceptable_threshold() - 0.7).abs() < f32::EPSILON);
}

#[test]
fn test_severity_display_all_variants() {
    assert_eq!(format!("{}", Severity::Medium), "MEDIUM");
    assert_eq!(format!("{}", Severity::High), "HIGH");
    assert_eq!(format!("{}", Severity::Critical), "CRITICAL");
}

#[test]
fn test_critical_issue_display() {
    let issue = CriticalIssue {
        severity: Severity::High,
        message: "Test issue".to_string(),
        action: "Fix this".to_string(),
    };
    let display = format!("{}", issue);
    assert!(display.contains("[HIGH]"));
    assert!(display.contains("Test issue"));
    assert!(display.contains("Action: Fix this"));
}

#[test]
fn test_quality_score_display_with_critical_issues() {
    let mut metadata = ModelMetadata::default();
    metadata
        .custom
        .insert("password".to_string(), "secret123".to_string());

    let config = ScoringConfig::default();
    let score = compute_quality_score(&metadata, &config);

    let display = format!("{}", score);
    assert!(display.contains("Critical Issues"));
}

#[test]
fn test_quality_score_display_with_warnings() {
    // Create metadata that generates warnings
    let mut metadata = ModelMetadata::default();
    metadata.model_type = Some(ScoredModelType::LinearRegression);
    // No regularization params -> warning

    let config = ScoringConfig::default();
    let score = compute_quality_score(&metadata, &config);

    // Should have warnings about missing info
    let display = format!("{}", score);
    assert!(display.contains("Warnings") || score.warning_count() > 0);
}

#[test]
fn test_compute_gap_score_all_branches() {
    // < 0.05 -> 5.0
    assert!((compute_gap_score(0.03) - 5.0).abs() < f32::EPSILON);
    // 0.05-0.1 -> 3.0
    assert!((compute_gap_score(0.07) - 3.0).abs() < f32::EPSILON);
    // 0.1-0.2 -> 1.0
    assert!((compute_gap_score(0.15) - 1.0).abs() < f32::EPSILON);
    // >= 0.2 -> 0.0
    assert!((compute_gap_score(0.25) - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_compute_efficiency_score_all_branches() {
    // < 0.1 -> 5.0
    assert!((compute_efficiency_score(0.05) - 5.0).abs() < f32::EPSILON);
    // 0.1-0.5 -> 4.0
    assert!((compute_efficiency_score(0.3) - 4.0).abs() < f32::EPSILON);
    // 0.5-1.0 -> 3.0
    assert!((compute_efficiency_score(0.7) - 3.0).abs() < f32::EPSILON);
    // 1.0-5.0 -> 2.0
    assert!((compute_efficiency_score(3.0) - 2.0).abs() < f32::EPSILON);
    // >= 5.0 -> 1.0
    assert!((compute_efficiency_score(10.0) - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_has_regularization_all_keys() {
    let mut params = HashMap::new();
    assert!(!has_regularization_params(&params));

    params.insert("alpha".to_string(), "0.1".to_string());
    assert!(has_regularization_params(&params));

    params.clear();
    params.insert("lambda".to_string(), "0.01".to_string());
    assert!(has_regularization_params(&params));

    params.clear();
    params.insert("l2_penalty".to_string(), "0.001".to_string());
    assert!(has_regularization_params(&params));

    params.clear();
    params.insert("weight_decay".to_string(), "0.0001".to_string());
    assert!(has_regularization_params(&params));
}

#[test]
fn test_high_cv_variance_warning() {
    let mut metadata = ModelMetadata::default();
    metadata.metrics.insert("cv_score_mean".to_string(), 0.85);
    metadata.metrics.insert("cv_score_std".to_string(), 0.15); // High variance

    let config = ScoringConfig {
        max_cv_std: 0.1,
        ..Default::default()
    };
    let score = compute_quality_score(&metadata, &config);

    assert!(score.findings.iter().any(|f| {
        if let Finding::Warning { message, .. } = f {
            message.contains("CV score variance")
        } else {
            false
        }
    }));
}

#[test]
fn test_input_bounds_documented() {
    let mut metadata = ModelMetadata::default();
    metadata
        .custom
        .insert("input_bounds".to_string(), "0.0-1.0".to_string());

    let config = ScoringConfig::default();
    let score = compute_quality_score(&metadata, &config);

    // Should get points for input validation
    assert!(score.dimensions.security_safety.score > 0.0);
}

#[test]
fn test_input_schema_documented() {
    let mut metadata = ModelMetadata::default();
    metadata
        .custom
        .insert("input_schema".to_string(), "json".to_string());

    let config = ScoringConfig::default();
    let score = compute_quality_score(&metadata, &config);

    // Should get points for input validation
    assert!(score.dimensions.security_safety.score > 0.0);
}

#[test]
fn test_feature_ranges_documented() {
    let mut metadata = ModelMetadata::default();
    metadata
        .custom
        .insert("feature_ranges".to_string(), "[[0,1],[0,100]]".to_string());

    let config = ScoringConfig::default();
    let score = compute_quality_score(&metadata, &config);

    // Should get points for input validation
    assert!(score.dimensions.security_safety.score > 0.0);
}

#[test]
fn test_secret_detection_variants() {
    let test_keys = [
        "password", "secret", "api_key", "token", "PASSWORD", "SECRET", "API_KEY", "TOKEN",
    ];

    for key in test_keys {
        let mut metadata = ModelMetadata::default();
        metadata.custom.insert(key.to_string(), "value".to_string());

        let config = ScoringConfig::default();
        let score = compute_quality_score(&metadata, &config);

        assert!(
            score.has_critical_issues(),
            "Expected critical issue for key: {}",
            key
        );
    }
}

#[test]
fn test_parameter_efficiency_high_ratio() {
    let metadata = ModelMetadata {
        n_parameters: Some(10000),
        training: Some(TrainingInfo {
            n_samples: Some(100), // Very few samples relative to params
            ..Default::default()
        }),
        ..Default::default()
    };

    let config = ScoringConfig::default();
    let score = compute_quality_score(&metadata, &config);

    // Should have info about high param count
    assert!(score.findings.iter().any(|f| {
        if let Finding::Info { message, .. } = f {
            message.contains("params/sample")
        } else {
            false
        }
    }));
}

#[test]
fn test_cv_score_low_std() {
    let mut metadata = ModelMetadata::default();
    metadata.metrics.insert("cv_score_mean".to_string(), 0.90);
    metadata.metrics.insert("cv_score_std".to_string(), 0.02); // Low variance - good

    let config = ScoringConfig::default();
    let score = compute_quality_score(&metadata, &config);

    // With low std, should get high CV score
    assert!(score
        .dimensions
        .accuracy_performance
        .breakdown
        .iter()
        .any(|b| b.criterion == "cross_validation" && (b.score - 8.0).abs() < f32::EPSILON));
}

#[test]
fn test_cv_score_medium_std() {
    let mut metadata = ModelMetadata::default();
    metadata.metrics.insert("cv_score_mean".to_string(), 0.90);
    metadata.metrics.insert("cv_score_std".to_string(), 0.07); // Medium variance

    let config = ScoringConfig::default();
    let score = compute_quality_score(&metadata, &config);

    // With medium std, should get 6.0 for CV
    assert!(score
        .dimensions
        .accuracy_performance
        .breakdown
        .iter()
        .any(|b| b.criterion == "cross_validation" && (b.score - 6.0).abs() < f32::EPSILON));
}

#[test]
fn test_cv_score_high_std_no_max_cv_warning() {
    let mut metadata = ModelMetadata::default();
    metadata.metrics.insert("cv_score_mean".to_string(), 0.90);
    metadata.metrics.insert("cv_score_std".to_string(), 0.12); // High variance but below config threshold

    let config = ScoringConfig {
        max_cv_std: 0.2, // Higher threshold
        ..Default::default()
    };
    let score = compute_quality_score(&metadata, &config);

    // With high std but below config threshold, should get 4.0 for CV but no warning
    assert!(score
        .dimensions
        .accuracy_performance
        .breakdown
        .iter()
        .any(|b| b.criterion == "cross_validation" && (b.score - 4.0).abs() < f32::EPSILON));
}

#[test]
fn test_scored_model_type_primary_metric_other() {
    assert_eq!(ScoredModelType::Other.primary_metric(), "primary_score");
}

#[test]
fn test_scored_model_type_neural_custom_properties() {
    assert!(ScoredModelType::NeuralCustom.needs_regularization());
    assert_eq!(ScoredModelType::NeuralCustom.interpretability_score(), 1.0);
    assert_eq!(ScoredModelType::NeuralCustom.primary_metric(), "loss");
}

#[test]
fn test_scored_model_type_svm_properties() {
    assert!(!ScoredModelType::Svm.needs_regularization());
    assert_eq!(ScoredModelType::Svm.interpretability_score(), 2.5);
    assert_eq!(ScoredModelType::Svm.primary_metric(), "accuracy");
}

#[test]
fn test_scored_model_type_knn_properties() {
    assert!(!ScoredModelType::Knn.needs_regularization());
    assert_eq!(ScoredModelType::Knn.interpretability_score(), 2.0);
    assert_eq!(ScoredModelType::Knn.primary_metric(), "accuracy");
}

#[test]
fn test_quality_score_with_all_training_provenance() {
    let metadata = ModelMetadata {
        training: Some(TrainingInfo {
            source: Some("data.csv".to_string()),
            n_samples: Some(1000),
            duration_ms: Some(5000),
            random_seed: Some(42),
            n_features: Some(10),
            test_size: Some(0.2),
        }),
        ..Default::default()
    };

    let config = ScoringConfig::default();
    let score = compute_quality_score(&metadata, &config);

    // Should have good provenance score
    assert!(score
        .dimensions
        .documentation_provenance
        .breakdown
        .iter()
        .any(|b| b.criterion == "training_provenance" && b.score >= 4.0));
}

#[test]
fn test_incomplete_training_provenance_warning() {
    let metadata = ModelMetadata {
        training: Some(TrainingInfo {
            source: None, // Missing source
            n_samples: None,
            duration_ms: None,
            random_seed: None,
            ..Default::default()
        }),
        ..Default::default()
    };

    let config = ScoringConfig::default();
    let score = compute_quality_score(&metadata, &config);

    assert!(score.findings.iter().any(|f| {
        if let Finding::Warning { message, .. } = f {
            message.contains("Incomplete training provenance")
        } else {
            false
        }
    }));
}
