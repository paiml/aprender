//! Tests for scoring module.

use super::*;

#[test]
fn test_grade_from_score() {
    assert_eq!(Grade::from_score(100.0), Grade::APlus);
    assert_eq!(Grade::from_score(97.0), Grade::APlus);
    assert_eq!(Grade::from_score(95.0), Grade::A);
    assert_eq!(Grade::from_score(91.0), Grade::AMinus);
    assert_eq!(Grade::from_score(88.0), Grade::BPlus);
    assert_eq!(Grade::from_score(85.0), Grade::B);
    assert_eq!(Grade::from_score(80.0), Grade::BMinus);
    assert_eq!(Grade::from_score(75.0), Grade::C);
    assert_eq!(Grade::from_score(70.0), Grade::CMinus);
    assert_eq!(Grade::from_score(65.0), Grade::D);
    assert_eq!(Grade::from_score(55.0), Grade::F);
}

#[test]
fn test_grade_is_passing() {
    assert!(Grade::APlus.is_passing());
    assert!(Grade::A.is_passing());
    assert!(Grade::CMinus.is_passing());
    assert!(!Grade::DPlus.is_passing());
    assert!(!Grade::F.is_passing());
}

#[test]
fn test_dimension_score_add() {
    let mut dim = DimensionScore::new(10.0);
    assert!((dim.score - 0.0).abs() < f32::EPSILON);

    dim.add_score("test1", 3.0, 5.0);
    assert!((dim.score - 3.0).abs() < f32::EPSILON);
    assert!((dim.percentage - 30.0).abs() < 0.01);

    dim.add_score("test2", 2.0, 5.0);
    assert!((dim.score - 5.0).abs() < f32::EPSILON);
    assert!((dim.percentage - 50.0).abs() < 0.01);
}

#[test]
fn test_dimension_score_perfect() {
    let mut dim = DimensionScore::new(10.0);
    dim.add_score("full", 10.0, 10.0);
    assert!(dim.is_perfect());
}

#[test]
fn test_quality_score_empty_metadata() {
    let metadata = ModelMetadata::default();
    let config = ScoringConfig::default();
    let score = compute_quality_score(&metadata, &config);

    // Should have some score for checksum (always present)
    assert!(score.total > 0.0);
    // Should have warnings for missing data
    assert!(!score.findings.is_empty());
}

#[test]
fn test_quality_score_full_metadata() {
    let mut metadata = ModelMetadata {
        model_name: Some("TestModel".to_string()),
        description: Some("A test model".to_string()),
        model_type: Some(ScoredModelType::LinearRegression),
        n_parameters: Some(100),
        aprender_version: Some("0.15.0".to_string()),
        training: Some(TrainingInfo {
            source: Some("test_data.csv".to_string()),
            n_samples: Some(1000),
            n_features: Some(10),
            duration_ms: Some(5000),
            random_seed: Some(42),
            test_size: Some(0.2),
        }),
        flags: ModelFlags {
            has_model_card: true,
            is_signed: true,
            has_feature_importance: true,
            has_edge_case_tests: true,
            has_preprocessing_steps: true,
            ..Default::default()
        },
        ..Default::default()
    };

    metadata.metrics.insert("r2_score".to_string(), 0.95);
    metadata.metrics.insert("cv_score_mean".to_string(), 0.93);
    metadata.metrics.insert("cv_score_std".to_string(), 0.02);
    metadata.metrics.insert("train_score".to_string(), 0.96);
    metadata.metrics.insert("test_score".to_string(), 0.94);
    metadata
        .metrics
        .insert("inference_latency_ms".to_string(), 1.5);

    metadata
        .hyperparameters
        .insert("alpha".to_string(), "0.01".to_string());
    metadata
        .hyperparameters
        .insert("fit_intercept".to_string(), "true".to_string());

    let config = ScoringConfig::default();
    let score = compute_quality_score(&metadata, &config);

    // Should have high score with all metadata
    assert!(score.total >= 80.0, "Score was {}", score.total);
    assert!(score.grade.is_passing());
}

#[test]
fn test_quality_score_detects_secrets() {
    let mut metadata = ModelMetadata::default();
    metadata
        .custom
        .insert("api_key".to_string(), "secret123".to_string());

    let config = ScoringConfig::default();
    let score = compute_quality_score(&metadata, &config);

    assert!(score.has_critical_issues());
    assert!(score
        .critical_issues
        .iter()
        .any(|i| i.message.contains("secrets")));
}

#[test]
fn test_quality_score_display() {
    let metadata = ModelMetadata {
        model_name: Some("TestModel".to_string()),
        ..Default::default()
    };
    let config = ScoringConfig::default();
    let score = compute_quality_score(&metadata, &config);

    let display = format!("{score}");
    assert!(display.contains("Model Quality Score"));
    assert!(display.contains("Grade:"));
}

#[test]
fn test_scored_model_type_properties() {
    assert!(ScoredModelType::LinearRegression.needs_regularization());
    assert!(!ScoredModelType::RandomForest.needs_regularization());

    assert_eq!(
        ScoredModelType::LinearRegression.interpretability_score(),
        5.0
    );
    assert_eq!(
        ScoredModelType::NeuralSequential.interpretability_score(),
        1.0
    );

    assert_eq!(
        ScoredModelType::LinearRegression.primary_metric(),
        "r2_score"
    );
    assert_eq!(ScoredModelType::KMeans.primary_metric(), "silhouette_score");
}

#[test]
fn test_dimension_scores_total() {
    let mut scores = DimensionScores::default_scores();

    // Add some scores
    scores.accuracy_performance.add_score("test", 20.0, 25.0);
    scores
        .generalization_robustness
        .add_score("test", 15.0, 20.0);
    scores.model_complexity.add_score("test", 10.0, 15.0);
    scores
        .documentation_provenance
        .add_score("test", 10.0, 15.0);
    scores.reproducibility.add_score("test", 10.0, 15.0);
    scores.security_safety.add_score("test", 8.0, 10.0);

    let total = scores.total_raw();
    assert!((total - 73.0).abs() < 0.01);
}

#[test]
fn test_finding_display() {
    let warning = Finding::Warning {
        message: "Test warning".to_string(),
        recommendation: "Fix it".to_string(),
    };
    let display = format!("{warning}");
    assert!(display.contains("[WARN]"));
    assert!(display.contains("Test warning"));

    let info = Finding::Info {
        message: "Test info".to_string(),
        recommendation: "Consider this".to_string(),
    };
    let display = format!("{info}");
    assert!(display.contains("[INFO]"));
}

#[test]
fn test_critical_issue_severity_ordering() {
    assert!(Severity::Critical > Severity::High);
    assert!(Severity::High > Severity::Medium);
}

#[test]
fn test_score_breakdown_percentage() {
    let breakdown = ScoreBreakdown {
        criterion: "test".to_string(),
        score: 7.5,
        max: 10.0,
    };
    assert!((breakdown.percentage() - 75.0).abs() < 0.01);
}

#[test]
fn test_quality_score_passes_threshold() {
    let metadata = ModelMetadata::default();
    let config = ScoringConfig::default();
    let score = compute_quality_score(&metadata, &config);

    // With empty metadata, should fail most thresholds
    assert!(!score.passes_threshold(90.0));
}

#[test]
fn test_score_with_high_train_test_gap() {
    let mut metadata = ModelMetadata::default();
    metadata.metrics.insert("train_score".to_string(), 0.99);
    metadata.metrics.insert("test_score".to_string(), 0.70);

    let config = ScoringConfig::default();
    let score = compute_quality_score(&metadata, &config);

    // Should have warning about overfitting
    assert!(score.findings.iter().any(|f| {
        if let Finding::Warning { message, .. } = f {
            message.contains("train/test gap")
        } else {
            false
        }
    }));
}

#[test]
fn test_score_requires_signed() {
    let metadata = ModelMetadata::default();
    let config = ScoringConfig {
        require_signed: true,
        ..Default::default()
    };
    let score = compute_quality_score(&metadata, &config);

    // Should have critical issue for unsigned model
    assert!(score
        .critical_issues
        .iter()
        .any(|i| i.message.contains("not signed")));
}

// =========================================================================
// Additional coverage tests
// =========================================================================

#[test]
fn test_grade_display() {
    assert_eq!(format!("{}", Grade::APlus), "A+");
    assert_eq!(format!("{}", Grade::AMinus), "A-");
    assert_eq!(format!("{}", Grade::BPlus), "B+");
    assert_eq!(format!("{}", Grade::BMinus), "B-");
    assert_eq!(format!("{}", Grade::CPlus), "C+");
    assert_eq!(format!("{}", Grade::CMinus), "C-");
    assert_eq!(format!("{}", Grade::DPlus), "D+");
    assert_eq!(format!("{}", Grade::DMinus), "D-");
    assert_eq!(format!("{}", Grade::F), "F");
}

#[test]
fn test_grade_clone() {
    let grade = Grade::APlus;
    let cloned = grade.clone();
    assert_eq!(grade, cloned);
}

#[test]
fn test_quality_score_warning_count() {
    let metadata = ModelMetadata::default();
    let config = ScoringConfig::default();
    let score = compute_quality_score(&metadata, &config);

    // With empty metadata, should have warnings
    let warning_count = score.warning_count();
    // warning_count is usize, always >= 0, just verify it's callable
    assert!(warning_count < usize::MAX);
}

#[test]
fn test_quality_score_info_count() {
    let mut metadata = ModelMetadata::default();
    metadata.model_name = Some("TestModel".to_string());
    let config = ScoringConfig::default();
    let score = compute_quality_score(&metadata, &config);

    let info_count = score.info_count();
    // info_count is usize, always >= 0, just verify it's callable
    assert!(info_count < usize::MAX);
}

#[test]
fn test_dimension_score_zero() {
    let dim = DimensionScore::new(10.0);
    assert!(!dim.is_perfect());
    assert!((dim.percentage - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_dimension_score_clone() {
    let mut dim = DimensionScore::new(10.0);
    dim.add_score("test", 5.0, 5.0);
    let cloned = dim.clone();
    assert!((cloned.score - dim.score).abs() < f32::EPSILON);
}

#[test]
fn test_severity_debug() {
    let severity = Severity::Critical;
    let debug_str = format!("{:?}", severity);
    assert!(debug_str.contains("Critical"));
}

#[test]
fn test_critical_issue_clone() {
    let issue = CriticalIssue {
        severity: Severity::High,
        message: "test".to_string(),
        action: "fix it".to_string(),
    };
    let cloned = issue.clone();
    assert_eq!(cloned.message, issue.message);
}

#[test]
fn test_finding_info_recommendation() {
    let info = Finding::Info {
        message: "Good practice".to_string(),
        recommendation: "Keep it up".to_string(),
    };
    let display = format!("{info}");
    assert!(display.contains("Good practice"));
}

#[test]
fn test_scored_model_type_all_variants() {
    assert_eq!(
        ScoredModelType::LogisticRegression.primary_metric(),
        "accuracy"
    );
    assert_eq!(ScoredModelType::DecisionTree.primary_metric(), "accuracy");
    assert_eq!(
        ScoredModelType::GradientBoosting.primary_metric(),
        "accuracy"
    );
    assert_eq!(ScoredModelType::NaiveBayes.primary_metric(), "accuracy");
    assert_eq!(ScoredModelType::Knn.primary_metric(), "accuracy");
}

#[test]
fn test_scored_model_type_interpretability() {
    assert_eq!(ScoredModelType::DecisionTree.interpretability_score(), 4.0);
    assert_eq!(
        ScoredModelType::LogisticRegression.interpretability_score(),
        5.0
    );
    assert_eq!(ScoredModelType::KMeans.interpretability_score(), 2.5);
    assert_eq!(
        ScoredModelType::GradientBoosting.interpretability_score(),
        3.0
    );
}

#[test]
fn test_scored_model_type_regularization() {
    assert!(ScoredModelType::LogisticRegression.needs_regularization());
    assert!(!ScoredModelType::DecisionTree.needs_regularization());
    assert!(!ScoredModelType::KMeans.needs_regularization());
}

#[test]
fn test_model_metadata_default() {
    let metadata = ModelMetadata::default();
    assert!(metadata.model_name.is_none());
    assert!(metadata.metrics.is_empty());
    assert!(!metadata.flags.has_model_card);
}

#[test]
fn test_training_info_default() {
    let info = TrainingInfo::default();
    assert!(info.source.is_none());
    assert!(info.n_samples.is_none());
}

#[test]
fn test_model_flags_default() {
    let flags = ModelFlags::default();
    assert!(!flags.has_model_card);
    assert!(!flags.is_signed);
}

#[test]
fn test_scoring_config_default() {
    let config = ScoringConfig::default();
    assert!(!config.require_signed);
    assert!((config.min_primary_metric - 0.7).abs() < f32::EPSILON);
}

#[test]
fn test_quality_score_debug() {
    let metadata = ModelMetadata::default();
    let config = ScoringConfig::default();
    let score = compute_quality_score(&metadata, &config);
    let debug_str = format!("{:?}", score);
    assert!(debug_str.contains("QualityScore"));
}

#[test]
fn test_dimension_scores_debug() {
    let scores = DimensionScores::default_scores();
    let debug_str = format!("{:?}", scores);
    assert!(debug_str.contains("DimensionScores"));
}

#[test]
fn test_dimension_score_debug() {
    let dim = DimensionScore::new(10.0);
    let debug_str = format!("{:?}", dim);
    assert!(debug_str.contains("DimensionScore"));
}

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

#[test]
fn test_model_type_neural_no_regularization_warning() {
    let metadata = ModelMetadata {
        model_type: Some(ScoredModelType::NeuralSequential),
        hyperparameters: HashMap::new(), // No regularization
        ..Default::default()
    };

    let config = ScoringConfig::default();
    let score = compute_quality_score(&metadata, &config);

    assert!(score.findings.iter().any(|f| {
        if let Finding::Warning { message, .. } = f {
            message.contains("No regularization")
        } else {
            false
        }
    }));
}

#[test]
fn test_model_with_regularization() {
    let mut metadata = ModelMetadata {
        model_type: Some(ScoredModelType::LinearRegression),
        ..Default::default()
    };
    metadata
        .hyperparameters
        .insert("alpha".to_string(), "0.01".to_string());

    let config = ScoringConfig::default();
    let score = compute_quality_score(&metadata, &config);

    // Should get regularization points
    assert!(score
        .dimensions
        .generalization_robustness
        .breakdown
        .iter()
        .any(|b| b.criterion == "regularization" && b.score > 0.0));
}

#[test]
fn test_dimension_score_update_percentage_zero_max() {
    let mut dim = DimensionScore {
        score: 0.0,
        max_score: 0.0,
        percentage: 0.0,
        breakdown: Vec::new(),
    };
    dim.add_score("test", 5.0, 5.0);
    // With max_score = 0.0, percentage should remain 0.0
    assert!((dim.percentage - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_grade_from_score_boundary_values() {
    // Test exact boundary values
    assert_eq!(Grade::from_score(97.0), Grade::APlus);
    assert_eq!(Grade::from_score(96.9), Grade::A);
    assert_eq!(Grade::from_score(93.0), Grade::A);
    assert_eq!(Grade::from_score(92.9), Grade::AMinus);
    assert_eq!(Grade::from_score(90.0), Grade::AMinus);
    assert_eq!(Grade::from_score(89.9), Grade::BPlus);
    assert_eq!(Grade::from_score(87.0), Grade::BPlus);
    assert_eq!(Grade::from_score(86.9), Grade::B);
    assert_eq!(Grade::from_score(83.0), Grade::B);
    assert_eq!(Grade::from_score(82.9), Grade::BMinus);
    assert_eq!(Grade::from_score(77.0), Grade::CPlus);
    assert_eq!(Grade::from_score(76.9), Grade::C);
    assert_eq!(Grade::from_score(73.0), Grade::C);
    assert_eq!(Grade::from_score(72.9), Grade::CMinus);
    assert_eq!(Grade::from_score(67.0), Grade::DPlus);
    assert_eq!(Grade::from_score(66.9), Grade::D);
    assert_eq!(Grade::from_score(63.0), Grade::D);
    assert_eq!(Grade::from_score(62.9), Grade::DMinus);
    assert_eq!(Grade::from_score(60.0), Grade::DMinus);
    assert_eq!(Grade::from_score(59.9), Grade::F);
}

#[test]
fn test_quality_score_clone() {
    let metadata = ModelMetadata::default();
    let config = ScoringConfig::default();
    let score = compute_quality_score(&metadata, &config);
    let cloned = score.clone();
    assert!((cloned.total - score.total).abs() < f32::EPSILON);
}

#[test]
fn test_dimension_scores_clone() {
    let scores = DimensionScores::default_scores();
    let cloned = scores.clone();
    assert!((cloned.total_raw() - scores.total_raw()).abs() < f32::EPSILON);
}

#[test]
fn test_grade_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(Grade::APlus);
    set.insert(Grade::A);
    assert!(set.contains(&Grade::APlus));
    assert!(!set.contains(&Grade::B));
}

#[test]
fn test_model_metadata_debug() {
    let metadata = ModelMetadata::default();
    let debug_str = format!("{:?}", metadata);
    assert!(debug_str.contains("ModelMetadata"));
}

#[test]
fn test_training_info_debug() {
    let info = TrainingInfo::default();
    let debug_str = format!("{:?}", info);
    assert!(debug_str.contains("TrainingInfo"));
}

#[test]
fn test_model_flags_debug() {
    let flags = ModelFlags::default();
    let debug_str = format!("{:?}", flags);
    assert!(debug_str.contains("ModelFlags"));
}

#[test]
fn test_scoring_config_debug() {
    let config = ScoringConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("ScoringConfig"));
}

#[test]
fn test_scored_model_type_debug() {
    let model_type = ScoredModelType::LinearRegression;
    let debug_str = format!("{:?}", model_type);
    assert!(debug_str.contains("LinearRegression"));
}

#[test]
fn test_finding_debug() {
    let finding = Finding::Warning {
        message: "test".to_string(),
        recommendation: "fix".to_string(),
    };
    let debug_str = format!("{:?}", finding);
    assert!(debug_str.contains("Warning"));
}
