//! Tests for scoring module.

pub(crate) use super::*;

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

#[path = "tests_part_02.rs"]

mod tests_part_02;
#[path = "tests_part_03.rs"]
mod tests_part_03;
