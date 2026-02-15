
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
