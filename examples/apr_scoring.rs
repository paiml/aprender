//! APR 100-Point Quality Scoring Example
//!
//! Demonstrates the comprehensive model quality scoring system that evaluates
//! models across six dimensions based on ML best practices:
//!
//! | Dimension | Max Points | Toyota Way Principle |
//! |-----------|-----------|---------------------|
//! | Accuracy & Performance | 25 | Kaizen (continuous improvement) |
//! | Generalization & Robustness | 20 | Jidoka (quality built-in) |
//! | Model Complexity | 15 | Muda elimination (waste reduction) |
//! | Documentation & Provenance | 15 | Genchi Genbutsu (go and see) |
//! | Reproducibility | 15 | Standardization |
//! | Security & Safety | 10 | Poka-yoke (error-proofing) |
//!
//! Run with: `cargo run --example apr_scoring`

use aprender::scoring::{
    compute_quality_score, Finding, Grade, ModelFlags, ModelMetadata, ScoredModelType,
    ScoringConfig, TrainingInfo,
};

fn main() {
    println!("=== 100-Point Model Quality Scoring Demo ===\n");

    // Part 1: Grade System
    grade_system_demo();

    // Part 2: Model Types
    model_types_demo();

    // Part 3: Basic Scoring
    basic_scoring_demo();

    // Part 4: Full Metadata Scoring
    full_metadata_scoring_demo();

    // Part 5: Security Issues Detection
    security_issues_demo();

    // Part 6: Scoring Config
    scoring_config_demo();

    println!("\n=== Scoring Demo Complete! ===");
}

fn grade_system_demo() {
    println!("--- Part 1: Grade System ---\n");

    let test_scores = [
        100.0, 97.0, 95.0, 91.0, 88.0, 85.0, 81.0, 78.0, 75.0, 71.0, 68.0, 65.0, 55.0,
    ];

    println!(
        "{:>8} {:>6} {:>10} {:>10}",
        "Score", "Grade", "Min Score", "Passing"
    );
    println!("{}", "-".repeat(40));

    for score in test_scores {
        let grade = Grade::from_score(score);
        println!(
            "{:>8.1} {:>6} {:>10.1} {:>10}",
            score,
            format!("{}", grade),
            grade.min_score(),
            if grade.is_passing() { "Yes" } else { "No" }
        );
    }
    println!();
}

fn model_types_demo() {
    println!("--- Part 2: Model Types ---\n");

    let types = [
        ScoredModelType::LinearRegression,
        ScoredModelType::LogisticRegression,
        ScoredModelType::DecisionTree,
        ScoredModelType::RandomForest,
        ScoredModelType::GradientBoosting,
        ScoredModelType::Knn,
        ScoredModelType::KMeans,
        ScoredModelType::NaiveBayes,
        ScoredModelType::NeuralSequential,
        ScoredModelType::Svm,
    ];

    println!(
        "{:<20} {:>12} {:>15} {:>12} {:>10}",
        "Model Type", "Interpretable", "Primary Metric", "Threshold", "Needs Reg"
    );
    println!("{}", "-".repeat(75));

    for model_type in &types {
        println!(
            "{:<20} {:>12.1} {:>15} {:>12.2} {:>10}",
            format!("{:?}", model_type),
            model_type.interpretability_score(),
            model_type.primary_metric(),
            model_type.acceptable_threshold(),
            if model_type.needs_regularization() {
                "Yes"
            } else {
                "No"
            }
        );
    }
    println!();
}

fn basic_scoring_demo() {
    println!("--- Part 3: Basic Scoring ---\n");

    // Empty metadata (worst case)
    let empty_metadata = ModelMetadata::default();
    let config = ScoringConfig::default();
    let score = compute_quality_score(&empty_metadata, &config);

    println!("Empty Metadata Score:");
    println!("  Total: {:.1}/100 (Grade: {})", score.total, score.grade);
    println!("  Warnings: {}", score.warning_count());
    println!("  Info: {}", score.info_count());
    println!("  Critical Issues: {}", score.critical_issues.len());
    println!("  Passes 70% threshold: {}", score.passes_threshold(70.0));

    // Minimal metadata
    let mut minimal = ModelMetadata {
        model_name: Some("BasicModel".to_string()),
        model_type: Some(ScoredModelType::LinearRegression),
        ..Default::default()
    };
    minimal.metrics.insert("r2_score".to_string(), 0.85);

    let minimal_score = compute_quality_score(&minimal, &config);
    println!("\nMinimal Metadata Score:");
    println!(
        "  Total: {:.1}/100 (Grade: {})",
        minimal_score.total, minimal_score.grade
    );
    println!();
}

fn full_metadata_scoring_demo() {
    println!("--- Part 4: Full Metadata Scoring ---\n");

    // Create comprehensive metadata
    let mut metadata = ModelMetadata {
        model_name: Some("IrisRandomForest".to_string()),
        description: Some("Random Forest classifier for Iris species prediction".to_string()),
        model_type: Some(ScoredModelType::RandomForest),
        n_parameters: Some(5000),
        aprender_version: Some("0.15.0".to_string()),
        training: Some(TrainingInfo {
            source: Some("iris_dataset.csv".to_string()),
            n_samples: Some(150),
            n_features: Some(4),
            duration_ms: Some(2500),
            random_seed: Some(42),
            test_size: Some(0.2),
        }),
        flags: ModelFlags {
            has_model_card: true,
            is_signed: true,
            is_encrypted: false,
            has_feature_importance: true,
            has_edge_case_tests: true,
            has_preprocessing_steps: true,
        },
        ..Default::default()
    };

    // Add comprehensive metrics
    metadata.metrics.insert("accuracy".to_string(), 0.967);
    metadata.metrics.insert("cv_score_mean".to_string(), 0.953);
    metadata.metrics.insert("cv_score_std".to_string(), 0.025);
    metadata.metrics.insert("train_score".to_string(), 0.985);
    metadata.metrics.insert("test_score".to_string(), 0.967);
    metadata
        .metrics
        .insert("inference_latency_ms".to_string(), 0.5);
    metadata.metrics.insert("f1_score".to_string(), 0.965);
    metadata.metrics.insert("precision".to_string(), 0.968);
    metadata.metrics.insert("recall".to_string(), 0.962);

    // Add hyperparameters
    metadata
        .hyperparameters
        .insert("n_estimators".to_string(), "100".to_string());
    metadata
        .hyperparameters
        .insert("max_depth".to_string(), "10".to_string());
    metadata
        .hyperparameters
        .insert("min_samples_split".to_string(), "2".to_string());
    metadata
        .hyperparameters
        .insert("random_state".to_string(), "42".to_string());
    metadata
        .hyperparameters
        .insert("criterion".to_string(), "gini".to_string());

    // Add custom metadata
    metadata.custom.insert(
        "input_bounds".to_string(),
        "[[4.3,7.9],[2.0,4.4],[1.0,6.9],[0.1,2.5]]".to_string(),
    );

    let config = ScoringConfig::default();
    let score = compute_quality_score(&metadata, &config);

    // Display full score breakdown
    println!("{}", score);

    println!("\nDimension Details:");
    println!(
        "  Accuracy: {:.1}/{:.1} ({:.0}% complete)",
        score.dimensions.accuracy_performance.score,
        score.dimensions.accuracy_performance.max_score,
        score.dimensions.accuracy_performance.completion_ratio() * 100.0
    );
    println!(
        "  Generalization: {:.1}/{:.1}",
        score.dimensions.generalization_robustness.score,
        score.dimensions.generalization_robustness.max_score
    );
    println!(
        "  Complexity: {:.1}/{:.1}",
        score.dimensions.model_complexity.score, score.dimensions.model_complexity.max_score
    );
    println!(
        "  Documentation: {:.1}/{:.1}",
        score.dimensions.documentation_provenance.score,
        score.dimensions.documentation_provenance.max_score
    );
    println!(
        "  Reproducibility: {:.1}/{:.1}",
        score.dimensions.reproducibility.score, score.dimensions.reproducibility.max_score
    );
    println!(
        "  Security: {:.1}/{:.1}",
        score.dimensions.security_safety.score, score.dimensions.security_safety.max_score
    );
    println!();
}

fn security_issues_demo() {
    println!("--- Part 5: Security Issues Detection ---\n");

    // Model with security issues
    let mut bad_metadata = ModelMetadata::default();
    bad_metadata.model_name = Some("LeakyModel".to_string());
    bad_metadata
        .custom
        .insert("api_key".to_string(), "sk-secret123".to_string());
    bad_metadata
        .custom
        .insert("password".to_string(), "admin123".to_string());

    let config = ScoringConfig {
        require_signed: true,
        require_model_card: true,
        ..Default::default()
    };

    let score = compute_quality_score(&bad_metadata, &config);

    println!("Security-Sensitive Scoring:");
    println!("  Total: {:.1}/100 (Grade: {})", score.total, score.grade);
    println!("  Has Critical Issues: {}", score.has_critical_issues());

    if !score.critical_issues.is_empty() {
        println!("\nCritical Issues Found:");
        for issue in &score.critical_issues {
            println!("  [{:?}] {}", issue.severity, issue.message);
            println!("    Action: {}", issue.action);
        }
    }

    // High train/test gap (overfitting indicator)
    let mut overfit = ModelMetadata::default();
    overfit.metrics.insert("train_score".to_string(), 0.99);
    overfit.metrics.insert("test_score".to_string(), 0.65);

    let overfit_score = compute_quality_score(&overfit, &ScoringConfig::default());
    println!("\nOverfitting Detection:");
    println!("  Train Score: 99%, Test Score: 65%");

    let gap_warnings: Vec<_> = overfit_score
        .findings
        .iter()
        .filter(|f| match f {
            Finding::Warning { message, .. } => message.contains("train/test gap"),
            _ => false,
        })
        .collect();

    for warning in gap_warnings {
        println!("  Warning: {}", warning);
    }
    println!();
}

fn scoring_config_demo() {
    println!("--- Part 6: Scoring Configuration ---\n");

    // Default config
    let default_config = ScoringConfig::default();
    println!("Default Configuration:");
    println!(
        "  Min Primary Metric: {:.2}",
        default_config.min_primary_metric
    );
    println!("  Max CV Std: {:.2}", default_config.max_cv_std);
    println!(
        "  Max Train/Test Gap: {:.2}",
        default_config.max_train_test_gap
    );
    println!("  Require Signed: {}", default_config.require_signed);
    println!(
        "  Require Model Card: {}",
        default_config.require_model_card
    );

    // Strict config for production
    let strict_config = ScoringConfig {
        min_primary_metric: 0.9,
        max_cv_std: 0.05,
        max_train_test_gap: 0.05,
        require_signed: true,
        require_model_card: true,
    };

    println!("\nStrict (Production) Configuration:");
    println!(
        "  Min Primary Metric: {:.2}",
        strict_config.min_primary_metric
    );
    println!("  Max CV Std: {:.2}", strict_config.max_cv_std);
    println!(
        "  Max Train/Test Gap: {:.2}",
        strict_config.max_train_test_gap
    );
    println!("  Require Signed: {}", strict_config.require_signed);
    println!("  Require Model Card: {}", strict_config.require_model_card);

    // Compare scores with different configs
    let mut metadata = ModelMetadata {
        model_type: Some(ScoredModelType::RandomForest),
        ..Default::default()
    };
    metadata.metrics.insert("accuracy".to_string(), 0.85);

    let default_score = compute_quality_score(&metadata, &default_config);
    let strict_score = compute_quality_score(&metadata, &strict_config);

    println!("\nScore Comparison (same model, different configs):");
    println!(
        "  Default Config: {:.1}/100 (Grade: {}, Critical: {})",
        default_score.total,
        default_score.grade,
        default_score.critical_issues.len()
    );
    println!(
        "  Strict Config:  {:.1}/100 (Grade: {}, Critical: {})",
        strict_score.total,
        strict_score.grade,
        strict_score.critical_issues.len()
    );
    println!();
}
