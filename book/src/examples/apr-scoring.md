# Case Study: APR 100-Point Quality Scoring

This example demonstrates the comprehensive model quality scoring system that evaluates models across six dimensions based on ML best practices and Toyota Way principles.

## Overview

The scoring system provides a standardized 100-point quality assessment:

| Dimension | Max Points | Toyota Way Principle |
|-----------|-----------|---------------------|
| Accuracy & Performance | 25 | Kaizen (continuous improvement) |
| Generalization & Robustness | 20 | Jidoka (quality built-in) |
| Model Complexity | 15 | Muda elimination (waste reduction) |
| Documentation & Provenance | 15 | Genchi Genbutsu (go and see) |
| Reproducibility | 15 | Standardization |
| Security & Safety | 10 | Poka-yoke (error-proofing) |

## Running the Example

```bash
cargo run --example apr_scoring
```

## Grade System

| Grade | Score Range | Passing |
|-------|-------------|---------|
| A+ | 97-100 | Yes |
| A | 93-96 | Yes |
| A- | 90-92 | Yes |
| B+ | 87-89 | Yes |
| B | 83-86 | Yes |
| B- | 80-82 | Yes |
| C+ | 77-79 | Yes |
| C | 73-76 | Yes |
| C- | 70-72 | Yes |
| D | 60-69 | No |
| F | <60 | No |

## Model Types and Metrics

Each model type has specific scoring criteria:

```rust
let types = [
    ScoredModelType::LinearRegression,      // Primary: R2, needs regularization
    ScoredModelType::LogisticRegression,    // Primary: accuracy
    ScoredModelType::DecisionTree,          // High interpretability
    ScoredModelType::RandomForest,          // Ensemble, lower interpretability
    ScoredModelType::GradientBoosting,      // Ensemble, needs tuning
    ScoredModelType::Knn,                   // Instance-based
    ScoredModelType::KMeans,                // Clustering
    ScoredModelType::NaiveBayes,            // Probabilistic
    ScoredModelType::NeuralSequential,      // Deep learning
    ScoredModelType::Svm,                   // Kernel methods
];

// Each type has:
println!("Interpretability: {:.1}", model_type.interpretability_score());
println!("Primary Metric: {}", model_type.primary_metric());
println!("Acceptable Threshold: {:.2}", model_type.acceptable_threshold());
println!("Needs Regularization: {}", model_type.needs_regularization());
```

## Scoring a Model

### Minimal Metadata

```rust
let mut metadata = ModelMetadata {
    model_name: Some("BasicModel".to_string()),
    model_type: Some(ScoredModelType::LinearRegression),
    ..Default::default()
};
metadata.metrics.insert("r2_score".to_string(), 0.85);

let config = ScoringConfig::default();
let score = compute_quality_score(&metadata, &config);

println!("Total: {:.1}/100 (Grade: {})", score.total, score.grade);
```

### Comprehensive Metadata

```rust
let mut metadata = ModelMetadata {
    model_name: Some("IrisRandomForest".to_string()),
    description: Some("Random Forest classifier for Iris".to_string()),
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

// Add metrics
metadata.metrics.insert("accuracy".to_string(), 0.967);
metadata.metrics.insert("cv_score_mean".to_string(), 0.953);
metadata.metrics.insert("cv_score_std".to_string(), 0.025);
metadata.metrics.insert("train_score".to_string(), 0.985);
metadata.metrics.insert("test_score".to_string(), 0.967);
```

## Security Detection

The scoring system detects security issues:

```rust
// Model with leaked secrets
let mut bad_metadata = ModelMetadata::default();
bad_metadata.custom.insert("api_key".to_string(), "sk-secret123".to_string());
bad_metadata.custom.insert("password".to_string(), "admin123".to_string());

let config = ScoringConfig {
    require_signed: true,
    require_model_card: true,
    ..Default::default()
};

let score = compute_quality_score(&bad_metadata, &config);
println!("Critical Issues: {}", score.critical_issues.len());
```

### Critical Issues Detected

- Leaked API keys or passwords in metadata
- Missing required signatures
- Missing model cards in production
- Excessive train/test gap (overfitting)

## Scoring Configuration

```rust
// Default config
let default_config = ScoringConfig::default();

// Strict config for production
let strict_config = ScoringConfig {
    min_primary_metric: 0.9,    // Require 90% accuracy
    max_cv_std: 0.05,           // Max CV standard deviation
    max_train_test_gap: 0.05,   // Max overfitting tolerance
    require_signed: true,        // Require model signature
    require_model_card: true,    // Require documentation
};
```

## Source Code

- Example: `examples/apr_scoring.rs`
- Module: `src/scoring/mod.rs`
