
/// Critical issue severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Medium severity (should fix)
    Medium,
    /// High severity (must fix)
    High,
    /// Critical severity (blocks deployment)
    Critical,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Medium => write!(f, "MEDIUM"),
            Self::High => write!(f, "HIGH"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Critical issue that must be addressed
#[derive(Debug, Clone)]
pub struct CriticalIssue {
    /// Issue severity
    pub severity: Severity,
    /// Issue description
    pub message: String,
    /// Required action
    pub action: String,
}

impl fmt::Display for CriticalIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {} (Action: {})",
            self.severity, self.message, self.action
        )
    }
}

/// Model type for scoring context
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScoredModelType {
    /// Linear regression
    LinearRegression,
    /// Logistic regression
    LogisticRegression,
    /// Decision tree
    DecisionTree,
    /// Random forest
    RandomForest,
    /// Gradient boosting
    GradientBoosting,
    /// K-Nearest Neighbors
    Knn,
    /// K-Means clustering
    KMeans,
    /// Naive Bayes
    NaiveBayes,
    /// Neural network (sequential)
    NeuralSequential,
    /// Neural network (custom)
    NeuralCustom,
    /// Support Vector Machine
    Svm,
    /// Other/unknown model type
    Other,
}

impl ScoredModelType {
    /// Check if model type typically requires regularization
    #[must_use]
    pub const fn needs_regularization(&self) -> bool {
        matches!(
            self,
            Self::LinearRegression
                | Self::LogisticRegression
                | Self::NeuralSequential
                | Self::NeuralCustom
        )
    }

    /// Get interpretability score for this model type
    #[must_use]
    pub const fn interpretability_score(&self) -> f32 {
        match self {
            Self::LinearRegression | Self::LogisticRegression => 5.0, // Highly interpretable
            Self::DecisionTree | Self::NaiveBayes => 4.0,             // Interpretable
            Self::RandomForest | Self::GradientBoosting => 3.0,       // Partially interpretable
            Self::Knn => 2.0,                                         // Instance-based
            Self::NeuralSequential | Self::NeuralCustom => 1.0,       // Black box
            Self::Svm | Self::KMeans | Self::Other => 2.5,
        }
    }

    /// Get default primary metric name for this model type
    #[must_use]
    pub const fn primary_metric(&self) -> &'static str {
        match self {
            Self::LinearRegression => "r2_score",
            Self::LogisticRegression
            | Self::DecisionTree
            | Self::RandomForest
            | Self::GradientBoosting
            | Self::Knn
            | Self::NaiveBayes
            | Self::Svm => "accuracy",
            Self::KMeans => "silhouette_score",
            Self::NeuralSequential | Self::NeuralCustom => "loss",
            Self::Other => "primary_score",
        }
    }

    /// Get acceptable threshold for primary metric
    #[must_use]
    pub const fn acceptable_threshold(&self) -> f32 {
        match self {
            // Classification models: 80% accuracy
            Self::LogisticRegression
            | Self::DecisionTree
            | Self::RandomForest
            | Self::GradientBoosting
            | Self::Knn
            | Self::NaiveBayes
            | Self::Svm => 0.8,
            // Clustering: silhouette >= 0.5
            Self::KMeans => 0.5,
            // Neural: loss <= 0.1 (inverted scale)
            Self::NeuralSequential | Self::NeuralCustom => 0.1,
            // Regression and other: R^2 >= 0.7
            Self::LinearRegression | Self::Other => 0.7,
        }
    }
}

/// Model metadata for scoring
#[derive(Debug, Clone, Default)]
pub struct ModelMetadata {
    /// Model name
    pub model_name: Option<String>,
    /// Model description
    pub description: Option<String>,
    /// Model type
    pub model_type: Option<ScoredModelType>,
    /// Number of parameters
    pub n_parameters: Option<u64>,
    /// Aprender version used
    pub aprender_version: Option<String>,
    /// Recorded metrics
    pub metrics: HashMap<String, f64>,
    /// Hyperparameters
    pub hyperparameters: HashMap<String, String>,
    /// Training information
    pub training: Option<TrainingInfo>,
    /// Custom metadata
    pub custom: HashMap<String, String>,
    /// Feature flags
    pub flags: ModelFlags,
}

/// Training information
#[derive(Debug, Clone, Default)]
pub struct TrainingInfo {
    /// Data source
    pub source: Option<String>,
    /// Number of training samples
    pub n_samples: Option<u64>,
    /// Number of features
    pub n_features: Option<u64>,
    /// Training duration in milliseconds
    pub duration_ms: Option<u64>,
    /// Random seed used
    pub random_seed: Option<u64>,
    /// Test set size (fraction)
    pub test_size: Option<f64>,
}

/// Model feature flags
#[derive(Debug, Clone, Copy, Default)]
#[allow(clippy::struct_excessive_bools)] // Flags struct legitimately has independent booleans
pub struct ModelFlags {
    /// Has model card
    pub has_model_card: bool,
    /// Is signed
    pub is_signed: bool,
    /// Is encrypted
    pub is_encrypted: bool,
    /// Has feature importance
    pub has_feature_importance: bool,
    /// Has edge case tests
    pub has_edge_case_tests: bool,
    /// Has preprocessing steps documented
    pub has_preprocessing_steps: bool,
}

/// Configuration for quality scoring
#[derive(Debug, Clone)]
pub struct ScoringConfig {
    /// Minimum acceptable primary metric value
    pub min_primary_metric: f32,
    /// Maximum acceptable CV score standard deviation
    pub max_cv_std: f32,
    /// Maximum acceptable train/test gap
    pub max_train_test_gap: f32,
    /// Require signed models
    pub require_signed: bool,
    /// Require model card
    pub require_model_card: bool,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            min_primary_metric: 0.7,
            max_cv_std: 0.1,
            max_train_test_gap: 0.1,
            require_signed: false,
            require_model_card: false,
        }
    }
}

/// Compute quality score for a model
///
/// # Arguments
///
/// * `metadata` - Model metadata for scoring
/// * `config` - Scoring configuration
///
/// # Returns
///
/// Quality score with detailed breakdown
#[must_use]
pub fn compute_quality_score(metadata: &ModelMetadata, config: &ScoringConfig) -> QualityScore {
    let mut findings = Vec::new();
    let mut critical_issues = Vec::new();

    // Dimension 1: Accuracy & Performance (25 points)
    let accuracy = score_accuracy_performance(metadata, config, &mut findings);

    // Dimension 2: Generalization & Robustness (20 points)
    let generalization = score_generalization_robustness(metadata, config, &mut findings);

    // Dimension 3: Model Complexity (15 points)
    let complexity = score_model_complexity(metadata, &mut findings);

    // Dimension 4: Documentation & Provenance (15 points)
    let documentation =
        score_documentation_provenance(metadata, &mut findings, &mut critical_issues);

    // Dimension 5: Reproducibility (15 points)
    let reproducibility = score_reproducibility(metadata, &mut findings);

    // Dimension 6: Security & Safety (10 points)
    let security = score_security_safety(metadata, config, &mut findings, &mut critical_issues);

    let dimensions = DimensionScores {
        accuracy_performance: accuracy,
        generalization_robustness: generalization,
        model_complexity: complexity,
        documentation_provenance: documentation,
        reproducibility,
        security_safety: security,
    };

    QualityScore::new(dimensions, findings, critical_issues)
}

/// Score Dimension 1: Accuracy & Performance (25 points)
fn score_accuracy_performance(
    metadata: &ModelMetadata,
    config: &ScoringConfig,
    findings: &mut Vec<Finding>,
) -> DimensionScore {
    let mut dim = DimensionScore::new(25.0);

    // 1.1 Primary metric meets threshold (10 points)
    let model_type = metadata.model_type.unwrap_or(ScoredModelType::Other);
    let primary_metric_name = model_type.primary_metric();
    let threshold = model_type
        .acceptable_threshold()
        .max(config.min_primary_metric);

    if let Some(&value) = metadata.metrics.get(primary_metric_name) {
        let metric_score = (value as f32 / threshold).min(1.0) * 10.0;
        dim.add_score("primary_metric", metric_score, 10.0);
    } else {
        findings.push(Finding::Warning {
            message: "No primary metric recorded in model metadata".to_string(),
            recommendation: "Include primary evaluation metric during training".to_string(),
        });
    }

    // 1.2 Cross-validation performed (8 points)
    if let Some(&cv_mean) = metadata.metrics.get("cv_score_mean") {
        let cv_std = metadata.metrics.get("cv_score_std").copied().unwrap_or(0.0);

        // Penalize high variance (indicates overfitting risk)
        let cv_quality = if cv_std < 0.05 {
            8.0
        } else if cv_std < 0.1 {
            6.0
        } else {
            4.0
        };
        dim.add_score("cross_validation", cv_quality, 8.0);

        if cv_std >= f64::from(config.max_cv_std) {
            findings.push(Finding::Warning {
                message: format!(
                    "High CV score variance: {cv_std:.3} (target < {:.2})",
                    config.max_cv_std
                ),
                recommendation:
                    "High variance may indicate overfitting. Consider simpler model or more data."
                        .to_string(),
            });
        }

        // Use cv_mean to avoid unused variable warning
        let _ = cv_mean;
    } else {
        findings.push(Finding::Info {
            message: "No cross-validation results found".to_string(),
            recommendation: "Use k-fold cross-validation to estimate generalization".to_string(),
        });
    }

    // 1.3 Inference latency documented (4 points)
    if metadata.metrics.contains_key("inference_latency_ms") {
        dim.add_score("latency_documented", 4.0, 4.0);
    }

    // 1.4 Multiple metrics recorded (3 points)
    let metric_count = metadata.metrics.len();
    let multi_metric_score = (metric_count as f32 / 3.0).min(1.0) * 3.0;
    dim.add_score("multiple_metrics", multi_metric_score, 3.0);

    dim
}

/// Score Dimension 2: Generalization & Robustness (20 points)
fn score_generalization_robustness(
    metadata: &ModelMetadata,
    config: &ScoringConfig,
    findings: &mut Vec<Finding>,
) -> DimensionScore {
    let mut dim = DimensionScore::new(20.0);

    // 2.1 Train/test split used (5 points)
    score_train_test_split(metadata, &mut dim);

    // 2.2 Regularization applied (5 points, if applicable)
    score_regularization(metadata, &mut dim, findings);

    // 2.3 Training/test performance gap (5 points)
    score_generalization_gap(metadata, config, &mut dim, findings);

    // 2.4 Handles edge cases (5 points)
    if metadata.flags.has_edge_case_tests {
        dim.add_score("edge_cases", 5.0, 5.0);
    }

    dim
}

/// Score train/test split usage.
fn score_train_test_split(metadata: &ModelMetadata, dim: &mut DimensionScore) {
    if metadata
        .training
        .as_ref()
        .is_some_and(|t| t.test_size.is_some())
    {
        dim.add_score("train_test_split", 5.0, 5.0);
    }
}

/// Score regularization usage.
fn score_regularization(
    metadata: &ModelMetadata,
    dim: &mut DimensionScore,
    findings: &mut Vec<Finding>,
) {
    let model_type = metadata.model_type.unwrap_or(ScoredModelType::Other);

    if !model_type.needs_regularization() {
        dim.add_score("regularization", 5.0, 5.0);
        return;
    }

    let has_reg = has_regularization_params(&metadata.hyperparameters);
    if has_reg {
        dim.add_score("regularization", 5.0, 5.0);
    } else {
        findings.push(Finding::Warning {
            message: "No regularization detected for linear/neural model".to_string(),
            recommendation: "Consider adding L2 regularization to prevent overfitting".to_string(),
        });
    }
}

/// Check if hyperparameters contain regularization settings.
fn has_regularization_params(hyperparameters: &HashMap<String, String>) -> bool {
    hyperparameters.contains_key("alpha")
        || hyperparameters.contains_key("lambda")
        || hyperparameters.contains_key("l2_penalty")
        || hyperparameters.contains_key("weight_decay")
}

/// Score generalization gap between train and test performance.
fn score_generalization_gap(
    metadata: &ModelMetadata,
    config: &ScoringConfig,
    dim: &mut DimensionScore,
    findings: &mut Vec<Finding>,
) {
    let train_score = metadata.metrics.get("train_score").copied();
    let test_score = metadata.metrics.get("test_score").copied();

    let Some((train, test)) = train_score.zip(test_score) else {
        return;
    };

    let gap = train - test;
    let gap_score = compute_gap_score(gap);
    dim.add_score("generalization_gap", gap_score, 5.0);

    if gap >= f64::from(config.max_train_test_gap) {
        findings.push(Finding::Warning {
            message: format!("High train/test gap detected: {:.1}%", gap * 100.0),
            recommendation: "Model may be overfitting. Consider regularization or simpler model."
                .to_string(),
        });
    }
}
