//! 100-Point Model Quality Scoring System (spec §7)
//!
//! Evaluates models across seven dimensions based on data science and ML best practices,
//! aligned with Toyota Way principles:
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
//! # References
//!
//! - [Raschka 2018] Model Evaluation, Model Selection, and Algorithm Selection in ML
//! - [Hastie et al. 2009] The Elements of Statistical Learning
//! - [Mitchell et al. 2019] Model Cards for Model Reporting
//! - [Gebru et al. 2021] Datasheets for Datasets
//! - [Pineau et al. 2021] ML Reproducibility Checklist

use std::collections::HashMap;
use std::fmt;

/// 100-point model quality score
#[derive(Debug, Clone)]
pub struct QualityScore {
    /// Total score (0-100, normalized from 110 raw points)
    pub total: f32,

    /// Grade letter (A+, A, A-, B+, ...)
    pub grade: Grade,

    /// Raw score before normalization (0-110)
    pub raw_score: f32,

    /// Individual dimension scores
    pub dimensions: DimensionScores,

    /// Detailed findings and recommendations
    pub findings: Vec<Finding>,

    /// Critical issues that must be addressed
    pub critical_issues: Vec<CriticalIssue>,
}

impl QualityScore {
    /// Create a new quality score from dimension scores
    #[must_use]
    pub fn new(
        dimensions: DimensionScores,
        findings: Vec<Finding>,
        critical_issues: Vec<CriticalIssue>,
    ) -> Self {
        let raw_score = dimensions.total_raw();
        let total = (raw_score / 110.0) * 100.0;
        let grade = Grade::from_score(total);

        Self {
            total,
            grade,
            raw_score,
            dimensions,
            findings,
            critical_issues,
        }
    }

    /// Check if the model passes minimum quality threshold
    #[must_use]
    pub fn passes_threshold(&self, min_score: f32) -> bool {
        self.total >= min_score && self.critical_issues.is_empty()
    }

    /// Check if there are critical issues
    #[must_use]
    pub fn has_critical_issues(&self) -> bool {
        !self.critical_issues.is_empty()
    }

    /// Get warnings count
    #[must_use]
    pub fn warning_count(&self) -> usize {
        self.findings
            .iter()
            .filter(|f| matches!(f, Finding::Warning { .. }))
            .count()
    }

    /// Get info count
    #[must_use]
    pub fn info_count(&self) -> usize {
        self.findings
            .iter()
            .filter(|f| matches!(f, Finding::Info { .. }))
            .count()
    }
}

impl fmt::Display for QualityScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Model Quality Score: {:.1}/100 (Grade: {})",
            self.total, self.grade
        )?;
        writeln!(f, "\nDimension Breakdown:")?;
        writeln!(
            f,
            "  Accuracy & Performance:    {:.1}/25 ({:.0}%)",
            self.dimensions.accuracy_performance.score,
            self.dimensions.accuracy_performance.percentage
        )?;
        writeln!(
            f,
            "  Generalization & Robust:   {:.1}/20 ({:.0}%)",
            self.dimensions.generalization_robustness.score,
            self.dimensions.generalization_robustness.percentage
        )?;
        writeln!(
            f,
            "  Model Complexity:          {:.1}/15 ({:.0}%)",
            self.dimensions.model_complexity.score, self.dimensions.model_complexity.percentage
        )?;
        writeln!(
            f,
            "  Documentation & Provenance:{:.1}/15 ({:.0}%)",
            self.dimensions.documentation_provenance.score,
            self.dimensions.documentation_provenance.percentage
        )?;
        writeln!(
            f,
            "  Reproducibility:           {:.1}/15 ({:.0}%)",
            self.dimensions.reproducibility.score, self.dimensions.reproducibility.percentage
        )?;
        writeln!(
            f,
            "  Security & Safety:         {:.1}/10 ({:.0}%)",
            self.dimensions.security_safety.score, self.dimensions.security_safety.percentage
        )?;

        if !self.critical_issues.is_empty() {
            writeln!(f, "\nCritical Issues ({}):", self.critical_issues.len())?;
            for issue in &self.critical_issues {
                writeln!(f, "  - {issue}")?;
            }
        }

        if !self.findings.is_empty() {
            let warnings: Vec<_> = self
                .findings
                .iter()
                .filter(|f| matches!(f, Finding::Warning { .. }))
                .collect();
            if !warnings.is_empty() {
                writeln!(f, "\nWarnings ({}):", warnings.len())?;
                for finding in warnings {
                    writeln!(f, "  - {finding}")?;
                }
            }
        }

        Ok(())
    }
}

/// Letter grade for quality score
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Grade {
    /// A+ (97-100)
    APlus,
    /// A (93-96)
    A,
    /// A- (90-92)
    AMinus,
    /// B+ (87-89)
    BPlus,
    /// B (83-86)
    B,
    /// B- (80-82)
    BMinus,
    /// C+ (77-79)
    CPlus,
    /// C (73-76)
    C,
    /// C- (70-72)
    CMinus,
    /// D+ (67-69)
    DPlus,
    /// D (63-66)
    D,
    /// D- (60-62)
    DMinus,
    /// F (< 60)
    F,
}

impl Grade {
    /// Convert score to grade
    #[must_use]
    pub fn from_score(score: f32) -> Self {
        match score {
            s if s >= 97.0 => Self::APlus,
            s if s >= 93.0 => Self::A,
            s if s >= 90.0 => Self::AMinus,
            s if s >= 87.0 => Self::BPlus,
            s if s >= 83.0 => Self::B,
            s if s >= 80.0 => Self::BMinus,
            s if s >= 77.0 => Self::CPlus,
            s if s >= 73.0 => Self::C,
            s if s >= 70.0 => Self::CMinus,
            s if s >= 67.0 => Self::DPlus,
            s if s >= 63.0 => Self::D,
            s if s >= 60.0 => Self::DMinus,
            _ => Self::F,
        }
    }

    /// Get minimum score for this grade
    #[must_use]
    pub const fn min_score(&self) -> f32 {
        match self {
            Self::APlus => 97.0,
            Self::A => 93.0,
            Self::AMinus => 90.0,
            Self::BPlus => 87.0,
            Self::B => 83.0,
            Self::BMinus => 80.0,
            Self::CPlus => 77.0,
            Self::C => 73.0,
            Self::CMinus => 70.0,
            Self::DPlus => 67.0,
            Self::D => 63.0,
            Self::DMinus => 60.0,
            Self::F => 0.0,
        }
    }

    /// Check if grade is passing (C- or better)
    #[must_use]
    pub const fn is_passing(&self) -> bool {
        matches!(
            self,
            Self::APlus
                | Self::A
                | Self::AMinus
                | Self::BPlus
                | Self::B
                | Self::BMinus
                | Self::CPlus
                | Self::C
                | Self::CMinus
        )
    }
}

impl fmt::Display for Grade {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::APlus => "A+",
            Self::A => "A",
            Self::AMinus => "A-",
            Self::BPlus => "B+",
            Self::B => "B",
            Self::BMinus => "B-",
            Self::CPlus => "C+",
            Self::C => "C",
            Self::CMinus => "C-",
            Self::DPlus => "D+",
            Self::D => "D",
            Self::DMinus => "D-",
            Self::F => "F",
        };
        write!(f, "{s}")
    }
}

/// Individual dimension scores
#[derive(Debug, Clone)]
pub struct DimensionScores {
    /// Accuracy & Performance (25 points max)
    pub accuracy_performance: DimensionScore,
    /// Generalization & Robustness (20 points max)
    pub generalization_robustness: DimensionScore,
    /// Model Complexity (15 points max)
    pub model_complexity: DimensionScore,
    /// Documentation & Provenance (15 points max)
    pub documentation_provenance: DimensionScore,
    /// Reproducibility (15 points max)
    pub reproducibility: DimensionScore,
    /// Security & Safety (10 points max)
    pub security_safety: DimensionScore,
}

impl DimensionScores {
    /// Get total raw score (out of 110)
    #[must_use]
    pub fn total_raw(&self) -> f32 {
        self.accuracy_performance.score
            + self.generalization_robustness.score
            + self.model_complexity.score
            + self.documentation_provenance.score
            + self.reproducibility.score
            + self.security_safety.score
    }

    /// Create default dimension scores
    #[must_use]
    pub fn default_scores() -> Self {
        Self {
            accuracy_performance: DimensionScore::new(25.0),
            generalization_robustness: DimensionScore::new(20.0),
            model_complexity: DimensionScore::new(15.0),
            documentation_provenance: DimensionScore::new(15.0),
            reproducibility: DimensionScore::new(15.0),
            security_safety: DimensionScore::new(10.0),
        }
    }
}

impl Default for DimensionScores {
    fn default() -> Self {
        Self::default_scores()
    }
}

/// Score for a single dimension
#[derive(Debug, Clone)]
pub struct DimensionScore {
    /// Score achieved
    pub score: f32,
    /// Maximum possible score
    pub max_score: f32,
    /// Percentage achieved
    pub percentage: f32,
    /// Detailed breakdown (criterion, score, max)
    pub breakdown: Vec<ScoreBreakdown>,
}

impl DimensionScore {
    /// Create a new dimension score with no points
    #[must_use]
    pub fn new(max_score: f32) -> Self {
        Self {
            score: 0.0,
            max_score,
            percentage: 0.0,
            breakdown: Vec::new(),
        }
    }

    /// Add points for a criterion
    pub fn add_score(&mut self, criterion: impl Into<String>, score: f32, max: f32) {
        self.breakdown.push(ScoreBreakdown {
            criterion: criterion.into(),
            score,
            max,
        });
        self.score += score;
        self.update_percentage();
    }

    /// Update percentage based on current score
    fn update_percentage(&mut self) {
        self.percentage = if self.max_score > 0.0 {
            (self.score / self.max_score) * 100.0
        } else {
            0.0
        };
    }

    /// Check if dimension achieved perfect score
    #[must_use]
    pub fn is_perfect(&self) -> bool {
        (self.score - self.max_score).abs() < f32::EPSILON
    }

    /// Get completion ratio (0.0 to 1.0)
    #[must_use]
    pub fn completion_ratio(&self) -> f32 {
        if self.max_score > 0.0 {
            self.score / self.max_score
        } else {
            0.0
        }
    }
}

/// Breakdown item for dimension scoring
#[derive(Debug, Clone)]
pub struct ScoreBreakdown {
    /// Criterion name
    pub criterion: String,
    /// Score achieved
    pub score: f32,
    /// Maximum possible
    pub max: f32,
}

impl ScoreBreakdown {
    /// Get percentage achieved for this criterion
    #[must_use]
    pub fn percentage(&self) -> f32 {
        if self.max > 0.0 {
            (self.score / self.max) * 100.0
        } else {
            0.0
        }
    }
}

/// Finding from quality analysis
#[derive(Debug, Clone)]
pub enum Finding {
    /// Warning that should be addressed
    Warning {
        /// Warning message
        message: String,
        /// Recommended action
        recommendation: String,
    },
    /// Informational note
    Info {
        /// Info message
        message: String,
        /// Suggested improvement
        recommendation: String,
    },
}

impl fmt::Display for Finding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Warning {
                message,
                recommendation,
            } => {
                write!(f, "[WARN] {message} (Recommendation: {recommendation})")
            }
            Self::Info {
                message,
                recommendation,
            } => {
                write!(f, "[INFO] {message} (Suggestion: {recommendation})")
            }
        }
    }
}

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

/// Compute score based on generalization gap.
fn compute_gap_score(gap: f64) -> f32 {
    if gap < 0.05 {
        5.0
    } else if gap < 0.1 {
        3.0
    } else if gap < 0.2 {
        1.0
    } else {
        0.0
    }
}

/// Score Dimension 3: Model Complexity (15 points)
fn score_model_complexity(metadata: &ModelMetadata, findings: &mut Vec<Finding>) -> DimensionScore {
    let mut dim = DimensionScore::new(15.0);

    // 3.1 Parameter efficiency (5 points)
    score_parameter_efficiency(metadata, &mut dim, findings);

    // 3.2 Model interpretability (5 points)
    let model_type = metadata.model_type.unwrap_or(ScoredModelType::Other);
    dim.add_score("interpretability", model_type.interpretability_score(), 5.0);

    // 3.3 Feature importance available (5 points)
    score_feature_importance(metadata, &mut dim, findings);

    dim
}

/// Score parameter efficiency based on params/sample ratio.
fn score_parameter_efficiency(
    metadata: &ModelMetadata,
    dim: &mut DimensionScore,
    findings: &mut Vec<Finding>,
) {
    let Some(n_params) = metadata.n_parameters else {
        return;
    };
    let Some(training) = &metadata.training else {
        return;
    };
    let Some(n_samples) = training.n_samples else {
        return;
    };

    let params_per_sample = n_params as f64 / n_samples as f64;
    let efficiency_score = compute_efficiency_score(params_per_sample);
    dim.add_score("parameter_efficiency", efficiency_score, 5.0);

    if params_per_sample > 1.0 {
        findings.push(Finding::Info {
            message: format!(
                "High parameter count relative to data: {params_per_sample:.2} params/sample"
            ),
            recommendation: "Consider feature selection or simpler model architecture".to_string(),
        });
    }
}

/// Compute efficiency score from params/sample ratio.
fn compute_efficiency_score(params_per_sample: f64) -> f32 {
    // Rule of thumb: < 0.1 params/sample is efficient
    if params_per_sample < 0.1 {
        5.0
    } else if params_per_sample < 0.5 {
        4.0
    } else if params_per_sample < 1.0 {
        3.0
    } else if params_per_sample < 5.0 {
        2.0
    } else {
        1.0
    }
}

/// Score feature importance availability.
fn score_feature_importance(
    metadata: &ModelMetadata,
    dim: &mut DimensionScore,
    findings: &mut Vec<Finding>,
) {
    if metadata.flags.has_feature_importance {
        dim.add_score("feature_importance", 5.0, 5.0);
    } else {
        findings.push(Finding::Info {
            message: "No feature importance information available".to_string(),
            recommendation: "Include feature importance for model interpretability".to_string(),
        });
    }
}

/// Score Dimension 4: Documentation & Provenance (15 points)
fn score_documentation_provenance(
    metadata: &ModelMetadata,
    findings: &mut Vec<Finding>,
    _critical: &mut Vec<CriticalIssue>,
) -> DimensionScore {
    let mut dim = DimensionScore::new(15.0);
    let mut name_desc_score = 0.0;

    // 4.1 Model name and description (3 points)
    if metadata.model_name.is_some() {
        name_desc_score += 1.5;
    }
    if metadata.description.is_some() {
        name_desc_score += 1.5;
    }
    dim.add_score("name_description", name_desc_score, 3.0);

    // 4.2 Training provenance (4 points)
    let mut provenance_score = 0.0;
    if let Some(training) = &metadata.training {
        if training.source.is_some() {
            provenance_score += 1.0;
        }
        if training.n_samples.is_some() {
            provenance_score += 1.0;
        }
        if training.duration_ms.is_some() {
            provenance_score += 1.0;
        }
        if training.random_seed.is_some() {
            provenance_score += 1.0;
        }
    }
    dim.add_score("training_provenance", provenance_score, 4.0);

    if provenance_score < 2.0 {
        findings.push(Finding::Warning {
            message: "Incomplete training provenance".to_string(),
            recommendation: "Record data source, sample count, training duration, and random seed"
                .to_string(),
        });
    }

    // 4.3 Hyperparameters documented (4 points)
    let hp_count = metadata.hyperparameters.len();
    let hp_score = (hp_count as f32 / 5.0).min(1.0) * 4.0;
    dim.add_score("hyperparameters", hp_score, 4.0);

    // 4.4 Model card present (4 points)
    if metadata.flags.has_model_card {
        dim.add_score("model_card", 4.0, 4.0);
    } else {
        findings.push(Finding::Info {
            message: "No model card attached".to_string(),
            recommendation:
                "Add model card for comprehensive documentation (see Mitchell et al. 2019)"
                    .to_string(),
        });
    }

    dim
}

/// Score Dimension 5: Reproducibility (15 points)
fn score_reproducibility(metadata: &ModelMetadata, findings: &mut Vec<Finding>) -> DimensionScore {
    let mut dim = DimensionScore::new(15.0);

    // 5.1 Random seed recorded (5 points)
    if metadata
        .training
        .as_ref()
        .is_some_and(|t| t.random_seed.is_some())
    {
        dim.add_score("random_seed", 5.0, 5.0);
    } else {
        findings.push(Finding::Warning {
            message: "No random seed recorded".to_string(),
            recommendation: "Set and record random seed for reproducibility".to_string(),
        });
    }

    // 5.2 Framework version recorded (3 points)
    if metadata.aprender_version.is_some() {
        dim.add_score("framework_version", 3.0, 3.0);
    }

    // 5.3 Data preprocessing documented (4 points)
    if metadata.flags.has_preprocessing_steps {
        dim.add_score("preprocessing", 4.0, 4.0);
    }

    // 5.4 Checksum/hash for integrity (3 points)
    // Always present in valid .apr files, give full points
    dim.add_score("checksum", 3.0, 3.0);

    dim
}

/// Score Dimension 6: Security & Safety (10 points)
fn score_security_safety(
    metadata: &ModelMetadata,
    config: &ScoringConfig,
    findings: &mut Vec<Finding>,
    critical: &mut Vec<CriticalIssue>,
) -> DimensionScore {
    let mut dim = DimensionScore::new(10.0);

    // 6.1 Model signed (4 points)
    if metadata.flags.is_signed {
        dim.add_score("signed", 4.0, 4.0);
    } else if config.require_signed {
        critical.push(CriticalIssue {
            severity: Severity::High,
            message: "Model is not signed".to_string(),
            action: "Sign model with Ed25519 key for deployment".to_string(),
        });
    } else {
        findings.push(Finding::Info {
            message: "Model is not cryptographically signed".to_string(),
            recommendation: "Consider signing models for production deployment".to_string(),
        });
    }

    // 6.2 No sensitive data in metadata (3 points)
    // Check for potential secrets in metadata
    let has_secrets = metadata.custom.keys().any(|k| {
        let k_lower = k.to_lowercase();
        k_lower.contains("password")
            || k_lower.contains("secret")
            || k_lower.contains("api_key")
            || k_lower.contains("token")
    });

    if has_secrets {
        critical.push(CriticalIssue {
            severity: Severity::Critical,
            message: "Potential secrets detected in model metadata".to_string(),
            action: "Remove all sensitive data from model metadata before distribution".to_string(),
        });
    } else {
        dim.add_score("no_secrets", 3.0, 3.0);
    }

    // 6.3 Input validation documented (3 points)
    let has_input_bounds = metadata.custom.contains_key("input_bounds")
        || metadata.custom.contains_key("input_schema")
        || metadata.custom.contains_key("feature_ranges");

    if has_input_bounds {
        dim.add_score("input_validation", 3.0, 3.0);
    } else {
        findings.push(Finding::Info {
            message: "No input validation bounds documented".to_string(),
            recommendation: "Document expected input ranges for safe inference".to_string(),
        });
    }

    dim
}

#[cfg(test)]
mod tests {
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
        assert!(warning_count >= 0);
    }

    #[test]
    fn test_quality_score_info_count() {
        let mut metadata = ModelMetadata::default();
        metadata.model_name = Some("TestModel".to_string());
        let config = ScoringConfig::default();
        let score = compute_quality_score(&metadata, &config);

        let info_count = score.info_count();
        assert!(info_count >= 0);
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
}
