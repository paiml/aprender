//! Data types for profile comparison (PMAT-045)
//!
//! Contains error types, benchmark profiles, metric samples,
//! and all result/verdict structures.

use std::collections::HashMap;

/// Result type for profile comparison operations
pub type CompareResult<T> = Result<T, CompareError>;

/// Errors in profile comparison
#[derive(Debug, Clone, PartialEq)]
pub enum CompareError {
    /// Insufficient samples for comparison
    InsufficientSamples { got: usize, need: usize },
    /// Metric not found in profile
    MetricNotFound { name: String },
    /// Variance is zero (no variation in data)
    ZeroVariance { metric: String },
    /// Invalid confidence level
    InvalidConfidence { value: f64 },
    /// Profiles have no common metrics
    NoCommonMetrics,
}

impl std::fmt::Display for CompareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientSamples { got, need } => {
                write!(f, "Insufficient samples: got {}, need {}", got, need)
            }
            Self::MetricNotFound { name } => {
                write!(f, "Metric not found: {}", name)
            }
            Self::ZeroVariance { metric } => {
                write!(f, "Zero variance in metric: {}", metric)
            }
            Self::InvalidConfidence { value } => {
                write!(f, "Invalid confidence level: {}", value)
            }
            Self::NoCommonMetrics => {
                write!(f, "Profiles have no common metrics")
            }
        }
    }
}

impl std::error::Error for CompareError {}

/// A benchmark profile containing multiple metrics
#[derive(Debug, Clone)]
pub struct BenchmarkProfile {
    /// Profile name/identifier
    pub name: String,
    /// Profile description
    pub description: Option<String>,
    /// Metrics with their sample values
    pub metrics: HashMap<String, MetricSamples>,
    /// Profile metadata
    pub metadata: HashMap<String, String>,
    /// Timestamp when profile was captured
    pub timestamp_ns: u64,
}

impl BenchmarkProfile {
    /// Create a new benchmark profile
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            metrics: HashMap::new(),
            metadata: HashMap::new(),
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0),
        }
    }

    /// Set profile description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Add a metric with samples
    pub fn add_metric(&mut self, name: impl Into<String>, samples: Vec<f64>) {
        self.metrics
            .insert(name.into(), MetricSamples::new(samples));
    }

    /// Get metric by name
    pub fn get_metric(&self, name: &str) -> Option<&MetricSamples> {
        self.metrics.get(name)
    }

    /// Get all metric names
    pub fn metric_names(&self) -> impl Iterator<Item = &String> {
        self.metrics.keys()
    }

    /// Number of metrics
    pub fn metric_count(&self) -> usize {
        self.metrics.len()
    }
}

/// Samples for a single metric
#[derive(Debug, Clone)]
pub struct MetricSamples {
    /// Raw sample values
    pub values: Vec<f64>,
    /// Precomputed mean
    mean: f64,
    /// Precomputed variance
    variance: f64,
    /// Precomputed standard deviation
    std_dev: f64,
}

impl MetricSamples {
    /// Create new metric samples
    pub fn new(values: Vec<f64>) -> Self {
        let (mean, variance, std_dev) = if values.is_empty() {
            (0.0, 0.0, 0.0)
        } else {
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / (values.len() - 1).max(1) as f64;
            let std_dev = variance.sqrt();
            (mean, variance, std_dev)
        };

        Self {
            values,
            mean,
            variance,
            std_dev,
        }
    }

    /// Get sample count
    pub fn count(&self) -> usize {
        self.values.len()
    }

    /// Get mean
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Get variance
    pub fn variance(&self) -> f64 {
        self.variance
    }

    /// Get standard deviation
    pub fn std_dev(&self) -> f64 {
        self.std_dev
    }

    /// Get minimum value
    pub fn min(&self) -> f64 {
        self.values.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    /// Get maximum value
    pub fn max(&self) -> f64 {
        self.values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

/// Result of Welch's t-test comparison
#[derive(Debug, Clone)]
pub struct WelchTestResult {
    /// T-statistic
    pub t_statistic: f64,
    /// Degrees of freedom (Welch-Satterthwaite)
    pub degrees_of_freedom: f64,
    /// Two-tailed p-value
    pub p_value: f64,
    /// Whether result is statistically significant
    pub significant: bool,
    /// Confidence level used
    pub confidence_level: f64,
}

/// Effect size interpretation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectMagnitude {
    /// Negligible effect (|d| < 0.2)
    Negligible,
    /// Small effect (0.2 <= |d| < 0.5)
    Small,
    /// Medium effect (0.5 <= |d| < 0.8)
    Medium,
    /// Large effect (|d| >= 0.8)
    Large,
}

impl EffectMagnitude {
    /// Get magnitude from Cohen's d value
    pub fn from_cohens_d(d: f64) -> Self {
        let abs_d = d.abs();
        if abs_d < 0.2 {
            Self::Negligible
        } else if abs_d < 0.5 {
            Self::Small
        } else if abs_d < 0.8 {
            Self::Medium
        } else {
            Self::Large
        }
    }
}

/// Effect size result
#[derive(Debug, Clone)]
pub struct EffectSizeResult {
    /// Cohen's d effect size
    pub cohens_d: f64,
    /// Effect magnitude interpretation
    pub magnitude: EffectMagnitude,
    /// Percentage change (new - old) / old * 100
    pub percent_change: f64,
}

/// Direction of change
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeDirection {
    /// Performance improved (metric decreased for latency, increased for throughput)
    Improved,
    /// Performance regressed
    Regressed,
    /// No significant change
    NoChange,
}

/// Result of comparing a single metric
#[derive(Debug, Clone)]
pub struct MetricComparison {
    /// Metric name
    pub name: String,
    /// Baseline (A) statistics
    pub baseline_mean: f64,
    /// Baseline standard deviation
    pub baseline_std: f64,
    /// Comparison (B) statistics
    pub comparison_mean: f64,
    /// Comparison standard deviation
    pub comparison_std: f64,
    /// Welch's t-test result
    pub t_test: WelchTestResult,
    /// Effect size analysis
    pub effect_size: EffectSizeResult,
    /// Direction of change
    pub direction: ChangeDirection,
    /// Whether this is a regression
    pub is_regression: bool,
    /// Confidence interval for the difference
    pub ci_lower: f64,
    /// Upper bound of confidence interval
    pub ci_upper: f64,
}

/// Complete A/B comparison result
#[derive(Debug, Clone)]
pub struct ProfileComparison {
    /// Baseline profile name
    pub baseline_name: String,
    /// Comparison profile name
    pub comparison_name: String,
    /// Individual metric comparisons
    pub metrics: Vec<MetricComparison>,
    /// Metrics that regressed significantly
    pub regressions: Vec<String>,
    /// Metrics that improved significantly
    pub improvements: Vec<String>,
    /// Overall verdict
    pub verdict: ComparisonVerdict,
    /// Bonferroni-corrected alpha for multiple comparisons
    pub corrected_alpha: f64,
}

impl ProfileComparison {
    /// Get regression count
    pub fn regression_count(&self) -> usize {
        self.regressions.len()
    }

    /// Get improvement count
    pub fn improvement_count(&self) -> usize {
        self.improvements.len()
    }

    /// Get metrics with no significant change
    pub fn unchanged_count(&self) -> usize {
        self.metrics.len() - self.regression_count() - self.improvement_count()
    }

    /// Check if comparison detected any regressions
    pub fn has_regressions(&self) -> bool {
        !self.regressions.is_empty()
    }
}

/// Overall comparison verdict
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonVerdict {
    /// All metrics stable or improved
    Pass,
    /// Minor regressions detected (< 5%)
    Warning,
    /// Significant regressions detected (>= 5%)
    Fail,
}
