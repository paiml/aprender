//! Preflight Validation Protocol for Deterministic Benchmarking
//!
//! Implements the Toyota Way principles for benchmark quality:
//! - Jidoka: Fail-fast validation, stop on anomaly
//! - Poka-yoke: Error-proofing through type-safe configurations
//! - Genchi Genbutsu: Verify actual system state
//!
//! References:
//! - Hoefler & Belli SC'15 [1]: CV-based stopping
//! - Vitek & Kalibera EMSOFT'11 [7]: Reproducibility requirements
//! - Liker [9]: Toyota Way principles

use std::fmt;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ============================================================================
// Specification Constants (per spec v1.0.1 Section 4.3)
// ============================================================================

/// Canonical benchmark inputs - versioned for reproducibility
pub mod canonical_inputs {
    /// Version: Changing inputs MUST increment this version
    pub const VERSION: &str = "1.0.0";

    /// Fixed prompt for latency benchmarks
    pub const LATENCY_PROMPT: &str = "Explain the concept of machine learning in one sentence.";

    /// Fixed token sequence for throughput benchmarks
    pub const THROUGHPUT_TOKENS: &[u32] = &[1, 2, 3, 4, 5, 6, 7, 8];

    /// Fixed max tokens for generation benchmarks
    pub const MAX_TOKENS: usize = 50;

    /// Specification version for metadata
    pub const SPEC_VERSION: &str = "1.0.1";
}

// ============================================================================
// Error Types (Poka-yoke: Make errors explicit and typed)
// ============================================================================

/// Preflight validation errors with detailed context for diagnosis
#[derive(Debug, Error)]
pub enum PreflightError {
    /// Server is not reachable at the specified URL
    #[error("Server unreachable at {url}: {reason}")]
    ServerUnreachable {
        /// The URL that was unreachable
        url: String,
        /// The reason for the failure
        reason: String,
    },

    /// Health check endpoint failed
    #[error("Health check failed at {url}: HTTP {status}")]
    HealthCheckFailed {
        /// The health check URL
        url: String,
        /// HTTP status code returned
        status: u16,
    },

    /// Required model not found in backend
    #[error("Model not found: requested '{requested}', available: {available:?}")]
    ModelNotFound {
        /// Model name that was requested
        requested: String,
        /// List of available models
        available: Vec<String>,
    },

    /// Response schema does not match expected format
    #[error("Schema mismatch: missing field '{missing_field}'")]
    SchemaMismatch {
        /// The field that was expected but missing
        missing_field: String,
    },

    /// Field type does not match expected type
    #[error("Field type mismatch: '{field}' expected {expected}, got {actual}")]
    FieldTypeMismatch {
        /// Field name with type mismatch
        field: String,
        /// Expected type name
        expected: String,
        /// Actual type received
        actual: String,
    },

    /// Response parsing failed
    #[error("Response parse error: {reason}")]
    ResponseParseError {
        /// Description of the parse failure
        reason: String,
    },

    /// Timeout during preflight check
    #[error("Timeout after {duration:?} during {operation}")]
    Timeout {
        /// How long the operation waited before timing out
        duration: Duration,
        /// Name of the operation that timed out
        operation: String,
    },

    /// Configuration error
    #[error("Configuration error: {reason}")]
    ConfigError {
        /// Description of the configuration error
        reason: String,
    },
}

/// Result type for preflight operations
pub type PreflightResult<T> = Result<T, PreflightError>;

// ============================================================================
// Preflight Check Trait (Jidoka: Quality built into every step)
// ============================================================================

/// Trait for preflight validation checks
///
/// Implements Jidoka principle: each check validates one aspect of system state
/// and fails fast if the condition is not met.
pub trait PreflightCheck: fmt::Debug + Send + Sync {
    /// Unique identifier for this check (for logging and metrics)
    fn name(&self) -> &'static str;

    /// Validate the condition, returning Ok(()) if passed
    ///
    /// # Errors
    /// Returns `PreflightError` with detailed context if validation fails
    fn validate(&self) -> PreflightResult<()>;

    /// Optional: Description of what this check validates
    fn description(&self) -> &'static str {
        "Preflight validation check"
    }
}

// ============================================================================
// Deterministic Inference Configuration (per spec Section 4.1)
// ============================================================================

/// Configuration for deterministic inference
///
/// Per Fleming & Wallace [5], deterministic benchmarks require seed control
/// and elimination of randomness sources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeterministicInferenceConfig {
    /// Temperature = 0.0 disables sampling randomness
    pub temperature: f64,
    /// Fixed seed for any remaining randomness
    pub seed: u64,
    /// Top-k = 1 forces greedy decoding
    pub top_k: usize,
    /// Top-p = 1.0 disables nucleus sampling
    pub top_p: f64,
}

impl Default for DeterministicInferenceConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0, // Greedy decoding
            seed: 42,         // Fixed, reproducible seed
            top_k: 1,         // Deterministic token selection
            top_p: 1.0,       // Disable nucleus sampling
        }
    }
}

impl DeterministicInferenceConfig {
    /// Create a new deterministic config with custom seed
    #[must_use]
    pub fn with_seed(seed: u64) -> Self {
        Self {
            seed,
            ..Default::default()
        }
    }

    /// Validate that this config is truly deterministic
    ///
    /// # Errors
    /// Returns `PreflightError::ConfigError` if any parameter allows non-determinism:
    /// - temperature > 0.0 (allows sampling randomness)
    /// - top_k != 1 (allows multiple token choices)
    pub fn validate_determinism(&self) -> PreflightResult<()> {
        if self.temperature > 0.0 {
            return Err(PreflightError::ConfigError {
                reason: format!(
                    "Temperature {} > 0.0 allows randomness; set to 0.0 for determinism",
                    self.temperature
                ),
            });
        }
        if self.top_k != 1 {
            return Err(PreflightError::ConfigError {
                reason: format!(
                    "top_k {} != 1 allows multiple token choices; set to 1 for determinism",
                    self.top_k
                ),
            });
        }
        Ok(())
    }
}

// ============================================================================
// CV-Based Stopping Criterion (per spec Section 3.1)
// ============================================================================

/// Reason for stopping benchmark iteration
#[derive(Debug, Clone, PartialEq)]
pub enum StopReason {
    /// CV threshold achieved (statistically sufficient)
    CvConverged(f64),
    /// Maximum samples reached (bounded resource usage)
    MaxSamples,
    /// Minimum samples not yet reached
    Continue,
}

/// Decision from stopping criterion
#[derive(Debug, Clone, PartialEq)]
pub enum StopDecision {
    /// Continue collecting samples
    Continue,
    /// Stop with given reason
    Stop(StopReason),
}

/// CV-based stopping criterion per Hoefler & Belli SC'15 [1]
#[derive(Debug, Clone)]
pub struct CvStoppingCriterion {
    /// Minimum samples before CV check (prevents premature stopping)
    pub min_samples: usize,
    /// Maximum samples (bounded resource usage)
    pub max_samples: usize,
    /// Target CV threshold (e.g., 0.05 = 5%)
    pub cv_threshold: f64,
}

impl Default for CvStoppingCriterion {
    fn default() -> Self {
        Self {
            min_samples: 5,
            max_samples: 30,
            cv_threshold: 0.05, // 5% CV target per SC'15
        }
    }
}

impl CvStoppingCriterion {
    /// Create new criterion with custom parameters
    #[must_use]
    pub fn new(min_samples: usize, max_samples: usize, cv_threshold: f64) -> Self {
        Self {
            min_samples,
            max_samples,
            cv_threshold,
        }
    }

    /// Evaluate whether to stop based on current samples
    #[must_use]
    pub fn should_stop(&self, samples: &[f64]) -> StopDecision {
        if samples.len() < self.min_samples {
            return StopDecision::Continue;
        }
        if samples.len() >= self.max_samples {
            return StopDecision::Stop(StopReason::MaxSamples);
        }

        let cv = self.calculate_cv(samples);
        if cv < self.cv_threshold {
            StopDecision::Stop(StopReason::CvConverged(cv))
        } else {
            StopDecision::Continue
        }
    }

    /// Calculate coefficient of variation (std_dev / mean)
    #[must_use]
    pub fn calculate_cv(&self, samples: &[f64]) -> f64 {
        if samples.len() < 2 {
            return f64::MAX;
        }

        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;

        if mean.abs() < f64::EPSILON {
            return f64::MAX;
        }

        let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();
        std_dev / mean
    }
}

// ============================================================================
// Outlier Detection (per spec Section 3.3)
// ============================================================================

/// Outlier detection using Median Absolute Deviation (MAD)
///
/// Per Chen et al. [4], MAD is more robust than standard deviation
/// for samples with potential outliers.
pub struct OutlierDetector {
    /// k-factor for threshold (3.0 = ~99.7% for normal distribution)
    pub k_factor: f64,
}

impl Default for OutlierDetector {
    fn default() -> Self {
        Self { k_factor: 3.0 }
    }
}

impl OutlierDetector {
    /// Create detector with custom k-factor
    #[must_use]
    pub fn new(k_factor: f64) -> Self {
        Self { k_factor }
    }

    /// Detect outliers in samples
    ///
    /// Returns a boolean vector where `true` indicates an outlier
    #[must_use]
    pub fn detect(&self, samples: &[f64]) -> Vec<bool> {
        if samples.len() < 3 {
            return vec![false; samples.len()];
        }

        let median = Self::percentile(samples, 50.0);
        let deviations: Vec<f64> = samples.iter().map(|x| (x - median).abs()).collect();
        let mad = Self::percentile(&deviations, 50.0);

        // 1.4826 scales MAD to equivalent std dev for normal distribution
        let threshold = self.k_factor * mad * 1.4826;

        samples
            .iter()
            .map(|x| (x - median).abs() > threshold)
            .collect()
    }

    /// Filter outliers from samples, returning clean samples
    #[must_use]
    pub fn filter(&self, samples: &[f64]) -> Vec<f64> {
        let outliers = self.detect(samples);
        samples
            .iter()
            .zip(outliers.iter())
            .filter(|(_, is_outlier)| !**is_outlier)
            .map(|(sample, _)| *sample)
            .collect()
    }

    /// Calculate percentile of samples
    fn percentile(samples: &[f64], p: f64) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }

        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }
}

// ============================================================================
// Quality Metrics (per spec Section 6.1)
// ============================================================================

/// Quality metrics for benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// CV at the point where benchmark stopped
    pub cv_at_stop: f64,
    /// Whether CV threshold was achieved
    pub cv_converged: bool,
    /// Number of outliers detected
    pub outliers_detected: usize,
    /// Number of outliers excluded from statistics
    pub outliers_excluded: usize,
    /// List of preflight checks that passed
    pub preflight_checks_passed: Vec<String>,
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            cv_at_stop: f64::MAX,
            cv_converged: false,
            outliers_detected: 0,
            outliers_excluded: 0,
            preflight_checks_passed: Vec::new(),
        }
    }
}

// ============================================================================
// Preflight Check Implementations
// ============================================================================

/// Check that a configuration is deterministic
#[derive(Debug)]
pub struct DeterminismCheck {
    config: DeterministicInferenceConfig,
}

impl DeterminismCheck {
    /// Create a new determinism check for the given config
    #[must_use]
    pub fn new(config: DeterministicInferenceConfig) -> Self {
        Self { config }
    }
}

impl PreflightCheck for DeterminismCheck {
    fn name(&self) -> &'static str {
        "determinism_check"
    }

    fn description(&self) -> &'static str {
        "Validates inference configuration ensures deterministic output"
    }

    fn validate(&self) -> PreflightResult<()> {
        self.config.validate_determinism()
    }
}

include!("check.rs");
include!("bench_preflight_model_check.rs");
