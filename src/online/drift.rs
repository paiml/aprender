//! Drift Detection for Triggering Model Retraining
//!
//! This module provides statistical drift detectors that monitor prediction
//! error rates to detect concept drift and trigger retraining.
//!
//! # References
//!
//! - [Gama et al. 2004] "Learning with Drift Detection" - DDM algorithm
//! - [Bifet & Gavalda 2007] "Learning from Time-Changing Data with Adaptive Windowing" - ADWIN
//! - [Page 1954] "Continuous Inspection Schemes" - Page-Hinkley test
//!
//! # Toyota Way Principles
//!
//! - **Jidoka**: Drift detection stops bad predictions automatically
//! - **Just-in-Time**: Retrain only when drift detected, not on schedule

use std::collections::VecDeque;

/// Drift status indicating current model health
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DriftStatus {
    /// No drift detected - model performing well
    #[default]
    Stable,
    /// Warning level - performance degrading, start collecting data
    Warning,
    /// Drift confirmed - trigger retraining
    Drift,
}

/// Statistics from drift detection
#[derive(Debug, Clone)]
pub struct DriftStats {
    /// Number of samples observed
    pub n_samples: u64,
    /// Current error rate
    pub error_rate: f64,
    /// Minimum error rate observed
    pub min_error_rate: f64,
    /// Current standard deviation
    pub std_dev: f64,
    /// Current drift status
    pub status: DriftStatus,
}

/// Drift detection for triggering model retraining
///
/// Reference: [Gama et al. 2004] "Learning with Drift Detection"
pub trait DriftDetector: Send + Sync {
    /// Add new prediction outcome
    ///
    /// # Arguments
    /// * `error` - true if prediction was wrong, false if correct
    fn add_element(&mut self, error: bool);

    /// Check current drift status
    fn detected_change(&self) -> DriftStatus;

    /// Reset detector after handling drift
    fn reset(&mut self);

    /// Get detector statistics
    fn stats(&self) -> DriftStats;
}

/// DDM (Drift Detection Method) implementation
///
/// Reference: [Gama et al. 2004]
/// - Warning: p + s > `p_min` + 2*`s_min`
/// - Drift: p + s > `p_min` + 3*`s_min`
///
/// Suitable for detecting sudden concept drift.
#[derive(Debug, Clone)]
pub struct DDM {
    /// Minimum samples before detection
    min_samples: u64,
    /// Warning threshold (standard deviations)
    warning_level: f64,
    /// Drift threshold (standard deviations)
    drift_level: f64,
    /// Number of samples seen
    n: u64,
    /// Error probability
    p: f64,
    /// Standard deviation
    s: f64,
    /// Minimum error probability
    p_min: f64,
    /// Standard deviation at minimum
    s_min: f64,
    /// Current status
    status: DriftStatus,
}

impl Default for DDM {
    fn default() -> Self {
        Self::new()
    }
}

impl DDM {
    /// Create a new DDM detector with default thresholds
    ///
    /// Default: warning at 2σ, drift at 3σ (as per Gama et al.)
    #[must_use]
    pub fn new() -> Self {
        Self {
            min_samples: 30,
            warning_level: 2.0,
            drift_level: 3.0,
            n: 0,
            p: 1.0,
            s: 0.0,
            p_min: f64::MAX,
            s_min: f64::MAX,
            status: DriftStatus::Stable,
        }
    }

    /// Create DDM with custom thresholds
    ///
    /// # Arguments
    /// * `min_samples` - Minimum samples before detection
    /// * `warning_level` - Standard deviations for warning
    /// * `drift_level` - Standard deviations for drift
    #[must_use]
    pub fn with_thresholds(min_samples: u64, warning_level: f64, drift_level: f64) -> Self {
        Self {
            min_samples,
            warning_level,
            drift_level,
            ..Self::new()
        }
    }
}

impl DriftDetector for DDM {
    fn add_element(&mut self, error: bool) {
        self.n += 1;

        // Update error probability (online mean)
        let error_val = if error { 1.0 } else { 0.0 };
        self.p += (error_val - self.p) / self.n as f64;

        // Standard deviation of Bernoulli: sqrt(p(1-p)/n)
        self.s = (self.p * (1.0 - self.p) / self.n as f64).sqrt();

        if self.n >= self.min_samples {
            // Update minimums
            if self.p + self.s < self.p_min + self.s_min {
                self.p_min = self.p;
                self.s_min = self.s;
            }

            // Check drift conditions
            if self.p + self.s > self.p_min + self.drift_level * self.s_min {
                self.status = DriftStatus::Drift;
            } else if self.p + self.s > self.p_min + self.warning_level * self.s_min {
                self.status = DriftStatus::Warning;
            } else {
                self.status = DriftStatus::Stable;
            }
        }
    }

    fn detected_change(&self) -> DriftStatus {
        self.status
    }

    fn reset(&mut self) {
        self.n = 0;
        self.p = 1.0;
        self.s = 0.0;
        self.p_min = f64::MAX;
        self.s_min = f64::MAX;
        self.status = DriftStatus::Stable;
    }

    fn stats(&self) -> DriftStats {
        DriftStats {
            n_samples: self.n,
            error_rate: self.p,
            min_error_rate: if self.p_min == f64::MAX {
                0.0
            } else {
                self.p_min
            },
            std_dev: self.s,
            status: self.status,
        }
    }
}

/// Page-Hinkley Test for gradual drift
///
/// Reference: [Page 1954] "Continuous Inspection Schemes"
/// - Cumulative sum test for mean shift detection
/// - Better for gradual drift than DDM
#[derive(Debug, Clone)]
pub struct PageHinkley {
    /// Minimum magnitude of change to detect
    delta: f64,
    /// Detection threshold
    lambda: f64,
    /// Cumulative sum
    sum: f64,
    /// Running mean
    mean: f64,
    /// Number of samples
    n: u64,
    /// Minimum cumulative sum seen
    min_sum: f64,
    /// Current status
    status: DriftStatus,
}

impl Default for PageHinkley {
    fn default() -> Self {
        Self::new()
    }
}

impl PageHinkley {
    /// Create a new Page-Hinkley detector with default thresholds
    #[must_use]
    pub fn new() -> Self {
        Self {
            delta: 0.005,
            lambda: 50.0,
            sum: 0.0,
            mean: 0.0,
            n: 0,
            min_sum: f64::MAX,
            status: DriftStatus::Stable,
        }
    }

    /// Create with custom thresholds
    ///
    /// # Arguments
    /// * `delta` - Minimum magnitude of change to detect
    /// * `lambda` - Detection threshold
    #[must_use]
    pub fn with_thresholds(delta: f64, lambda: f64) -> Self {
        Self {
            delta,
            lambda,
            ..Self::new()
        }
    }
}

impl DriftDetector for PageHinkley {
    fn add_element(&mut self, error: bool) {
        self.n += 1;
        let x = if error { 1.0 } else { 0.0 };

        // Update running mean
        self.mean += (x - self.mean) / self.n as f64;

        // Update cumulative sum
        self.sum += x - self.mean - self.delta;

        // Track minimum
        if self.sum < self.min_sum {
            self.min_sum = self.sum;
        }

        // Check for drift
        if self.sum - self.min_sum > self.lambda {
            self.status = DriftStatus::Drift;
        } else if self.sum - self.min_sum > self.lambda * 0.5 {
            self.status = DriftStatus::Warning;
        } else {
            self.status = DriftStatus::Stable;
        }
    }

    fn detected_change(&self) -> DriftStatus {
        self.status
    }

    fn reset(&mut self) {
        self.sum = 0.0;
        self.mean = 0.0;
        self.n = 0;
        self.min_sum = f64::MAX;
        self.status = DriftStatus::Stable;
    }

    fn stats(&self) -> DriftStats {
        DriftStats {
            n_samples: self.n,
            error_rate: self.mean,
            min_error_rate: 0.0,
            std_dev: (self.sum - self.min_sum).abs(),
            status: self.status,
        }
    }
}

/// Bucket for ADWIN's exponential histogram
#[derive(Debug, Clone)]
struct Bucket {
    /// Total sum in bucket
    total: f64,
    /// Count of elements
    count: usize,
}

/// ADWIN (Adaptive Windowing) for variable-rate drift
///
/// Reference: [Bifet & Gavalda 2007] "Learning from Time-Changing Data
/// with Adaptive Windowing"
/// - Automatically adjusts window size
/// - Detects both abrupt and gradual drift
/// - Recommended as default per Toyota Way review
///
/// This is the **recommended default** drift detector as it handles both
/// sudden and gradual drift without manual threshold tuning.
#[derive(Debug, Clone)]
pub struct ADWIN {
    /// Confidence parameter (smaller = more sensitive)
    delta: f64,
    /// Maximum buckets per row
    max_buckets: usize,
    /// Bucket rows (exponential histogram)
    bucket_rows: Vec<VecDeque<Bucket>>,
    /// Total sum of all elements
    total: f64,
    /// Total count of elements
    count: usize,
    /// Window width
    width: usize,
    /// Current status
    status: DriftStatus,
    /// Last detected change position
    last_bucket_row: usize,
}

impl Default for ADWIN {
    fn default() -> Self {
        Self::new()
    }
}

include!("drift_part_02.rs");
include!("drift_part_03.rs");
