//! Data drift detection for model retraining triggers.
//!
//! Provides statistical tests to detect when model performance degrades
//! or input data distribution shifts, triggering retraining.

use crate::primitives::{Matrix, Vector};

/// Drift detection result.
#[derive(Clone, Debug, PartialEq)]
pub enum DriftStatus {
    /// No significant drift detected
    NoDrift,
    /// Warning level drift (monitor closely)
    Warning { score: f32 },
    /// Significant drift detected (retrain recommended)
    Drift { score: f32 },
}

impl DriftStatus {
    /// Check if drift requires retraining.
    #[must_use]
    pub fn needs_retraining(&self) -> bool {
        matches!(self, DriftStatus::Drift { .. })
    }

    /// Get the drift score if available.
    #[must_use]
    pub fn score(&self) -> Option<f32> {
        match self {
            DriftStatus::NoDrift => None,
            DriftStatus::Warning { score } | DriftStatus::Drift { score } => Some(*score),
        }
    }
}

/// Configuration for drift detection.
#[derive(Clone, Debug)]
pub struct DriftConfig {
    /// Threshold for warning level
    pub warning_threshold: f32,
    /// Threshold for drift level (retrain trigger)
    pub drift_threshold: f32,
    /// Minimum samples required for drift detection
    pub min_samples: usize,
    /// Window size for rolling statistics
    pub window_size: usize,
}

impl Default for DriftConfig {
    fn default() -> Self {
        Self {
            warning_threshold: 0.1,
            drift_threshold: 0.2,
            min_samples: 30,
            window_size: 100,
        }
    }
}

impl DriftConfig {
    /// Create a new config with custom thresholds.
    #[must_use]
    pub fn new(warning: f32, drift: f32) -> Self {
        Self {
            warning_threshold: warning,
            drift_threshold: drift,
            ..Default::default()
        }
    }

    /// Set minimum samples required.
    #[must_use]
    pub fn with_min_samples(mut self, min: usize) -> Self {
        self.min_samples = min;
        self
    }

    /// Set window size for rolling statistics.
    #[must_use]
    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }
}

/// Drift detector using statistical distance measures.
///
/// # Examples
///
/// ```
/// use aprender::metrics::drift::{DriftDetector, DriftConfig, DriftStatus};
/// use aprender::primitives::Vector;
///
/// let reference = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
/// let detector = DriftDetector::new(DriftConfig::default());
///
/// // Similar distribution - no drift
/// let current = Vector::from_slice(&[1.1, 2.1, 3.0, 4.0, 5.1]);
/// let status = detector.detect_univariate(&reference, &current);
/// assert!(matches!(status, DriftStatus::NoDrift | DriftStatus::Warning { .. }));
/// ```
#[derive(Debug)]
pub struct DriftDetector {
    config: DriftConfig,
}

impl DriftDetector {
    /// Create a new drift detector.
    #[must_use]
    pub fn new(config: DriftConfig) -> Self {
        Self { config }
    }

    /// Detect drift in univariate data using normalized mean difference.
    ///
    /// Uses (|`mean_ref` - `mean_cur`| / `std_ref`) as drift measure.
    #[must_use]
    pub fn detect_univariate(&self, reference: &Vector<f32>, current: &Vector<f32>) -> DriftStatus {
        if reference.len() < self.config.min_samples || current.len() < self.config.min_samples {
            return DriftStatus::NoDrift;
        }

        let ref_mean = reference.mean();
        let cur_mean = current.mean();
        let ref_std = std_dev(reference.as_slice(), ref_mean);

        if ref_std < 1e-10 {
            return DriftStatus::NoDrift;
        }

        let score = (ref_mean - cur_mean).abs() / ref_std;

        self.classify_drift(score)
    }

    /// Detect drift in multivariate data using feature-wise analysis.
    ///
    /// Returns drift status for each feature and overall status.
    #[must_use]
    pub fn detect_multivariate(
        &self,
        reference: &Matrix<f32>,
        current: &Matrix<f32>,
    ) -> (DriftStatus, Vec<DriftStatus>) {
        let n_features = reference.n_cols();
        let mut feature_statuses = Vec::with_capacity(n_features);
        let mut max_score: f32 = 0.0;

        for col in 0..n_features {
            let ref_col = reference.column(col);
            let cur_col = current.column(col);

            let status = self.detect_univariate(&ref_col, &cur_col);
            if let Some(score) = status.score() {
                max_score = max_score.max(score);
            }
            feature_statuses.push(status);
        }

        let overall = self.classify_drift(max_score);
        (overall, feature_statuses)
    }

    /// Detect performance drift using accuracy/score degradation.
    ///
    /// Compares baseline performance to current performance.
    #[must_use]
    pub fn detect_performance_drift(
        &self,
        baseline_scores: &[f32],
        current_scores: &[f32],
    ) -> DriftStatus {
        if baseline_scores.is_empty() || current_scores.is_empty() {
            return DriftStatus::NoDrift;
        }

        let baseline_mean = mean(baseline_scores);
        let current_mean = mean(current_scores);
        let baseline_std = std_dev(baseline_scores, baseline_mean);

        if baseline_std < 1e-10 {
            // Use relative drop instead
            let relative_drop = (baseline_mean - current_mean) / baseline_mean.abs().max(1e-10);
            return self.classify_drift(relative_drop.max(0.0));
        }

        // Negative normalized score difference (degradation)
        let score = (baseline_mean - current_mean) / baseline_std;

        // Only trigger on performance drop (positive score means current is worse)
        self.classify_drift(score.max(0.0))
    }

    /// Classify drift score into status.
    fn classify_drift(&self, score: f32) -> DriftStatus {
        if score >= self.config.drift_threshold {
            DriftStatus::Drift { score }
        } else if score >= self.config.warning_threshold {
            DriftStatus::Warning { score }
        } else {
            DriftStatus::NoDrift
        }
    }
}

/// Rolling drift monitor for streaming data.
///
/// Maintains a reference window and detects drift in incoming data.
#[derive(Debug)]
pub struct RollingDriftMonitor {
    /// Reference data window
    reference_window: Vec<f32>,
    /// Current data window
    current_window: Vec<f32>,
    /// Detector configuration
    detector: DriftDetector,
    /// Maximum window size
    max_window: usize,
}

impl RollingDriftMonitor {
    /// Create a new rolling monitor.
    #[must_use]
    pub fn new(config: DriftConfig) -> Self {
        let max_window = config.window_size;
        Self {
            reference_window: Vec::with_capacity(max_window),
            current_window: Vec::with_capacity(max_window),
            detector: DriftDetector::new(config),
            max_window,
        }
    }

    /// Set the reference distribution from baseline data.
    pub fn set_reference(&mut self, data: &[f32]) {
        self.reference_window.clear();
        let start = if data.len() > self.max_window {
            data.len() - self.max_window
        } else {
            0
        };
        self.reference_window.extend_from_slice(&data[start..]);
    }

    /// Add a new observation and check for drift.
    pub fn observe(&mut self, value: f32) -> DriftStatus {
        self.current_window.push(value);

        // Maintain window size
        if self.current_window.len() > self.max_window {
            self.current_window.remove(0);
        }

        self.check_drift()
    }

    /// Check current drift status.
    #[must_use]
    pub fn check_drift(&self) -> DriftStatus {
        if self.reference_window.is_empty() {
            return DriftStatus::NoDrift;
        }

        let ref_vec = Vector::from_slice(&self.reference_window);
        let cur_vec = Vector::from_slice(&self.current_window);

        self.detector.detect_univariate(&ref_vec, &cur_vec)
    }

    /// Reset the current window (e.g., after retraining).
    pub fn reset_current(&mut self) {
        self.current_window.clear();
    }

    /// Update reference to current (after retraining).
    pub fn update_reference(&mut self) {
        self.reference_window.clone_from(&self.current_window);
        self.current_window.clear();
    }
}

/// Retraining trigger that combines multiple drift signals.
#[derive(Debug)]
pub struct RetrainingTrigger {
    /// Performance monitor
    performance_monitor: RollingDriftMonitor,
    /// Feature drift monitors (one per feature)
    feature_monitors: Vec<RollingDriftMonitor>,
    /// Number of consecutive drift detections required
    consecutive_required: usize,
    /// Current consecutive count
    consecutive_count: usize,
}

impl RetrainingTrigger {
    /// Create a new retraining trigger.
    #[must_use]
    pub fn new(n_features: usize, config: DriftConfig) -> Self {
        let feature_monitors = (0..n_features)
            .map(|_| RollingDriftMonitor::new(config.clone()))
            .collect();

        Self {
            performance_monitor: RollingDriftMonitor::new(config),
            feature_monitors,
            consecutive_required: 3,
            consecutive_count: 0,
        }
    }

    /// Set consecutive detections required to trigger.
    #[must_use]
    pub fn with_consecutive_required(mut self, count: usize) -> Self {
        self.consecutive_required = count.max(1);
        self
    }

    /// Set baseline performance scores.
    pub fn set_baseline_performance(&mut self, scores: &[f32]) {
        self.performance_monitor.set_reference(scores);
    }

    /// Set baseline feature distributions.
    pub fn set_baseline_features(&mut self, features: &Matrix<f32>) {
        for (i, monitor) in self.feature_monitors.iter_mut().enumerate() {
            if i < features.n_cols() {
                let col: Vec<f32> = (0..features.n_rows()).map(|r| features.get(r, i)).collect();
                monitor.set_reference(&col);
            }
        }
    }

    /// Observe new performance score and check trigger.
    pub fn observe_performance(&mut self, score: f32) -> bool {
        let status = self.performance_monitor.observe(score);

        if status.needs_retraining() {
            self.consecutive_count += 1;
        } else {
            self.consecutive_count = 0;
        }

        self.consecutive_count >= self.consecutive_required
    }

    /// Check if retraining is triggered.
    #[must_use]
    pub fn is_triggered(&self) -> bool {
        self.consecutive_count >= self.consecutive_required
    }

    /// Reset after retraining.
    pub fn reset(&mut self) {
        self.consecutive_count = 0;
        self.performance_monitor.reset_current();
        for monitor in &mut self.feature_monitors {
            monitor.reset_current();
        }
    }
}

// Helper functions

fn mean(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f32>() / data.len() as f32
}

fn std_dev(data: &[f32], mean_val: f32) -> f32 {
    if data.len() < 2 {
        return 0.0;
    }
    let variance: f32 =
        data.iter().map(|x| (x - mean_val).powi(2)).sum::<f32>() / (data.len() - 1) as f32;
    variance.sqrt()
}

#[cfg(test)]
#[path = "drift_tests.rs"]
mod tests;
