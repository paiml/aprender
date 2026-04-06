//! Learned workload thresholds and configuration.

use std::time::{Duration, Instant};

use super::types::{TimeSeriesFeatures, WorkloadClass};

/// Learned threshold for a specific workload
#[derive(Debug, Clone)]
pub struct LearnedWorkloadThreshold {
    /// Workload class
    pub workload_class: WorkloadClass,
    /// Learned CV threshold
    pub cv_threshold: f64,
    /// Confidence in the threshold (0-1)
    pub confidence: f64,
    /// Number of training samples
    pub training_samples: usize,
    /// Last update time
    pub last_updated: Instant,
    /// Feature means for drift detection
    pub feature_means: Vec<f64>,
    /// Feature standard deviations for drift detection
    pub feature_stds: Vec<f64>,
}

impl LearnedWorkloadThreshold {
    /// Create a new learned threshold
    pub fn new(workload_class: WorkloadClass) -> Self {
        Self {
            workload_class,
            cv_threshold: workload_class.default_cv_threshold(),
            confidence: 0.0,
            training_samples: 0,
            last_updated: Instant::now(),
            feature_means: Vec::new(),
            feature_stds: Vec::new(),
        }
    }

    /// Update threshold from training data
    pub fn update(&mut self, features: &TimeSeriesFeatures, is_anomaly: bool) {
        self.training_samples += 1;
        self.last_updated = Instant::now();

        // Update threshold based on observed CV
        if !is_anomaly {
            // Normal sample: threshold should be above observed CV
            let observed_cv = features.cv;
            let margin = 1.2; // 20% margin above normal

            // Weighted update
            let weight = 0.1; // Learning rate
            let target = observed_cv * margin;

            if target > self.cv_threshold {
                self.cv_threshold = self.cv_threshold * (1.0 - weight) + target * weight;
            }
        }

        // Update confidence based on sample count
        self.confidence = (self.training_samples as f64 / 100.0).min(1.0);

        // Update feature statistics for drift detection
        let fv = features.to_vec();
        if self.feature_means.is_empty() {
            self.feature_means = fv.clone();
            self.feature_stds = vec![0.0; fv.len()];
        } else {
            // Online mean and variance update (Welford's algorithm)
            let n = self.training_samples as f64;
            for (i, &val) in fv.iter().enumerate() {
                let delta = val - self.feature_means[i];
                self.feature_means[i] += delta / n;
                let delta2 = val - self.feature_means[i];
                // Update variance estimate
                if n > 1.0 {
                    self.feature_stds[i] = ((n - 2.0) / (n - 1.0) * self.feature_stds[i].powi(2)
                        + delta * delta2 / n)
                        .sqrt();
                }
            }
        }
    }

    /// Check if drift is detected
    pub fn check_drift(&self, features: &TimeSeriesFeatures) -> Option<f64> {
        if self.feature_means.is_empty() {
            return None;
        }

        let fv = features.to_vec();
        let mut max_zscore = 0.0_f64;

        for (i, &val) in fv.iter().enumerate() {
            if self.feature_stds[i] > 1e-10 {
                let zscore = ((val - self.feature_means[i]) / self.feature_stds[i]).abs();
                max_zscore = max_zscore.max(zscore);
            }
        }

        if max_zscore > 3.0 {
            Some(max_zscore)
        } else {
            None
        }
    }

    /// Check if threshold is stale (needs re-calibration)
    pub fn is_stale(&self, max_age: Duration) -> bool {
        self.last_updated.elapsed() > max_age
    }
}

/// Configuration for ML threshold system
#[derive(Debug, Clone)]
pub struct MlThresholdConfig {
    /// Minimum training samples before using learned threshold
    pub min_training_samples: usize,
    /// Minimum confidence to use learned threshold
    pub min_confidence: f64,
    /// Maximum age before threshold is stale
    pub max_threshold_age: Duration,
    /// Drift detection z-score threshold
    pub drift_zscore_threshold: f64,
    /// Conservative threshold multiplier for cold start
    pub cold_start_multiplier: f64,
}

impl Default for MlThresholdConfig {
    fn default() -> Self {
        Self {
            min_training_samples: 50,
            min_confidence: 0.7,
            max_threshold_age: Duration::from_secs(24 * 60 * 60), // 24 hours
            drift_zscore_threshold: 3.0,
            cold_start_multiplier: 1.5,
        }
    }
}
