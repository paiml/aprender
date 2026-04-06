//! Dynamic Adaptive Thresholds with ML (PMAT-049)
//!
//! Self-learning workload-specific thresholds using multivariate models.
//!
//! # Design
//!
//! - Workload fingerprinting using CV pattern analysis
//! - Multivariate modeling with feature correlation
//! - Confidence scoring with uncertainty estimation
//! - Drift detection with 24h re-calibration triggers
//!
//! # Falsification (FKR-050)
//!
//! H₀: ML thresholds cannot reduce false positives compared to static thresholds
//! Test: Compare precision/recall on labeled dataset with injected anomalies

mod threshold;
mod types;

pub use threshold::{LearnedWorkloadThreshold, MlThresholdConfig};
pub use types::{
    AnomalyResult, ClassificationMetrics, MlThresholdError, MlThresholdResult, TimeSeriesFeatures,
    WorkloadClass,
};

use std::collections::HashMap;

/// ML-based adaptive threshold system
#[derive(Debug)]
pub struct AdaptiveThresholdMl {
    /// Configuration
    config: MlThresholdConfig,
    /// Per-workload learned thresholds
    thresholds: HashMap<WorkloadClass, LearnedWorkloadThreshold>,
    /// Classification metrics
    metrics: ClassificationMetrics,
    /// Training mode enabled
    training_mode: bool,
    /// Global baseline threshold (used during cold start)
    global_threshold: f64,
}

impl AdaptiveThresholdMl {
    /// Create a new adaptive threshold system
    pub fn new(config: MlThresholdConfig) -> Self {
        Self {
            config,
            thresholds: HashMap::new(),
            metrics: ClassificationMetrics::default(),
            training_mode: true,
            global_threshold: 15.0,
        }
    }

    /// Classify workload from features
    pub fn classify_workload(&self, features: &TimeSeriesFeatures) -> WorkloadClass {
        // Simple rule-based classification based on CV and autocorrelation
        if features.cv < 10.0 && features.autocorr_lag1 > 0.5 {
            WorkloadClass::ComputeBound
        } else if features.cv > 18.0 && features.autocorr_lag1 < 0.3 {
            WorkloadClass::MemoryBound
        } else if features.cv < 12.0 {
            WorkloadClass::Matmul
        } else if features.cv > 15.0 {
            WorkloadClass::Ffn
        } else {
            WorkloadClass::Attention
        }
    }

    /// Get threshold for a workload
    pub fn get_threshold(&self, workload: WorkloadClass) -> f64 {
        if let Some(learned) = self.thresholds.get(&workload) {
            if learned.confidence >= self.config.min_confidence
                && learned.training_samples >= self.config.min_training_samples
                && !learned.is_stale(self.config.max_threshold_age)
            {
                return learned.cv_threshold;
            }
        }

        // Cold start: use default with conservative multiplier
        workload.default_cv_threshold() * self.config.cold_start_multiplier
    }

    /// Detect anomaly in a time series
    pub fn detect_anomaly(&self, values: &[f64]) -> MlThresholdResult<AnomalyResult> {
        let features =
            TimeSeriesFeatures::extract(values).ok_or(MlThresholdError::InsufficientData {
                have: values.len(),
                need: 10,
            })?;

        let workload = self.classify_workload(&features);
        let threshold = self.get_threshold(workload);

        let is_anomaly = features.cv > threshold;
        let score = features.cv / threshold;

        let confidence = self
            .thresholds
            .get(&workload)
            .map(|t| t.confidence)
            .unwrap_or(0.0);

        let reason = if is_anomaly {
            format!("CV {:.2}% exceeds threshold {:.2}%", features.cv, threshold)
        } else {
            format!("CV {:.2}% within threshold {:.2}%", features.cv, threshold)
        };

        Ok(AnomalyResult {
            is_anomaly,
            score,
            threshold,
            confidence,
            workload_class: workload,
            reason,
        })
    }

    /// Train on labeled sample
    pub fn train(&mut self, values: &[f64], is_anomaly: bool) -> MlThresholdResult<()> {
        let features =
            TimeSeriesFeatures::extract(values).ok_or(MlThresholdError::InsufficientData {
                have: values.len(),
                need: 10,
            })?;

        let workload = self.classify_workload(&features);

        // Get or create threshold for this workload
        let threshold = self
            .thresholds
            .entry(workload)
            .or_insert_with(|| LearnedWorkloadThreshold::new(workload));

        threshold.update(&features, is_anomaly);

        // Update classification metrics
        let predicted = features.cv > threshold.cv_threshold;
        match (predicted, is_anomaly) {
            (true, true) => self.metrics.true_positives += 1,
            (true, false) => self.metrics.false_positives += 1,
            (false, false) => self.metrics.true_negatives += 1,
            (false, true) => self.metrics.false_negatives += 1,
        }

        Ok(())
    }

    /// Check for drift in recent samples
    pub fn check_drift(&self, values: &[f64]) -> MlThresholdResult<Option<f64>> {
        let features =
            TimeSeriesFeatures::extract(values).ok_or(MlThresholdError::InsufficientData {
                have: values.len(),
                need: 10,
            })?;

        let workload = self.classify_workload(&features);

        if let Some(threshold) = self.thresholds.get(&workload) {
            Ok(threshold.check_drift(&features))
        } else {
            Ok(None)
        }
    }

    /// Get classification metrics
    pub fn get_metrics(&self) -> &ClassificationMetrics {
        &self.metrics
    }

    /// Reset classification metrics
    pub fn reset_metrics(&mut self) {
        self.metrics = ClassificationMetrics::default();
    }

    /// Get learned threshold for workload
    pub fn get_learned_threshold(
        &self,
        workload: WorkloadClass,
    ) -> Option<&LearnedWorkloadThreshold> {
        self.thresholds.get(&workload)
    }

    /// Get all workload classes with learned thresholds
    pub fn learned_workloads(&self) -> Vec<WorkloadClass> {
        self.thresholds.keys().copied().collect()
    }

    /// Export model state for persistence
    pub fn export_state(&self) -> HashMap<String, (f64, f64, usize)> {
        self.thresholds
            .iter()
            .map(|(k, v)| {
                (
                    k.name().to_string(),
                    (v.cv_threshold, v.confidence, v.training_samples),
                )
            })
            .collect()
    }

    /// Import model state
    pub fn import_state(&mut self, state: HashMap<String, (f64, f64, usize)>) {
        for (name, (threshold, confidence, samples)) in state {
            let Some(workload) = WorkloadClass::from_name(&name) else {
                continue;
            };

            let mut learned = LearnedWorkloadThreshold::new(workload);
            learned.cv_threshold = threshold;
            learned.confidence = confidence;
            learned.training_samples = samples;

            self.thresholds.insert(workload, learned);
        }
    }

    /// Get configuration
    pub fn config(&self) -> &MlThresholdConfig {
        &self.config
    }
}

/// Default minimum training samples
pub const DEFAULT_MIN_TRAINING_SAMPLES: usize = 50;

/// Default minimum confidence
pub const DEFAULT_MIN_CONFIDENCE: f64 = 0.7;

/// Default drift z-score threshold
pub const DEFAULT_DRIFT_ZSCORE: f64 = 3.0;

#[cfg(test)]
mod tests;
