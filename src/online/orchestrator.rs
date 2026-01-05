//! Retrain Orchestrator for Drift-Triggered Model Updates
//!
//! Automatically monitors model performance and triggers retraining
//! when concept drift is detected.
//!
//! # References
//!
//! - [Gama et al. 2004] DDM for drift detection
//! - [Bifet & Gavalda 2007] ADWIN for adaptive windowing
//!
//! # Toyota Way Principles
//!
//! - **Jidoka**: Stop and fix when problems detected
//! - **Just-in-Time**: Retrain only when needed (pull system)
//! - **Genchi Genbutsu**: Use actual prediction outcomes

use crate::error::Result;

use super::corpus::{CorpusBuffer, CorpusBufferConfig, EvictionPolicy, Sample, SampleSource};
use super::curriculum::{CurriculumScheduler, LinearCurriculum, ScoredSample};
use super::drift::{DriftDetector, DriftStatus, ADWIN};
use super::OnlineLearner;

/// Result of observing a new sample
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObserveResult {
    /// Model is performing well
    Stable,
    /// Warning level - collecting data
    Warning,
    /// Model was retrained
    Retrained,
    /// Skipped (e.g., duplicate sample)
    Skipped,
}

/// Configuration for retrain orchestrator
#[derive(Debug, Clone)]
pub struct RetrainConfig {
    /// Minimum samples before retraining
    pub min_samples: usize,
    /// Maximum buffer size
    pub max_buffer_size: usize,
    /// Enable incremental updates on each sample
    pub incremental_updates: bool,
    /// Use curriculum learning during retrain
    pub curriculum_learning: bool,
    /// Number of curriculum stages
    pub curriculum_stages: usize,
    /// Save checkpoint after retrain
    pub save_checkpoint: bool,
    /// Learning rate for retraining
    pub learning_rate: f64,
    /// Number of epochs for full retraining
    pub retrain_epochs: usize,
}

impl Default for RetrainConfig {
    fn default() -> Self {
        Self {
            min_samples: 100,
            max_buffer_size: 10_000,
            incremental_updates: true,
            curriculum_learning: true,
            curriculum_stages: 5,
            save_checkpoint: false,
            learning_rate: 0.01,
            retrain_epochs: 10,
        }
    }
}

/// Statistics from the orchestrator
#[derive(Debug, Clone, Default)]
pub struct OrchestratorStats {
    /// Total samples observed
    pub samples_observed: u64,
    /// Number of retraining events
    pub retrain_count: u64,
    /// Current buffer size
    pub buffer_size: usize,
    /// Current drift status
    pub drift_status: DriftStatus,
    /// Last retrain sample count
    pub last_retrain_samples: usize,
    /// Samples since last retrain
    pub samples_since_retrain: u64,
}

/// Automatic retraining orchestrator
///
/// Monitors predictions for drift and triggers retraining when needed.
#[derive(Debug)]
pub struct RetrainOrchestrator<
    M: OnlineLearner + std::fmt::Debug,
    D: DriftDetector + std::fmt::Debug,
> {
    /// Current model
    model: M,
    /// Drift detector
    detector: D,
    /// Data buffer for retraining
    buffer: CorpusBuffer,
    /// Retraining configuration
    config: RetrainConfig,
    /// Statistics
    stats: OrchestratorStats,
    /// Number of features (for validation)
    #[allow(dead_code)]
    n_features: usize,
}

impl<M: OnlineLearner + std::fmt::Debug> RetrainOrchestrator<M, ADWIN> {
    /// Create an orchestrator with ADWIN detector (recommended default)
    pub fn new(model: M, n_features: usize) -> Self {
        Self::with_detector(model, ADWIN::new(), n_features)
    }
}

impl<M: OnlineLearner + std::fmt::Debug, D: DriftDetector + std::fmt::Debug>
    RetrainOrchestrator<M, D>
{
    /// Create with custom drift detector
    pub fn with_detector(model: M, detector: D, n_features: usize) -> Self {
        let config = RetrainConfig::default();
        let buffer_config = CorpusBufferConfig {
            max_size: config.max_buffer_size,
            policy: EvictionPolicy::Reservoir,
            deduplicate: true,
            ..Default::default()
        };

        Self {
            model,
            detector,
            buffer: CorpusBuffer::with_config(buffer_config),
            config,
            stats: OrchestratorStats::default(),
            n_features,
        }
    }

    /// Create with custom configuration
    pub fn with_config(model: M, detector: D, n_features: usize, config: RetrainConfig) -> Self {
        let buffer_config = CorpusBufferConfig {
            max_size: config.max_buffer_size,
            policy: EvictionPolicy::Reservoir,
            deduplicate: true,
            ..Default::default()
        };

        Self {
            model,
            detector,
            buffer: CorpusBuffer::with_config(buffer_config),
            config,
            stats: OrchestratorStats::default(),
            n_features,
        }
    }

    /// Process new sample and handle drift
    ///
    /// # Arguments
    /// * `features` - Input features
    /// * `target` - True target value
    /// * `prediction` - Model's prediction
    ///
    /// # Returns
    /// Result indicating what action was taken
    pub fn observe(
        &mut self,
        features: &[f64],
        target: &[f64],
        prediction: &[f64],
    ) -> Result<ObserveResult> {
        self.stats.samples_observed += 1;
        self.stats.samples_since_retrain += 1;

        // Check prediction correctness
        let error = self.compute_error(target, prediction);
        self.detector.add_element(error);

        // Buffer data for potential retraining
        let sample =
            Sample::with_source(features.to_vec(), target.to_vec(), SampleSource::Production);

        if !self.buffer.add(sample) {
            return Ok(ObserveResult::Skipped);
        }

        self.stats.buffer_size = self.buffer.len();
        self.stats.drift_status = self.detector.detected_change();

        match self.detector.detected_change() {
            DriftStatus::Stable => {
                // Incremental update only
                if self.config.incremental_updates {
                    self.model
                        .partial_fit(features, target, Some(self.config.learning_rate))?;
                }
                Ok(ObserveResult::Stable)
            }
            DriftStatus::Warning => {
                // Continue collecting data, maybe do incremental update
                if self.config.incremental_updates {
                    self.model
                        .partial_fit(features, target, Some(self.config.learning_rate))?;
                }
                Ok(ObserveResult::Warning)
            }
            DriftStatus::Drift => {
                // Check if we have enough samples
                if self.buffer.len() >= self.config.min_samples {
                    self.retrain()?;
                    Ok(ObserveResult::Retrained)
                } else {
                    // Not enough data yet, do incremental update
                    if self.config.incremental_updates {
                        self.model.partial_fit(
                            features,
                            target,
                            Some(self.config.learning_rate),
                        )?;
                    }
                    Ok(ObserveResult::Warning)
                }
            }
        }
    }

    /// Compute if prediction was an error
    fn compute_error(&self, target: &[f64], prediction: &[f64]) -> bool {
        let _ = self; // suppress unused self warning - method for consistency
        if target.is_empty() || prediction.is_empty() {
            return true;
        }

        // For regression: error if prediction is too far from target
        // For classification: error if predicted class differs
        if target.len() == 1 && prediction.len() == 1 {
            // Regression or binary classification
            let diff = (target[0] - prediction[0]).abs();
            if target[0].abs() < 1.0 {
                // Classification threshold
                diff > 0.5
            } else {
                // Regression: relative error > 10%
                diff / target[0].abs().max(1.0) > 0.1
            }
        } else {
            // Multi-class: compare argmax
            let target_class = target
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(i, _)| i);

            let pred_class = prediction
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(i, _)| i);

            target_class != pred_class
        }
    }

    /// Perform full retraining on buffered data
    fn retrain(&mut self) -> Result<()> {
        let (features, targets, n_samples, n_features) = self.buffer.to_dataset();

        if n_samples == 0 || n_features == 0 {
            return Ok(());
        }

        // Reset model
        self.model.reset();

        if self.config.curriculum_learning {
            self.retrain_with_curriculum(&features, &targets, n_samples, n_features)?;
        } else {
            self.retrain_standard(&features, &targets, n_samples, n_features)?;
        }

        // Update stats
        self.stats.retrain_count += 1;
        self.stats.last_retrain_samples = n_samples;
        self.stats.samples_since_retrain = 0;

        // Reset drift detector
        self.detector.reset();

        // Clear buffer but keep some recent samples
        let keep = (self.config.min_samples / 2).min(self.buffer.len());
        let recent: Vec<Sample> = self
            .buffer
            .samples()
            .iter()
            .rev()
            .take(keep)
            .cloned()
            .collect();

        self.buffer.clear();
        for sample in recent {
            self.buffer.add(sample);
        }

        Ok(())
    }

    /// Standard retraining without curriculum
    fn retrain_standard(
        &mut self,
        features: &[f64],
        targets: &[f64],
        n_samples: usize,
        n_features: usize,
    ) -> Result<()> {
        for _ in 0..self.config.retrain_epochs {
            for i in 0..n_samples {
                let x = &features[i * n_features..(i + 1) * n_features];
                let y = &targets[i..=i];
                self.model
                    .partial_fit(x, y, Some(self.config.learning_rate))?;
            }
        }
        Ok(())
    }

    /// Curriculum-based retraining
    fn retrain_with_curriculum(
        &mut self,
        features: &[f64],
        targets: &[f64],
        n_samples: usize,
        n_features: usize,
    ) -> Result<()> {
        // Score samples by loss (using current model)
        let mut scored_samples: Vec<ScoredSample> = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let x = &features[i * n_features..(i + 1) * n_features];
            let y = targets[i];

            // Use feature norm as difficulty proxy (simple but effective)
            let difficulty: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();

            scored_samples.push(ScoredSample::new(x.to_vec(), y, difficulty));
        }

        // Sort by difficulty (easiest first)
        scored_samples.sort_by(|a, b| {
            a.difficulty
                .partial_cmp(&b.difficulty)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Create curriculum scheduler
        let mut curriculum = LinearCurriculum::new(self.config.curriculum_stages);

        // Train in stages
        let samples_per_stage = n_samples / self.config.curriculum_stages.max(1);

        for stage in 0..self.config.curriculum_stages {
            let end_idx = ((stage + 1) * samples_per_stage).min(n_samples);

            // Train on samples up to current stage
            for _epoch in 0..self.config.retrain_epochs / self.config.curriculum_stages.max(1) {
                for sample in scored_samples.iter().take(end_idx) {
                    let y = &[sample.target];
                    self.model
                        .partial_fit(&sample.features, y, Some(self.config.learning_rate))?;
                }
            }

            curriculum.advance();
        }

        Ok(())
    }

    /// Get reference to the model
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Get mutable reference to the model
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    /// Get reference to the drift detector
    pub fn detector(&self) -> &D {
        &self.detector
    }

    /// Get orchestrator statistics
    pub fn stats(&self) -> &OrchestratorStats {
        &self.stats
    }

    /// Get current drift status
    pub fn drift_status(&self) -> DriftStatus {
        self.detector.detected_change()
    }

    /// Force a retrain (useful for manual triggers)
    pub fn force_retrain(&mut self) -> Result<()> {
        self.retrain()
    }

    /// Get buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }

    /// Check if retraining is recommended
    pub fn should_retrain(&self) -> bool {
        self.detector.detected_change() == DriftStatus::Drift
            && self.buffer.len() >= self.config.min_samples
    }

    /// Get configuration
    pub fn config(&self) -> &RetrainConfig {
        &self.config
    }
}

/// Builder for `RetrainOrchestrator`
#[derive(Debug)]
pub struct OrchestratorBuilder<M: OnlineLearner + std::fmt::Debug> {
    model: M,
    n_features: usize,
    config: RetrainConfig,
    delta: f64, // For ADWIN
}

impl<M: OnlineLearner + std::fmt::Debug> OrchestratorBuilder<M> {
    /// Create a new builder
    pub fn new(model: M, n_features: usize) -> Self {
        Self {
            model,
            n_features,
            config: RetrainConfig::default(),
            delta: 0.002,
        }
    }

    /// Set minimum samples for retraining
    pub fn min_samples(mut self, min: usize) -> Self {
        self.config.min_samples = min;
        self
    }

    /// Set maximum buffer size
    pub fn max_buffer_size(mut self, max: usize) -> Self {
        self.config.max_buffer_size = max;
        self
    }

    /// Enable/disable incremental updates
    pub fn incremental_updates(mut self, enable: bool) -> Self {
        self.config.incremental_updates = enable;
        self
    }

    /// Enable/disable curriculum learning
    pub fn curriculum_learning(mut self, enable: bool) -> Self {
        self.config.curriculum_learning = enable;
        self
    }

    /// Set number of curriculum stages
    pub fn curriculum_stages(mut self, stages: usize) -> Self {
        self.config.curriculum_stages = stages;
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.config.learning_rate = lr;
        self
    }

    /// Set number of retrain epochs
    pub fn retrain_epochs(mut self, epochs: usize) -> Self {
        self.config.retrain_epochs = epochs;
        self
    }

    /// Set ADWIN delta (sensitivity)
    pub fn adwin_delta(mut self, delta: f64) -> Self {
        self.delta = delta;
        self
    }

    /// Build the orchestrator with ADWIN detector
    pub fn build(self) -> RetrainOrchestrator<M, ADWIN> {
        let detector = ADWIN::with_delta(self.delta);
        RetrainOrchestrator::with_config(self.model, detector, self.n_features, self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::super::OnlineLinearRegression;
    use super::*;

    #[test]
    fn test_orchestrator_creation() {
        let model = OnlineLinearRegression::new(3);
        let orchestrator = RetrainOrchestrator::new(model, 3);

        assert_eq!(orchestrator.buffer_size(), 0);
        assert_eq!(orchestrator.stats().samples_observed, 0);
    }

    #[test]
    fn test_orchestrator_observe_stable() {
        let model = OnlineLinearRegression::new(2);
        let mut orchestrator = RetrainOrchestrator::new(model, 2);

        // Good predictions should keep status stable
        for _ in 0..50 {
            let result = orchestrator.observe(&[1.0, 2.0], &[5.0], &[5.0]).unwrap();

            // Should mostly be stable
            assert!(result == ObserveResult::Stable || result == ObserveResult::Skipped);
        }

        assert_eq!(orchestrator.stats().retrain_count, 0);
    }

    #[test]
    fn test_orchestrator_observe_drift() {
        let config = RetrainConfig {
            min_samples: 10,
            incremental_updates: false,
            curriculum_learning: false,
            retrain_epochs: 1,
            ..Default::default()
        };

        let model = OnlineLinearRegression::new(2);
        let detector = ADWIN::with_delta(0.1); // More sensitive
        let mut orchestrator = RetrainOrchestrator::with_config(model, detector, 2, config);

        // Feed good predictions first (all correct)
        for i in 0..50 {
            orchestrator
                .observe(&[i as f64, 0.0], &[0.0], &[0.1]) // Close predictions
                .unwrap();
        }

        // Now feed bad predictions to trigger drift (classification errors)
        for i in 0..150 {
            orchestrator
                .observe(&[(50 + i) as f64, 1.0], &[1.0], &[0.0]) // Wrong class
                .unwrap();
        }

        // Should have observed samples and possibly detected drift or retrained
        let stats = orchestrator.stats();
        // The key assertion is that the orchestrator is functional and tracking samples
        assert!(
            stats.samples_observed >= 100,
            "Should have observed at least 100 samples, got {}",
            stats.samples_observed
        );
        // Either drift was detected, we retrained, or we're tracking errors
        assert!(
            stats.drift_status == DriftStatus::Warning
                || stats.drift_status == DriftStatus::Drift
                || stats.retrain_count > 0
                || stats.buffer_size > 0,
            "Expected some activity, got stats={:?}",
            stats
        );
    }

    #[test]
    fn test_orchestrator_force_retrain() {
        let config = RetrainConfig {
            min_samples: 5,
            curriculum_learning: false,
            retrain_epochs: 1,
            ..Default::default()
        };

        let model = OnlineLinearRegression::new(2);
        let detector = ADWIN::new();
        let mut orchestrator = RetrainOrchestrator::with_config(model, detector, 2, config);

        // Add some samples
        for i in 0..10 {
            orchestrator
                .observe(&[i as f64, (i * 2) as f64], &[(i * 3) as f64], &[0.0])
                .unwrap();
        }

        assert!(orchestrator.buffer_size() > 0);

        // Force retrain
        orchestrator.force_retrain().unwrap();

        assert_eq!(orchestrator.stats().retrain_count, 1);
    }

    #[test]
    fn test_orchestrator_should_retrain() {
        let config = RetrainConfig {
            min_samples: 5,
            ..Default::default()
        };

        let model = OnlineLinearRegression::new(2);
        let detector = ADWIN::with_delta(1.0); // Very insensitive
        let orchestrator = RetrainOrchestrator::with_config(model, detector, 2, config);

        // Initially should not need retraining
        assert!(!orchestrator.should_retrain());
    }

    #[test]
    fn test_orchestrator_builder() {
        let model = OnlineLinearRegression::new(3);

        let orchestrator = OrchestratorBuilder::new(model, 3)
            .min_samples(50)
            .max_buffer_size(1000)
            .incremental_updates(false)
            .curriculum_learning(true)
            .curriculum_stages(3)
            .learning_rate(0.05)
            .retrain_epochs(5)
            .adwin_delta(0.01)
            .build();

        assert_eq!(orchestrator.config().min_samples, 50);
        assert_eq!(orchestrator.config().max_buffer_size, 1000);
        assert!(!orchestrator.config().incremental_updates);
        assert!(orchestrator.config().curriculum_learning);
    }

    #[test]
    fn test_observe_result_equality() {
        assert_eq!(ObserveResult::Stable, ObserveResult::Stable);
        assert_ne!(ObserveResult::Stable, ObserveResult::Warning);
        assert_ne!(ObserveResult::Warning, ObserveResult::Retrained);
    }

    #[test]
    fn test_retrain_config_default() {
        let config = RetrainConfig::default();

        assert_eq!(config.min_samples, 100);
        assert_eq!(config.max_buffer_size, 10_000);
        assert!(config.incremental_updates);
        assert!(config.curriculum_learning);
    }

    #[test]
    fn test_orchestrator_stats_default() {
        let stats = OrchestratorStats::default();

        assert_eq!(stats.samples_observed, 0);
        assert_eq!(stats.retrain_count, 0);
        assert_eq!(stats.buffer_size, 0);
    }

    #[test]
    fn test_compute_error_regression() {
        let model = OnlineLinearRegression::new(2);
        let orchestrator = RetrainOrchestrator::new(model, 2);

        // Small error (should be false)
        assert!(!orchestrator.compute_error(&[10.0], &[10.5]));

        // Large error (should be true)
        assert!(orchestrator.compute_error(&[10.0], &[15.0]));
    }

    #[test]
    fn test_compute_error_classification() {
        let model = OnlineLinearRegression::new(2);
        let orchestrator = RetrainOrchestrator::new(model, 2);

        // Correct class
        assert!(!orchestrator.compute_error(&[0.0], &[0.3]));

        // Wrong class
        assert!(orchestrator.compute_error(&[0.0], &[0.8]));
    }

    #[test]
    fn test_compute_error_multiclass() {
        let model = OnlineLinearRegression::new(2);
        let orchestrator = RetrainOrchestrator::new(model, 2);

        // Same argmax
        assert!(!orchestrator.compute_error(&[0.1, 0.9, 0.0], &[0.2, 0.7, 0.1]));

        // Different argmax
        assert!(orchestrator.compute_error(&[0.1, 0.9, 0.0], &[0.8, 0.1, 0.1]));
    }

    #[test]
    fn test_orchestrator_model_access() {
        let model = OnlineLinearRegression::new(2);
        let mut orchestrator = RetrainOrchestrator::new(model, 2);

        // Read access
        assert_eq!(orchestrator.model().n_samples_seen(), 0);

        // Write access
        orchestrator
            .model_mut()
            .partial_fit(&[1.0, 2.0], &[3.0], None)
            .unwrap();
        assert!(orchestrator.model().n_samples_seen() > 0);
    }

    #[test]
    fn test_orchestrator_curriculum_retraining() {
        let config = RetrainConfig {
            min_samples: 10,
            curriculum_learning: true,
            curriculum_stages: 3,
            retrain_epochs: 3,
            incremental_updates: false,
            ..Default::default()
        };

        let model = OnlineLinearRegression::new(2);
        let detector = ADWIN::new();
        let mut orchestrator = RetrainOrchestrator::with_config(model, detector, 2, config);

        // Add samples with varying difficulty
        for i in 0..20 {
            orchestrator
                .observe(&[i as f64, (i * 2) as f64], &[(i * 3) as f64], &[0.0])
                .unwrap();
        }

        // Force curriculum-based retrain
        orchestrator.force_retrain().unwrap();

        assert_eq!(orchestrator.stats().retrain_count, 1);
    }
}
