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

include!("orchestrator_builder.rs");
include!("orchestrator_part_03.rs");
