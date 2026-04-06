//! Training Data Collection
//!
//! Implements `TunerDataCollector` for collecting and persisting training samples.
//!
//! # Module Layout
//!
//! - [`types`] -- `TrainingSample`, `UserFeedback`, `ConceptDriftStatus`, `TrainingStats`
//! - [`persistence`] -- APR binary save/load, JSON import/export, cache paths
//! - [`drift`] -- Online learning, concept drift detection, feedback, auto-retrain

mod drift;
mod persistence;
mod types;

pub use types::{ConceptDriftStatus, TrainingSample, TrainingStats, UserFeedback};

use crate::brick::BrickProfiler;

use super::brick_tuner::BrickTuner;
use super::error::TunerError;
use super::features::{FeatureExtractor, RunConfig, TunerFeatures};
use super::helpers::chrono_lite_now;
use super::types::{BottleneckClass, KernelType};

// ============================================================================
// TunerDataCollector
// ============================================================================

/// Training data collector with online learning support (T-TUNER-005, GitHub #82)
#[derive(Debug, Default)]
pub struct TunerDataCollector {
    /// Collected samples
    pub(crate) samples: Vec<TrainingSample>,
    /// Feature extractor
    pub(crate) extractor: FeatureExtractor,
    /// Auto-retrain threshold
    pub(crate) retrain_threshold: usize,
    /// Number of samples at last training
    pub(crate) samples_at_last_train: usize,
    /// User feedback history (sample index -> feedback)
    pub(crate) feedback: Vec<UserFeedback>,
    /// Online learning enabled (privacy: opt-in only)
    pub(crate) online_learning_enabled: bool,
    /// Moving average of prediction errors (for concept drift)
    pub(crate) error_window: Vec<f32>,
    /// Error window size for drift detection
    error_window_size: usize,
}

impl TunerDataCollector {
    /// Default error window size for concept drift detection
    pub(super) const DEFAULT_ERROR_WINDOW_SIZE: usize = 50;

    /// Error threshold for drift detection (mean absolute error)
    pub(super) const DRIFT_ERROR_THRESHOLD: f32 = 0.15;

    /// Staleness threshold (samples since training) for recommending retrain
    pub(super) const STALENESS_THRESHOLD: usize = 100;

    /// Minimum samples required before training triggers
    pub const MIN_SAMPLES_FOR_TRAINING: usize = 1000;

    /// Create a new collector
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            extractor: FeatureExtractor::new(),
            retrain_threshold: 100,
            samples_at_last_train: 0,
            feedback: Vec::new(),
            online_learning_enabled: false, // Privacy: opt-in
            error_window: Vec::new(),
            error_window_size: Self::DEFAULT_ERROR_WINDOW_SIZE,
        }
    }

    /// Create a collector with online learning enabled
    pub fn with_online_learning() -> Self {
        let mut collector = Self::new();
        collector.online_learning_enabled = true;
        collector
    }

    /// Enable online learning (privacy: explicit opt-in)
    pub fn enable_online_learning(&mut self) {
        self.online_learning_enabled = true;
    }

    /// Disable online learning
    pub fn disable_online_learning(&mut self) {
        self.online_learning_enabled = false;
    }

    /// Check if online learning is enabled
    pub fn is_online_learning_enabled(&self) -> bool {
        self.online_learning_enabled
    }

    /// Record a profiling run as training data
    pub fn record(
        &mut self,
        profiler: &BrickProfiler,
        config: &RunConfig,
        kernel: KernelType,
    ) -> Option<()> {
        let throughput_tps = profiler.tokens_per_sec()?;
        let features = self.extractor.extract(profiler, config);
        let bottleneck = features.bottleneck_class.unwrap_or(BottleneckClass::Unknown);

        let sample = TrainingSample {
            features,
            throughput_tps,
            best_kernel: kernel,
            bottleneck,
            timestamp: chrono_lite_now(),
            hardware_id: "unknown".to_string(),
        };

        self.samples.push(sample);
        Some(())
    }

    /// Get all samples
    pub fn samples(&self) -> &[TrainingSample] {
        &self.samples
    }

    /// Get sample count
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Export to JSON
    pub fn to_json(&self) -> Result<String, TunerError> {
        serde_json::to_string_pretty(&self.samples)
            .map_err(|e| TunerError::Serialization(e.to_string()))
    }

    /// Prepare training data for model
    pub fn prepare_training_data(&self) -> Vec<(TunerFeatures, f32)> {
        self.samples.iter().map(|s| (s.features.clone(), s.throughput_tps)).collect()
    }

    /// Check if we have enough samples to train
    pub fn ready_to_train(&self) -> bool {
        self.samples.len() >= Self::MIN_SAMPLES_FOR_TRAINING
    }

    /// Train a BrickTuner from collected data if we have enough samples
    pub fn train_if_ready(&self) -> Option<BrickTuner> {
        if !self.ready_to_train() {
            return None;
        }

        let training_data = self.prepare_training_data();
        let mut tuner = BrickTuner::new();

        match tuner.train(&training_data) {
            Ok(()) => Some(tuner),
            Err(_) => None,
        }
    }

    /// Get training progress as (current, required)
    pub fn training_progress(&self) -> (usize, usize) {
        (self.samples.len(), Self::MIN_SAMPLES_FOR_TRAINING)
    }

    /// Merge samples from another collector
    pub fn merge(&mut self, other: &TunerDataCollector) {
        self.samples.extend(other.samples.iter().cloned());
    }
}
