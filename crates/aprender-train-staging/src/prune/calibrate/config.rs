//! Calibration configuration for pruning.

use serde::{Deserialize, Serialize};

/// Configuration for calibration data collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConfig {
    /// Number of calibration samples to collect.
    num_samples: usize,
    /// Sequence length for text models.
    sequence_length: usize,
    /// Dataset identifier for calibration.
    dataset: String,
    /// Batch size for calibration forward passes.
    batch_size: usize,
    /// Whether to normalize activation statistics.
    normalize: bool,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            num_samples: 128,
            sequence_length: 2048,
            dataset: "c4".to_string(),
            batch_size: 1,
            normalize: true,
        }
    }
}

impl CalibrationConfig {
    /// Create a new calibration configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of calibration samples.
    pub fn with_num_samples(mut self, n: usize) -> Self {
        self.num_samples = n;
        self
    }

    /// Set the sequence length.
    pub fn with_sequence_length(mut self, len: usize) -> Self {
        self.sequence_length = len;
        self
    }

    /// Set the dataset name.
    pub fn with_dataset(mut self, dataset: impl Into<String>) -> Self {
        self.dataset = dataset.into();
        self
    }

    /// Set the batch size.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set whether to normalize statistics.
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Get the number of samples.
    pub fn num_samples(&self) -> usize {
        self.num_samples
    }

    /// Get the sequence length.
    pub fn sequence_length(&self) -> usize {
        self.sequence_length
    }

    /// Get the dataset name.
    pub fn dataset(&self) -> &str {
        &self.dataset
    }

    /// Get the batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Check if normalization is enabled.
    pub fn normalize(&self) -> bool {
        self.normalize
    }
}
