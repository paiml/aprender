//! Configuration for calibration data loading.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Configuration for calibration data loading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationDataConfig {
    /// Number of calibration samples to load.
    num_samples: usize,
    /// Batch size for calibration.
    batch_size: usize,
    /// Sequence length (for text models).
    sequence_length: usize,
    /// Dataset name or path.
    dataset: String,
    /// Cache directory for downloaded data.
    cache_dir: Option<PathBuf>,
    /// Random seed for sampling.
    seed: u64,
}

impl Default for CalibrationDataConfig {
    fn default() -> Self {
        Self {
            num_samples: 128,
            batch_size: 1,
            sequence_length: 2048,
            dataset: "c4".to_string(),
            cache_dir: None,
            seed: 42,
        }
    }
}

impl CalibrationDataConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of calibration samples.
    pub fn with_num_samples(mut self, n: usize) -> Self {
        self.num_samples = n;
        self
    }

    /// Set the batch size.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1);
        self
    }

    /// Set the sequence length.
    pub fn with_sequence_length(mut self, len: usize) -> Self {
        self.sequence_length = len;
        self
    }

    /// Set the dataset name or path.
    pub fn with_dataset(mut self, dataset: impl Into<String>) -> Self {
        self.dataset = dataset.into();
        self
    }

    /// Set the cache directory.
    pub fn with_cache_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(dir.into());
        self
    }

    /// Set the random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Get the number of samples.
    pub fn num_samples(&self) -> usize {
        self.num_samples
    }

    /// Get the batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Get the sequence length.
    pub fn sequence_length(&self) -> usize {
        self.sequence_length
    }

    /// Get the dataset name.
    pub fn dataset(&self) -> &str {
        &self.dataset
    }

    /// Get the cache directory.
    pub fn cache_dir(&self) -> Option<&PathBuf> {
        self.cache_dir.as_ref()
    }

    /// Get the random seed.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Get the number of batches.
    pub fn num_batches(&self) -> usize {
        self.num_samples.div_ceil(self.batch_size)
    }
}
