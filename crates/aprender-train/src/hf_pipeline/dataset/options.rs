//! Dataset fetch options

use std::path::PathBuf;

use super::split::Split;

/// Dataset fetch options
#[derive(Debug, Clone)]
pub struct DatasetOptions {
    /// Dataset split to load
    pub split: Split,
    /// Maximum number of examples (None = all)
    pub max_examples: Option<usize>,
    /// Stream data instead of loading all at once
    pub streaming: bool,
    /// Shuffle data
    pub shuffle: bool,
    /// Random seed for shuffling
    pub seed: Option<u64>,
    /// Cache directory
    pub cache_dir: Option<PathBuf>,
}

impl Default for DatasetOptions {
    fn default() -> Self {
        Self {
            split: Split::Train,
            max_examples: None,
            streaming: false,
            shuffle: true,
            seed: Some(42),
            cache_dir: None,
        }
    }
}

impl DatasetOptions {
    /// Create new options for training split
    #[must_use]
    pub fn train() -> Self {
        Self::default()
    }

    /// Create new options for validation split
    #[must_use]
    pub fn validation() -> Self {
        Self { split: Split::Validation, shuffle: false, ..Default::default() }
    }

    /// Create new options for test split
    #[must_use]
    pub fn test() -> Self {
        Self { split: Split::Test, shuffle: false, ..Default::default() }
    }

    /// Set maximum examples
    #[must_use]
    pub fn max_examples(mut self, n: usize) -> Self {
        self.max_examples = Some(n);
        self
    }

    /// Enable streaming
    #[must_use]
    pub fn streaming(mut self, enabled: bool) -> Self {
        self.streaming = enabled;
        self
    }

    /// Set shuffle
    #[must_use]
    pub fn shuffle(mut self, enabled: bool) -> Self {
        self.shuffle = enabled;
        self
    }

    /// Set random seed
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}
