//! Dataset configuration

use serde::{Deserialize, Serialize};

/// Dataset configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    /// Dataset ID or path
    pub path: String,
    /// Maximum sequence length
    #[serde(default = "default_max_seq_length")]
    pub max_seq_length: usize,
    /// Maximum training examples (None = all)
    pub max_train_examples: Option<usize>,
    /// Maximum validation examples
    pub max_eval_examples: Option<usize>,
}

fn default_max_seq_length() -> usize {
    512
}
