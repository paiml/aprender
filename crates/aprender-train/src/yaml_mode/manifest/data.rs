//! Data Configuration
//!
//! Contains all data-related configuration types for training manifests.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::shorthand::deserialize_human_usize_opt;

/// Dataset configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    /// Data source URI (pacha://, hf://, s3://, or local path)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,

    /// Explicit format (auto-detected if omitted)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,

    /// Data split configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split: Option<DataSplit>,

    /// Explicit training data path
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub train: Option<String>,

    /// Explicit validation data path
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub val: Option<String>,

    /// Explicit test data path
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub test: Option<String>,

    /// Preprocessing pipeline
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preprocessing: Option<Vec<PreprocessingStep>>,

    /// Data augmentation pipeline
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub augmentation: Option<Vec<HashMap<String, serde_json::Value>>>,

    /// DataLoader settings
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loader: Option<DataLoader>,

    // === LLM data fields (mirrors TrainSpec DataConfig) ===
    /// Path to tokenizer.json (for transformer/LLM training)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer: Option<String>,

    /// Sequence length (for transformers). Accepts shorthand: `"2K"` = 2048.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_human_usize_opt"
    )]
    pub seq_len: Option<usize>,

    /// Input text column name (for transformer mode)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_column: Option<String>,

    /// Output/target text column name (for transformer mode)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_column: Option<String>,

    /// Maximum tokenization length. Accepts shorthand: `"512"`, `"1K"`.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_human_usize_opt"
    )]
    pub max_length: Option<usize>,
}

/// Data split ratios
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSplit {
    /// Training set ratio (0.0-1.0)
    pub train: f64,

    /// Validation set ratio (optional)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub val: Option<f64>,

    /// Test set ratio (optional)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub test: Option<f64>,

    /// Column name for stratified sampling
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stratify: Option<String>,

    /// Split seed (inherits global if omitted)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

/// Preprocessing step (normalize, encode, drop, fillna, tokenize)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PreprocessingStep {
    /// Normalization step
    Normalize { normalize: NormalizeConfig },
    /// Encoding step
    Encode { encode: EncodeConfig },
    /// Drop columns step
    Drop { drop: DropConfig },
    /// Fill NA step
    FillNa { fillna: FillNaConfig },
    /// Tokenization step
    Tokenize { tokenize: TokenizeConfig },
}

/// Normalization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizeConfig {
    pub columns: Vec<String>,
    pub method: String,
}

/// Encoding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodeConfig {
    pub columns: Vec<String>,
    pub method: String,
}

/// Drop columns configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropConfig {
    pub columns: Vec<String>,
}

/// Fill NA configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillNaConfig {
    pub strategy: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<serde_json::Value>,
}

/// Tokenization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizeConfig {
    pub tokenizer: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_length: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub padding: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncation: Option<bool>,
}

/// DataLoader settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataLoader {
    /// Batch size
    pub batch_size: usize,

    /// Shuffle data each epoch
    pub shuffle: bool,

    /// Number of worker processes
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_workers: Option<usize>,

    /// Pin memory for GPU transfer
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pin_memory: Option<bool>,

    /// Drop incomplete last batch
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drop_last: Option<bool>,

    /// Prefetch factor
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefetch_factor: Option<usize>,
}
