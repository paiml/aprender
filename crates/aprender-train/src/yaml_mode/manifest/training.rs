//! Training Configuration
//!
//! Contains training loop configuration types for training manifests.

use serde::{Deserialize, Serialize};

/// Training loop configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Number of epochs
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epochs: Option<usize>,

    /// Maximum training steps (mutually exclusive with epochs)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_steps: Option<usize>,

    /// Maximum wall-clock duration (mutually exclusive with epochs/max_steps)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration: Option<String>,

    /// Gradient settings
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gradient: Option<GradientConfig>,

    /// Mixed precision training
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mixed_precision: Option<MixedPrecisionConfig>,

    /// Distributed training
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distributed: Option<DistributedConfig>,

    /// Checkpointing
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint: Option<CheckpointConfig>,

    /// Early stopping (Jidoka)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub early_stopping: Option<EarlyStoppingConfig>,

    /// Validation configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub validation: Option<ValidationConfig>,

    /// Deterministic mode
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deterministic: Option<bool>,

    /// Benchmark mode (cuDNN autotuner)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub benchmark: Option<bool>,

    /// R-023: Curriculum learning — multi-stage data mixing
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub curriculum: Option<Vec<crate::config::CurriculumStage>>,
}

/// Gradient settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientConfig {
    /// Gradient accumulation steps
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accumulation_steps: Option<usize>,

    /// Gradient clipping (L2 norm)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clip_norm: Option<f64>,

    /// Gradient clipping (absolute value)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clip_value: Option<f64>,
}

/// Mixed precision training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedPrecisionConfig {
    /// Enable mixed precision
    pub enabled: bool,

    /// Data type (float16, bfloat16)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,

    /// Loss scale (dynamic, static, or float)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loss_scale: Option<String>,
}

/// Distributed training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// Strategy (ddp, fsdp, deepspeed)
    pub strategy: String,

    /// World size
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub world_size: Option<usize>,

    /// Gradient as bucket view
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gradient_as_bucket_view: Option<bool>,

    /// Find unused parameters
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub find_unused_parameters: Option<bool>,
}

/// Checkpoint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    /// Save every N steps
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub save_every: Option<usize>,

    /// Keep last N checkpoints
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keep_last: Option<usize>,

    /// Save best model by metric
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub save_best: Option<bool>,

    /// Metric for best model selection
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metric: Option<String>,

    /// Metric mode (min, max)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
}

/// Early stopping configuration (Jidoka - automatic halt on quality degradation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    /// Enable early stopping
    pub enabled: bool,

    /// Metric to monitor
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metric: Option<String>,

    /// Patience (epochs without improvement)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub patience: Option<usize>,

    /// Minimum delta for improvement
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_delta: Option<f64>,

    /// Metric mode (min, max)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
}

/// Validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Validate every N steps
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub every: Option<usize>,

    /// Validate each epoch
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub every_epoch: Option<bool>,

    /// Metrics to compute
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<Vec<String>>,

    /// Cross-validation configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cross_validation: Option<CrossValidationConfig>,
}

/// Cross-validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationConfig {
    /// Number of folds
    pub folds: usize,

    /// Stratified sampling
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stratified: Option<bool>,

    /// Shuffle data
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shuffle: Option<bool>,
}
