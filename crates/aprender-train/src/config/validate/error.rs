//! Validation error types
//!
//! Defines all validation error variants for training specifications.

/// Validation error type
#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    #[error("Model path does not exist: {0}")]
    ModelPathNotFound(String),

    #[error("Training data path does not exist: {0}")]
    TrainDataNotFound(String),

    #[error("Validation data path does not exist: {0}")]
    ValDataNotFound(String),

    #[error("Invalid learning rate: {0} (must be > 0.0 and <= 1.0)")]
    InvalidLearningRate(f32),

    #[error("Invalid batch size: {0} (must be > 0)")]
    InvalidBatchSize(usize),

    #[error("Invalid epochs: {0} (must be > 0)")]
    InvalidEpochs(usize),

    #[error("Invalid LoRA rank: {0} (must be > 0 and <= 1024)")]
    InvalidLoRARank(usize),

    #[error("Invalid LoRA alpha: {0} (must be > 0.0)")]
    InvalidLoRAAlpha(f32),

    #[error("Invalid LoRA dropout: {0} (must be in [0.0, 1.0))")]
    InvalidLoRADropout(f32),

    #[error("Invalid quantization bits: {0} (must be 4 or 8)")]
    InvalidQuantBits(u8),

    #[error("Invalid optimizer: {0} (must be one of: adam, adamw, sgd)")]
    InvalidOptimizer(String),

    #[error("Invalid merge method: {0} (must be one of: ties, dare, slerp)")]
    InvalidMergeMethod(String),

    #[error("Invalid gradient clip value: {0} (must be > 0.0)")]
    InvalidGradClip(f32),

    #[error("Invalid sequence length: {0} (must be > 0)")]
    InvalidSeqLen(usize),

    #[error("Invalid save interval: {0} (must be > 0)")]
    InvalidSaveInterval(usize),

    #[error("LoRA target modules cannot be empty")]
    EmptyLoRATargets,

    #[error("Invalid LR scheduler: {0} (must be one of: cosine, linear, constant)")]
    InvalidLRScheduler(String),

    #[error("Invalid publish repo: {0} (must be org/name format)")]
    InvalidPublishRepo(String),

    #[error("Invalid publish format: {0} (must be safetensors or gguf)")]
    InvalidPublishFormat(String),
}
