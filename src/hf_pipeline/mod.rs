//! HuggingFace Distillation & Learning Pipeline
//!
//! This module provides integration with HuggingFace Hub for:
//! - Model downloading with authentication
//! - Knowledge distillation from teacher to student models
//! - Fine-tuning with LoRA/QLoRA
//! - Dataset streaming
//!
//! # References
//!
//! - Hinton et al. (2015) "Distilling the Knowledge in a Neural Network"
//! - Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models"
//! - Dettmers et al. (2023) "QLoRA: Efficient Finetuning of Quantized LLMs"
//!
//! # Example
//!
//! ```ignore
//! use entrenar::hf_pipeline::{HfModelFetcher, FetchOptions};
//!
//! let fetcher = HfModelFetcher::new()?;
//! let artifact = fetcher.download_model("microsoft/codebert-base", FetchOptions::default())?;
//! ```

mod config;
mod dataset;
mod distillation;
mod error;
mod export;
mod fetcher;
mod fine_tune;
#[cfg(feature = "hub-publish")]
pub mod leaderboard;
mod loader;
#[cfg(feature = "hub-publish")]
pub mod publish;
mod trainer;

#[cfg(test)]
mod tests;

pub use config::DistillationYamlConfig;
pub use dataset::{
    Batch, CacheStats, Dataset, DatasetOptions, DistillationCollator, Example, HfDatasetFetcher,
    Split, TeacherCache,
};
pub use distillation::{AttentionTransfer, DistillationLoss, ProgressiveDistillation};
pub use error::{FetchError, Result};
pub use export::{
    quantize_and_export, verify_gguf, ExportFormat, ExportResult, Exporter, GgufQuantization,
    GgufSummary, GgufTensorInfo, ModelMetadata, ModelWeights, QuantExportResult,
};
#[cfg(feature = "hub-publish")]
pub use export::{quantize_export_publish, QuantPublishError, QuantPublishResult};
pub use fetcher::{Architecture, FetchOptions, HfModelFetcher, ModelArtifact, WeightFormat};
pub use fine_tune::{FineTuneConfig, FineTuneMethod, MemoryRequirement, MixedPrecision};
pub use loader::{MemoryEstimate, SafeTensorsTeacher, TeacherModel};
pub use trainer::{DistillationTrainer, TrainerConfig, TrainingState};
