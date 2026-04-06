//! YAML Configuration for Distillation Training
//!
//! Provides declarative configuration for the entire distillation pipeline.
//!
//! # Example Config
//!
//! ```yaml
//! teacher:
//!   model_id: "microsoft/codebert-base"
//!
//! student:
//!   model_id: "distilbert-base-uncased"
//!   lora:
//!     rank: 16
//!     alpha: 32
//!     target_modules: ["q_proj", "v_proj"]
//!
//! distillation:
//!   temperature: 4.0
//!   alpha: 0.7
//!   progressive:
//!     layer_mapping: [[0, 3], [1, 7], [2, 11]]
//!
//! training:
//!   epochs: 3
//!   batch_size: 16
//!   learning_rate: 2.0e-4
//! ```

mod dataset;
mod distillation;
mod output;
mod student;
mod teacher;
mod training;
mod yaml_config;

#[cfg(test)]
mod tests;

// Re-export all public types for API compatibility
#[allow(unused_imports)]
pub use dataset::DatasetConfig;
#[allow(unused_imports)]
pub use distillation::{AttentionTransferConfig, DistillationConfig, ProgressiveConfig};
#[allow(unused_imports)]
pub use output::OutputConfig;
#[allow(unused_imports)]
pub use student::{LoRAYamlConfig, StudentConfig};
#[allow(unused_imports)]
pub use teacher::TeacherConfig;
#[allow(unused_imports)]
pub use training::TrainingConfig;
pub use yaml_config::DistillationYamlConfig;
