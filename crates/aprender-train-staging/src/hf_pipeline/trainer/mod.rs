//! Distillation Trainer Orchestrator
//!
//! High-level API for knowledge distillation training.
//!
//! # Example
//!
//! ```ignore
//! use entrenar::hf_pipeline::{DistillationTrainer, TrainerConfig};
//!
//! let trainer = DistillationTrainer::new(TrainerConfig {
//!     teacher_model: "microsoft/codebert-base".into(),
//!     student_model: "distilbert-base-uncased".into(),
//!     ..Default::default()
//! });
//!
//! trainer.train(&dataset)?;
//! ```

mod config;
mod distillation_trainer;
mod state;

#[cfg(test)]
mod tests;

// Re-export all public types
pub use config::TrainerConfig;
pub use distillation_trainer::DistillationTrainer;
pub use state::TrainingState;
