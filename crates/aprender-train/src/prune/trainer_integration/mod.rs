//! Trainer integration for pruning
//!
//! Provides utilities for integrating pruning with the training pipeline,
//! including fine-tuning after pruning.
//!
//! # Toyota Way: Kaizen (Continuous Improvement)
//! Fine-tuning allows the model to recover from pruning-induced accuracy loss.

mod config;
mod trainer;

#[cfg(test)]
mod tests;

// Re-export all public types
pub use config::PruneTrainerConfig;
pub use trainer::PruneTrainer;
