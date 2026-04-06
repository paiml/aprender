//! Neural network pruning integration for Entrenar
//!
//! This module provides training-time pruning capabilities that integrate
//! with the Aprender pruning primitives. It implements:
//!
//! - **Pruning Schedules**: OneShot, Gradual, and Cubic sparsity schedules
//! - **Pruning Callback**: Integration with the training loop
//! - **Calibration Pipeline**: Activation collection for Wanda/SparseGPT
//! - **Prune-Finetune Pipeline**: End-to-end pruning workflow
//!
//! # Toyota Way Principles
//!
//! - **Kaizen** (Continuous Improvement): Gradual pruning schedules
//! - **Jidoka** (Quality at Source): Validates sparsity at each step
//! - **Genchi Genbutsu** (Go and See): Uses real activation data
//!
//! # Example
//!
//! ```ignore
//! use entrenar::prune::{PruningSchedule, PruningCallback, PruningConfig};
//! use aprender::pruning::{MagnitudeImportance, WandaPruner};
//!
//! let schedule = PruningSchedule::Gradual {
//!     start_step: 1000,
//!     end_step: 10000,
//!     initial_sparsity: 0.0,
//!     final_sparsity: 0.5,
//!     frequency: 100,
//! };
//!
//! let config = PruningConfig::default()
//!     .with_schedule(schedule)
//!     .with_target_sparsity(0.5);
//!
//! let callback = PruningCallback::new(config);
//! trainer.add_callback(callback);
//! ```
//!
//! # References
//!
//! - Han, S., et al. (2015). Learning both weights and connections. NeurIPS.
//! - Sun, M., et al. (2023). A simple and effective pruning approach. arXiv:2306.11695.
//! - Zhu, M., & Gupta, S. (2017). To prune, or not to prune. arXiv:1710.01878.

mod calibrate;
mod callback;
mod config;
mod data_loader;
#[cfg(test)]
mod falsification_checklist;
#[cfg(test)]
mod golden_traces;
mod pipeline;
mod schedule;
#[cfg(test)]
mod snapshot_tests;
mod trainer_integration;

pub use calibrate::{CalibrationCollector, CalibrationConfig};
pub use callback::PruningCallback;
pub use config::{PruneMethod, PruningConfig, SparsityPatternConfig};
pub use data_loader::{CalibrationDataConfig, CalibrationDataLoader};
pub use pipeline::{PruneFinetunePipeline, PruningMetrics, PruningStage};
pub use schedule::PruningSchedule;
pub use trainer_integration::{PruneTrainer, PruneTrainerConfig};
