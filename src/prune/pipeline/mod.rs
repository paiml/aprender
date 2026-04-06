//! Prune-Finetune-Export pipeline
//!
//! Provides end-to-end workflows for pruning models:
//! 1. Calibrate: Collect activation statistics
//! 2. Prune: Apply pruning with selected method
//! 3. Finetune: Recover accuracy with brief training
//! 4. Export: Save pruned model
//!
//! # Toyota Way Principles
//! - **Heijunka** (Level Loading): Balanced sparsity across layers
//! - **Hansei** (Reflection): Post-pruning evaluation and metrics

mod metrics;
mod orchestrator;
mod prune_quantize;
mod sparse_export;
mod stage;
#[cfg(test)]
mod tests;

pub use metrics::PruningMetrics;
pub use orchestrator::PruneFinetunePipeline;
pub use stage::PruningStage;
