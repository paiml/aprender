//! Pruning callback for training loop integration
//!
//! This module provides the `PruningCallback` that integrates with Entrenar's
//! training callback system to apply pruning during training.
//!
//! # Toyota Way: Kaizen (Continuous Improvement)
//! Gradual pruning allows the model to adapt incrementally to sparsity.

mod pruning_callback;

#[cfg(test)]
mod tests;

// Re-export all public types
pub use pruning_callback::PruningCallback;
