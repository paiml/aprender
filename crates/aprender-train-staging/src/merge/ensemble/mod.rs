//! ENT-032: Multi-model ensemble merging (>2 models)
//!
//! Provides unified interface for merging multiple models with various strategies:
//! - Weighted averaging
//! - Iterative SLERP (pairwise application)
//! - Hierarchical merging (tree-based)
//! - Layer-wise strategy selection

mod config;
mod hierarchical;
mod merge;
mod slerp;
mod strategy;
mod weighted;

#[cfg(test)]
mod tests;

// Re-export from parent module for internal use
use super::{
    dare_merge, slerp_merge, ties_merge, DareConfig, MergeError, Model, SlerpConfig, TiesConfig,
};

// Public API exports
pub use config::EnsembleConfig;
pub use merge::ensemble_merge;
pub use strategy::EnsembleStrategy;
