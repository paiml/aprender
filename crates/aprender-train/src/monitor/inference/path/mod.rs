//! Decision Path Types (ENT-102)
//!
//! Model-specific decision paths for explainability.
//!
//! GH-305: Types now live in `aprender::explainable::path` (source of truth).
//! This module re-exports them for backwards compatibility.

// Re-export all public types from aprender (source of truth)
pub use aprender::explainable::path::{
    DecisionPath, ForestPath, KNNPath, LeafInfo, LinearPath, NeuralPath, PathError, TreePath,
    TreeSplit,
};
