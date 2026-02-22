//! Decision Path Types and Explainability Trait
//!
//! Pure type definitions for model-specific decision paths.
//! These types were previously in `entrenar::monitor::inference::path` but belong
//! in aprender since they describe aprender's own models.
//!
//! # Types
//!
//! - [`DecisionPath`] — Common trait for all decision paths
//! - [`Explainable`] — Trait for models that can explain their predictions
//! - [`TreePath`], [`ForestPath`], [`LinearPath`], [`KNNPath`], [`NeuralPath`] — Model-specific paths

mod forest;
mod knn;
mod linear;
mod neural;
pub(crate) mod traits;
mod tree;

// Re-export all public types
pub use forest::ForestPath;
pub use knn::KNNPath;
pub use linear::LinearPath;
pub use neural::NeuralPath;
pub use traits::{DecisionPath, PathError};
pub use tree::{LeafInfo, TreePath, TreeSplit};

/// Trait for models that can explain their predictions
pub trait Explainable {
    /// Model-specific decision path type
    type Path: DecisionPath;

    /// Predict with full decision trace for each sample
    fn predict_explained(&self, x: &[f32], n_samples: usize) -> (Vec<f32>, Vec<Self::Path>);

    /// Single-sample explanation (for streaming)
    fn explain_one(&self, sample: &[f32]) -> Self::Path;
}
