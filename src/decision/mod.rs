//! Decision pattern storage and CITL trainer module
//!
//! This module provides:
//! - `PatternStore`: Stores and retrieves decision patterns using cosine similarity search
//! - `CitlTrainer`: Correlation-Informed Transfer Learning for error-fix prediction
//!
//! # GH-28: DecisionPatternStore
//!
//! Stores `DecisionPattern` instances with feature-weight vectors and supports
//! nearest-neighbor retrieval via cosine similarity.
//!
//! # GH-29: DecisionCITL trainer
//!
//! Trains a simple linear correlation model from `ErrorFixPair` samples,
//! then predicts fix feature vectors from error feature vectors.

mod citl;
mod pattern_store;

#[cfg(test)]
mod tests;

pub use citl::{CitlTrainer, ErrorFixPair};
pub use pattern_store::{DecisionPattern, PatternStore};
