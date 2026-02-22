//! Explainable AI Integration for Inference Monitoring
//!
//! This module provides native `Explainable` trait implementations for APR format models,
//! with decision path types for full prediction traceability.
//!
//! # Toyota Way: 現地現物 (Genchi Genbutsu)
//!
//! Every prediction can be traced to its decision path. All decisions are explainable.
//!
//! # Features
//!
//! - `LinearExplainable`: Wraps linear models with feature contribution tracking
//! - `TreeExplainable`: Wraps decision trees with split path tracking
//! - `EnsembleExplainable`: Wraps ensembles with per-model aggregation
//!
//! # Example
//!
//! ```ignore
//! use aprender::linear_model::LinearRegression;
//! use aprender::explainable::LinearExplainable;
//! use aprender::explainable::path::{Explainable, DecisionPath};
//!
//! let model = LinearRegression::new();
//! model.fit(&x, &y)?;
//!
//! // Wrap with explainability
//! let explainable = LinearExplainable::new(model);
//! let (outputs, paths) = explainable.predict_explained(&features, 1);
//! println!("{}", paths[0].explain());
//! ```

pub mod path;

mod ensemble;
mod linear;
mod logistic;
mod tree;

pub use ensemble::{EnsembleExplainable, IntoEnsembleExplainable};
pub use linear::{IntoExplainable, LinearExplainable};
pub use logistic::{IntoLogisticExplainable, LogisticExplainable};
pub use tree::{IntoTreeExplainable, TreeExplainable};

#[cfg(test)]
mod tests;
