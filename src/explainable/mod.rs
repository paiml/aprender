//! Explainable AI Integration for Inference Monitoring
//!
//! This module bridges aprender models with entrenar's inference monitoring system,
//! providing native `Explainable` trait implementations for APR format models.
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
//! use entrenar::monitor::inference::{InferenceMonitor, RingCollector};
//!
//! let model = LinearRegression::new();
//! model.fit(&x, &y)?;
//!
//! // Wrap with explainability
//! let explainable = LinearExplainable::new(model);
//!
//! // Create monitored inference
//! let collector = RingCollector::<_, 64>::new();
//! let mut monitor = InferenceMonitor::new(explainable, collector);
//!
//! // Predictions are now traced
//! let output = monitor.predict(&features, 1);
//! let trace = monitor.collector().recent(1)[0];
//! println!("{}", trace.explain());
//! ```

#[cfg(feature = "inference-monitoring")]
mod ensemble;
#[cfg(feature = "inference-monitoring")]
mod linear;
#[cfg(feature = "inference-monitoring")]
mod logistic;
#[cfg(feature = "inference-monitoring")]
mod tree;

#[cfg(feature = "inference-monitoring")]
pub use ensemble::{EnsembleExplainable, IntoEnsembleExplainable};
#[cfg(feature = "inference-monitoring")]
pub use linear::{IntoExplainable, LinearExplainable};
#[cfg(feature = "inference-monitoring")]
pub use logistic::{IntoLogisticExplainable, LogisticExplainable};
#[cfg(feature = "inference-monitoring")]
pub use tree::{IntoTreeExplainable, TreeExplainable};

#[cfg(feature = "inference-monitoring")]
#[cfg(test)]
mod tests;
