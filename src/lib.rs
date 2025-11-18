//! Aprender: Next-generation machine learning library in pure Rust.
//!
//! Aprender provides production-grade ML algorithms with a focus on
//! ergonomic APIs, comprehensive testing, and backend-agnostic compute.
//!
//! # Quick Start
//!
//! ```
//! use aprender::prelude::*;
//!
//! // Create training data (y = 2*x + 1)
//! let x = Matrix::from_vec(4, 1, vec![
//!     1.0,
//!     2.0,
//!     3.0,
//!     4.0,
//! ]).unwrap();
//! let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);
//!
//! // Train linear regression
//! let mut model = LinearRegression::new();
//! model.fit(&x, &y).unwrap();
//!
//! // Make predictions
//! let predictions = model.predict(&x);
//! let r2 = model.score(&x, &y);
//! assert!(r2 > 0.99);
//! ```
//!
//! # Modules
//!
//! - [`primitives`]: Core Vector and Matrix types
//! - [`data`]: DataFrame for named columns
//! - [`linear_model`]: Linear regression algorithms
//! - [`cluster`]: Clustering algorithms (K-Means)
//! - [`tree`]: Decision tree classifiers
//! - [`metrics`]: Evaluation metrics

pub mod cluster;
pub mod data;
pub mod linear_model;
pub mod metrics;
pub mod prelude;
pub mod primitives;
pub mod traits;
pub mod tree;

pub use primitives::{Matrix, Vector};
pub use traits::{Estimator, Transformer, UnsupervisedEstimator};
