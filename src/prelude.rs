//! Convenience re-exports for common usage.
//!
//! # Usage
//!
//! ```
//! use aprender::prelude::*;
//! ```

pub use crate::primitives::{Matrix, Vector};
pub use crate::traits::{Estimator, Transformer, UnsupervisedEstimator};
pub use crate::data::DataFrame;
pub use crate::linear_model::LinearRegression;
pub use crate::cluster::KMeans;
pub use crate::metrics::{r_squared, mse, mae, rmse, inertia, silhouette_score};
