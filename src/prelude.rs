//! Convenience re-exports for common usage.
//!
//! # Usage
//!
//! ```
//! use aprender::prelude::*;
//! ```

pub use crate::cluster::KMeans;
pub use crate::data::DataFrame;
pub use crate::linear_model::{LinearRegression, Ridge};
pub use crate::metrics::{inertia, mae, mse, r_squared, rmse, silhouette_score};
pub use crate::preprocessing::{MinMaxScaler, StandardScaler};
pub use crate::primitives::{Matrix, Vector};
pub use crate::traits::{Estimator, Transformer, UnsupervisedEstimator};
pub use crate::tree::DecisionTreeClassifier;
