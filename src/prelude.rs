//! Convenience re-exports for common usage.
//!
//! # Usage
//!
//! ```
//! use aprender::prelude::*;
//! ```

pub use crate::classification::LogisticRegression;
pub use crate::cluster::{
    Affinity, AgglomerativeClustering, CovarianceType, GaussianMixture, IsolationForest, KMeans,
    Linkage, LocalOutlierFactor, SpectralClustering, DBSCAN,
};
pub use crate::data::DataFrame;
pub use crate::linear_model::{ElasticNet, Lasso, LinearRegression, Ridge};
pub use crate::loss::{huber_loss, mae_loss, mse_loss, HuberLoss, Loss, MAELoss, MSELoss};
pub use crate::metrics::{inertia, mae, mse, r_squared, rmse, silhouette_score};
pub use crate::mining::Apriori;
pub use crate::optim::{Adam, Optimizer, SGD};
pub use crate::preprocessing::{MinMaxScaler, StandardScaler, TSNE};
pub use crate::primitives::{Matrix, Vector};
pub use crate::traits::{Estimator, Transformer, UnsupervisedEstimator};
pub use crate::tree::{DecisionTreeClassifier, DecisionTreeRegressor};

// Re-export model_selection types
pub use crate::model_selection::{grid_search_alpha, GridSearchResult, StratifiedKFold};
