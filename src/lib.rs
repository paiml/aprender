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
//! - [`classification`]: Classification algorithms (Logistic Regression)
//! - [`tree`]: Decision tree classifiers
//! - [`metrics`]: Evaluation metrics
//! - [`mining`]: Pattern mining algorithms (Apriori for association rules)
//! - [`model_selection`]: Cross-validation and train/test splitting
//! - [`preprocessing`]: Data transformers (scalers, encoders)
//! - [`optim`]: Optimization algorithms (SGD, Adam)
//! - [`loss`]: Loss functions for training (MSE, MAE, Huber)
//! - [`serialization`]: Model serialization (SafeTensors format)
//! - [`stats`]: Traditional descriptive statistics (quantiles, histograms)
//! - [`graph`]: Graph construction and analysis (centrality, community detection)
//! - [`bayesian`]: Bayesian inference (conjugate priors, MCMC, variational inference)
//! - [`glm`]: Generalized Linear Models (Poisson, Gamma, Binomial families)
//! - [`decomposition`]: Matrix decomposition (ICA, PCA)
//! - [`text`]: Text processing and NLP (tokenization, stop words, stemming)
//! - [`time_series`]: Time series analysis and forecasting (ARIMA)
//! - [`index`]: Approximate nearest neighbor search (HNSW)
//! - [`recommend`]: Recommendation systems (content-based, collaborative filtering)
//! - [`chaos`]: Chaos engineering configuration (from renacer)

pub mod autograd;
pub mod bayesian;
pub mod chaos;
pub mod classification;
pub mod cluster;
pub mod data;
pub mod decomposition;
pub mod error;
pub mod format;
pub mod glm;
pub mod graph;
pub mod index;
pub mod linear_model;
pub mod loss;
pub mod metrics;
pub mod mining;
pub mod model_selection;
pub mod nn;
pub mod optim;
pub mod prelude;
pub mod preprocessing;
pub mod primitives;
pub mod recommend;
pub mod serialization;
pub mod stats;
pub mod text;
pub mod time_series;
pub mod traits;
pub mod tree;

pub use error::{AprenderError, Result};
pub use primitives::{Matrix, Vector};
pub use traits::{Estimator, Transformer, UnsupervisedEstimator};
