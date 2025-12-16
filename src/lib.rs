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
//! - [`code`]: Code analysis and code2vec embeddings
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
//! - [`synthetic`]: Synthetic data generation for AutoML (EDA, back-translation, MixUp)
//! - [`bundle`]: Model bundling and memory paging for large models
//! - [`cache`]: Cache hierarchy and model registry for large model management
//! - [`chaos`]: Chaos engineering configuration (from renacer)
//! - [`inspect`]: Model inspection tooling (header analysis, diff, quality scoring)
//! - [`loading`]: Model loading subsystem with WCET and cryptographic agility
//! - [`scoring`]: 100-point model quality scoring system
//! - [`zoo`]: Model zoo protocol for sharing and discovery
//! - [`embed`]: Data embedding with test data and tiny model representations
//! - [`native`]: SIMD-native model format for zero-copy inference
//! - [`stack`]: Sovereign AI Stack integration types
//! - [`online`]: Online learning and dynamic retraining infrastructure

pub mod active_learning;
pub mod autograd;
pub mod automl;
pub mod bayesian;
/// Model evaluation and benchmarking framework (spec §7.10)
pub mod bench;
pub mod bundle;
pub mod cache;
pub mod calibration;
pub mod chaos;
/// Compiler-in-the-Loop Learning (CITL) for transpiler support.
pub mod citl;
pub mod classification;
pub mod cluster;
pub mod code;
pub mod data;
pub mod decomposition;
/// Data embedding with test data and tiny model representations (spec §4)
pub mod embed;
pub mod ensemble;
pub mod error;
/// Explainability wrappers for inference monitoring (entrenar integration)
#[cfg(feature = "inference-monitoring")]
pub mod explainable;
pub mod format;
pub mod glm;
pub mod gnn;
pub mod graph;
/// Hugging Face Hub integration (GH-100)
#[cfg(feature = "hf-hub-integration")]
pub mod hf_hub;
pub mod index;
/// Model inspection tooling (spec §7.2)
pub mod inspect;
pub mod interpret;
pub mod linear_model;
/// Model loading subsystem with WCET and cryptographic agility (spec §7.1)
pub mod loading;
pub mod loss;
pub mod metaheuristics;
pub mod metrics;
pub mod mining;
pub mod model_selection;
pub mod monte_carlo;
/// SIMD-native model format for zero-copy Trueno inference (spec §5)
pub mod native;
pub mod nn;
/// Online learning and dynamic retraining infrastructure
pub mod online;
pub mod optim;
pub mod prelude;
pub mod preprocessing;
pub mod primitives;
/// Model Quality Assurance module (spec §7.9)
pub mod qa;
pub mod recommend;
pub mod regularization;
/// 100-point model quality scoring system (spec §7)
pub mod scoring;
pub mod serialization;
/// Sovereign AI Stack integration types (spec §9)
pub mod stack;
pub mod stats;
pub mod synthetic;
pub mod text;
pub mod time_series;
pub mod traits;
pub mod transfer;
pub mod tree;
/// Pipeline verification & visualization system (APR-VERIFY-001)
pub mod verify;
pub mod weak_supervision;
/// Model zoo protocol for sharing and discovery (spec §8)
pub mod zoo;

pub use error::{AprenderError, Result};
pub use primitives::{Matrix, Vector};
pub use traits::{Estimator, Transformer, UnsupervisedEstimator};
