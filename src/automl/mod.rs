//! Automated Machine Learning (AutoML) module.
//!
//! Provides hyperparameter optimization with type-safe parameter definitions,
//! multiple search strategies, and integration with aprender's Estimator trait.
//!
//! # Quick Start
//!
//! ```
//! use aprender::automl::{SearchSpace, RandomSearch, Trial};
//! use aprender::automl::params::RandomForestParam as RF;
//!
//! // Define type-safe search space (Poka-Yoke: compile-time typo prevention)
//! let space = SearchSpace::new()
//!     .add(RF::NEstimators, 10..500)
//!     .add(RF::MaxDepth, 2..20);
//!
//! // Random search with 50 trials
//! let search = RandomSearch::new(50);
//! ```
//!
//! # Design Principles
//!
//! - **Type Safety**: Parameter keys are enums, not strings—typos caught at compile time
//! - **Zero Unsafe**: Pure Rust implementation leveraging trueno SIMD
//! - **Extensible**: Custom parameter enums for any model family
//!
//! # References
//!
//! - Bergstra & Bengio (2012). Random Search for Hyper-Parameter Optimization. JMLR.
//! - Snoek et al. (2012). Practical Bayesian Optimization. NeurIPS.

pub mod params;
mod search;
mod tpe;
mod tuner;

pub use params::{GradientBoostingParam, RandomForestParam};
pub use search::{
    GridSearch, HyperParam, LogScale, ParamValue, RandomSearch, SearchSpace, SearchStrategy, Trial,
    TrialResult,
};
pub use tpe::{TPEConfig, TPE};
pub use tuner::{AutoTuner, Callback, EarlyStopping, ProgressCallback, TimeBudget, TuneResult};
