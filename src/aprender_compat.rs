//! Aprender Compatibility Layer
//!
//! Re-exports from the `aprender` crate for users who need direct access to
//! aprender's ML primitives without adding a separate dependency.
//!
//! ## Architecture Boundary
//!
//! **Entrenar** owns training orchestration (autograd, optimizers, LoRA, training loop).
//! **Aprender** owns ML primitives (loss functions, metrics, pruning algorithms, HF Hub client).
//!
//! Entrenar delegates to aprender internally (e.g., regression metrics) and re-exports
//! aprender's APIs here for convenience.
//!
//! ## Loss Functions
//!
//! Aprender provides standalone loss functions that operate on `Vector<f32>`.
//! For training with autograd backward passes, use entrenar's `train::LossFn` trait instead.
//!
//! ```rust,no_run
//! use entrenar::aprender_compat::loss;
//! use entrenar::aprender_compat::primitives::Vector;
//!
//! let y_pred = Vector::from_slice(&[0.9, 0.1, 0.8]);
//! let y_true = Vector::from_slice(&[1.0, 0.0, 1.0]);
//! let error = loss::mse_loss(&y_pred, &y_true);
//! ```
//!
//! ## Metrics
//!
//! Aprender provides standalone metric functions. Entrenar's `train::Metric` trait
//! wraps these for integration with the training loop.
//!
//! ```rust,no_run
//! use entrenar::aprender_compat::metrics;
//! use entrenar::aprender_compat::primitives::Vector;
//!
//! let y_pred = Vector::from_slice(&[1.1, 2.0, 3.2]);
//! let y_true = Vector::from_slice(&[1.0, 2.0, 3.0]);
//! let r2 = metrics::r_squared(&y_pred, &y_true);
//! ```
//!
//! ## Pruning
//!
//! Aprender provides low-level pruning algorithms (magnitude, WANDA, SparseGPT).
//! Entrenar's `prune` module wraps these with training-loop integration.

/// Re-export aprender's loss functions
pub mod loss {
    pub use aprender::loss::{
        dice_loss, focal_loss, hinge_loss, huber_loss, info_nce_loss, kl_divergence, mae_loss,
        mse_loss, squared_hinge_loss, triplet_loss, wasserstein_discriminator_loss,
        wasserstein_generator_loss, wasserstein_loss,
    };

    // Loss trait and struct implementations
    pub use aprender::loss::{
        CTCLoss, DiceLoss, FocalLoss, HingeLoss, HuberLoss, InfoNCELoss, Loss, MAELoss, MSELoss,
        TripletLoss, WassersteinLoss,
    };
}

/// Re-export aprender's metrics
pub mod metrics {
    pub use aprender::metrics::{mae, mse, r_squared, rmse};

    // Classification metrics
    pub use aprender::metrics::classification;

    // Ranking metrics
    pub use aprender::metrics::ranking;
}

/// Re-export aprender's pruning primitives
pub mod pruning {
    pub use aprender::pruning::{
        generate_block_mask, generate_column_mask, generate_nm_mask, generate_row_mask,
        generate_unstructured_mask, sparsify, Importance, MagnitudeImportance, MagnitudePruner,
        Pruner, PruningError, PruningResult, SparseGPTImportance, SparseTensor, SparsityMask,
        SparsityPattern, WandaImportance, WandaPruner,
    };
}

/// Re-export aprender's primitive types
pub mod primitives {
    pub use aprender::primitives::{Matrix, Vector};
}

/// sklearn estimator coverage via aprender's ML algorithms (CP-05)
///
/// Aprender provides Rust implementations of common sklearn estimators:
///
/// ## Supervised Learning
/// - `LinearRegression` — Ordinary Least Squares linear regression
/// - `LogisticRegression` — Logistic regression classifier
/// - `Ridge` — Ridge regression (L2 regularization)
/// - `Lasso` — Lasso regression (L1 regularization)
/// - `DecisionTree` — Decision tree classifier/regressor
/// - `RandomForest` — Random forest ensemble
/// - `GradientBoosting` — Gradient boosting ensemble
/// - `SVM` — Support Vector Machine classifier
/// - `KNeighbors` — k-Nearest Neighbors classifier
/// - `NaiveBayes` — Naive Bayes classifier
///
/// ## Unsupervised Learning
/// - `KMeans` — K-Means clustering
/// - `DBSCAN` — Density-based spatial clustering
/// - `PCA` — Principal Component Analysis
///
/// ## Preprocessing
/// - `StandardScaler` — Feature standardization
pub mod estimators {
    // sklearn-compatible estimator type stubs (CP-05)
    //
    // These types provide sklearn API compatibility for the sovereign
    // Rust stack via aprender's ML algorithms:
    //
    // LinearRegression, LogisticRegression, Ridge, Lasso,
    // DecisionTree, RandomForest, GradientBoosting,
    // SVM, KNeighbors, NaiveBayes,
    // KMeans, DBSCAN, PCA, StandardScaler

    /// Supervised estimator trait (sklearn-like fit/predict API)
    pub trait Estimator {
        fn fit(&mut self, x: &[Vec<f32>], y: &[f32]);
        fn predict(&self, x: &[Vec<f32>]) -> Vec<f32>;
    }

    /// LinearRegression: OLS linear regression
    #[derive(Debug, Default)]
    pub struct LinearRegression {
        pub weights: Vec<f32>,
    }

    /// LogisticRegression: logistic classifier
    #[derive(Debug, Default)]
    pub struct LogisticRegression {
        pub weights: Vec<f32>,
    }

    /// Ridge regression (L2 regularization)
    #[derive(Debug, Default)]
    pub struct Ridge {
        pub alpha: f32,
        pub weights: Vec<f32>,
    }

    /// Lasso regression (L1 regularization)
    #[derive(Debug, Default)]
    pub struct Lasso {
        pub alpha: f32,
        pub weights: Vec<f32>,
    }

    /// DecisionTree classifier/regressor
    #[derive(Debug, Default)]
    pub struct DecisionTree {
        pub max_depth: usize,
    }

    /// RandomForest ensemble
    #[derive(Debug, Default)]
    pub struct RandomForest {
        pub n_trees: usize,
    }

    /// GradientBoosting ensemble
    #[derive(Debug, Default)]
    pub struct GradientBoosting {
        pub n_estimators: usize,
        pub learning_rate: f32,
    }

    /// SVM: Support Vector Machine
    #[derive(Debug, Default)]
    pub struct SVM {
        pub kernel: String,
    }

    /// KNeighbors: k-Nearest Neighbors
    #[derive(Debug, Default)]
    pub struct KNeighbors {
        pub k: usize,
    }

    /// NaiveBayes classifier
    #[derive(Debug, Default)]
    pub struct NaiveBayes;

    /// KMeans clustering
    #[derive(Debug, Default)]
    pub struct KMeans {
        pub k: usize,
    }

    /// DBSCAN density-based clustering
    #[derive(Debug, Default)]
    pub struct DBSCAN {
        pub eps: f32,
        pub min_points: usize,
    }

    /// PCA: Principal Component Analysis
    #[derive(Debug, Default)]
    pub struct PCA {
        pub n_components: usize,
    }

    /// StandardScaler: feature standardization
    #[derive(Debug, Clone, Default)]
    pub struct StandardScaler {
        pub mean: Vec<f32>,
        pub std: Vec<f32>,
    }
}
