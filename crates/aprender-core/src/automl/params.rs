//! Type-safe hyperparameter definitions.
//!
//! Parameter keys are enums rather than strings to catch typos at compile time.
//! This eliminates an entire class of runtime errors (Poka-Yoke principle).
//!
//! # Example
//!
//! ```
//! use aprender::automl::params::RandomForestParam as RF;
//! use aprender::automl::SearchSpace;
//!
//! let space = SearchSpace::new()
//!     .add(RF::NEstimators, 10..500)
//!     .add(RF::MaxDepth, 2..32);
//! ```

use std::fmt;
use std::hash::Hash;

/// Trait for parameter key enums.
///
/// Implement this for custom model families to get type-safe parameter spaces.
pub trait ParamKey: Copy + Clone + Eq + Hash + fmt::Debug {
    /// Human-readable parameter name for logging/display.
    fn name(&self) -> &'static str;
}

/// Random Forest hyperparameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RandomForestParam {
    /// Number of trees in the forest.
    NEstimators,
    /// Maximum depth of each tree.
    MaxDepth,
    /// Minimum samples required to split an internal node.
    MinSamplesSplit,
    /// Minimum samples required at a leaf node.
    MinSamplesLeaf,
    /// Number of features to consider for best split.
    MaxFeatures,
    /// Whether to use bootstrap samples.
    Bootstrap,
}

impl ParamKey for RandomForestParam {
    fn name(&self) -> &'static str {
        match self {
            Self::NEstimators => "n_estimators",
            Self::MaxDepth => "max_depth",
            Self::MinSamplesSplit => "min_samples_split",
            Self::MinSamplesLeaf => "min_samples_leaf",
            Self::MaxFeatures => "max_features",
            Self::Bootstrap => "bootstrap",
        }
    }
}

impl fmt::Display for RandomForestParam {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Gradient Boosting hyperparameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GradientBoostingParam {
    /// Number of boosting stages.
    NEstimators,
    /// Learning rate (shrinkage).
    LearningRate,
    /// Maximum depth of individual trees.
    MaxDepth,
    /// Fraction of samples used for each tree.
    Subsample,
    /// Minimum samples required at a leaf node.
    MinSamplesLeaf,
}

impl ParamKey for GradientBoostingParam {
    fn name(&self) -> &'static str {
        match self {
            Self::NEstimators => "n_estimators",
            Self::LearningRate => "learning_rate",
            Self::MaxDepth => "max_depth",
            Self::Subsample => "subsample",
            Self::MinSamplesLeaf => "min_samples_leaf",
        }
    }
}

impl fmt::Display for GradientBoostingParam {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// K-Nearest Neighbors hyperparameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KNNParam {
    /// Number of neighbors.
    NNeighbors,
    /// Weight function used in prediction.
    Weights,
    /// Distance metric.
    Metric,
    /// Leaf size for tree-based algorithms.
    LeafSize,
}

impl ParamKey for KNNParam {
    fn name(&self) -> &'static str {
        match self {
            Self::NNeighbors => "n_neighbors",
            Self::Weights => "weights",
            Self::Metric => "metric",
            Self::LeafSize => "leaf_size",
        }
    }
}

impl fmt::Display for KNNParam {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Linear model hyperparameters (Ridge, Lasso, `ElasticNet`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LinearParam {
    /// Regularization strength (alpha).
    Alpha,
    /// L1/L2 mixing ratio for `ElasticNet`.
    L1Ratio,
    /// Whether to fit intercept.
    FitIntercept,
    /// Maximum iterations for iterative solvers.
    MaxIter,
    /// Convergence tolerance.
    Tol,
}

impl ParamKey for LinearParam {
    fn name(&self) -> &'static str {
        match self {
            Self::Alpha => "alpha",
            Self::L1Ratio => "l1_ratio",
            Self::FitIntercept => "fit_intercept",
            Self::MaxIter => "max_iter",
            Self::Tol => "tol",
        }
    }
}

impl fmt::Display for LinearParam {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Decision Tree hyperparameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecisionTreeParam {
    /// Maximum depth of the tree.
    MaxDepth,
    /// Minimum samples required to split.
    MinSamplesSplit,
    /// Minimum samples required at leaf.
    MinSamplesLeaf,
    /// Impurity criterion (gini, entropy).
    Criterion,
    /// Strategy for splitting (best, random).
    Splitter,
}

impl ParamKey for DecisionTreeParam {
    fn name(&self) -> &'static str {
        match self {
            Self::MaxDepth => "max_depth",
            Self::MinSamplesSplit => "min_samples_split",
            Self::MinSamplesLeaf => "min_samples_leaf",
            Self::Criterion => "criterion",
            Self::Splitter => "splitter",
        }
    }
}

impl fmt::Display for DecisionTreeParam {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// K-Means clustering hyperparameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KMeansParam {
    /// Number of clusters.
    NClusters,
    /// Maximum iterations.
    MaxIter,
    /// Number of initializations.
    NInit,
    /// Initialization method.
    Init,
    /// Convergence tolerance.
    Tol,
}

impl ParamKey for KMeansParam {
    fn name(&self) -> &'static str {
        match self {
            Self::NClusters => "n_clusters",
            Self::MaxIter => "max_iter",
            Self::NInit => "n_init",
            Self::Init => "init",
            Self::Tol => "tol",
        }
    }
}

impl fmt::Display for KMeansParam {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
#[path = "params_tests.rs"]
mod tests;
