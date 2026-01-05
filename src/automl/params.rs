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
mod tests {
    use super::*;

    // ========================================================================
    // RandomForestParam Tests
    // ========================================================================

    #[test]
    fn test_random_forest_param_names() {
        assert_eq!(RandomForestParam::NEstimators.name(), "n_estimators");
        assert_eq!(RandomForestParam::MaxDepth.name(), "max_depth");
        assert_eq!(
            RandomForestParam::MinSamplesSplit.name(),
            "min_samples_split"
        );
        assert_eq!(RandomForestParam::MinSamplesLeaf.name(), "min_samples_leaf");
        assert_eq!(RandomForestParam::MaxFeatures.name(), "max_features");
        assert_eq!(RandomForestParam::Bootstrap.name(), "bootstrap");
    }

    #[test]
    fn test_random_forest_display() {
        assert_eq!(
            format!("{}", RandomForestParam::NEstimators),
            "n_estimators"
        );
        assert_eq!(format!("{}", RandomForestParam::MaxDepth), "max_depth");
        assert_eq!(
            format!("{}", RandomForestParam::MinSamplesSplit),
            "min_samples_split"
        );
        assert_eq!(
            format!("{}", RandomForestParam::MinSamplesLeaf),
            "min_samples_leaf"
        );
        assert_eq!(
            format!("{}", RandomForestParam::MaxFeatures),
            "max_features"
        );
        assert_eq!(format!("{}", RandomForestParam::Bootstrap), "bootstrap");
    }

    // ========================================================================
    // GradientBoostingParam Tests
    // ========================================================================

    #[test]
    fn test_gradient_boosting_param_names() {
        assert_eq!(GradientBoostingParam::NEstimators.name(), "n_estimators");
        assert_eq!(GradientBoostingParam::LearningRate.name(), "learning_rate");
        assert_eq!(GradientBoostingParam::MaxDepth.name(), "max_depth");
        assert_eq!(GradientBoostingParam::Subsample.name(), "subsample");
        assert_eq!(
            GradientBoostingParam::MinSamplesLeaf.name(),
            "min_samples_leaf"
        );
    }

    #[test]
    fn test_gradient_boosting_display() {
        assert_eq!(
            format!("{}", GradientBoostingParam::NEstimators),
            "n_estimators"
        );
        assert_eq!(
            format!("{}", GradientBoostingParam::LearningRate),
            "learning_rate"
        );
        assert_eq!(format!("{}", GradientBoostingParam::MaxDepth), "max_depth");
        assert_eq!(format!("{}", GradientBoostingParam::Subsample), "subsample");
        assert_eq!(
            format!("{}", GradientBoostingParam::MinSamplesLeaf),
            "min_samples_leaf"
        );
    }

    // ========================================================================
    // KNNParam Tests
    // ========================================================================

    #[test]
    fn test_knn_param_names() {
        assert_eq!(KNNParam::NNeighbors.name(), "n_neighbors");
        assert_eq!(KNNParam::Weights.name(), "weights");
        assert_eq!(KNNParam::Metric.name(), "metric");
        assert_eq!(KNNParam::LeafSize.name(), "leaf_size");
    }

    #[test]
    fn test_knn_display() {
        assert_eq!(format!("{}", KNNParam::NNeighbors), "n_neighbors");
        assert_eq!(format!("{}", KNNParam::Weights), "weights");
        assert_eq!(format!("{}", KNNParam::Metric), "metric");
        assert_eq!(format!("{}", KNNParam::LeafSize), "leaf_size");
    }

    #[test]
    fn test_knn_param_equality() {
        assert_eq!(KNNParam::NNeighbors, KNNParam::NNeighbors);
        assert_ne!(KNNParam::NNeighbors, KNNParam::Weights);
    }

    #[test]
    fn test_knn_param_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(KNNParam::NNeighbors);
        set.insert(KNNParam::Weights);
        set.insert(KNNParam::NNeighbors);
        assert_eq!(set.len(), 2);
    }

    // ========================================================================
    // LinearParam Tests
    // ========================================================================

    #[test]
    fn test_linear_param_names() {
        assert_eq!(LinearParam::Alpha.name(), "alpha");
        assert_eq!(LinearParam::L1Ratio.name(), "l1_ratio");
        assert_eq!(LinearParam::FitIntercept.name(), "fit_intercept");
        assert_eq!(LinearParam::MaxIter.name(), "max_iter");
        assert_eq!(LinearParam::Tol.name(), "tol");
    }

    #[test]
    fn test_linear_display() {
        assert_eq!(format!("{}", LinearParam::Alpha), "alpha");
        assert_eq!(format!("{}", LinearParam::L1Ratio), "l1_ratio");
        assert_eq!(format!("{}", LinearParam::FitIntercept), "fit_intercept");
        assert_eq!(format!("{}", LinearParam::MaxIter), "max_iter");
        assert_eq!(format!("{}", LinearParam::Tol), "tol");
    }

    #[test]
    fn test_linear_param_equality() {
        assert_eq!(LinearParam::Alpha, LinearParam::Alpha);
        assert_ne!(LinearParam::Alpha, LinearParam::L1Ratio);
    }

    #[test]
    fn test_linear_param_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(LinearParam::Alpha);
        set.insert(LinearParam::L1Ratio);
        set.insert(LinearParam::Alpha);
        assert_eq!(set.len(), 2);
    }

    // ========================================================================
    // DecisionTreeParam Tests
    // ========================================================================

    #[test]
    fn test_decision_tree_param_names() {
        assert_eq!(DecisionTreeParam::MaxDepth.name(), "max_depth");
        assert_eq!(
            DecisionTreeParam::MinSamplesSplit.name(),
            "min_samples_split"
        );
        assert_eq!(DecisionTreeParam::MinSamplesLeaf.name(), "min_samples_leaf");
        assert_eq!(DecisionTreeParam::Criterion.name(), "criterion");
        assert_eq!(DecisionTreeParam::Splitter.name(), "splitter");
    }

    #[test]
    fn test_decision_tree_display() {
        assert_eq!(format!("{}", DecisionTreeParam::MaxDepth), "max_depth");
        assert_eq!(
            format!("{}", DecisionTreeParam::MinSamplesSplit),
            "min_samples_split"
        );
        assert_eq!(
            format!("{}", DecisionTreeParam::MinSamplesLeaf),
            "min_samples_leaf"
        );
        assert_eq!(format!("{}", DecisionTreeParam::Criterion), "criterion");
        assert_eq!(format!("{}", DecisionTreeParam::Splitter), "splitter");
    }

    #[test]
    fn test_decision_tree_param_equality() {
        assert_eq!(DecisionTreeParam::MaxDepth, DecisionTreeParam::MaxDepth);
        assert_ne!(DecisionTreeParam::MaxDepth, DecisionTreeParam::Criterion);
    }

    #[test]
    fn test_decision_tree_param_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(DecisionTreeParam::MaxDepth);
        set.insert(DecisionTreeParam::Criterion);
        set.insert(DecisionTreeParam::MaxDepth);
        assert_eq!(set.len(), 2);
    }

    // ========================================================================
    // KMeansParam Tests
    // ========================================================================

    #[test]
    fn test_kmeans_param_names() {
        assert_eq!(KMeansParam::NClusters.name(), "n_clusters");
        assert_eq!(KMeansParam::MaxIter.name(), "max_iter");
        assert_eq!(KMeansParam::NInit.name(), "n_init");
        assert_eq!(KMeansParam::Init.name(), "init");
        assert_eq!(KMeansParam::Tol.name(), "tol");
    }

    #[test]
    fn test_kmeans_display() {
        assert_eq!(format!("{}", KMeansParam::NClusters), "n_clusters");
        assert_eq!(format!("{}", KMeansParam::MaxIter), "max_iter");
        assert_eq!(format!("{}", KMeansParam::NInit), "n_init");
        assert_eq!(format!("{}", KMeansParam::Init), "init");
        assert_eq!(format!("{}", KMeansParam::Tol), "tol");
    }

    #[test]
    fn test_kmeans_param_equality() {
        assert_eq!(KMeansParam::NClusters, KMeansParam::NClusters);
        assert_ne!(KMeansParam::NClusters, KMeansParam::MaxIter);
    }

    #[test]
    fn test_kmeans_param_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(KMeansParam::NClusters);
        set.insert(KMeansParam::MaxIter);
        set.insert(KMeansParam::NClusters);
        assert_eq!(set.len(), 2);
    }

    // ========================================================================
    // Original Tests (kept for backwards compatibility)
    // ========================================================================

    #[test]
    fn test_param_display() {
        assert_eq!(format!("{}", RandomForestParam::MaxDepth), "max_depth");
        assert_eq!(
            format!("{}", GradientBoostingParam::LearningRate),
            "learning_rate"
        );
    }

    #[test]
    fn test_param_equality() {
        let p1 = RandomForestParam::NEstimators;
        let p2 = RandomForestParam::NEstimators;
        let p3 = RandomForestParam::MaxDepth;

        assert_eq!(p1, p2);
        assert_ne!(p1, p3);
    }

    #[test]
    fn test_param_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(RandomForestParam::NEstimators);
        set.insert(RandomForestParam::MaxDepth);
        set.insert(RandomForestParam::NEstimators); // duplicate

        assert_eq!(set.len(), 2);
    }

    // ========================================================================
    // Clone and Debug Tests
    // ========================================================================

    #[test]
    fn test_random_forest_clone_debug() {
        let p = RandomForestParam::NEstimators;
        let cloned = p;
        assert_eq!(p, cloned);
        assert!(format!("{p:?}").contains("NEstimators"));
    }

    #[test]
    fn test_gradient_boosting_clone_debug() {
        let p = GradientBoostingParam::LearningRate;
        let cloned = p;
        assert_eq!(p, cloned);
        assert!(format!("{p:?}").contains("LearningRate"));
    }

    #[test]
    fn test_knn_clone_debug() {
        let p = KNNParam::NNeighbors;
        let cloned = p;
        assert_eq!(p, cloned);
        assert!(format!("{p:?}").contains("NNeighbors"));
    }

    #[test]
    fn test_linear_clone_debug() {
        let p = LinearParam::Alpha;
        let cloned = p;
        assert_eq!(p, cloned);
        assert!(format!("{p:?}").contains("Alpha"));
    }

    #[test]
    fn test_decision_tree_clone_debug() {
        let p = DecisionTreeParam::MaxDepth;
        let cloned = p;
        assert_eq!(p, cloned);
        assert!(format!("{p:?}").contains("MaxDepth"));
    }

    #[test]
    fn test_kmeans_clone_debug() {
        let p = KMeansParam::NClusters;
        let cloned = p;
        assert_eq!(p, cloned);
        assert!(format!("{p:?}").contains("NClusters"));
    }
}
