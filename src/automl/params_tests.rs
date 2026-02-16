pub(crate) use super::*;

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
