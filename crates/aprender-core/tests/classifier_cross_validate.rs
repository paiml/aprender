//! Pillar 1 (beat scikit-learn): generic `cross_validate` now works over a
//! classifier, mirroring sklearn's `cross_val_score(any_estimator, ...)`.
//! Before the `Estimator` impl on `RandomForestClassifier` this didn't compile
//! (classifiers spoke `&[usize]`, the generic machinery speaks `Vector<f32>`).

use aprender::datasets::make_classification;
use aprender::model_selection::{cross_validate, KFold};
use aprender::tree::RandomForestClassifier;
use aprender::Vector;

#[test]
fn cross_validate_over_random_forest_classifier() {
    let (x, labels) = make_classification(150, 8, 4, 3, 42);
    let y = Vector::from_vec(labels.iter().map(|&l| l as f32).collect());

    let model = RandomForestClassifier::new(25)
        .with_max_depth(10)
        .with_random_state(42);

    let cv = KFold::new(5);
    let result =
        cross_validate(&model, &x, &y, &cv).expect("cross_validate must work over a classifier");

    assert_eq!(result.scores.len(), 5, "5-fold CV yields 5 scores");
    let mean = result.mean();
    // 3-class, learnable data: mean CV accuracy must be well above random (1/3).
    assert!(
        mean > 0.7,
        "RandomForestClassifier 5-fold CV accuracy {mean} not learnable (random ≈ 0.33)"
    );
}

#[test]
fn cross_validate_over_decision_tree_classifier() {
    use aprender::tree::DecisionTreeClassifier;
    let (x, labels) = make_classification(150, 8, 4, 3, 42);
    let y = Vector::from_vec(labels.iter().map(|&l| l as f32).collect());
    let model = DecisionTreeClassifier::new().with_max_depth(10);
    let result = cross_validate(&model, &x, &y, &KFold::new(5)).expect("cv over decision tree");
    assert_eq!(result.scores.len(), 5);
    assert!(
        result.mean() > 0.6,
        "DecisionTree CV acc {} not learnable",
        result.mean()
    );
}

#[test]
fn cross_validate_over_logistic_regression() {
    use aprender::classification::LogisticRegression;
    let (x, labels) = make_classification(150, 6, 4, 2, 7);
    let y = Vector::from_vec(labels.iter().map(|&l| l as f32).collect());
    let model = LogisticRegression::new().with_max_iter(300);
    let result =
        cross_validate(&model, &x, &y, &KFold::new(5)).expect("cv over logistic regression");
    assert_eq!(result.scores.len(), 5);
    assert!(
        result.mean() > 0.6,
        "LogReg CV acc {} not learnable",
        result.mean()
    );
}

#[test]
fn cross_val_score_returns_same_as_cross_validate_scores() {
    use aprender::model_selection::cross_val_score;
    let (x, labels) = make_classification(150, 8, 4, 3, 42);
    let y = Vector::from_vec(labels.iter().map(|&l| l as f32).collect());
    let model = RandomForestClassifier::new(25)
        .with_max_depth(10)
        .with_random_state(42);
    let cv = KFold::new(5);
    let scores = cross_val_score(&model, &x, &y, &cv).expect("cross_val_score");
    let result = cross_validate(&model, &x, &y, &cv).expect("cross_validate");
    assert_eq!(
        scores, result.scores,
        "cross_val_score must return exactly cross_validate().scores (sklearn parity)"
    );
    assert_eq!(scores.len(), 5);
}

#[test]
fn cross_validate_over_knn_classifier() {
    use aprender::classification::KNearestNeighbors;
    let (x, labels) = make_classification(150, 6, 4, 3, 11);
    let y = Vector::from_vec(labels.iter().map(|&l| l as f32).collect());
    let model = KNearestNeighbors::new(5);
    let result = cross_validate(&model, &x, &y, &KFold::new(5)).expect("cv over knn");
    assert_eq!(result.scores.len(), 5);
    assert!(
        result.mean() > 0.6,
        "KNN CV acc {} not learnable",
        result.mean()
    );
}

#[test]
fn cross_validate_over_gaussian_nb() {
    use aprender::classification::GaussianNB;
    let (x, labels) = make_classification(150, 6, 4, 3, 5);
    let y = Vector::from_vec(labels.iter().map(|&l| l as f32).collect());
    let model = GaussianNB::new();
    let result = cross_validate(&model, &x, &y, &KFold::new(5)).expect("cv over gaussian_nb");
    assert_eq!(result.scores.len(), 5);
    assert!(
        result.mean() > 0.6,
        "GaussianNB CV acc {} not learnable",
        result.mean()
    );
}

#[test]
fn cross_validate_over_gradient_boosting_classifier() {
    use aprender::tree::GradientBoostingClassifier;
    // GBM classifier is binary (sigmoid) -> use 2 classes
    let (x, labels) = make_classification(120, 6, 4, 2, 3);
    let y = Vector::from_vec(labels.iter().map(|&l| l as f32).collect());
    let model = GradientBoostingClassifier::new();
    let result = cross_validate(&model, &x, &y, &KFold::new(5)).expect("cv over gbm");
    assert_eq!(result.scores.len(), 5);
    assert!(
        result.mean() > 0.6,
        "GBM CV acc {} not learnable",
        result.mean()
    );
}

#[test]
fn cross_validate_over_random_forest_regressor() {
    use aprender::datasets::make_regression;
    use aprender::tree::RandomForestRegressor;
    let (x, y) = make_regression(150, 5, 0.1, 7);
    let model = RandomForestRegressor::new(20);
    let result = cross_validate(&model, &x, &y, &KFold::new(5)).expect("cv over RFR");
    assert_eq!(result.scores.len(), 5);
    // R² (regression score); low-noise linear data -> should beat the mean baseline
    assert!(
        result.mean() > 0.0 && result.mean().is_finite(),
        "RFR CV R² {} should be positive",
        result.mean()
    );
}

#[test]
fn cross_validate_over_decision_tree_regressor() {
    use aprender::datasets::make_regression;
    use aprender::tree::DecisionTreeRegressor;
    let (x, y) = make_regression(150, 5, 0.1, 7);
    let model = DecisionTreeRegressor::new();
    let result = cross_validate(&model, &x, &y, &KFold::new(5)).expect("cv over DTR");
    assert_eq!(result.scores.len(), 5);
    assert!(
        result.mean().is_finite(),
        "DTR CV R² {} must be finite",
        result.mean()
    );
}

#[test]
fn grid_search_picks_best_hyperparameters() {
    use aprender::model_selection::grid_search;
    use aprender::tree::RandomForestClassifier;
    let (x, labels) = make_classification(150, 8, 4, 3, 42);
    let y = Vector::from_vec(labels.iter().map(|&l| l as f32).collect());
    let depths = [2usize, 5, 12];
    let result = grid_search(
        &depths,
        |&d| {
            RandomForestClassifier::new(20)
                .with_max_depth(d)
                .with_random_state(42)
        },
        &x,
        &y,
        &KFold::new(5),
    )
    .expect("grid_search");
    assert_eq!(result.mean_scores.len(), 3);
    // best_score is the max over the grid, and best_params/index are consistent
    let max = result.mean_scores.iter().copied().fold(f32::MIN, f32::max);
    assert!((result.best_score - max).abs() < 1e-6);
    assert_eq!(result.best_params, depths[result.best_index]);
}

#[test]
fn randomized_search_samples_subset_and_picks_best() {
    use aprender::model_selection::randomized_search;
    use aprender::tree::RandomForestClassifier;
    let (x, labels) = make_classification(150, 8, 4, 3, 42);
    let y = Vector::from_vec(labels.iter().map(|&l| l as f32).collect());
    let depths: Vec<usize> = (1..=10).collect();
    let result = randomized_search(
        &depths,
        4, // sample 4 of 10
        7,
        |&d| {
            RandomForestClassifier::new(15)
                .with_max_depth(d)
                .with_random_state(0)
        },
        &x,
        &y,
        &KFold::new(5),
    )
    .expect("randomized_search");
    assert_eq!(
        result.mean_scores.len(),
        4,
        "samples exactly n_iter candidates"
    );
    let max = result.mean_scores.iter().copied().fold(f32::MIN, f32::max);
    assert!((result.best_score - max).abs() < 1e-6);
    assert_eq!(result.best_score, result.mean_scores[result.best_index]);
}
