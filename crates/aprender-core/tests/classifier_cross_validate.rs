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
