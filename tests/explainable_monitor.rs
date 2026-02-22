//! Integration tests: aprender explainability wrappers + entrenar InferenceMonitor
//!
//! GH-305: These tests must be in `tests/` (not in `src/`) because the test binary
//! is a separate crate instance from the library. entrenar depends on aprender-lib,
//! so its generic bounds require aprender-lib's traits. Integration tests use
//! `aprender::` which refers to the library crate, matching entrenar's dependency.

use aprender::classification::LogisticRegression;
use aprender::explainable::{
    EnsembleExplainable, LinearExplainable, LogisticExplainable, TreeExplainable,
};
use aprender::linear_model::LinearRegression;
use aprender::primitives::{Matrix, Vector};
use aprender::traits::Estimator;
use aprender::tree::{DecisionTreeRegressor, RandomForestRegressor};

use entrenar::monitor::inference::{
    ForestPath, InferenceMonitor, LinearPath, RingCollector, TreePath,
};

#[test]
fn test_linear_with_inference_monitor() {
    let x = Matrix::from_vec(5, 2, vec![1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 4.0, 5.0, 5.0, 4.0])
        .expect("Matrix creation");
    let y = Vector::from_slice(&[4.0, 8.0, 8.0, 14.0, 13.0]);

    let mut model = LinearRegression::new();
    model.fit(&x, &y).expect("fit");

    let explainable = LinearExplainable::new(model);
    let collector: RingCollector<LinearPath, 64> = RingCollector::new();
    let mut monitor = InferenceMonitor::new(explainable, collector);

    let sample = vec![2.0, 3.0];
    let outputs = monitor.predict(&sample, 1);

    assert_eq!(outputs.len(), 1);

    let traces = monitor.collector().recent(1);
    assert_eq!(traces.len(), 1);
}

#[test]
fn test_tree_with_inference_monitor() {
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("Matrix creation");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);

    let mut model = DecisionTreeRegressor::new();
    model.fit(&x, &y).expect("fit");

    let explainable = TreeExplainable::new(model, 1);
    let collector: RingCollector<TreePath, 64> = RingCollector::new();
    let mut monitor = InferenceMonitor::new(explainable, collector);

    let sample = vec![2.5];
    let outputs = monitor.predict(&sample, 1);

    assert_eq!(outputs.len(), 1);

    let traces = monitor.collector().recent(1);
    assert_eq!(traces.len(), 1);
}

#[test]
fn test_ensemble_with_inference_monitor() {
    let x = Matrix::from_vec(
        10,
        2,
        vec![
            1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 9.0,
            10.0, 10.0, 11.0,
        ],
    )
    .expect("Matrix creation");
    let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0]);

    let mut model = RandomForestRegressor::new(5);
    model.fit(&x, &y).expect("fit");

    let explainable = EnsembleExplainable::new(model, 2);
    let collector: RingCollector<ForestPath, 64> = RingCollector::new();
    let mut monitor = InferenceMonitor::new(explainable, collector);

    let sample = vec![5.0, 6.0];
    let outputs = monitor.predict(&sample, 1);

    assert_eq!(outputs.len(), 1);

    let traces = monitor.collector().recent(1);
    assert_eq!(traces.len(), 1);
}

#[test]
fn test_logistic_with_inference_monitor() {
    let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        .expect("Matrix creation");
    let y: &[usize] = &[0, 1, 1, 0];

    let mut model = LogisticRegression::new();
    model.fit(&x, y).expect("fit");

    let explainable = LogisticExplainable::new(model);
    let collector: RingCollector<LinearPath, 64> = RingCollector::new();
    let mut monitor = InferenceMonitor::new(explainable, collector);

    let sample = vec![0.5, 0.5];
    let outputs = monitor.predict(&sample, 1);

    assert_eq!(outputs.len(), 1);
    assert!(outputs[0] >= 0.0 && outputs[0] <= 1.0);

    let traces = monitor.collector().recent(1);
    assert_eq!(traces.len(), 1);
}
