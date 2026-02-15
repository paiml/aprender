
// =============================================================================
// Integration tests with InferenceMonitor
// =============================================================================

#[test]
fn test_linear_with_inference_monitor() {
    use entrenar::monitor::inference::{InferenceMonitor, LinearPath, RingCollector};

    let x = Matrix::from_vec(5, 2, vec![1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 4.0, 5.0, 5.0, 4.0])
        .expect("Matrix creation");
    let y = Vector::from_slice(&[4.0, 8.0, 8.0, 14.0, 13.0]);

    let mut model = LinearRegression::new();
    model.fit(&x, &y).expect("fit");

    let explainable = LinearExplainable::new(model);
    let collector: RingCollector<LinearPath, 64> = RingCollector::new();
    let mut monitor = InferenceMonitor::new(explainable, collector);

    // Make predictions through monitor
    let sample = vec![2.0, 3.0];
    let outputs = monitor.predict(&sample, 1);

    assert_eq!(outputs.len(), 1);

    // Check that traces were collected
    let traces = monitor.collector().recent(1);
    assert_eq!(traces.len(), 1);
}

#[test]
fn test_tree_with_inference_monitor() {
    use entrenar::monitor::inference::{InferenceMonitor, RingCollector, TreePath};

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
    use entrenar::monitor::inference::{ForestPath, InferenceMonitor, RingCollector};

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
