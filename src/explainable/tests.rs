//! Tests for explainability wrappers

pub(crate) use super::*;
pub(crate) use crate::classification::LogisticRegression;
pub(crate) use crate::linear_model::LinearRegression;
pub(crate) use crate::primitives::{Matrix, Vector};
pub(crate) use crate::traits::Estimator;
pub(crate) use crate::tree::{DecisionTreeRegressor, RandomForestRegressor};
pub(crate) use entrenar::monitor::inference::Explainable;

// Import extension traits from parent module
pub(crate) use super::{
    IntoEnsembleExplainable, IntoExplainable, IntoLogisticExplainable, IntoTreeExplainable,
};

// =============================================================================
// LinearExplainable tests
// =============================================================================

#[test]
fn test_linear_explainable_new() {
    // Use non-collinear data: x1 and x2 are independent
    let x = Matrix::from_vec(5, 2, vec![1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 4.0, 5.0, 5.0, 4.0])
        .expect("Matrix creation");
    let y = Vector::from_slice(&[4.0, 8.0, 8.0, 14.0, 13.0]); // Roughly y = x1 + 2*x2

    let mut model = LinearRegression::new();
    model.fit(&x, &y).expect("fit");

    let explainable = LinearExplainable::new(model);
    assert_eq!(explainable.n_features(), 2);
}

#[test]
fn test_linear_explainable_predict_explained() {
    let x = Matrix::from_vec(5, 2, vec![1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 4.0, 5.0, 5.0, 4.0])
        .expect("Matrix creation");
    let y = Vector::from_slice(&[4.0, 8.0, 8.0, 14.0, 13.0]);

    let mut model = LinearRegression::new();
    model.fit(&x, &y).expect("fit");

    let explainable = LinearExplainable::new(model);

    // Test single sample
    let sample = vec![2.0, 3.0];
    let (outputs, paths) = explainable.predict_explained(&sample, 1);

    assert_eq!(outputs.len(), 1);
    assert_eq!(paths.len(), 1);
    assert_eq!(paths[0].contributions.len(), 2);
}

#[test]
fn test_linear_explainable_explain_one() {
    let x = Matrix::from_vec(5, 2, vec![1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 4.0, 5.0, 5.0, 4.0])
        .expect("Matrix creation");
    let y = Vector::from_slice(&[4.0, 8.0, 8.0, 14.0, 13.0]);

    let mut model = LinearRegression::new();
    model.fit(&x, &y).expect("fit");

    let explainable = LinearExplainable::new(model);
    let sample = vec![2.0, 3.0];
    let path = explainable.explain_one(&sample);

    assert_eq!(path.contributions.len(), 2);
    // Contributions should sum to logit - intercept
    let contrib_sum: f32 = path.contributions.iter().sum();
    let expected = path.logit - path.intercept;
    assert!((contrib_sum - expected).abs() < 1e-5);
}

#[test]
fn test_linear_explainable_batch() {
    let x = Matrix::from_vec(5, 2, vec![1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 4.0, 5.0, 5.0, 4.0])
        .expect("Matrix creation");
    let y = Vector::from_slice(&[4.0, 8.0, 8.0, 14.0, 13.0]);

    let mut model = LinearRegression::new();
    model.fit(&x, &y).expect("fit");

    let explainable = LinearExplainable::new(model);

    // Test batch of 3 samples
    let samples = vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0];
    let (outputs, paths) = explainable.predict_explained(&samples, 3);

    assert_eq!(outputs.len(), 3);
    assert_eq!(paths.len(), 3);
}

#[test]
fn test_linear_explainable_into_trait() {
    let x = Matrix::from_vec(5, 2, vec![1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 4.0, 5.0, 5.0, 4.0])
        .expect("Matrix creation");
    let y = Vector::from_slice(&[4.0, 8.0, 8.0, 14.0, 13.0]);

    let mut model = LinearRegression::new();
    model.fit(&x, &y).expect("fit");

    let explainable = model.into_explainable();
    assert_eq!(explainable.n_features(), 2);
}

// =============================================================================
// LogisticExplainable tests
// =============================================================================

#[test]
fn test_logistic_explainable_new() {
    let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        .expect("Matrix creation");
    let y: &[usize] = &[0, 1, 1, 0];

    let mut model = LogisticRegression::new();
    model.fit(&x, y).expect("fit");

    let explainable = LogisticExplainable::new(model);
    assert_eq!(explainable.n_features(), 2);
}

#[test]
fn test_logistic_explainable_predict_explained() {
    let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        .expect("Matrix creation");
    let y: &[usize] = &[0, 1, 1, 0];

    let mut model = LogisticRegression::new();
    model.fit(&x, y).expect("fit");

    let explainable = LogisticExplainable::new(model);

    let sample = vec![0.5, 0.5];
    let (outputs, paths) = explainable.predict_explained(&sample, 1);

    assert_eq!(outputs.len(), 1);
    assert_eq!(paths.len(), 1);

    // Output should be a probability (0-1)
    assert!(outputs[0] >= 0.0 && outputs[0] <= 1.0);

    // Path should have probability set
    assert!(paths[0].probability.is_some());
}

#[test]
fn test_logistic_explainable_explain_one() {
    let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        .expect("Matrix creation");
    let y: &[usize] = &[0, 1, 1, 0];

    let mut model = LogisticRegression::new();
    model.fit(&x, y).expect("fit");

    let explainable = LogisticExplainable::new(model);
    let sample = vec![0.5, 0.5];
    let path = explainable.explain_one(&sample);

    assert_eq!(path.contributions.len(), 2);
    assert!(path.probability.is_some());
}

#[test]
fn test_logistic_explainable_into_trait() {
    let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        .expect("Matrix creation");
    let y: &[usize] = &[0, 1, 1, 0];

    let mut model = LogisticRegression::new();
    model.fit(&x, y).expect("fit");

    let explainable = model.into_explainable();
    assert_eq!(explainable.n_features(), 2);
}

// =============================================================================
// TreeExplainable tests
// =============================================================================

#[test]
fn test_tree_explainable_new() {
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("Matrix creation");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);

    let mut model = DecisionTreeRegressor::new();
    model.fit(&x, &y).expect("fit");

    let explainable = TreeExplainable::new(model, 1);
    assert_eq!(explainable.n_features(), 1);
}

#[test]
fn test_tree_explainable_predict_explained() {
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("Matrix creation");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);

    let mut model = DecisionTreeRegressor::new();
    model.fit(&x, &y).expect("fit");

    let explainable = TreeExplainable::new(model, 1);

    let sample = vec![2.5];
    let (outputs, paths) = explainable.predict_explained(&sample, 1);

    assert_eq!(outputs.len(), 1);
    assert_eq!(paths.len(), 1);
}

#[test]
fn test_tree_explainable_explain_one() {
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("Matrix creation");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);

    let mut model = DecisionTreeRegressor::new();
    model.fit(&x, &y).expect("fit");

    let explainable = TreeExplainable::new(model, 1);
    let sample = vec![2.5];
    let path = explainable.explain_one(&sample);

    // Path should have leaf info
    assert!(path.leaf.n_samples > 0 || path.leaf.prediction != 0.0);
}

#[test]
fn test_tree_explainable_into_trait() {
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("Matrix creation");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);

    let mut model = DecisionTreeRegressor::new();
    model.fit(&x, &y).expect("fit");

    let explainable = model.into_explainable(1);
    assert_eq!(explainable.n_features(), 1);
}

// =============================================================================
// EnsembleExplainable tests
// =============================================================================

#[test]
fn test_ensemble_explainable_new() {
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
    assert_eq!(explainable.n_features(), 2);
}

#[test]
fn test_ensemble_explainable_predict_explained() {
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

    let sample = vec![5.0, 6.0];
    let (outputs, paths) = explainable.predict_explained(&sample, 1);

    assert_eq!(outputs.len(), 1);
    assert_eq!(paths.len(), 1);
    assert!(paths[0].n_trees() > 0);
}

#[test]
fn test_ensemble_explainable_explain_one() {
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
    let sample = vec![5.0, 6.0];
    let path = explainable.explain_one(&sample);

    assert!(path.n_trees() > 0);
}

#[test]
fn test_ensemble_explainable_feature_importances() {
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
    let importances = explainable.feature_importances();

    // Feature importances may or may not be available depending on implementation
    if let Some(imp) = importances {
        assert!(!imp.is_empty());
    }
}

#[test]
fn test_ensemble_explainable_into_trait() {
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

    let explainable = model.into_explainable(2);
    assert_eq!(explainable.n_features(), 2);
}

// =============================================================================
// Path trait tests
// =============================================================================

#[test]
fn test_linear_path_explain() {
    let x = Matrix::from_vec(5, 2, vec![1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 4.0, 5.0, 5.0, 4.0])
        .expect("Matrix creation");
    let y = Vector::from_slice(&[4.0, 8.0, 8.0, 14.0, 13.0]);

    let mut model = LinearRegression::new();
    model.fit(&x, &y).expect("fit");

    let explainable = LinearExplainable::new(model);
    let sample = vec![2.0, 3.0];
    let path = explainable.explain_one(&sample);

    // Test DecisionPath trait methods
    use entrenar::monitor::inference::DecisionPath;

    let explanation = path.explain();
    assert!(explanation.contains("Prediction"));

    let contributions = path.feature_contributions();
    assert_eq!(contributions.len(), 2);

    let confidence = path.confidence();
    assert!(confidence >= 0.0 && confidence <= 1.0);
}

#[test]
fn test_tree_path_explain() {
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("Matrix creation");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);

    let mut model = DecisionTreeRegressor::new();
    model.fit(&x, &y).expect("fit");

    let explainable = TreeExplainable::new(model, 1);
    let sample = vec![2.5];
    let path = explainable.explain_one(&sample);

    // Test DecisionPath trait methods
    use entrenar::monitor::inference::DecisionPath;

    let explanation = path.explain();
    assert!(explanation.contains("Decision Path") || explanation.contains("LEAF"));

    let confidence = path.confidence();
    assert!(confidence >= 0.0 && confidence <= 1.0);
}

#[test]
fn test_forest_path_explain() {
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
    let sample = vec![5.0, 6.0];
    let path = explainable.explain_one(&sample);

    // Test DecisionPath trait methods
    use entrenar::monitor::inference::DecisionPath;

    let explanation = path.explain();
    assert!(explanation.contains("Ensemble"));

    let confidence = path.confidence();
    assert!(confidence >= 0.0 && confidence <= 1.0);
}

#[path = "tests_part_02.rs"]

mod tests_part_02;
