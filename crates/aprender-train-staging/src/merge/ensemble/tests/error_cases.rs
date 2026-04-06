//! Tests for error cases in ensemble merging

use super::common::make_model;
use crate::autograd::Tensor;
use crate::merge::ensemble::{ensemble_merge, EnsembleConfig, MergeError};
use std::collections::HashMap;

#[test]
fn test_insufficient_models() {
    let m = make_model(vec![1.0]);
    let config = EnsembleConfig::uniform_average();

    let result = ensemble_merge(&[m], &config);
    assert!(matches!(result, Err(MergeError::InsufficientModels { min: 2, got: 1 })));
}

#[test]
fn test_weighted_average_zero_sum() {
    let m1 = make_model(vec![1.0, 2.0]);
    let m2 = make_model(vec![3.0, 4.0]);

    let config = EnsembleConfig::weighted_average(vec![0.0, 0.0]);
    let result = ensemble_merge(&[m1, m2], &config);

    assert!(matches!(result, Err(MergeError::InvalidConfig(_))));
}

#[test]
fn test_weighted_average_negative_sum() {
    let m1 = make_model(vec![1.0, 2.0]);
    let m2 = make_model(vec![3.0, 4.0]);

    let config = EnsembleConfig::weighted_average(vec![-1.0, 0.5]);
    let result = ensemble_merge(&[m1, m2], &config);

    assert!(matches!(result, Err(MergeError::InvalidConfig(_))));
}

#[test]
fn test_weighted_average_missing_param() {
    // Model 1 has param "w", model 2 has param "x"
    let mut m1 = HashMap::new();
    m1.insert("w".to_string(), Tensor::from_vec(vec![1.0], false));

    let mut m2 = HashMap::new();
    m2.insert("x".to_string(), Tensor::from_vec(vec![2.0], false));

    let config = EnsembleConfig::uniform_average();
    let result = ensemble_merge(&[m1, m2], &config);

    assert!(matches!(result, Err(MergeError::IncompatibleArchitectures(_))));
}

#[test]
fn test_weighted_average_shape_mismatch() {
    let m1 = make_model(vec![1.0, 2.0, 3.0]);
    let m2 = make_model(vec![4.0, 5.0]); // Different length

    let config = EnsembleConfig::uniform_average();
    let result = ensemble_merge(&[m1, m2], &config);

    assert!(matches!(result, Err(MergeError::ShapeMismatch(_))));
}
