//! Test helper functions for commutativity tests

use crate::autograd::Tensor;
use crate::merge::Model;
use std::collections::HashMap;

/// Create a model with single parameter
pub fn make_model(values: Vec<f32>) -> Model {
    let mut m = HashMap::new();
    m.insert("w".to_string(), Tensor::from_vec(values, false));
    m
}

/// Create a model with multiple parameters
pub fn make_multi_param_model(params: Vec<(&str, Vec<f32>)>) -> Model {
    let mut m = HashMap::new();
    for (name, values) in params {
        m.insert(name.to_string(), Tensor::from_vec(values, false));
    }
    m
}

/// Compare two models for approximate equality
pub fn models_approx_equal(m1: &Model, m2: &Model, tolerance: f32) -> bool {
    if m1.len() != m2.len() {
        return false;
    }
    m1.iter().all(|(name, t1)| tensors_approx_equal(t1, m2.get(name), tolerance))
}

/// Compare two tensors for approximate equality within tolerance.
fn tensors_approx_equal(t1: &Tensor, t2: Option<&Tensor>, tolerance: f32) -> bool {
    let Some(t2) = t2 else { return false };
    let (d1, d2) = (t1.data(), t2.data());
    d1.len() == d2.len() && d1.iter().zip(d2.iter()).all(|(a, b)| (a - b).abs() <= tolerance)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_model_creates_single_param() {
        let model = make_model(vec![1.0, 2.0, 3.0]);
        assert_eq!(model.len(), 1);
        assert!(model.contains_key("w"));
        assert_eq!(model.get("w").expect("key should exist").data().len(), 3);
    }

    #[test]
    fn test_make_model_empty() {
        let model = make_model(vec![]);
        assert_eq!(model.len(), 1);
        assert!(model.get("w").expect("key should exist").data().is_empty());
    }

    #[test]
    fn test_make_multi_param_model() {
        let model =
            make_multi_param_model(vec![("w1", vec![1.0, 2.0]), ("w2", vec![3.0, 4.0, 5.0])]);
        assert_eq!(model.len(), 2);
        assert_eq!(model.get("w1").expect("key should exist").data().len(), 2);
        assert_eq!(model.get("w2").expect("key should exist").data().len(), 3);
    }

    #[test]
    fn test_make_multi_param_model_empty() {
        let model = make_multi_param_model(vec![]);
        assert!(model.is_empty());
    }

    #[test]
    fn test_models_approx_equal_identical() {
        let m1 = make_model(vec![1.0, 2.0, 3.0]);
        let m2 = make_model(vec![1.0, 2.0, 3.0]);
        assert!(models_approx_equal(&m1, &m2, 1e-6));
    }

    #[test]
    fn test_models_approx_equal_within_tolerance() {
        let m1 = make_model(vec![1.0, 2.0, 3.0]);
        let m2 = make_model(vec![1.001, 2.001, 3.001]);
        assert!(models_approx_equal(&m1, &m2, 0.01));
        assert!(!models_approx_equal(&m1, &m2, 0.0001));
    }

    #[test]
    fn test_models_approx_equal_different_lengths() {
        let m1 = make_model(vec![1.0, 2.0, 3.0]);
        let m2 = make_multi_param_model(vec![("w", vec![1.0, 2.0, 3.0]), ("b", vec![0.0])]);
        assert!(!models_approx_equal(&m1, &m2, 1e-6));
    }

    #[test]
    fn test_models_approx_equal_missing_key() {
        let m1 = make_multi_param_model(vec![("w1", vec![1.0])]);
        let m2 = make_multi_param_model(vec![("w2", vec![1.0])]);
        assert!(!models_approx_equal(&m1, &m2, 1e-6));
    }

    #[test]
    fn test_models_approx_equal_different_tensor_lengths() {
        let m1 = make_model(vec![1.0, 2.0]);
        let m2 = make_model(vec![1.0, 2.0, 3.0]);
        assert!(!models_approx_equal(&m1, &m2, 1e-6));
    }

    #[test]
    fn test_models_approx_equal_both_empty() {
        let m1: Model = HashMap::new();
        let m2: Model = HashMap::new();
        assert!(models_approx_equal(&m1, &m2, 1e-6));
    }

    #[test]
    fn test_models_approx_equal_multi_param() {
        let m1 = make_multi_param_model(vec![("w1", vec![1.0, 2.0]), ("w2", vec![3.0, 4.0])]);
        let m2 = make_multi_param_model(vec![("w1", vec![1.0, 2.0]), ("w2", vec![3.0, 4.0])]);
        assert!(models_approx_equal(&m1, &m2, 1e-6));
    }
}
