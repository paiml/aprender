//! Common test utilities for ensemble merging tests

use crate::autograd::Tensor;
use crate::merge::ensemble::Model;
use std::collections::HashMap;

pub fn make_model(values: Vec<f32>) -> Model {
    let mut m = HashMap::new();
    m.insert("w".to_string(), Tensor::from_vec(values, false));
    m
}

pub fn models_approx_equal(m1: &Model, m2: &Model, tol: f32) -> bool {
    for (name, t1) in m1 {
        if let Some(t2) = m2.get(name) {
            for (a, b) in t1.data().iter().zip(t2.data().iter()) {
                if (a - b).abs() > tol {
                    return false;
                }
            }
        } else {
            return false;
        }
    }
    true
}
