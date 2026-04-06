#![allow(clippy::disallowed_methods)]
//! Property-based tests using proptest.
//!
//! These tests verify invariants and properties of the ML algorithms.

use aprender::model_selection::{train_test_split, KFold};
use aprender::prelude::*;
use proptest::prelude::*;

// Strategy for generating small matrices
fn matrix_strategy(rows: usize, cols: usize) -> impl Strategy<Value = Matrix<f32>> {
    proptest::collection::vec(-100.0f32..100.0, rows * cols).prop_map(move |data| {
        Matrix::from_vec(rows, cols, data).expect("Test data should be valid")
    })
}

// Strategy for generating vectors
fn vector_strategy(len: usize) -> impl Strategy<Value = Vector<f32>> {
    proptest::collection::vec(-100.0f32..100.0, len).prop_map(Vector::from_vec)
}

// Strategy for generating f64 vectors (for NLP similarity functions)
fn vector_f64_strategy(len: usize) -> impl Strategy<Value = Vector<f64>> {
    proptest::collection::vec(-100.0f64..100.0, len).prop_map(Vector::from_vec)
}

include!("includes/property_tests_vector.rs");
include!("includes/property_tests_logistic.rs");
include!("includes/property_tests_nlp_summarization.rs");
include!("includes/property_tests_noise_audio.rs");
