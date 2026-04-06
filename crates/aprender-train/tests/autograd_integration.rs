//! Integration tests for autograd core functionality.
//!
//! Tests the tape-based automatic differentiation engine across
//! various tensor operations.

use entrenar::autograd::{backward, Context, Tensor};

#[test]
fn test_tensor_creation_from_vec() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
    assert_eq!(t.data().len(), 3);
    assert!((t.data()[0] - 1.0).abs() < 1e-6);
    assert!((t.data()[1] - 2.0).abs() < 1e-6);
    assert!((t.data()[2] - 3.0).abs() < 1e-6);
}

#[test]
fn test_tensor_zeros() {
    let t = Tensor::zeros(5, true);
    assert_eq!(t.data().len(), 5);
    for i in 0..5 {
        assert!((t.data()[i]).abs() < 1e-6);
    }
}

#[test]
fn test_tensor_ones() {
    let t = Tensor::ones(4, true);
    assert_eq!(t.data().len(), 4);
    for i in 0..4 {
        assert!((t.data()[i] - 1.0).abs() < 1e-6);
    }
}

#[test]
fn test_tensor_requires_grad() {
    let t_grad = Tensor::from_vec(vec![1.0], true);
    assert!(t_grad.requires_grad());

    let t_no_grad = Tensor::from_vec(vec![1.0], false);
    assert!(!t_no_grad.requires_grad());
}

#[test]
fn test_tensor_zero_grad() {
    let t = Tensor::from_vec(vec![2.0], true);

    // Set a grad
    t.set_grad(ndarray::Array1::from(vec![5.0]));
    assert!(t.grad().is_some());

    // Zero grad
    t.zero_grad();
    assert!(t.grad().is_none());
}

#[test]
fn test_tensor_grad_accumulation() {
    let t = Tensor::from_vec(vec![1.0, 2.0], true);

    // First grad
    t.accumulate_grad(ndarray::Array1::from(vec![1.0, 1.0]));

    // Second grad
    t.accumulate_grad(ndarray::Array1::from(vec![2.0, 3.0]));

    let grad = t.grad().expect("should have grad");
    assert!((grad[0] - 3.0).abs() < 1e-6);
    assert!((grad[1] - 4.0).abs() < 1e-6);
}

#[test]
fn test_backward_initializes_grad() {
    let mut t = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
    backward(&mut t, None);

    let grad = t.grad().expect("should have grad after backward");
    // Should be initialized to ones
    assert_eq!(grad.len(), 3);
    for i in 0..3 {
        assert!((grad[i] - 1.0).abs() < 1e-6);
    }
}

#[test]
fn test_backward_with_custom_grad() {
    let mut t = Tensor::from_vec(vec![1.0, 2.0], true);
    let custom_grad = ndarray::Array1::from(vec![0.5, 0.5]);
    backward(&mut t, Some(custom_grad));

    let grad = t.grad().expect("should have grad");
    assert!((grad[0] - 0.5).abs() < 1e-6);
    assert!((grad[1] - 0.5).abs() < 1e-6);
}

#[test]
fn test_context_training_mode() {
    let ctx = Context::new();
    assert!(ctx.is_training());
}

#[test]
fn test_tensor_len() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], true);
    assert_eq!(t.len(), 5);
    assert!(!t.is_empty());

    let empty = Tensor::zeros(0, true);
    assert!(empty.is_empty());
}
