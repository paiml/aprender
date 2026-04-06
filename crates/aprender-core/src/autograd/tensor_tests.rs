pub(crate) use super::*;

#[test]
fn test_tensor_creation() {
    let t = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(t.shape(), &[2, 2]);
    assert_eq!(t.numel(), 4);
    assert_eq!(t.ndim(), 2);
}

#[test]
fn test_tensor_from_slice() {
    let t = Tensor::from_slice(&[1.0, 2.0, 3.0]);
    assert_eq!(t.shape(), &[3]);
    assert_eq!(t.numel(), 3);
}

#[test]
fn test_tensor_zeros_ones() {
    let z = Tensor::zeros(&[2, 3]);
    assert!(z.data().iter().all(|&x| x == 0.0));

    let o = Tensor::ones(&[2, 3]);
    assert!(o.data().iter().all(|&x| x == 1.0));
}

#[test]
fn test_requires_grad() {
    let t = Tensor::from_slice(&[1.0, 2.0]).requires_grad();
    assert!(t.requires_grad_enabled());

    let t2 = Tensor::from_slice(&[1.0, 2.0]);
    assert!(!t2.requires_grad_enabled());
}

#[test]
fn test_detach() {
    let t = Tensor::from_slice(&[1.0, 2.0]).requires_grad();
    let d = t.detach();

    assert!(t.requires_grad_enabled());
    assert!(!d.requires_grad_enabled());
    assert!(d.is_leaf());
}

#[test]
fn test_item() {
    let t = Tensor::new(&[42.0], &[1]);
    assert_eq!(t.item(), 42.0);

    let t2 = Tensor::new(&[42.0], &[]);
    assert_eq!(t2.item(), 42.0);
}

#[test]
#[should_panic(expected = "item() only works on tensors with exactly 1 element")]
fn test_item_panics_multi_element() {
    let t = Tensor::from_slice(&[1.0, 2.0]);
    let _ = t.item();
}

#[test]
fn test_tensor_id_unique() {
    let t1 = Tensor::from_slice(&[1.0]);
    let t2 = Tensor::from_slice(&[1.0]);
    assert_ne!(t1.id(), t2.id());
}

#[test]
fn test_gradient_accumulation() {
    let mut t = Tensor::from_slice(&[1.0, 2.0, 3.0]).requires_grad();

    t.accumulate_grad(Tensor::from_slice(&[0.1, 0.2, 0.3]));

    let grad1 = t
        .grad()
        .expect("grad should exist after accumulate")
        .data()
        .to_vec();
    assert_eq!(grad1, vec![0.1, 0.2, 0.3]);

    t.accumulate_grad(Tensor::from_slice(&[0.1, 0.2, 0.3]));
    let grad2 = t
        .grad()
        .expect("grad should exist after second accumulate")
        .data()
        .to_vec();
    assert_eq!(grad2, vec![0.2, 0.4, 0.6]);
}

// ========================================================================
// Additional Coverage Tests for tensor.rs
// ========================================================================

#[test]
fn test_tensor_id_default() {
    let id1 = TensorId::default();
    let id2 = TensorId::default();
    assert_ne!(id1, id2); // Each default generates unique ID
}

#[test]
fn test_tensor_id_clone() {
    let id1 = TensorId::new();
    let id2 = id1; // Copy
    assert_eq!(id1, id2);
}

#[test]
fn test_tensor_id_debug() {
    let id = TensorId::new();
    let debug_str = format!("{:?}", id);
    assert!(debug_str.contains("TensorId"));
}

#[test]
fn test_tensor_zeros_like() {
    let original = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let zeros = Tensor::zeros_like(&original);
    assert_eq!(zeros.shape(), &[2, 2]);
    assert!(zeros.data().iter().all(|&x| x == 0.0));
}

#[test]
fn test_tensor_ones_like() {
    let original = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let ones = Tensor::ones_like(&original);
    assert_eq!(ones.shape(), &[3]);
    assert!(ones.data().iter().all(|&x| x == 1.0));
}

#[test]
fn test_tensor_requires_grad_mut() {
    let mut t = Tensor::from_slice(&[1.0, 2.0]);
    assert!(!t.requires_grad_enabled());

    t.requires_grad_(true);
    assert!(t.requires_grad_enabled());

    t.requires_grad_(false);
    assert!(!t.requires_grad_enabled());
}

#[test]
fn test_tensor_data_mut() {
    let mut t = Tensor::from_slice(&[1.0, 2.0, 3.0]);
    {
        let data = t.data_mut();
        data[0] = 10.0;
        data[1] = 20.0;
    }
    assert_eq!(t.data()[0], 10.0);
    assert_eq!(t.data()[1], 20.0);
}

#[test]
fn test_tensor_zero_grad_() {
    let mut t = Tensor::from_slice(&[1.0, 2.0, 3.0]).requires_grad();
    t.accumulate_grad(Tensor::from_slice(&[0.1, 0.2, 0.3]));
    assert!(t.grad().is_some());

    t.zero_grad_();
    assert!(t.grad().is_none());
}

#[test]
fn test_tensor_grad_none_initially() {
    let t = Tensor::from_slice(&[1.0, 2.0, 3.0]);
    assert!(t.grad().is_none());
}

#[test]
fn test_tensor_item_scalar() {
    let t = Tensor::new(&[3.14], &[]); // Scalar (0-dimensional)
    assert!((t.item() - 3.14).abs() < 1e-6);
}

#[test]
fn test_tensor_clone() {
    let t1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).requires_grad();
    let t2 = t1.clone();

    assert_eq!(t1.shape(), t2.shape());
    assert_eq!(t1.data(), t2.data());
    assert_eq!(t1.requires_grad_enabled(), t2.requires_grad_enabled());
}

#[test]
fn test_tensor_debug() {
    let t = Tensor::new(&[1.0, 2.0], &[2]);
    let debug_str = format!("{:?}", t);
    assert!(debug_str.contains("Tensor"));
}

#[test]
fn test_tensor_with_grad_fn() {
    // Create tensor without grad_fn (leaf)
    let t = Tensor::from_slice(&[1.0, 2.0]).requires_grad();
    assert!(t.is_leaf());
    assert!(t.grad_fn().is_none());
}

#[test]
fn test_tensor_backward_scalar() {
    // Scalar tensor required for backward()
    let t = Tensor::new(&[5.0], &[]).requires_grad();
    // backward on scalar tensor should not panic
    t.backward();
}

#[test]
fn test_tensor_hash() {
    use std::collections::HashSet;
    let t1 = TensorId::new();
    let t2 = TensorId::new();

    let mut set = HashSet::new();
    set.insert(t1);
    set.insert(t2);
    assert_eq!(set.len(), 2);
}

#[test]
fn test_tensor_empty_shape() {
    // Scalar tensor (empty shape = 0-d tensor)
    let t = Tensor::new(&[42.0], &[]);
    assert_eq!(t.ndim(), 0);
    assert_eq!(t.numel(), 1);
    assert_eq!(t.item(), 42.0);
}
