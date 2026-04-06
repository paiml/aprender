//! Unit tests for autograd operations (forward and backward)

use super::test_utils::finite_difference;
use crate::autograd::{
    add, attention, backward, gelu, layer_norm, matmul, mul, relu, scale, softmax, sum, swish,
    Tensor,
};
use approx::assert_abs_diff_eq;

#[test]
fn test_tensor_creation() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
    assert_eq!(t.len(), 3);
    assert!(t.requires_grad());
    assert!(t.grad().is_none());
}

#[test]
fn test_tensor_grad_accumulation() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);

    t.accumulate_grad(ndarray::arr1(&[1.0, 1.0, 1.0]));
    let grad1 = t.grad().expect("gradient should be available");
    assert_eq!(grad1[0], 1.0);

    t.accumulate_grad(ndarray::arr1(&[1.0, 1.0, 1.0]));
    let grad2 = t.grad().expect("gradient should be available");
    assert_eq!(grad2[0], 2.0);
}

#[test]
fn test_add_forward() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
    let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], true);
    let c = add(&a, &b);

    assert_abs_diff_eq!(c.data()[0], 5.0);
    assert_abs_diff_eq!(c.data()[1], 7.0);
    assert_abs_diff_eq!(c.data()[2], 9.0);
}

#[test]
fn test_add_backward() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
    let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], true);
    let mut c = add(&a, &b);

    // Backward with gradient of ones
    backward(&mut c, Some(ndarray::arr1(&[1.0, 1.0, 1.0])));

    let grad_a = a.grad().expect("gradient should be available");
    let grad_b = b.grad().expect("gradient should be available");

    assert_abs_diff_eq!(grad_a[0], 1.0);
    assert_abs_diff_eq!(grad_b[0], 1.0);
}

#[test]
fn test_mul_forward() {
    let a = Tensor::from_vec(vec![2.0, 3.0, 4.0], true);
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0], true);
    let c = mul(&a, &b);

    assert_abs_diff_eq!(c.data()[0], 10.0);
    assert_abs_diff_eq!(c.data()[1], 18.0);
    assert_abs_diff_eq!(c.data()[2], 28.0);
}

#[test]
fn test_mul_backward() {
    let a = Tensor::from_vec(vec![2.0, 3.0], true);
    let b = Tensor::from_vec(vec![5.0, 7.0], true);
    let mut c = mul(&a, &b);

    backward(&mut c, Some(ndarray::arr1(&[1.0, 1.0])));

    let grad_a = a.grad().expect("gradient should be available");
    let grad_b = b.grad().expect("gradient should be available");

    // d(a*b)/da = b
    assert_abs_diff_eq!(grad_a[0], 5.0);
    assert_abs_diff_eq!(grad_a[1], 7.0);

    // d(a*b)/db = a
    assert_abs_diff_eq!(grad_b[0], 2.0);
    assert_abs_diff_eq!(grad_b[1], 3.0);
}

#[test]
fn test_relu_forward() {
    let a = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], true);
    let c = relu(&a);

    assert_abs_diff_eq!(c.data()[0], 0.0);
    assert_abs_diff_eq!(c.data()[1], 0.0);
    assert_abs_diff_eq!(c.data()[2], 1.0);
    assert_abs_diff_eq!(c.data()[3], 2.0);
}

#[test]
fn test_relu_backward() {
    let a = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], true);
    let mut c = relu(&a);

    backward(&mut c, Some(ndarray::arr1(&[1.0, 1.0, 1.0, 1.0])));

    let grad_a = a.grad().expect("gradient should be available");

    // Gradient is 0 for negative inputs, 1 for positive
    assert_abs_diff_eq!(grad_a[0], 0.0);
    assert_abs_diff_eq!(grad_a[1], 0.0);
    assert_abs_diff_eq!(grad_a[2], 1.0);
    assert_abs_diff_eq!(grad_a[3], 1.0);
}

#[test]
fn test_scale_forward() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
    let c = scale(&a, 2.5);

    assert_abs_diff_eq!(c.data()[0], 2.5);
    assert_abs_diff_eq!(c.data()[1], 5.0);
    assert_abs_diff_eq!(c.data()[2], 7.5);
}

#[test]
fn test_scale_backward() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
    let mut c = scale(&a, 3.0);

    backward(&mut c, Some(ndarray::arr1(&[1.0, 1.0, 1.0])));

    let grad_a = a.grad().expect("gradient should be available");

    // Gradient of scale is just the factor
    assert_abs_diff_eq!(grad_a[0], 3.0);
    assert_abs_diff_eq!(grad_a[1], 3.0);
    assert_abs_diff_eq!(grad_a[2], 3.0);
}

#[test]
fn test_scale_zero_factor() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
    let c = scale(&a, 0.0);

    // Scaling by zero should give zeros
    assert_abs_diff_eq!(c.data()[0], 0.0);
    assert_abs_diff_eq!(c.data()[1], 0.0);
    assert_abs_diff_eq!(c.data()[2], 0.0);
}

#[test]
fn test_scale_negative_factor() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
    let c = scale(&a, -2.0);

    assert_abs_diff_eq!(c.data()[0], -2.0);
    assert_abs_diff_eq!(c.data()[1], -4.0);
    assert_abs_diff_eq!(c.data()[2], -6.0);
}

#[test]
fn test_add_no_grad() {
    let a = Tensor::from_vec(vec![1.0, 2.0], false);
    let b = Tensor::from_vec(vec![3.0, 4.0], false);
    let c = add(&a, &b);

    assert!(!c.requires_grad());
}

#[test]
fn test_mul_no_grad() {
    let a = Tensor::from_vec(vec![1.0, 2.0], false);
    let b = Tensor::from_vec(vec![3.0, 4.0], false);
    let c = mul(&a, &b);

    assert!(!c.requires_grad());
}

#[test]
fn test_scale_no_grad() {
    let a = Tensor::from_vec(vec![1.0, 2.0], false);
    let c = scale(&a, 2.0);

    assert!(!c.requires_grad());
}

#[test]
fn test_relu_no_grad() {
    let a = Tensor::from_vec(vec![-1.0, 1.0], false);
    let c = relu(&a);

    assert!(!c.requires_grad());
}

#[test]
fn test_gelu_forward() {
    let a = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], true);
    let c = gelu(&a);

    // GELU is smooth, non-linear activation
    // GELU(0) = 0
    assert_abs_diff_eq!(c.data()[2], 0.0, epsilon = 1e-5);

    // GELU is approximately linear for positive values
    // GELU(x) approx x for large positive x
    assert!(c.data()[4] > 1.5); // GELU(2) should be close to 2
}

#[test]
fn test_gelu_backward() {
    let a = Tensor::from_vec(vec![-1.0, 0.0, 1.0], true);
    let mut c = gelu(&a);

    backward(&mut c, Some(ndarray::arr1(&[1.0, 1.0, 1.0])));

    let grad_a = a.grad().expect("gradient should be available");

    // Gradients should exist
    assert_eq!(grad_a.len(), 3);
    // GELU gradient at 0 is 0.5
    assert_abs_diff_eq!(grad_a[1], 0.5, epsilon = 1e-2);
}

#[test]
fn test_swish_forward() {
    let a = Tensor::from_vec(vec![-2.0, 0.0, 2.0], true);
    let c = swish(&a);

    // Swish(0) = 0
    assert_abs_diff_eq!(c.data()[1], 0.0, epsilon = 1e-5);

    // Swish is approximately linear for large positive x
    assert!(c.data()[2] > 1.5);
}

#[test]
fn test_swish_backward() {
    let a = Tensor::from_vec(vec![-1.0, 0.0, 1.0], true);
    let mut c = swish(&a);

    backward(&mut c, Some(ndarray::arr1(&[1.0, 1.0, 1.0])));

    let grad_a = a.grad().expect("gradient should be available");

    // Gradients should exist
    assert_eq!(grad_a.len(), 3);
    // Swish gradient at 0 is 0.5
    assert_abs_diff_eq!(grad_a[1], 0.5, epsilon = 1e-2);
}

#[test]
fn test_layer_norm_forward() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], true);
    let gamma = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], true);
    let beta = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], true);
    let c = layer_norm(&a, &gamma, &beta, 1e-5);

    // LayerNorm should have mean approx 0 and std approx 1
    let mean: f32 = c.data().iter().sum::<f32>() / c.len() as f32;
    assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-5);

    // Check variance approx 1
    let var: f32 = c.data().iter().map(|&x| x * x).sum::<f32>() / c.len() as f32;
    assert_abs_diff_eq!(var, 1.0, epsilon = 1e-5);
}

#[test]
fn test_layer_norm_backward() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], true);
    let gamma = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], true);
    let beta = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], true);
    let mut c = layer_norm(&a, &gamma, &beta, 1e-5);

    backward(&mut c, Some(ndarray::arr1(&[1.0, 1.0, 1.0, 1.0])));

    // Gradients should exist for all inputs
    let grad_a = a.grad().expect("gradient should be available");
    let grad_gamma = gamma.grad().expect("gradient should be available");
    let grad_beta = beta.grad().expect("gradient should be available");

    assert_eq!(grad_a.len(), 4);
    assert_eq!(grad_gamma.len(), 4);
    assert_eq!(grad_beta.len(), 4);

    // Gradient of beta should be the upstream gradient
    for i in 0..4 {
        assert_abs_diff_eq!(grad_beta[i], 1.0, epsilon = 1e-5);
    }
}

#[test]
fn test_attention_forward() {
    // Simple 2x2 attention example
    // Q, K, V are all 2x2 matrices
    let q = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], true); // 2x2 identity
    let k = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], true); // 2x2 identity
    let v = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], true); // 2x2 values

    let output = attention(&q, &k, &v, 2, 2, 2, 2);

    // Output should have shape seq_len x d_v = 2x2
    assert_eq!(output.len(), 4);
}

#[test]
fn test_attention_backward() {
    // Simple attention backward test
    let q = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], true);
    let k = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], true);
    let v = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], true);

    let mut output = attention(&q, &k, &v, 2, 2, 2, 2);

    // Backward with all ones gradient
    backward(&mut output, Some(ndarray::arr1(&[1.0, 1.0, 1.0, 1.0])));

    // All three should have gradients
    assert!(q.grad().is_some());
    assert!(k.grad().is_some());
    assert!(v.grad().is_some());

    let grad_q = q.grad().expect("gradient should be available");
    let grad_k = k.grad().expect("gradient should be available");
    let grad_v = v.grad().expect("gradient should be available");

    assert_eq!(grad_q.len(), 4);
    assert_eq!(grad_k.len(), 4);
    assert_eq!(grad_v.len(), 4);
}

#[test]
fn test_softmax_forward() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
    let c = softmax(&a);

    // Softmax should sum to 1
    let sum: f32 = c.data().iter().sum();
    assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);

    // Largest input should have largest output
    assert!(c.data()[2] > c.data()[1]);
    assert!(c.data()[1] > c.data()[0]);
}

#[test]
fn test_softmax_backward_gradient_check() {
    let x_vec = vec![1.0, 2.0, 3.0, 4.0];
    let a = Tensor::from_vec(x_vec.clone(), true);
    let mut y = softmax(&a);

    // Compute analytical gradient
    backward(&mut y, Some(ndarray::arr1(&[1.0, 0.0, 0.0, 0.0])));
    let analytical = a.grad().expect("gradient should be available");

    // Compute numerical gradient
    let numerical = finite_difference(
        |x| {
            let t = Tensor::from_vec(x.to_vec(), false);
            let s = softmax(&t);
            s.data()[0]
        },
        &x_vec,
        1e-4,
    );

    // Compare
    for i in 0..x_vec.len() {
        assert_abs_diff_eq!(analytical[i], numerical[i], epsilon = 1e-3);
    }
}

#[test]
fn test_sum_backward() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
    let mut c = sum(&a);

    backward(&mut c, Some(ndarray::arr1(&[1.0])));

    let grad_a = a.grad().expect("gradient should be available");

    // Sum gradient broadcasts to all inputs
    assert_abs_diff_eq!(grad_a[0], 1.0);
    assert_abs_diff_eq!(grad_a[1], 1.0);
    assert_abs_diff_eq!(grad_a[2], 1.0);
}

#[test]
fn test_chain_rule() {
    // Test: f(x) = sum(relu(x * 2))
    let a = Tensor::from_vec(vec![-1.0, 1.0, 2.0], true);
    let b = scale(&a, 2.0);
    let c = relu(&b);
    let mut d = sum(&c);

    backward(&mut d, None);

    let grad_a = a.grad().expect("gradient should be available");

    // For x = -1: relu(-2) = 0, grad = 0
    assert_abs_diff_eq!(grad_a[0], 0.0);

    // For x = 1: relu(2) = 2, grad = 2
    assert_abs_diff_eq!(grad_a[1], 2.0);

    // For x = 2: relu(4) = 4, grad = 2
    assert_abs_diff_eq!(grad_a[2], 2.0);
}

#[test]
fn test_matmul_forward() {
    // Matrix A: 2x3 (flattened)
    // [1, 2, 3]
    // [4, 5, 6]
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], true);

    // Matrix B: 3x2 (flattened)
    // [7,  8]
    // [9, 10]
    // [11, 12]
    let b = Tensor::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], true);

    // Expected: 2x2
    // [1*7+2*9+3*11,  1*8+2*10+3*12]   = [58,  64]
    // [4*7+5*9+6*11,  4*8+5*10+6*12]   = [139, 154]
    let c = matmul(&a, &b, 2, 3, 2);

    assert_eq!(c.len(), 4);
    assert_abs_diff_eq!(c.data()[0], 58.0);
    assert_abs_diff_eq!(c.data()[1], 64.0);
    assert_abs_diff_eq!(c.data()[2], 139.0);
    assert_abs_diff_eq!(c.data()[3], 154.0);
}

#[test]
fn test_matmul_backward() {
    // Simple 2x2 @ 2x2
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], true);
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], true);
    let mut c = matmul(&a, &b, 2, 2, 2);

    backward(&mut c, Some(ndarray::arr1(&[1.0, 1.0, 1.0, 1.0])));

    let grad_a = a.grad().expect("gradient should be available");
    let grad_b = b.grad().expect("gradient should be available");

    // dL/dA = dL/dC @ B^T
    // dL/dB = A^T @ dL/dC
    // Gradients should exist
    assert_eq!(grad_a.len(), 4);
    assert_eq!(grad_b.len(), 4);
}
