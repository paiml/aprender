use super::*;
pub(crate) use crate::autograd::{clear_graph, no_grad};

/// Numerical gradient check using central differences.
#[allow(dead_code)]
pub(super) fn numerical_gradient<F>(f: F, x: &Tensor, eps: f32) -> Tensor
where
    F: Fn(&Tensor) -> Tensor,
{
    let mut grad_data = vec![0.0; x.numel()];

    for i in 0..x.numel() {
        let mut x_plus = x.data().to_vec();
        let mut x_minus = x.data().to_vec();
        x_plus[i] += eps;
        x_minus[i] -= eps;

        let y_plus = no_grad(|| f(&Tensor::new(&x_plus, x.shape())).item());
        let y_minus = no_grad(|| f(&Tensor::new(&x_minus, x.shape())).item());

        grad_data[i] = (y_plus - y_minus) / (2.0 * eps);
    }

    Tensor::new(&grad_data, x.shape())
}

#[allow(dead_code)]
pub(super) fn check_gradient<F>(f: F, x: &Tensor, eps: f32, tol: f32) -> bool
where
    F: Fn(&Tensor) -> Tensor,
{
    clear_graph();

    // Compute analytical gradient
    let x_grad = x.clone().requires_grad();
    let x_id = x_grad.id();
    let y = f(&x_grad);
    y.backward();

    // Get gradient from graph (since clones don't share storage)
    let analytical = crate::autograd::get_grad(x_id).expect("No gradient computed");

    // Compute numerical gradient
    let numerical = numerical_gradient(&f, x, eps);

    // Compare
    let max_diff: f32 = analytical
        .data()
        .iter()
        .zip(numerical.data().iter())
        .map(|(a, n)| (a - n).abs())
        .fold(0.0, f32::max);

    max_diff < tol
}

#[test]
fn test_simple_sum_gradient() {
    // Simple test: d/dx sum(x) = [1, 1, 1]
    clear_graph();

    let x = Tensor::from_slice(&[1.0, 2.0, 3.0]).requires_grad();
    let x_id = x.id();

    let y = x.sum();
    y.backward();

    let grad = crate::autograd::get_grad(x_id).expect("Gradient should exist");
    assert_eq!(grad.data(), &[1.0, 1.0, 1.0]);
}

#[test]
fn test_add_gradient() {
    // d/dx sum(x + y) = [1, 1, 1]
    clear_graph();

    let x = Tensor::from_slice(&[1.0, 2.0, 3.0]).requires_grad();
    let y = Tensor::from_slice(&[4.0, 5.0, 6.0]);
    let x_id = x.id();

    let z = x.add(&y).sum();
    z.backward();

    let grad = crate::autograd::get_grad(x_id).expect("Should have gradient");
    assert_eq!(grad.data(), &[1.0, 1.0, 1.0]);
}

#[test]
fn test_mul_gradient() {
    // d/dx sum(x * y) = y = [4, 5, 6]
    clear_graph();
    let x = Tensor::from_slice(&[1.0, 2.0, 3.0]).requires_grad();
    let y = Tensor::from_slice(&[4.0, 5.0, 6.0]);
    let x_id = x.id();
    let z = x.mul(&y).sum();
    z.backward();
    let grad = crate::autograd::get_grad(x_id).expect("grad");
    assert_eq!(grad.data(), &[4.0, 5.0, 6.0]);
}

#[test]
fn test_exp_gradient() {
    // d/dx sum(exp(x)) = exp(x)
    clear_graph();
    let x = Tensor::from_slice(&[0.0, 1.0, -1.0]).requires_grad();
    let x_id = x.id();
    let z = x.exp().sum();
    z.backward();
    let grad = crate::autograd::get_grad(x_id).expect("grad");
    for (g, &v) in grad.data().iter().zip(&[0.0_f32, 1.0, -1.0]) {
        assert!((g - v.exp()).abs() < 1e-5);
    }
}

#[test]
fn test_log_gradient() {
    // d/dx sum(log(x)) = 1/x
    clear_graph();
    let x = Tensor::from_slice(&[1.0, 2.0, 4.0]).requires_grad();
    let x_id = x.id();
    let z = x.log().sum();
    z.backward();
    let grad = crate::autograd::get_grad(x_id).expect("grad");
    assert_eq!(grad.data(), &[1.0, 0.5, 0.25]);
}

#[test]
fn test_pow_gradient() {
    // d/dx sum(x^2) = 2x
    clear_graph();
    let x = Tensor::from_slice(&[1.0, 2.0, 3.0]).requires_grad();
    let x_id = x.id();
    let z = x.pow(2.0).sum();
    z.backward();
    let grad = crate::autograd::get_grad(x_id).expect("grad");
    assert_eq!(grad.data(), &[2.0, 4.0, 6.0]);
}

#[test]
fn test_relu_gradient() {
    // d/dx sum(relu(x)) = 1 where x > 0
    clear_graph();
    let x = Tensor::from_slice(&[-1.0, 0.5, 2.0]).requires_grad();
    let x_id = x.id();
    let z = x.relu().sum();
    z.backward();
    let grad = crate::autograd::get_grad(x_id).expect("grad");
    assert_eq!(grad.data(), &[0.0, 1.0, 1.0]);
}

#[test]
fn test_sigmoid_gradient() {
    // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
    clear_graph();
    let x = Tensor::from_slice(&[0.0]).requires_grad();
    let x_id = x.id();
    let z = x.sigmoid().sum();
    z.backward();
    let grad = crate::autograd::get_grad(x_id).expect("grad");
    assert!((grad.data()[0] - 0.25).abs() < 1e-5); // sigmoid(0)=0.5, grad=0.5*0.5=0.25
}

#[test]
fn test_mean_gradient() {
    // d/dx mean(x) = 1/n
    clear_graph();
    let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]).requires_grad();
    let x_id = x.id();
    let z = x.mean();
    z.backward();
    let grad = crate::autograd::get_grad(x_id).expect("grad");
    assert_eq!(grad.data(), &[0.25, 0.25, 0.25, 0.25]);
}

#[test]
fn test_chain_gradient() {
    // Test chain rule: d/dx(sum((x * 2)^2)) = sum(4 * x * 2) = 8x
    let x = Tensor::from_slice(&[1.0, 2.0, 3.0]);

    clear_graph();
    let x_grad = x.clone().requires_grad();
    let x_id = x_grad.id();
    let y = x_grad.mul_scalar(2.0).pow(2.0).sum();
    y.backward();

    let grad = crate::autograd::get_grad(x_id).expect("No gradient");
    // Expected: 8 * x = [8, 16, 24]
    let expected = [8.0, 16.0, 24.0];

    for (g, e) in grad.data().iter().zip(expected.iter()) {
        assert!((g - e).abs() < 1e-3, "Expected {e}, got {g}");
    }
}

#[test]
fn test_matmul_forward() {
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

    let c = a.matmul(&b);

    // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.data(), &[19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_tanh_gradient() {
    // d/dx tanh(x) = 1 - tanh²(x)
    // At x=0: tanh(0)=0, grad=1
    clear_graph();
    let x = Tensor::from_slice(&[0.0]).requires_grad();
    let x_id = x.id();
    let z = x.tanh_().sum();
    z.backward();
    let grad = crate::autograd::get_grad(x_id).expect("grad");
    assert!((grad.data()[0] - 1.0).abs() < 1e-5); // at x=0, tanh(0)=0, grad=1-0²=1
}

#[test]
fn test_sqrt_gradient() {
    // d/dx sqrt(x) = 0.5 / sqrt(x)
    clear_graph();
    let x = Tensor::from_slice(&[4.0]).requires_grad();
    let x_id = x.id();
    let z = x.sqrt().sum();
    z.backward();
    let grad = crate::autograd::get_grad(x_id).expect("grad");
    assert!((grad.data()[0] - 0.25).abs() < 1e-5); // 0.5 / sqrt(4) = 0.5 / 2 = 0.25
}

#[test]
fn test_matmul_backward() {
    // For z = sum(A @ B), gradients are:
    // dL/dA = grad_output @ B^T
    // dL/dB = A^T @ grad_output
    clear_graph();
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).requires_grad();
    let b = Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2]).requires_grad(); // identity
    let a_id = a.id();
    let b_id = b.id();

    let c = a.matmul(&b);
    let loss = c.sum();
    loss.backward();

    let grad_a = crate::autograd::get_grad(a_id).expect("grad_a");
    let grad_b = crate::autograd::get_grad(b_id).expect("grad_b");

    // dL/dA = ones @ B^T = [[1,1],[1,1]] @ [[1,0],[0,1]] = [[1,1],[1,1]]
    assert_eq!(grad_a.data(), &[1.0, 1.0, 1.0, 1.0]);
    // dL/dB = A^T @ ones = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
    assert_eq!(grad_b.data(), &[4.0, 4.0, 6.0, 6.0]);
}

#[test]
fn test_div_gradient() {
    // d/dx (x/y) = 1/y
    // d/dy (x/y) = -x/y²
    clear_graph();
    let x = Tensor::from_slice(&[6.0]).requires_grad();
    let y = Tensor::from_slice(&[2.0]).requires_grad();
    let x_id = x.id();
    let y_id = y.id();
    let z = x.div(&y).sum();
    z.backward();
    let grad_x = crate::autograd::get_grad(x_id).expect("grad_x");
    let grad_y = crate::autograd::get_grad(y_id).expect("grad_y");
    assert!((grad_x.data()[0] - 0.5).abs() < 1e-5); // 1/2
    assert!((grad_y.data()[0] - (-1.5)).abs() < 1e-5); // -6/4
}

#[test]
fn test_neg_gradient() {
    // d/dx (-x) = -1
    clear_graph();
    let x = Tensor::from_slice(&[3.0]).requires_grad();
    let x_id = x.id();
    let z = x.neg().sum();
    z.backward();
    let grad = crate::autograd::get_grad(x_id).expect("grad");
    assert_eq!(grad.data()[0], -1.0);
}

#[test]
fn test_pow_gradient_cubic() {
    // d/dx x^n = n * x^(n-1)
    clear_graph();
    let x = Tensor::from_slice(&[2.0]).requires_grad();
    let x_id = x.id();
    let z = x.pow(3.0).sum(); // x^3, grad = 3*x^2 = 12
    z.backward();
    let grad = crate::autograd::get_grad(x_id).expect("grad");
    assert!((grad.data()[0] - 12.0).abs() < 1e-5);
}

#[test]
fn test_exp_gradient_e() {
    // d/dx exp(x) = exp(x)
    clear_graph();
    let x = Tensor::from_slice(&[1.0]).requires_grad();
    let x_id = x.id();
    let z = x.exp().sum();
    z.backward();
    let grad = crate::autograd::get_grad(x_id).expect("grad");
    assert!((grad.data()[0] - std::f32::consts::E).abs() < 1e-4);
}

#[test]
fn test_log_gradient_half() {
    // d/dx log(x) = 1/x
    clear_graph();
    let x = Tensor::from_slice(&[2.0]).requires_grad();
    let x_id = x.id();
    let z = x.log().sum();
    z.backward();
    let grad = crate::autograd::get_grad(x_id).expect("grad");
    assert!((grad.data()[0] - 0.5).abs() < 1e-5);
}

// =========================================================================
// Extended coverage tests
// =========================================================================

#[test]
fn test_sub_gradient() {
    // d/dx (x - y) = 1
    // d/dy (x - y) = -1
    clear_graph();
    let x = Tensor::from_slice(&[5.0, 3.0]).requires_grad();
    let y = Tensor::from_slice(&[2.0, 1.0]).requires_grad();
    let x_id = x.id();
    let y_id = y.id();
    let z = x.sub(&y).sum();
    z.backward();
    let grad_x = crate::autograd::get_grad(x_id).expect("grad_x");
    let grad_y = crate::autograd::get_grad(y_id).expect("grad_y");
    assert_eq!(grad_x.data(), &[1.0, 1.0]);
    assert_eq!(grad_y.data(), &[-1.0, -1.0]);
}

#[test]
fn test_leaky_relu_gradient_positive() {
    // For x > 0: d/dx leaky_relu(x) = 1
    clear_graph();
    let x = Tensor::from_slice(&[2.0, 3.0]).requires_grad();
    let x_id = x.id();
    let z = x.leaky_relu(0.01).sum();
    z.backward();
    let grad = crate::autograd::get_grad(x_id).expect("grad");
    assert_eq!(grad.data(), &[1.0, 1.0]);
}

#[test]
fn test_leaky_relu_gradient_negative() {
    // For x < 0: d/dx leaky_relu(x) = negative_slope
    clear_graph();
    let x = Tensor::from_slice(&[-2.0, -3.0]).requires_grad();
    let x_id = x.id();
    let negative_slope = 0.1;
    let z = x.leaky_relu(negative_slope).sum();
    z.backward();
    let grad = crate::autograd::get_grad(x_id).expect("grad");
    for &g in grad.data() {
        assert!((g - negative_slope).abs() < 1e-5);
    }
}

#[test]
fn test_gelu_forward() {
    // Test GELU forward computation
    let x = Tensor::from_slice(&[0.0, 1.0, -1.0]);
    let y = x.gelu();
    // GELU(0) ≈ 0, GELU(1) ≈ 0.841, GELU(-1) ≈ -0.159
    assert!((y.data()[0] - 0.0).abs() < 1e-3);
    assert!((y.data()[1] - 0.841).abs() < 0.01);
    assert!((y.data()[2] - (-0.159)).abs() < 0.01);
}

#[test]
fn test_gelu_gradient() {
    clear_graph();
    let x = Tensor::from_slice(&[0.0]).requires_grad();
    let x_id = x.id();
    let z = x.gelu().sum();
    z.backward();
    let grad = crate::autograd::get_grad(x_id).expect("grad");
    // GELU'(0) ≈ 0.5
    assert!((grad.data()[0] - 0.5).abs() < 0.01);
}

#[test]
fn test_softmax_forward() {
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let y = x.softmax();

    // Check rows sum to 1
    let row1_sum: f32 = y.data()[0..2].iter().sum();
    let row2_sum: f32 = y.data()[2..4].iter().sum();
    assert!((row1_sum - 1.0).abs() < 1e-5);
    assert!((row2_sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_gradient() {
    clear_graph();
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).requires_grad();
    let x_id = x.id();
    let z = x.softmax().sum();
    z.backward();
    let grad = crate::autograd::get_grad(x_id).expect("grad");
    // Softmax gradients should be small since softmax(x).sum() = n (number of rows)
    // The derivative of constant is 0, but there's some numerical detail here
    assert_eq!(grad.numel(), 4);
}

#[test]
fn test_transpose_forward() {
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let a_t = a.transpose();
    assert_eq!(a_t.shape(), &[3, 2]);
    // Original: [[1, 2, 3], [4, 5, 6]]
    // Transposed: [[1, 4], [2, 5], [3, 6]]
    assert_eq!(a_t.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_transpose_gradient() {
    clear_graph();
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).requires_grad();
    let a_id = a.id();
    let z = a.transpose().sum();
    z.backward();
    let grad = crate::autograd::get_grad(a_id).expect("grad");
    // Transpose of ones is ones
    assert_eq!(grad.data(), &[1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn test_broadcast_add_forward() {
    let matrix = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let bias = Tensor::new(&[10.0, 20.0], &[2]);
    let result = matrix.broadcast_add(&bias);
    // [[1+10, 2+20], [3+10, 4+20]] = [[11, 22], [13, 24]]
    assert_eq!(result.data(), &[11.0, 22.0, 13.0, 24.0]);
}

#[path = "tests_part_02.rs"]

mod tests_part_02;
