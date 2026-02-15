
#[test]
fn test_broadcast_add_gradient() {
    clear_graph();
    let matrix = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).requires_grad();
    let bias = Tensor::new(&[10.0, 20.0], &[2]).requires_grad();
    let m_id = matrix.id();
    let b_id = bias.id();
    let z = matrix.broadcast_add(&bias).sum();
    z.backward();
    let grad_m = crate::autograd::get_grad(m_id).expect("grad_m");
    let grad_b = crate::autograd::get_grad(b_id).expect("grad_b");
    // Matrix grad = ones
    assert_eq!(grad_m.data(), &[1.0, 1.0, 1.0, 1.0]);
    // Bias grad = summed over rows = [2, 2] (since 2 rows)
    assert_eq!(grad_b.data(), &[2.0, 2.0]);
}

#[test]
fn test_view_forward() {
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = a.view(&[3, 2]);
    assert_eq!(b.shape(), &[3, 2]);
    assert_eq!(b.data(), a.data()); // Same data, different shape
}

#[test]
fn test_view_gradient() {
    clear_graph();
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).requires_grad();
    let a_id = a.id();
    let z = a.view(&[4]).sum();
    z.backward();
    let grad = crate::autograd::get_grad(a_id).expect("grad");
    assert_eq!(grad.shape(), &[2, 2]); // Same shape as input
    assert_eq!(grad.data(), &[1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn test_mul_scalar_gradient() {
    clear_graph();
    let x = Tensor::from_slice(&[1.0, 2.0, 3.0]).requires_grad();
    let x_id = x.id();
    let z = x.mul_scalar(5.0).sum();
    z.backward();
    let grad = crate::autograd::get_grad(x_id).expect("grad");
    // d/dx (5*x).sum() = 5 for each element
    assert_eq!(grad.data(), &[5.0, 5.0, 5.0]);
}

#[test]
fn test_operations_without_grad() {
    // Test that operations work correctly when grad is not required
    let x = Tensor::from_slice(&[1.0, 2.0, 3.0]);
    let y = Tensor::from_slice(&[4.0, 5.0, 6.0]);

    // All these should work without recording gradients
    let _ = x.add(&y);
    let _ = x.sub(&y);
    let _ = x.mul(&y);
    let _ = x.div(&y);
    let _ = x.neg();
    let _ = x.exp();
    let _ = x.log();
    let _ = x.pow(2.0);
    let _ = x.sqrt();
    let _ = x.sum();
    let _ = x.mean();
    let _ = x.relu();
    let _ = x.sigmoid();
    let _ = x.tanh_();
    let _ = x.leaky_relu(0.01);
    let _ = x.gelu();
    let _ = x.mul_scalar(2.0);
}

#[test]
fn test_no_grad_context() {
    clear_graph();
    let result = no_grad(|| {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0]).requires_grad();
        x.sum()
    });
    // Inside no_grad, operations don't record to graph
    assert_eq!(result.data(), &[6.0]);
}

#[test]
fn test_matmul_non_square() {
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]); // 2x3
    let b = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]); // 3x2

    let c = a.matmul(&b);

    assert_eq!(c.shape(), &[2, 2]);
    // Row 0: [1,2,3] @ [1,3,5; 2,4,6]^T = [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
    // Row 1: [4,5,6] @ [1,3,5; 2,4,6]^T = [4*1+5*3+6*5, 4*2+5*4+6*6] = [49, 64]
    assert_eq!(c.data(), &[22.0, 28.0, 49.0, 64.0]);
}

#[test]
fn test_sub_forward() {
    let x = Tensor::from_slice(&[5.0, 3.0, 1.0]);
    let y = Tensor::from_slice(&[2.0, 2.0, 2.0]);
    let z = x.sub(&y);
    assert_eq!(z.data(), &[3.0, 1.0, -1.0]);
}

#[test]
fn test_div_forward() {
    let x = Tensor::from_slice(&[6.0, 9.0, 12.0]);
    let y = Tensor::from_slice(&[2.0, 3.0, 4.0]);
    let z = x.div(&y);
    assert_eq!(z.data(), &[3.0, 3.0, 3.0]);
}

#[test]
fn test_neg_forward() {
    let x = Tensor::from_slice(&[1.0, -2.0, 3.0]);
    let z = x.neg();
    assert_eq!(z.data(), &[-1.0, 2.0, -3.0]);
}

#[test]
fn test_sqrt_forward() {
    let x = Tensor::from_slice(&[4.0, 9.0, 16.0]);
    let z = x.sqrt();
    assert_eq!(z.data(), &[2.0, 3.0, 4.0]);
}

#[test]
fn test_tanh_forward() {
    let x = Tensor::from_slice(&[0.0]);
    let z = x.tanh_();
    assert!((z.data()[0] - 0.0).abs() < 1e-5);
}

#[test]
fn test_relu_forward() {
    let x = Tensor::from_slice(&[-2.0, 0.0, 2.0]);
    let z = x.relu();
    assert_eq!(z.data(), &[0.0, 0.0, 2.0]);
}

#[test]
fn test_leaky_relu_forward() {
    let x = Tensor::from_slice(&[-2.0, 0.0, 2.0]);
    let z = x.leaky_relu(0.1);
    assert_eq!(z.data(), &[-0.2, 0.0, 2.0]);
}

#[test]
fn test_sigmoid_forward() {
    let x = Tensor::from_slice(&[0.0]);
    let z = x.sigmoid();
    assert!((z.data()[0] - 0.5).abs() < 1e-5);
}

#[test]
fn test_log_forward() {
    let x = Tensor::from_slice(&[1.0, std::f32::consts::E]);
    let z = x.log();
    assert!((z.data()[0] - 0.0).abs() < 1e-5);
    assert!((z.data()[1] - 1.0).abs() < 1e-5);
}

#[test]
fn test_exp_forward() {
    let x = Tensor::from_slice(&[0.0, 1.0]);
    let z = x.exp();
    assert!((z.data()[0] - 1.0).abs() < 1e-5);
    assert!((z.data()[1] - std::f32::consts::E).abs() < 1e-4);
}

#[test]
fn test_pow_forward() {
    let x = Tensor::from_slice(&[2.0, 3.0]);
    let z = x.pow(2.0);
    assert_eq!(z.data(), &[4.0, 9.0]);
}

#[test]
fn test_add_forward() {
    let x = Tensor::from_slice(&[1.0, 2.0, 3.0]);
    let y = Tensor::from_slice(&[4.0, 5.0, 6.0]);
    let z = x.add(&y);
    assert_eq!(z.data(), &[5.0, 7.0, 9.0]);
}

#[test]
fn test_mul_forward() {
    let x = Tensor::from_slice(&[1.0, 2.0, 3.0]);
    let y = Tensor::from_slice(&[4.0, 5.0, 6.0]);
    let z = x.mul(&y);
    assert_eq!(z.data(), &[4.0, 10.0, 18.0]);
}

#[test]
fn test_sum_forward() {
    let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let z = x.sum();
    assert_eq!(z.data(), &[10.0]);
}

#[test]
fn test_mean_forward() {
    let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let z = x.mean();
    assert_eq!(z.data(), &[2.5]);
}

#[test]
fn test_mul_scalar_forward() {
    let x = Tensor::from_slice(&[1.0, 2.0, 3.0]);
    let z = x.mul_scalar(3.0);
    assert_eq!(z.data(), &[3.0, 6.0, 9.0]);
}
