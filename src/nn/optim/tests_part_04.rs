use super::*;

// ========== Tests for Large Parameter Tensors ==========

#[test]
fn test_sgd_large_tensor() {
    clear_graph();

    let data: Vec<f32> = (0..100).map(|x| x as f32 / 10.0).collect();
    let mut param = Tensor::from_slice(&data).requires_grad();

    let loss = param.pow(2.0).sum();
    loss.backward();

    let mut sgd = SGD::new(vec![&mut param], 0.01);
    sgd.step_with_params(&mut [&mut param]);

    // All non-zero elements should decrease
    for (i, &val) in param.data().iter().enumerate() {
        let original = data[i];
        if original > 0.0 {
            assert!(val < original, "Element {} should decrease", i);
        }
    }
}

#[test]
fn test_adam_large_tensor() {
    clear_graph();

    let data: Vec<f32> = (1..51).map(|x| x as f32).collect();
    let mut param = Tensor::from_slice(&data).requires_grad();

    let loss = param.pow(2.0).sum();
    loss.backward();

    let mut adam = Adam::new(vec![&mut param], 0.1);
    adam.step_with_params(&mut [&mut param]);

    // All elements should decrease
    for (i, &val) in param.data().iter().enumerate() {
        assert!(val < data[i], "Element {} should decrease", i);
    }
}

// ========== Tests for Learning Rate Edge Cases ==========

#[test]
fn test_sgd_very_small_lr() {
    clear_graph();

    let mut param = Tensor::from_slice(&[100.0]).requires_grad();

    let loss = param.pow(2.0).sum();
    loss.backward();

    let mut sgd = SGD::new(vec![&mut param], 1e-10);
    sgd.step_with_params(&mut [&mut param]);

    // Should barely change due to tiny learning rate
    assert!((param.data()[0] - 100.0).abs() < 1e-5);
}

#[test]
fn test_adam_very_small_lr() {
    clear_graph();

    let mut param = Tensor::from_slice(&[100.0]).requires_grad();

    let loss = param.pow(2.0).sum();
    loss.backward();

    let mut adam = Adam::new(vec![&mut param], 1e-10);
    adam.step_with_params(&mut [&mut param]);

    // Should barely change
    assert!((param.data()[0] - 100.0).abs() < 1e-3);
}
