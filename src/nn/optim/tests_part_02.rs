use super::*;

#[test]
fn test_adam_lr_methods() {
    let mut param = Tensor::from_slice(&[1.0]).requires_grad();
    let mut adam = Adam::new(vec![&mut param], 0.01);

    assert!((adam.lr() - 0.01).abs() < 1e-6);
    adam.set_lr(0.001);
    assert!((adam.lr() - 0.001).abs() < 1e-6);
}

#[test]
fn test_adam_zero_grad() {
    clear_graph();

    let mut param = Tensor::from_slice(&[2.0]).requires_grad();
    let param_id = param.id();

    let loss = param.pow(2.0).sum();
    loss.backward();

    assert!(get_grad(param_id).is_some());

    let mut adam = Adam::new(vec![&mut param], 0.1);
    adam.zero_grad();

    assert!(get_grad(param_id).is_none());
}

#[test]
fn test_sgd_multi_element_tensor() {
    clear_graph();

    let mut param = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]).requires_grad();

    let loss = param.pow(2.0).sum();
    loss.backward();

    let mut sgd = SGD::new(vec![&mut param], 0.1);
    sgd.step_with_params(&mut [&mut param]);

    // All elements should have decreased
    assert!(param.data()[0] < 1.0);
    assert!(param.data()[1] < 2.0);
    assert!(param.data()[2] < 3.0);
    assert!(param.data()[3] < 4.0);
}

#[test]
fn test_adam_multi_element_tensor() {
    clear_graph();

    let mut param = Tensor::from_slice(&[1.0, 2.0, 3.0]).requires_grad();

    let loss = param.pow(2.0).sum();
    loss.backward();

    let mut adam = Adam::new(vec![&mut param], 0.1);
    adam.step_with_params(&mut [&mut param]);

    // All elements should have decreased
    assert!(param.data()[0] < 1.0);
    assert!(param.data()[1] < 2.0);
    assert!(param.data()[2] < 3.0);
}

#[test]
fn test_adamw_multi_step() {
    clear_graph();

    let mut param = Tensor::from_slice(&[5.0]).requires_grad();
    let mut adamw = AdamW::new(vec![&mut param], 0.5).weight_decay(0.01);

    // Multiple steps to test convergence
    for _ in 0..10 {
        clear_graph();
        let loss = param.pow(2.0).sum();
        loss.backward();
        adamw.step_with_params(&mut [&mut param]);
    }

    // Should have decreased significantly
    assert!(param.data()[0] < 1.0);
}

#[test]
fn test_rmsprop_convergence() {
    clear_graph();

    let mut param = Tensor::from_slice(&[5.0]).requires_grad();
    let mut rmsprop = RMSprop::new(vec![&mut param], 0.5);

    // Multiple steps to test convergence
    for _ in 0..10 {
        clear_graph();
        let loss = param.pow(2.0).sum();
        loss.backward();
        rmsprop.step_with_params(&mut [&mut param]);
    }

    // Should have decreased significantly
    assert!(param.data()[0] < 1.0);
}

// ========== Additional Coverage Tests ==========

#[test]
fn test_sgd_lr_accessor() {
    let mut param = Tensor::from_slice(&[1.0]).requires_grad();
    let sgd = SGD::new(vec![&mut param], 0.05);
    assert!((sgd.lr() - 0.05).abs() < 1e-6);
}

#[test]
fn test_adam_lr_accessor() {
    let mut param = Tensor::from_slice(&[1.0]).requires_grad();
    let adam = Adam::new(vec![&mut param], 0.001);
    assert!((adam.lr() - 0.001).abs() < 1e-6);
}

#[test]
fn test_adamw_lr_accessor() {
    let mut param = Tensor::from_slice(&[1.0]).requires_grad();
    let adamw = AdamW::new(vec![&mut param], 0.002);
    assert!((adamw.lr() - 0.002).abs() < 1e-6);
}

#[test]
fn test_rmsprop_lr_accessor() {
    let mut param = Tensor::from_slice(&[1.0]).requires_grad();
    let rmsprop = RMSprop::new(vec![&mut param], 0.003);
    assert!((rmsprop.lr() - 0.003).abs() < 1e-6);
}

#[test]
fn test_adam_set_lr() {
    let mut param = Tensor::from_slice(&[1.0]).requires_grad();
    let mut adam = Adam::new(vec![&mut param], 0.1);
    adam.set_lr(0.001);
    assert!((adam.lr() - 0.001).abs() < 1e-6);
}

#[test]
fn test_adamw_set_lr() {
    let mut param = Tensor::from_slice(&[1.0]).requires_grad();
    let mut adamw = AdamW::new(vec![&mut param], 0.1);
    adamw.set_lr(0.001);
    assert!((adamw.lr() - 0.001).abs() < 1e-6);
}

#[test]
fn test_rmsprop_set_lr() {
    let mut param = Tensor::from_slice(&[1.0]).requires_grad();
    let mut rmsprop = RMSprop::new(vec![&mut param], 0.1);
    rmsprop.set_lr(0.001);
    assert!((rmsprop.lr() - 0.001).abs() < 1e-6);
}

#[test]
fn test_adam_zero_grad_clears() {
    clear_graph();

    let mut param = Tensor::from_slice(&[1.0]).requires_grad();
    let param_id = param.id();

    let loss = param.pow(2.0).sum();
    loss.backward();

    assert!(get_grad(param_id).is_some());

    let mut adam = Adam::new(vec![&mut param], 0.1);
    adam.zero_grad();

    assert!(get_grad(param_id).is_none());
}

#[test]
fn test_adamw_zero_grad_clears() {
    clear_graph();

    let mut param = Tensor::from_slice(&[1.0]).requires_grad();
    let param_id = param.id();

    let loss = param.pow(2.0).sum();
    loss.backward();

    assert!(get_grad(param_id).is_some());

    let mut adamw = AdamW::new(vec![&mut param], 0.1);
    adamw.zero_grad();

    assert!(get_grad(param_id).is_none());
}

#[test]
fn test_rmsprop_zero_grad_clears() {
    clear_graph();

    let mut param = Tensor::from_slice(&[1.0]).requires_grad();
    let param_id = param.id();

    let loss = param.pow(2.0).sum();
    loss.backward();

    assert!(get_grad(param_id).is_some());

    let mut rmsprop = RMSprop::new(vec![&mut param], 0.1);
    rmsprop.zero_grad();

    assert!(get_grad(param_id).is_none());
}

#[test]
fn test_sgd_multiple_params() {
    clear_graph();

    let mut param1 = Tensor::from_slice(&[1.0]).requires_grad();
    let mut param2 = Tensor::from_slice(&[2.0]).requires_grad();

    // Create loss using both params (use add method instead of +)
    let loss1 = param1.pow(2.0).sum();
    let loss2 = param2.pow(2.0).sum();
    let loss = loss1.add(&loss2);
    loss.backward();

    let mut sgd = SGD::new(vec![&mut param1, &mut param2], 0.1);
    sgd.step_with_params(&mut [&mut param1, &mut param2]);

    // Both params should have decreased
    assert!(param1.data()[0] < 1.0);
    assert!(param2.data()[0] < 2.0);
}

#[test]
fn test_adam_multiple_params() {
    clear_graph();

    let mut param1 = Tensor::from_slice(&[1.0]).requires_grad();
    let mut param2 = Tensor::from_slice(&[2.0]).requires_grad();

    let loss1 = param1.pow(2.0).sum();
    let loss2 = param2.pow(2.0).sum();
    let loss = loss1.add(&loss2);
    loss.backward();

    let mut adam = Adam::new(vec![&mut param1, &mut param2], 0.1);
    adam.step_with_params(&mut [&mut param1, &mut param2]);

    assert!(param1.data()[0] < 1.0);
    assert!(param2.data()[0] < 2.0);
}

#[test]
fn test_rmsprop_alpha_builder() {
    clear_graph();

    let mut param = Tensor::from_slice(&[5.0]).requires_grad();

    let loss = param.pow(2.0).sum();
    loss.backward();

    // Create RMSprop with custom alpha using builder pattern
    let mut rmsprop = RMSprop::new(vec![&mut param], 0.1).alpha(0.9);
    rmsprop.step_with_params(&mut [&mut param]);

    assert!(param.data()[0] < 5.0);
}

#[test]
fn test_sgd_debug_trait() {
    let mut param = Tensor::from_slice(&[1.0]).requires_grad();
    let sgd = SGD::new(vec![&mut param], 0.1);
    let debug_str = format!("{:?}", sgd);
    assert!(debug_str.contains("SGD"));
}

#[test]
fn test_adam_debug_trait() {
    let mut param = Tensor::from_slice(&[1.0]).requires_grad();
    let adam = Adam::new(vec![&mut param], 0.1);
    let debug_str = format!("{:?}", adam);
    assert!(debug_str.contains("Adam"));
}

#[test]
fn test_adamw_debug_trait() {
    let mut param = Tensor::from_slice(&[1.0]).requires_grad();
    let adamw = AdamW::new(vec![&mut param], 0.1);
    let debug_str = format!("{:?}", adamw);
    assert!(debug_str.contains("AdamW"));
}

#[test]
fn test_rmsprop_debug_trait() {
    let mut param = Tensor::from_slice(&[1.0]).requires_grad();
    let rmsprop = RMSprop::new(vec![&mut param], 0.1);
    let debug_str = format!("{:?}", rmsprop);
    assert!(debug_str.contains("RMSprop"));
}

#[test]
fn test_sgd_empty_params() {
    let sgd = SGD::new(vec![], 0.1);
    assert!((sgd.lr() - 0.1).abs() < 1e-6);
}

#[test]
fn test_adam_empty_params() {
    let adam = Adam::new(vec![], 0.1);
    assert!((adam.lr() - 0.1).abs() < 1e-6);
}

#[test]
fn test_sgd_momentum_initialization() {
    clear_graph();

    let mut param = Tensor::from_slice(&[3.0, 4.0]).requires_grad();

    let loss = param.pow(2.0).sum();
    loss.backward();

    // First step initializes velocities
    let mut sgd = SGD::with_momentum(vec![&mut param], 0.1, 0.9);
    sgd.step_with_params(&mut [&mut param]);

    // After first step, momentum buffer should be initialized
    assert!(param.data()[0] < 3.0);
    assert!(param.data()[1] < 4.0);
}

#[test]
fn test_adam_step_counter() {
    clear_graph();

    let mut param = Tensor::from_slice(&[1.0]).requires_grad();
    let mut adam = Adam::new(vec![&mut param], 0.1);

    // Step multiple times
    for _ in 0..3 {
        clear_graph();
        let loss = param.pow(2.0).sum();
        loss.backward();
        adam.step_with_params(&mut [&mut param]);
    }

    // After 3 steps param should have decreased from 1.0
    assert!(param.data()[0] < 1.0);
}

// ========== Tests for No Gradient Cases ==========

#[test]
fn test_sgd_no_gradient_skips_update() {
    clear_graph();

    let mut param = Tensor::from_slice(&[1.0, 2.0, 3.0]).requires_grad();
    let original_data = param.data().to_vec();

    // Don't call backward(), so no gradient exists
    let mut sgd = SGD::new(vec![&mut param], 0.1);
    sgd.step_with_params(&mut [&mut param]);

    // Param should be unchanged since no gradient
    assert_eq!(param.data(), &original_data);
}

#[test]
fn test_sgd_momentum_no_gradient_skips_update() {
    clear_graph();

    let mut param = Tensor::from_slice(&[5.0]).requires_grad();
    let original = param.data()[0];

    // No backward(), so no gradient
    let mut sgd = SGD::with_momentum(vec![&mut param], 0.1, 0.9);
    sgd.step_with_params(&mut [&mut param]);

    // Should be unchanged
    assert!((param.data()[0] - original).abs() < 1e-6);
}

#[test]
fn test_adam_no_gradient_skips_update() {
    clear_graph();

    let mut param = Tensor::from_slice(&[2.0, 3.0]).requires_grad();
    let original_data = param.data().to_vec();

    // No backward()
    let mut adam = Adam::new(vec![&mut param], 0.1);
    adam.step_with_params(&mut [&mut param]);

    // Should be unchanged
    assert_eq!(param.data(), &original_data);
}

#[test]
fn test_adamw_no_gradient_skips_update() {
    clear_graph();

    let mut param = Tensor::from_slice(&[4.0]).requires_grad();
    let original = param.data()[0];

    // No backward()
    let mut adamw = AdamW::new(vec![&mut param], 0.1);
    adamw.step_with_params(&mut [&mut param]);

    // Should be unchanged
    assert!((param.data()[0] - original).abs() < 1e-6);
}

#[test]
fn test_rmsprop_no_gradient_skips_update() {
    clear_graph();

    let mut param = Tensor::from_slice(&[6.0]).requires_grad();
    let original = param.data()[0];

    // No backward()
    let mut rmsprop = RMSprop::new(vec![&mut param], 0.1);
    rmsprop.step_with_params(&mut [&mut param]);

    // Should be unchanged
    assert!((param.data()[0] - original).abs() < 1e-6);
}
