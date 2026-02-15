
// ========== Tests for State Buffer Resize ==========

#[test]
fn test_sgd_velocity_resize() {
    clear_graph();

    // Create SGD with params, but then call with more params to trigger resize
    let mut param1 = Tensor::from_slice(&[1.0]).requires_grad();
    let mut param2 = Tensor::from_slice(&[2.0]).requires_grad();
    let mut param3 = Tensor::from_slice(&[3.0]).requires_grad();

    // Initialize with first param
    let mut sgd = SGD::with_momentum(vec![&mut param1], 0.1, 0.9);

    let loss1 = param1.pow(2.0).sum();
    loss1.backward();
    sgd.step_with_params(&mut [&mut param1]);

    clear_graph();

    // Now step with all three params - triggers resize
    let loss1 = param1.pow(2.0).sum();
    let loss2 = param2.pow(2.0).sum();
    let loss3 = param3.pow(2.0).sum();
    let total = loss1.add(&loss2).add(&loss3);
    total.backward();

    // This should trigger velocity resize for idx 1 and 2
    sgd.step_with_params(&mut [&mut param1, &mut param2, &mut param3]);

    // All should have decreased
    assert!(param1.data()[0] < 1.0);
    assert!(param2.data()[0] < 2.0);
    assert!(param3.data()[0] < 3.0);
}

#[test]
fn test_adam_state_resize() {
    clear_graph();

    let mut param1 = Tensor::from_slice(&[1.0]).requires_grad();
    let mut param2 = Tensor::from_slice(&[2.0]).requires_grad();

    let mut adam = Adam::new(vec![&mut param1], 0.1);

    // First step with param1
    let loss = param1.pow(2.0).sum();
    loss.backward();
    adam.step_with_params(&mut [&mut param1]);

    clear_graph();

    // Now add param2 - triggers state resize
    let loss1 = param1.pow(2.0).sum();
    let loss2 = param2.pow(2.0).sum();
    let total = loss1.add(&loss2);
    total.backward();

    adam.step_with_params(&mut [&mut param1, &mut param2]);

    assert!(param1.data()[0] < 1.0);
    assert!(param2.data()[0] < 2.0);
}

#[test]
fn test_adamw_state_resize() {
    clear_graph();

    let mut param1 = Tensor::from_slice(&[3.0]).requires_grad();
    let mut param2 = Tensor::from_slice(&[4.0]).requires_grad();

    let mut adamw = AdamW::new(vec![&mut param1], 0.1);

    // First step
    let loss = param1.pow(2.0).sum();
    loss.backward();
    adamw.step_with_params(&mut [&mut param1]);

    clear_graph();

    // Add param2
    let loss1 = param1.pow(2.0).sum();
    let loss2 = param2.pow(2.0).sum();
    let total = loss1.add(&loss2);
    total.backward();

    adamw.step_with_params(&mut [&mut param1, &mut param2]);

    assert!(param1.data()[0] < 3.0);
    assert!(param2.data()[0] < 4.0);
}

#[test]
fn test_rmsprop_state_resize() {
    clear_graph();

    let mut param1 = Tensor::from_slice(&[5.0]).requires_grad();
    let mut param2 = Tensor::from_slice(&[6.0]).requires_grad();

    let mut rmsprop = RMSprop::new(vec![&mut param1], 0.1);

    // First step
    let loss = param1.pow(2.0).sum();
    loss.backward();
    rmsprop.step_with_params(&mut [&mut param1]);

    clear_graph();

    // Add param2
    let loss1 = param1.pow(2.0).sum();
    let loss2 = param2.pow(2.0).sum();
    let total = loss1.add(&loss2);
    total.backward();

    rmsprop.step_with_params(&mut [&mut param1, &mut param2]);

    assert!(param1.data()[0] < 5.0);
    assert!(param2.data()[0] < 6.0);
}

// ========== Tests for Combined Builder Patterns ==========

#[test]
fn test_sgd_full_builder_chain() {
    clear_graph();

    let mut param = Tensor::from_slice(&[2.0]).requires_grad();

    let loss = param.pow(2.0).sum();
    loss.backward();

    // Chain all SGD builder methods
    let mut sgd = SGD::with_momentum(vec![&mut param], 0.1, 0.9)
        .nesterov()
        .weight_decay(0.01);
    sgd.step_with_params(&mut [&mut param]);

    assert!(param.data()[0] < 2.0);
}

#[test]
fn test_adam_full_builder_chain() {
    clear_graph();

    let mut param = Tensor::from_slice(&[2.0]).requires_grad();

    let loss = param.pow(2.0).sum();
    loss.backward();

    // Chain all Adam builder methods
    let mut adam = Adam::new(vec![&mut param], 0.1)
        .betas(0.85, 0.99)
        .eps(1e-7)
        .weight_decay(0.01);
    adam.step_with_params(&mut [&mut param]);

    assert!(param.data()[0] < 2.0);
}

#[test]
fn test_adamw_full_builder_chain() {
    clear_graph();

    let mut param = Tensor::from_slice(&[2.0]).requires_grad();

    let loss = param.pow(2.0).sum();
    loss.backward();

    // Chain all AdamW builder methods
    let mut adamw = AdamW::new(vec![&mut param], 0.1)
        .betas(0.85, 0.99)
        .eps(1e-7)
        .weight_decay(0.05);
    adamw.step_with_params(&mut [&mut param]);

    assert!(param.data()[0] < 2.0);
}

#[test]
fn test_rmsprop_full_builder_chain() {
    clear_graph();

    let mut param = Tensor::from_slice(&[2.0]).requires_grad();

    let loss = param.pow(2.0).sum();
    loss.backward();

    // Chain all RMSprop builder methods
    let mut rmsprop = RMSprop::new(vec![&mut param], 0.1)
        .alpha(0.95)
        .eps(1e-7)
        .momentum(0.9)
        .weight_decay(0.01);
    rmsprop.step_with_params(&mut [&mut param]);

    assert!(param.data()[0] < 2.0);
}

// ========== Tests for Zero Weight Decay ==========

#[test]
fn test_sgd_zero_weight_decay() {
    clear_graph();

    let mut param = Tensor::from_slice(&[5.0]).requires_grad();

    let loss = param.pow(2.0).sum();
    loss.backward();

    // Explicitly set weight_decay to 0
    let mut sgd = SGD::new(vec![&mut param], 0.1).weight_decay(0.0);
    sgd.step_with_params(&mut [&mut param]);

    // grad = 2 * 5 = 10
    // param = 5 - 0.1 * 10 = 4.0
    assert!((param.data()[0] - 4.0).abs() < 1e-5);
}

#[test]
fn test_adam_zero_weight_decay() {
    clear_graph();

    let mut param = Tensor::from_slice(&[3.0]).requires_grad();

    let loss = param.pow(2.0).sum();
    loss.backward();

    let mut adam = Adam::new(vec![&mut param], 0.1).weight_decay(0.0);
    adam.step_with_params(&mut [&mut param]);

    assert!(param.data()[0] < 3.0);
}

#[test]
fn test_adamw_zero_weight_decay() {
    clear_graph();

    let mut param = Tensor::from_slice(&[3.0]).requires_grad();

    let loss = param.pow(2.0).sum();
    loss.backward();

    let mut adamw = AdamW::new(vec![&mut param], 0.1).weight_decay(0.0);
    adamw.step_with_params(&mut [&mut param]);

    assert!(param.data()[0] < 3.0);
}

#[test]
fn test_rmsprop_zero_weight_decay() {
    clear_graph();

    let mut param = Tensor::from_slice(&[4.0]).requires_grad();

    let loss = param.pow(2.0).sum();
    loss.backward();

    let mut rmsprop = RMSprop::new(vec![&mut param], 0.1).weight_decay(0.0);
    rmsprop.step_with_params(&mut [&mut param]);

    assert!(param.data()[0] < 4.0);
}

// ========== Tests for Zero Momentum ==========

#[test]
fn test_sgd_zero_momentum_explicit() {
    clear_graph();

    let mut param = Tensor::from_slice(&[3.0]).requires_grad();

    let loss = param.pow(2.0).sum();
    loss.backward();

    // Explicitly zero momentum
    let mut sgd = SGD::with_momentum(vec![&mut param], 0.1, 0.0);
    sgd.step_with_params(&mut [&mut param]);

    // Should behave like vanilla SGD: grad = 6, param = 3 - 0.1 * 6 = 2.4
    assert!((param.data()[0] - 2.4).abs() < 1e-5);
}

#[test]
fn test_rmsprop_zero_momentum_explicit() {
    clear_graph();

    let mut param = Tensor::from_slice(&[2.0]).requires_grad();

    let loss = param.pow(2.0).sum();
    loss.backward();

    // Zero momentum with builder
    let mut rmsprop = RMSprop::new(vec![&mut param], 0.1).momentum(0.0);
    rmsprop.step_with_params(&mut [&mut param]);

    assert!(param.data()[0] < 2.0);
}

// ========== Tests for Empty Optimizer Operations ==========

#[test]
fn test_adamw_empty_params() {
    let adamw = AdamW::new(vec![], 0.1);
    assert!((adamw.lr() - 0.1).abs() < 1e-6);
}

#[test]
fn test_rmsprop_empty_params() {
    let rmsprop = RMSprop::new(vec![], 0.1);
    assert!((rmsprop.lr() - 0.1).abs() < 1e-6);
}

#[test]
fn test_sgd_zero_grad_empty() {
    let mut sgd = SGD::new(vec![], 0.1);
    // Should not panic with empty params
    sgd.zero_grad();
}

#[test]
fn test_adam_zero_grad_empty() {
    let mut adam = Adam::new(vec![], 0.1);
    adam.zero_grad();
}

#[test]
fn test_adamw_zero_grad_empty() {
    let mut adamw = AdamW::new(vec![], 0.1);
    adamw.zero_grad();
}

#[test]
fn test_rmsprop_zero_grad_empty() {
    let mut rmsprop = RMSprop::new(vec![], 0.1);
    rmsprop.zero_grad();
}

// ========== Tests for Optimizer Trait Step Method ==========

#[test]
fn test_sgd_step_multiple_times() {
    let mut param = Tensor::from_slice(&[1.0]).requires_grad();
    let mut sgd = SGD::new(vec![&mut param], 0.1);

    for _ in 0..5 {
        sgd.step();
    }
    assert!(sgd.initialized);
}

#[test]
fn test_adam_step_multiple_times() {
    let mut param = Tensor::from_slice(&[1.0]).requires_grad();
    let mut adam = Adam::new(vec![&mut param], 0.1);

    for i in 1..=5 {
        adam.step();
        assert_eq!(adam.t, i);
    }
}

#[test]
fn test_adamw_step_multiple_times() {
    let mut param = Tensor::from_slice(&[1.0]).requires_grad();
    let mut adamw = AdamW::new(vec![&mut param], 0.1);

    for i in 1..=5 {
        adamw.step();
        assert_eq!(adamw.t, i);
    }
}

#[test]
fn test_rmsprop_step_multiple_times() {
    let mut param = Tensor::from_slice(&[1.0]).requires_grad();
    let mut rmsprop = RMSprop::new(vec![&mut param], 0.1);

    for _ in 0..5 {
        rmsprop.step();
    }
    assert!(rmsprop.initialized);
}

// ========== Tests for Multiple Steps with Varying Gradients ==========

#[test]
fn test_sgd_momentum_accumulation_over_steps() {
    clear_graph();

    let mut param = Tensor::from_slice(&[5.0]).requires_grad();
    let mut sgd = SGD::with_momentum(vec![&mut param], 0.1, 0.9);

    let mut prev_val = 5.0;
    for _ in 0..5 {
        clear_graph();
        let loss = param.pow(2.0).sum();
        loss.backward();
        sgd.step_with_params(&mut [&mut param]);

        // Each step should decrease the parameter
        assert!(param.data()[0] < prev_val);
        prev_val = param.data()[0];
    }
}

#[test]
fn test_adamw_decoupled_vs_coupled_weight_decay() {
    // Test that AdamW applies weight decay separately from Adam
    clear_graph();

    // AdamW with weight decay
    let mut param_adamw = Tensor::from_slice(&[10.0]).requires_grad();
    let mut adamw = AdamW::new(vec![&mut param_adamw], 0.1).weight_decay(0.1);

    let loss = param_adamw.pow(2.0).sum();
    loss.backward();
    adamw.step_with_params(&mut [&mut param_adamw]);

    clear_graph();

    // Adam with weight decay (applied to gradient)
    let mut param_adam = Tensor::from_slice(&[10.0]).requires_grad();
    let mut adam = Adam::new(vec![&mut param_adam], 0.1).weight_decay(0.1);

    let loss = param_adam.pow(2.0).sum();
    loss.backward();
    adam.step_with_params(&mut [&mut param_adam]);

    // Both should decrease but may have slightly different values
    // due to decoupled vs coupled weight decay
    assert!(param_adamw.data()[0] < 10.0);
    assert!(param_adam.data()[0] < 10.0);
}
