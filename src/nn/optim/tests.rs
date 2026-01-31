//! Tests for nn::optim optimizers
//! PMAT-085: Extracted from optim.rs for file health

use super::*;
use crate::autograd::clear_graph;

#[test]
fn test_sgd_basic() {
    clear_graph();

    // Create a simple tensor
    let mut param = Tensor::from_slice(&[1.0, 2.0, 3.0]).requires_grad();
    let param_id = param.id();

    // Simulate a loss: sum of squared elements
    let loss = param.pow(2.0).sum();
    loss.backward();

    // Check gradient exists
    let grad = get_grad(param_id).expect("Should have gradient");
    assert_eq!(grad.data(), &[2.0, 4.0, 6.0]); // d/dx(x²) = 2x

    // Create optimizer and step
    let mut sgd = SGD::new(vec![&mut param], 0.1);
    sgd.step_with_params(&mut [&mut param]);

    // param = param - lr * grad = [1, 2, 3] - 0.1 * [2, 4, 6] = [0.8, 1.6, 2.4]
    let expected = [0.8, 1.6, 2.4];
    for (p, e) in param.data().iter().zip(expected.iter()) {
        assert!((p - e).abs() < 1e-5, "Expected {e}, got {p}");
    }
}

#[test]
fn test_sgd_with_momentum() {
    clear_graph();

    let mut param = Tensor::from_slice(&[1.0]).requires_grad();

    // First step
    let loss = param.pow(2.0).sum();
    loss.backward();

    let mut sgd = SGD::with_momentum(vec![&mut param], 0.1, 0.9);
    sgd.step_with_params(&mut [&mut param]);

    // v = 0.9 * 0 + 2.0 = 2.0
    // param = 1.0 - 0.1 * 2.0 = 0.8
    assert!((param.data()[0] - 0.8).abs() < 1e-5);

    // Second step
    clear_graph();
    let loss = param.pow(2.0).sum();
    loss.backward();

    sgd.step_with_params(&mut [&mut param]);

    // grad = 2 * 0.8 = 1.6
    // v = 0.9 * 2.0 + 1.6 = 3.4
    // param = 0.8 - 0.1 * 3.4 = 0.46
    assert!((param.data()[0] - 0.46).abs() < 1e-5);
}

#[test]
fn test_adam_basic() {
    clear_graph();

    let mut param = Tensor::from_slice(&[1.0, 2.0]).requires_grad();

    let loss = param.pow(2.0).sum();
    loss.backward();

    let mut adam = Adam::new(vec![&mut param], 0.1);
    adam.step_with_params(&mut [&mut param]);

    // After one step, params should decrease
    assert!(param.data()[0] < 1.0);
    assert!(param.data()[1] < 2.0);
}

#[test]
fn test_adam_convergence() {
    // Test that Adam can minimize a simple quadratic
    clear_graph();

    let mut param = Tensor::from_slice(&[5.0]).requires_grad();
    let mut adam = Adam::new(vec![&mut param], 0.5);

    // Minimize x² (optimal at x=0)
    for _ in 0..100 {
        clear_graph();
        let loss = param.pow(2.0).sum();
        loss.backward();
        adam.step_with_params(&mut [&mut param]);
    }

    // Should be close to 0
    assert!(
        param.data()[0].abs() < 0.1,
        "Parameter should converge to 0, got {}",
        param.data()[0]
    );
}

#[test]
fn test_adamw_weight_decay() {
    clear_graph();

    let mut param = Tensor::from_slice(&[10.0]).requires_grad();

    // With zero gradient, only weight decay applies
    // We need a loss that has zero gradient at current point
    // Actually, let's just test the decoupled nature

    let loss = param.pow(2.0).sum();
    loss.backward();

    let mut adamw = AdamW::new(vec![&mut param], 0.1).weight_decay(0.1);
    adamw.step_with_params(&mut [&mut param]);

    // With weight decay, param should decrease more
    assert!(param.data()[0] < 10.0);
}

#[test]
fn test_rmsprop_basic() {
    clear_graph();

    let mut param = Tensor::from_slice(&[3.0]).requires_grad();

    let loss = param.pow(2.0).sum();
    loss.backward();

    let mut rmsprop = RMSprop::new(vec![&mut param], 0.1);
    rmsprop.step_with_params(&mut [&mut param]);

    // Param should decrease
    assert!(param.data()[0] < 3.0);
}

#[test]
fn test_zero_grad() {
    clear_graph();

    let mut param = Tensor::from_slice(&[1.0, 2.0]).requires_grad();
    let param_id = param.id();

    let loss = param.pow(2.0).sum();
    loss.backward();

    // Gradient should exist
    assert!(get_grad(param_id).is_some());

    // Zero grad
    let mut sgd = SGD::new(vec![&mut param], 0.1);
    sgd.zero_grad();

    // Gradient should be cleared
    assert!(get_grad(param_id).is_none());
}

#[test]
fn test_learning_rate_change() {
    let mut param = Tensor::from_slice(&[1.0]).requires_grad();
    let mut sgd = SGD::new(vec![&mut param], 0.1);

    assert!((sgd.lr() - 0.1).abs() < 1e-6);

    sgd.set_lr(0.01);
    assert!((sgd.lr() - 0.01).abs() < 1e-6);
}

#[test]
fn test_sgd_nesterov() {
    clear_graph();

    let mut param = Tensor::from_slice(&[2.0]).requires_grad();

    let loss = param.pow(2.0).sum();
    loss.backward();

    let mut sgd = SGD::with_momentum(vec![&mut param], 0.1, 0.9).nesterov();
    sgd.step_with_params(&mut [&mut param]);

    // Nesterov should apply a "look ahead" update
    // With nesterov: param = param - lr * (momentum * velocity + grad)
    // v = 0.9 * 0 + 4 = 4 (grad = 2 * 2 = 4)
    // param = 2 - 0.1 * (0.9 * 4 + 4) = 2 - 0.1 * 7.6 = 1.24
    assert!(
        (param.data()[0] - 1.24).abs() < 1e-5,
        "Nesterov update failed: {}",
        param.data()[0]
    );
}

#[test]
fn test_sgd_weight_decay() {
    clear_graph();

    let mut param = Tensor::from_slice(&[5.0]).requires_grad();

    let loss = param.pow(2.0).sum();
    loss.backward();

    let mut sgd = SGD::new(vec![&mut param], 0.1).weight_decay(0.1);
    sgd.step_with_params(&mut [&mut param]);

    // grad = 2 * 5 = 10, with weight_decay: g = 10 + 0.1 * 5 = 10.5
    // param = 5 - 0.1 * 10.5 = 3.95
    assert!(
        (param.data()[0] - 3.95).abs() < 1e-5,
        "Weight decay update failed: {}",
        param.data()[0]
    );
}

#[test]
fn test_adam_with_custom_betas() {
    clear_graph();

    let mut param = Tensor::from_slice(&[1.0]).requires_grad();

    let loss = param.pow(2.0).sum();
    loss.backward();

    let mut adam = Adam::new(vec![&mut param], 0.1).betas(0.8, 0.99);
    adam.step_with_params(&mut [&mut param]);

    // Param should decrease with custom betas
    assert!(param.data()[0] < 1.0);
}

#[test]
fn test_adam_with_eps() {
    clear_graph();

    let mut param = Tensor::from_slice(&[1.0]).requires_grad();

    let loss = param.pow(2.0).sum();
    loss.backward();

    let mut adam = Adam::new(vec![&mut param], 0.1).eps(1e-6);
    adam.step_with_params(&mut [&mut param]);

    assert!(param.data()[0] < 1.0);
}

#[test]
fn test_adam_with_weight_decay() {
    clear_graph();

    let mut param = Tensor::from_slice(&[10.0]).requires_grad();

    let loss = param.pow(2.0).sum();
    loss.backward();

    // Compare with and without weight decay
    let mut adam_wd = Adam::new(vec![&mut param], 0.1).weight_decay(0.1);
    adam_wd.step_with_params(&mut [&mut param]);

    // With weight decay, the update should be larger
    assert!(param.data()[0] < 10.0);
}

#[test]
fn test_adamw_with_custom_betas_and_eps() {
    clear_graph();

    let mut param = Tensor::from_slice(&[3.0]).requires_grad();

    let loss = param.pow(2.0).sum();
    loss.backward();

    let mut adamw = AdamW::new(vec![&mut param], 0.1)
        .betas(0.85, 0.995)
        .eps(1e-7);
    adamw.step_with_params(&mut [&mut param]);

    assert!(param.data()[0] < 3.0);
}

#[test]
fn test_adamw_lr_methods() {
    let mut param = Tensor::from_slice(&[1.0]).requires_grad();
    let mut adamw = AdamW::new(vec![&mut param], 0.01);

    assert!((adamw.lr() - 0.01).abs() < 1e-6);
    adamw.set_lr(0.001);
    assert!((adamw.lr() - 0.001).abs() < 1e-6);
}

#[test]
fn test_adamw_zero_grad() {
    clear_graph();

    let mut param = Tensor::from_slice(&[2.0]).requires_grad();
    let param_id = param.id();

    let loss = param.pow(2.0).sum();
    loss.backward();

    assert!(get_grad(param_id).is_some());

    let mut adamw = AdamW::new(vec![&mut param], 0.1);
    adamw.zero_grad();

    assert!(get_grad(param_id).is_none());
}

#[test]
fn test_adamw_step_trait() {
    let mut param = Tensor::from_slice(&[1.0]).requires_grad();
    let mut adamw = AdamW::new(vec![&mut param], 0.1);

    // Test the Optimizer trait step method
    adamw.step();
    assert!(adamw.initialized);
    assert_eq!(adamw.t, 1);
}

#[test]
fn test_rmsprop_with_alpha() {
    clear_graph();

    let mut param = Tensor::from_slice(&[2.0]).requires_grad();

    let loss = param.pow(2.0).sum();
    loss.backward();

    let mut rmsprop = RMSprop::new(vec![&mut param], 0.1).alpha(0.9);
    rmsprop.step_with_params(&mut [&mut param]);

    assert!(param.data()[0] < 2.0);
}

#[test]
fn test_rmsprop_with_eps() {
    clear_graph();

    let mut param = Tensor::from_slice(&[2.0]).requires_grad();

    let loss = param.pow(2.0).sum();
    loss.backward();

    let mut rmsprop = RMSprop::new(vec![&mut param], 0.1).eps(1e-6);
    rmsprop.step_with_params(&mut [&mut param]);

    assert!(param.data()[0] < 2.0);
}

#[test]
fn test_rmsprop_with_momentum() {
    clear_graph();

    let mut param = Tensor::from_slice(&[3.0]).requires_grad();

    // First step
    let loss = param.pow(2.0).sum();
    loss.backward();

    let mut rmsprop = RMSprop::new(vec![&mut param], 0.1).momentum(0.9);
    rmsprop.step_with_params(&mut [&mut param]);

    let after_first = param.data()[0];
    assert!(after_first < 3.0);

    // Second step with momentum accumulation
    clear_graph();
    let loss = param.pow(2.0).sum();
    loss.backward();

    rmsprop.step_with_params(&mut [&mut param]);

    assert!(param.data()[0] < after_first);
}

#[test]
fn test_rmsprop_with_weight_decay() {
    clear_graph();

    let mut param = Tensor::from_slice(&[5.0]).requires_grad();

    let loss = param.pow(2.0).sum();
    loss.backward();

    let mut rmsprop = RMSprop::new(vec![&mut param], 0.1).weight_decay(0.1);
    rmsprop.step_with_params(&mut [&mut param]);

    assert!(param.data()[0] < 5.0);
}

#[test]
fn test_rmsprop_lr_methods() {
    let mut param = Tensor::from_slice(&[1.0]).requires_grad();
    let mut rmsprop = RMSprop::new(vec![&mut param], 0.01);

    assert!((rmsprop.lr() - 0.01).abs() < 1e-6);
    rmsprop.set_lr(0.001);
    assert!((rmsprop.lr() - 0.001).abs() < 1e-6);
}

#[test]
fn test_rmsprop_zero_grad() {
    clear_graph();

    let mut param = Tensor::from_slice(&[2.0]).requires_grad();
    let param_id = param.id();

    let loss = param.pow(2.0).sum();
    loss.backward();

    assert!(get_grad(param_id).is_some());

    let mut rmsprop = RMSprop::new(vec![&mut param], 0.1);
    rmsprop.zero_grad();

    assert!(get_grad(param_id).is_none());
}

#[test]
fn test_rmsprop_step_trait() {
    let mut param = Tensor::from_slice(&[1.0]).requires_grad();
    let mut rmsprop = RMSprop::new(vec![&mut param], 0.1);

    rmsprop.step();
    assert!(rmsprop.initialized);
}

#[test]
fn test_sgd_step_trait() {
    let mut param = Tensor::from_slice(&[1.0]).requires_grad();
    let mut sgd = SGD::new(vec![&mut param], 0.1);

    sgd.step();
    assert!(sgd.initialized);
}

#[test]
fn test_adam_step_trait() {
    let mut param = Tensor::from_slice(&[1.0]).requires_grad();
    let mut adam = Adam::new(vec![&mut param], 0.1);

    adam.step();
    assert!(adam.initialized);
    assert_eq!(adam.t, 1);
}

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
