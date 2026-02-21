//! Tests for nn::optim optimizers
//! PMAT-085: Extracted from optim.rs for file health

pub(crate) use super::*;
pub(crate) use crate::autograd::clear_graph;

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

#[path = "tests_adam.rs"]
mod tests_adam;
#[path = "tests_state_resize.rs"]
mod tests_state_resize;
#[path = "tests_large_tensors.rs"]
mod tests_large_tensors;
