//! Tests for learning rate schedulers

use super::*;
use crate::optim::Optimizer;
use approx::assert_abs_diff_eq;

#[test]
fn test_cosine_annealing_initial_lr() {
    let scheduler = CosineAnnealingLR::new(1.0, 100, 0.0);
    // At step 0, should return lr_max
    assert_abs_diff_eq!(scheduler.get_lr(), 1.0, epsilon = 1e-6);
}

#[test]
fn test_cosine_annealing_final_lr() {
    let mut scheduler = CosineAnnealingLR::new(1.0, 100, 0.0);

    // Step to the end
    for _ in 0..100 {
        scheduler.step();
    }

    // At step t_max, should return lr_min
    assert_abs_diff_eq!(scheduler.get_lr(), 0.0, epsilon = 1e-6);
}

#[test]
fn test_cosine_annealing_midpoint() {
    let mut scheduler = CosineAnnealingLR::new(1.0, 100, 0.0);

    // Step to midpoint
    for _ in 0..50 {
        scheduler.step();
    }

    // At midpoint (t = T/2), cos(pi/2) = 0, so lr = lr_max / 2
    assert_abs_diff_eq!(scheduler.get_lr(), 0.5, epsilon = 1e-4);
}

#[test]
fn test_cosine_annealing_with_min() {
    let mut scheduler = CosineAnnealingLR::new(1.0, 100, 0.1);

    // At start
    assert_abs_diff_eq!(scheduler.get_lr(), 1.0, epsilon = 1e-6);

    // Step to end
    for _ in 0..100 {
        scheduler.step();
    }

    // At end, should be lr_min = 0.1
    assert_abs_diff_eq!(scheduler.get_lr(), 0.1, epsilon = 1e-6);
}

#[test]
fn test_cosine_annealing_decreases_monotonically() {
    let mut scheduler = CosineAnnealingLR::new(1.0, 100, 0.0);
    let mut prev_lr = scheduler.get_lr();

    for _ in 0..100 {
        scheduler.step();
        let current_lr = scheduler.get_lr();
        assert!(
            current_lr <= prev_lr,
            "Learning rate should decrease monotonically: prev={prev_lr}, current={current_lr}"
        );
        prev_lr = current_lr;
    }
}

#[test]
fn test_cosine_annealing_with_optimizer() {
    use crate::optim::SGD;

    let mut optimizer = SGD::new(1.0, 0.0);
    let mut scheduler = CosineAnnealingLR::default_min(1.0, 10);

    // Initial learning rate
    assert_abs_diff_eq!(optimizer.lr(), 1.0, epsilon = 1e-6);

    // Apply scheduler
    scheduler.apply(&mut optimizer);
    assert_abs_diff_eq!(optimizer.lr(), 1.0, epsilon = 1e-6);

    // Step and apply
    scheduler.step();
    scheduler.apply(&mut optimizer);

    // Learning rate should have decreased
    assert!(optimizer.lr() < 1.0);
}

#[test]
fn test_cosine_annealing_past_t_max() {
    let mut scheduler = CosineAnnealingLR::new(1.0, 10, 0.0);

    // Step past t_max
    for _ in 0..20 {
        scheduler.step();
    }

    // Should stay at lr_min
    assert_abs_diff_eq!(scheduler.get_lr(), 0.0, epsilon = 1e-6);
}

// =========================================================================
// LinearWarmupLR tests
// =========================================================================

#[test]
fn test_linear_warmup_initial() {
    let scheduler = LinearWarmupLR::new(0.001, 100);
    // At step 0, LR should be 0
    assert_abs_diff_eq!(scheduler.get_lr(), 0.0, epsilon = 1e-8);
}

#[test]
fn test_linear_warmup_midpoint() {
    let mut scheduler = LinearWarmupLR::new(0.001, 100);
    for _ in 0..50 {
        scheduler.step();
    }
    // At midpoint, should be half of target
    assert_abs_diff_eq!(scheduler.get_lr(), 0.0005, epsilon = 1e-7);
}

#[test]
fn test_linear_warmup_complete() {
    let mut scheduler = LinearWarmupLR::new(0.001, 100);
    for _ in 0..100 {
        scheduler.step();
    }
    // After warmup, should be at target
    assert_abs_diff_eq!(scheduler.get_lr(), 0.001, epsilon = 1e-7);
}

#[test]
fn test_linear_warmup_past_warmup() {
    let mut scheduler = LinearWarmupLR::new(0.001, 100);
    for _ in 0..200 {
        scheduler.step();
    }
    // Should stay at target after warmup
    assert_abs_diff_eq!(scheduler.get_lr(), 0.001, epsilon = 1e-7);
}

#[test]
fn test_linear_warmup_increases_monotonically() {
    let mut scheduler = LinearWarmupLR::new(0.001, 100);
    let mut prev_lr = scheduler.get_lr();

    for _ in 0..100 {
        scheduler.step();
        let current_lr = scheduler.get_lr();
        assert!(
            current_lr >= prev_lr,
            "LR should increase during warmup: prev={prev_lr}, current={current_lr}"
        );
        prev_lr = current_lr;
    }
}

// =========================================================================
// StepDecayLR tests
// =========================================================================

#[test]
fn test_step_decay_initial() {
    let scheduler = StepDecayLR::new(0.1, 10, 0.1);
    assert_abs_diff_eq!(scheduler.get_lr(), 0.1, epsilon = 1e-7);
}

#[test]
fn test_step_decay_first_decay() {
    let mut scheduler = StepDecayLR::new(0.1, 10, 0.1);
    for _ in 0..10 {
        scheduler.step();
    }
    // After 10 epochs, should decay by gamma
    assert_abs_diff_eq!(scheduler.get_lr(), 0.01, epsilon = 1e-7);
}

#[test]
fn test_step_decay_second_decay() {
    let mut scheduler = StepDecayLR::new(0.1, 10, 0.1);
    for _ in 0..20 {
        scheduler.step();
    }
    // After 20 epochs, should decay twice
    assert_abs_diff_eq!(scheduler.get_lr(), 0.001, epsilon = 1e-8);
}

#[test]
fn test_step_decay_between_steps() {
    let mut scheduler = StepDecayLR::new(0.1, 10, 0.1);
    for _ in 0..5 {
        scheduler.step();
    }
    // Between decay steps, should stay at initial
    assert_abs_diff_eq!(scheduler.get_lr(), 0.1, epsilon = 1e-7);
}

// =========================================================================
// WarmupCosineDecayLR tests
// =========================================================================

#[test]
fn test_warmup_cosine_initial() {
    let scheduler = WarmupCosineDecayLR::new(0.001, 0.0, 10, 100);
    // At step 0, should be 0 (warmup phase)
    assert_abs_diff_eq!(scheduler.get_lr(), 0.0, epsilon = 1e-8);
}

#[test]
fn test_warmup_cosine_warmup_midpoint() {
    let mut scheduler = WarmupCosineDecayLR::new(0.001, 0.0, 10, 100);
    for _ in 0..5 {
        scheduler.step();
    }
    // Midpoint of warmup: half of lr_max
    assert_abs_diff_eq!(scheduler.get_lr(), 0.0005, epsilon = 1e-7);
}

#[test]
fn test_warmup_cosine_warmup_complete() {
    let mut scheduler = WarmupCosineDecayLR::new(0.001, 0.0, 10, 100);
    for _ in 0..10 {
        scheduler.step();
    }
    // At end of warmup, should be at lr_max
    assert_abs_diff_eq!(scheduler.get_lr(), 0.001, epsilon = 1e-7);
}

#[test]
fn test_warmup_cosine_decay_complete() {
    let mut scheduler = WarmupCosineDecayLR::new(0.001, 0.0, 10, 100);
    for _ in 0..100 {
        scheduler.step();
    }
    // At end, should be at lr_min
    assert_abs_diff_eq!(scheduler.get_lr(), 0.0, epsilon = 1e-7);
}

#[test]
fn test_warmup_cosine_warmup_increases_then_decreases() {
    let mut scheduler = WarmupCosineDecayLR::new(0.001, 0.0, 10, 100);
    let mut prev_lr = scheduler.get_lr();

    // Warmup phase: should increase
    for _ in 0..10 {
        scheduler.step();
        let current_lr = scheduler.get_lr();
        assert!(
            current_lr >= prev_lr,
            "LR should increase during warmup: prev={prev_lr}, current={current_lr}"
        );
        prev_lr = current_lr;
    }

    // Decay phase: should decrease
    for _ in 10..100 {
        scheduler.step();
        let current_lr = scheduler.get_lr();
        assert!(
            current_lr <= prev_lr,
            "LR should decrease during decay: prev={prev_lr}, current={current_lr}"
        );
        prev_lr = current_lr;
    }
}

// =========================================================================
// Additional coverage tests
// =========================================================================

#[test]
fn test_linear_warmup_apply() {
    use crate::optim::SGD;
    let mut optimizer = SGD::new(0.0, 0.0);
    let mut scheduler = LinearWarmupLR::new(0.01, 10);

    scheduler.step();
    scheduler.apply(&mut optimizer);
    assert!(optimizer.lr() > 0.0);
}

#[test]
fn test_linear_warmup_zero_steps() {
    let scheduler = LinearWarmupLR::new(0.01, 0);
    // With warmup_steps = 0, should immediately return target
    assert_abs_diff_eq!(scheduler.get_lr(), 0.01, epsilon = 1e-8);
}

#[test]
fn test_step_decay_apply() {
    use crate::optim::SGD;
    let mut optimizer = SGD::new(0.0, 0.0);
    let scheduler = StepDecayLR::new(0.1, 10, 0.1);

    scheduler.apply(&mut optimizer);
    assert_abs_diff_eq!(optimizer.lr(), 0.1, epsilon = 1e-8);
}

#[test]
fn test_step_decay_zero_step_size() {
    let scheduler = StepDecayLR::new(0.1, 0, 0.1);
    // With step_size = 0, should always return initial
    assert_abs_diff_eq!(scheduler.get_lr(), 0.1, epsilon = 1e-8);
}

#[test]
fn test_warmup_cosine_apply() {
    use crate::optim::SGD;
    let mut optimizer = SGD::new(0.0, 0.0);
    let mut scheduler = WarmupCosineDecayLR::new(0.01, 0.0, 10, 100);

    for _ in 0..10 {
        scheduler.step();
    }
    scheduler.apply(&mut optimizer);
    assert_abs_diff_eq!(optimizer.lr(), 0.01, epsilon = 1e-8);
}

#[test]
fn test_warmup_cosine_zero_warmup_steps() {
    let scheduler = WarmupCosineDecayLR::new(0.01, 0.0, 0, 100);
    // With warmup_steps = 0, should start at lr_max
    assert_abs_diff_eq!(scheduler.get_lr(), 0.01, epsilon = 1e-8);
}

#[test]
fn test_warmup_cosine_zero_total_steps() {
    let scheduler = WarmupCosineDecayLR::new(0.01, 0.001, 0, 0);
    // With total_steps = 0 and warmup_steps = 0, decay_steps = 0
    assert_abs_diff_eq!(scheduler.get_lr(), 0.001, epsilon = 1e-8);
}

#[test]
fn test_warmup_cosine_past_total() {
    let mut scheduler = WarmupCosineDecayLR::new(0.01, 0.001, 10, 50);
    for _ in 0..100 {
        scheduler.step();
    }
    // Past total steps, should return lr_min
    assert_abs_diff_eq!(scheduler.get_lr(), 0.001, epsilon = 1e-8);
}
