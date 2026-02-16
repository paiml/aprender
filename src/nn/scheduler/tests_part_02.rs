use super::*;

#[test]
fn test_warmup_cosine_full_training() {
    let mut optimizer = MockOptimizer::new(0.1);
    let mut scheduler = WarmupCosineScheduler::with_min_lr(5, 15, 0.001);

    // Warmup phase (5 steps)
    for i in 1..=5 {
        scheduler.step(&mut optimizer);
        let expected = 0.1 * (i as f32 / 5.0);
        assert!((scheduler.get_lr() - expected).abs() < 1e-5);
    }

    // Decay phase (10 steps)
    let mut prev_lr = scheduler.get_lr();
    for _ in 0..10 {
        scheduler.step(&mut optimizer);
        assert!(scheduler.get_lr() <= prev_lr);
        prev_lr = scheduler.get_lr();
    }

    // At end, should be at or near min_lr
    assert!(scheduler.get_lr() <= 0.01);
}

#[test]
fn test_reduce_on_plateau_continuous_improvement() {
    let mut optimizer = MockOptimizer::new(0.1);
    let mut scheduler = ReduceLROnPlateau::new(PlateauMode::Min, 0.5, 3);

    // Continuous improvement - LR should never decrease
    scheduler.step_with_metric(&mut optimizer, 1.0);
    scheduler.step_with_metric(&mut optimizer, 0.9);
    scheduler.step_with_metric(&mut optimizer, 0.8);
    scheduler.step_with_metric(&mut optimizer, 0.7);
    scheduler.step_with_metric(&mut optimizer, 0.6);

    assert!((optimizer.lr() - 0.1).abs() < 1e-6);
}

#[test]
fn test_reduce_on_plateau_max_mode_improvement() {
    let mut optimizer = MockOptimizer::new(0.1);
    let mut scheduler = ReduceLROnPlateau::new(PlateauMode::Max, 0.5, 3);

    // Continuous improvement (increasing metric) - LR should stay
    scheduler.step_with_metric(&mut optimizer, 0.5);
    scheduler.step_with_metric(&mut optimizer, 0.6);
    scheduler.step_with_metric(&mut optimizer, 0.7);
    scheduler.step_with_metric(&mut optimizer, 0.8);

    assert!((optimizer.lr() - 0.1).abs() < 1e-6);
}

#[test]
fn test_step_lr_many_steps() {
    let mut optimizer = MockOptimizer::new(1.0);
    let mut scheduler = StepLR::new(2, 0.5);

    // 10 steps with decay every 2 steps
    for _ in 0..10 {
        scheduler.step(&mut optimizer);
    }

    // After 10 steps: 5 decays of 0.5 = 1.0 * 0.5^5 = 0.03125
    assert!((optimizer.lr() - 0.03125).abs() < 1e-6);
    assert_eq!(scheduler.last_epoch(), 10);
}

#[test]
fn test_exponential_lr_many_steps() {
    let mut optimizer = MockOptimizer::new(1.0);
    let mut scheduler = ExponentialLR::new(0.9);

    for _ in 0..5 {
        scheduler.step(&mut optimizer);
    }

    // After 5 steps: 1.0 * 0.9^5 ≈ 0.59049
    assert!((optimizer.lr() - 0.59049).abs() < 1e-4);
}

#[test]
fn test_reduce_on_plateau_multiple_reductions() {
    let mut optimizer = MockOptimizer::new(0.1);
    let mut scheduler = ReduceLROnPlateau::new(PlateauMode::Min, 0.5, 2).min_lr(0.001);

    scheduler.step_with_metric(&mut optimizer, 1.0);

    // First plateau - reduce from 0.1 to 0.05
    scheduler.step_with_metric(&mut optimizer, 1.0);
    scheduler.step_with_metric(&mut optimizer, 1.0);
    assert!((optimizer.lr() - 0.05).abs() < 1e-6);

    // Second plateau - reduce from 0.05 to 0.025
    scheduler.step_with_metric(&mut optimizer, 1.0);
    scheduler.step_with_metric(&mut optimizer, 1.0);
    assert!((optimizer.lr() - 0.025).abs() < 1e-6);
}
