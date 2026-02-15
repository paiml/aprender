use super::*;

// Mock optimizer for testing
struct MockOptimizer {
    lr: f32,
}

impl MockOptimizer {
    fn new(lr: f32) -> Self {
        Self { lr }
    }
}

impl Optimizer for MockOptimizer {
    fn step(&mut self) {}
    fn zero_grad(&mut self) {}
    fn lr(&self) -> f32 {
        self.lr
    }
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

#[test]
fn test_step_lr() {
    let mut optimizer = MockOptimizer::new(0.1);
    let mut scheduler = StepLR::new(3, 0.1);

    // First 3 epochs: lr = 0.1
    scheduler.step(&mut optimizer);
    assert!((optimizer.lr() - 0.1).abs() < 1e-6);
    scheduler.step(&mut optimizer);
    assert!((optimizer.lr() - 0.1).abs() < 1e-6);
    scheduler.step(&mut optimizer);
    // After step 3: lr = 0.1 * 0.1 = 0.01
    assert!((optimizer.lr() - 0.01).abs() < 1e-6);

    // Next 3 epochs
    scheduler.step(&mut optimizer);
    scheduler.step(&mut optimizer);
    scheduler.step(&mut optimizer);
    // After step 6: lr = 0.01 * 0.1 = 0.001
    assert!((optimizer.lr() - 0.001).abs() < 1e-6);
}

#[test]
fn test_exponential_lr() {
    let mut optimizer = MockOptimizer::new(0.1);
    let mut scheduler = ExponentialLR::new(0.9);

    scheduler.step(&mut optimizer);
    assert!((optimizer.lr() - 0.09).abs() < 1e-6);

    scheduler.step(&mut optimizer);
    assert!((optimizer.lr() - 0.081).abs() < 1e-6);
}

#[test]
fn test_cosine_annealing() {
    let mut optimizer = MockOptimizer::new(0.1);
    let mut scheduler = CosineAnnealingLR::new(10);

    // At epoch 0 (before step): lr = 0.1
    scheduler.step(&mut optimizer);
    // At epoch 1: should be close to initial (cosine starts at 1)
    assert!(optimizer.lr() < 0.1);
    assert!(optimizer.lr() > 0.09);

    // At epoch 5 (halfway): should be around 0.05
    for _ in 0..4 {
        scheduler.step(&mut optimizer);
    }
    assert!((optimizer.lr() - 0.05).abs() < 0.01);

    // At epoch 10: should be close to 0
    for _ in 0..5 {
        scheduler.step(&mut optimizer);
    }
    assert!(optimizer.lr() < 0.01);
}

#[test]
fn test_linear_warmup() {
    let mut optimizer = MockOptimizer::new(0.1);
    let mut scheduler = LinearWarmup::new(5);

    // During warmup
    scheduler.step(&mut optimizer);
    assert!((optimizer.lr() - 0.02).abs() < 1e-6); // 0.1 * 1/5

    scheduler.step(&mut optimizer);
    assert!((optimizer.lr() - 0.04).abs() < 1e-6); // 0.1 * 2/5

    // After warmup
    for _ in 0..3 {
        scheduler.step(&mut optimizer);
    }
    assert!((optimizer.lr() - 0.1).abs() < 1e-6);

    scheduler.step(&mut optimizer);
    assert!((optimizer.lr() - 0.1).abs() < 1e-6); // Stays at initial
}

#[test]
fn test_warmup_cosine() {
    let mut optimizer = MockOptimizer::new(0.1);
    let mut scheduler = WarmupCosineScheduler::new(5, 20);

    // Warmup phase
    scheduler.step(&mut optimizer);
    assert!((optimizer.lr() - 0.02).abs() < 1e-6);

    // Complete warmup
    for _ in 0..4 {
        scheduler.step(&mut optimizer);
    }
    assert!((optimizer.lr() - 0.1).abs() < 1e-6);

    // Decay phase starts
    scheduler.step(&mut optimizer);
    assert!(optimizer.lr() < 0.1);
}

#[test]
fn test_reduce_on_plateau() {
    let mut optimizer = MockOptimizer::new(0.1);
    let mut scheduler = ReduceLROnPlateau::new(PlateauMode::Min, 0.1, 3);

    // Improving
    scheduler.step_with_metric(&mut optimizer, 1.0);
    assert!((optimizer.lr() - 0.1).abs() < 1e-6);

    scheduler.step_with_metric(&mut optimizer, 0.9);
    assert!((optimizer.lr() - 0.1).abs() < 1e-6);

    // Plateau (no improvement for 3 epochs)
    scheduler.step_with_metric(&mut optimizer, 0.9);
    scheduler.step_with_metric(&mut optimizer, 0.9);
    scheduler.step_with_metric(&mut optimizer, 0.9);

    // LR should be reduced
    assert!((optimizer.lr() - 0.01).abs() < 1e-6);
}

#[test]
fn test_reduce_on_plateau_max_mode() {
    let mut optimizer = MockOptimizer::new(0.1);
    let mut scheduler = ReduceLROnPlateau::new(PlateauMode::Max, 0.5, 2);

    // Improving
    scheduler.step_with_metric(&mut optimizer, 0.5);
    scheduler.step_with_metric(&mut optimizer, 0.6);
    assert!((optimizer.lr() - 0.1).abs() < 1e-6);

    // Plateau
    scheduler.step_with_metric(&mut optimizer, 0.6);
    scheduler.step_with_metric(&mut optimizer, 0.6);

    // LR should be reduced
    assert!((optimizer.lr() - 0.05).abs() < 1e-6);
}

// Additional tests for coverage

#[test]
fn test_step_lr_with_lr() {
    let mut optimizer = MockOptimizer::new(0.1);
    let mut scheduler = StepLR::with_lr(0.2, 2, 0.5);

    assert_eq!(scheduler.get_lr(), 0.2);
    assert_eq!(scheduler.last_epoch(), 0);

    scheduler.step(&mut optimizer);
    assert_eq!(scheduler.last_epoch(), 1);
    scheduler.step(&mut optimizer);
    // After 2 steps: 0.2 * 0.5 = 0.1
    assert!((scheduler.get_lr() - 0.1).abs() < 1e-6);
}

#[test]
fn test_exponential_lr_with_lr() {
    let mut optimizer = MockOptimizer::new(0.1);
    let mut scheduler = ExponentialLR::with_lr(0.5, 0.8);

    assert_eq!(scheduler.get_lr(), 0.5);
    assert_eq!(scheduler.last_epoch(), 0);

    scheduler.step(&mut optimizer);
    assert!((scheduler.get_lr() - 0.4).abs() < 1e-6);
    assert_eq!(scheduler.last_epoch(), 1);
}

#[test]
fn test_cosine_annealing_with_min_lr() {
    let mut optimizer = MockOptimizer::new(0.1);
    let mut scheduler = CosineAnnealingLR::with_min_lr(10, 0.01);

    scheduler.step(&mut optimizer);
    assert!(scheduler.get_lr() > 0.01);
    assert!(scheduler.get_lr() < 0.1);
}

#[test]
fn test_cosine_annealing_with_lr() {
    let mut optimizer = MockOptimizer::new(0.05);
    let mut scheduler = CosineAnnealingLR::with_lr(0.2, 10, 0.02);

    assert_eq!(scheduler.get_lr(), 0.2);
    scheduler.step(&mut optimizer);
    // Should use initial_lr of 0.2, not optimizer's 0.05
    assert!(scheduler.get_lr() < 0.2);
    assert!(scheduler.get_lr() > 0.02);
}

#[test]
fn test_linear_warmup_with_lr() {
    let mut optimizer = MockOptimizer::new(0.05);
    let mut scheduler = LinearWarmup::with_lr(0.2, 4);

    assert_eq!(scheduler.get_lr(), 0.0); // before any step
    scheduler.step(&mut optimizer);
    assert!((scheduler.get_lr() - 0.05).abs() < 1e-6); // 0.2 * 1/4
    assert_eq!(scheduler.last_epoch(), 1);
}

#[test]
fn test_warmup_cosine_with_min_lr() {
    let mut optimizer = MockOptimizer::new(0.1);
    let mut scheduler = WarmupCosineScheduler::with_min_lr(5, 20, 0.001);

    // Complete warmup
    for _ in 0..5 {
        scheduler.step(&mut optimizer);
    }
    assert!((scheduler.get_lr() - 0.1).abs() < 1e-6);

    // Start decay
    scheduler.step(&mut optimizer);
    assert!(scheduler.get_lr() < 0.1);
    assert!(scheduler.get_lr() > 0.001);
    assert_eq!(scheduler.last_epoch(), 6);
}

#[test]
fn test_reduce_on_plateau_min_lr_builder() {
    let scheduler = ReduceLROnPlateau::new(PlateauMode::Min, 0.1, 3).min_lr(0.0001);
    assert!((scheduler.min_lr - 0.0001).abs() < 1e-8);
}

#[test]
fn test_reduce_on_plateau_threshold_builder() {
    let scheduler = ReduceLROnPlateau::new(PlateauMode::Min, 0.1, 3).threshold(0.001);
    assert!((scheduler.threshold - 0.001).abs() < 1e-8);
}

#[test]
fn test_reduce_on_plateau_step_without_metric() {
    let mut optimizer = MockOptimizer::new(0.1);
    let mut scheduler = ReduceLROnPlateau::new(PlateauMode::Min, 0.1, 2);

    // Call step without metric (should just increment epoch)
    scheduler.step(&mut optimizer);
    assert_eq!(scheduler.last_epoch(), 1);
    scheduler.step(&mut optimizer);
    assert_eq!(scheduler.last_epoch(), 2);
}

#[test]
fn test_reduce_on_plateau_min_lr_clamp() {
    let mut optimizer = MockOptimizer::new(0.001);
    let mut scheduler = ReduceLROnPlateau::new(PlateauMode::Min, 0.1, 1).min_lr(0.0005);

    // First metric establishes baseline
    scheduler.step_with_metric(&mut optimizer, 1.0);
    // No improvement triggers reduction
    scheduler.step_with_metric(&mut optimizer, 1.0);
    // LR should be clamped at min_lr
    assert!(scheduler.get_lr() >= 0.0005);
}

#[test]
fn test_step_lr_getters() {
    let scheduler = StepLR::with_lr(0.1, 5, 0.9);
    assert_eq!(scheduler.get_lr(), 0.1);
    assert_eq!(scheduler.last_epoch(), 0);
}

#[test]
fn test_exponential_lr_getters() {
    let scheduler = ExponentialLR::with_lr(0.1, 0.9);
    assert_eq!(scheduler.get_lr(), 0.1);
    assert_eq!(scheduler.last_epoch(), 0);
}

#[test]
fn test_cosine_annealing_getters() {
    let scheduler = CosineAnnealingLR::with_lr(0.1, 10, 0.01);
    assert_eq!(scheduler.get_lr(), 0.1);
    assert_eq!(scheduler.last_epoch(), 0);
}

#[test]
fn test_linear_warmup_getters() {
    let scheduler = LinearWarmup::with_lr(0.1, 5);
    assert_eq!(scheduler.get_lr(), 0.0);
    assert_eq!(scheduler.last_epoch(), 0);
}

#[test]
fn test_warmup_cosine_getters() {
    let scheduler = WarmupCosineScheduler::with_min_lr(5, 20, 0.01);
    assert_eq!(scheduler.get_lr(), 0.0);
    assert_eq!(scheduler.last_epoch(), 0);
}

#[test]
fn test_reduce_on_plateau_getters() {
    let scheduler = ReduceLROnPlateau::new(PlateauMode::Max, 0.5, 3);
    assert_eq!(scheduler.get_lr(), 0.0);
    assert_eq!(scheduler.last_epoch(), 0);
}

#[test]
fn test_plateau_mode_eq() {
    assert_eq!(PlateauMode::Min, PlateauMode::Min);
    assert_eq!(PlateauMode::Max, PlateauMode::Max);
    assert_ne!(PlateauMode::Min, PlateauMode::Max);
}

#[test]
fn test_scheduler_clone() {
    let scheduler = StepLR::with_lr(0.1, 5, 0.9);
    let cloned = scheduler.clone();
    assert_eq!(scheduler.get_lr(), cloned.get_lr());
}

#[test]
fn test_scheduler_debug() {
    let scheduler = StepLR::with_lr(0.1, 5, 0.9);
    let debug = format!("{scheduler:?}");
    assert!(debug.contains("StepLR"));
}

// ========================================================================
// Additional coverage tests
// ========================================================================

#[test]
fn test_exponential_lr_clone_debug() {
    let scheduler = ExponentialLR::with_lr(0.1, 0.95);
    let cloned = scheduler.clone();
    assert_eq!(scheduler.get_lr(), cloned.get_lr());
    let debug = format!("{scheduler:?}");
    assert!(debug.contains("ExponentialLR"));
}

#[test]
fn test_cosine_annealing_clone_debug() {
    let scheduler = CosineAnnealingLR::with_lr(0.1, 20, 0.001);
    let cloned = scheduler.clone();
    assert_eq!(scheduler.get_lr(), cloned.get_lr());
    let debug = format!("{scheduler:?}");
    assert!(debug.contains("CosineAnnealingLR"));
}

#[test]
fn test_linear_warmup_clone_debug() {
    let scheduler = LinearWarmup::with_lr(0.1, 10);
    let cloned = scheduler.clone();
    assert_eq!(scheduler.get_lr(), cloned.get_lr());
    let debug = format!("{scheduler:?}");
    assert!(debug.contains("LinearWarmup"));
}

#[test]
fn test_warmup_cosine_clone_debug() {
    let scheduler = WarmupCosineScheduler::with_min_lr(5, 50, 0.0001);
    let cloned = scheduler.clone();
    assert_eq!(scheduler.get_lr(), cloned.get_lr());
    let debug = format!("{scheduler:?}");
    assert!(debug.contains("WarmupCosineScheduler"));
}

#[test]
fn test_reduce_on_plateau_clone_debug() {
    let scheduler = ReduceLROnPlateau::new(PlateauMode::Min, 0.1, 5);
    let cloned = scheduler.clone();
    assert_eq!(scheduler.get_lr(), cloned.get_lr());
    let debug = format!("{scheduler:?}");
    assert!(debug.contains("ReduceLROnPlateau"));
}

#[test]
fn test_plateau_mode_debug() {
    assert!(format!("{:?}", PlateauMode::Min).contains("Min"));
    assert!(format!("{:?}", PlateauMode::Max).contains("Max"));
}

#[test]
fn test_plateau_mode_copy() {
    let mode = PlateauMode::Min;
    let copied = mode;
    assert_eq!(mode, copied);
}

#[test]
fn test_cosine_annealing_full_cycle() {
    let mut optimizer = MockOptimizer::new(0.1);
    let mut scheduler = CosineAnnealingLR::with_min_lr(10, 0.01);

    // Go through full cycle
    for _ in 0..10 {
        scheduler.step(&mut optimizer);
    }
    // At end of cycle, should be at or near min_lr
    assert!(scheduler.get_lr() <= 0.02);
}

#[test]
fn test_linear_warmup_after_warmup_complete() {
    let mut optimizer = MockOptimizer::new(0.1);
    let mut scheduler = LinearWarmup::with_lr(0.1, 3);

    // Complete warmup
    scheduler.step(&mut optimizer);
    scheduler.step(&mut optimizer);
    scheduler.step(&mut optimizer);
    assert!((scheduler.get_lr() - 0.1).abs() < 1e-6);

    // Continue stepping - should stay at initial_lr
    scheduler.step(&mut optimizer);
    scheduler.step(&mut optimizer);
    assert!((scheduler.get_lr() - 0.1).abs() < 1e-6);
    assert_eq!(scheduler.last_epoch(), 5);
}

include!("tests_part_02.rs");
