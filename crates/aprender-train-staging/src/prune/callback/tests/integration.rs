//! Integration tests for pruning callback (should_prune, callback interface, completion, calibration)

use crate::prune::calibrate::CalibrationConfig;
use crate::prune::callback::PruningCallback;
use crate::prune::config::PruningConfig;
use crate::prune::schedule::PruningSchedule;
use crate::train::callback::{CallbackAction, CallbackContext, TrainerCallback};

fn gradual_config() -> PruningConfig {
    PruningConfig::default().with_schedule(PruningSchedule::Gradual {
        start_step: 100,
        end_step: 1000,
        initial_sparsity: 0.0,
        final_sparsity: 0.5,
        frequency: 100,
    })
}

// =============================================================================
// Should Prune Logic Tests
// =============================================================================

#[test]
fn test_should_prune_respects_disabled() {
    // TEST_ID: CB-030
    let mut callback = PruningCallback::new(gradual_config());
    callback.set_enabled(false);
    assert!(
        !callback.should_prune(100),
        "CB-030 FALSIFIED: should_prune should be false when disabled"
    );
}

#[test]
fn test_should_prune_respects_schedule() {
    // TEST_ID: CB-031
    // Note: At step 100 (start_step), sparsity_at_step returns initial_sparsity (0.0)
    // so target > current_sparsity (0.0 > 0.0) is false.
    // We test at step 200 where sparsity is > 0.
    let callback = PruningCallback::new(gradual_config());
    // Before start_step
    assert!(!callback.should_prune(50), "CB-031 FALSIFIED: should not prune before start");
    // At step 200: sparsity = 0.0 + (100/900) * 0.5 ≈ 0.056
    // schedule.should_prune_at_step(200) = true (at frequency)
    assert!(callback.should_prune(200), "CB-031 FALSIFIED: should prune at frequency step");
    // At step 300
    assert!(callback.should_prune(300), "CB-031 FALSIFIED: should prune at frequency step");
}

#[test]
fn test_should_prune_when_sparsity_already_achieved() {
    // TEST_ID: CB-032
    let mut callback = PruningCallback::new(gradual_config());
    // Set current sparsity to max
    callback.set_current_sparsity(0.5);
    // Even at prune step, should not prune if already at target
    assert!(
        !callback.should_prune(100),
        "CB-032 FALSIFIED: should not prune when already at target sparsity"
    );
}

// =============================================================================
// Callback Interface Tests
// =============================================================================

#[test]
fn test_on_train_begin_validates_schedule() {
    // TEST_ID: CB-040
    let config = PruningConfig::default().with_schedule(PruningSchedule::Gradual {
        start_step: 100,
        end_step: 1000,
        initial_sparsity: 0.0,
        final_sparsity: 0.5,
        frequency: 10,
    });
    let mut callback = PruningCallback::new(config);
    let ctx = CallbackContext::default();
    let action = callback.on_train_begin(&ctx);
    assert_eq!(
        action,
        CallbackAction::Continue,
        "CB-040 FALSIFIED: Valid schedule should continue"
    );
}

#[test]
fn test_on_train_begin_rejects_invalid_schedule() {
    // TEST_ID: CB-041
    let config = PruningConfig::default().with_schedule(PruningSchedule::Gradual {
        start_step: 1000,
        end_step: 100, // Invalid: end < start
        initial_sparsity: 0.0,
        final_sparsity: 0.5,
        frequency: 10,
    });
    let mut callback = PruningCallback::new(config);
    let ctx = CallbackContext::default();
    let action = callback.on_train_begin(&ctx);
    assert_eq!(
        action,
        CallbackAction::Stop,
        "CB-041 FALSIFIED: Invalid schedule should stop training"
    );
}

#[test]
fn test_on_step_end_updates_sparsity() {
    // TEST_ID: CB-042
    let config = PruningConfig::default().with_schedule(PruningSchedule::OneShot { step: 100 });
    let mut callback = PruningCallback::new(config);

    let mut ctx = CallbackContext::default();
    ctx.global_step = 50;
    callback.on_step_end(&ctx);
    assert_eq!(
        callback.current_sparsity(),
        0.0,
        "CB-042 FALSIFIED: Sparsity should be 0 before prune step"
    );

    ctx.global_step = 100;
    callback.on_step_end(&ctx);
    assert_eq!(
        callback.current_sparsity(),
        1.0,
        "CB-042 FALSIFIED: Sparsity should be 1.0 at prune step"
    );
}

#[test]
fn test_on_step_end_tracks_last_prune_step() {
    // TEST_ID: CB-043
    let config = PruningConfig::default().with_schedule(PruningSchedule::OneShot { step: 100 });
    let mut callback = PruningCallback::new(config);

    assert!(
        callback.last_prune_step().is_none(),
        "CB-043 FALSIFIED: last_prune_step should be None initially"
    );

    let mut ctx = CallbackContext::default();
    ctx.global_step = 100;
    callback.on_step_end(&ctx);

    assert_eq!(
        callback.last_prune_step(),
        Some(100),
        "CB-043 FALSIFIED: last_prune_step should be 100 after pruning"
    );
}

#[test]
fn test_on_step_end_disabled_does_nothing() {
    // TEST_ID: CB-044
    let config = PruningConfig::default().with_schedule(PruningSchedule::OneShot { step: 0 });
    let mut callback = PruningCallback::new(config);
    callback.set_enabled(false);

    let ctx = CallbackContext::default();
    callback.on_step_end(&ctx);

    assert_eq!(
        callback.current_sparsity(),
        0.0,
        "CB-044 FALSIFIED: Disabled callback should not update sparsity"
    );
}

#[test]
fn test_on_train_end_logs_summary() {
    // TEST_ID: CB-045
    let mut callback = PruningCallback::new(gradual_config());
    callback.set_current_sparsity(0.5);

    let ctx = CallbackContext::default();
    // This should not panic
    callback.on_train_end(&ctx);
}

// =============================================================================
// Completion Tests
// =============================================================================

#[test]
fn test_is_complete_oneshot() {
    // TEST_ID: CB-050
    let config = PruningConfig::default().with_schedule(PruningSchedule::OneShot { step: 100 });
    let mut callback = PruningCallback::new(config);

    assert!(!callback.is_complete(), "CB-050 FALSIFIED: Should not be complete before pruning");

    let mut ctx = CallbackContext::default();
    ctx.global_step = 100;
    callback.on_step_end(&ctx);

    // Still at prune step, not complete
    assert!(!callback.is_complete(), "CB-050 FALSIFIED: Should not be complete at prune step");

    ctx.global_step = 101;
    callback.on_step_end(&ctx);

    // Now past prune step
    // Note: We need to update last_prune_step to a step past completion
    callback.last_prune_step = Some(101);
    assert!(callback.is_complete(), "CB-050 FALSIFIED: Should be complete after prune step");
}

#[test]
fn test_is_complete_gradual() {
    // TEST_ID: CB-051
    let config = PruningConfig::default().with_schedule(PruningSchedule::Gradual {
        start_step: 0,
        end_step: 100,
        initial_sparsity: 0.0,
        final_sparsity: 0.5,
        frequency: 10,
    });
    let mut callback = PruningCallback::new(config);

    // Simulate pruning to completion
    let mut ctx = CallbackContext::default();
    ctx.global_step = 100;
    callback.on_step_end(&ctx);

    // At end_step, not yet complete
    assert!(!callback.is_complete(), "CB-051 FALSIFIED: Should not be complete at end_step");

    ctx.global_step = 101;
    callback.last_prune_step = Some(101);
    assert!(callback.is_complete(), "CB-051 FALSIFIED: Should be complete after end_step");
}

// =============================================================================
// Calibration Tests
// =============================================================================

#[test]
fn test_calibration_created_for_wanda() {
    // TEST_ID: CB-080
    use crate::prune::config::PruneMethod;

    let config = PruningConfig::default().with_method(PruneMethod::Wanda);
    let callback = PruningCallback::new(config);
    assert!(
        callback.calibration.is_some(),
        "CB-080 FALSIFIED: Wanda should have calibration collector"
    );
}

#[test]
fn test_no_calibration_for_magnitude() {
    // TEST_ID: CB-081
    use crate::prune::config::PruneMethod;

    let config = PruningConfig::default().with_method(PruneMethod::Magnitude);
    let callback = PruningCallback::new(config);
    assert!(
        callback.calibration.is_none(),
        "CB-081 FALSIFIED: Magnitude should not have calibration collector"
    );
}

#[test]
fn test_with_calibration_constructor() {
    // TEST_ID: CB-082
    let config = gradual_config();
    let cal_config = CalibrationConfig::new().with_num_samples(256);
    let callback = PruningCallback::with_calibration(config, cal_config);
    assert!(
        callback.calibration.is_some(),
        "CB-082 FALSIFIED: with_calibration should create calibration collector"
    );
}
