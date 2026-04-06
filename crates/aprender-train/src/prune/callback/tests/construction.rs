//! Construction, sparsity, progress, clone, and config access tests
//!
//! Tests for callback construction, sparsity tracking, progress calculation,
//! cloning, and configuration access.

use crate::prune::callback::PruningCallback;
use crate::prune::config::PruningConfig;
use crate::prune::schedule::PruningSchedule;
use crate::train::callback::TrainerCallback;

fn default_config() -> PruningConfig {
    PruningConfig::default()
}

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
// Construction Tests
// =============================================================================

#[test]
fn test_callback_new_default() {
    // TEST_ID: CB-001
    let callback = PruningCallback::new(default_config());
    assert!(callback.is_enabled(), "CB-001 FALSIFIED: Default should be enabled");
    assert_eq!(
        callback.current_sparsity(),
        0.0,
        "CB-001 FALSIFIED: Initial sparsity should be 0.0"
    );
    assert_eq!(
        callback.parameters_pruned(),
        0,
        "CB-001 FALSIFIED: Initial parameters pruned should be 0"
    );
}

#[test]
fn test_callback_enabled_toggle() {
    // TEST_ID: CB-002
    let mut callback = PruningCallback::new(default_config());
    assert!(callback.is_enabled());
    callback.set_enabled(false);
    assert!(!callback.is_enabled(), "CB-002 FALSIFIED: Should be disabled");
    callback.set_enabled(true);
    assert!(callback.is_enabled(), "CB-002 FALSIFIED: Should be enabled again");
}

#[test]
fn test_callback_name() {
    // TEST_ID: CB-003
    let callback = PruningCallback::new(default_config());
    assert_eq!(
        callback.name(),
        "PruningCallback",
        "CB-003 FALSIFIED: Name should be PruningCallback"
    );
}

// =============================================================================
// Sparsity Tracking Tests
// =============================================================================

#[test]
fn test_current_sparsity_starts_at_zero() {
    // TEST_ID: CB-010
    let callback = PruningCallback::new(gradual_config());
    assert_eq!(callback.current_sparsity(), 0.0, "CB-010 FALSIFIED: Initial sparsity must be 0.0");
}

#[test]
fn test_set_current_sparsity_clamps_to_bounds() {
    // TEST_ID: CB-011
    let mut callback = PruningCallback::new(default_config());

    callback.set_current_sparsity(0.5);
    assert_eq!(callback.current_sparsity(), 0.5, "CB-011 FALSIFIED: Should set sparsity to 0.5");

    callback.set_current_sparsity(-0.5);
    assert_eq!(callback.current_sparsity(), 0.0, "CB-011 FALSIFIED: Should clamp negative to 0.0");

    callback.set_current_sparsity(1.5);
    assert_eq!(callback.current_sparsity(), 1.0, "CB-011 FALSIFIED: Should clamp >1.0 to 1.0");
}

#[test]
fn test_target_sparsity_from_config() {
    // TEST_ID: CB-012
    let config = PruningConfig::default().with_target_sparsity(0.7);
    let callback = PruningCallback::new(config);
    assert!(
        (callback.target_sparsity() - 0.7).abs() < 1e-6,
        "CB-012 FALSIFIED: Target sparsity should match config"
    );
}

// =============================================================================
// Progress Tracking Tests
// =============================================================================

#[test]
fn test_progress_zero_when_no_pruning() {
    // TEST_ID: CB-020
    let config = PruningConfig::default().with_target_sparsity(0.5);
    let callback = PruningCallback::new(config);
    assert_eq!(
        callback.progress(),
        0.0,
        "CB-020 FALSIFIED: Progress should be 0.0 when no pruning has occurred"
    );
}

#[test]
fn test_progress_increases_with_sparsity() {
    // TEST_ID: CB-021
    let config = PruningConfig::default().with_target_sparsity(0.5);
    let mut callback = PruningCallback::new(config);

    callback.set_current_sparsity(0.25);
    assert!(
        (callback.progress() - 0.5).abs() < 1e-6,
        "CB-021 FALSIFIED: Progress should be 0.5 at half target sparsity"
    );

    callback.set_current_sparsity(0.5);
    assert!(
        (callback.progress() - 1.0).abs() < 1e-6,
        "CB-021 FALSIFIED: Progress should be 1.0 at target sparsity"
    );
}

#[test]
fn test_progress_clamped_to_one() {
    // TEST_ID: CB-022
    let config = PruningConfig::default().with_target_sparsity(0.5);
    let mut callback = PruningCallback::new(config);
    callback.set_current_sparsity(0.6); // Exceeds target
    assert_eq!(callback.progress(), 1.0, "CB-022 FALSIFIED: Progress should be clamped to 1.0");
}

#[test]
fn test_progress_one_when_zero_target() {
    // TEST_ID: CB-023
    // Edge case: zero target sparsity
    let config = PruningConfig::default().with_target_sparsity(0.0);
    let callback = PruningCallback::new(config);
    assert_eq!(
        callback.progress(),
        1.0,
        "CB-023 FALSIFIED: Progress should be 1.0 when target is 0"
    );
}

// =============================================================================
// Clone Tests
// =============================================================================

#[test]
fn test_callback_clone() {
    // TEST_ID: CB-060
    let mut callback = PruningCallback::new(gradual_config());
    callback.set_current_sparsity(0.3);
    callback.set_enabled(false);

    let cloned = callback.clone();
    assert_eq!(
        callback.current_sparsity(),
        cloned.current_sparsity(),
        "CB-060 FALSIFIED: Cloned sparsity should match"
    );
    assert_eq!(
        callback.is_enabled(),
        cloned.is_enabled(),
        "CB-060 FALSIFIED: Cloned enabled state should match"
    );
}

// =============================================================================
// Config Access Tests
// =============================================================================

#[test]
fn test_config_access() {
    // TEST_ID: CB-070
    let config = gradual_config();
    let callback = PruningCallback::new(config.clone());
    assert_eq!(
        callback.config().target_sparsity(),
        config.target_sparsity(),
        "CB-070 FALSIFIED: Config access should return the config"
    );
}

#[test]
fn test_schedule_access() {
    // TEST_ID: CB-071
    let callback = PruningCallback::new(gradual_config());
    match callback.schedule() {
        PruningSchedule::Gradual { end_step, .. } => {
            assert_eq!(*end_step, 1000, "CB-071 FALSIFIED: Schedule should be accessible");
        }
        _ => panic!("CB-071 FALSIFIED: Expected Gradual schedule"),
    }
}
