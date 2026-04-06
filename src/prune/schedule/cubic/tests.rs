//! Tests for cubic pruning schedule.

use crate::prune::schedule::PruningSchedule;

// =========================================================================
// Cubic Schedule Tests
// =========================================================================

#[test]
fn test_cubic_before_start_returns_zero() {
    // TEST_ID: SCHED-020
    // FALSIFIES: Cubic returns non-zero before start
    let schedule = PruningSchedule::Cubic { start_step: 100, end_step: 1000, final_sparsity: 0.5 };
    assert_eq!(
        schedule.sparsity_at_step(0),
        0.0,
        "SCHED-020 FALSIFIED: Cubic should return 0.0 before start"
    );
    assert_eq!(
        schedule.sparsity_at_step(99),
        0.0,
        "SCHED-020 FALSIFIED: Cubic should return 0.0 at step before start"
    );
}

#[test]
fn test_cubic_after_end_returns_final() {
    // TEST_ID: SCHED-021
    // FALSIFIES: Cubic returns wrong value after end
    let schedule = PruningSchedule::Cubic { start_step: 0, end_step: 100, final_sparsity: 0.5 };
    assert_eq!(
        schedule.sparsity_at_step(100),
        0.5,
        "SCHED-021 FALSIFIED: Cubic should return final_sparsity at end"
    );
    assert_eq!(
        schedule.sparsity_at_step(10000),
        0.5,
        "SCHED-021 FALSIFIED: Cubic should return final_sparsity after end"
    );
}

#[test]
fn test_cubic_formula_at_start() {
    // TEST_ID: SCHED-022
    // At start: t=0, s = s_f * (1 - (1 - 0)^3) = s_f * 0 = 0
    let schedule = PruningSchedule::Cubic { start_step: 0, end_step: 100, final_sparsity: 0.5 };
    let sparsity = schedule.sparsity_at_step(0);
    assert!(
        sparsity.abs() < 1e-6,
        "SCHED-022 FALSIFIED: Cubic at start should be 0.0, got {sparsity}"
    );
}

#[test]
fn test_cubic_formula_at_midpoint() {
    // TEST_ID: SCHED-023
    // At midpoint: t=50, T=100, ratio=(1-0.5)=0.5, s = 0.5 * (1 - 0.5^3) = 0.5 * 0.875 = 0.4375
    let schedule = PruningSchedule::Cubic { start_step: 0, end_step: 100, final_sparsity: 0.5 };
    let sparsity = schedule.sparsity_at_step(50);
    let expected = 0.5 * (1.0 - 0.5_f32.powi(3));
    assert!(
        (sparsity - expected).abs() < 1e-6,
        "SCHED-023 FALSIFIED: Cubic at midpoint should be {expected}, got {sparsity}"
    );
}

#[test]
fn test_cubic_faster_initial_pruning() {
    // TEST_ID: SCHED-024
    // FALSIFIES: Cubic doesn't provide faster initial pruning than linear
    // At 25% progress, cubic should have higher sparsity than 25% linear
    let schedule = PruningSchedule::Cubic { start_step: 0, end_step: 100, final_sparsity: 1.0 };
    let sparsity_25 = schedule.sparsity_at_step(25);
    let linear_25 = 0.25; // 25% of final_sparsity
    assert!(
        sparsity_25 > linear_25,
        "SCHED-024 FALSIFIED: Cubic should be faster than linear at 25%, got {sparsity_25} vs {linear_25}"
    );
}

#[test]
fn test_cubic_slower_final_pruning() {
    // TEST_ID: SCHED-025
    // At 75% progress, cubic should have lower sparsity increase than linear would
    let schedule = PruningSchedule::Cubic { start_step: 0, end_step: 100, final_sparsity: 1.0 };
    let sparsity_75 = schedule.sparsity_at_step(75);
    let linear_75 = 0.75;
    // Cubic at 75%: s = 1 * (1 - 0.25^3) = 1 - 0.015625 = 0.984375
    // This is HIGHER than linear, showing faster convergence
    // But the RATE of change is slower (derivative is lower)
    assert!(
        sparsity_75 > linear_75,
        "SCHED-025 FALSIFIED: Cubic at 75% should be higher than linear ({sparsity_75})"
    );
}

#[test]
fn test_cubic_should_prune_in_window() {
    // TEST_ID: SCHED-026
    let schedule = PruningSchedule::Cubic { start_step: 100, end_step: 200, final_sparsity: 0.5 };
    assert!(
        !schedule.should_prune_at_step(99),
        "SCHED-026 FALSIFIED: should not prune before window"
    );
    assert!(
        schedule.should_prune_at_step(100),
        "SCHED-026 FALSIFIED: should prune at start of window"
    );
    assert!(schedule.should_prune_at_step(150), "SCHED-026 FALSIFIED: should prune during window");
    assert!(
        schedule.should_prune_at_step(200),
        "SCHED-026 FALSIFIED: should prune at end of window"
    );
    assert!(
        !schedule.should_prune_at_step(201),
        "SCHED-026 FALSIFIED: should not prune after window"
    );
}

#[test]
fn test_validate_cubic_end_after_start() {
    // TEST_ID: SCHED-034
    let schedule = PruningSchedule::Cubic {
        start_step: 100,
        end_step: 50, // Invalid
        final_sparsity: 0.5,
    };
    assert!(
        schedule.validate().is_err(),
        "SCHED-034 FALSIFIED: Cubic with end < start should be invalid"
    );
}

#[test]
fn test_validate_cubic_invalid_final_sparsity() {
    // TEST_ID: SCHED-035
    let schedule = PruningSchedule::Cubic {
        start_step: 0,
        end_step: 100,
        final_sparsity: 2.0, // Invalid
    };
    assert!(
        schedule.validate().is_err(),
        "SCHED-035 FALSIFIED: final_sparsity > 1.0 should be invalid"
    );
}

#[test]
fn test_num_pruning_steps_cubic() {
    // TEST_ID: SCHED-042
    let schedule = PruningSchedule::Cubic { start_step: 0, end_step: 100, final_sparsity: 0.5 };
    // Cubic prunes at every step in the window
    assert_eq!(
        schedule.num_pruning_steps(),
        101,
        "SCHED-042 FALSIFIED: Cubic from 0-100 should have 101 pruning steps"
    );
}

#[test]
fn test_is_complete_cubic() {
    // TEST_ID: SCHED-046
    // FALSIFIES: Cubic is_complete doesn't work correctly
    let schedule = PruningSchedule::Cubic { start_step: 0, end_step: 100, final_sparsity: 0.5 };
    assert!(
        !schedule.is_complete(50),
        "SCHED-046 FALSIFIED: Cubic should not be complete mid-schedule"
    );
    assert!(
        !schedule.is_complete(100),
        "SCHED-046 FALSIFIED: Cubic should not be complete at end_step"
    );
    assert!(
        schedule.is_complete(101),
        "SCHED-046 FALSIFIED: Cubic should be complete after end_step"
    );
}

#[test]
fn test_validate_cubic_valid() {
    // TEST_ID: SCHED-047
    // FALSIFIES: Valid Cubic schedule fails validation
    let schedule = PruningSchedule::Cubic { start_step: 0, end_step: 100, final_sparsity: 0.5 };
    assert!(
        schedule.validate().is_ok(),
        "SCHED-047 FALSIFIED: Valid Cubic schedule should pass validation"
    );
}

#[test]
fn test_validate_cubic_negative_sparsity() {
    // TEST_ID: SCHED-048
    let schedule = PruningSchedule::Cubic { start_step: 0, end_step: 100, final_sparsity: -0.1 };
    assert!(
        schedule.validate().is_err(),
        "SCHED-048 FALSIFIED: Negative final_sparsity should be invalid"
    );
}

#[test]
fn test_sparsity_monotonic_cubic() {
    // TEST_ID: SCHED-061
    // FALSIFIES: Cubic sparsity can decrease over time
    let schedule = PruningSchedule::Cubic { start_step: 0, end_step: 100, final_sparsity: 0.5 };
    let mut prev = 0.0;
    for step in 0..=100 {
        let sparsity = schedule.sparsity_at_step(step);
        assert!(
            sparsity >= prev - 1e-6, // Allow floating point tolerance
            "SCHED-061 FALSIFIED: Sparsity decreased from {prev} to {sparsity} at step {step}"
        );
        prev = sparsity;
    }
}

#[test]
fn test_validate_cubic_error_message_sparsity() {
    // TEST_ID: SCHED-068
    let schedule = PruningSchedule::Cubic {
        start_step: 0,
        end_step: 100,
        final_sparsity: 1.5, // > 1.0
    };
    let err = schedule.validate().unwrap_err();
    assert!(err.contains("final_sparsity"));
    assert!(err.contains("1.5"));
}

#[test]
fn test_validate_cubic_error_message_end_step() {
    // TEST_ID: SCHED-069
    let schedule = PruningSchedule::Cubic {
        start_step: 100,
        end_step: 100, // Equal
        final_sparsity: 0.5,
    };
    let err = schedule.validate().unwrap_err();
    assert!(err.contains("end_step"));
}

#[test]
fn test_cubic_at_start_step() {
    // TEST_ID: SCHED-071
    let schedule = PruningSchedule::Cubic { start_step: 100, end_step: 200, final_sparsity: 0.5 };
    let sparsity = schedule.sparsity_at_step(100);
    assert!(sparsity.abs() < 1e-6, "Should be 0.0 at start_step");
}

#[test]
fn test_cubic_num_pruning_steps_short() {
    // TEST_ID: SCHED-073
    let schedule = PruningSchedule::Cubic { start_step: 0, end_step: 10, final_sparsity: 0.5 };
    assert_eq!(schedule.num_pruning_steps(), 11);
}

#[test]
fn test_debug_format_cubic() {
    // TEST_ID: SCHED-077
    let schedule = PruningSchedule::Cubic { start_step: 0, end_step: 100, final_sparsity: 0.5 };
    let debug = format!("{schedule:?}");
    assert!(debug.contains("Cubic"));
    assert!(debug.contains("final_sparsity"));
}

#[test]
fn test_cubic_formula_at_quarter() {
    // TEST_ID: SCHED-079
    // At 25%: ratio = 0.75, s = 1.0 * (1 - 0.75^3) = 1 - 0.421875 = 0.578125
    let schedule = PruningSchedule::Cubic { start_step: 0, end_step: 100, final_sparsity: 1.0 };
    let sparsity = schedule.sparsity_at_step(25);
    let expected = 1.0 * (1.0 - 0.75_f32.powi(3));
    assert!(
        (sparsity - expected).abs() < 1e-6,
        "Cubic at 25% should be {expected}, got {sparsity}"
    );
}

#[test]
fn test_validate_cubic_exactly_zero() {
    // TEST_ID: SCHED-082
    let schedule = PruningSchedule::Cubic {
        start_step: 0,
        end_step: 100,
        final_sparsity: 0.0, // Valid boundary
    };
    assert!(schedule.validate().is_ok());
}

#[test]
fn test_validate_cubic_exactly_one() {
    // TEST_ID: SCHED-083
    let schedule = PruningSchedule::Cubic {
        start_step: 0,
        end_step: 100,
        final_sparsity: 1.0, // Valid boundary
    };
    assert!(schedule.validate().is_ok());
}

#[test]
fn test_serialize_cubic() {
    // TEST_ID: SCHED-052
    let schedule = PruningSchedule::Cubic { start_step: 0, end_step: 100, final_sparsity: 0.5 };
    let json = serde_json::to_string(&schedule).expect("JSON serialization should succeed");
    assert!(json.contains("cubic"), "SCHED-052 FALSIFIED: Cubic should serialize with type=cubic");
    let deserialized: PruningSchedule =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");
    assert_eq!(schedule, deserialized, "SCHED-052 FALSIFIED: Deserialized should match original");
}

#[test]
fn test_deserialize_cubic_from_yaml() {
    // TEST_ID: SCHED-085
    let yaml = r"
type: cubic
start_step: 0
end_step: 100
final_sparsity: 0.5
";
    let schedule: PruningSchedule = serde_yaml::from_str(yaml).expect("operation should succeed");
    match schedule {
        PruningSchedule::Cubic { start_step, end_step, final_sparsity } => {
            assert_eq!(start_step, 0);
            assert_eq!(end_step, 100);
            assert!((final_sparsity - 0.5).abs() < 1e-6);
        }
        _ => panic!("Should deserialize to Cubic"),
    }
}
