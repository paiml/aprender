//! Validation and serialization tests for gradual pruning schedule.

use crate::prune::schedule::PruningSchedule;

// =========================================================================
// Validation Tests
// =========================================================================

#[test]
fn test_validate_gradual_end_after_start() {
    // TEST_ID: SCHED-031
    let schedule = PruningSchedule::Gradual {
        start_step: 100,
        end_step: 50, // Invalid: end < start
        initial_sparsity: 0.0,
        final_sparsity: 0.5,
        frequency: 10,
    };
    assert!(
        schedule.validate().is_err(),
        "SCHED-031 FALSIFIED: Gradual with end < start should be invalid"
    );
}

#[test]
fn test_validate_gradual_invalid_initial_sparsity() {
    // TEST_ID: SCHED-032
    let schedule = PruningSchedule::Gradual {
        start_step: 0,
        end_step: 100,
        initial_sparsity: -0.1, // Invalid
        final_sparsity: 0.5,
        frequency: 10,
    };
    assert!(
        schedule.validate().is_err(),
        "SCHED-032 FALSIFIED: Negative initial_sparsity should be invalid"
    );
}

#[test]
fn test_validate_gradual_invalid_final_sparsity() {
    // TEST_ID: SCHED-033
    let schedule = PruningSchedule::Gradual {
        start_step: 0,
        end_step: 100,
        initial_sparsity: 0.0,
        final_sparsity: 1.5, // Invalid
        frequency: 10,
    };
    assert!(
        schedule.validate().is_err(),
        "SCHED-033 FALSIFIED: final_sparsity > 1.0 should be invalid"
    );
}

#[test]
fn test_validate_gradual_error_message_end_step() {
    // TEST_ID: SCHED-065
    // Test specific error message content
    let schedule = PruningSchedule::Gradual {
        start_step: 100,
        end_step: 100, // Equal, not greater
        initial_sparsity: 0.0,
        final_sparsity: 0.5,
        frequency: 10,
    };
    let err = schedule.validate().unwrap_err();
    assert!(err.contains("end_step"));
    assert!(err.contains("start_step"));
}

#[test]
fn test_validate_gradual_error_message_initial_over_one() {
    // TEST_ID: SCHED-066
    let schedule = PruningSchedule::Gradual {
        start_step: 0,
        end_step: 100,
        initial_sparsity: 1.5, // > 1.0
        final_sparsity: 0.5,
        frequency: 10,
    };
    let err = schedule.validate().unwrap_err();
    assert!(err.contains("initial_sparsity"));
    assert!(err.contains("1.5"));
}

#[test]
fn test_validate_gradual_valid() {
    // TEST_ID: SCHED-067
    let schedule = PruningSchedule::Gradual {
        start_step: 0,
        end_step: 100,
        initial_sparsity: 0.0,
        final_sparsity: 1.0, // Valid boundary
        frequency: 10,
    };
    assert!(schedule.validate().is_ok());
}

#[test]
fn test_validate_gradual_final_negative() {
    // TEST_ID: SCHED-080
    let schedule = PruningSchedule::Gradual {
        start_step: 0,
        end_step: 100,
        initial_sparsity: 0.0,
        final_sparsity: -0.1,
        frequency: 10,
    };
    assert!(schedule.validate().is_err());
}

#[test]
fn test_validate_gradual_initial_exactly_one() {
    // TEST_ID: SCHED-081
    let schedule = PruningSchedule::Gradual {
        start_step: 0,
        end_step: 100,
        initial_sparsity: 1.0, // Valid boundary
        final_sparsity: 1.0,
        frequency: 10,
    };
    assert!(schedule.validate().is_ok());
}

// =========================================================================
// Serialization Tests
// =========================================================================

#[test]
fn test_serialize_gradual() {
    // TEST_ID: SCHED-051
    let schedule = PruningSchedule::Gradual {
        start_step: 100,
        end_step: 1000,
        initial_sparsity: 0.0,
        final_sparsity: 0.5,
        frequency: 10,
    };
    let json = serde_json::to_string(&schedule).expect("JSON serialization should succeed");
    assert!(
        json.contains("gradual"),
        "SCHED-051 FALSIFIED: Gradual should serialize with type=gradual"
    );
    let deserialized: PruningSchedule =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");
    assert_eq!(schedule, deserialized, "SCHED-051 FALSIFIED: Deserialized should match original");
}

#[test]
fn test_deserialize_from_yaml() {
    // TEST_ID: SCHED-053
    let yaml = r"
type: gradual
start_step: 100
end_step: 1000
initial_sparsity: 0.0
final_sparsity: 0.5
frequency: 10
";
    let schedule: PruningSchedule = serde_yaml::from_str(yaml).expect("operation should succeed");
    match schedule {
        PruningSchedule::Gradual {
            start_step,
            end_step,
            initial_sparsity,
            final_sparsity,
            frequency,
        } => {
            assert_eq!(start_step, 100);
            assert_eq!(end_step, 1000);
            assert!((initial_sparsity - 0.0).abs() < 1e-6);
            assert!((final_sparsity - 0.5).abs() < 1e-6);
            assert_eq!(frequency, 10);
        }
        _ => panic!("SCHED-053 FALSIFIED: Should deserialize to Gradual variant"),
    }
}
