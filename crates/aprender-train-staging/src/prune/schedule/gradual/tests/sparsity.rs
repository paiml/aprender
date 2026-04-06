//! Sparsity computation and interpolation tests for gradual pruning schedule.

use crate::prune::schedule::PruningSchedule;

// =========================================================================
// Sparsity Computation Tests
// =========================================================================

#[test]
fn test_gradual_before_start_returns_initial() {
    // TEST_ID: SCHED-010
    // FALSIFIES: Gradual returns wrong value before start
    let schedule = PruningSchedule::Gradual {
        start_step: 100,
        end_step: 1000,
        initial_sparsity: 0.0,
        final_sparsity: 0.5,
        frequency: 10,
    };
    assert_eq!(
        schedule.sparsity_at_step(0),
        0.0,
        "SCHED-010 FALSIFIED: Gradual should return initial_sparsity before start"
    );
    assert_eq!(
        schedule.sparsity_at_step(99),
        0.0,
        "SCHED-010 FALSIFIED: Gradual should return initial_sparsity at step before start"
    );
}

#[test]
fn test_gradual_after_end_returns_final() {
    // TEST_ID: SCHED-011
    // FALSIFIES: Gradual returns wrong value after end
    let schedule = PruningSchedule::Gradual {
        start_step: 100,
        end_step: 1000,
        initial_sparsity: 0.0,
        final_sparsity: 0.5,
        frequency: 10,
    };
    assert_eq!(
        schedule.sparsity_at_step(1000),
        0.5,
        "SCHED-011 FALSIFIED: Gradual should return final_sparsity at end"
    );
    assert_eq!(
        schedule.sparsity_at_step(10000),
        0.5,
        "SCHED-011 FALSIFIED: Gradual should return final_sparsity after end"
    );
}

#[test]
fn test_gradual_linear_interpolation_midpoint() {
    // TEST_ID: SCHED-012
    // FALSIFIES: Gradual doesn't perform linear interpolation correctly
    let schedule = PruningSchedule::Gradual {
        start_step: 0,
        end_step: 100,
        initial_sparsity: 0.0,
        final_sparsity: 1.0,
        frequency: 10,
    };
    // At step 50, should be 50% through the schedule
    let sparsity = schedule.sparsity_at_step(50);
    assert!(
        (sparsity - 0.5).abs() < 1e-6,
        "SCHED-012 FALSIFIED: Gradual at midpoint should be 0.5, got {sparsity}"
    );
}

#[test]
fn test_gradual_linear_interpolation_quarter() {
    // TEST_ID: SCHED-013
    let schedule = PruningSchedule::Gradual {
        start_step: 0,
        end_step: 100,
        initial_sparsity: 0.0,
        final_sparsity: 0.8,
        frequency: 10,
    };
    // At step 25, should be 25% * 0.8 = 0.2
    let sparsity = schedule.sparsity_at_step(25);
    assert!(
        (sparsity - 0.2).abs() < 1e-6,
        "SCHED-013 FALSIFIED: Gradual at 25% should be 0.2, got {sparsity}"
    );
}

#[test]
fn test_gradual_with_nonzero_initial() {
    // TEST_ID: SCHED-014
    // FALSIFIES: Gradual doesn't handle non-zero initial sparsity
    let schedule = PruningSchedule::Gradual {
        start_step: 0,
        end_step: 100,
        initial_sparsity: 0.2,
        final_sparsity: 0.8,
        frequency: 10,
    };
    // At step 50, should be 0.2 + 0.5 * (0.8 - 0.2) = 0.5
    let sparsity = schedule.sparsity_at_step(50);
    assert!(
        (sparsity - 0.5).abs() < 1e-6,
        "SCHED-014 FALSIFIED: Gradual with initial 0.2 at midpoint should be 0.5, got {sparsity}"
    );
}

#[test]
fn test_gradual_should_prune_at_frequency() {
    // TEST_ID: SCHED-015
    let schedule = PruningSchedule::Gradual {
        start_step: 100,
        end_step: 200,
        initial_sparsity: 0.0,
        final_sparsity: 0.5,
        frequency: 10,
    };
    assert!(
        !schedule.should_prune_at_step(99),
        "SCHED-015 FALSIFIED: should not prune before start"
    );
    assert!(schedule.should_prune_at_step(100), "SCHED-015 FALSIFIED: should prune at start");
    assert!(
        !schedule.should_prune_at_step(105),
        "SCHED-015 FALSIFIED: should not prune between frequencies"
    );
    assert!(schedule.should_prune_at_step(110), "SCHED-015 FALSIFIED: should prune at frequency");
    assert!(!schedule.should_prune_at_step(201), "SCHED-015 FALSIFIED: should not prune after end");
}

#[test]
fn test_gradual_zero_frequency_prunes_once() {
    // TEST_ID: SCHED-016
    let schedule = PruningSchedule::Gradual {
        start_step: 100,
        end_step: 200,
        initial_sparsity: 0.0,
        final_sparsity: 0.5,
        frequency: 0,
    };
    assert!(
        schedule.should_prune_at_step(100),
        "SCHED-016 FALSIFIED: should prune at start with freq=0"
    );
    assert!(
        !schedule.should_prune_at_step(150),
        "SCHED-016 FALSIFIED: should not prune mid-schedule with freq=0"
    );
}

#[test]
fn test_num_pruning_steps_gradual() {
    // TEST_ID: SCHED-041
    let schedule = PruningSchedule::Gradual {
        start_step: 0,
        end_step: 100,
        initial_sparsity: 0.0,
        final_sparsity: 0.5,
        frequency: 10,
    };
    // Steps: 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 = 11 steps
    assert_eq!(
        schedule.num_pruning_steps(),
        11,
        "SCHED-041 FALSIFIED: Gradual with freq=10 over 100 steps should have 11 pruning steps"
    );
}

#[test]
fn test_is_complete_gradual() {
    // TEST_ID: SCHED-044
    let schedule = PruningSchedule::Gradual {
        start_step: 0,
        end_step: 100,
        initial_sparsity: 0.0,
        final_sparsity: 0.5,
        frequency: 10,
    };
    assert!(
        !schedule.is_complete(100),
        "SCHED-044 FALSIFIED: Gradual should not be complete at end_step"
    );
    assert!(
        schedule.is_complete(101),
        "SCHED-044 FALSIFIED: Gradual should be complete after end_step"
    );
}

#[test]
fn test_num_pruning_steps_gradual_zero_frequency() {
    // TEST_ID: SCHED-049
    // FALSIFIES: num_pruning_steps doesn't handle freq=0
    let schedule = PruningSchedule::Gradual {
        start_step: 0,
        end_step: 100,
        initial_sparsity: 0.0,
        final_sparsity: 0.5,
        frequency: 0,
    };
    assert_eq!(
        schedule.num_pruning_steps(),
        1,
        "SCHED-049 FALSIFIED: Gradual with freq=0 should have exactly 1 pruning step"
    );
}

#[test]
fn test_sparsity_monotonic_gradual() {
    // TEST_ID: SCHED-060
    // FALSIFIES: Gradual sparsity can decrease over time
    let schedule = PruningSchedule::Gradual {
        start_step: 0,
        end_step: 100,
        initial_sparsity: 0.0,
        final_sparsity: 0.5,
        frequency: 1,
    };
    let mut prev = 0.0;
    for step in 0..=100 {
        let sparsity = schedule.sparsity_at_step(step);
        assert!(
            sparsity >= prev,
            "SCHED-060 FALSIFIED: Sparsity decreased from {prev} to {sparsity} at step {step}"
        );
        prev = sparsity;
    }
}

#[test]
fn test_sparsity_bounded_zero_to_final() {
    // TEST_ID: SCHED-062
    // FALSIFIES: Sparsity can exceed bounds
    let schedule = PruningSchedule::Gradual {
        start_step: 0,
        end_step: 100,
        initial_sparsity: 0.0,
        final_sparsity: 0.5,
        frequency: 1,
    };
    for step in 0..=200 {
        let sparsity = schedule.sparsity_at_step(step);
        assert!(
            (0.0..=0.5).contains(&sparsity),
            "SCHED-062 FALSIFIED: Sparsity {sparsity} out of bounds [0.0, 0.5] at step {step}"
        );
    }
}

#[test]
fn test_clone_produces_equal_schedule() {
    // TEST_ID: SCHED-063
    let schedule = PruningSchedule::Gradual {
        start_step: 100,
        end_step: 1000,
        initial_sparsity: 0.1,
        final_sparsity: 0.9,
        frequency: 50,
    };
    let cloned = schedule.clone();
    assert_eq!(schedule, cloned, "SCHED-063 FALSIFIED: Clone should equal original");
}

#[test]
fn test_gradual_at_start_step() {
    // TEST_ID: SCHED-070
    let schedule = PruningSchedule::Gradual {
        start_step: 100,
        end_step: 200,
        initial_sparsity: 0.2,
        final_sparsity: 0.8,
        frequency: 10,
    };
    let sparsity = schedule.sparsity_at_step(100);
    assert!((sparsity - 0.2).abs() < 1e-6, "Should be initial_sparsity at start_step");
}

#[test]
fn test_gradual_should_prune_at_end() {
    // TEST_ID: SCHED-075
    let schedule = PruningSchedule::Gradual {
        start_step: 0,
        end_step: 100,
        initial_sparsity: 0.0,
        final_sparsity: 0.5,
        frequency: 10,
    };
    assert!(schedule.should_prune_at_step(100));
}

#[test]
fn test_debug_format_gradual() {
    // TEST_ID: SCHED-076
    let schedule = PruningSchedule::Gradual {
        start_step: 100,
        end_step: 200,
        initial_sparsity: 0.1,
        final_sparsity: 0.9,
        frequency: 10,
    };
    let debug = format!("{schedule:?}");
    assert!(debug.contains("Gradual"));
    assert!(debug.contains("100"));
    assert!(debug.contains("200"));
}

#[test]
fn test_gradual_progress_at_75_percent() {
    // TEST_ID: SCHED-078
    let schedule = PruningSchedule::Gradual {
        start_step: 0,
        end_step: 100,
        initial_sparsity: 0.0,
        final_sparsity: 1.0,
        frequency: 1,
    };
    let sparsity = schedule.sparsity_at_step(75);
    assert!((sparsity - 0.75).abs() < 1e-6, "Gradual at 75% should be 0.75");
}
