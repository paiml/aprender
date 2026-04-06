// ============================================================================
// Supervisor Integration -- GPU Worker Management
// ============================================================================

use trueno_cuda_edge::supervisor::{
    GpuHealthMonitor, HealthAction, HeartbeatStatus, SupervisionStrategy, SupervisionTree,
};

/// Test supervision strategies for GPU workers.
#[test]
fn supervision_strategies() {
    // One-for-one: isolated restarts
    assert!(SupervisionStrategy::OneForOne.is_isolated());

    // One-for-all: restart all on any failure
    assert!(!SupervisionStrategy::OneForAll.is_isolated());

    // Rest-for-one: restart crashed + dependents
    assert!(!SupervisionStrategy::RestForOne.is_isolated());
}

/// Test supervision tree crash handling.
#[test]
fn supervision_tree_operations() {
    let mut tree = SupervisionTree::new(SupervisionStrategy::OneForOne, 4);

    // Crash worker 2 at time 0
    let action = tree.handle_crash(2, 0);
    match action {
        trueno_cuda_edge::supervisor::SupervisorAction::Restart(indices) => {
            assert_eq!(indices, vec![2]);
        }
        _ => panic!("Expected Restart action"),
    }
}

/// Test one-for-all strategy.
#[test]
fn one_for_all_restarts() {
    let mut tree = SupervisionTree::new(SupervisionStrategy::OneForAll, 3);

    let action = tree.handle_crash(1, 0);
    match action {
        trueno_cuda_edge::supervisor::SupervisorAction::Restart(indices) => {
            assert_eq!(indices, vec![0, 1, 2]);
        }
        _ => panic!("Expected Restart action"),
    }
}

/// Test health monitoring for GPU workers.
#[test]
fn health_monitoring() {
    let monitor =
        GpuHealthMonitor::builder().max_missed(3).throttle_temp(85).shutdown_temp(95).build();

    // Alive: healthy
    assert_eq!(monitor.check_status(HeartbeatStatus::Alive), HealthAction::Healthy);

    // Missed beats below threshold: healthy
    assert_eq!(monitor.check_status(HeartbeatStatus::MissedBeats(2)), HealthAction::Healthy);

    // Missed beats at threshold: restart
    assert_eq!(monitor.check_status(HeartbeatStatus::MissedBeats(3)), HealthAction::RestartWorker);

    // Dead: shutdown
    assert_eq!(monitor.check_status(HeartbeatStatus::Dead), HealthAction::Shutdown);
}

/// Test thermal monitoring thresholds.
#[test]
fn thermal_monitoring() {
    let monitor = GpuHealthMonitor::new(3, 85, 95);

    // Below throttle: healthy
    assert_eq!(monitor.check_temperature(70), HealthAction::Healthy);

    // At throttle threshold: throttle
    assert_eq!(monitor.check_temperature(85), HealthAction::Throttle);

    // Between throttle and shutdown: throttle
    assert_eq!(monitor.check_temperature(90), HealthAction::Throttle);

    // At shutdown threshold: shutdown
    assert_eq!(monitor.check_temperature(95), HealthAction::Shutdown);
}
