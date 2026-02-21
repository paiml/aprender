use super::*;

#[test]
fn test_health_status_transitions() {
    let checker = HealthChecker::default();
    let node = NodeId("test-node".to_string());

    checker.register_node(node.clone());

    // Initially unknown
    let health = checker.get_cached_health(&node).expect("node should exist");
    assert_eq!(health.status, HealthState::Unknown);

    // Report successes to become healthy
    for _ in 0..3 {
        checker.report_success(&node, Duration::from_millis(50));
    }

    let health = checker.get_cached_health(&node).expect("node should exist");
    assert_eq!(health.status, HealthState::Healthy);
}

#[test]
fn test_health_degraded_on_high_latency() {
    let checker = HealthChecker::default();
    let node = NodeId("slow-node".to_string());

    checker.register_node(node.clone());

    // Report high latency
    checker.report_success(&node, Duration::from_secs(2));

    let health = checker.get_cached_health(&node).expect("node should exist");
    assert_eq!(health.status, HealthState::Degraded);
}

#[test]
fn test_health_unhealthy_on_failures() {
    let config = HealthConfig {
        failure_threshold: 3,
        ..Default::default()
    };
    let checker = HealthChecker::new(config);
    let node = NodeId("failing-node".to_string());

    checker.register_node(node.clone());

    // Report failures
    for _ in 0..3 {
        checker.report_failure(&node);
    }

    let health = checker.get_cached_health(&node).expect("node should exist");
    assert_eq!(health.status, HealthState::Unhealthy);
}

#[test]
fn test_circuit_breaker_opens_on_failures() {
    let config = CircuitBreakerConfig {
        failure_threshold: 3,
        ..Default::default()
    };
    let breaker = CircuitBreaker::new(config);
    let node = NodeId("failing-node".to_string());

    // Initially closed
    assert!(!breaker.is_open(&node));
    assert_eq!(breaker.state(&node), CircuitState::Closed);

    // Record failures
    for _ in 0..3 {
        breaker.record_failure(&node);
    }

    assert!(breaker.is_open(&node));
    assert_eq!(breaker.state(&node), CircuitState::Open);
}

#[test]
fn test_circuit_breaker_success_resets() {
    let breaker = CircuitBreaker::default();
    let node = NodeId("flaky-node".to_string());

    // Some failures
    breaker.record_failure(&node);
    breaker.record_failure(&node);

    // Success should reset
    breaker.record_success(&node);

    let state = breaker.get_or_create_state(&node);
    assert_eq!(state.failures, 0);
}

#[test]
fn test_circuit_breaker_half_open_recovery() {
    let config = CircuitBreakerConfig {
        failure_threshold: 2,
        half_open_successes: 2,
        reset_timeout: Duration::from_millis(10),
    };
    let breaker = CircuitBreaker::new(config);
    let node = NodeId("recovering-node".to_string());

    // Open the circuit
    breaker.record_failure(&node);
    breaker.record_failure(&node);
    assert_eq!(breaker.state(&node), CircuitState::Open);

    // Wait for reset timeout
    std::thread::sleep(Duration::from_millis(20));

    // Should transition to half-open on next check
    assert!(!breaker.is_open(&node));
    assert_eq!(breaker.state(&node), CircuitState::HalfOpen);

    // Successes in half-open should close
    breaker.record_success(&node);
    breaker.record_success(&node);
    assert_eq!(breaker.state(&node), CircuitState::Closed);
}

// =========================================================================
// HealthConfig tests
// =========================================================================

#[test]
fn test_health_config_default() {
    let config = HealthConfig::default();
    assert_eq!(config.check_interval, Duration::from_secs(10));
    assert_eq!(config.probe_timeout, Duration::from_secs(5));
    assert_eq!(config.failure_threshold, 3);
    assert_eq!(config.recovery_threshold, 2);
    assert_eq!(config.degraded_latency, Duration::from_secs(1));
}

#[test]
fn test_health_config_custom() {
    let config = HealthConfig {
        check_interval: Duration::from_secs(30),
        probe_timeout: Duration::from_secs(10),
        failure_threshold: 5,
        recovery_threshold: 3,
        degraded_latency: Duration::from_millis(500),
    };
    assert_eq!(config.check_interval, Duration::from_secs(30));
    assert_eq!(config.failure_threshold, 5);
}

#[test]
fn test_health_config_clone() {
    let config = HealthConfig::default();
    let cloned = config.clone();
    assert_eq!(cloned.failure_threshold, config.failure_threshold);
}

// =========================================================================
// HealthChecker extended tests
// =========================================================================

#[test]
fn test_health_checker_register_and_deregister() {
    let checker = HealthChecker::default();
    let node = NodeId("temp-node".to_string());

    checker.register_node(node.clone());
    assert_eq!(checker.total_count(), 1);

    checker.deregister_node(&node);
    assert_eq!(checker.total_count(), 0);
}

#[test]
fn test_health_checker_deregister_unknown_node() {
    let checker = HealthChecker::default();
    let unknown = NodeId("unknown".to_string());

    // Deregistering a non-existent node should not panic
    checker.deregister_node(&unknown);
    assert_eq!(checker.total_count(), 0);
}

#[test]
fn test_health_checker_all_statuses_empty() {
    let checker = HealthChecker::default();
    let statuses = checker.all_statuses();
    assert!(statuses.is_empty());
}

#[test]
fn test_health_checker_all_statuses_multiple() {
    let checker = HealthChecker::default();
    checker.register_node(NodeId("n1".to_string()));
    checker.register_node(NodeId("n2".to_string()));
    checker.register_node(NodeId("n3".to_string()));

    let statuses = checker.all_statuses();
    assert_eq!(statuses.len(), 3);
}

#[test]
fn test_health_checker_healthy_count_none() {
    let checker = HealthChecker::default();
    checker.register_node(NodeId("n1".to_string()));
    // All nodes start Unknown, not Healthy
    assert_eq!(checker.healthy_count(), 0);
}

#[test]
fn test_health_checker_healthy_count_some() {
    let checker = HealthChecker::default();
    let n1 = NodeId("n1".to_string());
    let n2 = NodeId("n2".to_string());
    checker.register_node(n1.clone());
    checker.register_node(n2.clone());

    // Make n1 healthy
    for _ in 0..3 {
        checker.report_success(&n1, Duration::from_millis(10));
    }
    // Leave n2 as Unknown

    assert_eq!(checker.healthy_count(), 1);
    assert_eq!(checker.total_count(), 2);
}

#[test]
fn test_health_checker_is_monitoring_default() {
    let checker = HealthChecker::default();
    assert!(!checker.is_monitoring());
}

#[test]
fn test_health_checker_report_success_unknown_node() {
    let checker = HealthChecker::default();
    let unknown = NodeId("unknown".to_string());

    // Reporting on unknown node should not panic
    checker.report_success(&unknown, Duration::from_millis(50));
    assert_eq!(checker.total_count(), 0);
}

#[test]
fn test_health_checker_report_failure_unknown_node() {
    let checker = HealthChecker::default();
    let unknown = NodeId("unknown".to_string());

    // Reporting on unknown node should not panic
    checker.report_failure(&unknown);
    assert_eq!(checker.total_count(), 0);
}

#[test]
fn test_health_checker_degraded_then_healthy() {
    let config = HealthConfig {
        recovery_threshold: 2,
        ..Default::default()
    };
    let checker = HealthChecker::new(config);
    let node = NodeId("recovering".to_string());

    checker.register_node(node.clone());

    // First failure -> degraded
    checker.report_failure(&node);
    let health = checker.get_cached_health(&node).expect("should exist");
    assert_eq!(health.status, HealthState::Degraded);

    // Successful recoveries
    for _ in 0..2 {
        checker.report_success(&node, Duration::from_millis(50));
    }
    let health = checker.get_cached_health(&node).expect("should exist");
    assert_eq!(health.status, HealthState::Healthy);
}

#[test]
fn test_health_checker_failure_below_threshold_is_degraded() {
    let config = HealthConfig {
        failure_threshold: 5,
        ..Default::default()
    };
    let checker = HealthChecker::new(config);
    let node = NodeId("flaky".to_string());
    checker.register_node(node.clone());

    // 3 failures (below threshold of 5)
    for _ in 0..3 {
        checker.report_failure(&node);
    }
    let health = checker.get_cached_health(&node).expect("should exist");
    assert_eq!(health.status, HealthState::Degraded);

    // 2 more failures (reach threshold of 5)
    for _ in 0..2 {
        checker.report_failure(&node);
    }
    let health = checker.get_cached_health(&node).expect("should exist");
    assert_eq!(health.status, HealthState::Unhealthy);
}

#[test]
fn test_health_checker_latency_moving_average() {
    let checker = HealthChecker::default();
    let node = NodeId("avg-node".to_string());
    checker.register_node(node.clone());

    // Report several successes with known latency
    for _ in 0..10 {
        checker.report_success(&node, Duration::from_millis(100));
    }

    let health = checker.get_cached_health(&node).expect("should exist");
    // After enough iterations, p50 should approach 100ms
    assert!(health.latency_p50.as_millis() > 0);
}

#[test]
fn test_health_checker_get_cached_health_none() {
    let checker = HealthChecker::default();
    let unknown = NodeId("no-such-node".to_string());
    assert!(checker.get_cached_health(&unknown).is_none());
}

// =========================================================================
// HealthStatus From<NodeHealth> tests
// =========================================================================

#[test]
fn test_health_status_from_node_health() {
    let now = Instant::now();
    let health = NodeHealth {
        node_id: NodeId("test".to_string()),
        status: HealthState::Degraded,
        latency_p50: Duration::from_millis(100),
        latency_p99: Duration::from_millis(500),
        throughput: 200,
        gpu_utilization: Some(0.5),
        queue_depth: 5,
        last_check: now,
    };

    let status = HealthStatus::from(health);
    assert_eq!(status.node_id, NodeId("test".to_string()));
    assert_eq!(status.state, HealthState::Degraded);
    assert_eq!(status.latency_p50, Duration::from_millis(100));
    assert_eq!(status.latency_p99, Duration::from_millis(500));
    assert_eq!(status.queue_depth, 5);
}

// =========================================================================
// HealthCheckerTrait async tests
// =========================================================================

#[tokio::test]
async fn test_health_checker_check_node_registered() {
    let checker = HealthChecker::default();
    let node = NodeId("registered".to_string());
    checker.register_node(node.clone());

    let result = checker.check_node(&node).await;
    assert!(result.is_ok());
    let health = result.expect("check_node failed");
    assert_eq!(health.node_id, node);
    assert_eq!(health.status, HealthState::Unknown);
}

#[tokio::test]
async fn test_health_checker_check_node_unregistered() {
    let checker = HealthChecker::default();
    let node = NodeId("missing".to_string());

    let result = checker.check_node(&node).await;
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        FederationError::NodeUnreachable(_)
    ));
}

#[tokio::test]
async fn test_health_checker_start_stop_monitoring() {
    let checker = HealthChecker::default();

    assert!(!checker.is_monitoring());

    checker.start_monitoring(Duration::from_secs(10)).await;
    assert!(checker.is_monitoring());

    checker.stop_monitoring().await;
    assert!(!checker.is_monitoring());
}

// =========================================================================
// CircuitBreakerConfig tests
// =========================================================================

#[test]
fn test_circuit_breaker_config_default() {
    let config = CircuitBreakerConfig::default();
    assert_eq!(config.failure_threshold, 5);
    assert_eq!(config.reset_timeout, Duration::from_secs(30));
    assert_eq!(config.half_open_successes, 3);
}

#[test]
fn test_circuit_breaker_config_custom() {
    let config = CircuitBreakerConfig {
        failure_threshold: 10,
        reset_timeout: Duration::from_secs(60),
        half_open_successes: 5,
    };
    assert_eq!(config.failure_threshold, 10);
}

#[test]
fn test_circuit_breaker_config_clone() {
    let config = CircuitBreakerConfig::default();
    let cloned = config.clone();
    assert_eq!(cloned.half_open_successes, config.half_open_successes);
}

// =========================================================================
// CircuitBreaker extended tests
// =========================================================================

#[test]
fn test_circuit_breaker_all_states_empty() {
    let breaker = CircuitBreaker::default();
    let states = breaker.all_states();
    assert!(states.is_empty());
}

include!("health_tests_circuit_breaker.rs");
