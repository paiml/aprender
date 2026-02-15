
#[test]
fn test_circuit_breaker_all_states_multiple() {
    let config = CircuitBreakerConfig {
        failure_threshold: 2,
        ..Default::default()
    };
    let breaker = CircuitBreaker::new(config);

    let n1 = NodeId("n1".to_string());
    let n2 = NodeId("n2".to_string());

    // n1: make it open
    breaker.record_failure(&n1);
    breaker.record_failure(&n1);

    // n2: keep it closed
    breaker.record_success(&n2);

    let states = breaker.all_states();
    assert_eq!(states.len(), 2);

    let n1_state = states.iter().find(|(id, _)| *id == n1).map(|(_, s)| *s);
    let n2_state = states.iter().find(|(id, _)| *id == n2).map(|(_, s)| *s);

    assert_eq!(n1_state, Some(CircuitState::Open));
    assert_eq!(n2_state, Some(CircuitState::Closed));
}

#[test]
fn test_circuit_breaker_unknown_node_defaults_closed() {
    let breaker = CircuitBreaker::default();
    let unknown = NodeId("unknown".to_string());

    assert_eq!(breaker.state(&unknown), CircuitState::Closed);
    assert!(!breaker.is_open(&unknown));
}

#[test]
fn test_circuit_breaker_record_success_in_open_resets() {
    let config = CircuitBreakerConfig {
        failure_threshold: 2,
        ..Default::default()
    };
    let breaker = CircuitBreaker::new(config);
    let node = NodeId("node".to_string());

    // Open circuit
    breaker.record_failure(&node);
    breaker.record_failure(&node);
    assert_eq!(breaker.state(&node), CircuitState::Open);

    // Record success while open (edge case)
    breaker.record_success(&node);
    // Should reset to closed
    assert_eq!(breaker.state(&node), CircuitState::Closed);
}

#[test]
fn test_circuit_breaker_failure_in_half_open_reopens() {
    let config = CircuitBreakerConfig {
        failure_threshold: 2,
        half_open_successes: 3,
        reset_timeout: Duration::from_millis(10),
    };
    let breaker = CircuitBreaker::new(config);
    let node = NodeId("node".to_string());

    // Open circuit
    breaker.record_failure(&node);
    breaker.record_failure(&node);
    assert_eq!(breaker.state(&node), CircuitState::Open);

    // Wait for reset
    std::thread::sleep(Duration::from_millis(20));

    // Transition to half-open
    assert!(!breaker.is_open(&node));
    assert_eq!(breaker.state(&node), CircuitState::HalfOpen);

    // Failure in half-open -> back to open
    breaker.record_failure(&node);
    assert_eq!(breaker.state(&node), CircuitState::Open);
}

#[test]
fn test_circuit_breaker_failure_while_already_open() {
    let config = CircuitBreakerConfig {
        failure_threshold: 2,
        ..Default::default()
    };
    let breaker = CircuitBreaker::new(config);
    let node = NodeId("node".to_string());

    // Open circuit
    breaker.record_failure(&node);
    breaker.record_failure(&node);
    assert_eq!(breaker.state(&node), CircuitState::Open);

    // Additional failure while open should not panic
    breaker.record_failure(&node);
    assert_eq!(breaker.state(&node), CircuitState::Open);
}

#[test]
fn test_circuit_breaker_is_open_before_timeout() {
    let config = CircuitBreakerConfig {
        failure_threshold: 2,
        reset_timeout: Duration::from_secs(60), // Long timeout
        ..Default::default()
    };
    let breaker = CircuitBreaker::new(config);
    let node = NodeId("node".to_string());

    breaker.record_failure(&node);
    breaker.record_failure(&node);

    // Still open (timeout hasn't passed)
    assert!(breaker.is_open(&node));
}

#[test]
fn test_circuit_breaker_half_open_partial_success() {
    let config = CircuitBreakerConfig {
        failure_threshold: 2,
        half_open_successes: 3,
        reset_timeout: Duration::from_millis(10),
    };
    let breaker = CircuitBreaker::new(config);
    let node = NodeId("node".to_string());

    // Open circuit
    breaker.record_failure(&node);
    breaker.record_failure(&node);
    std::thread::sleep(Duration::from_millis(20));

    // Transition to half-open
    assert!(!breaker.is_open(&node));

    // 2 successes (need 3)
    breaker.record_success(&node);
    breaker.record_success(&node);
    assert_eq!(breaker.state(&node), CircuitState::HalfOpen);

    // 3rd success -> closed
    breaker.record_success(&node);
    assert_eq!(breaker.state(&node), CircuitState::Closed);
}
