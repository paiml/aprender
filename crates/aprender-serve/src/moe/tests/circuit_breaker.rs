//! Tests for CircuitBreaker (Nygard 2018)

use crate::moe::{CircuitBreaker, CircuitBreakerConfig, CircuitState};

#[test]
fn test_circuit_breaker_starts_closed() {
    let cb = CircuitBreaker::new(CircuitBreakerConfig {
        failure_threshold: 5,
        success_threshold: 3,
        timeout_ms: 1000,
    });
    assert_eq!(cb.state(), CircuitState::Closed);
}

#[test]
fn test_circuit_breaker_opens_on_failures() {
    let cb = CircuitBreaker::new(CircuitBreakerConfig {
        failure_threshold: 3,
        success_threshold: 2,
        timeout_ms: 1000,
    });

    cb.record_failure();
    cb.record_failure();
    assert_eq!(cb.state(), CircuitState::Closed);

    cb.record_failure();
    assert_eq!(cb.state(), CircuitState::Open);
}

#[test]
fn test_circuit_breaker_blocks_when_open() {
    let cb = CircuitBreaker::new(CircuitBreakerConfig {
        failure_threshold: 1,
        success_threshold: 1,
        timeout_ms: 100_000,
    });

    cb.record_failure();
    assert!(!cb.allow_request());
}

#[test]
fn test_circuit_breaker_half_open_after_timeout() {
    let cb = CircuitBreaker::new(CircuitBreakerConfig {
        failure_threshold: 1,
        success_threshold: 1,
        timeout_ms: 1,
    });

    cb.record_failure();
    std::thread::sleep(std::time::Duration::from_millis(5));

    assert_eq!(cb.state(), CircuitState::HalfOpen);
    assert!(cb.allow_request());
}

#[test]
fn test_circuit_breaker_closes_on_success_in_half_open() {
    let cb = CircuitBreaker::new(CircuitBreakerConfig {
        failure_threshold: 1,
        success_threshold: 2,
        timeout_ms: 1,
    });

    cb.record_failure();
    std::thread::sleep(std::time::Duration::from_millis(5));

    cb.record_success();
    cb.record_success();

    assert_eq!(cb.state(), CircuitState::Closed);
}

#[test]
fn test_circuit_breaker_success_resets_failure_count_in_closed() {
    let cb = CircuitBreaker::new(CircuitBreakerConfig {
        failure_threshold: 3,
        success_threshold: 2,
        timeout_ms: 1000,
    });

    cb.record_failure();
    cb.record_failure();
    assert_eq!(cb.state(), CircuitState::Closed);

    cb.record_success();

    cb.record_failure();
    cb.record_failure();
    assert_eq!(cb.state(), CircuitState::Closed);

    cb.record_failure();
    assert_eq!(cb.state(), CircuitState::Open);
}

#[test]
fn test_circuit_breaker_success_while_open_no_effect() {
    let cb = CircuitBreaker::new(CircuitBreakerConfig {
        failure_threshold: 1,
        success_threshold: 2,
        timeout_ms: 100_000,
    });

    cb.record_failure();
    assert_eq!(cb.state(), CircuitState::Open);

    cb.record_success();
    assert_eq!(cb.state(), CircuitState::Open);
    assert!(!cb.allow_request());
}

#[test]
fn test_circuit_breaker_partial_success_in_half_open() {
    let cb = CircuitBreaker::new(CircuitBreakerConfig {
        failure_threshold: 1,
        success_threshold: 3,
        timeout_ms: 1,
    });

    cb.record_failure();
    std::thread::sleep(std::time::Duration::from_millis(5));
    assert_eq!(cb.state(), CircuitState::HalfOpen);

    cb.record_success();
    assert_eq!(cb.state(), CircuitState::HalfOpen);

    cb.record_success();
    assert_eq!(cb.state(), CircuitState::HalfOpen);

    cb.record_success();
    assert_eq!(cb.state(), CircuitState::Closed);
}

#[test]
fn test_circuit_breaker_failure_in_half_open_reopens() {
    let cb = CircuitBreaker::new(CircuitBreakerConfig {
        failure_threshold: 1,
        success_threshold: 3,
        timeout_ms: 1,
    });

    cb.record_failure();
    std::thread::sleep(std::time::Duration::from_millis(5));
    assert_eq!(cb.state(), CircuitState::HalfOpen);

    cb.record_failure();
    assert_eq!(cb.state(), CircuitState::Open);
}

#[test]
fn test_circuit_breaker_config_debug() {
    let config = CircuitBreakerConfig {
        failure_threshold: 5,
        success_threshold: 3,
        timeout_ms: 1000,
    };
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("failure_threshold: 5"));
    assert!(debug_str.contains("success_threshold: 3"));
    assert!(debug_str.contains("timeout_ms: 1000"));
}

#[test]
fn test_circuit_breaker_config_clone() {
    let config = CircuitBreakerConfig {
        failure_threshold: 5,
        success_threshold: 3,
        timeout_ms: 1000,
    };
    let cloned = config.clone();
    assert_eq!(config.failure_threshold, cloned.failure_threshold);
    assert_eq!(config.success_threshold, cloned.success_threshold);
    assert_eq!(config.timeout_ms, cloned.timeout_ms);
}

#[test]
fn test_circuit_state_debug_and_eq() {
    assert_eq!(CircuitState::Closed, CircuitState::Closed);
    assert_eq!(CircuitState::Open, CircuitState::Open);
    assert_eq!(CircuitState::HalfOpen, CircuitState::HalfOpen);
    assert_ne!(CircuitState::Closed, CircuitState::Open);

    let debug_str = format!("{:?}", CircuitState::HalfOpen);
    assert_eq!(debug_str, "HalfOpen");
}

#[test]
fn test_circuit_state_copy() {
    let state = CircuitState::Closed;
    let copied = state;
    assert_eq!(state, copied);
}

// ============================================================================
// Additional CircuitBreaker Coverage Tests
// ============================================================================

#[test]
fn test_circuit_breaker_rapid_failures() {
    let cb = CircuitBreaker::new(CircuitBreakerConfig {
        failure_threshold: 5,
        success_threshold: 2,
        timeout_ms: 1000,
    });

    // Rapid fire failures
    for _ in 0..4 {
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    cb.record_failure();
    assert_eq!(cb.state(), CircuitState::Open);
}

#[test]
fn test_circuit_breaker_interleaved_success_failure() {
    let cb = CircuitBreaker::new(CircuitBreakerConfig {
        failure_threshold: 3,
        success_threshold: 2,
        timeout_ms: 1000,
    });

    // Interleaved pattern - success resets failure count
    cb.record_failure();
    cb.record_success(); // Resets
    cb.record_failure();
    cb.record_success(); // Resets
    cb.record_failure();
    cb.record_success(); // Resets

    // Still closed because we never hit 3 consecutive
    assert_eq!(cb.state(), CircuitState::Closed);
}

#[test]
fn test_circuit_breaker_threshold_boundary() {
    let cb = CircuitBreaker::new(CircuitBreakerConfig {
        failure_threshold: 1,
        success_threshold: 1,
        timeout_ms: 1,
    });

    // Exactly at threshold
    cb.record_failure();
    assert_eq!(cb.state(), CircuitState::Open);

    // Wait for half-open
    std::thread::sleep(std::time::Duration::from_millis(5));
    assert_eq!(cb.state(), CircuitState::HalfOpen);

    // One success closes it
    cb.record_success();
    assert_eq!(cb.state(), CircuitState::Closed);
}

#[test]
fn test_circuit_breaker_allow_request_updates_state() {
    let cb = CircuitBreaker::new(CircuitBreakerConfig {
        failure_threshold: 1,
        success_threshold: 1,
        timeout_ms: 1,
    });

    cb.record_failure();
    assert!(!cb.allow_request()); // Open

    std::thread::sleep(std::time::Duration::from_millis(5));

    // allow_request should also transition to half-open
    assert!(cb.allow_request());
    assert_eq!(cb.state(), CircuitState::HalfOpen);
}

#[test]
fn test_circuit_breaker_high_threshold() {
    let cb = CircuitBreaker::new(CircuitBreakerConfig {
        failure_threshold: 100,
        success_threshold: 50,
        timeout_ms: 1000,
    });

    // 99 failures - still closed
    for _ in 0..99 {
        cb.record_failure();
    }
    assert_eq!(cb.state(), CircuitState::Closed);

    // 100th failure opens it
    cb.record_failure();
    assert_eq!(cb.state(), CircuitState::Open);
}

#[test]
fn test_circuit_breaker_zero_timeout() {
    let cb = CircuitBreaker::new(CircuitBreakerConfig {
        failure_threshold: 1,
        success_threshold: 1,
        timeout_ms: 0,
    });

    cb.record_failure();
    // With zero timeout, should immediately be half-open on next state check
    assert_eq!(cb.state(), CircuitState::HalfOpen);
}

#[test]
fn test_circuit_state_clone() {
    let state = CircuitState::HalfOpen;
    let cloned = state;
    assert_eq!(state, cloned);
}

#[test]
fn test_circuit_breaker_multiple_half_open_cycles() {
    let cb = CircuitBreaker::new(CircuitBreakerConfig {
        failure_threshold: 1,
        success_threshold: 2,
        timeout_ms: 1,
    });

    // First cycle
    cb.record_failure();
    std::thread::sleep(std::time::Duration::from_millis(5));
    assert_eq!(cb.state(), CircuitState::HalfOpen);
    cb.record_failure(); // Back to open
    assert_eq!(cb.state(), CircuitState::Open);

    // Second cycle
    std::thread::sleep(std::time::Duration::from_millis(5));
    assert_eq!(cb.state(), CircuitState::HalfOpen);
    cb.record_success();
    cb.record_success();
    assert_eq!(cb.state(), CircuitState::Closed);
}
