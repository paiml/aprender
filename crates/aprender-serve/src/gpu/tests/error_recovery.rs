
#[test]
fn test_error_recovery_strategy_calculate_delay_exponential() {
    let strategy = ErrorRecoveryStrategy::new()
        .with_base_delay(Duration::from_millis(100))
        .with_jitter(0.0); // No jitter for predictable test

    let delay0 = strategy.calculate_delay(0).as_millis();
    let delay1 = strategy.calculate_delay(1).as_millis();
    let delay2 = strategy.calculate_delay(2).as_millis();

    // Exponential backoff: 100, 200, 400
    assert!(delay0 >= 100);
    assert!(delay1 >= 200);
    assert!(delay2 >= 400);
}

// ============================================================================
// DegradationManager Tests
// ============================================================================

#[test]
fn test_degradation_manager_new() {
    let manager = DegradationManager::new();
    assert_eq!(manager.current_mode(), DegradationMode::Normal);
}

#[test]
fn test_degradation_manager_default() {
    let manager = DegradationManager::default();
    assert_eq!(manager.current_mode(), DegradationMode::Normal);
}

#[test]
fn test_degradation_manager_set_gpu_available() {
    let mut manager = DegradationManager::new();

    manager.set_gpu_available(false);
    assert_eq!(manager.current_mode(), DegradationMode::CpuFallback);

    manager.set_gpu_available(true);
    assert_eq!(manager.current_mode(), DegradationMode::Normal);
}

#[test]
fn test_degradation_manager_memory_pressure() {
    let mut manager = DegradationManager::new();

    manager.update_memory_pressure(0.9); // High pressure
    assert_eq!(manager.current_mode(), DegradationMode::MemoryPressure);

    manager.update_memory_pressure(0.5); // Normal
    assert_eq!(manager.current_mode(), DegradationMode::Normal);
}

#[test]
fn test_degradation_manager_latency_priority() {
    let mut manager = DegradationManager::new();

    manager.set_latency_priority(true);
    assert_eq!(manager.current_mode(), DegradationMode::LowLatency);

    manager.set_latency_priority(false);
    assert_eq!(manager.current_mode(), DegradationMode::Normal);
}

#[test]
fn test_degradation_manager_system_load() {
    let mut manager = DegradationManager::new();

    let high_load = SystemLoad {
        cpu_percent: 95.0,
        memory_percent: 90.0,
        queue_depth: 100,
    };

    manager.update_system_load(high_load);
    assert_eq!(manager.current_mode(), DegradationMode::MemoryPressure);
}

#[test]
fn test_degradation_manager_recommended_batch_size() {
    let mut manager = DegradationManager::new();

    // Normal pressure
    assert_eq!(manager.recommended_batch_size(32), 32);

    // High pressure reduces batch size
    manager.update_memory_pressure(0.9);
    let reduced = manager.recommended_batch_size(32);
    assert!(reduced < 32);
}

#[test]
fn test_degradation_manager_recommended_max_context() {
    let mut manager = DegradationManager::new();

    // No load info
    assert_eq!(manager.recommended_max_context(4096), 4096);

    // High load
    let high_load = SystemLoad {
        cpu_percent: 95.0,
        memory_percent: 85.0,
        queue_depth: 60,
    };
    manager.update_system_load(high_load);

    let reduced = manager.recommended_max_context(4096);
    assert!(reduced < 4096);
}

// ============================================================================
// FailureIsolator Tests
// ============================================================================

#[test]
fn test_failure_isolator_new() {
    let isolator = FailureIsolator::new();
    assert_eq!(isolator.active_requests(), 0);
    assert_eq!(isolator.success_count(), 0);
    assert_eq!(isolator.failure_count(), 0);
    assert!(!isolator.is_circuit_open());
}

#[test]
fn test_failure_isolator_default() {
    let isolator = FailureIsolator::default();
    assert_eq!(isolator.active_requests(), 0);
}

#[test]
fn test_failure_isolator_start_request() {
    let isolator = FailureIsolator::new();

    let id1 = isolator.start_request();
    assert_eq!(isolator.active_requests(), 1);

    let id2 = isolator.start_request();
    assert_eq!(isolator.active_requests(), 2);

    assert_ne!(id1, id2);
}

#[test]
fn test_failure_isolator_try_start_request() {
    let isolator = FailureIsolator::new();

    let result = isolator.try_start_request();
    assert!(result.is_ok());
    assert_eq!(isolator.active_requests(), 1);
}

#[test]
fn test_failure_isolator_complete_request_success() {
    let isolator = FailureIsolator::new();

    let id = isolator.start_request();
    isolator.complete_request(id, &RequestOutcome::Success);

    assert_eq!(isolator.active_requests(), 0);
    assert_eq!(isolator.success_count(), 1);
    assert_eq!(isolator.failure_count(), 0);
}

#[test]
fn test_failure_isolator_complete_request_failure() {
    let isolator = FailureIsolator::new();

    let id = isolator.start_request();
    isolator.complete_request(id, &RequestOutcome::Failed("test error".to_string()));

    assert_eq!(isolator.active_requests(), 0);
    assert_eq!(isolator.success_count(), 0);
    assert_eq!(isolator.failure_count(), 1);
}

#[test]
fn test_failure_isolator_circuit_opens_on_consecutive_failures() {
    let isolator = FailureIsolator::new();

    // Need 5 consecutive failures to open circuit
    for _ in 0..5 {
        let id = isolator.start_request();
        isolator.complete_request(id, &RequestOutcome::Failed("error".to_string()));
    }

    assert!(isolator.is_circuit_open());
}

#[test]
fn test_failure_isolator_success_resets_consecutive_failures() {
    let isolator = FailureIsolator::new();

    // 4 failures
    for _ in 0..4 {
        let id = isolator.start_request();
        isolator.complete_request(id, &RequestOutcome::Failed("error".to_string()));
    }

    // 1 success
    let id = isolator.start_request();
    isolator.complete_request(id, &RequestOutcome::Success);

    // 4 more failures
    for _ in 0..4 {
        let id = isolator.start_request();
        isolator.complete_request(id, &RequestOutcome::Failed("error".to_string()));
    }

    // Circuit should still be closed (success reset counter)
    assert!(!isolator.is_circuit_open());
}

#[test]
fn test_failure_isolator_reset_circuit() {
    let isolator = FailureIsolator::new();

    // Open circuit
    for _ in 0..5 {
        let id = isolator.start_request();
        isolator.complete_request(id, &RequestOutcome::Failed("error".to_string()));
    }
    assert!(isolator.is_circuit_open());

    // Reset
    isolator.reset_circuit();
    assert!(!isolator.is_circuit_open());
}

#[test]
fn test_failure_isolator_try_start_with_open_circuit() {
    let isolator = FailureIsolator::new();

    // Open circuit
    for _ in 0..5 {
        let id = isolator.start_request();
        isolator.complete_request(id, &RequestOutcome::Failed("error".to_string()));
    }

    // Should fail with open circuit
    let result = isolator.try_start_request();
    assert!(result.is_err());
}

#[test]
fn test_failure_isolator_register_cleanup() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let isolator = FailureIsolator::new();
    let cleanup_called = Arc::new(AtomicBool::new(false));
    let cleanup_called_clone = cleanup_called.clone();

    let id = isolator.start_request();
    isolator.register_cleanup(id, move || {
        cleanup_called_clone.store(true, Ordering::SeqCst);
    });

    // Cleanup should be called on failure
    isolator.complete_request(id, &RequestOutcome::Failed("error".to_string()));

    assert!(cleanup_called.load(Ordering::SeqCst));
}

// ============================================================================
// ConnectionPool Tests
// ============================================================================

#[test]
fn test_connection_config_new() {
    let config = ConnectionConfig::new();
    // Just verify it doesn't panic
    let _ = config;
}

#[test]
fn test_connection_config_default() {
    let config = ConnectionConfig::default();
    let _ = config;
}

#[test]
fn test_connection_config_with_max_connections() {
    let config = ConnectionConfig::new().with_max_connections(20);
    let pool = ConnectionPool::new(config);
    assert_eq!(pool.max_connections(), 20);
}

#[test]
fn test_connection_config_with_min_connections() {
    let config = ConnectionConfig::new().with_min_connections(5);
    let pool = ConnectionPool::new(config);
    assert_eq!(pool.min_connections(), 5);
}

#[test]
fn test_connection_config_with_idle_timeout() {
    let config = ConnectionConfig::new().with_idle_timeout(Duration::from_secs(600));
    let _ = ConnectionPool::new(config);
}

#[test]
fn test_connection_pool_new() {
    let config = ConnectionConfig::new();
    let pool = ConnectionPool::new(config);

    assert_eq!(pool.active_connections(), 0);
    assert_eq!(pool.idle_connections(), 0);
}

#[test]
fn test_connection_pool_acquire() {
    let config = ConnectionConfig::new().with_max_connections(5);
    let pool = ConnectionPool::new(config);

    let conn = pool.acquire();
    assert!(conn.is_ok());
    assert_eq!(pool.active_connections(), 1);
}

#[test]
fn test_connection_pool_release() {
    let config = ConnectionConfig::new();
    let pool = ConnectionPool::new(config);

    let conn = pool.acquire().expect("test value should be present");
    pool.release(conn);

    assert_eq!(pool.active_connections(), 0);
    assert_eq!(pool.idle_connections(), 1);
}

#[test]
fn test_connection_pool_reuse() {
    let config = ConnectionConfig::new();
    let pool = ConnectionPool::new(config);

    // Acquire and release
    let conn1 = pool.acquire().expect("test value should be present");
    pool.release(conn1);

    // Should reuse from idle pool
    let _conn2 = pool.acquire().expect("test value should be present");
    assert_eq!(pool.idle_connections(), 0);
}

#[test]
fn test_connection_pool_exhausted() {
    let config = ConnectionConfig::new().with_max_connections(2);
    let pool = ConnectionPool::new(config);

    let _conn1 = pool.acquire().expect("test value should be present");
    let _conn2 = pool.acquire().expect("test value should be present");

    // Third should fail
    let result = pool.acquire();
    assert!(result.is_err());
}

#[test]
fn test_connection_pool_try_acquire() {
    let config = ConnectionConfig::new();
    let pool = ConnectionPool::new(config);

    let result = pool.try_acquire();
    assert!(result.is_ok());
}

#[test]
fn test_connection_pool_check_health() {
    let config = ConnectionConfig::new().with_idle_timeout(Duration::from_millis(1));
    let pool = ConnectionPool::new(config);

    let conn = pool.acquire().expect("test value should be present");

    // New connection should be healthy
    let health = pool.check_health(&conn);
    assert!(health == ConnectionState::Healthy || health == ConnectionState::Stale);
}

#[test]
fn test_connection_pool_warm() {
    let config = ConnectionConfig::new().with_min_connections(3);
    let pool = ConnectionPool::new(config);

    pool.warm();

    assert_eq!(pool.idle_connections(), 3);
}

// ============================================================================
// ResourceLimiter Tests
// ============================================================================

#[test]
fn test_resource_config_new() {
    let config = ResourceConfig::new();
    let _ = config;
}

#[test]
fn test_resource_config_default() {
    let config = ResourceConfig::default();
    let _ = config;
}

#[test]
fn test_resource_config_with_max_memory_per_request() {
    let config = ResourceConfig::new().with_max_memory_per_request(1024 * 1024);
    let _ = ResourceLimiter::new(config);
}

#[test]
fn test_resource_config_with_max_total_memory() {
    let config = ResourceConfig::new().with_max_total_memory(2 * 1024 * 1024 * 1024);
    let _ = ResourceLimiter::new(config);
}

#[test]
fn test_resource_config_with_max_compute_time() {
    let config = ResourceConfig::new().with_max_compute_time(Duration::from_secs(60));
    let _ = ResourceLimiter::new(config);
}

#[test]
fn test_resource_config_with_max_queue_depth() {
    let config = ResourceConfig::new().with_max_queue_depth(50);
    let _ = ResourceLimiter::new(config);
}

#[test]
fn test_resource_limiter_new() {
    let config = ResourceConfig::new();
    let limiter = ResourceLimiter::new(config);
    assert_eq!(limiter.current_memory(), 0);
}

#[test]
fn test_resource_limiter_check_memory_allowed() {
    let config = ResourceConfig::new()
        .with_max_memory_per_request(1024)
        .with_max_total_memory(10240);
    let limiter = ResourceLimiter::new(config);

    let result = limiter.check_memory(512);
    assert!(matches!(result, LimitResult::Allowed));
}
