
    // ==================== ErrorCategory Tests ====================

    #[test]
    fn test_error_category_transient() {
        let cat = ErrorCategory::Transient;
        assert_eq!(cat, ErrorCategory::Transient);
        assert_ne!(cat, ErrorCategory::Permanent);
    }

    #[test]
    fn test_error_category_permanent() {
        let cat = ErrorCategory::Permanent;
        assert_eq!(cat, ErrorCategory::Permanent);
    }

    #[test]
    fn test_error_category_clone_copy() {
        let cat = ErrorCategory::Transient;
        let cloned = cat;
        let copied: ErrorCategory = cat;
        assert_eq!(cat, cloned);
        assert_eq!(cat, copied);
    }

    // ==================== RetryDecision Tests ====================

    #[test]
    fn test_retry_decision_retry() {
        let decision = RetryDecision::Retry {
            delay: Duration::from_millis(100),
        };
        if let RetryDecision::Retry { delay } = decision {
            assert_eq!(delay.as_millis(), 100);
        } else {
            panic!("Expected Retry variant");
        }
    }

    #[test]
    fn test_retry_decision_abort() {
        let decision = RetryDecision::Abort {
            reason: "test".to_string(),
        };
        if let RetryDecision::Abort { reason } = decision {
            assert_eq!(reason, "test");
        } else {
            panic!("Expected Abort variant");
        }
    }

    #[test]
    fn test_retry_decision_clone() {
        let decision = RetryDecision::Retry {
            delay: Duration::from_secs(1),
        };
        let cloned = decision.clone();
        if let RetryDecision::Retry { delay } = cloned {
            assert_eq!(delay.as_secs(), 1);
        } else {
            panic!("Expected Retry variant");
        }
    }

    // ==================== RetryConfig Tests ====================

    #[test]
    fn test_retry_config_new() {
        let config = RetryConfig::new();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.base_delay.as_millis(), 100);
        assert_eq!(config.max_delay.as_secs(), 30);
    }

    #[test]
    fn test_retry_config_default() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_retry_config_with_max_retries() {
        let config = RetryConfig::new().with_max_retries(5);
        assert_eq!(config.max_retries, 5);
    }

    #[test]
    fn test_retry_config_with_base_delay() {
        let config = RetryConfig::new().with_base_delay(Duration::from_millis(200));
        assert_eq!(config.base_delay.as_millis(), 200);
    }

    #[test]
    fn test_retry_config_with_max_delay() {
        let config = RetryConfig::new().with_max_delay(Duration::from_secs(60));
        assert_eq!(config.max_delay.as_secs(), 60);
    }

    #[test]
    fn test_retry_config_with_jitter_factor() {
        let config = RetryConfig::new().with_jitter_factor(0.2);
        assert!((config.jitter_factor - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_retry_config_jitter_clamped() {
        let config = RetryConfig::new().with_jitter_factor(2.0);
        assert!((config.jitter_factor - 1.0).abs() < 1e-6);

        let config2 = RetryConfig::new().with_jitter_factor(-0.5);
        assert!((config2.jitter_factor - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_retry_config_clone() {
        let config = RetryConfig::new().with_max_retries(10);
        let cloned = config.clone();
        assert_eq!(cloned.max_retries, 10);
    }

    // ==================== RetryPolicy Tests ====================

    #[test]
    fn test_retry_policy_new() {
        let config = RetryConfig::new();
        let policy = RetryPolicy::new(config);
        assert_eq!(policy.max_retries(), 3);
    }

    #[test]
    fn test_retry_policy_should_retry_transient() {
        let config = RetryConfig::new().with_max_retries(3);
        let policy = RetryPolicy::new(config);

        let decision = policy.should_retry(1, ErrorCategory::Transient);
        assert!(matches!(decision, RetryDecision::Retry { .. }));
    }

    #[test]
    fn test_retry_policy_should_retry_permanent() {
        let config = RetryConfig::new();
        let policy = RetryPolicy::new(config);

        let decision = policy.should_retry(1, ErrorCategory::Permanent);
        if let RetryDecision::Abort { reason } = decision {
            assert!(reason.contains("Permanent"));
        } else {
            panic!("Expected Abort for permanent error");
        }
    }

    #[test]
    fn test_retry_policy_max_retries_exceeded() {
        let config = RetryConfig::new().with_max_retries(2);
        let policy = RetryPolicy::new(config);

        let decision = policy.should_retry(3, ErrorCategory::Transient);
        if let RetryDecision::Abort { reason } = decision {
            assert!(reason.contains("exceeded"));
        } else {
            panic!("Expected Abort when max retries exceeded");
        }
    }

    #[test]
    fn test_retry_policy_calculate_delay_exponential() {
        let config = RetryConfig::new()
            .with_base_delay(Duration::from_millis(100))
            .with_max_delay(Duration::from_secs(60));
        let policy = RetryPolicy::new(config);

        // attempt 0: 100 * 2^0 = 100ms
        assert_eq!(policy.calculate_delay(0).as_millis(), 100);
        // attempt 1: 100 * 2^1 = 200ms
        assert_eq!(policy.calculate_delay(1).as_millis(), 200);
        // attempt 2: 100 * 2^2 = 400ms
        assert_eq!(policy.calculate_delay(2).as_millis(), 400);
    }

    #[test]
    fn test_retry_policy_calculate_delay_capped() {
        let config = RetryConfig::new()
            .with_base_delay(Duration::from_millis(100))
            .with_max_delay(Duration::from_millis(500));
        let policy = RetryPolicy::new(config);

        // attempt 5: 100 * 2^5 = 3200ms, but capped at 500ms
        let delay = policy.calculate_delay(5);
        assert_eq!(delay.as_millis(), 500);
    }

    // ==================== CircuitState Tests ====================

    #[test]
    fn test_circuit_state_values() {
        assert_eq!(CircuitState::Closed, CircuitState::Closed);
        assert_eq!(CircuitState::Open, CircuitState::Open);
        assert_eq!(CircuitState::HalfOpen, CircuitState::HalfOpen);
        assert_ne!(CircuitState::Closed, CircuitState::Open);
    }

    #[test]
    fn test_circuit_state_clone_copy() {
        let state = CircuitState::HalfOpen;
        let cloned = state;
        let copied: CircuitState = state;
        assert_eq!(state, cloned);
        assert_eq!(state, copied);
    }

    // ==================== CircuitConfig Tests ====================

    #[test]
    fn test_circuit_config_new() {
        let config = CircuitConfig::new();
        assert_eq!(config.failure_threshold, 5);
        assert_eq!(config.success_threshold, 2);
        assert_eq!(config.timeout.as_secs(), 30);
    }

    #[test]
    fn test_circuit_config_default() {
        let config = CircuitConfig::default();
        assert_eq!(config.failure_threshold, 5);
    }

    #[test]
    fn test_circuit_config_with_failure_threshold() {
        let config = CircuitConfig::new().with_failure_threshold(10);
        assert_eq!(config.failure_threshold, 10);
    }

    #[test]
    fn test_circuit_config_with_success_threshold() {
        let config = CircuitConfig::new().with_success_threshold(3);
        assert_eq!(config.success_threshold, 3);
    }

    #[test]
    fn test_circuit_config_with_timeout() {
        let config = CircuitConfig::new().with_timeout(Duration::from_secs(60));
        assert_eq!(config.timeout.as_secs(), 60);
    }

    #[test]
    fn test_circuit_config_clone() {
        let config = CircuitConfig::new().with_failure_threshold(8);
        let cloned = config.clone();
        assert_eq!(cloned.failure_threshold, 8);
    }

    // ==================== CircuitBreaker Tests ====================

    #[test]
    fn test_circuit_breaker_new() {
        let config = CircuitConfig::new();
        let breaker = CircuitBreaker::new(config);
        assert_eq!(breaker.state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_allow_request_closed() {
        let config = CircuitConfig::new();
        let breaker = CircuitBreaker::new(config);
        assert!(breaker.allow_request());
    }

    #[test]
    fn test_circuit_breaker_record_failure_opens() {
        let config = CircuitConfig::new().with_failure_threshold(2);
        let breaker = CircuitBreaker::new(config);

        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Closed);
        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);
    }

    #[test]
    fn test_circuit_breaker_open_blocks_requests() {
        let config = CircuitConfig::new()
            .with_failure_threshold(1)
            .with_timeout(Duration::from_secs(60));
        let breaker = CircuitBreaker::new(config);

        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);
        assert!(!breaker.allow_request());
    }

    #[test]
    fn test_circuit_breaker_record_success_resets_failures() {
        let config = CircuitConfig::new().with_failure_threshold(3);
        let breaker = CircuitBreaker::new(config);

        breaker.record_failure();
        breaker.record_failure();
        breaker.record_success(); // Should reset failure count
        breaker.record_failure();
        breaker.record_failure();
        // Still closed because success reset the count
        assert_eq!(breaker.state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_half_open_to_closed() {
        let config = CircuitConfig::new()
            .with_failure_threshold(1)
            .with_success_threshold(2)
            .with_timeout(Duration::from_millis(1));
        let breaker = CircuitBreaker::new(config);

        // Open the circuit
        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);

        // Wait for timeout to transition to half-open
        std::thread::sleep(Duration::from_millis(5));
        assert!(breaker.allow_request()); // Should transition to HalfOpen
        assert_eq!(breaker.state(), CircuitState::HalfOpen);

        // Record successes to close
        breaker.record_success();
        breaker.record_success();
        assert_eq!(breaker.state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_half_open_failure_reopens() {
        let config = CircuitConfig::new()
            .with_failure_threshold(1)
            .with_timeout(Duration::from_millis(1));
        let breaker = CircuitBreaker::new(config);

        breaker.record_failure();
        std::thread::sleep(Duration::from_millis(5));
        let _ = breaker.allow_request(); // Transition to HalfOpen
        assert_eq!(breaker.state(), CircuitState::HalfOpen);

        breaker.record_failure(); // Should reopen
        assert_eq!(breaker.state(), CircuitState::Open);
    }

    // ==================== RequestType Tests ====================

    #[test]
    fn test_request_type_values() {
        assert_eq!(RequestType::Inference, RequestType::Inference);
        assert_eq!(RequestType::Embedding, RequestType::Embedding);
        assert_eq!(RequestType::Batch, RequestType::Batch);
        assert_ne!(RequestType::Inference, RequestType::Batch);
    }

    #[test]
    fn test_request_type_clone_copy() {
        let rt = RequestType::Embedding;
        let cloned = rt;
        let copied: RequestType = rt;
        assert_eq!(rt, cloned);
        assert_eq!(rt, copied);
    }

    // ==================== BulkheadConfig Tests ====================

    #[test]
    fn test_bulkhead_config_new() {
        let config = BulkheadConfig::new();
        assert!(config.pools.is_empty());
    }

    #[test]
    fn test_bulkhead_config_default() {
        let config = BulkheadConfig::default();
        assert!(config.pools.is_empty());
    }

    #[test]
    fn test_bulkhead_config_with_pool() {
        let config = BulkheadConfig::new()
            .with_pool("inference", 10)
            .with_pool("embedding", 5);
        assert_eq!(config.pools.get("inference"), Some(&10));
        assert_eq!(config.pools.get("embedding"), Some(&5));
    }

    // ==================== BulkheadStats Tests ====================

    #[test]
    fn test_bulkhead_stats_fields() {
        let stats = BulkheadStats {
            pool_count: 3,
            total_capacity: 17,
        };
        assert_eq!(stats.pool_count, 3);
        assert_eq!(stats.total_capacity, 17);
    }

    #[test]
    fn test_bulkhead_stats_clone() {
        let stats = BulkheadStats {
            pool_count: 2,
            total_capacity: 10,
        };
        let cloned = stats.clone();
        assert_eq!(cloned.pool_count, stats.pool_count);
    }

    // ==================== BulkheadPermit Tests ====================

    #[test]
    fn test_bulkhead_permit_debug() {
        let permit = BulkheadPermit {
            request_type: RequestType::Inference,
        };
        let debug_str = format!("{:?}", permit);
        assert!(debug_str.contains("Inference"));
    }

    // ==================== BulkheadManager Tests ====================

    #[test]
    fn test_bulkhead_manager_new_defaults() {
        let config = BulkheadConfig::new();
        let manager = BulkheadManager::new(&config);

        assert_eq!(manager.available(RequestType::Inference), 10);
        assert_eq!(manager.available(RequestType::Embedding), 5);
        assert_eq!(manager.available(RequestType::Batch), 2);
    }

    #[test]
    fn test_bulkhead_manager_new_custom() {
        let config = BulkheadConfig::new()
            .with_pool("inference", 20)
            .with_pool("embedding", 8);
        let manager = BulkheadManager::new(&config);

        assert_eq!(manager.available(RequestType::Inference), 20);
        assert_eq!(manager.available(RequestType::Embedding), 8);
    }
