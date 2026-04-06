
// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // WARM-001: Configuration Tests
    #[test]
    fn test_warmup_config_default_and_builder() {
        let def = WarmupConfig::default();
        assert_eq!(def.warmup_iterations, 3);
        assert_eq!(def.timeout, Duration::from_secs(60));
        assert!(def.validate_output);

        let built = WarmupConfig::new()
            .with_warmup_iterations(5)
            .with_timeout(Duration::from_secs(120))
            .with_sample_prompt("Test")
            .with_sample_max_tokens(20)
            .with_validate_output(false)
            .with_gc_after_warmup(false)
            .with_verbose(true);
        assert_eq!(built.warmup_iterations, 5);
        assert_eq!(built.sample_prompt, "Test");
        assert!(!built.validate_output);
        assert!(!built.gc_after_warmup);
        assert!(built.verbose);

        // Min iterations clamping
        assert_eq!(
            WarmupConfig::new()
                .with_warmup_iterations(0)
                .warmup_iterations,
            1
        );
    }

    // WARM-002: Status Tests
    #[test]
    fn test_warmup_status_methods() {
        assert!(!WarmupStatus::NotStarted.is_ready());
        assert!(WarmupStatus::Ready.is_ready());
        assert!(WarmupStatus::InProgress.is_in_progress());
        assert!(!WarmupStatus::Ready.is_in_progress());
        assert!(WarmupStatus::Failed.has_failed());
        assert!(WarmupStatus::TimedOut.has_failed());
        assert!(!WarmupStatus::Ready.has_failed());
    }

    // WARM-003: Result Tests
    #[test]
    fn test_warmup_result_variants() {
        // Success with latencies
        let latencies = vec![
            Duration::from_millis(100),
            Duration::from_millis(50),
            Duration::from_millis(25),
        ];
        let success = WarmupResult::success(3, Duration::from_millis(200), &latencies);
        assert_eq!(success.status, WarmupStatus::Ready);
        assert_eq!(success.first_latency, Duration::from_millis(100));
        assert_eq!(success.last_latency, Duration::from_millis(25));
        assert!(success.speedup_factor > 1.0);
        assert!(success.error.is_none());

        // Failed
        let failed = WarmupResult::failed("Test error", 2, Duration::from_secs(5));
        assert_eq!(failed.status, WarmupStatus::Failed);
        assert_eq!(failed.error, Some("Test error".to_string()));

        // Timed out
        let timeout = WarmupResult::timed_out(1, Duration::from_secs(60));
        assert_eq!(timeout.status, WarmupStatus::TimedOut);
        assert!(timeout.error.expect("err").contains("timed out"));

        // Empty latencies
        let empty = WarmupResult::success(0, Duration::ZERO, &[]);
        assert_eq!(empty.first_latency, Duration::ZERO);
        assert!((empty.speedup_factor - 1.0).abs() < f64::EPSILON);

        // Zero last latency edge case
        let zero = WarmupResult::success(
            2,
            Duration::from_millis(100),
            &[Duration::from_millis(100), Duration::ZERO],
        );
        assert!((zero.speedup_factor - 1.0).abs() < f64::EPSILON);

        // Average latency calculation
        let avg_test = WarmupResult::success(
            3,
            Duration::from_millis(600),
            &[
                Duration::from_millis(100),
                Duration::from_millis(200),
                Duration::from_millis(300),
            ],
        );
        assert_eq!(avg_test.avg_latency, Duration::from_millis(200));
    }

    // WARM-004: Health Tests
    #[test]
    fn test_model_health_basic() {
        let health = ModelHealth::new();
        assert!(!health.is_ready());
        assert_eq!(health.status(), WarmupStatus::NotStarted);
        assert_eq!(health.total_requests(), 0);

        health.set_ready(true);
        assert!(health.is_ready());
        health.set_ready(false);
        assert!(!health.is_ready());

        health.set_status(WarmupStatus::Ready);
        assert!(health.is_ready());
        assert_eq!(health.status(), WarmupStatus::Ready);

        health.record_success();
        health.record_success();
        health.record_failure();
        assert_eq!(health.total_requests(), 2);
        assert_eq!(health.failed_requests(), 1);
        assert!((health.error_rate() - 0.5).abs() < f64::EPSILON);

        // Default trait
        let def = ModelHealth::default();
        assert!(!def.is_ready());
    }

    #[test]
    fn test_model_health_timing_and_report() {
        let health = ModelHealth::new();
        std::thread::sleep(Duration::from_millis(5));
        assert!(health.uptime() >= Duration::from_millis(5));

        let before = health.time_since_last_check();
        health.touch();
        assert!(health.time_since_last_check() < before);

        health.set_status(WarmupStatus::Ready);
        health.record_success();
        let report = health.report();
        assert!(report.ready);
        assert_eq!(report.status, WarmupStatus::Ready);
        assert_eq!(report.total_requests, 1);
    }

    #[test]
    fn test_model_health_clone_shares_state() {
        let health = ModelHealth::new();
        health.set_status(WarmupStatus::Ready);
        health.record_success();
        let cloned = health.clone();
        assert!(cloned.is_ready());
        health.record_success();
        assert_eq!(cloned.total_requests(), 2); // Shared Arc state
    }

    // WARM-005: Executor Tests
    #[test]
    fn test_warmup_executor() {
        let executor = WarmupExecutor::new(WarmupConfig::new().with_warmup_iterations(3));
        assert_eq!(executor.config().warmup_iterations, 3);

        let result = executor.simulate_warmup();
        assert_eq!(result.status, WarmupStatus::Ready);
        assert!(result.first_latency > result.last_latency);

        // Default trait
        assert_eq!(WarmupExecutor::default().config().warmup_iterations, 3);

        // Timeout check
        let timeout_exec =
            WarmupExecutor::new(WarmupConfig::new().with_timeout(Duration::from_millis(1)));
        let start = Instant::now();
        std::thread::sleep(Duration::from_millis(10));
        assert!(timeout_exec.check_timeout(start, 0).is_some());

        // No timeout
        let long_exec =
            WarmupExecutor::new(WarmupConfig::new().with_timeout(Duration::from_secs(60)));
        assert!(long_exec.check_timeout(Instant::now(), 5).is_none());

        // Many iterations (jitter clamping)
        let many = WarmupExecutor::new(WarmupConfig::new().with_warmup_iterations(10));
        let res = many.simulate_warmup();
        assert_eq!(res.iterations_completed, 10);
        assert!(res.avg_latency > Duration::ZERO);
    }

    // WARM-006: Preload Config Tests
    #[test]
    fn test_preload_config() {
        let def = PreloadConfig::default();
        assert!(def.models.is_empty());
        assert!(def.parallel_loading);
        assert_eq!(def.max_concurrent, 4);

        let model = PreloadModelConfig::new("llama", "pacha://llama:7b");
        let config = PreloadConfig::new()
            .with_model(model)
            .with_parallel_loading(false)
            .with_max_concurrent(2)
            .with_fail_fast(true);
        assert_eq!(config.models.len(), 1);
        assert!(!config.parallel_loading);
        assert!(config.fail_fast);

        // Min concurrent
        assert_eq!(
            PreloadConfig::new().with_max_concurrent(0).max_concurrent,
            1
        );

        // Multiple models
        let multi = PreloadConfig::new()
            .with_model(PreloadModelConfig::new("m1", "f://1").with_priority(10))
            .with_model(PreloadModelConfig::new("m2", "f://2").with_priority(5));
        assert_eq!(multi.models.len(), 2);
    }

    #[test]
    fn test_preload_model_config() {
        let basic = PreloadModelConfig::new("gpt2", "hf://gpt2");
        assert_eq!(basic.model_id, "gpt2");
        assert_eq!(basic.priority, 100);
        assert!(basic.warmup);
        assert!(basic.warmup_config.is_none());

        let built = PreloadModelConfig::new("llama", "file://model.gguf")
            .with_priority(10)
            .with_warmup(false)
            .with_warmup_config(WarmupConfig::new().with_warmup_iterations(5));
        assert_eq!(built.priority, 10);
        assert!(built.warmup); // with_warmup_config enables warmup
        assert_eq!(built.warmup_config.expect("cfg").warmup_iterations, 5);
    }

    // Serialization Tests
    #[test]
    fn test_serialization() {
        // WarmupConfig roundtrip
        let config = WarmupConfig::new()
            .with_warmup_iterations(8)
            .with_sample_prompt("Test");
        let json = serde_json::to_string(&config).expect("ser");
        let deser: WarmupConfig = serde_json::from_str(&json).expect("de");
        assert_eq!(deser.warmup_iterations, 8);

        // All WarmupStatus variants
        for status in [
            WarmupStatus::NotStarted,
            WarmupStatus::InProgress,
            WarmupStatus::Ready,
            WarmupStatus::Failed,
            WarmupStatus::TimedOut,
        ] {
            let j = serde_json::to_string(&status).expect("ser");
            let d: WarmupStatus = serde_json::from_str(&j).expect("de");
            assert_eq!(d, status);
        }

        // WarmupResult
        let result =
            WarmupResult::success(3, Duration::from_millis(100), &[Duration::from_millis(50)]);
        assert!(serde_json::to_string(&result)
            .expect("ser")
            .contains("Ready"));

        // PreloadConfig roundtrip
        let model = PreloadModelConfig::new("test", "file://test.gguf")
            .with_warmup_config(WarmupConfig::new().with_warmup_iterations(7));
        let pc = PreloadConfig::new().with_model(model).with_fail_fast(true);
        let pc_json = serde_json::to_string(&pc).expect("ser");
        let pc_de: PreloadConfig = serde_json::from_str(&pc_json).expect("de");
        assert_eq!(
            pc_de.models[0]
                .warmup_config
                .as_ref()
                .expect("cfg")
                .warmup_iterations,
            7
        );

        // HealthReport
        let report = HealthReport {
            ready: true,
            status: WarmupStatus::Ready,
            uptime_secs: 100.0,
            total_requests: 1000,
            failed_requests: 5,
            error_rate: 0.005,
            time_since_last_check_secs: 1.5,
        };
        assert!(serde_json::to_string(&report)
            .expect("ser")
            .contains("1000"));

        let json2 = r#"{"ready":false,"status":"InProgress","uptime_secs":42.5,"total_requests":500,"failed_requests":10,"error_rate":0.02,"time_since_last_check_secs":0.5}"#;
        let r2: HealthReport = serde_json::from_str(json2).expect("de");
        assert_eq!(r2.total_requests, 500);
    }

    // Debug trait coverage
    #[test]
    fn test_debug_traits() {
        assert!(format!(
            "{:?}",
            WarmupResult::failed("err", 1, Duration::from_secs(1))
        )
        .contains("Failed"));
        assert!(format!("{:?}", WarmupConfig::new()).contains("warmup_iterations"));
        assert!(format!("{:?}", WarmupExecutor::default()).contains("WarmupExecutor"));
    }
}
