
// ============================================================================
// Tests (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // LoadTestConfig tests
    // =========================================================================

    #[test]
    fn test_load_test_config_default() {
        let config = LoadTestConfig::default();
        assert_eq!(config.concurrency, 10);
        assert_eq!(config.duration_secs, 60);
        assert_eq!(config.target_rps, 0.0);
        assert_eq!(config.timeout_ms, 5000);
        assert_eq!(config.warmup_secs, 5);
        assert!((config.latency_threshold_ms - 500.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_load_test_config_for_stress_test() {
        let config = LoadTestConfig::for_stress_test();
        assert_eq!(config.concurrency, 100);
        assert_eq!(config.duration_secs, 300);
        assert_eq!(config.timeout_ms, 10_000);
        assert_eq!(config.warmup_secs, 10);
        assert!((config.latency_threshold_ms - 1000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_load_test_config_for_latency_test() {
        let config = LoadTestConfig::for_latency_test();
        assert_eq!(config.concurrency, 1);
        assert_eq!(config.duration_secs, 60);
        assert!((config.target_rps - 10.0).abs() < f64::EPSILON);
        assert_eq!(config.timeout_ms, 2000);
        assert!((config.latency_threshold_ms - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_load_test_config_is_valid() {
        let config = LoadTestConfig::default();
        assert!(config.is_valid());
    }

    #[test]
    fn test_load_test_config_is_valid_zero_concurrency() {
        let config = LoadTestConfig {
            concurrency: 0,
            ..Default::default()
        };
        assert!(!config.is_valid());
    }

    #[test]
    fn test_load_test_config_is_valid_zero_duration() {
        let config = LoadTestConfig {
            duration_secs: 0,
            ..Default::default()
        };
        assert!(!config.is_valid());
    }

    #[test]
    fn test_load_test_config_is_valid_zero_timeout() {
        let config = LoadTestConfig {
            timeout_ms: 0,
            ..Default::default()
        };
        assert!(!config.is_valid());
    }

    #[test]
    fn test_load_test_config_is_valid_zero_threshold() {
        let config = LoadTestConfig {
            latency_threshold_ms: 0.0,
            ..Default::default()
        };
        assert!(!config.is_valid());
    }

    #[test]
    fn test_load_test_config_is_valid_negative_threshold() {
        let config = LoadTestConfig {
            latency_threshold_ms: -1.0,
            ..Default::default()
        };
        assert!(!config.is_valid());
    }

    #[test]
    fn test_load_test_config_clone() {
        let config = LoadTestConfig::for_stress_test();
        let cloned = config.clone();
        assert_eq!(config.concurrency, cloned.concurrency);
        assert_eq!(config.duration_secs, cloned.duration_secs);
    }

    #[test]
    fn test_load_test_config_debug() {
        let config = LoadTestConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("LoadTestConfig"));
        assert!(debug.contains("concurrency"));
    }

    #[test]
    fn test_load_test_config_serialization() {
        let config = LoadTestConfig::default();
        let json = serde_json::to_string(&config).expect("serialize");
        assert!(json.contains("concurrency"));
        assert!(json.contains("60")); // duration_secs
    }

    #[test]
    fn test_load_test_config_deserialization() {
        let json = r#"{"concurrency": 50, "duration_secs": 120, "target_rps": 100.0, "timeout_ms": 3000, "warmup_secs": 10, "latency_threshold_ms": 300.0}"#;
        let config: LoadTestConfig = serde_json::from_str(json).expect("deserialize");
        assert_eq!(config.concurrency, 50);
        assert_eq!(config.duration_secs, 120);
        assert!((config.target_rps - 100.0).abs() < f64::EPSILON);
    }

    // =========================================================================
    // LoadTestResult tests
    // =========================================================================

    #[test]
    fn test_load_test_result_is_passing_true() {
        let result = LoadTestResult {
            total_requests: 1000,
            successful_requests: 995,
            failed_requests: 5,
            rps_achieved: 100.0,
            latency_p50_ms: 10.0,
            latency_p95_ms: 20.0,
            latency_p99_ms: 30.0,
            latency_max_ms: 50.0,
            data_transferred_bytes: 1_000_000,
            duration_secs: 10.0,
            error_rate: 0.005, // 0.5% < 1%
            passed_latency_threshold: true,
        };
        assert!(result.is_passing());
    }

    #[test]
    fn test_load_test_result_is_passing_false_latency() {
        let result = LoadTestResult {
            total_requests: 1000,
            successful_requests: 995,
            failed_requests: 5,
            rps_achieved: 100.0,
            latency_p50_ms: 10.0,
            latency_p95_ms: 20.0,
            latency_p99_ms: 30.0,
            latency_max_ms: 50.0,
            data_transferred_bytes: 1_000_000,
            duration_secs: 10.0,
            error_rate: 0.005,
            passed_latency_threshold: false,
        };
        assert!(!result.is_passing());
    }

    #[test]
    fn test_load_test_result_is_passing_false_error_rate() {
        let result = LoadTestResult {
            total_requests: 100,
            successful_requests: 80,
            failed_requests: 20,
            rps_achieved: 10.0,
            latency_p50_ms: 10.0,
            latency_p95_ms: 20.0,
            latency_p99_ms: 30.0,
            latency_max_ms: 50.0,
            data_transferred_bytes: 100_000,
            duration_secs: 10.0,
            error_rate: 0.20, // 20% > 1%
            passed_latency_threshold: true,
        };
        assert!(!result.is_passing());
    }

    #[test]
    fn test_load_test_result_throughput_mbps() {
        let result = LoadTestResult {
            total_requests: 1000,
            successful_requests: 1000,
            failed_requests: 0,
            rps_achieved: 100.0,
            latency_p50_ms: 10.0,
            latency_p95_ms: 20.0,
            latency_p99_ms: 30.0,
            latency_max_ms: 50.0,
            data_transferred_bytes: 10_000_000, // 10 MB
            duration_secs: 10.0,
            error_rate: 0.0,
            passed_latency_threshold: true,
        };
        // 10 MB / 10 sec = 1 MB/s
        assert!((result.throughput_mbps() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_load_test_result_throughput_mbps_zero_duration() {
        let result = LoadTestResult {
            total_requests: 1000,
            successful_requests: 1000,
            failed_requests: 0,
            rps_achieved: 0.0,
            latency_p50_ms: 10.0,
            latency_p95_ms: 20.0,
            latency_p99_ms: 30.0,
            latency_max_ms: 50.0,
            data_transferred_bytes: 10_000_000,
            duration_secs: 0.0,
            error_rate: 0.0,
            passed_latency_threshold: true,
        };
        assert!((result.throughput_mbps() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_load_test_result_clone() {
        let result = LoadTestResult {
            total_requests: 500,
            successful_requests: 490,
            failed_requests: 10,
            rps_achieved: 50.0,
            latency_p50_ms: 15.0,
            latency_p95_ms: 25.0,
            latency_p99_ms: 35.0,
            latency_max_ms: 55.0,
            data_transferred_bytes: 500_000,
            duration_secs: 10.0,
            error_rate: 0.02,
            passed_latency_threshold: true,
        };
        let cloned = result.clone();
        assert_eq!(result.total_requests, cloned.total_requests);
        assert!((result.latency_p50_ms - cloned.latency_p50_ms).abs() < f64::EPSILON);
    }

    #[test]
    fn test_load_test_result_debug() {
        let result = LoadTestResult {
            total_requests: 100,
            successful_requests: 100,
            failed_requests: 0,
            rps_achieved: 10.0,
            latency_p50_ms: 10.0,
            latency_p95_ms: 20.0,
            latency_p99_ms: 30.0,
            latency_max_ms: 40.0,
            data_transferred_bytes: 100_000,
            duration_secs: 10.0,
            error_rate: 0.0,
            passed_latency_threshold: true,
        };
        let debug = format!("{:?}", result);
        assert!(debug.contains("LoadTestResult"));
        assert!(debug.contains("total_requests"));
    }

    #[test]
    fn test_load_test_result_serialization() {
        let result = LoadTestResult {
            total_requests: 1000,
            successful_requests: 990,
            failed_requests: 10,
            rps_achieved: 100.0,
            latency_p50_ms: 10.0,
            latency_p95_ms: 20.0,
            latency_p99_ms: 30.0,
            latency_max_ms: 50.0,
            data_transferred_bytes: 1_000_000,
            duration_secs: 10.0,
            error_rate: 0.01,
            passed_latency_threshold: true,
        };
        let json = serde_json::to_string(&result).expect("serialize");
        assert!(json.contains("total_requests"));
        assert!(json.contains("1000"));
    }

    // =========================================================================
    // LoadTestRunner tests
    // =========================================================================

    #[test]
    fn test_load_test_runner_new() {
        let config = LoadTestConfig::default();
        let runner = LoadTestRunner::new(config.clone());
        assert_eq!(runner.config().concurrency, config.concurrency);
    }

    #[test]
    fn test_load_test_runner_config() {
        let config = LoadTestConfig::for_stress_test();
        let runner = LoadTestRunner::new(config);
        assert_eq!(runner.config().concurrency, 100);
    }

    #[test]
    fn test_load_test_runner_debug() {
        let runner = LoadTestRunner::new(LoadTestConfig::default());
        let debug = format!("{:?}", runner);
        assert!(debug.contains("LoadTestRunner"));
    }

    #[test]
    fn test_load_test_runner_simulate_run_basic() {
        let config = LoadTestConfig::default();
        let runner = LoadTestRunner::new(config);
        let result = runner.simulate_run();

        assert!(result.total_requests > 0);
        assert!(result.successful_requests <= result.total_requests);
        assert_eq!(
            result.total_requests,
            result.successful_requests + result.failed_requests
        );
        assert!(result.rps_achieved > 0.0);
    }

    #[test]
    fn test_load_test_runner_simulate_run_stress() {
        let config = LoadTestConfig::for_stress_test();
        let runner = LoadTestRunner::new(config);
        let result = runner.simulate_run();

        // Stress test should have more requests
        assert!(result.total_requests > 10_000);
        assert!(result.latency_p50_ms > 0.0);
        assert!(result.latency_p99_ms >= result.latency_p50_ms);
    }

    #[test]
    fn test_load_test_runner_simulate_run_latency() {
        let config = LoadTestConfig::for_latency_test();
        let runner = LoadTestRunner::new(config);
        let result = runner.simulate_run();

        // Latency test has concurrency=1, so fewer requests
        assert!(result.total_requests > 0);
        // Lower concurrency should mean lower latencies
        assert!(result.latency_p50_ms < 50.0);
    }

    #[test]
    fn test_load_test_runner_simulate_run_error_rate() {
        let config = LoadTestConfig::default();
        let runner = LoadTestRunner::new(config);
        let result = runner.simulate_run();

        // Simulated error rate is ~1%
        assert!(result.error_rate >= 0.0);
        assert!(result.error_rate <= 0.02);
    }

    #[test]
    fn test_load_test_runner_simulate_run_latency_percentiles_order() {
        let config = LoadTestConfig::default();
        let runner = LoadTestRunner::new(config);
        let result = runner.simulate_run();

        // Percentiles should be in ascending order
        assert!(result.latency_p50_ms <= result.latency_p95_ms);
        assert!(result.latency_p95_ms <= result.latency_p99_ms);
        assert!(result.latency_p99_ms <= result.latency_max_ms);
    }

    #[test]
    fn test_load_test_runner_simulate_run_data_transferred() {
        let config = LoadTestConfig::default();
        let runner = LoadTestRunner::new(config);
        let result = runner.simulate_run();

        // Data should be approximately total_requests * 1KB
        let expected_data = (result.total_requests * 1024) as u64;
        assert_eq!(result.data_transferred_bytes, expected_data);
    }

    #[test]
    fn test_load_test_runner_simulate_run_duration() {
        let config = LoadTestConfig {
            duration_secs: 30,
            ..Default::default()
        };
        let runner = LoadTestRunner::new(config);
        let result = runner.simulate_run();

        assert!((result.duration_secs - 30.0).abs() < f64::EPSILON);
    }

    // =========================================================================
    // Edge case tests
    // =========================================================================

    #[test]
    fn test_load_test_config_extreme_values() {
        let config = LoadTestConfig {
            concurrency: 10000,
            duration_secs: 3600,
            target_rps: 1_000_000.0,
            timeout_ms: 60_000,
            warmup_secs: 60,
            latency_threshold_ms: 10.0,
        };
        assert!(config.is_valid());
    }

    #[test]
    fn test_load_test_result_boundary_error_rate() {
        // Exactly 1% error rate - should still fail (< not <=)
        let result = LoadTestResult {
            total_requests: 100,
            successful_requests: 99,
            failed_requests: 1,
            rps_achieved: 10.0,
            latency_p50_ms: 10.0,
            latency_p95_ms: 20.0,
            latency_p99_ms: 30.0,
            latency_max_ms: 40.0,
            data_transferred_bytes: 100_000,
            duration_secs: 10.0,
            error_rate: 0.01, // Exactly 1%
            passed_latency_threshold: true,
        };
        // 0.01 is NOT < 0.01, so should NOT pass
        assert!(!result.is_passing());
    }

    #[test]
    fn test_load_test_result_just_under_error_threshold() {
        let result = LoadTestResult {
            total_requests: 1000,
            successful_requests: 991,
            failed_requests: 9,
            rps_achieved: 100.0,
            latency_p50_ms: 10.0,
            latency_p95_ms: 20.0,
            latency_p99_ms: 30.0,
            latency_max_ms: 40.0,
            data_transferred_bytes: 1_000_000,
            duration_secs: 10.0,
            error_rate: 0.009, // Just under 1%
            passed_latency_threshold: true,
        };
        assert!(result.is_passing());
    }
}
