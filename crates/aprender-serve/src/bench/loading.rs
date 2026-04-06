//! Part 2 of bench module tests - Configuration, Metrics, Error Handling
//!
//! Focus areas:
//! - Benchmark configuration validation
//! - Metric collection edge cases
//! - Error handling paths
//! - Statistics module coverage
//! - Load testing module coverage

#[cfg(test)]
mod tests {
    use crate::bench::*;
    use std::time::Duration;

    // ========================================================================
    // MeasurementProtocol Tests
    // ========================================================================

    #[test]
    fn test_measurement_protocol_default() {
        let protocol = MeasurementProtocol::default();
        assert_eq!(protocol.latency_samples, 100);
        assert_eq!(
            protocol.latency_percentiles,
            vec![50.0, 90.0, 95.0, 99.0, 99.9]
        );
        assert_eq!(protocol.throughput_duration, Duration::from_secs(60));
        assert_eq!(protocol.throughput_ramp_up, Duration::from_secs(10));
        assert_eq!(protocol.memory_samples, 10);
        assert_eq!(protocol.memory_interval, Duration::from_secs(1));
    }

    #[test]
    fn test_measurement_protocol_builder_chain() {
        let protocol = MeasurementProtocol::new()
            .with_latency_samples(200)
            .with_percentiles(vec![50.0, 99.0, 99.9])
            .with_throughput_duration(Duration::from_secs(30))
            .with_memory_samples(20);

        assert_eq!(protocol.latency_samples, 200);
        assert_eq!(protocol.latency_percentiles, vec![50.0, 99.0, 99.9]);
        assert_eq!(protocol.throughput_duration, Duration::from_secs(30));
        assert_eq!(protocol.memory_samples, 20);
    }

    // ========================================================================
    // LatencyStatistics Tests
    // ========================================================================

    #[test]
    fn test_latency_statistics_from_samples() {
        let samples: Vec<Duration> = (1..=100).map(|i| Duration::from_millis(i * 10)).collect();
        let stats = LatencyStatistics::from_samples(&samples);

        assert_eq!(stats.samples, 100);
        assert!(stats.min == Duration::from_millis(10));
        assert!(stats.max == Duration::from_millis(1000));
        assert!(stats.p50 <= stats.p90);
        assert!(stats.p90 <= stats.p95);
        assert!(stats.p95 <= stats.p99);
        assert!(stats.p99 <= stats.p999);
    }

    #[test]
    fn test_latency_statistics_constant_samples() {
        let samples = vec![Duration::from_millis(50); 100];
        let stats = LatencyStatistics::from_samples(&samples);

        assert_eq!(stats.min, Duration::from_millis(50));
        assert_eq!(stats.max, Duration::from_millis(50));
        assert_eq!(stats.p50, Duration::from_millis(50));
        assert!(stats.std_dev < Duration::from_millis(1));
    }

    #[test]
    fn test_latency_statistics_confidence_interval() {
        let samples: Vec<Duration> = (1..=50).map(|i| Duration::from_millis(i * 2)).collect();
        let stats = LatencyStatistics::from_samples(&samples);
        let (lower, upper) = stats.confidence_interval_95;
        assert!(lower <= stats.mean);
        assert!(upper >= stats.mean);
    }

    // ========================================================================
    // Outlier Detection Tests
    // ========================================================================

    #[test]
    fn test_detect_outliers_no_outliers() {
        let samples: Vec<f64> = (1..=100).map(|i| 50.0 + (i as f64) * 0.1).collect();
        let outliers = detect_outliers(&samples, 3.5);
        assert!(outliers.len() <= 5);
    }

    #[test]
    fn test_detect_outliers_with_outliers() {
        let mut samples: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        samples.push(1000.0);
        let outliers = detect_outliers(&samples, 3.5);
        assert!(!outliers.is_empty());
        assert!(outliers.contains(&100));
    }

    #[test]
    fn test_detect_outliers_edge_cases() {
        assert!(detect_outliers(&[], 3.5).is_empty());
        assert!(detect_outliers(&[1.0, 2.0], 3.5).is_empty());
        assert!(detect_outliers(&vec![50.0; 100], 3.5).is_empty());
    }

    // ========================================================================
    // RegressionDetector Tests
    // ========================================================================

    #[test]
    fn test_regression_detector_no_change() {
        let detector = RegressionDetector::default();
        let baseline = BenchmarkMetrics {
            name: "latency".to_string(),
            mean: 100.0,
            std_dev: 5.0,
            samples: 50,
        };
        let current = baseline.clone();
        let report = detector.compare(&baseline, &current);
        assert!(report.passed);
        assert!(report.regressions.is_empty());
    }

    #[test]
    fn test_regression_detector_regression() {
        let detector = RegressionDetector::default();
        let baseline = BenchmarkMetrics {
            name: "latency".to_string(),
            mean: 100.0,
            std_dev: 5.0,
            samples: 50,
        };
        let current = BenchmarkMetrics {
            name: "latency".to_string(),
            mean: 110.0,
            std_dev: 5.0,
            samples: 50,
        };
        let report = detector.compare(&baseline, &current);
        assert!(!report.passed);
        assert_eq!(report.regressions.len(), 1);
    }

    #[test]
    fn test_regression_detector_improvement() {
        let detector = RegressionDetector::default();
        let baseline = BenchmarkMetrics {
            name: "latency".to_string(),
            mean: 100.0,
            std_dev: 5.0,
            samples: 50,
        };
        let current = BenchmarkMetrics {
            name: "latency".to_string(),
            mean: 95.0,
            std_dev: 5.0,
            samples: 50,
        };
        let report = detector.compare(&baseline, &current);
        assert!(report.passed);
        assert_eq!(report.improvements.len(), 1);
    }

    // ========================================================================
    // Welch's t-test Tests
    // ========================================================================

    #[test]
    fn test_welch_t_test_identical_samples() {
        let a = vec![10.0, 11.0, 10.5, 10.2, 10.8];
        let result = welch_t_test(&a, &a.clone(), 0.05);
        assert!(!result.significant);
        assert!(result.t_statistic.abs() < 0.001);
    }

    #[test]
    fn test_welch_t_test_different_means() {
        let a = vec![10.0, 11.0, 10.5, 10.2, 10.8];
        let b = vec![20.0, 21.0, 20.5, 20.2, 20.8];
        let result = welch_t_test(&a, &b, 0.05);
        assert!(result.significant);
        assert!(result.t_statistic.abs() > 1.0);
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_welch_t_test_zero_variance() {
        let a = vec![10.0; 10];
        let b = vec![10.0; 10];
        let result = welch_t_test(&a, &b, 0.05);
        assert!(!result.significant);
        assert_eq!(result.p_value, 1.0);
    }

    // ========================================================================
    // LoadTestConfig Tests
    // ========================================================================

    #[test]
    fn test_load_test_config_variants() {
        let default = LoadTestConfig::default();
        assert_eq!(default.concurrency, 10);
        assert!(default.is_valid());

        let stress = LoadTestConfig::for_stress_test();
        assert_eq!(stress.concurrency, 100);
        assert!(stress.is_valid());

        let latency = LoadTestConfig::for_latency_test();
        assert_eq!(latency.concurrency, 1);
        assert!(latency.is_valid());

        let invalid = LoadTestConfig {
            concurrency: 0,
            ..Default::default()
        };
        assert!(!invalid.is_valid());
    }

    // ========================================================================
    // LoadTestResult Tests
    // ========================================================================

    fn create_load_test_result(
        successful: usize,
        failed: usize,
        error_rate: f64,
        passed_latency: bool,
    ) -> LoadTestResult {
        LoadTestResult {
            total_requests: successful + failed,
            successful_requests: successful,
            failed_requests: failed,
            rps_achieved: 100.0,
            latency_p50_ms: 20.0,
            latency_p95_ms: 40.0,
            latency_p99_ms: 60.0,
            latency_max_ms: 100.0,
            data_transferred_bytes: 1_000_000,
            duration_secs: 10.0,
            error_rate,
            passed_latency_threshold: passed_latency,
        }
    }

    #[test]
    fn test_load_test_result_passing_and_failing() {
        let passing = create_load_test_result(995, 5, 0.005, true);
        assert!(passing.is_passing());

        let failing_latency = create_load_test_result(995, 5, 0.005, false);
        assert!(!failing_latency.is_passing());

        let failing_errors = create_load_test_result(900, 100, 0.10, true);
        assert!(!failing_errors.is_passing());
    }

    #[test]
    fn test_load_test_result_throughput_mbps() {
        let result = LoadTestResult {
            total_requests: 100,
            successful_requests: 100,
            failed_requests: 0,
            rps_achieved: 10.0,
            latency_p50_ms: 10.0,
            latency_p95_ms: 20.0,
            latency_p99_ms: 30.0,
            latency_max_ms: 50.0,
            data_transferred_bytes: 10_000_000,
            duration_secs: 10.0,
            error_rate: 0.0,
            passed_latency_threshold: true,
        };
        assert!((result.throughput_mbps() - 1.0).abs() < 0.01);

        let zero_duration = LoadTestResult {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            rps_achieved: 0.0,
            latency_p50_ms: 0.0,
            latency_p95_ms: 0.0,
            latency_p99_ms: 0.0,
            latency_max_ms: 0.0,
            data_transferred_bytes: 0,
            duration_secs: 0.0,
            error_rate: 0.0,
            passed_latency_threshold: true,
        };
        assert_eq!(zero_duration.throughput_mbps(), 0.0);
    }

    // ========================================================================
    // LoadTestRunner Tests
    // ========================================================================

    #[test]
    fn test_load_test_runner() {
        let config = LoadTestConfig {
            concurrency: 10,
            duration_secs: 10,
            ..Default::default()
        };
        let runner = LoadTestRunner::new(config.clone());
        assert_eq!(runner.config().concurrency, config.concurrency);

        let result = runner.simulate_run();
        assert!(result.total_requests > 0);
        assert!(result.successful_requests > 0);
        assert!(result.rps_achieved > 0.0);
        assert!(result.latency_p50_ms > 0.0);
    }

    // ========================================================================
    // ThermalGuard Tests
    // ========================================================================

    #[test]
    fn test_thermal_guard() {
        let guard = ThermalGuard::new(85.0, 75.0, 15_000, 3.0);
        assert!((guard.max_temp_c - 85.0).abs() < 0.001);
        assert!((guard.cooldown_threshold_c - 75.0).abs() < 0.001);
        assert_eq!(guard.cooldown_sleep_ms, 15_000);

        let default_guard = ThermalGuard::default();
        assert!(!default_guard.needs_cooldown(75.0));
        assert!(!default_guard.needs_cooldown(80.0));
        assert!(default_guard.needs_cooldown(81.0));
    }

    // ========================================================================
    // BenchmarkMetrics Clone Test
    // ========================================================================

    #[test]
    fn test_benchmark_metrics_clone() {
        let metrics = BenchmarkMetrics {
            name: "test".to_string(),
            mean: 100.0,
            std_dev: 5.0,
            samples: 50,
        };
        let cloned = metrics.clone();
        assert_eq!(metrics.name, cloned.name);
        assert_eq!(metrics.mean, cloned.mean);
    }

    // ========================================================================
    // RegressionDetector Warning Threshold Test
    // ========================================================================

    #[test]
    fn test_regression_detector_warning() {
        let detector = RegressionDetector::default();
        let baseline = BenchmarkMetrics {
            name: "latency".to_string(),
            mean: 100.0,
            std_dev: 5.0,
            samples: 50,
        };
        let current = BenchmarkMetrics {
            name: "latency".to_string(),
            mean: 103.5,
            std_dev: 5.0,
            samples: 50,
        };
        let report = detector.compare(&baseline, &current);
        assert!(report.passed);
        assert!(report.regressions.is_empty());
        assert_eq!(report.warnings.len(), 1);
    }
}
