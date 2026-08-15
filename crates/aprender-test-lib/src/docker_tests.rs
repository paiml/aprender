    use super::*;

    // =========================================================================
    // Browser Tests
    // =========================================================================

    #[test]
    fn test_browser_default_cdp_ports() {
        assert_eq!(Browser::Chrome.default_cdp_port(), 9222);
        assert_eq!(Browser::Firefox.default_cdp_port(), 9223);
        assert_eq!(Browser::WebKit.default_cdp_port(), 9224);
    }

    #[test]
    fn test_browser_image_names() {
        assert_eq!(Browser::Chrome.image_name(), "probar-chrome:latest");
        assert_eq!(Browser::Firefox.image_name(), "probar-firefox:latest");
        assert_eq!(Browser::WebKit.image_name(), "probar-webkit:latest");
    }

    #[test]
    fn test_browser_container_prefix() {
        assert_eq!(Browser::Chrome.container_prefix(), "probar-chrome");
        assert_eq!(Browser::Firefox.container_prefix(), "probar-firefox");
        assert_eq!(Browser::WebKit.container_prefix(), "probar-webkit");
    }

    #[test]
    fn test_browser_all() {
        let all = Browser::all();
        assert_eq!(all.len(), 3);
        assert!(all.contains(&Browser::Chrome));
        assert!(all.contains(&Browser::Firefox));
        assert!(all.contains(&Browser::WebKit));
    }

    #[test]
    fn test_browser_from_str() {
        assert_eq!(Browser::from_str("chrome"), Some(Browser::Chrome));
        assert_eq!(Browser::from_str("CHROME"), Some(Browser::Chrome));
        assert_eq!(Browser::from_str("chromium"), Some(Browser::Chrome));
        assert_eq!(Browser::from_str("firefox"), Some(Browser::Firefox));
        assert_eq!(Browser::from_str("ff"), Some(Browser::Firefox));
        assert_eq!(Browser::from_str("webkit"), Some(Browser::WebKit));
        assert_eq!(Browser::from_str("safari"), Some(Browser::WebKit));
        assert_eq!(Browser::from_str("invalid"), None);
    }

    #[test]
    fn test_browser_display() {
        assert_eq!(format!("{}", Browser::Chrome), "chrome");
        assert_eq!(format!("{}", Browser::Firefox), "firefox");
        assert_eq!(format!("{}", Browser::WebKit), "webkit");
    }

    // =========================================================================
    // Container State Tests
    // =========================================================================

    #[test]
    fn test_container_state_default() {
        let state = ContainerState::default();
        assert_eq!(state, ContainerState::NotCreated);
    }

    #[test]
    fn test_container_state_display() {
        assert_eq!(format!("{}", ContainerState::NotCreated), "not_created");
        assert_eq!(format!("{}", ContainerState::Creating), "creating");
        assert_eq!(format!("{}", ContainerState::Starting), "starting");
        assert_eq!(format!("{}", ContainerState::Running), "running");
        assert_eq!(
            format!("{}", ContainerState::HealthChecking),
            "health_checking"
        );
        assert_eq!(format!("{}", ContainerState::Stopping), "stopping");
        assert_eq!(format!("{}", ContainerState::Stopped), "stopped");
        assert_eq!(format!("{}", ContainerState::Error), "error");
    }

    // =========================================================================
    // COOP/COEP Config Tests
    // =========================================================================

    #[test]
    fn test_coop_coep_config_default() {
        let config = CoopCoepConfig::default();
        assert_eq!(config.coop, "same-origin");
        assert_eq!(config.coep, "require-corp");
        assert_eq!(config.corp, "cross-origin");
        assert!(config.enabled);
    }

    #[test]
    fn test_coop_coep_config_new() {
        let config = CoopCoepConfig::new();
        assert!(config.enabled);
        assert_eq!(config.coop, "same-origin");
    }

    #[test]
    fn test_coop_coep_config_disabled() {
        let config = CoopCoepConfig::disabled();
        assert!(!config.enabled);
    }

    #[test]
    fn test_coop_coep_shared_array_buffer_available() {
        let config = CoopCoepConfig::default();
        assert!(config.shared_array_buffer_available());

        let mut disabled = CoopCoepConfig::default();
        disabled.enabled = false;
        assert!(!disabled.shared_array_buffer_available());

        let mut wrong_coop = CoopCoepConfig::default();
        wrong_coop.coop = "unsafe-none".to_string();
        assert!(!wrong_coop.shared_array_buffer_available());

        let mut wrong_coep = CoopCoepConfig::default();
        wrong_coep.coep = "unsafe-none".to_string();
        assert!(!wrong_coep.shared_array_buffer_available());
    }

    // =========================================================================
    // Container Config Tests
    // =========================================================================

    #[test]
    fn test_container_config_default() {
        let config = ContainerConfig::default();
        assert_eq!(config.image, "probar-wasm-test:latest");
        assert_eq!(config.name, "probar-test");
        assert!(config.ports.is_empty());
        assert!(config.environment.is_empty());
        assert_eq!(config.memory_limit, Some(2 * 1024 * 1024 * 1024));
        assert_eq!(config.cpu_limit, Some(2.0));
    }

    #[test]
    fn test_container_config_for_browser() {
        let chrome_config = ContainerConfig::for_browser(Browser::Chrome);
        assert_eq!(chrome_config.image, "probar-chrome:latest");
        assert!(chrome_config.name.starts_with("probar-chrome-"));
        assert_eq!(chrome_config.ports, vec![(9222, 9222)]);
        assert_eq!(
            chrome_config.environment.get("PROBAR_BROWSER"),
            Some(&"chrome".to_string())
        );

        let firefox_config = ContainerConfig::for_browser(Browser::Firefox);
        assert_eq!(firefox_config.image, "probar-firefox:latest");
        assert_eq!(firefox_config.ports, vec![(9223, 9223)]);

        let webkit_config = ContainerConfig::for_browser(Browser::WebKit);
        assert_eq!(webkit_config.image, "probar-webkit:latest");
        assert_eq!(webkit_config.ports, vec![(9224, 9224)]);
    }

    // =========================================================================
    // Docker Config Tests
    // =========================================================================

    #[test]
    fn test_docker_config_default() {
        let config = DockerConfig::default();
        assert_eq!(config.browser, Browser::Chrome);
        assert!(config.coop_coep.enabled);
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.parallel, 4);
        assert!(config.cleanup);
        assert!(config.capture_logs);
    }

    // =========================================================================
    // DockerTestRunner Builder Tests
    // =========================================================================

    #[test]
    fn test_docker_test_runner_builder_new() {
        let builder = DockerTestRunnerBuilder::new();
        let runner = builder.build().expect("Should build successfully");
        assert_eq!(runner.state(), ContainerState::NotCreated);
    }

    #[test]
    fn test_docker_test_runner_builder_browser() {
        let runner = DockerTestRunner::builder()
            .browser(Browser::Firefox)
            .build()
            .expect("Should build successfully");
        assert_eq!(runner.config().browser, Browser::Firefox);
    }

    #[test]
    fn test_docker_test_runner_builder_coop_coep() {
        let runner = DockerTestRunner::builder()
            .with_coop_coep(false)
            .build()
            .expect("Should build successfully");
        assert!(!runner.config().coop_coep.enabled);
    }

    #[test]
    fn test_docker_test_runner_builder_timeout() {
        let runner = DockerTestRunner::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("Should build successfully");
        assert_eq!(runner.config().timeout, Duration::from_secs(120));
    }

    #[test]
    fn test_docker_test_runner_builder_parallel() {
        let runner = DockerTestRunner::builder()
            .parallel(8)
            .build()
            .expect("Should build successfully");
        assert_eq!(runner.config().parallel, 8);
    }

    #[test]
    fn test_docker_test_runner_builder_pull_images() {
        let runner = DockerTestRunner::builder()
            .pull_images(false)
            .build()
            .expect("Should build successfully");
        assert!(!runner.config().pull_images);
    }

    #[test]
    fn test_docker_test_runner_builder_cleanup() {
        let runner = DockerTestRunner::builder()
            .cleanup(false)
            .build()
            .expect("Should build successfully");
        assert!(!runner.config().cleanup);
    }

    #[test]
    fn test_docker_test_runner_builder_capture_logs() {
        let runner = DockerTestRunner::builder()
            .capture_logs(false)
            .build()
            .expect("Should build successfully");
        assert!(!runner.config().capture_logs);
    }

    #[test]
    fn test_docker_test_runner_builder_docker_socket() {
        let runner = DockerTestRunner::builder()
            .docker_socket("/custom/docker.sock".to_string())
            .build()
            .expect("Should build successfully");
        assert_eq!(runner.config().docker_socket, "/custom/docker.sock");
    }

    #[test]
    fn test_docker_test_runner_builder_volume() {
        let runner = DockerTestRunner::builder()
            .volume(PathBuf::from("/host/path"), "/container/path".to_string())
            .build()
            .expect("Should build successfully");
        assert_eq!(runner.config().container.volumes.len(), 1);
    }

    #[test]
    fn test_docker_test_runner_builder_env() {
        let runner = DockerTestRunner::builder()
            .env("MY_VAR".to_string(), "my_value".to_string())
            .build()
            .expect("Should build successfully");
        assert_eq!(
            runner
                .config()
                .container
                .environment
                .get("MY_VAR")
                .map(String::as_str),
            Some("my_value")
        );
    }

    // =========================================================================
    // DockerTestRunner Tests
    // =========================================================================

    #[test]
    fn test_docker_test_runner_default() {
        let runner = DockerTestRunner::default();
        assert_eq!(runner.state(), ContainerState::NotCreated);
        assert!(runner.container_id().is_none());
        assert!(runner.logs().is_empty());
    }

    #[test]
    fn test_docker_test_runner_cdp_url() {
        let chrome_runner = DockerTestRunner::builder()
            .browser(Browser::Chrome)
            .build()
            .expect("Should build successfully");
        assert_eq!(chrome_runner.cdp_url(), "http://localhost:9222");

        let firefox_runner = DockerTestRunner::builder()
            .browser(Browser::Firefox)
            .build()
            .expect("Should build successfully");
        assert_eq!(firefox_runner.cdp_url(), "http://localhost:9223");
    }

    #[test]
    fn test_docker_test_runner_check_docker_available() {
        let runner = DockerTestRunner::default();
        assert!(runner.check_docker_available().is_ok());

        let empty_socket_runner = DockerTestRunner::builder()
            .docker_socket(String::new())
            .build()
            .expect("Should build");
        assert!(empty_socket_runner.check_docker_available().is_err());
    }

    #[test]
    fn test_docker_test_runner_validate_config() {
        let runner = DockerTestRunner::default();
        assert!(runner.validate_config().is_ok());
    }

    #[test]
    fn test_docker_test_runner_validate_config_empty_image() {
        let mut runner = DockerTestRunner::default();
        runner.config.container.image = String::new();
        assert!(runner.validate_config().is_err());
    }

    #[test]
    fn test_docker_test_runner_validate_config_empty_name() {
        let mut runner = DockerTestRunner::default();
        runner.config.container.name = String::new();
        assert!(runner.validate_config().is_err());
    }

    #[test]
    fn test_docker_test_runner_validate_config_zero_timeout() {
        let mut runner = DockerTestRunner::default();
        runner.config.timeout = Duration::ZERO;
        assert!(runner.validate_config().is_err());
    }

    #[test]
    fn test_docker_test_runner_simulate_start() {
        let mut runner = DockerTestRunner::default();
        runner.simulate_start().expect("Should start");
        assert_eq!(runner.state(), ContainerState::Running);
        assert!(runner.container_id().is_some());
        assert!(!runner.logs().is_empty());
    }

    #[test]
    fn test_docker_test_runner_simulate_stop() {
        let mut runner = DockerTestRunner::default();
        runner.simulate_start().expect("Should start");
        runner.simulate_stop().expect("Should stop");
        assert_eq!(runner.state(), ContainerState::Stopped);
        assert!(runner.container_id().is_none());
    }

    #[test]
    fn test_docker_test_runner_simulate_stop_not_running() {
        let mut runner = DockerTestRunner::default();
        assert!(runner.simulate_stop().is_err());
    }

    #[test]
    fn test_docker_test_runner_simulate_run_tests() {
        let mut runner = DockerTestRunner::default();
        runner.simulate_start().expect("Should start");
        let results = runner
            .simulate_run_tests(&["test1.rs", "test2.rs"])
            .expect("Should run tests");
        assert_eq!(results.passed, 2);
        assert_eq!(results.failed, 0);
        assert!(results.all_passed());
    }

    #[test]
    fn test_docker_test_runner_simulate_run_tests_not_running() {
        let mut runner = DockerTestRunner::default();
        assert!(runner.simulate_run_tests(&["test.rs"]).is_err());
    }

    // =========================================================================
    // TestResult Tests
    // =========================================================================

    #[test]
    fn test_test_result_passed() {
        let result = TestResult::passed("my_test".to_string(), Duration::from_millis(50));
        assert!(result.passed);
        assert!(result.error.is_none());
        assert_eq!(result.name, "my_test");
    }

    #[test]
    fn test_test_result_failed() {
        let result = TestResult::failed(
            "my_test".to_string(),
            Duration::from_millis(50),
            "assertion failed".to_string(),
        );
        assert!(!result.passed);
        assert_eq!(result.error, Some("assertion failed".to_string()));
    }

    // =========================================================================
    // TestResults Tests
    // =========================================================================

    #[test]
    fn test_test_results_new() {
        let results = TestResults::new(Browser::Chrome);
        assert_eq!(results.browser, Browser::Chrome);
        assert!(results.results.is_empty());
        assert_eq!(results.passed, 0);
        assert_eq!(results.failed, 0);
    }

    #[test]
    fn test_test_results_add_result() {
        let mut results = TestResults::new(Browser::Firefox);
        results.add_result(TestResult::passed(
            "test1".to_string(),
            Duration::from_secs(1),
        ));
        results.add_result(TestResult::failed(
            "test2".to_string(),
            Duration::from_secs(2),
            "error".to_string(),
        ));
        assert_eq!(results.passed, 1);
        assert_eq!(results.failed, 1);
        assert_eq!(results.total(), 2);
        assert_eq!(results.total_duration, Duration::from_secs(3));
    }

    #[test]
    fn test_test_results_all_passed() {
        let mut results = TestResults::new(Browser::Chrome);
        assert!(!results.all_passed()); // Empty results

        results.add_result(TestResult::passed(
            "test1".to_string(),
            Duration::from_secs(1),
        ));
        assert!(results.all_passed());

        results.add_result(TestResult::failed(
            "test2".to_string(),
            Duration::from_secs(1),
            "error".to_string(),
        ));
        assert!(!results.all_passed());
    }

    #[test]
    fn test_test_results_pass_rate() {
        let mut results = TestResults::new(Browser::WebKit);
        assert_eq!(results.pass_rate(), 0.0);

        results.add_result(TestResult::passed(
            "test1".to_string(),
            Duration::from_secs(1),
        ));
        assert_eq!(results.pass_rate(), 100.0);

        results.add_result(TestResult::failed(
            "test2".to_string(),
            Duration::from_secs(1),
            "error".to_string(),
        ));
        assert_eq!(results.pass_rate(), 50.0);
    }

    #[test]
    fn test_test_results_display() {
        let mut results = TestResults::new(Browser::Chrome);
        results.add_result(TestResult::passed(
            "test1".to_string(),
            Duration::from_secs(1),
        ));
        results.add_result(TestResult::passed(
            "test2".to_string(),
            Duration::from_secs(1),
        ));
        let display = format!("{results}");
        assert!(display.contains("chrome"));
        assert!(display.contains("2 passed"));
        assert!(display.contains("0 failed"));
        assert!(display.contains("100.0%"));
    }

    // =========================================================================
    // ParallelRunner Tests
    // =========================================================================

    #[test]
    fn test_parallel_runner_builder_new() {
        let builder = ParallelRunnerBuilder::new();
        let result = builder.build();
        assert!(result.is_err()); // No browsers configured
    }

    #[test]
    fn test_parallel_runner_builder_no_browsers() {
        let result = ParallelRunner::builder().tests(&["test.rs"]).build();
        assert!(result.is_err());
        match result {
            Err(DockerError::ConfigError(msg)) => {
                assert!(msg.contains("No browsers"));
            }
            _ => panic!("Expected ConfigError"),
        }
    }

    #[test]
    fn test_parallel_runner_builder_no_tests() {
        let result = ParallelRunner::builder()
            .browsers(&[Browser::Chrome])
            .build();
        assert!(result.is_err());
        match result {
            Err(DockerError::ConfigError(msg)) => {
                assert!(msg.contains("No tests"));
            }
            _ => panic!("Expected ConfigError"),
        }
    }

    #[test]
    fn test_parallel_runner_builder_success() {
        let runner = ParallelRunner::builder()
            .browsers(&[Browser::Chrome, Browser::Firefox])
            .tests(&["test1.rs", "test2.rs"])
            .timeout(Duration::from_secs(120))
            .build()
            .expect("Should build successfully");

        assert_eq!(runner.browsers().len(), 2);
        assert_eq!(runner.tests().len(), 2);
    }

    #[test]
    fn test_parallel_runner_simulate_run() {
        let mut runner = ParallelRunner::builder()
            .browsers(&[Browser::Chrome, Browser::Firefox])
            .tests(&["test1.rs", "test2.rs"])
            .build()
            .expect("Should build");

        runner.simulate_run().expect("Should run");

        assert!(runner.all_passed());
        let results = runner.results_by_browser();
        assert_eq!(results.len(), 2);
        assert!(results.contains_key(&Browser::Chrome));
        assert!(results.contains_key(&Browser::Firefox));
    }

    #[test]
    fn test_parallel_runner_aggregate_stats() {
        let mut runner = ParallelRunner::builder()
            .browsers(&[Browser::Chrome, Browser::Firefox, Browser::WebKit])
            .tests(&["test1.rs", "test2.rs"])
            .build()
            .expect("Should build");

        runner.simulate_run().expect("Should run");

        let (passed, failed, duration) = runner.aggregate_stats();
        assert_eq!(passed, 6); // 2 tests × 3 browsers
        assert_eq!(failed, 0);
        assert!(duration > Duration::ZERO);
    }

    #[test]
    fn test_parallel_runner_default() {
        let runner = ParallelRunner::default();
        assert!(runner.browsers().is_empty());
        assert!(runner.tests().is_empty());
        assert!(!runner.all_passed());
    }

    // =========================================================================
    // Header Validation Tests
    // =========================================================================

    #[test]
    fn test_validate_coop_coep_headers_valid() {
        let mut headers = HashMap::new();
        headers.insert(
            "cross-origin-opener-policy".to_string(),
            "same-origin".to_string(),
        );
        headers.insert(
            "cross-origin-embedder-policy".to_string(),
            "require-corp".to_string(),
        );
        assert!(validate_coop_coep_headers(&headers).is_ok());
    }

    #[test]
    fn test_validate_coop_coep_headers_valid_capitalized() {
        let mut headers = HashMap::new();
        headers.insert(
            "Cross-Origin-Opener-Policy".to_string(),
            "same-origin".to_string(),
        );
        headers.insert(
            "Cross-Origin-Embedder-Policy".to_string(),
            "require-corp".to_string(),
        );
        assert!(validate_coop_coep_headers(&headers).is_ok());
    }

    #[test]
    fn test_validate_coop_coep_headers_missing_coop() {
        let mut headers = HashMap::new();
        headers.insert(
            "cross-origin-embedder-policy".to_string(),
            "require-corp".to_string(),
        );
        let result = validate_coop_coep_headers(&headers);
        assert!(result.is_err());
        match result {
            Err(DockerError::ConfigError(msg)) => {
                assert!(msg.contains("Opener-Policy"));
            }
            _ => panic!("Expected ConfigError"),
        }
    }

    #[test]
    fn test_validate_coop_coep_headers_missing_coep() {
        let mut headers = HashMap::new();
        headers.insert(
            "cross-origin-opener-policy".to_string(),
            "same-origin".to_string(),
        );
        let result = validate_coop_coep_headers(&headers);
        assert!(result.is_err());
        match result {
            Err(DockerError::ConfigError(msg)) => {
                assert!(msg.contains("Embedder-Policy"));
            }
            _ => panic!("Expected ConfigError"),
        }
    }

    #[test]
    fn test_validate_coop_coep_headers_wrong_values() {
        let mut headers = HashMap::new();
        headers.insert(
            "cross-origin-opener-policy".to_string(),
            "unsafe-none".to_string(),
        );
        headers.insert(
            "cross-origin-embedder-policy".to_string(),
            "require-corp".to_string(),
        );
        let result = validate_coop_coep_headers(&headers);
        assert!(result.is_err());
    }

    #[test]
    fn test_check_shared_array_buffer_support() {
        let config = CoopCoepConfig::default();
        assert!(check_shared_array_buffer_support(&config));

        let disabled = CoopCoepConfig::disabled();
        assert!(!check_shared_array_buffer_support(&disabled));
    }

    // =========================================================================
    // Error Tests
    // =========================================================================

    #[test]
    fn test_docker_error_display() {
        let err = DockerError::DaemonUnavailable("not running".to_string());
        assert!(format!("{err}").contains("Docker daemon not available"));

        let err = DockerError::ContainerStartFailed("exit 1".to_string());
        assert!(format!("{err}").contains("Container failed to start"));

        let err = DockerError::ContainerNotFound("abc123".to_string());
        assert!(format!("{err}").contains("Container not found"));

        let err = DockerError::ImageNotFound("probar:latest".to_string());
        assert!(format!("{err}").contains("Image not found"));

        let err = DockerError::CdpConnectionFailed("timeout".to_string());
        assert!(format!("{err}").contains("CDP connection failed"));

        let err = DockerError::TestExecutionFailed("assertion".to_string());
        assert!(format!("{err}").contains("Test execution failed"));

        let err = DockerError::Timeout("30s".to_string());
        assert!(format!("{err}").contains("Timeout"));

        let err = DockerError::HealthCheckFailed("unhealthy".to_string());
        assert!(format!("{err}").contains("Health check failed"));

        let err = DockerError::ConfigError("invalid".to_string());
        assert!(format!("{err}").contains("Configuration error"));

        let err = DockerError::IoError("permission denied".to_string());
        assert!(format!("{err}").contains("IO error"));

        let err = DockerError::NetworkError("connection refused".to_string());
        assert!(format!("{err}").contains("Network error"));
    }

    // =========================================================================
    // Integration-style Tests
    // =========================================================================

    #[test]
    fn test_full_lifecycle_chrome() {
        let mut runner = DockerTestRunner::builder()
            .browser(Browser::Chrome)
            .with_coop_coep(true)
            .timeout(Duration::from_secs(30))
            .cleanup(true)
            .build()
            .expect("Should build");

        // Verify initial state
        assert_eq!(runner.state(), ContainerState::NotCreated);

        // Start container
        runner.simulate_start().expect("Should start");
        assert_eq!(runner.state(), ContainerState::Running);

        // Run tests
        let results = runner
            .simulate_run_tests(&["worker_tests.rs", "shared_memory_tests.rs"])
            .expect("Should run tests");
        assert!(results.all_passed());
        assert_eq!(results.passed, 2);

        // Stop container
        runner.simulate_stop().expect("Should stop");
        assert_eq!(runner.state(), ContainerState::Stopped);
    }

    #[test]
    fn test_full_lifecycle_firefox() {
        let mut runner = DockerTestRunner::builder()
            .browser(Browser::Firefox)
            .build()
            .expect("Should build");

        runner.simulate_start().expect("Should start");
        let results = runner
            .simulate_run_tests(&["e2e_tests.rs"])
            .expect("Should run");
        assert!(results.all_passed());
        runner.simulate_stop().expect("Should stop");
    }

    #[test]
    fn test_full_lifecycle_webkit() {
        let mut runner = DockerTestRunner::builder()
            .browser(Browser::WebKit)
            .build()
            .expect("Should build");

        runner.simulate_start().expect("Should start");
        let results = runner
            .simulate_run_tests(&["visual_regression.rs"])
            .expect("Should run");
        assert!(results.all_passed());
        runner.simulate_stop().expect("Should stop");
    }

    #[test]
    fn test_parallel_cross_browser() {
        let mut runner = ParallelRunner::builder()
            .browsers(&Browser::all())
            .tests(&[
                "worker_tests.rs",
                "shared_memory_tests.rs",
                "ring_buffer_tests.rs",
            ])
            .build()
            .expect("Should build");

        runner.simulate_run().expect("Should run");

        assert!(runner.all_passed());

        let (passed, failed, _) = runner.aggregate_stats();
        assert_eq!(passed, 9); // 3 tests × 3 browsers
        assert_eq!(failed, 0);

        // Check each browser
        let results = runner.results_by_browser();
        for browser in Browser::all() {
            let browser_results = results.get(&browser).expect("Should have results");
            assert!(browser_results.all_passed());
            assert_eq!(browser_results.passed, 3);
        }
    }

    // =========================================================================
    // Edge Cases and Boundary Tests
    // =========================================================================

    #[test]
    fn test_empty_test_list() {
        let mut runner = DockerTestRunner::default();
        runner.simulate_start().expect("Should start");
        let results = runner.simulate_run_tests(&[]).expect("Should handle empty");
        assert_eq!(results.total(), 0);
        assert!(!results.all_passed()); // No tests = not passing
    }

    #[test]
    fn test_single_test() {
        let mut runner = DockerTestRunner::default();
        runner.simulate_start().expect("Should start");
        let results = runner
            .simulate_run_tests(&["single_test.rs"])
            .expect("Should run");
        assert_eq!(results.total(), 1);
        assert!(results.all_passed());
    }

    #[test]
    fn test_many_tests() {
        let mut runner = DockerTestRunner::default();
        runner.simulate_start().expect("Should start");

        let tests: Vec<String> = (0..100).map(|i| format!("test_{i}.rs")).collect();
        let test_refs: Vec<&str> = tests.iter().map(String::as_str).collect();

        let results = runner.simulate_run_tests(&test_refs).expect("Should run");
        assert_eq!(results.total(), 100);
        assert!(results.all_passed());
    }

    #[test]
    fn test_pass_rate_precision() {
        let mut results = TestResults::new(Browser::Chrome);

        // Add 1 passed, 2 failed = 33.33...%
        results.add_result(TestResult::passed("t1".to_string(), Duration::from_secs(1)));
        results.add_result(TestResult::failed(
            "t2".to_string(),
            Duration::from_secs(1),
            "err".to_string(),
        ));
        results.add_result(TestResult::failed(
            "t3".to_string(),
            Duration::from_secs(1),
            "err".to_string(),
        ));

        let rate = results.pass_rate();
        assert!((rate - 33.333_333_333_333_336).abs() < 0.001);
    }

    // =========================================================================
    // Serialization Tests
    // =========================================================================

    #[test]
    fn test_browser_serialization() {
        let browser = Browser::Chrome;
        let json = serde_json::to_string(&browser).expect("Should serialize");
        assert_eq!(json, "\"chrome\"");

        let deserialized: Browser = serde_json::from_str(&json).expect("Should deserialize");
        assert_eq!(deserialized, Browser::Chrome);
    }

    #[test]
    fn test_container_state_serialization() {
        let state = ContainerState::Running;
        let json = serde_json::to_string(&state).expect("Should serialize");
        let deserialized: ContainerState = serde_json::from_str(&json).expect("Should deserialize");
        assert_eq!(deserialized, ContainerState::Running);
    }

    #[test]
    fn test_coop_coep_config_serialization() {
        let config = CoopCoepConfig::default();
        let json = serde_json::to_string(&config).expect("Should serialize");
        assert!(json.contains("same-origin"));
        assert!(json.contains("require-corp"));

        let deserialized: CoopCoepConfig = serde_json::from_str(&json).expect("Should deserialize");
        assert_eq!(deserialized.coop, "same-origin");
    }

    #[test]
    fn test_test_result_serialization() {
        let result = TestResult::passed("my_test".to_string(), Duration::from_millis(123));
        let json = serde_json::to_string(&result).expect("Should serialize");
        assert!(json.contains("my_test"));
        assert!(json.contains("true"));

        let deserialized: TestResult = serde_json::from_str(&json).expect("Should deserialize");
        assert!(deserialized.passed);
    }

    #[test]
    fn test_test_results_serialization() {
        let mut results = TestResults::new(Browser::Firefox);
        results.add_result(TestResult::passed("t1".to_string(), Duration::from_secs(1)));
        results.add_result(TestResult::failed(
            "t2".to_string(),
            Duration::from_secs(2),
            "error".to_string(),
        ));

        let json = serde_json::to_string(&results).expect("Should serialize");
        assert!(json.contains("firefox"));
        assert!(json.contains("t1"));
        assert!(json.contains("t2"));

        let deserialized: TestResults = serde_json::from_str(&json).expect("Should deserialize");
        assert_eq!(deserialized.passed, 1);
        assert_eq!(deserialized.failed, 1);
    }

    // =========================================================================
    // Additional Edge Case Tests for 100% Coverage
    // =========================================================================

    #[test]
    fn test_docker_config_serialization() {
        let config = DockerConfig::default();
        let json = serde_json::to_string(&config).expect("Should serialize");
        assert!(json.contains("chrome"));
        assert!(json.contains("timeout"));
    }

    #[test]
    fn test_container_config_serialization() {
        let config = ContainerConfig::default();
        let json = serde_json::to_string(&config).expect("Should serialize");
        assert!(json.contains("probar-wasm-test"));
    }

    #[test]
    fn test_container_config_for_all_browsers() {
        for browser in Browser::all() {
            let config = ContainerConfig::for_browser(browser);
            assert!(!config.image.is_empty());
            assert!(!config.name.is_empty());
            assert!(!config.ports.is_empty());
            assert!(config.health_check.is_some());
        }
    }

    #[test]
    fn test_browser_serialization_all_variants() {
        for browser in Browser::all() {
            let json = serde_json::to_string(&browser).expect("Should serialize");
            let deserialized: Browser = serde_json::from_str(&json).expect("Should deserialize");
            assert_eq!(deserialized, browser);
        }
    }

    #[test]
    fn test_container_state_all_variants_serialization() {
        let states = [
            ContainerState::NotCreated,
            ContainerState::Creating,
            ContainerState::Starting,
            ContainerState::Running,
            ContainerState::HealthChecking,
            ContainerState::Stopping,
            ContainerState::Stopped,
            ContainerState::Error,
        ];
        for state in states {
            let json = serde_json::to_string(&state).expect("Should serialize");
            let deserialized: ContainerState =
                serde_json::from_str(&json).expect("Should deserialize");
            assert_eq!(deserialized, state);
        }
    }

    #[test]
    fn test_parallel_runner_all_passed_no_results() {
        let runner = ParallelRunner::default();
        assert!(!runner.all_passed()); // Empty results = not passed
    }

    #[test]
    fn test_test_results_with_only_failures() {
        let mut results = TestResults::new(Browser::Chrome);
        results.add_result(TestResult::failed(
            "fail1".to_string(),
            Duration::from_secs(1),
            "error".to_string(),
        ));
        results.add_result(TestResult::failed(
            "fail2".to_string(),
            Duration::from_secs(1),
            "error".to_string(),
        ));
        assert!(!results.all_passed());
        assert_eq!(results.pass_rate(), 0.0);
    }

    #[test]
    fn test_coop_coep_custom_values() {
        let mut config = CoopCoepConfig::default();
        config.coop = "same-origin-allow-popups".to_string();
        config.coep = "credentialless".to_string();
        assert!(!config.shared_array_buffer_available());
    }

    #[test]
    fn test_docker_test_runner_config_accessors() {
        let runner = DockerTestRunner::builder()
            .browser(Browser::WebKit)
            .parallel(8)
            .timeout(Duration::from_secs(300))
            .build()
            .expect("Should build");

        assert_eq!(runner.config().browser, Browser::WebKit);
        assert_eq!(runner.config().parallel, 8);
        assert_eq!(runner.config().timeout, Duration::from_secs(300));
        assert_eq!(runner.cdp_url(), "http://localhost:9224");
    }

    #[test]
    fn test_container_config_environment_variables() {
        let config = ContainerConfig::for_browser(Browser::Chrome);
        assert!(config.environment.contains_key("PROBAR_BROWSER"));
        assert!(config.environment.contains_key("PROBAR_CDP_PORT"));
        assert!(config.environment.contains_key("PROBAR_COOP_COEP"));
    }

    #[test]
    fn test_container_config_default_resources() {
        let config = ContainerConfig::default();
        assert_eq!(config.memory_limit, Some(2 * 1024 * 1024 * 1024));
        assert_eq!(config.cpu_limit, Some(2.0));
        assert_eq!(config.health_check_interval, Duration::from_secs(5));
        assert_eq!(config.health_check_timeout, Duration::from_secs(5));
        assert_eq!(config.health_check_retries, 3);
    }

    #[test]
    fn test_docker_error_variants_debug() {
        let errors = vec![
            DockerError::DaemonUnavailable("test".to_string()),
            DockerError::ContainerStartFailed("test".to_string()),
            DockerError::ContainerNotFound("test".to_string()),
            DockerError::ImageNotFound("test".to_string()),
            DockerError::CdpConnectionFailed("test".to_string()),
            DockerError::TestExecutionFailed("test".to_string()),
            DockerError::Timeout("test".to_string()),
            DockerError::HealthCheckFailed("test".to_string()),
            DockerError::ConfigError("test".to_string()),
            DockerError::IoError("test".to_string()),
            DockerError::NetworkError("test".to_string()),
        ];
        for err in errors {
            let debug = format!("{:?}", err);
            assert!(!debug.is_empty());
        }
    }

    #[test]
    fn test_parallel_runner_tests_accessor() {
        let runner = ParallelRunner::builder()
            .browsers(&[Browser::Chrome])
            .tests(&["test1.rs", "test2.rs", "test3.rs"])
            .build()
            .expect("Should build");

        assert_eq!(runner.tests().len(), 3);
        assert!(runner.tests().contains(&"test1.rs".to_string()));
    }

    #[test]
    fn test_docker_test_runner_logs_accumulate() {
        let mut runner = DockerTestRunner::default();
        runner.simulate_start().expect("Should start");
        let initial_logs = runner.logs().len();

        runner.simulate_run_tests(&["t1.rs"]).expect("Should run");
        assert!(runner.logs().len() > initial_logs);

        runner
            .simulate_run_tests(&["t2.rs", "t3.rs"])
            .expect("Should run");
        assert!(runner.logs().len() > initial_logs + 1);
    }

    #[test]
    fn test_test_result_duration() {
        let result = TestResult::passed("test".to_string(), Duration::from_millis(42));
        assert_eq!(result.duration, Duration::from_millis(42));

        let failed = TestResult::failed(
            "test".to_string(),
            Duration::from_millis(100),
            "err".to_string(),
        );
        assert_eq!(failed.duration, Duration::from_millis(100));
    }

    #[test]
    fn test_test_results_total_duration() {
        let mut results = TestResults::new(Browser::Firefox);
        results.add_result(TestResult::passed(
            "t1".to_string(),
            Duration::from_millis(100),
        ));
        results.add_result(TestResult::passed(
            "t2".to_string(),
            Duration::from_millis(200),
        ));
        results.add_result(TestResult::passed(
            "t3".to_string(),
            Duration::from_millis(300),
        ));

        assert_eq!(results.total_duration, Duration::from_millis(600));
    }

    #[test]
    fn test_browser_from_str_case_insensitive() {
        assert_eq!(Browser::from_str("CHROME"), Some(Browser::Chrome));
        assert_eq!(Browser::from_str("Chrome"), Some(Browser::Chrome));
        assert_eq!(Browser::from_str("chrome"), Some(Browser::Chrome));
        assert_eq!(Browser::from_str("FIREFOX"), Some(Browser::Firefox));
        assert_eq!(Browser::from_str("Firefox"), Some(Browser::Firefox));
        assert_eq!(Browser::from_str("WEBKIT"), Some(Browser::WebKit));
        assert_eq!(Browser::from_str("WebKit"), Some(Browser::WebKit));
    }

    #[test]
    fn test_parallel_runner_timeout_configuration() {
        let runner = ParallelRunner::builder()
            .browsers(&[Browser::Chrome])
            .tests(&["test.rs"])
            .timeout(Duration::from_secs(180))
            .build()
            .expect("Should build");

        // Just verify it builds - timeout is stored in config
        assert!(!runner.browsers().is_empty());
    }

    #[test]
    fn test_docker_test_runner_chain_configuration() {
        let runner = DockerTestRunner::builder()
            .browser(Browser::Firefox)
            .with_coop_coep(true)
            .timeout(Duration::from_secs(90))
            .parallel(2)
            .pull_images(false)
            .cleanup(true)
            .capture_logs(true)
            .build()
            .expect("Should build");

        assert_eq!(runner.config().browser, Browser::Firefox);
        assert!(runner.config().coop_coep.enabled);
        assert_eq!(runner.config().timeout, Duration::from_secs(90));
        assert_eq!(runner.config().parallel, 2);
        assert!(!runner.config().pull_images);
        assert!(runner.config().cleanup);
        assert!(runner.config().capture_logs);
    }
