
    /// IMP-900: M3 target achievement
    #[test]
    fn test_imp900_m3_achievement() {
        let result = Imp900Result::from_baseline(13.1)
            .with_gemm_improvement(2.5)
            .with_memory_improvement(1.5);

        // 13.1 * 2.5 * 1.5 = 49.125 tok/s
        assert!((result.optimized_tps - 49.125).abs() < 0.1);
        assert!((result.gap_ratio - 4.89).abs() < 0.1); // 240/49.125

        assert!(result.achieves_m3()); // >48 tok/s, <5x gap
        assert_eq!(result.milestone, Some("M2".to_string())); // Actually at M2 threshold
    }

    /// IMP-900: M4 target achievement (full parity)
    #[test]
    fn test_imp900_m4_achievement() {
        let result = Imp900Result::from_baseline(13.1)
            .with_gemm_improvement(3.0)
            .with_fusion_improvement(2.0)
            .with_flash_attention_improvement(2.5)
            .with_memory_improvement(1.5);

        // 13.1 * 3.0 * 2.0 * 2.5 * 1.5 = 294.75 tok/s
        let expected_tps = 13.1 * 3.0 * 2.0 * 2.5 * 1.5;
        assert!((result.optimized_tps - expected_tps).abs() < 0.1);
        assert!((result.gap_ratio - 0.81).abs() < 0.1); // 240/294.75

        assert!(result.achieves_m4()); // >192 tok/s, <1.25x gap
        assert_eq!(result.milestone, Some("M4".to_string()));
    }

    /// IMP-900: Total improvement factor
    #[test]
    fn test_imp900_total_improvement() {
        let result = Imp900Result::from_baseline(13.1)
            .with_gemm_improvement(2.0)
            .with_fusion_improvement(1.5)
            .with_flash_attention_improvement(2.0)
            .with_memory_improvement(1.5);

        // Total: 2.0 * 1.5 * 2.0 * 1.5 = 9.0x
        assert!((result.total_improvement() - 9.0).abs() < 0.1);
    }

    // ========================================================================
    // Additional Load Testing Coverage Tests (Phase 4)
    // ========================================================================

    #[test]
    fn test_load_test_runner_config_accessor() {
        let config = LoadTestConfig {
            concurrency: 50,
            duration_secs: 120,
            target_rps: 100.0,
            timeout_ms: 3000,
            warmup_secs: 10,
            latency_threshold_ms: 300.0,
        };
        let runner = LoadTestRunner::new(config.clone());
        let retrieved = runner.config();

        assert_eq!(retrieved.concurrency, 50);
        assert_eq!(retrieved.duration_secs, 120);
        assert!((retrieved.target_rps - 100.0).abs() < 0.001);
        assert_eq!(retrieved.timeout_ms, 3000);
        assert_eq!(retrieved.warmup_secs, 10);
        assert!((retrieved.latency_threshold_ms - 300.0).abs() < 0.001);
    }

    #[test]
    fn test_load_test_result_throughput_zero_duration() {
        let result = LoadTestResult {
            total_requests: 1000,
            successful_requests: 1000,
            failed_requests: 0,
            rps_achieved: 100.0,
            latency_p50_ms: 20.0,
            latency_p95_ms: 50.0,
            latency_p99_ms: 80.0,
            latency_max_ms: 200.0,
            data_transferred_bytes: 10_000_000,
            duration_secs: 0.0, // Zero duration
            error_rate: 0.0,
            passed_latency_threshold: true,
        };
        assert_eq!(result.throughput_mbps(), 0.0);
    }

    #[test]
    fn test_load_test_result_throughput_negative_duration() {
        let result = LoadTestResult {
            total_requests: 1000,
            successful_requests: 1000,
            failed_requests: 0,
            rps_achieved: 100.0,
            latency_p50_ms: 20.0,
            latency_p95_ms: 50.0,
            latency_p99_ms: 80.0,
            latency_max_ms: 200.0,
            data_transferred_bytes: 10_000_000,
            duration_secs: -5.0, // Negative duration edge case
            error_rate: 0.0,
            passed_latency_threshold: true,
        };
        assert_eq!(result.throughput_mbps(), 0.0);
    }

    #[test]
    fn test_load_test_config_validation_edge_cases() {
        // Zero duration
        let zero_duration = LoadTestConfig {
            duration_secs: 0,
            ..LoadTestConfig::default()
        };
        assert!(!zero_duration.is_valid());

        // Zero timeout
        let zero_timeout = LoadTestConfig {
            timeout_ms: 0,
            ..LoadTestConfig::default()
        };
        assert!(!zero_timeout.is_valid());

        // Zero latency threshold
        let zero_latency = LoadTestConfig {
            latency_threshold_ms: 0.0,
            ..LoadTestConfig::default()
        };
        assert!(!zero_latency.is_valid());

        // Negative latency threshold
        let negative_latency = LoadTestConfig {
            latency_threshold_ms: -100.0,
            ..LoadTestConfig::default()
        };
        assert!(!negative_latency.is_valid());
    }

    #[test]
    fn test_load_test_runner_simulate_with_stress_config() {
        let config = LoadTestConfig::for_stress_test();
        let runner = LoadTestRunner::new(config);
        let result = runner.simulate_run();

        // Stress test should have more requests due to higher concurrency
        assert!(result.total_requests > 0);
        assert!(result.rps_achieved > 0.0);
        // Higher concurrency = higher latency in simulation
        assert!(result.latency_p50_ms > 0.0);
    }

    #[test]
    fn test_load_test_runner_simulate_with_latency_config() {
        let config = LoadTestConfig::for_latency_test();
        let runner = LoadTestRunner::new(config);
        let result = runner.simulate_run();

        // Latency test has low concurrency
        assert!(result.total_requests > 0);
        // Low concurrency = lower latencies
        assert!(result.latency_p50_ms > 0.0);
    }

    // ========================================================================
    // Additional Matrix Benchmark Coverage Tests (Phase 4)
    // ========================================================================

    #[test]
    fn test_matrix_benchmark_entry_default() {
        let entry = MatrixBenchmarkEntry::default();

        assert_eq!(entry.runtime, RuntimeType::Realizar);
        assert_eq!(entry.backend, ComputeBackendType::Cpu);
        assert!(entry.model.is_empty());
        assert!(!entry.available);
        assert_eq!(entry.p50_latency_ms, 0.0);
        assert_eq!(entry.p99_latency_ms, 0.0);
        assert_eq!(entry.throughput_tps, 0.0);
        assert_eq!(entry.cold_start_ms, 0.0);
        assert_eq!(entry.samples, 0);
        assert_eq!(entry.cv_at_stop, 0.0);
        assert!(entry.notes.is_empty());
    }

    #[test]
    fn test_benchmark_matrix_get_entry_not_found() {
        let hardware = HardwareSpec::default();
        let matrix = BenchmarkMatrix::new("test-model", hardware);

        // Empty matrix should return None for any query
        assert!(matrix
            .get_entry(RuntimeType::Realizar, ComputeBackendType::Cpu)
            .is_none());
        assert!(matrix
            .get_entry(RuntimeType::LlamaCpp, ComputeBackendType::Cuda)
            .is_none());
    }

    #[test]
    fn test_benchmark_matrix_fastest_for_backend_empty() {
        let hardware = HardwareSpec::default();
        let matrix = BenchmarkMatrix::new("test-model", hardware);

        // Empty matrix should return None
        assert!(matrix
            .fastest_for_backend(ComputeBackendType::Cpu)
            .is_none());
        assert!(matrix
            .fastest_for_backend(ComputeBackendType::Cuda)
            .is_none());
    }

    #[test]
    fn test_benchmark_matrix_highest_throughput_for_backend_empty() {
        let hardware = HardwareSpec::default();
        let matrix = BenchmarkMatrix::new("test-model", hardware);

        // Empty matrix should return None
        assert!(matrix
            .highest_throughput_for_backend(ComputeBackendType::Cpu)
            .is_none());
        assert!(matrix
            .highest_throughput_for_backend(ComputeBackendType::Wgpu)
            .is_none());
    }

    #[test]
    fn test_benchmark_matrix_fastest_excludes_unavailable() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("test-model", hardware);

        // Add an unavailable entry with "better" metrics (0 latency)
        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
        ));

        // Add an available entry with real metrics
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cpu,
            "test",
            &[100.0, 105.0],
            &[50.0],
            90.0,
        ));

        // Should return the available entry, not the unavailable one
        let fastest = matrix.fastest_for_backend(ComputeBackendType::Cpu);
        assert!(fastest.is_some());
        assert_eq!(fastest.unwrap().runtime, RuntimeType::LlamaCpp);
    }

    #[test]
    fn test_benchmark_matrix_highest_throughput_excludes_unavailable() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("test-model", hardware);

        // Add an unavailable entry
        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::Realizar,
            ComputeBackendType::Wgpu,
        ));

        // Add an available entry
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Wgpu,
            "test",
            &[100.0],
            &[75.0],
            90.0,
        ));

        // Should return the available entry
        let highest = matrix.highest_throughput_for_backend(ComputeBackendType::Wgpu);
        assert!(highest.is_some());
        assert_eq!(highest.unwrap().runtime, RuntimeType::LlamaCpp);
    }

    #[test]
    fn test_benchmark_matrix_summary_empty() {
        let hardware = HardwareSpec::default();
        let matrix = BenchmarkMatrix::new("test-model", hardware);

        let summary = matrix.summary();

        assert_eq!(summary.total_entries, 0);
        assert_eq!(summary.available_entries, 0);
        assert!(summary.overall_fastest.is_none());
        assert!(summary.overall_highest_throughput.is_none());
        assert_eq!(summary.backend_summaries.len(), 3); // CPU, Wgpu, Cuda
    }

    #[test]
    fn test_benchmark_matrix_summary_with_entries() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("test-model", hardware);

        // Add entries for different backends
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test",
            &[100.0, 110.0], // p50 ~= 105
            &[50.0, 55.0],   // avg ~= 52.5
            90.0,
        ));

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cpu,
            "test",
            &[80.0, 85.0], // p50 ~= 82.5 (faster)
            &[60.0, 65.0], // avg ~= 62.5 (higher throughput)
            100.0,
        ));

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cuda,
            "test",
            &[50.0, 55.0], // p50 ~= 52.5 (fastest overall)
            &[80.0, 85.0], // avg ~= 82.5 (highest throughput)
            80.0,
        ));

        let summary = matrix.summary();

        assert_eq!(summary.total_entries, 3);
        assert_eq!(summary.available_entries, 3);

        // Overall fastest should be CUDA (lowest latency)
        assert!(summary.overall_fastest.is_some());
        let (runtime, backend) = summary.overall_fastest.unwrap();
        assert_eq!(runtime, "realizar");
        assert_eq!(backend, "cuda");

        // Overall highest throughput should also be CUDA
        assert!(summary.overall_highest_throughput.is_some());
        let (runtime, backend) = summary.overall_highest_throughput.unwrap();
        assert_eq!(runtime, "realizar");
        assert_eq!(backend, "cuda");
    }

    #[test]
    fn test_benchmark_matrix_summary_backend_details() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("test-model", hardware);

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test",
            &[100.0],
            &[50.0],
            90.0,
        ));

        let summary = matrix.summary();

        // Find CPU backend summary
        let cpu_summary = summary
            .backend_summaries
            .iter()
            .find(|s| s.backend == ComputeBackendType::Cpu)
            .expect("Should have CPU summary");

        assert_eq!(cpu_summary.available_runtimes, 1);
        assert!(cpu_summary.fastest_runtime.is_some());
        assert!(cpu_summary.fastest_p50_ms > 0.0);
        assert!(cpu_summary.highest_throughput_runtime.is_some());
        assert!(cpu_summary.highest_throughput_tps > 0.0);
    }

    #[test]
    fn test_benchmark_matrix_summary_all_unavailable() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("test-model", hardware);

        // Add only unavailable entries
        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cuda,
        ));

        let summary = matrix.summary();

        assert_eq!(summary.total_entries, 2);
        assert_eq!(summary.available_entries, 0);
        assert!(summary.overall_fastest.is_none());
        assert!(summary.overall_highest_throughput.is_none());
    }

    #[test]
    fn test_matrix_benchmark_config_default_comprehensive() {
        let config = MatrixBenchmarkConfig::default();

        assert!(config.runtimes.contains(&RuntimeType::Realizar));
        assert!(config.runtimes.contains(&RuntimeType::LlamaCpp));
        assert!(config.runtimes.contains(&RuntimeType::Ollama));
        assert!(!config.runtimes.contains(&RuntimeType::Vllm)); // Not in default

        assert!(config.backends.contains(&ComputeBackendType::Cpu));
        assert!(config.backends.contains(&ComputeBackendType::Wgpu));
        assert!(!config.backends.contains(&ComputeBackendType::Cuda)); // Not in default

        assert!(config.model_path.is_empty());
        assert!(!config.prompt.is_empty());
        assert_eq!(config.max_tokens, 50);
        assert!((config.cv_threshold - 0.05).abs() < 0.001);
        assert_eq!(config.min_samples, 30);
        assert_eq!(config.max_samples, 200);
        assert_eq!(config.warmup_iterations, 5);
    }

    #[test]
    fn test_backend_summary_struct_fields() {
        let summary = BackendSummary {
            backend: ComputeBackendType::Cuda,
            available_runtimes: 3,
            fastest_runtime: Some("realizar".to_string()),
            fastest_p50_ms: 25.5,
            highest_throughput_runtime: Some("llama-cpp".to_string()),
            highest_throughput_tps: 150.0,
        };

        assert_eq!(summary.backend, ComputeBackendType::Cuda);
        assert_eq!(summary.available_runtimes, 3);
        assert_eq!(summary.fastest_runtime, Some("realizar".to_string()));
        assert!((summary.fastest_p50_ms - 25.5).abs() < 0.001);
        assert_eq!(
            summary.highest_throughput_runtime,
            Some("llama-cpp".to_string())
        );
        assert!((summary.highest_throughput_tps - 150.0).abs() < 0.001);
    }
