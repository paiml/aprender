
    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_distributed_bench_communication() {
        let config = DistributedBenchConfig::default();
        let mut suite = DistributedBenchSuite::new(config);
        suite.run_communication_benchmark();

        let results = suite.comm_results();
        // 4 data sizes × 2 operations (all_reduce, all_gather)
        assert_eq!(results.len(), 8);

        for result in results {
            assert!(result.latency_us > 0.0);
            assert!(result.bandwidth_gbps > 0.0);
            assert!(result.world_size > 0);
            assert!(!result.operation.is_empty());
            assert!(result.data_size_bytes > 0);
        }

        // All-gather should be faster than all-reduce for same size
        let reduce_1kb = results
            .iter()
            .find(|r| r.operation == "all_reduce" && r.data_size_bytes == 1024)
            .expect("test");
        let gather_1kb = results
            .iter()
            .find(|r| r.operation == "all_gather" && r.data_size_bytes == 1024)
            .expect("test");
        assert!(gather_1kb.latency_us < reduce_1kb.latency_us);
    }

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_distributed_bench_run_all() {
        let config = DistributedBenchConfig::for_small_model();
        let mut suite = DistributedBenchSuite::new(config);
        suite.run_all();

        assert!(!suite.scaling_results().is_empty());
        assert!(!suite.tp_results().is_empty());
        assert!(!suite.pp_results().is_empty());
        assert!(!suite.comm_results().is_empty());
    }

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_distributed_bench_summary() {
        let config = DistributedBenchConfig::default();
        let mut suite = DistributedBenchSuite::new(config);
        suite.run_all();

        let summary = suite.summary();
        assert_eq!(summary.max_scaling, 8);
        assert!(summary.max_efficiency > 0.0);
        assert!(summary.min_efficiency > 0.0);
        assert!(summary.max_efficiency >= summary.min_efficiency);
        assert!(summary.max_throughput_tps > 0.0);
        assert!(summary.avg_tp_comm_overhead_pct >= 0.0);
        assert!(summary.avg_pp_bubble_ratio >= 0.0);
    }

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_distributed_bench_all_meet_threshold() {
        // Use small model config (only 1-2 GPUs) where efficiency stays high
        let config = DistributedBenchConfig::for_small_model();
        let mut suite = DistributedBenchSuite::new(config);
        suite.run_scaling_benchmark();

        // With 80% threshold and only 1-2 GPUs, all should pass
        assert!(suite.all_meet_efficiency_threshold());
    }

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_distributed_bench_fail_threshold() {
        let config = DistributedBenchConfig {
            efficiency_threshold: 0.99, // Very high threshold
            ..DistributedBenchConfig::default()
        };
        let mut suite = DistributedBenchSuite::new(config);
        suite.run_scaling_benchmark();

        // With 99% threshold, multi-GPU configs should fail
        assert!(!suite.all_meet_efficiency_threshold());
    }

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_distributed_bench_empty_summary() {
        let config = DistributedBenchConfig::default();
        let suite = DistributedBenchSuite::new(config);

        // Summary on empty results should handle gracefully
        let summary = suite.summary();
        assert_eq!(summary.max_scaling, 1);
        assert!((summary.max_efficiency - 0.0).abs() < 0.001);
        assert!((summary.avg_tp_comm_overhead_pct - 0.0).abs() < 0.001);
        assert!((summary.avg_pp_bubble_ratio - 0.0).abs() < 0.001);
    }

    // ========================================================================
    // Load Testing Tests
    // ========================================================================

    #[test]
    fn test_load_test_config_default() {
        let config = LoadTestConfig::default();
        assert_eq!(config.concurrency, 10);
        assert_eq!(config.duration_secs, 60);
        assert!((config.target_rps - 0.0).abs() < 0.001);
        assert_eq!(config.timeout_ms, 5000);
        assert_eq!(config.warmup_secs, 5);
        assert!((config.latency_threshold_ms - 500.0).abs() < 0.001);
    }

    #[test]
    fn test_load_test_config_stress_test() {
        let config = LoadTestConfig::for_stress_test();
        assert_eq!(config.concurrency, 100);
        assert_eq!(config.duration_secs, 300);
        assert!((config.latency_threshold_ms - 1000.0).abs() < 0.001);
    }

    #[test]
    fn test_load_test_config_latency_test() {
        let config = LoadTestConfig::for_latency_test();
        assert_eq!(config.concurrency, 1);
        assert!((config.target_rps - 10.0).abs() < 0.001);
        assert!((config.latency_threshold_ms - 200.0).abs() < 0.001);
    }

    #[test]
    fn test_load_test_config_validation() {
        let valid = LoadTestConfig::default();
        assert!(valid.is_valid());

        let invalid = LoadTestConfig {
            concurrency: 0,
            ..LoadTestConfig::default()
        };
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_load_test_runner_simulate() {
        let config = LoadTestConfig::default();
        let runner = LoadTestRunner::new(config);
        let result = runner.simulate_run();

        assert!(result.total_requests > 0);
        assert!(result.successful_requests > 0);
        assert!(result.rps_achieved > 0.0);
        assert!(result.latency_p50_ms > 0.0);
        assert!(result.latency_p95_ms > result.latency_p50_ms);
        assert!(result.latency_p99_ms > result.latency_p95_ms);
        assert!(result.latency_max_ms > result.latency_p99_ms);
        assert!(result.data_transferred_bytes > 0);
        assert!(result.duration_secs > 0.0);
        assert!(result.error_rate >= 0.0 && result.error_rate < 1.0);
    }

    #[test]
    fn test_load_test_result_is_passing() {
        let passing = LoadTestResult {
            total_requests: 1000,
            successful_requests: 995,
            failed_requests: 5,
            rps_achieved: 100.0,
            latency_p50_ms: 20.0,
            latency_p95_ms: 50.0,
            latency_p99_ms: 80.0,
            latency_max_ms: 200.0,
            data_transferred_bytes: 1_000_000,
            duration_secs: 10.0,
            error_rate: 0.005,
            passed_latency_threshold: true,
        };
        assert!(passing.is_passing());

        let failing_error_rate = LoadTestResult {
            error_rate: 0.05, // 5% errors
            ..passing.clone()
        };
        assert!(!failing_error_rate.is_passing());

        let failing_latency = LoadTestResult {
            passed_latency_threshold: false,
            ..passing
        };
        assert!(!failing_latency.is_passing());
    }

    #[test]
    fn test_load_test_result_throughput() {
        let result = LoadTestResult {
            total_requests: 1000,
            successful_requests: 1000,
            failed_requests: 0,
            rps_achieved: 100.0,
            latency_p50_ms: 20.0,
            latency_p95_ms: 50.0,
            latency_p99_ms: 80.0,
            latency_max_ms: 200.0,
            data_transferred_bytes: 10_000_000, // 10 MB
            duration_secs: 10.0,
            error_rate: 0.0,
            passed_latency_threshold: true,
        };
        assert!((result.throughput_mbps() - 1.0).abs() < 0.001); // 10MB / 10s = 1 MB/s
    }

    // ========================================================================
    // LlamaCppBackend REAL CLI Output Parsing Tests (P0 - No Stubs)
    // ========================================================================

    #[test]
    #[cfg(feature = "bench-http")]
    fn test_parse_llama_cli_timing_prompt_eval() {
        let output = r"llama_perf_context_print: prompt eval time =      12.34 ms /    10 tokens (    1.23 ms per token,   810.37 tokens per second)";
        let timing = LlamaCppBackend::parse_timing_line(output, "prompt eval time");
        assert!(timing.is_some());
        let (total_ms, tokens) = timing.expect("test");
        assert!((total_ms - 12.34).abs() < 0.01);
        assert_eq!(tokens, 10);
    }

    #[test]
    #[cfg(feature = "bench-http")]
    fn test_parse_llama_cli_timing_eval() {
        let output = r"llama_perf_context_print:        eval time =      22.60 ms /     5 runs   (    4.52 ms per token,   221.28 tokens per second)";
        let timing = LlamaCppBackend::parse_timing_line(output, "eval time");
        assert!(timing.is_some());
        let (total_ms, runs) = timing.expect("test");
        assert!((total_ms - 22.60).abs() < 0.01);
        assert_eq!(runs, 5);
    }

    #[test]
    #[cfg(feature = "bench-http")]
    fn test_parse_llama_cli_timing_total() {
        let output = r"llama_perf_context_print:       total time =      23.27 ms /     6 tokens";
        let timing = LlamaCppBackend::parse_timing_line(output, "total time");
        assert!(timing.is_some());
        let (total_ms, tokens) = timing.expect("test");
        assert!((total_ms - 23.27).abs() < 0.01);
        assert_eq!(tokens, 6);
    }

    #[test]
    #[cfg(feature = "bench-http")]
    fn test_parse_llama_cli_full_output() {
        let output = r#"Hello world",
```

llama_perf_sampler_print:    sampling time =       0.14 ms /     6 runs   (    0.02 ms per token, 42553.19 tokens per second)
llama_perf_context_print:        load time =    1349.68 ms
llama_perf_context_print: prompt eval time =       5.00 ms /     1 tokens (    5.00 ms per token,   200.00 tokens per second)
llama_perf_context_print:        eval time =      22.60 ms /     5 runs   (    4.52 ms per token,   221.28 tokens per second)
llama_perf_context_print:       total time =      27.60 ms /     6 tokens"#;

        let result = LlamaCppBackend::parse_cli_output(output);
        assert!(result.is_ok());
        let response = result.expect("test");
        // TTFT = prompt eval time (time to first token)
        assert!((response.ttft_ms - 5.0).abs() < 0.1);
        // Total time from parse
        assert!((response.total_time_ms - 27.60).abs() < 0.1);
        // Tokens generated = eval runs (5 tokens generated after prompt)
        assert_eq!(response.tokens_generated, 5);
        // Text should contain the generated content
        assert!(response.text.contains("Hello world"));
    }

    #[test]
    #[cfg(feature = "bench-http")]
    fn test_parse_llama_cli_extract_generated_text() {
        let output =
            "The answer is 42.\n\nllama_perf_context_print: total time = 100.0 ms / 10 tokens";
        let text = LlamaCppBackend::extract_generated_text(output);
        assert_eq!(text, "The answer is 42.");
    }

    #[test]
    #[cfg(feature = "bench-http")]
    fn test_llama_cpp_backend_build_command() {
        let config = LlamaCppConfig {
            binary_path: "/path/to/llama-cli".to_string(),
            model_path: Some("/path/to/model.gguf".to_string()),
            n_gpu_layers: 99,
            ctx_size: 4096,
            threads: 8,
        };
        let backend = LlamaCppBackend::new(config);
        let request = InferenceRequest {
            prompt: "Hello".to_string(),
            max_tokens: 10,
            temperature: 0.7,
            stop: vec![],
        };
        let args = backend.build_cli_args(&request);

        assert!(args.contains(&"-m".to_string()));
        assert!(args.contains(&"/path/to/model.gguf".to_string()));
        assert!(args.contains(&"-p".to_string()));
        assert!(args.contains(&"Hello".to_string()));
        assert!(args.contains(&"-n".to_string()));
        assert!(args.contains(&"10".to_string()));
        assert!(args.contains(&"-ngl".to_string()));
        assert!(args.contains(&"99".to_string()));
        assert!(args.contains(&"-c".to_string()));
        assert!(args.contains(&"4096".to_string()));
        assert!(args.contains(&"-t".to_string()));
        assert!(args.contains(&"8".to_string()));
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_llama_cpp_backend_no_model_path_error() {
        let config = LlamaCppConfig {
            binary_path: "/path/to/llama-cli".to_string(),
            model_path: None, // No model
            n_gpu_layers: 0,
            ctx_size: 2048,
            threads: 4,
        };
        let backend = LlamaCppBackend::new(config);
        let request = InferenceRequest {
            prompt: "Hello".to_string(),
            max_tokens: 10,
            temperature: 0.7,
            stop: vec![],
        };
        let result = backend.inference(&request);
        assert!(result.is_err());
    }

    // ========================================================================
    // Benchmark Matrix Tests (EXTREME TDD)
    // ========================================================================

    #[test]
    fn test_compute_backend_type_display() {
        assert_eq!(format!("{}", ComputeBackendType::Cpu), "cpu");
        assert_eq!(format!("{}", ComputeBackendType::Wgpu), "wgpu");
        assert_eq!(format!("{}", ComputeBackendType::Cuda), "cuda");
    }

    #[test]
    fn test_compute_backend_type_from_str() {
        assert_eq!(
            ComputeBackendType::parse("cpu"),
            Some(ComputeBackendType::Cpu)
        );
        assert_eq!(
            ComputeBackendType::parse("CPU"),
            Some(ComputeBackendType::Cpu)
        );
        assert_eq!(
            ComputeBackendType::parse("wgpu"),
            Some(ComputeBackendType::Wgpu)
        );
        assert_eq!(
            ComputeBackendType::parse("gpu"),
            Some(ComputeBackendType::Wgpu)
        );
        assert_eq!(
            ComputeBackendType::parse("cuda"),
            Some(ComputeBackendType::Cuda)
        );
        assert_eq!(
            ComputeBackendType::parse("nvidia"),
            Some(ComputeBackendType::Cuda)
        );
        assert_eq!(ComputeBackendType::parse("unknown"), None);
    }

    #[test]
    fn test_compute_backend_type_all() {
        let all = ComputeBackendType::all();
        assert_eq!(all.len(), 3);
        assert!(all.contains(&ComputeBackendType::Cpu));
        assert!(all.contains(&ComputeBackendType::Wgpu));
        assert!(all.contains(&ComputeBackendType::Cuda));
    }

    #[test]
    fn test_matrix_benchmark_entry_unavailable() {
        let entry =
            MatrixBenchmarkEntry::unavailable(RuntimeType::Realizar, ComputeBackendType::Cuda);
        assert!(!entry.available);
        assert_eq!(entry.runtime, RuntimeType::Realizar);
        assert_eq!(entry.backend, ComputeBackendType::Cuda);
        assert!(entry.notes.contains("not available"));
    }

    #[test]
    fn test_matrix_benchmark_entry_from_samples() {
        let latencies = vec![100.0, 105.0, 110.0, 95.0, 102.0];
        let throughputs = vec![50.0, 48.0, 52.0, 49.0, 51.0];
        let entry = MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Wgpu,
            "phi-2",
            &latencies,
            &throughputs,
            150.0,
        );

        assert!(entry.available);
        assert_eq!(entry.runtime, RuntimeType::LlamaCpp);
        assert_eq!(entry.backend, ComputeBackendType::Wgpu);
        assert_eq!(entry.model, "phi-2");
        assert_eq!(entry.samples, 5);
        assert!((entry.p50_latency_ms - 102.0).abs() < 1.0); // Median
        assert!((entry.cold_start_ms - 150.0).abs() < 0.1);
        assert!(entry.throughput_tps > 0.0);
    }

    #[test]
    fn test_matrix_benchmark_entry_from_empty_samples() {
        let entry = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "model",
            &[],
            &[],
            0.0,
        );
        assert!(!entry.available);
    }

    #[test]
    fn test_matrix_benchmark_entry_with_notes() {
        let entry =
            MatrixBenchmarkEntry::unavailable(RuntimeType::Realizar, ComputeBackendType::Cuda)
                .with_notes("GPU layers: 99");
        assert_eq!(entry.notes, "GPU layers: 99");
    }
