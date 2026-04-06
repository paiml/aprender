
    // ========================================================================
    // Distributed Benchmark Suite Tests
    // ========================================================================

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_distributed_bench_config_default() {
        let config = DistributedBenchConfig::default();
        assert_eq!(config.gpu_counts, vec![1, 2, 4, 8]);
        assert_eq!(config.iterations, 100);
        assert_eq!(config.warmup, 10);
        assert_eq!(config.model_params, 7_000_000_000);
        assert_eq!(config.seq_len, 2048);
        assert_eq!(config.batch_size, 1);
        assert!((config.efficiency_threshold - 0.85).abs() < 0.001);
    }

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_distributed_bench_config_small_model() {
        let config = DistributedBenchConfig::for_small_model();
        assert_eq!(config.gpu_counts, vec![1, 2]);
        assert_eq!(config.model_params, 125_000_000);
        assert!((config.efficiency_threshold - 0.80).abs() < 0.001);
    }

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_distributed_bench_config_large_model() {
        let config = DistributedBenchConfig::for_large_model();
        assert_eq!(config.gpu_counts, vec![2, 4, 8]);
        assert_eq!(config.model_params, 70_000_000_000);
        assert_eq!(config.seq_len, 4096);
    }

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_distributed_bench_suite_new() {
        let config = DistributedBenchConfig::default();
        let suite = DistributedBenchSuite::new(config.clone());
        assert_eq!(suite.config().gpu_counts, config.gpu_counts);
        assert!(suite.scaling_results().is_empty());
        assert!(suite.tp_results().is_empty());
        assert!(suite.pp_results().is_empty());
        assert!(suite.comm_results().is_empty());
    }

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_distributed_bench_scaling() {
        let config = DistributedBenchConfig::default();
        let mut suite = DistributedBenchSuite::new(config);
        suite.run_scaling_benchmark();

        let results = suite.scaling_results();
        assert_eq!(results.len(), 4); // 1, 2, 4, 8 GPUs

        // First result should be 1 GPU (baseline)
        assert_eq!(results[0].gpu_count, 1);
        assert!((results[0].efficiency - 1.0).abs() < 0.001);
        assert!(results[0].comm_overhead_ms.abs() < 0.001);

        // Multi-GPU should have lower efficiency due to overhead
        for result in results.iter().skip(1) {
            assert!(result.efficiency < 1.0);
            assert!(result.efficiency > 0.0); // Efficiency is always positive
            assert!(result.comm_overhead_ms > 0.0);
            assert!(result.throughput_tps > 0.0);
            assert!(result.latency_p50_ms > 0.0);
            assert!(result.latency_p99_ms > result.latency_p50_ms);
        }

        // 2 GPUs should be >85% efficient (spec target for 2-8 GPUs)
        let gpu2 = results.iter().find(|r| r.gpu_count == 2).expect("test");
        assert!(gpu2.efficiency > 0.85, "2-GPU efficiency should be >85%");
    }

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_scaling_efficiency_result_meets_threshold() {
        let result = ScalingEfficiencyResult {
            gpu_count: 4,
            throughput_tps: 400.0,
            latency_p50_ms: 2.5,
            latency_p99_ms: 3.75,
            efficiency: 0.90,
            comm_overhead_ms: 0.5,
            theoretical_speedup: 3.6,
            achieved_speedup: 3.4,
        };

        assert!(result.meets_threshold(0.85));
        assert!(result.meets_threshold(0.90));
        assert!(!result.meets_threshold(0.95));
    }

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_scaling_efficiency_parallel_fraction() {
        let result = ScalingEfficiencyResult {
            gpu_count: 4,
            throughput_tps: 400.0,
            latency_p50_ms: 2.5,
            latency_p99_ms: 3.75,
            efficiency: 0.85,
            comm_overhead_ms: 0.5,
            theoretical_speedup: 3.6,
            achieved_speedup: 3.4,
        };

        let parallel = result.parallel_fraction();
        assert!(parallel > 0.8); // Should be highly parallelizable
        assert!(parallel <= 1.0);

        // Single GPU case
        let single = ScalingEfficiencyResult {
            gpu_count: 1,
            throughput_tps: 100.0,
            latency_p50_ms: 10.0,
            latency_p99_ms: 15.0,
            efficiency: 1.0,
            comm_overhead_ms: 0.0,
            theoretical_speedup: 1.0,
            achieved_speedup: 1.0,
        };
        assert!((single.parallel_fraction() - 1.0).abs() < 0.001);
    }

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_distributed_bench_tensor_parallel() {
        let config = DistributedBenchConfig::default();
        let mut suite = DistributedBenchSuite::new(config);
        suite.run_tensor_parallel_benchmark();

        let results = suite.tp_results();
        assert!(!results.is_empty());

        // Check that TP=1 has no communication overhead
        let tp1 = results.iter().find(|r| r.tp_degree == 1).expect("test");
        assert!(tp1.all_reduce_ms.abs() < 0.001);
        assert!(tp1.comm_overhead_pct.abs() < 0.001);

        // Check that higher TP degrees have communication overhead
        for result in results.iter().filter(|r| r.tp_degree > 1) {
            assert!(result.all_reduce_ms > 0.0);
            assert!(result.comm_overhead_pct > 0.0);
            assert!(result.memory_per_gpu_mb > 0.0);
            assert!(result.effective_tflops > 0.0);
        }
    }

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_distributed_bench_pipeline_parallel() {
        let config = DistributedBenchConfig::default();
        let mut suite = DistributedBenchSuite::new(config);
        suite.run_pipeline_parallel_benchmark();

        let results = suite.pp_results();
        assert!(!results.is_empty());

        // Check PP=1 has no bubble
        let pp1 = results.iter().find(|r| r.pp_degree == 1).expect("test");
        assert!(pp1.bubble_ratio.abs() < 0.001);
        assert!(pp1.inter_stage_ms.abs() < 0.001);

        // Check higher PP degrees have bubble and inter-stage latency
        for result in results.iter().filter(|r| r.pp_degree > 1) {
            assert!(result.bubble_ratio > 0.0);
            assert!(result.bubble_ratio < 1.0); // Should be <100%
            assert!(result.inter_stage_ms > 0.0);
            assert!(result.micro_batches > 0);
            assert!(result.throughput_tps > 0.0);
            assert!(result.memory_per_stage_mb > 0.0);
        }
    }

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
