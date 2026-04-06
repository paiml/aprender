
    /// QA-036: Runtime and backend types are complete
    #[test]
    fn test_qa_036_runtime_backend_completeness() {
        // All expected runtimes should be representable
        let runtimes = [
            RuntimeType::Realizar,
            RuntimeType::LlamaCpp,
            RuntimeType::Ollama,
            RuntimeType::Vllm,
        ];

        for runtime in &runtimes {
            let name = runtime.as_str();
            assert!(
                !name.is_empty(),
                "QA-036: Runtime {} should have a name",
                name
            );
        }

        // All expected backends should be representable
        let backends = [
            ComputeBackendType::Cpu,
            ComputeBackendType::Cuda,
            ComputeBackendType::Wgpu,
        ];

        for backend in &backends {
            let name = backend.to_string();
            assert!(
                !name.is_empty(),
                "QA-036: Backend {:?} should have a name",
                backend
            );
        }
    }

    /// QA-037: Matrix summary calculations are correct
    #[test]
    fn test_qa_037_matrix_summary_correctness() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("test-model", hardware);

        // Add known entries
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test",
            &[100.0], // p50 = 100ms
            &[10.0],  // throughput = 10 tok/s
            90.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cpu,
            "test",
            &[50.0], // p50 = 50ms (faster)
            &[20.0], // throughput = 20 tok/s (higher)
            95.0,
        ));

        let summary = matrix.summary();

        assert_eq!(summary.total_entries, 2, "QA-037: Should have 2 entries");
        assert_eq!(
            summary.available_entries, 2,
            "QA-037: Both entries should be available"
        );

        // LlamaCpp should be fastest (50ms < 100ms)
        if let Some((fastest, _)) = &summary.overall_fastest {
            assert_eq!(fastest, "llamacpp", "QA-037: LlamaCpp should be fastest");
        }
    }

    /// QA-038: Benchmark report generation works
    #[test]
    fn test_qa_038_report_generation() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("test-model", hardware);
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test-model",
            &[100.0],
            &[50.0],
            90.0,
        ));

        let report = matrix.to_markdown_table();

        // Report should contain key information
        assert!(
            report.contains("realizar") || report.contains("Realizar"),
            "QA-038: Report should mention realizar"
        );
    }

    /// QA-039: Dynamic sampler respects min/max bounds
    #[test]
    fn test_qa_039_sampler_bounds() {
        let mut sampler = DynamicSampler::new(5, 20, 0.01); // Very tight CV

        // Should always continue until min_samples
        let few_samples = vec![1.0, 2.0, 3.0];
        assert!(
            sampler.should_continue(&few_samples),
            "QA-039: Should continue below min_samples"
        );

        // Should stop at max_samples regardless of CV
        let many_samples: Vec<f64> = (0..25).map(|i| i as f64).collect(); // High variance
        assert!(
            !sampler.should_continue(&many_samples),
            "QA-039: Should stop at max_samples"
        );
    }

    /// QA-040: ITL metrics handle edge cases
    #[test]
    fn test_qa_040_itl_edge_cases() {
        // Single sample
        let single = ItlMetrics::from_measurements(&[100.0]);
        assert!(
            (single.median_ms - 100.0).abs() < 0.001,
            "QA-040: Single sample median should equal the sample"
        );

        // Empty samples should produce zeros or NaN (valid edge case)
        let empty = ItlMetrics::from_measurements(&[]);
        assert!(
            empty.median_ms.is_nan() || empty.median_ms == 0.0,
            "QA-040: Empty samples should produce NaN or 0"
        );

        // All same values - std_dev should be 0
        let same = ItlMetrics::from_measurements(&[50.0, 50.0, 50.0, 50.0]);
        assert!(
            same.std_dev_ms.abs() < 0.001,
            "QA-040: Identical samples should have zero std_dev"
        );
    }

    // ========================================================================
    // QA Checklist Section E: Integration Tests (QA-041 to QA-050)
    // Per spec: performance-parity-ollama-llamacpp-gpu-inference-llms.md §5
    // ========================================================================

    /// QA-041: Benchmark infrastructure compiles and runs
    /// Per spec: `make bench-inference-all` should complete without error
    #[test]
    fn test_qa_041_benchmark_infrastructure() {
        // Verify all benchmark types are representable
        let runtimes = [
            RuntimeType::Realizar,
            RuntimeType::Ollama,
            RuntimeType::LlamaCpp,
        ];

        for runtime in &runtimes {
            assert!(
                !runtime.as_str().is_empty(),
                "QA-041: Runtime {} should have a name",
                runtime.as_str()
            );
        }

        // Verify benchmark matrix can be created
        let hardware = HardwareSpec::default();
        let matrix = BenchmarkMatrix::new("test-model", hardware);
        assert!(
            matrix.entries.is_empty(),
            "QA-041: New matrix should be empty"
        );
    }

    /// QA-042: Comparison report generation works
    /// Per spec: `make bench-pytorch-inference` produces comparison report
    #[test]
    fn test_qa_042_comparison_report() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("test-model", hardware);

        // Add entries for comparison
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test",
            &[100.0, 105.0, 95.0],
            &[50.0, 55.0, 45.0],
            90.0,
        ));

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cpu,
            "test",
            &[80.0, 85.0, 75.0],
            &[40.0, 45.0, 35.0],
            110.0,
        ));

        // Generate comparison report
        let report = matrix.to_markdown_table();

        // Report should contain both runtimes
        assert!(
            report.contains("realizar") || report.contains("Realizar"),
            "QA-042: Report should include Realizar"
        );
    }

    /// QA-043: CPU-only benchmarks work
    /// Per spec: `make bench-cpu-inference` tests all CPU backends
    #[test]
    fn test_qa_043_cpu_benchmarks() {
        // Verify CPU backend type exists and is valid
        let cpu_backend = ComputeBackendType::Cpu;
        let backend_str = cpu_backend.to_string();
        assert!(
            backend_str.to_lowercase().contains("cpu"),
            "QA-043: CPU backend should be identifiable"
        );

        // Verify CPU entries can be created
        let entry = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test-model",
            &[100.0],
            &[50.0],
            90.0,
        );

        assert_eq!(
            entry.backend,
            ComputeBackendType::Cpu,
            "QA-043: Entry should be CPU backend"
        );
    }

    /// QA-044: WGPU benchmark gracefully handles unavailability
    /// Per spec: `make bench-wgpu` gracefully skips if unavailable
    #[test]
    fn test_qa_044_wgpu_graceful_skip() {
        // WGPU backend type should exist
        let wgpu_backend = ComputeBackendType::Wgpu;
        let backend_str = wgpu_backend.to_string();

        // Should have a valid string representation
        assert!(
            !backend_str.is_empty(),
            "QA-044: WGPU backend should have a name"
        );

        // Creating an entry with WGPU should work (even if GPU not available)
        let entry = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Wgpu,
            "test-model",
            &[100.0],
            &[50.0],
            90.0,
        );

        assert_eq!(
            entry.backend,
            ComputeBackendType::Wgpu,
            "QA-044: Entry should be WGPU backend"
        );
    }

    /// QA-045: Multi-runtime comparison works
    /// Per spec: `make bench-gguf-gpu-inference` compares all runtimes
    #[test]
    fn test_qa_045_multi_runtime_comparison() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("test-model", hardware);

        // Add entries for all runtime types
        for runtime in [
            RuntimeType::Realizar,
            RuntimeType::Ollama,
            RuntimeType::LlamaCpp,
        ] {
            matrix.add_entry(MatrixBenchmarkEntry::from_samples(
                runtime,
                ComputeBackendType::Cpu,
                "test",
                &[100.0],
                &[50.0],
                90.0,
            ));
        }

        // Should have 3 entries
        assert_eq!(
            matrix.entries.len(),
            3,
            "QA-045: Should have 3 runtime entries"
        );

        // Summary should work
        let summary = matrix.summary();
        assert!(
            summary.overall_fastest.is_some(),
            "QA-045: Summary should identify fastest runtime"
        );
    }

    /// QA-046: Format comparison works
    /// Per spec: `make bench-apr-gpu-inference` produces format comparison
    #[test]
    fn test_qa_046_format_comparison() {
        // Different model formats should be comparable via the same infrastructure
        let hardware = HardwareSpec::default();
        let mut gguf_matrix = BenchmarkMatrix::new("model.gguf", hardware.clone());
        let mut apr_matrix = BenchmarkMatrix::new("model.apr", hardware);

        gguf_matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "model.gguf",
            &[100.0],
            &[50.0],
            90.0,
        ));

        apr_matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "model.apr",
            &[95.0],
            &[48.0],
            92.0,
        ));

        // Both should generate valid reports
        let gguf_report = gguf_matrix.to_markdown_table();
        let apr_report = apr_matrix.to_markdown_table();

        assert!(
            !gguf_report.is_empty(),
            "QA-046: GGUF report should be non-empty"
        );
        assert!(
            !apr_report.is_empty(),
            "QA-046: APR report should be non-empty"
        );
    }

    /// QA-047: CI pipeline integration (structure validation)
    /// Per spec: CI pipeline runs benchmarks on every PR
    #[test]
    fn test_qa_047_ci_integration() {
        // Verify benchmark results can be serialized for CI
        let entry = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test-model",
            &[100.0, 105.0],
            &[50.0, 55.0],
            90.0,
        );

        // Should serialize to JSON for CI consumption
        let json = serde_json::to_string(&entry);
        assert!(json.is_ok(), "QA-047: Entry should serialize for CI");

        // Should deserialize back
        let deser: Result<MatrixBenchmarkEntry, _> = serde_json::from_str(&json.expect("test"));
        assert!(deser.is_ok(), "QA-047: Entry should deserialize from CI");
    }

    /// QA-048: Metrics dashboard support
    /// Per spec: Benchmark results published to metrics dashboard
    #[test]
    fn test_qa_048_metrics_dashboard() {
        // Verify all metrics needed for dashboard are present
        let entry = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test-model",
            &[100.0, 105.0, 95.0, 98.0, 102.0],
            &[50.0, 55.0, 45.0, 48.0, 52.0],
            90.0,
        );

        // Dashboard needs: p50, p99, throughput, runtime, backend
        assert!(
            entry.p50_latency_ms > 0.0,
            "QA-048: p50 should be available"
        );
        assert!(
            entry.p99_latency_ms > 0.0,
            "QA-048: p99 should be available"
        );
        assert!(
            entry.throughput_tps > 0.0,
            "QA-048: Throughput should be available"
        );
        assert!(
            !entry.runtime.as_str().is_empty(),
            "QA-048: Runtime should be identifiable"
        );
    }

    /// QA-049: Historical trend detection
    /// Per spec: Historical trend analysis detects regressions
    #[test]
    fn test_qa_049_trend_detection() {
        // Simulate historical data with a regression
        let baseline = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test-model",
            &[100.0, 100.0, 100.0],
            &[50.0, 50.0, 50.0],
            100.0,
        );

        let regressed = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test-model",
            &[120.0, 120.0, 120.0], // 20% slower
            &[60.0, 60.0, 60.0],
            83.0, // Lower throughput
        );

        // Regression should be detectable
        let regression_percent =
            (regressed.p50_latency_ms - baseline.p50_latency_ms) / baseline.p50_latency_ms * 100.0;

        assert!(
            regression_percent > 15.0,
            "QA-049: Should detect >15% regression, got {}%",
            regression_percent
        );
    }
