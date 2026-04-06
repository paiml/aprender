
    #[test]
    fn test_benchmark_matrix_creation() {
        let hardware = HardwareSpec::default();
        let matrix = BenchmarkMatrix::new("phi-2", hardware);

        assert_eq!(matrix.model, "phi-2");
        assert_eq!(matrix.version, "1.1");
        assert!(matrix.methodology.contains("Hoefler"));
        assert!(matrix.entries.is_empty());
    }

    #[test]
    fn test_benchmark_matrix_add_entry() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("phi-2", hardware);

        let entry1 = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[100.0, 102.0, 98.0],
            &[50.0, 51.0, 49.0],
            100.0,
        );
        matrix.add_entry(entry1);

        assert_eq!(matrix.entries.len(), 1);

        // Add another entry
        let entry2 = MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Wgpu,
            "phi-2",
            &[80.0, 82.0, 78.0],
            &[60.0, 61.0, 59.0],
            120.0,
        );
        matrix.add_entry(entry2);

        assert_eq!(matrix.entries.len(), 2);

        // Replace existing entry
        let entry1_updated = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[90.0, 92.0, 88.0],
            &[55.0, 56.0, 54.0],
            95.0,
        );
        matrix.add_entry(entry1_updated);

        assert_eq!(matrix.entries.len(), 2); // Still 2, replaced
    }

    #[test]
    fn test_benchmark_matrix_get_entry() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("phi-2", hardware);

        let entry = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[100.0],
            &[50.0],
            100.0,
        );
        matrix.add_entry(entry);

        let found = matrix.get_entry(RuntimeType::Realizar, ComputeBackendType::Cpu);
        assert!(found.is_some());
        assert_eq!(found.expect("test").runtime, RuntimeType::Realizar);

        let not_found = matrix.get_entry(RuntimeType::LlamaCpp, ComputeBackendType::Cuda);
        assert!(not_found.is_none());
    }

    #[test]
    fn test_benchmark_matrix_entries_for_runtime() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("phi-2", hardware);

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[100.0],
            &[50.0],
            100.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Wgpu,
            "phi-2",
            &[80.0],
            &[60.0],
            90.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cpu,
            "phi-2",
            &[90.0],
            &[55.0],
            95.0,
        ));

        let realizar_entries = matrix.entries_for_runtime(RuntimeType::Realizar);
        assert_eq!(realizar_entries.len(), 2);

        let llama_entries = matrix.entries_for_runtime(RuntimeType::LlamaCpp);
        assert_eq!(llama_entries.len(), 1);
    }

    #[test]
    fn test_benchmark_matrix_entries_for_backend() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("phi-2", hardware);

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[100.0],
            &[50.0],
            100.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cpu,
            "phi-2",
            &[90.0],
            &[55.0],
            95.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Wgpu,
            "phi-2",
            &[80.0],
            &[60.0],
            90.0,
        ));

        let cpu_entries = matrix.entries_for_backend(ComputeBackendType::Cpu);
        assert_eq!(cpu_entries.len(), 2);

        let wgpu_entries = matrix.entries_for_backend(ComputeBackendType::Wgpu);
        assert_eq!(wgpu_entries.len(), 1);

        let cuda_entries = matrix.entries_for_backend(ComputeBackendType::Cuda);
        assert!(cuda_entries.is_empty());
    }

    #[test]
    fn test_benchmark_matrix_fastest_for_backend() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("phi-2", hardware);

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[100.0, 102.0, 98.0],
            &[50.0],
            100.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cpu,
            "phi-2",
            &[80.0, 82.0, 78.0], // Faster
            &[55.0],
            95.0,
        ));

        let fastest = matrix.fastest_for_backend(ComputeBackendType::Cpu);
        assert!(fastest.is_some());
        assert_eq!(fastest.expect("test").runtime, RuntimeType::LlamaCpp);
    }

    #[test]
    fn test_benchmark_matrix_highest_throughput_for_backend() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("phi-2", hardware);

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[100.0],
            &[50.0, 51.0, 49.0],
            100.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cpu,
            "phi-2",
            &[90.0],
            &[70.0, 71.0, 69.0], // Higher throughput
            95.0,
        ));

        let highest = matrix.highest_throughput_for_backend(ComputeBackendType::Cpu);
        assert!(highest.is_some());
        assert_eq!(highest.expect("test").runtime, RuntimeType::LlamaCpp);
    }

    #[test]
    fn test_benchmark_matrix_to_markdown_table() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("phi-2", hardware);

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[100.0, 110.0, 105.0],
            &[50.0, 51.0, 49.0],
            100.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::Realizar,
            ComputeBackendType::Cuda,
        ));

        let table = matrix.to_markdown_table();
        assert!(table.contains("| Runtime | Backend |"));
        assert!(table.contains("| **realizar** |"));
        assert!(table.contains("| - | - |")); // Unavailable entry
    }

    #[test]
    fn test_benchmark_matrix_json_roundtrip() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("phi-2", hardware);

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[100.0, 102.0, 98.0],
            &[50.0, 51.0, 49.0],
            100.0,
        ));

        let json = matrix.to_json().expect("serialization should succeed");
        assert!(json.contains("\"model\": \"phi-2\""));

        let parsed = BenchmarkMatrix::from_json(&json).expect("deserialization should succeed");
        assert_eq!(parsed.model, "phi-2");
        assert_eq!(parsed.entries.len(), 1);
    }

    #[test]
    fn test_benchmark_matrix_summary() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("phi-2", hardware);

        // Add entries for different combinations
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[100.0, 102.0, 98.0],
            &[50.0, 51.0, 49.0],
            100.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cpu,
            "phi-2",
            &[80.0, 82.0, 78.0],
            &[70.0, 71.0, 69.0],
            95.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Wgpu,
            "phi-2",
            &[60.0, 62.0, 58.0], // Fastest overall
            &[80.0, 81.0, 79.0], // Highest throughput overall
            90.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::Realizar,
            ComputeBackendType::Cuda,
        ));

        let summary = matrix.summary();

        assert_eq!(summary.total_entries, 4);
        assert_eq!(summary.available_entries, 3);

        // Overall fastest should be llama-cpp with wgpu (p50 ~60ms)
        assert!(summary.overall_fastest.is_some());
        let (fastest_runtime, fastest_backend) = summary.overall_fastest.expect("test");
        assert_eq!(fastest_runtime, "llamacpp");
        assert_eq!(fastest_backend, "wgpu");

        // Overall highest throughput should also be llama-cpp with wgpu (~80 tok/s)
        assert!(summary.overall_highest_throughput.is_some());
        let (tp_runtime, tp_backend) = summary.overall_highest_throughput.expect("test");
        assert_eq!(tp_runtime, "llamacpp");
        assert_eq!(tp_backend, "wgpu");
    }

    #[test]
    fn test_matrix_benchmark_config_default() {
        let config = MatrixBenchmarkConfig::default();

        assert!(config.runtimes.contains(&RuntimeType::Realizar));
        assert!(config.runtimes.contains(&RuntimeType::LlamaCpp));
        assert!(config.runtimes.contains(&RuntimeType::Ollama));
        assert!(config.backends.contains(&ComputeBackendType::Cpu));
        assert!(config.backends.contains(&ComputeBackendType::Wgpu));
        assert_eq!(config.cv_threshold, 0.05);
        assert_eq!(config.min_samples, 30);
        assert_eq!(config.max_samples, 200);
        assert_eq!(config.warmup_iterations, 5);
    }

    // ========================================================================
    // QA Checklist Validation Tests
    // Per spec: performance-parity-ollama-llamacpp-gpu-inference-llms.md §5
    // ========================================================================

    /// QA-031: Benchmark framework produces valid statistical metrics
    #[test]
    fn test_qa_031_benchmark_statistical_validity() {
        // DynamicSampler must produce valid CV calculations
        let sampler = DynamicSampler::new(10, 100, 0.05);

        // Stable samples should produce low CV
        let stable_samples: Vec<f64> = (0..50).map(|_| 100.0).collect();
        let cv = sampler.current_cv(&stable_samples);
        assert!(
            cv.abs() < 0.001,
            "QA-031: Stable samples should have near-zero CV, got {}",
            cv
        );

        // Variable samples should produce higher CV
        let variable_samples: Vec<f64> = (0..50).map(|i| 50.0 + (i as f64) * 2.0).collect();
        let cv_var = sampler.current_cv(&variable_samples);
        assert!(
            cv_var > 0.1,
            "QA-031: Variable samples should have measurable CV, got {}",
            cv_var
        );
    }

    /// QA-032: Thermal guard validates temperature variance correctly
    #[test]
    fn test_qa_032_thermal_guard_validation() {
        let guard = ThermalGuard::default();

        // Default thermal guard should have sensible thresholds
        assert!(
            guard.max_temp_c > 70.0 && guard.max_temp_c <= 95.0,
            "QA-032: Max temp should be in safe GPU range"
        );
        assert!(
            guard.cooldown_threshold_c < guard.max_temp_c,
            "QA-032: Cooldown threshold must be below max temp"
        );
        assert!(
            guard.temp_variance_c > 0.0 && guard.temp_variance_c <= 5.0,
            "QA-032: Temperature variance threshold should be reasonable"
        );
    }

    /// QA-033: ITL metrics capture variance correctly
    #[test]
    fn test_qa_033_itl_variance_capture() {
        let samples = vec![10.0, 12.0, 11.0, 13.0, 10.0, 15.0, 11.0, 12.0];
        let metrics = ItlMetrics::from_measurements(&samples);

        // p99 should be >= p999 (order check)
        // Actually p999 >= p99 in expectation (tail values)
        assert!(
            metrics.p999_ms >= metrics.p99_ms,
            "QA-033: p999 should be >= p99"
        );
        assert!(
            metrics.p99_ms >= metrics.median_ms,
            "QA-033: p99 should be >= median"
        );

        // Median and std_dev should be non-negative
        assert!(metrics.median_ms > 0.0, "QA-033: Median should be positive");
        assert!(
            metrics.std_dev_ms >= 0.0,
            "QA-033: Std dev should be non-negative"
        );
    }

    /// QA-034: CV-based stopping rule converges
    #[test]
    #[allow(clippy::similar_names)] // sampler vs samples are related but distinct concepts
    fn test_qa_034_cv_stopping_convergence() {
        let mut sampler = DynamicSampler::new(10, 1000, 0.05);
        sampler.stability_count = 3;

        // Feed stable samples - should eventually stop
        let mut samples = Vec::new();
        let mut stopped = false;

        for i in 0..100 {
            samples.push(100.0 + (i as f64 % 3.0)); // Small variance
            if !sampler.should_continue(&samples) {
                stopped = true;
                break;
            }
        }

        assert!(
            stopped,
            "QA-034: CV-based stopping should converge for stable samples"
        );
    }

    /// QA-035: Benchmark results are serializable
    #[test]
    fn test_qa_035_benchmark_serialization() {
        let entry = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test-model",
            &[50.0, 52.0, 48.0],
            &[100.0, 98.0, 102.0],
            95.0,
        );

        // Should serialize to JSON without error
        let json = serde_json::to_string(&entry);
        assert!(
            json.is_ok(),
            "QA-035: MatrixBenchmarkEntry should serialize"
        );

        // Should deserialize back
        let deser: Result<MatrixBenchmarkEntry, _> = serde_json::from_str(&json.expect("test"));
        assert!(
            deser.is_ok(),
            "QA-035: MatrixBenchmarkEntry should deserialize"
        );
    }
