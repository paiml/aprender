//! Part 4 of bench module tests - Statistical Tools, Regression Detection, HTTP Backends
//!
//! Focus areas:
//! - BENCH-006: OutlierDetector (MAD-based)
//! - BENCH-007: RegressionDetector
//! - BENCH-008: Welch's t-test
//! - BENCH-009: ThermalGuard (TDD RED)
//! - BENCH-010: KL-Divergence Quality Validation
//! - OllamaBackend HTTP Integration
//! - Distributed Benchmark Suite
//! - Load Testing
//! - LlamaCppBackend CLI Output Parsing
//! - Benchmark Matrix (EXTREME TDD)
//! - QA Checklist Validation Tests (QA-031 to QA-040)

#[cfg(test)]
#[allow(unexpected_cfgs)]
mod tests {
    use crate::bench::*;

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
include!("tests_outlier_detector.rs");
include!("tests_distributed_bench_02.rs");
include!("tests_parse_llama.rs");
}
