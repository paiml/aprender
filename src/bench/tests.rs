#[cfg(test)]
#[allow(unexpected_cfgs)]
mod tests {
    use crate::bench::*;

    #[test]
    fn test_matrix_summary_struct_fields() {
        let summary = MatrixSummary {
            total_entries: 10,
            available_entries: 8,
            backend_summaries: vec![],
            overall_fastest: Some(("realizar".to_string(), "cuda".to_string())),
            overall_highest_throughput: Some(("llama-cpp".to_string(), "cuda".to_string())),
        };

        assert_eq!(summary.total_entries, 10);
        assert_eq!(summary.available_entries, 8);
        assert!(summary.backend_summaries.is_empty());
        assert_eq!(
            summary.overall_fastest,
            Some(("realizar".to_string(), "cuda".to_string()))
        );
        assert_eq!(
            summary.overall_highest_throughput,
            Some(("llama-cpp".to_string(), "cuda".to_string()))
        );
    }

    #[test]
    fn test_benchmark_matrix_to_markdown_unavailable_entries() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("test-model", hardware);

        // Add both available and unavailable entries
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test",
            &[100.0],
            &[50.0],
            90.0,
        ));

        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cuda,
        ));

        let markdown = matrix.to_markdown_table();

        // Should contain table structure
        assert!(markdown.contains("|"));
        assert!(markdown.contains("Runtime"));
        // Should contain dash for unavailable metrics
        assert!(markdown.contains("-"));
    }

    #[test]
    fn test_compute_backend_type_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(ComputeBackendType::Cpu);
        set.insert(ComputeBackendType::Wgpu);
        set.insert(ComputeBackendType::Cuda);

        assert_eq!(set.len(), 3);
        assert!(set.contains(&ComputeBackendType::Cpu));
        assert!(set.contains(&ComputeBackendType::Wgpu));
        assert!(set.contains(&ComputeBackendType::Cuda));
    }

    #[test]
    fn test_compute_backend_type_eq() {
        assert_eq!(ComputeBackendType::Cpu, ComputeBackendType::Cpu);
        assert_eq!(ComputeBackendType::Wgpu, ComputeBackendType::Wgpu);
        assert_eq!(ComputeBackendType::Cuda, ComputeBackendType::Cuda);
        assert_ne!(ComputeBackendType::Cpu, ComputeBackendType::Wgpu);
        assert_ne!(ComputeBackendType::Cpu, ComputeBackendType::Cuda);
        assert_ne!(ComputeBackendType::Wgpu, ComputeBackendType::Cuda);
    }

    #[test]
    fn test_compute_backend_type_copy() {
        let original = ComputeBackendType::Cuda;
        let copied = original;
        assert_eq!(original, copied);
    }

    #[test]
    fn test_matrix_benchmark_entry_serialization_roundtrip() {
        let entry = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cuda,
            "phi-2",
            &[50.0, 55.0, 52.0],
            &[100.0, 105.0, 102.0],
            80.0,
        )
        .with_notes("GPU layers: 99");

        let json = serde_json::to_string(&entry).expect("Serialization should succeed");
        let deser: MatrixBenchmarkEntry =
            serde_json::from_str(&json).expect("Deserialization should succeed");

        assert_eq!(deser.runtime, RuntimeType::Realizar);
        assert_eq!(deser.backend, ComputeBackendType::Cuda);
        assert_eq!(deser.model, "phi-2");
        assert!(deser.available);
        assert_eq!(deser.samples, 3);
        assert_eq!(deser.notes, "GPU layers: 99");
    }

    #[test]
    fn test_benchmark_matrix_entries_for_runtime_empty() {
        let hardware = HardwareSpec::default();
        let matrix = BenchmarkMatrix::new("test-model", hardware);

        let entries = matrix.entries_for_runtime(RuntimeType::Realizar);
        assert!(entries.is_empty());
    }

    #[test]
    fn test_benchmark_matrix_entries_for_backend_with_multiple() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("test-model", hardware);

        // Add multiple entries for the same backend
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test",
            &[100.0],
            &[50.0],
            90.0,
        ));

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cpu,
            "test",
            &[80.0],
            &[60.0],
            85.0,
        ));

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Ollama,
            ComputeBackendType::Cpu,
            "test",
            &[90.0],
            &[55.0],
            95.0,
        ));

        let cpu_entries = matrix.entries_for_backend(ComputeBackendType::Cpu);
        assert_eq!(cpu_entries.len(), 3);

        // Verify all are for CPU backend
        for entry in &cpu_entries {
            assert_eq!(entry.backend, ComputeBackendType::Cpu);
        }
    }

    #[test]
    fn test_load_test_config_clone() {
        let config = LoadTestConfig::for_stress_test();
        let cloned = config.clone();

        assert_eq!(cloned.concurrency, config.concurrency);
        assert_eq!(cloned.duration_secs, config.duration_secs);
        assert!((cloned.target_rps - config.target_rps).abs() < 0.001);
        assert_eq!(cloned.timeout_ms, config.timeout_ms);
        assert_eq!(cloned.warmup_secs, config.warmup_secs);
        assert!((cloned.latency_threshold_ms - config.latency_threshold_ms).abs() < 0.001);
    }

    #[test]
    fn test_load_test_result_clone() {
        let result = LoadTestResult {
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
        let cloned = result.clone();

        assert_eq!(cloned.total_requests, result.total_requests);
        assert_eq!(cloned.successful_requests, result.successful_requests);
        assert!((cloned.error_rate - result.error_rate).abs() < 0.0001);
    }
include!("tests_dynamic_sampler.rs");
include!("tests_convoy_saturation.rs");
include!("tests_thermal_guard.rs");
include!("tests_mock_backend_registry.rs");
include!("tests_welch_thermal_guard.rs");
include!("tests_distributed_bench.rs");
include!("tests_benchmark_matrix.rs");
include!("tests_036.rs");
include!("tests_050.rs");
include!("tests_imp900.rs");
}
