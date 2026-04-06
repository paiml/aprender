
// ============================================================================
// Tests (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // MatrixSummary Tests
    // =========================================================================

    #[test]
    fn test_benchmark_matrix_summary() {
        let hw = make_hardware_spec();
        let mut matrix = BenchmarkMatrix::new("model", hw);

        let mut entry1 = MatrixBenchmarkEntry::default();
        entry1.runtime = RuntimeType::Realizar;
        entry1.backend = ComputeBackendType::Cpu;
        entry1.available = true;
        entry1.p50_latency_ms = 20.0;
        entry1.throughput_tps = 100.0;
        matrix.add_entry(entry1);

        let mut entry2 = MatrixBenchmarkEntry::default();
        entry2.runtime = RuntimeType::LlamaCpp;
        entry2.backend = ComputeBackendType::Cpu;
        entry2.available = true;
        entry2.p50_latency_ms = 10.0;
        entry2.throughput_tps = 150.0;
        matrix.add_entry(entry2);

        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::Vllm,
            ComputeBackendType::Cuda,
        ));

        let summary = matrix.summary();
        assert_eq!(summary.total_entries, 3);
        assert_eq!(summary.available_entries, 2);

        // Overall fastest should be llama-cpp (10ms)
        assert!(summary.overall_fastest.is_some());
        assert_eq!(summary.overall_fastest.as_ref().expect("as_ref").0, "llamacpp");

        // Overall highest throughput should be llama-cpp (150 tok/s)
        assert!(summary.overall_highest_throughput.is_some());
        assert_eq!(
            summary.overall_highest_throughput.as_ref().expect("as_ref").0,
            "llamacpp"
        );
    }

    #[test]
    fn test_matrix_summary_serialize() {
        let summary = MatrixSummary {
            total_entries: 5,
            available_entries: 3,
            backend_summaries: vec![],
            overall_fastest: Some(("realizar".to_string(), "cuda".to_string())),
            overall_highest_throughput: None,
        };
        let json = serde_json::to_string(&summary).expect("json");
        assert!(json.contains("total_entries"));
        assert!(json.contains("realizar"));
    }
include!("compute_backend_type_tests.rs");
}
