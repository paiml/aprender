
// ============================================================================
// Tests (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Constants Tests
    // =========================================================================

    #[test]
    fn test_apr_cpu_decode_threshold() {
        assert!((APR_CPU_DECODE_THRESHOLD_TOK_S - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_apr_prefill_threshold() {
        assert!((APR_PREFILL_THRESHOLD_TOK_S - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_apr_parity_threshold() {
        assert!((APR_PARITY_THRESHOLD_PCT - 95.0).abs() < f64::EPSILON);
    }

    // =========================================================================
    // AprBenchmarkResult Tests
    // =========================================================================

    #[test]
    fn test_benchmark_result_default() {
        let result = AprBenchmarkResult::default();
        assert_eq!(result.tokens_generated, 0);
        assert_eq!(result.total_time_ms, 0.0);
        assert_eq!(result.tokens_per_second, 0.0);
        assert_eq!(result.throughput_p50, 0.0);
        assert_eq!(result.throughput_p99, 0.0);
        assert_eq!(result.throughput_std_dev, 0.0);
        assert_eq!(result.peak_memory_mb, 0.0);
        assert_eq!(result.model_memory_mb, 0.0);
    }

    #[test]
    fn test_benchmark_result_clone() {
        let result = AprBenchmarkResult {
            tokens_generated: 100,
            total_time_ms: 1000.0,
            tokens_per_second: 100.0,
            throughput_p50: 95.0,
            throughput_p99: 80.0,
            throughput_std_dev: 5.0,
            peak_memory_mb: 512.0,
            model_memory_mb: 256.0,
        };
        let cloned = result.clone();
        assert_eq!(cloned.tokens_generated, 100);
        assert_eq!(cloned.tokens_per_second, 100.0);
    }

    #[test]
    fn test_benchmark_result_debug() {
        let result = AprBenchmarkResult::default();
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("AprBenchmarkResult"));
        assert!(debug_str.contains("tokens_generated"));
    }

    #[test]
    fn test_meets_threshold_pass() {
        let result = AprBenchmarkResult {
            tokens_per_second: 60.0,
            ..Default::default()
        };
        assert!(result.meets_threshold(50.0));
        assert!(result.meets_threshold(60.0)); // equal
    }

    #[test]
    fn test_meets_threshold_fail() {
        let result = AprBenchmarkResult {
            tokens_per_second: 40.0,
            ..Default::default()
        };
        assert!(!result.meets_threshold(50.0));
    }

    #[test]
    fn test_meets_threshold_zero() {
        let result = AprBenchmarkResult::default();
        assert!(!result.meets_threshold(50.0));
        assert!(result.meets_threshold(0.0)); // 0 >= 0
    }

    #[test]
    fn test_compare_to_baseline_normal() {
        let result = AprBenchmarkResult {
            tokens_per_second: 90.0,
            peak_memory_mb: 500.0,
            ..Default::default()
        };
        let baseline = AprBenchmarkResult {
            tokens_per_second: 100.0,
            peak_memory_mb: 400.0,
            ..Default::default()
        };
        let comparison = result.compare_to_baseline(&baseline);
        assert!((comparison.throughput_ratio - 0.9).abs() < 0.001);
        assert!((comparison.memory_ratio - 1.25).abs() < 0.001);
        assert_eq!(comparison.parity_threshold_pct, APR_PARITY_THRESHOLD_PCT);
    }

    #[test]
    fn test_compare_to_baseline_zero_baseline() {
        let result = AprBenchmarkResult {
            tokens_per_second: 100.0,
            peak_memory_mb: 500.0,
            ..Default::default()
        };
        let baseline = AprBenchmarkResult::default(); // zeros
        let comparison = result.compare_to_baseline(&baseline);
        assert_eq!(comparison.throughput_ratio, 1.0);
        assert_eq!(comparison.memory_ratio, 1.0);
    }

    #[test]
    fn test_compare_to_baseline_equal() {
        let result = AprBenchmarkResult {
            tokens_per_second: 100.0,
            peak_memory_mb: 500.0,
            ..Default::default()
        };
        let comparison = result.compare_to_baseline(&result);
        assert!((comparison.throughput_ratio - 1.0).abs() < 0.001);
        assert!((comparison.memory_ratio - 1.0).abs() < 0.001);
    }

    // =========================================================================
    // AprPrefillResult Tests
    // =========================================================================

    #[test]
    fn test_prefill_result_default() {
        let result = AprPrefillResult::default();
        assert_eq!(result.prompt_tokens, 0);
        assert_eq!(result.prefill_time_ms, 0.0);
        assert_eq!(result.prefill_tok_s, 0.0);
    }

    #[test]
    fn test_prefill_result_clone() {
        let result = AprPrefillResult {
            prompt_tokens: 512,
            prefill_time_ms: 50.0,
            prefill_tok_s: 10240.0,
        };
        let cloned = result.clone();
        assert_eq!(cloned.prompt_tokens, 512);
        assert_eq!(cloned.prefill_time_ms, 50.0);
        assert_eq!(cloned.prefill_tok_s, 10240.0);
    }

    #[test]
    fn test_prefill_result_debug() {
        let result = AprPrefillResult::default();
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("AprPrefillResult"));
        assert!(debug_str.contains("prompt_tokens"));
    }

    // =========================================================================
    // AprLoadResult Tests
    // =========================================================================

    #[test]
    fn test_load_result_default() {
        let result = AprLoadResult::default();
        assert_eq!(result.load_time_ms, 0.0);
    }

    #[test]
    fn test_load_result_clone() {
        let result = AprLoadResult {
            load_time_ms: 1234.5,
        };
        let cloned = result.clone();
        assert_eq!(cloned.load_time_ms, 1234.5);
    }

    #[test]
    fn test_load_result_debug() {
        let result = AprLoadResult {
            load_time_ms: 999.0,
        };
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("AprLoadResult"));
        assert!(debug_str.contains("load_time_ms"));
    }

    // =========================================================================
    // AprParityComparison Tests
    // =========================================================================

    #[test]
    fn test_parity_comparison_clone() {
        let comparison = AprParityComparison {
            throughput_ratio: 0.96,
            memory_ratio: 1.1,
            parity_threshold_pct: 95.0,
        };
        let cloned = comparison.clone();
        assert!((cloned.throughput_ratio - 0.96).abs() < 0.001);
        assert!((cloned.memory_ratio - 1.1).abs() < 0.001);
    }

    #[test]
    fn test_parity_comparison_debug() {
        let comparison = AprParityComparison {
            throughput_ratio: 0.96,
            memory_ratio: 1.1,
            parity_threshold_pct: 95.0,
        };
        let debug_str = format!("{:?}", comparison);
        assert!(debug_str.contains("AprParityComparison"));
        assert!(debug_str.contains("throughput_ratio"));
    }

    #[test]
    fn test_is_parity_pass() {
        let comparison = AprParityComparison {
            throughput_ratio: 0.96, // 96% >= 95%
            memory_ratio: 1.0,
            parity_threshold_pct: 95.0,
        };
        assert!(comparison.is_parity());
    }

    #[test]
    fn test_is_parity_exact_threshold() {
        let comparison = AprParityComparison {
            throughput_ratio: 0.95, // 95% == 95%
            memory_ratio: 1.0,
            parity_threshold_pct: 95.0,
        };
        assert!(comparison.is_parity());
    }

    #[test]
    fn test_is_parity_fail() {
        let comparison = AprParityComparison {
            throughput_ratio: 0.94, // 94% < 95%
            memory_ratio: 1.0,
            parity_threshold_pct: 95.0,
        };
        assert!(!comparison.is_parity());
    }

    #[test]
    fn test_is_parity_exceed() {
        let comparison = AprParityComparison {
            throughput_ratio: 1.1, // 110% > 95%
            memory_ratio: 0.9,
            parity_threshold_pct: 95.0,
        };
        assert!(comparison.is_parity());
    }

    #[test]
    fn test_is_parity_custom_threshold() {
        let comparison = AprParityComparison {
            throughput_ratio: 0.85,
            memory_ratio: 1.0,
            parity_threshold_pct: 80.0, // 85% >= 80%
        };
        assert!(comparison.is_parity());
    }

    #[test]
    fn test_is_parity_zero_threshold() {
        let comparison = AprParityComparison {
            throughput_ratio: 0.01,
            memory_ratio: 1.0,
            parity_threshold_pct: 0.0, // any positive passes
        };
        assert!(comparison.is_parity());
    }
}
