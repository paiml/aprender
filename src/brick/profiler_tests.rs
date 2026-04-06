//! BrickProfiler Tests - Coverage for PMAT-112
//!
//! Tests for the BrickProfiler real-time telemetry system.
//! Refs PMAT-802: Protocol T-COV-95

#[cfg(test)]
mod tests {
    use super::super::profiler::*;
    use std::thread;
    use std::time::Duration;

    // =========================================================================
    // OpStats Tests
    // =========================================================================

    #[test]
    fn test_op_stats_default() {
        let stats = OpStats::default();
        assert!((stats.min_us - 0.0).abs() < 0.001);
        assert!((stats.max_us - 0.0).abs() < 0.001);
        assert!((stats.avg_us - 0.0).abs() < 0.001);
        assert!((stats.total_us - 0.0).abs() < 0.001);
        assert_eq!(stats.count, 0);
        assert!(stats.per_layer.is_empty());
    }

    #[test]
    fn test_op_stats_clone() {
        let mut stats = OpStats::default();
        stats.min_us = 1.0;
        stats.max_us = 10.0;
        stats.avg_us = 5.0;
        stats.total_us = 50.0;
        stats.count = 10;
        stats.per_layer = vec![1.0, 2.0, 3.0];

        let cloned = stats.clone();
        assert!((cloned.min_us - 1.0).abs() < 0.001);
        assert!((cloned.max_us - 10.0).abs() < 0.001);
        assert_eq!(cloned.count, 10);
        assert_eq!(cloned.per_layer, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_op_stats_debug() {
        let stats = OpStats::default();
        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("OpStats"));
    }

    // =========================================================================
    // ProfileReport Tests
    // =========================================================================

    #[test]
    fn test_profile_report_hottest_empty() {
        let report = ProfileReport {
            operations: std::collections::HashMap::new(),
            total_inference_us: 0.0,
            tokens_processed: 0,
            num_layers: 0,
            throughput_tok_s: 0.0,
            is_real_data: false,
        };
        assert!(report.hottest().is_none());
    }

    #[test]
    fn test_profile_report_hottest_single() {
        let mut ops = std::collections::HashMap::new();
        let mut stats = OpStats::default();
        stats.total_us = 100.0;
        ops.insert("attention".to_string(), stats);

        let report = ProfileReport {
            operations: ops,
            total_inference_us: 100.0,
            tokens_processed: 1,
            num_layers: 1,
            throughput_tok_s: 10000.0,
            is_real_data: true,
        };

        let (name, _) = report.hottest().unwrap();
        assert_eq!(name, "attention");
    }

    #[test]
    fn test_profile_report_hottest_multiple() {
        let mut ops = std::collections::HashMap::new();

        let mut stats1 = OpStats::default();
        stats1.total_us = 50.0;
        ops.insert("attention".to_string(), stats1);

        let mut stats2 = OpStats::default();
        stats2.total_us = 150.0;
        ops.insert("mlp".to_string(), stats2);

        let report = ProfileReport {
            operations: ops,
            total_inference_us: 200.0,
            tokens_processed: 1,
            num_layers: 1,
            throughput_tok_s: 5000.0,
            is_real_data: true,
        };

        let (name, stats) = report.hottest().unwrap();
        assert_eq!(name, "mlp");
        assert!((stats.total_us - 150.0).abs() < 0.001);
    }

    #[test]
    fn test_profile_report_sorted_by_time() {
        let mut ops = std::collections::HashMap::new();

        let mut stats1 = OpStats::default();
        stats1.total_us = 50.0;
        ops.insert("attention".to_string(), stats1);

        let mut stats2 = OpStats::default();
        stats2.total_us = 150.0;
        ops.insert("mlp".to_string(), stats2);

        let mut stats3 = OpStats::default();
        stats3.total_us = 100.0;
        ops.insert("norm".to_string(), stats3);

        let report = ProfileReport {
            operations: ops,
            total_inference_us: 300.0,
            tokens_processed: 1,
            num_layers: 1,
            throughput_tok_s: 3333.0,
            is_real_data: true,
        };

        let sorted = report.sorted_by_time();
        assert_eq!(sorted.len(), 3);
        assert_eq!(sorted[0].0, "mlp");
        assert_eq!(sorted[1].0, "norm");
        assert_eq!(sorted[2].0, "attention");
    }

    #[test]
    fn test_profile_report_percentage_breakdown_empty() {
        let report = ProfileReport {
            operations: std::collections::HashMap::new(),
            total_inference_us: 0.0,
            tokens_processed: 0,
            num_layers: 0,
            throughput_tok_s: 0.0,
            is_real_data: false,
        };

        let breakdown = report.percentage_breakdown();
        assert!(breakdown.is_empty());
    }

    #[test]
    fn test_profile_report_percentage_breakdown() {
        let mut ops = std::collections::HashMap::new();

        let mut stats1 = OpStats::default();
        stats1.total_us = 50.0;
        ops.insert("attention".to_string(), stats1);

        let mut stats2 = OpStats::default();
        stats2.total_us = 150.0;
        ops.insert("mlp".to_string(), stats2);

        let report = ProfileReport {
            operations: ops,
            total_inference_us: 200.0,
            tokens_processed: 1,
            num_layers: 1,
            throughput_tok_s: 5000.0,
            is_real_data: true,
        };

        let breakdown = report.percentage_breakdown();
        assert!(breakdown.contains_key("attention"));
        assert!(breakdown.contains_key("mlp"));
        assert!((breakdown["attention"] - 25.0).abs() < 0.001);
        assert!((breakdown["mlp"] - 75.0).abs() < 0.001);
    }

    #[test]
    fn test_profile_report_clone() {
        let report = ProfileReport {
            operations: std::collections::HashMap::new(),
            total_inference_us: 100.0,
            tokens_processed: 5,
            num_layers: 12,
            throughput_tok_s: 50000.0,
            is_real_data: true,
        };

        let cloned = report.clone();
        assert!((cloned.total_inference_us - 100.0).abs() < 0.001);
        assert_eq!(cloned.tokens_processed, 5);
        assert_eq!(cloned.num_layers, 12);
        assert!(cloned.is_real_data);
    }

    #[test]
    fn test_profile_report_debug() {
        let report = ProfileReport {
            operations: std::collections::HashMap::new(),
            total_inference_us: 0.0,
            tokens_processed: 0,
            num_layers: 0,
            throughput_tok_s: 0.0,
            is_real_data: false,
        };
        let debug_str = format!("{:?}", report);
        assert!(debug_str.contains("ProfileReport"));
    }

    // =========================================================================
    // BrickProfiler Tests
    // =========================================================================

    #[test]
    fn test_profiler_new() {
        let profiler = BrickProfiler::new();
        assert!(profiler.is_enabled());
        assert!(profiler.stats().is_empty());
    }

    #[test]
    fn test_profiler_default() {
        let profiler = BrickProfiler::default();
        assert!(profiler.is_enabled());
    }

    #[test]
    fn test_profiler_disabled() {
        let profiler = BrickProfiler::disabled();
        assert!(!profiler.is_enabled());
    }

    #[test]
    fn test_profiler_set_tokens() {
        let mut profiler = BrickProfiler::new();
        profiler.set_tokens(10);

        let report = profiler.report();
        assert_eq!(report.tokens_processed, 10);
    }

    #[test]
    fn test_profiler_set_num_layers() {
        let mut profiler = BrickProfiler::new();
        profiler.set_num_layers(32);

        let report = profiler.report();
        assert_eq!(report.num_layers, 32);
    }

    #[test]
    fn test_profiler_set_current_layer() {
        let mut profiler = BrickProfiler::new();
        profiler.set_current_layer(5);
        // Just verify it doesn't panic
    }

    #[test]
    fn test_profiler_start_stop() {
        let mut profiler = BrickProfiler::new();

        profiler.start("test_op");
        thread::sleep(Duration::from_micros(100));
        profiler.stop("test_op");

        let stats = profiler.stats();
        assert!(stats.contains_key("test_op"));
        assert!(stats["test_op"].count == 1);
        assert!(stats["test_op"].total_us > 0.0);
    }

    #[test]
    fn test_profiler_start_stop_multiple() {
        let mut profiler = BrickProfiler::new();

        for _ in 0..5 {
            profiler.start("repeated_op");
            thread::sleep(Duration::from_micros(10));
            profiler.stop("repeated_op");
        }

        let stats = profiler.stats();
        assert_eq!(stats["repeated_op"].count, 5);
    }

    #[test]
    fn test_profiler_start_stop_disabled() {
        let mut profiler = BrickProfiler::disabled();

        profiler.start("test_op");
        profiler.stop("test_op");

        // Should be empty because profiler is disabled
        assert!(profiler.stats().is_empty());
    }

    #[test]
    fn test_profiler_start_inference() {
        let mut profiler = BrickProfiler::new();

        profiler.start_inference();
        thread::sleep(Duration::from_micros(100));
        profiler.stop_inference();

        let report = profiler.report();
        assert!(report.total_inference_us > 0.0);
    }

    #[test]
    fn test_profiler_inference_disabled() {
        let mut profiler = BrickProfiler::disabled();

        profiler.start_inference();
        profiler.stop_inference();

        // Should still work but with no real timing
        let report = profiler.report();
        assert!(!report.is_real_data);
    }

    #[test]
    fn test_profiler_record() {
        let mut profiler = BrickProfiler::new();

        profiler.record("external_op", 100.0);
        profiler.record("external_op", 200.0);

        let stats = profiler.stats();
        assert_eq!(stats["external_op"].count, 2);
        assert!((stats["external_op"].total_us - 300.0).abs() < 0.001);
        assert!((stats["external_op"].min_us - 100.0).abs() < 0.001);
        assert!((stats["external_op"].max_us - 200.0).abs() < 0.001);
    }

    #[test]
    fn test_profiler_record_disabled() {
        let mut profiler = BrickProfiler::disabled();

        profiler.record("external_op", 100.0);

        assert!(profiler.stats().is_empty());
    }

    #[test]
    fn test_profiler_measure() {
        let mut profiler = BrickProfiler::new();

        let result = profiler.measure("compute", || {
            thread::sleep(Duration::from_micros(50));
            42
        });

        assert_eq!(result, 42);
        let stats = profiler.stats();
        assert!(stats.contains_key("compute"));
        assert_eq!(stats["compute"].count, 1);
    }

    #[test]
    fn test_profiler_measure_disabled() {
        let mut profiler = BrickProfiler::disabled();

        let result = profiler.measure("compute", || 123);

        assert_eq!(result, 123);
        assert!(profiler.stats().is_empty());
    }

    #[test]
    fn test_profiler_clear() {
        let mut profiler = BrickProfiler::new();

        profiler.set_tokens(10);
        profiler.record("op1", 100.0);
        profiler.record("op2", 200.0);

        profiler.clear();

        assert!(profiler.stats().is_empty());
        let report = profiler.report();
        assert_eq!(report.tokens_processed, 0);
    }

    #[test]
    fn test_profiler_report_empty() {
        let profiler = BrickProfiler::new();

        let report = profiler.report();
        assert!(report.operations.is_empty());
        assert_eq!(report.tokens_processed, 0);
        assert_eq!(report.num_layers, 0);
        assert!((report.throughput_tok_s - 0.0).abs() < 0.001);
        assert!(!report.is_real_data);
    }

    #[test]
    fn test_profiler_report_with_data() {
        let mut profiler = BrickProfiler::new();

        profiler.set_tokens(100);
        profiler.set_num_layers(12);
        profiler.start_inference();
        profiler.record("attention", 1000.0); // 1ms
        profiler.record("mlp", 2000.0); // 2ms
        profiler.stop_inference();

        let report = profiler.report();
        assert_eq!(report.operations.len(), 2);
        assert!(report.is_real_data);
        assert!(report.throughput_tok_s > 0.0);
    }

    #[test]
    fn test_profiler_throughput_calculation() {
        let mut profiler = BrickProfiler::new();

        profiler.set_tokens(100);
        // 10ms total = 10,000 us
        profiler.record("op", 10000.0);

        let report = profiler.report();
        // throughput = 100 tokens / 0.01 seconds = 10,000 tok/s
        assert!((report.throughput_tok_s - 10000.0).abs() < 1.0);
    }

    #[test]
    fn test_profiler_debug() {
        let profiler = BrickProfiler::new();
        let debug_str = format!("{:?}", profiler);
        assert!(debug_str.contains("BrickProfiler"));
    }

    #[test]
    fn test_profiler_stop_without_start() {
        let mut profiler = BrickProfiler::new();

        // Stop without start should be a no-op
        profiler.stop("nonexistent");

        assert!(profiler.stats().is_empty());
    }

    #[test]
    fn test_profiler_multiple_operations() {
        let mut profiler = BrickProfiler::new();

        profiler.start("op1");
        profiler.start("op2");
        thread::sleep(Duration::from_micros(50));
        profiler.stop("op1");
        profiler.stop("op2");

        let stats = profiler.stats();
        assert!(stats.contains_key("op1"));
        assert!(stats.contains_key("op2"));
    }
}
