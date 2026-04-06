
// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_basic_profiling() {
        let mut profiler = BrickProfiler::new();

        profiler.start("test_op");
        thread::sleep(Duration::from_micros(100));
        profiler.stop("test_op");

        let report = profiler.report();
        assert!(report.operations.contains_key("test_op"));

        let stats = &report.operations["test_op"];
        assert_eq!(stats.count, 1);
        assert!(stats.avg_us >= 100.0); // At least 100µs
    }

    #[test]
    fn test_multiple_measurements() {
        let mut profiler = BrickProfiler::new();

        for _ in 0..5 {
            profiler.start("multi_op");
            thread::sleep(Duration::from_micros(50));
            profiler.stop("multi_op");
        }

        let report = profiler.report();
        let stats = &report.operations["multi_op"];
        assert_eq!(stats.count, 5);
        assert!(stats.min_us <= stats.avg_us);
        assert!(stats.avg_us <= stats.max_us);
        assert_eq!(stats.per_layer.len(), 5);
    }

    #[test]
    fn test_measure_closure() {
        let mut profiler = BrickProfiler::new();

        let result = profiler.measure("closure_op", || {
            thread::sleep(Duration::from_micros(100));
            42
        });

        assert_eq!(result, 42);

        let report = profiler.report();
        assert!(report.operations.contains_key("closure_op"));
    }

    #[test]
    fn test_disabled_profiler() {
        let mut profiler = BrickProfiler::disabled();
        assert!(!profiler.is_enabled());

        profiler.start("disabled_op");
        profiler.stop("disabled_op");

        let report = profiler.report();
        assert!(!report.is_real_data);
        assert!(report.operations.is_empty());
    }

    #[test]
    fn test_throughput_calculation() {
        let mut profiler = BrickProfiler::new();
        profiler.set_tokens(100);

        profiler.start_inference();
        thread::sleep(Duration::from_millis(10));
        profiler.stop_inference();

        let report = profiler.report();
        assert_eq!(report.tokens_processed, 100);
        // Should be roughly 10,000 tok/s (100 tokens in 10ms)
        assert!(report.throughput_tok_s > 1000.0);
    }

    #[test]
    fn test_percentage_breakdown() {
        let mut profiler = BrickProfiler::new();

        profiler.record("op1", 500.0); // 50%
        profiler.record("op2", 300.0); // 30%
        profiler.record("op3", 200.0); // 20%

        let report = profiler.report();
        let breakdown = report.percentage_breakdown();

        assert!((breakdown["op1"] - 50.0).abs() < 0.1);
        assert!((breakdown["op2"] - 30.0).abs() < 0.1);
        assert!((breakdown["op3"] - 20.0).abs() < 0.1);
    }

    #[test]
    fn test_sorted_by_time() {
        let mut profiler = BrickProfiler::new();

        profiler.record("small", 100.0);
        profiler.record("large", 500.0);
        profiler.record("medium", 300.0);

        let report = profiler.report();
        let sorted = report.sorted_by_time();

        assert_eq!(sorted[0].0, "large");
        assert_eq!(sorted[1].0, "medium");
        assert_eq!(sorted[2].0, "small");
    }

    #[test]
    fn test_hottest() {
        let mut profiler = BrickProfiler::new();

        profiler.record("op1", 100.0);
        profiler.record("op2", 500.0);
        profiler.record("op3", 200.0);

        let report = profiler.report();
        let (name, _) = report.hottest().expect("hottest");

        assert_eq!(name, "op2");
    }

    #[test]
    fn test_clear() {
        let mut profiler = BrickProfiler::new();

        profiler.record("op1", 100.0);
        assert_eq!(profiler.stats().len(), 1);

        profiler.clear();
        assert_eq!(profiler.stats().len(), 0);
    }

    #[test]
    fn test_record_direct() {
        let mut profiler = BrickProfiler::new();

        profiler.record("external", 123.45);

        let report = profiler.report();
        let stats = &report.operations["external"];

        assert!((stats.avg_us - 123.45).abs() < 0.01);
    }
}
