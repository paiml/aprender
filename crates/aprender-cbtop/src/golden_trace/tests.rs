use super::*;
use tempfile::TempDir;

#[test]
fn test_trace_metrics_default() {
    let metrics = TraceMetrics::default();
    assert!(metrics.is_valid());
    assert_eq!(metrics.total_time_us, 0.0);
}

#[test]
fn test_golden_trace_creation() {
    let metrics = TraceMetrics::new()
        .total_time_us(1000.0)
        .p50_latency_us(50.0)
        .throughput(10000.0);

    let trace = GoldenTrace::new("test", metrics);
    assert_eq!(trace.name, "test");
    assert_eq!(trace.version, "1.0");
    assert!(!trace.hash.is_empty());
}

#[test]
fn test_hash_verification() {
    let metrics = TraceMetrics::new().total_time_us(1000.0);
    let trace = GoldenTrace::new("test", metrics);
    assert!(trace.verify_hash());
}

#[test]
fn test_comparison_no_regression() {
    let golden = GoldenTrace::new(
        "baseline",
        TraceMetrics::new()
            .total_time_us(1000.0)
            .p50_latency_us(50.0)
            .throughput(10000.0),
    );

    let current = TraceMetrics::new()
        .total_time_us(1050.0) // 5% slower
        .p50_latency_us(52.0)
        .throughput(9800.0);

    let comparator = GoldenComparator::new().with_threshold(10.0);
    let result = comparator.compare(&current, &golden).unwrap();

    assert!(!result.is_regression);
}

#[test]
fn test_comparison_regression() {
    let golden = GoldenTrace::new(
        "baseline",
        TraceMetrics::new()
            .total_time_us(1000.0)
            .p50_latency_us(50.0)
            .throughput(10000.0),
    );

    let current = TraceMetrics::new()
        .total_time_us(1200.0) // 20% slower - regression!
        .p50_latency_us(60.0)
        .throughput(8000.0);

    let comparator = GoldenComparator::new().with_threshold(10.0);
    let result = comparator.compare(&current, &golden).unwrap();

    assert!(result.is_regression);
    assert!(result.time_delta_percent > 10.0);
}

#[test]
fn test_syscall_breakdown() {
    let baseline = SyscallBreakdown {
        read_count: 100,
        write_count: 50,
        mmap_count: 10,
        futex_count: 20,
        other_count: 5,
    };

    let current = SyscallBreakdown {
        read_count: 110, // 10% more
        write_count: 50,
        mmap_count: 15, // 50% more
        futex_count: 20,
        other_count: 5,
    };

    let delta = current.percentage_diff(&baseline);
    assert!((delta.read_delta - 10.0).abs() < 0.1);
    assert!((delta.mmap_delta - 50.0).abs() < 0.1);
}

#[test]
fn test_manager_save_load() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = GoldenTraceManager::new(temp_dir.path().to_path_buf());

    let metrics = TraceMetrics::new()
        .total_time_us(1000.0)
        .p50_latency_us(50.0);

    manager.capture_golden("test_trace", metrics).unwrap();

    let loaded = manager.load_golden("test_trace").unwrap();
    assert_eq!(loaded.name, "test_trace");
    assert_eq!(loaded.metrics.total_time_us, 1000.0);
}

#[test]
fn test_manager_list() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = GoldenTraceManager::new(temp_dir.path().to_path_buf());

    manager
        .capture_golden("trace_a", TraceMetrics::new())
        .unwrap();
    manager
        .capture_golden("trace_b", TraceMetrics::new())
        .unwrap();

    let names = manager.list_goldens().unwrap();
    assert_eq!(names.len(), 2);
    assert!(names.contains(&"trace_a".to_string()));
    assert!(names.contains(&"trace_b".to_string()));
}

#[test]
fn test_json_serialization() {
    let metrics = TraceMetrics::new().total_time_us(1000.0);
    let trace = GoldenTrace::new("json_test", metrics);

    let json = trace.to_json().unwrap();
    assert!(json.contains("json_test"));

    let parsed = GoldenTrace::from_json(&json).unwrap();
    assert_eq!(parsed.name, "json_test");
}

#[test]
fn test_toml_serialization() {
    let metrics = TraceMetrics::new().total_time_us(1000.0);
    let trace = GoldenTrace::new("toml_test", metrics);

    let toml = trace.to_toml().unwrap();
    assert!(toml.contains("toml_test"));

    let parsed = GoldenTrace::from_toml(&toml).unwrap();
    assert_eq!(parsed.name, "toml_test");
}
