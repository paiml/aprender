//! golden_trace_f1211 - Part 1

use cbtop::{
    GoldenComparator, GoldenTrace, TraceMetrics, TraceSyscallBreakdown as SyscallBreakdown,
};

// =============================================================================
// F1211: Golden Trace Capture Tests
// =============================================================================

/// F1211.1: Golden trace captures metrics
#[test]
fn f1211_capture_metrics() {
    let metrics = TraceMetrics::new()
        .total_time_us(1234.5)
        .p50_latency_us(50.0)
        .p99_latency_us(100.0)
        .throughput(10000.0)
        .peak_memory_bytes(1024 * 1024);

    let trace = GoldenTrace::new("capture_test", metrics);

    // All fields should be populated
    assert_eq!(trace.name, "capture_test");
    assert_eq!(trace.metrics.total_time_us, 1234.5);
    assert_eq!(trace.metrics.p50_latency_us, 50.0);
    assert_eq!(trace.metrics.p99_latency_us, 100.0);
    assert_eq!(trace.metrics.throughput, 10000.0);
    assert_eq!(trace.metrics.peak_memory_bytes, 1024 * 1024);
    assert!(!trace.hash.is_empty());
    assert!(trace.timestamp > 0);
}

/// F1211.2: Capture with syscall breakdown
#[test]
fn f1211_capture_with_syscalls() {
    let syscalls = SyscallBreakdown {
        read_count: 100,
        write_count: 50,
        mmap_count: 10,
        futex_count: 25,
        other_count: 15,
    };

    let metrics = TraceMetrics::new().total_time_us(1000.0).syscalls(syscalls);

    let trace = GoldenTrace::new("syscall_test", metrics);

    assert_eq!(trace.metrics.syscalls.read_count, 100);
    assert_eq!(trace.metrics.syscalls.write_count, 50);
    assert_eq!(trace.metrics.syscalls.total(), 200);
}

// =============================================================================
// F1212: Trace Comparison Delta Tests
// =============================================================================

/// F1212.1: Trace comparison calculates delta correctly
#[test]
fn f1212_delta_calculation() {
    let golden = GoldenTrace::new(
        "baseline",
        TraceMetrics::new()
            .total_time_us(1000.0)
            .p50_latency_us(50.0)
            .throughput(10000.0),
    );

    let current = TraceMetrics::new()
        .total_time_us(1100.0) // 10% slower
        .p50_latency_us(55.0) // 10% slower
        .throughput(9000.0); // 10% lower

    let comparator = GoldenComparator::new();
    let result = comparator.compare(&current, &golden).unwrap();

    // Check delta calculations
    assert!((result.time_delta_percent - 10.0).abs() < 0.1);
    assert!((result.p50_delta_percent - 10.0).abs() < 0.1);
    assert!((result.throughput_delta_percent - (-10.0)).abs() < 0.1);
}

/// F1212.2: Zero baseline handled correctly
#[test]
fn f1212_zero_baseline() {
    let golden = GoldenTrace::new(
        "zero_baseline",
        TraceMetrics::new().total_time_us(0.0).throughput(0.0),
    );

    let current = TraceMetrics::new().total_time_us(100.0).throughput(1000.0);

    let comparator = GoldenComparator::new();
    let result = comparator.compare(&current, &golden).unwrap();

    // New values from zero should be 100% delta
    assert!(result.time_delta_percent > 0.0);
}

// =============================================================================
// F1213: Regression Detection Tests
// =============================================================================

/// F1213.1: Regression detected at >10% threshold
#[test]
fn f1213_regression_detection() {
    let golden = GoldenTrace::new(
        "baseline",
        TraceMetrics::new()
            .total_time_us(1000.0)
            .p50_latency_us(50.0)
            .throughput(10000.0),
    );

    // Current is 15% worse
    let current = TraceMetrics::new()
        .total_time_us(1150.0) // 15% slower
        .p50_latency_us(57.5)
        .throughput(8500.0);

    let comparator = GoldenComparator::new().with_threshold(10.0);
    let result = comparator.compare(&current, &golden).unwrap();

    assert!(result.is_regression);
    assert!(result.time_delta_percent > 10.0);
}

/// F1213.2: No regression below threshold
#[test]
fn f1213_no_regression_below_threshold() {
    let golden = GoldenTrace::new(
        "baseline",
        TraceMetrics::new()
            .total_time_us(1000.0)
            .p50_latency_us(50.0)
            .throughput(10000.0),
    );

    // Current is 5% worse (within 10% threshold)
    let current = TraceMetrics::new()
        .total_time_us(1050.0)
        .p50_latency_us(52.5)
        .throughput(9500.0);

    let comparator = GoldenComparator::new().with_threshold(10.0);
    let result = comparator.compare(&current, &golden).unwrap();

    assert!(!result.is_regression);
}

/// F1213.3: Custom threshold works
#[test]
fn f1213_custom_threshold() {
    let golden = GoldenTrace::new("baseline", TraceMetrics::new().total_time_us(1000.0));

    let current = TraceMetrics::new().total_time_us(1060.0); // 6% slower

    // 5% threshold - should be regression
    let comparator = GoldenComparator::new().with_threshold(5.0);
    let result = comparator.compare(&current, &golden).unwrap();
    assert!(result.is_regression);

    // 10% threshold - should not be regression
    let comparator = GoldenComparator::new().with_threshold(10.0);
    let result = comparator.compare(&current, &golden).unwrap();
    assert!(!result.is_regression);
}

// =============================================================================
// F1214: Golden Trace Versioning Tests
// =============================================================================

/// F1214.1: Golden trace versioned
#[test]
fn f1214_version_stored() {
    let trace = GoldenTrace::with_version(
        "versioned",
        "2.0",
        TraceMetrics::new().total_time_us(1000.0),
    );

    assert_eq!(trace.version, "2.0");
}

/// F1214.2: Git commit stored
#[test]
fn f1214_git_commit_stored() {
    let trace = GoldenTrace::new("git_test", TraceMetrics::new()).git_commit("abc123def456");

    assert_eq!(trace.git_commit, Some("abc123def456".to_string()));
}

// =============================================================================
// F1215: Trace Timestamps Tests
// =============================================================================

/// F1215.1: Trace timestamps preserved
#[test]
fn f1215_timestamps_preserved() {
    let trace1 = GoldenTrace::new("first", TraceMetrics::new());
    std::thread::sleep(std::time::Duration::from_millis(10));
    let trace2 = GoldenTrace::new("second", TraceMetrics::new());

    // Timestamps should be chronological
    assert!(trace2.timestamp >= trace1.timestamp);
}

/// F1215.2: Timestamp through serialization
#[test]
fn f1215_timestamp_serialization() {
    let original = GoldenTrace::new("timestamp_test", TraceMetrics::new());
    let timestamp = original.timestamp;

    let json = original.to_json().unwrap();
    let parsed = GoldenTrace::from_json(&json).unwrap();

    assert_eq!(parsed.timestamp, timestamp);
}

// =============================================================================
// F1216: Breakdown Delta Tests
// =============================================================================

/// F1216.1: Per-syscall diff calculated
#[test]
fn f1216_syscall_diff() {
    let baseline = SyscallBreakdown {
        read_count: 100,
        write_count: 50,
        mmap_count: 10,
        futex_count: 20,
        other_count: 5,
    };

    let current = SyscallBreakdown {
        read_count: 120, // 20% increase
        write_count: 50, // unchanged
        mmap_count: 15,  // 50% increase
        futex_count: 22, // 10% increase
        other_count: 5,  // unchanged
    };

    let delta = current.percentage_diff(&baseline);

    assert!((delta.read_delta - 20.0).abs() < 0.1);
    assert!(delta.write_delta.abs() < 0.1);
    assert!((delta.mmap_delta - 50.0).abs() < 0.1);
    assert!((delta.futex_delta - 10.0).abs() < 0.1);
}

/// F1216.2: Max delta detection
#[test]
fn f1216_max_delta() {
    let baseline = SyscallBreakdown {
        read_count: 100,
        write_count: 100,
        mmap_count: 100,
        futex_count: 100,
        other_count: 100,
    };

    let current = SyscallBreakdown {
        read_count: 100,
        write_count: 100,
        mmap_count: 200, // 100% increase - max delta
        futex_count: 100,
        other_count: 100,
    };

    let delta = current.percentage_diff(&baseline);
    assert!((delta.max_delta() - 100.0).abs() < 0.1);
}
