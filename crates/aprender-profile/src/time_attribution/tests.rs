// Integration tests for time-weighted attribution
//
// Toyota Way Principle: Genchi Genbutsu (Go and See)
// - Test with realistic transpiler syscall patterns
// - Validate hotspot identification accuracy
// - Ensure actionable explanations

use super::*;
use crate::cluster::ClusterRegistry;
use crate::unified_trace::SyscallSpan;
use std::borrow::Cow;
use std::time::Duration;

fn make_span(
    name: &'static str,
    duration_nanos: u64,
    args: Vec<(Cow<'static, str>, String)>,
) -> SyscallSpan {
    SyscallSpan {
        span_id: 1,
        parent_span_id: 0,
        name: Cow::Borrowed(name),
        args,
        return_value: 0,
        timestamp_nanos: 0,
        duration_nanos,
        errno: None,
    }
}

/// Test realistic transpiler pattern: FileIO dominates
#[test]
fn test_transpiler_file_io_dominant() {
    let registry = ClusterRegistry::default_transpiler_clusters().expect("test");

    // Realistic: transpiler spends 70% time in I/O, 20% memory, 10% misc
    let spans = vec![
        // File I/O (70% of time)
        make_span("open", 1_000_000, vec![]),   // 1ms
        make_span("read", 50_000_000, vec![]),  // 50ms
        make_span("write", 18_000_000, vec![]), // 18ms
        make_span("close", 1_000_000, vec![]),  // 1ms
        // Memory allocation (20% of time)
        make_span("mmap", 10_000_000, vec![]),  // 10ms
        make_span("munmap", 5_000_000, vec![]), // 5ms
        make_span("brk", 5_000_000, vec![]),    // 5ms
        // Dynamic linking (10% of time)
        make_span("openat", 5_000_000, vec![]), // 5ms (ld.so)
        make_span("mmap", 5_000_000, vec![]),   // 5ms (ld.so)
    ];

    let attributions = calculate_time_attribution(&spans, &registry);

    // FileIO should be the top cluster
    assert_eq!(attributions[0].cluster, "FileIO");
    // FileIO: open(1ms) + read(50ms) + write(18ms) + close(1ms) = 70ms total
    // But mmap appears twice (10ms + 5ms in DynamicLinking), so need to account for that
    // Total should still be dominated by FileIO
    assert!(
        attributions[0].percentage > 60.0,
        "FileIO should dominate with >60%, got {}",
        attributions[0].percentage
    );

    let hotspots = identify_hotspots(&attributions);

    // Should identify FileIO and MemoryAllocation as hotspots
    assert!(hotspots.iter().any(|h| h.cluster == "FileIO"));
    assert!(hotspots.iter().any(|h| h.cluster == "MemoryAllocation"));

    // FileIO hotspot should be marked as expected
    let file_io_hotspot = hotspots.iter().find(|h| h.cluster == "FileIO").expect("test");
    assert!(file_io_hotspot.is_expected);
}

/// Test anomaly: Unexpected networking (telemetry leak)
#[test]
fn test_unexpected_networking_hotspot() {
    let registry = ClusterRegistry::default_transpiler_clusters().expect("test");

    let spans = vec![
        // Normal I/O (40%)
        make_span("read", 40_000_000, vec![]),
        // UNEXPECTED: Networking (60%)
        make_span("socket", 10_000_000, vec![]),
        make_span("connect", 20_000_000, vec![]),
        make_span("send", 30_000_000, vec![]),
    ];

    let attributions = calculate_time_attribution(&spans, &registry);
    let hotspots = identify_hotspots(&attributions);

    // Should identify Networking hotspot
    let networking = hotspots.iter().find(|h| h.cluster == "Networking");
    assert!(networking.is_some());

    let networking = networking.expect("test");
    assert!(!networking.is_expected); // NOT expected for transpilers
    assert!(networking.explanation.contains("UNEXPECTED"));
}

/// Test anomaly: Unexpected GPU usage
#[test]
fn test_unexpected_gpu_hotspot() {
    let registry = ClusterRegistry::default_transpiler_clusters().expect("test");

    let spans = vec![
        make_span("read", 30_000_000, vec![]),
        // UNEXPECTED: GPU operations (70%)
        make_span(
            "ioctl",
            70_000_000,
            vec![(Cow::Borrowed("fd_path"), "/dev/nvidia0".to_string())],
        ),
    ];

    let attributions = calculate_time_attribution(&spans, &registry);
    let hotspots = identify_hotspots(&attributions);

    // Should identify GPU hotspot
    let gpu = hotspots.iter().find(|h| h.cluster == "GPU");
    assert!(gpu.is_some());

    let gpu = gpu.expect("test");
    assert!(!gpu.is_expected);
    assert!(gpu.explanation.contains("UNEXPECTED"));
}

/// Test blocking I/O dominates over frequent fast calls
#[test]
fn test_blocking_io_dominates_fast_calls() {
    let registry = ClusterRegistry::default_transpiler_clusters().expect("test");

    let spans = vec![
        // 1000 fast mmap calls (1ms total)
        make_span("mmap", 1, vec![]),
        make_span("mmap", 1, vec![]),
        make_span("mmap", 1, vec![]),
        // ... (imagine 997 more)
        // 1 blocking read (99ms)
        make_span("read", 99_000_000, vec![]),
    ];

    let attributions = calculate_time_attribution(&spans, &registry);

    // FileIO should dominate despite only 1 call
    let file_io = attributions.iter().find(|a| a.cluster == "FileIO").expect("test");
    assert!(file_io.percentage > 99.0);
    assert_eq!(file_io.call_count, 1);
}

/// Test hotspot threshold (5%)
#[test]
fn test_hotspot_threshold_filtering() {
    let registry = ClusterRegistry::default_transpiler_clusters().expect("test");

    let spans = vec![
        make_span("read", 60_000_000, vec![]), // FileIO: 60%
        make_span("mmap", 30_000_000, vec![]), // MemoryAllocation: 30%
        make_span("brk", 6_000_000, vec![]),   // MemoryAllocation: 6% (total 36%)
        make_span("close", 4_000_000, vec![]), // FileIO: 4% (total 64%)
    ];

    let attributions = calculate_time_attribution(&spans, &registry);
    let hotspots = identify_hotspots(&attributions);

    // Should identify FileIO (64%) and MemoryAllocation (36%)
    assert_eq!(hotspots.len(), 2);
    assert!(hotspots.iter().all(|h| h.percentage > 5.0));
}

/// Test empty trace handling
#[test]
fn test_empty_trace() {
    let registry = ClusterRegistry::default_transpiler_clusters().expect("test");
    let spans: Vec<SyscallSpan> = vec![];

    let attributions = calculate_time_attribution(&spans, &registry);
    assert!(attributions.is_empty());

    let hotspots = identify_hotspots(&attributions);
    assert!(hotspots.is_empty());
}

/// Test all-zero duration handling
#[test]
fn test_zero_duration_trace() {
    let registry = ClusterRegistry::default_transpiler_clusters().expect("test");
    let spans = vec![make_span("read", 0, vec![]), make_span("write", 0, vec![])];

    let attributions = calculate_time_attribution(&spans, &registry);
    assert!(attributions.is_empty()); // Should handle gracefully

    let hotspots = identify_hotspots(&attributions);
    assert!(hotspots.is_empty());
}

/// Test hotspot report formatting
#[test]
fn test_hotspot_report_formatting() {
    let hotspot = Hotspot {
        cluster: "FileIO".to_string(),
        time: Duration::from_secs(1),
        percentage: 65.4,
        explanation: "File I/O dominates execution".to_string(),
        is_expected: true,
    };

    let report = hotspot.to_report_string();

    // Should contain key information
    assert!(report.contains("FileIO"));
    assert!(report.contains("65.4%"));
    assert!(report.contains("File I/O dominates"));
    assert!(report.contains("✓")); // Expected marker
}

/// Test avg_per_call calculation
#[test]
fn test_avg_per_call_accuracy() {
    let registry = ClusterRegistry::default_transpiler_clusters().expect("test");

    let spans = vec![
        make_span("read", 10_000_000, vec![]), // 10ms
        make_span("read", 20_000_000, vec![]), // 20ms
        make_span("read", 30_000_000, vec![]), // 30ms
    ];

    let attributions = calculate_time_attribution(&spans, &registry);
    let file_io = attributions.iter().find(|a| a.cluster == "FileIO").expect("test");

    // Average: (10 + 20 + 30) / 3 = 20ms
    assert_eq!(file_io.avg_per_call, Duration::from_millis(20));
}

/// Test percentage calculation accuracy
#[test]
fn test_percentage_accuracy() {
    let registry = ClusterRegistry::default_transpiler_clusters().expect("test");

    let spans = vec![
        make_span("read", 25_000_000, vec![]), // 25% (FileIO)
        make_span("mmap", 75_000_000, vec![]), // 75% (MemoryAllocation)
    ];

    let attributions = calculate_time_attribution(&spans, &registry);

    let file_io = attributions.iter().find(|a| a.cluster == "FileIO").expect("test");
    assert!((file_io.percentage - 25.0).abs() < 0.01);

    let mem = attributions.iter().find(|a| a.cluster == "MemoryAllocation").expect("test");
    assert!((mem.percentage - 75.0).abs() < 0.01);
}
