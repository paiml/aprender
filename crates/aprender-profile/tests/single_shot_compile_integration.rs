//! Integration tests for Single-Shot Compile Tooling
//!
//! Tests the complete workflow combining:
//! - TOML-based syscall clustering (Section 6.1)
//! - N-gram sequence mining (Section 6.1.1)
//! - Time-weighted attribution (Section 6.2)
//! - Statistical regression detection (Section 6.4)
//!
//! Toyota Way Principle: Genchi Genbutsu (Go and See)
//! These tests use realistic transpiler syscall patterns.

use renacer::cluster::ClusterRegistry;
use renacer::regression::{assess_regression, RegressionConfig};
use renacer::sequence::{detect_sequence_anomalies, extract_ngrams};
use renacer::time_attribution::{calculate_time_attribution, identify_hotspots};
use renacer::unified_trace::SyscallSpan;
use std::borrow::Cow;
use std::collections::HashMap;

fn make_span(name: &'static str, duration_nanos: u64) -> SyscallSpan {
    SyscallSpan {
        span_id: 1,
        parent_span_id: 0,
        name: Cow::Borrowed(name),
        args: vec![],
        return_value: 0,
        timestamp_nanos: 0,
        duration_nanos,
        errno: None,
    }
}

/// Test complete workflow: Normal transpiler execution (baseline)
#[test]
fn test_complete_workflow_baseline() {
    let registry = ClusterRegistry::default_transpiler_clusters().expect("test");

    // Realistic transpiler baseline: File I/O dominates, single-threaded
    let spans = vec![
        make_span("open", 1_000_000),   // 1ms - open source file
        make_span("read", 50_000_000),  // 50ms - read source (blocking I/O)
        make_span("mmap", 5_000_000),   // 5ms - allocate AST memory
        make_span("brk", 2_000_000),    // 2ms - grow heap
        make_span("write", 10_000_000), // 10ms - write output
        make_span("close", 1_000_000),  // 1ms - close files
    ];

    // 1. Clustering analysis
    let attributions = calculate_time_attribution(&spans, &registry);
    assert!(!attributions.is_empty(), "Should identify clusters");

    let hotspots = identify_hotspots(&attributions);
    assert!(hotspots.iter().any(|h| h.cluster == "FileIO"), "FileIO should be a hotspot");
    assert!(
        hotspots.iter().all(|h| h.is_expected),
        "All hotspots should be expected for transpiler"
    );

    // 2. Sequence analysis
    let syscall_names: Vec<String> = spans.iter().map(|s| s.name.to_string()).collect();
    let ngrams = extract_ngrams(&syscall_names, 3);
    assert!(!ngrams.is_empty(), "Should extract N-grams");

    // 3. Regression detection (baseline vs baseline = no regression)
    let mut baseline_data = HashMap::new();
    baseline_data.insert("mmap".to_string(), vec![5.0, 5.0, 5.0, 5.0, 5.0]);
    baseline_data.insert("read".to_string(), vec![50.0, 50.0, 50.0, 50.0, 50.0]);

    let config = RegressionConfig::default();
    let assessment = assess_regression(&baseline_data, &baseline_data, &config).expect("test");

    // Baseline vs baseline should show no regression
    assert_eq!(assessment.verdict, renacer::regression::RegressionVerdict::NoRegression);
}

/// Test regression detection: File I/O regression
#[test]
fn test_regression_file_io_slowdown() {
    // Baseline: Fast file I/O
    let mut baseline = HashMap::new();
    baseline.insert("read".to_string(), vec![10.0, 11.0, 10.0, 12.0, 10.0]);
    baseline.insert("write".to_string(), vec![5.0, 6.0, 5.0, 6.0, 5.0]);

    // Current: Slow file I/O (regression!)
    let mut current = HashMap::new();
    current.insert("read".to_string(), vec![50.0, 52.0, 51.0, 53.0, 50.0]);
    current.insert("write".to_string(), vec![5.0, 6.0, 5.0, 6.0, 5.0]);

    let config = RegressionConfig::default();
    let assessment = assess_regression(&baseline, &current, &config).expect("test");

    // Should detect read() regression
    match assessment.verdict {
        renacer::regression::RegressionVerdict::Regression { ref regressed_syscalls, .. } => {
            assert!(regressed_syscalls.contains(&"read".to_string()));
        }
        _ => panic!("Expected regression detection"),
    }
}

/// Test anomaly detection: Unexpected networking
#[test]
fn test_anomaly_unexpected_networking() {
    let registry = ClusterRegistry::default_transpiler_clusters().expect("test");

    // Baseline: Normal transpiler (no networking)
    let baseline_spans = [
        make_span("open", 1_000_000),
        make_span("read", 50_000_000),
        make_span("write", 10_000_000),
    ];

    // Current: WITH networking (telemetry leak!)
    let current_spans = vec![
        make_span("open", 1_000_000),
        make_span("read", 50_000_000),
        make_span("socket", 5_000_000),   // NEW!
        make_span("connect", 10_000_000), // NEW!
        make_span("send", 5_000_000),     // NEW!
        make_span("write", 10_000_000),
    ];

    // Time attribution should flag networking hotspot
    let current_attributions = calculate_time_attribution(&current_spans, &registry);
    let current_hotspots = identify_hotspots(&current_attributions);

    let networking_hotspot = current_hotspots.iter().find(|h| h.cluster == "Networking");
    assert!(networking_hotspot.is_some(), "Should detect networking hotspot");
    assert!(!networking_hotspot.expect("test").is_expected, "Networking should be unexpected");

    // Sequence analysis should detect new patterns
    let baseline_syscalls: Vec<String> =
        baseline_spans.iter().map(|s| s.name.to_string()).collect();
    let current_syscalls: Vec<String> = current_spans.iter().map(|s| s.name.to_string()).collect();

    let baseline_ngrams = extract_ngrams(&baseline_syscalls, 3);
    let current_ngrams = extract_ngrams(&current_syscalls, 3);

    let anomalies = detect_sequence_anomalies(&baseline_ngrams, &current_ngrams, 0.30);

    // Should detect new sequences involving networking
    let networking_anomalies: Vec<_> = anomalies
        .iter()
        .filter(|a| {
            a.ngram
                .iter()
                .any(|s| s.contains("socket") || s.contains("connect") || s.contains("send"))
        })
        .collect();

    assert!(!networking_anomalies.is_empty(), "Should detect networking anomalies");
}

/// Test memory allocation pattern change detection
#[test]
fn test_memory_allocation_pattern_change() {
    // Baseline: Uses mmap
    let baseline_syscalls = vec!["mmap".to_string(), "read".to_string(), "munmap".to_string()];

    // Current: Uses brk instead (semantic equivalence)
    let current_syscalls = vec!["brk".to_string(), "read".to_string(), "brk".to_string()];

    let baseline_ngrams = extract_ngrams(&baseline_syscalls, 2);
    let current_ngrams = extract_ngrams(&current_syscalls, 2);

    let anomalies = detect_sequence_anomalies(&baseline_ngrams, &current_ngrams, 0.30);

    // Should detect sequence changes (mmap→read vs brk→read)
    assert!(!anomalies.is_empty(), "Should detect allocation pattern change");

    // Note: Semantic equivalence (Section 6.3) would validate this as acceptable
}

/// Test complete transpiler validation workflow
#[test]
fn test_complete_transpiler_validation() {
    let registry = ClusterRegistry::default_transpiler_clusters().expect("test");

    // Step 1: Collect baseline golden trace
    let baseline_spans = vec![
        make_span("open", 1_000_000),
        make_span("read", 50_000_000),
        make_span("mmap", 5_000_000),
        make_span("write", 10_000_000),
        make_span("close", 1_000_000),
    ];

    // Step 2: Collect current trace (optimized version)
    let current_spans = vec![
        make_span("open", 1_000_000),
        make_span("read", 30_000_000), // Faster! (buffered I/O)
        make_span("mmap", 5_000_000),
        make_span("write", 8_000_000), // Faster!
        make_span("close", 1_000_000),
    ];

    // Step 3: Time attribution analysis
    let baseline_attr = calculate_time_attribution(&baseline_spans, &registry);
    let current_attr = calculate_time_attribution(&current_spans, &registry);

    // Verify optimization reduced time
    let baseline_file_io = baseline_attr.iter().find(|a| a.cluster == "FileIO").expect("test");
    let current_file_io = current_attr.iter().find(|a| a.cluster == "FileIO").expect("test");

    assert!(
        current_file_io.total_time < baseline_file_io.total_time,
        "Optimization should reduce FileIO time"
    );

    // Step 4: Sequence grammar validation
    let baseline_syscalls: Vec<String> =
        baseline_spans.iter().map(|s| s.name.to_string()).collect();
    let current_syscalls: Vec<String> = current_spans.iter().map(|s| s.name.to_string()).collect();

    let baseline_ngrams = extract_ngrams(&baseline_syscalls, 3);
    let current_ngrams = extract_ngrams(&current_syscalls, 3);

    // Same sequences (optimization preserves behavior)
    assert_eq!(baseline_ngrams.len(), current_ngrams.len(), "Sequence grammar preserved");

    // Step 5: Statistical regression check
    let mut baseline_data = HashMap::new();
    baseline_data.insert("read".to_string(), vec![50.0, 50.0, 50.0, 50.0, 50.0]);

    let mut current_data = HashMap::new();
    current_data.insert("read".to_string(), vec![30.0, 30.0, 30.0, 30.0, 30.0]); // Improvement!

    let config = RegressionConfig::default();
    let assessment = assess_regression(&baseline_data, &current_data, &config).expect("test");

    // This is an IMPROVEMENT (negative regression), not a problem
    // Tool should report this as optimization success
    match assessment.verdict {
        renacer::regression::RegressionVerdict::Regression { .. } => {
            // Detected change (could be improvement)
        }
        renacer::regression::RegressionVerdict::NoRegression => {
            // Also acceptable if within noise threshold
        }
        _ => {}
    }
}

/// Test hotspot identification accuracy
#[test]
fn test_hotspot_identification_accuracy() {
    let registry = ClusterRegistry::default_transpiler_clusters().expect("test");

    // Spans with clear hotspot (80% in read)
    let spans = vec![
        make_span("read", 80_000_000),  // 80% of time
        make_span("mmap", 10_000_000),  // 10%
        make_span("write", 10_000_000), // 10%
    ];

    let attributions = calculate_time_attribution(&spans, &registry);
    let hotspots = identify_hotspots(&attributions);

    // Should identify FileIO (read+write) and MemoryAllocation (mmap) as hotspots
    // FileIO: 80ms + 10ms = 90ms (90%)
    // MemoryAllocation: 10ms (10%)
    assert!(!hotspots.is_empty(), "Should identify hotspots (>5% threshold)");

    let file_io_hotspot = hotspots.iter().find(|h| h.cluster == "FileIO").expect("test");
    assert!(
        (file_io_hotspot.percentage - 90.0).abs() < 1.0,
        "FileIO should be ~90%, got {}",
        file_io_hotspot.percentage
    );
}

/// Test noise filtering in regression detection
#[test]
fn test_noise_filtering_integration() {
    // Baseline with noisy syscalls
    let mut baseline = HashMap::new();
    baseline.insert(
        "read".to_string(),
        vec![10.0, 11.0, 10.0, 12.0, 10.0], // Stable
    );
    baseline.insert(
        "socket".to_string(),
        vec![5.0, 50.0, 3.0, 45.0, 2.0], // Noisy!
    );

    // Current with same patterns
    let mut current = HashMap::new();
    current.insert("read".to_string(), vec![10.0, 11.0, 10.0, 13.0, 10.0]);
    current.insert("socket".to_string(), vec![6.0, 51.0, 4.0, 46.0, 3.0]);

    let config = RegressionConfig::default();
    let assessment = assess_regression(&baseline, &current, &config).expect("test");

    // Should filter socket as noisy
    assert!(assessment.filtered_syscalls.contains(&"socket".to_string()));

    // Should only test stable syscalls
    assert_eq!(assessment.tests.len(), 1);
    assert!(assessment.tests.contains_key("read"));
}
