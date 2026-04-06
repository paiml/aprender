//! golden_trace_f1211 - Part 2

use cbtop::{
    GoldenComparator, GoldenTrace, GoldenTraceError, GoldenTraceManager, TraceMetrics,
    TraceSyscallBreakdown as SyscallBreakdown,
};
use tempfile::TempDir;

// =============================================================================
// F1217: Empty Golden Handling Tests
// =============================================================================

/// F1217.1: Empty golden returns baseline error
#[test]
fn f1217_no_baseline() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = GoldenTraceManager::new(temp_dir.path().to_path_buf());
    manager.ensure_directory().unwrap();

    let result = manager.load_golden("nonexistent");
    assert!(matches!(result, Err(GoldenTraceError::NoBaseline)));
}

/// F1217.2: Compare without golden fails
#[test]
fn f1217_compare_no_golden() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = GoldenTraceManager::new(temp_dir.path().to_path_buf());
    manager.ensure_directory().unwrap();

    let current = TraceMetrics::new().total_time_us(1000.0);
    let result = manager.compare_to_golden("missing", &current);

    assert!(matches!(result, Err(GoldenTraceError::NoBaseline)));
}

// =============================================================================
// F1218: Trace Hash Tests
// =============================================================================

/// F1218.1: Trace hash computed
#[test]
fn f1218_hash_computed() {
    let trace = GoldenTrace::new("hash_test", TraceMetrics::new().total_time_us(1000.0));

    assert!(!trace.hash.is_empty());
    assert_eq!(trace.hash.len(), 16); // 16 hex chars
}

/// F1218.2: Hash is deterministic
#[test]
fn f1218_hash_deterministic() {
    let metrics = TraceMetrics::new().total_time_us(1000.0);

    // Same name and metrics should produce same hash
    let trace1 = GoldenTrace::new("deterministic", metrics.clone());
    let computed = trace1.compute_hash();

    // Hash should be consistent
    assert_eq!(trace1.hash, computed);
    assert!(trace1.verify_hash());
}

// =============================================================================
// F1219: Multiple Goldens Tests
// =============================================================================

/// F1219.1: Multiple goldens supported
#[test]
fn f1219_multiple_goldens() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = GoldenTraceManager::new(temp_dir.path().to_path_buf());

    // Save multiple goldens
    manager.capture_golden("v1", TraceMetrics::new().total_time_us(1000.0)).unwrap();
    manager.capture_golden("v2", TraceMetrics::new().total_time_us(900.0)).unwrap();
    manager.capture_golden("v3", TraceMetrics::new().total_time_us(800.0)).unwrap();

    let names = manager.list_goldens().unwrap();
    assert_eq!(names.len(), 3);
}

/// F1219.2: Version selection works
#[test]
fn f1219_version_selection() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = GoldenTraceManager::new(temp_dir.path().to_path_buf());

    manager.capture_golden("release_1_0", TraceMetrics::new().total_time_us(1000.0)).unwrap();
    manager.capture_golden("release_2_0", TraceMetrics::new().total_time_us(800.0)).unwrap();

    // Can load specific version
    let v1 = manager.load_golden("release_1_0").unwrap();
    assert_eq!(v1.metrics.total_time_us, 1000.0);

    let v2 = manager.load_golden("release_2_0").unwrap();
    assert_eq!(v2.metrics.total_time_us, 800.0);
}

// =============================================================================
// F1220: Export Format Tests
// =============================================================================

/// F1220.1: JSON export valid
#[test]
fn f1220_json_export() {
    let metrics = TraceMetrics::new().total_time_us(1000.0).p50_latency_us(50.0);
    let trace = GoldenTrace::new("json_export", metrics);

    let json = trace.to_json().unwrap();

    // Should be valid JSON
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed["name"], "json_export");
    assert_eq!(parsed["metrics"]["total_time_us"], 1000.0);
}

/// F1220.2: TOML export valid
#[test]
fn f1220_toml_export() {
    let metrics = TraceMetrics::new().total_time_us(1000.0).p50_latency_us(50.0);
    let trace = GoldenTrace::new("toml_export", metrics);

    let toml_str = trace.to_toml().unwrap();

    // Should be valid TOML
    let parsed: toml::Value = toml::from_str(&toml_str).unwrap();
    assert_eq!(parsed["name"].as_str(), Some("toml_export"));
}

/// F1220.3: Export to file
#[test]
fn f1220_export_to_file() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = GoldenTraceManager::new(temp_dir.path().join("goldens"));

    manager.capture_golden("exportable", TraceMetrics::new().total_time_us(1000.0)).unwrap();

    // Export to different location
    let export_path = temp_dir.path().join("exported.toml");
    manager.export_trace("exportable", &export_path).unwrap();

    // File should exist and be valid
    assert!(export_path.exists());
    let content = std::fs::read_to_string(&export_path).unwrap();
    assert!(content.contains("exportable"));
}

// =============================================================================
// Additional Tests
// =============================================================================

/// Test trace deletion
#[test]
fn test_trace_deletion() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = GoldenTraceManager::new(temp_dir.path().to_path_buf());

    manager.capture_golden("deletable", TraceMetrics::new()).unwrap();
    assert!(manager.golden_exists("deletable"));

    manager.delete_golden("deletable").unwrap();
    assert!(!manager.golden_exists("deletable"));
}

/// Test regression summary message
#[test]
fn test_regression_summary() {
    let golden = GoldenTrace::new("baseline", TraceMetrics::new().total_time_us(1000.0));

    let current = TraceMetrics::new().total_time_us(1500.0); // 50% regression

    let comparator = GoldenComparator::new();
    let result = comparator.compare(&current, &golden).unwrap();

    assert!(result.is_regression);
    assert!(result.summary.contains("REGRESSION"));
}

/// Test regressions list
#[test]
fn test_regressions_list() {
    let golden = GoldenTrace::new(
        "baseline",
        TraceMetrics::new().total_time_us(1000.0).p50_latency_us(50.0).throughput(10000.0),
    );

    let current = TraceMetrics::new()
        .total_time_us(1200.0) // 20% regression
        .p50_latency_us(65.0) // 30% regression
        .throughput(7000.0); // 30% regression

    let comparator = GoldenComparator::new().with_threshold(10.0);
    let result = comparator.compare(&current, &golden).unwrap();

    let regressions = result.regressions();
    assert!(!regressions.is_empty());
    assert!(regressions.iter().any(|(name, _)| name == "total_time"));
    assert!(regressions.iter().any(|(name, _)| name == "p50_latency"));
}

/// Test custom metrics
#[test]
fn test_custom_metrics() {
    let metrics = TraceMetrics::new()
        .total_time_us(1000.0)
        .with_custom("gpu_utilization", 95.5)
        .with_custom("memory_bandwidth", 800.0);

    let trace = GoldenTrace::new("custom_test", metrics);

    assert_eq!(trace.metrics.custom.get("gpu_utilization"), Some(&95.5));
    assert_eq!(trace.metrics.custom.get("memory_bandwidth"), Some(&800.0));
}

/// Test environment info
#[test]
fn test_environment_info() {
    let trace = GoldenTrace::new("env_test", TraceMetrics::new())
        .with_env("os", "linux")
        .with_env("cpu", "x86_64");

    assert_eq!(trace.environment.get("os"), Some(&"linux".to_string()));
    assert_eq!(trace.environment.get("cpu"), Some(&"x86_64".to_string()));
}

/// Test metrics validity
#[test]
fn test_metrics_validity() {
    let valid = TraceMetrics::new().total_time_us(1000.0);
    assert!(valid.is_valid());

    let mut invalid = TraceMetrics::new();
    invalid.total_time_us = f64::NAN;
    assert!(!invalid.is_valid());

    let mut negative = TraceMetrics::new();
    negative.total_time_us = -1.0;
    assert!(!negative.is_valid());
}

/// Test invalid metrics comparison
#[test]
fn test_invalid_metrics_comparison() {
    let golden = GoldenTrace::new("baseline", TraceMetrics::new().total_time_us(1000.0));

    let mut invalid = TraceMetrics::new();
    invalid.total_time_us = f64::NAN;

    let comparator = GoldenComparator::new();
    let result = comparator.compare(&invalid, &golden);

    assert!(matches!(result, Err(GoldenTraceError::InvalidTrace(_))));
}

/// Test syscall threshold detection
#[test]
fn test_syscall_threshold_detection() {
    let baseline = SyscallBreakdown {
        read_count: 100,
        write_count: 100,
        mmap_count: 10,
        futex_count: 20,
        other_count: 10,
    };

    let current = SyscallBreakdown {
        read_count: 130, // 30% increase
        write_count: 100,
        mmap_count: 10,
        futex_count: 20,
        other_count: 10,
    };

    let delta = current.percentage_diff(&baseline);

    assert!(delta.exceeds_threshold(20.0));
    assert!(!delta.exceeds_threshold(50.0));
}
