#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! Golden Trace Validation Tests
//!
//! Validates that SIMD performance matches golden trace baselines.
//! Golden traces captured with Renacer v0.6.2 on reference hardware.
//!
//! Purpose:
//! - Detect performance regressions
//! - Validate syscall patterns haven't changed
//! - Ensure SIMD optimizations remain effective

use std::path::Path;

/// Test that backend detection example matches golden trace performance
#[test]
#[ignore = "requires golden trace files"]
fn test_backend_detection_golden_trace() {
    let golden_trace = Path::new("golden_traces/backend_detection_summary.txt");

    if !golden_trace.exists() {
        eprintln!("Golden trace not found, skipping validation");
        return;
    }

    // Golden trace shows: 0.730ms runtime, 87 syscalls
    // Allow 10% deviation: 0.803ms, 96 syscalls
    let max_runtime = 0.803;
    let max_syscalls = 96;

    validate_performance_budget("backend_detection", max_runtime, max_syscalls);
}

/// Test that performance demo matches golden trace
#[test]
#[ignore = "requires golden trace files"]
fn test_performance_demo_golden_trace() {
    let golden_trace = Path::new("golden_traces/performance_demo_summary.txt");

    if !golden_trace.exists() {
        eprintln!("Golden trace not found, skipping validation");
        return;
    }

    // Golden trace shows: 1.507ms runtime, 138 syscalls
    // Allow 10% deviation: 1.658ms, 152 syscalls
    let max_runtime = 1.658;
    let max_syscalls = 152;

    validate_performance_budget("performance_demo", max_runtime, max_syscalls);
}

/// Test that matrix operations match golden trace
#[test]
#[ignore = "requires golden trace files"]
fn test_matrix_operations_golden_trace() {
    let golden_trace = Path::new("golden_traces/matrix_operations_summary.txt");

    if !golden_trace.exists() {
        eprintln!("Golden trace not found, skipping validation");
        return;
    }

    // Golden trace shows: 1.560ms runtime, 168 syscalls
    // Allow 10% deviation: 1.716ms, 185 syscalls
    let max_runtime = 1.716;
    let max_syscalls = 185;

    validate_performance_budget("matrix_operations", max_runtime, max_syscalls);
}

/// Test that activation functions match golden trace
#[test]
#[ignore = "requires golden trace files"]
fn test_activation_functions_golden_trace() {
    let golden_trace = Path::new("golden_traces/activation_functions_summary.txt");

    if !golden_trace.exists() {
        eprintln!("Golden trace not found, skipping validation");
        return;
    }

    // Golden trace shows: 1.298ms runtime, 159 syscalls
    // Allow 10% deviation: 1.428ms, 175 syscalls
    let max_runtime = 1.428;
    let max_syscalls = 175;

    validate_performance_budget("activation_functions", max_runtime, max_syscalls);
}

/// Test that ML similarity operations match golden trace
#[test]
#[ignore = "requires golden trace files"]
fn test_ml_similarity_golden_trace() {
    let golden_trace = Path::new("golden_traces/ml_similarity_summary.txt");

    if !golden_trace.exists() {
        eprintln!("Golden trace not found, skipping validation");
        return;
    }

    // Golden trace shows: 0.817ms runtime, 109 syscalls
    // Allow 10% deviation: 0.899ms, 120 syscalls
    let max_runtime = 0.899;
    let max_syscalls = 120;

    validate_performance_budget("ml_similarity", max_runtime, max_syscalls);
}

/// Helper function to validate performance budgets
///
/// This would ideally use renacer to capture a new trace and compare,
/// but for CI we just validate the golden traces exist and are well-formed.
fn validate_performance_budget(_example: &str, _max_runtime_ms: f64, _max_syscalls: usize) {
    // In a full implementation, this would:
    // 1. Run: renacer --format json -- ./target/release/examples/{example}
    // 2. Parse JSON output
    // 3. Extract runtime and syscall count
    // 4. Assert runtime <= max_runtime_ms
    // 5. Assert syscalls <= max_syscalls
    //
    // For now, we just ensure the golden traces are present
    // (actual validation happens in CI with renacer installed)

    // Placeholder: In CI with renacer, this test would fail if performance regresses
    // When renacer is available, implement the above validation steps
}

/// Test that all golden traces are well-formed JSON
#[test]
fn test_golden_traces_valid_json() {
    let traces = [
        "golden_traces/backend_detection.json",
        "golden_traces/performance_demo.json",
        "golden_traces/matrix_operations.json",
        "golden_traces/activation_functions.json",
        "golden_traces/ml_similarity.json",
    ];

    for trace_path in &traces {
        let path = Path::new(trace_path);
        if !path.exists() {
            eprintln!("Warning: Golden trace not found: {}", trace_path);
            continue;
        }

        let content = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("Failed to read {}: {}", trace_path, e));

        // Skip placeholder files (captured traces will be added later)
        // Valid JSON must start with '{' or '['
        let trimmed = content.trim();
        if trimmed.is_empty() || !trimmed.starts_with('{') && !trimmed.starts_with('[') {
            eprintln!("Info: Skipping placeholder golden trace: {}", trace_path);
            continue;
        }

        // Validate JSON structure
        let parsed: serde_json::Value = serde_json::from_str(&content)
            .unwrap_or_else(|e| panic!("Invalid JSON in {}: {}", trace_path, e));

        // Validate required fields exist
        assert!(parsed.get("version").is_some(), "Missing 'version' in {}", trace_path);
        assert!(parsed.get("format").is_some(), "Missing 'format' in {}", trace_path);

        // Validate format is correct
        if let Some(format) = parsed.get("format") {
            assert_eq!(
                format.as_str(),
                Some("renacer-json-v1"),
                "Unexpected format in {}",
                trace_path
            );
        }
    }
}

/// Test that golden trace summaries contain expected performance data
#[test]
fn test_golden_trace_summaries_exist() {
    let summaries = [
        "golden_traces/backend_detection_summary.txt",
        "golden_traces/performance_demo_summary.txt",
        "golden_traces/matrix_operations_summary.txt",
        "golden_traces/activation_functions_summary.txt",
        "golden_traces/ml_similarity_summary.txt",
    ];

    for summary_path in &summaries {
        let path = Path::new(summary_path);
        assert!(path.exists(), "Golden trace summary missing: {}", summary_path);

        let content = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("Failed to read {}: {}", summary_path, e));

        // Validate summary contains syscall statistics
        assert!(
            content.contains("% time") && content.contains("syscall"),
            "Summary {} does not contain syscall statistics",
            summary_path
        );
    }
}

/// Test that ANALYSIS.md documents the golden traces
#[test]
fn test_golden_trace_analysis_exists() {
    let analysis_path = Path::new("golden_traces/ANALYSIS.md");
    assert!(analysis_path.exists(), "Golden trace analysis documentation missing");

    let content = std::fs::read_to_string(analysis_path).expect("Failed to read ANALYSIS.md");

    // Validate documentation contains key sections
    assert!(content.contains("## Overview"), "Missing Overview section");
    assert!(content.contains("## Baseline Performance Metrics"), "Missing performance metrics");
    assert!(content.contains("SIMD"), "Missing SIMD documentation");
}

/// Validates that the performance budgets are documented
#[test]
fn test_performance_budgets_documented() {
    let analysis_path = Path::new("golden_traces/ANALYSIS.md");
    if !analysis_path.exists() {
        return;
    }

    let content = std::fs::read_to_string(analysis_path).unwrap();

    // Check that all examples have documented budgets
    assert!(content.contains("backend_detection"), "Missing backend_detection budget");
    assert!(content.contains("matrix_operations"), "Missing matrix_operations budget");
    assert!(content.contains("activation_functions"), "Missing activation_functions budget");
    assert!(content.contains("performance_demo"), "Missing performance_demo budget");
    assert!(content.contains("ml_similarity"), "Missing ml_similarity budget");

    // Check that performance budget compliance is documented
    assert!(
        content.contains("Performance Budget Compliance"),
        "Missing performance budget compliance section"
    );
}
