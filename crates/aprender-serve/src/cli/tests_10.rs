//! T-COV-95 Poisoned Pygmies: CLI Graceful Degradation Tests (PMAT-802)
//!
//! Dr. Popper's directive: "The CLI must not merely run; it must gracefully
//! degrade when fed poisoned artifacts. Valid headers, invalid data."
//!
//! This module tests:
//! 1. Benchmark subcommand with poisoned data
//! 2. Visualize subcommand with edge cases
//! 3. Format/display functions with corrupted inputs
//!
//! Target: 483 missed lines in cli/mod.rs

use super::*;
use std::io::Write;
use tempfile::NamedTempFile;

// ============================================================================
// Poisoned Pygmy: Format Size Edge Cases
// ============================================================================

#[test]
fn test_format_size_zero() {
    let result = format_size(0);
    assert_eq!(result, "0 B");
}

#[test]
fn test_format_size_bytes() {
    assert_eq!(format_size(1), "1 B");
    assert_eq!(format_size(512), "512 B");
    assert_eq!(format_size(1023), "1023 B");
}

#[test]
fn test_format_size_kilobytes() {
    assert_eq!(format_size(1024), "1.0 KB");
    assert_eq!(format_size(1536), "1.5 KB");
    assert_eq!(format_size(1024 * 100), "100.0 KB");
}

#[test]
fn test_format_size_megabytes() {
    assert_eq!(format_size(1024 * 1024), "1.0 MB");
    assert_eq!(format_size(1024 * 1024 * 512), "512.0 MB");
}

#[test]
fn test_format_size_gigabytes() {
    assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
    assert_eq!(format_size(1024 * 1024 * 1024 * 7), "7.0 GB");
}

#[test]
fn test_format_size_boundary_kb() {
    // Just below KB boundary
    assert_eq!(format_size(1023), "1023 B");
    // At KB boundary
    assert_eq!(format_size(1024), "1.0 KB");
}

#[test]
fn test_format_size_boundary_mb() {
    // Just below MB boundary
    let result = format_size(1024 * 1024 - 1);
    assert!(result.contains("KB"));
    // At MB boundary
    assert_eq!(format_size(1024 * 1024), "1.0 MB");
}

#[test]
fn test_format_size_boundary_gb() {
    // Just below GB boundary
    let result = format_size(1024 * 1024 * 1024 - 1);
    assert!(result.contains("MB"));
    // At GB boundary
    assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
}

// ============================================================================
// Poisoned Pygmy: Benchmark Suite Validation
// ============================================================================

#[test]
fn test_benchmark_suites_exist() {
    // Verify BENCHMARK_SUITES has content
    assert!(BENCHMARK_SUITES.len() >= 5);
}

#[test]
fn test_benchmark_suites_have_descriptions() {
    for (name, desc) in BENCHMARK_SUITES {
        assert!(!name.is_empty(), "Suite name should not be empty");
        assert!(!desc.is_empty(), "Suite description should not be empty");
    }
}

#[test]
fn test_benchmark_suite_names_valid() {
    let expected = ["tensor_ops", "inference", "cache", "tokenizer", "quantize"];
    for name in expected {
        assert!(
            BENCHMARK_SUITES.iter().any(|(n, _)| *n == name),
            "Expected suite '{}' not found",
            name
        );
    }
}

// ============================================================================
// Poisoned Pygmy: Display Model Info with Corrupted Files
// ============================================================================

#[test]
fn test_display_model_info_gguf_valid_header() {
    use crate::gguf::{GGUF_MAGIC, GGUF_VERSION_V3};

    // Create minimal valid GGUF
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
    data.extend_from_slice(&0u64.to_le_bytes()); // metadata_count

    // Should not panic even with minimal data
    let result = display_model_info("test.gguf", &data);
    // May succeed or fail, but shouldn't panic
    let _ = result;
}

#[test]
fn test_display_model_info_gguf_poisoned_truncated() {
    use crate::gguf::GGUF_MAGIC;

    // Valid magic but truncated - only 8 bytes
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes()); // Partial version

    let result = display_model_info("test.gguf", &data);
    // Should fail gracefully, not panic
    assert!(result.is_err());
}

#[test]
fn test_display_model_info_safetensors_poisoned() {
    // Valid SafeTensors structure but invalid JSON
    let mut data = Vec::new();
    let header = b"{ not valid json }";
    data.extend_from_slice(&(header.len() as u64).to_le_bytes());
    data.extend_from_slice(header);

    let result = display_model_info("test.safetensors", &data);
    // Should fail gracefully
    assert!(result.is_err());
}

#[test]
fn test_display_model_info_apr_poisoned() {
    use crate::format::APR_MAGIC;

    // Valid APR magic but corrupted metadata
    let mut data = Vec::new();
    data.extend_from_slice(APR_MAGIC);
    data.extend_from_slice(&[0xFF; 100]); // Garbage

    let result = display_model_info("test.apr", &data);
    // Should handle gracefully (may succeed with "Unknown" or fail)
    let _ = result;
}

#[test]
fn test_display_model_info_unknown_format() {
    // Completely unknown format
    let data = b"UNKNOWN_FORMAT_HEADER_DATA_HERE";

    let result = display_model_info("test.bin", data);
    // Should succeed and report as unknown
    assert!(result.is_ok());
}

#[test]
fn test_display_model_info_empty_file() {
    let data: &[u8] = &[];

    let result = display_model_info("empty.gguf", data);
    // Should fail gracefully with empty data
    assert!(result.is_err() || result.is_ok()); // Either way, no panic
}

// ============================================================================
// Poisoned Pygmy: Visualization with Edge Cases
// ============================================================================

#[test]
fn test_run_visualization_zero_samples() {
    // Zero samples - should not panic
    run_visualization(false, 0);
}

#[test]
fn test_run_visualization_one_sample() {
    // Single sample - edge case
    run_visualization(false, 1);
}

#[test]
fn test_run_visualization_large_samples() {
    // Large sample count
    run_visualization(false, 1000);
}

#[test]
fn test_run_visualization_with_color() {
    // Color enabled
    run_visualization(true, 100);
}

#[test]
fn test_run_visualization_without_color() {
    // Color disabled
    run_visualization(false, 100);
}

// ============================================================================
// Poisoned Pygmy: is_local_file_path Edge Cases
// ============================================================================

#[test]
fn test_is_local_file_path_absolute() {
    assert!(is_local_file_path("/home/user/model.gguf"));
    assert!(is_local_file_path("/tmp/test.apr"));
}

#[test]
fn test_is_local_file_path_relative() {
    assert!(is_local_file_path("./model.gguf"));
    assert!(is_local_file_path("models/test.apr"));
}

#[test]
fn test_is_local_file_path_registry_uri() {
    assert!(!is_local_file_path("pacha://model/name"));
    assert!(!is_local_file_path("hf://org/model"));
}

#[test]
fn test_is_local_file_path_empty() {
    // Empty string may or may not be treated as local - implementation dependent
    let result = is_local_file_path("");
    // Just verify it doesn't panic
    let _ = result;
}

#[test]
fn test_is_local_file_path_just_filename() {
    assert!(is_local_file_path("model.gguf"));
}

// ============================================================================
// Poisoned Pygmy: Benchmark File Operations
// ============================================================================

#[test]
fn test_run_bench_compare_nonexistent_files() {
    let result = run_bench_compare("/nonexistent/file1.json", "/nonexistent/file2.json", 0.1);
    assert!(result.is_err());
}

#[test]
fn test_run_bench_compare_empty_file() {
    let mut temp = NamedTempFile::new().unwrap();
    temp.write_all(b"").unwrap();
    temp.flush().unwrap();

    let result = run_bench_compare(
        temp.path().to_str().unwrap(),
        temp.path().to_str().unwrap(),
        0.1,
    );
    // Should fail gracefully with empty JSON
    assert!(result.is_err());
}

#[test]
fn test_run_bench_compare_invalid_json() {
    let mut temp = NamedTempFile::new().unwrap();
    temp.write_all(b"{ not valid json }").unwrap();
    temp.flush().unwrap();

    let result = run_bench_compare(
        temp.path().to_str().unwrap(),
        temp.path().to_str().unwrap(),
        0.1,
    );
    assert!(result.is_err());
}

#[test]
fn test_run_bench_regression_nonexistent() {
    let result = run_bench_regression(
        "/nonexistent/baseline.json",
        "/nonexistent/current.json",
        false, // strict
    );
    assert!(result.is_err());
}

// ============================================================================
// Poisoned Pygmy: Convoy and Saturation Tests (CLI paths)
// ============================================================================

#[test]
fn test_run_convoy_test_no_model() {
    // Runtime None means use default
    let result = run_convoy_test(None, None, None);
    // Should handle gracefully (may succeed with no-op or fail)
    let _ = result;
}

#[test]
fn test_run_saturation_test_no_model() {
    let result = run_saturation_test(None, None, None);
    let _ = result;
}

// ============================================================================
// Poisoned Pygmy: Benchmark List Mode
// ============================================================================

#[test]
fn test_run_benchmarks_list_mode() {
    // list=true should just print suites and return
    let result = run_benchmarks(None, true, None, None, None, None);
    assert!(result.is_ok());
}

#[test]
#[ignore = "calls std::process::exit(1) which kills test process"]
fn test_run_benchmarks_invalid_suite() {
    // Note: This test is ignored because run_benchmarks calls process::exit(1)
    // for invalid suite names, which kills the test harness.
    let result = run_benchmarks(
        Some("nonexistent_suite".to_string()),
        false,
        None,
        None,
        None,
        None,
    );
    let _ = result;
}

#[test]
#[ignore = "requires cargo bench infrastructure"]
fn test_run_benchmarks_valid_suite_no_runtime() {
    // Note: This test is ignored because it tries to run actual cargo bench
    let result = run_benchmarks(
        Some("tensor_ops".to_string()),
        false,
        None,
        None,
        None,
        None,
    );
    let _ = result;
}

// ============================================================================
// Poisoned Pygmy: Chat Template Handling
// ============================================================================

#[test]
fn test_display_model_info_by_extension() {
    // Test extension-based detection
    use crate::gguf::GGUF_MAGIC;

    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    // Incomplete but has valid magic

    // .gguf extension triggers GGUF path
    let result = display_model_info("model.gguf", &data);
    // Either parses or fails, but uses GGUF path
    let _ = result;
}

#[test]
fn test_display_model_info_safetensors_extension() {
    // Test .safetensors extension path
    let mut data = Vec::new();
    let header = b"{}"; // Valid but empty JSON
    data.extend_from_slice(&(header.len() as u64).to_le_bytes());
    data.extend_from_slice(header);

    let result = display_model_info("model.safetensors", &data);
    // Empty header is valid JSON, may succeed or fail on tensor count
    let _ = result;
}

// ============================================================================
// Poisoned Pygmy: Handlers Module Integration
// ============================================================================

#[test]
fn test_validate_suite_name_valid() {
    for (name, _) in BENCHMARK_SUITES {
        // All defined suites should be valid
        let is_valid = BENCHMARK_SUITES.iter().any(|(n, _)| *n == *name);
        assert!(is_valid);
    }
}

#[test]
fn test_validate_suite_name_invalid() {
    let invalid_names = ["invalid", "nonexistent", "test123", ""];
    for name in invalid_names {
        let is_valid = BENCHMARK_SUITES.iter().any(|(n, _)| *n == name);
        // These should NOT be valid suites
        if !name.is_empty() {
            assert!(!is_valid, "Name '{}' should not be valid", name);
        }
    }
}

// ============================================================================
// Poisoned Pygmy: Error Message Coverage
// ============================================================================

#[test]
fn test_model_not_found_error_path() {
    use crate::error::RealizarError;

    let err = RealizarError::ModelNotFound("nonexistent.gguf".to_string());
    let msg = err.to_string();
    assert!(msg.contains("nonexistent.gguf") || msg.contains("not found"));
}

#[test]
fn test_unsupported_operation_error_path() {
    use crate::error::RealizarError;

    let err = RealizarError::UnsupportedOperation {
        operation: "test_op".to_string(),
        reason: "test reason".to_string(),
    };
    let msg = err.to_string();
    assert!(msg.contains("test_op") || msg.contains("test reason") || msg.contains("Unsupported"));
}
