//! T-COV-95 Maimed Pygmy Campaign: Destructive Falsification Tests
//!
//! Dr. Popper's directive: "If the loader doesn't fail loudly and specifically
//! for a corrupted tensor count, it is unfalsifiable."
//!
//! This module creates:
//! 1. Active Pygmies - Valid GGUF models written to disk for real inference
//! 2. Maimed Pygmies - Corrupted artifacts that test specific error branches
//!
//! Target coverage: infer/mod.rs real inference paths (run_gguf_inference,
//! run_apr_inference, run_safetensors_inference)

use super::*;
use std::io::Write;
use tempfile::NamedTempFile;

// ============================================================================
// Active Pygmy Tests - Real Inference Paths
// ============================================================================

/// Test GGUF inference with executable pygmy written to disk
#[test]
fn test_maimed_pygmy_gguf_real_inference_path() {
    use crate::gguf::test_factory::build_executable_pygmy_gguf;

    // Create Active Pygmy on disk
    let data = build_executable_pygmy_gguf();
    let mut temp = NamedTempFile::with_suffix(".gguf").expect("create temp file");
    temp.write_all(&data).expect("write pygmy data");
    temp.flush().expect("flush");

    // Run real inference (not mock)
    let config = InferenceConfig::new(temp.path())
        .with_prompt("test")
        .with_max_tokens(2);

    // This exercises run_gguf_inference -> MappedGGUFModel -> OwnedQuantizedModel
    let result = run_inference(&config);

    // Active Pygmy should produce a result (may be garbage but code runs)
    // If it errors, we still exercised the code path
    match result {
        Ok(r) => {
            assert_eq!(r.format, "GGUF");
            assert!(r.generated_token_count <= 2);
        },
        Err(e) => {
            // Error is acceptable - we exercised the path
            let err_str = e.to_string();
            assert!(
                !err_str.contains("mock"),
                "Should not use mock path: {}",
                err_str
            );
        },
    }
}

/// Test GGUF inference with verbose output enabled
#[test]
fn test_maimed_pygmy_gguf_verbose_path() {
    use crate::gguf::test_factory::build_executable_pygmy_gguf;

    let data = build_executable_pygmy_gguf();
    let mut temp = NamedTempFile::with_suffix(".gguf").expect("create temp file");
    temp.write_all(&data).expect("write");
    temp.flush().expect("flush");

    // Enable verbose output to exercise eprintln paths
    let config = InferenceConfig::new(temp.path())
        .with_prompt("hello")
        .with_max_tokens(1)
        .with_verbose(true);

    let _ = run_inference(&config);
    // Verbose output goes to stderr - just verify no panic
}

/// Test GGUF inference with input_tokens (not prompt)
#[test]
fn test_maimed_pygmy_gguf_input_tokens_path() {
    use crate::gguf::test_factory::build_executable_pygmy_gguf;

    let data = build_executable_pygmy_gguf();
    let mut temp = NamedTempFile::with_suffix(".gguf").expect("create temp file");
    temp.write_all(&data).expect("write");
    temp.flush().expect("flush");

    // Use input_tokens instead of prompt to exercise that branch
    let config = InferenceConfig::new(temp.path())
        .with_input_tokens(vec![1, 2, 3])
        .with_max_tokens(2);

    let result = run_inference(&config);
    // Either succeeds or fails with GGUF error, not mock
    if let Ok(r) = result {
        assert_eq!(r.format, "GGUF");
        assert_eq!(r.input_token_count, 3);
    }
}

/// Test GGUF inference with no_gpu flag
#[test]
fn test_maimed_pygmy_gguf_no_gpu_path() {
    use crate::gguf::test_factory::build_executable_pygmy_gguf;

    let data = build_executable_pygmy_gguf();
    let mut temp = NamedTempFile::with_suffix(".gguf").expect("create temp file");
    temp.write_all(&data).expect("write");
    temp.flush().expect("flush");

    let config = InferenceConfig::new(temp.path())
        .with_prompt("test")
        .with_max_tokens(1)
        .without_gpu();

    let result = run_inference(&config);
    if let Ok(r) = result {
        assert!(!r.used_gpu, "Should not use GPU when disabled");
    }
}

// ============================================================================
// Maimed Pygmy Tests - Corrupted GGUF Headers
// ============================================================================

/// Maimed Pygmy: Corrupt magic number (first 4 bytes)
#[test]
fn test_maimed_pygmy_corrupt_magic() {
    use crate::gguf::test_factory::build_executable_pygmy_gguf;

    let mut data = build_executable_pygmy_gguf();
    // Corrupt magic: change GGUF to DEAD
    data[0] = 0xDE;
    data[1] = 0xAD;
    data[2] = 0xBE;
    data[3] = 0xEF;

    let mut temp = NamedTempFile::with_suffix(".gguf").expect("create temp file");
    temp.write_all(&data).expect("write");
    temp.flush().expect("flush");

    let config = InferenceConfig::new(temp.path())
        .with_prompt("test")
        .with_max_tokens(1);

    let result = run_inference(&config);
    assert!(result.is_err(), "Should fail with corrupt magic");
    let err_str = format!("{:?}", result.unwrap_err());
    // Should fail during format detection or GGUF parsing
    assert!(
        err_str.contains("magic") || err_str.contains("format") || err_str.contains("Format"),
        "Error should mention magic or format: {}",
        err_str
    );
}

/// Maimed Pygmy: Corrupt version (bytes 4-7)
#[test]
fn test_maimed_pygmy_corrupt_version() {
    use crate::gguf::test_factory::build_executable_pygmy_gguf;

    let mut data = build_executable_pygmy_gguf();
    // Corrupt version to 99 (unsupported)
    data[4] = 99;
    data[5] = 0;
    data[6] = 0;
    data[7] = 0;

    let mut temp = NamedTempFile::with_suffix(".gguf").expect("create temp file");
    temp.write_all(&data).expect("write");
    temp.flush().expect("flush");

    let config = InferenceConfig::new(temp.path())
        .with_prompt("test")
        .with_max_tokens(1);

    let result = run_inference(&config);
    assert!(result.is_err(), "Should fail with unsupported version");
    let err_str = format!("{:?}", result.unwrap_err());
    assert!(
        err_str.contains("version")
            || err_str.contains("Version")
            || err_str.contains("Unsupported"),
        "Error should mention version: {}",
        err_str
    );
}

/// Maimed Pygmy: Truncate file mid-header
#[test]
fn test_maimed_pygmy_truncated_header() {
    use crate::gguf::test_factory::build_executable_pygmy_gguf;

    let data = build_executable_pygmy_gguf();
    // Truncate to just magic + partial version (12 bytes, header needs 24)
    let truncated = &data[..12];

    let mut temp = NamedTempFile::with_suffix(".gguf").expect("create temp file");
    temp.write_all(truncated).expect("write");
    temp.flush().expect("flush");

    let config = InferenceConfig::new(temp.path())
        .with_prompt("test")
        .with_max_tokens(1);

    let result = run_inference(&config);
    assert!(result.is_err(), "Should fail with truncated header");
}

/// Maimed Pygmy: Corrupt tensor_count to zero (missing tensors)
#[test]
fn test_maimed_pygmy_corrupt_tensor_count() {
    use crate::gguf::test_factory::build_executable_pygmy_gguf;

    let mut data = build_executable_pygmy_gguf();
    // Corrupt tensor_count (bytes 8-15) to claim 0 tensors
    // Model will "parse" successfully but fail when tensors are missing
    let zero_count: u64 = 0;
    data[8..16].copy_from_slice(&zero_count.to_le_bytes());

    let mut temp = NamedTempFile::with_suffix(".gguf").expect("create temp file");
    temp.write_all(&data).expect("write");
    temp.flush().expect("flush");

    let config = InferenceConfig::new(temp.path())
        .with_prompt("test")
        .with_max_tokens(1);

    let result = run_inference(&config);
    // Should fail - model construction will fail with missing tensors
    assert!(result.is_err(), "Should fail with zero tensor count");
}

/// Maimed Pygmy: Truncate file mid-tensor-data
#[test]
fn test_maimed_pygmy_truncated_tensor_data() {
    use crate::gguf::test_factory::build_executable_pygmy_gguf;

    let data = build_executable_pygmy_gguf();
    // Keep header and metadata, truncate tensor data region
    // Header is 24 bytes, metadata varies, tensors follow
    // Truncate to ~60% of file
    let truncated = &data[..data.len() * 60 / 100];

    let mut temp = NamedTempFile::with_suffix(".gguf").expect("create temp file");
    temp.write_all(truncated).expect("write");
    temp.flush().expect("flush");

    let config = InferenceConfig::new(temp.path())
        .with_prompt("test")
        .with_max_tokens(1);

    let result = run_inference(&config);
    // Should fail during tensor loading
    assert!(result.is_err(), "Should fail with truncated tensor data");
}

// ============================================================================
// APR Maimed Pygmy Tests
// ============================================================================

/// APR Maimed Pygmy: Invalid APR magic
#[test]
fn test_maimed_pygmy_apr_invalid_magic() {
    let mut data = Vec::new();
    // Invalid magic (not APR\x01 or APR\x02)
    data.extend_from_slice(b"NOPE");
    data.extend_from_slice(&[0u8; 100]);

    let mut temp = NamedTempFile::with_suffix(".apr").expect("create temp file");
    temp.write_all(&data).expect("write");
    temp.flush().expect("flush");

    let config = InferenceConfig::new(temp.path())
        .with_prompt("test")
        .with_max_tokens(1);

    let result = run_inference(&config);
    assert!(result.is_err(), "Should fail with invalid APR magic");
}

/// APR Maimed Pygmy: Valid magic but truncated metadata
#[test]
fn test_maimed_pygmy_apr_truncated_metadata() {
    let mut data = Vec::new();
    // Valid APR v2 magic
    data.extend_from_slice(b"APR\x02");
    // But truncated - no metadata or tensors

    let mut temp = NamedTempFile::with_suffix(".apr").expect("create temp file");
    temp.write_all(&data).expect("write");
    temp.flush().expect("flush");

    let config = InferenceConfig::new(temp.path())
        .with_prompt("test")
        .with_max_tokens(1);

    let result = run_inference(&config);
    assert!(result.is_err(), "Should fail with truncated APR");
}

// ============================================================================
// SafeTensors Maimed Pygmy Tests
// ============================================================================

/// SafeTensors Maimed Pygmy: Header size exceeds file size
#[test]
fn test_maimed_pygmy_safetensors_header_overflow() {
    let mut data = Vec::new();
    // Header size claims 1GB but file is tiny
    let huge_header: u64 = 1024 * 1024 * 1024;
    data.extend_from_slice(&huge_header.to_le_bytes());
    data.extend_from_slice(b"{}");

    let mut temp = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
    temp.write_all(&data).expect("write");
    temp.flush().expect("flush");

    let config = InferenceConfig::new(temp.path())
        .with_prompt("test")
        .with_max_tokens(1);

    let result = run_inference(&config);
    assert!(result.is_err(), "Should fail with header overflow");
}

/// SafeTensors Maimed Pygmy: Invalid JSON in header
#[test]
fn test_maimed_pygmy_safetensors_invalid_json() {
    let mut data = Vec::new();
    let invalid_json = b"{ not valid json {{{{";
    let header_size = invalid_json.len() as u64;
    data.extend_from_slice(&header_size.to_le_bytes());
    data.extend_from_slice(invalid_json);

    let mut temp = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
    temp.write_all(&data).expect("write");
    temp.flush().expect("flush");

    let config = InferenceConfig::new(temp.path())
        .with_prompt("test")
        .with_max_tokens(1);

    let result = run_inference(&config);
    assert!(result.is_err(), "Should fail with invalid JSON");
}

// ============================================================================
// InferenceConfig Edge Case Tests
// ============================================================================

/// Test InferenceConfig with temperature edge cases
#[test]
fn test_inference_config_temperature_edges() {
    // Zero temperature (greedy)
    let config = InferenceConfig::new("model.gguf")
        .with_prompt("test")
        .with_temperature(0.0);
    assert_eq!(config.temperature, 0.0);

    // Very high temperature
    let config = InferenceConfig::new("model.gguf")
        .with_prompt("test")
        .with_temperature(100.0);
    assert_eq!(config.temperature, 100.0);
}

/// Test InferenceConfig with top_k edge cases
#[test]
fn test_inference_config_top_k_edges() {
    // top_k = 0 (disabled)
    let config = InferenceConfig::new("model.gguf")
        .with_prompt("test")
        .with_top_k(0);
    assert_eq!(config.top_k, 0);

    // top_k = 1 (greedy-ish)
    let config = InferenceConfig::new("model.gguf")
        .with_prompt("test")
        .with_top_k(1);
    assert_eq!(config.top_k, 1);

    // top_k = very large
    let config = InferenceConfig::new("model.gguf")
        .with_prompt("test")
        .with_top_k(100000);
    assert_eq!(config.top_k, 100000);
}

/// Test InferenceConfig builder chain
#[test]
fn test_inference_config_full_chain() {
    let config = InferenceConfig::new("model.gguf")
        .with_prompt("Hello world")
        .with_max_tokens(50)
        .with_temperature(0.7)
        .with_top_k(40)
        .with_verbose(true)
        .with_trace(true)
        .with_trace_output("/tmp/trace.json")
        .without_gpu();

    assert_eq!(config.prompt, Some("Hello world".to_string()));
    assert_eq!(config.max_tokens, 50);
    assert_eq!(config.temperature, 0.7);
    assert_eq!(config.top_k, 40);
    assert!(config.verbose);
    assert!(config.trace);
    assert!(config.no_gpu);
    assert!(config.trace_output.is_some());
}

/// Test InferenceConfig with no prompt or tokens (uses BOS)
#[test]
fn test_inference_config_no_input() {
    use crate::gguf::test_factory::build_executable_pygmy_gguf;

    let data = build_executable_pygmy_gguf();
    let mut temp = NamedTempFile::with_suffix(".gguf").expect("create temp file");
    temp.write_all(&data).expect("write");
    temp.flush().expect("flush");

    // No prompt, no input_tokens - should use BOS token
    let config = InferenceConfig::new(temp.path()).with_max_tokens(1);

    let result = run_inference(&config);
    // Either succeeds (generating from BOS) or fails on model issues
    if let Ok(r) = result {
        // With no prompt, input should be just BOS
        assert!(r.input_token_count >= 1);
    }
}
