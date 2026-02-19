//! CLI Integration Tests for apr-cli
//!
//! Extreme TDD: Tests written BEFORE implementation.
//! All tests should FAIL initially (RED phase).
//!
//! Toyota Way: Jidoka - Build quality in through testing.

#![allow(clippy::unwrap_used)] // Tests can use unwrap
#![allow(deprecated)] // cargo_bin still works, just deprecated for custom build-dir

use assert_cmd::Command;
use predicates::prelude::*;
use std::io::Write;
use tempfile::NamedTempFile;

// ============================================================================
// Helper Functions
// ============================================================================

/// Create an apr command
fn apr() -> Command {
    Command::cargo_bin("apr").expect("Failed to find apr binary")
}

/// Create a minimal valid APR v2 file for testing
fn create_test_apr_file() -> NamedTempFile {
    use aprender::format::v2::{AprV2Metadata, AprV2Writer};

    let file = NamedTempFile::new().expect("Failed to create temp file");

    let mut metadata = AprV2Metadata::new("test");
    metadata.architecture = Some("llama".to_string());
    metadata.hidden_size = Some(8);
    metadata.vocab_size = Some(16);
    metadata.num_layers = Some(1);

    let mut writer = AprV2Writer::new(metadata);

    // Add a minimal tensor with non-zero data
    let data: Vec<f32> = (0..128).map(|i| (i as f32 + 1.0) * 0.01).collect();
    writer.add_f32_tensor("model.embed_tokens.weight", vec![16, 8], &data);

    let bytes = writer.write().expect("Failed to write APR v2");
    std::fs::write(file.path(), bytes).expect("write file");

    file
}

// ============================================================================
// QA-001: Help and Version (Falsification tests 1-5 foundation)
// ============================================================================

#[test]
fn test_qa_001_help_flag() {
    apr()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("apr"))
        .stdout(predicate::str::contains("inspect"))
        .stdout(predicate::str::contains("debug"))
        .stdout(predicate::str::contains("Usage").or(predicate::str::contains("USAGE")));
}

#[test]
fn test_qa_001_version_flag() {
    apr()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("apr"));
}

#[test]
fn test_qa_001_no_args_shows_help() {
    apr()
        .assert()
        .failure()
        .stderr(predicate::str::contains("USAGE").or(predicate::str::contains("Usage")));
}

// ============================================================================
// QA-006: Inspect Command - Basic (Falsification test 6-10)
// ============================================================================

#[test]
fn test_qa_006_inspect_help() {
    apr()
        .args(["inspect", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Inspect"))
        .stdout(predicate::str::contains("metadata"));
}

#[test]
fn test_qa_006_inspect_missing_file() {
    apr()
        .args(["inspect", "/nonexistent/file.apr"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not found").or(predicate::str::contains("No such file")));
}

#[test]
fn test_qa_006_inspect_invalid_magic() {
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(b"XXXX").unwrap(); // Wrong magic
    file.write_all(&[0u8; 28]).unwrap(); // Rest of header

    apr()
        .args(["inspect", file.path().to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Invalid").or(predicate::str::contains("magic")));
}

#[test]
fn test_qa_006_inspect_shows_model_type() {
    let file = create_test_apr_file();

    apr()
        .args(["inspect", file.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("Type:").or(predicate::str::contains("Model")));
}

#[test]
fn test_qa_006_inspect_shows_version() {
    let file = create_test_apr_file();

    apr()
        .args(["inspect", file.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("Version").or(predicate::str::contains("2.0")));
}

#[test]
fn test_qa_006_inspect_json_output() {
    let file = create_test_apr_file();

    apr()
        .args(["inspect", file.path().to_str().unwrap(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("{"))
        .stdout(predicate::str::contains("}"));
}

// ============================================================================
// QA-011: Debug Command (Falsification test 11-15)
// ============================================================================

#[test]
fn test_qa_011_debug_help() {
    apr()
        .args(["debug", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("debug"))
        .stdout(predicate::str::contains("drama").or(predicate::str::contains("Drama")));
}

#[test]
fn test_qa_011_debug_basic() {
    let file = create_test_apr_file();

    apr()
        .args(["debug", file.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("APR").or(predicate::str::contains("apr")));
}

#[test]
fn test_qa_011_debug_drama_mode() {
    let file = create_test_apr_file();

    apr()
        .args(["debug", file.path().to_str().unwrap(), "--drama"])
        .assert()
        .success()
        .stdout(predicate::str::contains("DRAMA").or(predicate::str::contains("ACT")));
}

#[test]
fn test_qa_011_debug_hex_mode() {
    let file = create_test_apr_file();

    apr()
        .args(["debug", file.path().to_str().unwrap(), "--hex"])
        .assert()
        .success()
        .stdout(predicate::str::contains("41505200").or(predicate::str::contains("APR")));
}

#[test]
fn test_qa_011_debug_strings_mode() {
    let file = create_test_apr_file();

    apr()
        .args(["debug", file.path().to_str().unwrap(), "--strings"])
        .assert()
        .success()
        .stdout(predicate::str::contains("llama").or(predicate::str::contains("model")));
}

// ============================================================================
// QA-016: Validate Command (Quality scoring)
// ============================================================================

#[test]
fn test_qa_016_validate_help() {
    apr()
        .args(["validate", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Validate"));
}

#[test]
fn test_qa_016_validate_basic() {
    let file = create_test_apr_file();

    apr()
        .args(["validate", file.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("PASS").or(predicate::str::contains("valid")));
}

#[test]
fn test_qa_016_validate_corrupted() {
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(b"APRN").unwrap();
    file.write_all(&[0u8; 10]).unwrap(); // Truncated header

    apr()
        .args(["validate", file.path().to_str().unwrap()])
        .assert()
        .failure()
        .stdout(
            predicate::str::contains("FAIL")
                .or(predicate::str::contains("INVALID"))
                .or(predicate::str::contains("Invalid")),
        );
}

#[test]
fn test_qa_016_validate_quality_score() {
    let file = create_test_apr_file();

    apr()
        .args(["validate", file.path().to_str().unwrap(), "--quality"])
        .assert()
        .success()
        .stdout(predicate::str::contains("/100").or(predicate::str::contains("points")));
}

// ============================================================================
// QA-021: Diff Command (Falsification test 21-25)
// ============================================================================

#[test]
fn test_qa_021_diff_help() {
    apr()
        .args(["diff", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Compare").or(predicate::str::contains("diff")));
}

#[test]
fn test_qa_021_diff_identical() {
    let file = create_test_apr_file();
    let path = file.path().to_str().unwrap();

    apr().args(["diff", path, path]).assert().success().stdout(
        predicate::str::contains("100%")
            .or(predicate::str::contains("identical"))
            .or(predicate::str::contains("IDENTICAL")),
    );
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_error_invalid_subcommand() {
    apr()
        .arg("notacommand")
        .assert()
        .failure()
        .stderr(predicate::str::contains("error").or(predicate::str::contains("invalid")));
}

#[test]
fn test_error_inspect_directory() {
    apr().args(["inspect", "/tmp"]).assert().failure().stderr(
        predicate::str::contains("directory")
            .or(predicate::str::contains("not a file"))
            .or(predicate::str::contains("Not a file"))
            .or(predicate::str::contains("Invalid")),
    );
}

// ============================================================================
// QA-026: Tensors Command (Falsification tests 26-30)
// ============================================================================

#[test]
fn test_qa_026_tensors_help() {
    apr()
        .args(["tensors", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("tensor").or(predicate::str::contains("Tensor")));
}

#[test]
fn test_qa_026_tensors_basic() {
    let file = create_test_apr_file();

    apr()
        .args(["tensors", file.path().to_str().unwrap()])
        .assert()
        .success();
}

#[test]
fn test_qa_026_tensors_json_output() {
    let file = create_test_apr_file();

    apr()
        .args(["tensors", file.path().to_str().unwrap(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("{"))
        .stdout(predicate::str::contains("tensor_count"));
}

#[test]
fn test_qa_026_tensors_missing_file() {
    apr()
        .args(["tensors", "/nonexistent/model.apr"])
        .assert()
        .failure();
}

// ============================================================================
// Performance Tests (Falsification test 10)
// ============================================================================

#[test]
#[ignore = "Flaky latency test - fails under CI/coverage load"]
fn test_perf_inspect_latency() {
    use std::time::Instant;

    let file = create_test_apr_file();

    // Warmup
    apr()
        .args(["inspect", file.path().to_str().unwrap()])
        .assert()
        .success();

    let start = Instant::now();
    apr()
        .args(["inspect", file.path().to_str().unwrap()])
        .assert()
        .success();
    let elapsed = start.elapsed();

    // Per spec: inspect should complete in < 100ms
    assert!(
        elapsed.as_millis() < 500, // Allow 500ms for CI overhead
        "Inspect took {}ms, should be <500ms",
        elapsed.as_millis()
    );
}

// ============================================================================
// Helper: Create APR1 format file for hex/tree/flow tests
// ============================================================================

/// Create a valid APR1 format file with test tensors
fn create_apr1_test_file() -> NamedTempFile {
    use aprender::format::v2::{AprV2Metadata, AprV2Writer};

    let file = NamedTempFile::new().expect("Failed to create temp file");

    let mut metadata = AprV2Metadata::new("test");
    metadata.architecture = Some("whisper".to_string());
    metadata.hidden_size = Some(64);
    metadata.vocab_size = Some(64);
    metadata.num_layers = Some(2);

    let mut writer = AprV2Writer::new(metadata);

    // Add test tensors mimicking Whisper structure
    writer.add_f32_tensor(
        "encoder.layers.0.self_attn.q_proj.weight",
        vec![64, 64],
        &vec![0.01; 64 * 64],
    );
    writer.add_f32_tensor(
        "encoder.layers.0.self_attn.k_proj.weight",
        vec![64, 64],
        &vec![0.02; 64 * 64],
    );
    writer.add_f32_tensor(
        "decoder.layers.0.cross_attn.q_proj.weight",
        vec![64, 64],
        &vec![0.03; 64 * 64],
    );
    writer.add_f32_tensor(
        "decoder.layers.0.cross_attn.k_proj.weight",
        vec![64, 64],
        &vec![0.04; 64 * 64],
    );

    let bytes = writer.write().expect("Failed to write APR v2 test file");
    std::fs::write(file.path(), bytes).expect("write file");

    file
}

// ============================================================================
// GH-122: Hex Command Tests
// ============================================================================

#[test]
fn test_gh122_hex_help() {
    apr()
        .args(["hex", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("forensics"))
        .stdout(predicate::str::contains("tensor"));
}

include!("includes/cli_integration_part_02.rs");
include!("includes/cli_integration_part_03.rs");
include!("includes/cli_integration_part_04.rs");
