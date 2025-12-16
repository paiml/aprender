//! CLI Integration Tests for apr-cli
//!
//! Extreme TDD: Tests written BEFORE implementation.
//! All tests should FAIL initially (RED phase).
//!
//! Toyota Way: Jidoka - Build quality in through testing.

#![allow(clippy::unwrap_used)] // Tests can use unwrap

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

/// Create a minimal valid .apr file for testing
fn create_test_apr_file() -> NamedTempFile {
    use std::collections::HashMap;

    // We'll create a simple model and save it
    let file = NamedTempFile::new().expect("Failed to create temp file");
    let path = file.path();

    // Use aprender to create a real .apr file
    // For now, create a minimal binary that looks like an APR file
    let mut f = std::fs::File::create(path).expect("create file");

    // Magic: APRN
    f.write_all(b"APRN").unwrap();
    // Version: 1.0
    f.write_all(&[1, 0]).unwrap();
    // Model type: Custom (0x00FF)
    f.write_all(&[0xFF, 0x00]).unwrap();
    // Metadata size: 64 bytes (little endian)
    f.write_all(&64u32.to_le_bytes()).unwrap();
    // Payload size: 32 bytes
    f.write_all(&32u32.to_le_bytes()).unwrap();
    // Uncompressed size: 32 bytes
    f.write_all(&32u32.to_le_bytes()).unwrap();
    // Compression: None (0x00)
    f.write_all(&[0x00]).unwrap();
    // Flags: None (0x00)
    f.write_all(&[0x00]).unwrap();
    // Reserved: 6 bytes
    f.write_all(&[0u8; 6]).unwrap();

    // Minimal MessagePack metadata (empty map)
    let metadata = rmp_serde::to_vec_named(&HashMap::<String, String>::new()).unwrap();
    // Pad to 64 bytes
    let mut meta_padded = metadata.clone();
    meta_padded.resize(64, 0);
    f.write_all(&meta_padded).unwrap();

    // Minimal payload (32 bytes of zeros)
    f.write_all(&[0u8; 32]).unwrap();

    // CRC32 checksum (placeholder)
    f.write_all(&[0xDE, 0xAD, 0xBE, 0xEF]).unwrap();

    drop(f);
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
        .stdout(predicate::str::contains("Version").or(predicate::str::contains("1.0")));
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
        .stdout(predicate::str::contains("APRN").or(predicate::str::contains("apr")));
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
        .stdout(predicate::str::contains("4150524e").or(predicate::str::contains("APRN")));
}

#[test]
fn test_qa_011_debug_strings_mode() {
    let file = create_test_apr_file();

    apr()
        .args(["debug", file.path().to_str().unwrap(), "--strings"])
        .assert()
        .success()
        .stdout(predicate::str::contains("APRN"));
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
        .stdout(predicate::str::contains("FAIL").or(predicate::str::contains("incomplete")));
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
