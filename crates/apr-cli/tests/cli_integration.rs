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

// ============================================================================
// Helper: Create APR1 format file for hex/tree/flow tests
// ============================================================================

/// Create a valid APR1 format file with test tensors
fn create_apr1_test_file() -> NamedTempFile {
    use aprender::serialization::apr::AprWriter;
    use serde_json::json;

    let file = NamedTempFile::new().expect("Failed to create temp file");

    let mut writer = AprWriter::new();

    // Model metadata
    writer.set_metadata("model_type", json!("test"));
    writer.set_metadata("n_layers", json!(2));

    // Add test tensors mimicking Whisper structure
    writer.add_tensor_f32(
        "encoder.layers.0.self_attn.q_proj.weight",
        vec![64, 64],
        &vec![0.01; 64 * 64],
    );
    writer.add_tensor_f32(
        "encoder.layers.0.self_attn.k_proj.weight",
        vec![64, 64],
        &vec![0.02; 64 * 64],
    );
    writer.add_tensor_f32(
        "decoder.layers.0.cross_attn.q_proj.weight",
        vec![64, 64],
        &vec![0.03; 64 * 64],
    );
    writer.add_tensor_f32(
        "decoder.layers.0.cross_attn.k_proj.weight",
        vec![64, 64],
        &vec![0.04; 64 * 64],
    );

    writer
        .write(file.path())
        .expect("Failed to write APR1 test file");

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
        .stdout(predicate::str::contains("Hex"))
        .stdout(predicate::str::contains("tensor"));
}

#[test]
fn test_gh122_hex_list_tensors() {
    let file = create_apr1_test_file();

    apr()
        .args(["hex", file.path().to_str().unwrap(), "--list"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "encoder.layers.0.self_attn.q_proj.weight",
        ))
        .stdout(predicate::str::contains("decoder.layers.0.cross_attn"));
}

#[test]
fn test_gh122_hex_with_filter() {
    let file = create_apr1_test_file();

    apr()
        .args([
            "hex",
            file.path().to_str().unwrap(),
            "--tensor",
            "cross_attn",
            "--list",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("cross_attn"))
        .stdout(predicate::str::contains("2 tensors").or(predicate::str::contains("tensors")));
}

#[test]
fn test_gh122_hex_with_stats() {
    let file = create_apr1_test_file();

    apr()
        .args([
            "hex",
            file.path().to_str().unwrap(),
            "--tensor",
            "encoder.layers.0.self_attn.q_proj.weight",
            "--stats",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("min="))
        .stdout(predicate::str::contains("max="))
        .stdout(predicate::str::contains("mean="))
        .stdout(predicate::str::contains("std="));
}

#[test]
fn test_gh122_hex_json_output() {
    let file = create_apr1_test_file();

    apr()
        .args([
            "hex",
            file.path().to_str().unwrap(),
            "--tensor",
            "encoder",
            "--json",
            "--stats",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"name\""))
        .stdout(predicate::str::contains("\"shape\""))
        .stdout(predicate::str::contains("\"stats\""));
}

#[test]
fn test_gh122_hex_dump_display() {
    let file = create_apr1_test_file();

    apr()
        .args([
            "hex",
            file.path().to_str().unwrap(),
            "--tensor",
            "encoder.layers.0.self_attn.q_proj.weight",
            "--limit",
            "8",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Hex dump"))
        .stdout(predicate::str::contains("00000000:")); // Hex offset
}

#[test]
fn test_gh122_hex_no_match() {
    let file = create_apr1_test_file();

    apr()
        .args([
            "hex",
            file.path().to_str().unwrap(),
            "--tensor",
            "nonexistent_tensor",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("No tensors match"));
}

// ============================================================================
// GH-122: Tree Command Tests
// ============================================================================

#[test]
fn test_gh122_tree_help() {
    apr()
        .args(["tree", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("tree"))
        .stdout(predicate::str::contains("format"));
}

#[test]
fn test_gh122_tree_ascii_default() {
    let file = create_apr1_test_file();

    apr()
        .args(["tree", file.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("encoder"))
        .stdout(predicate::str::contains("decoder"))
        .stdout(predicate::str::contains("tensors"));
}

#[test]
fn test_gh122_tree_with_sizes() {
    let file = create_apr1_test_file();

    apr()
        .args(["tree", file.path().to_str().unwrap(), "--sizes"])
        .assert()
        .success()
        .stdout(predicate::str::contains("KB").or(predicate::str::contains("MB")));
}

#[test]
fn test_gh122_tree_with_filter() {
    let file = create_apr1_test_file();

    apr()
        .args(["tree", file.path().to_str().unwrap(), "--filter", "encoder"])
        .assert()
        .success()
        .stdout(predicate::str::contains("encoder"));
}

#[test]
fn test_gh122_tree_depth_limit() {
    let file = create_apr1_test_file();

    apr()
        .args(["tree", file.path().to_str().unwrap(), "--depth", "2"])
        .assert()
        .success()
        .stdout(predicate::str::contains("encoder"))
        .stdout(predicate::str::contains("layers"));
}

#[test]
fn test_gh122_tree_mermaid_format() {
    let file = create_apr1_test_file();

    apr()
        .args(["tree", file.path().to_str().unwrap(), "--format", "mermaid"])
        .assert()
        .success()
        .stdout(predicate::str::contains("```mermaid"))
        .stdout(predicate::str::contains("graph TD"));
}

#[test]
fn test_gh122_tree_dot_format() {
    let file = create_apr1_test_file();

    apr()
        .args(["tree", file.path().to_str().unwrap(), "--format", "dot"])
        .assert()
        .success()
        .stdout(predicate::str::contains("digraph"))
        .stdout(predicate::str::contains("rankdir"));
}

#[test]
fn test_gh122_tree_json_format() {
    let file = create_apr1_test_file();

    apr()
        .args(["tree", file.path().to_str().unwrap(), "--format", "json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"name\""))
        .stdout(predicate::str::contains("\"children\""));
}

// ============================================================================
// GH-122: Flow Command Tests
// ============================================================================

#[test]
fn test_gh122_flow_help() {
    apr()
        .args(["flow", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("flow"))
        .stdout(predicate::str::contains("component"));
}

#[test]
fn test_gh122_flow_full_model() {
    let file = create_apr1_test_file();

    apr()
        .args(["flow", file.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("encoder-decoder").or(predicate::str::contains("Model")));
}

#[test]
fn test_gh122_flow_cross_attn() {
    let file = create_apr1_test_file();

    apr()
        .args([
            "flow",
            file.path().to_str().unwrap(),
            "--component",
            "cross_attn",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("CROSS-ATTENTION"))
        .stdout(predicate::str::contains("encoder_output"))
        .stdout(predicate::str::contains("softmax"));
}

#[test]
fn test_gh122_flow_self_attn() {
    let file = create_apr1_test_file();

    apr()
        .args([
            "flow",
            file.path().to_str().unwrap(),
            "--component",
            "self_attn",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("SELF-ATTENTION"));
}

#[test]
fn test_gh122_flow_encoder() {
    let file = create_apr1_test_file();

    apr()
        .args([
            "flow",
            file.path().to_str().unwrap(),
            "--component",
            "encoder",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("ENCODER"));
}

#[test]
fn test_gh122_flow_decoder() {
    let file = create_apr1_test_file();

    apr()
        .args([
            "flow",
            file.path().to_str().unwrap(),
            "--component",
            "decoder",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("DECODER"));
}

#[test]
fn test_gh122_flow_verbose() {
    let file = create_apr1_test_file();

    apr()
        .args(["flow", file.path().to_str().unwrap(), "--verbose"])
        .assert()
        .success();
}

#[test]
fn test_gh122_flow_with_layer_filter() {
    let file = create_apr1_test_file();

    apr()
        .args([
            "flow",
            file.path().to_str().unwrap(),
            "--layer",
            "decoder.layers.0",
        ])
        .assert()
        .success();
}
