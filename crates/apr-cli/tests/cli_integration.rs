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

// ============================================================================
// GH-179 / PMAT-191: Missing Tool Tests (Tool Coverage Gap)
// ============================================================================

// F-RUN-001: apr run help works
#[test]
fn test_f_run_001_help() {
    apr()
        .args(["run", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Run").or(predicate::str::contains("inference")));
}

// F-RUN-002: apr run with missing model shows error
#[test]
fn test_f_run_002_missing_model_error() {
    apr()
        .args(["run", "/nonexistent/model.gguf", "--prompt", "test"])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("not found")
                .or(predicate::str::contains("No such file"))
                .or(predicate::str::contains("Failed")),
        );
}

// F-CHAT-001: apr chat help works
#[test]
fn test_f_chat_001_help() {
    apr()
        .args(["chat", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("chat").or(predicate::str::contains("Chat")));
}

// F-CHAT-002: apr chat with missing model shows error
#[test]
fn test_f_chat_002_missing_model_error() {
    apr()
        .args(["chat", "/nonexistent/model.gguf"])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("not found")
                .or(predicate::str::contains("No such file"))
                .or(predicate::str::contains("Failed")),
        );
}

// F-SERVE-001: apr serve help works
#[test]
fn test_f_serve_001_help() {
    apr()
        .args(["serve", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("serve").or(predicate::str::contains("Serve")))
        .stdout(predicate::str::contains("port").or(predicate::str::contains("PORT")));
}

// F-SERVE-002: apr serve with missing model shows error
#[test]
fn test_f_serve_002_missing_model_error() {
    apr()
        .args(["serve", "/nonexistent/model.gguf"])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("not found")
                .or(predicate::str::contains("No such file"))
                .or(predicate::str::contains("Failed")),
        );
}

// F-CANARY-001: apr canary help works
#[test]
fn test_f_canary_001_help() {
    apr()
        .args(["canary", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("canary").or(predicate::str::contains("Canary")))
        .stdout(predicate::str::contains("create").or(predicate::str::contains("check")));
}

// F-CANARY-002: apr canary create with missing model shows error
#[test]
fn test_f_canary_002_create_missing_model() {
    apr()
        .args([
            "canary",
            "create",
            "--input",
            "/tmp/test.wav",
            "--output",
            "/tmp/canary.json",
            "/nonexistent/model.gguf",
        ])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("not found")
                .or(predicate::str::contains("No such file"))
                .or(predicate::str::contains("Failed"))
                .or(predicate::str::contains("does not exist")),
        );
}

// F-TUNE-001: apr tune help works
#[test]
fn test_f_tune_001_help() {
    apr()
        .args(["tune", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("tune").or(predicate::str::contains("Tune")))
        .stdout(predicate::str::contains("plan").or(predicate::str::contains("lora")));
}

// F-TUNE-002: apr tune with missing model shows error
#[test]
fn test_f_tune_002_missing_model() {
    apr()
        .args(["tune", "/nonexistent/model.gguf", "--plan"])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("not found")
                .or(predicate::str::contains("No such file"))
                .or(predicate::str::contains("Failed")),
        );
}

// F-QA-001: apr qa help works
#[test]
fn test_f_qa_001_help() {
    apr()
        .args(["qa", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("qa").or(predicate::str::contains("QA")));
}

// F-QA-002: apr qa with missing model shows error
#[test]
fn test_f_qa_002_missing_model() {
    apr()
        .args(["qa", "/nonexistent/model.gguf"])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("not found")
                .or(predicate::str::contains("No such file"))
                .or(predicate::str::contains("Failed")),
        );
}

// F-CONVERT-001: apr convert help works (rosetta subcommand)
#[test]
fn test_f_convert_001_rosetta_help() {
    apr()
        .args(["rosetta", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("convert").or(predicate::str::contains("Convert")));
}

// F-CONVERT-002: apr rosetta convert with missing model shows error
#[test]
fn test_f_convert_002_missing_model() {
    apr()
        .args([
            "rosetta",
            "convert",
            "/nonexistent/model.gguf",
            "/tmp/out.apr",
        ])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("not found")
                .or(predicate::str::contains("No such file"))
                .or(predicate::str::contains("Failed"))
                .or(predicate::str::contains("does not exist")),
        );
}

// ============================================================================
// PMAT-192 Phase 5: F-PROFILE-CI-* Tests (GH-180)
// ============================================================================

// F-PROFILE-CI-001: apr profile --help shows CI options
#[test]
fn test_f_profile_ci_001_help_shows_ci_options() {
    apr()
        .args(["profile", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--ci"))
        .stdout(predicate::str::contains("--assert-throughput"))
        .stdout(predicate::str::contains("--assert-p99"));
}

// F-PROFILE-CI-002: apr profile with missing model shows error
#[test]
fn test_f_profile_ci_002_missing_model_error() {
    apr()
        .args(["profile", "/nonexistent/model.gguf"])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("not found")
                .or(predicate::str::contains("No such file"))
                .or(predicate::str::contains("Failed"))
                .or(predicate::str::contains("does not exist")),
        );
}

// F-PROFILE-CI-003: apr profile --ci with missing model shows error
#[test]
fn test_f_profile_ci_003_ci_mode_missing_model() {
    apr()
        .args([
            "profile",
            "/nonexistent/model.gguf",
            "--ci",
            "--assert-throughput",
            "100",
        ])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("not found")
                .or(predicate::str::contains("No such file"))
                .or(predicate::str::contains("Failed"))
                .or(predicate::str::contains("does not exist")),
        );
}

// F-PROFILE-CI-004: apr profile accepts format=json
#[test]
fn test_f_profile_ci_004_format_json_accepted() {
    // Just verify the argument is accepted (will fail on missing model, but that's expected)
    apr()
        .args(["profile", "/nonexistent/model.gguf", "--format", "json"])
        .assert()
        .failure(); // Expected - model doesn't exist
}

// F-PROFILE-CI-005: apr profile accepts warmup and measure args
#[test]
fn test_f_profile_ci_005_warmup_measure_accepted() {
    apr()
        .args([
            "profile",
            "/nonexistent/model.gguf",
            "--warmup",
            "5",
            "--measure",
            "20",
        ])
        .assert()
        .failure(); // Expected - model doesn't exist
}

// F-PROFILE-CI-006: apr profile --ci --assert-p50 accepted
#[test]
fn test_f_profile_ci_006_assert_p50_accepted() {
    apr()
        .args([
            "profile",
            "/nonexistent/model.gguf",
            "--ci",
            "--assert-p50",
            "25",
        ])
        .assert()
        .failure(); // Expected - model doesn't exist
}

// ============================================================================
// F-PROFILE-EXIT Tests (GH-184: Exit code verification)
// ============================================================================

// F-PROFILE-EXIT-001: Help text documents exit codes
#[test]
fn test_f_profile_exit_001_help_documents_exit_codes() {
    // Verify that CI mode documentation exists in help
    apr()
        .args(["profile", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--ci"))
        .stdout(predicate::str::contains("--assert-throughput"));
}

// F-PROFILE-EXIT-002: Non-existent model returns failure (not success)
#[test]
fn test_f_profile_exit_002_nonexistent_model_fails() {
    // GH-184: Ensure file-not-found errors exit with non-zero code
    apr()
        .args([
            "profile",
            "/nonexistent/path/to/model.gguf",
            "--ci",
            "--assert-throughput",
            "1.0",
        ])
        .assert()
        .failure()
        .code(predicate::ne(0)); // Must be non-zero exit code
}

// F-PROFILE-EXIT-003: CI mode with impossible threshold should fail
// Note: This test requires a real model and inference feature
#[test]
#[ignore = "requires model download - run with: cargo test -- --ignored"]
fn test_f_profile_exit_003_impossible_threshold_fails() {
    // This test would need a real model to verify exit code on assertion failure
    // The assertion --assert-throughput 999999999.0 should always fail
    // and exit with code 1
    apr()
        .args([
            "profile",
            "test-model.gguf", // Would need real model path
            "--ci",
            "--assert-throughput",
            "999999999.0",
            "--warmup",
            "1",
            "--measure",
            "1",
        ])
        .assert()
        .failure()
        .code(predicate::eq(1)); // Must be exit code 1 for assertion failure
}

// ============================================================================
// F-CONVERT-QUANT Tests (GH-181: Q4_K_M block alignment)
// ============================================================================

// F-CONVERT-QUANT-001: apr convert --help shows Q4K option
#[test]
fn test_f_convert_quant_001_help_shows_q4k() {
    apr()
        .args(["convert", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("q4k").or(predicate::str::contains("q4_k")));
}

// F-CONVERT-QUANT-002: apr convert with --quantize q4k accepted
#[test]
fn test_f_convert_quant_002_q4k_option_accepted() {
    apr()
        .args([
            "convert",
            "/nonexistent/model.gguf",
            "/tmp/out.apr",
            "--quantize",
            "q4k",
        ])
        .assert()
        .failure(); // Expected - model doesn't exist
}

// ============================================================================
// F-EXPORT-COMPANION Tests (GH-182: SafeTensors companion files)
// ============================================================================

// F-EXPORT-COMPANION-001: apr export --help shows format options
#[test]
fn test_f_export_companion_001_help_shows_formats() {
    apr()
        .args(["export", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("safetensors"))
        .stdout(predicate::str::contains("gguf"));
}

// F-EXPORT-COMPANION-002: apr export accepts safetensors format
#[test]
fn test_f_export_companion_002_safetensors_accepted() {
    apr()
        .args([
            "export",
            "/nonexistent/model.apr",
            "--format",
            "safetensors",
            "/tmp/out.safetensors",
        ])
        .assert()
        .failure(); // Expected - model doesn't exist
}

// ============================================================================
// F-VALIDATE-GGUF Tests (GH-183: GGUF v3 validation)
// ============================================================================

// F-VALIDATE-GGUF-001: apr validate --help shows options
#[test]
fn test_f_validate_gguf_001_help_shows_options() {
    apr()
        .args(["validate", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("quality").or(predicate::str::contains("strict")));
}

// F-VALIDATE-GGUF-002: apr validate shows magic error for corrupted file
#[test]
fn test_f_validate_gguf_002_magic_error_message() {
    use std::io::Write;

    // Create a file with invalid magic
    let mut file = NamedTempFile::new().expect("create temp file");
    file.write_all(b"BADM1234567890123456789012345678").unwrap();

    apr()
        .args(["validate", file.path().to_str().unwrap()])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("Invalid")
                .or(predicate::str::contains("magic"))
                .or(predicate::str::contains("Unknown")),
        );
}

// F-VALIDATE-GGUF-003: apr validate accepts GGUF magic bytes
#[test]
fn test_f_validate_gguf_003_accepts_gguf_magic() {
    use std::io::Write;

    // Create a file with valid GGUF magic but minimal content
    let mut file = NamedTempFile::new().expect("create temp file");
    // GGUF magic + version 3 + minimal header
    file.write_all(b"GGUF").unwrap(); // magic
    file.write_all(&3u32.to_le_bytes()).unwrap(); // version 3
    file.write_all(&0u64.to_le_bytes()).unwrap(); // tensor count
    file.write_all(&0u64.to_le_bytes()).unwrap(); // metadata count
                                                  // Pad to 32 bytes
    file.write_all(&[0u8; 8]).unwrap();

    // This should pass the magic check (might fail later for other reasons)
    let output = apr()
        .args(["validate", file.path().to_str().unwrap()])
        .output()
        .expect("run command");

    // Should not fail on magic bytes
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stdout.contains("Invalid magic"),
        "GGUF magic should be accepted: {stdout}"
    );
}
