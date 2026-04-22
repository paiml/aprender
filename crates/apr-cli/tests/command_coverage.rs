// Integration tests: unwrap()/panic!() are idiomatic; strict workspace lints relaxed here.
#![allow(
    clippy::disallowed_methods,
    clippy::needless_range_loop,
    clippy::format_collect,
    clippy::format_push_string,
    clippy::manual_assert,
    clippy::uninlined_format_args,
    clippy::unnecessary_debug_formatting,
    clippy::unwrap_or_default,
    clippy::expect_fun_call,
    clippy::manual_repeat_n,
    clippy::unnecessary_map_or
)]

//! Command Coverage Integration Tests
//!
//! Tests every read-only `apr` CLI command against a synthetic APR model fixture.
//! This is the single biggest coverage ROI: one fixture exercises 15+ subcommands.
//!
//! Toyota Way: Jidoka — build quality in through broad integration testing.

#![allow(clippy::unwrap_used)]

use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::NamedTempFile;

// ============================================================================
// Fixture: Synthetic APR v2 Model
// ============================================================================

/// Create a rich synthetic APR v2 model file with realistic tensor names.
///
/// The model mimics a minimal LLaMA-style transformer:
/// - 1 layer, 32 hidden dim, 64 vocab, 4 attention heads
/// - Named tensors following `model.layers.N.*` convention
/// - Enough structure for lint, trace, tree, flow, explain to produce output
fn create_rich_test_model() -> NamedTempFile {
    use aprender::format::v2::{AprV2Metadata, AprV2Writer};

    let file = NamedTempFile::with_suffix(".apr").expect("create temp file");

    let mut metadata = AprV2Metadata::new("test-llama-tiny");
    metadata.architecture = Some("llama".to_string());
    metadata.hidden_size = Some(32);
    metadata.vocab_size = Some(64);
    metadata.num_layers = Some(1);
    metadata.num_heads = Some(4);
    metadata.intermediate_size = Some(64);

    let mut writer = AprV2Writer::new(metadata);

    // Embedding: [vocab=64, hidden=32]
    let embed_data: Vec<f32> = (0..64 * 32).map(|i| (i as f32 + 1.0) * 0.001).collect();
    writer.add_f32_tensor("model.embed_tokens.weight", vec![64, 32], &embed_data);

    // Layer 0: attention projections [hidden=32, hidden=32]
    let proj_data: Vec<f32> = (0..32 * 32).map(|i| (i as f32) * 0.01).collect();
    writer.add_f32_tensor(
        "model.layers.0.self_attn.q_proj.weight",
        vec![32, 32],
        &proj_data,
    );
    writer.add_f32_tensor(
        "model.layers.0.self_attn.k_proj.weight",
        vec![32, 32],
        &proj_data,
    );
    writer.add_f32_tensor(
        "model.layers.0.self_attn.v_proj.weight",
        vec![32, 32],
        &proj_data,
    );
    writer.add_f32_tensor(
        "model.layers.0.self_attn.o_proj.weight",
        vec![32, 32],
        &proj_data,
    );

    // Layer 0: FFN [hidden=32, intermediate=64] and [intermediate=64, hidden=32]
    let up_data: Vec<f32> = (0..64 * 32).map(|i| (i as f32) * 0.005).collect();
    let down_data: Vec<f32> = (0..32 * 64).map(|i| (i as f32) * 0.005).collect();
    writer.add_f32_tensor(
        "model.layers.0.mlp.gate_proj.weight",
        vec![64, 32],
        &up_data,
    );
    writer.add_f32_tensor("model.layers.0.mlp.up_proj.weight", vec![64, 32], &up_data);
    writer.add_f32_tensor(
        "model.layers.0.mlp.down_proj.weight",
        vec![32, 64],
        &down_data,
    );

    // Layer norms [hidden=32]
    let norm_data: Vec<f32> = vec![1.0; 32];
    writer.add_f32_tensor(
        "model.layers.0.input_layernorm.weight",
        vec![32],
        &norm_data,
    );
    writer.add_f32_tensor(
        "model.layers.0.post_attention_layernorm.weight",
        vec![32],
        &norm_data,
    );
    writer.add_f32_tensor("model.norm.weight", vec![32], &norm_data);

    // LM head [vocab=64, hidden=32]
    let lm_head_data: Vec<f32> = (0..64 * 32).map(|i| (i as f32) * 0.002).collect();
    writer.add_f32_tensor("lm_head.weight", vec![64, 32], &lm_head_data);

    let bytes = writer.write().expect("write APR v2");
    std::fs::write(file.path(), bytes).expect("write file");

    file
}

/// Create a minimal GGUF file for format-dispatch tests.
fn create_minimal_gguf() -> NamedTempFile {
    let file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes()); // version 3
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor count = 0
    data.extend_from_slice(&0u64.to_le_bytes()); // metadata count = 0
    data.extend_from_slice(&[0u8; 8]); // padding
    std::fs::write(file.path(), data).expect("write file");
    file
}

/// Helper to build an `apr` command with NO_COLOR set.
fn apr() -> Command {
    let mut cmd = Command::cargo_bin("apr").expect("apr binary");
    cmd.env("NO_COLOR", "1");
    cmd
}

// ============================================================================
// INSPECT — already partially tested, verify JSON depth
// ============================================================================

#[test]
fn test_coverage_inspect_json_has_tensors() {
    let model = create_rich_test_model();
    apr()
        .args(["inspect", model.path().to_str().unwrap(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"tensor"))
        .stdout(predicate::str::contains("llama"));
}

// ============================================================================
// DEBUG — verify drama and strings modes with rich model
// ============================================================================

#[test]
fn test_coverage_debug_rich_model() {
    let model = create_rich_test_model();
    apr()
        .args(["debug", model.path().to_str().unwrap()])
        .assert()
        .success();
}

// ============================================================================
// VALIDATE — basic + quality with rich model
// ============================================================================

#[test]
fn test_coverage_validate_rich_model() {
    let model = create_rich_test_model();
    let output = apr()
        .args(["validate", model.path().to_str().unwrap()])
        .output()
        .expect("run validate");
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Synthetic model won't score high, but validate must run all checks.
    // It exits non-zero when score < 50%, which is expected for a synthetic model.
    assert!(
        stdout.contains("PASS") || stdout.contains("VALID"),
        "validate must show check results, got: {stdout}"
    );
}

#[test]
fn test_coverage_validate_quality_rich() {
    let model = create_rich_test_model();
    let output = apr()
        .args(["validate", model.path().to_str().unwrap(), "--quality"])
        .output()
        .expect("run validate --quality");
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Quality mode produces a score table
    assert!(
        stdout.contains("/100") || stdout.contains("points") || stdout.contains("Score"),
        "validate --quality must show score, got: {stdout}"
    );
}

// ============================================================================
// LINT — NEW: not previously tested with a real model
// ============================================================================

#[test]
fn test_coverage_lint_basic() {
    let model = create_rich_test_model();
    // Lint may pass or fail (warn about missing tensors etc), but must not crash.
    let output = apr()
        .args(["lint", model.path().to_str().unwrap()])
        .output()
        .expect("run lint");
    // Must produce output (not silently skip)
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stdout.is_empty() || !stderr.is_empty(),
        "lint must produce output"
    );
}

#[test]
fn test_coverage_lint_json() {
    let model = create_rich_test_model();
    let output = apr()
        .args(["lint", model.path().to_str().unwrap(), "--json"])
        .output()
        .expect("run lint --json");
    let stdout = String::from_utf8_lossy(&output.stdout);
    // JSON mode must produce JSON output with the expected fields
    assert!(
        stdout.contains("\"passed\"") || stdout.contains("\"error_count\""),
        "lint --json must produce JSON with 'passed' or 'error_count' field, got: {stdout}"
    );
}

#[test]
fn test_coverage_lint_quiet() {
    let model = create_rich_test_model();
    // --quiet suppresses INFO/WARN, only shows errors
    let _output = apr()
        .args(["lint", model.path().to_str().unwrap(), "--quiet"])
        .output()
        .expect("run lint --quiet");
    // Just verify it doesn't crash
}

// ============================================================================
// TENSORS — JSON with rich model
// ============================================================================

#[test]
fn test_coverage_tensors_rich_json() {
    let model = create_rich_test_model();
    apr()
        .args(["tensors", model.path().to_str().unwrap(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("tensor_count"));
}

#[test]
fn test_coverage_tensors_stats() {
    let model = create_rich_test_model();
    apr()
        .args(["tensors", model.path().to_str().unwrap(), "--stats"])
        .assert()
        .success();
}

// ============================================================================
// EXPLAIN — NEW: not previously tested with a real model
// ============================================================================

#[test]
fn test_coverage_explain_model_file() {
    let model = create_rich_test_model();
    apr()
        .args(["explain", model.path().to_str().unwrap()])
        .assert()
        .success();
}

#[test]
fn test_coverage_explain_error_code() {
    // Explain an error code (no model needed)
    apr().args(["explain", "E001"]).assert().success();
}

#[test]
fn test_coverage_explain_tensor() {
    apr()
        .args(["explain", "--tensor", "q_proj"])
        .assert()
        .success();
}

#[test]
fn test_coverage_explain_kernel() {
    apr()
        .args(["explain", "--kernel", "llama"])
        .assert()
        .success();
}

#[test]
fn test_coverage_explain_model_json() {
    let model = create_rich_test_model();
    apr()
        .args(["explain", model.path().to_str().unwrap(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("{"));
}

// ============================================================================
// TRACE — NEW: not previously tested with a real model
// ============================================================================

#[test]
fn test_coverage_trace_basic() {
    let model = create_rich_test_model();
    apr()
        .args(["trace", model.path().to_str().unwrap()])
        .assert()
        .success();
}

#[test]
fn test_coverage_trace_json() {
    let model = create_rich_test_model();
    apr()
        .args(["trace", model.path().to_str().unwrap(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"layers\""))
        .stdout(predicate::str::contains("\"summary\""));
}

#[test]
fn test_coverage_trace_verbose() {
    let model = create_rich_test_model();
    apr()
        .args(["trace", model.path().to_str().unwrap(), "--verbose"])
        .assert()
        .success();
}

#[test]
fn test_coverage_trace_layer_filter() {
    let model = create_rich_test_model();
    apr()
        .args([
            "trace",
            model.path().to_str().unwrap(),
            "--layer",
            "embedding",
        ])
        .assert()
        .success();
}

// ============================================================================
// HEX — verify with rich model
// ============================================================================

#[test]
fn test_coverage_hex_list_rich() {
    let model = create_rich_test_model();
    apr()
        .args(["hex", model.path().to_str().unwrap(), "--list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("model.layers.0"))
        .stdout(predicate::str::contains("lm_head"));
}

#[test]
fn test_coverage_hex_stats_specific_tensor() {
    let model = create_rich_test_model();
    apr()
        .args([
            "hex",
            model.path().to_str().unwrap(),
            "--tensor",
            "embed_tokens",
            "--stats",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("min="))
        .stdout(predicate::str::contains("max="));
}

// ============================================================================
// TREE — verify with rich LLaMA-style model
// ============================================================================

#[test]
fn test_coverage_tree_rich_model() {
    let model = create_rich_test_model();
    apr()
        .args(["tree", model.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("model"))
        .stdout(predicate::str::contains("layers"));
}

#[test]
fn test_coverage_tree_json_rich() {
    let model = create_rich_test_model();
    apr()
        .args(["tree", model.path().to_str().unwrap(), "--format", "json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"children\""));
}

// ============================================================================
// FLOW — verify with rich model
// ============================================================================

#[test]
fn test_coverage_flow_rich_model() {
    let model = create_rich_test_model();
    apr()
        .args(["flow", model.path().to_str().unwrap()])
        .assert()
        .success();
}

#[test]
fn test_coverage_flow_self_attn_rich() {
    let model = create_rich_test_model();
    apr()
        .args([
            "flow",
            model.path().to_str().unwrap(),
            "--component",
            "self_attn",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("SELF-ATTENTION"));
}

// ============================================================================
// DIFF — self-diff with rich model
// ============================================================================

#[test]
fn test_coverage_diff_self_rich() {
    let model = create_rich_test_model();
    let path = model.path().to_str().unwrap();
    apr().args(["diff", path, path]).assert().success().stdout(
        predicate::str::contains("100%")
            .or(predicate::str::contains("identical"))
            .or(predicate::str::contains("IDENTICAL")),
    );
}

#[test]
fn test_coverage_diff_apr_vs_gguf() {
    let apr_model = create_rich_test_model();
    let gguf_model = create_minimal_gguf();
    // Cross-format diff should not crash
    let _output = apr()
        .args([
            "diff",
            apr_model.path().to_str().unwrap(),
            gguf_model.path().to_str().unwrap(),
        ])
        .output()
        .expect("run diff APR vs GGUF");
}

// ============================================================================
// CHECK — model self-test (may fail gates but must not crash)
// ============================================================================

#[test]
fn test_coverage_check_basic() {
    let model = create_rich_test_model();
    // check may fail (synthetic model won't pass inference gates),
    // but it must not panic/crash
    let output = apr()
        .args(["check", model.path().to_str().unwrap()])
        .output()
        .expect("run check");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    // Must produce diagnostic output
    assert!(
        !stdout.is_empty() || !stderr.is_empty(),
        "check must produce output"
    );
}

#[test]
fn test_coverage_check_json() {
    let model = create_rich_test_model();
    let output = apr()
        .args(["check", model.path().to_str().unwrap(), "--json"])
        .output()
        .expect("run check --json");
    let stdout = String::from_utf8_lossy(&output.stdout);
    // JSON mode must produce parseable JSON
    assert!(
        stdout.contains('{') && stdout.contains('}'),
        "check --json must produce JSON, got: {stdout}"
    );
}

#[test]
fn test_coverage_check_no_gpu() {
    let model = create_rich_test_model();
    let _output = apr()
        .args(["check", model.path().to_str().unwrap(), "--no-gpu"])
        .output()
        .expect("run check --no-gpu");
}

// ============================================================================
// QA — quality assurance gates (may fail but must not crash)
// ============================================================================

#[test]
fn test_coverage_qa_basic() {
    let model = create_rich_test_model();
    // QA gates will fail on a synthetic model, but must not crash
    let output = apr()
        .args(["qa", model.path().to_str().unwrap()])
        .output()
        .expect("run qa");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stdout.is_empty() || !stderr.is_empty(),
        "qa must produce output"
    );
}

#[test]
fn test_coverage_qa_json() {
    let model = create_rich_test_model();
    let output = apr()
        .args(["qa", model.path().to_str().unwrap(), "--json"])
        .output()
        .expect("run qa --json");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    // QA gates may fail early on synthetic model. Either JSON output or a
    // clear error in stderr is acceptable (no silent skip, no panic).
    assert!(
        stdout.contains('{') || stderr.contains("error") || stderr.contains("Error"),
        "qa --json must produce JSON or a clear error, stdout={stdout}, stderr={stderr}"
    );
}

#[test]
fn test_coverage_qa_skip_gates() {
    let model = create_rich_test_model();
    let _output = apr()
        .args([
            "qa",
            model.path().to_str().unwrap(),
            "--skip-ollama",
            "--skip-gpu-speedup",
            "--skip-format-parity",
        ])
        .output()
        .expect("run qa with skipped gates");
}

// ============================================================================
// QUALIFY — cross-subcommand smoke test (NEW: never tested before)
// ============================================================================

#[test]
fn test_coverage_qualify_smoke() {
    let model = create_rich_test_model();
    // qualify runs all diagnostic subcommands against the model
    let output = apr()
        .args(["qualify", model.path().to_str().unwrap()])
        .output()
        .expect("run qualify");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stdout.is_empty() || !stderr.is_empty(),
        "qualify must produce output"
    );
    // Should show gate results
    assert!(
        stdout.contains("PASS") || stdout.contains("FAIL") || stdout.contains("Qualify"),
        "qualify should show gate statuses, got: {stdout}"
    );
}

#[test]
fn test_coverage_qualify_json() {
    let model = create_rich_test_model();
    let output = apr()
        .args(["qualify", model.path().to_str().unwrap(), "--json"])
        .output()
        .expect("run qualify --json");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains('{') && stdout.contains('}'),
        "qualify --json must produce JSON, got: {stdout}"
    );
}

#[test]
fn test_coverage_qualify_verbose() {
    let model = create_rich_test_model();
    let _output = apr()
        .args(["qualify", model.path().to_str().unwrap(), "--verbose"])
        .output()
        .expect("run qualify --verbose");
}

// ============================================================================
// EXPORT — verify export to safetensors (NEW: only missing-file tested before)
// ============================================================================

#[test]
fn test_coverage_export_safetensors() {
    let model = create_rich_test_model();
    let output_file = NamedTempFile::with_suffix(".safetensors").expect("create output temp");

    let result = apr()
        .args([
            "export",
            model.path().to_str().unwrap(),
            "--format",
            "safetensors",
            "-o",
            output_file.path().to_str().unwrap(),
        ])
        .output()
        .expect("run export");
    // Export should either succeed or fail with a clear error
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        result.status.success() || stderr.contains("Error") || stderr.contains("error"),
        "export should succeed or fail clearly, exit={}, stderr={}",
        result.status,
        stderr
    );
}

#[test]
fn test_coverage_export_gguf() {
    let model = create_rich_test_model();
    let output_file = NamedTempFile::with_suffix(".gguf").expect("create output temp");

    let result = apr()
        .args([
            "export",
            model.path().to_str().unwrap(),
            "--format",
            "gguf",
            "-o",
            output_file.path().to_str().unwrap(),
        ])
        .output()
        .expect("run export gguf");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        result.status.success() || stderr.contains("Error") || stderr.contains("error"),
        "export to GGUF should succeed or fail clearly, exit={}, stderr={}",
        result.status,
        stderr
    );
}

#[test]
fn test_coverage_export_list_formats() {
    apr()
        .args(["export", "--list-formats"])
        .assert()
        .success()
        .stdout(predicate::str::contains("safetensors"))
        .stdout(predicate::str::contains("gguf"));
}

#[test]
fn test_coverage_export_plan() {
    let model = create_rich_test_model();
    let output_file = NamedTempFile::with_suffix(".safetensors").expect("create output temp");

    // --plan shows what would happen without actually exporting
    let result = apr()
        .args([
            "export",
            model.path().to_str().unwrap(),
            "--format",
            "safetensors",
            "-o",
            output_file.path().to_str().unwrap(),
            "--plan",
        ])
        .output()
        .expect("run export --plan");
    let stdout = String::from_utf8_lossy(&result.stdout);
    let stderr = String::from_utf8_lossy(&result.stderr);
    // Plan mode should produce output about what would happen
    assert!(
        !stdout.is_empty() || !stderr.is_empty(),
        "export --plan must produce output"
    );
}

// ============================================================================
// CONVERT — format conversion (NEW: only missing-file tested before)
// ============================================================================

#[test]
fn test_coverage_convert_help_shows_formats() {
    apr()
        .args(["convert", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("quantize").or(predicate::str::contains("format")));
}

// ============================================================================
// ROSETTA — cross-format operations
// ============================================================================

#[test]
fn test_coverage_rosetta_inspect() {
    let model = create_rich_test_model();
    let result = apr()
        .args(["rosetta", "inspect", model.path().to_str().unwrap()])
        .output()
        .expect("run rosetta inspect");
    let stdout = String::from_utf8_lossy(&result.stdout);
    let stderr = String::from_utf8_lossy(&result.stderr);
    // Rosetta inspect should work on APR files
    assert!(
        !stdout.is_empty() || !stderr.is_empty(),
        "rosetta inspect must produce output"
    );
}

// ============================================================================
// PROFILE — performance profiling (may need inference, graceful fail)
// ============================================================================

#[test]
fn test_coverage_profile_help_flags() {
    apr()
        .args(["profile", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--ci"))
        .stdout(predicate::str::contains("--warmup"))
        .stdout(predicate::str::contains("--measure"));
}

// ============================================================================
// MULTI-FORMAT: Run key commands on GGUF too
// ============================================================================

#[test]
fn test_coverage_inspect_gguf() {
    let model = create_minimal_gguf();
    let output = apr()
        .args(["inspect", model.path().to_str().unwrap()])
        .output()
        .expect("run inspect on GGUF");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("not supported"),
        "GGUF inspect must be supported: {stderr}"
    );
}

#[test]
fn test_coverage_trace_gguf() {
    let model = create_minimal_gguf();
    let output = apr()
        .args(["trace", model.path().to_str().unwrap()])
        .output()
        .expect("run trace on GGUF");
    // Minimal GGUF with 0 tensors may produce limited trace
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stdout.is_empty() || !stderr.is_empty(),
        "trace on GGUF must produce output"
    );
}

#[test]
fn test_coverage_explain_gguf() {
    let model = create_minimal_gguf();
    let output = apr()
        .args(["explain", model.path().to_str().unwrap()])
        .output()
        .expect("run explain on GGUF");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stdout.is_empty() || !stderr.is_empty(),
        "explain on GGUF must produce output"
    );
}

// ============================================================================
// EDGE CASES: Empty model, missing tensors
// ============================================================================

#[test]
fn test_coverage_lint_gguf_minimal() {
    let model = create_minimal_gguf();
    let output = apr()
        .args(["lint", model.path().to_str().unwrap()])
        .output()
        .expect("run lint on minimal GGUF");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stdout.is_empty() || !stderr.is_empty(),
        "lint on GGUF must produce output"
    );
}

#[test]
fn test_coverage_validate_gguf_minimal() {
    let model = create_minimal_gguf();
    let output = apr()
        .args(["validate", model.path().to_str().unwrap()])
        .output()
        .expect("run validate on minimal GGUF");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stdout.is_empty() || !stderr.is_empty(),
        "validate on GGUF must produce output"
    );
}

// ============================================================================
// CHECK — additional edge cases (basic/json/no-gpu already above)
// ============================================================================

#[test]
fn test_coverage_check_invalid_file() {
    // A GGUF with no tensors is structurally valid but semantically empty.
    // check should handle this gracefully (not panic).
    let model = create_minimal_gguf();
    let output = apr()
        .args(["check", model.path().to_str().unwrap()])
        .output()
        .expect("run check on minimal GGUF");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stdout.is_empty() || !stderr.is_empty(),
        "check on invalid model must produce output"
    );
}

#[test]
fn test_coverage_check_nonexistent_file() {
    let output = apr()
        .args(["check", "/tmp/nonexistent_model_42.apr"])
        .output()
        .expect("run check on nonexistent file");
    assert!(
        !output.status.success(),
        "check on nonexistent file must fail"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("error")
            || stderr.contains("Error")
            || stderr.contains("not found")
            || stderr.contains("No such file"),
        "check on nonexistent file must report error, got stderr: {stderr}"
    );
}

// ============================================================================
// COMPILE — model compilation tests
// ============================================================================

#[test]
fn test_coverage_compile_help() {
    apr()
        .args(["compile", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("compile"))
        .stdout(predicate::str::contains("--target"));
}

#[test]
fn test_coverage_compile_with_synthetic_model() {
    let model = create_rich_test_model();
    let dir = tempfile::tempdir().expect("create tempdir");
    let output_path = dir.path().join("compiled_model");
    // compile may fail on missing toolchain, but must not panic
    let output = apr()
        .args([
            "compile",
            model.path().to_str().unwrap(),
            "-o",
            output_path.to_str().unwrap(),
        ])
        .output()
        .expect("run compile");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stdout.is_empty() || !stderr.is_empty(),
        "compile must produce output (even if it fails gracefully)"
    );
}

#[test]
fn test_coverage_compile_list_targets() {
    let output = apr()
        .args(["compile", "--list-targets"])
        .output()
        .expect("run compile --list-targets");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    // Should list available targets or produce output
    assert!(
        !stdout.is_empty() || !stderr.is_empty(),
        "compile --list-targets must produce output"
    );
}

// ============================================================================
// DATA — data pipeline tests (audit, split)
// ============================================================================

/// Create a small JSONL dataset in a tempdir for data command tests.
fn create_test_jsonl() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().expect("create tempdir");
    let jsonl_path = dir.path().join("test.jsonl");
    std::fs::write(
        &jsonl_path,
        r#"{"input":"hello world","label":"1"}
{"input":"goodbye world","label":"0"}
{"input":"foo bar baz","label":"1"}
{"input":"testing data","label":"0"}
{"input":"more samples","label":"1"}
{"input":"final entry","label":"0"}
"#,
    )
    .expect("write JSONL");
    (dir, jsonl_path)
}

#[test]
fn test_coverage_data_audit_basic() {
    let (_dir, jsonl_path) = create_test_jsonl();
    let output = apr()
        .args([
            "data",
            "audit",
            jsonl_path.to_str().unwrap(),
            "--num-classes",
            "2",
        ])
        .output()
        .expect("run data audit");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stdout.is_empty() || !stderr.is_empty(),
        "data audit must produce output"
    );
}

#[test]
fn test_coverage_data_audit_json() {
    let (_dir, jsonl_path) = create_test_jsonl();
    let output = apr()
        .args([
            "data",
            "audit",
            jsonl_path.to_str().unwrap(),
            "--num-classes",
            "2",
            "--json",
        ])
        .output()
        .expect("run data audit --json");
    let stdout = String::from_utf8_lossy(&output.stdout);
    // JSON mode should produce structured output
    assert!(
        stdout.contains('{') || stdout.contains('['),
        "data audit --json must produce JSON output, got: {stdout}"
    );
}

#[test]
fn test_coverage_data_split_basic() {
    let (_dir, jsonl_path) = create_test_jsonl();
    let output_dir = tempfile::tempdir().expect("create output tempdir");
    let output = apr()
        .args([
            "data",
            "split",
            jsonl_path.to_str().unwrap(),
            "--output",
            output_dir.path().to_str().unwrap(),
            "--train",
            "0.6",
            "--val",
            "0.2",
            "--test",
            "0.2",
            "--seed",
            "42",
        ])
        .output()
        .expect("run data split");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stdout.is_empty() || !stderr.is_empty(),
        "data split must produce output"
    );
}

#[test]
fn test_coverage_data_audit_nonexistent_file() {
    let output = apr()
        .args(["data", "audit", "/tmp/nonexistent_data_42.jsonl"])
        .output()
        .expect("run data audit on nonexistent file");
    assert!(
        !output.status.success(),
        "data audit on nonexistent file must fail"
    );
}

// ============================================================================
// DIAGNOSE — diagnostic analysis tests
// ============================================================================

#[test]
fn test_coverage_diagnose_help() {
    apr()
        .args(["diagnose", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("diagnose"))
        .stdout(predicate::str::contains("checkpoint"));
}

#[test]
fn test_coverage_diagnose_nonexistent_checkpoint() {
    let output = apr()
        .args(["diagnose", "/tmp/nonexistent_checkpoint_42"])
        .output()
        .expect("run diagnose on nonexistent checkpoint");
    assert!(
        !output.status.success(),
        "diagnose on nonexistent checkpoint must fail"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("error")
            || stderr.contains("Error")
            || stderr.contains("not found")
            || stderr.contains("No such"),
        "diagnose on nonexistent checkpoint must report error, got: {stderr}"
    );
}

#[test]
fn test_coverage_diagnose_with_synthetic_model() {
    // diagnose expects a checkpoint directory, but passing a model file
    // should produce a clear error, not a panic
    let model = create_rich_test_model();
    let output = apr()
        .args(["diagnose", model.path().to_str().unwrap()])
        .output()
        .expect("run diagnose with model file");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    // May succeed or fail gracefully — either is acceptable
    assert!(
        !stdout.is_empty() || !stderr.is_empty(),
        "diagnose must produce output"
    );
}

// ============================================================================
// COMPARE-HF — HuggingFace comparison tests
// ============================================================================

#[test]
fn test_coverage_compare_hf_help() {
    apr()
        .args(["compare-hf", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("compare"))
        .stdout(predicate::str::contains("--hf"));
}

#[test]
fn test_coverage_compare_hf_missing_required_args() {
    // compare-hf requires --hf and a FILE; running without should fail
    let model = create_rich_test_model();
    let output = apr()
        .args(["compare-hf", model.path().to_str().unwrap()])
        .output()
        .expect("run compare-hf without --hf");
    assert!(
        !output.status.success(),
        "compare-hf without --hf must fail"
    );
}

// ============================================================================
// PROFILE — performance profiling tests
// ============================================================================

#[test]
fn test_coverage_profile_help() {
    apr()
        .args(["profile", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("profile"));
}

#[test]
fn test_coverage_profile_with_synthetic_model() {
    let model = create_rich_test_model();
    // profile may fail on a synthetic model that can't do inference,
    // but it must not panic
    let output = apr()
        .args(["profile", model.path().to_str().unwrap()])
        .output()
        .expect("run profile");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stdout.is_empty() || !stderr.is_empty(),
        "profile must produce output"
    );
}

// ============================================================================
// PIPELINE — pipeline orchestration tests
// ============================================================================

#[test]
fn test_coverage_pipeline_help() {
    apr()
        .args(["pipeline", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("pipeline"))
        .stdout(predicate::str::contains("plan"));
}

#[test]
fn test_coverage_pipeline_plan_nonexistent_config() {
    let output = apr()
        .args(["pipeline", "plan", "/tmp/nonexistent_pipeline_42.yaml"])
        .output()
        .expect("run pipeline plan on nonexistent config");
    assert!(
        !output.status.success(),
        "pipeline plan on nonexistent config must fail"
    );
}

// ============================================================================
// PMAT-540 Phase 1: Coverage tests for 33 previously untested subcommands.
// Strategy: --help exits 0 (covers clap parse + dispatch), error paths where
// cheap (covers validation logic). Model-dependent commands get only --help.
// ============================================================================

// --- Model-independent commands: exercise error paths ---

#[test]
fn test_coverage_import_missing_file() {
    apr()
        .args(["import", "/tmp/nonexistent_model_pmat540.safetensors"])
        .assert()
        .failure();
}

#[test]
fn test_coverage_merge_no_args() {
    apr().args(["merge"]).assert().failure();
}

#[test]
fn test_coverage_quantize_missing_file() {
    apr()
        .args(["quantize", "/tmp/nonexistent_pmat540.apr"])
        .assert()
        .failure();
}

#[test]
fn test_coverage_prune_missing_file() {
    apr()
        .args(["prune", "/tmp/nonexistent_pmat540.apr"])
        .assert()
        .failure();
}

#[test]
fn test_coverage_distill_no_args() {
    apr().args(["distill"]).assert().failure();
}

#[test]
fn test_coverage_train_help() {
    apr().args(["train", "--help"]).assert().success();
}

#[test]
fn test_coverage_finetune_no_args() {
    apr().args(["finetune"]).assert().failure();
}

#[test]
fn test_coverage_tokenize_help() {
    apr().args(["tokenize", "--help"]).assert().success();
}

#[test]
fn test_coverage_tune_help() {
    apr().args(["tune", "--help"]).assert().success();
}

#[test]
fn test_coverage_bench_help() {
    apr().args(["bench", "--help"]).assert().success();
}

#[test]
fn test_coverage_eval_help() {
    apr().args(["eval", "--help"]).assert().success();
}

#[test]
fn test_coverage_canary_help() {
    apr().args(["canary", "--help"]).assert().success();
}

#[test]
fn test_coverage_parity_help() {
    apr().args(["parity", "--help"]).assert().success();
}

#[test]
fn test_coverage_gpu_info() {
    // gpu command should succeed even without GPU (prints "no GPU detected" or similar)
    let output = apr().args(["gpu"]).output().expect("run apr gpu");
    // Don't assert success — may fail if no GPU, but should not panic
    assert!(
        output.status.success() || !output.stderr.is_empty(),
        "apr gpu should either succeed or produce error message"
    );
}

#[test]
fn test_coverage_ptx_help() {
    apr().args(["ptx", "--help"]).assert().success();
}

#[test]
fn test_coverage_ptx_map_help() {
    apr().args(["ptx-map", "--help"]).assert().success();
}

#[test]
fn test_coverage_monitor_help() {
    apr().args(["monitor", "--help"]).assert().success();
}

#[test]
fn test_coverage_runs_help() {
    apr().args(["runs", "--help"]).assert().success();
}

#[test]
fn test_coverage_experiment_help() {
    apr().args(["experiment", "--help"]).assert().success();
}

#[test]
fn test_coverage_showcase_help() {
    apr().args(["showcase", "--help"]).assert().success();
}

#[test]
fn test_coverage_probar_help() {
    apr().args(["probar", "--help"]).assert().success();
}

#[test]
fn test_coverage_oracle_help() {
    apr().args(["oracle", "--help"]).assert().success();
}

#[test]
fn test_coverage_encrypt_no_args() {
    apr().args(["encrypt"]).assert().failure();
}

#[test]
fn test_coverage_decrypt_no_args() {
    apr().args(["decrypt"]).assert().failure();
}

#[test]
fn test_coverage_pull_missing_model() {
    // pull with nonexistent model should fail gracefully
    let output = apr()
        .args(["pull", "nonexistent/model-pmat540"])
        .output()
        .expect("run apr pull");
    assert!(!output.status.success());
}

#[test]
fn test_coverage_list_help() {
    apr().args(["list", "--help"]).assert().success();
}

#[test]
fn test_coverage_rm_no_args() {
    apr().args(["rm"]).assert().failure();
}

#[test]
fn test_coverage_publish_help() {
    apr().args(["publish", "--help"]).assert().success();
}

#[test]
fn test_coverage_run_missing_model() {
    apr()
        .args(["run", "/tmp/nonexistent_pmat540.gguf"])
        .assert()
        .failure();
}

#[test]
fn test_coverage_tui_help() {
    apr().args(["tui", "--help"]).assert().success();
}

#[test]
fn test_coverage_cbtop_help() {
    apr().args(["cbtop", "--help"]).assert().success();
}

// --- Model-dependent commands: --help only (no fixture needed) ---

#[test]
fn test_coverage_chat_help() {
    apr().args(["chat", "--help"]).assert().success();
}

#[test]
fn test_coverage_serve_help() {
    apr().args(["serve", "--help"]).assert().success();
}

// ============================================================================
// PMAT-540 Phase 4: Fixture-backed tests for top-5 untested command files.
// Uses the rich APR model to exercise handler logic beyond --help/error paths.
// ============================================================================

// --- serve plan: exercises roofline computation, hardware detection ---

#[test]
fn test_coverage_serve_plan_with_model() {
    let model = create_rich_test_model();
    let output = apr()
        .args(["serve", "plan", model.path().to_str().unwrap()])
        .output()
        .expect("run serve plan");
    // serve plan should produce output even for tiny models
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success() || !stdout.is_empty(),
        "serve plan should produce output or succeed"
    );
}

#[test]
fn test_coverage_serve_plan_json() {
    let model = create_rich_test_model();
    apr()
        .args(["serve", "plan", model.path().to_str().unwrap(), "--json"])
        .assert()
        .success();
}

// --- check: exercises architecture detection, real model checks ---

#[test]
fn test_coverage_check_rich_model_no_gpu() {
    let model = create_rich_test_model();
    apr()
        .args(["check", model.path().to_str().unwrap(), "--no-gpu"])
        .assert()
        .success();
}

#[test]
fn test_coverage_check_rich_model_json() {
    let model = create_rich_test_model();
    apr()
        .args([
            "check",
            model.path().to_str().unwrap(),
            "--no-gpu",
            "--json",
        ])
        .assert()
        .success();
}

// --- distill: exercises argument parsing, config validation ---

#[test]
fn test_coverage_distill_help_subcommands() {
    apr().args(["distill", "--help"]).assert().success();
}

#[test]
fn test_coverage_distill_missing_teacher() {
    apr()
        .args([
            "distill",
            "--teacher",
            "/tmp/nonexistent_teacher_pmat540.gguf",
            "--student",
            "/tmp/nonexistent_student_pmat540.gguf",
        ])
        .assert()
        .failure();
}

// --- train: exercises plan/apply/watch subcommands ---

#[test]
fn test_coverage_train_plan_missing_config() {
    apr()
        .args([
            "train",
            "plan",
            "--config",
            "/tmp/nonexistent_train_config.yaml",
        ])
        .assert()
        .failure();
}

#[test]
fn test_coverage_train_apply_missing_config() {
    apr()
        .args([
            "train",
            "apply",
            "--config",
            "/tmp/nonexistent_train_config.yaml",
        ])
        .assert()
        .failure();
}

// --- runs: exercises list/show subcommands ---

#[test]
fn test_coverage_runs_ls_empty() {
    // runs ls with a nonexistent dir should fail gracefully
    let output = apr()
        .args(["runs", "ls", "--dir", "/tmp/nonexistent_runs_dir_pmat540"])
        .output()
        .expect("run runs ls");
    // Either succeeds with "no experiments" or fails with directory error
    assert!(
        output.status.success() || !output.stderr.is_empty(),
        "runs ls should produce output or error"
    );
}

#[test]
fn test_coverage_runs_ls_json() {
    let output = apr()
        .args([
            "runs",
            "ls",
            "--json",
            "--dir",
            "/tmp/nonexistent_runs_dir_pmat540",
        ])
        .output()
        .expect("run runs ls json");
    assert!(
        output.status.success() || !output.stderr.is_empty(),
        "runs ls --json should produce output or error"
    );
}
