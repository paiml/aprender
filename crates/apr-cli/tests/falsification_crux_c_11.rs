//! E2E falsification tests for `apr tool-use-lint` (CRUX-C-11).
//!
//! Discharges g3 of CRUX-SHIP-001 for PR #974: exercise the CLI surface
//! end-to-end on captured JSON observations and assert the classifier
//! verdicts + non-zero exit codes on known-bad input.

use std::io::Write;
use std::process::Command;

fn apr_binary() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_apr"));
    cmd.env("NO_COLOR", "1");
    cmd
}

fn write_tmp_json(name: &str, body: &str) -> tempfile::NamedTempFile {
    let mut f = tempfile::Builder::new()
        .prefix(name)
        .suffix(".json")
        .tempfile()
        .expect("create tempfile");
    f.write_all(body.as_bytes()).expect("write tempfile");
    f.flush().expect("flush tempfile");
    f
}

// ---- help surface (g2 proof) ----------------------------------------------

#[test]
fn falsify_crux_c_11_help_advertises_observation_file_flag() {
    let out = apr_binary()
        .args(["tool-use-lint", "--help"])
        .output()
        .expect("run apr tool-use-lint --help");
    assert!(out.status.success(), "apr tool-use-lint --help must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("--observation-file"),
        "--help must advertise --observation-file; got:\n{stdout}"
    );
}

#[test]
fn falsify_crux_c_11_rejects_bare_invocation_without_file() {
    let out = apr_binary()
        .arg("tool-use-lint")
        .output()
        .expect("run apr tool-use-lint without args");
    assert!(
        !out.status.success(),
        "bare `apr tool-use-lint` must exit non-zero"
    );
}

// ---- shape gate (FALSIFY-CRUX-C-11-001) -----------------------------------

#[test]
fn falsify_crux_c_11_001_shape_ok_on_well_formed_call() {
    let tmp = write_tmp_json(
        "tu-shape-ok",
        r#"{ "shape": {
               "declared_tools": [ { "name": "get_weather",
                                     "parameters": { "type": "object",
                                                     "properties": { "location": { "type": "string" } },
                                                     "required": ["location"] } } ],
               "tool_calls": [ { "id": "call_1", "type": "function",
                                 "name": "get_weather",
                                 "arguments": "{\"location\":\"Paris\"}" } ],
               "finish_reason": "tool_calls"
             } }"#,
    );
    let out = apr_binary()
        .args(["tool-use-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr tool-use-lint");
    assert!(
        out.status.success(),
        "shape well-formed must pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_c_11_001_shape_rejects_unknown_name() {
    let tmp = write_tmp_json(
        "tu-shape-unk",
        r#"{ "shape": {
               "declared_tools": [ { "name": "get_weather", "parameters": {} } ],
               "tool_calls": [ { "id": "c", "type": "function",
                                 "name": "get_time", "arguments": "{}" } ],
               "finish_reason": "tool_calls"
             } }"#,
    );
    let out = apr_binary()
        .args(["tool-use-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr tool-use-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-11-001"));
}

#[test]
fn falsify_crux_c_11_001_shape_rejects_non_function_type() {
    let tmp = write_tmp_json(
        "tu-shape-type",
        r#"{ "shape": {
               "declared_tools": [ { "name": "get_weather", "parameters": {} } ],
               "tool_calls": [ { "id": "c", "type": "code_interpreter",
                                 "name": "get_weather", "arguments": "{}" } ],
               "finish_reason": "tool_calls"
             } }"#,
    );
    let out = apr_binary()
        .args(["tool-use-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr tool-use-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-11-001"));
}

#[test]
fn falsify_crux_c_11_001_shape_rejects_non_json_arguments() {
    let tmp = write_tmp_json(
        "tu-shape-args",
        r#"{ "shape": {
               "declared_tools": [ { "name": "get_weather", "parameters": {} } ],
               "tool_calls": [ { "id": "c", "type": "function",
                                 "name": "get_weather",
                                 "arguments": "not-json{" } ],
               "finish_reason": "tool_calls"
             } }"#,
    );
    let out = apr_binary()
        .args(["tool-use-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr tool-use-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-11-001"));
}

#[test]
fn falsify_crux_c_11_001_shape_rejects_wrong_finish_reason() {
    let tmp = write_tmp_json(
        "tu-shape-fr",
        r#"{ "shape": {
               "declared_tools": [ { "name": "get_weather", "parameters": {} } ],
               "tool_calls": [ { "id": "c", "type": "function",
                                 "name": "get_weather", "arguments": "{}" } ],
               "finish_reason": "stop"
             } }"#,
    );
    let out = apr_binary()
        .args(["tool-use-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr tool-use-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-11-001"));
}

// ---- schema gate (FALSIFY-CRUX-C-11-002) ----------------------------------

#[test]
fn falsify_crux_c_11_002_schema_ok_on_matching_object() {
    let tmp = write_tmp_json(
        "tu-schema-ok",
        r#"{ "schema": {
               "arguments": "{\"location\":\"Paris\"}",
               "parameters": { "type":"object",
                               "properties":{"location":{"type":"string"}},
                               "required":["location"] }
             } }"#,
    );
    let out = apr_binary()
        .args(["tool-use-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr tool-use-lint");
    assert!(
        out.status.success(),
        "schema match must pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_c_11_002_schema_rejects_missing_required() {
    let tmp = write_tmp_json(
        "tu-schema-miss",
        r#"{ "schema": {
               "arguments": "{}",
               "parameters": { "type":"object",
                               "properties":{"location":{"type":"string"}},
                               "required":["location"] }
             } }"#,
    );
    let out = apr_binary()
        .args(["tool-use-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr tool-use-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-11-002"));
}

#[test]
fn falsify_crux_c_11_002_schema_rejects_wrong_type() {
    let tmp = write_tmp_json(
        "tu-schema-type",
        r#"{ "schema": {
               "arguments": "{\"location\":42}",
               "parameters": { "type":"object",
                               "properties":{"location":{"type":"string"}},
                               "required":["location"] }
             } }"#,
    );
    let out = apr_binary()
        .args(["tool-use-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr tool-use-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-11-002"));
}

// ---- passthrough gate (FALSIFY-CRUX-C-11-003) -----------------------------

#[test]
fn falsify_crux_c_11_003_passthrough_ok_with_stop() {
    let tmp = write_tmp_json(
        "tu-pass-ok",
        r#"{ "passthrough": { "tool_calls": [], "finish_reason": "stop" } }"#,
    );
    let out = apr_binary()
        .args(["tool-use-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr tool-use-lint");
    assert!(
        out.status.success(),
        "empty tool_calls + stop must pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_c_11_003_passthrough_rejects_synthesized_calls() {
    let tmp = write_tmp_json(
        "tu-pass-syn",
        r#"{ "passthrough": {
               "tool_calls": [ { "id": "c", "type": "function",
                                 "name": "get_weather", "arguments": "{}" } ],
               "finish_reason": "stop"
             } }"#,
    );
    let out = apr_binary()
        .args(["tool-use-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr tool-use-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-11-003"));
}

#[test]
fn falsify_crux_c_11_003_passthrough_rejects_tool_calls_finish_reason() {
    let tmp = write_tmp_json(
        "tu-pass-fr",
        r#"{ "passthrough": { "tool_calls": [], "finish_reason": "tool_calls" } }"#,
    );
    let out = apr_binary()
        .args(["tool-use-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr tool-use-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-11-003"));
}

// ---- input validation -----------------------------------------------------

#[test]
fn falsify_crux_c_11_empty_file_rejected_via_cli() {
    let tmp = write_tmp_json("tu-empty", "");
    let out = apr_binary()
        .args(["tool-use-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr tool-use-lint");
    assert!(!out.status.success(), "empty file must be rejected");
}

#[test]
fn falsify_crux_c_11_nonexistent_file_rejected() {
    let out = apr_binary()
        .args([
            "tool-use-lint",
            "--observation-file",
            "/nonexistent/path/obs.json",
        ])
        .output()
        .expect("run apr tool-use-lint");
    assert!(!out.status.success());
}

// ---- --json shape ---------------------------------------------------------

#[test]
fn falsify_crux_c_11_json_output_shape() {
    let tmp = write_tmp_json(
        "tu-json",
        r#"{
          "shape": {
            "declared_tools": [ { "name": "get_weather",
                                  "parameters": { "type":"object",
                                                  "properties":{"location":{"type":"string"}},
                                                  "required":["location"] } } ],
            "tool_calls":   [ { "id": "c", "type": "function",
                                "name": "get_weather",
                                "arguments": "{\"location\":\"NYC\"}" } ],
            "finish_reason": "tool_calls"
          },
          "schema": {
            "arguments": "{\"location\":\"NYC\"}",
            "parameters": { "type":"object",
                            "properties":{"location":{"type":"string"}},
                            "required":["location"] }
          },
          "passthrough": { "tool_calls": [], "finish_reason": "stop" }
        }"#,
    );
    let out = apr_binary()
        .args(["--json", "tool-use-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr tool-use-lint --json");
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("\"shape\""));
    assert!(stdout.contains("\"schema\""));
    assert!(stdout.contains("\"passthrough\""));
}
