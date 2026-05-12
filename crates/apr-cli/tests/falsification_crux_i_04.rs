//! End-to-end falsification tests for CRUX-I-04 — Ollama function calling.
//!
//! Contract: `contracts/crux-I-04-v1.yaml`.
//!
//! CRUX-SHIP-001 compliance:
//! - g1_classifier_green: `commands::ollama_tool_call_classifier` in-crate (36 tests).
//! - g2_cli_reachable: `apr ollama-tools-lint --help` surfaces `--response-file`,
//!   `--request-file`, and `--stream`.
//! - g3_e2e_runs: subprocess invocation of the real `apr` binary runs the
//!   classifier end-to-end over a user-supplied captured /api/chat tool-call
//!   response. Live /api/chat handler with `tools[]` in aprender-serve remains
//!   PARTIAL_ALGORITHM_LEVEL under BLOCKER-UPSTREAM-MISSING.

#![allow(clippy::unwrap_used)]

use std::io::Write;

use assert_cmd::Command;
use predicates::prelude::*;

fn write_json(v: &serde_json::Value) -> tempfile::NamedTempFile {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    let body = serde_json::to_vec(v).unwrap();
    f.write_all(&body).unwrap();
    f.flush().unwrap();
    f
}

fn write_text(body: &str) -> tempfile::NamedTempFile {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(body.as_bytes()).unwrap();
    f.flush().unwrap();
    f
}

fn well_formed_tool_response() -> serde_json::Value {
    serde_json::json!({
        "model": "tiny",
        "created_at": "2026-04-22T00:00:00Z",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {
                    "name": "get_weather",
                    "arguments": {"location": "San Francisco", "unit": "celsius"}
                }}
            ]
        },
        "done": true
    })
}

fn weather_request() -> serde_json::Value {
    serde_json::json!({
        "model": "tiny",
        "messages": [{"role": "user", "content": "What's the weather in SF?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Look up current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            }
        }]
    })
}

// ═══ g2_cli_reachable ═══

#[test]
fn falsify_crux_i_04_help_advertises_response_file_flag() {
    Command::cargo_bin("apr")
        .unwrap()
        .args(["ollama-tools-lint", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--response-file"));
}

#[test]
fn falsify_crux_i_04_help_advertises_request_file_flag() {
    Command::cargo_bin("apr")
        .unwrap()
        .args(["ollama-tools-lint", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--request-file"));
}

#[test]
fn falsify_crux_i_04_help_advertises_stream_flag() {
    Command::cargo_bin("apr")
        .unwrap()
        .args(["ollama-tools-lint", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--stream"));
}

#[test]
fn falsify_crux_i_04_rejects_bare_invocation_without_response_file() {
    let output = Command::cargo_bin("apr")
        .unwrap()
        .args(["ollama-tools-lint"])
        .output()
        .expect("apr binary runs");
    assert!(
        !output.status.success(),
        "bare `apr ollama-tools-lint` without --response-file must fail"
    );
}

// ═══ g3_e2e_runs: non-streaming schema ═══

#[test]
fn falsify_crux_i_04_non_streaming_accepts_well_formed_tool_call() {
    let resp_f = write_json(&well_formed_tool_response());
    let req_f = write_json(&weather_request());
    let output = Command::cargo_bin("apr")
        .unwrap()
        .args([
            "--json",
            "ollama-tools-lint",
            "--response-file",
            resp_f.path().to_str().unwrap(),
            "--request-file",
            req_f.path().to_str().unwrap(),
        ])
        .output()
        .expect("apr binary runs");
    assert!(
        output.status.success(),
        "well-formed tool-call response must pass: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let v: serde_json::Value =
        serde_json::from_str(&String::from_utf8_lossy(&output.stdout)).unwrap();
    assert_eq!(v["schema_ok"].as_bool(), Some(true));
    assert_eq!(v["allowlist_ok"].as_bool(), Some(true));
    assert_eq!(v["mode"].as_str(), Some("non_streaming"));
}

#[test]
fn falsify_crux_i_04_without_request_file_flags_no_declared_tools() {
    // Response has tool_calls but no declared tools → NoDeclaredTools.
    let f = write_json(&well_formed_tool_response());
    let output = Command::cargo_bin("apr")
        .unwrap()
        .args([
            "ollama-tools-lint",
            "--response-file",
            f.path().to_str().unwrap(),
        ])
        .output()
        .expect("apr binary runs");
    assert!(
        !output.status.success(),
        "tool_calls with no declared tools must be rejected"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("NoDeclaredTools") || stderr.contains("allowlist"),
        "stderr should explain NoDeclaredTools; got: {stderr}"
    );
}

/// FALSIFY-CRUX-I-04-002: `arguments` MUST be an object, not a stringified
/// JSON blob — this is the canonical Ollama/OpenAI drift bug.
#[test]
fn falsify_crux_i_04_rejects_stringified_arguments() {
    let mut resp = well_formed_tool_response();
    resp["message"]["tool_calls"][0]["function"]["arguments"] =
        serde_json::json!("{\"location\":\"San Francisco\"}");
    let f = write_json(&resp);

    let output = Command::cargo_bin("apr")
        .unwrap()
        .args([
            "ollama-tools-lint",
            "--response-file",
            f.path().to_str().unwrap(),
        ])
        .output()
        .expect("apr binary runs");
    assert!(
        !output.status.success(),
        "stringified arguments must be rejected (drift bug)"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("FunctionArgumentsIsString") || stderr.contains("schema"),
        "stderr should explain stringified-arguments violation; got: {stderr}"
    );
}

#[test]
fn falsify_crux_i_04_rejects_missing_function_name() {
    let mut resp = well_formed_tool_response();
    resp["message"]["tool_calls"][0]["function"]
        .as_object_mut()
        .unwrap()
        .remove("name");
    let f = write_json(&resp);

    let output = Command::cargo_bin("apr")
        .unwrap()
        .args([
            "ollama-tools-lint",
            "--response-file",
            f.path().to_str().unwrap(),
        ])
        .output()
        .expect("apr binary runs");
    assert!(
        !output.status.success(),
        "missing function.name must be rejected"
    );
}

#[test]
fn falsify_crux_i_04_rejects_empty_tool_calls_array() {
    let resp = serde_json::json!({
        "message": {"role": "assistant", "tool_calls": []},
        "done": true
    });
    let f = write_json(&resp);

    let output = Command::cargo_bin("apr")
        .unwrap()
        .args([
            "ollama-tools-lint",
            "--response-file",
            f.path().to_str().unwrap(),
        ])
        .output()
        .expect("apr binary runs");
    assert!(
        !output.status.success(),
        "empty tool_calls[] must be rejected by this gate"
    );
}

// ═══ g3_e2e_runs: tool-name allowlist (FALSIFY-CRUX-I-04-005) ═══

#[test]
fn falsify_crux_i_04_allowlist_accepts_declared_tool() {
    let resp_f = write_json(&well_formed_tool_response());
    let req_f = write_json(&weather_request());

    let output = Command::cargo_bin("apr")
        .unwrap()
        .args([
            "--json",
            "ollama-tools-lint",
            "--response-file",
            resp_f.path().to_str().unwrap(),
            "--request-file",
            req_f.path().to_str().unwrap(),
        ])
        .output()
        .expect("apr binary runs");
    assert!(
        output.status.success(),
        "declared tool must pass allowlist: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let v: serde_json::Value =
        serde_json::from_str(&String::from_utf8_lossy(&output.stdout)).unwrap();
    assert_eq!(v["allowlist_ok"].as_bool(), Some(true));
    let declared = v["declared_tool_names"].as_array().unwrap();
    assert_eq!(declared.len(), 1);
    assert_eq!(declared[0].as_str(), Some("get_weather"));
}

#[test]
fn falsify_crux_i_04_allowlist_rejects_hallucinated_tool() {
    let mut resp = well_formed_tool_response();
    resp["message"]["tool_calls"][0]["function"]["name"] = serde_json::json!("make_coffee");
    let resp_f = write_json(&resp);
    let req_f = write_json(&weather_request());

    let output = Command::cargo_bin("apr")
        .unwrap()
        .args([
            "ollama-tools-lint",
            "--response-file",
            resp_f.path().to_str().unwrap(),
            "--request-file",
            req_f.path().to_str().unwrap(),
        ])
        .output()
        .expect("apr binary runs");
    assert!(
        !output.status.success(),
        "hallucinated tool name must be rejected"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("HallucinatedToolName") || stderr.contains("allowlist"),
        "stderr should explain allowlist violation; got: {stderr}"
    );
}

// ═══ g3_e2e_runs: streaming NDJSON (FALSIFY-CRUX-I-04-003) ═══

#[test]
fn falsify_crux_i_04_stream_accepts_well_formed_tool_call_ndjson() {
    let body = r#"{"message":{"role":"assistant","content":""},"done":false}
{"message":{"role":"assistant","content":""},"done":false}
{"message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"get_weather","arguments":{"location":"SF"}}}]},"done":true}
"#;
    let f = write_text(body);
    let output = Command::cargo_bin("apr")
        .unwrap()
        .args([
            "--json",
            "ollama-tools-lint",
            "--response-file",
            f.path().to_str().unwrap(),
            "--stream",
        ])
        .output()
        .expect("apr binary runs");
    assert!(
        output.status.success(),
        "well-formed tool-call NDJSON must pass: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let v: serde_json::Value =
        serde_json::from_str(&String::from_utf8_lossy(&output.stdout)).unwrap();
    assert_eq!(v["ndjson_ok"].as_bool(), Some(true));
    assert_eq!(v["num_frames"].as_u64(), Some(3));
    assert_eq!(v["mode"].as_str(), Some("streaming_ndjson"));
}

#[test]
fn falsify_crux_i_04_stream_rejects_tool_calls_in_non_terminator() {
    // A non-terminator frame carrying a non-empty tool_calls[] violates
    // the atomicity invariant (tool_calls MUST only appear in done=true frame).
    let body = r#"{"message":{"role":"assistant","tool_calls":[{"function":{"name":"get_weather","arguments":{}}}]},"done":false}
{"message":{"role":"assistant","content":""},"done":true}
"#;
    let f = write_text(body);
    let output = Command::cargo_bin("apr")
        .unwrap()
        .args([
            "ollama-tools-lint",
            "--response-file",
            f.path().to_str().unwrap(),
            "--stream",
        ])
        .output()
        .expect("apr binary runs");
    assert!(
        !output.status.success(),
        "tool_calls in non-terminator frame must be rejected"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("ToolCallsInNonTerminatorFrame") || stderr.contains("NDJSON"),
        "stderr should explain atomicity violation; got: {stderr}"
    );
}

#[test]
fn falsify_crux_i_04_stream_rejects_early_done_true() {
    let body = r#"{"message":{"role":"assistant","content":"hi"},"done":false}
{"done":true}
{"message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"get_weather","arguments":{}}}]},"done":true}
"#;
    let f = write_text(body);
    let output = Command::cargo_bin("apr")
        .unwrap()
        .args([
            "ollama-tools-lint",
            "--response-file",
            f.path().to_str().unwrap(),
            "--stream",
        ])
        .output()
        .expect("apr binary runs");
    assert!(
        !output.status.success(),
        "early done=true must be rejected (terminator uniqueness)"
    );
}

#[test]
fn falsify_crux_i_04_stream_rejects_missing_terminator() {
    let body = r#"{"message":{"role":"assistant","content":"a"},"done":false}
{"message":{"role":"assistant","content":"b"},"done":false}
"#;
    let f = write_text(body);
    let output = Command::cargo_bin("apr")
        .unwrap()
        .args([
            "ollama-tools-lint",
            "--response-file",
            f.path().to_str().unwrap(),
            "--stream",
        ])
        .output()
        .expect("apr binary runs");
    assert!(
        !output.status.success(),
        "NDJSON without terminal done=true must be rejected"
    );
}

#[test]
fn falsify_crux_i_04_stream_rejects_empty_file() {
    let f = write_text("");
    let output = Command::cargo_bin("apr")
        .unwrap()
        .args([
            "ollama-tools-lint",
            "--response-file",
            f.path().to_str().unwrap(),
            "--stream",
        ])
        .output()
        .expect("apr binary runs");
    assert!(
        !output.status.success(),
        "empty NDJSON stream must be rejected (no silent pass)"
    );
}
