//! End-to-end falsification tests for CRUX-C-04 — Ollama /api/chat.
//!
//! Contract: `contracts/crux-C-04-v1.yaml` (v1.2.0).
//!
//! CRUX-SHIP-001 compliance:
//! - g1_classifier_green: `commands::ollama_chat_classifier` in-crate (25 tests).
//! - g2_cli_reachable: `apr ollama-chat-lint --help` surfaces `--response-file`.
//! - g3_e2e_runs: subprocess invocation of the real binary runs the
//!   classifier end-to-end over a user-supplied captured /api/chat response.
//!   Live /api/chat handler in aprender-serve remains PARTIAL_ALGORITHM_LEVEL
//!   under BLOCKER-UPSTREAM-MISSING.

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

fn well_formed_response() -> serde_json::Value {
    serde_json::json!({
        "model": "tiny",
        "created_at": "2026-04-21T00:00:00Z",
        "message": {"role": "assistant", "content": "ok"},
        "done": true,
        "total_duration": 1_000u64,
        "load_duration": 100u64,
        "prompt_eval_count": 2u64,
        "prompt_eval_duration": 200u64,
        "eval_count": 1u64,
        "eval_duration": 500u64,
    })
}

// ═══ g2_cli_reachable ═══

#[test]
fn falsify_crux_c_04_help_advertises_response_file_flag() {
    Command::cargo_bin("apr")
        .unwrap()
        .args(["ollama-chat-lint", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--response-file"));
}

#[test]
fn falsify_crux_c_04_help_advertises_stream_flag() {
    Command::cargo_bin("apr")
        .unwrap()
        .args(["ollama-chat-lint", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--stream"));
}

#[test]
fn falsify_crux_c_04_rejects_bare_invocation_without_file() {
    let output = Command::cargo_bin("apr")
        .unwrap()
        .args(["ollama-chat-lint"])
        .output()
        .expect("apr binary runs");
    assert!(
        !output.status.success(),
        "bare `apr ollama-chat-lint` without --response-file must fail"
    );
}

// ═══ g3_e2e_runs: non-streaming ═══

#[test]
fn falsify_crux_c_04_non_streaming_accepts_well_formed_response() {
    let f = write_json(&well_formed_response());
    let output = Command::cargo_bin("apr")
        .unwrap()
        .args([
            "--json",
            "ollama-chat-lint",
            "--response-file",
            f.path().to_str().unwrap(),
        ])
        .output()
        .expect("apr binary runs");
    assert!(
        output.status.success(),
        "well-formed response must pass: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let v: serde_json::Value =
        serde_json::from_str(&String::from_utf8_lossy(&output.stdout)).unwrap();
    assert_eq!(v["schema_ok"].as_bool(), Some(true));
    assert_eq!(v["eval_metrics_ok"].as_bool(), Some(true));
    assert_eq!(v["mode"].as_str(), Some("non_streaming"));
}

#[test]
fn falsify_crux_c_04_non_streaming_rejects_missing_required_key() {
    // Drop `eval_duration` — one of the 10 Ollama required keys.
    let mut resp = well_formed_response();
    resp.as_object_mut().unwrap().remove("eval_duration");
    let f = write_json(&resp);

    let output = Command::cargo_bin("apr")
        .unwrap()
        .args([
            "ollama-chat-lint",
            "--response-file",
            f.path().to_str().unwrap(),
        ])
        .output()
        .expect("apr binary runs");
    assert!(
        !output.status.success(),
        "response missing eval_duration must be rejected"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("MissingRequiredKey") || stderr.contains("schema"),
        "stderr should explain schema violation; got: {stderr}"
    );
}

#[test]
fn falsify_crux_c_04_non_streaming_rejects_wrong_message_role() {
    let mut resp = well_formed_response();
    resp["message"] = serde_json::json!({"role": "user", "content": "x"});
    let f = write_json(&resp);

    let output = Command::cargo_bin("apr")
        .unwrap()
        .args([
            "ollama-chat-lint",
            "--response-file",
            f.path().to_str().unwrap(),
        ])
        .output()
        .expect("apr binary runs");
    assert!(
        !output.status.success(),
        "message.role != 'assistant' must be rejected"
    );
}

#[test]
fn falsify_crux_c_04_non_streaming_rejects_done_false() {
    let mut resp = well_formed_response();
    resp["done"] = serde_json::json!(false);
    let f = write_json(&resp);

    let output = Command::cargo_bin("apr")
        .unwrap()
        .args([
            "ollama-chat-lint",
            "--response-file",
            f.path().to_str().unwrap(),
        ])
        .output()
        .expect("apr binary runs");
    assert!(!output.status.success(), "done=false must be rejected");
}

#[test]
fn falsify_crux_c_04_non_streaming_rejects_zero_eval_count_with_content() {
    let mut resp = well_formed_response();
    resp["eval_count"] = serde_json::json!(0u64);
    let f = write_json(&resp);

    let output = Command::cargo_bin("apr")
        .unwrap()
        .args([
            "ollama-chat-lint",
            "--response-file",
            f.path().to_str().unwrap(),
        ])
        .output()
        .expect("apr binary runs");
    assert!(
        !output.status.success(),
        "eval_count=0 with non-empty content must be rejected"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("EvalCountZero") || stderr.contains("eval-metrics"),
        "stderr should explain eval metrics violation; got: {stderr}"
    );
}

// ═══ g3_e2e_runs: streaming NDJSON ═══

#[test]
fn falsify_crux_c_04_stream_accepts_well_formed_ndjson() {
    let body = r#"{"message":{"role":"assistant","content":"he"},"done":false}
{"message":{"role":"assistant","content":"llo"},"done":false}
{"done":true,"eval_count":3,"eval_duration":500}
"#;
    let f = write_text(body);
    let output = Command::cargo_bin("apr")
        .unwrap()
        .args([
            "--json",
            "ollama-chat-lint",
            "--response-file",
            f.path().to_str().unwrap(),
            "--stream",
        ])
        .output()
        .expect("apr binary runs");
    assert!(
        output.status.success(),
        "well-formed NDJSON must pass: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let v: serde_json::Value =
        serde_json::from_str(&String::from_utf8_lossy(&output.stdout)).unwrap();
    assert_eq!(v["ndjson_ok"].as_bool(), Some(true));
    assert_eq!(v["num_frames"].as_u64(), Some(3));
    assert_eq!(v["mode"].as_str(), Some("streaming_ndjson"));
}

#[test]
fn falsify_crux_c_04_stream_rejects_early_done_true() {
    let body = r#"{"message":{"role":"assistant","content":"hi"},"done":false}
{"done":true}
{"message":{"role":"assistant","content":"!"},"done":false}
{"done":true,"eval_count":1,"eval_duration":10}
"#;
    let f = write_text(body);
    let output = Command::cargo_bin("apr")
        .unwrap()
        .args([
            "ollama-chat-lint",
            "--response-file",
            f.path().to_str().unwrap(),
            "--stream",
        ])
        .output()
        .expect("apr binary runs");
    assert!(
        !output.status.success(),
        "NDJSON with early done=true must be rejected"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("EarlyDone") || stderr.contains("NDJSON"),
        "stderr should explain early-done violation; got: {stderr}"
    );
}

#[test]
fn falsify_crux_c_04_stream_rejects_missing_terminal_done() {
    let body = r#"{"message":{"role":"assistant","content":"a"},"done":false}
{"message":{"role":"assistant","content":"b"},"done":false}
"#;
    let f = write_text(body);
    let output = Command::cargo_bin("apr")
        .unwrap()
        .args([
            "ollama-chat-lint",
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
fn falsify_crux_c_04_stream_rejects_empty_file() {
    let f = write_text("");
    let output = Command::cargo_bin("apr")
        .unwrap()
        .args([
            "ollama-chat-lint",
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
