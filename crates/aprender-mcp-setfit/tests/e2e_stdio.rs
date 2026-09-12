//! E2E-SETFIT-PMCP-001 — the thin server classifies over live stdio MCP.
//!
//! Env-gated on `APR_MCP_E2E_SETFIT_MODEL`, with a println! SKIP + early return
//! rather than `#[ignore]`, so a run without an artifact says so out loud
//! instead of reporting a silent pass. No SetFit-tagged artifact can be checked
//! in (F-10), and no in-tree command produces one in a single step yet, so the
//! model has to come from a training run of your own — build the artifact with
//! `aprender::setfit::import` + `write_setfit_apr` and point the variable at it.
//!
//! The binary under test is pinned with `env!("CARGO_BIN_EXE_...")` — no PATH
//! lookup and no cargo_bin guess, so there is no shadowed-artifact ambiguity
//! about which build answered.

#![allow(clippy::disallowed_methods)] // serde_json::json! expands to .unwrap() internally

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

const ENV_MODEL: &str = "APR_MCP_E2E_SETFIT_MODEL";
/// The initialize response arrives only after the model loads, and a DEBUG
/// build spends ~47s of CPU on the artifact verification ladder (measured
/// 2026-08-18; release loads in under a second). 10s here failed for exactly
/// that reason.
const INIT_TIMEOUT: Duration = Duration::from_secs(180);
/// The call budget covers a debug-build forward pass on CPU.
const CALL_TIMEOUT: Duration = Duration::from_secs(120);

/// Kill the server when the test unwinds — a panicking assertion must not
/// orphan a child that holds inherited pipes open (that orphan turned one
/// 10-second failure into a 600-second harness timeout).
struct KillOnDrop(std::process::Child);

impl Drop for KillOnDrop {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn request(id: u64, method: &str, params: serde_json::Value) -> String {
    serde_json::json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params })
        .to_string()
}

fn send(stdin: &mut impl Write, line: &str) {
    stdin.write_all(line.as_bytes()).expect("write request");
    stdin.write_all(b"\n").expect("write newline");
    stdin.flush().expect("flush stdin");
}

fn recv_json(rx: &mpsc::Receiver<String>, timeout: Duration, what: &str) -> serde_json::Value {
    let line = rx
        .recv_timeout(timeout)
        .unwrap_or_else(|e| panic!("timed out waiting for {what}: {e}"));
    serde_json::from_str(&line)
        .unwrap_or_else(|e| panic!("non-JSON line while waiting for {what}: {e}\n{line}"))
}

#[test]
fn the_thin_server_classifies_a_batch_over_live_stdio() {
    let Some(model) = std::env::var_os(ENV_MODEL) else {
        println!(
            "E2E-SETFIT-PMCP-001 SKIP: {ENV_MODEL} unset — train a setfit-apr-v1 artifact \
             (see this file's header) and export the var"
        );
        return;
    };
    let model = PathBuf::from(model);
    assert!(
        model.is_file(),
        "{ENV_MODEL} points at {}, which is not a file",
        model.display()
    );

    let mut child = KillOnDrop(
        Command::new(env!("CARGO_BIN_EXE_aprender-mcp-setfit"))
            .arg("--model")
            .arg(&model)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn aprender-mcp-setfit"),
    );
    let mut stdin = child.0.stdin.take().expect("piped stdin");
    let stdout = child.0.stdout.take().expect("piped stdout");

    let (tx, rx) = mpsc::channel::<String>();
    let reader = thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            if line.trim().is_empty() {
                continue;
            }
            if tx.send(line).is_err() {
                break;
            }
        }
    });

    send(
        &mut stdin,
        &request(
            1,
            "initialize",
            serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": { "name": "e2e", "version": "0" }
            }),
        ),
    );
    let init = recv_json(&rx, INIT_TIMEOUT, "initialize response");
    assert_eq!(init["id"], 1);
    assert!(init.get("error").is_none(), "initialize failed: {init:?}");

    // The thin philosophy is observable: exactly ONE tool.
    send(&mut stdin, &request(2, "tools/list", serde_json::json!({})));
    let list = recv_json(&rx, INIT_TIMEOUT, "tools/list response");
    let tools = list["result"]["tools"].as_array().expect("tools array");
    assert_eq!(
        tools.len(),
        1,
        "a thin single-model server advertises exactly one tool: {tools:?}"
    );
    assert_eq!(tools[0]["name"], aprender_mcp_setfit::TOOL_NAME);
    assert_eq!(
        tools[0]["inputSchema"]["additionalProperties"],
        serde_json::json!(false),
        "deny_unknown_fields must be visible to clients"
    );

    send(
        &mut stdin,
        &request(
            3,
            "tools/call",
            serde_json::json!({
                "name": aprender_mcp_setfit::TOOL_NAME,
                "arguments": {
                    "texts": [
                        "Every woman deserves the right to make her own healthcare decisions.",
                        "The weather in Lisbon was lovely this weekend."
                    ]
                }
            }),
        ),
    );
    let call = recv_json(&rx, CALL_TIMEOUT, "classify tools/call response");
    assert_eq!(call["id"], 3);
    assert!(call.get("error").is_none(), "tools/call errored: {call:?}");
    let result = &call["result"];
    assert_ne!(
        result["isError"],
        serde_json::json!(true),
        "classify must not be an in-band error: {result:?}"
    );

    // pmcp serializes a Value-returning tool as JSON text content — parse it
    // back and hold it to the ClassifyResponse envelope every surface shares.
    let text = result["content"][0]["text"]
        .as_str()
        .expect("content[0].text must be a string");
    let payload: serde_json::Value = serde_json::from_str(text).expect("CLI-shaped JSON");
    assert_eq!(payload["schema_version"], 1);
    let sha = payload["artifact_sha256"].as_str().expect("artifact sha");
    assert_eq!(sha.len(), 64);
    let results = payload["results"].as_array().expect("results array");
    assert_eq!(results.len(), 2, "one result per text, in order");
    for (i, row) in results.iter().enumerate() {
        assert!(
            !row["label"].as_str().unwrap_or_default().is_empty(),
            "results[{i}].label must be non-empty"
        );
        let sum: f64 = row["probabilities"]
            .as_array()
            .unwrap_or_else(|| panic!("results[{i}].probabilities: {row}"))
            .iter()
            .filter_map(serde_json::Value::as_f64)
            .sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "results[{i}] probabilities must sum to 1.0, got {sum}"
        );
    }

    // A strictness probe THROUGH the live server: an unmodeled key must come
    // back as an error, not a silent classification.
    send(
        &mut stdin,
        &request(
            4,
            "tools/call",
            serde_json::json!({
                "name": aprender_mcp_setfit::TOOL_NAME,
                "arguments": { "texts": ["a"], "temperature": 0.7 }
            }),
        ),
    );
    let strict = recv_json(&rx, CALL_TIMEOUT, "unknown-key refusal");
    let refused =
        strict.get("error").is_some() || strict["result"]["isError"] == serde_json::json!(true);
    assert!(
        refused,
        "an unknown argument key must be refused end-to-end: {strict:?}"
    );

    // pmcp 2.9's `run_stdio` does NOT exit on stdin EOF (measured 2026-08-18:
    // a /dev/null stdin left the process alive past 590s), so shutdown here is
    // the client's kill — which is exactly what real MCP clients (Claude
    // Desktop, Cursor) do to stdio servers. No exit-status assertion: a killed
    // process reports a signal, not success. KillOnDrop does the reaping.
    drop(stdin);
    drop(child);
    reader.join().expect("stdout reader thread panicked");
}
