//! E2E-FORECAST-PMCP-001 — the thin fit server forecasts over live stdio MCP.
//!
//! stdio is the transport MCP clients (Claude Desktop, Claude Code, Cursor) actually
//! spawn, and it is NOT the transport the in-process suite in `src/lib.rs` drives — that
//! one speaks streamable-HTTP. This test closes that gap against the REAL binary:
//! `env!("CARGO_BIN_EXE_...")` pins the executable this crate just built, so there is no
//! PATH lookup and no shadowed-artifact ambiguity (CLAUDE.md Verification Discipline #3
//! and #8).
//!
//! Harness copied whole from `crates/aprender-mcp-setfit/tests/e2e_stdio.rs` (D-06):
//! `KillOnDrop`, the line-oriented reader thread, `request`/`send`/`recv_json`, the
//! "exactly ONE tool" assertion and the shutdown note. ONE thing is deliberately dropped:
//! the SetFit test is gated on `APR_MCP_E2E_SETFIT_MODEL` because no SetFit artifact can
//! be checked in. The fit server needs no model, no credentials and no network — the
//! request carries the series (D-01) — so there is NO environment gate here and this test
//! ALWAYS runs. A test that can silently not run proves nothing (D-18).
//!
//! DARK IN CI. `.github/workflows/ci.yml` runs `--lib` across the workspace plus one
//! explicit line listing individual `--test` targets; this target is not on that line, so
//! it does not execute in CI yet. Adding it is a human check-in (plan 06-08), not
//! something this plan does.

#![allow(clippy::disallowed_methods)] // serde_json::json! expands to .unwrap() internally

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use aprender_forecast::dates::{days_from_civil, format_ymd};

/// No model to load, so startup is process spawn only — but a cold debug binary on a
/// loaded box still deserves room.
const INIT_TIMEOUT: Duration = Duration::from_secs(60);
/// One Prophet fit on 60 points, in a debug build.
const CALL_TIMEOUT: Duration = Duration::from_secs(120);

/// Kill the server when the test unwinds — a panicking assertion must not orphan a child
/// that holds inherited pipes open.
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

fn notification(method: &str, params: serde_json::Value) -> String {
    serde_json::json!({ "jsonrpc": "2.0", "method": method, "params": params }).to_string()
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

/// A short synthetic daily series, strictly ascending from 2020-01-01.
fn synth(n: usize) -> (Vec<String>, Vec<f64>) {
    let t0 = days_from_civil(2020, 1, 1);
    let mut ds = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        ds.push(format_ymd(t0 + i as i64));
        let t = i as f64;
        y.push(10.0 + 0.05 * t + (2.0 * std::f64::consts::PI * t / 7.0).sin());
    }
    (ds, y)
}

#[test]
fn the_thin_server_forecasts_over_live_stdio() {
    // No arguments: stdio is the DEFAULT transport, and that default is part of what
    // MCP clients rely on when they spawn this binary.
    let mut child = KillOnDrop(
        Command::new(env!("CARGO_BIN_EXE_aprender-mcp-forecast"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn aprender-mcp-forecast"),
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
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": { "name": "e2e-stdio", "version": "0" }
            }),
        ),
    );
    let init = recv_json(&rx, INIT_TIMEOUT, "initialize response");
    assert_eq!(init["id"], 1);
    assert!(init.get("error").is_none(), "initialize failed: {init:?}");

    // The handshake the specification asks for. It is a notification, so nothing comes
    // back and nothing may be waited on.
    send(
        &mut stdin,
        &notification("notifications/initialized", serde_json::json!({})),
    );

    // The thin philosophy is observable over stdio too: exactly ONE tool, strict schema.
    send(&mut stdin, &request(2, "tools/list", serde_json::json!({})));
    let list = recv_json(&rx, INIT_TIMEOUT, "tools/list response");
    let tools = list["result"]["tools"].as_array().expect("tools array");
    assert_eq!(
        tools.len(),
        1,
        "a thin single-purpose server advertises exactly one tool: {tools:?}"
    );
    assert_eq!(tools[0]["name"], aprender_mcp_forecast::TOOL_NAME);
    assert_eq!(
        tools[0]["inputSchema"]["additionalProperties"],
        serde_json::json!(false),
        "deny_unknown_fields must be visible to clients over stdio as well"
    );

    // One real fit, inside one call — the D-01 shape, over the spawned binary.
    let (ds, y) = synth(60);
    send(
        &mut stdin,
        &request(
            3,
            "tools/call",
            serde_json::json!({
                "name": aprender_mcp_forecast::TOOL_NAME,
                "arguments": { "ds": ds, "y": y, "horizon": 7 }
            }),
        ),
    );
    let call = recv_json(&rx, CALL_TIMEOUT, "forecast tools/call response");
    assert_eq!(call["id"], 3);
    assert!(call.get("error").is_none(), "tools/call errored: {call:?}");
    let result = &call["result"];
    assert_ne!(
        result["isError"],
        serde_json::json!(true),
        "forecast must not be an in-band error: {result:?}"
    );

    // pmcp serializes a Value-returning tool as JSON text content — parse it back and
    // hold it to the shared response shape.
    let payload: serde_json::Value = if let Some(structured) = result.get("structuredContent") {
        structured.clone()
    } else {
        let text = result["content"][0]["text"]
            .as_str()
            .expect("content[0].text must be a string");
        serde_json::from_str(text).expect("tool JSON")
    };
    let out_ds = payload["ds"].as_array().expect("ds array");
    assert_eq!(out_ds.len(), 7, "one row per horizon step: {payload}");
    let yhat = payload["yhat"].as_array().expect("yhat array");
    assert_eq!(yhat.len(), 7);
    for (i, v) in yhat.iter().enumerate() {
        assert!(
            v.as_f64().expect("yhat entry is a number").is_finite(),
            "yhat[{i}] must be finite: {payload}"
        );
    }

    // A strictness probe THROUGH the live binary: an unmodeled key must come back as an
    // error, not a silent forecast that ignored it (D-11).
    send(
        &mut stdin,
        &request(
            4,
            "tools/call",
            serde_json::json!({
                "name": aprender_mcp_forecast::TOOL_NAME,
                "arguments": { "ds": ds, "y": y, "horizon": 7, "temperature": 0.7 }
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
    assert!(
        strict.to_string().contains("temperature"),
        "the refusal must name the offending key: {strict:?}"
    );

    // pmcp's `run_stdio` does NOT exit on stdin EOF (measured for the SetFit server: a
    // /dev/null stdin left the process alive past 590 s), so shutdown here is the client's
    // kill — exactly what real MCP clients do to stdio servers. No exit-status assertion:
    // a killed process reports a signal, not success. KillOnDrop does the reaping.
    drop(stdin);
    drop(child);
    reader.join().expect("stdout reader thread panicked");
}
