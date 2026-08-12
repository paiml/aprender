//! FALSIFY-MCP-010/-011 — stdio TRANSPORT conformance for `apr mcp`.
//!
//! These two defects are invisible to an in-process dispatcher test, because
//! both live in the read loop rather than in request handling:
//!
//! * **FALSIFY-MCP-010** — `tools/call` runs on a worker thread that writes
//!   its own response. The read loop returned on EOF without joining workers,
//!   so `printf '<initialize>\n<tools/call>\n' | apr mcp` answered initialize,
//!   exited 0, and silently dropped the tool result. 3/3 deterministic on the
//!   shipped 0.63.0 binary.
//! * **FALSIFY-MCP-011** — one invalid UTF-8 byte on stdin propagated an
//!   `io::Error` out of `BufRead::lines()` and killed the process (exit 1),
//!   losing every request after it. Well-formed-UTF-8 malformed JSON was
//!   already handled correctly (-32700, keep serving), so the two disagreed.
//!
//! Both are asserted here by driving the real `apr mcp` binary, because a
//! unit test cannot observe "the process exited before the answer was
//! written".

#![allow(clippy::disallowed_methods)] // serde_json::json! expands to code that hits unwrap()

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

/// Hard cap on a whole stdio session. Anything slower is a hang, not a slow
/// machine — `apr.version` is answered in-process with no subprocess spawn.
const SESSION_TIMEOUT: Duration = Duration::from_secs(30);

const INITIALIZE: &str = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"falsifier","version":"1"}}}"#;
const TOOLS_CALL_VERSION: &str = r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"apr.version","arguments":{}}}"#;

/// Locate the workspace-built `apr`, building it on demand when the test
/// crate is exercised in isolation. Same approach as
/// `falsify_mcp_dogfood_001.rs`.
fn apr_binary() -> PathBuf {
    // ALWAYS build; never short-circuit on "the file exists".
    //
    // Returning an existing `cargo_bin("apr")` unconditionally means a binary
    // left in the shared target dir by ANY other commit is silently preferred.
    // That happened: these six falsifiers all failed against
    // `apr 0.63.0 (d16c608b1)` while the worktree was at 11f958f25 — the exact
    // pre-fix symptom ("stream did not contain valid UTF-8", exit 1), so the
    // fix under test looked broken when it was simply not the code running.
    // All six pass once the binary's embedded SHA matches HEAD.
    //
    // `cargo build` is a cheap no-op when the binary is already current, so
    // this costs nothing in the common case and removes the failure mode.
    // Same doctrine as scripts/apr_bin.sh, which hard-fails on a stale SHA.
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let pkg_spec = format!("aprender@{}", env!("CARGO_PKG_VERSION"));
    let status = Command::new(&cargo)
        .args(["build", "--bin", "apr", "-p", &pkg_spec, "--quiet"])
        .status()
        .expect("invoke `cargo build --bin apr`");
    assert!(
        status.success(),
        "cargo build --bin apr -p {pkg_spec} failed"
    );
    let path = assert_cmd::cargo::cargo_bin("apr");
    assert!(path.is_file(), "apr binary missing after cargo build");
    path
}

/// One complete stdio session: write `input`, CLOSE stdin (the whole point —
/// this is the EOF the server used to exit through), then read stdout to EOF.
///
/// Returns `(exit_code, parsed_response_lines, stderr)`.
fn drive_stdio(input: &[u8]) -> (Option<i32>, Vec<serde_json::Value>, String) {
    let mut child = Command::new(apr_binary())
        .arg("mcp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn `apr mcp`");

    let mut stdin = child.stdin.take().expect("piped stdin");
    stdin.write_all(input).expect("write stdin");
    stdin.flush().expect("flush stdin");
    drop(stdin); // EOF

    let mut stdout = child.stdout.take().expect("piped stdout");
    let (tx, rx) = mpsc::channel::<String>();
    let reader = std::thread::spawn(move || {
        let mut buf = String::new();
        let _ = stdout.read_to_string(&mut buf);
        let _ = tx.send(buf);
    });

    let out = match rx.recv_timeout(SESSION_TIMEOUT) {
        Ok(s) => s,
        Err(e) => {
            let _ = child.kill();
            panic!("`apr mcp` did not close stdout within {SESSION_TIMEOUT:?}: {e}");
        }
    };
    let _ = reader.join();

    let status = child.wait().expect("wait for `apr mcp`");
    let mut stderr = String::new();
    if let Some(mut e) = child.stderr.take() {
        let _ = e.read_to_string(&mut stderr);
    }

    let responses = out
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            serde_json::from_str::<serde_json::Value>(l)
                .unwrap_or_else(|e| panic!("`apr mcp` emitted a non-JSON stdout line {l:?}: {e}"))
        })
        .collect();

    (status.code(), responses, stderr)
}

fn find_id(responses: &[serde_json::Value], id: i64) -> Option<&serde_json::Value> {
    responses.iter().find(|r| r["id"] == serde_json::json!(id))
}

/// FALSIFY-MCP-010: every request carrying an `id` must be answered before
/// the server exits, INCLUDING a `tools/call` still in flight when stdin
/// closes. Run three times because the shipped defect was deterministic and
/// a single green run could otherwise be a scheduling accident.
#[test]
fn falsify_mcp_010_tools_call_is_answered_before_eof_exit() {
    let input = format!("{INITIALIZE}\n{TOOLS_CALL_VERSION}\n");

    for attempt in 1..=3 {
        let (code, responses, stderr) = drive_stdio(input.as_bytes());

        assert_eq!(
            code,
            Some(0),
            "attempt {attempt}: clean exit; stderr: {stderr}"
        );
        let call = find_id(&responses, 2).unwrap_or_else(|| {
            panic!(
                "attempt {attempt}: tools/call response (id=2) was DROPPED on stdin EOF. \
                 Got {} response(s): {:?}",
                responses.len(),
                responses
            )
        });
        assert!(
            call.get("error").is_none(),
            "attempt {attempt}: tools/call must succeed, got {call:?}"
        );
        let text = call["result"]["content"][0]["text"]
            .as_str()
            .unwrap_or_else(|| panic!("attempt {attempt}: missing content text in {call:?}"));
        let payload: serde_json::Value =
            serde_json::from_str(text).expect("apr.version payload is JSON");
        assert_eq!(
            payload["server"], "aprender-mcp",
            "attempt {attempt}: tool payload must be the real apr.version result"
        );
        assert!(
            find_id(&responses, 1).is_some(),
            "attempt {attempt}: initialize must still be answered"
        );
    }
}

// REMOVED: an "ordering variant" that sent an inline `tools/list` AFTER the
// tools/call. It was mutation-tested against the unfixed binary and PASSED —
// serializing the large tools/list response gives the worker enough time to
// finish writing before EOF, so the test could never observe the drop it
// claimed to guard. A test that stays green on the defect is theater; the two
// FALSIFY-MCP-010 cases that remain both turn RED (verbatim: "tools/call
// response (id=2) was DROPPED on stdin EOF").

/// FALSIFY-MCP-010 (concurrency): several `tools/call` requests in one
/// pipeline must ALL be answered, not just the ones that happened to finish
/// before EOF.
#[test]
fn falsify_mcp_010_every_pipelined_tools_call_is_answered() {
    let mut input = String::from(INITIALIZE);
    input.push('\n');
    for id in 2..=6 {
        input.push_str(&format!(
            r#"{{"jsonrpc":"2.0","id":{id},"method":"tools/call","params":{{"name":"apr.version","arguments":{{}}}}}}"#
        ));
        input.push('\n');
    }

    let (code, responses, stderr) = drive_stdio(input.as_bytes());
    assert_eq!(code, Some(0), "clean exit; stderr: {stderr}");
    for id in 1..=6 {
        assert!(
            find_id(&responses, id).is_some(),
            "id={id} unanswered; got {} of 6: {responses:?}",
            responses.len()
        );
    }
}

/// FALSIFY-MCP-011: an invalid UTF-8 byte is a malformed MESSAGE. It must
/// cost exactly that one message — a -32700 — and the session must keep
/// serving, matching how the server already treats malformed JSON.
#[test]
fn falsify_mcp_011_invalid_utf8_line_does_not_kill_the_session() {
    let mut input: Vec<u8> = Vec::new();
    input.extend_from_slice(br#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#);
    input.push(b'\n');
    input.push(0xFF); // lone continuation byte — never valid UTF-8
    input.push(b'\n');
    input.extend_from_slice(br#"{"jsonrpc":"2.0","id":2,"method":"ping"}"#);
    input.push(b'\n');

    let (code, responses, stderr) = drive_stdio(&input);

    assert_eq!(
        code,
        Some(0),
        "one bad byte must not kill the server; stderr: {stderr}"
    );
    assert!(
        find_id(&responses, 1).is_some(),
        "request before the bad byte must be answered: {responses:?}"
    );
    let after = find_id(&responses, 2)
        .unwrap_or_else(|| panic!("request AFTER the bad byte was lost; got {responses:?}"));
    assert!(
        after.get("error").is_none(),
        "request after the bad byte must be served normally, got {after:?}"
    );
    let parse_err = responses
        .iter()
        .find(|r| r["error"]["code"] == serde_json::json!(-32700))
        .unwrap_or_else(|| panic!("the bad line itself must be reported: {responses:?}"));
    assert!(
        parse_err["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("UTF-8"),
        "the -32700 must name the cause, got {parse_err:?}"
    );
}

/// FALSIFY-MCP-011 (leading byte): the bad byte arriving before any valid
/// request must not prevent the session from ever starting.
#[test]
fn falsify_mcp_011_leading_invalid_utf8_still_serves_the_first_request() {
    let mut input: Vec<u8> = vec![0x80, b'\n'];
    input.extend_from_slice(br#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#);
    input.push(b'\n');

    let (code, responses, stderr) = drive_stdio(&input);
    assert_eq!(code, Some(0), "clean exit; stderr: {stderr}");
    assert!(
        find_id(&responses, 1).is_some(),
        "the request after a leading bad byte must be answered: {responses:?}"
    );
}

/// FALSIFY-MCP-011 (embedded): an invalid 2-byte sequence inside an otherwise
/// well-formed request is rejected with -32700 and never silently repaired
/// into a different string value.
#[test]
fn falsify_mcp_011_invalid_utf8_inside_a_request_is_rejected_not_repaired() {
    let mut input: Vec<u8> = Vec::new();
    input.extend_from_slice(
        br#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"apr.validate","arguments":{"model_path":"/tmp/"#,
    );
    input.extend_from_slice(&[0xC3, 0x28]); // invalid 2-byte sequence
    input.extend_from_slice(br#".gguf"}}}"#);
    input.push(b'\n');
    input.extend_from_slice(br#"{"jsonrpc":"2.0","id":2,"method":"ping"}"#);
    input.push(b'\n');

    let (code, responses, stderr) = drive_stdio(&input);
    assert_eq!(code, Some(0), "clean exit; stderr: {stderr}");
    assert!(
        responses
            .iter()
            .any(|r| r["error"]["code"] == serde_json::json!(-32700)),
        "the malformed line must be reported as a parse error: {responses:?}"
    );
    assert!(
        find_id(&responses, 2).is_some(),
        "the following request must still be served: {responses:?}"
    );
}

/// End-to-end confirmation of the four request-handling fixes over the real
/// transport: version negotiation, `ping`, Invalid-Request classification and
/// batch-array diagnostics.
#[test]
fn falsify_mcp_stdio_protocol_surface_matches_jsonrpc_and_mcp() {
    let input = concat!(
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}"#,
        "\n",
        r#"{"jsonrpc":"2.0","id":2,"method":"ping"}"#,
        "\n",
        r#"{"id":3,"method":"tools/list"}"#,
        "\n",
        r#"{"jsonrpc":"2.0","id":4}"#,
        "\n",
        r#"[{"jsonrpc":"2.0","id":5,"method":"tools/list"}]"#,
        "\n",
        r#"{not json"#,
        "\n",
        r#"{"jsonrpc":"2.0","id":7,"method":"tools/list"}"#,
        "\n",
    );

    let (code, responses, stderr) = drive_stdio(input.as_bytes());
    assert_eq!(code, Some(0), "clean exit; stderr: {stderr}");

    // A newer proposed version negotiates DOWN to ours, it does not abort.
    let init = find_id(&responses, 1).expect("initialize answered");
    assert!(
        init.get("error").is_none(),
        "a newer protocolVersion must not abort the handshake: {init:?}"
    );
    assert_eq!(init["result"]["protocolVersion"], "2024-11-05");

    // ping is base protocol — empty result, never -32601.
    let pong = find_id(&responses, 2).expect("ping answered");
    assert_eq!(
        pong["result"],
        serde_json::json!({}),
        "ping must pong: {pong:?}"
    );

    // Valid JSON that isn't a valid Request is -32600, with the id echoed.
    for id in [3, 4] {
        let resp = find_id(&responses, id).unwrap_or_else(|| {
            panic!("id={id} must be echoed on an Invalid Request: {responses:?}")
        });
        assert_eq!(
            resp["error"]["code"],
            serde_json::json!(-32600),
            "id={id} must be Invalid Request, not Parse error: {resp:?}"
        );
    }

    // A batch array names batching.
    let batch = responses
        .iter()
        .find(|r| {
            r["error"]["message"]
                .as_str()
                .is_some_and(|m| m.contains("batch"))
        })
        .unwrap_or_else(|| panic!("batch array must be diagnosed as such: {responses:?}"));
    assert_eq!(batch["error"]["code"], serde_json::json!(-32600));

    // Genuinely malformed JSON is still -32700, and the session survives it.
    assert!(
        responses
            .iter()
            .any(|r| r["error"]["code"] == serde_json::json!(-32700)),
        "`{{not json` must remain a Parse error: {responses:?}"
    );
    assert!(
        find_id(&responses, 7).is_some(),
        "the server must keep serving after every malformed line: {responses:?}"
    );
}
