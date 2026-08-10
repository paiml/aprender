//! FALSIFY-MCP-011 / -012 — stdio transport survivability and delivery.
//!
//! These two gates cover the failure modes that a real MCP client hits and
//! that no in-process dispatcher test can see, because they live in
//! `run_stdio`'s read loop rather than in `handle_request`.
//!
//! **FALSIFY-MCP-011 — a bad line must cost one line, not the session.**
//! `apr` 0.63.0 read stdin with `BufRead::lines()`, which yields an
//! `io::Error` on the first non-UTF-8 byte, and propagated it straight out
//! of `run_stdio`. One stray `0xFF` killed the server with exit 1 and every
//! later request on the session was silently lost. This falsifier feeds a
//! raw `0xFF` line mid-stream and asserts the requests on *both* sides of it
//! are answered. It also pins the JSON-RPC error taxonomy the same loop
//! owns: a syntactically valid object missing `jsonrpc` or `method` is
//! `-32600 Invalid Request` with its id echoed (0.63.0 said `-32700` with
//! `id: null`), and a batch array is declined by name instead of surfacing
//! serde's "invalid type: map, expected a string".
//!
//! **FALSIFY-MCP-012 — every request carrying an id gets an answer.**
//! `tools/call` is dispatched on a worker thread. 0.63.0 dropped the
//! `JoinHandle` and returned from `run_stdio` the moment stdin hit EOF, so
//! the worker's response was never written. `printf ... | apr mcp` — the
//! canonical scripted-client form — exited 0 having answered `initialize`
//! and `tools/list` while silently discarding the tool result, which is
//! indistinguishable from a tool that produced no output.
//!
//! Both tests drive the shipped `apr mcp` binary over real pipes. Neither
//! needs a model or a mock subprocess: `apr.version` is answered in-process.

#![allow(clippy::disallowed_methods)] // serde_json::json! expands to code that hits unwrap()

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

/// Hard cap on a whole session. Anything slower is a hang, not a slow
/// machine — every request here is answered without touching disk.
const SESSION_TIMEOUT: Duration = Duration::from_secs(30);

/// Locate the workspace-built `apr`, building it on demand when the test
/// crate is exercised in isolation. Same approach as
/// `tests/falsify_mcp_dogfood_001.rs`.
fn apr_binary() -> PathBuf {
    let candidate = assert_cmd::cargo::cargo_bin("apr");
    if candidate.is_file() {
        return candidate;
    }
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let pkg_spec = format!("aprender@{}", env!("CARGO_PKG_VERSION"));
    let status = Command::new(&cargo)
        .args(["build", "--bin", "apr", "-p", &pkg_spec, "--quiet"])
        .status()
        .expect("invoke `cargo build --bin apr`");
    assert!(
        status.success(),
        "cargo build --bin apr -p {pkg_spec} failed with status {status:?}"
    );
    let path = assert_cmd::cargo::cargo_bin("apr");
    assert!(
        path.is_file(),
        "expected apr binary at {} after `cargo build`",
        path.display()
    );
    path
}

/// One completed `apr mcp` session: the raw stdin bytes we wrote, and what
/// came back.
struct Session {
    exit_code: Option<i32>,
    stdout: String,
    stderr: String,
}

impl Session {
    /// Every stdout line parsed as JSON. Panics with the offending line if
    /// the server emitted anything that is not one JSON object per line —
    /// that is itself a transport violation.
    fn responses(&self) -> Vec<serde_json::Value> {
        self.stdout
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                serde_json::from_str(l)
                    .unwrap_or_else(|e| panic!("stdout line was not JSON ({e}): {l}"))
            })
            .collect()
    }

    /// The single response carrying `id`, or a panic naming everything that
    /// *was* returned — the message a dropped response needs to be readable.
    fn by_id(&self, id: u64) -> serde_json::Value {
        let all = self.responses();
        all.iter()
            .find(|r| r["id"] == serde_json::json!(id))
            .cloned()
            .unwrap_or_else(|| {
                panic!(
                    "no response for id={id}; server returned {} response(s): {}\nstderr: {}",
                    all.len(),
                    self.stdout.trim(),
                    self.stderr.trim()
                )
            })
    }
}

/// Run one `apr mcp` session: write `input` to stdin, close it, read stdout
/// to EOF, wait for exit. Fails loudly rather than hanging forever.
fn run_session(input: &[u8]) -> Session {
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
    // Closing stdin is the whole point of FALSIFY-MCP-012: EOF must not be
    // permission to abandon an in-flight worker.
    drop(stdin);

    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(child.wait_with_output());
    });

    let output = rx
        .recv_timeout(SESSION_TIMEOUT)
        .unwrap_or_else(|e| panic!("`apr mcp` did not exit within {SESSION_TIMEOUT:?}: {e}"))
        .expect("collect `apr mcp` output");

    Session {
        exit_code: output.status.code(),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

/// FALSIFY-MCP-011: one invalid UTF-8 byte costs one line, not the session;
/// and shape errors carry the right JSON-RPC code with the id echoed.
#[test]
#[cfg(unix)]
fn falsify_mcp_011_malformed_line_does_not_kill_the_server() {
    let mut input: Vec<u8> = Vec::new();
    input.extend_from_slice(br#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#);
    input.push(b'\n');
    // The byte that used to end the session.
    input.push(0xFF);
    input.push(b'\n');
    // An invalid 2-byte sequence embedded in an otherwise well-formed request.
    input.extend_from_slice(br#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"apr."#);
    input.extend_from_slice(&[0xC3, 0x28]);
    input.extend_from_slice(br#"version"}}"#);
    input.push(b'\n');
    // Well-formed JSON that is not a well-formed Request object.
    input.extend_from_slice(b"{\"id\":3,\"method\":\"tools/list\"}\n");
    input.extend_from_slice(b"{\"jsonrpc\":\"2.0\",\"id\":4}\n");
    // A JSON-RPC batch array.
    input.extend_from_slice(
        br#"[{"jsonrpc":"2.0","id":90,"method":"tools/list"},{"jsonrpc":"2.0","id":91,"method":"tools/list"}]"#,
    );
    input.push(b'\n');
    // Genuinely malformed JSON — the one true -32700.
    input.extend_from_slice(b"{not json\n");
    // The server must still be serving.
    input.extend_from_slice(b"{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"tools/list\"}\n");

    let session = run_session(&input);

    assert_eq!(
        session.exit_code,
        Some(0),
        "a malformed line must not terminate the server; stderr: {}",
        session.stderr.trim()
    );
    assert!(
        !session.stderr.contains("valid UTF-8"),
        "server must not surface a UTF-8 decode failure as a fatal error; stderr: {}",
        session.stderr.trim()
    );

    // Requests on BOTH sides of the bad byte are answered.
    let first = session.by_id(1);
    assert!(
        first["result"]["tools"].is_array(),
        "request before the invalid byte must be answered normally, got: {first}"
    );
    let last = session.by_id(7);
    assert!(
        last["result"]["tools"].is_array(),
        "request after the invalid byte must still be answered, got: {last}"
    );

    // Missing `jsonrpc` is Invalid Request, not Parse error, and the id is
    // echoed so the client can correlate the failure.
    let missing_jsonrpc = session.by_id(3);
    assert_eq!(
        missing_jsonrpc["error"]["code"], -32600,
        "a valid JSON object missing `jsonrpc` is -32600 Invalid Request, not -32700; got: {missing_jsonrpc}"
    );
    let missing_method = session.by_id(4);
    assert_eq!(
        missing_method["error"]["code"], -32600,
        "a valid JSON object missing `method` is -32600 Invalid Request, not -32700; got: {missing_method}"
    );

    // The batch array is declined by name. It has no usable id, so find it
    // by its message rather than by id.
    let batch_error = session
        .responses()
        .into_iter()
        .find(|r| {
            r["error"]["message"]
                .as_str()
                .is_some_and(|m| m.contains("batch"))
        })
        .unwrap_or_else(|| {
            panic!(
                "no response mentioned batching; server returned: {}",
                session.stdout.trim()
            )
        });
    assert_eq!(
        batch_error["error"]["code"], -32600,
        "batch arrays are refused as Invalid Request, got: {batch_error}"
    );
    let batch_message = batch_error["error"]["message"]
        .as_str()
        .expect("batch error message is a string");
    assert!(
        !batch_message.contains("invalid type"),
        "batch refusal must not leak serde's internal wording, got: {batch_message}"
    );

    // Genuinely malformed JSON is still -32700.
    let parse_errors: Vec<_> = session
        .responses()
        .into_iter()
        .filter(|r| r["error"]["code"] == serde_json::json!(-32700))
        .collect();
    assert!(
        !parse_errors.is_empty(),
        "`{{not json` must still yield -32700 Parse error; server returned: {}",
        session.stdout.trim()
    );
}

/// FALSIFY-MCP-012: a `tools/call` answer is delivered even when stdin
/// reaches EOF immediately after the request — the `printf ... | apr mcp`
/// form every scripted client uses.
#[test]
#[cfg(unix)]
fn falsify_mcp_012_tools_call_response_survives_stdin_eof() {
    let input = concat!(
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"t","version":"1"}}}"#,
        "\n",
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"apr.version","arguments":{}}}"#,
        "\n",
    );

    let session = run_session(input.as_bytes());

    assert_eq!(
        session.exit_code,
        Some(0),
        "server must exit cleanly; stderr: {}",
        session.stderr.trim()
    );

    // The regression: id=1 was present and id=2 was not.
    let init = session.by_id(1);
    assert_eq!(init["result"]["protocolVersion"], "2024-11-05");

    let call = session.by_id(2);
    assert!(
        call.get("error").is_none(),
        "tools/call must not be a JSON-RPC error, got: {call}"
    );
    let text = call["result"]["content"][0]["text"]
        .as_str()
        .unwrap_or_else(|| panic!("tools/call result must carry text content, got: {call}"));
    let payload: serde_json::Value =
        serde_json::from_str(text).expect("apr.version payload parses as JSON");
    assert_eq!(
        payload["server"], "aprender-mcp",
        "the delivered response must be the real tool result, got: {payload}"
    );
}

/// FALSIFY-MCP-007 / -010 at the binary surface: a client negotiating a
/// newer protocol version completes the handshake and can go on to use the
/// server, and `ping` is answered.
///
/// 0.63.0 hard-errored `-32602` for every `protocolVersion` other than the
/// literal `"2024-11-05"` — including OLDER dated versions, so it was not
/// even a floor check — which locked out Claude Code, Cursor and Cline, all
/// of which negotiate 2025-03-26 or 2025-06-18.
#[test]
#[cfg(unix)]
fn falsify_mcp_007_010_newer_version_client_completes_handshake_and_pings() {
    let mut input = String::new();
    for (id, version) in [
        (1_u64, "2025-06-18"),
        (2, "2025-03-26"),
        (3, "2024-11-05"),
        (4, "2024-10-07"),
    ] {
        input.push_str(&format!(
            r#"{{"jsonrpc":"2.0","id":{id},"method":"initialize","params":{{"protocolVersion":"{version}","capabilities":{{}},"clientInfo":{{"name":"claude-code","version":"1"}}}}}}"#
        ));
        input.push('\n');
    }
    input.push_str(r#"{"jsonrpc":"2.0","id":5,"method":"ping"}"#);
    input.push('\n');
    // Negotiation must actually advance — not just avoid an error.
    input.push_str(r#"{"jsonrpc":"2.0","id":6,"method":"tools/list"}"#);
    input.push('\n');

    let session = run_session(input.as_bytes());
    assert_eq!(session.exit_code, Some(0), "stderr: {}", session.stderr.trim());

    for id in 1..=4_u64 {
        let resp = session.by_id(id);
        assert!(
            resp.get("error").is_none(),
            "initialize must never abort the handshake over a version string, got: {resp}"
        );
        assert_eq!(
            resp["result"]["protocolVersion"], "2024-11-05",
            "the server must answer with the version it DOES support, got: {resp}"
        );
        assert_eq!(resp["result"]["serverInfo"]["name"], "aprender-mcp");
    }

    let pong = session.by_id(5);
    assert!(
        pong.get("error").is_none(),
        "ping is a base-protocol method, not an advertised capability; got: {pong}"
    );
    assert_eq!(
        pong["result"],
        serde_json::json!({}),
        "ping must answer with an empty result, got: {pong}"
    );

    let list = session.by_id(6);
    assert!(
        list["result"]["tools"].is_array(),
        "the session must remain usable after negotiating a newer version, got: {list}"
    );
}
