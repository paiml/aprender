//! e2e: the `mcp` transport (JSON-RPC 2.0 over stdio), against the SHIPPED BINARY.
//!
//! Declared in the root `Cargo.toml` as
//! `[package.metadata.transports] mcp = { e2e = "e2e_mcp_stdio_t", features = ["cli"] }`.
//!
//! `apr mcp` is what `.mcp.json` spawns (`{"command":"apr","args":["mcp"]}`),
//! so the only faithful test is the same thing: spawn the artifact with piped
//! stdio, speak newline-delimited JSON-RPC at it, and read what comes back.
//!
//! Hermetic: one child process, pipes only, no network, no model. Every wait
//! carries a deadline — a stdio server that never answers must fail the test,
//! not hang the suite.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

/// Per-line read budget. Generous: the first response follows a cold start of a
/// debug-profile binary on a shared, deliberately over-subscribed box.
const LINE_TIMEOUT: Duration = Duration::from_secs(60);
/// Budget for the process to exit once its stdin is closed.
const EXIT_TIMEOUT: Duration = Duration::from_secs(30);

/// A spawned `apr mcp` whose stdout is drained by a reader thread.
///
/// The thread matters: reading the child's stdout inline would deadlock the
/// moment the child wrote more than a pipe buffer while we were writing to its
/// stdin. Handing lines to an `mpsc` channel also gives us the only timeout
/// primitive std offers for a blocking read — `Receiver::recv_timeout`.
struct McpServer {
    child: Child,
    stdin: Option<ChildStdin>,
    lines: Receiver<String>,
}

impl McpServer {
    fn spawn() -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_apr"))
            .arg("mcp")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn the shipped apr binary with `mcp`");

        let stdout = child.stdout.take().expect("piped stdout");
        let stdin = child.stdin.take().expect("piped stdin");
        let (tx, lines) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if tx.send(line).is_err() {
                    break;
                }
            }
        });

        Self {
            child,
            stdin: Some(stdin),
            lines,
        }
    }

    fn send(&mut self, msg: &str) {
        let pipe = self.stdin.as_mut().expect("stdin still open");
        pipe.write_all(msg.as_bytes())
            .expect("write JSON-RPC request");
        pipe.write_all(b"\n").expect("write newline delimiter");
        pipe.flush().expect("flush JSON-RPC request");
    }

    /// Next stdout line, or a described failure. Blank lines and any line that
    /// is not a JSON object are skipped: a server is free to log, and this test
    /// asserts about its JSON-RPC replies, not its chatter.
    fn recv_json(&self) -> Result<String, String> {
        let deadline = Instant::now() + LINE_TIMEOUT;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(format!("no JSON line within {LINE_TIMEOUT:?}"));
            }
            match self.lines.recv_timeout(left) {
                Ok(line) if line.trim_start().starts_with('{') => return Ok(line),
                Ok(_) => {}
                Err(RecvTimeoutError::Timeout) => {
                    return Err(format!("no JSON line within {LINE_TIMEOUT:?}"))
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err("stdout closed before a JSON line arrived".to_string())
                }
            }
        }
    }

    /// The reply carrying this id, skipping notifications and anything else the
    /// server interleaves.
    fn recv_reply(&self, id: u32) -> String {
        let needle = format!("\"id\":{id}");
        let deadline = Instant::now() + LINE_TIMEOUT;
        loop {
            let line = self.recv_json().unwrap_or_else(|e| {
                panic!("waiting for JSON-RPC reply id={id}: {e}");
            });
            if line.contains(&needle) {
                return line;
            }
            assert!(
                Instant::now() < deadline,
                "no reply with id={id} within {LINE_TIMEOUT:?}"
            );
        }
    }

    /// Close stdin and require the server to exit — the shutdown path a real
    /// MCP client uses when it drops the connection.
    fn close_and_wait(&mut self) -> bool {
        drop(self.stdin.take());
        let deadline = Instant::now() + EXIT_TIMEOUT;
        while Instant::now() < deadline {
            match self.child.try_wait().expect("poll the mcp child") {
                Some(_) => return true,
                None => thread::sleep(Duration::from_millis(50)),
            }
        }
        false
    }
}

impl Drop for McpServer {
    fn drop(&mut self) {
        // A panicking assertion must not leave a stdio server behind.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Value of the first `"<key>":"<string>"` at or after `from`, if any.
fn string_field_after(hay: &str, from: usize, key: &str) -> Option<String> {
    let pat = format!("\"{key}\":\"");
    let at = hay[from..].find(&pat)? + from + pat.len();
    let end = hay[at..].find('"')? + at;
    Some(hay[at..end].to_string())
}

#[test]
fn initialize_then_tools_list_over_stdio() {
    let mut server = McpServer::spawn();

    server.send(
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"e2e_mcp_stdio_t","version":"0"}}}"#,
    );
    let init = server.recv_reply(1);

    assert!(
        init.contains("\"jsonrpc\":\"2.0\"") && init.contains("\"result\""),
        "initialize did not answer with a JSON-RPC 2.0 result: {init}"
    );
    assert!(
        init.contains("\"capabilities\""),
        "initialize result declares no capabilities: {init}"
    );
    let si = init
        .find("\"serverInfo\"")
        .unwrap_or_else(|| panic!("initialize result has no serverInfo: {init}"));
    let name = string_field_after(&init, si, "name")
        .unwrap_or_else(|| panic!("serverInfo has no name: {init}"));
    assert!(
        !name.is_empty(),
        "serverInfo.name is empty — the server does not identify itself: {init}"
    );

    server.send(r#"{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}"#);
    let listed = server.recv_reply(2);

    assert!(
        listed.contains("\"result\""),
        "tools/list answered with an error, not a result: {listed}"
    );
    let at = listed
        .find("\"tools\":[")
        .unwrap_or_else(|| panic!("tools/list result has no tools array: {listed}"));
    let tools = &listed[at..];
    let count = tools.matches("\"inputSchema\"").count();
    assert!(
        count >= 1,
        "tools/list returned an EMPTY tool list — an MCP transport that \
         advertises nothing is unreachable in practice: {listed}"
    );
    let first = string_field_after(tools, 0, "name")
        .unwrap_or_else(|| panic!("first tool has no name: {listed}"));
    assert!(!first.is_empty(), "first tool has an empty name: {listed}");

    eprintln!("mcp stdio: server={name}, {count} tool(s), first={first}");

    assert!(
        server.close_and_wait(),
        "apr mcp did not exit within {EXIT_TIMEOUT:?} of its stdin closing — a \
         stdio server that outlives its client leaks a process per session"
    );
}
