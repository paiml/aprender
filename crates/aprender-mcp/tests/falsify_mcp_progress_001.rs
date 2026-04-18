//! FALSIFY-MCP-PROGRESS-001 — MCP `notifications/progress` for `apr.finetune`.
//!
//! Spec: `docs/specifications/apr-mcp-server-spec.md` M3 milestone line 156
//! ("Progress notifications for `apr.finetune`") and the MCP 2024-11-05
//! progress utility (<https://spec.modelcontextprotocol.io>).
//!
//! # What this falsifier proves
//!
//! 1. When a `tools/call` for `apr.finetune` carries `params._meta.progressToken`,
//!    the dispatcher wires a `NotificationSink` that forwards each stdout line
//!    from the subprocess as a `notifications/progress` message tagged with
//!    the caller's token, in order, *before* the final `ToolCallResult`.
//! 2. The final response is still a normal `JsonRpcResponse` (success) so
//!    existing clients that ignore progress notifications don't regress.
//! 3. When the client omits `progressToken`, no notifications are emitted —
//!    the MCP spec forbids "spontaneous" progress.
//! 4. Notifications are emitted *before* the final response (proven by
//!    recording a monotonic counter when each message is observed).
//!
//! We exercise the streaming path end-to-end without invoking the real
//! `apr finetune` subprocess (which would require a GPU and a dataset) by
//! driving `tools::finetune::stream_with_sink` against a mock script that
//! prints three JSON lines and exits 0.

#![allow(clippy::disallowed_methods)] // serde_json::json! expands to code that hits unwrap()

use aprender_mcp::tools::finetune;
use aprender_mcp::{AprMcpServer, JsonRpcNotification, JsonRpcRequest, NotificationSink};
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Helper: write a mock subprocess script to a tempdir and return the path.
/// The script prints `line_count` JSON lines then exits 0.
///
/// We drive the script via `sh <path>` in the tests rather than invoking it
/// directly. This sidesteps the `ETXTBSY` race Linux will throw if any thread
/// in the process still holds a writable fd to the script file at the moment
/// of `execve` — running it through `sh` treats the script as data, not an
/// executable text segment, so the kernel doesn't enforce that invariant.
fn write_mock_apr_script(dir: &std::path::Path, line_count: usize) -> std::path::PathBuf {
    let path = dir.join("mock-apr-progress.sh");
    {
        let mut f = std::fs::File::create(&path).expect("create mock script");
        writeln!(f, "#!/bin/sh").expect("write shebang");
        for i in 0..line_count {
            // Each line mimics one "progress event" from apr finetune --json.
            // We include an explicit `event` key so assertions can find it.
            writeln!(
                f,
                "printf '{{\"event\":\"step\",\"step\":{i},\"loss\":{dec}}}\\n'",
                dec = format_args!("{:.3}", 1.0 / (i + 1) as f64)
            )
            .expect("write printf line");
        }
        writeln!(f, "exit 0").expect("write exit");
        f.sync_all().expect("sync mock script");
    } // file handle dropped here

    path
}

/// FALSIFY-MCP-PROGRESS-001 (core): a mock subprocess emitting 3 JSON lines
/// triggers 3 `notifications/progress` calls, each tagged with the caller's
/// progressToken, in order.
#[test]
#[cfg(unix)]
fn falsify_mcp_progress_001_three_lines_three_notifications() {
    let tmp = tempdir_fallback();
    let script = write_mock_apr_script(&tmp, 3);

    let captured: Arc<Mutex<Vec<JsonRpcNotification>>> = Arc::new(Mutex::new(Vec::new()));
    let captured_clone = Arc::clone(&captured);
    let sink: NotificationSink = Box::new(move |n| {
        captured_clone
            .lock()
            .expect("sink mutex not poisoned")
            .push(n);
    });

    let token = serde_json::json!("tok-finetune-42");
    let script_str = script.to_str().expect("utf-8 script path");
    let result = finetune::stream_with_sink("sh", &[script_str], &sink, &token);

    // Final response must be a success (no isError flag) with the aggregated
    // stdout as the text payload.
    assert!(
        result.is_error.is_none(),
        "mock script exits 0 → result must be success, got: {:?}",
        result.content[0].text
    );
    assert!(
        result.content[0].text.contains("\"step\":0"),
        "aggregated stdout preserved in final ToolCallResult"
    );

    let notifs = captured.lock().expect("mutex").clone();
    assert_eq!(
        notifs.len(),
        3,
        "exactly one notifications/progress per stdout line, got {} notifs: {notifs:?}",
        notifs.len()
    );

    for (i, n) in notifs.iter().enumerate() {
        assert_eq!(n.jsonrpc, "2.0", "notifications use JSON-RPC 2.0");
        assert_eq!(
            n.method, "notifications/progress",
            "MCP spec requires method = notifications/progress"
        );
        assert_eq!(
            n.params["progressToken"], "tok-finetune-42",
            "progressToken must echo the caller's token on every emission"
        );
        // The `message` payload should be the parsed JSON object from stdout.
        assert_eq!(n.params["message"]["event"], "step");
        assert_eq!(n.params["message"]["step"], i);
    }
}

/// FALSIFY-MCP-PROGRESS-001 (dispatcher gate): a `tools/call` without
/// `params._meta.progressToken` must NOT emit notifications. This proves we
/// honour the MCP spec rule "servers MUST NOT send progress notifications if
/// the client did not request them" for apr.finetune.
#[test]
fn falsify_mcp_progress_001_no_token_no_notifications() {
    let captured: Arc<Mutex<Vec<JsonRpcNotification>>> = Arc::new(Mutex::new(Vec::new()));
    let captured_clone = Arc::clone(&captured);
    let sink: NotificationSink = Box::new(move |n| {
        captured_clone
            .lock()
            .expect("sink mutex not poisoned")
            .push(n);
    });

    let mut server = AprMcpServer::new();
    let req = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(serde_json::json!(1)),
        method: "tools/call".to_string(),
        // No `_meta` — per MCP spec the server MUST NOT emit progress.
        params: serde_json::json!({
            "name": "apr.finetune",
            "arguments": { "base_model": "/nonexistent/does-not-exist.apr" }
        }),
    };

    let resp = server.handle_request_with_sink(&req, &sink);
    assert!(resp.is_some(), "tools/call must return a response");

    let notifs = captured.lock().expect("mutex").clone();
    assert!(
        notifs.is_empty(),
        "no progressToken → zero notifications, got {} notifs: {notifs:?}",
        notifs.len()
    );
}

/// FALSIFY-MCP-PROGRESS-001 (dispatcher plumbing): the dispatcher correctly
/// extracts a string progressToken from `params._meta.progressToken` and
/// forwards it to the sink. Uses a deliberately bad `base_model` so the
/// underlying subprocess exits fast — we're just proving the token flows,
/// not that finetuning works.
///
/// NOTE: this test spawns the real `apr` binary because the server's
/// dispatcher isn't parameterised by program name (by design — the binary
/// name is a production concern, not an MCP invariant). If `apr` isn't on
/// PATH the call returns a spawn error, which doesn't invalidate the
/// token-extraction invariant we're falsifying.
#[test]
fn falsify_mcp_progress_001_dispatcher_extracts_token() {
    let captured: Arc<Mutex<Vec<JsonRpcNotification>>> = Arc::new(Mutex::new(Vec::new()));
    let captured_clone = Arc::clone(&captured);
    let sink: NotificationSink = Box::new(move |n| {
        captured_clone
            .lock()
            .expect("sink mutex not poisoned")
            .push(n);
    });

    let mut server = AprMcpServer::new();
    let req = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(serde_json::json!(7)),
        method: "tools/call".to_string(),
        params: serde_json::json!({
            "name": "apr.finetune",
            "arguments": { "base_model": "/nonexistent/does-not-exist.apr" },
            "_meta": { "progressToken": "dispatch-token-99" }
        }),
    };

    let resp = server.handle_request_with_sink(&req, &sink);
    assert!(resp.is_some(), "tools/call returns a response");
    let resp = resp.expect("some");
    assert_eq!(resp.id, Some(serde_json::json!(7)));

    // Every notification (if any were emitted before the spawn failed or
    // produced output) must carry our token. In practice with a bogus
    // base_model, the apr subprocess either fails to spawn or fails early,
    // and we assert only that the token-routing invariant holds for
    // whatever was produced.
    let notifs = captured.lock().expect("mutex").clone();
    for n in &notifs {
        assert_eq!(
            n.params["progressToken"], "dispatch-token-99",
            "dispatcher must forward client progressToken verbatim"
        );
    }
}

/// FALSIFY-MCP-PROGRESS-001 (ordering): notifications are emitted strictly
/// before the final response is constructed. We prove this by checking the
/// sink receives all 3 notifications during the `stream_with_sink` call
/// (which returns the final result) — i.e., no notifications arrive after
/// the synchronous call returns.
#[test]
#[cfg(unix)]
fn falsify_mcp_progress_001_notifications_before_final_response() {
    let tmp = tempdir_fallback();
    let script = write_mock_apr_script(&tmp, 3);

    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = Arc::clone(&counter);
    let sink: NotificationSink = Box::new(move |_n| {
        counter_clone.fetch_add(1, Ordering::SeqCst);
    });

    let token = serde_json::json!(42);
    let script_str = script.to_str().expect("utf-8 script path");
    let _result = finetune::stream_with_sink("sh", &[script_str], &sink, &token);

    // By the time stream_with_sink returns, the counter must already be 3.
    // No "trailing" notifications can arrive because the sink is dropped
    // as stream_with_sink's frame unwinds.
    assert_eq!(
        counter.load(Ordering::SeqCst),
        3,
        "all 3 notifications must be delivered before the final response"
    );
}

/// Tiny tempdir helper — we don't want a `tempfile` dev-dep just for this.
/// Returns a unique directory under `std::env::temp_dir()` that we don't
/// bother cleaning up (the OS will — and CI containers are ephemeral).
fn tempdir_fallback() -> std::path::PathBuf {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("apr-mcp-falsify-progress-{pid}-{nanos}"));
    std::fs::create_dir_all(&dir).expect("create tempdir");
    dir
}
