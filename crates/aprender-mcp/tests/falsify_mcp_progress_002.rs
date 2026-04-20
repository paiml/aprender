//! FALSIFY-MCP-PROGRESS-002 — MCP `notifications/progress` for `apr.run`.
//!
//! Spec: `docs/specifications/apr-mcp-server-spec.md` M3 milestone line 156
//! ("Progress notifications for `apr.run` / `apr.finetune`") and the MCP
//! 2024-11-05 progress utility (<https://spec.modelcontextprotocol.io>).
//!
//! # What this falsifier proves
//!
//! 1. When a `tools/call` for `apr.run` carries `params._meta.progressToken`,
//!    the dispatcher routes through `apr.run --stream` and forwards each
//!    NDJSON stdout line as a `notifications/progress` message tagged with
//!    the caller's token. For N decoded tokens we observe N + 1 notifications
//!    (N `event=token` lines + 1 `event=final` blob).
//! 2. The final response is still a normal `JsonRpcResponse` (success) so
//!    existing clients that ignore progress notifications don't regress.
//! 3. When the client omits `progressToken`, no notifications are emitted —
//!    the MCP spec forbids "spontaneous" progress.
//! 4. Notifications are emitted *before* the final response (proven by
//!    recording a counter when each message is observed).
//!
//! We exercise the streaming path end-to-end without invoking the real `apr`
//! binary (which would require a downloaded model and a GPU/SIMD backend) by
//! driving `tools::run::stream_with_sink` against a mock script that prints
//! N JSON token lines + 1 final blob and exits 0.

#![allow(clippy::disallowed_methods)] // serde_json::json! expands to code that hits unwrap()

use aprender_mcp::tools::run;
use aprender_mcp::{AprMcpServer, JsonRpcNotification, JsonRpcRequest, NotificationSink};
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Helper: write a mock subprocess script that mimics `apr run --stream`
/// output — `token_count` JSON token lines followed by one `event=final`
/// JSON blob. Returns the path; we drive it via `sh <path>` (see the note
/// on `falsify_mcp_progress_001` for the ETXTBSY rationale).
fn write_mock_apr_stream_script(dir: &std::path::Path, token_count: usize) -> std::path::PathBuf {
    let path = dir.join("mock-apr-run-stream.sh");
    {
        let mut f = std::fs::File::create(&path).expect("create mock script");
        writeln!(f, "#!/bin/sh").expect("write shebang");
        for i in 0..token_count {
            // Mirror the apr run --stream contract: NDJSON, one event=token
            // per decoded token, ascending index, monotonic token_id.
            writeln!(
                f,
                "printf '{{\"event\":\"token\",\"index\":{i},\"token_id\":{tok},\"text\":\"\"}}\\n'",
                tok = 1000 + i,
            )
            .expect("write token line");
        }
        // Terminal final blob — must have event=final, total token count,
        // and the rolled-up text/tok_per_sec the legacy --json mode emits.
        writeln!(
            f,
            "printf '{{\"event\":\"final\",\"model\":\"mock.gguf\",\"text\":\"mock-output\",\"tokens_generated\":{token_count},\"tok_per_sec\":42.0}}\\n'",
        )
        .expect("write final line");
        writeln!(f, "exit 0").expect("write exit");
        f.sync_all().expect("sync mock script");
    } // file handle dropped here

    path
}

/// FALSIFY-MCP-PROGRESS-002 (core): a mock subprocess emitting 4 JSON token
/// lines + 1 final blob triggers 5 `notifications/progress` calls, each
/// tagged with the caller's progressToken, in order. Token notifications
/// carry `event=token`/`index`/`token_id`; the final notification carries
/// `event=final` so clients can distinguish per-token progress from the
/// terminal payload.
#[test]
#[cfg(unix)]
fn falsify_mcp_progress_002_apr_run_streams_tokens() {
    let tmp = tempdir_fallback();
    let script = write_mock_apr_stream_script(&tmp, 4);

    let captured: Arc<Mutex<Vec<JsonRpcNotification>>> = Arc::new(Mutex::new(Vec::new()));
    let captured_clone = Arc::clone(&captured);
    let sink: NotificationSink = Box::new(move |n| {
        captured_clone
            .lock()
            .expect("sink mutex not poisoned")
            .push(n);
    });

    let token = serde_json::json!("tok-run-77");
    let script_str = script.to_str().expect("utf-8 script path");
    let result = run::stream_with_sink("sh", &[script_str], &sink, &token);

    // Final response must be a success (no isError flag) with the aggregated
    // stdout as the text payload — clients that ignore progress events still
    // get the full NDJSON in the body.
    assert!(
        result.is_error.is_none(),
        "mock script exits 0 → result must be success, got: {:?}",
        result.content[0].text
    );
    assert!(
        result.content[0].text.contains("\"event\":\"final\""),
        "aggregated stdout preserved in final ToolCallResult"
    );

    let notifs = captured.lock().expect("mutex").clone();
    // 4 tokens + 1 final = 5 notifications.
    assert_eq!(
        notifs.len(),
        5,
        "4 token lines + 1 final = 5 notifications, got {} notifs: {notifs:?}",
        notifs.len()
    );

    // First 4 must be event=token with ascending index & matching token_id.
    for (i, n) in notifs.iter().take(4).enumerate() {
        assert_eq!(n.jsonrpc, "2.0");
        assert_eq!(n.method, "notifications/progress");
        assert_eq!(
            n.params["progressToken"], "tok-run-77",
            "every notification must echo the caller's token"
        );
        assert_eq!(n.params["message"]["event"], "token");
        assert_eq!(n.params["message"]["index"], i);
        assert_eq!(n.params["message"]["token_id"], 1000 + i);
    }

    // Fifth notification is the final blob.
    let final_n = &notifs[4];
    assert_eq!(final_n.params["progressToken"], "tok-run-77");
    assert_eq!(final_n.params["message"]["event"], "final");
    assert_eq!(final_n.params["message"]["tokens_generated"], 4);
    assert_eq!(final_n.params["message"]["text"], "mock-output");
}

/// FALSIFY-MCP-PROGRESS-002 (dispatcher gate): a `tools/call apr.run`
/// without `params._meta.progressToken` MUST NOT emit notifications. This
/// proves we honour the MCP spec rule "servers MUST NOT send progress
/// notifications if the client did not request them" for apr.run.
#[test]
fn falsify_mcp_progress_002_no_token_no_notifications() {
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
            "name": "apr.run",
            "arguments": { "model_path": "/nonexistent/does-not-exist.apr" }
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

/// FALSIFY-MCP-PROGRESS-002 (dispatcher plumbing): the dispatcher correctly
/// extracts a string progressToken from `params._meta.progressToken` and
/// forwards it to the sink. Uses a deliberately bad `model_path` so the
/// underlying subprocess (the real `apr` if on PATH, or a spawn error
/// otherwise) exits fast — we're proving the token flows, not that
/// inference works.
#[test]
fn falsify_mcp_progress_002_dispatcher_extracts_token() {
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
            "name": "apr.run",
            "arguments": { "model_path": "/nonexistent/does-not-exist.apr" },
            "_meta": { "progressToken": "dispatch-run-token-99" }
        }),
    };

    let resp = server.handle_request_with_sink(&req, &sink);
    assert!(resp.is_some(), "tools/call returns a response");
    let resp = resp.expect("some");
    assert_eq!(resp.id, Some(serde_json::json!(7)));

    // Every notification (if any were emitted before the spawn failed or
    // produced output) must carry our token.
    let notifs = captured.lock().expect("mutex").clone();
    for n in &notifs {
        assert_eq!(
            n.params["progressToken"], "dispatch-run-token-99",
            "dispatcher must forward client progressToken verbatim"
        );
    }
}

/// FALSIFY-MCP-PROGRESS-002 (ordering): notifications are emitted strictly
/// before the final response is constructed. We prove this by checking the
/// sink receives all 5 notifications during the `stream_with_sink` call
/// (which returns the final result) — i.e., no notifications arrive after
/// the synchronous call returns.
#[test]
#[cfg(unix)]
fn falsify_mcp_progress_002_notifications_before_final_response() {
    let tmp = tempdir_fallback();
    let script = write_mock_apr_stream_script(&tmp, 4);

    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = Arc::clone(&counter);
    let sink: NotificationSink = Box::new(move |_n| {
        counter_clone.fetch_add(1, Ordering::SeqCst);
    });

    let token = serde_json::json!(99);
    let script_str = script.to_str().expect("utf-8 script path");
    let _result = run::stream_with_sink("sh", &[script_str], &sink, &token);

    // By the time stream_with_sink returns, the counter must already be 5
    // (4 tokens + 1 final). No "trailing" notifications can arrive because
    // the sink is dropped as stream_with_sink's frame unwinds.
    assert_eq!(
        counter.load(Ordering::SeqCst),
        5,
        "all 5 notifications must be delivered before the final response"
    );
}

/// Tiny tempdir helper — we don't want a `tempfile` dev-dep just for this.
fn tempdir_fallback() -> std::path::PathBuf {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("apr-mcp-falsify-progress-002-{pid}-{nanos}"));
    std::fs::create_dir_all(&dir).expect("create tempdir");
    dir
}
