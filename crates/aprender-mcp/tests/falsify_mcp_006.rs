//! FALSIFY-MCP-006 — `notifications/cancelled` during `apr.run` stops the
//! spawned subprocess within the grace window and returns a partial result.
//!
//! Spec: `docs/specifications/apr-mcp-server-spec.md` lines 95, 137.
//!
//! This file exercises the cancellation machinery at two layers:
//!
//! 1. The subprocess poll loop — proves that sending a cancel signal to a
//!    long-running child (`sleep 60`) triggers SIGTERM and the call returns
//!    far faster than the child's natural lifetime.
//! 2. The server's in-flight registry — proves that
//!    `AprMcpServer::cancel_in_flight` signals the right worker's cancel
//!    channel keyed by JSON-RPC id, is idempotent, and is a no-op for ids
//!    that never existed.

#![allow(clippy::disallowed_methods)] // serde_json::json! expands to code that hits unwrap()

use aprender_mcp::AprMcpServer;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

/// FALSIFY-MCP-006 (core): cancelling a long-running subprocess returns
/// within the configured grace window and flags the result as an error.
///
/// Uses `sleep 60` rather than a real `apr run` call because (a) we don't
/// need a real model in the MCP crate's test suite and (b) the
/// cancellation logic lives entirely in the spawn-and-signal poll loop —
/// it's binary-agnostic. The equivalent with the `apr` binary would be
/// tested end-to-end in the M4 integration suite.
#[test]
#[cfg(unix)]
fn falsify_mcp_006_cancel_stops_subprocess_within_grace() {
    let (tx, rx) = mpsc::channel::<()>();

    // Fire the cancel 100ms after spawn.
    let cancel_sender = thread::spawn(move || {
        thread::sleep(Duration::from_millis(100));
        let _ = tx.send(());
    });

    let t0 = Instant::now();
    // Grace of 500ms — SIGTERM to `sleep` is immediate, so we should see
    // the call return within ~150ms in practice.
    let result = aprender_mcp::tools::subprocess::spawn_cancellable("sleep", &["60"], &rx, 500);
    let elapsed = t0.elapsed();

    cancel_sender.join().expect("cancel-sender joins cleanly");

    assert_eq!(
        result.is_error,
        Some(true),
        "cancelled calls return isError: true"
    );
    assert!(
        result.content[0].text.starts_with("Cancelled:"),
        "message must lead with 'Cancelled:', got: {}",
        result.content[0].text
    );
    assert!(
        result.content[0].text.contains("partial stdout"),
        "message must acknowledge partial stdout, got: {}",
        result.content[0].text
    );
    // Spec budget: grace_ms + 200ms slack. sleep 60 would otherwise take
    // 60s to complete, so any elapsed << that is definitive.
    assert!(
        elapsed < Duration::from_millis(700),
        "cancel must finish within grace_ms + slack, took {elapsed:?}"
    );
}

/// FALSIFY-MCP-006 (registry): cancelling a live in-flight id signals the
/// registered receiver, removes the entry, and is idempotent on repeat.
#[test]
fn falsify_mcp_006_registry_routes_cancel_by_id() {
    let server = AprMcpServer::new();
    let handle = server.in_flight_handle();

    let id = serde_json::json!(42);
    let rx = AprMcpServer::register_in_flight(&handle, id.clone());

    // Live id → signalled, then removed.
    let first = AprMcpServer::cancel_in_flight(&handle, &id);
    assert!(first, "live id should be signalled");

    // Receiver observes the signal (or a disconnected sender — either
    // way, the signal reached it).
    let received = rx.recv_timeout(Duration::from_millis(100));
    assert!(
        received.is_ok(),
        "cancel signal must reach the receiver within 100ms, got: {received:?}"
    );

    // Idempotent: second cancel for the same id is a no-op.
    let second = AprMcpServer::cancel_in_flight(&handle, &id);
    assert!(!second, "cancelling an already-removed id is a no-op");
}

/// FALSIFY-MCP-006 (registry): cancelling an id that was never registered
/// is a safe no-op — MCP notifications can arrive after a call completes
/// and we must not panic or mis-route.
#[test]
fn falsify_mcp_006_registry_unknown_id_is_noop() {
    let server = AprMcpServer::new();
    let handle = server.in_flight_handle();

    let never_registered = serde_json::json!("phantom");
    let signalled = AprMcpServer::cancel_in_flight(&handle, &never_registered);
    assert!(
        !signalled,
        "cancel for unknown id must return false, not panic"
    );
}

/// FALSIFY-MCP-006 (registry): string-valued ids work just like numeric
/// ids. MCP permits both and we key the registry on `serde_json::Value`.
#[test]
fn falsify_mcp_006_registry_accepts_string_ids() {
    let server = AprMcpServer::new();
    let handle = server.in_flight_handle();

    let id = serde_json::json!("abc-123");
    let _rx = AprMcpServer::register_in_flight(&handle, id.clone());

    let signalled = AprMcpServer::cancel_in_flight(&handle, &id);
    assert!(signalled, "string id should cancel like numeric id");
}
