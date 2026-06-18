//! `AprMcpServer` — JSON-RPC 2.0 dispatcher for aprender MCP tools.
//!
//! # Cancellation model (FALSIFY-MCP-006)
//!
//! `tools/call` requests that target `apr.run` are dispatched on a worker
//! thread so the main stdio loop can continue reading and honour
//! `notifications/cancelled`. Each in-flight call registers a [`CancelHandle`]
//! in [`AprMcpServer::in_flight`], keyed by request id. A matching
//! `notifications/cancelled` signals the worker's cancel channel; the worker
//! then SIGTERMs the spawned `apr` subprocess, waits
//! [`crate::tools::subprocess::CANCEL_GRACE_MS`], and SIGKILLs if still alive.
//!
//! Non-cancellable tool calls still run on a worker (so future concurrent
//! calls don't block notifications/cancelled routing) but their cancel
//! channels are never signalled. `initialize`, `tools/list`, and other
//! fast synchronous methods dispatch inline on the main thread.

#![allow(clippy::disallowed_methods)] // serde_json::json! macro expands to .unwrap() internally

use crate::types::{
    JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, ToolCallResult, ToolDefinition,
};
use std::collections::HashMap;
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};

/// Callback used by tools to emit `notifications/progress` messages back to
/// the MCP client while a long-running `tools/call` is still in flight.
///
/// FALSIFY-MCP-PROGRESS-001: in stdio mode the dispatcher passes a sink that
/// writes each notification as one JSON line to the shared stdout handle
/// (guarded by the same mutex as final responses). In-process tests use an
/// `Arc<Mutex<Vec<_>>>`-backed sink to assert the outgoing wire format.
///
/// Must be `Send` because the sink is moved into the worker thread that
/// `run_stdio` spawns for every `tools/call`.
pub type NotificationSink = Box<dyn Fn(JsonRpcNotification) + Send + Sync>;

/// Per-request cancellation record held in [`AprMcpServer::in_flight`].
///
/// Only `apr.run` currently honours cancellation. Entries for other tools
/// are still registered (so a stray `notifications/cancelled` doesn't log
/// a warning) but their senders are never used.
#[derive(Debug)]
pub struct CancelHandle {
    /// Sender side of the worker's cancel mpsc. `send(())` causes the
    /// subprocess poll loop to SIGTERM its child.
    pub cancel_tx: Sender<()>,
}

/// Map of in-flight `tools/call` requests keyed by JSON-RPC id.
///
/// The id is stored as a raw `serde_json::Value` because the MCP spec
/// permits both integer and string ids.
type InFlight = Arc<Mutex<HashMap<serde_json::Value, CancelHandle>>>;

/// MCP server exposing the `apr` CLI as tools.
///
/// M1: `initialize`, `tools/list`, `tools/call` with `apr.version`.
/// M3: `notifications/cancelled` routed to in-flight `apr.run` workers.
#[derive(Debug, Default)]
pub struct AprMcpServer {
    in_flight: InFlight,
}

impl AprMcpServer {
    /// Construct a new server.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Dispatch a single JSON-RPC request synchronously.
    ///
    /// This is the in-process test entry point. It does NOT exercise the
    /// threading / cancellation machinery — `apr.run` runs inline with a
    /// dummy never-firing cancel receiver and NO notification sink is
    /// attached, so `apr.finetune` silently falls back to its synchronous
    /// path even if the request carries `params._meta.progressToken`. Use
    /// [`Self::run_stdio`] for the full M3 dispatcher or
    /// [`Self::handle_request_with_sink`] to drive FALSIFY-MCP-PROGRESS-001
    /// in tests.
    ///
    /// The dispatcher enforces two protocol-level invariants before routing:
    /// FALSIFY-MCP-005 (`jsonrpc` must be exactly `"2.0"` or the response is
    /// `-32600 Invalid Request`) and FALSIFY-MCP-007 (an `initialize` whose
    /// `params.protocolVersion` mismatches ours returns `-32602 Invalid Params`
    /// instead of advancing to tools/list).
    #[must_use]
    pub fn handle_request(&mut self, request: &JsonRpcRequest) -> JsonRpcResponse {
        if request.jsonrpc != "2.0" {
            return JsonRpcResponse::error(
                request.id.clone(),
                -32600,
                format!(
                    "Invalid Request: jsonrpc must be \"2.0\", got \"{}\"",
                    request.jsonrpc
                ),
            );
        }

        match request.method.as_str() {
            "initialize" => self.handle_initialize(request),
            "tools/list" => self.handle_tools_list(request),
            "tools/call" => self.handle_tools_call_sync(request),
            other => JsonRpcResponse::error(
                request.id.clone(),
                -32601,
                format!("Method not found: {other}"),
            ),
        }
    }

    fn handle_initialize(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        // FALSIFY-MCP-007: if the client advertises a protocolVersion, it must
        // match ours. Missing field is permitted (some clients omit it on the
        // very first handshake); only a *mismatch* is rejected.
        if let Some(client_version) = request
            .params
            .get("protocolVersion")
            .and_then(|v| v.as_str())
        {
            if client_version != crate::PROTOCOL_VERSION {
                return JsonRpcResponse::error(
                    request.id.clone(),
                    -32602,
                    format!(
                        "Unsupported protocolVersion: client requested \"{}\", server speaks \"{}\"",
                        client_version,
                        crate::PROTOCOL_VERSION
                    ),
                );
            }
        }

        JsonRpcResponse::success(
            request.id.clone(),
            serde_json::json!({
                "protocolVersion": crate::PROTOCOL_VERSION,
                "capabilities": {
                    "tools": { "listChanged": false }
                },
                "serverInfo": {
                    "name": crate::SERVER_NAME,
                    "version": env!("CARGO_PKG_VERSION"),
                },
            }),
        )
    }

    fn handle_tools_list(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let tools: Vec<ToolDefinition> = self.tool_definitions();
        JsonRpcResponse::success(request.id.clone(), serde_json::json!({ "tools": tools }))
    }

    /// Synchronous fallback used by [`Self::handle_request`]. `apr.run`
    /// runs with a never-firing cancel receiver — cancellation is only
    /// wired by the stdio loop in [`Self::run_stdio`]. No notifications are
    /// emitted from this path.
    fn handle_tools_call_sync(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let (_tx, rx) = mpsc::channel::<()>();
        let result = dispatch_tool_call(&request.params, &rx, None);
        JsonRpcResponse::success(
            request.id.clone(),
            serde_json::to_value(result).unwrap_or_else(|_| serde_json::json!({})),
        )
    }

    /// Dispatch one request with an explicit notification sink (test entry
    /// point for FALSIFY-MCP-PROGRESS-001).
    ///
    /// The sink is only exercised for `tools/call` dispatches where
    /// (a) the client supplied `params._meta.progressToken` on the original
    /// request AND (b) the target tool supports progress streaming
    /// (currently `apr.finetune` and `apr.run`). Other methods ignore the
    /// sink.
    ///
    /// `handle_request_with_sink` returns `None` for notifications (methods
    /// prefixed with `notifications/`) because notifications have no id and
    /// MUST NOT receive a response per JSON-RPC 2.0. All other methods
    /// return `Some(response)`.
    #[must_use]
    pub fn handle_request_with_sink(
        &mut self,
        request: &JsonRpcRequest,
        sink: &NotificationSink,
    ) -> Option<JsonRpcResponse> {
        if request.jsonrpc != "2.0" {
            return Some(JsonRpcResponse::error(
                request.id.clone(),
                -32600,
                format!(
                    "Invalid Request: jsonrpc must be \"2.0\", got \"{}\"",
                    request.jsonrpc
                ),
            ));
        }

        if request.method.starts_with("notifications/") {
            return None;
        }

        if request.method != "tools/call" {
            return Some(self.handle_request(request));
        }

        let progress_token = extract_progress_token(&request.params);
        let (_tx, rx) = mpsc::channel::<()>();
        let sink_for_dispatch = progress_token.as_ref().map(|_| sink);
        let result =
            dispatch_tool_call_with_sink(&request.params, &rx, sink_for_dispatch, progress_token);
        Some(JsonRpcResponse::success(
            request.id.clone(),
            serde_json::to_value(result).unwrap_or_else(|_| serde_json::json!({})),
        ))
    }

    /// All tool definitions registered on this server.
    ///
    /// HELIX-IDEA-002 / FALSIFY-INVENTORY-001: returns whatever
    /// [`crate::tools::ToolIndex::definitions`] contains, which is
    /// populated at startup by iterating
    /// `inventory::iter::<McpToolEntry>`. Adding a new tool requires only
    /// a `register_mcp_tool!` invocation in that tool's module — no
    /// edit here.
    #[must_use]
    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        tool_index().definitions().to_vec()
    }

    /// Register a new in-flight request and return its cancel receiver.
    ///
    /// Exposed for testing the cancellation routing without spawning a real
    /// worker. Production code calls this from [`Self::run_stdio`].
    #[must_use]
    pub fn register_in_flight(in_flight: &InFlight, id: serde_json::Value) -> mpsc::Receiver<()> {
        let (tx, rx) = mpsc::channel::<()>();
        let mut guard = in_flight
            .lock()
            .expect("in_flight mutex not poisoned during register");
        guard.insert(id, CancelHandle { cancel_tx: tx });
        rx
    }

    /// Route a `notifications/cancelled` to the matching in-flight request.
    ///
    /// Idempotent: repeated cancels for the same id after the first are
    /// silently dropped. References to completed / unknown ids are no-ops.
    /// Returns `true` iff a live handle was signalled.
    pub fn cancel_in_flight(in_flight: &InFlight, id: &serde_json::Value) -> bool {
        let mut guard = in_flight
            .lock()
            .expect("in_flight mutex not poisoned during cancel");
        if let Some(handle) = guard.remove(id) {
            // Best-effort: if the worker already completed and dropped its
            // receiver, the send fails silently — exactly the no-op we want.
            let _ = handle.cancel_tx.send(());
            true
        } else {
            false
        }
    }

    /// Deregister an in-flight id after its worker finishes. Safe to call
    /// even if the id was already removed by a concurrent cancel.
    fn deregister_in_flight(in_flight: &InFlight, id: &serde_json::Value) {
        if let Ok(mut guard) = in_flight.lock() {
            guard.remove(id);
        }
    }

    /// Run the server over stdio (blocking).
    ///
    /// Reads one JSON-RPC message per line from stdin. `initialize`,
    /// `tools/list`, and unknown methods dispatch inline. `tools/call`
    /// spawns a worker thread so a subsequent `notifications/cancelled`
    /// message can flow through the main loop and signal the worker's
    /// cancel channel. Workers write their responses directly to stdout
    /// (guarded by a mutex) so the main loop never has to wait on them.
    ///
    /// # Errors
    /// Returns an error if stdin/stdout I/O fails.
    #[cfg(feature = "native")]
    pub fn run_stdio(&mut self) -> anyhow::Result<()> {
        use std::io::{self, BufRead};

        let stdin = io::stdin();
        let stdout = Arc::new(Mutex::new(io::stdout()));

        for line in stdin.lock().lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let parsed: Result<JsonRpcRequest, _> = serde_json::from_str(&line);
            match parsed {
                Ok(req) => self.route_stdio_message(req, &stdout)?,
                Err(e) => {
                    let resp = JsonRpcResponse::error(None, -32700, format!("Parse error: {e}"));
                    write_response(&stdout, &resp)?;
                }
            }
        }

        Ok(())
    }

    /// Dispatch one parsed request within the stdio loop. Separated from
    /// [`Self::run_stdio`] for testability.
    #[cfg(feature = "native")]
    fn route_stdio_message(
        &mut self,
        req: JsonRpcRequest,
        stdout: &Arc<Mutex<std::io::Stdout>>,
    ) -> anyhow::Result<()> {
        // FALSIFY-MCP-005: jsonrpc field gate runs before method dispatch.
        if req.jsonrpc != "2.0" {
            let resp = JsonRpcResponse::error(
                req.id.clone(),
                -32600,
                format!(
                    "Invalid Request: jsonrpc must be \"2.0\", got \"{}\"",
                    req.jsonrpc
                ),
            );
            return write_response(stdout, &resp);
        }

        match req.method.as_str() {
            // Notifications have no `id` and MUST NOT receive a response.
            "notifications/cancelled" => {
                if let Some(request_id) = req.params.get("requestId").cloned() {
                    let _ = Self::cancel_in_flight(&self.in_flight, &request_id);
                }
                Ok(())
            }
            "notifications/initialized" => {
                // Client handshake ack — no response, no state change.
                Ok(())
            }
            "tools/call" => self.spawn_tools_call_worker(req, stdout),
            // Fast inline paths.
            _ => {
                // FALSIFY-MCP-009: JSON-RPC 2.0 §4.1 — a Request object
                // without an `id` member is a *Notification*, and "The Server
                // MUST NOT reply to a Notification." The `notifications/*`
                // method prefix is an MCP convention, but conformance is
                // determined by the *absence of an id*, not the method name. A
                // client that sends e.g. `{"jsonrpc":"2.0","method":"initialize"}`
                // (no id) or an unknown method with no id is issuing a
                // notification; emitting a response with `id:null` would
                // corrupt the stream for a strict peer. Drop it silently.
                if req.id.is_none() {
                    return Ok(());
                }
                let resp = self.handle_request(&req);
                write_response(stdout, &resp)
            }
        }
    }

    #[cfg(feature = "native")]
    fn spawn_tools_call_worker(
        &mut self,
        req: JsonRpcRequest,
        stdout: &Arc<Mutex<std::io::Stdout>>,
    ) -> anyhow::Result<()> {
        // Notifications would arrive with id = None; tools/call must have
        // an id per JSON-RPC. Defensive: if it's missing, respond inline
        // with an error so the client sees the failure immediately.
        let Some(id) = req.id.clone() else {
            let resp =
                JsonRpcResponse::error(None, -32600, "Invalid Request: tools/call requires an id");
            return write_response(stdout, &resp);
        };

        let cancel_rx = Self::register_in_flight(&self.in_flight, id.clone());
        let stdout_clone = Arc::clone(stdout);
        let in_flight_clone = Arc::clone(&self.in_flight);
        let params = req.params.clone();
        let id_for_worker = id.clone();
        let progress_token = extract_progress_token(&params);

        // Build a stdout-backed notification sink for this worker. The sink
        // shares the response stdout mutex so progress lines and the final
        // response can never interleave. Per MCP spec the sink is only
        // wired when the client advertised a progressToken.
        let sink_stdout = Arc::clone(stdout);
        let sink: NotificationSink = Box::new(move |notif| {
            // Best-effort: a broken stdout means the client disconnected.
            let _ = write_notification(&sink_stdout, &notif);
        });

        // Thread spawn is infallible here in practice, but propagate the
        // error rather than unwrapping so we stay in the "no panics" lane.
        let builder = std::thread::Builder::new().name(format!("apr-mcp-call-{id}"));
        let spawn_result = builder.spawn(move || {
            let sink_ref = progress_token.as_ref().map(|_| &sink);
            let result =
                dispatch_tool_call_with_sink(&params, &cancel_rx, sink_ref, progress_token);
            let resp = JsonRpcResponse::success(
                Some(id_for_worker.clone()),
                serde_json::to_value(result).unwrap_or_else(|_| serde_json::json!({})),
            );
            // Best-effort: a broken stdout means the client disconnected,
            // which we can't recover from anyway.
            let _ = write_response(&stdout_clone, &resp);
            Self::deregister_in_flight(&in_flight_clone, &id_for_worker);
        });

        match spawn_result {
            Ok(_handle) => Ok(()),
            Err(e) => {
                // Failed to spawn — clean up the registry entry we just
                // inserted and report the failure inline.
                Self::deregister_in_flight(&self.in_flight, &id);
                let resp = JsonRpcResponse::error(
                    Some(id),
                    -32603,
                    format!("Internal error: failed to spawn worker thread: {e}"),
                );
                write_response(stdout, &resp)
            }
        }
    }

    /// Handle for tests that want to inspect the in-flight registry.
    #[must_use]
    pub fn in_flight_handle(&self) -> InFlight {
        Arc::clone(&self.in_flight)
    }
}

/// Shared tool-call dispatch logic used by both the sync and stdio paths.
///
/// `cancel_rx` is forwarded to `apr.run` only; the other tools ignore it.
/// Callers that never need progress streaming can keep using this wrapper;
/// the [`dispatch_tool_call_with_sink`] variant exposes the
/// FALSIFY-MCP-PROGRESS-001 path.
fn dispatch_tool_call(
    params: &serde_json::Value,
    cancel_rx: &mpsc::Receiver<()>,
    sink: Option<&NotificationSink>,
) -> ToolCallResult {
    dispatch_tool_call_with_sink(params, cancel_rx, sink, None)
}

/// Full dispatch variant with optional `NotificationSink` + `progressToken`.
///
/// FALSIFY-MCP-PROGRESS-001 / FALSIFY-MCP-PROGRESS-002: when `sink` and
/// `progress_token` are both `Some`, tools that support streaming
/// (`apr.finetune` and `apr.run`) forward each stdout line as a
/// `notifications/progress` message via `sink` before returning the final
/// `ToolCallResult`. Tools that don't support streaming ignore the sink and
/// run synchronously.
fn dispatch_tool_call_with_sink(
    params: &serde_json::Value,
    cancel_rx: &mpsc::Receiver<()>,
    sink: Option<&NotificationSink>,
    progress_token: Option<serde_json::Value>,
) -> ToolCallResult {
    let name = params.get("name").and_then(|v| v.as_str());
    let arguments = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));

    // HELIX-IDEA-002 / FALSIFY-INVENTORY-003: dispatch goes through the
    // inventory-built name → fn-pointer index. Every shipped tool's
    // module owns a `dispatch` shim that adapts to the unified
    // `DispatchFn` signature (FALSIFY-MCP-PROGRESS-002 still applies for
    // `apr.run` and `apr.finetune`; sink + progress_token forward through
    // the shim as before).
    let Some(name) = name else {
        return ToolCallResult::error("Missing tool name");
    };
    match tool_index().dispatch_for(name) {
        Some(dispatch_fn) => dispatch_fn(&arguments, cancel_rx, sink, progress_token),
        None => ToolCallResult::error(format!("Unknown tool: {name}")),
    }
}

/// Module-local inventory cache. Built once on first access via
/// [`crate::tools::ToolIndex::from_inventory`]; that call panics
/// (FALSIFY-INVENTORY-002) if two tools advertise the same name, so a
/// duplicate-registration regression fails every test that hits the
/// dispatcher rather than silently shadowing one entry.
fn tool_index() -> &'static crate::tools::ToolIndex {
    static INDEX: std::sync::OnceLock<crate::tools::ToolIndex> = std::sync::OnceLock::new();
    INDEX.get_or_init(crate::tools::ToolIndex::from_inventory)
}

/// Pull `params._meta.progressToken` out of a `tools/call` request. Returns
/// `None` when the field is absent — per MCP 2024-11-05 the server MUST NOT
/// emit progress notifications in that case.
fn extract_progress_token(params: &serde_json::Value) -> Option<serde_json::Value> {
    params
        .get("_meta")
        .and_then(|m| m.get("progressToken"))
        .cloned()
}

#[cfg(feature = "native")]
fn write_response(
    stdout: &Arc<Mutex<std::io::Stdout>>,
    resp: &JsonRpcResponse,
) -> anyhow::Result<()> {
    use std::io::Write;

    let json = serde_json::to_string(resp)?;
    let mut guard = stdout
        .lock()
        .map_err(|e| anyhow::anyhow!("stdout mutex poisoned: {e}"))?;
    writeln!(&mut *guard, "{json}")?;
    guard.flush()?;
    Ok(())
}

/// FALSIFY-MCP-PROGRESS-001: write one `notifications/progress` line to
/// stdout under the same mutex used for final responses. Called from the
/// worker-local `NotificationSink` built in
/// [`AprMcpServer::spawn_tools_call_worker`].
#[cfg(feature = "native")]
fn write_notification(
    stdout: &Arc<Mutex<std::io::Stdout>>,
    notif: &JsonRpcNotification,
) -> anyhow::Result<()> {
    use std::io::Write;

    let json = notif.to_json_line()?;
    let mut guard = stdout
        .lock()
        .map_err(|e| anyhow::anyhow!("stdout mutex poisoned: {e}"))?;
    writeln!(&mut *guard, "{json}")?;
    guard.flush()?;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // serde_json::json! expands to code that hits unwrap()
mod tests {
    use super::*;

    fn make_request(method: &str, params: serde_json::Value) -> JsonRpcRequest {
        JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(serde_json::json!(1)),
            method: method.to_string(),
            params,
        }
    }

    /// FALSIFY-MCP-001: initialize returns protocolVersion "2024-11-05".
    #[test]
    fn initialize_returns_protocol_version() {
        let mut server = AprMcpServer::new();
        let req = make_request("initialize", serde_json::json!({}));
        let resp = server.handle_request(&req);

        assert!(resp.error.is_none());
        let result = resp.result.expect("result present");
        assert_eq!(result["protocolVersion"], "2024-11-05");
        assert_eq!(result["serverInfo"]["name"], "aprender-mcp");
        assert!(result["capabilities"]["tools"].is_object());
    }

    /// FALSIFY-MCP-002: tools/list returns every registered tool. The
    /// Phase-1 8-tool set (M2 subprocess wrappers + M3 `apr.finetune`) plus
    /// the `apr.version` M1 scaffold is what a conforming dispatcher now
    /// advertises; adding a new tool should fail this test until the contract
    /// YAML and codegen are updated in lockstep.
    #[test]
    fn tools_list_returns_registered_tools() {
        let mut server = AprMcpServer::new();
        let req = make_request("tools/list", serde_json::json!({}));
        let resp = server.handle_request(&req);

        let result = resp.result.expect("result present");
        let tools = result["tools"].as_array().expect("tools array");
        let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
        for expected in [
            "apr.version",
            "apr.validate",
            "apr.tensors",
            "apr.bench",
            "apr.qa",
            "apr.trace",
            "apr.run",
            "apr.serve",
            "apr.finetune",
        ] {
            assert!(names.contains(&expected), "{expected} registered");
        }

        for tool in tools {
            assert_eq!(tool["inputSchema"]["type"], "object");
        }
    }

    #[test]
    fn tools_call_version_returns_metadata() {
        let mut server = AprMcpServer::new();
        let req = make_request(
            "tools/call",
            serde_json::json!({ "name": "apr.version", "arguments": {} }),
        );
        let resp = server.handle_request(&req);

        let result = resp.result.expect("result present");
        let text = result["content"][0]["text"].as_str().expect("text");
        let parsed: serde_json::Value = serde_json::from_str(text).expect("json");
        assert_eq!(parsed["server"], "aprender-mcp");
        assert_eq!(parsed["protocol_version"], "2024-11-05");
    }

    #[test]
    fn unknown_method_returns_method_not_found() {
        let mut server = AprMcpServer::new();
        let req = make_request("tools/explode", serde_json::json!({}));
        let resp = server.handle_request(&req);

        assert!(resp.result.is_none());
        let err = resp.error.expect("error present");
        assert_eq!(err.code, -32601);
    }

    /// `apr.validate` without `model_path` must return `isError: true` via
    /// the argument-validation branch (no subprocess spawn).
    #[test]
    fn tools_call_validate_missing_model_path_is_error() {
        let mut server = AprMcpServer::new();
        let req = make_request(
            "tools/call",
            serde_json::json!({ "name": "apr.validate", "arguments": {} }),
        );
        let resp = server.handle_request(&req);

        let result = resp.result.expect("result present");
        assert_eq!(result["isError"], true);
        let text = result["content"][0]["text"].as_str().expect("text");
        assert!(text.contains("model_path"));
    }

    #[test]
    fn tools_call_unknown_tool_returns_is_error() {
        let mut server = AprMcpServer::new();
        let req = make_request(
            "tools/call",
            serde_json::json!({ "name": "apr.nonexistent" }),
        );
        let resp = server.handle_request(&req);

        let result = resp.result.expect("result present");
        assert_eq!(result["isError"], true);
    }

    #[test]
    fn tools_call_missing_name_returns_is_error() {
        let mut server = AprMcpServer::new();
        let req = make_request("tools/call", serde_json::json!({}));
        let resp = server.handle_request(&req);

        let result = resp.result.expect("result present");
        assert_eq!(result["isError"], true);
    }

    #[test]
    fn id_is_echoed_back() {
        let mut server = AprMcpServer::new();
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(serde_json::json!("req-42")),
            method: "initialize".to_string(),
            params: serde_json::json!({}),
        };
        let resp = server.handle_request(&req);
        assert_eq!(resp.id, Some(serde_json::json!("req-42")));
    }

    /// FALSIFY-MCP-006 (unit): registering an id and then cancelling it
    /// signals the receiver and removes the entry.
    #[test]
    fn cancel_in_flight_signals_and_deregisters() {
        let server = AprMcpServer::new();
        let id = serde_json::json!(99);
        let rx = AprMcpServer::register_in_flight(&server.in_flight, id.clone());

        let signalled = AprMcpServer::cancel_in_flight(&server.in_flight, &id);
        assert!(signalled, "live id should signal");
        // Sender was dropped by cancel_in_flight (removed from the map), so
        // try_recv must see either the signal or a disconnected channel —
        // both prove the cancel reached the receiver side.
        let received = rx.try_recv();
        assert!(received.is_ok(), "cancel signal must be deliverable");

        // Idempotent: second call is a no-op.
        let signalled_again = AprMcpServer::cancel_in_flight(&server.in_flight, &id);
        assert!(
            !signalled_again,
            "cancelling an already-removed id is a no-op"
        );
    }

    /// FALSIFY-MCP-006 (unit): cancelling an unknown id is a safe no-op.
    #[test]
    fn cancel_unknown_id_is_noop() {
        let server = AprMcpServer::new();
        let id = serde_json::json!("never-registered");
        let signalled = AprMcpServer::cancel_in_flight(&server.in_flight, &id);
        assert!(!signalled);
    }
}
