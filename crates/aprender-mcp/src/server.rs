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
    /// Join handles for `tools/call` workers spawned by
    /// [`Self::spawn_tools_call_worker`]. The read loop MUST join these
    /// before returning on EOF, otherwise the process exits while a worker
    /// still owes the client a response and the answer is lost — see
    /// [`Self::serve_stream`].
    #[cfg(feature = "native")]
    workers: Vec<std::thread::JoinHandle<()>>,
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
    /// The dispatcher enforces one protocol-level invariant before routing:
    /// FALSIFY-MCP-005 (`jsonrpc` must be exactly `"2.0"` or the response is
    /// `-32600 Invalid Request`). Version negotiation is NOT a gate — see
    /// [`Self::handle_initialize`] (FALSIFY-MCP-007).
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
            // MCP base protocol utility: `ping` is not a capability and is
            // never advertised, so a client may send it at any time to check
            // liveness. The receiver "MUST respond promptly with an empty
            // response". Answering -32601 makes keepalive clients conclude
            // the server is dead and restart it.
            "ping" => JsonRpcResponse::success(request.id.clone(), serde_json::json!({})),
            other => JsonRpcResponse::error(
                request.id.clone(),
                -32601,
                format!("Method not found: {other}"),
            ),
        }
    }

    /// Handle `initialize`.
    ///
    /// FALSIFY-MCP-007: version negotiation is a *proposal*, not a gate. The
    /// MCP lifecycle says that if the server supports the requested version it
    /// responds with that version, and OTHERWISE responds with a version it
    /// does support, leaving the client to decide whether to proceed or
    /// disconnect. Returning `-32602` on a mismatch aborts the handshake, so a
    /// client negotiating anything newer than ours (Claude Code and Cursor
    /// both propose 2025-03-26 / 2025-06-18) can never connect at all — even
    /// though the wire protocol it would then speak is one we handle.
    ///
    /// We support exactly one version, so the reply always carries
    /// [`crate::PROTOCOL_VERSION`] regardless of what was proposed.
    fn handle_initialize(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
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
    /// Thin wrapper: binds [`Self::serve_stream`] to the real stdin/stdout.
    /// All loop behaviour — and every falsifier for it — lives in
    /// `serve_stream`, which is generic over its streams precisely so the
    /// read loop can be exercised in-process. `run_stdio` itself is the one
    /// piece that cannot be unit-tested, so it is kept to two lines with no
    /// logic of its own.
    ///
    /// # Errors
    /// Returns an error if stdin/stdout I/O fails.
    #[cfg(feature = "native")]
    pub fn run_stdio(&mut self) -> anyhow::Result<()> {
        let stdin = std::io::stdin();
        let reader = stdin.lock();
        self.serve_stream(reader, Arc::new(Mutex::new(std::io::stdout())))
    }

    /// Serve one JSON-RPC-over-newline-delimited-JSON session to completion.
    ///
    /// Reads one message per line from `reader`. `initialize`, `tools/list`,
    /// `ping`, and unknown methods dispatch inline. `tools/call` spawns a
    /// worker thread so a subsequent `notifications/cancelled` message can
    /// flow through the main loop and signal the worker's cancel channel.
    /// Workers write their responses directly to `out` (guarded by a mutex)
    /// so the main loop never has to wait on them mid-stream.
    ///
    /// Two transport invariants live here, both found by the 0.63.0 dogfood
    /// and both invisible to a dispatcher-level test:
    ///
    /// * **FALSIFY-MCP-010** — on EOF the loop MUST join every worker it
    ///   spawned before returning. Without that join the process exits the
    ///   instant stdin closes, so the canonical `printf ... | apr mcp`
    ///   invocation loses every `tools/call` result while still exiting 0 —
    ///   indistinguishable, to the client, from a tool that produced no
    ///   output.
    /// * **FALSIFY-MCP-011** — lines are read as BYTES and decoded per line.
    ///   A line that is not valid UTF-8 is a malformed *message*, answered
    ///   with `-32700`, not a transport failure that takes the session down
    ///   with it.
    ///
    /// `W` must be `Send + 'static` because the same handle is shared with
    /// every worker thread.
    ///
    /// # Errors
    /// Returns an error if reading or writing the streams fails.
    #[cfg(feature = "native")]
    pub fn serve_stream<R, W>(&mut self, mut reader: R, out: Arc<Mutex<W>>) -> anyhow::Result<()>
    where
        R: std::io::BufRead,
        W: std::io::Write + Send + 'static,
    {
        let mut buf: Vec<u8> = Vec::new();

        loop {
            buf.clear();
            if reader.read_until(b'\n', &mut buf)? == 0 {
                break; // EOF
            }
            while matches!(buf.last(), Some(b'\n' | b'\r')) {
                buf.pop();
            }

            // FALSIFY-MCP-011: one bad byte must cost one message, not the
            // session. `BufRead::lines()` surfaced this as an io::Error that
            // propagated out of the loop and killed the process (exit 1), so
            // every request after the bad byte went unanswered.
            let Ok(line) = std::str::from_utf8(&buf) else {
                let resp =
                    JsonRpcResponse::error(None, -32700, "Parse error: message is not valid UTF-8");
                write_response(&out, &resp)?;
                continue;
            };

            if line.trim().is_empty() {
                continue;
            }

            match parse_incoming(line) {
                Ok(req) => self.route_stdio_message(req, &out)?,
                Err(resp) => write_response(&out, &resp)?,
            }

            self.reap_finished_workers();
        }

        // FALSIFY-MCP-010: drain before returning. EOF means the client sent
        // everything it intends to, NOT that it stopped wanting answers.
        self.join_workers();
        Ok(())
    }

    /// Drop join handles for workers that have already finished, so a
    /// long-lived session does not accumulate one handle per tool call.
    /// Never blocks — `is_finished` is a non-blocking check.
    #[cfg(feature = "native")]
    fn reap_finished_workers(&mut self) {
        self.workers.retain(|h| !h.is_finished());
    }

    /// Block until every in-flight `tools/call` worker has written its
    /// response. Called once, on EOF, by [`Self::serve_stream`].
    ///
    /// A panicking worker is ignored rather than propagated: the client is
    /// owed whatever the surviving workers produced, and the panic message
    /// has already reached stderr.
    #[cfg(feature = "native")]
    fn join_workers(&mut self) {
        for handle in std::mem::take(&mut self.workers) {
            let _ = handle.join();
        }
    }

    /// Dispatch one parsed request within the read loop. Separated from
    /// [`Self::serve_stream`] for testability.
    #[cfg(feature = "native")]
    fn route_stdio_message<W>(
        &mut self,
        req: JsonRpcRequest,
        stdout: &Arc<Mutex<W>>,
    ) -> anyhow::Result<()>
    where
        W: std::io::Write + Send + 'static,
    {
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
    fn spawn_tools_call_worker<W>(
        &mut self,
        req: JsonRpcRequest,
        stdout: &Arc<Mutex<W>>,
    ) -> anyhow::Result<()>
    where
        W: std::io::Write + Send + 'static,
    {
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
            Ok(handle) => {
                // FALSIFY-MCP-010: keep the handle so EOF can wait for this
                // worker's response instead of exiting out from under it.
                self.workers.push(handle);
                Ok(())
            }
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
/// JSON type name, in JSON Schema vocabulary, for diagnostics.
fn json_type_name(value: &serde_json::Value) -> &'static str {
    crate::tools::args::json_type_name(value)
}

/// Parse one incoming line into a [`JsonRpcRequest`], or into the JSON-RPC
/// error response that must be sent instead.
///
/// FALSIFY-MCP-012: JSON-RPC 2.0 draws a line the old
/// `serde_json::from_str::<JsonRpcRequest>` path could not see. `-32700 Parse
/// error` means "the payload was not valid JSON". A payload that IS valid JSON
/// but is not a valid Request object is `-32600 Invalid Request`, and its
/// response must echo the request's `id` so the client can correlate the
/// failure. Deserializing straight into the struct reported a missing
/// `jsonrpc` or `method` field as a *parse* error with `id: null` — while a
/// jsonrpc field with the WRONG VALUE was already correctly reported as
/// -32600 with the id echoed, so the server disagreed with itself.
///
/// A batch ARRAY gets its own message. Batching is optional for a 2024-11-05
/// server and we decline it, but the old behaviour surfaced serde's attempt to
/// read the first array element as the `jsonrpc` string — "invalid type: map,
/// expected a string at line 1 column 1" — which names neither batching nor
/// arrays and points at a '[' that is perfectly valid JSON.
/// The error side is boxed because `JsonRpcResponse` is large enough that
/// clippy's `result_large_err` fires on the bare form, and the error path is
/// the rare one.
fn parse_incoming(line: &str) -> Result<JsonRpcRequest, Box<JsonRpcResponse>> {
    let value: serde_json::Value = serde_json::from_str(line).map_err(|e| {
        Box::new(JsonRpcResponse::error(
            None,
            -32700,
            format!("Parse error: {e}"),
        ))
    })?;

    if value.is_array() {
        return Err(Box::new(JsonRpcResponse::error(
            None,
            -32600,
            "Invalid Request: JSON-RPC batch arrays are not supported; \
             send one request per line",
        )));
    }

    let Some(obj) = value.as_object() else {
        return Err(Box::new(JsonRpcResponse::error(
            None,
            -32600,
            format!(
                "Invalid Request: a request must be a JSON object, got {}",
                json_type_name(&value)
            ),
        )));
    };

    // A null id is the same as an absent one (serde maps JSON null to None for
    // `Option<Value>`), which is what keeps FALSIFY-MCP-009's notification
    // rule intact.
    let id = obj.get("id").filter(|v| !v.is_null()).cloned();

    let jsonrpc = match obj.get("jsonrpc") {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(other) => {
            return Err(Box::new(JsonRpcResponse::error(
                id,
                -32600,
                format!(
                    "Invalid Request: \"jsonrpc\" must be the string \"2.0\", got {}",
                    json_type_name(other)
                ),
            )));
        }
        None => {
            return Err(Box::new(JsonRpcResponse::error(
                id,
                -32600,
                "Invalid Request: missing required field \"jsonrpc\"",
            )));
        }
    };

    let method = match obj.get("method") {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(other) => {
            return Err(Box::new(JsonRpcResponse::error(
                id,
                -32600,
                format!(
                    "Invalid Request: \"method\" must be a string, got {}",
                    json_type_name(other)
                ),
            )));
        }
        None => {
            return Err(Box::new(JsonRpcResponse::error(
                id,
                -32600,
                "Invalid Request: missing required field \"method\"",
            )));
        }
    };

    Ok(JsonRpcRequest {
        jsonrpc,
        id,
        method,
        params: obj
            .get("params")
            .cloned()
            .unwrap_or(serde_json::Value::Null),
    })
}

fn write_response<W: std::io::Write>(
    stdout: &Arc<Mutex<W>>,
    resp: &JsonRpcResponse,
) -> anyhow::Result<()> {
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
fn write_notification<W: std::io::Write>(
    stdout: &Arc<Mutex<W>>,
    notif: &JsonRpcNotification,
) -> anyhow::Result<()> {
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

    /// Drive a whole session through [`AprMcpServer::serve_stream`] with
    /// in-memory streams and return the response lines it wrote.
    ///
    /// This is the point of `serve_stream` being generic: FALSIFY-MCP-010 and
    /// -011 live in the read loop, not in request handling, so a
    /// `handle_request` test cannot see either. Driving the real loop over a
    /// byte slice reproduces both defects exactly — including invalid UTF-8,
    /// which cannot even be expressed as a `&str` input.
    #[cfg(feature = "native")]
    fn drive(input: &[u8]) -> Vec<serde_json::Value> {
        let out = Arc::new(Mutex::new(Vec::<u8>::new()));
        let mut server = AprMcpServer::new();
        server
            .serve_stream(std::io::Cursor::new(input.to_vec()), Arc::clone(&out))
            .expect("serve_stream must not propagate an error out of the session");

        let guard = out.lock().expect("output mutex not poisoned");
        String::from_utf8_lossy(&guard)
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                serde_json::from_str::<serde_json::Value>(l)
                    .unwrap_or_else(|e| panic!("non-JSON output line {l:?}: {e}"))
            })
            .collect()
    }

    #[cfg(feature = "native")]
    fn find_id(responses: &[serde_json::Value], id: i64) -> Option<&serde_json::Value> {
        responses.iter().find(|r| r["id"] == serde_json::json!(id))
    }

    #[cfg(feature = "native")]
    const INIT_LINE: &str = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05"}}"#;
    #[cfg(feature = "native")]
    const CALL_LINE: &str = r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"apr.version","arguments":{}}}"#;

    /// FALSIFY-MCP-010: every request carrying an `id` must be answered before
    /// the loop returns, INCLUDING a `tools/call` still in flight when the
    /// input reaches EOF.
    ///
    /// The shipped 0.63.0 loop returned the instant stdin closed, without
    /// joining the worker that owed the client its answer, so
    /// `printf '<initialize>\n<tools/call>\n' | apr mcp` answered initialize,
    /// exited 0, and silently dropped the tool result — indistinguishable
    /// from a tool that produced no output.
    #[cfg(feature = "native")]
    #[test]
    fn serve_stream_answers_tools_call_before_returning_on_eof() {
        let responses = drive(format!("{INIT_LINE}\n{CALL_LINE}\n").as_bytes());

        let call = find_id(&responses, 2).unwrap_or_else(|| {
            panic!(
                "tools/call response (id=2) was DROPPED at EOF; got {} response(s): {responses:?}",
                responses.len()
            )
        });
        assert!(
            call.get("error").is_none(),
            "tools/call must succeed, got {call:?}"
        );
        let text = call["result"]["content"][0]["text"]
            .as_str()
            .unwrap_or_else(|| panic!("missing content text in {call:?}"));
        let payload: serde_json::Value =
            serde_json::from_str(text).expect("apr.version payload is JSON");
        assert_eq!(
            payload["server"], "aprender-mcp",
            "must be the real apr.version result, not an empty envelope"
        );
        assert!(
            find_id(&responses, 1).is_some(),
            "initialize still answered"
        );
    }

    /// FALSIFY-MCP-010 (concurrency): several pipelined `tools/call` requests
    /// must ALL be answered, not just the ones that happened to finish before
    /// EOF.
    #[cfg(feature = "native")]
    #[test]
    fn serve_stream_answers_every_pipelined_tools_call() {
        let mut input = format!("{INIT_LINE}\n");
        for id in 2..=6 {
            input.push_str(&format!(
                r#"{{"jsonrpc":"2.0","id":{id},"method":"tools/call","params":{{"name":"apr.version","arguments":{{}}}}}}"#
            ));
            input.push('\n');
        }

        let responses = drive(input.as_bytes());
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
    /// serving, matching how the loop already treats malformed JSON.
    ///
    /// The shipped loop propagated an `io::Error` out of `BufRead::lines()`
    /// and killed the process (exit 1), losing every later request. Here the
    /// same failure would surface as `serve_stream` returning `Err`, which
    /// `drive` turns into a panic.
    #[cfg(feature = "native")]
    #[test]
    fn serve_stream_survives_invalid_utf8_line() {
        let mut input: Vec<u8> = Vec::new();
        input.extend_from_slice(br#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#);
        input.push(b'\n');
        input.push(0xFF); // never valid UTF-8
        input.push(b'\n');
        input.extend_from_slice(br#"{"jsonrpc":"2.0","id":2,"method":"ping"}"#);
        input.push(b'\n');

        let responses = drive(&input);

        assert!(
            find_id(&responses, 1).is_some(),
            "request before the bad byte must be answered: {responses:?}"
        );
        let after = find_id(&responses, 2)
            .unwrap_or_else(|| panic!("request AFTER the bad byte was lost: {responses:?}"));
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

    /// FALSIFY-MCP-011 (leading byte): a bad byte arriving before any valid
    /// request must not stop the session from ever starting.
    #[cfg(feature = "native")]
    #[test]
    fn serve_stream_survives_leading_invalid_utf8() {
        let mut input: Vec<u8> = vec![0x80, b'\n'];
        input.extend_from_slice(br#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#);
        input.push(b'\n');

        let responses = drive(&input);
        assert!(
            find_id(&responses, 1).is_some(),
            "the request after a leading bad byte must be answered: {responses:?}"
        );
    }

    /// The whole request-handling surface, over the real loop: negotiation,
    /// ping, Invalid-Request classification, batch diagnostics, and the
    /// -32700 case that must NOT regress.
    #[cfg(feature = "native")]
    #[test]
    fn serve_stream_protocol_surface_matches_jsonrpc_and_mcp() {
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

        let responses = drive(input.as_bytes());

        let init = find_id(&responses, 1).expect("initialize answered");
        assert!(
            init.get("error").is_none(),
            "a newer protocolVersion must not abort the handshake: {init:?}"
        );
        assert_eq!(init["result"]["protocolVersion"], crate::PROTOCOL_VERSION);

        let pong = find_id(&responses, 2).expect("ping answered");
        assert_eq!(pong["result"], serde_json::json!({}), "ping must pong");

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

        let batch = responses
            .iter()
            .find(|r| {
                r["error"]["message"]
                    .as_str()
                    .is_some_and(|m| m.contains("batch"))
            })
            .unwrap_or_else(|| panic!("batch array must be diagnosed as such: {responses:?}"));
        assert_eq!(batch["error"]["code"], serde_json::json!(-32600));

        assert!(
            responses
                .iter()
                .any(|r| r["error"]["code"] == serde_json::json!(-32700)),
            "`{{not json` must remain a Parse error: {responses:?}"
        );
        assert!(
            find_id(&responses, 7).is_some(),
            "the loop must keep serving after every malformed line: {responses:?}"
        );
    }

    /// FALSIFY-MCP-009 over the real loop: a request with no id is a
    /// notification and MUST NOT be answered.
    #[cfg(feature = "native")]
    #[test]
    fn serve_stream_never_answers_a_notification() {
        let responses = drive(
            concat!(
                r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
                "\n",
                r#"{"jsonrpc":"2.0","method":"tools/list"}"#,
                "\n",
                r#"{"jsonrpc":"2.0","id":9,"method":"ping"}"#,
                "\n",
            )
            .as_bytes(),
        );

        assert_eq!(
            responses.len(),
            1,
            "only the id-bearing request may be answered: {responses:?}"
        );
        assert!(find_id(&responses, 9).is_some());
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

    /// FALSIFY-MCP-007: a client proposing a version we do not speak must get
    /// the version we DO speak, not a handshake-aborting error. Claude Code
    /// and Cursor propose 2025-03-26 / 2025-06-18; under the old -32602 gate
    /// neither could ever connect.
    #[test]
    fn initialize_negotiates_down_instead_of_erroring() {
        for proposed in ["2025-06-18", "2025-03-26", "2024-10-07", "latest", ""] {
            let mut server = AprMcpServer::new();
            let req = make_request(
                "initialize",
                serde_json::json!({ "protocolVersion": proposed }),
            );
            let resp = server.handle_request(&req);

            assert!(
                resp.error.is_none(),
                "proposing {proposed:?} must not abort the handshake, got {:?}",
                resp.error
            );
            let result = resp.result.expect("result present");
            assert_eq!(
                result["protocolVersion"],
                crate::PROTOCOL_VERSION,
                "server must answer with the version it actually speaks"
            );
        }
    }

    /// A non-string `protocolVersion` must not be treated as a proposal we
    /// somehow honoured — the reply still carries our version.
    #[test]
    fn initialize_ignores_non_string_protocol_version() {
        let mut server = AprMcpServer::new();
        let req = make_request("initialize", serde_json::json!({ "protocolVersion": 2025 }));
        let resp = server.handle_request(&req);
        assert!(resp.error.is_none());
        let result = resp.result.expect("result present");
        assert_eq!(result["protocolVersion"], crate::PROTOCOL_VERSION);
    }

    /// `ping` is MCP base protocol: an empty result, not -32601. A keepalive
    /// client reads an error (or silence) as a dead server and restarts it.
    #[test]
    fn ping_returns_empty_result() {
        let mut server = AprMcpServer::new();
        let req = make_request("ping", serde_json::json!({}));
        let resp = server.handle_request(&req);

        assert!(
            resp.error.is_none(),
            "ping must not error: {:?}",
            resp.error
        );
        assert_eq!(resp.result, Some(serde_json::json!({})));
        assert_eq!(resp.id, Some(serde_json::json!(1)), "id echoed");
    }

    /// FALSIFY-MCP-012: valid JSON that is not a valid Request object is
    /// -32600 Invalid Request with the id echoed — NOT -32700 with id null.
    /// The server already got this right for a WRONG jsonrpc value, so the
    /// missing-field path disagreed with its own neighbour.
    #[test]
    fn missing_required_field_is_invalid_request_with_id_echoed() {
        for line in [
            r#"{"id":1,"method":"tools/list"}"#,
            r#"{"jsonrpc":"2.0","id":1}"#,
            r#"{"jsonrpc":2.0,"id":1,"method":"tools/list"}"#,
            r#"{"jsonrpc":"2.0","id":1,"method":42}"#,
        ] {
            let resp = parse_incoming(line).expect_err("must be rejected");
            let err = resp.error.as_ref().expect("error present");
            assert_eq!(
                err.code, -32600,
                "{line} must be Invalid Request, got {err:?}"
            );
            assert_eq!(
                resp.id,
                Some(serde_json::json!(1)),
                "{line} must echo the client's id so it can correlate"
            );
        }
    }

    /// Genuinely malformed JSON must STAY -32700 — the fix above must not
    /// swallow the case the code already handled correctly.
    #[test]
    fn malformed_json_is_still_parse_error() {
        for line in [
            r#"{not json"#,
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/li"#,
        ] {
            let resp = parse_incoming(line).expect_err("must be rejected");
            let err = resp.error.as_ref().expect("error present");
            assert_eq!(err.code, -32700, "{line} must remain a Parse error");
        }
    }

    /// A batch array must be diagnosed as a batch array. The old message was
    /// serde's "invalid type: map, expected a string at line 1 column 1",
    /// which names neither batching nor arrays.
    #[test]
    fn batch_array_is_diagnosed_as_unsupported_batching() {
        let line = r#"[{"jsonrpc":"2.0","id":1,"method":"tools/list"},{"jsonrpc":"2.0","id":2,"method":"tools/list"}]"#;
        let resp = parse_incoming(line).expect_err("batch must be rejected");
        let err = resp.error.as_ref().expect("error present");
        assert_eq!(err.code, -32600);
        assert!(
            err.message.contains("batch"),
            "message must name batching, got: {}",
            err.message
        );
        assert!(
            !err.message.contains("expected a string"),
            "must not leak serde's field-level error, got: {}",
            err.message
        );
    }

    #[test]
    fn non_object_request_is_invalid_request() {
        for line in ["42", r#""hello""#, "null", "true"] {
            let resp = parse_incoming(line).expect_err("must be rejected");
            let err = resp.error.as_ref().expect("error present");
            assert_eq!(err.code, -32600, "{line} must be Invalid Request");
        }
    }

    /// Happy path: a well-formed request survives the new shape validation
    /// with every field intact, including an absent `params`.
    #[test]
    fn well_formed_request_parses_unchanged() {
        let req = parse_incoming(r#"{"jsonrpc":"2.0","id":"abc","method":"tools/list"}"#)
            .expect("well-formed request must parse");
        assert_eq!(req.jsonrpc, "2.0");
        assert_eq!(req.method, "tools/list");
        assert_eq!(req.id, Some(serde_json::json!("abc")));
        assert_eq!(req.params, serde_json::Value::Null);

        let with_params = parse_incoming(
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"apr.version"}}"#,
        )
        .expect("params must round-trip");
        assert_eq!(with_params.params["name"], "apr.version");
    }

    /// FALSIFY-MCP-009 must survive the rewrite: a null or absent id still
    /// means "notification", which `route_stdio_message` relies on to stay
    /// silent.
    #[test]
    fn null_and_absent_id_both_parse_as_notification() {
        let absent = parse_incoming(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#)
            .expect("parse");
        assert!(absent.id.is_none());
        let null_id =
            parse_incoming(r#"{"jsonrpc":"2.0","id":null,"method":"tools/list"}"#).expect("parse");
        assert!(null_id.id.is_none(), "a null id is not an id");
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
