//! `apr.run` — M2 tool. Synchronous inference via subprocess wrapper.
//!
//! Wraps `apr run <model> --json [--prompt X] [--max-tokens N] [--temperature T] [--top-p P]`.
//!
//! M3 (FALSIFY-MCP-006) adds cancellation: the call accepts a cancel receiver
//! and forwards it to [`run_apr_cancellable`], which SIGTERMs the spawned
//! subprocess on signal and SIGKILLs after the grace window.
//!
//! M3 (FALSIFY-MCP-PROGRESS-002) adds streaming: when the originating
//! `tools/call` carries `params._meta.progressToken`, [`call_with_sink`]
//! invokes `apr run ... --stream` (NDJSON: one `event=token` line per
//! decoded token, then one `event=final` blob) and forwards each line as a
//! `notifications/progress` message tagged with the caller's token. When the
//! sink is absent (no progressToken), we fall back to the original
//! cancellable sync path so existing clients see no behaviour change.

#![allow(clippy::disallowed_methods)] // serde_json::json! macro expands to .unwrap() internally

use crate::server::NotificationSink;
use crate::tools::subprocess::{run_apr_cancellable, spawn_streaming, CANCEL_GRACE_MS};
use crate::types::{InputSchema, JsonRpcNotification, ToolCallResult, ToolDefinition};
use std::sync::mpsc::Receiver;

/// Tool name registered with MCP clients.
pub const NAME: &str = "apr.run";

/// Return the MCP tool definition for `apr.run`.
///
/// FALSIFY-MCP-008: the `inputSchema` is parsed from the build-time codegen
/// constant `crate::schemas::APR_RUN_SCHEMA`, which `build.rs` emits from
/// `contracts/apr-mcp-tool-schemas-v1.yaml`. The contract is the single
/// source of truth — the live `tools/list` response and the YAML must agree
/// byte-for-byte after JSON canonicalization (asserted by
/// `tests/falsify_mcp_008.rs`).
#[must_use]
pub fn run_tool_definition() -> ToolDefinition {
    let input_schema: InputSchema = serde_json::from_str(crate::schemas::APR_RUN_SCHEMA).expect(
        "FALSIFY-MCP-008: apr.run codegen constant must parse as InputSchema; \
             regenerate by editing contracts/apr-mcp-tool-schemas-v1.yaml and rebuilding",
    );
    ToolDefinition {
        name: NAME.to_string(),
        description: crate::schemas::APR_RUN_DESCRIPTION.to_string(),
        input_schema,
    }
}

/// Execute `apr.run` by spawning `apr run <model> --json [...flags]`.
///
/// `cancel_rx` is signalled by the MCP dispatcher when a matching
/// `notifications/cancelled` arrives on the same request id (FALSIFY-MCP-006).
/// Pass a never-firing channel for tests or direct non-MCP callers.
///
/// Back-compat entry point used by callers that don't opt into progress
/// streaming. Equivalent to `call_with_sink(args, cancel_rx, None, None)` but
/// preserves the cancellable code path for the no-stream case.
#[must_use]
pub fn call(args: &serde_json::Value, cancel_rx: &Receiver<()>) -> ToolCallResult {
    call_with_sink(args, cancel_rx, None, None)
}

/// Execute `apr.run` with optional `notifications/progress` streaming.
///
/// FALSIFY-MCP-PROGRESS-002: when both `sink` and `progress_token` are
/// `Some`, the subprocess is spawned with `apr run ... --stream` so each
/// decoded token (NDJSON `event=token` line) and the terminal `event=final`
/// blob is forwarded as a `notifications/progress` message tagged with the
/// caller's token. When either argument is `None` (no progressToken on the
/// originating `tools/call`) we fall back to the synchronous
/// [`run_apr_cancellable`] path so existing clients see identical behaviour.
///
/// Note: the streaming path does NOT honour `cancel_rx` today (the MCP
/// `apr.finetune` streaming path made the same trade-off in #887). Wiring
/// SIGTERM into [`spawn_streaming`] is tracked separately — clients that
/// require both streaming AND cancellation should not yet supply a
/// progressToken on `apr.run`. The non-streaming path remains fully
/// cancellable.
#[must_use]
pub fn call_with_sink(
    args: &serde_json::Value,
    cancel_rx: &Receiver<()>,
    sink: Option<&NotificationSink>,
    progress_token: Option<serde_json::Value>,
) -> ToolCallResult {
    let model_path = match crate::tools::args::require_str(args, "model_path") {
        Ok(p) => p,
        Err(e) => return e,
    };

    let streaming = sink.is_some() && progress_token.is_some();

    let mut owned: Vec<String> = vec!["run".to_string(), model_path.to_string()];
    // --stream emits NDJSON (one event per line); the legacy --json path
    // emits a single pretty-printed blob. Pick whichever matches the
    // intended consumer.
    if streaming {
        owned.push("--stream".to_string());
    } else {
        owned.push("--json".to_string());
    }

    if let Some(prompt) = args.get("prompt").and_then(|v| v.as_str()) {
        if !prompt.is_empty() {
            owned.push("--prompt".to_string());
            owned.push(prompt.to_string());
        }
    }
    if let Some(n) = args.get("max_tokens").and_then(serde_json::Value::as_u64) {
        owned.push("--max-tokens".to_string());
        owned.push(n.to_string());
    }
    if let Some(t) = args.get("temperature").and_then(serde_json::Value::as_f64) {
        owned.push("--temperature".to_string());
        owned.push(t.to_string());
    }
    if let Some(p) = args.get("top_p").and_then(serde_json::Value::as_f64) {
        owned.push("--top-p".to_string());
        owned.push(p.to_string());
    }

    let argv: Vec<&str> = owned.iter().map(String::as_str).collect();

    match (streaming, sink, progress_token) {
        (true, Some(sink), Some(token)) => {
            // #2384: the streaming path must target the same self-resolved
            // binary as the non-streaming path, not a `$PATH` lookup.
            stream_with_sink(crate::apr_bin::apr_binary(), &argv, sink, &token)
        }
        _ => run_apr_cancellable(&argv, cancel_rx, CANCEL_GRACE_MS),
    }
}

/// Test-visible: stream `program args...` and forward each stdout line as a
/// `notifications/progress` notification through `sink`, tagged with
/// `progress_token`. Each stdout line is JSON-parsed if possible (the
/// `apr run --stream` NDJSON contract guarantees JSON) so downstream MCP
/// clients receive structured `message.event = "token"` / `"final"` events;
/// non-JSON lines fall back to a bare string. The returned `ToolCallResult`
/// is the aggregated stdout (same shape as `run_apr_cancellable`'s success
/// body) so non-streaming consumers get the full payload too.
#[must_use]
pub fn stream_with_sink<P: AsRef<std::ffi::OsStr>>(
    program: P,
    args: &[&str],
    sink: &NotificationSink,
    progress_token: &serde_json::Value,
) -> ToolCallResult {
    spawn_streaming(program, args, |line| {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return;
        }
        let payload = serde_json::from_str::<serde_json::Value>(trimmed)
            .unwrap_or_else(|_| serde_json::Value::String(line.to_string()));
        let notif = JsonRpcNotification::progress(progress_token.clone(), payload);
        sink(notif);
    })
}

/// HELIX-IDEA-002 — unified-signature shim for the inventory dispatcher.
/// `apr.run` honours both `cancel_rx` and the optional notification sink.
pub fn dispatch(
    args: &serde_json::Value,
    cancel_rx: &Receiver<()>,
    sink: Option<&NotificationSink>,
    progress_token: Option<serde_json::Value>,
) -> ToolCallResult {
    call_with_sink(args, cancel_rx, sink, progress_token)
}

crate::register_mcp_tool!(
    name: NAME,
    definition: run_tool_definition,
    dispatch: dispatch,
);

#[cfg(test)]
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;

    #[test]
    fn definition_has_correct_name_and_required_field() {
        let def = run_tool_definition();
        assert_eq!(def.name, "apr.run");
        assert_eq!(def.input_schema.schema_type, "object");
        assert_eq!(def.input_schema.required, vec!["model_path".to_string()]);
        for field in ["model_path", "prompt", "max_tokens", "temperature", "top_p"] {
            assert!(
                def.input_schema.properties.contains_key(field),
                "property {field} present"
            );
        }
    }

    #[test]
    fn missing_model_path_returns_error() {
        let (_tx, rx) = std::sync::mpsc::channel::<()>();
        let result = call(&serde_json::json!({}), &rx);
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("model_path"));
    }
}
