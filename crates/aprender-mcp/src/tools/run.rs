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
use crate::tools::args::{self, try_arg};
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

/// Build the `apr run ...` argv from `tools/call` arguments.
///
/// `streaming` selects `--stream` (NDJSON) over `--json` (one blob).
///
/// # Errors
/// Returns the client-facing message when an argument is present but not
/// usable at its declared type.
pub fn build_argv(args: &serde_json::Value, streaming: bool) -> Result<Vec<String>, String> {
    let model_path = args::required_str(args, "model_path")?;

    let mut owned: Vec<String> = vec!["run".to_string(), model_path.to_string()];
    // --stream emits NDJSON (one event per line); the legacy --json path
    // emits a single pretty-printed blob. Pick whichever matches the
    // intended consumer.
    if streaming {
        owned.push("--stream".to_string());
    } else {
        owned.push("--json".to_string());
    }

    if let Some(prompt) = args::opt_str(args, "prompt")? {
        if !prompt.is_empty() {
            owned.push("--prompt".to_string());
            owned.push(prompt.to_string());
        }
    }
    if let Some(n) = args::opt_u64(args, "max_tokens")? {
        owned.push("--max-tokens".to_string());
        owned.push(n.to_string());
    }
    if let Some(t) = args::opt_f64(args, "temperature")? {
        owned.push("--temperature".to_string());
        owned.push(t.to_string());
    }
    if let Some(p) = args::opt_f64(args, "top_p")? {
        owned.push("--top-p".to_string());
        owned.push(p.to_string());
    }
    Ok(owned)
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
    let streaming = sink.is_some() && progress_token.is_some();
    let owned = try_arg!(build_argv(args, streaming));
    let argv: Vec<&str> = owned.iter().map(String::as_str).collect();

    match (streaming, sink, progress_token) {
        (true, Some(sink), Some(token)) => stream_with_sink("apr", &argv, sink, &token),
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
pub fn stream_with_sink(
    program: &str,
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

    /// #2403 — `max_tokens: "8"` produced a 32-token generation whose result
    /// JSON echoed `"max_tokens": 32`, so the client could not even detect
    /// that its request had been ignored.
    #[test]
    fn string_max_tokens_reaches_the_cli() {
        let argv = build_argv(
            &serde_json::json!({ "model_path": "m.gguf", "prompt": "hi", "max_tokens": "8" }),
            false,
        )
        .expect("numeric string is usable");
        let idx = argv
            .iter()
            .position(|a| a == "--max-tokens")
            .unwrap_or_else(|| panic!("max_tokens dropped: {argv:?}"));
        assert_eq!(argv[idx + 1], "8");
    }

    #[test]
    fn unusable_temperature_is_an_error_not_a_dropped_flag() {
        let (_tx, rx) = std::sync::mpsc::channel::<()>();
        let result = call(
            &serde_json::json!({ "model_path": "m.gguf", "temperature": "warm" }),
            &rx,
        );
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("temperature"));
    }

    /// Streaming selects `--stream`; the argument handling is otherwise
    /// identical, so a wrong-typed value must not sneak through that path.
    #[test]
    fn streaming_argv_uses_stream_flag_and_same_coercion() {
        let argv = build_argv(
            &serde_json::json!({ "model_path": "m.gguf", "top_p": "0.9" }),
            true,
        )
        .expect("numeric string is usable");
        assert!(argv.contains(&"--stream".to_string()), "{argv:?}");
        assert!(!argv.contains(&"--json".to_string()), "{argv:?}");
        assert!(argv.contains(&"--top-p".to_string()), "{argv:?}");
    }

    /// #2419: `max_tokens:"eight"` was dropped and the run proceeded with the
    /// default 32 — a caller asking for 8 got 4x the generation it requested
    /// with no diagnostic. The tool must refuse before spawning anything.
    #[test]
    fn non_integer_max_tokens_is_rejected_before_the_subprocess_runs() {
        let (_tx, rx) = std::sync::mpsc::channel::<()>();
        let result = call(
            &serde_json::json!({
                "model_path": "/nonexistent/model.gguf",
                "prompt": "hi",
                "max_tokens": "eight",
            }),
            &rx,
        );
        assert_eq!(result.is_error, Some(true));
        let text = &result.content[0].text;
        assert!(text.contains("max_tokens"), "must name the argument: {text}");
        assert!(text.contains("integer"), "must state the expected type: {text}");
        assert!(text.contains("eight"), "must quote what was received: {text}");
    }

    #[test]
    fn non_numeric_temperature_is_rejected() {
        let (_tx, rx) = std::sync::mpsc::channel::<()>();
        let result = call(
            &serde_json::json!({
                "model_path": "/nonexistent/model.gguf",
                "temperature": "hot",
            }),
            &rx,
        );
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("Invalid temperature"));
    }
}
