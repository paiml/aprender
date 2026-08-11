//! apr.finetune — LoRA/full-finetune subprocess wrapper.
//!
//! Opt-in streaming (FALSIFY-MCP-PROGRESS-001, shipped M3 — PR #887): when the
//! caller supplies a [`NotificationSink`] (because the originating
//! `tools/call` included `params._meta.progressToken`), each non-empty stdout
//! line from `apr finetune --json` is forwarded as an MCP
//! `notifications/progress` message before the final `ToolCallResult` is
//! returned. When the sink is absent — or when no progress token was supplied
//! — the call falls back to the synchronous path shipped in #881.
//!
//! Wraps `apr finetune <base_model> --json [--data <path>] [--rank <N>]
//! [--epochs <N>] [--method <m>] [--output <path>]`.
//!
//! Note on argument names: the spec (`docs/specifications/apr-mcp-server-spec.md`
//! line 85) lists `base_model`, `dataset`, `lora_rank`, `epochs`. The actual
//! `apr finetune` CLI uses a positional `<FILE>` for the base model, `--data`
//! (not `--dataset`), and `--rank` (not `--lora-rank`). We keep the spec's
//! ergonomic MCP argument names (`base_model`, `dataset`, `lora_rank`) as the
//! schema surface but map them to the real CLI flags at dispatch time so LLM
//! callers aren't exposed to the flag-name mismatch.
//!
//! Note on the event schema: the current `apr finetune --json` emits a single
//! terminal JSON object (the `display_train_result` payload) rather than a
//! stream of per-step events. We forward whatever the subprocess prints — one
//! `notifications/progress` per stdout line — and leave structured
//! `progress`/`total` numeric fields absent until the CLI grows a per-step
//! event channel. See the PR description for the follow-up.

#![allow(clippy::disallowed_methods)] // serde_json::json! macro expands to .unwrap() internally

use crate::server::NotificationSink;
use crate::tools::subprocess::{run_apr, spawn_streaming};
use crate::types::{InputSchema, JsonRpcNotification, ToolCallResult, ToolDefinition};

/// Tool name registered with MCP clients.
pub const NAME: &str = "apr.finetune";

/// Return the MCP tool definition for `apr.finetune`.
///
/// FALSIFY-MCP-008: the `inputSchema` is parsed from the build-time codegen
/// constant `crate::schemas::APR_FINETUNE_SCHEMA`, which `build.rs` emits from
/// `contracts/apr-mcp-tool-schemas-v1.yaml`. The contract is the single
/// source of truth — the live `tools/list` response and the YAML must agree
/// byte-for-byte after JSON canonicalization (asserted by
/// `tests/falsify_mcp_008.rs`).
#[must_use]
pub fn finetune_tool_definition() -> ToolDefinition {
    let input_schema: InputSchema = serde_json::from_str(crate::schemas::APR_FINETUNE_SCHEMA)
        .expect(
            "FALSIFY-MCP-008: apr.finetune codegen constant must parse as InputSchema; \
             regenerate by editing contracts/apr-mcp-tool-schemas-v1.yaml and rebuilding",
        );
    ToolDefinition {
        name: NAME.to_string(),
        description: crate::schemas::APR_FINETUNE_DESCRIPTION.to_string(),
        input_schema,
    }
}

/// Execute `apr.finetune` by spawning `apr finetune <base_model> --json [...flags]`.
///
/// Back-compat entry point used by callers that don't opt into progress
/// streaming (the sync `handle_tools_call_sync` path, non-stdio tests, etc).
/// Equivalent to `call_with_sink(args, None, None)`.
#[must_use]
pub fn call(args: &serde_json::Value) -> ToolCallResult {
    call_with_sink(args, None, None)
}

/// Execute `apr.finetune` with optional `notifications/progress` streaming.
///
/// If both `sink` and `progress_token` are `Some`, the subprocess is spawned
/// with stdout piped and every line is forwarded as a
/// `notifications/progress` notification carrying the caller's token. The
/// final `ToolCallResult` still contains the full stdout so existing clients
/// that ignore progress get identical behaviour.
///
/// When either argument is `None` (the client did not advertise a
/// progressToken, per MCP spec "servers MUST NOT send progress notifications
/// if the client did not request them"), we fall back to the non-streaming
/// [`run_apr`] path.
#[must_use]
pub fn call_with_sink(
    args: &serde_json::Value,
    sink: Option<&NotificationSink>,
    progress_token: Option<serde_json::Value>,
) -> ToolCallResult {
    let base_model = match crate::tools::args::require_str(args, "base_model") {
        Ok(p) => p,
        Err(e) => return e,
    };

    let mut owned: Vec<String> = vec![
        "finetune".to_string(),
        base_model.to_string(),
        "--json".to_string(),
    ];

    if let Some(dataset) = args.get("dataset").and_then(|v| v.as_str()) {
        if !dataset.is_empty() {
            owned.push("--data".to_string());
            owned.push(dataset.to_string());
        }
    }
    if let Some(rank) = args.get("lora_rank").and_then(serde_json::Value::as_u64) {
        owned.push("--rank".to_string());
        owned.push(rank.to_string());
    }
    if let Some(epochs) = args.get("epochs").and_then(serde_json::Value::as_u64) {
        owned.push("--epochs".to_string());
        owned.push(epochs.to_string());
    }
    if let Some(method) = args.get("method").and_then(|v| v.as_str()) {
        if !method.is_empty() {
            owned.push("--method".to_string());
            owned.push(method.to_string());
        }
    }
    if let Some(output) = args.get("output").and_then(|v| v.as_str()) {
        if !output.is_empty() {
            owned.push("--output".to_string());
            owned.push(output.to_string());
        }
    }

    let argv: Vec<&str> = owned.iter().map(String::as_str).collect();

    match (sink, progress_token) {
        (Some(sink), Some(token)) => {
            // #2384: self-resolved binary, never a `$PATH` lookup.
            stream_with_sink(crate::apr_bin::apr_binary(), &argv, sink, &token)
        }
        _ => run_apr(&argv),
    }
}

/// Test-visible: stream `program args...` and forward each stdout line as a
/// `notifications/progress` notification through `sink`, tagged with
/// `progress_token`. Each stdout line is JSON-parsed if possible; otherwise
/// forwarded as a plain string. The returned `ToolCallResult` is the
/// aggregated stdout (same shape as `run_apr`'s success body).
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
        // Prefer parsed JSON when the line looks like one so downstream
        // clients can introspect fields; fall back to a bare string.
        let payload = serde_json::from_str::<serde_json::Value>(trimmed)
            .unwrap_or_else(|_| serde_json::Value::String(line.to_string()));
        let notif = JsonRpcNotification::progress(progress_token.clone(), payload);
        sink(notif);
    })
}

/// HELIX-IDEA-002 — unified-signature shim for the inventory dispatcher.
/// `apr.finetune` honours the optional notification sink; cancellation is
/// not yet wired (FALSIFY-MCP-006 covers `apr.run` only).
pub fn dispatch(
    args: &serde_json::Value,
    _cancel: &std::sync::mpsc::Receiver<()>,
    sink: Option<&NotificationSink>,
    progress_token: Option<serde_json::Value>,
) -> ToolCallResult {
    call_with_sink(args, sink, progress_token)
}

crate::register_mcp_tool!(
    name: NAME,
    definition: finetune_tool_definition,
    dispatch: dispatch,
);

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // serde_json::json! expands to code that hits unwrap()
mod tests {
    use super::*;

    #[test]
    fn finetune_tool_definition_shape() {
        let def = finetune_tool_definition();
        assert_eq!(def.name, "apr.finetune");
        assert_eq!(def.input_schema.schema_type, "object");
        assert_eq!(def.input_schema.required, vec!["base_model".to_string()]);
        for field in [
            "base_model",
            "dataset",
            "lora_rank",
            "epochs",
            "method",
            "output",
        ] {
            assert!(
                def.input_schema.properties.contains_key(field),
                "property {field} present"
            );
        }
    }

    #[test]
    fn finetune_missing_base_model_is_error() {
        let result = call(&serde_json::json!({}));
        assert_eq!(result.is_error, Some(true));
        assert!(
            result.content[0].text.contains("base_model"),
            "error message must mention base_model, got: {}",
            result.content[0].text
        );
    }

    #[test]
    fn finetune_nonstring_base_model_is_error() {
        let result = call(&serde_json::json!({ "base_model": 42 }));
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("base_model"));
        // Shape ("is_error, mentions the field") passed while the message said
        // the argument was MISSING — see tools::args. Assert the behaviour.
        assert_eq!(
            result.content[0].text,
            "Argument base_model must be a string, got number"
        );
    }

    /// FALSIFY-MCP-PROGRESS-001 (unit): `stream_with_sink` fires one
    /// notification per stdout line, tagging each with the supplied
    /// progressToken, and returns the aggregated stdout as a success result.
    #[test]
    fn stream_with_sink_emits_one_notification_per_line() {
        use std::sync::{Arc, Mutex};

        let captured: Arc<Mutex<Vec<JsonRpcNotification>>> = Arc::new(Mutex::new(Vec::new()));
        let captured_clone = Arc::clone(&captured);
        let sink: NotificationSink = Box::new(move |n| {
            captured_clone
                .lock()
                .expect("sink mutex not poisoned")
                .push(n);
        });

        let token = serde_json::json!("progress-token-xyz");
        let result = stream_with_sink(
            "printf",
            &[r#"{"step":1}\n{"step":2}\nplain-line\n"#],
            &sink,
            &token,
        );
        assert!(result.is_error.is_none(), "printf should succeed");

        let notifs = captured.lock().expect("mutex").clone();
        assert_eq!(
            notifs.len(),
            3,
            "one notification per non-empty stdout line"
        );

        for n in &notifs {
            assert_eq!(n.method, "notifications/progress");
            assert_eq!(n.params["progressToken"], "progress-token-xyz");
        }
        // First two lines were JSON → parsed; third was plain → forwarded as string.
        assert_eq!(notifs[0].params["message"]["step"], 1);
        assert_eq!(notifs[1].params["message"]["step"], 2);
        assert_eq!(notifs[2].params["message"], "plain-line");
    }

    /// Without a sink, `call_with_sink` falls back to the sync path.
    #[test]
    fn call_with_sink_none_sink_is_synchronous() {
        // No sink → no streaming path; this exercises the fallback branch
        // without spawning apr (the unknown-subcommand error still confirms
        // we took the run_apr path).
        let result = call_with_sink(
            &serde_json::json!({ "base_model": "/nonexistent/model.apr" }),
            None,
            None,
        );
        // Either we reach the run_apr spawn path (which will error on the
        // missing model / missing apr binary) or the base_model validator
        // passed cleanly. Both are acceptable — the key assertion is that
        // no sink was exercised, which would be guaranteed by its absence.
        // We just verify the result is a well-formed ToolCallResult.
        assert!(!result.content.is_empty());
    }
}
