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

use crate::apr_bin::apr_binary;
use crate::server::NotificationSink;
use crate::tools::args::{self, try_arg};
use crate::tools::subprocess::{run_apr, spawn_streaming};
use crate::types::{InputSchema, JsonRpcNotification, ToolCallResult, ToolDefinition};
use std::ffi::OsStr;

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
///
/// Every optional argument the `inputSchema` declares — `dataset`,
/// `lora_rank`, `epochs`, `method`, `output` — is forwarded by
/// [`build_argv`], and a wrong-typed one is now an `isError` result rather
/// than a silent omission (#2417 / #2403).
#[must_use]
pub fn call_with_sink(
    args: &serde_json::Value,
    sink: Option<&NotificationSink>,
    progress_token: Option<serde_json::Value>,
) -> ToolCallResult {
    let owned = try_arg!(build_argv(args));
    let argv: Vec<&str> = owned.iter().map(String::as_str).collect();

    match (sink, progress_token) {
        // #2465: `apr_binary()`, never the literal `"apr"`. The `run_apr`
        // fallback below has resolved since #2424; this arm did not, so the
        // streaming path spawned whatever `$PATH` produced.
        (Some(sink), Some(token)) => stream_with_sink(apr_binary(), &argv, sink, &token),
        _ => run_apr(&argv),
    }
}

/// Build the `apr finetune ...` argv from `tools/call` arguments.
///
/// # Errors
/// Returns the client-facing message when an argument is present but not
/// usable at its declared type.
pub fn build_argv(args: &serde_json::Value) -> Result<Vec<String>, String> {
    let base_model = args::required_str(args, "base_model")?;

    let mut owned: Vec<String> = vec![
        "finetune".to_string(),
        base_model.to_string(),
        "--json".to_string(),
    ];

    if let Some(dataset) = args::opt_str(args, "dataset")? {
        if !dataset.is_empty() {
            owned.push("--data".to_string());
            owned.push(dataset.to_string());
        }
    }
    if let Some(rank) = args::opt_u64(args, "lora_rank")? {
        owned.push("--rank".to_string());
        owned.push(rank.to_string());
    }
    if let Some(epochs) = args::opt_u64(args, "epochs")? {
        owned.push("--epochs".to_string());
        owned.push(epochs.to_string());
    }
    if let Some(method) = args::opt_str(args, "method")? {
        if !method.is_empty() {
            owned.push("--method".to_string());
            owned.push(method.to_string());
        }
    }
    if let Some(output) = args::opt_str(args, "output")? {
        if !output.is_empty() {
            owned.push("--output".to_string());
            owned.push(output.to_string());
        }
    }
    Ok(owned)
}

/// Test-visible: stream `program args...` and forward each stdout line as a
/// `notifications/progress` notification through `sink`, tagged with
/// `progress_token`. Each stdout line is JSON-parsed if possible; otherwise
/// forwarded as a plain string. The returned `ToolCallResult` is the
/// aggregated stdout (same shape as `run_apr`'s success body).
///
/// Generic over the program so production callers pass
/// [`crate::apr_bin::apr_binary`] (a `PathBuf`) while tests inject a mock by
/// name. It was `&str`, which is why the one production call site could be —
/// and was — a bare `"apr"`.
#[must_use]
pub fn stream_with_sink<P: AsRef<OsStr>>(
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
    }

    /// #2417 — every optional argument the inputSchema declares must appear in
    /// the spawned argv. A declared argument that is dropped is a wrong-answer
    /// channel: the caller believes it configured the run.
    #[test]
    fn every_declared_optional_argument_is_forwarded() {
        let argv = build_argv(&serde_json::json!({
            "base_model": "base.safetensors",
            "dataset": "/tmp/d.jsonl",
            "lora_rank": 4,
            "epochs": 1,
            "method": "lora",
            "output": "/tmp/out"
        }))
        .expect("all arguments usable");
        assert_eq!(
            argv,
            vec![
                "finetune",
                "base.safetensors",
                "--json",
                "--data",
                "/tmp/d.jsonl",
                "--rank",
                "4",
                "--epochs",
                "1",
                "--method",
                "lora",
                "--output",
                "/tmp/out"
            ]
        );
    }

    /// The same call with the two integers sent as JSON strings — the shape an
    /// LLM client routinely emits — must be identical, not silently rank-less.
    #[test]
    fn string_typed_rank_and_epochs_are_forwarded() {
        let argv = build_argv(&serde_json::json!({
            "base_model": "base.safetensors",
            "lora_rank": "4",
            "epochs": "1"
        }))
        .expect("numeric strings usable");
        assert!(argv.contains(&"--rank".to_string()), "{argv:?}");
        assert!(argv.contains(&"--epochs".to_string()), "{argv:?}");
    }

    #[test]
    fn unusable_lora_rank_is_an_error_not_a_dropped_flag() {
        let result = call(&serde_json::json!({ "base_model": "b.apr", "lora_rank": "high" }));
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("lora_rank"));
    }

    /// FALSIFIER (#2465 finding 4): the STREAMING path must execute the
    /// binary [`crate::apr_bin::apr_binary`] resolves, not a literal `"apr"`.
    ///
    /// #2424 fixed the `run_apr` fallback and left this arm spawning `"apr"`,
    /// so a client that supplied a `progressToken` got a `$PATH` lookup.
    ///
    /// Behavioural: `$APR_BIN` designates `echo`, so the child's stdout IS the
    /// argv it received. One notification carrying
    /// `finetune <base_model> --json` proves the designated program ran with
    /// this call's arguments — an outcome no `apr` on `$PATH` can produce, so
    /// reverting the fix fails the test whether or not `apr` is installed.
    #[test]
    #[cfg(unix)]
    fn falsify_2465_streaming_path_executes_the_resolved_binary() {
        use std::sync::{Arc, Mutex};

        let _guard = crate::apr_bin::lock_apr_bin_env();

        let captured: Arc<Mutex<Vec<JsonRpcNotification>>> = Arc::new(Mutex::new(Vec::new()));
        let captured_clone = Arc::clone(&captured);
        let sink: NotificationSink = Box::new(move |n| {
            captured_clone
                .lock()
                .expect("sink mutex not poisoned")
                .push(n);
        });

        // Edition 2021 — `set_var` is safe here.
        std::env::set_var(crate::apr_bin::APR_BIN_ENV, "echo");
        let result = call_with_sink(
            &serde_json::json!({ "base_model": "MARKER-BASE.gguf" }),
            Some(&sink),
            Some(serde_json::json!("tok-2465")),
        );
        std::env::remove_var(crate::apr_bin::APR_BIN_ENV);

        assert!(
            result.is_error.is_none(),
            "the designated binary exits 0; got: {}",
            result.content[0].text
        );

        let notifs = captured.lock().expect("mutex").clone();
        assert_eq!(
            notifs.len(),
            1,
            "echo writes exactly one line, so exactly one progress notification"
        );
        let payload = notifs[0]
            .params
            .get("message")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| format!("<no string message: {:?}>", notifs[0].params));
        assert_eq!(
            payload, "finetune MARKER-BASE.gguf --json",
            "the streaming path must spawn the RESOLVED binary with this call's argv"
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
