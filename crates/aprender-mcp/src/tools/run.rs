//! `apr.run` — M2 tool. Synchronous inference via subprocess wrapper.
//!
//! Wraps `apr run <model> --json [--prompt X] [--max-tokens N] [--temperature T] [--top-p P]`.
//!
//! M3 (FALSIFY-MCP-006) adds cancellation: the call accepts a cancel receiver
//! and forwards it to [`run_apr_cancellable`], which SIGTERMs the spawned
//! subprocess on signal and SIGKILLs after the grace window. Per-token
//! `notifications/progress` streaming is deferred to M4 — it needs an
//! `apr run --stream` CLI flag prereq that doesn't yet exist.

#![allow(clippy::disallowed_methods)] // serde_json::json! macro expands to .unwrap() internally

use crate::tools::subprocess::{run_apr_cancellable, CANCEL_GRACE_MS};
use crate::types::{InputSchema, ToolCallResult, ToolDefinition};
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
        description:
            "Run synchronous inference on a model. Wraps `apr run <model> --json` and returns tokens + tok/s + stop reason."
                .to_string(),
        input_schema,
    }
}

/// Execute `apr.run` by spawning `apr run <model> --json [...flags]`.
///
/// `cancel_rx` is signalled by the MCP dispatcher when a matching
/// `notifications/cancelled` arrives on the same request id (FALSIFY-MCP-006).
/// Pass a never-firing channel for tests or direct non-MCP callers.
#[must_use]
pub fn call(args: &serde_json::Value, cancel_rx: &Receiver<()>) -> ToolCallResult {
    let Some(model_path) = args.get("model_path").and_then(|v| v.as_str()) else {
        return ToolCallResult::error("Missing required argument: model_path");
    };

    let mut owned: Vec<String> = vec![
        "run".to_string(),
        model_path.to_string(),
        "--json".to_string(),
    ];

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
    run_apr_cancellable(&argv, cancel_rx, CANCEL_GRACE_MS)
}

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
