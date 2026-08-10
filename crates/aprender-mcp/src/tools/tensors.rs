//! `apr.tensors` — M2 tool. List tensor names, shapes, and (optionally) stats.
//!
//! Wraps `apr tensors <model> --json [--stats] [--filter <pat>] [--limit <n>]`.

#![allow(clippy::disallowed_methods)] // serde_json::json! macro expands to .unwrap() internally

use crate::tools::args::{self, try_arg};
use crate::tools::subprocess::run_apr;
use crate::types::{InputSchema, ToolCallResult, ToolDefinition};

/// Tool name registered with MCP clients.
pub const NAME: &str = "apr.tensors";

/// Return the MCP tool definition for `apr.tensors`.
///
/// FALSIFY-MCP-008: the `inputSchema` is parsed from the build-time codegen
/// constant `crate::schemas::APR_TENSORS_SCHEMA`, which `build.rs` emits from
/// `contracts/apr-mcp-tool-schemas-v1.yaml`. The contract is the single
/// source of truth — the live `tools/list` response and the YAML must agree
/// byte-for-byte after JSON canonicalization (asserted by
/// `tests/falsify_mcp_008.rs`).
#[must_use]
pub fn tensors_tool_definition() -> ToolDefinition {
    let input_schema: InputSchema = serde_json::from_str(crate::schemas::APR_TENSORS_SCHEMA)
        .expect(
            "FALSIFY-MCP-008: apr.tensors codegen constant must parse as InputSchema; \
             regenerate by editing contracts/apr-mcp-tool-schemas-v1.yaml and rebuilding",
        );
    ToolDefinition {
        name: NAME.to_string(),
        description: crate::schemas::APR_TENSORS_DESCRIPTION.to_string(),
        input_schema,
    }
}

/// Build the `apr tensors ...` argv from `tools/call` arguments.
///
/// # Errors
/// Returns the client-facing message when an argument is present but not
/// usable at its declared type.
pub fn build_argv(args: &serde_json::Value) -> Result<Vec<String>, String> {
    let model_path = args::required_str(args, "model_path")?;

    let mut argv: Vec<String> = vec![
        "tensors".to_string(),
        model_path.to_string(),
        "--json".to_string(),
    ];
    if args::opt_bool(args, "stats")?.unwrap_or(false) {
        argv.push("--stats".to_string());
    }
    let filter = args::opt_str(args, "filter")?.unwrap_or("");
    if !filter.is_empty() {
        argv.push("--filter".to_string());
        argv.push(filter.to_string());
    }
    Ok(argv)
}

/// Execute `apr.tensors` by spawning `apr tensors <model> --json [...flags]`.
#[must_use]
pub fn call(args: &serde_json::Value) -> ToolCallResult {
    let owned = try_arg!(build_argv(args));
    let argv: Vec<&str> = owned.iter().map(String::as_str).collect();

    run_apr(&argv)
}

/// HELIX-IDEA-002 — unified-signature shim for the inventory dispatcher.
pub fn dispatch(
    args: &serde_json::Value,
    _cancel: &std::sync::mpsc::Receiver<()>,
    _sink: Option<&crate::server::NotificationSink>,
    _token: Option<serde_json::Value>,
) -> ToolCallResult {
    call(args)
}

crate::register_mcp_tool!(
    name: NAME,
    definition: tensors_tool_definition,
    dispatch: dispatch,
);

#[cfg(test)]
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;

    #[test]
    fn definition_has_correct_name_and_required_field() {
        let def = tensors_tool_definition();
        assert_eq!(def.name, "apr.tensors");
        assert_eq!(def.input_schema.schema_type, "object");
        assert_eq!(def.input_schema.required, vec!["model_path".to_string()]);
        assert!(def.input_schema.properties.contains_key("model_path"));
        assert!(def.input_schema.properties.contains_key("stats"));
        assert!(def.input_schema.properties.contains_key("filter"));
    }

    #[test]
    fn missing_model_path_returns_error() {
        let result = call(&serde_json::json!({}));
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("model_path"));
    }

    /// #2403 — `stats: "true"` used to drop `--stats`, so the caller got a
    /// bare listing back and no indication its request was ignored.
    #[test]
    fn string_stats_still_asks_for_stats() {
        let argv = build_argv(&serde_json::json!({ "model_path": "m.gguf", "stats": "true" }))
            .expect("\"true\" is a usable boolean");
        assert!(argv.contains(&"--stats".to_string()), "{argv:?}");
    }

    #[test]
    fn unusable_stats_is_an_error_not_a_dropped_flag() {
        let result = call(&serde_json::json!({ "model_path": "m.gguf", "stats": 1 }));
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("stats"));
    }
}
