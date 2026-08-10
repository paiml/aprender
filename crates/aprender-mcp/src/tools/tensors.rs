//! `apr.tensors` — M2 tool. List tensor names, shapes, and (optionally) stats.
//!
//! Wraps `apr tensors <model> --json [--stats] [--filter <pat>] [--limit <n>]`.

#![allow(clippy::disallowed_methods)] // serde_json::json! macro expands to .unwrap() internally

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

/// Execute `apr.tensors` by spawning `apr tensors <model> --json [...flags]`.
#[must_use]
pub fn call(args: &serde_json::Value) -> ToolCallResult {
    // FALSIFY-MCP-013: distinguish absent from wrong-typed.
    let model_path = match crate::tools::require_str(args, "model_path") {
        Ok(value) => value,
        Err(message) => return ToolCallResult::error(message),
    };

    let mut argv: Vec<&str> = vec!["tensors", model_path, "--json"];
    if args
        .get("stats")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        argv.push("--stats");
    }
    let filter = args.get("filter").and_then(|v| v.as_str()).unwrap_or("");
    if !filter.is_empty() {
        argv.push("--filter");
        argv.push(filter);
    }

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
}
