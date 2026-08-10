//! `apr.trace` — M2 tool. Layer-by-layer tensor trace for debugging a model.
//!
//! Wraps `apr trace <model> --json [--layer <pat>] [--reference <path>]`.

#![allow(clippy::disallowed_methods)] // serde_json::json! macro expands to .unwrap() internally

use crate::tools::subprocess::run_apr;
use crate::types::{InputSchema, ToolCallResult, ToolDefinition};

/// Tool name registered with MCP clients.
pub const NAME: &str = "apr.trace";

/// Return the MCP tool definition for `apr.trace`.
///
/// FALSIFY-MCP-008: the `inputSchema` is parsed from the build-time codegen
/// constant `crate::schemas::APR_TRACE_SCHEMA`, which `build.rs` emits from
/// `contracts/apr-mcp-tool-schemas-v1.yaml`. The contract is the single
/// source of truth — the live `tools/list` response and the YAML must agree
/// byte-for-byte after JSON canonicalization (asserted by
/// `tests/falsify_mcp_008.rs`).
#[must_use]
pub fn trace_tool_definition() -> ToolDefinition {
    let input_schema: InputSchema = serde_json::from_str(crate::schemas::APR_TRACE_SCHEMA).expect(
        "FALSIFY-MCP-008: apr.trace codegen constant must parse as InputSchema; \
             regenerate by editing contracts/apr-mcp-tool-schemas-v1.yaml and rebuilding",
    );
    ToolDefinition {
        name: NAME.to_string(),
        description: crate::schemas::APR_TRACE_DESCRIPTION.to_string(),
        input_schema,
    }
}

/// Execute `apr.trace` by spawning `apr trace <model> --json [...flags]`.
#[must_use]
pub fn call(args: &serde_json::Value) -> ToolCallResult {
    // FALSIFY-MCP-013: distinguish absent from wrong-typed.
    let model_path = match crate::tools::require_str(args, "model_path") {
        Ok(value) => value,
        Err(message) => return ToolCallResult::error(message),
    };

    let mut owned: Vec<String> = vec![
        "trace".to_string(),
        model_path.to_string(),
        "--json".to_string(),
    ];

    if let Some(pat) = args.get("layer").and_then(|v| v.as_str()) {
        if !pat.is_empty() {
            owned.push("--layer".to_string());
            owned.push(pat.to_string());
        }
    }
    if let Some(ref_path) = args.get("reference").and_then(|v| v.as_str()) {
        if !ref_path.is_empty() {
            owned.push("--reference".to_string());
            owned.push(ref_path.to_string());
        }
    }

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
    definition: trace_tool_definition,
    dispatch: dispatch,
);

#[cfg(test)]
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;

    #[test]
    fn definition_has_correct_name_and_required_field() {
        let def = trace_tool_definition();
        assert_eq!(def.name, "apr.trace");
        assert_eq!(def.input_schema.schema_type, "object");
        assert_eq!(def.input_schema.required, vec!["model_path".to_string()]);
        for field in ["model_path", "layer", "reference"] {
            assert!(
                def.input_schema.properties.contains_key(field),
                "property {field} present"
            );
        }
    }

    #[test]
    fn missing_model_path_returns_error() {
        let result = call(&serde_json::json!({}));
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("model_path"));
    }
}
