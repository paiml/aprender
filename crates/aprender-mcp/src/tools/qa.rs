//! `apr.qa` — M2 tool. The 8-gate quality checklist; first stop for any model issue.
//!
//! Wraps `apr qa <model> --json [--assert-tps N] [--max-tokens N] [--iterations N]`.

#![allow(clippy::disallowed_methods)] // serde_json::json! macro expands to .unwrap() internally

use crate::tools::subprocess::run_apr;
use crate::types::{InputSchema, ToolCallResult, ToolDefinition};

/// Tool name registered with MCP clients.
pub const NAME: &str = "apr.qa";

/// Return the MCP tool definition for `apr.qa`.
///
/// FALSIFY-MCP-008: the `inputSchema` is parsed from the build-time codegen
/// constant `crate::schemas::APR_QA_SCHEMA`, which `build.rs` emits from
/// `contracts/apr-mcp-tool-schemas-v1.yaml`. The contract is the single
/// source of truth — the live `tools/list` response and the YAML must agree
/// byte-for-byte after JSON canonicalization (asserted by
/// `tests/falsify_mcp_008.rs`).
#[must_use]
pub fn qa_tool_definition() -> ToolDefinition {
    let input_schema: InputSchema = serde_json::from_str(crate::schemas::APR_QA_SCHEMA).expect(
        "FALSIFY-MCP-008: apr.qa codegen constant must parse as InputSchema; \
             regenerate by editing contracts/apr-mcp-tool-schemas-v1.yaml and rebuilding",
    );
    ToolDefinition {
        name: NAME.to_string(),
        description: crate::schemas::APR_QA_DESCRIPTION.to_string(),
        input_schema,
    }
}

/// Execute `apr.qa` by spawning `apr qa <model> --json [...flags]`.
#[must_use]
pub fn call(args: &serde_json::Value) -> ToolCallResult {
    let model_path = match crate::tools::args::require_str(args, "model_path") {
        Ok(p) => p,
        Err(e) => return e,
    };

    let mut owned: Vec<String> = vec![
        "qa".to_string(),
        model_path.to_string(),
        "--json".to_string(),
    ];

    if let Some(tps) = args.get("assert_tps").and_then(serde_json::Value::as_f64) {
        owned.push("--assert-tps".to_string());
        owned.push(tps.to_string());
    }
    if let Some(n) = args.get("max_tokens").and_then(serde_json::Value::as_u64) {
        owned.push("--max-tokens".to_string());
        owned.push(n.to_string());
    }
    if let Some(n) = args.get("iterations").and_then(serde_json::Value::as_u64) {
        owned.push("--iterations".to_string());
        owned.push(n.to_string());
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
    definition: qa_tool_definition,
    dispatch: dispatch,
);

#[cfg(test)]
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;

    #[test]
    fn definition_has_correct_name_and_required_field() {
        let def = qa_tool_definition();
        assert_eq!(def.name, "apr.qa");
        assert_eq!(def.input_schema.schema_type, "object");
        assert_eq!(def.input_schema.required, vec!["model_path".to_string()]);
        for field in ["model_path", "assert_tps", "max_tokens", "iterations"] {
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
