//! `apr.validate` — M2 subprocess wrapper over `apr validate <model> --json`.
//!
//! This was the first M2 tool shipped and established the subprocess pattern
//! every M2/M3 `apr.*` wrapper follows: spawn `apr <subcommand> --json`,
//! capture stdout, pass through to the MCP client as a single text content
//! block. Non-zero exit maps to `isError: true` with stderr attached. All 7
//! M2 wrappers (`apr.validate`, `apr.tensors`, `apr.bench`, `apr.qa`,
//! `apr.trace`, `apr.run`, `apr.serve`) and the M3 addition `apr.finetune`
//! now ship on this pattern.

#![allow(clippy::disallowed_methods)] // serde_json::json! macro expands to .unwrap() internally

use crate::tools::args::{self, try_arg};
use crate::tools::subprocess::run_apr;
use crate::types::{InputSchema, ToolCallResult, ToolDefinition};

/// Tool name registered with MCP clients.
pub const NAME: &str = "apr.validate";

/// Return the MCP tool definition for `apr.validate`.
///
/// FALSIFY-MCP-008: the `inputSchema` is parsed from the build-time codegen
/// constant `crate::schemas::APR_VALIDATE_SCHEMA`, which `build.rs` emits from
/// `contracts/apr-mcp-tool-schemas-v1.yaml`. The contract is the single
/// source of truth — the live `tools/list` response and the YAML must agree
/// byte-for-byte after JSON canonicalization (asserted by
/// `tests/falsify_mcp_008.rs`).
#[must_use]
pub fn validate_tool_definition() -> ToolDefinition {
    let input_schema: InputSchema = serde_json::from_str(crate::schemas::APR_VALIDATE_SCHEMA)
        .expect(
            "FALSIFY-MCP-008: apr.validate codegen constant must parse as InputSchema; \
             regenerate by editing contracts/apr-mcp-tool-schemas-v1.yaml and rebuilding",
        );
    ToolDefinition {
        name: NAME.to_string(),
        description: crate::schemas::APR_VALIDATE_DESCRIPTION.to_string(),
        input_schema,
    }
}

/// Execute `apr.validate` by spawning `apr validate <model_path> --json`.
#[must_use]
pub fn call(args: &serde_json::Value) -> ToolCallResult {
    let model_path = try_arg!(args::required_str(args, "model_path"));
    run_apr(&["validate", model_path, "--json"])
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
    definition: validate_tool_definition,
    dispatch: dispatch,
);

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // serde_json::json! expands to code that hits unwrap()
mod tests {
    use super::*;

    #[test]
    fn definition_has_correct_name_and_required_field() {
        let def = validate_tool_definition();
        assert_eq!(def.name, "apr.validate");
        assert_eq!(def.input_schema.schema_type, "object");
        assert_eq!(def.input_schema.required, vec!["model_path".to_string()]);
        assert!(def.input_schema.properties.contains_key("model_path"));
    }

    #[test]
    fn missing_model_path_returns_error() {
        let result = call(&serde_json::json!({}));
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("model_path"));
    }

    /// #2419: this test used to assert only `is_error`, which the defect
    /// satisfied — the tool answered "Missing required argument: model_path"
    /// for an argument that was present. The message is the behaviour the
    /// caller acts on, so the message is what is asserted.
    #[test]
    fn nonstring_model_path_is_reported_as_a_type_error_not_as_missing() {
        let result = call(&serde_json::json!({ "model_path": 42 }));
        assert_eq!(result.is_error, Some(true));
        let text = &result.content[0].text;
        assert!(
            !text.contains("Missing"),
            "model_path WAS supplied; reporting it as missing sends the caller \
             to fix the wrong thing. got: {text}"
        );
        assert!(
            text.contains("model_path"),
            "must name the argument: {text}"
        );
        assert!(
            text.contains("string"),
            "must state the expected type: {text}"
        );
        assert!(text.contains("42"), "must quote what was received: {text}");
    }
}
