//! `apr.capability` — subprocess wrapper over `apr capability --json`
//! (aprender#3856 row 3).
//!
//! The capability facts are declared once, in `contracts/apr-model-capability-v1.yaml`,
//! and the `apr` binary embeds that file. This tool does not carry a second copy:
//! it asks the binary, so the MCP answer is the CLI answer by construction and
//! cannot drift from it.

use crate::tools::subprocess::run_apr;
use crate::types::{InputSchema, ToolCallResult, ToolDefinition};

/// Tool name registered with MCP clients.
pub const NAME: &str = "apr.capability";

/// Return the MCP tool definition for `apr.capability`.
///
/// FALSIFY-MCP-008: the `inputSchema` is parsed from the build-time codegen
/// constant `crate::schemas::APR_CAPABILITY_SCHEMA`, which `build.rs` emits from
/// `contracts/apr-mcp-tool-schemas-v1.yaml`.
#[must_use]
pub fn capability_tool_definition() -> ToolDefinition {
    let input_schema: InputSchema = serde_json::from_str(crate::schemas::APR_CAPABILITY_SCHEMA)
        .expect(
            "FALSIFY-MCP-008: apr.capability codegen constant must parse as InputSchema; \
             regenerate by editing contracts/apr-mcp-tool-schemas-v1.yaml and rebuilding",
        );
    ToolDefinition {
        name: NAME.to_string(),
        description: crate::schemas::APR_CAPABILITY_DESCRIPTION.to_string(),
        input_schema,
    }
}

/// Execute `apr.capability` by spawning `apr capability --json`.
#[must_use]
pub fn call(_args: &serde_json::Value) -> ToolCallResult {
    run_apr(&["capability", "--json"])
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
    definition: capability_tool_definition,
    dispatch: dispatch,
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn definition_takes_no_arguments() {
        let def = capability_tool_definition();
        assert_eq!(def.name, "apr.capability");
        assert_eq!(def.input_schema.schema_type, "object");
        assert!(def.input_schema.required.is_empty());
        assert!(def.input_schema.properties.is_empty());
    }
}
