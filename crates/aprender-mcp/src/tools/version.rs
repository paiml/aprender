//! `apr.version` — M1 stub tool that reports the aprender-mcp crate version.

#![allow(clippy::disallowed_methods)] // serde_json::json! macro expands to .unwrap() internally

use crate::types::{InputSchema, ToolCallResult, ToolDefinition};

/// Tool name registered with MCP clients.
pub const NAME: &str = "apr.version";

/// Crate version baked in at compile time.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Return the MCP tool definition for `apr.version`.
///
/// FALSIFY-MCP-008: the `inputSchema` is parsed from the build-time codegen
/// constant `crate::schemas::APR_VERSION_SCHEMA`, which `build.rs` emits from
/// `contracts/apr-mcp-tool-schemas-v1.yaml`. The contract is the single
/// source of truth — the live `tools/list` response and the YAML must agree
/// byte-for-byte after JSON canonicalization (asserted by
/// `tests/falsify_mcp_008.rs`).
#[must_use]
pub fn version_tool_definition() -> ToolDefinition {
    let input_schema: InputSchema = serde_json::from_str(crate::schemas::APR_VERSION_SCHEMA)
        .expect(
            "FALSIFY-MCP-008: apr.version codegen constant must parse as InputSchema; \
             regenerate by editing contracts/apr-mcp-tool-schemas-v1.yaml and rebuilding",
        );
    ToolDefinition {
        name: NAME.to_string(),
        description: "Return the aprender-mcp server version. Takes no arguments.".to_string(),
        input_schema,
    }
}

/// Execute the `apr.version` tool.
#[must_use]
pub fn call(_args: &serde_json::Value) -> ToolCallResult {
    let payload = serde_json::json!({
        "server": crate::SERVER_NAME,
        "version": VERSION,
        "protocol_version": crate::PROTOCOL_VERSION,
    });
    ToolCallResult::success(payload.to_string())
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // serde_json::json! expands to code that hits unwrap()
mod tests {
    use super::*;

    #[test]
    fn definition_has_correct_name() {
        let def = version_tool_definition();
        assert_eq!(def.name, "apr.version");
        assert!(def.input_schema.required.is_empty());
        assert_eq!(def.input_schema.schema_type, "object");
    }

    #[test]
    fn call_returns_version_payload() {
        let result = call(&serde_json::json!({}));
        assert!(result.is_error.is_none());
        let text = &result.content[0].text;
        let parsed: serde_json::Value = serde_json::from_str(text).expect("valid json");
        assert_eq!(parsed["server"], "aprender-mcp");
        assert_eq!(parsed["version"], VERSION);
        assert_eq!(parsed["protocol_version"], "2024-11-05");
    }
}
