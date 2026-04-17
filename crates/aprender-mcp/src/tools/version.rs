//! `apr.version` — M1 stub tool that reports the aprender-mcp crate version.

#![allow(clippy::disallowed_methods)] // serde_json::json! macro expands to .unwrap() internally

use crate::types::{InputSchema, ToolCallResult, ToolDefinition};
use std::collections::HashMap;

/// Tool name registered with MCP clients.
pub const NAME: &str = "apr.version";

/// Crate version baked in at compile time.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Return the MCP tool definition for `apr.version`.
#[must_use]
pub fn version_tool_definition() -> ToolDefinition {
    ToolDefinition {
        name: NAME.to_string(),
        description: "Return the aprender-mcp server version. Takes no arguments.".to_string(),
        input_schema: InputSchema {
            schema_type: "object".to_string(),
            properties: HashMap::new(),
            required: vec![],
        },
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
