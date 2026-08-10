//! JSON-RPC 2.0 + MCP protocol types.
//!
//! Mirrors `aprender-orchestrate::mcp::types` so downstream tools can depend on
//! `aprender-mcp` without pulling the entire batuta dependency tree.

#![allow(clippy::disallowed_methods)] // serde_json::json! macro expands to .unwrap() internally

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Human-readable JSON type name, for diagnostics that must tell a client
/// what it actually sent.
///
/// Used by the request-shape validator in [`crate::server`] and by
/// [`crate::tools::require_str`]. Both previously reported a wrong-typed
/// value as if it were absent, which sends a client (or an LLM) off to
/// re-send a key it already supplied.
#[must_use]
pub fn json_type_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

/// JSON-RPC 2.0 request envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: Option<serde_json::Value>,
    pub method: String,
    #[serde(default)]
    pub params: serde_json::Value,
}

/// JSON-RPC 2.0 response envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// JSON-RPC 2.0 notification envelope — server → client, no `id` field.
///
/// Used for MCP `notifications/progress` per the 2024-11-05 spec:
/// <https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/utilities/progress/>.
/// A notification MUST NOT carry an `id`; that's the serialization rule the
/// peer uses to route it as a fire-and-forget message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcNotification {
    pub jsonrpc: String,
    pub method: String,
    #[serde(default)]
    pub params: serde_json::Value,
}

impl JsonRpcNotification {
    /// Construct a `notifications/progress` payload with the spec-mandated
    /// `progressToken` echoed back from the originating request's
    /// `params._meta.progressToken`.
    ///
    /// `message` is the opaque JSON payload emitted by the subprocess on one
    /// line of stdout. The MCP spec's optional `progress`/`total` numeric
    /// fields are left absent for now — we forward the parsed JSON verbatim
    /// so higher-level clients can introspect whatever shape the CLI emits.
    #[must_use]
    pub fn progress(progress_token: serde_json::Value, message: serde_json::Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            method: "notifications/progress".to_string(),
            params: serde_json::json!({
                "progressToken": progress_token,
                "message": message,
            }),
        }
    }

    /// Serialize this notification as a single JSON line (no trailing
    /// newline). Suitable for writing to stdio with `writeln!`.
    ///
    /// # Errors
    /// Returns a `serde_json::Error` if the payload cannot be serialized,
    /// which in practice only happens for non-finite floats that a well-formed
    /// subprocess would not produce.
    pub fn to_json_line(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

/// JSON-RPC 2.0 error object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl JsonRpcResponse {
    #[must_use]
    pub fn success(id: Option<serde_json::Value>, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: Option<serde_json::Value>, code: i64, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
                data: None,
            }),
        }
    }
}

/// `initialize` response capabilities block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerCapabilities {
    pub tools: ToolsCapability,
}

/// Tools capability flags.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsCapability {
    #[serde(rename = "listChanged")]
    pub list_changed: bool,
}

/// MCP tool registration record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: InputSchema,
}

/// JSON Schema for a tool's input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputSchema {
    #[serde(rename = "type")]
    pub schema_type: String,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub properties: HashMap<String, PropertySchema>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required: Vec<String>,
}

/// JSON Schema for one input property.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertySchema {
    #[serde(rename = "type")]
    pub prop_type: String,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#enum: Option<Vec<String>>,
}

/// Result returned by `tools/call`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallResult {
    pub content: Vec<ContentBlock>,
    #[serde(rename = "isError", skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

/// One content block in a tool result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentBlock {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: String,
}

impl ContentBlock {
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content_type: "text".to_string(),
            text: content.into(),
        }
    }
}

impl ToolCallResult {
    pub fn success(text: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::text(text)],
            is_error: None,
        }
    }

    pub fn error(text: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::text(text)],
            is_error: Some(true),
        }
    }
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // serde_json::json! expands to code that hits unwrap()
mod tests {
    use super::*;

    #[test]
    fn json_rpc_response_success_round_trips() {
        let resp = JsonRpcResponse::success(Some(serde_json::json!(1)), serde_json::json!("ok"));
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
        assert_eq!(resp.jsonrpc, "2.0");
    }

    #[test]
    fn json_rpc_response_error_sets_code() {
        let resp = JsonRpcResponse::error(Some(serde_json::json!(2)), -32600, "Invalid Request");
        assert!(resp.result.is_none());
        let err = resp.error.expect("error present");
        assert_eq!(err.code, -32600);
        assert_eq!(err.message, "Invalid Request");
    }

    #[test]
    fn tool_call_result_success_has_no_error_flag() {
        let result = ToolCallResult::success("hello");
        assert_eq!(result.content.len(), 1);
        assert_eq!(result.content[0].text, "hello");
        assert!(result.is_error.is_none());
    }

    #[test]
    fn tool_call_result_error_flags_error() {
        let result = ToolCallResult::error("fail");
        assert_eq!(result.is_error, Some(true));
    }

    #[test]
    fn content_block_text_defaults_type() {
        let block = ContentBlock::text("test");
        assert_eq!(block.content_type, "text");
    }

    #[test]
    fn json_rpc_request_deserializes_tools_list() {
        let json = r#"{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}"#;
        let req: JsonRpcRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(req.method, "tools/list");
    }

    /// FALSIFY-MCP-PROGRESS-001 (wire-format unit): a `notifications/progress`
    /// serializes without an `id` field and carries `progressToken` +
    /// `message` inside `params`.
    #[test]
    fn json_rpc_notification_progress_has_no_id_field() {
        let notif = JsonRpcNotification::progress(
            serde_json::json!("tok-1"),
            serde_json::json!({"step": 1, "loss": 0.42}),
        );
        let json = notif.to_json_line().expect("serialize");
        assert!(json.contains("\"jsonrpc\":\"2.0\""));
        assert!(json.contains("\"method\":\"notifications/progress\""));
        assert!(json.contains("\"progressToken\":\"tok-1\""));
        assert!(!json.contains("\"id\""), "notifications MUST NOT carry id");
    }

    #[test]
    fn json_rpc_notification_accepts_numeric_token() {
        let notif = JsonRpcNotification::progress(serde_json::json!(7), serde_json::json!("tick"));
        let json = notif.to_json_line().expect("serialize");
        assert!(json.contains("\"progressToken\":7"));
    }

    #[test]
    fn input_schema_serializes_object_type() {
        let schema = InputSchema {
            schema_type: "object".to_string(),
            properties: HashMap::new(),
            required: vec![],
        };
        let json = serde_json::to_string(&schema).expect("serialize");
        assert!(json.contains("\"type\":\"object\""));
    }
}
