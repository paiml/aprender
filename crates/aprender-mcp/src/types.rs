//! JSON-RPC 2.0 + MCP protocol types.
//!
//! Mirrors `aprender-orchestrate::mcp::types` so downstream tools can depend on
//! `aprender-mcp` without pulling the entire batuta dependency tree.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
