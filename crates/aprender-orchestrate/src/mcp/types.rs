//! MCP Protocol Types
//!
//! JSON-RPC 2.0 types for Model Context Protocol communication.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// JSON-RPC 2.0 Types
// ============================================================================

/// JSON-RPC 2.0 request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: Option<serde_json::Value>,
    pub method: String,
    #[serde(default)]
    pub params: serde_json::Value,
}

/// JSON-RPC 2.0 response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// JSON-RPC 2.0 error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl JsonRpcResponse {
    pub fn success(id: Option<serde_json::Value>, result: serde_json::Value) -> Self {
        Self { jsonrpc: "2.0".to_string(), id, result: Some(result), error: None }
    }

    pub fn error(id: Option<serde_json::Value>, code: i64, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError { code, message: message.into(), data: None }),
        }
    }
}

// ============================================================================
// MCP Protocol Types
// ============================================================================

/// MCP server capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerCapabilities {
    pub tools: ToolsCapability,
}

/// Tools capability declaration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsCapability {
    #[serde(rename = "listChanged")]
    pub list_changed: bool,
}

/// MCP tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: InputSchema,
}

/// JSON Schema for tool input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputSchema {
    #[serde(rename = "type")]
    pub schema_type: String,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub properties: HashMap<String, PropertySchema>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required: Vec<String>,
}

/// Property schema for tool parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertySchema {
    #[serde(rename = "type")]
    pub prop_type: String,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#enum: Option<Vec<String>>,
}

/// MCP tool call result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallResult {
    pub content: Vec<ContentBlock>,
    #[serde(rename = "isError", skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

/// Content block in tool result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentBlock {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: String,
}

impl ContentBlock {
    pub fn text(content: impl Into<String>) -> Self {
        Self { content_type: "text".to_string(), text: content.into() }
    }
}

impl ToolCallResult {
    pub fn success(text: impl Into<String>) -> Self {
        Self { content: vec![ContentBlock::text(text)], is_error: None }
    }

    pub fn error(text: impl Into<String>) -> Self {
        Self { content: vec![ContentBlock::text(text)], is_error: Some(true) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_rpc_response_success() {
        let resp = JsonRpcResponse::success(Some(serde_json::json!(1)), serde_json::json!("ok"));
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
        assert_eq!(resp.jsonrpc, "2.0");
    }

    #[test]
    fn test_json_rpc_response_error() {
        let resp = JsonRpcResponse::error(Some(serde_json::json!(1)), -32600, "Invalid Request");
        assert!(resp.result.is_none());
        assert!(resp.error.is_some());
        assert_eq!(resp.error.as_ref().map(|e| e.code), Some(-32600));
    }

    #[test]
    fn test_tool_call_result_success() {
        let result = ToolCallResult::success("hello");
        assert_eq!(result.content.len(), 1);
        assert_eq!(result.content[0].text, "hello");
        assert!(result.is_error.is_none());
    }

    #[test]
    fn test_tool_call_result_error() {
        let result = ToolCallResult::error("fail");
        assert!(result.is_error == Some(true));
    }

    #[test]
    fn test_content_block_text() {
        let block = ContentBlock::text("test");
        assert_eq!(block.content_type, "text");
        assert_eq!(block.text, "test");
    }

    #[test]
    fn test_json_rpc_request_deserialize() {
        let json = r#"{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}"#;
        let req: JsonRpcRequest = serde_json::from_str(json).expect("json deserialize failed");
        assert_eq!(req.method, "tools/list");
        assert_eq!(req.jsonrpc, "2.0");
    }

    #[test]
    fn test_input_schema_serialization() {
        let schema = InputSchema {
            schema_type: "object".to_string(),
            properties: HashMap::new(),
            required: vec![],
        };
        let json = serde_json::to_string(&schema).expect("json serialize failed");
        assert!(json.contains("\"type\":\"object\""));
    }
}
