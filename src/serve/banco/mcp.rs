//! Model Context Protocol (MCP) handler — JSON-RPC 2.0 endpoint.
//!
//! Exposes Banco's tools as MCP tools, enabling Claude Desktop, Cursor,
//! and other MCP-compatible clients to connect via HTTP SSE transport.
//!
//! Implements: initialize, tools/list, tools/call, resources/list, prompts/list.

use serde::{Deserialize, Serialize};

/// MCP JSON-RPC 2.0 request.
#[derive(Debug, Clone, Deserialize)]
pub struct McpRequest {
    pub jsonrpc: String,
    pub id: Option<serde_json::Value>,
    pub method: String,
    #[serde(default)]
    pub params: serde_json::Value,
}

/// MCP JSON-RPC 2.0 response.
#[derive(Debug, Clone, Serialize)]
pub struct McpResponse {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<McpError>,
}

/// MCP JSON-RPC error.
#[derive(Debug, Clone, Serialize)]
pub struct McpError {
    pub code: i32,
    pub message: String,
}

impl McpResponse {
    pub fn success(id: Option<serde_json::Value>, result: serde_json::Value) -> Self {
        Self { jsonrpc: "2.0".to_string(), id, result: Some(result), error: None }
    }

    pub fn error(id: Option<serde_json::Value>, code: i32, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(McpError { code, message: message.into() }),
        }
    }
}

/// MCP server info returned by initialize.
#[derive(Debug, Clone, Serialize)]
pub struct McpServerInfo {
    pub name: String,
    pub version: String,
}

/// MCP tool definition (MCP protocol format).
#[derive(Debug, Clone, Serialize)]
pub struct McpTool {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: serde_json::Value,
}

/// MCP resource definition.
#[derive(Debug, Clone, Serialize)]
pub struct McpResource {
    pub uri: String,
    pub name: String,
    pub description: String,
    #[serde(rename = "mimeType")]
    pub mime_type: String,
}

/// MCP prompt definition.
#[derive(Debug, Clone, Serialize)]
pub struct McpPrompt {
    pub name: String,
    pub description: String,
}

/// Process an MCP JSON-RPC request using the Banco tool registry.
pub fn handle_mcp_request(
    request: &McpRequest,
    tools: &super::tools::ToolRegistry,
    prompts: &super::prompts::PromptStore,
) -> McpResponse {
    match request.method.as_str() {
        "initialize" => handle_initialize(request),
        "tools/list" => handle_tools_list(request, tools),
        "tools/call" => handle_tools_call(request, tools),
        "resources/list" => handle_resources_list(request),
        "prompts/list" => handle_prompts_list(request, prompts),
        "ping" => McpResponse::success(request.id.clone(), serde_json::json!({})),
        _ => McpResponse::error(
            request.id.clone(),
            -32601,
            format!("Method not found: {}", request.method),
        ),
    }
}

fn handle_initialize(request: &McpRequest) -> McpResponse {
    McpResponse::success(
        request.id.clone(),
        serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {},
                "resources": {},
                "prompts": {}
            },
            "serverInfo": {
                "name": "banco",
                "version": env!("CARGO_PKG_VERSION")
            }
        }),
    )
}

fn handle_tools_list(request: &McpRequest, tools: &super::tools::ToolRegistry) -> McpResponse {
    let mcp_tools: Vec<McpTool> = tools
        .list()
        .into_iter()
        .filter(|t| t.enabled)
        .map(|t| McpTool { name: t.name, description: t.description, input_schema: t.parameters })
        .collect();

    McpResponse::success(request.id.clone(), serde_json::json!({ "tools": mcp_tools }))
}

fn handle_tools_call(request: &McpRequest, tools: &super::tools::ToolRegistry) -> McpResponse {
    let name = request.params.get("name").and_then(|v| v.as_str()).unwrap_or("");
    let arguments = request.params.get("arguments").cloned().unwrap_or(serde_json::json!({}));

    if tools.get(name).is_none() {
        return McpResponse::error(request.id.clone(), -32602, format!("Unknown tool: {name}"));
    }

    let call = super::tools::ToolCall {
        id: format!("mcp-{}", epoch_secs()),
        name: name.to_string(),
        arguments,
    };

    let result = tools.execute(&call);

    if let Some(err) = &result.error {
        McpResponse::success(
            request.id.clone(),
            serde_json::json!({
                "content": [{"type": "text", "text": err}],
                "isError": true
            }),
        )
    } else {
        McpResponse::success(
            request.id.clone(),
            serde_json::json!({
                "content": [{"type": "text", "text": result.content}]
            }),
        )
    }
}

fn handle_resources_list(request: &McpRequest) -> McpResponse {
    let resources = vec![
        McpResource {
            uri: "banco://system".to_string(),
            name: "System Info".to_string(),
            description: "Banco system status and configuration".to_string(),
            mime_type: "application/json".to_string(),
        },
        McpResource {
            uri: "banco://models".to_string(),
            name: "Models".to_string(),
            description: "Available and loaded models".to_string(),
            mime_type: "application/json".to_string(),
        },
    ];
    McpResponse::success(request.id.clone(), serde_json::json!({ "resources": resources }))
}

fn handle_prompts_list(request: &McpRequest, prompts: &super::prompts::PromptStore) -> McpResponse {
    let mcp_prompts: Vec<McpPrompt> = prompts
        .list()
        .into_iter()
        .map(|p| McpPrompt { name: p.name, description: p.content })
        .collect();

    McpResponse::success(request.id.clone(), serde_json::json!({ "prompts": mcp_prompts }))
}

fn epoch_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()
}
