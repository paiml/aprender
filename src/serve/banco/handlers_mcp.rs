//! MCP (Model Context Protocol) HTTP handler.
//!
//! POST /api/v1/mcp — JSON-RPC 2.0 endpoint for MCP clients.
//! GET /api/v1/mcp/info — MCP server metadata.

use axum::{extract::State, response::Json};

use super::mcp::{handle_mcp_request, McpRequest, McpResponse};
use super::state::BancoState;

/// POST /api/v1/mcp — handle MCP JSON-RPC request.
pub async fn mcp_handler(
    State(state): State<BancoState>,
    Json(request): Json<McpRequest>,
) -> Json<McpResponse> {
    let response = handle_mcp_request(&request, &state.tools, &state.prompts);
    Json(response)
}

/// GET /api/v1/mcp/info — MCP server info (for discovery).
pub async fn mcp_info_handler() -> Json<McpInfoResponse> {
    Json(McpInfoResponse {
        protocol: "mcp".to_string(),
        version: "2024-11-05".to_string(),
        server: "banco".to_string(),
        server_version: env!("CARGO_PKG_VERSION").to_string(),
        transport: "http".to_string(),
        endpoint: "/api/v1/mcp".to_string(),
    })
}

/// MCP server info response.
#[derive(Debug, serde::Serialize)]
pub struct McpInfoResponse {
    pub protocol: String,
    pub version: String,
    pub server: String,
    pub server_version: String,
    pub transport: String,
    pub endpoint: String,
}
