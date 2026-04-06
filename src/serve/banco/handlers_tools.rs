//! Tool calling endpoint handlers — list, configure, execute tools.

use axum::{extract::State, http::StatusCode, response::Json};
use serde::Deserialize;

use super::state::BancoState;
use super::tools::{ToolCall, ToolDefinition, ToolResult};
use super::types::ErrorResponse;

/// GET /api/v1/tools — list available tools for current privacy tier.
pub async fn list_tools_handler(State(state): State<BancoState>) -> Json<ToolsListResponse> {
    let tier = format!("{:?}", state.privacy_tier);
    let tools = state.tools.list_for_tier(&tier);
    Json(ToolsListResponse { tools })
}

/// POST /api/v1/tools/execute — execute a tool call directly.
pub async fn execute_tool_handler(
    State(state): State<BancoState>,
    Json(call): Json<ToolCall>,
) -> Result<Json<ToolResult>, (StatusCode, Json<ErrorResponse>)> {
    // Check tool exists
    if state.tools.get(&call.name).is_none() {
        return Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse::new(
                format!("Tool '{}' not found", call.name),
                "tool_not_found",
                404,
            )),
        ));
    }

    let result = state.tools.execute(&call);
    Ok(Json(result))
}

/// PUT /api/v1/tools/:name/config — enable/disable a tool.
pub async fn configure_tool_handler(
    State(state): State<BancoState>,
    axum::extract::Path(name): axum::extract::Path<String>,
    Json(config): Json<ToolConfigRequest>,
) -> Result<StatusCode, (StatusCode, Json<ErrorResponse>)> {
    if !state.tools.set_enabled(&name, config.enabled) {
        return Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse::new(format!("Tool '{name}' not found"), "tool_not_found", 404)),
        ));
    }
    Ok(StatusCode::OK)
}

/// POST /api/v1/tools — register a custom tool.
pub async fn register_tool_handler(
    State(state): State<BancoState>,
    Json(tool): Json<ToolDefinition>,
) -> Json<ToolDefinition> {
    state.tools.register(tool.clone());
    Json(tool)
}

// ============================================================================
// Types
// ============================================================================

#[derive(Debug, serde::Serialize)]
pub struct ToolsListResponse {
    pub tools: Vec<ToolDefinition>,
}

#[derive(Debug, Deserialize)]
pub struct ToolConfigRequest {
    pub enabled: bool,
}
