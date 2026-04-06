//! MCP (Model Context Protocol) tests.

use super::mcp::{handle_mcp_request, McpRequest};

fn make_request(method: &str, params: serde_json::Value) -> McpRequest {
    McpRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(serde_json::json!(1)),
        method: method.to_string(),
        params,
    }
}

// ============================================================================
// MCP protocol tests
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_MCP_001_initialize() {
    let tools = super::tools::ToolRegistry::new();
    let prompts = super::prompts::PromptStore::new();
    let req = make_request("initialize", serde_json::json!({}));
    let resp = handle_mcp_request(&req, &tools, &prompts);
    assert!(resp.error.is_none());
    let result = resp.result.expect("result");
    assert_eq!(result["protocolVersion"], "2024-11-05");
    assert_eq!(result["serverInfo"]["name"], "banco");
    assert!(result["capabilities"]["tools"].is_object());
}

#[test]
#[allow(non_snake_case)]
fn test_MCP_002_tools_list() {
    let tools = super::tools::ToolRegistry::new();
    let prompts = super::prompts::PromptStore::new();
    let req = make_request("tools/list", serde_json::json!({}));
    let resp = handle_mcp_request(&req, &tools, &prompts);
    assert!(resp.error.is_none());
    let result = resp.result.expect("result");
    let tool_list = result["tools"].as_array().expect("tools array");
    // calculator and code_execution are enabled by default
    assert!(tool_list.len() >= 2);
    assert!(tool_list.iter().any(|t| t["name"] == "calculator"));
}

#[test]
#[allow(non_snake_case)]
fn test_MCP_003_tools_call_calculator() {
    let tools = super::tools::ToolRegistry::new();
    let prompts = super::prompts::PromptStore::new();
    let req = make_request(
        "tools/call",
        serde_json::json!({
            "name": "calculator",
            "arguments": {"expression": "6 * 7"}
        }),
    );
    let resp = handle_mcp_request(&req, &tools, &prompts);
    assert!(resp.error.is_none());
    let result = resp.result.expect("result");
    let content = result["content"].as_array().expect("content");
    assert_eq!(content[0]["text"], "42");
}

#[test]
#[allow(non_snake_case)]
fn test_MCP_004_tools_call_unknown() {
    let tools = super::tools::ToolRegistry::new();
    let prompts = super::prompts::PromptStore::new();
    let req =
        make_request("tools/call", serde_json::json!({"name": "nonexistent", "arguments": {}}));
    let resp = handle_mcp_request(&req, &tools, &prompts);
    assert!(resp.error.is_some());
    assert_eq!(resp.error.unwrap().code, -32602);
}

#[test]
#[allow(non_snake_case)]
fn test_MCP_005_resources_list() {
    let tools = super::tools::ToolRegistry::new();
    let prompts = super::prompts::PromptStore::new();
    let req = make_request("resources/list", serde_json::json!({}));
    let resp = handle_mcp_request(&req, &tools, &prompts);
    assert!(resp.error.is_none());
    let result = resp.result.expect("result");
    let resources = result["resources"].as_array().expect("resources");
    assert!(resources.len() >= 2);
}

#[test]
#[allow(non_snake_case)]
fn test_MCP_006_prompts_list() {
    let tools = super::tools::ToolRegistry::new();
    let prompts = super::prompts::PromptStore::new();
    let req = make_request("prompts/list", serde_json::json!({}));
    let resp = handle_mcp_request(&req, &tools, &prompts);
    assert!(resp.error.is_none());
    let result = resp.result.expect("result");
    let prompt_list = result["prompts"].as_array().expect("prompts");
    // PromptStore has 3 built-in presets
    assert!(prompt_list.len() >= 3);
}

#[test]
#[allow(non_snake_case)]
fn test_MCP_007_ping() {
    let tools = super::tools::ToolRegistry::new();
    let prompts = super::prompts::PromptStore::new();
    let req = make_request("ping", serde_json::json!({}));
    let resp = handle_mcp_request(&req, &tools, &prompts);
    assert!(resp.error.is_none());
}

#[test]
#[allow(non_snake_case)]
fn test_MCP_008_unknown_method() {
    let tools = super::tools::ToolRegistry::new();
    let prompts = super::prompts::PromptStore::new();
    let req = make_request("nonexistent/method", serde_json::json!({}));
    let resp = handle_mcp_request(&req, &tools, &prompts);
    assert!(resp.error.is_some());
    assert_eq!(resp.error.unwrap().code, -32601);
}

#[test]
#[allow(non_snake_case)]
fn test_MCP_009_tool_call_error_is_not_rpc_error() {
    let tools = super::tools::ToolRegistry::new();
    let prompts = super::prompts::PromptStore::new();
    // Calculator with empty expression returns error, but it's a tool error not RPC error
    let req = make_request(
        "tools/call",
        serde_json::json!({
            "name": "calculator",
            "arguments": {"expression": ""}
        }),
    );
    let resp = handle_mcp_request(&req, &tools, &prompts);
    // MCP spec: tool errors are returned as isError=true in result, not as RPC error
    assert!(resp.error.is_none(), "tool errors should not be RPC errors");
    let result = resp.result.expect("result");
    assert!(result["isError"].as_bool().unwrap_or(false));
}

// ============================================================================
// MCP endpoint tests
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_MCP_HDL_001_endpoint() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {}
    });
    let response = app
        .oneshot(
            Request::post("/api/v1/mcp")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["jsonrpc"], "2.0");
    assert_eq!(json["result"]["serverInfo"]["name"], "banco");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_MCP_HDL_002_info() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let response = app
        .oneshot(Request::get("/api/v1/mcp/info").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["protocol"], "mcp");
    assert_eq!(json["server"], "banco");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_MCP_HDL_003_tools_call_via_endpoint() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": "calculator", "arguments": {"expression": "10+20"}}
    });
    let response = app
        .oneshot(
            Request::post("/api/v1/mcp")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["result"]["content"][0]["text"], "30");
}
