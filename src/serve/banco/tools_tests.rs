//! Tool calling framework tests.

use super::tools::{ToolCall, ToolDefinition, ToolRegistry, ToolResult};

// ============================================================================
// ToolRegistry unit tests
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_TOOL_001_built_in_tools() {
    let registry = ToolRegistry::new();
    let tools = registry.list();
    assert!(tools.len() >= 3, "should have at least 3 built-in tools");
    let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    assert!(names.contains(&"calculator"));
    assert!(names.contains(&"code_execution"));
    assert!(names.contains(&"web_search"));
}

#[test]
#[allow(non_snake_case)]
fn test_TOOL_002_calculator_basic() {
    let registry = ToolRegistry::new();
    let call = ToolCall {
        id: "call-1".to_string(),
        name: "calculator".to_string(),
        arguments: serde_json::json!({"expression": "2 + 3 * 4"}),
    };
    let result = registry.execute(&call);
    assert!(result.error.is_none(), "error: {:?}", result.error);
    assert_eq!(result.content, "14");
}

#[test]
#[allow(non_snake_case)]
fn test_TOOL_003_calculator_parentheses() {
    let registry = ToolRegistry::new();
    let call = ToolCall {
        id: "call-2".to_string(),
        name: "calculator".to_string(),
        arguments: serde_json::json!({"expression": "(2 + 3) * 4"}),
    };
    let result = registry.execute(&call);
    assert!(result.error.is_none());
    assert_eq!(result.content, "20");
}

#[test]
#[allow(non_snake_case)]
fn test_TOOL_004_calculator_division() {
    let registry = ToolRegistry::new();
    let call = ToolCall {
        id: "call-3".to_string(),
        name: "calculator".to_string(),
        arguments: serde_json::json!({"expression": "10 / 3"}),
    };
    let result = registry.execute(&call);
    assert!(result.error.is_none());
    let val: f64 = result.content.parse().expect("parse");
    assert!((val - 3.333_333_333_333_333_5).abs() < 0.01);
}

#[test]
#[allow(non_snake_case)]
fn test_TOOL_005_calculator_division_by_zero() {
    let registry = ToolRegistry::new();
    let call = ToolCall {
        id: "call-4".to_string(),
        name: "calculator".to_string(),
        arguments: serde_json::json!({"expression": "5 / 0"}),
    };
    let result = registry.execute(&call);
    assert!(result.error.is_some());
    assert!(result.error.unwrap().contains("Division by zero"));
}

#[test]
#[allow(non_snake_case)]
fn test_TOOL_006_calculator_negation() {
    let registry = ToolRegistry::new();
    let call = ToolCall {
        id: "call-5".to_string(),
        name: "calculator".to_string(),
        arguments: serde_json::json!({"expression": "-5 + 3"}),
    };
    let result = registry.execute(&call);
    assert!(result.error.is_none());
    assert_eq!(result.content, "-2");
}

#[test]
#[allow(non_snake_case)]
fn test_TOOL_007_code_execution_dry_run() {
    let registry = ToolRegistry::new();
    let call = ToolCall {
        id: "call-6".to_string(),
        name: "code_execution".to_string(),
        arguments: serde_json::json!({"language": "python", "code": "print('hello')"}),
    };
    let result = registry.execute(&call);
    assert!(result.error.is_none());
    assert!(result.content.contains("sandbox dry-run"));
    assert!(result.content.contains("python"));
}

#[test]
#[allow(non_snake_case)]
fn test_TOOL_008_unknown_tool() {
    let registry = ToolRegistry::new();
    let call = ToolCall {
        id: "call-7".to_string(),
        name: "nonexistent".to_string(),
        arguments: serde_json::json!({}),
    };
    let result = registry.execute(&call);
    assert!(result.error.is_some());
    assert!(result.error.unwrap().contains("Unknown tool"));
}

#[test]
#[allow(non_snake_case)]
fn test_TOOL_009_register_custom_tool() {
    let registry = ToolRegistry::new();
    registry.register(ToolDefinition {
        name: "my_tool".to_string(),
        description: "Custom tool".to_string(),
        parameters: serde_json::json!({"type": "object"}),
        enabled: true,
        required_tier: None,
    });
    assert!(registry.get("my_tool").is_some());
    assert!(registry.list().iter().any(|t| t.name == "my_tool"));
}

#[test]
#[allow(non_snake_case)]
fn test_TOOL_010_enable_disable() {
    let registry = ToolRegistry::new();
    assert!(!registry.get("web_search").unwrap().enabled);
    assert!(registry.set_enabled("web_search", true));
    assert!(registry.get("web_search").unwrap().enabled);
    assert!(registry.set_enabled("web_search", false));
    assert!(!registry.get("web_search").unwrap().enabled);
}

#[test]
#[allow(non_snake_case)]
fn test_TOOL_011_tier_filtering() {
    let registry = ToolRegistry::new();
    // web_search requires Standard tier and is disabled by default
    registry.set_enabled("web_search", true);

    let sovereign = registry.list_for_tier("Sovereign");
    assert!(!sovereign.iter().any(|t| t.name == "web_search"));

    let standard = registry.list_for_tier("Standard");
    assert!(standard.iter().any(|t| t.name == "web_search"));
}

#[test]
#[allow(non_snake_case)]
fn test_TOOL_012_tool_call_serde() {
    let json = serde_json::json!({
        "id": "call-1",
        "name": "calculator",
        "arguments": {"expression": "2+2"}
    });
    let call: ToolCall = serde_json::from_value(json).expect("parse");
    assert_eq!(call.name, "calculator");

    let result = ToolResult {
        tool_call_id: "call-1".to_string(),
        name: "calculator".to_string(),
        content: "4".to_string(),
        error: None,
    };
    let json = serde_json::to_value(&result).expect("serialize");
    assert_eq!(json["content"], "4");
    // error is None → skip_serializing_if means key is absent
    assert!(json.get("error").is_none() || json["error"].is_null());
}

// ============================================================================
// Tool endpoint tests
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_TOOL_HDL_001_list_tools() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let response = app
        .oneshot(Request::get("/api/v1/tools").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    let tools = json["tools"].as_array().expect("tools");
    assert!(tools.len() >= 2); // calculator + code_execution (web_search disabled)
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_TOOL_HDL_002_execute_calculator() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({
        "id": "test-call",
        "name": "calculator",
        "arguments": {"expression": "100 / 4"}
    });
    let response = app
        .oneshot(
            Request::post("/api/v1/tools/execute")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["content"], "25");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_TOOL_HDL_003_execute_unknown_tool() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({
        "id": "test-call",
        "name": "nonexistent",
        "arguments": {}
    });
    let response = app
        .oneshot(
            Request::post("/api/v1/tools/execute")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::NOT_FOUND);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_TOOL_HDL_004_register_custom_tool() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({
        "name": "my_custom",
        "description": "A custom tool",
        "parameters": {"type": "object"},
        "enabled": true
    });
    let response = app
        .oneshot(
            Request::post("/api/v1/tools")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["name"], "my_custom");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_TOOL_HDL_005_configure_tool() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({"enabled": true});
    let response = app
        .oneshot(
            Request::put("/api/v1/tools/web_search/config")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
}

// ============================================================================
// Self-healing retry tests
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_TOOL_013_self_heal_success_no_retry() {
    let registry = ToolRegistry::new();
    let call = ToolCall {
        id: "call-1".to_string(),
        name: "calculator".to_string(),
        arguments: serde_json::json!({"expression": "2+2"}),
    };
    let outcome = registry.execute_with_retry(&call, 3);
    assert!(!outcome.should_retry);
    assert_eq!(outcome.result.content, "4");
    assert_eq!(outcome.retries_remaining, 0);
}

#[test]
#[allow(non_snake_case)]
fn test_TOOL_014_self_heal_error_triggers_retry() {
    let registry = ToolRegistry::new();
    let call = ToolCall {
        id: "call-2".to_string(),
        name: "calculator".to_string(),
        arguments: serde_json::json!({"expression": ""}),
    };
    let outcome = registry.execute_with_retry(&call, 3);
    assert!(outcome.should_retry);
    assert!(outcome.error_context.is_some());
    assert_eq!(outcome.retries_remaining, 2);
}

#[test]
#[allow(non_snake_case)]
fn test_TOOL_015_self_heal_zero_retries() {
    let registry = ToolRegistry::new();
    let call = ToolCall {
        id: "call-3".to_string(),
        name: "calculator".to_string(),
        arguments: serde_json::json!({"expression": ""}),
    };
    let outcome = registry.execute_with_retry(&call, 0);
    assert!(!outcome.should_retry);
    assert!(outcome.result.error.is_some());
}

#[test]
#[allow(non_snake_case)]
fn test_TOOL_016_self_heal_unknown_tool() {
    let registry = ToolRegistry::new();
    let call = ToolCall {
        id: "call-4".to_string(),
        name: "nonexistent".to_string(),
        arguments: serde_json::json!({}),
    };
    let outcome = registry.execute_with_retry(&call, 2);
    assert!(outcome.should_retry);
    assert!(outcome.error_context.unwrap().contains("Unknown tool"));
    assert_eq!(outcome.retries_remaining, 1);
}

// ============================================================================
// Chat attachment endpoint tests
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_ATTACH_HDL_001_chat_with_attachment() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "Summarize this document"}],
        "attachments": [
            {"name": "readme.txt", "content": "This is a Rust project for ML."}
        ]
    });
    let response = app
        .oneshot(
            Request::post("/api/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    // Attachment content should be reflected in the response context
    let content = json["choices"][0]["message"]["content"].as_str().unwrap_or("");
    assert!(!content.is_empty());
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_ATTACH_HDL_002_chat_with_tools_field() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Evaluate math",
                    "parameters": {"type": "object"}
                }
            }
        ],
        "tool_choice": "auto"
    });
    let response = app
        .oneshot(
            Request::post("/api/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
}

#[test]
#[allow(non_snake_case)]
fn test_ATTACH_003_attachment_serde() {
    let json = serde_json::json!({
        "messages": [{"role": "user", "content": "hi"}],
        "attachments": [
            {"name": "code.py", "content": "print('hello')"},
            {"name": "data.csv", "content": "a,b\n1,2", "content_type": "text/csv"}
        ]
    });
    let req: super::types::BancoChatRequest = serde_json::from_value(json).expect("parse");
    assert_eq!(req.attachments.len(), 2);
    assert_eq!(req.attachments[0].name, "code.py");
    assert_eq!(req.attachments[1].content_type, Some("text/csv".to_string()));
}

#[test]
#[allow(non_snake_case)]
fn test_ATTACH_004_tool_spec_serde() {
    let json = serde_json::json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}}
            }
        }
    });
    let spec: super::types::ToolSpec = serde_json::from_value(json).expect("parse");
    assert_eq!(spec.tool_type, "function");
    assert_eq!(spec.function.name, "get_weather");
}
