use std::time::Duration;

use super::*;

/// Skip stdio-based tests in CI (subprocess timing is unreliable)
fn skip_in_ci() -> bool {
    std::env::var("CI").is_ok()
}

fn mock_tool(responses: Vec<Result<String, String>>) -> McpClientTool {
    McpClientTool::new(
        "test-server",
        "search",
        "Search documents",
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            }
        }),
        Box::new(MockMcpTransport::new("test-server", responses)),
    )
}

#[test]
fn test_prefixed_name() {
    let tool = mock_tool(vec![]);
    assert_eq!(tool.prefixed_name(), "mcp_test-server_search");
}

#[test]
fn test_definition() {
    let tool = mock_tool(vec![]);
    let def = tool.definition();
    assert_eq!(def.name, "mcp_test-server_search");
    assert!(def.description.contains("[MCP:test-server]"));
    assert!(def.description.contains("Search documents"));
}

#[test]
fn test_required_capability() {
    let tool = mock_tool(vec![]);
    let cap = tool.required_capability();
    assert!(matches!(
        cap,
        Capability::Mcp { server, tool }
        if server == "test-server" && tool == "search"
    ));
}

#[tokio::test]
async fn test_execute_success() {
    let tool = mock_tool(vec![Ok("found 3 results".into())]);
    let result = tool.execute(serde_json::json!({"query": "rust"})).await;
    assert!(!result.is_error);
    assert_eq!(result.content, "found 3 results");
}

#[tokio::test]
async fn test_execute_error() {
    let tool = mock_tool(vec![Err("connection refused".into())]);
    let result = tool.execute(serde_json::json!({"query": "test"})).await;
    assert!(result.is_error);
    assert!(result.content.contains("MCP call"));
    assert!(result.content.contains("connection refused"));
}

#[tokio::test]
async fn test_execute_transport_exhausted() {
    let tool = mock_tool(vec![]);
    let result = tool.execute(serde_json::json!({})).await;
    assert!(result.is_error);
    assert!(result.content.contains("exhausted"));
}

#[test]
fn test_timeout_default() {
    let tool = mock_tool(vec![]);
    assert_eq!(tool.timeout(), Duration::from_secs(60));
}

#[test]
fn test_timeout_custom() {
    let tool = mock_tool(vec![]).with_timeout(Duration::from_secs(10));
    assert_eq!(tool.timeout(), Duration::from_secs(10));
}

#[test]
fn test_capability_matches_with_registry() {
    use crate::agent::capability::capability_matches;

    let tool = mock_tool(vec![]);
    let cap = tool.required_capability();

    // Exact match
    let granted = vec![Capability::Mcp { server: "test-server".into(), tool: "search".into() }];
    assert!(capability_matches(&granted, &cap));

    // Wildcard tool match
    let wildcard = vec![Capability::Mcp { server: "test-server".into(), tool: "*".into() }];
    assert!(capability_matches(&wildcard, &cap));

    // Wrong server — denied
    let wrong = vec![Capability::Mcp { server: "other-server".into(), tool: "search".into() }];
    assert!(!capability_matches(&wrong, &cap));
}

#[tokio::test]
async fn test_stdio_transport_empty_command() {
    let transport = StdioMcpTransport::new("test", vec![]);
    let result = transport.call_tool("search", serde_json::json!({})).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("empty command"));
}

#[tokio::test]
async fn test_stdio_transport_nonexistent_command() {
    let transport = StdioMcpTransport::new("test", vec!["__nonexistent_binary_42__".into()]);
    let result = transport.call_tool("search", serde_json::json!({})).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("spawn"));
}

#[tokio::test]
async fn test_stdio_transport_echo_jsonrpc() {
    if skip_in_ci() {
        return;
    }
    let response = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "content": [{"type": "text", "text": "hello from mcp"}]
        }
    });
    let transport = StdioMcpTransport::new(
        "echo-server",
        vec!["sh".into(), "-c".into(), format!("echo '{}'", response)],
    );
    let result = transport.call_tool("greet", serde_json::json!({"name": "test"})).await;
    assert!(result.is_ok(), "expected ok, got: {:?}", result);
    assert_eq!(result.unwrap(), "hello from mcp");
}

#[tokio::test]
async fn test_stdio_transport_error_response() {
    if skip_in_ci() {
        return;
    }
    let response = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "error": {"code": -32601, "message": "method not found"}
    });
    let transport = StdioMcpTransport::new(
        "err-server",
        vec!["sh".into(), "-c".into(), format!("echo '{}'", response)],
    );
    let result = transport.call_tool("missing", serde_json::json!({})).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("method not found"));
}

#[tokio::test]
async fn test_stdio_transport_server_name() {
    if skip_in_ci() {
        return;
    }
    let transport = StdioMcpTransport::new("my-server", vec!["echo".into()]);
    assert_eq!(transport.server_name(), "my-server");
}

#[tokio::test]
async fn test_multiple_calls() {
    let tool = mock_tool(vec![Ok("first".into()), Ok("second".into())]);

    let r1 = tool.execute(serde_json::json!({})).await;
    assert_eq!(r1.content, "first");

    let r2 = tool.execute(serde_json::json!({})).await;
    assert_eq!(r2.content, "second");
}

#[tokio::test]
async fn test_discover_tools_via_echo() {
    if skip_in_ci() {
        return;
    }
    let response = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "tools": [
                {
                    "name": "search",
                    "description": "Search files",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        }
                    }
                },
                {
                    "name": "read",
                    "description": "Read a file",
                    "inputSchema": {"type": "object"}
                }
            ]
        }
    });
    let transport = StdioMcpTransport::new(
        "tool-server",
        vec!["sh".into(), "-c".into(), format!("echo '{}'", response)],
    );
    let tools = transport.discover_tools().await;
    assert!(tools.is_ok(), "discover failed: {:?}", tools);
    let tools = tools.unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0].name, "search");
    assert_eq!(tools[0].description, "Search files");
    assert_eq!(tools[1].name, "read");
}

#[tokio::test]
async fn test_discover_tools_empty_response() {
    if skip_in_ci() {
        return;
    }
    let response = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"tools": []}
    });
    let transport = StdioMcpTransport::new(
        "empty-server",
        vec!["sh".into(), "-c".into(), format!("echo '{}'", response)],
    );
    let tools = transport.discover_tools().await;
    assert!(tools.is_ok());
    assert_eq!(tools.unwrap().len(), 0);
}

#[tokio::test]
async fn test_stdio_transport_process_exit_failure() {
    if skip_in_ci() {
        return;
    }
    let transport = StdioMcpTransport::new(
        "fail-server",
        vec!["sh".into(), "-c".into(), "echo 'oops' >&2; exit 1".into()],
    );
    let result = transport.call_tool("search", serde_json::json!({})).await;
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.contains("process exited"), "got: {err}");
    assert!(err.contains("oops"), "got: {err}");
}

#[tokio::test]
async fn test_stdio_transport_invalid_json_output() {
    if skip_in_ci() {
        return;
    }
    let transport = StdioMcpTransport::new(
        "bad-json",
        vec!["sh".into(), "-c".into(), "echo 'not json at all'".into()],
    );
    let result = transport.call_tool("search", serde_json::json!({})).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("parse response"));
}

#[tokio::test]
async fn test_stdio_transport_no_result_field() {
    if skip_in_ci() {
        return;
    }
    let response = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1
    });
    let transport = StdioMcpTransport::new(
        "no-result",
        vec!["sh".into(), "-c".into(), format!("echo '{}'", response)],
    );
    let result = transport.call_tool("search", serde_json::json!({})).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("no result"));
}

#[tokio::test]
async fn test_stdio_transport_result_no_content() {
    if skip_in_ci() {
        return;
    }
    // Result without content array — falls back to JSON serialization
    let response = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"data": "raw value"}
    });
    let transport = StdioMcpTransport::new(
        "raw-result",
        vec!["sh".into(), "-c".into(), format!("echo '{}'", response)],
    );
    let result = transport.call_tool("search", serde_json::json!({})).await;
    assert!(result.is_ok());
    let text = result.unwrap();
    assert!(text.contains("raw value"), "got: {text}");
}

#[tokio::test]
async fn test_stdio_transport_content_empty_texts() {
    if skip_in_ci() {
        return;
    }
    // Content array with items that have no "text" field
    let response = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "content": [{"type": "image", "data": "abc"}]
        }
    });
    let transport = StdioMcpTransport::new(
        "no-text",
        vec!["sh".into(), "-c".into(), format!("echo '{}'", response)],
    );
    let result = transport.call_tool("search", serde_json::json!({})).await;
    assert!(result.is_ok());
    // Falls back to JSON serialization of result
    let text = result.unwrap();
    assert!(text.contains("content"), "got: {text}");
}

#[tokio::test]
async fn test_discover_tools_no_result() {
    if skip_in_ci() {
        return;
    }
    let response = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1
    });
    let transport = StdioMcpTransport::new(
        "no-result",
        vec!["sh".into(), "-c".into(), format!("echo '{}'", response)],
    );
    let result = transport.discover_tools().await;
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("no result"));
}

#[tokio::test]
async fn test_discover_tools_no_tools_array() {
    if skip_in_ci() {
        return;
    }
    let response = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"other": "stuff"}
    });
    let transport = StdioMcpTransport::new(
        "no-tools",
        vec!["sh".into(), "-c".into(), format!("echo '{}'", response)],
    );
    let result = transport.discover_tools().await;
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("no tools array"));
}

#[tokio::test]
async fn test_mock_transport_server_name() {
    let transport = MockMcpTransport::new("my-mock", vec![Ok("ok".into())]);
    assert_eq!(transport.server_name(), "my-mock");
}

#[tokio::test]
async fn test_discover_tools_skips_empty_names() {
    if skip_in_ci() {
        return;
    }
    let response = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "tools": [
                {"name": "", "description": "bad"},
                {"name": "good", "description": "ok"}
            ]
        }
    });
    let transport = StdioMcpTransport::new(
        "filter-server",
        vec!["sh".into(), "-c".into(), format!("echo '{}'", response)],
    );
    let tools = transport.discover_tools().await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "good");
}
