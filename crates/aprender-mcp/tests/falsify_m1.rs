//! FALSIFY-MCP-001 + FALSIFY-MCP-002 (M1 subset) — protocol-level gates.
//!
//! These mirror the spec in `docs/specifications/apr-mcp-server-spec.md` and
//! exercise `AprMcpServer` through its public JSON-RPC surface only.

#![allow(clippy::disallowed_methods)] // serde_json::json! expands to code that hits unwrap()

use aprender_mcp::{AprMcpServer, JsonRpcRequest, PROTOCOL_VERSION};
use std::time::Instant;

fn request(id: u64, method: &str, params: serde_json::Value) -> JsonRpcRequest {
    JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(serde_json::json!(id)),
        method: method.to_string(),
        params,
    }
}

/// FALSIFY-MCP-001: `initialize` responds within 500ms with the correct
/// `protocolVersion`. In-process dispatch, no stdio overhead — the spec budget
/// includes transport; we assert ≤50ms which is ~10× headroom for CI noise.
#[test]
fn falsify_mcp_001_initialize_under_500ms() {
    let mut server = AprMcpServer::new();
    let req = request(1, "initialize", serde_json::json!({}));

    let t0 = Instant::now();
    let resp = server.handle_request(&req);
    let elapsed = t0.elapsed();

    assert!(
        elapsed.as_millis() < 50,
        "initialize took {elapsed:?}, spec budget 500ms / test budget 50ms"
    );

    let result = resp.result.expect("initialize result");
    assert_eq!(result["protocolVersion"], PROTOCOL_VERSION);
}

/// FALSIFY-MCP-002 (progressive): `tools/list` must include every tool that
/// has shipped so far (apr.version from M1, apr.validate from M2 slice 1) and
/// every registered schema must be a valid JSON Schema object. The full
/// 8-tool check lands when M2 completes.
#[test]
fn falsify_mcp_002_tools_list_schema_shape() {
    let mut server = AprMcpServer::new();
    let resp = server.handle_request(&request(2, "tools/list", serde_json::json!({})));

    let result = resp.result.expect("tools/list result");
    let tools = result["tools"].as_array().expect("tools array");

    let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
    for expected in [
        "apr.version",
        "apr.validate",
        "apr.tensors",
        "apr.bench",
        "apr.qa",
        "apr.trace",
    ] {
        assert!(names.contains(&expected), "{expected} registered");
    }

    for tool in tools {
        assert!(tool["name"].is_string(), "name present");
        assert!(tool["description"].is_string(), "description present");
        let schema = &tool["inputSchema"];
        assert_eq!(schema["type"], "object", "schema is object type");
    }
}

/// Parse errors on malformed JSON must NOT panic — the stdio loop returns
/// JSON-RPC code -32700. The types layer handles this; we assert at the
/// dispatch layer that unknown methods return -32601.
#[test]
fn unknown_method_maps_to_minus_32601() {
    let mut server = AprMcpServer::new();
    let resp = server.handle_request(&request(3, "nonexistent/method", serde_json::json!({})));

    assert!(resp.result.is_none());
    assert_eq!(resp.error.expect("error").code, -32601);
}

/// FALSIFY-MCP-VALIDATE-001: calling `apr.validate` without `model_path` must
/// surface a tool-level error (isError:true) rather than a JSON-RPC error —
/// the MCP spec requires argument validation failures to come back as
/// tool-call results so the LLM sees and can react to them.
#[test]
fn falsify_validate_missing_model_path_is_tool_error() {
    let mut server = AprMcpServer::new();
    let resp = server.handle_request(&request(
        20,
        "tools/call",
        serde_json::json!({ "name": "apr.validate", "arguments": {} }),
    ));

    // JSON-RPC layer must succeed; the error belongs inside the result.
    assert!(resp.error.is_none(), "JSON-RPC error should be none");
    let result = resp.result.expect("tools/call result");
    assert_eq!(result["isError"], true);
    assert!(result["content"][0]["text"]
        .as_str()
        .expect("text")
        .contains("model_path"));
}

/// End-to-end `initialize` → `tools/list` → `tools/call` works on one server
/// instance without state leaking between requests.
#[test]
fn sequential_dispatch_keeps_server_healthy() {
    let mut server = AprMcpServer::new();

    let init = server.handle_request(&request(10, "initialize", serde_json::json!({})));
    assert!(init.error.is_none());

    let list = server.handle_request(&request(11, "tools/list", serde_json::json!({})));
    assert!(list.error.is_none());

    let call = server.handle_request(&request(
        12,
        "tools/call",
        serde_json::json!({ "name": "apr.version", "arguments": {} }),
    ));
    let text = call.result.expect("call result")["content"][0]["text"]
        .as_str()
        .expect("text")
        .to_string();
    let payload: serde_json::Value = serde_json::from_str(&text).expect("json");
    assert_eq!(payload["server"], "aprender-mcp");
}
