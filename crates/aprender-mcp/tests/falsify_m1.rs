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

/// FALSIFY-MCP-002 (M1 subset): `tools/list` returns the M1 tool set and each
/// schema is a valid JSON Schema object. Full 8-tool check lands in M2.
#[test]
fn falsify_mcp_002_tools_list_schema_shape() {
    let mut server = AprMcpServer::new();
    let resp = server.handle_request(&request(2, "tools/list", serde_json::json!({})));

    let result = resp.result.expect("tools/list result");
    let tools = result["tools"].as_array().expect("tools array");

    assert_eq!(tools.len(), 1, "M1 registers exactly one tool");
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
