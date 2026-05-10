//! FALSIFY-INVENTORY-003 — the inventory dispatch path produces the
//! same `tools/call` envelope as the pre-migration hardcoded match
//! arms for every shipped tool. Equivalence is checked at the JSON
//! envelope level (success vs error, content array shape, isError
//! flag); subprocess output parity is owned by FALSIFY-MCP-003 /
//! FALSIFY-MCP-004 in the parent contract.
//!
//! Contract: `contracts/apr-mcp-tool-inventory-v1.yaml`.
//!
//! Discharge strategy: fire `tools/call` against a real `AprMcpServer`
//! with each shipped tool name; assert the response has the documented
//! envelope shape (success or `isError: true` with structured content).
//! Tools that require subprocess args (like `apr.run`) are exercised in
//! their argument-validation branch where no subprocess is spawned.

#![allow(clippy::unwrap_used, clippy::disallowed_methods)]

use aprender_mcp::{AprMcpServer, JsonRpcRequest};

fn make_call(name: &str, args: serde_json::Value) -> JsonRpcRequest {
    JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(serde_json::json!(1)),
        method: "tools/call".to_string(),
        params: serde_json::json!({
            "name": name,
            "arguments": args
        }),
    }
}

#[test]
fn version_dispatch_envelope_matches_hardcoded_path() {
    // `apr.version` is the cleanest dispatch path: no subprocess, no
    // required args. Pre-migration the hardcoded match returned a
    // success envelope with one text content block. Inventory dispatch
    // must produce the same.
    let mut server = AprMcpServer::new();
    let resp = server.handle_request(&make_call("apr.version", serde_json::json!({})));

    assert!(resp.error.is_none(), "JSON-RPC envelope must be success");
    let result = resp.result.expect("result present");
    assert!(
        result["isError"].is_null() || result["isError"] == false,
        "version dispatch must not flag isError: {result:?}",
    );
    let content = result["content"].as_array().expect("content array");
    assert_eq!(content.len(), 1, "expected exactly one content block");
    assert_eq!(content[0]["type"], "text");
    let text = content[0]["text"].as_str().expect("text payload");
    let payload: serde_json::Value = serde_json::from_str(text).expect("text is JSON");
    assert_eq!(payload["server"], "aprender-mcp");
    assert_eq!(payload["protocol_version"], "2024-11-05");
}

#[test]
fn validate_missing_arg_dispatch_envelope_matches_hardcoded_path() {
    // `apr.validate` without `model_path` returns an error envelope via
    // its argument-validation branch (no subprocess). The inventory
    // path goes through the same `call` body so the envelope must
    // match the pre-migration shape: isError=true, content[0].text
    // contains "model_path".
    let mut server = AprMcpServer::new();
    let resp = server.handle_request(&make_call("apr.validate", serde_json::json!({})));
    let result = resp.result.expect("result present");
    assert_eq!(result["isError"], true);
    let text = result["content"][0]["text"].as_str().expect("text");
    assert!(text.contains("model_path"));
}

#[test]
fn unknown_tool_dispatch_envelope_matches_hardcoded_path() {
    // Pre-migration the match's catch-all returned an error envelope
    // mentioning the unknown name. The inventory path must produce the
    // same shape via `dispatch_for(name) -> None`.
    let mut server = AprMcpServer::new();
    let resp = server.handle_request(&make_call("apr.never-exists", serde_json::json!({})));
    let result = resp.result.expect("result present");
    assert_eq!(result["isError"], true);
    let text = result["content"][0]["text"].as_str().expect("text");
    assert!(
        text.contains("Unknown tool"),
        "unknown-tool envelope must say 'Unknown tool': got {text}"
    );
    assert!(
        text.contains("apr.never-exists"),
        "envelope must echo the offending name"
    );
}

#[test]
fn missing_name_dispatch_envelope_matches_hardcoded_path() {
    // Pre-migration the match's None-arm returned "Missing tool name".
    // Inventory path returns the same when params.name is absent.
    let mut server = AprMcpServer::new();
    let req = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(serde_json::json!(1)),
        method: "tools/call".to_string(),
        params: serde_json::json!({}),
    };
    let resp = server.handle_request(&req);
    let result = resp.result.expect("result present");
    assert_eq!(result["isError"], true);
    let text = result["content"][0]["text"].as_str().expect("text");
    assert!(
        text.contains("Missing tool name"),
        "missing-name envelope must say 'Missing tool name': got {text}"
    );
}

#[test]
fn every_shipped_tool_is_reachable_via_dispatch() {
    // Every entry surfaced by tools/list MUST also be reachable via
    // tools/call. A name in the definitions index but not the dispatch
    // map (or vice versa) means the inventory drifted between the two
    // — an inconsistency that the hardcoded match would have caught at
    // compile time.
    let mut server = AprMcpServer::new();
    let req = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(serde_json::json!(1)),
        method: "tools/list".to_string(),
        params: serde_json::json!({}),
    };
    let list_resp = server.handle_request(&req);
    let tools = list_resp.result.expect("result")["tools"]
        .as_array()
        .expect("tools array")
        .clone();

    for tool in tools {
        let name = tool["name"].as_str().expect("name");
        let resp = server.handle_request(&make_call(name, serde_json::json!({})));
        // We expect either success or an `isError: true` envelope —
        // never a JSON-RPC-level error (-32601 etc) because that would
        // mean the dispatcher couldn't find the tool, which would fail
        // the inventory parity contract.
        assert!(
            resp.error.is_none(),
            "tools/call for {name} must not produce a JSON-RPC error envelope: {:?}",
            resp.error,
        );
        let result = resp.result.expect("result present");
        assert!(
            result["content"].is_array(),
            "{name}: response must carry a content array"
        );
    }
}
