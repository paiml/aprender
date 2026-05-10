//! FALSIFY-INVENTORY-001 — the inventory-built tool set is exactly the
//! pre-migration Phase-1 9-tool set, in deterministic alphabetical
//! order. No tool dropped, no duplicate, no name drift.
//!
//! Contract: `contracts/apr-mcp-tool-inventory-v1.yaml`.
//!
//! Discharge strategy: fire `tools/list` against a real `AprMcpServer`,
//! collect the tool names, and assert byte-equality with a frozen
//! golden array. Two-way drift detection: a name typo or missing tool
//! breaks the equality; a stray new tool also breaks it (forcing the
//! contract's golden set to be updated in lockstep).

#![allow(clippy::unwrap_used, clippy::disallowed_methods)]

use aprender_mcp::{AprMcpServer, JsonRpcRequest};

const GOLDEN_TOOL_NAMES: &[&str] = &[
    "apr.bench",
    "apr.finetune",
    "apr.qa",
    "apr.run",
    "apr.serve",
    "apr.tensors",
    "apr.trace",
    "apr.validate",
    "apr.version",
];

#[test]
fn inventory_yields_same_tool_set_as_hardcoded_list() {
    let mut server = AprMcpServer::new();
    let req = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(serde_json::json!(1)),
        method: "tools/list".to_string(),
        params: serde_json::json!({}),
    };
    let resp = server.handle_request(&req);
    let result = resp.result.expect("tools/list result");
    let tools = result["tools"].as_array().expect("tools is an array");

    let mut names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
    names.sort();

    let mut expected: Vec<&str> = GOLDEN_TOOL_NAMES.to_vec();
    expected.sort();

    assert_eq!(
        names, expected,
        "inventory-derived tool set must equal the golden Phase-1 9-tool set; \
         migration must not drop or add tools without updating both this \
         test and the contract."
    );
}

#[test]
fn tool_definitions_method_yields_same_set() {
    // Same gate but via `tool_definitions()` directly (cheaper than the
    // JSON-RPC path; same source of truth).
    let server = AprMcpServer::new();
    let mut names: Vec<String> = server
        .tool_definitions()
        .iter()
        .map(|t| t.name.clone())
        .collect();
    names.sort();

    let mut expected: Vec<String> = GOLDEN_TOOL_NAMES.iter().map(|s| (*s).to_string()).collect();
    expected.sort();

    assert_eq!(names, expected);
}

#[test]
fn every_tool_definition_carries_input_schema() {
    let server = AprMcpServer::new();
    for def in &server.tool_definitions() {
        assert_eq!(
            def.input_schema.schema_type, "object",
            "{}: every MCP tool must declare an `inputSchema.type=object`",
            def.name
        );
    }
}
