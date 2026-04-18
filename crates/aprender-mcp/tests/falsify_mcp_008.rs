//! FALSIFY-MCP-008: tools/list output for each migrated tool must be
//! byte-identical to the corresponding entry in
//! `contracts/apr-mcp-tool-schemas-v1.yaml` — covering both the
//! `inputSchema` object and the tool-level `description` string.
//!
//! This gate proves the YAML contract IS the single source of truth. The
//! harness:
//!   1. Reads `contracts/apr-mcp-tool-schemas-v1.yaml` at test runtime.
//!   2. Finds the entry for each migrated tool.
//!   3. Reconstructs the expected JSON Schema from the contract.
//!   4. Fetches the live schema via the server's `tools/list` response.
//!   5. Asserts `serde_json::Value` equality (order-independent object match).
//!   6. Separately asserts `ToolDefinition.description ==
//!      tools[*].description` as a raw string compare (PMAT-514).
//!
//! **Scope (M3 shipped, extended by PMAT-514 on 2026-04-18):** all 9
//! registered tools (`apr.version` + 8 Phase-1 wrappers) are wired through
//! the build.rs codegen path (`crate::schemas::APR_*_SCHEMA`). `MIGRATED_TOOLS`
//! below is sourced from `schemas::TOOL_NAMES` (itself derived from the
//! YAML at build time), so adding a new tool to the YAML automatically
//! pulls it into this harness with zero test edits. Adding a description
//! field to the new tool is likewise auto-covered by
//! `tool_descriptions_match_yaml_contract`.

#![allow(clippy::disallowed_methods)] // test-only serde_json::json! / to_value expansions

use aprender_mcp::AprMcpServer;
use serde_json::{Map, Value};
use std::collections::BTreeSet;
use std::path::PathBuf;

/// Tools currently routed through the build.rs codegen (`crate::schemas::*`).
///
/// Sourced from the build-time `schemas::TOOL_NAMES` constant so that every
/// tool in `contracts/apr-mcp-tool-schemas-v1.yaml` is covered by this gate
/// automatically. When a new tool is added to the YAML, it is picked up here
/// on the next build without editing this file.
fn migrated_tools() -> Vec<&'static str> {
    aprender_mcp::schemas::TOOL_NAMES.to_vec()
}

/// Per-tool codegen constants for the narrower parse-and-match test below.
/// Must be kept in sync with `schemas::TOOL_NAMES` (guarded by
/// `codegen_constants_cover_every_tool_name`).
const CODEGEN_CONSTANTS: &[(&str, &str)] = &[
    ("apr.version", aprender_mcp::schemas::APR_VERSION_SCHEMA),
    ("apr.validate", aprender_mcp::schemas::APR_VALIDATE_SCHEMA),
    ("apr.tensors", aprender_mcp::schemas::APR_TENSORS_SCHEMA),
    ("apr.bench", aprender_mcp::schemas::APR_BENCH_SCHEMA),
    ("apr.qa", aprender_mcp::schemas::APR_QA_SCHEMA),
    ("apr.trace", aprender_mcp::schemas::APR_TRACE_SCHEMA),
    ("apr.run", aprender_mcp::schemas::APR_RUN_SCHEMA),
    ("apr.serve", aprender_mcp::schemas::APR_SERVE_SCHEMA),
    ("apr.finetune", aprender_mcp::schemas::APR_FINETUNE_SCHEMA),
];

fn contract_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("contracts")
        .join("apr-mcp-tool-schemas-v1.yaml")
}

/// Parse the YAML contract and return the `tools` list as a `serde_yaml::Value`.
fn load_contract_tools() -> Vec<serde_yaml::Value> {
    let path = contract_path();
    let yaml =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let root: serde_yaml::Value = serde_yaml::from_str(&yaml)
        .unwrap_or_else(|e| panic!("parse {} as YAML: {e}", path.display()));
    root.get("tools")
        .and_then(|v| v.as_sequence())
        .cloned()
        .unwrap_or_else(|| panic!("{} has no `tools:` sequence", path.display()))
}

/// Reconstruct the expected JSON Schema object from one contract entry.
///
/// Shape (matches what the live `InputSchema` serializes to, modulo key order):
/// ```json
/// {"type":"object","properties":{...},"required":[...]}
/// ```
/// Empty `properties` and empty `required` are omitted (InputSchema marks
/// both with `skip_serializing_if`).
fn expected_schema_from_yaml(tool: &serde_yaml::Value) -> Value {
    let mut schema = Map::new();
    schema.insert("type".to_string(), Value::String("object".to_string()));

    if let Some(args) = tool.get("args").and_then(|v| v.as_sequence()) {
        if !args.is_empty() {
            let mut props = Map::new();
            for arg in args {
                let name = arg
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or_else(|| panic!("arg missing `name`: {arg:?}"));
                let ty = arg
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or_else(|| panic!("arg {name} missing `type`"));
                let desc = arg
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or_else(|| panic!("arg {name} missing `description`"));
                let mut prop = Map::new();
                prop.insert("type".to_string(), Value::String(ty.to_string()));
                prop.insert("description".to_string(), Value::String(desc.to_string()));
                props.insert(name.to_string(), Value::Object(prop));
            }
            schema.insert("properties".to_string(), Value::Object(props));
        }
    }

    if let Some(req) = tool.get("required").and_then(|v| v.as_sequence()) {
        if !req.is_empty() {
            let list: Vec<Value> = req
                .iter()
                .map(|v| {
                    v.as_str()
                        .map(|s| Value::String(s.to_string()))
                        .unwrap_or_else(|| panic!("`required` entry not a string: {v:?}"))
                })
                .collect();
            schema.insert("required".to_string(), Value::Array(list));
        }
    }

    Value::Object(schema)
}

/// Pull the live `inputSchema` for a tool out of the server's `tools/list` result.
fn live_schema(server: &AprMcpServer, tool_name: &str) -> Value {
    let defs = server.tool_definitions();
    let def = defs
        .iter()
        .find(|d| d.name == tool_name)
        .unwrap_or_else(|| panic!("tool {tool_name} not registered on the server"));
    serde_json::to_value(&def.input_schema).expect("serialize inputSchema")
}

/// FALSIFY-MCP-008: for each migrated tool, the live `inputSchema` is
/// structurally identical to the schema derived from the YAML contract.
///
/// `serde_json::Value` equality ignores insignificant whitespace and
/// property order — both sides are canonicalized by serde into a Map and
/// compared key-by-key.
#[test]
fn migrated_tools_match_yaml_contract_byte_for_byte() {
    let server = AprMcpServer::new();
    let contract_tools = load_contract_tools();
    let tools = migrated_tools();

    assert!(
        !tools.is_empty(),
        "FALSIFY-MCP-008: at least one tool must be wired through the codegen path"
    );

    for tool_name in &tools {
        let entry = contract_tools
            .iter()
            .find(|t| t.get("name").and_then(|v| v.as_str()) == Some(*tool_name))
            .unwrap_or_else(|| panic!("FALSIFY-MCP-008: contract has no entry for {tool_name}"));

        let expected = expected_schema_from_yaml(entry);
        let actual = live_schema(&server, tool_name);

        assert_eq!(
            actual,
            expected,
            "FALSIFY-MCP-008 FAIL: live schema for {tool_name} drifted from YAML contract\n\
             expected (from contracts/apr-mcp-tool-schemas-v1.yaml):\n{}\n\
             actual (from tools/list):\n{}",
            serde_json::to_string_pretty(&expected).unwrap_or_default(),
            serde_json::to_string_pretty(&actual).unwrap_or_default(),
        );
    }
}

/// Each codegen constant must itself parse as valid JSON and match the YAML
/// — this is the narrower assertion that the build.rs output is usable for
/// every tool, independent of the live server wiring above.
#[test]
fn codegen_constants_parse_and_match_yaml_for_every_tool() {
    let contract_tools = load_contract_tools();

    for (tool_name, schema_json) in CODEGEN_CONSTANTS {
        let parsed: Value = serde_json::from_str(schema_json)
            .unwrap_or_else(|e| panic!("codegen constant for {tool_name} must be valid JSON: {e}"));
        let entry = contract_tools
            .iter()
            .find(|t| t.get("name").and_then(|v| v.as_str()) == Some(*tool_name))
            .unwrap_or_else(|| panic!("contract has no entry for {tool_name}"));
        let expected = expected_schema_from_yaml(entry);

        assert_eq!(
            parsed, expected,
            "codegen constant for {tool_name} diverged from YAML contract"
        );
    }
}

/// Guardrail: every name in `schemas::TOOL_NAMES` must have a matching entry
/// in `CODEGEN_CONSTANTS`. Catches the case where a new tool is added to the
/// YAML without being added to the narrower per-constant test above.
#[test]
fn codegen_constants_cover_every_tool_name() {
    let covered: BTreeSet<&str> = CODEGEN_CONSTANTS.iter().map(|(n, _)| *n).collect();
    for name in aprender_mcp::schemas::TOOL_NAMES {
        assert!(
            covered.contains(*name),
            "CODEGEN_CONSTANTS is missing an entry for {name} (present in schemas::TOOL_NAMES)"
        );
    }
}

/// The exported `TOOL_NAMES` list must cover every `tools[*].name` in the
/// contract in declaration order — follow-up PRs migrating each tool depend
/// on this invariant to drive the harness without a hard-coded list.
#[test]
fn tool_names_constant_mirrors_yaml() {
    let contract_tools = load_contract_tools();
    let yaml_names: Vec<String> = contract_tools
        .iter()
        .filter_map(|t| t.get("name").and_then(|v| v.as_str()).map(String::from))
        .collect();
    let gen_names: Vec<String> = aprender_mcp::schemas::TOOL_NAMES
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    assert_eq!(yaml_names, gen_names, "TOOL_NAMES drifted from YAML order");
}

/// Guardrail: every migrated tool must also be a contract entry. Catches a
/// TOOL_NAMES/YAML desync before it becomes a silent false-pass.
#[test]
fn migrated_tools_are_all_in_contract() {
    let contract_tools = load_contract_tools();
    let contract_names: BTreeSet<String> = contract_tools
        .iter()
        .filter_map(|t| t.get("name").and_then(|v| v.as_str()).map(String::from))
        .collect();
    for name in migrated_tools() {
        assert!(
            contract_names.contains(name),
            "migrated tool {name} is not in contracts/apr-mcp-tool-schemas-v1.yaml"
        );
    }
}

/// FALSIFY-MCP-008 (extended): live `ToolDefinition.description` must be
/// byte-identical to `tools[*].description` in the YAML contract.
///
/// The base gate above compares only `inputSchema` — letting tool-level
/// descriptions silently drift out of the contract (observed twice in
/// 2026-04-18: apr.serve commit 715781df5 and apr.run commit 91a613968). This
/// test closes that class of bug by making the contract's own assertion —
/// "each tool's `description` matches `tools[*].description` byte-for-byte"
/// (apr-mcp-tool-schemas-v1.yaml line 282) — actually enforced at test time.
#[test]
fn tool_descriptions_match_yaml_contract() {
    let server = AprMcpServer::new();
    let contract_tools = load_contract_tools();
    let defs = server.tool_definitions();

    for tool_name in migrated_tools() {
        let entry = contract_tools
            .iter()
            .find(|t| t.get("name").and_then(|v| v.as_str()) == Some(tool_name))
            .unwrap_or_else(|| panic!("contract has no entry for {tool_name}"));
        let expected = entry
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| panic!("contract entry for {tool_name} missing `description`"));
        let def = defs
            .iter()
            .find(|d| d.name == tool_name)
            .unwrap_or_else(|| panic!("tool {tool_name} not registered on the server"));

        assert_eq!(
            def.description, expected,
            "FALSIFY-MCP-008 FAIL: description for {tool_name} drifted from YAML contract\n\
             expected (contracts/apr-mcp-tool-schemas-v1.yaml):\n{expected}\n\
             actual (ToolDefinition.description):\n{}",
            def.description,
        );
    }
}
