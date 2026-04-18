//! FALSIFY-MCP-008: tools/list output for each migrated tool must be
//! byte-identical to the schema in `contracts/apr-mcp-tool-schemas-v1.yaml`.
//!
//! This gate proves the YAML contract IS the single source of truth. The
//! harness:
//!   1. Reads `contracts/apr-mcp-tool-schemas-v1.yaml` at test runtime.
//!   2. Finds the entry for each migrated tool.
//!   3. Reconstructs the expected JSON Schema from the contract.
//!   4. Fetches the live schema via the server's `tools/list` response.
//!   5. Asserts `serde_json::Value` equality (order-independent object match).
//!
//! **Scope of this PR:** only `apr.version` is migrated to the codegen path.
//! The other 5 tools in the contract still ship hand-written schemas;
//! follow-up PRs migrate them tool-by-tool for reviewability. When a new
//! tool is added to `MIGRATED_TOOLS` below, this harness automatically
//! asserts byte-identity for it — no test edits required.

#![allow(clippy::disallowed_methods)] // test-only serde_json::json! / to_value expansions

use aprender_mcp::AprMcpServer;
use serde_json::{Map, Value};
use std::collections::BTreeSet;
use std::path::PathBuf;

/// Tools currently routed through the build.rs codegen (`crate::schemas::*`).
/// Extend as follow-up PRs migrate the rest of the registry.
const MIGRATED_TOOLS: &[&str] = &["apr.version"];

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

    assert!(
        !MIGRATED_TOOLS.is_empty(),
        "FALSIFY-MCP-008: at least one tool must be wired through the codegen path"
    );

    for tool_name in MIGRATED_TOOLS {
        let entry = contract_tools
            .iter()
            .find(|t| t.get("name").and_then(|v| v.as_str()) == Some(tool_name))
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

/// The codegen constant must itself parse as valid JSON and match the YAML
/// — this is the narrower assertion that the build.rs output is usable.
#[test]
fn codegen_constant_parses_and_matches_yaml_for_apr_version() {
    let parsed: Value = serde_json::from_str(aprender_mcp::schemas::APR_VERSION_SCHEMA)
        .expect("APR_VERSION_SCHEMA must be valid JSON");

    let contract_tools = load_contract_tools();
    let entry = contract_tools
        .iter()
        .find(|t| t.get("name").and_then(|v| v.as_str()) == Some("apr.version"))
        .expect("contract has apr.version entry");
    let expected = expected_schema_from_yaml(entry);

    assert_eq!(
        parsed, expected,
        "codegen constant APR_VERSION_SCHEMA diverged from YAML contract"
    );
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
/// typo in `MIGRATED_TOOLS` before it becomes a silent false-pass.
#[test]
fn migrated_tools_are_all_in_contract() {
    let contract_tools = load_contract_tools();
    let contract_names: BTreeSet<String> = contract_tools
        .iter()
        .filter_map(|t| t.get("name").and_then(|v| v.as_str()).map(String::from))
        .collect();
    for name in MIGRATED_TOOLS {
        assert!(
            contract_names.contains(*name),
            "MIGRATED_TOOLS entry {name} is not in contracts/apr-mcp-tool-schemas-v1.yaml"
        );
    }
}
