//! FALSIFY-MCP-002 (strict slice) — every registered tool's `inputSchema` must
//! compile as a valid JSON Schema Draft 7 document.
//!
//! The base FALSIFY-MCP-002 in `falsify_m1.rs` only asserts that each schema
//! has `type: "object"` — necessary but not sufficient. The MCP spec requires
//! that the schema validates against JSONSchema Draft 7. Compiling with
//! `jsonschema::validator_for` performs meta-schema validation as a side
//! effect, so a successful compile proves the schema is well-formed.

#![allow(clippy::disallowed_methods)] // serde_json::json! / to_value expand to unwrap()

use aprender_mcp::AprMcpServer;

/// FALSIFY-MCP-002 (strict): every tool's `inputSchema` is a valid JSON Schema.
/// Failure mode caught by this test: a tool definition that names a
/// nonexistent type, uses a malformed `properties` map, or otherwise emits
/// JSON that no spec-compliant client would accept.
#[test]
fn every_tool_input_schema_is_valid_jsonschema_draft_7() {
    let server = AprMcpServer::new();
    let definitions = server.tool_definitions();
    assert!(
        !definitions.is_empty(),
        "expected at least one tool registered"
    );

    for def in &definitions {
        let schema_value = serde_json::to_value(&def.input_schema)
            .unwrap_or_else(|e| panic!("serialize {}: {e}", def.name));

        // Compiling is the validation: jsonschema rejects malformed schemas
        // here. We do not validate any instance — just the schema itself.
        jsonschema::validator_for(&schema_value).unwrap_or_else(|e| {
            panic!(
                "tool {} has an invalid JSON Schema inputSchema: {e}\nschema: {}",
                def.name,
                serde_json::to_string_pretty(&schema_value).unwrap_or_default()
            )
        });
    }
}

/// Spot-check: a known-malformed schema must be rejected by the same
/// validator-compile path. This is belt-and-braces: if some future
/// `jsonschema` upgrade silently accepts garbage, the strict test above would
/// give a false pass — this guard fails first.
#[test]
fn invalid_schema_is_rejected_by_validator_for() {
    let bogus = serde_json::json!({
        "type": "object",
        "properties": "this-should-be-an-object-not-a-string"
    });

    assert!(
        jsonschema::validator_for(&bogus).is_err(),
        "validator_for should reject schema with non-object `properties`"
    );
}
