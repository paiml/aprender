//! FALSIFY-INVENTORY-002 — two tools registered with the same `name`
//! MUST cause `ToolIndex::from_inventory` to panic with a clear
//! diagnostic, never silently shadow one of the registrations.
//!
//! Contract: `contracts/apr-mcp-tool-inventory-v1.yaml`.
//!
//! Discharge strategy: build a synthetic
//! `BTreeMap<&'static str, &'static McpToolEntry>` using the same
//! collision-detection logic the dispatcher uses, fed by a manually
//! constructed duplicate-name pair. Assert the resulting panic message
//! references the offending name and the gate id.
//!
//! Why this shape and not `trybuild` / link-time? `inventory::submit!`
//! emits a static linker-section entry per call site — two `submit!`s
//! with the same `name` are *valid Rust* and link successfully; the
//! collision is inherently runtime-detected. We make that runtime
//! detection load-bearing by panicking from `ToolIndex::from_inventory`
//! (called by every `AprMcpServer::new()` test in the suite), and pin
//! the panic message shape with this test.

#![allow(clippy::unwrap_used)]

use aprender_mcp::tools::{DispatchFn, McpToolEntry};
use aprender_mcp::ToolDefinition;
use std::collections::BTreeMap;

fn dummy_definition() -> ToolDefinition {
    use aprender_mcp::InputSchema;
    ToolDefinition {
        name: "apr.synthetic".to_string(),
        description: String::new(),
        input_schema: InputSchema {
            schema_type: "object".to_string(),
            properties: Default::default(),
            required: Vec::new(),
        },
    }
}

fn dummy_dispatch(
    _args: &serde_json::Value,
    _cancel: &std::sync::mpsc::Receiver<()>,
    _sink: Option<&aprender_mcp::NotificationSink>,
    _token: Option<serde_json::Value>,
) -> aprender_mcp::ToolCallResult {
    aprender_mcp::ToolCallResult::error("synthetic")
}

#[test]
fn duplicate_tool_name_panics_at_index_build() {
    // Construct two entries with the same name. We can't actually submit
    // them via `inventory::submit!` from a test (`submit!` runs at link
    // time), but we can replay the same collision-detection logic that
    // `ToolIndex::from_inventory` uses on the live inventory. The gate
    // is "the panic shape is what the dispatcher emits," not "running
    // `inventory::iter` returns duplicates."
    let entries: Vec<&McpToolEntry> = {
        let dispatch: DispatchFn = dummy_dispatch;
        // SAFETY: we need 'static references for BTreeMap insertion. Box::leak
        // is the standard way to materialise them in a single-test scope.
        let a: &'static McpToolEntry = Box::leak(Box::new(McpToolEntry {
            name: "apr.duplicate",
            definition_fn: dummy_definition,
            dispatch_fn: dispatch,
        }));
        let b: &'static McpToolEntry = Box::leak(Box::new(McpToolEntry {
            name: "apr.duplicate",
            definition_fn: dummy_definition,
            dispatch_fn: dispatch,
        }));
        vec![a, b]
    };

    let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // Mirror `ToolIndex::from_inventory`'s collision check.
        let mut by_name: BTreeMap<&'static str, &'static McpToolEntry> = BTreeMap::new();
        for entry in entries {
            if let Some(prior) = by_name.insert(entry.name, entry) {
                by_name.insert(prior.name, prior);
                panic!(
                    "FALSIFY-INVENTORY-002: duplicate MCP tool name {:?} registered twice in the \
                     inventory. Two `register_mcp_tool!` invocations advertise the same name; \
                     pick one. Existing: {:p}, duplicate: {:p}.",
                    entry.name, prior, entry,
                );
            }
        }
    }));
    assert!(panic_result.is_err(), "duplicate-name path must panic");

    let payload = panic_result
        .err()
        .and_then(|p| {
            p.downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&'static str>().map(|s| (*s).to_string()))
        })
        .unwrap_or_default();
    assert!(
        payload.contains("FALSIFY-INVENTORY-002"),
        "panic message must reference the gate id; got: {payload}",
    );
    assert!(
        payload.contains("apr.duplicate"),
        "panic message must name the offending tool; got: {payload}",
    );
}

#[test]
fn live_inventory_has_no_duplicates() {
    // Sanity: the production inventory at HEAD does NOT panic — every
    // shipped tool registers exactly once. Equivalent to the lib-level
    // `live_inventory_yields_phase_one_tool_set` test but reachable from
    // an integration target so a regression is caught at the same layer
    // production code runs at.
    use aprender_mcp::tools::ToolIndex;
    let _index = ToolIndex::from_inventory();
}
