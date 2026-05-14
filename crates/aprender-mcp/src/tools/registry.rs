//! HELIX-IDEA-002 — link-time tool registration via `inventory`.
//!
//! Contract: `contracts/apr-mcp-tool-inventory-v1.yaml`.
//!
//! Each `tools/<tool>.rs` submits one [`McpToolEntry`] to the global
//! inventory using the [`register_mcp_tool!`] macro. The dispatcher
//! ([`AprMcpServer`](crate::server::AprMcpServer)) iterates
//! `inventory::iter::<McpToolEntry>` once at startup and builds:
//!
//! * a `Vec<ToolDefinition>` (replaces the hardcoded vec at
//!   `server.rs:221-233`),
//! * a `BTreeMap<&str, DispatchFn>` (replaces the hardcoded match arms
//!   at `server.rs:461-483`).
//!
//! Adding a new tool is therefore a one-file edit under `tools/`: write
//! the dispatcher, the definition factory, and one
//! `register_mcp_tool!` invocation. No more edits to `server.rs`.
//!
//! `inputSchema` remains contracts-driven (FALSIFY-MCP-008); this
//! module only owns the *registration* path.

use crate::server::NotificationSink;
use crate::types::{ToolCallResult, ToolDefinition};
use std::collections::BTreeMap;
use std::sync::mpsc;

/// Unified dispatch signature shared by every MCP tool. Tools that do
/// not need cancel / sink simply ignore those parameters.
///
/// * `args` — the `params.arguments` JSON value from the MCP request.
/// * `cancel_rx` — an mpsc channel signalled by `notifications/cancelled`.
///   Honoured today only by `apr.run`; other tools accept it for ABI
///   uniformity.
/// * `sink` — `Some` when the client supplied `params._meta.progressToken`
///   AND the tool supports streaming. Honoured today by `apr.finetune`
///   and `apr.run`; other tools accept it for ABI uniformity.
/// * `progress_token` — the JSON value of `params._meta.progressToken`
///   passed back verbatim in each `notifications/progress`.
pub type DispatchFn = fn(
    &serde_json::Value,
    &mpsc::Receiver<()>,
    Option<&NotificationSink>,
    Option<serde_json::Value>,
) -> ToolCallResult;

/// Submitted by every tool module via [`register_mcp_tool!`]. The
/// dispatcher reads these out of [`inventory`] at startup.
#[derive(Debug)]
pub struct McpToolEntry {
    /// MCP tool name advertised in `tools/list` and matched in
    /// `tools/call`. Examples: `"apr.version"`, `"apr.run"`.
    pub name: &'static str,
    /// Returns the full [`ToolDefinition`] (name + description + input
    /// schema). Called once per `tools/list` request.
    pub definition_fn: fn() -> ToolDefinition,
    /// Returns the [`ToolCallResult`] for a `tools/call` request.
    pub dispatch_fn: DispatchFn,
}

inventory::collect!(McpToolEntry);

/// One-time index built from the global inventory at startup. Holds
/// definitions in registration order (so `tools/list` output is
/// deterministic across builds — `inventory::iter` itself orders entries
/// by submission link order, but we sort by name to remove that
/// dependency for tests) and a name → dispatch map for `tools/call`.
#[derive(Debug)]
pub struct ToolIndex {
    definitions: Vec<ToolDefinition>,
    dispatch: BTreeMap<&'static str, DispatchFn>,
}

impl ToolIndex {
    /// Build the index from `inventory::iter::<McpToolEntry>()`.
    ///
    /// FALSIFY-INVENTORY-002: panics with a clear diagnostic if two
    /// entries share the same `name`. The panic fires the first time
    /// any test or production path constructs an `AprMcpServer`, so a
    /// duplicate-name regression cannot escape CI.
    #[must_use]
    pub fn from_inventory() -> Self {
        let mut by_name: BTreeMap<&'static str, &'static McpToolEntry> = BTreeMap::new();
        for entry in inventory::iter::<McpToolEntry> {
            if let Some(prior) = by_name.insert(entry.name, entry) {
                // Restore the prior entry so the panic message lists both.
                by_name.insert(prior.name, prior);
                panic!(
                    "FALSIFY-INVENTORY-002: duplicate MCP tool name {:?} registered twice in the \
                     inventory. Two `register_mcp_tool!` invocations advertise the same name; \
                     pick one. Existing: {:p}, duplicate: {:p}.",
                    entry.name, prior, entry,
                );
            }
        }

        // Deterministic order — sorted by name so test goldens are stable
        // across linker reorderings of `inventory::submit!`.
        let mut definitions: Vec<ToolDefinition> =
            by_name.values().map(|e| (e.definition_fn)()).collect();
        definitions.sort_by(|a, b| a.name.cmp(&b.name));

        let dispatch: BTreeMap<&'static str, DispatchFn> = by_name
            .iter()
            .map(|(name, entry)| (*name, entry.dispatch_fn))
            .collect();

        Self {
            definitions,
            dispatch,
        }
    }

    /// All tool definitions advertised by `tools/list`, sorted by name.
    #[must_use]
    pub fn definitions(&self) -> &[ToolDefinition] {
        &self.definitions
    }

    /// Look up the dispatch function for a `tools/call` request.
    /// Returns `None` if no tool with that name is registered — the
    /// caller is responsible for emitting the appropriate `isError`
    /// envelope.
    #[must_use]
    pub fn dispatch_for(&self, name: &str) -> Option<&DispatchFn> {
        self.dispatch.get(name)
    }

    /// Names of all registered tools, alphabetically sorted. Used by
    /// FALSIFY-INVENTORY-001 to assert the migrated set matches the
    /// pre-migration golden list.
    #[must_use]
    pub fn names(&self) -> Vec<&'static str> {
        self.dispatch.keys().copied().collect()
    }
}

/// Submit one [`McpToolEntry`] to the global inventory.
///
/// Usage (one invocation per `tools/<tool>.rs`):
///
/// ```ignore
/// register_mcp_tool!(
///     name: "apr.foo",
///     definition: foo_tool_definition,
///     dispatch: dispatch,
/// );
/// ```
///
/// The `dispatch` argument MUST point at a function with signature
/// [`DispatchFn`] — typically a thin shim that calls the tool's
/// existing `call` / `call_with_sink`.
#[macro_export]
macro_rules! register_mcp_tool {
    (name: $name:expr, definition: $def:path, dispatch: $dispatch:path $(,)?) => {
        ::inventory::submit! {
            $crate::tools::registry::McpToolEntry {
                name: $name,
                definition_fn: $def,
                dispatch_fn: $dispatch,
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity: building the index from the live inventory does not panic
    /// and yields the 9 Phase-1 tool names in alphabetical order.
    #[test]
    fn live_inventory_yields_phase_one_tool_set() {
        let index = ToolIndex::from_inventory();
        let names = index.names();
        let expected = [
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
        assert_eq!(names, expected);
        assert_eq!(index.definitions().len(), 9);
    }

    #[test]
    fn dispatch_for_known_tool_returns_some() {
        let index = ToolIndex::from_inventory();
        assert!(index.dispatch_for("apr.version").is_some());
        assert!(index.dispatch_for("apr.qa").is_some());
    }

    #[test]
    fn dispatch_for_unknown_tool_returns_none() {
        let index = ToolIndex::from_inventory();
        assert!(index.dispatch_for("apr.never").is_none());
    }
}
