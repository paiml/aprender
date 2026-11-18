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
#[cfg(feature = "apr-tools")]
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

#[cfg(feature = "apr-tools")]
inventory::collect!(McpToolEntry);

/// One-time index built from the global inventory at startup. Holds
/// definitions in registration order (so `tools/list` output is
/// deterministic across builds — `inventory::iter` itself orders entries
/// by submission link order, but we sort by name to remove that
/// dependency for tests) and a name → dispatch map for `tools/call`.
#[derive(Debug)]
pub struct ToolIndex {
    definitions: Vec<ToolDefinition>,
    /// Keyed by an OWNED name (PMAT-3954): a caller-supplied tool's name is a `String`
    /// in its [`ToolDefinition`], never `'static`.
    dispatch: BTreeMap<String, DispatchFn>,
}

impl ToolIndex {
    /// Build an index from a caller-supplied tool set (PMAT-3954) — a downstream server
    /// serving ITS OWN tools on this crate, with no `inventory` and no apr tool linked in.
    ///
    /// Definitions are sorted by name, as [`Self::from_inventory`] sorts them.
    ///
    /// # Errors
    ///
    /// Two entries with one name are refused, naming the tool — the same rule as
    /// FALSIFY-INVENTORY-002, returned rather than panicked because the caller built it.
    pub fn from_entries(entries: Vec<(ToolDefinition, DispatchFn)>) -> Result<Self, String> {
        let mut definitions = Vec::with_capacity(entries.len());
        let mut dispatch: BTreeMap<String, DispatchFn> = BTreeMap::new();
        for (definition, dispatch_fn) in entries {
            if dispatch
                .insert(definition.name.clone(), dispatch_fn)
                .is_some()
            {
                return Err(format!(
                    "duplicate MCP tool name {:?}: two entries advertise the same name",
                    definition.name
                ));
            }
            definitions.push(definition);
        }
        definitions.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(Self {
            definitions,
            dispatch,
        })
    }

    /// Build the index from `inventory::iter::<McpToolEntry>()`.
    ///
    /// FALSIFY-INVENTORY-002: panics with a clear diagnostic if two
    /// entries share the same `name`. The panic fires the first time
    /// any test or production path constructs an `AprMcpServer`, so a
    /// duplicate-name regression cannot escape CI.
    #[cfg(feature = "apr-tools")]
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

        let dispatch: BTreeMap<String, DispatchFn> = by_name
            .iter()
            .map(|(name, entry)| ((*name).to_string(), entry.dispatch_fn))
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
    pub fn names(&self) -> Vec<&str> {
        self.dispatch.keys().map(String::as_str).collect()
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
#[cfg(feature = "apr-tools")]
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

#[cfg(all(test, feature = "apr-tools"))] // from_inventory exists only with apr's tools
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

    fn fixture_tool(
        _args: &serde_json::Value,
        _cancel: &mpsc::Receiver<()>,
        _sink: Option<&NotificationSink>,
        _token: Option<serde_json::Value>,
    ) -> ToolCallResult {
        ToolCallResult::success("fixture".to_string())
    }

    fn fixture_def(name: &str) -> ToolDefinition {
        ToolDefinition {
            name: name.to_string(),
            description: "fixture".to_string(),
            input_schema: crate::types::InputSchema {
                schema_type: "object".to_string(),
                properties: Default::default(),
                required: vec![],
            },
        }
    }

    /// FALSIFY-INVENTORY-002: two entries with one name are refused, and the
    /// diagnostic names the offending tool.
    #[test]
    fn from_entries_refuses_duplicate_tool_name() {
        let result = ToolIndex::from_entries(vec![
            (fixture_def("fx.dup"), fixture_tool as DispatchFn),
            (fixture_def("fx.dup"), fixture_tool as DispatchFn),
        ]);
        match result {
            Ok(_) => panic!("a duplicate tool name must be refused"),
            Err(msg) => assert!(msg.contains("fx.dup"), "diagnostic names the tool: {msg}"),
        }
        assert!(
            ToolIndex::from_entries(vec![
                (fixture_def("fx.a"), fixture_tool as DispatchFn),
                (fixture_def("fx.b"), fixture_tool as DispatchFn),
            ])
            .is_ok(),
            "distinct names are accepted"
        );
    }

    /// FALSIFY-MCP-008 (lib-level): the live tool set and every live tool
    /// description equal what `contracts/apr-mcp-tool-schemas-v1.yaml` declares.
    #[test]
    fn live_tools_match_schema_contract_names_and_descriptions() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../contracts/apr-mcp-tool-schemas-v1.yaml");
        let text = std::fs::read_to_string(&path).expect("schema contract readable");
        let mut declared: Vec<(String, String)> = Vec::new();
        let mut current: Option<String> = None;
        for line in text.lines() {
            if let Some(name) = line.strip_prefix("  - name: apr.") {
                current = Some(format!("apr.{}", name.trim()));
            } else if let (Some(name), Some(desc)) =
                (current.as_ref(), line.strip_prefix("    description: \""))
            {
                let desc = desc.strip_suffix('"').expect("description is quoted");
                declared.push((name.clone(), desc.to_string()));
                current = None;
            }
        }
        declared.sort();
        let index = ToolIndex::from_inventory();
        let mut live: Vec<(String, String)> = index
            .definitions()
            .iter()
            .map(|d| (d.name.clone(), d.description.clone()))
            .collect();
        live.sort();
        assert_eq!(live.len(), declared.len(), "same tool count");
        for ((ln, ld), (dn, dd)) in live.iter().zip(declared.iter()) {
            assert_eq!(ln, dn, "tool names match the contract");
            assert_eq!(
                ld, dd,
                "{ln}: description matches the contract byte-for-byte"
            );
        }
    }
}
