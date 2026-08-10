//! Model Context Protocol (MCP) server for aprender.
//!
//! Exposes the `apr` CLI as MCP tools for Claude Code, Cursor, Cline, and
//! other MCP clients over JSON-RPC 2.0 stdio transport.
//!
//! Spec: `docs/specifications/apr-mcp-server-spec.md`.
//! Protocol: MCP v2024-11-05 (<https://spec.modelcontextprotocol.io>).
//!
//! # Example
//!
//! ```no_run
//! # #[cfg(feature = "native")]
//! # fn main() -> anyhow::Result<()> {
//! let mut server = aprender_mcp::AprMcpServer::new();
//! server.run_stdio()?;
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "native"))]
//! # fn main() {}
//! ```
//!
//! # Scope (M1–M3 shipped)
//!
//! - M1: `initialize` + `tools/list` + the `apr.version` stub.
//! - M2: 7 subprocess wrappers (`apr.validate`, `apr.tensors`, `apr.bench`,
//!   `apr.qa`, `apr.trace`, `apr.run`, `apr.serve`) + FALSIFY-MCP-005/-007
//!   dispatcher gates + Draft-7 schema meta-validation.
//! - M3: `apr.finetune` (8th Phase-1 tool), `notifications/cancelled` →
//!   SIGTERM→SIGKILL for `apr.run`, `build.rs` codegen from
//!   `contracts/apr-mcp-tool-schemas-v1.yaml` for both `inputSchema`
//!   (`schemas::APR_<TOOL>_SCHEMA`) and tool-level `description`
//!   (`schemas::APR_<TOOL>_DESCRIPTION`, PMAT-514), and opt-in per-line
//!   `notifications/progress` for `apr.finetune` via
//!   `params._meta.progressToken` (FALSIFY-MCP-PROGRESS-001).
//!
//! M4 (in progress) promotes FALSIFY-MCP-003/-004 from surface tests to
//! real-model end-to-end; M5 ports the dispatcher to `pmcp = "2.3"` and
//! extends cancellation to `apr.serve`.

pub mod apr_bin;
pub mod server;
pub mod tools;
pub mod types;

/// FALSIFY-MCP-008: per-tool JSON Schema + description constants codegen'd
/// at build time from `contracts/apr-mcp-tool-schemas-v1.yaml` by `build.rs`.
///
/// For each tool, `build.rs` emits two constants:
/// * `APR_<TOOL>_SCHEMA: &str` — a JSON string whose parsed body is the
///   tool's MCP `inputSchema`. Consume via `serde_json::from_str`.
/// * `APR_<TOOL>_DESCRIPTION: &str` — the tool-level description.
///   Consume via `.to_string()`.
///
/// Tools MUST source both fields from this module — hand-coding either in
/// the tool's source file is caught by `tests/falsify_mcp_008.rs`
/// (PMAT-514, 2026-04-18).
pub mod schemas {
    include!(concat!(env!("OUT_DIR"), "/schemas.rs"));
}

pub use server::{AprMcpServer, NotificationSink};
pub use types::{
    ContentBlock, InputSchema, JsonRpcError, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse,
    PropertySchema, ServerCapabilities, ToolCallResult, ToolDefinition, ToolsCapability,
};

/// MCP protocol version this server implements.
pub const PROTOCOL_VERSION: &str = "2024-11-05";

/// Server identity reported in `initialize` response.
pub const SERVER_NAME: &str = "aprender-mcp";
