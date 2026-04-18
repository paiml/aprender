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
//! # M1 Scope
//!
//! M1 (skeleton) ships `initialize` + `tools/list` + 1 stub tool (`apr.version`).
//! M2 adds the 8 Phase-1 tools. FALSIFY-MCP-001/-002 gate M1.

pub mod server;
pub mod tools;
pub mod types;

/// FALSIFY-MCP-008: per-tool JSON Schema constants codegen'd at build time
/// from `contracts/apr-mcp-tool-schemas-v1.yaml` by `build.rs`.
///
/// Each `<TOOL>_SCHEMA` constant is a JSON string whose parsed body is the
/// tool's MCP `inputSchema`. Consume via `serde_json::from_str`.
pub mod schemas {
    include!(concat!(env!("OUT_DIR"), "/schemas.rs"));
}

pub use server::AprMcpServer;
pub use types::{
    ContentBlock, InputSchema, JsonRpcError, JsonRpcRequest, JsonRpcResponse, PropertySchema,
    ServerCapabilities, ToolCallResult, ToolDefinition, ToolsCapability,
};

/// MCP protocol version this server implements.
pub const PROTOCOL_VERSION: &str = "2024-11-05";

/// Server identity reported in `initialize` response.
pub const SERVER_NAME: &str = "aprender-mcp";
