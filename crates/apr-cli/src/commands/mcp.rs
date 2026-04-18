//! `apr mcp` — start the aprender MCP server over stdio.
//!
//! Wraps `aprender_mcp::AprMcpServer::run_stdio` so that Claude Code, Cursor,
//! Cline, and other MCP clients can configure `apr` as a tool server via
//! `.mcp.json`.

use crate::error::CliError;

/// Start the MCP server on stdio (blocking until stdin closes).
pub fn run() -> Result<(), CliError> {
    let mut server = aprender_mcp::AprMcpServer::new();
    server
        .run_stdio()
        .map_err(|e| CliError::Aprender(format!("mcp server: {e}")))
}
