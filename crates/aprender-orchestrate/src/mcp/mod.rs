//! MCP (Model Context Protocol) Server Module
//!
//! Exposes batuta functionality as MCP tools for Claude Code and other MCP clients.
//! Uses JSON-RPC 2.0 over stdio transport.
//!
//! ## Tools
//!
//! - `hf_search` - Search HuggingFace Hub for models, datasets, spaces
//! - `hf_info` - Get metadata for a HuggingFace asset
//! - `hf_tree` - Show HuggingFace ecosystem component tree
//! - `hf_integration` - Show PAIML ↔ HuggingFace integration map
//! - `stack_status` - Show PAIML stack component status
//! - `stack_check` - Run stack health check

mod server;
mod types;

pub use server::McpServer;
pub use types::*;
