//! MCP Server — expose agent tools to external MCP clients.
//!
//! Implements handler dispatch for agent tools (memory, rag, compute)
//! so external LLM clients (Claude Code, other agents) can call
//! the agent's tools over MCP protocol.
//!
//! Uses a trait-based handler abstraction compatible with pforge.
//! Refs: arXiv:2505.02279, arXiv:2503.23278

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use super::ToolResult;
use crate::agent::memory::MemorySubstrate;

/// Handler for a single MCP tool endpoint.
///
/// Mirrors pforge `Handler` trait pattern for forward compatibility.
#[async_trait]
pub trait McpHandler: Send + Sync {
    /// Tool name as exposed via MCP (e.g., `memory_store`).
    fn name(&self) -> &'static str;

    /// Human-readable description for tool discovery.
    fn description(&self) -> &'static str;

    /// JSON Schema for the tool's input parameters.
    fn input_schema(&self) -> serde_json::Value;

    /// Execute the tool with the given parameters.
    async fn handle(&self, params: serde_json::Value) -> ToolResult;
}

/// Registry of MCP handlers for dispatch.
pub struct HandlerRegistry {
    handlers: HashMap<String, Box<dyn McpHandler>>,
}

impl HandlerRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self { handlers: HashMap::new() }
    }

    /// Register a handler.
    pub fn register(&mut self, handler: Box<dyn McpHandler>) {
        let name = handler.name().to_string();
        self.handlers.insert(name, handler);
    }

    /// Dispatch a tool call to the appropriate handler.
    pub async fn dispatch(&self, method: &str, params: serde_json::Value) -> ToolResult {
        match self.handlers.get(method) {
            Some(handler) => handler.handle(params).await,
            None => ToolResult::error(format!("unknown method: {method}")),
        }
    }

    /// List available tools for MCP discovery.
    pub fn list_tools(&self) -> Vec<McpToolInfo> {
        self.handlers
            .values()
            .map(|h| McpToolInfo {
                name: h.name().to_string(),
                description: h.description().to_string(),
                input_schema: h.input_schema(),
            })
            .collect()
    }

    /// Number of registered handlers.
    pub fn len(&self) -> usize {
        self.handlers.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.handlers.is_empty()
    }
}

impl Default for HandlerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Tool info returned by MCP tools/list.
#[derive(Debug, Clone, serde::Serialize)]
pub struct McpToolInfo {
    /// Tool name.
    pub name: String,
    /// Tool description.
    pub description: String,
    /// JSON Schema for input.
    pub input_schema: serde_json::Value,
}

/// Memory handler — exposes agent memory via MCP.
///
/// Supports `store` (remember) and `recall` (search) actions.
pub struct MemoryHandler {
    memory: Arc<dyn MemorySubstrate>,
    agent_id: String,
}

impl MemoryHandler {
    /// Create a new memory handler.
    pub fn new(memory: Arc<dyn MemorySubstrate>, agent_id: impl Into<String>) -> Self {
        Self { memory, agent_id: agent_id.into() }
    }
}

#[async_trait]
impl McpHandler for MemoryHandler {
    fn name(&self) -> &'static str {
        "memory"
    }

    fn description(&self) -> &'static str {
        "Store and recall agent memory fragments"
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["store", "recall"]
                },
                "content": { "type": "string" },
                "query": { "type": "string" },
                "limit": { "type": "integer" }
            },
            "required": ["action"]
        })
    }

    async fn handle(&self, params: serde_json::Value) -> ToolResult {
        let action = params.get("action").and_then(|v| v.as_str()).unwrap_or("");

        match action {
            "store" => {
                let content = params.get("content").and_then(|v| v.as_str()).unwrap_or("");
                if content.is_empty() {
                    return ToolResult::error("content is required for store");
                }
                match self
                    .memory
                    .remember(
                        &self.agent_id,
                        content,
                        crate::agent::memory::MemorySource::User,
                        None,
                    )
                    .await
                {
                    Ok(id) => ToolResult::success(format!("Stored with id: {id}")),
                    Err(e) => ToolResult::error(format!("store failed: {e}")),
                }
            }
            "recall" => {
                let query = params.get("query").and_then(|v| v.as_str()).unwrap_or("");
                let limit = params
                    .get("limit")
                    .and_then(serde_json::Value::as_u64)
                    .map_or(5, |v| usize::try_from(v).unwrap_or(5));
                match self.memory.recall(query, limit, None, None).await {
                    Ok(fragments) => {
                        if fragments.is_empty() {
                            return ToolResult::success("No matching memories found.");
                        }
                        let mut out = String::new();
                        for f in &fragments {
                            use std::fmt::Write;
                            let _ =
                                writeln!(out, "- {} (score: {:.2})", f.content, f.relevance_score,);
                        }
                        ToolResult::success(out)
                    }
                    Err(e) => ToolResult::error(format!("recall failed: {e}")),
                }
            }
            _ => ToolResult::error(format!("unknown action: {action} (expected: store, recall)")),
        }
    }
}

/// RAG handler — exposes document search via MCP.
///
/// Wraps `RagOracle` to allow external clients to search
/// indexed Sovereign AI Stack documentation.
#[cfg(feature = "rag")]
pub struct RagHandler {
    oracle: Arc<crate::oracle::rag::RagOracle>,
    max_results: usize,
}

#[cfg(feature = "rag")]
impl RagHandler {
    /// Create a new RAG handler.
    pub fn new(oracle: Arc<crate::oracle::rag::RagOracle>, max_results: usize) -> Self {
        Self { oracle, max_results }
    }
}

#[cfg(feature = "rag")]
#[async_trait]
impl McpHandler for RagHandler {
    fn name(&self) -> &'static str {
        "rag"
    }

    fn description(&self) -> &'static str {
        "Search indexed Sovereign AI Stack documentation"
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for documentation"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results (default: 5)"
                }
            },
            "required": ["query"]
        })
    }

    async fn handle(&self, params: serde_json::Value) -> ToolResult {
        let query = params.get("query").and_then(|v| v.as_str()).unwrap_or("");
        if query.is_empty() {
            return ToolResult::error("query is required for search");
        }

        let limit = params
            .get("limit")
            .and_then(serde_json::Value::as_u64)
            .map_or(self.max_results, |v| usize::try_from(v).unwrap_or(self.max_results));

        let results = self.oracle.query(query);
        let truncated: Vec<_> = results.into_iter().take(limit).collect();

        if truncated.is_empty() {
            return ToolResult::success("No results found.");
        }

        let mut out = String::new();
        for (i, r) in truncated.iter().enumerate() {
            use std::fmt::Write;
            let _ =
                writeln!(out, "{}. [{}] {} (score: {:.3})", i + 1, r.component, r.source, r.score,);
            let _ = writeln!(out, "   {}", r.content);
        }
        ToolResult::success(out)
    }
}

/// Compute handler — exposes task execution via MCP.
///
/// Supports `run` (single command) and `parallel` (multiple commands)
/// actions. Output is truncated to prevent context overflow.
pub struct ComputeHandler {
    working_dir: String,
    max_output_bytes: usize,
}

impl ComputeHandler {
    /// Create a new compute handler.
    pub fn new(working_dir: impl Into<String>) -> Self {
        Self { working_dir: working_dir.into(), max_output_bytes: 8192 }
    }
}

#[async_trait]
impl McpHandler for ComputeHandler {
    fn name(&self) -> &'static str {
        "compute"
    }

    fn description(&self) -> &'static str {
        "Execute shell commands with output capture"
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["run", "parallel"]
                },
                "command": { "type": "string" },
                "commands": {
                    "type": "array",
                    "items": { "type": "string" }
                }
            },
            "required": ["action"]
        })
    }

    async fn handle(&self, params: serde_json::Value) -> ToolResult {
        let action = params.get("action").and_then(|v| v.as_str()).unwrap_or("");

        match action {
            "run" => {
                let command = params.get("command").and_then(|v| v.as_str()).unwrap_or("");
                if command.is_empty() {
                    return ToolResult::error("command is required for run");
                }
                execute_command(command, &self.working_dir, self.max_output_bytes).await
            }
            "parallel" => {
                let commands: Vec<String> = params
                    .get("commands")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                    .unwrap_or_default();
                if commands.is_empty() {
                    return ToolResult::error("commands array is required for parallel");
                }
                let mut results = Vec::new();
                for cmd in &commands {
                    let r = execute_command(cmd, &self.working_dir, self.max_output_bytes).await;
                    results.push(format!("$ {cmd}\n{}", r.content));
                }
                ToolResult::success(results.join("\n---\n"))
            }
            _ => ToolResult::error(format!("unknown action: {action} (expected: run, parallel)")),
        }
    }
}

/// Execute a single shell command and capture output.
async fn execute_command(command: &str, working_dir: &str, max_bytes: usize) -> ToolResult {
    let output = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(command)
        .current_dir(working_dir)
        .output()
        .await;

    match output {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            let mut text = stdout.to_string();
            if !stderr.is_empty() {
                text.push_str("\nstderr: ");
                text.push_str(&stderr);
            }
            if text.len() > max_bytes {
                text.truncate(max_bytes);
                text.push_str("\n[truncated]");
            }
            if out.status.success() {
                ToolResult::success(text)
            } else {
                ToolResult::error(format!("exit {}: {}", out.status.code().unwrap_or(-1), text,))
            }
        }
        Err(e) => ToolResult::error(format!("exec failed: {e}")),
    }
}

#[cfg(test)]
#[path = "mcp_server_tests.rs"]
mod tests;
