//! Memory tool — read/write agent persistent state.
//!
//! Wraps a `MemorySubstrate` as a `Tool` for use in the agent loop.
//! Supports two actions: "remember" (store) and "recall" (retrieve).

use async_trait::async_trait;
use std::sync::Arc;

use super::{Tool, ToolResult};
use crate::agent::capability::Capability;
use crate::agent::driver::ToolDefinition;
use crate::agent::memory::MemorySubstrate;

/// Tool for reading and writing agent memory.
pub struct MemoryTool {
    substrate: Arc<dyn MemorySubstrate>,
    agent_id: String,
}

impl MemoryTool {
    /// Create a new memory tool for the given agent.
    pub fn new(substrate: Arc<dyn MemorySubstrate>, agent_id: String) -> Self {
        Self { substrate, agent_id }
    }
}

#[async_trait]
impl Tool for MemoryTool {
    fn name(&self) -> &'static str {
        "memory"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "memory".into(),
            description: "Read and write agent memory. \
                Actions: 'remember' stores content, \
                'recall' retrieves relevant memories."
                .into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["remember", "recall"],
                        "description": "Action to perform"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to store (remember) or query (recall)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max memories to recall (default 5)"
                    }
                },
                "required": ["action", "content"]
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let action = input.get("action").and_then(|v| v.as_str()).unwrap_or("");
        let content = input.get("content").and_then(|v| v.as_str()).unwrap_or("");

        match action {
            "remember" => self.do_remember(content).await,
            "recall" => {
                #[allow(clippy::cast_possible_truncation)]
                let limit =
                    input.get("limit").and_then(serde_json::Value::as_u64).unwrap_or(5) as usize;
                self.do_recall(content, limit).await
            }
            other => ToolResult::error(format!(
                "unknown action '{other}', expected 'remember' or 'recall'"
            )),
        }
    }

    fn required_capability(&self) -> Capability {
        Capability::Memory
    }
}

impl MemoryTool {
    async fn do_remember(&self, content: &str) -> ToolResult {
        match self
            .substrate
            .remember(&self.agent_id, content, crate::agent::memory::MemorySource::ToolResult, None)
            .await
        {
            Ok(id) => ToolResult::success(format!("Stored memory: {id}")),
            Err(e) => ToolResult::error(format!("Failed to store: {e}")),
        }
    }

    async fn do_recall(&self, query: &str, limit: usize) -> ToolResult {
        match self.substrate.recall(query, limit, None, None).await {
            Ok(fragments) => {
                if fragments.is_empty() {
                    return ToolResult::success("No memories found.");
                }
                let text = fragments
                    .iter()
                    .enumerate()
                    .map(|(i, f)| {
                        format!("{}. [score={:.2}] {}", i + 1, f.relevance_score, f.content)
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                ToolResult::success(text)
            }
            Err(e) => ToolResult::error(format!("Failed to recall: {e}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::memory::InMemorySubstrate;

    fn make_tool() -> MemoryTool {
        let substrate = Arc::new(InMemorySubstrate::new());
        MemoryTool::new(substrate, "test-agent".into())
    }

    #[tokio::test]
    async fn test_remember_and_recall() {
        let tool = make_tool();

        // Remember
        let result = tool
            .execute(serde_json::json!({
                "action": "remember",
                "content": "Rust is great for systems programming"
            }))
            .await;
        assert!(!result.is_error);
        assert!(result.content.contains("Stored memory"));

        // Recall
        let result = tool
            .execute(serde_json::json!({
                "action": "recall",
                "content": "Rust",
                "limit": 3
            }))
            .await;
        assert!(!result.is_error);
        assert!(result.content.contains("systems programming"));
    }

    #[tokio::test]
    async fn test_recall_empty() {
        let tool = make_tool();

        let result = tool
            .execute(serde_json::json!({
                "action": "recall",
                "content": "nonexistent"
            }))
            .await;
        assert!(!result.is_error);
        assert!(result.content.contains("No memories found"));
    }

    #[tokio::test]
    async fn test_unknown_action() {
        let tool = make_tool();

        let result = tool
            .execute(serde_json::json!({
                "action": "delete",
                "content": "test"
            }))
            .await;
        assert!(result.is_error);
        assert!(result.content.contains("unknown action"));
    }

    #[test]
    fn test_tool_metadata() {
        let tool = make_tool();
        assert_eq!(tool.name(), "memory");
        assert_eq!(tool.required_capability(), Capability::Memory);

        let def = tool.definition();
        assert_eq!(def.name, "memory");
        assert!(def.description.contains("recall"));
    }
}
