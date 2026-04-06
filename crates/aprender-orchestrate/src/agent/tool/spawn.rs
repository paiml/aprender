//! Sub-agent spawning tool.
//!
//! Allows an agent to delegate work to a child agent running
//! its own perceive-reason-act loop. The child shares the parent's
//! LLM driver and memory substrate but gets its own loop guard.
//!
//! Requires `Capability::Spawn { max_depth }` — recursion is
//! bounded by depth tracking (Jidoka: stop on runaway spawning).

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use crate::agent::capability::Capability;
use crate::agent::driver::ToolDefinition;
use crate::agent::manifest::AgentManifest;
use crate::agent::pool::{AgentPool, SpawnConfig};

use super::{Tool, ToolResult};

/// Tool that spawns a sub-agent, waits for completion, and
/// returns the child's response as the tool result.
pub struct SpawnTool {
    pool: Arc<Mutex<AgentPool>>,
    parent_manifest: AgentManifest,
    current_depth: u32,
    max_depth: u32,
}

impl SpawnTool {
    /// Create a spawn tool with depth tracking.
    pub fn new(
        pool: Arc<Mutex<AgentPool>>,
        parent_manifest: AgentManifest,
        current_depth: u32,
        max_depth: u32,
    ) -> Self {
        Self { pool, parent_manifest, current_depth, max_depth }
    }
}

#[async_trait]
impl Tool for SpawnTool {
    fn name(&self) -> &'static str {
        "spawn_agent"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "spawn_agent".into(),
            description: "Spawn a sub-agent to handle a delegated task. \
                The child agent runs its own perceive-reason-act loop \
                and returns its final response."
                .into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The task to delegate to the sub-agent"
                    },
                    "name": {
                        "type": "string",
                        "description": "Optional name for the sub-agent (defaults to parent name + '-sub')"
                    }
                },
                "required": ["query"]
            }),
        }
    }

    #[cfg_attr(
        feature = "agents-contracts",
        provable_contracts_macros::contract("agent-loop-v1", equation = "spawn_depth_bound")
    )]
    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        // Jidoka: depth guard
        if self.current_depth >= self.max_depth {
            return ToolResult::error(format!(
                "spawn depth limit reached ({}/{})",
                self.current_depth, self.max_depth,
            ));
        }

        let query = match input.get("query").and_then(|v| v.as_str()) {
            Some(q) => q.to_string(),
            None => {
                return ToolResult::error("missing required field: query");
            }
        };

        let name = match input.get("name").and_then(|v| v.as_str()) {
            Some(n) => n.to_string(),
            None => format!("{}-sub", self.parent_manifest.name),
        };

        // Build child manifest (inherits parent config, new name)
        let mut child_manifest = self.parent_manifest.clone();
        child_manifest.name = name;
        // Reduce child iterations to prevent runaway
        child_manifest.resources.max_iterations = child_manifest.resources.max_iterations.min(10);

        let config = SpawnConfig { manifest: child_manifest, query };

        // Spawn and await
        let mut pool = self.pool.lock().await;
        let id = match pool.spawn(config) {
            Ok(id) => id,
            Err(e) => {
                return ToolResult::error(format!("spawn failed: {e}"));
            }
        };

        match pool.join_next().await {
            Some((completed_id, Ok(result))) if completed_id == id => {
                ToolResult::success(result.text)
            }
            Some((_, Ok(result))) => {
                // Different agent finished first — still return it
                ToolResult::success(result.text)
            }
            Some((_, Err(e))) => ToolResult::error(format!("sub-agent error: {e}")),
            None => ToolResult::error("sub-agent produced no result"),
        }
    }

    fn required_capability(&self) -> Capability {
        Capability::Spawn { max_depth: self.max_depth }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::driver::mock::MockDriver;

    fn make_pool() -> Arc<Mutex<AgentPool>> {
        let driver = MockDriver::single_response("child response");
        Arc::new(Mutex::new(AgentPool::new(Arc::new(driver), 4)))
    }

    #[test]
    fn test_spawn_tool_definition() {
        let pool = make_pool();
        let manifest = AgentManifest::default();
        let tool = SpawnTool::new(pool, manifest, 0, 3);
        let def = tool.definition();
        assert_eq!(def.name, "spawn_agent");
        assert!(def.description.contains("sub-agent"));
    }

    #[test]
    fn test_spawn_tool_capability() {
        let pool = make_pool();
        let manifest = AgentManifest::default();
        let tool = SpawnTool::new(pool, manifest, 0, 3);
        assert_eq!(tool.required_capability(), Capability::Spawn { max_depth: 3 },);
    }

    #[tokio::test]
    async fn test_spawn_tool_depth_limit() {
        let pool = make_pool();
        let manifest = AgentManifest::default();
        // current_depth == max_depth → blocked
        let tool = SpawnTool::new(pool, manifest, 3, 3);
        let result = tool.execute(serde_json::json!({ "query": "hello" })).await;
        assert!(result.is_error);
        assert!(result.content.contains("depth limit"));
    }

    #[tokio::test]
    async fn test_spawn_tool_missing_query() {
        let pool = make_pool();
        let manifest = AgentManifest::default();
        let tool = SpawnTool::new(pool, manifest, 0, 3);
        let result = tool.execute(serde_json::json!({})).await;
        assert!(result.is_error);
        assert!(result.content.contains("missing"));
    }

    #[tokio::test]
    async fn test_spawn_tool_executes_child() {
        let pool = make_pool();
        let manifest = AgentManifest::default();
        let tool = SpawnTool::new(pool, manifest, 0, 3);
        let result = tool
            .execute(serde_json::json!({
                "query": "do something",
                "name": "worker"
            }))
            .await;
        assert!(!result.is_error, "error: {}", result.content);
        assert_eq!(result.content, "child response");
    }

    #[tokio::test]
    async fn test_spawn_tool_default_name() {
        let pool = make_pool();
        let mut manifest = AgentManifest::default();
        manifest.name = "parent".into();
        let tool = SpawnTool::new(pool, manifest, 0, 3);
        let result = tool.execute(serde_json::json!({ "query": "hello" })).await;
        assert!(!result.is_error, "error: {}", result.content);
    }
}
