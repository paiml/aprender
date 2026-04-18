//! Task tool — Claude-Code Agent-equivalent for typed sub-agent delegation.
//!
//! PMAT-CODE-SPAWN-PARITY-001: the `Task` tool mirrors Claude Code's
//! `Agent` tool, which takes `subagent_type` + `description` + `prompt`
//! and returns the child's final response. This is the 1:1 equivalent
//! for `apr code` and is **default-registered** (no capability gate) so
//! the agent can always delegate, matching Claude Code 1:1.
//!
//! Distinct from `SpawnTool`:
//!   - `SpawnTool` is untyped — caller passes any manifest inline.
//!   - `TaskTool` resolves `subagent_type` via [`SubagentRegistry`] to a
//!     preset personality (system-prompt + narrow tool allowlist). This
//!     matches Claude's pattern where each agent type has a fixed role.
//!
//! Toyota Production System:
//!   - **Poka-Yoke**: subagent_type must exist in the registry — an
//!     unknown type errors out before spawn, so typos can't launch a
//!     misconfigured child.
//!   - **Jidoka**: child `max_iterations` is clamped and recursion is
//!     bounded by `max_depth`.
//!   - **Muda**: child shares parent's driver + memory Arc, so no
//!     duplicate model load.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use crate::agent::capability::Capability;
use crate::agent::driver::{LlmDriver, ToolDefinition};
use crate::agent::manifest::{AgentManifest, ResourceQuota};
use crate::agent::pool::{AgentPool, SpawnConfig};

use super::tool::{Tool, ToolResult};

/// A pre-configured subagent personality resolved by `subagent_type`.
///
/// Holds the pieces of a child `AgentManifest` that vary per type
/// (system prompt, iteration budget). Other fields inherit from the
/// parent manifest so settings like `model_path` stay consistent.
#[derive(Debug, Clone)]
pub struct SubagentSpec {
    /// Identifier used by the parent agent (e.g. `"general-purpose"`).
    pub name: String,
    /// One-line purpose shown in the tool description.
    pub description: String,
    /// Prepended to the child's system prompt to scope its role.
    pub system_prompt: String,
    /// Hard cap on child iterations (Jidoka).
    pub max_iterations: u32,
}

impl SubagentSpec {
    /// The canonical `general-purpose` subagent used by default.
    pub fn general_purpose() -> Self {
        Self {
            name: "general-purpose".into(),
            description:
                "General-purpose agent for researching questions, searching for code, \
                 and executing multi-step tasks. Use when the target is not known \
                 and you are not confident you will find the right match in the \
                 first few tries."
                    .into(),
            system_prompt:
                "You are a general-purpose research subagent. Your job is to answer \
                 the user's question by gathering evidence from the codebase. Return \
                 a concise summary of findings, not running commentary."
                    .into(),
            max_iterations: 10,
        }
    }

    /// `explore` subagent — codebase search specialist.
    pub fn explore() -> Self {
        Self {
            name: "explore".into(),
            description:
                "Fast agent specialized for exploring codebases. Use to find files \
                 by pattern, search code for keywords, or answer questions about \
                 structure. Returns a digest of findings."
                    .into(),
            system_prompt:
                "You are a codebase exploration subagent. Prefer pmat_query over \
                 raw grep. Never edit files. Return a short digest of what you found \
                 with file:line citations."
                    .into(),
            max_iterations: 8,
        }
    }

    /// `plan` subagent — designs implementation strategies.
    pub fn plan() -> Self {
        Self {
            name: "plan".into(),
            description:
                "Software architect subagent for designing implementation plans. \
                 Use when you need a step-by-step plan, critical-file list, or \
                 architectural trade-offs before implementing."
                    .into(),
            system_prompt:
                "You are a planning subagent. Read the relevant code, then return a \
                 numbered plan with (a) files to change, (b) order of changes, and \
                 (c) the key trade-off. Do not write code."
                    .into(),
            max_iterations: 6,
        }
    }
}

/// Registry of subagent types available to [`TaskTool`].
///
/// Claude Code ships with `general-purpose`, `Explore`, and `Plan`.
/// `apr code` mirrors the same three by default; users can register
/// additional types via `register`.
#[derive(Debug, Clone, Default)]
pub struct SubagentRegistry {
    by_name: BTreeMap<String, SubagentSpec>,
}

impl SubagentRegistry {
    /// Empty registry — use [`default_registry`] for the stock 3.
    pub fn new() -> Self {
        Self { by_name: BTreeMap::new() }
    }

    /// Register a subagent spec (replaces any existing entry with the
    /// same name).
    pub fn register(&mut self, spec: SubagentSpec) {
        self.by_name.insert(spec.name.clone(), spec);
    }

    /// Resolve a subagent by type name. Returns `None` if not found.
    pub fn resolve(&self, name: &str) -> Option<&SubagentSpec> {
        self.by_name.get(name)
    }

    /// Number of registered subagent types.
    pub fn len(&self) -> usize {
        self.by_name.len()
    }

    /// `true` when no subagent types are registered.
    pub fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }

    /// All registered names, alphabetical.
    pub fn names(&self) -> Vec<String> {
        self.by_name.keys().cloned().collect()
    }
}

/// Default registry populated with the 3 canonical subagent types
/// (`general-purpose`, `explore`, `plan`) — identical roster to
/// Claude Code's built-in agents.
pub fn default_registry() -> SubagentRegistry {
    let mut r = SubagentRegistry::new();
    r.register(SubagentSpec::general_purpose());
    r.register(SubagentSpec::explore());
    r.register(SubagentSpec::plan());
    r
}

/// Task tool — Claude-Code-equivalent `Agent`/`Task` sub-agent dispatcher.
///
/// Unlike [`crate::agent::tool::spawn::SpawnTool`], this tool is
/// **default-registered** in `apr code` with no capability gate: the
/// parent agent can always delegate via `Task`, matching Claude Code's
/// behavior where `Agent` is part of the default toolbelt.
pub struct TaskTool {
    registry: Arc<SubagentRegistry>,
    pool: Arc<Mutex<AgentPool>>,
    parent_manifest: AgentManifest,
    current_depth: u32,
    max_depth: u32,
}

impl TaskTool {
    /// Create a Task tool.
    ///
    /// * `registry` — resolves `subagent_type` to a [`SubagentSpec`].
    /// * `pool` — shared agent pool; child executions run here.
    /// * `parent_manifest` — used as the base for child manifests
    ///    (model, capabilities, memory settings are inherited).
    /// * `current_depth` / `max_depth` — recursion guard.
    pub fn new(
        registry: Arc<SubagentRegistry>,
        pool: Arc<Mutex<AgentPool>>,
        parent_manifest: AgentManifest,
        current_depth: u32,
        max_depth: u32,
    ) -> Self {
        Self { registry, pool, parent_manifest, current_depth, max_depth }
    }

    /// Convenience constructor — driver + default subagent registry.
    /// Used by `apr code` to install a task tool in the default
    /// registry without the caller threading the pool.
    pub fn from_driver(
        driver: Arc<dyn LlmDriver>,
        parent_manifest: AgentManifest,
        max_depth: u32,
    ) -> Self {
        Self::from_driver_with_registry(driver, parent_manifest, max_depth, default_registry())
    }

    /// Like [`from_driver`] but uses a caller-supplied registry. Used by
    /// [`register_task_tool`] to install the tool with user-defined
    /// subagents (from `.apr/agents/` / `.claude/agents/`) merged in on
    /// top of the 3 canonical built-ins.
    pub fn from_driver_with_registry(
        driver: Arc<dyn LlmDriver>,
        parent_manifest: AgentManifest,
        max_depth: u32,
        registry: SubagentRegistry,
    ) -> Self {
        let pool = Arc::new(Mutex::new(AgentPool::new(driver, 4)));
        Self::new(Arc::new(registry), pool, parent_manifest, 0, max_depth)
    }

    fn build_child_manifest(&self, spec: &SubagentSpec) -> AgentManifest {
        let mut child = self.parent_manifest.clone();
        child.name = format!("{}/{}", self.parent_manifest.name, spec.name);
        child.model.system_prompt = spec.system_prompt.clone();
        child.resources = ResourceQuota {
            max_iterations: child.resources.max_iterations.min(spec.max_iterations),
            ..child.resources.clone()
        };
        child
    }
}

#[async_trait]
impl Tool for TaskTool {
    fn name(&self) -> &'static str {
        "task"
    }

    fn definition(&self) -> ToolDefinition {
        let names = self.registry.names();
        let listed = if names.is_empty() { "(none registered)".into() } else { names.join(", ") };
        ToolDefinition {
            name: "task".into(),
            description: format!(
                "Launch a typed sub-agent to handle a delegated task. \
                 Registered subagent types: {listed}. \
                 The child runs its own perceive-reason-act loop and its \
                 final response is returned as the tool result."
            ),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "subagent_type": {
                        "type": "string",
                        "description": "Registered subagent type (e.g. general-purpose, explore, plan)"
                    },
                    "description": {
                        "type": "string",
                        "description": "Short (3-7 word) label for the task — shown in telemetry"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The full task delegated to the child subagent"
                    }
                },
                "required": ["subagent_type", "prompt"]
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        if self.current_depth >= self.max_depth {
            return ToolResult::error(format!(
                "task depth limit reached ({}/{})",
                self.current_depth, self.max_depth,
            ));
        }

        let subagent_type = match input.get("subagent_type").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => {
                return ToolResult::error("missing required field: subagent_type");
            }
        };

        let prompt = match input.get("prompt").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => {
                return ToolResult::error("missing required field: prompt");
            }
        };

        let spec = match self.registry.resolve(subagent_type) {
            Some(s) => s.clone(),
            None => {
                let names = self.registry.names().join(", ");
                return ToolResult::error(format!(
                    "unknown subagent_type '{subagent_type}' (registered: {names})"
                ));
            }
        };

        let child_manifest = self.build_child_manifest(&spec);
        let config = SpawnConfig { manifest: child_manifest, query: prompt };

        let mut pool = self.pool.lock().await;
        let id = match pool.spawn(config) {
            Ok(id) => id,
            Err(e) => return ToolResult::error(format!("task spawn failed: {e}")),
        };

        match pool.join_next().await {
            Some((completed_id, Ok(result))) if completed_id == id => {
                ToolResult::success(result.text)
            }
            Some((_, Ok(result))) => ToolResult::success(result.text),
            Some((_, Err(e))) => ToolResult::error(format!("subagent error: {e}")),
            None => ToolResult::error("subagent produced no result"),
        }
    }

    fn required_capability(&self) -> Capability {
        // Task tool is part of the default toolbelt (like Claude Code's
        // Agent), but we still model it under the Spawn capability so
        // manifests that *deny* spawning can keep denying it. The
        // default manifest grants Spawn implicitly via [`register_task_tool`].
        Capability::Spawn { max_depth: self.max_depth }
    }
}

/// Register `TaskTool` + `Spawn { max_depth }` capability in the given
/// registry, unless the manifest already denies `Spawn` explicitly.
///
/// Also discovers user-defined subagents from the current working
/// directory's `.apr/agents/` (or `.claude/agents/` for cross-compat)
/// and merges them on top of the 3 canonical built-ins. This is the
/// default-registration hook used by `apr code` so the `Task` tool
/// ships in the default toolbelt (Claude-Code parity).
pub fn register_task_tool(
    registry: &mut super::tool::ToolRegistry,
    manifest: &AgentManifest,
    driver: Arc<dyn LlmDriver>,
    max_depth: u32,
) {
    let mut subagents = default_registry();
    if let Ok(cwd) = std::env::current_dir() {
        let _ = super::custom_agents::register_discovered_into(&mut subagents, &cwd);
    }
    let tool = TaskTool::from_driver_with_registry(
        Arc::clone(&driver),
        manifest.clone(),
        max_depth,
        subagents,
    );
    registry.register(Box::new(tool));
}

#[cfg(test)]
mod tests;
