//! Unit tests for [`super::TaskTool`] and [`super::SubagentRegistry`].

use std::sync::Arc;

use tokio::sync::Mutex;

use super::*;
use crate::agent::driver::mock::MockDriver;
use crate::agent::manifest::AgentManifest;
use crate::agent::pool::AgentPool;

fn make_pool(response: &'static str) -> Arc<Mutex<AgentPool>> {
    let driver = MockDriver::single_response(response);
    Arc::new(Mutex::new(AgentPool::new(Arc::new(driver), 4)))
}

#[test]
fn default_registry_has_three_canonical_types() {
    let r = default_registry();
    assert_eq!(r.len(), 3, "Claude parity: general-purpose + explore + plan");
    assert!(r.resolve("general-purpose").is_some());
    assert!(r.resolve("explore").is_some());
    assert!(r.resolve("plan").is_some());
    assert!(r.resolve("nonexistent").is_none());
}

#[test]
fn registry_register_replaces_by_name() {
    let mut r = SubagentRegistry::new();
    assert!(r.is_empty());
    r.register(SubagentSpec::general_purpose());
    assert_eq!(r.len(), 1);
    r.register(SubagentSpec {
        name: "general-purpose".into(),
        description: "override".into(),
        system_prompt: "custom".into(),
        max_iterations: 99,
    });
    assert_eq!(r.len(), 1, "same name → replace, not dup");
    assert_eq!(r.resolve("general-purpose").unwrap().max_iterations, 99);
}

#[test]
fn registry_names_is_alphabetical() {
    let r = default_registry();
    let names = r.names();
    assert_eq!(names, vec!["explore", "general-purpose", "plan"]);
}

#[tokio::test]
async fn task_tool_definition_includes_registered_types() {
    let pool = make_pool("ok");
    let registry = Arc::new(default_registry());
    let tool = TaskTool::new(registry, pool, AgentManifest::default(), 0, 3);
    let def = tool.definition();
    assert_eq!(def.name, "task");
    assert!(def.description.contains("general-purpose"));
    assert!(def.description.contains("explore"));
    assert!(def.description.contains("plan"));
    let schema = def.input_schema.to_string();
    assert!(schema.contains("subagent_type"));
    assert!(schema.contains("prompt"));
}

#[tokio::test]
async fn task_tool_rejects_unknown_subagent_type() {
    let pool = make_pool("never-runs");
    let registry = Arc::new(default_registry());
    let tool = TaskTool::new(registry, pool, AgentManifest::default(), 0, 3);
    let result = tool
        .execute(serde_json::json!({
            "subagent_type": "unicorn",
            "prompt": "hi"
        }))
        .await;
    assert!(result.is_error, "unknown type must error");
    assert!(result.content.contains("unknown subagent_type"));
    assert!(result.content.contains("unicorn"));
    assert!(result.content.contains("general-purpose"), "lists known types");
}

#[tokio::test]
async fn task_tool_missing_subagent_type_errors() {
    let pool = make_pool("x");
    let registry = Arc::new(default_registry());
    let tool = TaskTool::new(registry, pool, AgentManifest::default(), 0, 3);
    let result = tool.execute(serde_json::json!({ "prompt": "hi" })).await;
    assert!(result.is_error);
    assert!(result.content.contains("subagent_type"));
}

#[tokio::test]
async fn task_tool_missing_prompt_errors() {
    let pool = make_pool("x");
    let registry = Arc::new(default_registry());
    let tool = TaskTool::new(registry, pool, AgentManifest::default(), 0, 3);
    let result = tool
        .execute(serde_json::json!({ "subagent_type": "explore" }))
        .await;
    assert!(result.is_error);
    assert!(result.content.contains("prompt"));
}

#[tokio::test]
async fn task_tool_depth_limit_blocks_spawn() {
    let pool = make_pool("never");
    let registry = Arc::new(default_registry());
    // current == max → blocked
    let tool = TaskTool::new(registry, pool, AgentManifest::default(), 3, 3);
    let result = tool
        .execute(serde_json::json!({
            "subagent_type": "general-purpose",
            "prompt": "go"
        }))
        .await;
    assert!(result.is_error);
    assert!(result.content.contains("depth limit"));
}

#[tokio::test]
async fn task_tool_spawns_child_and_returns_response() {
    let pool = make_pool("child-done");
    let registry = Arc::new(default_registry());
    let tool = TaskTool::new(registry, pool, AgentManifest::default(), 0, 3);
    let result = tool
        .execute(serde_json::json!({
            "subagent_type": "general-purpose",
            "description": "test",
            "prompt": "investigate X"
        }))
        .await;
    assert!(!result.is_error, "execute errored: {}", result.content);
    assert_eq!(result.content, "child-done");
}

#[tokio::test]
async fn task_tool_required_capability_is_spawn() {
    let pool = make_pool("x");
    let registry = Arc::new(default_registry());
    let tool = TaskTool::new(registry, pool, AgentManifest::default(), 0, 5);
    let cap = tool.required_capability();
    assert_eq!(cap, Capability::Spawn { max_depth: 5 });
}

#[tokio::test]
async fn task_tool_from_driver_constructor() {
    let driver: Arc<dyn LlmDriver> = Arc::new(MockDriver::single_response("ok"));
    let tool = TaskTool::from_driver(driver, AgentManifest::default(), 3);
    assert_eq!(tool.name(), "task");
    assert_eq!(tool.max_depth, 3);
}

#[tokio::test]
async fn register_task_tool_adds_to_registry() {
    use crate::agent::tool::ToolRegistry;
    let driver: Arc<dyn LlmDriver> = Arc::new(MockDriver::single_response("ok"));
    let mut reg = ToolRegistry::new();
    let before = reg.len();
    register_task_tool(&mut reg, &AgentManifest::default(), driver, 3);
    assert_eq!(reg.len(), before + 1, "register_task_tool adds one tool");
    assert!(reg.get("task").is_some(), "registered under `task` name");
}

#[test]
fn subagent_spec_presets_have_nonempty_system_prompts() {
    for spec in [
        SubagentSpec::general_purpose(),
        SubagentSpec::explore(),
        SubagentSpec::plan(),
    ] {
        assert!(!spec.system_prompt.trim().is_empty(), "{}: system_prompt empty", spec.name);
        assert!(!spec.description.trim().is_empty(), "{}: description empty", spec.name);
        assert!(spec.max_iterations > 0);
    }
}
