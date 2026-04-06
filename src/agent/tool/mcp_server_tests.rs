use super::*;
use crate::agent::memory::in_memory::InMemorySubstrate;

fn test_registry() -> HandlerRegistry {
    let memory = Arc::new(InMemorySubstrate::new());
    let mut registry = HandlerRegistry::new();
    registry.register(Box::new(MemoryHandler::new(memory, "test-agent")));
    registry
}

#[test]
fn test_registry_creation() {
    let registry = HandlerRegistry::new();
    assert!(registry.is_empty());
    assert_eq!(registry.len(), 0);
}

#[test]
fn test_registry_register() {
    let registry = test_registry();
    assert_eq!(registry.len(), 1);
    assert!(!registry.is_empty());
}

#[test]
fn test_list_tools() {
    let registry = test_registry();
    let tools = registry.list_tools();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "memory");
}

#[tokio::test]
async fn test_dispatch_unknown_method() {
    let registry = test_registry();
    let result = registry.dispatch("nonexistent", serde_json::json!({})).await;
    assert!(result.is_error);
    assert!(result.content.contains("unknown method"));
}

#[tokio::test]
async fn test_memory_store() {
    let registry = test_registry();
    let result = registry
        .dispatch(
            "memory",
            serde_json::json!({
                "action": "store",
                "content": "test memory content"
            }),
        )
        .await;
    assert!(!result.is_error);
    assert!(result.content.contains("Stored with id"));
}

#[tokio::test]
async fn test_memory_store_empty_content() {
    let registry = test_registry();
    let result = registry
        .dispatch(
            "memory",
            serde_json::json!({
                "action": "store",
                "content": ""
            }),
        )
        .await;
    assert!(result.is_error);
    assert!(result.content.contains("required"));
}

#[tokio::test]
async fn test_memory_recall_empty() {
    let registry = test_registry();
    let result = registry
        .dispatch(
            "memory",
            serde_json::json!({
                "action": "recall",
                "query": "nothing"
            }),
        )
        .await;
    assert!(!result.is_error);
    assert!(result.content.contains("No matching"));
}

#[tokio::test]
async fn test_memory_store_then_recall() {
    let memory: Arc<dyn MemorySubstrate> = Arc::new(InMemorySubstrate::new());
    let mut registry = HandlerRegistry::new();
    registry.register(Box::new(MemoryHandler::new(Arc::clone(&memory), "test")));

    // Store
    let store_result = registry
        .dispatch(
            "memory",
            serde_json::json!({
                "action": "store",
                "content": "Rust is a systems language"
            }),
        )
        .await;
    assert!(!store_result.is_error);

    // Recall
    let recall_result = registry
        .dispatch(
            "memory",
            serde_json::json!({
                "action": "recall",
                "query": "Rust",
                "limit": 3
            }),
        )
        .await;
    assert!(!recall_result.is_error);
    assert!(recall_result.content.contains("systems language"));
}

#[tokio::test]
async fn test_memory_unknown_action() {
    let registry = test_registry();
    let result = registry
        .dispatch(
            "memory",
            serde_json::json!({
                "action": "delete"
            }),
        )
        .await;
    assert!(result.is_error);
    assert!(result.content.contains("unknown action"));
}

#[test]
fn test_memory_handler_schema() {
    let memory = Arc::new(InMemorySubstrate::new());
    let handler = MemoryHandler::new(memory, "test");
    let schema = handler.input_schema();
    assert!(schema.get("properties").is_some());
    assert_eq!(handler.name(), "memory");
    assert!(!handler.description().is_empty());
}

#[test]
fn test_default_registry() {
    let registry = HandlerRegistry::default();
    assert!(registry.is_empty());
}

#[tokio::test]
async fn test_compute_handler_run() {
    let handler = ComputeHandler::new("/tmp");
    let result = handler
        .handle(serde_json::json!({
            "action": "run",
            "command": "echo hello"
        }))
        .await;
    assert!(!result.is_error);
    assert!(result.content.contains("hello"));
}

#[tokio::test]
async fn test_compute_handler_run_empty_command() {
    let handler = ComputeHandler::new("/tmp");
    let result = handler
        .handle(serde_json::json!({
            "action": "run",
            "command": ""
        }))
        .await;
    assert!(result.is_error);
    assert!(result.content.contains("required"));
}

#[tokio::test]
async fn test_compute_handler_parallel() {
    let handler = ComputeHandler::new("/tmp");
    let result = handler
        .handle(serde_json::json!({
            "action": "parallel",
            "commands": ["echo first", "echo second"]
        }))
        .await;
    assert!(!result.is_error);
    assert!(result.content.contains("first"));
    assert!(result.content.contains("second"));
}

#[tokio::test]
async fn test_compute_handler_unknown_action() {
    let handler = ComputeHandler::new("/tmp");
    let result = handler
        .handle(serde_json::json!({
            "action": "delete"
        }))
        .await;
    assert!(result.is_error);
    assert!(result.content.contains("unknown action"));
}

#[test]
fn test_compute_handler_metadata() {
    let handler = ComputeHandler::new("/tmp");
    assert_eq!(handler.name(), "compute");
    assert!(!handler.description().is_empty());
    let schema = handler.input_schema();
    assert!(schema.get("properties").is_some());
}

#[tokio::test]
async fn test_compute_handler_parallel_empty_commands() {
    let handler = ComputeHandler::new("/tmp");
    let result = handler
        .handle(serde_json::json!({
            "action": "parallel",
            "commands": []
        }))
        .await;
    assert!(result.is_error);
    assert!(result.content.contains("required"));
}

#[tokio::test]
async fn test_compute_handler_missing_action() {
    let handler = ComputeHandler::new("/tmp");
    let result = handler
        .handle(serde_json::json!({
            "command": "echo hi"
        }))
        .await;
    assert!(result.is_error);
    assert!(result.content.contains("unknown action"));
}

#[tokio::test]
async fn test_compute_handler_failing_command() {
    let handler = ComputeHandler::new("/tmp");
    let result = handler
        .handle(serde_json::json!({
            "action": "run",
            "command": "exit 1"
        }))
        .await;
    assert!(result.is_error);
    assert!(result.content.contains("exit"));
}

#[tokio::test]
async fn test_compute_handler_command_with_stderr() {
    let handler = ComputeHandler::new("/tmp");
    let result = handler
        .handle(serde_json::json!({
            "action": "run",
            "command": "echo ok && echo warn >&2"
        }))
        .await;
    assert!(!result.is_error);
    assert!(result.content.contains("ok"));
    assert!(result.content.contains("warn"));
}

#[tokio::test]
async fn test_memory_recall_with_limit() {
    let memory: Arc<dyn MemorySubstrate> = Arc::new(InMemorySubstrate::new());
    let mut registry = HandlerRegistry::new();
    registry.register(Box::new(MemoryHandler::new(Arc::clone(&memory), "test")));

    // Store multiple items
    for i in 0..5 {
        registry
            .dispatch(
                "memory",
                serde_json::json!({
                    "action": "store",
                    "content": format!("memory item {i} about Rust")
                }),
            )
            .await;
    }

    // Recall with limit
    let result = registry
        .dispatch(
            "memory",
            serde_json::json!({
                "action": "recall",
                "query": "Rust",
                "limit": 2
            }),
        )
        .await;
    assert!(!result.is_error);
    assert!(result.content.contains("score"));
}

#[tokio::test]
async fn test_memory_store_no_content_field() {
    let registry = test_registry();
    let result = registry.dispatch("memory", serde_json::json!({"action": "store"})).await;
    assert!(result.is_error);
    assert!(result.content.contains("required"));
}

#[tokio::test]
async fn test_compute_handler_parallel_with_failure() {
    let handler = ComputeHandler::new("/tmp");
    let result = handler
        .handle(serde_json::json!({
            "action": "parallel",
            "commands": ["echo pass", "false"]
        }))
        .await;
    // Should contain output from both
    assert!(result.content.contains("pass"));
    assert!(result.content.contains("exit"));
}

#[tokio::test]
async fn test_execute_command_truncation() {
    // Generate output exceeding max_output_bytes (8192)
    let handler = ComputeHandler::new("/tmp");
    let result = handler
        .handle(serde_json::json!({
            "action": "run",
            "command": "head -c 20000 /dev/urandom | base64"
        }))
        .await;
    // Should truncate at max_output_bytes and include marker
    if !result.is_error {
        assert!(result.content.len() <= 8192 + 50, "content should be truncated");
    }
}

#[tokio::test]
async fn test_execute_command_nonexistent() {
    let handler = ComputeHandler::new("/tmp");
    let result = handler
        .handle(serde_json::json!({
            "action": "run",
            "command": "__nonexistent_binary_xyz_123__"
        }))
        .await;
    assert!(result.is_error);
}

#[tokio::test]
async fn test_compute_handler_parallel_no_commands_key() {
    let handler = ComputeHandler::new("/tmp");
    let result = handler
        .handle(serde_json::json!({
            "action": "parallel"
        }))
        .await;
    assert!(result.is_error);
    assert!(result.content.contains("required"));
}

#[test]
fn test_mcp_tool_info_serialization() {
    let info = McpToolInfo {
        name: "test".into(),
        description: "Test tool".into(),
        input_schema: serde_json::json!({}),
    };
    let json = serde_json::to_string(&info).expect("serialize");
    assert!(json.contains("\"name\":\"test\""));
}
