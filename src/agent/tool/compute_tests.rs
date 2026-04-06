use super::*;
use std::env;

fn test_tool() -> ComputeTool {
    ComputeTool::new(env::current_dir().expect("cwd").to_string_lossy().into())
}

#[test]
fn test_tool_metadata() {
    let tool = test_tool();
    assert_eq!(tool.name(), "compute");
    let def = tool.definition();
    assert_eq!(def.name, "compute");
    assert!(def.description.contains("parallel"));
}

#[test]
fn test_required_capability() {
    let tool = test_tool();
    assert_eq!(tool.required_capability(), Capability::Compute);
}

#[test]
fn test_default_timeout() {
    let tool = test_tool();
    assert_eq!(tool.timeout(), Duration::from_secs(300));
}

#[test]
fn test_custom_timeout() {
    let tool = test_tool().with_timeout(Duration::from_secs(60));
    assert_eq!(tool.timeout(), Duration::from_secs(60));
}

#[test]
fn test_max_concurrent() {
    let tool = test_tool().with_max_concurrent(8);
    assert_eq!(tool.max_concurrent, 8);
}

#[test]
fn test_truncate_output() {
    assert_eq!(ComputeTool::truncate_output("hello"), "hello");
    let long = "x".repeat(MAX_TASK_OUTPUT_BYTES + 100);
    assert!(ComputeTool::truncate_output(&long).contains("[output truncated"));
}

#[test]
fn test_schema_structure() {
    let def = test_tool().definition();
    assert_eq!(def.input_schema["type"], "object");
    assert!(def.input_schema["required"]
        .as_array()
        .expect("required")
        .iter()
        .any(|v| v == "action"));
}

#[tokio::test]
async fn test_run_single_command() {
    let tool = test_tool();
    let result = tool
        .execute(serde_json::json!({
            "action": "run",
            "command": "echo compute_test"
        }))
        .await;
    assert!(!result.is_error, "error: {}", result.content);
    assert!(result.content.contains("compute_test"));
}

#[tokio::test]
async fn test_run_failing_command() {
    let tool = test_tool();
    let result = tool
        .execute(serde_json::json!({
            "action": "run",
            "command": "false"
        }))
        .await;
    assert!(result.is_error);
    assert!(result.content.contains("exit code"));
}

#[tokio::test]
async fn test_parallel_commands() {
    let tool = test_tool();
    let result = tool
        .execute(serde_json::json!({
            "action": "parallel",
            "commands": ["echo task1", "echo task2"]
        }))
        .await;
    assert!(!result.is_error, "error: {}", result.content);
    assert!(result.content.contains("Task 1"));
    assert!(result.content.contains("Task 2"));
    assert!(result.content.contains("task1"));
    assert!(result.content.contains("task2"));
}

#[tokio::test]
async fn test_missing_action() {
    let tool = test_tool();
    let result = tool.execute(serde_json::json!({"command": "echo hi"})).await;
    assert!(result.is_error);
    assert!(result.content.contains("missing"));
}

#[tokio::test]
async fn test_missing_command() {
    let tool = test_tool();
    let result = tool.execute(serde_json::json!({"action": "run"})).await;
    assert!(result.is_error);
    assert!(result.content.contains("missing"));
}

#[tokio::test]
async fn test_empty_commands_array() {
    let tool = test_tool();
    let result = tool
        .execute(serde_json::json!({
            "action": "parallel",
            "commands": []
        }))
        .await;
    assert!(result.is_error);
    assert!(result.content.contains("empty"));
}

#[tokio::test]
async fn test_parallel_exceeds_max_concurrent() {
    let tool = test_tool().with_max_concurrent(2);
    let result = tool
        .execute(serde_json::json!({
            "action": "parallel",
            "commands": [
                "echo one",
                "echo two",
                "echo three",
                "echo four"
            ]
        }))
        .await;
    // Should only run 2 tasks (max_concurrent)
    assert!(!result.is_error, "error: {}", result.content);
    assert!(result.content.contains("Task 1"));
    assert!(result.content.contains("Task 2"));
    // Tasks 3 and 4 are dropped
    assert!(!result.content.contains("Task 3"));
}

#[tokio::test]
async fn test_run_command_with_stderr() {
    let tool = test_tool();
    let result = tool
        .execute(serde_json::json!({
            "action": "run",
            "command": "echo ok && echo warning >&2"
        }))
        .await;
    assert!(!result.is_error);
    assert!(result.content.contains("ok"));
    assert!(result.content.contains("warning"));
}

#[tokio::test]
async fn test_parallel_with_mixed_results() {
    let tool = test_tool();
    let result = tool
        .execute(serde_json::json!({
            "action": "parallel",
            "commands": ["echo success", "false"]
        }))
        .await;
    // One task fails — should be reported as error
    assert!(result.is_error);
    assert!(result.content.contains("Task 1"));
    assert!(result.content.contains("Task 2"));
}

#[tokio::test]
async fn test_parallel_missing_commands() {
    let tool = test_tool();
    let result = tool
        .execute(serde_json::json!({
            "action": "parallel"
        }))
        .await;
    assert!(result.is_error);
    assert!(result.content.contains("missing"));
}

#[tokio::test]
async fn test_unknown_action() {
    let tool = test_tool();
    let result = tool
        .execute(serde_json::json!({
            "action": "cancel"
        }))
        .await;
    assert!(result.is_error);
    assert!(result.content.contains("unknown action"));
}
