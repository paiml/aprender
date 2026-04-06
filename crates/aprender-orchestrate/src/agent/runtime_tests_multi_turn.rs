//! Multi-turn conversation tests for agent runtime.
//!
//! Tests run_agent_turn() — the multi-turn variant that accepts
//! &mut Vec<Message> for persistent conversation history.
//! See: apr-code.md §3.3, PMAT-115.

use super::*;
use crate::agent::capability::Capability;
use crate::agent::driver::mock::MockDriver;
use crate::agent::driver::ToolDefinition;
use crate::agent::memory::InMemorySubstrate;
use crate::agent::result::TokenUsage;
use crate::agent::tool::{Tool, ToolResult as TResult};
use async_trait::async_trait;

struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &'static str {
        "echo"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "echo".into(),
            description: "Echoes input".into(),
            input_schema: serde_json::json!({"type": "object"}),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> TResult {
        TResult::success(format!("echo: {input}"))
    }

    fn required_capability(&self) -> Capability {
        Capability::Memory
    }
}

fn default_manifest() -> AgentManifest {
    AgentManifest {
        capabilities: vec![Capability::Memory, Capability::Rag],
        ..AgentManifest::default()
    }
}

#[tokio::test]
async fn test_multi_turn_history_accumulates() {
    let manifest = default_manifest();
    let driver = MockDriver::new(vec![
        CompletionResponse {
            text: "answer 1".into(),
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: TokenUsage { input_tokens: 10, output_tokens: 5 },
        },
        CompletionResponse {
            text: "answer 2".into(),
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: TokenUsage { input_tokens: 20, output_tokens: 10 },
        },
    ]);
    let tools = ToolRegistry::new();
    let memory = InMemorySubstrate::new();

    let mut history = Vec::new();

    // Turn 1
    let r1 = run_agent_turn(&manifest, &mut history, "hello", &driver, &tools, &memory, None)
        .await
        .expect("turn 1 failed");
    assert_eq!(r1.text, "answer 1");
    assert_eq!(history.len(), 2, "history after turn 1: {:?}", history);
    assert!(matches!(&history[0], Message::User(s) if s == "hello"));
    assert!(matches!(&history[1], Message::Assistant(s) if s == "answer 1"));

    // Turn 2 — driver sees history from turn 1
    let r2 = run_agent_turn(&manifest, &mut history, "followup", &driver, &tools, &memory, None)
        .await
        .expect("turn 2 failed");
    assert_eq!(r2.text, "answer 2");
    assert_eq!(history.len(), 4, "history after turn 2: {:?}", history);
    assert!(matches!(&history[2], Message::User(s) if s == "followup"));
    assert!(matches!(&history[3], Message::Assistant(s) if s == "answer 2"));
}

#[tokio::test]
async fn test_multi_turn_with_tool_calls() {
    let manifest = default_manifest();
    let driver = MockDriver::new(vec![
        CompletionResponse {
            text: String::new(),
            stop_reason: StopReason::ToolUse,
            tool_calls: vec![ToolCall {
                id: "1".into(),
                name: "echo".into(),
                input: serde_json::json!({"text": "hello"}),
            }],
            usage: TokenUsage { input_tokens: 10, output_tokens: 5 },
        },
        CompletionResponse {
            text: "done with tools".into(),
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: TokenUsage { input_tokens: 15, output_tokens: 8 },
        },
        CompletionResponse {
            text: "I remember the tool call".into(),
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: TokenUsage { input_tokens: 30, output_tokens: 10 },
        },
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(EchoTool));
    let memory = InMemorySubstrate::new();

    let mut history = Vec::new();

    // Turn 1 with tool call
    let r1 = run_agent_turn(&manifest, &mut history, "use echo", &driver, &tools, &memory, None)
        .await
        .expect("turn 1 failed");
    assert_eq!(r1.text, "done with tools");
    assert_eq!(r1.tool_calls, 1);
    assert!(history.len() >= 4, "expected tool history, got {}", history.len());

    // Turn 2 should have full context
    let r2 =
        run_agent_turn(&manifest, &mut history, "what did you do?", &driver, &tools, &memory, None)
            .await
            .expect("turn 2 failed");
    assert_eq!(r2.text, "I remember the tool call");
    assert!(history.len() >= 6, "expected accumulated history, got {}", history.len());
}

#[tokio::test]
async fn test_run_agent_loop_delegates_to_turn() {
    let manifest = default_manifest();
    let driver = MockDriver::single_response("compat");
    let tools = ToolRegistry::new();
    let memory = InMemorySubstrate::new();

    let result = run_agent_loop(&manifest, "test", &driver, &tools, &memory, None)
        .await
        .expect("run_agent_loop failed");
    assert_eq!(result.text, "compat");
}
