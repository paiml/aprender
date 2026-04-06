//! Tests for agent runtime loop.

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
async fn test_single_turn_response() {
    let manifest = default_manifest();
    let driver = MockDriver::single_response("Hello!");
    let tools = ToolRegistry::new();
    let memory = InMemorySubstrate::new();

    let result =
        run_agent_loop(&manifest, "hi", &driver, &tools, &memory, None).await.expect("loop failed");

    assert_eq!(result.text, "Hello!");
    assert_eq!(result.iterations, 1);
    assert_eq!(result.tool_calls, 0);
}

#[tokio::test]
async fn test_tool_call_and_response() {
    let manifest = default_manifest();
    let driver =
        MockDriver::tool_then_response("echo", serde_json::json!({"text": "test"}), "Final answer");
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(EchoTool));
    let memory = InMemorySubstrate::new();

    let result = run_agent_loop(&manifest, "do it", &driver, &tools, &memory, None)
        .await
        .expect("loop failed");

    assert_eq!(result.text, "Final answer");
    assert_eq!(result.iterations, 2);
    assert_eq!(result.tool_calls, 1);
}

#[tokio::test]
async fn test_max_iterations_reached() {
    let mut manifest = default_manifest();
    manifest.resources.max_iterations = 1;

    let driver = MockDriver::new(vec![
        CompletionResponse {
            text: String::new(),
            stop_reason: StopReason::ToolUse,
            tool_calls: vec![ToolCall {
                id: "1".into(),
                name: "echo".into(),
                input: serde_json::json!({}),
            }],
            usage: Default::default(),
        },
        CompletionResponse {
            text: String::new(),
            stop_reason: StopReason::ToolUse,
            tool_calls: vec![ToolCall {
                id: "2".into(),
                name: "echo".into(),
                input: serde_json::json!({"x": 1}),
            }],
            usage: Default::default(),
        },
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(EchoTool));
    let memory = InMemorySubstrate::new();

    let err = run_agent_loop(&manifest, "go", &driver, &tools, &memory, None).await.unwrap_err();

    assert!(matches!(err, AgentError::CircuitBreak(_)), "expected CircuitBreak, got: {err}");
}

#[tokio::test]
async fn test_unknown_tool_handled() {
    let manifest = default_manifest();
    let driver = MockDriver::new(vec![
        CompletionResponse {
            text: String::new(),
            stop_reason: StopReason::ToolUse,
            tool_calls: vec![ToolCall {
                id: "1".into(),
                name: "nonexistent".into(),
                input: serde_json::json!({}),
            }],
            usage: Default::default(),
        },
        CompletionResponse {
            text: "recovered".into(),
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: Default::default(),
        },
    ]);
    let tools = ToolRegistry::new();
    let memory = InMemorySubstrate::new();

    let result = run_agent_loop(&manifest, "q", &driver, &tools, &memory, None)
        .await
        .expect("loop should recover from unknown tool");
    assert_eq!(result.text, "recovered");
}

#[tokio::test]
async fn test_capability_denied_handled() {
    let mut manifest = default_manifest();
    manifest.capabilities = vec![Capability::Rag];

    let driver = MockDriver::new(vec![
        CompletionResponse {
            text: String::new(),
            stop_reason: StopReason::ToolUse,
            tool_calls: vec![ToolCall {
                id: "1".into(),
                name: "echo".into(),
                input: serde_json::json!({}),
            }],
            usage: Default::default(),
        },
        CompletionResponse {
            text: "denied path".into(),
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: Default::default(),
        },
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(EchoTool));
    let memory = InMemorySubstrate::new();

    let result = run_agent_loop(&manifest, "q", &driver, &tools, &memory, None)
        .await
        .expect("loop should handle capability denial");
    assert_eq!(result.text, "denied path");
}

#[tokio::test]
async fn test_stream_events_emitted() {
    let manifest = default_manifest();
    let driver = MockDriver::single_response("done");
    let tools = ToolRegistry::new();
    let memory = InMemorySubstrate::new();

    let (tx, mut rx) = mpsc::channel(32);

    run_agent_loop(&manifest, "hi", &driver, &tools, &memory, Some(tx)).await.expect("loop failed");

    let mut phases = vec![];
    while let Ok(event) = rx.try_recv() {
        if let StreamEvent::PhaseChange { phase } = event {
            phases.push(phase);
        }
    }

    assert!(phases.contains(&LoopPhase::Perceive));
    assert!(phases.contains(&LoopPhase::Reason));
    assert!(phases.contains(&LoopPhase::Done));
}

#[tokio::test]
async fn test_memories_recalled_into_system() {
    let manifest = default_manifest();
    let memory = InMemorySubstrate::new();
    memory
        .remember("unnamed-agent", "prior knowledge about SIMD", MemorySource::Conversation, None)
        .await
        .expect("remember failed");

    let driver = MockDriver::single_response("answer");
    let tools = ToolRegistry::new();

    let result = run_agent_loop(&manifest, "SIMD", &driver, &tools, &memory, None)
        .await
        .expect("loop failed");
    assert_eq!(result.text, "answer");
}

#[tokio::test]
async fn test_conversation_stored_in_memory() {
    let manifest = default_manifest();
    let driver = MockDriver::single_response("the answer");
    let tools = ToolRegistry::new();
    let memory = Arc::new(InMemorySubstrate::new());

    run_agent_loop(&manifest, "my question", &driver, &tools, memory.as_ref(), None)
        .await
        .expect("loop failed");

    let recalled = memory.recall("my question", 10, None, None).await.expect("recall failed");
    assert!(!recalled.is_empty());
    assert!(recalled[0].content.contains("the answer"));
}

#[tokio::test]
async fn test_max_tokens_continues_loop() {
    let manifest = default_manifest();
    let driver = MockDriver::new(vec![
        CompletionResponse {
            text: "partial".into(),
            stop_reason: StopReason::MaxTokens,
            tool_calls: vec![],
            usage: Default::default(),
        },
        CompletionResponse {
            text: "complete".into(),
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: Default::default(),
        },
    ]);
    let tools = ToolRegistry::new();
    let memory = InMemorySubstrate::new();

    let result = run_agent_loop(&manifest, "q", &driver, &tools, &memory, None)
        .await
        .expect("loop should continue after MaxTokens");
    assert_eq!(result.text, "complete");
    assert_eq!(result.iterations, 2);
}

#[tokio::test]
async fn test_stop_sequence_terminates() {
    let manifest = default_manifest();
    let driver = MockDriver::new(vec![CompletionResponse {
        text: "stopped by sequence".into(),
        stop_reason: StopReason::StopSequence,
        tool_calls: vec![],
        usage: Default::default(),
    }]);
    let tools = ToolRegistry::new();
    let memory = InMemorySubstrate::new();

    let result = run_agent_loop(&manifest, "test", &driver, &tools, &memory, None)
        .await
        .expect("loop failed");
    assert_eq!(result.text, "stopped by sequence");
    assert_eq!(result.iterations, 1);
}

#[tokio::test]
async fn test_tool_stream_events() {
    let manifest = default_manifest();
    let driver = MockDriver::tool_then_response("echo", serde_json::json!({"text": "hi"}), "done");
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(EchoTool));
    let memory = InMemorySubstrate::new();

    let (tx, mut rx) = mpsc::channel(64);
    run_agent_loop(&manifest, "go", &driver, &tools, &memory, Some(tx)).await.expect("loop failed");

    // Check for ToolUseStart and ToolUseEnd events
    let mut saw_tool_start = false;
    let mut saw_tool_end = false;
    let mut saw_act_phase = false;
    while let Ok(event) = rx.try_recv() {
        match event {
            StreamEvent::ToolUseStart { name, .. } if name == "echo" => {
                saw_tool_start = true;
            }
            StreamEvent::ToolUseEnd { name, .. } if name == "echo" => {
                saw_tool_end = true;
            }
            StreamEvent::PhaseChange { phase: LoopPhase::Act { tool_name } }
                if tool_name == "echo" =>
            {
                saw_act_phase = true;
            }
            _ => {}
        }
    }
    assert!(saw_tool_start, "should emit ToolUseStart");
    assert!(saw_tool_end, "should emit ToolUseEnd");
    assert!(saw_act_phase, "should emit Act phase");
}

#[tokio::test]
async fn test_cost_budget_circuit_break() {
    let mut manifest = default_manifest();
    manifest.resources.max_cost_usd = 0.001;

    // MockDriver with high cost per token
    let driver = MockDriver::new(vec![CompletionResponse {
        text: String::new(),
        stop_reason: StopReason::ToolUse,
        tool_calls: vec![ToolCall {
            id: "1".into(),
            name: "echo".into(),
            input: serde_json::json!({}),
        }],
        usage: TokenUsage { input_tokens: 10000, output_tokens: 10000 },
    }])
    .with_cost_per_token(0.001);

    let mut tools = ToolRegistry::new();
    tools.register(Box::new(EchoTool));
    let memory = InMemorySubstrate::new();

    let err = run_agent_loop(&manifest, "go", &driver, &tools, &memory, None).await.unwrap_err();
    assert!(matches!(err, AgentError::CircuitBreak(_)), "expected CircuitBreak, got: {err}");
}

#[tokio::test]
async fn test_pingpong_blocked_in_tool_calls() {
    let manifest = default_manifest();
    // Three identical tool calls will trigger ping-pong detection
    let driver = MockDriver::new(vec![
        CompletionResponse {
            text: String::new(),
            stop_reason: StopReason::ToolUse,
            tool_calls: vec![ToolCall {
                id: "1".into(),
                name: "echo".into(),
                input: serde_json::json!({"x": "same"}),
            }],
            usage: Default::default(),
        },
        CompletionResponse {
            text: String::new(),
            stop_reason: StopReason::ToolUse,
            tool_calls: vec![ToolCall {
                id: "2".into(),
                name: "echo".into(),
                input: serde_json::json!({"x": "same"}),
            }],
            usage: Default::default(),
        },
        CompletionResponse {
            text: String::new(),
            stop_reason: StopReason::ToolUse,
            tool_calls: vec![ToolCall {
                id: "3".into(),
                name: "echo".into(),
                input: serde_json::json!({"x": "same"}),
            }],
            usage: Default::default(),
        },
        CompletionResponse {
            text: "after block".into(),
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: Default::default(),
        },
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(EchoTool));
    let memory = InMemorySubstrate::new();

    let result = run_agent_loop(&manifest, "q", &driver, &tools, &memory, None)
        .await
        .expect("loop should recover from ping-pong block");
    assert_eq!(result.text, "after block");
}
