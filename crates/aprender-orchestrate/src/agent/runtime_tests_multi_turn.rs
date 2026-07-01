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

// ═══════════════════════════════════════════════════════════════════════════
// CCPA-m296 FALSIFIERS — OBLIG-APR-CODE-TOOLCALL-RETENTION
//
// The apr-code agent loop had a HARNESS bug independent of the model: a prior
// assistant TOOL_CALL turn was retained / re-rendered as raw Markdown prose,
// re-priming prose mode and eroding a format-correct model's tool-calling to
// 0/N across a multi-turn run (a self-reinforcing text loop).
//
// Oracle = the STRUCTURED, tool-call-preserving render. These tests are
// mutation-verified: reverting the fix (pushing raw response.text into history,
// or dropping the structured AssistantToolUse/ToolResult render) turns them RED.
// ═══════════════════════════════════════════════════════════════════════════

use crate::agent::driver::chat_template::{format_prompt_with_template, ChatTemplate};
use crate::agent::driver::{CompletionRequest, ToolResultMsg};

/// FALSIFY-TOOLCALL-RETENTION-001: a prior format-correct assistant TOOL_CALL
/// turn renders STRUCTURALLY on the next turn — the canonical assistant
/// `<tool_call>` + the `<tool_result>` — and NEVER as re-flattened raw Markdown
/// with a capability-breaking "### Continue:" prose nudge.
///
/// Oracle: the structured ChatML render. Mutation: re-rendering the prior turn
/// as raw Markdown (or appending "### Continue:") makes the negative assertions
/// FAIL.
#[test]
fn falsify_toolcall_retention_001_structured_render_no_continue_nudge() {
    // History as produced by handle_tool_calls() after a format-correct turn.
    let history = vec![
        Message::User("fix the off-by-one in src/lib.rs".into()),
        Message::AssistantToolUse(ToolCall {
            id: "local-1".into(),
            name: "file_read".into(),
            input: serde_json::json!({"path": "src/lib.rs"}),
        }),
        Message::ToolResult(ToolResultMsg {
            tool_use_id: "local-1".into(),
            content: "fn f() { return (i, j); }".into(),
            is_error: false,
        }),
    ];

    let request = CompletionRequest {
        model: "qwen3-1.7b".into(),
        messages: history,
        tools: vec![],
        max_tokens: 256,
        temperature: 0.2,
        system: Some("You are apr code.".into()),
    };

    let prompt = format_prompt_with_template(&request, ChatTemplate::ChatMl);

    // (a) STRUCTURE PRESERVED: the tool_call + tool_result survive into the
    //     next-turn prompt as the canonical envelope, not as flattened prose.
    assert!(
        prompt.contains("<tool_call>"),
        "tool_call must be preserved structurally across turns, got:\n{prompt}"
    );
    assert!(
        prompt.contains("\"name\":\"file_read\"") || prompt.contains("\"name\": \"file_read\""),
        "the tool name must survive in the structured render, got:\n{prompt}"
    );
    assert!(
        prompt.contains("<tool_result>"),
        "tool_result must be preserved structurally across turns, got:\n{prompt}"
    );

    // (b) NO PROSE RE-PRIMING: the harness must not re-inject a capability-
    //     breaking "### Continue:" prose nudge after a tool-using turn.
    assert!(
        !prompt.contains("### Continue"),
        "must NOT re-inject a '### Continue:' prose nudge after a tool turn, got:\n{prompt}"
    );
    assert!(
        !prompt.contains("Continue:"),
        "must NOT append any prose Continue nudge that re-primes text mode, got:\n{prompt}"
    );
}

/// FALSIFY-TOOLCALL-RETENTION-002: when a format-correct turn yields a tool
/// call, the multi-turn history retains the STRUCTURED AssistantToolUse +
/// ToolResult — NOT a raw `Message::Assistant(markdown)` prose blob.
///
/// Oracle: `Message::AssistantToolUse` / `Message::ToolResult` present in
/// history; no Assistant message carrying raw `<tool_call>` markup. Mutation:
/// retaining raw tool-call text as Assistant prose makes the final assertion
/// FAIL.
#[tokio::test]
async fn falsify_toolcall_retention_002_history_keeps_structure_not_prose() {
    let manifest = default_manifest();
    let driver = MockDriver::new(vec![
        // Turn 1: a clean tool call (format-correct model).
        CompletionResponse {
            text: String::new(),
            stop_reason: StopReason::ToolUse,
            tool_calls: vec![ToolCall {
                id: "1".into(),
                name: "echo".into(),
                input: serde_json::json!({"text": "hi"}),
            }],
            usage: TokenUsage { input_tokens: 10, output_tokens: 5 },
        },
        // Then a clean text conclusion.
        CompletionResponse {
            text: "done".into(),
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: TokenUsage { input_tokens: 12, output_tokens: 4 },
        },
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(EchoTool));
    let memory = InMemorySubstrate::new();
    let mut history = Vec::new();

    run_agent_turn(&manifest, &mut history, "use echo", &driver, &tools, &memory, None)
        .await
        .expect("turn failed");

    // History must carry the STRUCTURED tool call + result.
    assert!(
        history.iter().any(|m| matches!(m, Message::AssistantToolUse(_))),
        "structured AssistantToolUse must be retained, got: {history:?}"
    );
    assert!(
        history.iter().any(|m| matches!(m, Message::ToolResult(_))),
        "structured ToolResult must be retained, got: {history:?}"
    );
    // No Assistant prose message may carry raw tool-call markup back into history.
    assert!(
        !history.iter().any(|m| matches!(m, Message::Assistant(s) if s.contains("<tool_call>"))),
        "raw <tool_call> markup must NOT be retained as Assistant prose, got: {history:?}"
    );
}

/// FALSIFY-TOOLCALL-RETENTION-003: `retain_assistant_text` strips lingering
/// tool-call markup so a tool-using turn cannot be re-rendered as prose that
/// re-primes text mode. Genuine prose passes through unchanged.
///
/// Oracle: residue stripped, prose preserved. Mutation: making
/// `retain_assistant_text` an identity function makes the strip assertions FAIL.
#[test]
fn falsify_toolcall_retention_003_retain_strips_residue_keeps_prose() {
    // Pure tool-call residue collapses to empty (structure lives elsewhere).
    let only_call =
        "<tool_call>\n{\"name\": \"shell\", \"input\": {\"command\": \"ls\"}}\n</tool_call>";
    assert_eq!(
        super::retain_assistant_text(only_call),
        "",
        "a turn that was entirely tool-call markup must not enter history as prose"
    );

    // Unclosed trailing <tool_call> residue is truncated.
    let unclosed = "Let me check.\n<tool_call> {\"name\": \"glob\", \"input\": {}}";
    assert_eq!(super::retain_assistant_text(unclosed), "Let me check.");

    // Genuine prose passes through untouched.
    let prose = "The bug is an off-by-one on line 42.";
    assert_eq!(super::retain_assistant_text(prose), prose);

    // Mixed: prose is kept, the tool-call span is removed.
    let mixed =
        "Here is the fix.\n<tool_call>\n{\"name\": \"file_edit\", \"input\": {}}\n</tool_call>";
    assert_eq!(super::retain_assistant_text(mixed), "Here is the fix.");
}
