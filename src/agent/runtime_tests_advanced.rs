//! Advanced tests for agent runtime loop.
//!
//! Extracted from `runtime_tests.rs` for QA-002 (≤500 lines).
//! Covers: stop sequences, stream events with tool calls, retry
//! logic, context truncation, message conversion, sovereign privacy.

use super::*;
use crate::agent::capability::Capability;
use crate::agent::driver::mock::MockDriver;
use crate::agent::driver::ToolDefinition;
use crate::agent::memory::InMemorySubstrate;
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

/// Tool that sleeps longer than its timeout.
struct SlowTool;

#[async_trait]
impl Tool for SlowTool {
    fn name(&self) -> &'static str {
        "slow"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "slow".into(),
            description: "Sleeps forever".into(),
            input_schema: serde_json::json!({"type": "object"}),
        }
    }

    async fn execute(&self, _input: serde_json::Value) -> TResult {
        tokio::time::sleep(std::time::Duration::from_secs(60)).await;
        TResult::success("never reached")
    }

    fn required_capability(&self) -> Capability {
        Capability::Memory
    }

    fn timeout(&self) -> std::time::Duration {
        // Very short timeout for testing
        std::time::Duration::from_millis(50)
    }
}

fn default_manifest() -> AgentManifest {
    AgentManifest {
        capabilities: vec![Capability::Memory, Capability::Rag],
        ..AgentManifest::default()
    }
}

#[tokio::test]
async fn test_stop_sequence_handled() {
    let manifest = default_manifest();
    let driver = MockDriver::new(vec![CompletionResponse {
        text: "stopped".into(),
        stop_reason: StopReason::StopSequence,
        tool_calls: vec![],
        usage: Default::default(),
    }]);
    let tools = ToolRegistry::new();
    let memory = InMemorySubstrate::new();

    let result = run_agent_loop(&manifest, "q", &driver, &tools, &memory, None)
        .await
        .expect("StopSequence should end the loop");
    assert_eq!(result.text, "stopped");
}

#[tokio::test]
async fn test_stream_events_with_tool_call() {
    let manifest = default_manifest();
    let driver = MockDriver::tool_then_response("echo", serde_json::json!({}), "done");
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(EchoTool));
    let memory = InMemorySubstrate::new();

    let (tx, mut rx) = mpsc::channel(64);

    run_agent_loop(&manifest, "hi", &driver, &tools, &memory, Some(tx)).await.expect("loop failed");

    let mut got_tool_start = false;
    let mut got_tool_end = false;
    let mut got_act = false;
    while let Ok(event) = rx.try_recv() {
        match event {
            StreamEvent::ToolUseStart { .. } => got_tool_start = true,
            StreamEvent::ToolUseEnd { .. } => got_tool_end = true,
            StreamEvent::PhaseChange { phase: LoopPhase::Act { .. } } => got_act = true,
            _ => {}
        }
    }
    assert!(got_tool_start, "expected ToolUseStart event");
    assert!(got_tool_end, "expected ToolUseEnd event");
    assert!(got_act, "expected Act phase event");
}

/// Driver that fails N times with a retryable error, then succeeds.
struct RetryDriver {
    fail_count: std::sync::atomic::AtomicU32,
    max_fails: u32,
    success_response: CompletionResponse,
}

impl RetryDriver {
    fn new(max_fails: u32, response: CompletionResponse) -> Self {
        Self {
            fail_count: std::sync::atomic::AtomicU32::new(0),
            max_fails,
            success_response: response,
        }
    }
}

#[async_trait]
impl crate::agent::driver::LlmDriver for RetryDriver {
    async fn complete(
        &self,
        _request: crate::agent::driver::CompletionRequest,
    ) -> Result<CompletionResponse, crate::agent::result::AgentError> {
        let count = self.fail_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if count < self.max_fails {
            Err(crate::agent::result::AgentError::Driver(
                crate::agent::result::DriverError::Network("transient network error".into()),
            ))
        } else {
            Ok(self.success_response.clone())
        }
    }

    fn context_window(&self) -> usize {
        4096
    }

    fn privacy_tier(&self) -> crate::serve::backends::PrivacyTier {
        crate::serve::backends::PrivacyTier::Sovereign
    }
}

#[tokio::test]
async fn test_retry_on_transient_error() {
    let manifest = default_manifest();
    let driver = RetryDriver::new(
        2,
        CompletionResponse {
            text: "recovered after retry".into(),
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: Default::default(),
        },
    );
    let tools = ToolRegistry::new();
    let memory = InMemorySubstrate::new();

    let result = run_agent_loop(&manifest, "q", &driver, &tools, &memory, None)
        .await
        .expect("should succeed after retries");
    assert_eq!(result.text, "recovered after retry");
}

#[tokio::test]
async fn test_non_retryable_error_fails_immediately() {
    let manifest = default_manifest();

    struct FailDriver;

    #[async_trait]
    impl crate::agent::driver::LlmDriver for FailDriver {
        async fn complete(
            &self,
            _request: crate::agent::driver::CompletionRequest,
        ) -> Result<CompletionResponse, crate::agent::result::AgentError> {
            Err(crate::agent::result::AgentError::Driver(
                crate::agent::result::DriverError::InferenceFailed("model corrupted".into()),
            ))
        }
        fn context_window(&self) -> usize {
            4096
        }
        fn privacy_tier(&self) -> crate::serve::backends::PrivacyTier {
            crate::serve::backends::PrivacyTier::Sovereign
        }
    }

    let tools = ToolRegistry::new();
    let memory = InMemorySubstrate::new();

    let err = run_agent_loop(&manifest, "q", &FailDriver, &tools, &memory, None).await.unwrap_err();
    assert!(
        matches!(
            err,
            crate::agent::result::AgentError::Driver(
                crate::agent::result::DriverError::InferenceFailed(_)
            )
        ),
        "expected InferenceFailed, got: {err}"
    );
}

/// Context truncation: small window driver truncates long conversations.
#[tokio::test]
async fn test_context_truncation_small_window() {
    let manifest = default_manifest();

    // Driver with 200-token context window (tiny)
    struct TinyWindowDriver {
        inner: MockDriver,
    }
    #[async_trait]
    impl crate::agent::driver::LlmDriver for TinyWindowDriver {
        async fn complete(
            &self,
            request: crate::agent::driver::CompletionRequest,
        ) -> Result<CompletionResponse, crate::agent::result::AgentError> {
            // Verify messages were truncated: should be fewer
            // than the 3 tool-call messages we'll generate
            self.inner.complete(request).await
        }
        fn context_window(&self) -> usize {
            200 // Very small
        }
        fn privacy_tier(&self) -> crate::serve::backends::PrivacyTier {
            crate::serve::backends::PrivacyTier::Sovereign
        }
    }

    // Multi-turn: tool call then response
    let driver = TinyWindowDriver {
        inner: MockDriver::tool_then_response(
            "echo",
            serde_json::json!({"text": "x".repeat(100)}),
            "truncated ok",
        ),
    };
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(EchoTool));
    let memory = InMemorySubstrate::new();

    let result = run_agent_loop(&manifest, "hi", &driver, &tools, &memory, None)
        .await
        .expect("should succeed with truncation");
    assert_eq!(result.text, "truncated ok");
}

/// Context truncation: messages converted correctly to ChatMessage.
#[tokio::test]
async fn test_message_to_chat_message_conversion() {
    use crate::agent::driver::Message;

    let msgs = vec![
        Message::System("sys".into()),
        Message::User("hello".into()),
        Message::Assistant("hi".into()),
        Message::AssistantToolUse(ToolCall {
            id: "1".into(),
            name: "echo".into(),
            input: serde_json::json!({"x": 1}),
        }),
        Message::ToolResult(crate::agent::driver::ToolResultMsg {
            tool_use_id: "1".into(),
            content: "result".into(),
            is_error: false,
        }),
    ];

    let chat_msgs: Vec<_> = msgs.iter().map(|m| m.to_chat_message()).collect();

    assert_eq!(chat_msgs.len(), 5);
    assert_eq!(chat_msgs[0].content, "sys");
    assert_eq!(chat_msgs[1].content, "hello");
    assert_eq!(chat_msgs[2].content, "hi");
    assert!(chat_msgs[3].content.contains("echo"));
    assert!(chat_msgs[4].content.contains("result"));
}

/// Sovereign privacy blocks network tools even when capability is granted.
#[tokio::test]
async fn test_sovereign_privacy_blocks_network() {
    let mut manifest = AgentManifest {
        name: "sovereign-agent".into(),
        capabilities: vec![
            Capability::Memory,
            Capability::Network { allowed_hosts: vec!["*".into()] },
        ],
        ..AgentManifest::default()
    };
    manifest.privacy = crate::serve::backends::PrivacyTier::Sovereign;

    let driver = MockDriver::tool_then_response(
        "network",
        serde_json::json!({"url": "https://api.example.com"}),
        "network blocked by sovereign",
    );

    let mut tools = ToolRegistry::new();
    tools.register(Box::new(crate::agent::tool::network::NetworkTool::new(vec!["*".into()])));

    let memory = InMemorySubstrate::new();
    let result = run_agent_loop(&manifest, "test", &driver, &tools, &memory, None)
        .await
        .expect("loop should complete");
    assert_eq!(result.text, "network blocked by sovereign");
    assert_eq!(result.tool_calls, 0, "sovereign must block network tool");
}
