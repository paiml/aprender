//! Guard and circuit-break tests for agent runtime loop.
//!
//! Extracted from `runtime_tests_advanced.rs` for QA-002 (≤500 lines).
//! Covers: MCP privacy gates, token budget, max iterations,
//! tool timeout, retry exhaustion, consecutive MaxTokens.

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

fn default_manifest() -> AgentManifest {
    AgentManifest {
        capabilities: vec![Capability::Memory, Capability::Rag],
        ..AgentManifest::default()
    }
}

/// MCP privacy gate blocks SSE transport in sovereign mode.
#[cfg(feature = "agents-mcp")]
#[tokio::test]
async fn test_mcp_privacy_gate_blocks_sse() {
    use crate::agent::manifest::{McpServerConfig, McpTransport};

    let mut manifest = default_manifest();
    manifest.privacy = crate::serve::backends::PrivacyTier::Sovereign;
    manifest.mcp_servers = vec![McpServerConfig {
        name: "remote".into(),
        transport: McpTransport::Sse,
        command: vec![],
        url: Some("https://example.com/mcp".into()),
        capabilities: vec!["*".into()],
    }];

    let driver = MockDriver::single_response("unreachable");
    let tools = ToolRegistry::new();
    let memory = InMemorySubstrate::new();

    let err = run_agent_loop(&manifest, "q", &driver, &tools, &memory, None).await.unwrap_err();
    assert!(
        matches!(err, crate::agent::result::AgentError::CircuitBreak(ref msg) if msg.contains("sovereign")),
        "expected sovereign privacy CircuitBreak, got: {err}"
    );
}

/// MCP privacy gate blocks WebSocket transport in sovereign mode.
#[cfg(feature = "agents-mcp")]
#[tokio::test]
async fn test_mcp_privacy_gate_blocks_websocket() {
    use crate::agent::manifest::{McpServerConfig, McpTransport};

    let mut manifest = default_manifest();
    manifest.privacy = crate::serve::backends::PrivacyTier::Sovereign;
    manifest.mcp_servers = vec![McpServerConfig {
        name: "ws-server".into(),
        transport: McpTransport::WebSocket,
        command: vec![],
        url: Some("wss://example.com/mcp".into()),
        capabilities: vec![],
    }];

    let driver = MockDriver::single_response("unreachable");
    let tools = ToolRegistry::new();
    let memory = InMemorySubstrate::new();

    let err = run_agent_loop(&manifest, "q", &driver, &tools, &memory, None).await.unwrap_err();
    assert!(
        matches!(err, crate::agent::result::AgentError::CircuitBreak(ref msg) if msg.contains("ws-server")),
        "expected CircuitBreak for ws-server, got: {err}"
    );
}

/// MCP privacy gate allows stdio transport in sovereign mode.
#[cfg(feature = "agents-mcp")]
#[tokio::test]
async fn test_mcp_privacy_gate_allows_stdio() {
    use crate::agent::manifest::{McpServerConfig, McpTransport};

    let mut manifest = default_manifest();
    manifest.privacy = crate::serve::backends::PrivacyTier::Sovereign;
    manifest.mcp_servers = vec![McpServerConfig {
        name: "local".into(),
        transport: McpTransport::Stdio,
        command: vec!["echo".into()],
        url: None,
        capabilities: vec!["*".into()],
    }];

    let driver = MockDriver::single_response("ok");
    let tools = ToolRegistry::new();
    let memory = InMemorySubstrate::new();

    let result = run_agent_loop(&manifest, "q", &driver, &tools, &memory, None)
        .await
        .expect("stdio should be allowed in sovereign");
    assert_eq!(result.text, "ok");
}

/// Token budget circuit break from guard triggers AgentError.
#[tokio::test]
async fn test_token_budget_circuit_break() {
    use crate::agent::result::TokenUsage;

    let mut manifest = default_manifest();
    manifest.resources.max_tokens_budget = Some(100);

    let driver = MockDriver::new(vec![CompletionResponse {
        text: "expensive".into(),
        stop_reason: StopReason::EndTurn,
        tool_calls: vec![],
        usage: TokenUsage { input_tokens: 80, output_tokens: 80 },
    }]);
    let tools = ToolRegistry::new();
    let memory = InMemorySubstrate::new();

    let err = run_agent_loop(&manifest, "q", &driver, &tools, &memory, None).await.unwrap_err();
    assert!(
        matches!(err, crate::agent::result::AgentError::CircuitBreak(ref msg) if msg.contains("token budget")),
        "expected token budget CircuitBreak, got: {err}"
    );
}

/// Max iterations with warn before circuit break.
#[tokio::test]
async fn test_max_iterations_with_tool_calls() {
    let mut manifest = default_manifest();
    manifest.resources.max_iterations = 2;

    let driver = MockDriver::new(vec![
        CompletionResponse {
            text: String::new(),
            stop_reason: StopReason::ToolUse,
            tool_calls: vec![ToolCall {
                id: "1".into(),
                name: "echo".into(),
                input: serde_json::json!({"a": 1}),
            }],
            usage: Default::default(),
        },
        CompletionResponse {
            text: String::new(),
            stop_reason: StopReason::ToolUse,
            tool_calls: vec![ToolCall {
                id: "2".into(),
                name: "echo".into(),
                input: serde_json::json!({"a": 2}),
            }],
            usage: Default::default(),
        },
        CompletionResponse {
            text: "done".into(),
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: Default::default(),
        },
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(EchoTool));
    let memory = InMemorySubstrate::new();

    let err = run_agent_loop(&manifest, "q", &driver, &tools, &memory, None).await.unwrap_err();
    assert!(
        matches!(err, crate::agent::result::AgentError::CircuitBreak(_)),
        "expected CircuitBreak, got: {err}"
    );
}

/// Tool timeout produces error result and loop continues.
#[tokio::test]
async fn test_tool_timeout_produces_error() {
    let manifest = default_manifest();

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
            std::time::Duration::from_millis(50)
        }
    }

    let driver = MockDriver::tool_then_response("slow", serde_json::json!({}), "after timeout");
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(SlowTool));
    let memory = InMemorySubstrate::new();

    let result = run_agent_loop(&manifest, "q", &driver, &tools, &memory, None)
        .await
        .expect("loop should continue after tool timeout");
    assert_eq!(result.text, "after timeout");
    assert_eq!(result.tool_calls, 1);
}

/// Retry exhaustion: all retries fail with retryable error.
#[tokio::test]
async fn test_retry_exhaustion() {
    let manifest = default_manifest();

    struct AlwaysFailDriver;

    #[async_trait]
    impl crate::agent::driver::LlmDriver for AlwaysFailDriver {
        async fn complete(
            &self,
            _request: crate::agent::driver::CompletionRequest,
        ) -> Result<CompletionResponse, crate::agent::result::AgentError> {
            Err(crate::agent::result::AgentError::Driver(
                crate::agent::result::DriverError::Network("transient network error".into()),
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

    let err =
        run_agent_loop(&manifest, "q", &AlwaysFailDriver, &tools, &memory, None).await.unwrap_err();
    assert!(
        matches!(err, crate::agent::result::AgentError::Driver(_)),
        "expected Driver error after exhaustion, got: {err}"
    );
}

/// Consecutive MaxTokens triggers circuit break.
#[tokio::test]
async fn test_consecutive_max_tokens_circuit_break() {
    let manifest = default_manifest();

    let driver = MockDriver::new(
        (0..6)
            .map(|_| CompletionResponse {
                text: "partial".into(),
                stop_reason: StopReason::MaxTokens,
                tool_calls: vec![],
                usage: Default::default(),
            })
            .collect(),
    );
    let tools = ToolRegistry::new();
    let memory = InMemorySubstrate::new();

    let err = run_agent_loop(&manifest, "q", &driver, &tools, &memory, None).await.unwrap_err();
    assert!(
        matches!(err, crate::agent::result::AgentError::CircuitBreak(ref msg) if msg.contains("MaxTokens")),
        "expected consecutive MaxTokens CircuitBreak, got: {err}"
    );
}
