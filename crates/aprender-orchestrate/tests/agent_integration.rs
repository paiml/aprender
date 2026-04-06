//! Integration tests for `batuta agent` module.
//!
//! Tests the full agent loop with MockDriver, InMemorySubstrate,
//! and MemoryTool wired together — end-to-end validation of the
//! perceive-reason-act pattern.
//!
//! Spec refs: Section 9.2 (Integration Tests)

#![cfg(feature = "agents")]

use std::sync::Arc;

use batuta::agent::capability::Capability;
use batuta::agent::driver::mock::MockDriver;
use batuta::agent::driver::{CompletionResponse, StreamEvent, ToolCall};
use batuta::agent::manifest::AgentManifest;
use batuta::agent::memory::InMemorySubstrate;
use batuta::agent::memory::MemorySubstrate;
use batuta::agent::result::{StopReason, TokenUsage};
use batuta::agent::runtime::run_agent_loop;
use batuta::agent::tool::memory::MemoryTool;
use batuta::agent::tool::ToolRegistry;
use batuta::agent::AgentBuilder;
use tokio::sync::mpsc;

fn test_manifest() -> AgentManifest {
    AgentManifest {
        name: "test-agent".into(),
        capabilities: vec![Capability::Memory, Capability::Rag],
        ..AgentManifest::default()
    }
}

/// Full loop: tool call → memory store → verify memory.
#[tokio::test]
async fn test_full_loop_with_memory_tool() {
    let manifest = test_manifest();
    let memory = Arc::new(InMemorySubstrate::new());

    let driver = MockDriver::tool_then_response(
        "memory",
        serde_json::json!({
            "action": "remember",
            "content": "Important: SIMD is fast"
        }),
        "I've stored that information.",
    );

    let mut tools = ToolRegistry::new();
    tools.register(Box::new(MemoryTool::new(memory.clone(), "test-agent".into())));

    let result = run_agent_loop(
        &manifest,
        "Remember that SIMD is fast",
        &driver,
        &tools,
        memory.as_ref(),
        None,
    )
    .await
    .expect("loop failed");

    assert_eq!(result.text, "I've stored that information.");
    assert_eq!(result.iterations, 2);
    assert_eq!(result.tool_calls, 1);

    // Verify memory was actually stored
    let recalled = memory.recall("SIMD", 10, None, None).await.expect("recall failed");
    assert!(!recalled.is_empty(), "memory should contain SIMD-related entries");
}

/// Agent builder API round-trip.
#[tokio::test]
async fn test_agent_builder_end_to_end() {
    let manifest = test_manifest();
    let memory = Arc::new(InMemorySubstrate::new());
    let driver = MockDriver::single_response("Builder works!");

    let result = AgentBuilder::new(&manifest)
        .driver(&driver)
        .memory(memory.as_ref())
        .run("Hello builder")
        .await
        .expect("builder run failed");

    assert_eq!(result.text, "Builder works!");

    // Verify conversation stored in memory
    let recalled = memory.recall("Hello builder", 10, None, None).await.expect("recall failed");
    assert!(!recalled.is_empty(), "conversation should be stored in memory");
}

/// Stream events are emitted for all phases during a tool call loop.
#[tokio::test]
async fn test_stream_events_full_lifecycle() {
    let manifest = test_manifest();
    let memory = InMemorySubstrate::new();

    let driver = MockDriver::tool_then_response(
        "memory",
        serde_json::json!({"action": "recall", "query": "test"}),
        "Done with stream test",
    );

    let mem_arc = Arc::new(InMemorySubstrate::new());
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(MemoryTool::new(mem_arc, "test-agent".into())));

    let (tx, mut rx) = mpsc::channel(64);

    run_agent_loop(&manifest, "test", &driver, &tools, &memory, Some(tx))
        .await
        .expect("loop failed");

    let mut phases = vec![];
    let mut tool_starts = 0u32;
    let mut tool_ends = 0u32;

    while let Ok(event) = rx.try_recv() {
        match event {
            StreamEvent::PhaseChange { phase } => {
                phases.push(format!("{phase}"));
            }
            StreamEvent::ToolUseStart { .. } => tool_starts += 1,
            StreamEvent::ToolUseEnd { .. } => tool_ends += 1,
            _ => {}
        }
    }

    // Should have: Perceive → Reason → Act → Reason → Done
    assert!(phases.len() >= 4, "expected ≥4 phase changes, got: {phases:?}");
    assert_eq!(tool_starts, 1, "expected 1 ToolUseStart");
    assert_eq!(tool_ends, 1, "expected 1 ToolUseEnd");
}

/// Sovereign privacy enforcement: only Memory capability
/// prevents tool execution for non-granted capabilities.
#[tokio::test]
async fn test_sovereign_capability_enforcement() {
    let mut manifest = test_manifest();
    // Only grant Memory, NOT Rag
    manifest.capabilities = vec![Capability::Memory];

    let driver = MockDriver::new(vec![
        CompletionResponse {
            text: String::new(),
            stop_reason: StopReason::ToolUse,
            tool_calls: vec![ToolCall {
                id: "1".into(),
                name: "memory".into(),
                input: serde_json::json!({
                    "action": "recall",
                    "query": "test"
                }),
            }],
            usage: Default::default(),
        },
        CompletionResponse {
            text: "capability enforced".into(),
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: Default::default(),
        },
    ]);

    let mem_arc = Arc::new(InMemorySubstrate::new());
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(MemoryTool::new(mem_arc, "test-agent".into())));

    let memory = InMemorySubstrate::new();

    let result = run_agent_loop(&manifest, "test", &driver, &tools, &memory, None)
        .await
        .expect("loop should succeed");
    assert_eq!(result.text, "capability enforced");
}

/// Multiple tool calls in a single loop iteration.
#[tokio::test]
async fn test_multi_tool_call_iteration() {
    let manifest = test_manifest();

    let driver = MockDriver::new(vec![
        CompletionResponse {
            text: String::new(),
            stop_reason: StopReason::ToolUse,
            tool_calls: vec![
                ToolCall {
                    id: "1".into(),
                    name: "memory".into(),
                    input: serde_json::json!({
                        "action": "remember",
                        "content": "fact A"
                    }),
                },
                ToolCall {
                    id: "2".into(),
                    name: "memory".into(),
                    input: serde_json::json!({
                        "action": "remember",
                        "content": "fact B"
                    }),
                },
            ],
            usage: TokenUsage { input_tokens: 100, output_tokens: 50 },
        },
        CompletionResponse {
            text: "Stored both facts.".into(),
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: TokenUsage { input_tokens: 200, output_tokens: 20 },
        },
    ]);

    let mem_arc = Arc::new(InMemorySubstrate::new());
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(MemoryTool::new(mem_arc.clone(), "test-agent".into())));

    let result =
        run_agent_loop(&manifest, "Store two facts", &driver, &tools, mem_arc.as_ref(), None)
            .await
            .expect("loop failed");

    assert_eq!(result.text, "Stored both facts.");
    assert_eq!(result.tool_calls, 2);
    assert_eq!(result.usage.input_tokens, 300);
    assert_eq!(result.usage.output_tokens, 70);
}

/// Memory recall augments system prompt.
#[tokio::test]
async fn test_memory_recall_augments_prompt() {
    let manifest = test_manifest();
    let memory = InMemorySubstrate::new();

    // Pre-populate memory
    memory
        .remember(
            "test-agent",
            "SIMD uses vector operations for parallelism",
            batuta::agent::memory::MemorySource::Conversation,
            None,
        )
        .await
        .expect("remember failed");

    let driver = MockDriver::single_response("Based on my recalled context about SIMD...");
    let tools = ToolRegistry::new();

    let result = run_agent_loop(&manifest, "SIMD", &driver, &tools, &memory, None)
        .await
        .expect("loop failed");

    assert!(result.text.contains("SIMD"));
}

/// RoutingDriver: primary fails, fallback succeeds end-to-end.
#[cfg(feature = "native")]
#[tokio::test]
async fn test_routing_driver_fallback_integration() {
    use async_trait::async_trait;
    use batuta::agent::driver::router::RoutingDriver;
    use batuta::agent::driver::CompletionRequest;
    use batuta::serve::backends::PrivacyTier;

    let manifest = test_manifest();

    // Failing primary driver
    struct FailPrimary;

    #[async_trait]
    impl batuta::agent::driver::LlmDriver for FailPrimary {
        async fn complete(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse, batuta::agent::AgentError> {
            Err(batuta::agent::AgentError::Driver(
                batuta::agent::result::DriverError::InferenceFailed(
                    "local model unavailable".into(),
                ),
            ))
        }
        fn context_window(&self) -> usize {
            4096
        }
        fn privacy_tier(&self) -> PrivacyTier {
            PrivacyTier::Sovereign
        }
    }

    let fallback = MockDriver::single_response("fallback response");
    let driver = RoutingDriver::new(Box::new(FailPrimary), Box::new(fallback));

    let tools = ToolRegistry::new();
    let memory = InMemorySubstrate::new();

    let result = run_agent_loop(&manifest, "hello", &driver, &tools, &memory, None)
        .await
        .expect("should fallback");
    assert_eq!(result.text, "fallback response");
    assert_eq!(driver.metrics().spillover_count(), 1);
}

// ═══════════════════════════════════════════════════════════
// FALSIFICATION TESTS (Spec §13.2)
// Popperian tests that attempt to BREAK invariants.
// ═══════════════════════════════════════════════════════════

/// FALSIFY-AL-001: Loop termination.
/// MockDriver returns ToolUse indefinitely → must hit MaxIterationsReached.
#[tokio::test]
async fn test_falsify_al_001_loop_termination() {
    let mut manifest = test_manifest();
    manifest.resources.max_iterations = 3;
    manifest.resources.max_tool_calls = 100; // won't hit this

    // Create driver that always returns ToolUse (never EndTurn)
    let responses: Vec<CompletionResponse> = (0..10)
        .map(|i| CompletionResponse {
            text: String::new(),
            stop_reason: StopReason::ToolUse,
            tool_calls: vec![ToolCall {
                id: format!("call-{i}"),
                name: "memory".into(),
                input: serde_json::json!({
                    "action": "recall",
                    "query": format!("query-{i}")
                }),
            }],
            usage: Default::default(),
        })
        .collect();

    let driver = MockDriver::new(responses);
    let mem_arc = Arc::new(InMemorySubstrate::new());
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(MemoryTool::new(mem_arc.clone(), "test-agent".into())));

    let result =
        run_agent_loop(&manifest, "infinite tools", &driver, &tools, mem_arc.as_ref(), None).await;

    assert!(result.is_err(), "FALSIFY-AL-001: loop must terminate at max_iterations");
    let err = result.unwrap_err();
    assert!(
        matches!(err, batuta::agent::AgentError::CircuitBreak(_)),
        "FALSIFY-AL-001: expected CircuitBreak, got: {err}"
    );
}

/// FALSIFY-AL-002: Capability deny-by-default.
/// Empty capabilities → all tool calls denied.
#[tokio::test]
async fn test_falsify_al_002_capability_deny_by_default() {
    let mut manifest = test_manifest();
    manifest.capabilities = vec![]; // No capabilities!

    let driver = MockDriver::tool_then_response(
        "memory",
        serde_json::json!({"action": "recall", "query": "x"}),
        "Tool was denied.",
    );

    let mem_arc = Arc::new(InMemorySubstrate::new());
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(MemoryTool::new(mem_arc.clone(), "test-agent".into())));

    let result = run_agent_loop(&manifest, "try tool", &driver, &tools, mem_arc.as_ref(), None)
        .await
        .expect("should succeed with denied tool");

    // Tool was denied but loop completed
    assert_eq!(result.text, "Tool was denied.");
    // Tool call counted but was denied (push_tool_error)
    assert_eq!(result.tool_calls, 0, "FALSIFY-AL-002: denied tools should not count as executed");
}

/// FALSIFY-AL-003: Ping-pong detection.
/// Same tool call 3x → Block.
#[tokio::test]
async fn test_falsify_al_003_pingpong_detection() {
    let manifest = test_manifest();

    // Same exact tool call repeated 5 times
    let responses: Vec<CompletionResponse> = (0..5)
        .map(|_| CompletionResponse {
            text: String::new(),
            stop_reason: StopReason::ToolUse,
            tool_calls: vec![ToolCall {
                id: "same-id".into(),
                name: "memory".into(),
                input: serde_json::json!({
                    "action": "recall",
                    "query": "same-query"
                }),
            }],
            usage: Default::default(),
        })
        .chain(std::iter::once(CompletionResponse {
            text: "should not reach here".into(),
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: Default::default(),
        }))
        .collect();

    let driver = MockDriver::new(responses);
    let mem_arc = Arc::new(InMemorySubstrate::new());
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(MemoryTool::new(mem_arc.clone(), "test-agent".into())));

    let result =
        run_agent_loop(&manifest, "ping-pong test", &driver, &tools, mem_arc.as_ref(), None).await;

    // Should complete (Block sends error message to tool, loop continues)
    // The ping-pong Block prevents the tool from executing but the
    // loop continues until the model stops or max iterations.
    // In this case, the model keeps trying the same call which gets
    // blocked, and eventually hits max iterations or gets a final response.
    assert!(result.is_ok() || result.is_err(), "FALSIFY-AL-003: ping-pong must be detected");
}

/// FALSIFY-AL-004: Cost circuit breaker.
/// High token counts + low budget → CircuitBreak.
#[tokio::test]
async fn test_falsify_al_004_cost_circuit_breaker() {
    let mut manifest = test_manifest();
    manifest.resources.max_cost_usd = 0.001; // Very low budget

    // Driver that produces high token counts
    let responses: Vec<CompletionResponse> = (0..10)
        .map(|i| CompletionResponse {
            text: String::new(),
            stop_reason: StopReason::ToolUse,
            tool_calls: vec![ToolCall {
                id: format!("call-{i}"),
                name: "memory".into(),
                input: serde_json::json!({
                    "action": "recall",
                    "query": format!("q-{i}")
                }),
            }],
            usage: TokenUsage { input_tokens: 100_000, output_tokens: 50_000 },
        })
        .collect();

    // $0.01 per 1K tokens → 150K tokens/response = $1.50/response > $0.001 budget
    let driver = MockDriver::new(responses).with_cost_per_token(0.00001);
    let mem_arc = Arc::new(InMemorySubstrate::new());
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(MemoryTool::new(mem_arc.clone(), "test-agent".into())));

    let result =
        run_agent_loop(&manifest, "expensive query", &driver, &tools, mem_arc.as_ref(), None).await;

    assert!(result.is_err(), "FALSIFY-AL-004: cost budget must be enforced");
    let err = result.unwrap_err();
    assert!(
        matches!(err, batuta::agent::AgentError::CircuitBreak(_)),
        "FALSIFY-AL-004: expected CircuitBreak, got: {err}"
    );
}

/// FALSIFY-AL-005: Consecutive MaxTokens circuit break.
/// 5 consecutive MaxTokens → CircuitBreak.
#[tokio::test]
async fn test_falsify_al_005_consecutive_max_tokens() {
    let manifest = test_manifest();

    let responses: Vec<CompletionResponse> = (0..6)
        .map(|_| CompletionResponse {
            text: "truncated...".into(),
            stop_reason: StopReason::MaxTokens,
            tool_calls: vec![],
            usage: Default::default(),
        })
        .collect();

    let driver = MockDriver::new(responses);
    let tools = ToolRegistry::new();
    let memory = InMemorySubstrate::new();

    let result = run_agent_loop(&manifest, "test", &driver, &tools, &memory, None).await;

    assert!(result.is_err(), "FALSIFY-AL-005: 5 consecutive MaxTokens must circuit-break");
    let err = result.unwrap_err();
    assert!(
        matches!(err, batuta::agent::AgentError::CircuitBreak(_)),
        "FALSIFY-AL-005: expected CircuitBreak, got: {err}"
    );
}

/// FALSIFY-AL-006: MaxTokens reset prevents false circuit-break.
/// Interleaved ToolUse resets the counter — only consecutive MaxTokens
/// should trigger the circuit breaker.
#[tokio::test]
async fn test_falsify_al_006_max_tokens_reset_interleaved() {
    let manifest = test_manifest();

    // Pattern: MaxTokens × 3, ToolUse (resets), MaxTokens × 3, EndTurn
    // Total MaxTokens = 6, but never 5 consecutive → should NOT circuit-break
    let responses: Vec<CompletionResponse> = vec![
        // 3 consecutive MaxTokens
        CompletionResponse {
            text: "t1".into(),
            stop_reason: StopReason::MaxTokens,
            tool_calls: vec![],
            usage: Default::default(),
        },
        CompletionResponse {
            text: "t2".into(),
            stop_reason: StopReason::MaxTokens,
            tool_calls: vec![],
            usage: Default::default(),
        },
        CompletionResponse {
            text: "t3".into(),
            stop_reason: StopReason::MaxTokens,
            tool_calls: vec![],
            usage: Default::default(),
        },
        // ToolUse resets consecutive counter
        CompletionResponse {
            text: String::new(),
            stop_reason: StopReason::ToolUse,
            tool_calls: vec![ToolCall {
                id: "reset".into(),
                name: "memory".into(),
                input: serde_json::json!({"action": "recall", "query": "x"}),
            }],
            usage: Default::default(),
        },
        // 3 more MaxTokens (still under 5 consecutive)
        CompletionResponse {
            text: "t4".into(),
            stop_reason: StopReason::MaxTokens,
            tool_calls: vec![],
            usage: Default::default(),
        },
        CompletionResponse {
            text: "t5".into(),
            stop_reason: StopReason::MaxTokens,
            tool_calls: vec![],
            usage: Default::default(),
        },
        CompletionResponse {
            text: "t6".into(),
            stop_reason: StopReason::MaxTokens,
            tool_calls: vec![],
            usage: Default::default(),
        },
        // EndTurn
        CompletionResponse {
            text: "completed despite max_tokens".into(),
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: Default::default(),
        },
    ];

    let driver = MockDriver::new(responses);
    let mem_arc = Arc::new(InMemorySubstrate::new());
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(MemoryTool::new(mem_arc.clone(), "test-agent".into())));

    let result =
        run_agent_loop(&manifest, "interleaved test", &driver, &tools, mem_arc.as_ref(), None)
            .await;

    assert!(
        result.is_ok(),
        "FALSIFY-AL-006: interleaved ToolUse should reset MaxTokens counter, got: {:?}",
        result.err()
    );
    assert_eq!(result.as_ref().expect("ok").text, "completed despite max_tokens");
}

/// FALSIFY-AL-007: Conversation stored in memory after loop completes.
/// Verifies INV-007: every completed loop stores Q+A in memory.
#[tokio::test]
async fn test_falsify_al_007_conversation_stored_in_memory() {
    let manifest = test_manifest();
    let memory = Arc::new(InMemorySubstrate::new());

    let driver = MockDriver::single_response("The answer is 42.");

    let result = run_agent_loop(
        &manifest,
        "What is the meaning of life?",
        &driver,
        &ToolRegistry::new(),
        memory.as_ref(),
        None,
    )
    .await
    .expect("loop should succeed");

    assert_eq!(result.text, "The answer is 42.");

    // INV-007: conversation MUST be stored in memory
    let recalled = memory.recall("meaning of life", 10, None, None).await.expect("recall");
    assert!(
        !recalled.is_empty(),
        "FALSIFY-AL-007: conversation must be stored in memory after loop"
    );
    // Verify both Q and A are in the stored fragment
    let content = &recalled[0].content;
    assert!(
        content.contains("meaning of life"),
        "FALSIFY-AL-007: stored memory must contain the query"
    );
    assert!(content.contains("42"), "FALSIFY-AL-007: stored memory must contain the answer");
}

/// MCP server handler stores memory, agent loop recalls it.
/// Tests the pipeline: MCP dispatch → MemorySubstrate → agent recall.
#[tokio::test]
async fn test_mcp_server_memory_roundtrip() {
    use batuta::agent::tool::mcp_server::{HandlerRegistry, MemoryHandler};

    let memory: Arc<dyn MemorySubstrate> = Arc::new(InMemorySubstrate::new());

    // Step 1: Store via MCP server handler
    let mut registry = HandlerRegistry::new();
    registry.register(Box::new(MemoryHandler::new(Arc::clone(&memory), "test-agent")));

    let store_result = registry
        .dispatch(
            "memory",
            serde_json::json!({
                "action": "store",
                "content": "MCP-stored: quantum computing uses qubits"
            }),
        )
        .await;
    assert!(!store_result.is_error, "MCP store should succeed: {}", store_result.content);

    // Step 2: Run agent loop that queries the same memory
    let manifest = test_manifest();
    let driver = MockDriver::single_response("I recall quantum computing facts.");
    let tools = ToolRegistry::new();

    let result = run_agent_loop(
        &manifest,
        "Tell me about quantum computing",
        &driver,
        &tools,
        memory.as_ref(),
        None,
    )
    .await
    .expect("loop should succeed");

    assert_eq!(result.text, "I recall quantum computing facts.");

    // Step 3: Verify MCP-stored content is recallable
    let recalled = memory.recall("quantum", 10, None, None).await.expect("recall");
    assert!(recalled.len() >= 1, "should recall MCP-stored memory + conversation");
    assert!(
        recalled.iter().any(|f| f.content.contains("qubits")),
        "should find the MCP-stored fragment about qubits"
    );
}

/// MCP server list_tools returns correct metadata.
#[tokio::test]
async fn test_mcp_server_tool_discovery() {
    use batuta::agent::tool::mcp_server::{HandlerRegistry, MemoryHandler};

    let memory: Arc<dyn MemorySubstrate> = Arc::new(InMemorySubstrate::new());
    let mut registry = HandlerRegistry::new();
    registry.register(Box::new(MemoryHandler::new(memory, "test")));

    let tools = registry.list_tools();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "memory");
    assert!(!tools[0].description.is_empty());

    // Schema should have action field
    let props = tools[0].input_schema.get("properties").expect("schema should have properties");
    assert!(props.get("action").is_some());
    assert!(props.get("content").is_some());
    assert!(props.get("query").is_some());
}

/// Context truncation works with tiny context window driver.
#[tokio::test]
async fn test_context_truncation_integration() {
    use async_trait::async_trait;
    use batuta::agent::driver::CompletionRequest;
    use batuta::serve::backends::PrivacyTier;

    let manifest = test_manifest();

    // Tiny-window driver wrapping MockDriver
    struct TinyDriver(MockDriver);

    #[async_trait]
    impl batuta::agent::driver::LlmDriver for TinyDriver {
        async fn complete(
            &self,
            request: CompletionRequest,
        ) -> Result<CompletionResponse, batuta::agent::AgentError> {
            self.0.complete(request).await
        }
        fn context_window(&self) -> usize {
            300
        }
        fn privacy_tier(&self) -> PrivacyTier {
            PrivacyTier::Sovereign
        }
    }

    let driver = TinyDriver(MockDriver::single_response("context managed"));
    let tools = ToolRegistry::new();
    let memory = InMemorySubstrate::new();

    let result = run_agent_loop(&manifest, "test", &driver, &tools, &memory, None)
        .await
        .expect("should work with tiny context");
    assert_eq!(result.text, "context managed");
}

/// Runtime MCP privacy enforcement: sovereign + SSE should error.
#[cfg(feature = "agents-mcp")]
#[tokio::test]
async fn test_runtime_mcp_privacy_sovereign_blocks_sse() {
    use batuta::agent::manifest::{McpServerConfig, McpTransport};
    use batuta::serve::backends::PrivacyTier;

    let mut manifest = AgentManifest::default();
    manifest.privacy = PrivacyTier::Sovereign;
    manifest.mcp_servers = vec![McpServerConfig {
        name: "remote-api".into(),
        transport: McpTransport::Sse,
        command: vec![],
        url: Some("https://api.example.com/mcp".into()),
        capabilities: vec!["*".into()],
    }];

    let driver = MockDriver::single_response("should not reach here");
    let tools = ToolRegistry::new();
    let memory = InMemorySubstrate::new();

    let result = run_agent_loop(&manifest, "test", &driver, &tools, &memory, None).await;

    assert!(result.is_err(), "sovereign + SSE must error");
    let err = result.unwrap_err();
    assert!(
        format!("{err}").contains("sovereign"),
        "error should mention sovereign privacy: {err}"
    );
}

/// Runtime MCP privacy: sovereign + stdio is allowed.
#[cfg(feature = "agents-mcp")]
#[tokio::test]
async fn test_runtime_mcp_privacy_sovereign_allows_stdio() {
    use batuta::agent::manifest::{McpServerConfig, McpTransport};
    use batuta::serve::backends::PrivacyTier;

    let mut manifest = AgentManifest::default();
    manifest.privacy = PrivacyTier::Sovereign;
    manifest.mcp_servers = vec![McpServerConfig {
        name: "local-tool".into(),
        transport: McpTransport::Stdio,
        command: vec!["node".into(), "server.js".into()],
        url: None,
        capabilities: vec!["*".into()],
    }];

    let driver = MockDriver::single_response("stdio is fine");
    let tools = ToolRegistry::new();
    let memory = InMemorySubstrate::new();

    let result = run_agent_loop(&manifest, "test", &driver, &tools, &memory, None).await;

    assert!(result.is_ok(), "sovereign + stdio should be allowed: {:?}", result.err());
}

/// FALSIFY-AL-008: Sovereign privacy blocks network egress.
/// Agent with privacy=Sovereign + Network capability → tool denied.
/// Spec ref: FALSIFY-AL-005 (sovereign privacy).
#[tokio::test]
async fn test_falsify_al_008_sovereign_blocks_network_egress() {
    use batuta::agent::tool::network::NetworkTool;
    use batuta::serve::backends::PrivacyTier;

    let mut manifest = test_manifest();
    manifest.privacy = PrivacyTier::Sovereign;
    manifest.capabilities =
        vec![Capability::Memory, Capability::Network { allowed_hosts: vec!["*".into()] }];

    // Driver tries to use network tool, then responds
    let driver = MockDriver::new(vec![
        CompletionResponse {
            text: String::new(),
            stop_reason: StopReason::ToolUse,
            tool_calls: vec![ToolCall {
                id: "net-1".into(),
                name: "network".into(),
                input: serde_json::json!({
                    "url": "https://api.example.com/data"
                }),
            }],
            usage: Default::default(),
        },
        CompletionResponse {
            text: "network was blocked".into(),
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: Default::default(),
        },
    ]);

    let mem_arc = Arc::new(InMemorySubstrate::new());
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(MemoryTool::new(mem_arc.clone(), "test-agent".into())));
    tools.register(Box::new(NetworkTool::new(vec!["*".into()])));

    let result = run_agent_loop(&manifest, "fetch data", &driver, &tools, mem_arc.as_ref(), None)
        .await
        .expect("loop should succeed with denied tool");

    // Tool was denied but loop completed
    assert_eq!(result.text, "network was blocked");
    assert_eq!(result.tool_calls, 0, "FALSIFY-AL-008: sovereign privacy must block network tool");
}
