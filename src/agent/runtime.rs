//! Agent runtime — the core perceive-reason-act loop.
//!
//! Orchestrates perceive (recall) → reason (LLM) → act (tools) → guard (Jidoka).
//! See: arXiv:2512.10350 (loop dynamics), arXiv:2501.09136 (agentic RAG).

use std::sync::Arc;

use tokio::sync::mpsc;
use tracing::{debug, info, instrument, warn};

use super::capability::capability_matches;
use super::driver::{
    CompletionRequest, CompletionResponse, LlmDriver, Message, StreamEvent, ToolCall, ToolResultMsg,
};
use super::guard::{LoopGuard, LoopVerdict};
use super::manifest::AgentManifest;
use super::memory::{MemorySource, MemorySubstrate};
use super::phase::LoopPhase;
use super::result::{AgentError, AgentLoopResult, StopReason};
use super::runtime_helpers::{call_with_retry, emit, truncate_messages};
use super::tool::ToolRegistry;
use crate::serve::context::{
    ContextConfig, ContextManager, ContextWindow, TokenEstimator, TruncationStrategy,
};

/// Run the agent loop to completion (single-turn, no history).
#[instrument(skip_all, fields(agent = %manifest.name, query_len = query.len()))]
#[cfg_attr(
    feature = "agents-contracts",
    provable_contracts_macros::contract("agent-loop-v1", equation = "loop_termination")
)]
pub async fn run_agent_loop(
    manifest: &AgentManifest,
    query: &str,
    driver: &dyn LlmDriver,
    tools: &ToolRegistry,
    memory: &dyn MemorySubstrate,
    stream_tx: Option<mpsc::Sender<StreamEvent>>,
) -> Result<AgentLoopResult, AgentError> {
    let mut history = Vec::new();
    run_agent_turn(manifest, &mut history, query, driver, tools, memory, stream_tx).await
}

/// PMAT-177: Agent loop with tool-use nudge. If first turn has no tool calls, retries once.
pub async fn run_agent_loop_with_nudge(
    manifest: &AgentManifest,
    query: &str,
    driver: &dyn LlmDriver,
    tools: &ToolRegistry,
    memory: &dyn MemorySubstrate,
    stream_tx: Option<mpsc::Sender<StreamEvent>>,
) -> Result<AgentLoopResult, AgentError> {
    let mut history = Vec::new();
    let r = run_agent_turn(manifest, &mut history, query, driver, tools, memory, stream_tx.clone())
        .await?;
    if r.tool_calls == 0 && tools.len() > 0 {
        info!("no tool calls on first turn, nudging");
        let nudge =
            "Use a tool to answer. Emit a <tool_call> block with glob, file_read, or shell.";
        return run_agent_turn(manifest, &mut history, nudge, driver, tools, memory, stream_tx)
            .await;
    }
    Ok(r)
}

/// Run one agent turn with full conversation history (multi-turn).
///
/// Appends the new query and all resulting messages (tool calls, assistant
/// response) to `history`. On next call, prior turns provide context so
/// the agent can maintain coherent multi-turn conversation.
///
/// This is the core primitive for the REPL — each user prompt is one turn,
/// and `history` accumulates across the session.
#[instrument(skip_all, fields(agent = %manifest.name, query_len = query.len(), history_len = history.len()))]
pub async fn run_agent_turn(
    manifest: &AgentManifest,
    history: &mut Vec<Message>,
    query: &str,
    driver: &dyn LlmDriver,
    tools: &ToolRegistry,
    memory: &dyn MemorySubstrate,
    stream_tx: Option<mpsc::Sender<StreamEvent>>,
) -> Result<AgentLoopResult, AgentError> {
    // ═══ PRIVACY GATE (Poka-Yoke) ═══
    // Defense-in-depth: block non-local MCP transports under Sovereign tier.
    #[cfg(feature = "agents-mcp")]
    validate_mcp_privacy(manifest)?;

    let mut guard = LoopGuard::new(
        manifest.resources.max_iterations,
        manifest.resources.max_tool_calls,
        manifest.resources.max_cost_usd,
    )
    .with_token_budget(manifest.resources.max_tokens_budget);

    // ═══ PERCEIVE ═══
    emit(stream_tx.as_ref(), StreamEvent::PhaseChange { phase: LoopPhase::Perceive }).await;

    let system = build_system_prompt(manifest, query, memory).await;
    let tool_defs = tools.definitions_for(&manifest.capabilities);
    info!(
        tools = tool_defs.len(),
        capabilities = manifest.capabilities.len(),
        history_messages = history.len(),
        "agent turn initialized"
    );
    let context = build_context(driver, &system, &tool_defs, manifest);

    // Build messages: prior history + new user query
    let mut messages = history.clone();
    messages.push(Message::User(query.to_string()));

    let mut last_tool_sig: Option<String> = None; // PMAT-172: stuck-loop detection
    let mut repeat_count: u32 = 0;

    loop {
        check_verdict(guard.check_iteration())?;
        debug!(
            iteration = guard.current_iteration(),
            tool_calls = guard.total_tool_calls(),
            "loop iteration start"
        );

        // ═══ REASON ═══
        emit(stream_tx.as_ref(), StreamEvent::PhaseChange { phase: LoopPhase::Reason }).await;

        let response =
            reason_step(driver, &messages, &tool_defs, manifest, &system, &context).await?;
        check_verdict(guard.record_usage(&response.usage))?;

        // INV-005: Estimate cost and enforce budget (Muda)
        let cost = driver.estimate_cost(&response.usage);
        check_verdict(guard.record_cost(cost))?;

        match response.stop_reason {
            StopReason::EndTurn | StopReason::StopSequence => {
                info!(
                    iterations = guard.current_iteration(),
                    tool_calls = guard.total_tool_calls(),
                    "turn complete"
                );
                let new_start = history.len();
                for msg in &messages[new_start..] {
                    history.push(msg.clone());
                }
                if !response.text.is_empty() {
                    history.push(Message::Assistant(response.text.clone()));
                }
                return finish_loop(&response, &guard, manifest, query, memory, stream_tx.as_ref())
                    .await;
            }
            StopReason::ToolUse => {
                // PMAT-172: detect stuck loops (4+ identical tool calls → break)
                let sig = response.tool_calls.first().map(|tc| format!("{}:{}", tc.name, tc.input));
                if sig == last_tool_sig {
                    repeat_count += 1;
                } else {
                    last_tool_sig = sig;
                    repeat_count = 1;
                }
                if repeat_count >= 4 {
                    warn!("stuck loop: same tool call repeated {repeat_count} times");
                    return finish_loop(
                        &response,
                        &guard,
                        manifest,
                        query,
                        memory,
                        stream_tx.as_ref(),
                    )
                    .await;
                }
                debug!(num_calls = response.tool_calls.len(), "processing tool calls");
                guard.reset_max_tokens();
                handle_tool_calls(
                    &response,
                    &mut messages,
                    &mut guard,
                    manifest,
                    tools,
                    stream_tx.as_ref(),
                )
                .await?;
            }
            StopReason::MaxTokens => {
                warn!("max tokens reached, continuing loop");
                check_verdict(guard.record_max_tokens())?;
                messages.push(Message::Assistant(response.text));
            }
        }
    }
}

fn check_verdict(verdict: LoopVerdict) -> Result<(), AgentError> {
    match verdict {
        LoopVerdict::CircuitBreak(msg) | LoopVerdict::Block(msg) => {
            Err(AgentError::CircuitBreak(msg))
        }
        LoopVerdict::Allow | LoopVerdict::Warn(_) => Ok(()),
    }
}

async fn reason_step(
    driver: &dyn LlmDriver,
    messages: &[Message],
    tool_defs: &[super::driver::ToolDefinition],
    manifest: &AgentManifest,
    system: &str,
    context: &ContextManager,
) -> Result<CompletionResponse, AgentError> {
    let truncated_messages = truncate_messages(messages, context)?;

    let request = CompletionRequest {
        model: String::new(),
        messages: truncated_messages,
        tools: tool_defs.to_vec(),
        max_tokens: manifest.model.max_tokens,
        temperature: manifest.model.temperature,
        system: Some(system.to_string()),
    };

    call_with_retry(driver, &request).await
}

async fn finish_loop(
    response: &CompletionResponse,
    guard: &LoopGuard,
    manifest: &AgentManifest,
    query: &str,
    memory: &dyn MemorySubstrate,
    stream_tx: Option<&mpsc::Sender<StreamEvent>>,
) -> Result<AgentLoopResult, AgentError> {
    let _ = memory
        .remember(
            &manifest.name,
            &format!("Q: {query}\nA: {}", response.text),
            MemorySource::Conversation,
            None,
        )
        .await;

    emit(stream_tx, StreamEvent::PhaseChange { phase: LoopPhase::Done }).await;

    Ok(AgentLoopResult {
        text: response.text.clone(),
        usage: guard.usage().clone(),
        iterations: guard.current_iteration(),
        tool_calls: guard.total_tool_calls(),
    })
}

async fn build_system_prompt(
    manifest: &AgentManifest,
    query: &str,
    memory: &dyn MemorySubstrate,
) -> String {
    let memories = memory.recall(query, 5, None, None).await.unwrap_or_default();

    let mut system = manifest.model.system_prompt.clone();
    if !memories.is_empty() {
        use std::fmt::Write;
        system.push_str("\n\n## Recalled Context\n");
        for m in &memories {
            let _ = writeln!(system, "- {}", m.content);
        }
    }
    system
}

fn build_context(
    driver: &dyn LlmDriver,
    system: &str,
    tool_defs: &[super::driver::ToolDefinition],
    manifest: &AgentManifest,
) -> ContextManager {
    let estimator = TokenEstimator::new();
    let system_tokens = estimator.estimate(system);
    let tool_json = serde_json::to_string(tool_defs).unwrap_or_default();
    let tool_tokens = estimator.estimate(&tool_json);
    let context_window = driver.context_window();
    let effective_window = context_window.saturating_sub(system_tokens).saturating_sub(tool_tokens);
    ContextManager::new(ContextConfig {
        window: ContextWindow::new(effective_window, manifest.model.max_tokens as usize),
        strategy: TruncationStrategy::SlidingWindow,
        preserve_system: false,
        min_messages: 2,
    })
}

/// Process tool calls from a completion response.
#[instrument(skip_all, fields(num_calls = response.tool_calls.len()))]
async fn handle_tool_calls(
    response: &CompletionResponse,
    messages: &mut Vec<Message>,
    guard: &mut LoopGuard,
    manifest: &AgentManifest,
    tools: &ToolRegistry,
    stream_tx: Option<&mpsc::Sender<StreamEvent>>,
) -> Result<(), AgentError> {
    for call in &response.tool_calls {
        let Some(tool) = tools.get(&call.name) else {
            push_tool_error(messages, call, &format!("unknown tool: {}", call.name));
            continue;
        };

        // Poka-Yoke: capability check
        let cap = tool.required_capability();
        if !capability_matches(&manifest.capabilities, &cap) {
            push_tool_error(messages, call, &format!("capability denied for tool '{}'", call.name));
            continue;
        }

        // Poka-Yoke: sovereign privacy blocks network egress
        if manifest.privacy == crate::serve::backends::PrivacyTier::Sovereign
            && matches!(cap, super::capability::Capability::Network { .. })
        {
            push_tool_error(messages, call, "sovereign privacy blocks network egress");
            continue;
        }

        // Jidoka: loop guard check
        match guard.check_tool_call(&call.name, &call.input) {
            LoopVerdict::Allow | LoopVerdict::Warn(_) => {}
            LoopVerdict::Block(msg) => {
                push_tool_error(messages, call, &msg);
                continue;
            }
            LoopVerdict::CircuitBreak(msg) => {
                return Err(AgentError::CircuitBreak(msg));
            }
        }

        // ═══ ACT ═══
        let result = execute_tool(call, tool, stream_tx).await;

        messages.push(Message::AssistantToolUse(ToolCall {
            id: call.id.clone(),
            name: call.name.clone(),
            input: call.input.clone(),
        }));
        messages.push(Message::ToolResult(ToolResultMsg {
            tool_use_id: call.id.clone(),
            content: result.content,
            is_error: result.is_error,
        }));
    }
    Ok(())
}

/// Execute a single tool call with tracing, timeout, and sanitization.
async fn execute_tool(
    call: &ToolCall,
    tool: &dyn super::tool::Tool,
    stream_tx: Option<&mpsc::Sender<StreamEvent>>,
) -> super::tool::ToolResult {
    let tool_span = tracing::info_span!(
        "tool_execute",
        tool = %call.name,
        id = %call.id,
    );
    let _enter = tool_span.enter();

    emit(
        stream_tx,
        StreamEvent::PhaseChange { phase: LoopPhase::Act { tool_name: call.name.clone() } },
    )
    .await;

    emit(stream_tx, StreamEvent::ToolUseStart { id: call.id.clone(), name: call.name.clone() })
        .await;

    let result = tokio::time::timeout(tool.timeout(), tool.execute(call.input.clone()))
        .await
        .unwrap_or_else(|elapsed| {
            warn!(tool = %call.name, timeout = ?elapsed, "tool execution timed out");
            super::tool::ToolResult::error(format!(
                "tool '{}' timed out after {:?}",
                call.name, elapsed
            ))
        })
        .sanitized(); // Poka-Yoke: strip injection patterns from tool output

    debug!(
        tool = %call.name,
        is_error = result.is_error,
        output_len = result.content.len(),
        "tool execution complete"
    );

    emit(
        stream_tx,
        StreamEvent::ToolUseEnd {
            id: call.id.clone(),
            name: call.name.clone(),
            result: result.content.clone(),
        },
    )
    .await;

    result
}

fn push_tool_error(messages: &mut Vec<Message>, call: &ToolCall, error: &str) {
    messages.push(Message::AssistantToolUse(ToolCall {
        id: call.id.clone(),
        name: call.name.clone(),
        input: call.input.clone(),
    }));
    messages.push(Message::ToolResult(ToolResultMsg {
        tool_use_id: call.id.clone(),
        content: error.to_string(),
        is_error: true,
    }));
}

// truncate_messages, call_with_retry, emit, validate_mcp_privacy
// extracted to runtime_helpers.rs (PMAT-190: keep under 500-line threshold)
#[cfg(feature = "agents-mcp")]
use super::runtime_helpers::validate_mcp_privacy;
#[cfg(test)]
#[path = "runtime_tests.rs"]
mod tests;
#[cfg(test)]
#[path = "runtime_tests_advanced.rs"]
mod tests_advanced;
#[cfg(test)]
#[path = "runtime_tests_guards.rs"]
mod tests_guards;
#[cfg(test)]
#[path = "runtime_tests_multi_turn.rs"]
mod tests_multi_turn;
