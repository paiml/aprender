//! Helper functions for the agent runtime loop.
//!
//! Extracted from runtime.rs to keep module under 500-line threshold.
//! Contains: context truncation, retry logic, event emission, MCP validation,
//! assistant-text retention.

use std::time::Duration;

use tokio::sync::mpsc;
use tracing::{instrument, warn};

use super::driver::{CompletionRequest, CompletionResponse, LlmDriver, Message, StreamEvent};
use super::result::AgentError;
use crate::serve::context::ContextManager;

/// Maximum retry attempts for retryable driver errors.
const MAX_RETRIES: u32 = 3;
/// Base delay for exponential backoff (milliseconds).
const RETRY_BASE_MS: u64 = 1000;

/// Truncate agent messages to fit within context window.
pub(super) fn truncate_messages(
    messages: &[Message],
    context: &ContextManager,
) -> Result<Vec<Message>, AgentError> {
    let chat_msgs: Vec<_> = messages.iter().map(Message::to_chat_message).collect();

    if context.fits(&chat_msgs) {
        return Ok(messages.to_vec());
    }

    let truncated = context.truncate(&chat_msgs).map_err(
        |crate::serve::context::ContextError::ExceedsLimit { tokens, limit }| {
            AgentError::ContextOverflow { required: tokens, available: limit }
        },
    )?;

    // Map truncated ChatMessages back to original Messages
    // by matching content. SlidingWindow keeps most recent,
    // so iterate from end of original list.
    let mut kept = Vec::with_capacity(truncated.len());
    let mut msg_idx = messages.len();
    for chat_msg in truncated.iter().rev() {
        while msg_idx > 0 {
            msg_idx -= 1;
            if messages[msg_idx].to_chat_message().content == chat_msg.content {
                kept.push(msg_idx);
                break;
            }
        }
    }
    kept.reverse();

    // #4599: the current turn's prompt (the last `User` message) is never evicted to make room
    // for its own tool results. Dropping it would answer a conversation that no longer asks
    // anything, with rc 0; refuse instead.
    if let Some(prompt) = messages.iter().rposition(|m| matches!(m, Message::User(_))) {
        if !kept.contains(&prompt) {
            return Err(AgentError::ContextOverflow {
                required: context.estimate_tokens(&chat_msgs[prompt..]),
                available: context.available_tokens(),
            });
        }
    }
    Ok(kept.into_iter().map(|i| messages[i].clone()).collect())
}

/// Retry `driver.complete()` with exponential backoff for retryable errors.
#[instrument(skip_all)]
pub(super) async fn call_with_retry(
    driver: &dyn LlmDriver,
    request: &CompletionRequest,
) -> Result<CompletionResponse, AgentError> {
    let mut last_err = None;
    for attempt in 0..=MAX_RETRIES {
        match driver.complete(request.clone()).await {
            Ok(response) => return Ok(response),
            Err(AgentError::Driver(ref e)) if e.is_retryable() => {
                warn!(
                    attempt = attempt + 1,
                    max = MAX_RETRIES,
                    error = %e,
                    "retryable driver error"
                );
                last_err = Some(AgentError::Driver(e.clone()));
                if attempt < MAX_RETRIES {
                    let delay = RETRY_BASE_MS * 2u64.pow(attempt);
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                }
            }
            Err(e) => return Err(e),
        }
    }
    Err(last_err.unwrap_or_else(|| AgentError::CircuitBreak("retry loop exhausted".into())))
}

pub(super) async fn emit(tx: Option<&mpsc::Sender<StreamEvent>>, event: StreamEvent) {
    if let Some(tx) = tx {
        let _ = tx.send(event).await;
    }
}

/// Validate MCP transports against privacy tier (Poka-Yoke).
/// Defense-in-depth: blocks SSE/WebSocket under Sovereign even if
/// `manifest.validate()` was skipped.
#[cfg(feature = "agents-mcp")]
pub(super) fn validate_mcp_privacy(
    manifest: &super::manifest::AgentManifest,
) -> Result<(), AgentError> {
    use crate::agent::manifest::McpTransport;
    if manifest.privacy != crate::serve::backends::PrivacyTier::Sovereign {
        return Ok(());
    }
    for server in &manifest.mcp_servers {
        if matches!(server.transport, McpTransport::Sse | McpTransport::WebSocket) {
            return Err(AgentError::CircuitBreak(format!(
                "sovereign privacy blocks network MCP transport for '{}'",
                server.name,
            )));
        }
    }
    Ok(())
}

/// CCPA-m296: sanitize an assistant turn's text before it enters multi-turn
/// history. Strips any lingering tool-call markup so a prior tool-using turn is
/// NEVER re-rendered as capability-breaking raw Markdown that re-primes prose
/// mode on the next turn.
///
/// Genuine text turns pass through unchanged. Turns whose entire content was
/// tool-call markup collapse to empty (the structured `AssistantToolUse` /
/// `ToolResult` messages already carry that turn's tool semantics into history).
///
/// Returns the trimmed prose with `<tool_call>...</tool_call>` / unclosed
/// `<tool_call>` / `<tool_result>...</tool_result>` spans removed.
pub(super) fn retain_assistant_text(text: &str) -> String {
    let mut out = text.to_string();

    // Remove well-formed <tool_call>...</tool_call> spans.
    out = strip_spans(&out, "<tool_call>", "</tool_call>");
    // Remove <tool_result>...</tool_result> spans (model sometimes echoes them).
    out = strip_spans(&out, "<tool_result>", "</tool_result>");

    // Remove an unclosed trailing <tool_call> (small models omit the close tag):
    // everything from the marker to end-of-string is tool-call residue, not prose.
    if let Some(pos) = out.find("<tool_call>") {
        out.truncate(pos);
    }

    out.trim().to_string()
}

/// Remove every `open..close` span (inclusive) from `text`. Spans without a
/// matching close are left untouched (handled separately by the caller).
fn strip_spans(text: &str, open: &str, close: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut cursor = text;
    loop {
        let Some(start) = cursor.find(open) else {
            out.push_str(cursor);
            break;
        };
        let after_open = &cursor[start + open.len()..];
        let Some(rel_end) = after_open.find(close) else {
            // No closing tag — leave the remainder for the caller to handle.
            out.push_str(cursor);
            break;
        };
        out.push_str(&cursor[..start]);
        cursor = &after_open[rel_end + close.len()..];
    }
    out
}
