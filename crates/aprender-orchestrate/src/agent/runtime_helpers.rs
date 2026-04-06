//! Helper functions for the agent runtime loop.
//!
//! Extracted from runtime.rs to keep module under 500-line threshold.
//! Contains: context truncation, retry logic, event emission, MCP validation.

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
    let mut result = Vec::with_capacity(truncated.len());
    let mut msg_idx = messages.len();
    for chat_msg in truncated.iter().rev() {
        while msg_idx > 0 {
            msg_idx -= 1;
            if messages[msg_idx].to_chat_message().content == chat_msg.content {
                result.push(messages[msg_idx].clone());
                break;
            }
        }
    }
    result.reverse();
    Ok(result)
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
