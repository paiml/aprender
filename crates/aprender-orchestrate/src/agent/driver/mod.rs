//! LLM driver abstraction.
//!
//! Defines the `LlmDriver` trait — the interface between the agent
//! loop and LLM inference backends. The default implementation is
//! `RealizarDriver` (sovereign, local GGUF/APR inference).

#[cfg(feature = "native")]
pub mod apr_serve;
pub mod chat_template;
pub mod mock;
#[cfg(feature = "inference")]
pub mod realizar;
#[cfg(feature = "native")]
pub mod remote;
#[cfg(feature = "native")]
pub mod router;
pub mod validate;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::agent::phase::LoopPhase;
use crate::agent::result::{AgentError, StopReason, TokenUsage};
use crate::serve::backends::PrivacyTier;

/// Message in the agent conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Message {
    /// System prompt (injected once at start).
    System(String),
    /// User query or follow-up.
    User(String),
    /// Assistant text response.
    Assistant(String),
    /// Assistant tool use request.
    AssistantToolUse(ToolCall),
    /// Tool execution result.
    ToolResult(ToolResultMsg),
}

impl Message {
    /// Convert to `ChatMessage` for context window truncation.
    ///
    /// Tool-use and tool-result messages are serialized as
    /// assistant/user text so the token estimator can size them.
    pub fn to_chat_message(&self) -> crate::serve::templates::ChatMessage {
        use crate::serve::templates::ChatMessage;
        match self {
            Self::System(s) => ChatMessage::system(s),
            Self::User(s) => ChatMessage::user(s),
            Self::Assistant(s) => ChatMessage::assistant(s),
            Self::AssistantToolUse(call) => {
                ChatMessage::assistant(format!("[tool_use: {} {}]", call.name, call.input))
            }
            Self::ToolResult(result) => {
                ChatMessage::user(format!("[tool_result: {}]", result.content))
            }
        }
    }
}

/// A tool call request from the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique ID for this tool call.
    pub id: String,
    /// Tool name.
    pub name: String,
    /// Tool input as JSON.
    pub input: serde_json::Value,
}

/// A tool result message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultMsg {
    /// ID of the tool call this is responding to.
    pub tool_use_id: String,
    /// Result content.
    pub content: String,
    /// Whether the tool call errored.
    pub is_error: bool,
}

/// Tool definition for the LLM (JSON Schema).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool name (must match Tool trait name).
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// JSON Schema for the tool's input.
    pub input_schema: serde_json::Value,
}

/// Request to the LLM driver.
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    /// Model identifier.
    pub model: String,
    /// Conversation messages.
    pub messages: Vec<Message>,
    /// Available tools.
    pub tools: Vec<ToolDefinition>,
    /// Maximum tokens to generate.
    pub max_tokens: u32,
    /// Sampling temperature.
    pub temperature: f32,
    /// System prompt (separate from messages).
    pub system: Option<String>,
}

/// Response from the LLM driver.
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    /// Generated text (may be empty if only tool calls).
    pub text: String,
    /// Why the model stopped generating.
    pub stop_reason: StopReason,
    /// Tool calls requested by the model.
    pub tool_calls: Vec<ToolCall>,
    /// Token usage for this completion.
    pub usage: TokenUsage,
}

/// Streaming event from the LLM driver.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Agent loop phase changed.
    PhaseChange {
        /// New phase.
        phase: LoopPhase,
    },
    /// Incremental text from the model.
    TextDelta {
        /// Text fragment.
        text: String,
    },
    /// Tool call started.
    ToolUseStart {
        /// Tool call ID.
        id: String,
        /// Tool name.
        name: String,
    },
    /// Tool call completed.
    ToolUseEnd {
        /// Tool call ID.
        id: String,
        /// Tool name.
        name: String,
        /// Tool result.
        result: String,
    },
    /// Completion finished.
    ContentComplete {
        /// Stop reason.
        stop_reason: StopReason,
        /// Usage for this completion.
        usage: TokenUsage,
    },
}

/// Abstraction over LLM inference backends.
///
/// Default implementation: `RealizarDriver` (sovereign, local).
#[async_trait]
pub trait LlmDriver: Send + Sync {
    /// Non-streaming completion.
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, AgentError>;

    /// Streaming completion with channel-based events.
    ///
    /// Default wraps `complete()` for drivers that don't support
    /// native streaming. Override for token-by-token output.
    async fn stream(
        &self,
        request: CompletionRequest,
        tx: tokio::sync::mpsc::Sender<StreamEvent>,
    ) -> Result<CompletionResponse, AgentError> {
        let response = self.complete(request).await?;
        let _ = tx.send(StreamEvent::TextDelta { text: response.text.clone() }).await;
        let _ = tx
            .send(StreamEvent::ContentComplete {
                stop_reason: response.stop_reason.clone(),
                usage: response.usage.clone(),
            })
            .await;
        Ok(response)
    }

    /// Maximum context window in tokens.
    fn context_window(&self) -> usize;

    /// Privacy tier this driver operates at.
    fn privacy_tier(&self) -> PrivacyTier;

    /// Estimate cost in USD for a single completion's token usage.
    ///
    /// Default: 0.0 (sovereign/local inference is free).
    /// Remote drivers override with their pricing model.
    fn estimate_cost(&self, _usage: &TokenUsage) -> f64 {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_serialization() {
        let msgs = vec![
            Message::System("sys".into()),
            Message::User("hello".into()),
            Message::Assistant("hi".into()),
        ];
        for msg in &msgs {
            let json = serde_json::to_string(msg).expect("serialize failed");
            let back: Message = serde_json::from_str(&json).expect("deserialize failed");
            match (msg, &back) {
                (Message::System(a), Message::System(b)) => {
                    assert_eq!(a, b);
                }
                (Message::User(a), Message::User(b)) => assert_eq!(a, b),
                (Message::Assistant(a), Message::Assistant(b)) => {
                    assert_eq!(a, b);
                }
                _ => panic!("mismatch"),
            }
        }
    }

    #[test]
    fn test_tool_call_serialization() {
        let call = ToolCall {
            id: "1".into(),
            name: "rag".into(),
            input: serde_json::json!({"query": "test"}),
        };
        let json = serde_json::to_string(&call).expect("serialize failed");
        let back: ToolCall = serde_json::from_str(&json).expect("deserialize failed");
        assert_eq!(back.name, "rag");
    }

    #[test]
    fn test_tool_definition_serialization() {
        let def = ToolDefinition {
            name: "memory".into(),
            description: "Read/write memory".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {"type": "string"}
                }
            }),
        };
        let json = serde_json::to_string(&def).expect("serialize failed");
        assert!(json.contains("memory"));
    }

    #[tokio::test]
    async fn test_stream_default_wraps_complete() {
        use crate::agent::driver::mock::MockDriver;
        use tokio::sync::mpsc;

        let driver = MockDriver::single_response("streamed");
        let (tx, mut rx) = mpsc::channel(16);

        let request = CompletionRequest {
            model: String::new(),
            messages: vec![Message::User("hi".into())],
            tools: vec![],
            max_tokens: 100,
            temperature: 0.5,
            system: None,
        };

        let response = driver.stream(request, tx).await.expect("stream failed");
        assert_eq!(response.text, "streamed");

        let mut got_text = false;
        let mut got_complete = false;
        while let Ok(event) = rx.try_recv() {
            match event {
                StreamEvent::TextDelta { text } => {
                    assert_eq!(text, "streamed");
                    got_text = true;
                }
                StreamEvent::ContentComplete { .. } => {
                    got_complete = true;
                }
                _ => {}
            }
        }
        assert!(got_text, "expected TextDelta event");
        assert!(got_complete, "expected ContentComplete event");
    }
}
