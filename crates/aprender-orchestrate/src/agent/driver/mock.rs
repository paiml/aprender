//! Mock LLM driver for deterministic testing.
//!
//! Returns pre-configured responses in sequence. Essential for
//! testing the agent loop without actual model inference.

use async_trait::async_trait;
use std::sync::Mutex;

use super::{CompletionRequest, CompletionResponse, LlmDriver, ToolCall};
use crate::agent::result::{AgentError, StopReason, TokenUsage};
use crate::serve::backends::PrivacyTier;

/// Mock driver that returns pre-configured responses.
pub struct MockDriver {
    responses: Mutex<Vec<CompletionResponse>>,
    context_window: usize,
    /// Cost per token (input + output) for testing cost budgets.
    cost_per_token: f64,
}

impl MockDriver {
    /// Create a mock driver with a sequence of responses.
    ///
    /// Responses are returned in order. If exhausted, returns
    /// a default "end of mock responses" response.
    pub fn new(responses: Vec<CompletionResponse>) -> Self {
        Self { responses: Mutex::new(responses), context_window: 4096, cost_per_token: 0.0 }
    }

    /// Create a mock that returns a single text response.
    pub fn single_response(text: &str) -> Self {
        Self::new(vec![CompletionResponse {
            text: text.to_string(),
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: TokenUsage { input_tokens: 10, output_tokens: 5 },
        }])
    }

    /// Create a mock that first requests a tool call, then responds.
    pub fn tool_then_response(
        tool_name: &str,
        tool_input: serde_json::Value,
        final_text: &str,
    ) -> Self {
        Self::new(vec![
            CompletionResponse {
                text: String::new(),
                stop_reason: StopReason::ToolUse,
                tool_calls: vec![ToolCall {
                    id: "mock-1".into(),
                    name: tool_name.to_string(),
                    input: tool_input,
                }],
                usage: TokenUsage { input_tokens: 10, output_tokens: 5 },
            },
            CompletionResponse {
                text: final_text.to_string(),
                stop_reason: StopReason::EndTurn,
                tool_calls: vec![],
                usage: TokenUsage { input_tokens: 20, output_tokens: 10 },
            },
        ])
    }

    /// Set context window size.
    #[must_use]
    pub fn with_context_window(mut self, size: usize) -> Self {
        self.context_window = size;
        self
    }

    /// Set cost per token for testing cost budget enforcement.
    #[must_use]
    pub fn with_cost_per_token(mut self, cost: f64) -> Self {
        self.cost_per_token = cost;
        self
    }
}

#[async_trait]
impl LlmDriver for MockDriver {
    async fn complete(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse, AgentError> {
        let mut responses = self.responses.lock().map_err(|e| {
            AgentError::Driver(crate::agent::result::DriverError::InferenceFailed(format!(
                "mock lock poisoned: {e}"
            )))
        })?;

        if responses.is_empty() {
            Ok(CompletionResponse {
                text: "[mock exhausted]".into(),
                stop_reason: StopReason::EndTurn,
                tool_calls: vec![],
                usage: TokenUsage::default(),
            })
        } else {
            Ok(responses.remove(0))
        }
    }

    fn context_window(&self) -> usize {
        self.context_window
    }

    fn privacy_tier(&self) -> PrivacyTier {
        PrivacyTier::Sovereign
    }

    #[allow(clippy::cast_precision_loss)] // token counts fit in f64 mantissa
    fn estimate_cost(&self, usage: &TokenUsage) -> f64 {
        self.cost_per_token * usage.total() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_single_response() {
        let driver = MockDriver::single_response("hello world");
        let req = CompletionRequest {
            model: "mock".into(),
            messages: vec![],
            tools: vec![],
            max_tokens: 100,
            temperature: 0.0,
            system: None,
        };

        let resp = driver.complete(req).await.expect("complete failed");
        assert_eq!(resp.text, "hello world");
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
        assert!(resp.tool_calls.is_empty());
    }

    #[tokio::test]
    async fn test_sequenced_responses() {
        let driver = MockDriver::new(vec![
            CompletionResponse {
                text: "first".into(),
                stop_reason: StopReason::EndTurn,
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
            CompletionResponse {
                text: "second".into(),
                stop_reason: StopReason::EndTurn,
                tool_calls: vec![],
                usage: TokenUsage::default(),
            },
        ]);

        let req = CompletionRequest {
            model: "mock".into(),
            messages: vec![],
            tools: vec![],
            max_tokens: 100,
            temperature: 0.0,
            system: None,
        };

        let r1 = driver.complete(req.clone()).await.expect("first failed");
        assert_eq!(r1.text, "first");

        let r2 = driver.complete(req).await.expect("second failed");
        assert_eq!(r2.text, "second");
    }

    #[tokio::test]
    async fn test_exhausted_responses() {
        let driver = MockDriver::new(vec![]);
        let req = CompletionRequest {
            model: "mock".into(),
            messages: vec![],
            tools: vec![],
            max_tokens: 100,
            temperature: 0.0,
            system: None,
        };

        let resp = driver.complete(req).await.expect("complete failed");
        assert_eq!(resp.text, "[mock exhausted]");
    }

    #[tokio::test]
    async fn test_tool_call_response() {
        let driver = MockDriver::tool_then_response(
            "rag",
            serde_json::json!({"query": "test"}),
            "final answer",
        );

        let req = CompletionRequest {
            model: "mock".into(),
            messages: vec![],
            tools: vec![],
            max_tokens: 100,
            temperature: 0.0,
            system: None,
        };

        let r1 = driver.complete(req.clone()).await.expect("first failed");
        assert_eq!(r1.stop_reason, StopReason::ToolUse);
        assert_eq!(r1.tool_calls.len(), 1);
        assert_eq!(r1.tool_calls[0].name, "rag");

        let r2 = driver.complete(req).await.expect("second failed");
        assert_eq!(r2.text, "final answer");
        assert_eq!(r2.stop_reason, StopReason::EndTurn);
    }

    #[test]
    fn test_context_window() {
        let driver = MockDriver::single_response("hi");
        assert_eq!(driver.context_window(), 4096);

        let driver = driver.with_context_window(8192);
        assert_eq!(driver.context_window(), 8192);
    }

    #[test]
    fn test_privacy_tier_sovereign() {
        let driver = MockDriver::single_response("hi");
        assert_eq!(driver.privacy_tier(), PrivacyTier::Sovereign);
    }
}
