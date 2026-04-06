//! `InferenceTool` — sub-model invocation for agent delegation.
//!
//! Allows an agent to run a secondary LLM completion via the
//! same driver, useful for chain-of-thought delegation or
//! specialized reasoning sub-tasks.

use async_trait::async_trait;
use std::sync::Arc;

use super::{Tool, ToolResult};
use crate::agent::capability::Capability;
use crate::agent::driver::{CompletionRequest, LlmDriver, Message, ToolDefinition};

/// Tool that runs a sub-inference via the agent's LLM driver.
pub struct InferenceTool {
    driver: Arc<dyn LlmDriver>,
    max_tokens: u32,
}

impl InferenceTool {
    /// Create a new `InferenceTool` with the given driver.
    pub fn new(driver: Arc<dyn LlmDriver>, max_tokens: u32) -> Self {
        Self { driver, max_tokens }
    }
}

#[async_trait]
impl Tool for InferenceTool {
    fn name(&self) -> &'static str {
        "inference"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "inference".into(),
            description: "Run a sub-inference completion for \
                          delegation or chain-of-thought reasoning"
                .into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The prompt to send for completion"
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "Optional system prompt override"
                    }
                },
                "required": ["prompt"]
            }),
        }
    }

    #[cfg_attr(
        feature = "agents-contracts",
        provable_contracts_macros::contract("agent-loop-v1", equation = "inference_timeout")
    )]
    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let Some(prompt) = input.get("prompt").and_then(|p| p.as_str()) else {
            return ToolResult::error("missing required field: prompt");
        };

        let system = input.get("system_prompt").and_then(|s| s.as_str()).map(String::from);

        let request = CompletionRequest {
            model: String::new(),
            messages: vec![Message::User(prompt.into())],
            max_tokens: self.max_tokens,
            temperature: 0.0,
            tools: vec![],
            system,
        };

        match self.driver.complete(request).await {
            Ok(response) => {
                if response.text.is_empty() {
                    ToolResult::error("inference returned empty response")
                } else {
                    ToolResult::success(response.text)
                }
            }
            Err(e) => ToolResult::error(format!("inference error: {e}")),
        }
    }

    fn required_capability(&self) -> Capability {
        Capability::Inference
    }

    fn timeout(&self) -> std::time::Duration {
        std::time::Duration::from_secs(300)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::driver::mock::MockDriver;

    #[test]
    fn test_inference_tool_definition() {
        let driver = Arc::new(MockDriver::single_response("ok"));
        let tool = InferenceTool::new(driver, 256);
        let def = tool.definition();
        assert_eq!(def.name, "inference");
        assert!(def.description.contains("sub-inference"));
        let props = def.input_schema.get("properties").expect("schema properties");
        assert!(props.get("prompt").is_some());
        assert!(props.get("system_prompt").is_some());
    }

    #[test]
    fn test_inference_tool_capability() {
        let driver = Arc::new(MockDriver::single_response("ok"));
        let tool = InferenceTool::new(driver, 256);
        assert_eq!(tool.required_capability(), Capability::Inference);
    }

    #[test]
    fn test_inference_tool_timeout() {
        let driver = Arc::new(MockDriver::single_response("ok"));
        let tool = InferenceTool::new(driver, 256);
        assert_eq!(tool.timeout(), std::time::Duration::from_secs(300),);
    }

    #[tokio::test]
    async fn test_inference_missing_prompt() {
        let driver = Arc::new(MockDriver::single_response("ok"));
        let tool = InferenceTool::new(driver, 256);
        let result = tool.execute(serde_json::json!({})).await;
        assert!(result.is_error);
        assert!(result.content.contains("missing"));
    }

    #[tokio::test]
    async fn test_inference_executes() {
        let driver = Arc::new(MockDriver::single_response("The answer is 42."));
        let tool = InferenceTool::new(driver, 256);
        let result = tool
            .execute(serde_json::json!({
                "prompt": "What is the meaning of life?"
            }))
            .await;
        assert!(!result.is_error);
        assert!(result.content.contains("42"));
    }

    #[tokio::test]
    async fn test_inference_with_system_prompt() {
        let driver = Arc::new(MockDriver::single_response("I am a math tutor."));
        let tool = InferenceTool::new(driver, 256);
        let result = tool
            .execute(serde_json::json!({
                "prompt": "Help me with algebra",
                "system_prompt": "You are a math tutor."
            }))
            .await;
        assert!(!result.is_error);
        assert!(result.content.contains("math tutor"));
    }
}
