//! Autonomous Agent Runtime (perceive-reason-act loop).
//!
//! Implements a sovereign agent that uses local LLM inference
//! (realizar), RAG retrieval (trueno-rag), and persistent memory
//! (trueno-db) — all running locally with zero API dependencies.
//!
//! # Architecture
//!
//! ```text
//! AgentManifest (TOML)
//!   → PERCEIVE: recall memories
//!   → REASON:   LlmDriver.complete()
//!   → ACT:      Tool.execute()
//!   → repeat until Done or guard triggers
//! ```
//!
//! # Toyota Production System Principles
//!
//! - **Jidoka**: `LoopGuard` stops on ping-pong, budget, max iterations
//! - **Poka-Yoke**: Capability system prevents unauthorized tool access
//! - **Muda**: `CostCircuitBreaker` prevents runaway spend
//! - **Genchi Genbutsu**: Default sovereign — local hardware, no proxies
//!
//! # References
//!
//! - arXiv:2512.10350 — Geometric dynamics of agentic loops
//! - arXiv:2501.09136 — Agentic RAG survey
//! - arXiv:2406.09187 — `GuardAgent` safety

pub mod capability;
pub mod code;
mod code_prompts;
pub mod contracts;
pub mod driver;
pub mod guard;
pub mod manifest;
pub mod memory;
pub mod phase;
pub mod pool;
pub mod repl;
mod repl_display;
pub mod result;
pub mod runtime;
mod runtime_helpers;
pub mod session;
pub mod signing;
pub mod tool;
pub mod tui;

// Re-export key types for convenience.
pub use capability::{capability_matches, Capability};
pub use guard::{LoopGuard, LoopVerdict};
pub use manifest::{AgentManifest, AutoPullError, ModelConfig, ResourceQuota};
pub use memory::InMemorySubstrate;
pub use phase::LoopPhase;
pub use pool::{AgentId, AgentMessage, AgentPool, MessageRouter, SpawnConfig, ToolBuilder};
pub use result::{AgentError, AgentLoopResult, DriverError, StopReason, TokenUsage};

use driver::{LlmDriver, StreamEvent};
use memory::MemorySubstrate;
use tokio::sync::mpsc;
use tool::ToolRegistry;

/// Ergonomic builder for constructing and running agent loops.
///
/// ```rust,ignore
/// let result = AgentBuilder::new(manifest)
///     .driver(&my_driver)
///     .tool(Box::new(rag_tool))
///     .memory(&substrate)
///     .run("What is SIMD?")
///     .await?;
/// ```
pub struct AgentBuilder<'a> {
    manifest: &'a AgentManifest,
    driver: Option<&'a dyn LlmDriver>,
    tools: ToolRegistry,
    memory: Option<&'a dyn MemorySubstrate>,
    stream_tx: Option<mpsc::Sender<StreamEvent>>,
}

impl<'a> AgentBuilder<'a> {
    /// Create a new builder from an agent manifest.
    pub fn new(manifest: &'a AgentManifest) -> Self {
        Self { manifest, driver: None, tools: ToolRegistry::new(), memory: None, stream_tx: None }
    }

    /// Set the LLM driver for inference.
    #[must_use]
    pub fn driver(mut self, driver: &'a dyn LlmDriver) -> Self {
        self.driver = Some(driver);
        self
    }

    /// Register a tool in the tool registry.
    #[must_use]
    pub fn tool(mut self, tool: Box<dyn tool::Tool>) -> Self {
        self.tools.register(tool);
        self
    }

    /// Set the memory substrate.
    #[must_use]
    pub fn memory(mut self, memory: &'a dyn MemorySubstrate) -> Self {
        self.memory = Some(memory);
        self
    }

    /// Set the stream event channel for real-time events.
    #[must_use]
    pub fn stream(mut self, tx: mpsc::Sender<StreamEvent>) -> Self {
        self.stream_tx = Some(tx);
        self
    }

    /// Run the agent loop with the given query.
    ///
    /// Uses `InMemorySubstrate` if no memory was provided.
    pub async fn run(self, query: &str) -> Result<AgentLoopResult, AgentError> {
        let driver = self
            .driver
            .ok_or_else(|| AgentError::ManifestError("no LLM driver configured".into()))?;

        let default_memory = InMemorySubstrate::new();
        let memory = self.memory.unwrap_or(&default_memory);

        runtime::run_agent_loop(self.manifest, query, driver, &self.tools, memory, self.stream_tx)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use driver::mock::MockDriver;

    #[tokio::test]
    async fn test_builder_minimal() {
        let manifest = AgentManifest::default();
        let driver = MockDriver::single_response("built!");

        let result = AgentBuilder::new(&manifest)
            .driver(&driver)
            .run("hello")
            .await
            .expect("builder run failed");

        assert_eq!(result.text, "built!");
    }

    #[tokio::test]
    async fn test_builder_no_driver_errors() {
        let manifest = AgentManifest::default();

        let err = AgentBuilder::new(&manifest).run("hello").await.unwrap_err();

        assert!(matches!(err, AgentError::ManifestError(_)), "expected ManifestError, got: {err}");
    }

    #[tokio::test]
    async fn test_builder_with_memory() {
        let manifest = AgentManifest::default();
        let driver = MockDriver::single_response("remembered");
        let memory = InMemorySubstrate::new();

        let result = AgentBuilder::new(&manifest)
            .driver(&driver)
            .memory(&memory)
            .run("test")
            .await
            .expect("builder run failed");

        assert_eq!(result.text, "remembered");
    }

    #[tokio::test]
    async fn test_builder_with_stream() {
        let manifest = AgentManifest::default();
        let driver = MockDriver::single_response("streamed");
        let (tx, mut rx) = mpsc::channel(32);

        let result = AgentBuilder::new(&manifest)
            .driver(&driver)
            .stream(tx)
            .run("test")
            .await
            .expect("builder run failed");

        assert_eq!(result.text, "streamed");

        let mut got_events = false;
        while let Ok(_event) = rx.try_recv() {
            got_events = true;
        }
        assert!(got_events, "expected stream events");
    }

    #[tokio::test]
    async fn test_builder_with_tool() {
        use crate::agent::driver::ToolDefinition;
        use crate::agent::tool::ToolResult as TResult;

        struct DummyTool;

        #[async_trait::async_trait]
        impl tool::Tool for DummyTool {
            fn name(&self) -> &'static str {
                "dummy"
            }
            fn definition(&self) -> ToolDefinition {
                ToolDefinition {
                    name: "dummy".into(),
                    description: "Dummy tool".into(),
                    input_schema: serde_json::json!(
                        {"type": "object"}
                    ),
                }
            }
            async fn execute(&self, _input: serde_json::Value) -> TResult {
                TResult::success("dummy result")
            }
            fn required_capability(&self) -> capability::Capability {
                capability::Capability::Memory
            }
        }

        let manifest = AgentManifest::default();
        let driver = MockDriver::single_response("with tool");

        let result = AgentBuilder::new(&manifest)
            .driver(&driver)
            .tool(Box::new(DummyTool))
            .run("test")
            .await
            .expect("builder run with tool failed");

        assert_eq!(result.text, "with tool");
    }
}
