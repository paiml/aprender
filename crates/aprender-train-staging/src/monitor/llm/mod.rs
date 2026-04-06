//! LLM Evaluation Metrics Module (#71)
//!
//! Direct observation of LLM behavior through comprehensive metrics tracking.
//!
//! # Toyota Way: 現地現物 (Genchi Genbutsu)
//!
//! "Go and see" - Direct observation of LLM behavior through metrics enables
//! data-driven decisions about prompt engineering and model selection.
//!
//! # Example
//!
//! ```ignore
//! use entrenar::monitor::llm::{LLMMetrics, PromptVersion, EvalResult, InMemoryLLMEvaluator};
//!
//! let mut evaluator = InMemoryLLMEvaluator::new();
//!
//! // Track prompt version
//! let prompt = PromptVersion::new("Summarize: {text}", vec!["text".to_string()]);
//! evaluator.track_prompt("run-1", &prompt)?;
//!
//! // Log LLM call metrics
//! let metrics = LLMMetrics::new("gpt-4")
//!     .with_tokens(100, 50)
//!     .with_latency(1500.0);
//! evaluator.log_llm_call("run-1", metrics)?;
//!
//! // Evaluate response quality
//! let result = evaluator.evaluate_response("What is 2+2?", "4", Some("4"))?;
//! ```

mod error;
mod eval_result;
pub mod heuristics;
mod memory_evaluator;
mod metrics;
mod prompt;
pub mod stats;
mod traits;

#[cfg(test)]
mod tests;

// Re-export all public types for API compatibility
pub use error::{LLMError, Result};
pub use eval_result::EvalResult;
pub use memory_evaluator::InMemoryLLMEvaluator;
pub use metrics::LLMMetrics;
pub use prompt::{PromptId, PromptVersion};
pub use stats::LLMStats;
pub use traits::LLMEvaluator;
