//! LLM evaluator trait definition.

use crate::monitor::llm::{EvalResult, LLMMetrics, LLMStats, PromptVersion, Result};

/// Trait for LLM evaluation systems
pub trait LLMEvaluator: Send + Sync {
    /// Evaluate response quality
    fn evaluate_response(
        &self,
        prompt: &str,
        response: &str,
        reference: Option<&str>,
    ) -> Result<EvalResult>;

    /// Log LLM call metrics
    fn log_llm_call(&mut self, run_id: &str, metrics: LLMMetrics) -> Result<()>;

    /// Track prompt version
    fn track_prompt(&mut self, run_id: &str, prompt: &PromptVersion) -> Result<()>;

    /// Get metrics for a run
    fn get_metrics(&self, run_id: &str) -> Result<Vec<LLMMetrics>>;

    /// Get prompts for a run
    fn get_prompts(&self, run_id: &str) -> Result<Vec<PromptVersion>>;

    /// Get aggregate statistics
    fn get_stats(&self, run_id: &str) -> Result<LLMStats>;
}
