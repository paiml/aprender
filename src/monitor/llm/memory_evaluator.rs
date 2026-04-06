//! In-memory LLM evaluator implementation.

use crate::monitor::llm::{
    heuristics::{compute_coherence, compute_groundedness, compute_harmfulness, compute_relevance},
    EvalResult, LLMError, LLMEvaluator, LLMMetrics, LLMStats, PromptVersion, Result,
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// In-memory LLM evaluator for testing
#[derive(Debug, Default)]
pub struct InMemoryLLMEvaluator {
    /// Metrics by run ID
    metrics: Arc<RwLock<HashMap<String, Vec<LLMMetrics>>>>,
    /// Prompts by run ID
    prompts: Arc<RwLock<HashMap<String, Vec<PromptVersion>>>>,
}

impl InMemoryLLMEvaluator {
    /// Create a new in-memory evaluator
    pub fn new() -> Self {
        Self::default()
    }
}

impl LLMEvaluator for InMemoryLLMEvaluator {
    fn evaluate_response(
        &self,
        prompt: &str,
        response: &str,
        reference: Option<&str>,
    ) -> Result<EvalResult> {
        // Simple heuristic evaluation (production would use a model)
        let relevance = compute_relevance(prompt, response);
        let coherence = compute_coherence(response);
        let groundedness = if let Some(ref_text) = reference {
            compute_groundedness(response, ref_text)
        } else {
            0.5 // Unknown without reference
        };
        let harmfulness = compute_harmfulness(response);

        Ok(EvalResult::new(relevance, coherence, groundedness, harmfulness))
    }

    fn log_llm_call(&mut self, run_id: &str, metrics: LLMMetrics) -> Result<()> {
        let mut store =
            self.metrics.write().map_err(|e| LLMError::Internal(format!("Lock error: {e}")))?;

        store.entry(run_id.to_string()).or_default().push(metrics);

        Ok(())
    }

    fn track_prompt(&mut self, run_id: &str, prompt: &PromptVersion) -> Result<()> {
        let mut store =
            self.prompts.write().map_err(|e| LLMError::Internal(format!("Lock error: {e}")))?;

        store.entry(run_id.to_string()).or_default().push(prompt.clone());

        Ok(())
    }

    fn get_metrics(&self, run_id: &str) -> Result<Vec<LLMMetrics>> {
        let store =
            self.metrics.read().map_err(|e| LLMError::Internal(format!("Lock error: {e}")))?;

        store.get(run_id).cloned().ok_or_else(|| LLMError::RunNotFound(run_id.to_string()))
    }

    fn get_prompts(&self, run_id: &str) -> Result<Vec<PromptVersion>> {
        let store =
            self.prompts.read().map_err(|e| LLMError::Internal(format!("Lock error: {e}")))?;

        store.get(run_id).cloned().ok_or_else(|| LLMError::RunNotFound(run_id.to_string()))
    }

    fn get_stats(&self, run_id: &str) -> Result<LLMStats> {
        let metrics = self.get_metrics(run_id)?;
        Ok(LLMStats::from_metrics(&metrics))
    }
}
