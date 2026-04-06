//! Evaluation result scores for LLM responses.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Evaluation result scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResult {
    /// Relevance to the query (0-1, higher is better)
    pub relevance: f64,
    /// Logical consistency (0-1, higher is better)
    pub coherence: f64,
    /// Factual accuracy (0-1, higher is better)
    pub groundedness: f64,
    /// Potential harm score (0-1, lower is better)
    pub harmfulness: f64,
    /// Optional detailed scores
    pub details: HashMap<String, f64>,
    /// Overall score (weighted average)
    pub overall: f64,
}

impl EvalResult {
    /// Create a new evaluation result
    pub fn new(relevance: f64, coherence: f64, groundedness: f64, harmfulness: f64) -> Self {
        let overall = Self::compute_overall(relevance, coherence, groundedness, harmfulness);
        Self {
            relevance: relevance.clamp(0.0, 1.0),
            coherence: coherence.clamp(0.0, 1.0),
            groundedness: groundedness.clamp(0.0, 1.0),
            harmfulness: harmfulness.clamp(0.0, 1.0),
            details: HashMap::new(),
            overall,
        }
    }

    /// Add a detail score
    pub fn with_detail(mut self, name: &str, score: f64) -> Self {
        self.details.insert(name.to_string(), score.clamp(0.0, 1.0));
        self
    }

    /// Compute overall score
    fn compute_overall(relevance: f64, coherence: f64, groundedness: f64, harmfulness: f64) -> f64 {
        // Weighted average (harmfulness is inverted since lower is better)
        let weights = [0.3, 0.2, 0.3, 0.2]; // relevance, coherence, groundedness, safety
        let scores = [
            relevance,
            coherence,
            groundedness,
            1.0 - harmfulness, // Invert harmfulness
        ];

        let weighted_sum: f64 = weights.iter().zip(scores.iter()).map(|(w, s)| w * s).sum();
        weighted_sum.clamp(0.0, 1.0)
    }

    /// Check if result passes quality threshold
    pub fn passes_threshold(&self, min_overall: f64, max_harmfulness: f64) -> bool {
        self.overall >= min_overall && self.harmfulness <= max_harmfulness
    }
}

impl Default for EvalResult {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }
}
