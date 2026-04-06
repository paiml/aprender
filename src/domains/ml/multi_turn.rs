use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::engine::rng::SimRng;
use crate::error::{SimError, SimResult};

// ============================================================================
// Multi-Turn Simulation
// ============================================================================

/// A single turn in multi-turn interaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    /// Turn index.
    pub index: usize,
    /// Input query/prompt.
    pub input: String,
    /// Model response.
    pub output: String,
    /// Ground truth (if available).
    pub expected: Option<String>,
    /// Turn metrics.
    pub metrics: TurnMetrics,
    /// Context window usage (tokens).
    pub context_tokens: usize,
}

/// Metrics for a single turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnMetrics {
    /// Generation latency in milliseconds.
    pub latency_ms: f64,
    /// Input tokens.
    pub input_tokens: usize,
    /// Output tokens.
    pub output_tokens: usize,
    /// Estimated cost (normalized).
    pub cost: f64,
    /// Accuracy vs oracle (if available).
    pub accuracy: Option<f64>,
}

impl Default for TurnMetrics {
    fn default() -> Self {
        Self {
            latency_ms: 0.0,
            input_tokens: 0,
            output_tokens: 0,
            cost: 0.0,
            accuracy: None,
        }
    }
}

/// Multi-turn evaluation results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiTurnEvaluation {
    /// Mean accuracy across runs.
    pub mean_accuracy: Option<f64>,
    /// Mean latency across runs.
    pub mean_latency: Option<f64>,
    /// Total cost across runs.
    pub total_cost: f64,
    /// Confidence interval level.
    pub confidence_interval: f64,
    /// Number of runs performed.
    pub n_runs: usize,
}

/// Point on Pareto frontier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoPoint {
    /// Model identifier.
    pub model_id: String,
    /// Accuracy score.
    pub accuracy: f64,
    /// Cost metric.
    pub cost: f64,
    /// Latency metric.
    pub latency: f64,
    /// Models that dominate this one.
    pub dominated_by: Vec<String>,
}

/// Pareto frontier analysis results.
#[derive(Debug, Clone, Default)]
pub struct ParetoAnalysis {
    /// Non-dominated solutions (Pareto frontier).
    pub frontier: Vec<ParetoPoint>,
    /// Value scores per model.
    pub value_scores: HashMap<String, f64>,
}

/// Multi-turn simulation for conversational/iterative model evaluation.
///
/// Implements Pareto frontier analysis across accuracy, cost, and latency.
pub struct MultiTurnSimulation {
    /// Conversation history.
    history: Vec<Turn>,
    /// Deterministic RNG.
    rng: SimRng,
    /// Cost per input token.
    input_token_cost: f64,
    /// Cost per output token.
    output_token_cost: f64,
    /// Base latency per token (ms).
    latency_per_token_ms: f64,
}

impl MultiTurnSimulation {
    /// Create new multi-turn simulation.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            history: Vec::new(),
            rng: SimRng::new(seed),
            input_token_cost: 0.00001,
            output_token_cost: 0.00003,
            latency_per_token_ms: 10.0,
        }
    }

    /// Set cost parameters.
    #[must_use]
    pub fn with_costs(mut self, input_cost: f64, output_cost: f64) -> Self {
        self.input_token_cost = input_cost;
        self.output_token_cost = output_cost;
        self
    }

    /// Set latency per token.
    #[must_use]
    pub fn with_latency_per_token(mut self, latency_ms: f64) -> Self {
        self.latency_per_token_ms = latency_ms;
        self
    }

    /// Execute a single turn using a response generator.
    ///
    /// The `generate_fn` takes (input, history) and returns response string.
    ///
    /// # Errors
    ///
    /// Returns error if turn execution fails.
    pub fn turn<F>(
        &mut self,
        input: &str,
        expected: Option<&str>,
        generate_fn: F,
    ) -> SimResult<Turn>
    where
        F: FnOnce(&str, &[Turn]) -> String,
    {
        let input_tokens = self.count_tokens(input);

        // Generate response
        let output = generate_fn(input, &self.history);
        let output_tokens = self.count_tokens(&output);

        // Compute latency with noise
        let base_latency = (input_tokens + output_tokens) as f64 * self.latency_per_token_ms;
        let noise = (self.rng.gen_f64() * 0.2 - 0.1) * base_latency;
        let latency_ms = (base_latency + noise).max(1.0);

        // Compute cost
        let cost = input_tokens as f64 * self.input_token_cost
            + output_tokens as f64 * self.output_token_cost;

        // Compute accuracy if expected is provided
        let accuracy = expected.map(|exp| self.compute_accuracy(&output, exp));

        let context_tokens = self
            .history
            .iter()
            .map(|t| t.metrics.input_tokens + t.metrics.output_tokens)
            .sum::<usize>()
            + input_tokens;

        let turn = Turn {
            index: self.history.len(),
            input: input.to_string(),
            output,
            expected: expected.map(String::from),
            metrics: TurnMetrics {
                latency_ms,
                input_tokens,
                output_tokens,
                cost,
                accuracy,
            },
            context_tokens,
        };

        self.history.push(turn.clone());
        Ok(turn)
    }

    /// Simplified token counting (words * 1.3).
    #[allow(clippy::unused_self)]
    fn count_tokens(&self, text: &str) -> usize {
        let words = text.split_whitespace().count();
        (words as f64 * 1.3).ceil() as usize
    }

    /// Compute accuracy between output and expected (Levenshtein similarity).
    #[allow(clippy::unused_self)]
    fn compute_accuracy(&self, output: &str, expected: &str) -> f64 {
        if expected.is_empty() && output.is_empty() {
            return 1.0;
        }
        if expected.is_empty() || output.is_empty() {
            return 0.0;
        }

        // Simple word overlap similarity
        let output_words: std::collections::HashSet<&str> = output.split_whitespace().collect();
        let expected_words: std::collections::HashSet<&str> = expected.split_whitespace().collect();

        let intersection = output_words.intersection(&expected_words).count();
        let union = output_words.union(&expected_words).count();

        if union == 0 {
            return 1.0;
        }

        intersection as f64 / union as f64
    }

    /// Run complete multi-turn evaluation with statistical analysis.
    ///
    /// Following Princeton methodology: minimum 5 runs, 95% CI.
    ///
    /// # Errors
    ///
    /// Returns error if fewer than 5 runs are requested or if evaluation fails.
    pub fn evaluate<F>(
        &mut self,
        queries: &[(String, Option<String>)],
        n_runs: usize,
        generate_fn: F,
    ) -> SimResult<MultiTurnEvaluation>
    where
        F: Fn(&str, &[Turn]) -> String,
    {
        if n_runs < 5 {
            return Err(SimError::config(
                "Princeton methodology requires minimum 5 runs".to_string(),
            ));
        }

        let mut all_accuracies: Vec<f64> = Vec::new();
        let mut all_latencies: Vec<f64> = Vec::new();
        let mut total_cost = 0.0;

        for run in 0..n_runs {
            // Reset for each run with derived seed
            let derived_seed = self.rng.gen_u64().wrapping_add(run as u64);
            self.reset(derived_seed);

            for (query, expected) in queries {
                let turn = self.turn(query, expected.as_deref(), &generate_fn)?;
                if let Some(acc) = turn.metrics.accuracy {
                    all_accuracies.push(acc);
                }
                all_latencies.push(turn.metrics.latency_ms);
                total_cost += turn.metrics.cost;
            }
        }

        let mean_accuracy = if all_accuracies.is_empty() {
            None
        } else {
            Some(all_accuracies.iter().sum::<f64>() / all_accuracies.len() as f64)
        };

        let mean_latency = if all_latencies.is_empty() {
            None
        } else {
            Some(all_latencies.iter().sum::<f64>() / all_latencies.len() as f64)
        };

        Ok(MultiTurnEvaluation {
            mean_accuracy,
            mean_latency,
            total_cost: total_cost / n_runs as f64,
            confidence_interval: 0.95,
            n_runs,
        })
    }

    /// Compute Pareto frontier across multiple model evaluations.
    #[must_use]
    pub fn pareto_analysis(evaluations: &[(String, MultiTurnEvaluation)]) -> ParetoAnalysis {
        let mut points: Vec<ParetoPoint> = evaluations
            .iter()
            .map(|(id, eval)| ParetoPoint {
                model_id: id.clone(),
                accuracy: eval.mean_accuracy.unwrap_or(0.0),
                cost: eval.total_cost,
                latency: eval.mean_latency.unwrap_or(f64::MAX),
                dominated_by: Vec::new(),
            })
            .collect();

        // Identify dominated points
        // First pass: identify dominance relationships
        let mut dominance: Vec<Vec<String>> = vec![Vec::new(); points.len()];
        for i in 0..points.len() {
            for j in 0..points.len() {
                if i != j && Self::dominates(&points[j], &points[i]) {
                    dominance[i].push(points[j].model_id.clone());
                }
            }
        }
        // Second pass: assign dominated_by
        for (i, dominated_by) in dominance.into_iter().enumerate() {
            points[i].dominated_by = dominated_by;
        }

        // Compute value scores
        let baseline_accuracy = points.iter().map(|p| p.accuracy).fold(0.0_f64, f64::max);
        let baseline_cost = points.iter().map(|p| p.cost).fold(f64::INFINITY, f64::min);
        let baseline_latency = points
            .iter()
            .map(|p| p.latency)
            .fold(f64::INFINITY, f64::min);

        let value_scores: HashMap<String, f64> = points
            .iter()
            .map(|p| {
                let accuracy_gap = baseline_accuracy - p.accuracy;
                let cost_ratio = baseline_cost / p.cost.max(1e-10);
                let latency_ratio = baseline_latency / p.latency.max(1e-10);
                let value = (1.0 - accuracy_gap) * cost_ratio * latency_ratio;
                (p.model_id.clone(), value)
            })
            .collect();

        let frontier: Vec<ParetoPoint> = points
            .into_iter()
            .filter(|p| p.dominated_by.is_empty())
            .collect();

        ParetoAnalysis {
            frontier,
            value_scores,
        }
    }

    /// Check if point a dominates point b (better in all objectives).
    fn dominates(a: &ParetoPoint, b: &ParetoPoint) -> bool {
        a.accuracy >= b.accuracy
            && a.cost <= b.cost
            && a.latency <= b.latency
            && (a.accuracy > b.accuracy || a.cost < b.cost || a.latency < b.latency)
    }

    /// Get conversation history.
    #[must_use]
    pub fn history(&self) -> &[Turn] {
        &self.history
    }

    /// Reset simulation state.
    pub fn reset(&mut self, seed: u64) {
        self.rng = SimRng::new(seed);
        self.history.clear();
    }
}
