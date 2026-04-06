//! Aggregate LLM statistics.

use crate::monitor::llm::LLMMetrics;
use serde::{Deserialize, Serialize};

/// Aggregate LLM statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LLMStats {
    /// Number of calls
    pub n_calls: usize,
    /// Total tokens used
    pub total_tokens: u64,
    /// Total prompt tokens
    pub total_prompt_tokens: u64,
    /// Total completion tokens
    pub total_completion_tokens: u64,
    /// Total estimated cost
    pub total_cost: f64,
    /// Average latency
    pub avg_latency_ms: f64,
    /// Average tokens per second
    pub avg_tokens_per_second: f64,
    /// P50 latency
    pub p50_latency_ms: f64,
    /// P95 latency
    pub p95_latency_ms: f64,
    /// P99 latency
    pub p99_latency_ms: f64,
}

impl LLMStats {
    /// Compute stats from metrics
    pub fn from_metrics(metrics: &[LLMMetrics]) -> Self {
        if metrics.is_empty() {
            return Self::default();
        }

        let n = metrics.len();
        let total_tokens: u64 = metrics.iter().map(|m| u64::from(m.total_tokens)).sum();
        let total_prompt: u64 = metrics.iter().map(|m| u64::from(m.prompt_tokens)).sum();
        let total_completion: u64 = metrics.iter().map(|m| u64::from(m.completion_tokens)).sum();
        let total_cost: f64 =
            metrics.iter().map(|m| m.cost_usd.unwrap_or_else(|| m.estimate_cost())).sum();

        let avg_latency: f64 = metrics.iter().map(|m| m.latency_ms).sum::<f64>() / n as f64;
        let avg_tps: f64 = metrics.iter().map(|m| m.tokens_per_second).sum::<f64>() / n as f64;

        // Compute percentiles
        let mut latencies: Vec<f64> = metrics.iter().map(|m| m.latency_ms).collect();
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let p50 = percentile(&latencies, 50.0);
        let p95 = percentile(&latencies, 95.0);
        let p99 = percentile(&latencies, 99.0);

        Self {
            n_calls: n,
            total_tokens,
            total_prompt_tokens: total_prompt,
            total_completion_tokens: total_completion,
            total_cost,
            avg_latency_ms: avg_latency,
            avg_tokens_per_second: avg_tps,
            p50_latency_ms: p50,
            p95_latency_ms: p95,
            p99_latency_ms: p99,
        }
    }
}

/// Compute percentile from sorted array
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}
