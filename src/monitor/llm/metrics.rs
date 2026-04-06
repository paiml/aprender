//! LLM call metrics tracking.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// LLM call metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMMetrics {
    /// Number of tokens in the prompt
    pub prompt_tokens: u32,
    /// Number of tokens in the completion
    pub completion_tokens: u32,
    /// Total tokens (prompt + completion)
    pub total_tokens: u32,
    /// Time to first token in milliseconds
    pub time_to_first_token_ms: f64,
    /// Tokens generated per second
    pub tokens_per_second: f64,
    /// Total latency in milliseconds
    pub latency_ms: f64,
    /// Estimated cost in USD (if known)
    pub cost_usd: Option<f64>,
    /// Model name (e.g., "gpt-4", "claude-3-opus")
    pub model_name: String,
    /// Timestamp of the call
    pub timestamp: DateTime<Utc>,
    /// Optional request ID
    pub request_id: Option<String>,
    /// Optional tags
    pub tags: HashMap<String, String>,
}

impl LLMMetrics {
    /// Create new LLM metrics with model name
    pub fn new(model_name: &str) -> Self {
        Self {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            time_to_first_token_ms: 0.0,
            tokens_per_second: 0.0,
            latency_ms: 0.0,
            cost_usd: None,
            model_name: model_name.to_string(),
            timestamp: Utc::now(),
            request_id: None,
            tags: HashMap::new(),
        }
    }

    /// Set token counts
    pub fn with_tokens(mut self, prompt: u32, completion: u32) -> Self {
        self.prompt_tokens = prompt;
        self.completion_tokens = completion;
        self.total_tokens = prompt + completion;
        self
    }

    /// Set latency
    pub fn with_latency(mut self, latency_ms: f64) -> Self {
        self.latency_ms = latency_ms;
        if latency_ms > 0.0 && self.completion_tokens > 0 {
            self.tokens_per_second = f64::from(self.completion_tokens) / (latency_ms / 1000.0);
        }
        self
    }

    /// Set time to first token
    pub fn with_ttft(mut self, ttft_ms: f64) -> Self {
        self.time_to_first_token_ms = ttft_ms;
        self
    }

    /// Set cost
    pub fn with_cost(mut self, cost_usd: f64) -> Self {
        self.cost_usd = Some(cost_usd);
        self
    }

    /// Set request ID
    pub fn with_request_id(mut self, id: &str) -> Self {
        self.request_id = Some(id.to_string());
        self
    }

    /// Add a tag
    pub fn with_tag(mut self, key: &str, value: &str) -> Self {
        self.tags.insert(key.to_string(), value.to_string());
        self
    }

    /// Calculate cost based on model pricing (approximate)
    ///
    /// N-07 (Meyer DbC): Prices are approximate and may be stale. Order matters:
    /// more specific patterns (e.g., "gpt-4-turbo") must precede generic ones
    /// ("gpt-4") to avoid mis-categorization. Unknown models warn and use a
    /// conservative default.
    pub fn estimate_cost(&self) -> f64 {
        // Approximate pricing per 1K tokens (as of late 2024).
        // Table-driven: (pattern, prompt_price, completion_price).
        // Order matters — more specific patterns first.
        const PRICING: &[(&str, f64, f64)] = &[
            ("gpt-4-turbo", 0.01, 0.03),
            ("gpt-4o", 0.005, 0.015),
            ("gpt-4", 0.03, 0.06),
            ("gpt-3.5", 0.0005, 0.0015),
            ("claude-3-opus", 0.015, 0.075),
            ("claude-3-sonnet", 0.003, 0.015),
            ("claude-3-haiku", 0.00025, 0.00125),
            ("gemini", 0.00025, 0.0005),
            ("mistral", 0.0002, 0.0006),
            ("llama", 0.0002, 0.0006),
        ];

        let (prompt_price, completion_price) = PRICING
            .iter()
            .find(|(pattern, _, _)| self.model_name.contains(pattern))
            .map_or_else(
                || {
                    eprintln!(
                        "Warning: unknown model '{}' for cost estimation, using conservative default \
                         ($0.001/$0.002 per 1K tokens)",
                        self.model_name
                    );
                    (0.001, 0.002)
                },
                |&(_, p, c)| (p, c),
            );

        let prompt_cost = (f64::from(self.prompt_tokens) / 1000.0) * prompt_price;
        let completion_cost = (f64::from(self.completion_tokens) / 1000.0) * completion_price;
        prompt_cost + completion_cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_metrics_new() {
        let metrics = LLMMetrics::new("gpt-4");
        assert_eq!(metrics.model_name, "gpt-4");
        assert_eq!(metrics.prompt_tokens, 0);
        assert_eq!(metrics.completion_tokens, 0);
        assert_eq!(metrics.total_tokens, 0);
        assert!(metrics.cost_usd.is_none());
        assert!(metrics.request_id.is_none());
        assert!(metrics.tags.is_empty());
    }

    #[test]
    fn test_llm_metrics_with_tokens() {
        let metrics = LLMMetrics::new("gpt-4").with_tokens(100, 50);
        assert_eq!(metrics.prompt_tokens, 100);
        assert_eq!(metrics.completion_tokens, 50);
        assert_eq!(metrics.total_tokens, 150);
    }

    #[test]
    fn test_llm_metrics_with_latency() {
        let metrics = LLMMetrics::new("gpt-4").with_tokens(100, 100).with_latency(1000.0);
        assert!((metrics.latency_ms - 1000.0).abs() < 1e-9);
        // tokens_per_second = 100 / 1.0 = 100
        assert!((metrics.tokens_per_second - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_llm_metrics_with_latency_zero() {
        let metrics = LLMMetrics::new("gpt-4").with_tokens(100, 100).with_latency(0.0);
        assert!((metrics.latency_ms - 0.0).abs() < 1e-9);
        // Should not calculate tokens_per_second for zero latency
        assert!((metrics.tokens_per_second - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_llm_metrics_with_ttft() {
        let metrics = LLMMetrics::new("gpt-4").with_ttft(150.0);
        assert!((metrics.time_to_first_token_ms - 150.0).abs() < 1e-9);
    }

    #[test]
    fn test_llm_metrics_with_cost() {
        let metrics = LLMMetrics::new("gpt-4").with_cost(0.05);
        assert_eq!(metrics.cost_usd, Some(0.05));
    }

    #[test]
    fn test_llm_metrics_with_request_id() {
        let metrics = LLMMetrics::new("gpt-4").with_request_id("req-12345");
        assert_eq!(metrics.request_id, Some("req-12345".to_string()));
    }

    #[test]
    fn test_llm_metrics_with_tag() {
        let metrics = LLMMetrics::new("gpt-4")
            .with_tag("environment", "production")
            .with_tag("user_id", "user123");
        assert_eq!(metrics.tags.get("environment"), Some(&"production".to_string()));
        assert_eq!(metrics.tags.get("user_id"), Some(&"user123".to_string()));
    }

    #[test]
    fn test_llm_metrics_estimate_cost_gpt4() {
        let metrics = LLMMetrics::new("gpt-4").with_tokens(1000, 1000);
        // GPT-4: 0.03 per 1K prompt + 0.06 per 1K completion = 0.09
        let cost = metrics.estimate_cost();
        assert!((cost - 0.09).abs() < 0.001);
    }

    #[test]
    fn test_llm_metrics_estimate_cost_gpt4_turbo() {
        let metrics = LLMMetrics::new("gpt-4-turbo").with_tokens(1000, 1000);
        // GPT-4-turbo: 0.01 + 0.03 = 0.04
        let cost = metrics.estimate_cost();
        assert!((cost - 0.04).abs() < 0.001);
    }

    #[test]
    fn test_llm_metrics_estimate_cost_gpt35() {
        let metrics = LLMMetrics::new("gpt-3.5-turbo").with_tokens(1000, 1000);
        // GPT-3.5: 0.0005 + 0.0015 = 0.002
        let cost = metrics.estimate_cost();
        assert!((cost - 0.002).abs() < 0.0001);
    }

    #[test]
    fn test_llm_metrics_estimate_cost_claude_opus() {
        let metrics = LLMMetrics::new("claude-3-opus").with_tokens(1000, 1000);
        // Claude-3-opus: 0.015 + 0.075 = 0.09
        let cost = metrics.estimate_cost();
        assert!((cost - 0.09).abs() < 0.001);
    }

    #[test]
    fn test_llm_metrics_estimate_cost_claude_sonnet() {
        let metrics = LLMMetrics::new("claude-3-sonnet").with_tokens(1000, 1000);
        // Claude-3-sonnet: 0.003 + 0.015 = 0.018
        let cost = metrics.estimate_cost();
        assert!((cost - 0.018).abs() < 0.001);
    }

    #[test]
    fn test_llm_metrics_estimate_cost_claude_haiku() {
        let metrics = LLMMetrics::new("claude-3-haiku").with_tokens(1000, 1000);
        // Claude-3-haiku: 0.00025 + 0.00125 = 0.0015
        let cost = metrics.estimate_cost();
        assert!((cost - 0.0015).abs() < 0.0001);
    }

    #[test]
    fn test_llm_metrics_estimate_cost_unknown_model() {
        let metrics = LLMMetrics::new("some-unknown-model").with_tokens(1000, 1000);
        // Default: 0.001 + 0.002 = 0.003
        let cost = metrics.estimate_cost();
        assert!((cost - 0.003).abs() < 0.001);
    }

    #[test]
    fn test_llm_metrics_clone() {
        let metrics = LLMMetrics::new("gpt-4").with_tokens(100, 50).with_latency(500.0);
        let cloned = metrics.clone();
        assert_eq!(metrics.model_name, cloned.model_name);
        assert_eq!(metrics.prompt_tokens, cloned.prompt_tokens);
    }

    #[test]
    fn test_llm_metrics_serde() {
        let metrics =
            LLMMetrics::new("gpt-4").with_tokens(100, 50).with_latency(500.0).with_cost(0.01);

        let json = serde_json::to_string(&metrics).expect("JSON serialization should succeed");
        let deserialized: LLMMetrics =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        assert_eq!(metrics.model_name, deserialized.model_name);
        assert_eq!(metrics.prompt_tokens, deserialized.prompt_tokens);
        assert_eq!(metrics.cost_usd, deserialized.cost_usd);
    }

    #[test]
    fn test_llm_metrics_debug() {
        let metrics = LLMMetrics::new("gpt-4");
        let debug_str = format!("{metrics:?}");
        assert!(debug_str.contains("LLMMetrics"));
        assert!(debug_str.contains("gpt-4"));
    }

    #[test]
    fn test_llm_metrics_chained_builders() {
        let metrics = LLMMetrics::new("claude-3-opus")
            .with_tokens(500, 200)
            .with_latency(2000.0)
            .with_ttft(100.0)
            .with_cost(0.05)
            .with_request_id("req-abc")
            .with_tag("feature", "summarization");

        assert_eq!(metrics.model_name, "claude-3-opus");
        assert_eq!(metrics.total_tokens, 700);
        assert!((metrics.latency_ms - 2000.0).abs() < 1e-9);
        assert!((metrics.time_to_first_token_ms - 100.0).abs() < 1e-9);
        assert_eq!(metrics.cost_usd, Some(0.05));
        assert_eq!(metrics.request_id, Some("req-abc".to_string()));
        assert_eq!(metrics.tags.get("feature"), Some(&"summarization".to_string()));
    }

    // =========================================================================
    // FALSIFY tests — contract violation sweep (N-07)
    // =========================================================================

    #[test]
    fn test_falsify_n07_gpt4_turbo_before_gpt4() {
        // N-07: "gpt-4-turbo-preview" must match "gpt-4-turbo" tier ($0.01),
        // not "gpt-4" tier ($0.03). Order of match arms matters.
        let turbo = LLMMetrics::new("gpt-4-turbo-preview").with_tokens(1000, 0);
        let base = LLMMetrics::new("gpt-4-0613").with_tokens(1000, 0);

        let turbo_cost = turbo.estimate_cost();
        let base_cost = base.estimate_cost();

        assert!(
            turbo_cost < base_cost,
            "gpt-4-turbo-preview ({turbo_cost}) must be cheaper than gpt-4 ({base_cost})"
        );
    }

    #[test]
    fn test_falsify_n07_gpt4o_distinct_from_gpt4() {
        // N-07: "gpt-4o" must match its own tier, not fall through to "gpt-4".
        let gpt4o = LLMMetrics::new("gpt-4o-2024-05-13").with_tokens(1000, 1000);
        let gpt4 = LLMMetrics::new("gpt-4-0613").with_tokens(1000, 1000);

        let gpt4o_cost = gpt4o.estimate_cost();
        let gpt4_cost = gpt4.estimate_cost();

        assert!(
            gpt4o_cost < gpt4_cost,
            "gpt-4o ({gpt4o_cost}) must be cheaper than gpt-4 ({gpt4_cost})"
        );
    }

    #[test]
    fn test_falsify_n07_unknown_model_uses_conservative_default() {
        // N-07: Unknown models must use the conservative default, never $0.
        let metrics = LLMMetrics::new("totally-unknown-model-v9").with_tokens(1000, 1000);
        let cost = metrics.estimate_cost();

        assert!(cost > 0.0, "Unknown model cost must be > 0, got {cost}");
        // Conservative default: $0.001 prompt + $0.002 completion = $0.003 per 1K
        assert!((cost - 0.003).abs() < 1e-6, "Expected conservative default ~$0.003, got {cost}");
    }

    #[test]
    fn test_estimate_cost_all_model_variants() {
        // (model_name, expected_total_cost_per_1K_prompt_1K_completion)
        let models = [
            ("gpt-4-turbo-preview", 0.01 + 0.03),
            ("gpt-4o-2024-05-13", 0.005 + 0.015),
            ("gpt-4-0613", 0.03 + 0.06),
            ("gpt-3.5-turbo", 0.0005 + 0.0015),
            ("claude-3-opus-20240229", 0.015 + 0.075),
            ("claude-3-sonnet-20240229", 0.003 + 0.015),
            ("claude-3-haiku-20240307", 0.00025 + 0.00125),
            ("gemini-pro", 0.00025 + 0.0005),
            ("mistral-medium", 0.0002 + 0.0006),
            ("llama-3-70b", 0.0002 + 0.0006),
            ("unknown-model", 0.001 + 0.002),
        ];

        for (model_name, expected_cost) in &models {
            let metrics = LLMMetrics::new(model_name).with_tokens(1000, 1000);
            let cost = metrics.estimate_cost();
            assert!(
                (cost - expected_cost).abs() < 1e-6,
                "cost mismatch for {model_name}: got {cost}, expected {expected_cost}"
            );
        }
    }
}
