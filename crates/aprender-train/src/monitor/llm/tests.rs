//! Tests for LLM evaluation module.

#![allow(clippy::module_inception)]
#[cfg(test)]
mod tests {
    use crate::monitor::llm::{
        heuristics::{
            compute_coherence, compute_groundedness, compute_harmfulness, compute_relevance,
        },
        EvalResult, InMemoryLLMEvaluator, LLMEvaluator, LLMMetrics, LLMStats, PromptVersion,
    };
    use std::collections::HashMap;

    // -------------------------------------------------------------------------
    // LLMMetrics Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_llm_metrics_new() {
        let metrics = LLMMetrics::new("gpt-4");
        assert_eq!(metrics.model_name, "gpt-4");
        assert_eq!(metrics.total_tokens, 0);
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
        let metrics = LLMMetrics::new("gpt-4").with_tokens(100, 50).with_latency(1000.0);
        assert_eq!(metrics.latency_ms, 1000.0);
        assert!((metrics.tokens_per_second - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_llm_metrics_estimate_cost() {
        let metrics = LLMMetrics::new("gpt-4").with_tokens(1000, 500);
        let cost = metrics.estimate_cost();
        // gpt-4: $0.03/1K prompt + $0.06/1K completion
        // = 0.03 + 0.03 = 0.06
        assert!((cost - 0.06).abs() < 0.001);
    }

    #[test]
    fn test_llm_metrics_with_tag() {
        let metrics = LLMMetrics::new("gpt-4").with_tag("purpose", "summarization");
        assert_eq!(metrics.tags.get("purpose"), Some(&"summarization".to_string()));
    }

    // -------------------------------------------------------------------------
    // PromptVersion Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_prompt_version_new() {
        let prompt = PromptVersion::new("Hello {name}!", vec!["name".to_string()]);
        assert!(!prompt.id.is_empty());
        assert_eq!(prompt.template, "Hello {name}!");
        assert_eq!(prompt.variables, vec!["name"]);
        assert_eq!(prompt.version, 1);
    }

    #[test]
    fn test_prompt_version_hash_deterministic() {
        let p1 = PromptVersion::new("Test template", vec![]);
        let p2 = PromptVersion::new("Test template", vec![]);
        assert_eq!(p1.sha256, p2.sha256);
        assert_eq!(p1.id, p2.id);
    }

    #[test]
    fn test_prompt_version_hash_different() {
        let p1 = PromptVersion::new("Template A", vec![]);
        let p2 = PromptVersion::new("Template B", vec![]);
        assert_ne!(p1.sha256, p2.sha256);
    }

    #[test]
    fn test_prompt_version_render() {
        let prompt = PromptVersion::new(
            "Hello {name}! You are {age} years old.",
            vec!["name".to_string(), "age".to_string()],
        );

        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "Alice".to_string());
        vars.insert("age".to_string(), "30".to_string());

        let rendered = prompt.render(&vars).expect("rendering should succeed");
        assert_eq!(rendered, "Hello Alice! You are 30 years old.");
    }

    #[test]
    fn test_prompt_version_render_missing_var() {
        let prompt = PromptVersion::new("Hello {name}!", vec!["name".to_string()]);
        let vars = HashMap::new();
        let result = prompt.render(&vars);
        assert!(result.is_err());
    }

    #[test]
    fn test_prompt_version_extract_variables() {
        let vars = PromptVersion::extract_variables("Hello {name}, your ID is {id}.");
        assert_eq!(vars, vec!["name", "id"]);
    }

    #[test]
    fn test_prompt_version_with_version() {
        let prompt = PromptVersion::new("Test", vec![]).with_version(5);
        assert_eq!(prompt.version, 5);
    }

    // -------------------------------------------------------------------------
    // EvalResult Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_eval_result_new() {
        let result = EvalResult::new(0.8, 0.9, 0.7, 0.1);
        assert_eq!(result.relevance, 0.8);
        assert_eq!(result.coherence, 0.9);
        assert_eq!(result.groundedness, 0.7);
        assert_eq!(result.harmfulness, 0.1);
    }

    #[test]
    fn test_eval_result_clamped() {
        let result = EvalResult::new(1.5, -0.1, 0.5, 2.0);
        assert_eq!(result.relevance, 1.0);
        assert_eq!(result.coherence, 0.0);
        assert_eq!(result.harmfulness, 1.0);
    }

    #[test]
    fn test_eval_result_overall() {
        let result = EvalResult::new(1.0, 1.0, 1.0, 0.0);
        // Perfect scores: overall should be 1.0
        assert!((result.overall - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_eval_result_passes_threshold() {
        let good = EvalResult::new(0.9, 0.9, 0.9, 0.1);
        let bad = EvalResult::new(0.3, 0.3, 0.3, 0.8);

        assert!(good.passes_threshold(0.7, 0.3));
        assert!(!bad.passes_threshold(0.7, 0.3));
    }

    #[test]
    fn test_eval_result_with_detail() {
        let result = EvalResult::new(0.8, 0.9, 0.7, 0.1).with_detail("fluency", 0.95);
        assert_eq!(result.details.get("fluency"), Some(&0.95));
    }

    // -------------------------------------------------------------------------
    // LLMStats Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_llm_stats_empty() {
        let stats = LLMStats::from_metrics(&[]);
        assert_eq!(stats.n_calls, 0);
        assert_eq!(stats.total_tokens, 0);
    }

    #[test]
    fn test_llm_stats_single() {
        let metrics = vec![LLMMetrics::new("gpt-4").with_tokens(100, 50).with_latency(1000.0)];
        let stats = LLMStats::from_metrics(&metrics);
        assert_eq!(stats.n_calls, 1);
        assert_eq!(stats.total_tokens, 150);
        assert_eq!(stats.avg_latency_ms, 1000.0);
    }

    #[test]
    fn test_llm_stats_multiple() {
        let metrics = vec![
            LLMMetrics::new("gpt-4").with_tokens(100, 50).with_latency(1000.0),
            LLMMetrics::new("gpt-4").with_tokens(200, 100).with_latency(2000.0),
        ];
        let stats = LLMStats::from_metrics(&metrics);
        assert_eq!(stats.n_calls, 2);
        assert_eq!(stats.total_tokens, 450);
        assert_eq!(stats.avg_latency_ms, 1500.0);
    }

    // -------------------------------------------------------------------------
    // InMemoryLLMEvaluator Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_evaluator_new() {
        let evaluator = InMemoryLLMEvaluator::new();
        let result = evaluator.get_metrics("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_evaluator_log_metrics() {
        let mut evaluator = InMemoryLLMEvaluator::new();
        let metrics = LLMMetrics::new("gpt-4").with_tokens(100, 50);

        evaluator.log_llm_call("run-1", metrics).expect("operation should succeed");

        let retrieved = evaluator.get_metrics("run-1").expect("operation should succeed");
        assert_eq!(retrieved.len(), 1);
        assert_eq!(retrieved[0].total_tokens, 150);
    }

    #[test]
    fn test_evaluator_track_prompt() {
        let mut evaluator = InMemoryLLMEvaluator::new();
        let prompt = PromptVersion::new("Test prompt", vec![]);

        evaluator.track_prompt("run-1", &prompt).expect("operation should succeed");

        let retrieved = evaluator.get_prompts("run-1").expect("operation should succeed");
        assert_eq!(retrieved.len(), 1);
        assert_eq!(retrieved[0].template, "Test prompt");
    }

    #[test]
    fn test_evaluator_evaluate_response() {
        let evaluator = InMemoryLLMEvaluator::new();

        let result = evaluator
            .evaluate_response(
                "What is the capital of France?",
                "The capital of France is Paris.",
                Some("Paris is the capital of France"),
            )
            .expect("operation should succeed");

        assert!(result.relevance > 0.3);
        assert!(result.coherence > 0.5);
        assert!(result.groundedness > 0.5);
        assert!(result.harmfulness < 0.5);
    }

    #[test]
    fn test_evaluator_get_stats() {
        let mut evaluator = InMemoryLLMEvaluator::new();

        evaluator
            .log_llm_call(
                "run-1",
                LLMMetrics::new("gpt-4").with_tokens(100, 50).with_latency(500.0),
            )
            .expect("operation should succeed");
        evaluator
            .log_llm_call(
                "run-1",
                LLMMetrics::new("gpt-4").with_tokens(200, 100).with_latency(1500.0),
            )
            .expect("operation should succeed");

        let stats = evaluator.get_stats("run-1").expect("operation should succeed");
        assert_eq!(stats.n_calls, 2);
        assert_eq!(stats.total_tokens, 450);
    }

    // -------------------------------------------------------------------------
    // Heuristic Function Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_compute_relevance() {
        let high = compute_relevance("capital France", "The capital of France is Paris");
        let low = compute_relevance("weather today", "The capital of France is Paris");
        assert!(high > low);
    }

    #[test]
    fn test_compute_coherence() {
        let good = compute_coherence("This is a well-formed sentence. It has good structure.");
        let bad = compute_coherence("BAD ALL CAPS TEXT");
        assert!(good > bad);
    }

    #[test]
    fn test_compute_groundedness() {
        let grounded =
            compute_groundedness("Paris is the capital", "Paris is the capital of France");
        let ungrounded = compute_groundedness("Tokyo is in Asia", "Paris is the capital of France");
        assert!(grounded > ungrounded);
    }

    #[test]
    fn test_compute_harmfulness() {
        let safe = compute_harmfulness("The weather is nice today.");
        let harmful = compute_harmfulness("How to hack into systems and steal data.");
        assert!(safe < harmful);
    }
}

// =============================================================================
// Property Tests
// =============================================================================

#[cfg(test)]
mod property_tests {
    use crate::monitor::llm::{stats::percentile, EvalResult, LLMMetrics, LLMStats, PromptVersion};
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_eval_result_overall_bounded(
            relevance in 0.0f64..1.0,
            coherence in 0.0f64..1.0,
            groundedness in 0.0f64..1.0,
            harmfulness in 0.0f64..1.0
        ) {
            let result = EvalResult::new(relevance, coherence, groundedness, harmfulness);
            prop_assert!(result.overall >= 0.0);
            prop_assert!(result.overall <= 1.0);
        }

        #[test]
        fn prop_llm_metrics_tokens_sum(prompt in 0u32..10000, completion in 0u32..10000) {
            let metrics = LLMMetrics::new("test").with_tokens(prompt, completion);
            prop_assert_eq!(metrics.total_tokens, prompt + completion);
        }

        #[test]
        fn prop_prompt_hash_deterministic(template in "[a-zA-Z0-9 ]{1,100}") {
            let p1 = PromptVersion::new(&template, vec![]);
            let p2 = PromptVersion::new(&template, vec![]);
            prop_assert_eq!(p1.sha256, p2.sha256);
        }

        #[test]
        fn prop_llm_stats_tokens_consistent(
            calls in prop::collection::vec(
                (0u32..1000, 0u32..1000),
                1..10
            )
        ) {
            let metrics: Vec<LLMMetrics> = calls
                .iter()
                .map(|(p, c)| LLMMetrics::new("test").with_tokens(*p, *c))
                .collect();

            let stats = LLMStats::from_metrics(&metrics);
            let expected_total: u64 = calls.iter().map(|(p, c)| u64::from(*p + *c)).sum();
            prop_assert_eq!(stats.total_tokens, expected_total);
        }

        #[test]
        fn prop_percentile_bounded(values in prop::collection::vec(0.0f64..1000.0, 1..100)) {
            let mut sorted = values.clone();
            sorted.sort_by(f64::total_cmp);

            let p50 = percentile(&sorted, 50.0);
            let min = sorted.first().expect("collection should not be empty");
            let max = sorted.last().expect("collection should not be empty");

            prop_assert!(p50 >= *min);
            prop_assert!(p50 <= *max);
        }
    }
}
