//! EXTREME TDD coverage tests for generate.rs
//!
//! Targets edge cases and coverage gaps to increase from 95% to 98%+.
//! All tests are designed to pass clippy with -D warnings.

use realizar::generate::{
    analyze_token_healing, apply_all_penalties, apply_cfg, apply_dry_penalty,
    apply_dynamic_temperature, apply_infill_sampling, apply_logit_bias,
    apply_presence_frequency_penalty, apply_repetition_penalty, apply_temperature, apply_xtc,
    sample_eta, sample_greedy, sample_min_p, sample_mirostat, sample_tfs, sample_token,
    sample_top_k, sample_top_p, sample_typical, AdvancedGenerationConfig, BeamHypothesis,
    BeamSearchConfig, BeamSearchState, CfgConfig, DryConfig, DynTempConfig, EtaConfig,
    GenerationConfig, InfillConfig, LogitBias, LogitProcessor, LogitProcessorChain,
    LogitProcessorContext, MirostatState, PresenceFrequencyPenalty, PromptCache, RepetitionPenalty,
    RepetitionPenaltyConfig, SamplerChain, SamplerContext, SamplingStrategy, StopSequenceDetector,
    StreamingGenerator, TemperatureScaler, TokenHealingConfig, TokenSuppressor, XtcConfig,
};
use realizar::tensor::Tensor;

// =============================================================================
// Edge Case Tests: Single-Element and Boundary Inputs
// =============================================================================

#[test]
fn test_sample_greedy_single_element() {
    let logits = Tensor::from_vec(vec![1], vec![42.0]).expect("single element");
    let result = sample_greedy(&logits);
    assert!(result.is_ok());
    assert_eq!(result.expect("test"), 0);
}

#[test]
fn test_sample_top_k_single_element() {
    let logits = Tensor::from_vec(vec![1], vec![10.0]).expect("single element");
    let result = sample_top_k(&logits, 5, 0.5);
    assert!(result.is_ok());
    assert_eq!(result.expect("test"), 0);
}

#[test]
fn test_sample_top_p_single_element() {
    let logits = Tensor::from_vec(vec![1], vec![5.0]).expect("single element");
    let result = sample_top_p(&logits, 0.9, 0.5);
    assert!(result.is_ok());
    assert_eq!(result.expect("test"), 0);
}

#[test]
fn test_sample_min_p_invalid_threshold() {
    let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
    // min_p must be in [0, 1]
    assert!(sample_min_p(&logits, -0.1, 0.5).is_err());
    assert!(sample_min_p(&logits, 1.1, 0.5).is_err());
}

#[test]
fn test_sample_mirostat_single_element() {
    let logits = Tensor::from_vec(vec![1], vec![5.0]).expect("single element");
    let mut state = MirostatState::default();
    let result = sample_mirostat(&logits, &mut state, 0.5);
    assert!(result.is_ok());
    assert_eq!(result.expect("test"), 0);
}

#[test]
fn test_sample_tfs_single_element() {
    let logits = Tensor::from_vec(vec![1], vec![5.0]).expect("single element");
    let result = sample_tfs(&logits, 0.95, 0.5);
    assert!(result.is_ok());
    assert_eq!(result.expect("test"), 0);
}

#[test]
fn test_sample_typical_single_element() {
    let logits = Tensor::from_vec(vec![1], vec![5.0]).expect("single element");
    let result = sample_typical(&logits, 0.95, 0.5);
    assert!(result.is_ok());
    assert_eq!(result.expect("test"), 0);
}

#[test]
fn test_sample_eta_single_element() {
    let logits = Tensor::from_vec(vec![1], vec![5.0]).expect("single element");
    let config = EtaConfig::default();
    let result = sample_eta(&logits, &config, 0.5);
    assert!(result.is_ok());
    assert_eq!(result.expect("test"), 0);
}

// =============================================================================
// GenerationConfig Builder Edge Cases
// =============================================================================

#[test]
fn test_generation_config_with_seed() {
    let config = GenerationConfig::greedy().with_seed(12345);
    assert_eq!(config.seed, Some(12345));
}

#[test]
fn test_generation_config_all_strategies_combined() {
    // Test all strategy builders with all modifiers
    let greedy = GenerationConfig::greedy()
        .with_temperature(0.7)
        .with_max_tokens(50)
        .with_eos_token_id(2)
        .with_seed(42);
    assert_eq!(greedy.strategy, SamplingStrategy::Greedy);
    assert!((greedy.temperature - 0.7).abs() < 1e-6);
    assert_eq!(greedy.max_tokens, 50);
    assert_eq!(greedy.eos_token_id, Some(2));
    assert_eq!(greedy.seed, Some(42));

    let top_k = GenerationConfig::top_k(40);
    assert_eq!(top_k.strategy, SamplingStrategy::TopK { k: 40 });

    let top_p = GenerationConfig::top_p(0.95);
    assert_eq!(top_p.strategy, SamplingStrategy::TopP { p: 0.95 });
}

// =============================================================================
// sample_token Strategy Dispatch
// =============================================================================

#[test]
fn test_sample_token_with_invalid_temperature_error() {
    let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
    let mut config = GenerationConfig::greedy();
    config.temperature = 0.0; // Invalid
    let result = sample_token(&logits, &config, 0.5);
    assert!(result.is_err());
}

#[test]
fn test_sample_token_top_k_with_k_larger_than_vocab() {
    let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
    let config = GenerationConfig::top_k(100); // k > vocab size
    let result = sample_token(&logits, &config, 0.0);
    assert!(result.is_ok());
    assert_eq!(result.expect("test"), 2); // Should still get max
}

// =============================================================================
// StopSequenceDetector Edge Cases
// =============================================================================

#[test]
fn test_stop_sequence_detector_empty_sequences_ignored() {
    let detector = StopSequenceDetector::new()
        .with_token_sequence(vec![]) // Empty - should be ignored
        .with_string_pattern(""); // Empty - should be ignored
                                  // Empty sequences should not be added
    assert!(!detector.has_conditions());
}

#[test]
fn test_stop_sequence_detector_multiple_patterns_first_match() {
    let detector = StopSequenceDetector::new()
        .with_string_pattern("first")
        .with_string_pattern("second");

    // First pattern matches
    let result = detector.check_text("this is first and second");
    assert!(result.is_some());
    assert_eq!(result.expect("test"), 8); // Position of "first"
}

// =============================================================================
// Repetition Penalty Edge Cases
// =============================================================================

#[test]
fn test_repetition_penalty_empty_context() {
    let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
    let context: Vec<usize> = vec![];
    let config = RepetitionPenaltyConfig::new(2.0);

    let result = apply_repetition_penalty(&logits, &context, &config);
    // No change with empty context
    assert_eq!(result.data(), logits.data());
}

#[test]
fn test_repetition_penalty_token_out_of_vocab() {
    let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
    let context = vec![100, 200]; // Out of vocab range
    let config = RepetitionPenaltyConfig::new(2.0);

    let result = apply_repetition_penalty(&logits, &context, &config);
    // Should not panic, no change
    assert_eq!(result.data(), logits.data());
}

#[test]
fn test_repetition_penalty_zero_window() {
    let logits = Tensor::from_vec(vec![5], vec![2.0, 2.0, 2.0, 2.0, 2.0]).expect("test");
    let context = vec![0, 1, 2, 3, 4];
    let config = RepetitionPenaltyConfig::new(2.0).with_window(0); // Window 0 = all tokens

    let result = apply_repetition_penalty(&logits, &context, &config);
    // All tokens should be penalized
    for &val in result.data() {
        assert!((val - 1.0).abs() < 1e-6); // 2.0 / 2.0 = 1.0
    }
}

// =============================================================================
// Presence/Frequency Penalty Edge Cases
// =============================================================================

#[test]
fn test_presence_frequency_penalty_empty_context() {
    let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
    let context: Vec<usize> = vec![];
    let config = PresenceFrequencyPenalty::new(1.0, 1.0);

    let result = apply_presence_frequency_penalty(&logits, &context, &config);
    assert_eq!(result.data(), logits.data());
}

#[test]
fn test_presence_frequency_penalty_token_out_of_vocab() {
    let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
    let context = vec![100]; // Out of range
    let config = PresenceFrequencyPenalty::new(1.0, 1.0);

    let result = apply_presence_frequency_penalty(&logits, &context, &config);
    assert_eq!(result.data(), logits.data());
}

// =============================================================================
// Logit Bias Edge Cases
// =============================================================================

#[test]
fn test_logit_bias_with_biases_method() {
    let mut biases = std::collections::HashMap::new();
    biases.insert(0, 5.0);
    biases.insert(2, -5.0);

    let bias = LogitBias::new().with_biases(biases);
    assert_eq!(bias.get(0), 5.0);
    assert_eq!(bias.get(2), -5.0);
    assert_eq!(bias.get(1), 0.0);
}

#[test]
fn test_apply_logit_bias_empty() {
    let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
    let bias = LogitBias::new(); // Empty

    let result = apply_logit_bias(&logits, &bias);
    assert_eq!(result.data(), logits.data());
}

// =============================================================================
// DRY Penalty Edge Cases
// =============================================================================

#[test]
fn test_dry_penalty_empty_context() {
    let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
    let config = DryConfig::new(1.0);
    let context: Vec<usize> = vec![];

    let result = apply_dry_penalty(&logits, &context, &config);
    assert_eq!(result.data(), logits.data());
}

// =============================================================================
// XTC (Exclude Top Choices) Edge Cases
// =============================================================================

#[test]
fn test_xtc_small_vocab_no_exclusion() {
    let logits = Tensor::from_vec(vec![1], vec![10.0]).expect("test");
    let config = XtcConfig::new(1.0).with_min_keep(1);
    // Only 1 token, can't exclude anything
    let result = apply_xtc(&logits, &config, 0.0);
    assert_eq!(result.data(), logits.data());
}

// =============================================================================
// CFG (Classifier-Free Guidance) Edge Cases
// =============================================================================

#[test]
fn test_cfg_negative_scale() {
    let cond = Tensor::from_vec(vec![3], vec![2.0, 1.0, 0.0]).expect("test");
    let uncond = Tensor::from_vec(vec![3], vec![1.0, 1.0, 1.0]).expect("test");
    // Negative scale inverts the direction
    let result = apply_cfg(&cond, &uncond, -1.0).expect("test");
    // uncond + (-1) * (cond - uncond) = uncond - cond + uncond = 2*uncond - cond
    assert_eq!(result.data(), &[0.0, 1.0, 2.0]);
}

// =============================================================================
// Dynamic Temperature Edge Cases
// =============================================================================

#[test]
fn test_dynamic_temperature_with_negative_delta() {
    let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
    let config = DynTempConfig::new(1.0, -0.5, 1.0); // Negative delta treated as 0

    let result = apply_dynamic_temperature(&logits, &config);
    // Should use static temperature with delta <= 0
    let expected = apply_temperature(&logits, 1.0).expect("test");
    assert_eq!(result.data(), expected.data());
}

#[test]
fn test_dynamic_temperature_single_element() {
    let logits = Tensor::from_vec(vec![1], vec![5.0]).expect("single");
    let config = DynTempConfig::new(1.0, 0.5, 1.0);

    let result = apply_dynamic_temperature(&logits, &config);
    // Single element returns unchanged
    assert_eq!(result.data().len(), 1);
    assert!((result.data()[0] - 5.0).abs() < 1e-6);
}

// =============================================================================
// Infill Sampling Edge Cases
// =============================================================================

#[test]
fn test_infill_sampling_single_element() {
    let logits = Tensor::from_vec(vec![1], vec![5.0]).expect("single");
    let config = InfillConfig::new(vec![0]); // Token 0 is EOG

    let result = apply_infill_sampling(&logits, &config);
    // Single token (0) is EOG, so force_eog might be true
    assert!(result.logits.data().len() == 1);
}

#[test]
fn test_infill_eog_token_out_of_range() {
    let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
    let config = InfillConfig::new(vec![100]); // Out of range

    let result = apply_infill_sampling(&logits, &config);
    // Should not panic, p_eog = 0 since token 100 doesn't exist
    assert!((result.p_eog - 0.0).abs() < 1e-6);
}

// =============================================================================
// Beam Search Edge Cases
// =============================================================================

#[test]
fn test_beam_hypothesis_normalized_score_zero_length_penalty() {
    let hyp = BeamHypothesis::new(vec![1, 2, 3], -6.0);
    // length_penalty = 0 means divide by len^0 = 1
    let score = hyp.normalized_score(0.0);
    assert!((score - (-6.0)).abs() < 1e-6);
}

#[test]
fn test_beam_search_state_best_hypotheses_mixed() {
    let config = BeamSearchConfig::new(3).with_num_return(2);
    let mut state = BeamSearchState::new(config, vec![0]);

    // Add some finished and unfinished hypotheses
    state.finished.push(BeamHypothesis {
        tokens: vec![1, 2, 99],
        score: -0.5,
        finished: true,
    });
    state.hypotheses = vec![
        BeamHypothesis::new(vec![1, 3], -1.0),
        BeamHypothesis::new(vec![1, 4], -2.0),
    ];

    let best = state.best_hypotheses();
    assert_eq!(best.len(), 2);
    // Best should be the finished one (higher score)
    assert!(best[0].finished);
}

#[test]
fn test_beam_search_step_with_eos() {
    let config = BeamSearchConfig::new(2);
    let mut state = BeamSearchState::new(config, vec![0]);

    // Log probs where token 4 (EOS) is highly likely
    let log_probs = vec![vec![-10.0, -10.0, -10.0, -10.0, -0.1]];
    state.step(&log_probs, Some(4));

    // EOS hypothesis should be in finished
    assert!(!state.finished.is_empty());
}

// =============================================================================
// Streaming Generator Edge Cases
// =============================================================================

#[test]
fn test_streaming_generator_mixed_tokens() {
    let mut gen = StreamingGenerator::new();
    gen.add_token(1, Some("Hello"));
    gen.add_token(2, None); // No text
    gen.add_token(3, Some(" World"));
    gen.finish();

    assert_eq!(gen.tokens, vec![1, 2, 3]);
    assert_eq!(gen.text, "Hello World");
    assert!(gen.finished);
    assert_eq!(gen.token_count(), 3);
}

// =============================================================================
// Prompt Cache Edge Cases
// =============================================================================

#[test]
fn test_prompt_cache_lru_eviction_order() {
    let mut cache = PromptCache::new(2);
    cache.add(vec![1], 111);
    cache.add(vec![2], 222);

    // Access first entry to make it more recent
    let _ = cache.find_prefix(&[1]);

    // Add third entry - should evict [2] (least recently used)
    cache.add(vec![3], 333);

    assert!(cache.find_prefix(&[1]).is_some()); // Still present
    assert!(cache.find_prefix(&[2]).is_none()); // Evicted
    assert!(cache.find_prefix(&[3]).is_some()); // Present
}

#[test]
fn test_prompt_cache_hit_count_tracking() {
    let mut cache = PromptCache::new(10);
    cache.add(vec![1, 2, 3], 12345);

    // Multiple hits
    for _ in 0..5 {
        let _ = cache.find_prefix(&[1, 2, 3]);
    }

    let stats = cache.stats();
    assert_eq!(stats.total_hits, 5);
}

// =============================================================================
// Token Healing Edge Cases
// =============================================================================

#[test]
fn test_token_healing_none_text() {
    let tokens = vec![1, 2, 3];
    let result = analyze_token_healing(&tokens, None);
    assert_eq!(result.adjusted_tokens, tokens);
    assert!(result.prefix_constraint.is_none());
    assert_eq!(result.tokens_removed, 0);
}

#[test]
fn test_token_healing_long_token_no_heal() {
    let tokens = vec![1, 2, 3];
    // Token longer than 3 chars - no healing
    let result = analyze_token_healing(&tokens, Some("word"));
    assert_eq!(result.adjusted_tokens, tokens);
    assert!(result.prefix_constraint.is_none());
}

#[test]
fn test_token_healing_non_alphanumeric() {
    let tokens = vec![1, 2, 3];
    // Non-alphanumeric characters - no healing
    let result = analyze_token_healing(&tokens, Some("@#"));
    assert_eq!(result.adjusted_tokens, tokens);
    assert!(result.prefix_constraint.is_none());
}

// =============================================================================
// AdvancedGenerationConfig Edge Cases
// =============================================================================

#[test]
fn test_advanced_generation_config_all_options() {
    let config = AdvancedGenerationConfig::new(GenerationConfig::top_k(40))
        .with_stop_sequences(vec!["END".to_string()])
        .with_repetition_penalty(1.2)
        .with_presence_frequency(0.3, 0.5)
        .with_logit_bias(LogitBias::new().with_bias(0, -100.0));

    assert!(config.stop_detector.is_some());
    assert!(config.repetition_penalty.is_some());
    assert!(config.presence_frequency.is_some());
    assert!(config.logit_bias.is_some());
}

#[test]
fn test_apply_all_penalties_with_all_options() {
    let logits = Tensor::from_vec(vec![5], vec![10.0, 10.0, 10.0, 10.0, 10.0]).expect("test");
    let context = vec![0, 0, 1];

    let config = AdvancedGenerationConfig::new(GenerationConfig::greedy())
        .with_repetition_penalty(2.0)
        .with_presence_frequency(0.5, 0.5)
        .with_logit_bias(LogitBias::new().with_bias(4, 50.0));

    let result = apply_all_penalties(&logits, &context, &config);

    // Token 4 should have the highest value (big bias)
    let max_idx = result
        .data()
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .expect("test")
        .0;
    assert_eq!(max_idx, 4);
}

// =============================================================================
// SamplerChain and Context Edge Cases
// =============================================================================

#[test]
fn test_sampler_chain_empty_apply() {
    let chain = SamplerChain::new();
    let mut logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
    let original = logits.data().to_vec();
    let context = SamplerContext::new();

    chain.apply(&mut logits, &context);
    assert_eq!(logits.data(), original.as_slice());
}

#[test]
fn test_sampler_context_all_builders() {
    let ctx = SamplerContext::new()
        .with_tokens(vec![1, 2, 3, 4, 5])
        .with_rng(0.999)
        .with_step(100);

    assert_eq!(ctx.tokens.len(), 5);
    assert!(ctx.rng_value > 0.99);
    assert_eq!(ctx.step, 100);
}

// =============================================================================
// LogitProcessor Edge Cases
// =============================================================================

#[test]
fn test_token_suppressor_from_slice() {
    let suppressor = TokenSuppressor::from_slice(&[0, 1, 2]);
    let mut logits: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let ctx = LogitProcessorContext::new(&[], 0, 4);

    suppressor.process(&mut logits, &ctx);

    assert!(logits[0].is_infinite() && logits[0] < 0.0);
    assert!(logits[1].is_infinite() && logits[1] < 0.0);
    assert!(logits[2].is_infinite() && logits[2] < 0.0);
    assert!((logits[3] - 4.0).abs() < 1e-6);
}

#[test]
fn test_repetition_penalty_processor_window() {
    let penalty = RepetitionPenalty::new(2.0, 1); // Window of 1
    let tokens = vec![0u32, 1, 2, 3]; // Only last token (3) in window
    let mut logits: Vec<f32> = vec![1.0, 1.0, 1.0, 2.0, 1.0];
    let ctx = LogitProcessorContext::new(&tokens, 0, 5);

    penalty.process(&mut logits, &ctx);

    // Only token 3 should be penalized
    assert!((logits[0] - 1.0).abs() < 1e-6);
    assert!((logits[1] - 1.0).abs() < 1e-6);
    assert!((logits[2] - 1.0).abs() < 1e-6);
    assert!((logits[3] - 1.0).abs() < 1e-6); // 2.0 / 2.0 = 1.0
    assert!((logits[4] - 1.0).abs() < 1e-6);
}

#[test]
fn test_logit_processor_chain_with_boxed() {
    let chain =
        LogitProcessorChain::new().with_boxed_processor(Box::new(TemperatureScaler::new(2.0)));

    assert_eq!(chain.len(), 1);
    assert!(!chain.is_empty());
}

// =============================================================================
// Mirostat Edge Cases
// =============================================================================

#[test]
fn test_mirostat_state_converges() {
    let mut state = MirostatState::new(5.0).with_eta(0.5);

    // High surprise should push mu down
    state.update(10.0);
    assert!(state.mu < 10.0);

    // Very low surprise should push mu up
    state.mu = 5.0;
    state.update(0.1);
    assert!(state.mu > 5.0);
}

// =============================================================================
// Typical Sampling Edge Cases
// =============================================================================

#[test]
fn test_typical_sampling_very_low_p() {
    let logits = Tensor::from_vec(vec![5], vec![10.0, 5.0, 1.0, 0.0, -5.0]).expect("test");
    // Very low p should still return a valid token
    let result = sample_typical(&logits, 0.001, 0.5);
    assert!(result.is_ok());
    assert!(result.expect("test") < 5);
}

// =============================================================================
// TFS Edge Cases
// =============================================================================

#[test]
fn test_tfs_three_tokens_minimum() {
    // TFS requires at least 3 tokens for second derivative
    let logits = Tensor::from_vec(vec![3], vec![1.0, 0.5, 0.1]).expect("test");
    let result = sample_tfs(&logits, 0.95, 0.0);
    assert!(result.is_ok());
}

// =============================================================================
// Config Enabled/Disabled Checks
// =============================================================================

#[test]
fn test_all_config_is_enabled_methods() {
    // RepetitionPenaltyConfig
    assert!(!RepetitionPenaltyConfig::new(1.0).is_enabled());
    assert!(RepetitionPenaltyConfig::new(1.1).is_enabled());

    // PresenceFrequencyPenalty
    assert!(!PresenceFrequencyPenalty::new(0.0, 0.0).is_enabled());
    assert!(PresenceFrequencyPenalty::new(0.0, 0.1).is_enabled());
    assert!(PresenceFrequencyPenalty::new(0.1, 0.0).is_enabled());

    // DryConfig
    assert!(!DryConfig::new(0.0).is_enabled());
    assert!(DryConfig::new(0.1).is_enabled());

    // XtcConfig
    assert!(!XtcConfig::default().is_enabled());
    assert!(XtcConfig::new(0.5).is_enabled());

    // EtaConfig
    assert!(!EtaConfig::new(0.0).is_enabled());
    assert!(EtaConfig::new(0.5).is_enabled());

    // CfgConfig
    assert!(!CfgConfig::default().is_enabled());
    assert!(CfgConfig::new(1.5).is_enabled());
}

// =============================================================================
// TokenHealingConfig
// =============================================================================

#[test]
fn test_token_healing_config_builder() {
    let config = TokenHealingConfig::new(true).with_max_backup(20);
    assert!(config.enabled);
    assert_eq!(config.max_backup_chars, 20);
}
