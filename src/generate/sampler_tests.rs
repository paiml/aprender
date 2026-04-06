use super::*;

// ========================================================================
// StopSequenceDetector tests
// ========================================================================

#[test]
fn test_stop_sequence_detector_new() {
    let detector = StopSequenceDetector::new();
    assert!(!detector.has_conditions());
}

#[test]
fn test_stop_sequence_detector_with_token_sequence() {
    let detector = StopSequenceDetector::new().with_token_sequence(vec![1, 2, 3]);
    assert!(detector.has_conditions());
}

#[test]
fn test_stop_sequence_detector_with_string_pattern() {
    let detector = StopSequenceDetector::new().with_string_pattern("stop");
    assert!(detector.has_conditions());
}

#[test]
fn test_stop_sequence_detector_with_stop_strings() {
    let detector =
        StopSequenceDetector::new().with_stop_strings(vec!["end".to_string(), "stop".to_string()]);
    assert!(detector.has_conditions());
}

#[test]
fn test_stop_sequence_detector_check_token() {
    let mut detector = StopSequenceDetector::new().with_token_sequence(vec![1, 2, 3]);
    assert!(!detector.check_token(1));
    assert!(!detector.check_token(2));
    assert!(detector.check_token(3)); // Sequence complete
}

#[test]
fn test_stop_sequence_detector_check_text() {
    let detector = StopSequenceDetector::new().with_string_pattern("stop");
    assert!(detector.check_text("please stop now").is_some());
    assert!(detector.check_text("continue going").is_none());
}

#[test]
fn test_stop_sequence_detector_check_text_position() {
    let detector = StopSequenceDetector::new().with_string_pattern("stop");
    let pos = detector.check_text("please stop now");
    assert_eq!(pos, Some(7)); // "stop" starts at position 7
}

#[test]
fn test_stop_sequence_detector_reset() {
    let mut detector = StopSequenceDetector::new().with_token_sequence(vec![1, 2]);
    detector.check_token(1);
    detector.reset();
    // After reset, partial match should be cleared
    assert!(!detector.check_token(2));
}

#[test]
fn test_stop_sequence_detector_empty_patterns() {
    // Empty patterns should be ignored
    let detector = StopSequenceDetector::new()
        .with_token_sequence(vec![])
        .with_string_pattern("")
        .with_stop_strings(vec![String::new()]);
    assert!(!detector.has_conditions());
}

// ========================================================================
// RepetitionPenaltyConfig tests
// ========================================================================

#[test]
fn test_repetition_penalty_config_default() {
    let config = RepetitionPenaltyConfig::default();
    assert_eq!(config.penalty, 1.0);
    assert_eq!(config.window_size, 64);
    assert!(!config.is_enabled()); // 1.0 = no penalty
}

#[test]
fn test_repetition_penalty_config_new() {
    let config = RepetitionPenaltyConfig::new(1.2);
    assert_eq!(config.penalty, 1.2);
    assert!(config.is_enabled());
}

#[test]
fn test_repetition_penalty_config_with_window() {
    let config = RepetitionPenaltyConfig::new(1.5).with_window(128);
    assert_eq!(config.window_size, 128);
    assert_eq!(config.penalty, 1.5);
}

#[test]
fn test_repetition_penalty_config_is_enabled() {
    assert!(!RepetitionPenaltyConfig::new(1.0).is_enabled());
    assert!(RepetitionPenaltyConfig::new(1.1).is_enabled());
    assert!(RepetitionPenaltyConfig::new(0.9).is_enabled());
}

// ========================================================================
// PresenceFrequencyPenalty tests
// ========================================================================

#[test]
fn test_presence_frequency_penalty_default() {
    let config = PresenceFrequencyPenalty::default();
    assert_eq!(config.presence_penalty, 0.0);
    assert_eq!(config.frequency_penalty, 0.0);
    assert!(!config.is_enabled());
}

#[test]
fn test_presence_frequency_penalty_new() {
    let config = PresenceFrequencyPenalty::new(0.5, 0.3);
    assert_eq!(config.presence_penalty, 0.5);
    assert_eq!(config.frequency_penalty, 0.3);
    assert!(config.is_enabled());
}

#[test]
fn test_presence_frequency_penalty_is_enabled() {
    assert!(!PresenceFrequencyPenalty::new(0.0, 0.0).is_enabled());
    assert!(PresenceFrequencyPenalty::new(0.1, 0.0).is_enabled());
    assert!(PresenceFrequencyPenalty::new(0.0, 0.1).is_enabled());
}

// ========================================================================
// LogitBias tests
// ========================================================================

#[test]
fn test_logit_bias_new() {
    let bias = LogitBias::new();
    assert!(bias.is_empty());
}

#[test]
fn test_logit_bias_with_bias() {
    let bias = LogitBias::new().with_bias(10, 5.0);
    assert!(!bias.is_empty());
    assert_eq!(bias.get(10), 5.0);
    assert_eq!(bias.get(20), 0.0); // Not set
}

#[test]
fn test_logit_bias_with_biases() {
    let mut biases = HashMap::new();
    biases.insert(1, 1.0);
    biases.insert(2, 2.0);
    let bias = LogitBias::new().with_biases(biases);
    assert_eq!(bias.get(1), 1.0);
    assert_eq!(bias.get(2), 2.0);
}

#[test]
fn test_logit_bias_get_default() {
    let bias = LogitBias::new();
    assert_eq!(bias.get(999), 0.0); // Default for unset
}

// ========================================================================
// apply_repetition_penalty tests
// ========================================================================

#[test]
fn test_apply_repetition_penalty_no_penalty() {
    let logits = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let config = RepetitionPenaltyConfig::new(1.0); // No penalty
    let result = apply_repetition_penalty(&logits, &[0, 1, 2], &config);
    assert_eq!(result.data(), logits.data());
}

#[test]
fn test_apply_repetition_penalty_empty_context() {
    let logits = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let config = RepetitionPenaltyConfig::new(1.5);
    let result = apply_repetition_penalty(&logits, &[], &config);
    assert_eq!(result.data(), logits.data());
}

#[test]
fn test_apply_repetition_penalty_positive_logits() {
    let logits = Tensor::from_vec(vec![4], vec![2.0, 2.0, 2.0, 2.0]).unwrap();
    let config = RepetitionPenaltyConfig::new(2.0);
    let result = apply_repetition_penalty(&logits, &[0, 2], &config);
    let data = result.data();
    // Token 0 and 2 should be penalized (divided by 2.0)
    assert_eq!(data[0], 1.0);
    assert_eq!(data[1], 2.0); // Not in context
    assert_eq!(data[2], 1.0);
    assert_eq!(data[3], 2.0); // Not in context
}

#[test]
fn test_apply_repetition_penalty_negative_logits() {
    let logits = Tensor::from_vec(vec![4], vec![-2.0, -2.0, -2.0, -2.0]).unwrap();
    let config = RepetitionPenaltyConfig::new(2.0);
    let result = apply_repetition_penalty(&logits, &[0], &config);
    let data = result.data();
    // Negative logits get multiplied by penalty
    assert_eq!(data[0], -4.0);
    assert_eq!(data[1], -2.0); // Not in context
}

// ========================================================================
// apply_presence_frequency_penalty tests
// ========================================================================

#[test]
fn test_apply_presence_frequency_penalty_disabled() {
    let logits = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let config = PresenceFrequencyPenalty::default();
    let result = apply_presence_frequency_penalty(&logits, &[0, 1], &config);
    assert_eq!(result.data(), logits.data());
}

#[test]
fn test_apply_presence_frequency_penalty_presence() {
    let logits = Tensor::from_vec(vec![4], vec![5.0, 5.0, 5.0, 5.0]).unwrap();
    let config = PresenceFrequencyPenalty::new(1.0, 0.0); // Only presence
    let result = apply_presence_frequency_penalty(&logits, &[0, 2], &config);
    let data = result.data();
    assert_eq!(data[0], 4.0); // 5.0 - 1.0
    assert_eq!(data[1], 5.0); // Not in context
    assert_eq!(data[2], 4.0); // 5.0 - 1.0
    assert_eq!(data[3], 5.0); // Not in context
}

#[test]
fn test_apply_presence_frequency_penalty_frequency() {
    let logits = Tensor::from_vec(vec![4], vec![10.0, 10.0, 10.0, 10.0]).unwrap();
    let config = PresenceFrequencyPenalty::new(0.0, 0.5); // Only frequency
    // Token 0 appears 3 times
    let result = apply_presence_frequency_penalty(&logits, &[0, 0, 0, 1], &config);
    let data = result.data();
    assert_eq!(data[0], 8.5); // 10.0 - 0.5 * 3
    assert_eq!(data[1], 9.5); // 10.0 - 0.5 * 1
}

// ========================================================================
// apply_logit_bias tests
// ========================================================================

#[test]
fn test_apply_logit_bias_empty() {
    let logits = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let bias = LogitBias::new();
    let result = apply_logit_bias(&logits, &bias);
    assert_eq!(result.data(), logits.data());
}

#[test]
fn test_apply_logit_bias_applied() {
    let logits = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let bias = LogitBias::new().with_bias(0, 10.0).with_bias(3, -5.0);
    let result = apply_logit_bias(&logits, &bias);
    let data = result.data();
    assert_eq!(data[0], 11.0); // 1.0 + 10.0
    assert_eq!(data[1], 2.0); // No bias
    assert_eq!(data[2], 3.0); // No bias
    assert_eq!(data[3], -1.0); // 4.0 + (-5.0)
}

// ========================================================================
// TemperatureSampler tests
// ========================================================================

#[test]
fn test_temperature_sampler_name() {
    let sampler = TemperatureSampler::new(0.5);
    assert_eq!(sampler.name(), "temperature");
}

#[test]
fn test_temperature_sampler_clone_box() {
    let sampler = TemperatureSampler::new(0.7);
    let _boxed = sampler.clone_box();
}

// ========================================================================
// DynTempConfig tests
// ========================================================================

#[test]
fn test_dyn_temp_config_default() {
    let config = DynTempConfig::default();
    assert_eq!(config.temp, 1.0);
    assert_eq!(config.delta, 0.0);
    assert_eq!(config.exponent, 1.0);
}

// ========================================================================
// InfillConfig tests
// ========================================================================

#[test]
fn test_infill_config_default() {
    let config = InfillConfig::default();
    assert!(config.eog_tokens.is_empty());
    assert!(config.eog_ratio_threshold > 0.0);
}

// ========================================================================
// BeamSearchConfig tests
// ========================================================================

#[test]
fn test_beam_search_config_default() {
    let config = BeamSearchConfig::default();
    assert!(config.num_beams > 0);
    assert_eq!(config.length_penalty, 1.0);
}

// ========================================================================
// TokenSuppressor tests
// ========================================================================

#[test]
fn test_token_suppressor_new() {
    let suppressor = TokenSuppressor::new(vec![0, 1, 2]);
    // Just verify it can be created
    assert_eq!(suppressor.name(), "token_suppressor");
}

#[test]
fn test_token_suppressor_from_slice() {
    let ids: &[u32] = &[10, 20, 30];
    let suppressor = TokenSuppressor::from_slice(ids);
    assert_eq!(suppressor.name(), "token_suppressor");
}

// ========================================================================
// PromptCacheStats tests
// ========================================================================

#[test]
fn test_prompt_cache_stats_fields() {
    let stats = PromptCacheStats {
        entries: 5,
        total_hits: 10,
        max_entries: 100,
    };
    assert_eq!(stats.entries, 5);
    assert_eq!(stats.total_hits, 10);
    assert_eq!(stats.max_entries, 100);
}

// ========================================================================
// SamplerContext tests
// ========================================================================

#[test]
fn test_sampler_context() {
    let ctx = SamplerContext {
        tokens: vec![1, 2, 3],
        rng_value: 0.5,
        step: 5,
    };
    assert_eq!(ctx.tokens, vec![1, 2, 3]);
    assert_eq!(ctx.rng_value, 0.5);
    assert_eq!(ctx.step, 5);
}

// ========================================================================
// LogitProcessorContext tests
// ========================================================================

#[test]
fn test_logit_processor_context() {
    let ctx = LogitProcessorContext::new(&[1, 2, 3], 5, 1000);
    assert_eq!(ctx.tokens, &[1, 2, 3]);
    assert_eq!(ctx.step, 5);
    assert_eq!(ctx.n_vocab, 1000);
}

// ========================================================================
// LogitProcessorChain tests
// ========================================================================

#[test]
fn test_logit_processor_chain_new() {
    let chain = LogitProcessorChain::new();
    let mut logits = vec![1.0, 2.0, 3.0];
    let ctx = LogitProcessorContext::new(&[0, 1], 0, 100);
    chain.process(&mut logits, &ctx);
    // Empty chain should not modify logits
    assert_eq!(logits, vec![1.0, 2.0, 3.0]);
}

// ========================================================================
// AdvancedGenerationConfig tests
// ========================================================================

#[test]
fn test_advanced_generation_config_new() {
    let base = GenerationConfig::default();
    let config = AdvancedGenerationConfig::new(base);
    assert!(config.stop_detector.is_none());
    assert!(config.repetition_penalty.is_none());
}

#[test]
fn test_advanced_generation_config_with_stop_sequences() {
    let base = GenerationConfig::default();
    let config = AdvancedGenerationConfig::new(base).with_stop_sequences(vec!["stop".to_string()]);
    assert!(config.stop_detector.is_some());
}
