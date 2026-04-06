//! Additional tests for generate/mod.rs coverage (Part 2)
//!
//! Focus: Token generation edge cases, sampling boundaries, stop conditions

#[cfg(test)]
mod tests {
    use crate::generate::*;
    use crate::tensor::Tensor;

    // =========================================================================
    // Token Generation Edge Cases
    // =========================================================================

    #[test]
    fn test_sample_greedy_very_large_logits() {
        let logits = Tensor::from_vec(vec![3], vec![1e30, 1e31, 1e32]).expect("test");
        let token = sample_greedy(&logits).expect("test");
        assert_eq!(token, 2);
    }

    #[test]
    fn test_sample_greedy_very_small_logits() {
        let logits = Tensor::from_vec(vec![3], vec![-1e30, -1e31, -1e32]).expect("test");
        let token = sample_greedy(&logits).expect("test");
        assert_eq!(token, 0);
    }

    #[test]
    fn test_sample_greedy_mixed_inf_values() {
        let logits = Tensor::from_vec(vec![4], vec![f32::NEG_INFINITY, 1.0, f32::INFINITY, 2.0])
            .expect("test");
        let token = sample_greedy(&logits).expect("test");
        assert_eq!(token, 2);
    }

    #[test]
    fn test_sample_greedy_all_same_values() {
        let logits = Tensor::from_vec(vec![5], vec![3.14; 5]).expect("test");
        let token = sample_greedy(&logits).expect("test");
        assert_eq!(token, 0);
    }

    #[test]
    fn test_sample_greedy_nan_handling() {
        let logits = Tensor::from_vec(vec![3], vec![1.0, f32::NAN, 2.0]).expect("test");
        assert!(sample_greedy(&logits).is_ok());
    }

    // =========================================================================
    // Top-K Sampling Edge Cases
    // =========================================================================

    #[test]
    fn test_sample_top_k_k_equals_one_always_greedy() {
        let logits = Tensor::from_vec(vec![5], vec![1.0, 5.0, 3.0, 2.0, 4.0]).expect("test");
        for rng in [0.0, 0.5, 0.99] {
            assert_eq!(sample_top_k(&logits, 1, rng).expect("test"), 1);
        }
    }

    #[test]
    fn test_sample_top_k_rng_boundary_zero() {
        let logits = Tensor::from_vec(vec![3], vec![10.0, 5.0, 1.0]).expect("test");
        assert_eq!(sample_top_k(&logits, 3, 0.0).expect("test"), 0);
    }

    #[test]
    fn test_sample_top_k_rng_boundary_almost_one() {
        let logits = Tensor::from_vec(vec![3], vec![10.0, 5.0, 1.0]).expect("test");
        assert!(sample_top_k(&logits, 3, 0.9999).expect("test") < 3);
    }

    #[test]
    fn test_sample_top_k_equal_logits() {
        let logits = Tensor::from_vec(vec![5], vec![1.0; 5]).expect("test");
        assert!(sample_top_k(&logits, 5, 0.1).expect("test") < 5);
        assert!(sample_top_k(&logits, 5, 0.9).expect("test") < 5);
    }

    // =========================================================================
    // Top-P Sampling Edge Cases
    // =========================================================================

    #[test]
    fn test_sample_top_p_single_dominant_token() {
        let logits = Tensor::from_vec(vec![5], vec![100.0, 0.0, 0.0, 0.0, 0.0]).expect("test");
        assert_eq!(sample_top_p(&logits, 0.9, 0.5).expect("test"), 0);
    }

    #[test]
    fn test_sample_top_p_rng_selects_within_nucleus() {
        let logits = Tensor::from_vec(vec![4], vec![2.0, 2.0, 1.0, 0.0]).expect("test");
        let token = sample_top_p(&logits, 0.99, 0.0).expect("test");
        assert!(token == 0 || token == 1);
    }

    #[test]
    fn test_sample_top_p_with_negative_logits() {
        let logits = Tensor::from_vec(vec![4], vec![-1.0, -2.0, -3.0, -4.0]).expect("test");
        assert!(sample_top_p(&logits, 0.8, 0.5).expect("test") < 4);
    }

    // =========================================================================
    // Temperature Scaling Edge Cases
    // =========================================================================

    #[test]
    fn test_apply_temperature_epsilon() {
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
        let scaled = apply_temperature(&logits, 1.0 + 1e-7).expect("test");
        for i in 0..3 {
            assert!((scaled.data()[i] - logits.data()[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_apply_temperature_zero_logits() {
        let logits = Tensor::from_vec(vec![4], vec![0.0; 4]).expect("test");
        let scaled = apply_temperature(&logits, 2.0).expect("test");
        for &val in scaled.data() {
            assert!((val - 0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_apply_temperature_tiny_positive() {
        let logits = Tensor::from_vec(vec![2], vec![1.0, 2.0]).expect("test");
        let scaled = apply_temperature(&logits, 1e-6).expect("test");
        assert!(scaled.data()[0] > 1e5);
    }

    // =========================================================================
    // Sample Token Integration
    // =========================================================================

    #[test]
    fn test_sample_token_greedy_ignores_rng() {
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 10.0, 3.0, 4.0]).expect("test");
        let config = GenerationConfig::greedy();
        for rng in [0.0, 0.5, 1.0] {
            assert_eq!(sample_token(&logits, &config, rng).expect("test"), 2);
        }
    }

    #[test]
    fn test_sample_token_top_k_with_high_temp() {
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
        let config = GenerationConfig::top_k(3).with_temperature(100.0);
        assert!(sample_token(&logits, &config, 0.5).expect("test") < 5);
    }

    #[test]
    fn test_sample_token_top_p_with_low_temp() {
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
        let config = GenerationConfig::top_p(0.95).with_temperature(0.01);
        assert_eq!(sample_token(&logits, &config, 0.5).expect("test"), 4);
    }

    // =========================================================================
    // Stop Sequence Detector Edge Cases
    // =========================================================================

    #[test]
    fn test_stop_sequence_detector_single_token_sequence() {
        let mut detector = StopSequenceDetector::new().with_token_sequence(vec![42]);
        assert!(!detector.check_token(10));
        assert!(detector.check_token(42));
    }

    #[test]
    fn test_stop_sequence_detector_long_sequence() {
        let mut detector =
            StopSequenceDetector::new().with_token_sequence(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        for i in 1..10 {
            assert!(!detector.check_token(i));
        }
        assert!(detector.check_token(10));
    }

    #[test]
    fn test_stop_sequence_detector_overlapping_sequences() {
        let mut detector = StopSequenceDetector::new()
            .with_token_sequence(vec![1, 2])
            .with_token_sequence(vec![2, 3]);
        detector.check_token(1);
        assert!(detector.check_token(2));
        detector.reset();
        detector.check_token(2);
        assert!(detector.check_token(3));
    }

    #[test]
    fn test_stop_sequence_detector_repeated_token() {
        let mut detector = StopSequenceDetector::new().with_token_sequence(vec![5, 5, 5]);
        detector.check_token(5);
        detector.check_token(5);
        assert!(detector.check_token(5));
    }

    #[test]
    fn test_stop_sequence_detector_text_positions() {
        let detector = StopSequenceDetector::new().with_string_pattern("end");
        assert_eq!(detector.check_text("the end"), Some(4));
        assert_eq!(detector.check_text("no match"), None);
    }

    #[test]
    fn test_stop_sequence_detector_case_sensitive() {
        let detector = StopSequenceDetector::new().with_string_pattern("STOP");
        assert!(detector.check_text("stop").is_none());
        assert!(detector.check_text("STOP").is_some());
    }

    #[test]
    fn test_stop_sequence_detector_unicode() {
        let detector = StopSequenceDetector::new().with_string_pattern("\u{2603}");
        assert!(detector.check_text("Winter \u{2603} here").is_some());
    }

    // =========================================================================
    // Generation Config Variants
    // =========================================================================

    #[test]
    fn test_generation_config_sampling_strategy_equality() {
        assert_eq!(SamplingStrategy::Greedy, SamplingStrategy::Greedy);
        assert_eq!(
            SamplingStrategy::TopK { k: 50 },
            SamplingStrategy::TopK { k: 50 }
        );
        assert_ne!(
            SamplingStrategy::TopK { k: 50 },
            SamplingStrategy::TopK { k: 100 }
        );
    }

    #[test]
    fn test_generation_config_copy() {
        let config = GenerationConfig::top_k(50).with_temperature(0.8);
        let copied = config;
        assert_eq!(copied.strategy, SamplingStrategy::TopK { k: 50 });
    }

    #[test]
    fn test_generation_config_debug_format() {
        let config = GenerationConfig::greedy();
        assert!(format!("{:?}", config).contains("Greedy"));
    }

    // =========================================================================
    // Helper Function Edge Cases
    // =========================================================================

    #[test]
    fn test_sample_from_distribution_single_element() {
        assert_eq!(sample_from_distribution(&[1.0], &[42], 0.5), 42);
    }

    #[test]
    fn test_sample_from_distribution_rng_exactly_zero() {
        assert_eq!(sample_from_distribution(&[0.5, 0.5], &[10, 20], 0.0), 10);
    }

    #[test]
    fn test_build_nucleus_very_small_p() {
        let indexed = vec![(0, 0.5), (1, 0.3)];
        assert!(!build_nucleus(&indexed, 0.01).is_empty());
    }

    #[test]
    fn test_logits_to_probs_with_large_difference() {
        let probs = logits_to_probs(&[(0, 100.0), (1, 0.0)]);
        assert!(probs[0] > 0.99);
    }

    #[test]
    fn test_logits_to_probs_all_negative() {
        let probs = logits_to_probs(&[(0, -10.0), (1, -20.0), (2, -30.0)]);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    // =========================================================================
    // Min-P Sampling
    // =========================================================================

    #[test]
    fn test_sample_min_p_filters_correctly() {
        let logits = Tensor::from_vec(vec![5], vec![5.0, 4.0, 3.0, 2.0, 1.0]).expect("test");
        assert!(sample_min_p(&logits, 0.5, 0.0).expect("test") < 3);
    }

    #[test]
    fn test_sample_min_p_all_filtered_fallback() {
        let logits = Tensor::from_vec(vec![3], vec![10.0, 1.0, 0.1]).expect("test");
        assert_eq!(sample_min_p(&logits, 0.99, 0.5).expect("test"), 0);
    }

    // =========================================================================
    // Mirostat State
    // =========================================================================

    #[test]
    fn test_mirostat_state_mu_stays_finite() {
        let mut state = MirostatState::new(5.0);
        for _ in 0..100 {
            state.update(100.0);
        }
        assert!(state.mu.is_finite());
    }

    // =========================================================================
    // Advanced Generation Config
    // =========================================================================

    #[test]
    fn test_advanced_generation_config_multiple_penalties() {
        let config = AdvancedGenerationConfig::new(GenerationConfig::top_k(50))
            .with_repetition_penalty(1.2)
            .with_presence_frequency(0.5, 0.3);
        assert!(config.repetition_penalty.is_some());
        assert!(config.presence_frequency.is_some());
    }

    #[test]
    fn test_apply_all_penalties_with_repetition_only() {
        let logits = Tensor::from_vec(vec![5], vec![5.0, 4.0, 3.0, 2.0, 1.0]).expect("test");
        let config =
            AdvancedGenerationConfig::new(GenerationConfig::greedy()).with_repetition_penalty(2.0);
        let result = apply_all_penalties(&logits, &[0, 1, 0], &config);
        assert!(result.data()[0] < logits.data()[0]);
    }

    #[test]
    fn test_apply_all_penalties_with_logit_bias_only() {
        let logits = Tensor::from_vec(vec![5], vec![1.0; 5]).expect("test");
        let config = AdvancedGenerationConfig::new(GenerationConfig::greedy())
            .with_logit_bias(LogitBias::new().with_bias(2, 100.0));
        let result = apply_all_penalties(&logits, &[], &config);
        assert!((result.data()[2] - 101.0).abs() < 1e-6);
    }

    // =========================================================================
    // Repetition Penalty Edge Cases
    // =========================================================================

    #[test]
    fn test_repetition_penalty_zero_logit() {
        let logits = Tensor::from_vec(vec![3], vec![0.0, 1.0, -1.0]).expect("test");
        let result = apply_repetition_penalty(&logits, &[0], &RepetitionPenaltyConfig::new(2.0));
        assert!((result.data()[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_repetition_penalty_out_of_bounds_token() {
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
        let result = apply_repetition_penalty(&logits, &[100], &RepetitionPenaltyConfig::new(2.0));
        assert_eq!(result.data(), logits.data());
    }

    // =========================================================================
    // Presence/Frequency Penalty Edge Cases
    // =========================================================================

    #[test]
    fn test_presence_frequency_high_frequency() {
        let logits = Tensor::from_vec(vec![3], vec![10.0, 10.0, 10.0]).expect("test");
        let config = PresenceFrequencyPenalty::new(1.0, 1.0);
        let result = apply_presence_frequency_penalty(&logits, &[0, 0, 0, 0, 0], &config);
        assert!((result.data()[0] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_presence_frequency_empty_context() {
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
        let config = PresenceFrequencyPenalty::new(1.0, 1.0);
        let result = apply_presence_frequency_penalty(&logits, &[], &config);
        assert_eq!(result.data(), logits.data());
    }

    // =========================================================================
    // Streaming Generator State
    // =========================================================================

    #[test]
    fn test_streaming_generator_large_text() {
        let mut generator = StreamingGenerator::new();
        generator.add_token(1, Some(&"a".repeat(10000)));
        assert_eq!(generator.text.len(), 10000);
    }

    #[test]
    fn test_streaming_generator_empty_text() {
        let mut generator = StreamingGenerator::new();
        generator.add_token(1, Some(""));
        generator.add_token(2, Some(""));
        assert!(generator.text.is_empty());
        assert_eq!(generator.tokens.len(), 2);
    }
}
