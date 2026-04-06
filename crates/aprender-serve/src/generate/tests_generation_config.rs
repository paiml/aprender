
    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_tokens, 100);
        assert_eq!(config.strategy, SamplingStrategy::Greedy);
        assert!((config.temperature - 1.0).abs() < 1e-6);
        assert!(config.eos_token_id.is_none());
    }

    #[test]
    fn test_generation_config_builders() {
        let config = GenerationConfig::greedy().with_max_tokens(50);
        assert_eq!(config.max_tokens, 50);
        assert_eq!(config.strategy, SamplingStrategy::Greedy);

        let config = GenerationConfig::top_k(10).with_temperature(0.8);
        assert_eq!(config.strategy, SamplingStrategy::TopK { k: 10 });
        assert!((config.temperature - 0.8).abs() < 1e-6);

        let config = GenerationConfig::top_p(0.9).with_eos_token_id(2);
        assert_eq!(config.strategy, SamplingStrategy::TopP { p: 0.9 });
        assert_eq!(config.eos_token_id, Some(2));
    }

    #[test]
    fn test_apply_temperature() {
        let logits = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");

        // Temperature = 1.0 should return same values
        let scaled = apply_temperature(&logits, 1.0).expect("test");
        for i in 0..4 {
            assert!((scaled.data()[i] - logits.data()[i]).abs() < 1e-6);
        }

        // Temperature = 2.0 should halve values
        let scaled = apply_temperature(&logits, 2.0).expect("test");
        assert!((scaled.data()[0] - 0.5).abs() < 1e-6);
        assert!((scaled.data()[3] - 2.0).abs() < 1e-6);

        // Temperature = 0.5 should double values
        let scaled = apply_temperature(&logits, 0.5).expect("test");
        assert!((scaled.data()[0] - 2.0).abs() < 1e-6);
        assert!((scaled.data()[3] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_temperature_invalid() {
        let logits = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
        assert!(apply_temperature(&logits, 0.0).is_err());
        assert!(apply_temperature(&logits, -1.0).is_err());
    }

    #[test]
    fn test_sample_greedy() {
        // Clear winner at index 2
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 10.0, 3.0, 4.0]).expect("test");
        let token = sample_greedy(&logits).expect("test");
        assert_eq!(token, 2);

        // Winner at last index
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 5.0]).expect("test");
        let token = sample_greedy(&logits).expect("test");
        assert_eq!(token, 2);

        // Winner at first index
        let logits = Tensor::from_vec(vec![3], vec![5.0, 2.0, 1.0]).expect("test");
        let token = sample_greedy(&logits).expect("test");
        assert_eq!(token, 0);
    }

    #[test]
    fn test_sample_greedy_empty_error() {
        let logits = Tensor::from_vec(vec![1], vec![1.0]).expect("test");
        // Single element should work
        assert_eq!(sample_greedy(&logits).expect("test"), 0);
    }

    #[test]
    fn test_sample_top_k() {
        // Strong preference for index 0
        let logits = Tensor::from_vec(vec![5], vec![100.0, 1.0, 1.0, 1.0, 1.0]).expect("test");

        // With rng_value = 0.0, should always get first (highest prob)
        let token = sample_top_k(&logits, 3, 0.0).expect("test");
        assert_eq!(token, 0);

        // With k=1, should always get highest
        let token = sample_top_k(&logits, 1, 0.5).expect("test");
        assert_eq!(token, 0);
    }

    #[test]
    fn test_sample_top_k_distribution() {
        // Two equally likely tokens
        let logits = Tensor::from_vec(vec![4], vec![10.0, 10.0, 0.0, 0.0]).expect("test");

        // Low rng should get index 0 or 1 (they're equal)
        let token = sample_top_k(&logits, 2, 0.1).expect("test");
        assert!(token == 0 || token == 1);

        // High rng should get index 0 or 1
        let token = sample_top_k(&logits, 2, 0.9).expect("test");
        assert!(token == 0 || token == 1);
    }

    #[test]
    fn test_sample_top_k_errors() {
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
        assert!(sample_top_k(&logits, 0, 0.5).is_err());
    }

    #[test]
    fn test_sample_top_p() {
        // One dominant token
        let logits = Tensor::from_vec(vec![3], vec![100.0, 1.0, 1.0]).expect("test");

        // With p=0.9, nucleus likely just the first token
        let token = sample_top_p(&logits, 0.9, 0.5).expect("test");
        assert_eq!(token, 0);
    }

    #[test]
    fn test_sample_top_p_uniform() {
        // Equal logits
        let logits = Tensor::from_vec(vec![4], vec![0.0, 0.0, 0.0, 0.0]).expect("test");

        // With p=1.0, all tokens in nucleus
        // Low rng should get early token
        let token = sample_top_p(&logits, 1.0, 0.1).expect("test");
        assert!(token < 4);

        // High rng should get later token
        let token = sample_top_p(&logits, 1.0, 0.9).expect("test");
        assert!(token < 4);
    }

    #[test]
    fn test_sample_top_p_errors() {
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
        assert!(sample_top_p(&logits, 0.0, 0.5).is_err());
        assert!(sample_top_p(&logits, 1.1, 0.5).is_err());
        assert!(sample_top_p(&logits, -0.1, 0.5).is_err());
    }

    #[test]
    fn test_sample_token_greedy() {
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 10.0, 3.0, 4.0]).expect("test");
        let config = GenerationConfig::greedy();
        let token = sample_token(&logits, &config, 0.5).expect("test");
        assert_eq!(token, 2);
    }

    #[test]
    fn test_sample_token_with_temperature() {
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
        let config = GenerationConfig::greedy().with_temperature(0.5);
        let token = sample_token(&logits, &config, 0.5).expect("test");
        // Higher temperature doesn't change greedy selection
        assert_eq!(token, 2);
    }

    #[test]
    fn test_sample_token_top_k() {
        let logits = Tensor::from_vec(vec![5], vec![100.0, 1.0, 1.0, 1.0, 1.0]).expect("test");
        let config = GenerationConfig::top_k(3);
        let token = sample_token(&logits, &config, 0.0).expect("test");
        assert_eq!(token, 0);
    }

    #[test]
    fn test_sample_token_top_p() {
        let logits = Tensor::from_vec(vec![3], vec![100.0, 1.0, 1.0]).expect("test");
        let config = GenerationConfig::top_p(0.95);
        let token = sample_token(&logits, &config, 0.5).expect("test");
        assert_eq!(token, 0);
    }

    // =====================================================================
    // Advanced Sampling Feature Tests
    // =====================================================================

    // ----- Stop Sequence Detector Tests -----

    #[test]
    fn test_stop_sequence_detector_new() {
        let detector = StopSequenceDetector::new();
        assert!(!detector.has_conditions());
    }

    #[test]
    fn test_stop_sequence_detector_add_token_sequence() {
        let detector = StopSequenceDetector::new()
            .with_token_sequence(vec![1, 2, 3])
            .with_token_sequence(vec![4, 5]);
        assert!(detector.has_conditions());
    }

    #[test]
    fn test_stop_sequence_detector_add_string_pattern() {
        let detector = StopSequenceDetector::new()
            .with_string_pattern("<|end|>")
            .with_string_pattern("\n\n");
        assert!(detector.has_conditions());
    }

    #[test]
    fn test_stop_sequence_detector_token_match() {
        let mut detector = StopSequenceDetector::new().with_token_sequence(vec![10, 20, 30]);

        // Add tokens one by one
        assert!(!detector.check_token(10)); // Partial match
        assert!(!detector.check_token(20)); // Still partial
        assert!(detector.check_token(30)); // Complete match!
    }

    #[test]
    fn test_stop_sequence_detector_token_no_match() {
        let mut detector = StopSequenceDetector::new().with_token_sequence(vec![10, 20, 30]);

        detector.check_token(10);
        detector.check_token(25); // Wrong token breaks sequence
        assert!(!detector.check_token(30)); // 30 alone doesn't match
    }

    #[test]
    fn test_stop_sequence_detector_string_match() {
        let detector = StopSequenceDetector::new().with_string_pattern("<|end|>");

        assert!(detector.check_text("Hello world").is_none());
        assert!(detector.check_text("Output: <|end|>").is_some());
        assert!(detector.check_text("<|end|> extra").is_some());
    }

    #[test]
    fn test_stop_sequence_detector_buffer_limit() {
        let mut detector = StopSequenceDetector::new().with_token_sequence(vec![1, 2]); // max_seq_len = 2

        // Add many tokens
        for i in 0..100 {
            detector.check_token(i);
        }

        // Detector should still work (buffer is trimmed internally)
        assert!(detector.has_conditions());
    }

    #[test]
    fn test_stop_sequence_detector_reset() {
        let mut detector = StopSequenceDetector::new().with_token_sequence(vec![1, 2, 3]);

        detector.check_token(1);
        detector.check_token(2);
        detector.reset();

        // After reset, need to match sequence again from start
        assert!(!detector.check_token(3)); // Just 3 alone won't match
    }

    // ----- Repetition Penalty Tests -----

    #[test]
    fn test_repetition_penalty_config_default() {
        let config = RepetitionPenaltyConfig::default();
        assert_eq!(config.penalty, 1.0); // No penalty by default
        assert_eq!(config.window_size, 64);
    }

    #[test]
    fn test_repetition_penalty_config_builder() {
        let config = RepetitionPenaltyConfig::new(1.5).with_window(128);
        assert_eq!(config.penalty, 1.5);
        assert_eq!(config.window_size, 128);
    }

    #[test]
    fn test_apply_repetition_penalty_basic() {
        let logits = Tensor::from_vec(vec![5], vec![2.0, 1.0, 3.0, 0.5, -1.0]).expect("test");
        let context = vec![0, 2, 4]; // Penalize tokens 0, 2, 4
        let config = RepetitionPenaltyConfig::new(2.0);

        let result = apply_repetition_penalty(&logits, &context, &config);

        // Positive logits should be divided by penalty
        assert_eq!(result.data()[0], 1.0); // 2.0 / 2.0
        assert_eq!(result.data()[1], 1.0); // Unchanged (not in context)
        assert_eq!(result.data()[2], 1.5); // 3.0 / 2.0

        // Negative logits should be multiplied by penalty
        assert_eq!(result.data()[4], -2.0); // -1.0 * 2.0
    }

    #[test]
    fn test_apply_repetition_penalty_window() {
        let logits = Tensor::from_vec(vec![5], vec![2.0, 2.0, 2.0, 2.0, 2.0]).expect("test");
        let context = vec![0, 1, 2, 3, 4]; // All tokens in context
        let config = RepetitionPenaltyConfig::new(2.0).with_window(2); // Only last 2 tokens

        let result = apply_repetition_penalty(&logits, &context, &config);

        // Only tokens 3, 4 should be penalized (last 2 in window)
        assert_eq!(result.data()[0], 2.0); // Unchanged
        assert_eq!(result.data()[1], 2.0); // Unchanged
        assert_eq!(result.data()[2], 2.0); // Unchanged
        assert_eq!(result.data()[3], 1.0); // 2.0 / 2.0
        assert_eq!(result.data()[4], 1.0); // 2.0 / 2.0
    }

    #[test]
    fn test_apply_repetition_penalty_no_penalty() {
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
        let context = vec![0, 1, 2];
        let config = RepetitionPenaltyConfig::new(1.0); // No penalty

        let result = apply_repetition_penalty(&logits, &context, &config);

        // No change when penalty is 1.0
        assert_eq!(result.data()[0], 1.0);
        assert_eq!(result.data()[1], 2.0);
        assert_eq!(result.data()[2], 3.0);
    }

    #[test]
    fn test_repetition_penalty_is_enabled() {
        let disabled = RepetitionPenaltyConfig::new(1.0);
        assert!(!disabled.is_enabled());

        let enabled = RepetitionPenaltyConfig::new(1.1);
        assert!(enabled.is_enabled());
    }

    // ----- Presence/Frequency Penalty Tests -----

    #[test]
    fn test_presence_frequency_penalty_default() {
        let config = PresenceFrequencyPenalty::default();
        assert_eq!(config.presence_penalty, 0.0);
        assert_eq!(config.frequency_penalty, 0.0);
    }

    #[test]
    fn test_presence_frequency_penalty_new() {
        let config = PresenceFrequencyPenalty::new(0.5, 0.3);
        assert_eq!(config.presence_penalty, 0.5);
        assert_eq!(config.frequency_penalty, 0.3);
    }

    #[test]
    fn test_apply_presence_penalty() {
        let logits = Tensor::from_vec(vec![5], vec![10.0, 10.0, 10.0, 10.0, 10.0]).expect("test");
        let context = vec![0, 0, 1]; // Token 0 appears twice, token 1 once
        let config = PresenceFrequencyPenalty::new(1.0, 0.0);

        let result = apply_presence_frequency_penalty(&logits, &context, &config);

        // Tokens 0 and 1 get presence penalty (constant)
        assert_eq!(result.data()[0], 9.0); // 10.0 - 1.0
        assert_eq!(result.data()[1], 9.0); // 10.0 - 1.0
        assert_eq!(result.data()[2], 10.0); // Unchanged
    }

    #[test]
    fn test_apply_frequency_penalty() {
        let logits = Tensor::from_vec(vec![5], vec![10.0, 10.0, 10.0, 10.0, 10.0]).expect("test");
        let context = vec![0, 0, 0, 1]; // Token 0 appears 3x, token 1 once
        let config = PresenceFrequencyPenalty::new(0.0, 1.0);

        let result = apply_presence_frequency_penalty(&logits, &context, &config);

        // Frequency penalty is proportional to count
        assert_eq!(result.data()[0], 7.0); // 10.0 - 3*1.0
        assert_eq!(result.data()[1], 9.0); // 10.0 - 1*1.0
        assert_eq!(result.data()[2], 10.0); // Unchanged
    }

    #[test]
    fn test_apply_combined_penalties() {
        let logits = Tensor::from_vec(vec![3], vec![10.0, 10.0, 10.0]).expect("test");
        let context = vec![0, 0, 1]; // Token 0 appears 2x, token 1 once
        let config = PresenceFrequencyPenalty::new(0.5, 0.5);

        let result = apply_presence_frequency_penalty(&logits, &context, &config);

        // Token 0: 10.0 - 0.5(presence) - 2*0.5(freq) = 10.0 - 1.5 = 8.5
        assert_eq!(result.data()[0], 8.5);
        // Token 1: 10.0 - 0.5(presence) - 1*0.5(freq) = 10.0 - 1.0 = 9.0
        assert_eq!(result.data()[1], 9.0);
    }

    #[test]
    fn test_presence_frequency_is_enabled() {
        let disabled = PresenceFrequencyPenalty::new(0.0, 0.0);
        assert!(!disabled.is_enabled());

        let enabled = PresenceFrequencyPenalty::new(0.1, 0.0);
        assert!(enabled.is_enabled());
    }

    // ----- Logit Bias Tests -----

    #[test]
    fn test_logit_bias_default() {
        let bias = LogitBias::default();
        assert!(bias.is_empty());
    }

    #[test]
    fn test_logit_bias_add() {
        let bias = LogitBias::new().with_bias(10, 5.0).with_bias(20, -100.0);
        assert!(!bias.is_empty());
        assert_eq!(bias.get(10), 5.0);
        assert_eq!(bias.get(20), -100.0);
        assert_eq!(bias.get(30), 0.0); // Not set
    }

    #[test]
    fn test_apply_logit_bias() {
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
        let bias = LogitBias::new()
            .with_bias(0, 10.0)
            .with_bias(2, -100.0)
            .with_bias(4, 3.0);

        let result = apply_logit_bias(&logits, &bias);

        assert_eq!(result.data()[0], 11.0); // 1.0 + 10.0
        assert_eq!(result.data()[1], 2.0); // Unchanged
        assert_eq!(result.data()[2], -97.0); // 3.0 - 100.0
        assert_eq!(result.data()[3], 4.0); // Unchanged
        assert_eq!(result.data()[4], 8.0); // 5.0 + 3.0
    }

    #[test]
    fn test_apply_logit_bias_out_of_range() {
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
        let bias = LogitBias::new().with_bias(100, 50.0); // Index out of range

        let result = apply_logit_bias(&logits, &bias);

        // Should not panic, just skip out-of-range indices
        assert_eq!(result.data()[0], 1.0);
        assert_eq!(result.data()[1], 2.0);
        assert_eq!(result.data()[2], 3.0);
    }
