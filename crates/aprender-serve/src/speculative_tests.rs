
    // === Error Type Coverage ===

    #[test]
    fn test_speculative_error_debug() {
        let err = SpeculativeError::DraftModelError("test error".to_string());
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("DraftModelError"));
        assert!(debug_str.contains("test error"));

        let err2 = SpeculativeError::VerificationFailed { position: 7 };
        let debug_str2 = format!("{:?}", err2);
        assert!(debug_str2.contains("VerificationFailed"));
        assert!(debug_str2.contains("7"));
    }

    // === MockModel Extended Tests ===

    #[test]
    fn test_mock_model_vocab_size() {
        let model = MockModel::new(256, 42);
        assert_eq!(model.vocab_size(), 256);
    }

    #[test]
    fn test_mock_model_eos_token() {
        let model = MockModel::new(100, 5);
        assert_eq!(model.eos_token(), 0);
    }

    #[test]
    fn test_mock_model_forward_returns_correct_size() {
        let model = MockModel::new(50, 1);
        let logits = model.forward(&[1, 2, 3]).expect("forward");
        assert_eq!(logits.len(), 50);
    }

    // === SpeculativeResult all_accepted Edge Cases ===

    #[test]
    fn test_speculative_result_all_accepted_zero() {
        let result = SpeculativeResult {
            accepted_tokens: vec![],
            num_speculated: 0,
            num_accepted: 0,
            resampled_token: None,
            draft_time_ms: 0.0,
            target_time_ms: 0.0,
        };
        // 0 == 0, so all_accepted is true
        assert!(result.all_accepted());
    }

    // === Config adapt_spec_length boundary tests ===

    #[test]
    fn test_speculative_config_adapt_exactly_at_threshold() {
        let mut config = SpeculativeConfig {
            spec_length: 4,
            min_acceptance_rate: 0.5,
            adaptive: true,
            max_spec_length: 8,
        };

        // Exactly at min_acceptance_rate - should not decrease
        config.adapt_spec_length(0.5);
        assert_eq!(config.spec_length, 4);

        // Exactly at 0.8 - should not increase (> 0.8 is required)
        config.adapt_spec_length(0.8);
        assert_eq!(config.spec_length, 4);
    }

    #[test]
    fn test_speculative_config_adapt_just_above_high_threshold() {
        let mut config = SpeculativeConfig {
            spec_length: 4,
            min_acceptance_rate: 0.5,
            adaptive: true,
            max_spec_length: 8,
        };

        // Just above 0.8 - should increase
        config.adapt_spec_length(0.81);
        assert_eq!(config.spec_length, 5);
    }

    #[test]
    fn test_speculative_config_adapt_just_below_low_threshold() {
        let mut config = SpeculativeConfig {
            spec_length: 4,
            min_acceptance_rate: 0.5,
            adaptive: true,
            max_spec_length: 8,
        };

        // Just below min_acceptance_rate - should decrease
        config.adapt_spec_length(0.49);
        assert_eq!(config.spec_length, 3);
    }

    // ============================================================================
    // Part 02: Additional Coverage Tests for Edge Cases
    // ============================================================================

    // === SpeculativeStats Debug Trait ===

    #[test]
    fn test_speculative_stats_debug() {
        let stats = SpeculativeStats {
            iterations: 10,
            tokens_speculated: 40,
            tokens_accepted: 32,
            acceptance_rate: 0.8,
            avg_spec_length: 4.0,
            time_saved_ms: 120.5,
            draft_time_ms: 15.0,
            target_time_ms: 150.0,
        };
        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("iterations"));
        assert!(debug_str.contains("tokens_speculated"));
        assert!(debug_str.contains("acceptance_rate"));
        assert!(debug_str.contains("time_saved_ms"));
    }

    // === SpeculativeStats Edge Cases ===

    #[test]
    fn test_speculative_stats_speedup_zero_iterations_zero_speculated() {
        // Edge case: manually created stats with no activity
        let stats = SpeculativeStats {
            iterations: 0,
            tokens_speculated: 0,
            tokens_accepted: 0,
            acceptance_rate: 0.0,
            avg_spec_length: 0.0,
            time_saved_ms: 0.0,
            draft_time_ms: 0.0,
            target_time_ms: 0.0,
        };
        // tokens_accepted == 0 means early return with 1.0
        assert_eq!(stats.speedup(), 1.0);
    }

    #[test]
    fn test_speculative_stats_speedup_actual_time_calculation() {
        // Test the actual speedup calculation path
        let mut stats = SpeculativeStats::default();
        // 10 speculated, 8 accepted, 1 iteration
        // draft_tokens_equivalent = 10 * 0.1 = 1.0
        // baseline_time = 8
        // actual_time = 1.0 + 1 = 2.0
        // speedup = 8 / 2 = 4.0
        stats.record_iteration(10, 8, 1.0, 10.0);
        let speedup = stats.speedup();
        assert!((speedup - 4.0).abs() < 0.1);
    }

    #[test]
    fn test_speculative_stats_multiple_records_accumulate() {
        let mut stats = SpeculativeStats::default();
        stats.record_iteration(4, 3, 1.0, 10.0);
        stats.record_iteration(4, 4, 1.5, 12.0);
        stats.record_iteration(4, 2, 0.8, 8.0);

        assert_eq!(stats.iterations, 3);
        assert_eq!(stats.tokens_speculated, 12);
        assert_eq!(stats.tokens_accepted, 9);
        assert!((stats.draft_time_ms - 3.3).abs() < 0.01);
        assert!((stats.target_time_ms - 30.0).abs() < 0.01);
    }

    #[test]
    fn test_speculative_stats_acceptance_rate_precision() {
        let mut stats = SpeculativeStats::default();
        stats.record_iteration(3, 2, 1.0, 10.0);
        // 2/3 = 0.6666...
        assert!((stats.acceptance_rate - 0.6666667).abs() < 0.0001);
    }

    // === SpeculativeResult Edge Cases ===

    #[test]
    fn test_speculative_result_partial_acceptance() {
        let result = SpeculativeResult {
            accepted_tokens: vec![100, 200],
            num_speculated: 5,
            num_accepted: 2,
            resampled_token: Some(300),
            draft_time_ms: 2.5,
            target_time_ms: 25.0,
        };
        // 2/5 = 0.4
        assert!((result.acceptance_rate() - 0.4).abs() < 0.001);
        assert!(!result.all_accepted());
    }

    #[test]
    fn test_speculative_result_100_percent_acceptance() {
        let result = SpeculativeResult {
            accepted_tokens: vec![10, 20, 30, 40, 50],
            num_speculated: 5,
            num_accepted: 5,
            resampled_token: None,
            draft_time_ms: 1.0,
            target_time_ms: 10.0,
        };
        assert!((result.acceptance_rate() - 1.0).abs() < 0.001);
        assert!(result.all_accepted());
    }

    // === TokenProb Edge Cases ===

    #[test]
    fn test_token_prob_zero_log_prob() {
        // log_prob = 0 means prob = exp(0) = 1.0
        let tp = TokenProb::new(999, 0.0);
        assert!((tp.prob() - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_token_prob_very_positive_log_prob() {
        // Edge case: positive log_prob (prob > 1, mathematically unusual but handled)
        let tp = TokenProb::new(1, 2.0);
        let expected = std::f32::consts::E.powi(2);
        assert!((tp.prob() - expected).abs() < 0.1);
    }

    #[test]
    fn test_token_prob_negative_infinity_log_prob() {
        // Very negative log_prob approximates zero probability
        let tp = TokenProb::new(1, f32::NEG_INFINITY);
        assert_eq!(tp.prob(), 0.0);
    }

    // === SpeculativeConfig Edge Cases ===

    #[test]
    fn test_speculative_config_adapt_spec_length_two_decreases() {
        let mut config = SpeculativeConfig {
            spec_length: 3,
            min_acceptance_rate: 0.5,
            adaptive: true,
            max_spec_length: 8,
        };
        // First decrease
        config.adapt_spec_length(0.3);
        assert_eq!(config.spec_length, 2);
        // Second decrease
        config.adapt_spec_length(0.3);
        assert_eq!(config.spec_length, 1);
        // Third attempt - should stay at 1 (minimum)
        config.adapt_spec_length(0.3);
        assert_eq!(config.spec_length, 1);
    }

    #[test]
    fn test_speculative_config_adapt_spec_length_two_increases() {
        let mut config = SpeculativeConfig {
            spec_length: 6,
            min_acceptance_rate: 0.5,
            adaptive: true,
            max_spec_length: 8,
        };
        // First increase
        config.adapt_spec_length(0.9);
        assert_eq!(config.spec_length, 7);
        // Second increase
        config.adapt_spec_length(0.9);
        assert_eq!(config.spec_length, 8);
        // Third attempt - should stay at 8 (maximum)
        config.adapt_spec_length(0.9);
        assert_eq!(config.spec_length, 8);
    }

    #[test]
    fn test_speculative_config_custom_min_acceptance_rate() {
        let mut config = SpeculativeConfig {
            spec_length: 4,
            min_acceptance_rate: 0.7, // Custom higher threshold
            adaptive: true,
            max_spec_length: 8,
        };
        // 0.65 is below 0.7, should decrease
        config.adapt_spec_length(0.65);
        assert_eq!(config.spec_length, 3);
    }

    // === SpeculativeDecoder Edge Cases ===

    /// Model that generates multiple tokens then hits EOS
    struct DelayedEosModel {
        vocab_size: usize,
        eos_token: u32,
        call_count: std::cell::RefCell<usize>,
        eos_after: usize,
    }

    impl DelayedEosModel {
        fn new(vocab_size: usize, eos_token: u32, eos_after: usize) -> Self {
            Self {
                vocab_size,
                eos_token,
                call_count: std::cell::RefCell::new(0),
                eos_after,
            }
        }
    }

    impl SpeculativeModel for DelayedEosModel {
        fn forward(&self, _input_ids: &[u32]) -> Result<Vec<f32>, SpeculativeError> {
            Ok(vec![0.0; self.vocab_size])
        }

        fn sample(&self, _logits: &[f32]) -> Result<TokenProb, SpeculativeError> {
            let mut count = self.call_count.borrow_mut();
            *count += 1;
            if *count > self.eos_after {
                Ok(TokenProb::new(self.eos_token, -0.5))
            } else {
                Ok(TokenProb::new(42, -1.0)) // Non-EOS token
            }
        }

        fn vocab_size(&self) -> usize {
            self.vocab_size
        }

        fn eos_token(&self) -> u32 {
            self.eos_token
        }
    }

    #[test]
    fn test_speculative_decoder_eos_after_two_tokens() {
        let draft = DelayedEosModel::new(100, 0, 2); // EOS after 2 tokens
        let target = MockModel::new(100, 42);

        let mut decoder = SpeculativeDecoder::new(draft, target, 4).expect("create");
        let result = decoder.decode_iteration(&[1]).expect("decode");

        // Should have 3 tokens: 2 non-EOS + 1 EOS
        assert_eq!(result.num_speculated, 3);
    }

    #[test]
    fn test_speculative_decoder_large_context() {
        let draft = MockModel::new(100, 1);
        let target = MockModel::new(100, 1);

        let mut decoder = SpeculativeDecoder::new(draft, target, 2).expect("create");
        // Large context of 1000 tokens
        let context: Vec<u32> = (0..1000).collect();
        let result = decoder.decode_iteration(&context).expect("decode");

        // Should still work with large context
        assert!(result.num_speculated > 0);
    }

    #[test]
    fn test_speculative_decoder_single_token_spec_length() {
        let draft = MockModel::new(100, 1);
        let target = MockModel::new(100, 1);

        let mut decoder = SpeculativeDecoder::new(draft, target, 1).expect("create");
        let result = decoder.decode_iteration(&[10, 20]).expect("decode");

        // With spec_length=1, should only speculate 1 token
        assert_eq!(result.num_speculated, 1);
    }

    #[test]
    fn test_speculative_decoder_max_spec_length_32() {
        // Model that doesn't hit EOS
        struct NoEosModel {
            vocab_size: usize,
        }

        impl SpeculativeModel for NoEosModel {
            fn forward(&self, _input_ids: &[u32]) -> Result<Vec<f32>, SpeculativeError> {
                Ok(vec![0.0; self.vocab_size])
            }

            fn sample(&self, _logits: &[f32]) -> Result<TokenProb, SpeculativeError> {
                Ok(TokenProb::new(999, -1.0)) // Never EOS (EOS is 0)
            }

            fn vocab_size(&self) -> usize {
                self.vocab_size
            }

            fn eos_token(&self) -> u32 {
                0
            }
        }

        let draft = NoEosModel { vocab_size: 100 };
        let target = NoEosModel { vocab_size: 100 };

        let mut decoder = SpeculativeDecoder::new(draft, target, 32).expect("create");
        let result = decoder.decode_iteration(&[1]).expect("decode");

        // Should speculate exactly 32 tokens (max spec_length)
        assert_eq!(result.num_speculated, 32);
    }

    // === should_accept Comprehensive Tests ===

    #[test]
    fn test_should_accept_ratio_exactly_one() {
        let draft = MockModel::new(100, 1);
        let target = MockModel::new(100, 1);
        let decoder = SpeculativeDecoder::new(draft, target, 4).expect("create");

        // Same probability, different tokens
        let draft_prob = TokenProb::new(5, -2.0);
        let target_prob = TokenProb::new(10, -2.0);

        // ratio = exp(-2) / exp(-2) = 1.0, should accept (ratio >= 1.0)
        assert!(decoder.should_accept(&draft_prob, &target_prob));
    }

    #[test]
    fn test_should_accept_ratio_slightly_above_half() {
        let draft = MockModel::new(100, 1);
        let target = MockModel::new(100, 1);
        let decoder = SpeculativeDecoder::new(draft, target, 4).expect("create");

        // Calculate log probs to get ratio just above 0.5
        // draft_prob = 0.6, target_prob = 0.31 => ratio ≈ 0.517
        let _draft_prob = TokenProb::new(5, (-0.6_f32).ln());
        let _target_prob = TokenProb::new(10, (-0.31_f32).ln());

        // This is a negative log prob scenario - let's use cleaner values
        // draft prob = exp(-1) ≈ 0.368, target prob = exp(-1.3) ≈ 0.273
        // ratio = 0.273 / 0.368 ≈ 0.74 > 0.5, should accept
        let draft_prob = TokenProb::new(5, -1.0);
        let target_prob = TokenProb::new(10, -1.3);
        assert!(decoder.should_accept(&draft_prob, &target_prob));
    }
