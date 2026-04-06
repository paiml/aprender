
    /// Mock model for testing
    struct MockModel {
        vocab_size: usize,
        eos_token: u32,
        /// Fixed token to return
        fixed_token: u32,
    }

    impl MockModel {
        fn new(vocab_size: usize, fixed_token: u32) -> Self {
            Self {
                vocab_size,
                eos_token: 0,
                fixed_token,
            }
        }
    }

    impl SpeculativeModel for MockModel {
        fn forward(&self, _input_ids: &[u32]) -> Result<Vec<f32>, SpeculativeError> {
            // Return uniform logits
            Ok(vec![0.0; self.vocab_size])
        }

        fn sample(&self, _logits: &[f32]) -> Result<TokenProb, SpeculativeError> {
            Ok(TokenProb::new(self.fixed_token, -1.0))
        }

        fn vocab_size(&self) -> usize {
            self.vocab_size
        }

        fn eos_token(&self) -> u32 {
            self.eos_token
        }
    }

    // === TokenProb Tests ===

    #[test]
    fn test_token_prob_new() {
        let tp = TokenProb::new(42, -1.0);
        assert_eq!(tp.token, 42);
        assert_eq!(tp.log_prob, -1.0);
    }

    #[test]
    fn test_token_prob_prob() {
        let tp = TokenProb::new(42, 0.0);
        assert!((tp.prob() - 1.0).abs() < 0.001);

        let tp2 = TokenProb::new(42, -1.0);
        assert!((tp2.prob() - 0.368).abs() < 0.01);
    }

    // === SpeculativeStats Tests ===

    #[test]
    fn test_speculative_stats_default() {
        let stats = SpeculativeStats::default();
        assert_eq!(stats.iterations, 0);
        assert_eq!(stats.tokens_speculated, 0);
        assert_eq!(stats.acceptance_rate, 0.0);
    }

    #[test]
    fn test_speculative_stats_record() {
        let mut stats = SpeculativeStats::default();
        stats.record_iteration(4, 3, 1.0, 10.0);

        assert_eq!(stats.iterations, 1);
        assert_eq!(stats.tokens_speculated, 4);
        assert_eq!(stats.tokens_accepted, 3);
        assert_eq!(stats.acceptance_rate, 0.75);
    }

    #[test]
    fn test_speculative_stats_speedup() {
        let mut stats = SpeculativeStats::default();
        stats.record_iteration(4, 4, 1.0, 10.0);

        let speedup = stats.speedup();
        assert!(speedup > 1.0);
    }

    #[test]
    fn test_speculative_stats_serialization() {
        let stats = SpeculativeStats {
            iterations: 10,
            tokens_speculated: 40,
            tokens_accepted: 30,
            acceptance_rate: 0.75,
            avg_spec_length: 4.0,
            time_saved_ms: 100.0,
            draft_time_ms: 10.0,
            target_time_ms: 100.0,
        };

        let json = serde_json::to_string(&stats).expect("test");
        let parsed: SpeculativeStats = serde_json::from_str(&json).expect("test");

        assert_eq!(parsed.iterations, stats.iterations);
        assert_eq!(parsed.acceptance_rate, stats.acceptance_rate);
    }

    // === SpeculativeResult Tests ===

    #[test]
    fn test_speculative_result_acceptance_rate() {
        let result = SpeculativeResult {
            accepted_tokens: vec![1, 2, 3],
            num_speculated: 4,
            num_accepted: 3,
            resampled_token: Some(4),
            draft_time_ms: 1.0,
            target_time_ms: 10.0,
        };

        assert_eq!(result.acceptance_rate(), 0.75);
        assert!(!result.all_accepted());
    }

    #[test]
    fn test_speculative_result_all_accepted() {
        let result = SpeculativeResult {
            accepted_tokens: vec![1, 2, 3, 4],
            num_speculated: 4,
            num_accepted: 4,
            resampled_token: None,
            draft_time_ms: 1.0,
            target_time_ms: 10.0,
        };

        assert!(result.all_accepted());
    }

    // === SpeculativeConfig Tests ===

    #[test]
    fn test_speculative_config_default() {
        let config = SpeculativeConfig::default();
        assert_eq!(config.spec_length, 4);
        assert!(config.adaptive);
    }

    #[test]
    fn test_speculative_config_builder() {
        let config = SpeculativeConfig::new()
            .with_spec_length(6)
            .with_adaptive(false);

        assert_eq!(config.spec_length, 6);
        assert!(!config.adaptive);
    }

    #[test]
    fn test_speculative_config_adapt_increase() {
        let mut config = SpeculativeConfig::new().with_spec_length(4);
        config.adapt_spec_length(0.9); // High acceptance

        assert!(config.spec_length >= 4);
    }

    #[test]
    fn test_speculative_config_adapt_decrease() {
        let mut config = SpeculativeConfig::new().with_spec_length(4);
        config.adapt_spec_length(0.3); // Low acceptance

        assert!(config.spec_length <= 4);
    }

    #[test]
    fn test_speculative_config_no_adapt_when_disabled() {
        let mut config = SpeculativeConfig::new()
            .with_spec_length(4)
            .with_adaptive(false);

        config.adapt_spec_length(0.1);
        assert_eq!(config.spec_length, 4); // Should not change
    }

    // === SpeculativeDecoder Tests ===

    #[test]
    fn test_speculative_decoder_new() {
        let draft = MockModel::new(100, 1);
        let target = MockModel::new(100, 1);

        let decoder = SpeculativeDecoder::new(draft, target, 4).expect("test");
        assert_eq!(decoder.spec_length(), 4);
    }

    #[test]
    fn test_speculative_decoder_invalid_spec_length() {
        let draft = MockModel::new(100, 1);
        let target = MockModel::new(100, 1);

        let result = SpeculativeDecoder::new(draft, target, 0);
        assert!(matches!(
            result,
            Err(SpeculativeError::InvalidSpecLength(0))
        ));
    }

    #[test]
    fn test_speculative_decoder_decode_iteration() {
        let draft = MockModel::new(100, 1);
        let target = MockModel::new(100, 1);

        let mut decoder = SpeculativeDecoder::new(draft, target, 4).expect("test");
        let result = decoder.decode_iteration(&[10, 20, 30]).expect("test");

        assert!(!result.accepted_tokens.is_empty());
        assert!(result.num_speculated > 0);
    }

    #[test]
    fn test_speculative_decoder_set_spec_length() {
        let draft = MockModel::new(100, 1);
        let target = MockModel::new(100, 1);

        let mut decoder = SpeculativeDecoder::new(draft, target, 4).expect("test");
        decoder.set_spec_length(8).expect("test");
        assert_eq!(decoder.spec_length(), 8);

        let err = decoder.set_spec_length(0);
        assert!(err.is_err());
    }

    #[test]
    fn test_speculative_decoder_stats() {
        let draft = MockModel::new(100, 1);
        let target = MockModel::new(100, 1);

        let mut decoder = SpeculativeDecoder::new(draft, target, 4).expect("test");
        let _ = decoder.decode_iteration(&[10]).expect("test");

        let stats = decoder.stats();
        assert_eq!(stats.iterations, 1);
    }

    #[test]
    fn test_speculative_decoder_reset_stats() {
        let draft = MockModel::new(100, 1);
        let target = MockModel::new(100, 1);

        let mut decoder = SpeculativeDecoder::new(draft, target, 4).expect("test");
        let _ = decoder.decode_iteration(&[10]).expect("test");

        decoder.reset_stats();
        assert_eq!(decoder.stats().iterations, 0);
    }

    // === Error Tests ===

    #[test]
    fn test_speculative_error_display() {
        let err = SpeculativeError::DraftModelError("test".to_string());
        assert!(err.to_string().contains("Draft"));

        let err = SpeculativeError::TargetModelError("test".to_string());
        assert!(err.to_string().contains("Target"));

        let err = SpeculativeError::InvalidSpecLength(0);
        assert!(err.to_string().contains("0"));

        let err = SpeculativeError::VerificationFailed { position: 3 };
        assert!(err.to_string().contains("3"));
    }

    // ============================================================================
    // Additional Coverage Tests
    // ============================================================================

    // === TokenProb Extended Tests ===

    #[test]
    fn test_token_prob_clone() {
        let tp = TokenProb::new(42, -2.0);
        let tp_clone = tp.clone();
        assert_eq!(tp.token, tp_clone.token);
        assert_eq!(tp.log_prob, tp_clone.log_prob);
    }

    #[test]
    fn test_token_prob_debug() {
        let tp = TokenProb::new(42, -1.5);
        let debug_str = format!("{:?}", tp);
        assert!(debug_str.contains("42"));
        assert!(debug_str.contains("-1.5"));
    }

    #[test]
    fn test_token_prob_extreme_values() {
        // Very negative log prob (near-zero probability)
        let tp = TokenProb::new(1, -100.0);
        assert!(tp.prob() < 1e-40);

        // Positive log prob (probability > 1, edge case)
        let tp2 = TokenProb::new(1, 1.0);
        assert!((tp2.prob() - std::f32::consts::E).abs() < 0.01);
    }

    // === SpeculativeStats Extended Tests ===

    #[test]
    fn test_speculative_stats_speedup_zero_accepted() {
        let stats = SpeculativeStats::default();
        // tokens_accepted is 0, should return 1.0
        assert_eq!(stats.speedup(), 1.0);
    }

    #[test]
    fn test_speculative_stats_speedup_with_many_iterations() {
        let mut stats = SpeculativeStats::default();
        // Simulate many iterations
        for _ in 0..100 {
            stats.record_iteration(4, 3, 1.0, 10.0);
        }
        let speedup = stats.speedup();
        // With 400 speculated, 300 accepted, 100 iterations
        // baseline_time = 300, draft_equiv = 400 * 0.1 = 40, actual = 40 + 100 = 140
        // speedup = 300 / 140 ≈ 2.14
        assert!(speedup > 1.5);
        assert!(speedup < 3.0);
    }

    #[test]
    fn test_speculative_stats_record_zero_speculated() {
        let mut stats = SpeculativeStats::default();
        // Edge case: zero speculated tokens
        stats.record_iteration(0, 0, 0.5, 5.0);

        assert_eq!(stats.iterations, 1);
        assert_eq!(stats.tokens_speculated, 0);
        // acceptance_rate remains 0 when tokens_speculated is 0
        assert_eq!(stats.acceptance_rate, 0.0);
        // avg_spec_length = 0 / 1 = 0
        assert_eq!(stats.avg_spec_length, 0.0);
    }

    #[test]
    fn test_speculative_stats_time_saved_calculation() {
        let mut stats = SpeculativeStats::default();
        // 4 speculated, 4 accepted, target took 40ms
        // time_per_token = 40 / 4 = 10ms
        // time_saved = (4-1) * 10 = 30ms
        stats.record_iteration(4, 4, 1.0, 40.0);
        assert!((stats.time_saved_ms - 30.0).abs() < 0.01);
    }

    #[test]
    fn test_speculative_stats_time_saved_one_accepted() {
        let mut stats = SpeculativeStats::default();
        // Only 1 accepted: saturating_sub(1) = 0, no time saved
        stats.record_iteration(4, 1, 1.0, 40.0);
        assert_eq!(stats.time_saved_ms, 0.0);
    }

    #[test]
    fn test_speculative_stats_clone() {
        let stats = SpeculativeStats {
            iterations: 5,
            tokens_speculated: 20,
            tokens_accepted: 15,
            acceptance_rate: 0.75,
            avg_spec_length: 4.0,
            time_saved_ms: 50.0,
            draft_time_ms: 5.0,
            target_time_ms: 50.0,
        };
        let cloned = stats.clone();
        assert_eq!(stats.iterations, cloned.iterations);
        assert_eq!(stats.acceptance_rate, cloned.acceptance_rate);
    }

    // === SpeculativeResult Extended Tests ===

    #[test]
    fn test_speculative_result_acceptance_rate_zero_speculated() {
        let result = SpeculativeResult {
            accepted_tokens: vec![],
            num_speculated: 0,
            num_accepted: 0,
            resampled_token: None,
            draft_time_ms: 0.0,
            target_time_ms: 0.0,
        };
        // Should return 0.0 when num_speculated is 0
        assert_eq!(result.acceptance_rate(), 0.0);
    }

    #[test]
    fn test_speculative_result_clone() {
        let result = SpeculativeResult {
            accepted_tokens: vec![1, 2, 3],
            num_speculated: 4,
            num_accepted: 3,
            resampled_token: Some(5),
            draft_time_ms: 2.5,
            target_time_ms: 25.0,
        };
        let cloned = result.clone();
        assert_eq!(result.accepted_tokens, cloned.accepted_tokens);
        assert_eq!(result.resampled_token, cloned.resampled_token);
    }

    #[test]
    fn test_speculative_result_debug() {
        let result = SpeculativeResult {
            accepted_tokens: vec![1, 2],
            num_speculated: 3,
            num_accepted: 2,
            resampled_token: Some(4),
            draft_time_ms: 1.0,
            target_time_ms: 10.0,
        };
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("accepted_tokens"));
        assert!(debug_str.contains("resampled_token"));
    }

    // === SpeculativeConfig Extended Tests ===

    #[test]
    fn test_speculative_config_adapt_at_max() {
        let mut config = SpeculativeConfig {
            spec_length: 8, // Already at max
            min_acceptance_rate: 0.5,
            adaptive: true,
            max_spec_length: 8,
        };
        config.adapt_spec_length(0.95); // Very high acceptance
        // Should not exceed max
        assert_eq!(config.spec_length, 8);
    }

    #[test]
    fn test_speculative_config_adapt_at_min() {
        let mut config = SpeculativeConfig {
            spec_length: 1, // Already at min
            min_acceptance_rate: 0.5,
            adaptive: true,
            max_spec_length: 8,
        };
        config.adapt_spec_length(0.1); // Very low acceptance
        // Should not go below 1
        assert_eq!(config.spec_length, 1);
    }
